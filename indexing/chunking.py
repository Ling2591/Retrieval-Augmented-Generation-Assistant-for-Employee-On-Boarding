# indexing/chunking.py
# PHASE 1, step 2. Cuts the cleaned page records into chunks.
#
# Input   the page records from loaders.load_documents()
# Output  a list of chunk records, each a dictionary with five fields.
#   id       unique and stable, for example "brightline_05_leave_policy_2026_md_002"
#   text     what gets embedded, searched and shown to the language model
#   source   the file name
#   page     the page number, or 0 for files without pages
#   section  the heading path, for example "Leave Policy (effective 1 January 2026) > Annual leave"
#
# The strategy, in plain words
#   1. Follow the document structure. Headings define sections and a chunk never
#      mixes two sections.
#   2. Start every chunk with its section path. Without it, a chunk that says
#      "20 days of annual leave" would not reveal that it belongs to the 2026 policy.
#   3. Fill a chunk with whole paragraphs until it is full, then start the next one.
#      Long paragraphs are cut at sentence ends. Neighbouring chunks of the same
#      section overlap a little so a sentence on the border is not lost.
#   4. Tables are never cut in the middle of a row. A long table is split by rows
#      and the header row is repeated in every piece, so each row keeps its meaning.
#      A single row that is too long for one chunk is cut like ordinary text.
#   5. Every chunk must fit both the character limit and the embedding model's
#      token limit, because the model silently ignores anything beyond its limit.

import re

import config
from shared.embeddings import count_tokens

# A little room under the model limit, because the model adds 2 special tokens.
_TOKEN_LIMIT = config.EMBEDDING_MAX_TOKENS - 6

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_SEPARATOR = re.compile(r"^\|[\s:\-|]+\|?$")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _with_header(header, body):
    return f"{header}\n{body}" if header else body


def _fits(header, body):
    """True when header plus body stays inside both limits."""
    text = _with_header(header, body)
    return len(text) <= config.CHUNK_SIZE_CHARS and count_tokens(text) <= _TOKEN_LIMIT


def _breadcrumb(stack):
    crumb = " > ".join(title for _, title in stack)
    if len(crumb) > 160:                       # keep very long paths from eating the chunk
        crumb = "..." + crumb[-157:]
    return crumb


def _slug(source, used):
    """Make a safe, unique id prefix from a file name."""
    slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_") or "doc"
    candidate, number = slug, 2
    while candidate in used:
        candidate = f"{slug}_{number}"
        number += 1
    used.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Step 1. Turn the text of a document into ordered "units"
# A unit is one paragraph or one table, tagged with its page and section path.
# ---------------------------------------------------------------------------
def _blocks(lines):
    """Split consecutive non-blank lines into paragraphs and tables."""
    blocks, current, current_is_table = [], [], None
    for line in lines:
        is_table = line.startswith("|")
        if current and is_table != current_is_table:
            blocks.append(("table" if current_is_table else "para", "\n".join(current)))
            current = []
        current.append(line)
        current_is_table = is_table
    if current:
        blocks.append(("table" if current_is_table else "para", "\n".join(current)))
    return blocks


def _units_for_source(records):
    units, stack = [], []          # stack holds the headings that are currently open
    for record in records:
        buffer = []
        lines = record["text"].split("\n") + [""]          # the final blank line flushes the buffer
        for raw in lines:
            line = raw.strip()
            heading = _HEADING.match(line)
            if heading or not line:
                for kind, text in _blocks(buffer):
                    units.append({"page": record["page"], "section": _breadcrumb(stack),
                                  "kind": kind, "text": text})
                buffer = []
                if heading:
                    level, title = len(heading.group(1)), heading.group(2).strip()
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
            else:
                buffer.append(line)
    return units


# ---------------------------------------------------------------------------
# Step 2. Splitting things that are too big for one chunk
# Every function here returns pieces that fit both limits.
# ---------------------------------------------------------------------------
def _force_fit(header, text):
    """Last resort for text that cannot be cut at a sentence or a space, for
    example a very long web address. Cut by characters and halve the step until
    every piece fits."""
    step = max(20, config.CHUNK_SIZE_CHARS - len(header) - 20)
    while True:
        pieces = [text[i:i + step] for i in range(0, len(text), step)]
        if step <= 10 or all(_fits(header, piece) for piece in pieces):
            return pieces
        step //= 2


def _split_text(header, text):
    """Cut a long paragraph at sentence ends or line breaks, then at spaces,
    then by characters. Every piece returned fits both limits."""
    atoms = [a for a in re.split(r"(?<=[.!?])\s+|\n", text) if a.strip()]
    pieces, current = [], ""
    for atom in atoms:
        units = [atom] if _fits(header, atom) else atom.split(" ")
        for unit in units:
            parts = [unit] if _fits(header, unit) else _force_fit(header, unit)
            for part in parts:
                candidate = f"{current} {part}".strip()
                if current and not _fits(header, candidate):
                    pieces.append(current)
                    current = part
                else:
                    current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_table(header, table):
    """Cut a long table by rows and repeat the header rows in every piece.
    A single row that is too long even on its own is cut like ordinary text.
    Every piece returned fits both limits."""
    lines = table.split("\n")
    if len(lines) >= 2 and _TABLE_SEPARATOR.match(lines[1]):
        head_rows, rows = lines[:2], lines[2:]
    else:
        head_rows, rows = [], lines
    pieces, current = [], []
    for row in rows:
        if not _fits(header, "\n".join(head_rows + [row])):
            if current:                                   # finish the piece in progress first
                pieces.append("\n".join(head_rows + current))
                current = []
            pieces.extend(_split_text(header, row))       # then cut the oversized row
            continue
        candidate = "\n".join(head_rows + current + [row])
        if current and not _fits(header, candidate):
            pieces.append("\n".join(head_rows + current))
            current = [row]
        else:
            current.append(row)
    if current:
        pieces.append("\n".join(head_rows + current))
    return pieces


def _overlap_tail(body):
    """The last few words of a finished chunk, started at a clean boundary,
    to be repeated at the top of the next chunk of the same section."""
    size = config.CHUNK_OVERLAP_CHARS
    if size <= 0 or len(body) <= size:
        return ""
    tail = body[-size:]
    match = re.search(r"[.!?]\s+|\n", tail)
    if match and match.end() < len(tail) - 20:
        return tail[match.end():].strip()
    space = tail.find(" ")
    return tail[space + 1:].strip() if space != -1 else ""


# ---------------------------------------------------------------------------
# Step 3. Fill chunks
# ---------------------------------------------------------------------------
class _Packer:
    """Collects pieces into chunks, one section at a time."""

    def __init__(self):
        self.chunks = []           # finished chunks, each {"section", "page", "body"}
        self.section = ""
        self.parts = []            # pieces of the chunk being built
        self.page = 0
        self.has_content = False   # False while parts holds nothing but an overlap
        self.carry = ""            # overlap text waiting for the next chunk

    def start_section(self, section):
        if section != self.section:
            self.close(overlap=False)          # a new section always starts a new chunk
            self.section = section

    def close(self, overlap):
        carry = ""
        if self.has_content:
            body = "\n\n".join(self.parts)
            self.chunks.append({"section": self.section, "page": self.page, "body": body})
            if overlap:
                carry = _overlap_tail(body)
        self.carry = carry
        self.parts, self.has_content = [], False

    def add(self, piece, page, overlap=True):
        if self.parts and not _fits(self.section, "\n\n".join(self.parts + [piece])):
            self.close(overlap=overlap)
            if self.carry and _fits(self.section, self.carry + "\n\n" + piece):
                self.parts = [self.carry]
            self.carry = ""
        if not self.has_content:
            self.page = page
            self.has_content = True
        self.parts.append(piece)


def _chunks_for_source(records):
    packer = _Packer()
    for unit in _units_for_source(records):
        packer.start_section(unit["section"])
        header = unit["section"]

        if unit["kind"] == "table":
            joined = "\n\n".join(packer.parts + [unit["text"]])
            if _fits(header, joined):
                packer.add(unit["text"], unit["page"])        # small table joins the chunk
            else:
                packer.close(overlap=False)
                if _fits(header, unit["text"]):
                    pieces = [unit["text"]]
                else:
                    pieces = _split_table(header, unit["text"])
                for piece in pieces:                           # each table piece stands alone
                    packer.add(piece, unit["page"])
                    packer.close(overlap=False)
        else:
            if _fits(header, unit["text"]):
                pieces = [unit["text"]]
            else:
                pieces = _split_text(header, unit["text"])
            for piece in pieces:
                packer.add(piece, unit["page"])
    packer.close(overlap=False)
    return packer.chunks


# ---------------------------------------------------------------------------
# Public function used by the rest of the project
# ---------------------------------------------------------------------------
def chunk_documents(pages, verbose=True):
    """Turn page records into chunk records."""
    by_source = {}
    for record in sorted(pages, key=lambda r: (r["source"], r["page"])):
        by_source.setdefault(record["source"], []).append(record)

    used_slugs, chunks = set(), []
    for source, records in by_source.items():
        prefix = _slug(source, used_slugs)
        for number, item in enumerate(_chunks_for_source(records), start=1):
            chunks.append({
                "id": f"{prefix}_{number:03d}",
                "text": _with_header(item["section"], item["body"]),
                "source": source,
                "page": item["page"],
                "section": item["section"],
            })

    if verbose:
        _print_stats(chunks)
    return chunks


def _print_stats(chunks):
    if not chunks:
        print("No chunks were produced.")
        return
    lengths = sorted(len(c["text"]) for c in chunks)
    tokens = [count_tokens(c["text"]) for c in chunks]
    too_long = sum(1 for t in tokens if t > config.EMBEDDING_MAX_TOKENS)
    tiny = sum(1 for n in lengths if n < config.MIN_CHUNK_CHARS)
    print(f"Produced {len(chunks)} chunks from {len({c['source'] for c in chunks})} files")
    print(f"  characters  shortest {lengths[0]}, median {lengths[len(lengths) // 2]}, longest {lengths[-1]}")
    print(f"  tokens      average {sum(tokens) // len(tokens)}, largest {max(tokens)}, model limit {config.EMBEDDING_MAX_TOKENS}")
    print(f"  chunks over the token limit {too_long}, chunks under {config.MIN_CHUNK_CHARS} characters {tiny}")
    if too_long:
        print("  WARNING some chunks exceed the token limit. Run  python -m indexing.chunking  to list them.")


# ---------------------------------------------------------------------------
# Self test. Run with  python -m indexing.chunking
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from indexing.loaders import load_documents

    all_chunks = chunk_documents(load_documents(verbose=False))

    def show(label, test, limit=2):
        print(f"\n--- {label} ---")
        shown = 0
        for chunk in all_chunks:
            if test(chunk):
                print(f"[{chunk['id']}  page {chunk['page']}]")
                print(chunk["text"])
                print()
                shown += 1
                if shown >= limit:
                    break

    limit_tokens = config.EMBEDDING_MAX_TOKENS
    oversize = [c for c in all_chunks if count_tokens(c["text"]) > limit_tokens]
    tiny = [c for c in all_chunks if len(c["text"]) < config.MIN_CHUNK_CHARS]
    print(f"\n--- {len(oversize)} chunks over the token limit (this must be 0) ---")
    for c in oversize[:10]:
        print(f"  {c['id']}  {len(c['text'])} characters, {count_tokens(c['text'])} tokens")
    print(f"\n--- {len(tiny)} chunks under {config.MIN_CHUNK_CHARS} characters ---")
    for c in tiny[:15]:
        print(f"  {c['id']}  {c['text'][:90]!r}")

    show("the 2026 annual leave chunk (the section path must mention 2026)",
         lambda c: c["source"].startswith("brightline_05") and "Annual leave" in c["section"], limit=1)
    show("the product catalog table (every piece must start with the column names)",
         lambda c: c["source"].startswith("brightline_06") and "ITM-" in c["text"], limit=2)