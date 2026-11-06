# indexing/loaders.py
# PHASE 1, step 1. Reads every file in data/raw and returns clean text.
#
# Output is a list of "page records", each a dictionary with three fields.
#   source   the file name, for example "brightline_02_it_setup_guide.md"
#   page     the page number for PDFs, or 0 for files that have no pages
#   text     the cleaned text
# (Page 0 instead of None because the vector database rejects empty values.)
#
# The next file (chunking.py) takes this list and cuts it into chunks.

import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

import config


# ---------------------------------------------------------------------------
# Cleaning that applies to every file type
# ---------------------------------------------------------------------------
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _base_clean(text):
    """Normalise characters and whitespace without touching the content."""
    text = unicodedata.normalize("NFKC", text)        # fixes ligatures and odd spaces
    text = _CONTROL_CHARS.sub("", text)               # removes invisible control characters
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        line = re.sub(r"(?<=\S)[ \t]{2,}", " ", line)  # collapse runs of spaces inside a line
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)            # at most one blank line in a row
    return text.strip()


# ---------------------------------------------------------------------------
# Markdown and plain text
# ---------------------------------------------------------------------------
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")


def _clean_markdown(text):
    """Remove markdown noise that would only confuse the embedding model.
    Headings and tables are kept, because chunking and search use them."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")

    # Some files start with a metadata block between two --- lines. Drop it,
    # but remember the title so the text still starts with a clear heading.
    title = None
    match = _FRONT_MATTER.match(text)
    if match:
        for line in match.group(1).splitlines():
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip().strip("\"'")
                break
        text = text[match.end():]

    text = _HTML_COMMENT.sub("", text)
    text = _MD_IMAGE.sub(r"\1", text)   # keep the alt text, drop the image address
    text = _MD_LINK.sub(r"\1", text)    # keep the link words, drop the web address
    text = _MD_BOLD.sub(r"\1", text)    # drop the ** bold markers, keep the words
    text = _HTML_TAG.sub("", text)
    text = _base_clean(text)

    if title and not re.match(r"#\s", text):
        text = f"# {title}\n\n{text}"
    return text


def _load_text_file(path, source, is_markdown):
    text = path.read_text(encoding="utf-8", errors="replace")
    text = _clean_markdown(text) if is_markdown else _base_clean(text.lstrip("\ufeff"))
    if len(text) < config.MIN_PAGE_CHARS:
        return []
    return [{"source": source, "page": 0, "text": text}]


# ---------------------------------------------------------------------------
# PDF
# Web pages saved as PDF carry a lot of "page furniture" that is not content.
# Two passes remove it.
#   Pass 1 (one file at a time)  lines repeated on most pages of that file,
#                                for example the browser header and page number.
#   Pass 2 (across all PDFs)     lines found in several different PDFs, for
#                                example a website menu printed at the top.
# ---------------------------------------------------------------------------
def _normalize_for_repeat(line):
    """Lowercase and replace digits, so '1/7' and '2/7' count as the same line."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _strip_repeated_lines(page_texts):
    """Pass 1. Remove short lines that appear on most pages of one PDF."""
    if len(page_texts) < 3:
        return page_texts

    counts = Counter()
    for text in page_texts:
        # a set, so a line is counted at most once per page
        counts.update({_normalize_for_repeat(line) for line in text.splitlines() if line.strip()})

    threshold = max(3, int(config.REPEATED_LINE_FRACTION * len(page_texts)))
    repeated = {
        line for line, seen in counts.items()
        if seen >= threshold and len(line) <= config.REPEATED_LINE_MAX_CHARS
    }

    cleaned = []
    for text in page_texts:
        kept = [line for line in text.splitlines() if _normalize_for_repeat(line) not in repeated]
        cleaned.append("\n".join(kept))
    return cleaned


def _strip_cross_file_lines(pdf_pages):
    """Pass 2. pdf_pages maps a file name to its list of page texts.
    Remove short lines that appear in several different PDFs. Only lines with
    at least three letters count, so numbered lists are never mistaken for menus."""
    if len(pdf_pages) < config.CROSS_FILE_MIN_FILES:
        return pdf_pages

    counts = Counter()
    for page_texts in pdf_pages.values():
        lines = {
            _normalize_for_repeat(line)
            for text in page_texts
            for line in text.splitlines()
            if line.strip()
        }
        counts.update(lines)    # each line is counted once per file

    threshold = max(config.CROSS_FILE_MIN_FILES, int(config.CROSS_FILE_FRACTION * len(pdf_pages)))
    boilerplate = {
        line for line, files in counts.items()
        if files >= threshold
        and len(line) <= config.REPEATED_LINE_MAX_CHARS
        and len(re.sub(r"[^a-z]", "", line)) >= 3
    }

    cleaned = {}
    for source, page_texts in pdf_pages.items():
        cleaned[source] = [
            "\n".join(line for line in text.splitlines() if _normalize_for_repeat(line) not in boilerplate)
            for text in page_texts
        ]
    return cleaned


def _join_wrapped_lines(text):
    """PDFs break lines in the middle of sentences. Glue a line onto the previous
    one when the previous line has no sentence ending and this one starts in
    lowercase. A hyphen at the end of the previous line is treated as a word break,
    which is occasionally wrong for real compound words but is right far more often."""
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        previous = out[-1] if out else ""
        ends_sentence = re.search(r"[.!?:;][\"')\]]?$", previous)
        if previous and not ends_sentence and stripped[0].islower():
            if previous.endswith("-"):
                out[-1] = previous[:-1] + stripped
            else:
                out[-1] = previous + " " + stripped
        else:
            out.append(stripped)
    return "\n".join(out)


def _title_from_source(source):
    """Turn a file name into a readable title, used as the heading of a PDF."""
    return re.sub(r"[_\s]+", " ", Path(source).stem).strip()


def _extract_pdf_pages(path):
    reader = PdfReader(str(path))
    page_texts = [_base_clean(page.extract_text() or "") for page in reader.pages]
    return _strip_repeated_lines(page_texts)


def _pages_to_records(source, page_texts):
    records = []
    for number, text in enumerate(page_texts, start=1):
        text = _base_clean(_join_wrapped_lines(text))
        if len(text) >= config.MIN_PAGE_CHARS:   # skips blank pages and image-only pages
            records.append({"source": source, "page": number, "text": text})
    if records:
        # Web menus are gone, so give the document a title taken from its file name.
        # The chunker turns this heading into the section label of every chunk.
        records[0]["text"] = f"# {_title_from_source(source)}\n\n{records[0]['text']}"
    return records


# ---------------------------------------------------------------------------
# Word documents (not exercised by your current files, but supported for later)
# ---------------------------------------------------------------------------
def _load_docx(path, source):
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    parts = []

    # Walk the document body in order so tables stay where they belong.
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else ""
            if style == "Title":
                text = "# " + text
            elif style.startswith("Heading"):
                digits = "".join(ch for ch in style if ch.isdigit())
                level = int(digits) if digits else 1
                text = "#" * min(level, 6) + " " + text
            parts.append(text)
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
            parts.append("\n".join(rows))

    text = _base_clean("\n\n".join(parts))
    if len(text) < config.MIN_PAGE_CHARS:
        return []
    return [{"source": source, "page": 0, "text": text}]


# ---------------------------------------------------------------------------
# Public function used by the rest of the project
# ---------------------------------------------------------------------------
def load_documents(raw_dir=None, verbose=True):
    """Read every supported file under raw_dir and return a list of page records."""
    raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
    if not raw_dir.exists():
        raise FileNotFoundError(f"Folder not found. {raw_dir}")

    everything = sorted(p for p in raw_dir.rglob("*") if p.is_file())
    files = [p for p in everything if p.suffix.lower() in config.SUPPORTED_EXTENSIONS]
    ignored = len(everything) - len(files)

    pages, skipped = [], []
    pdf_pages = {}                      # file name -> page texts after pass 1

    for path in files:
        source = path.relative_to(raw_dir).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                pdf_pages[source] = _extract_pdf_pages(path)
                continue                # PDFs are finished below, after pass 2
            if suffix == ".docx":
                records = _load_docx(path, source)
            else:
                records = _load_text_file(path, source, is_markdown=(suffix == ".md"))
        except Exception as error:      # one bad file must not stop the whole run
            skipped.append((source, f"error, {error}"))
            continue

        if not records:
            skipped.append((source, "no usable text"))
            continue
        pages.extend(records)

    # PDFs, second pass and conversion to page records
    pdf_report = []
    for source, page_texts in _strip_cross_file_lines(pdf_pages).items():
        records = _pages_to_records(source, page_texts)
        if not records:
            skipped.append((source, "no usable text"))
            continue
        pages.extend(records)
        pdf_report.append((source, len(page_texts), len(records), sum(len(r["text"]) for r in records)))

    pages.sort(key=lambda r: (r["source"], r["page"]))   # same order on every run

    if verbose:
        total_chars = sum(len(r["text"]) for r in pages)
        loaded_files = len({r["source"] for r in pages})
        print(f"Read {loaded_files} files from {raw_dir}")
        print(f"Produced {len(pages)} page records, {total_chars:,} characters after cleaning")
        print(f"Ignored {ignored} files with unsupported types, skipped {len(skipped)} files")
        for source, reason in skipped:
            print(f"  skipped {source} ({reason})")
        if pdf_report:
            print("PDF pages (pages in file, pages kept, characters kept)")
            for source, total, kept, chars in pdf_report:
                print(f"  {source[:58]:58} {total:>3} {kept:>3} {chars:>7}")
    return pages


# ---------------------------------------------------------------------------
# Self test. Run with  python -m indexing.loaders
# Writes a readable copy of every cleaned file to data/cleaned_preview so you
# can open them in VS Code and check the result with your own eyes.
# ---------------------------------------------------------------------------
def _write_previews(pages):
    config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old in config.PREVIEW_DIR.glob("*.txt"):
        old.unlink()

    by_source = {}
    for record in pages:
        by_source.setdefault(record["source"], []).append(record)

    for source, records in by_source.items():
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source) + ".txt"
        blocks = []
        for record in records:
            label = f"[page {record['page']}]" if record["page"] else "[text]"
            blocks.append(label + "\n" + record["text"])
        (config.PREVIEW_DIR / safe_name).write_text("\n\n".join(blocks), encoding="utf-8")
    return len(by_source)


if __name__ == "__main__":
    all_pages = load_documents()
    written = _write_previews(all_pages)
    print(f"\nWrote {written} preview files to {config.PREVIEW_DIR}")

    samples = [
        ("first PDF page", lambda r: r["source"].lower().endswith(".pdf")),
        ("first Dwarves file", lambda r: r["source"].startswith("dwarves_")),
    ]
    for label, matches in samples:
        record = next((r for r in all_pages if matches(r)), None)
        if record:
            print(f"\n--- {label} ({record['source']}) ---")
            print(record["text"][:700])