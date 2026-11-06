# evaluation/metrics.py
# The scoring rules and the summary tables. No model is called in this file.
# run_eval.py does the running, this file does the judging.
#
# The columns of evaluation/questions.csv
#   id, question, type
#   answerable        yes when the documents contain the answer, no otherwise
#   expected_answer   for you to read when reviewing, not used by the code
#   gold_source       file that holds the answer. Several files are separated by ||
#   gold_phrase       text that must appear in a retrieved chunk of that file for
#                     retrieval to count as a hit. Several phrases are separated by ||
#   must_include      words or numbers that a correct answer must contain, separated by ||
#
# What is measured, and how
#   Retrieval   the smallest k so that the top k chunks contain every gold phrase
#               (hit at 1, 3, 5 and 20, and the mean reciprocal rank, MRR)
#   Answers     automatic check that must_include appears in the answer
#   Refusals    for questions with no answer in the documents, did the system decline
#   Figures     numbers and codes in the answer that do not appear in the sources.
#               This is a cheap warning sign for invented facts, not a proof.
#               A number the model calculated itself, such as a price difference,
#               is also flagged, so read the flagged ones by hand.

import csv
import re
import unicodedata
from collections import defaultdict

import config

HIT_KS = (1, 3, 5)


# ---------------------------------------------------------------------------
# Reading the questions
# ---------------------------------------------------------------------------
def load_questions(path=None):
    path = path or config.EVAL_QUESTIONS_PATH
    with open(path, newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if (row.get("id") or "").strip() and (row.get("question") or "").strip()]
    for row in rows:
        gold_pairs(row)          # fails early, with the question id, if gold columns do not line up
    return rows


def is_answerable(row):
    return str(row.get("answerable", "")).strip().lower() in ("yes", "y", "true", "1")


def split_list(value):
    return [part.strip() for part in str(value or "").split("||") if part.strip()]


def normalize(text):
    """Lowercase, plain hyphens and apostrophes, single spaces. Used before every comparison."""
    text = unicodedata.normalize("NFKC", str(text))
    for character in "\u2010\u2011\u2012\u2212":
        text = text.replace(character, "-")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", text).lower().strip()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def gold_pairs(row):
    """The (file, phrase) pairs a retrieval result must contain for this question."""
    sources = split_list(row.get("gold_source"))
    phrases = split_list(row.get("gold_phrase"))
    if len(sources) == 1 and len(phrases) > 1:
        sources = sources * len(phrases)
    if len(sources) != len(phrases):
        raise ValueError(f"Question {row.get('id')} has {len(sources)} gold sources but {len(phrases)} gold phrases.")
    return [(source, normalize(phrase)) for source, phrase in zip(sources, phrases)]


def rank_needed(hits, row):
    """The smallest k so that hits[:k] contain every gold pair, or None when some pair is missing."""
    pairs = gold_pairs(row)
    if not pairs:
        return None
    found = []
    for source, phrase in pairs:
        rank = next((i for i, hit in enumerate(hits, start=1)
                     if hit["source"] == source and phrase in normalize(hit["text"])), None)
        if rank is None:
            return None
        found.append(rank)
    return max(found)


def summarize_retrieval(rows):
    """rows are dictionaries with keys type, answerable, rank (a number or None).
    Prints hit rates for answerable questions."""
    scored = [r for r in rows if r["answerable"] and r["gold"]]
    if not scored:
        print("  no answerable questions with gold columns to score")
        return

    def rate(subset, k):
        return sum(1 for r in subset if r["rank"] is not None and r["rank"] <= k) / len(subset)

    def line(label, subset):
        found = sum(1 for r in subset if r["rank"] is not None)
        mrr = sum(1 / r["rank"] for r in subset if r["rank"]) / len(subset)
        cells = "  ".join(f"{rate(subset, k):5.0%}" for k in HIT_KS)
        print(f"  {label:14} {len(subset):>3}  {cells}  {found / len(subset):5.0%}  {mrr:5.2f}")

    print(f"  {'':14} {'n':>3}  " + "  ".join(f"hit@{k}" for k in HIT_KS) + "  hit@20    MRR")
    line("all", scored)
    by_type = defaultdict(list)
    for r in scored:
        by_type[r["type"]].append(r)
    for name, subset in sorted(by_type.items()):
        line(name, subset)
    misses = [r["id"] for r in scored if r["rank"] is None or r["rank"] > 5]
    print("  not in the top 5 ->", ", ".join(misses) if misses else "none")


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------
def loose(text):
    """normalize() plus hyphens treated as spaces, so home-office and home office match."""
    return re.sub(r"\s+", " ", normalize(text).replace("-", " ")).strip()


def must_include_ok(answer, row):
    """True or False when the row has must_include terms, None when it has none."""
    terms = split_list(row.get("must_include"))
    if not terms:
        return None
    text = loose(answer)
    for term in terms:
        # A single | inside a term separates acceptable wordings, for example  1 week|one week
        options = [loose(option) for option in term.split("|") if option.strip()]
        pattern = "|".join(r"(?<![\w.])" + re.escape(option) + r"(?!\w)" for option in options)
        if not re.search(pattern, text):
            return False
    return True


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CODE = re.compile(r"[A-Za-z]{2,5}-[A-Za-z]?\d[\w-]*")
_CITATION = re.compile(r"\[\s*\d+\s*\]")


def unsupported_figures(answer, sources):
    """Numbers and codes that appear in the answer but in none of the source chunks."""
    text = normalize(_CITATION.sub(" ", answer))
    support = normalize(" ".join(hit["text"] for hit in sources))
    support_plain = support.replace(",", "")
    missing = []
    for token in _CODE.findall(text) + _NUMBER.findall(text):
        cleaned = token.strip(".,")
        if cleaned and cleaned not in support and cleaned.replace(",", "") not in support_plain:
            missing.append(cleaned)
    return sorted(set(missing))


_DECLINE = re.compile(
    r"(do not|don't|cannot|can't|unable to|no) (have|know|access|find|provide|see)"
    r"|not (sure|aware|provided|available|specified)|i lack|no (specific )?information",
    re.IGNORECASE,
)


def baseline_declined(answer):
    """A rough guess that the baseline model admitted it did not know. Check these by hand."""
    return bool(_DECLINE.search(normalize(answer)))


def summarize_answers(rows, title):
    """rows are the dictionaries saved in an answers file. Prints the headline numbers."""
    print(f"\n{title}  ({len(rows)} questions)")
    if not rows:
        return

    answerable = [r for r in rows if r["answerable"] == "yes"]
    unanswerable = [r for r in rows if r["answerable"] != "yes"]

    checked = [r for r in answerable if r["auto_correct"] in ("yes", "no")]
    correct = sum(1 for r in checked if r["auto_correct"] == "yes")
    print(f"  Questions with an answer in the documents  {len(answerable)}")
    if checked:
        print(f"    automatic check passed   {correct}/{len(checked)}  ({correct / len(checked):.0%})")
    wrong_refusals = [r["id"] for r in answerable if r["refused"] == "yes"]
    print(f"    wrongly declined         {len(wrong_refusals)}  {', '.join(wrong_refusals)}")
    failed = [r["id"] for r in checked if r["auto_correct"] == "no"]
    print(f"    failed the check         {', '.join(failed) if failed else 'none'}")

    print(f"  Questions with no answer in the documents  {len(unanswerable)}")
    refused = [r for r in unanswerable if r["refused"] == "yes"]
    print(f"    declined correctly       {len(refused)}/{len(unanswerable)}")
    invented = [r["id"] for r in unanswerable if r["refused"] != "yes"]
    print(f"    answered anyway          {', '.join(invented) if invented else 'none'}")

    flagged = [r["id"] for r in rows if r["unsupported"]]
    print(f"  Answers with figures missing from the sources  {', '.join(flagged) if flagged else 'none'}")

        # Final verdicts. Your label wins when you gave one, otherwise the automatic result is used.
    def final_correct(row):
        label = (row.get("human_correct") or "").strip().lower()
        if label in ("y", "yes"):
            return True
        if label in ("n", "no"):
            return False
        if row["answerable"] == "yes":
            return {"yes": True, "no": False}.get(row["auto_correct"])
        return row["refused"] == "yes"            # no answer exists, so declining is correct

    verdicts = [(r, final_correct(r)) for r in rows]
    for kind, label in (("yes", "Final, answerable questions answered correctly"),
                        ("no", "Final, unanswerable questions declined correctly")):
        pool = [v for r, v in verdicts if r["answerable"] == kind and v is not None]
        if pool:
            print(f"  {label}  {sum(pool)}/{len(pool)}  ({sum(pool) / len(pool):.0%})")

    reviewed = [r for r in rows if (r.get("human_correct") or "").strip()]
    if reviewed:
        good = sum(1 for r in reviewed if r["human_correct"].strip().lower() in ("y", "yes"))
        print(f"  Rows you reviewed, labelled correct   {good}/{len(reviewed)}")
    grounded = [r for r in rows if (r.get("human_grounded") or "").strip()]
    if grounded:
        good = sum(1 for r in grounded if r["human_grounded"].strip().lower() in ("y", "yes"))
        print(f"  Rows you reviewed, labelled grounded  {good}/{len(grounded)}")

    tokens = sum(int(r["tokens"] or 0) for r in rows)
    print(f"  Tokens used  {tokens:,}")