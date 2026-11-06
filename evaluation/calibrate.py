# evaluation/calibrate.py
# Helps you choose the confidence threshold from your own evaluation data.
#
#   python -m evaluation.calibrate rewrite
#
# It reads  evaluation/results/retrieval_<name>.csv , which  run_eval retrieval  writes.
# Each question has a "top score", the score of the best chunk found. Questions with an
# answer in the documents should score high and questions without one should score low.
# A threshold turns that into a rule. When the best chunk scores below it, the system
# declines to answer and the language model is never called.
#
# No threshold is perfect, because the two groups can overlap. The table shows the trade.
#   kept      share of answerable questions that score at or above the threshold (these still get answered)
#   declined  share of unanswerable questions that score below it (these are correctly refused)
#
# After the table, the script names the exact questions that the suggested threshold would
# wrongly decline and the ones it would let through, so you can judge the cost yourself.
#
# Honest warning. The threshold is chosen using the same questions it is then scored on,
# so it will look a little better here than it would on brand new questions. Say so in the report.

import csv
import sys

import config


def load_scores(name):
    path = config.EVAL_RESULTS_DIR / f"retrieval_{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run  python -m evaluation.run_eval retrieval --config {name}  first.")
    with open(path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    # older files have no score_kind column, so fall back to guessing from the name
    kind = rows[0].get("score_kind") or ("rerank" if ("rerank" in name or "final" in name or "rewrite" in name) else "vector")
    answerable = [(r["id"], float(r["top_score"])) for r in rows if r["answerable"] == "True"]
    unanswerable = [(r["id"], float(r["top_score"])) for r in rows if r["answerable"] != "True"]
    return kind, answerable, unanswerable


def rates(threshold, answerable, unanswerable):
    kept = sum(1 for _, s in answerable if s >= threshold) / len(answerable)
    declined = sum(1 for _, s in unanswerable if s < threshold) / len(unanswerable)
    return kept, declined


def main(name):
    kind, answerable, unanswerable = load_scores(name)
    if not answerable or not unanswerable:
        raise SystemExit("Need both answerable and unanswerable questions in the results.")
    setting = "MIN_RERANK_SCORE" if kind == "rerank" else "MIN_VECTOR_SIMILARITY"

    scores_a = sorted(s for _, s in answerable)
    scores_u = sorted(s for _, s in unanswerable)
    everything = sorted(set(scores_a + scores_u))
    if len(everything) < 2:
        raise SystemExit("Every question has the same top score, so no threshold can separate them. "
                         "Check that the retrieval file was produced by a normal run.")
    low, high = everything[0], everything[-1]
    print(f"Configuration {name}, {kind} scores from {low:.3f} to {high:.3f}.")
    print(f"  answerable questions    {len(scores_a)}, middle score {scores_a[len(scores_a) // 2]:.3f}")
    print(f"  unanswerable questions  {len(scores_u)}, middle score {scores_u[len(scores_u) // 2]:.3f}\n")

    print("  threshold    kept  declined")
    steps = 12
    for i in range(steps + 1):
        threshold = low + (high - low) * i / steps
        kept, declined = rates(threshold, answerable, unanswerable)
        print(f"  {threshold:9.3f}  {kept:6.0%}  {declined:7.0%}")

    # Every cut between two neighbouring scores is a possible threshold.
    cuts = [(a + b) / 2 for a, b in zip(everything, everything[1:])]
    best = max(cuts, key=lambda t: sum(rates(t, answerable, unanswerable)))
    careful = [t for t in cuts if rates(t, answerable, unanswerable)[0] >= 0.90]
    safest = max(careful, key=lambda t: rates(t, answerable, unanswerable)[1]) if careful else None

    kept, declined = rates(best, answerable, unanswerable)
    print(f"\nBest balance          {best:.3f}  keeps {kept:.0%} of answerable, declines {declined:.0%} of unanswerable")
    if safest is not None:
        kept, declined = rates(safest, answerable, unanswerable)
        print(f"Keeps at least 90%    {safest:.3f}  keeps {kept:.0%} of answerable, declines {declined:.0%} of unanswerable")
    chosen = safest if safest is not None else best

    wrongly_declined = sorted(((s, i) for i, s in answerable if s < chosen))
    let_through = sorted(((s, i) for i, s in unanswerable if s >= chosen), reverse=True)
    print(f"\nAt {chosen:.2f}")
    print("  answerable questions that would be declined   ",
          ", ".join(f"{i} ({s:.2f})" for s, i in wrongly_declined) or "none")
    print("  unanswerable questions that would get through ",
          ", ".join(f"{i} ({s:.2f})" for s, i in let_through) or "none")

    print(f"\nTo use this threshold, put this line in config.py and replace the old {setting} line.")
    print(f"{setting} = {chosen:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rewrite")