# evaluation/run_eval.py
# Runs the questions in evaluation/questions.csv and saves what happened.
# It calls the same functions the app uses (find_chunks and answer_question),
# so you are testing exactly what ships.
#
#   python -m evaluation.run_eval retrieval
#       Free and instant, no language model, except for the configurations that
#       rewrite questions. Those make one small model call per question the first
#       time, and the rewrites are saved in results/rewrites.json for later runs.
#       Measures how well each available configuration finds the right chunks and
#       saves the top scores that calibrate.py uses to set the confidence threshold.
#
#   python -m evaluation.run_eval answers --baseline
#       Asks the model every question with NO documents.
#
#   python -m evaluation.run_eval answers --config final
#       Asks every question through the full pipeline.
#
#   python -m evaluation.run_eval report
#       Prints the summaries again from the saved files, no model calls.
#
# Extra options
#   --config NAME ...  which configurations to run (see CONFIGS below)
#   --only D01,C01     run just these question ids
#   --limit 5          run only the first 5 selected questions
#   --fresh            ignore saved answers and start over
#
# The free Groq plan allows roughly 200,000 tokens per day. Answers are saved
# after every question, so if a limit stops the run, start the same command
# again later and it continues where it stopped. Close the results files in
# Excel before running, because Excel locks them.

import argparse
import csv
import json
import statistics
import time

import config
from evaluation import metrics
from evaluation.baseline import run_baseline
from querying import guardrails
from querying.pipeline import Options, answer_question, find_chunks
from querying.rewrite import rewrite_question

CONFIGS = {
    "vector": Options(),
    "hybrid": Options(use_hybrid=True),
    "hybrid_rerank": Options(use_hybrid=True, use_rerank=True),
    "final": Options(use_hybrid=True, use_rerank=True, use_threshold=True),
    "rewrite": Options(use_hybrid=True, use_rerank=True, use_rewrite=True),
    "final_rewrite": Options(use_hybrid=True, use_rerank=True, use_rewrite=True, use_threshold=True),    
    "company": Options(use_hybrid=True, use_rerank=True, use_rewrite=True, company_only=True),
    "company_final": Options(use_hybrid=True, use_rerank=True, use_rewrite=True, company_only=True, use_threshold=True),
}

ANSWER_COLUMNS = [
    "id", "type", "answerable", "config", "question", "search_query", "status", "answer",
    "auto_correct", "refused", "unsupported", "gold_rank", "sources",
    "tokens", "seconds", "human_correct", "human_grounded", "notes",
]
REFUSAL_STATUSES = ("model_declined", "low_confidence", "no_context")
REWRITE_CACHE = config.EVAL_RESULTS_DIR / "rewrites.json"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def available_configs():
    """Only the configurations whose code exists. Files that are still empty are skipped."""
    usable = {}
    for name, options in CONFIGS.items():
        try:
            find_chunks("probe question", options)
            usable[name] = options
        except (NotImplementedError, ImportError, ValueError) as error:
            print(f"  skipping {name}, not available yet ({type(error).__name__})")
    return usable


def choose_questions(args):
    questions = metrics.load_questions()
    if args.only:
        wanted = {item.strip().upper() for item in args.only.split(",")}
        questions = [q for q in questions if q["id"].upper() in wanted]
    if args.limit:
        questions = questions[:args.limit]
    return questions


def answers_path(name):
    return config.EVAL_RESULTS_DIR / f"answers_{name}.csv"


def saved_ids(path, fresh):
    if fresh and path.exists():
        path.unlink()
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8-sig") as file:
        return {row["id"] for row in csv.DictReader(file)}


def append_row(path, columns, row):
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    # the first write adds a byte order mark so Excel shows accents and symbols correctly
    with open(path, "a", newline="", encoding="utf-8-sig" if is_new else "utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_answer_rows(path):
    """Read a saved answers file and score it again with the current rules in metrics.py,
    so an improved rule applies to old runs without calling the model again."""
    rows = read_rows(path)
    questions = {q["id"]: q for q in metrics.load_questions()}
    for row in rows:
        question = questions.get(row["id"])
        if question is None or row["answerable"] != "yes":
            continue
        check = metrics.must_include_ok(row["answer"], question)
        if check is not None and row["refused"] == "yes":
            check = False                       # declining is not a correct answer
        row["auto_correct"] = "" if check is None else ("yes" if check else "no")
    return rows


def describe(values):
    if not values:
        return "no values"
    return f"lowest {min(values):.3f}  median {statistics.median(values):.3f}  highest {max(values):.3f}"


def load_rewrites():
    if REWRITE_CACHE.exists():
        return json.loads(REWRITE_CACHE.read_text(encoding="utf-8"))
    return {}


def get_rewrite(text, cache):
    """The rewritten question, from the saved file when possible, otherwise from the model."""
    if text in cache:
        return cache[text]
    rewritten, tokens = rewrite_question(text, [])
    cache[text] = rewritten
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REWRITE_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    {text!r} -> {rewritten!r}  ({tokens} tokens)")
    time.sleep(config.EVAL_SECONDS_BETWEEN_CALLS)
    return rewritten


# ---------------------------------------------------------------------------
# Mode 1. Retrieval
# ---------------------------------------------------------------------------
def run_retrieval(questions, names):
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    usable = available_configs()
    cache = load_rewrites()
    for name in names or usable:
        if name not in usable:
            continue
        options = usable[name]
        rows, scored = [], []
        for question in questions:
            query = question["question"]
            if options.use_rewrite:
                try:
                    query = get_rewrite(query, cache)
                except RuntimeError as error:
                    print(f"\nStopped at {question['id']}. {error}")
                    print("The rewrites finished so far are saved. Run the same command again to continue.")
                    return
            candidates = find_chunks(query, options)
            answerable = metrics.is_answerable(question)
            has_gold = bool(metrics.gold_pairs(question))
            rank = metrics.rank_needed(candidates, question) if (answerable and has_gold) else None
            top = candidates[0] if candidates else {}
            rows.append({
                "id": question["id"], "type": question["type"], "answerable": answerable,
                "gold": has_gold, "rank": rank,
                "top_score": guardrails.confidence_of(top) if top else 0.0,
                "score_kind": "rerank" if "rerank_score" in top else "vector",
                "top_source": top.get("source", ""), "search_query": query,
            })
            scored.append({"id": question["id"], "type": question["type"], "answerable": answerable, "gold": has_gold, "rank": rank})

        with open(config.EVAL_RESULTS_DIR / f"retrieval_{name}.csv", "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nRetrieval, configuration {name}")
        metrics.summarize_retrieval(scored)
        print("  top score, answerable      ", describe([r["top_score"] for r in rows if r["answerable"]]))
        print("  top score, not answerable  ", describe([r["top_score"] for r in rows if not r["answerable"]]))


# ---------------------------------------------------------------------------
# Mode 2. Answers from the language model
# ---------------------------------------------------------------------------
def build_row(question, name, status, answer, sources, gold_rank, tokens, seconds, refused, search_query):
    answerable = metrics.is_answerable(question)
    check = metrics.must_include_ok(answer, question) if answerable else None
    if answerable and refused and check is not None:
        check = False                           # declining is not a correct answer
    figures = [] if refused or name == "baseline" else metrics.unsupported_figures(answer, sources)
    return {
        "id": question["id"], "type": question["type"], "answerable": "yes" if answerable else "no",
        "config": name, "question": question["question"], "search_query": search_query,
        "status": status, "answer": answer,
        "auto_correct": "" if check is None else ("yes" if check else "no"),
        "refused": "yes" if refused else "no",
        "unsupported": " ".join(figures),
        "gold_rank": "" if gold_rank is None else gold_rank,
        "sources": " | ".join(f"{h['source']} > {h['section']}" for h in sources),
        "tokens": tokens, "seconds": f"{seconds:.1f}",
        "human_correct": "", "human_grounded": "", "notes": "",
    }


def run_answers(questions, name, options, fresh):
    path = answers_path(name)
    done = saved_ids(path, fresh)
    todo = [q for q in questions if q["id"] not in done]
    print(f"\nAnswers, configuration {name}. {len(done)} already saved, {len(todo)} to run.")

    called_model = False
    for question in todo:
        if called_model:
            time.sleep(config.EVAL_SECONDS_BETWEEN_CALLS)
        started = time.time()
        try:
            if name == "baseline":
                llm = run_baseline(question["question"])
                answer, sources, status = llm["text"], [], "baseline"
                refused, candidates, search_query = metrics.baseline_declined(answer), [], ""
                tokens, called_model = llm["total_tokens"], True
            else:
                result = answer_question(question["question"], options=options)
                llm, answer, sources, status = result["llm"], result["answer"], result["sources"], result["status"]
                refused, candidates, search_query = status in REFUSAL_STATUSES, result["candidates"], result["search_query"]
                tokens = (llm["total_tokens"] if llm else 0) + result.get("rewrite_tokens", 0)
                called_model = tokens > 0
        except RuntimeError as error:
            print(f"\nStopped at {question['id']}. {error}")
            print("Everything finished so far is saved. Run the same command again later to continue.")
            return

        gold_rank = None
        if metrics.is_answerable(question) and metrics.gold_pairs(question):
            gold_rank = metrics.rank_needed(candidates, question) if candidates else None
        row = build_row(question, name, status, answer, sources, gold_rank, tokens,
                        time.time() - started, refused, search_query)
        append_row(path, ANSWER_COLUMNS, row)
        print(f"  {question['id']:4} {status:15} {answer[:120]!r}")

    metrics.summarize_answers(read_answer_rows(path), f"Summary for {name}")


# ---------------------------------------------------------------------------
# Mode 3. Report
# ---------------------------------------------------------------------------
def run_report():
    paths = sorted(config.EVAL_RESULTS_DIR.glob("answers_*.csv"))
    if not paths:
        print("No answer files yet. Run  python -m evaluation.run_eval answers --baseline  first.")
    for path in paths:
        metrics.summarize_answers(read_answer_rows(path), f"Summary for {path.stem.replace('answers_', '')}")

    for path in sorted(config.EVAL_RESULTS_DIR.glob("retrieval_*.csv")):
        rows = [{"id": r["id"], "type": r["type"], "answerable": r["answerable"] == "True",
                 "gold": r["gold"] == "True", "rank": int(r["rank"]) if r["rank"] else None}
                for r in read_rows(path)]
        print(f"\nRetrieval, configuration {path.stem.replace('retrieval_', '')}")
        metrics.summarize_retrieval(rows)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate the onboarding assistant.")
    parser.add_argument("mode", choices=["retrieval", "answers", "report"])
    parser.add_argument("--config", nargs="+", choices=list(CONFIGS), help="configuration names to run")
    parser.add_argument("--baseline", action="store_true", help="run the no-documents baseline")
    parser.add_argument("--only", help="comma separated question ids")
    parser.add_argument("--limit", type=int, help="only the first N selected questions")
    parser.add_argument("--fresh", action="store_true", help="start over instead of continuing")
    args = parser.parse_args()

    if args.mode == "report":
        run_report()
        return

    questions = choose_questions(args)
    if not questions:
        print("No questions selected. Check the ids given to --only.")
        return

    if args.mode == "retrieval":
        run_retrieval(questions, args.config)
        return

    if not args.baseline and not args.config:
        print("Say what to run, for example  --baseline  or  --config vector")
        return
    if args.baseline:
        run_answers(questions, "baseline", None, args.fresh)
    for name in args.config or []:
        run_answers(questions, name, CONFIGS[name], args.fresh)


if __name__ == "__main__":
    main()