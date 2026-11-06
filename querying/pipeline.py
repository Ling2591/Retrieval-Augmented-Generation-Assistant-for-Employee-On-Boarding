# querying/pipeline.py
# PHASE 2, the conductor. answer_question() runs every step in order and returns
# one dictionary that the Streamlit app (Part 7) and the evaluation (Part 5) both use.
#
#   1. validate      reject empty or oversized questions            guardrails.py
#   2. rewrite       make a vague follow-up standalone  (Part 7)    rewrite.py
#   3. retrieve      find candidate chunks                          retrieve.py
#   4. rerank        reorder candidates with a better model (Part 6) rerank.py
#   5. confidence    decline to answer when the chunks are weak     guardrails.py
#   6. prompt        build the messages                             prompt.py
#   7. generate      ask the language model                         generate.py
#
# Every optional step is a switch in Options, off by default. The evaluation
# flips these switches to compare configurations. Files for steps 2 and 4 are
# only imported when their switch is on, so they can stay empty until their part.

import re
import time
from dataclasses import asdict, dataclass

import config
from querying import guardrails, prompt
from querying.generate import generate
from querying.retrieve import retrieve


@dataclass
class Options:
    use_rewrite: bool = False      # Part 7
    use_hybrid: bool = False       # Part 6
    use_rerank: bool = False       # Part 6
    use_threshold: bool = False    # Part 6
    company_only: bool = False     # Part 7, search only the company's own documents
    top_k: int = config.FINAL_TOP_K


# ---------------------------------------------------------------------------
# Two helpers that the evaluation (Part 5) reuses, so it tests exactly what the app runs.
# ---------------------------------------------------------------------------
# The model sometimes writes typographic characters that look right to a person but break
# exact matching, for example a non-breaking hyphen inside a code such as ITM-20417, or
# source numbers in fullwidth brackets. tidy_answer() turns them back into plain ASCII.
_HYPHENS = {ord(ch): "-" for ch in "\u2010\u2011\u2012\u2212"}
_SPACES = {ord(ch): " " for ch in "\u00a0\u202f\u2009"}


def tidy_answer(text):
    text = text.translate(_HYPHENS).translate(_SPACES)
    # gpt-oss sometimes writes citations such as 【1†L5-L7】. Keep only the source number.
    text = re.sub(r"\u3010\s*(\d+)[^\u3011]*\u3011", r"[\1]", text)
    return re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", text)


def find_chunks(search_query, options, timings=None):
    """The retrieval half of the pipeline, with no language model call, so the
    evaluation can measure retrieval quality for free. Returns candidates, best first."""
    timings = {} if timings is None else timings

    started = time.time()    
    candidates = retrieve(search_query, use_hybrid=options.use_hybrid, company_only=options.company_only)
    timings["retrieve"] = time.time() - started

    if options.use_rerank:
        from querying.rerank import rerank
        started = time.time()
        candidates = rerank(search_query, candidates)
        timings["rerank"] = time.time() - started
    return candidates


def answer_question(question, history=None, options=None):
    """Answer one question. history is a list of earlier messages shaped like
    {"role": "user" or "assistant", "content": "..."} and is only used by the rewrite step.

    The returned dictionary has these keys.
      status        answered, model_declined, low_confidence, no_context or invalid_question
      answer        the text to show the user
      sources       the chunks that were given to the model, best first
      candidates    every chunk retrieval found, for the evaluation
      search_query  the question that was actually searched (differs after a rewrite)
      llm           model name, token counts and seconds, or None when the model was not called
      timings       seconds spent in each step
    """
    options = options or Options()
    timings = {}
    result = {
        "question": question, "search_query": question, "status": "", "answer": "",
        "sources": [], "candidates": [], "llm": None, "rewrite_tokens": 0,
        "options": asdict(options), "timings": timings,
    }

    ok, cleaned, message = guardrails.validate_question(question)
    if not ok:
        result.update(status="invalid_question", answer=message)
        return result
    result["question"] = result["search_query"] = cleaned

    if options.use_rewrite:
        from querying.rewrite import rewrite_question
        started = time.time()
        try:
            result["search_query"], result["rewrite_tokens"] = rewrite_question(cleaned, history)
        except RuntimeError as error:              # carry on with the original wording
            result["rewrite_error"] = str(error)
        timings["rewrite"] = time.time() - started
    search_query = result["search_query"]

    candidates = find_chunks(search_query, options, timings)
    hits = candidates[:options.top_k]
    result["candidates"], result["sources"] = candidates, hits

    confident, status = guardrails.check_confidence(hits, options.use_threshold)
    if not confident:
        result.update(status=status, answer=guardrails.fallback_message(status, hits))
        return result

    started = time.time()
    llm = generate(prompt.build_messages(search_query, hits))
    timings["generate"] = time.time() - started

    llm["text"] = tidy_answer(llm["text"])
    result["llm"] = llm
    result["answer"] = llm["text"]
    result["status"] = "model_declined" if guardrails.is_refusal(llm["text"]) else "answered"
    return result


# ---------------------------------------------------------------------------
# Self test. Run with  python -m querying.pipeline
# or with your own question  python -m querying.pipeline "What is the expense limit?"
# ---------------------------------------------------------------------------
DEFAULT_QUESTIONS = [
    "How many days of annual leave do I get?",
    "What is the price of item ITM-20417?",
    "What does VPN error 612 mean?",
    "What is the name of the company's founder?",     # not in the documents on purpose
]


def _print_result(result):
    print("=" * 78)
    print("Question", result["question"])
    print("Status  ", result["status"])
    print("\n" + result["answer"] + "\n")
    print("Sources given to the model")
    for number, hit in enumerate(result["sources"], start=1):
        print(f"  [{number}] {hit['vector_similarity']:.3f}  {hit['source']}  >  {hit['section']}")
    seconds = "  ".join(f"{name} {value:.1f}s" for name, value in result["timings"].items())
    print("Timings ", seconds)
    if result["llm"]:
        print("Tokens  ", result["llm"]["total_tokens"], "used by", result["llm"]["model"])


if __name__ == "__main__":
    import sys

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEFAULT_QUESTIONS
    for index, text in enumerate(questions):
        if index:
            time.sleep(3)          # stay well inside the free per-minute limits
        _print_result(answer_question(text))