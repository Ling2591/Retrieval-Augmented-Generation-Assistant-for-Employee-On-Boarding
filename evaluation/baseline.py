# evaluation/baseline.py
# The comparison point for the report. It sends the question to the SAME model
# with NO documents at all, so any difference in the results comes from retrieval.
#
# Run a single question with  python -m evaluation.baseline "How many days of annual leave do I get?"

from querying import prompt
from querying.generate import generate
from querying.pipeline import tidy_answer


def run_baseline(question):
    """Return the same kind of dictionary generate() returns, with the text tidied."""
    result = generate(prompt.build_baseline_messages(question))
    result["text"] = tidy_answer(result["text"])
    return result


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "How many days of annual leave do I get?"
    outcome = run_baseline(question)
    print("Question", question)
    print("\n" + outcome["text"] + "\n")
    print("Tokens", outcome["total_tokens"])