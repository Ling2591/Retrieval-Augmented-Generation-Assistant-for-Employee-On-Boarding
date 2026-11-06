# querying/rewrite.py
# PHASE 2, step 1. Turns the user's message into a clear, standalone question
# before anything is searched.
#
# Two problems it solves.
#   Casual wording    "how much time off do i get" never says "annual leave", and
#                     "paid back for my train ticket" never says "expense claim".
#                     Search compares words and meaning, so the question is rewritten
#                     in the words that company documents use.
#   Follow-ups        "what about the second one?" means nothing alone. The earlier
#                     messages of the conversation are used to fill in what it refers to.
#
# The rewritten question is used for searching AND for the final answer, so it must stay
# a faithful, natural question. The rules below forbid answering, guessing or adding facts.
# It costs one small model call per question, about 250 to 400 tokens.

import config
from querying.generate import generate

REWRITE_PROMPT = """You prepare questions for a search over a company knowledge base that holds HR, IT, finance and warehouse documents.

Rewrite the latest user message as one standalone question that a document search can match.
1. Use the earlier conversation to resolve words such as it, that and the second one.
2. When the message uses casual or vague wording, use the formal term a policy document would use. For example write annual leave when the user says time off, and write expense claim when the user says paid back.
3. Never replace a precise word with a vaguer one. For example keep laptop, do not change it to computer.
4. Keep every code, number, name and unit exactly as written.
5. Do not answer the question and do not add facts.
6. If the message is already clear, return it unchanged.
7. Reply with the question only, on one line, with no quotation marks."""

MAX_QUERY_CHARS = 300


def format_history(history):
    """The last few messages as plain text. Long answers are shortened."""
    recent = [m for m in (history or []) if m.get("role") in ("user", "assistant")]
    recent = recent[-config.HISTORY_TURNS:] if config.HISTORY_TURNS else []
    return "\n".join(f"{m['role'].upper()} {str(m['content'])[:300]}" for m in recent)


def rewrite_question(question, history=None):
    """Return (rewritten_question, tokens_used). Raises RuntimeError when the model
    cannot be reached, so the caller decides whether to continue with the original."""
    from querying.pipeline import tidy_answer       # imported here to avoid a circular import

    conversation = format_history(history)
    user_text = (f"CONVERSATION\n{conversation}\n\n" if conversation else "") + f"LATEST MESSAGE\n{question}"
    result = generate(
        [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": user_text}],
        max_tokens=400,
    )
    lines = [line.strip() for line in tidy_answer(result["text"]).splitlines() if line.strip()]
    rewritten = lines[0].strip("\"' ") if lines else ""
    if not rewritten or len(rewritten) > MAX_QUERY_CHARS:
        rewritten = question                       # an empty or runaway reply is not used
    return rewritten, result["total_tokens"]


# ---------------------------------------------------------------------------
# Self test. Run with  python -m querying.rewrite "how much time off do i get"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    samples = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "how much time off do i get",
        "can i get paid back for my train ticket",
        "first day what do i do",
        "What is the price of item ITM-20417?",
    ]
    for text in samples:
        rewritten, tokens = rewrite_question(text)
        print(f"{text}\n  -> {rewritten}   ({tokens} tokens)")

    history = [
        {"role": "user", "content": "What is the price of ITM-20417?"},
        {"role": "assistant", "content": "The price of ITM-20417 is 96.50 USD."},
    ]
    follow_up = "and how long is the lead time for it?"
    rewritten, tokens = rewrite_question(follow_up, history)
    print(f"{follow_up}  (after a question about ITM-20417)\n  -> {rewritten}   ({tokens} tokens)")