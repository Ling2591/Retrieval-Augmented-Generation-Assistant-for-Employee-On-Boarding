# querying/guardrails.py
# PHASE 2, safety checks around the language model. Part 4 uses the simple
# version, Part 6 turns on the confidence thresholds after you calibrate them.
#
#   validate_question()   rejects empty or oversized input before any work is done
#   check_confidence()    decides whether the retrieved chunks are good enough to answer from
#   fallback_message()    the reply shown when we decline to call the model
#   is_refusal()          detects that the model itself said it does not know

import config


def validate_question(question):
    """Return (ok, cleaned_question, message_for_the_user)."""
    cleaned = (question or "").strip()
    if not cleaned:
        return False, cleaned, "Please type a question."
    if len(cleaned) > config.MAX_QUESTION_CHARS:
        return False, cleaned, f"Please keep the question under {config.MAX_QUESTION_CHARS} characters."
    return True, cleaned, ""


def confidence_of(hit):
    """The score of one chunk. A reranker score is used when there is one,
    otherwise the vector similarity."""
    if "rerank_score" in hit:
        return hit["rerank_score"]
    return hit.get("vector_similarity", 0.0)


def check_confidence(hits, use_threshold):
    """Return (ok, status). Status is "ok", "no_context" or "low_confidence".
    The threshold is only applied when use_threshold is on AND a number has been
    set in config.py, so nothing changes until you calibrate it in Part 6."""
    if not hits:
        return False, "no_context"

    if use_threshold:
        reranked = "rerank_score" in hits[0]
        threshold = config.MIN_RERANK_SCORE if reranked else config.MIN_VECTOR_SIMILARITY
        if threshold is not None and confidence_of(hits[0]) < threshold:
            return False, "low_confidence"
    return True, "ok"


def fallback_message(status, hits=None):
    """The reply used when the model is not called. Includes the closest topics
    as suggestions, so the user has something to try next."""
    if status == "no_context":
        text = "I could not find anything in the company documents about that."
    else:
        text = ("I could not find a reliable answer in the company documents. "
                "Try rephrasing the question or adding more detail.")

    topics = []
    for hit in hits or []:
        label = hit.get("section") or hit["source"]
        if label not in topics:
            topics.append(label)
    if topics:
        text += "\n\nThe closest topics I found were\n" + "\n".join(f"- {t}" for t in topics[:3])
    return text


def is_refusal(answer):
    """True when the model answered with the agreed "I do not know" sentence."""
    return answer.strip().lower().startswith(config.NO_ANSWER_TEXT.lower())