# querying/display.py
# Turns a pipeline result into things a screen can show. No Streamlit in this file, so it
# can be tested on its own and reused by any other front end.
#
#   history_for_rewrite()   the earlier chat as plain messages for the rewrite step
#   escape_for_markdown()   protects dollar signs, which Streamlit would read as maths
#   needs_warning()         True when the best source matched only weakly
#   source_cards()          the sources as simple dictionaries
#   summarize()             one compact dictionary per answer, kept in the chat history

import config


def history_for_rewrite(messages):
    """Earlier chat messages reduced to role and text, which is all the rewrite step needs."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("role") in ("user", "assistant") and str(message.get("content", "")).strip()
    ]


def escape_for_markdown(text):
    """Streamlit treats text between two dollar signs as a formula, so a price list such as
    $96.50 and $38.90 would be garbled. A backslash in front of each one stops that."""
    return str(text).replace("$", "\\$")


def top_score(result):
    """The score of the best source, or None when there were no sources."""
    sources = result.get("sources") or []
    if not sources:
        return None
    first = sources[0]
    return first["rerank_score"] if "rerank_score" in first else first.get("vector_similarity")


def needs_warning(result):
    """True when an answer was given but its best source matched only weakly."""
    limit = getattr(config, "LOW_CONFIDENCE_RERANK_SCORE", None)
    score = top_score(result)
    sources = result.get("sources") or []
    reranked = bool(sources) and "rerank_score" in sources[0]
    return bool(
        limit is not None and reranked and score is not None
        and result.get("status") == "answered" and score < limit
    )


def source_cards(result, snippet_chars=600):
    cards = []
    for number, hit in enumerate(result.get("sources") or [], start=1):
        title = hit["source"] + (f", page {hit['page']}" if hit.get("page") else "")
        score = hit["rerank_score"] if "rerank_score" in hit else hit.get("vector_similarity", 0.0)
        cards.append({
            "number": number,
            "title": title,
            "section": hit.get("section", ""),
            "score": float(score),
            "snippet": hit["text"][:snippet_chars] + ("..." if len(hit["text"]) > snippet_chars else ""),
        })
    return cards


def summarize(result):
    """The compact record stored in the chat history for one answer."""
    llm = result.get("llm") or {}
    return {
        "role": "assistant",
        "content": result["answer"],
        "status": result["status"],
        "sources": source_cards(result),
        "question": result["question"],
        "search_query": result["search_query"],
        "warning": needs_warning(result),
        "top_score": top_score(result),
        "tokens": llm.get("total_tokens", 0) + result.get("rewrite_tokens", 0),
        "timings": {name: round(value, 2) for name, value in result["timings"].items()},
    }