# querying/rerank.py
# PHASE 2, step 4. Reorders the candidates with a more careful model.
#
# Retrieval compares a question with each chunk separately, using numbers computed
# in advance. That is fast but rough. A cross-encoder reads the question and one
# chunk TOGETHER and judges how well the chunk answers the question. It is far more
# accurate and far slower, so it is only used on the 20 or so candidates that
# retrieval already found, never on the whole knowledge base.
#
# The score is a raw number, not a percentage. On the model's own examples a good
# match scores about +8 and a poor match about -4. Part 6 sets the confidence
# threshold from your own data, so the scale does not matter, only the ordering.

from functools import lru_cache

from sentence_transformers import CrossEncoder

import config


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder once. The first call downloads about 90 MB."""
    return CrossEncoder(config.RERANKER_MODEL_NAME)


def rerank(question, hits, top_n=None):
    """Return the hits ordered by relevance, each with a new  rerank_score  field.
    All hits are returned unless top_n is given, so the evaluation can see where
    the right chunk landed."""
    if not hits:
        return []
    pairs = [(question, hit["text"]) for hit in hits]
    scores = get_reranker().predict(pairs, batch_size=32, show_progress_bar=False)

    ranked = []
    for hit, score in zip(hits, scores):
        ranked.append({**hit, "rerank_score": float(score)})
    ranked.sort(key=lambda hit: hit["rerank_score"], reverse=True)
    return ranked[:top_n] if top_n else ranked


# ---------------------------------------------------------------------------
# Self test. Run with  python -m querying.rerank "your question"
# Shows the hybrid candidates before and after reranking. No language model is called.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from querying.retrieve import retrieve

    question = " ".join(sys.argv[1:]) or "How much can I claim for a chair or other home office equipment?"
    candidates = retrieve(question, use_hybrid=True)
    reranked = rerank(question, candidates)
    print("Question", question)
    print("  new  old  rerank  chunk")
    for new_rank, hit in enumerate(reranked[:8], start=1):
        old_rank = candidates.index(next(c for c in candidates if c["id"] == hit["id"])) + 1
        print(f"  {new_rank:>3}  {old_rank:>3}  {hit['rerank_score']:6.2f}  {hit['source']}  >  {hit['section']}")