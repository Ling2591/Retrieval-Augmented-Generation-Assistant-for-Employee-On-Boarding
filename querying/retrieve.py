# querying/retrieve.py
# PHASE 2, step 2 (step 1 is the optional rewrite). Finds candidate chunks.
#
# Two modes.
#   vector only   the k chunks closest in meaning to the question
#   hybrid        vector search AND keyword search, merged into one ranked list
# And one filter.
#   company_only  search only the company's own documents (COMPANY_SOURCE_PREFIXES in config.py)
#
# How the merge works (reciprocal rank fusion). Each search gives a ranked list.
# A chunk earns  1 / (RRF_K + rank)  from every list it appears in, and the chunks
# are ordered by their total. A chunk that ranks well in BOTH lists beats a chunk
# that ranks first in only one. No score scaling is needed, because only ranks are used.
#
# Every hit is a chunk record (id, text, source, page, section) plus these fields.
#   vector_similarity   1.0 means identical meaning, about 0 means unrelated (always present)
#   vector_rank         position in the vector list, None when only keyword search found it
#   bm25_rank           position in the keyword list, None when keyword search did not find it
#   bm25_score          the keyword score, 0 when not found
#   fused_score         the merged score, hybrid mode only

import config
from shared import store
from shared.embeddings import embed_texts


def retrieve(search_query, use_hybrid=False, k=None, company_only=False):
    """Return up to k candidate chunks for the question, best first.
    With company_only, only the company's own documents are searched, never the
    public handbooks of other organizations."""
    k = k or config.VECTOR_TOP_K
    query_vector = embed_texts(search_query)[0]      # same model that embedded the chunks

    sources = None
    if company_only:
        sources = store.company_sources()
        if sources is None:
            raise ValueError("company_only needs COMPANY_SOURCE_PREFIXES to be set in config.py.")

    vector_hits = store.vector_search(query_vector, k, sources)
    if not use_hybrid:
        for rank, hit in enumerate(vector_hits, start=1):
            hit.update(vector_rank=rank, bm25_rank=None, bm25_score=0.0)
        return vector_hits

    keyword_hits = store.keyword_search(search_query, config.BM25_TOP_K, sources)

    merged = {}
    for rank, hit in enumerate(vector_hits, start=1):
        merged[hit["id"]] = {**hit, "vector_rank": rank, "bm25_rank": None, "bm25_score": 0.0}
    for rank, hit in enumerate(keyword_hits, start=1):
        if hit["id"] in merged:
            merged[hit["id"]].update(bm25_rank=rank, bm25_score=hit["bm25_score"])
        else:
            merged[hit["id"]] = {**hit, "vector_rank": None, "bm25_rank": rank}

    # Chunks found only by keyword search still need a meaning-based score.
    missing = [chunk_id for chunk_id, hit in merged.items() if "vector_similarity" not in hit]
    similarities = store.vector_similarities(query_vector, missing)
    for chunk_id in missing:
        similarity = similarities.get(chunk_id, 0.0)
        merged[chunk_id]["vector_similarity"] = similarity
        merged[chunk_id]["vector_distance"] = 2.0 * (1.0 - similarity)

    for hit in merged.values():
        score = 0.0
        if hit["vector_rank"]:
            score += 1.0 / (config.RRF_K + hit["vector_rank"])
        if hit["bm25_rank"]:
            score += 1.0 / (config.RRF_K + hit["bm25_rank"])
        hit["fused_score"] = score

    ranked = sorted(merged.values(), key=lambda hit: (-hit["fused_score"], -hit["vector_similarity"]))
    return ranked[:k]


# ---------------------------------------------------------------------------
# Self test. Run with  python -m querying.retrieve "your question"
# or with both searches merged  python -m querying.retrieve --hybrid "your question"
# or limited to company documents  python -m querying.retrieve --hybrid --company "your question"
# No language model is called here, so this costs nothing and is instant.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    arguments = sys.argv[1:]
    hybrid = "--hybrid" in arguments
    company = "--company" in arguments
    question = " ".join(a for a in arguments if not a.startswith("--")) or "What is the price of item ITM-20417?"
    print("Question", question, "| hybrid" if hybrid else "| vector only", "| company documents only" if company else "")
    print("  rank  similarity  vector_rank  bm25_rank  chunk")
    for number, hit in enumerate(retrieve(question, use_hybrid=hybrid, k=8, company_only=company), start=1):
        print(f"  {number:>4}  {hit['vector_similarity']:10.3f}  {str(hit['vector_rank']):>11}  "
              f"{str(hit['bm25_rank']):>9}  {hit['source']}  >  {hit['section']}")