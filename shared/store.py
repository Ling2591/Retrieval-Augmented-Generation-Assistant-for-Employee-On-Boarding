# shared/store.py
# Everything that touches the saved Phase 1 results lives here.
#   load_chunks()               reads data/chunks.jsonl (the chunk records)
#   create_fresh_collection()   makes an empty Chroma collection (used by build_index)
#   get_collection()            opens the finished collection (used by Phase 2)
#   vector_search()             finds the chunks closest in meaning to a question vector
#   keyword_search()            finds the chunks sharing the most distinctive words (BM25)
#   vector_similarities()       scores chosen chunks against a question vector
#
# About distances. Chroma reports a "distance", where smaller means closer.
# Its default distance is the squared straight-line distance. Our vectors are
# normalised to length 1, so  similarity = 1 - distance / 2  gives exactly the
# cosine similarity you saw in your 6x6 grid (1.0 identical, about 0 unrelated).
# The self test at the bottom proves this conversion on your machine.
#
# About keyword search. BM25 scores a chunk higher when it contains the question's
# words, and counts rare words far more than common ones. That is why it finds
# exact codes such as ITM-20417, which meaning-based search tends to blur.

import json
import re
from functools import lru_cache

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

import config


def distance_to_similarity(distance):
    return 1.0 - distance / 2.0


def _client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def create_fresh_collection():
    """Delete any old collection and return a new empty one."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = _client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass                       # there was nothing to delete on the very first run
    return client.create_collection(config.COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_collection():
    """Open the finished collection. Cached, so the app opens it only once."""
    return _client().get_collection(config.COLLECTION_NAME)


@lru_cache(maxsize=1)
def load_chunks():
    """Return every chunk record. Treat the list as read-only."""
    if not config.CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{config.CHUNKS_PATH} not found. Run  python -m indexing.build_index  first."
        )
    with open(config.CHUNKS_PATH, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def company_sources():
    """File names of the company's own documents, found with COMPANY_SOURCE_PREFIXES
    in config.py. Returns None when no prefixes are set."""
    prefixes = tuple(config.COMPANY_SOURCE_PREFIXES)
    if not prefixes:
        return None
    return sorted({chunk["source"] for chunk in load_chunks() if chunk["source"].startswith(prefixes)})


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------
def vector_search(query_vector, k, sources=None):
    """Return the k chunks closest to query_vector, best first.
    When sources is a list of file names, only chunks from those files are considered.
    Each hit is a chunk record plus  vector_distance  and  vector_similarity."""
    if sources is not None and not sources:
        return []
    collection = get_collection()
    k = max(1, min(k, collection.count()))
    filters = {"where": {"source": {"$in": list(sources)}}} if sources is not None else {}
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        **filters,
    )
    hits = []
    for i, chunk_id in enumerate(result["ids"][0]):
        meta = result["metadatas"][0][i]
        distance = float(result["distances"][0][i])
        hits.append({
            "id": chunk_id,
            "text": result["documents"][0][i],
            "source": meta["source"],
            "page": int(meta["page"]),
            "section": meta["section"],
            "vector_distance": distance,
            "vector_similarity": distance_to_similarity(distance),
        })
    return hits

def vector_similarities(query_vector, ids):
    """Cosine similarity between the question and each chunk id given.
    Used for chunks that keyword search found but vector search did not."""
    if not ids:
        return {}
    result = get_collection().get(ids=list(ids), include=["embeddings"])
    query = np.asarray(query_vector, dtype=float)
    return {
        chunk_id: float(np.dot(query, np.asarray(embedding, dtype=float)))
        for chunk_id, embedding in zip(result["ids"], result["embeddings"])
    }


# ---------------------------------------------------------------------------
# Keyword search (BM25)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were",
    "do", "does", "did", "i", "my", "me", "we", "our", "you", "your", "how", "what", "which",
    "who", "when", "where", "can", "be", "it", "at", "by", "with", "get", "much", "many",
}
_WORD = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")


def tokenize(text):
    """Lowercase words. A code such as itm-20417 is kept whole AND also split into
    itm and 20417, so a question that says "error 612" still matches the chunk
    that says "VPN-612", while "ITM-20417" matches only that exact code."""
    tokens = []
    for word in _WORD.findall(text.lower()):
        parts = re.split(r"[-_./]", word)
        if len(parts) > 1:
            tokens.append(word)
        tokens.extend(part for part in parts if part not in _STOPWORDS)
    return tokens


@lru_cache(maxsize=1)
def get_bm25():
    """Build the keyword index from chunks.jsonl. Takes a moment, then stays in memory."""
    return BM25Okapi([tokenize(chunk["text"]) for chunk in load_chunks()])


def keyword_search(query, k, sources=None):
    """Return up to k chunks that share words with the query, best first.
    When sources is a list of file names, only chunks from those files are considered.
    Each hit is a chunk record plus  bm25_score. Chunks sharing no words are left out."""
    tokens = tokenize(query)
    if not tokens:
        return []
    chunks = load_chunks()
    scores = np.array(get_bm25().get_scores(tokens), dtype=float)
    if sources is not None:
        allowed = set(sources)
        scores = np.where([chunk["source"] in allowed for chunk in chunks], scores, 0.0)
    hits = []
    for position in np.argsort(scores)[::-1][:k]:
        score = float(scores[position])
        if score <= 0:
            break
        hits.append({**chunks[int(position)], "bm25_score": score})
    return hits

# ---------------------------------------------------------------------------
# Self test. Run with  python -m shared.store
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from shared.embeddings import embed_texts

    chunks = load_chunks()
    collection = get_collection()
    print("chunks in chunks.jsonl", len(chunks), "| vectors in Chroma", collection.count())

    query = embed_texts("How many days of annual leave do I get?")[0]
    hits = vector_search(query, 3)

    # Cross-check the distance conversion with a plain dot product.
    stored = collection.get(ids=[hits[0]["id"]], include=["embeddings"])["embeddings"][0]
    manual = sum(float(a) * float(b) for a, b in zip(query, stored))
    print("similarity from Chroma", round(hits[0]["vector_similarity"], 4),
          "| manual dot product", round(manual, 4), "(these two numbers must match)")
    print("similarity from vector_similarities", round(vector_similarities(query, [hits[0]["id"]])[hits[0]["id"]], 4))

    for hit in hits:
        print(f"  {hit['vector_similarity']:.3f}  {hit['source']}  >  {hit['section']}")

    print("\nKeyword search for  ITM-20417 price")
    for hit in keyword_search("ITM-20417 price", 3):
        print(f"  {hit['bm25_score']:6.2f}  {hit['source']}  >  {hit['section']}")