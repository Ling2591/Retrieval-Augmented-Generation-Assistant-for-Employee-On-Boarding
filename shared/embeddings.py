# shared/embeddings.py
# Phase 1 (indexing documents) and Phase 2 (embedding a question) both call
# embed_texts(), which guarantees both sides use the same model.

from functools import lru_cache

from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_model():
    """Load the model once and reuse it. The first call takes a few seconds,
    
    every later call is instant because lru_cache remembers the result."""
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def embed_texts(texts, show_progress=False):
    """Turn text into vectors.

    texts          a single string or a list of strings
    returns        a list of vectors, one per input text, each a list of floats

    Vectors are normalised to length 1. That means a plain dot product between
    two vectors equals their cosine similarity, which keeps every other file
    (Chroma, reranking, thresholds) consistent.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    if not texts:
        return []

    vectors = get_model().encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def count_tokens(text):
    """Count the word pieces the model would see in this text.
    Part 3 uses it to check that chunks fit inside the model limit."""
    tokenizer = get_model().tokenizer
    return len(tokenizer.encode(text, add_special_tokens=False))


if __name__ == "__main__":
    # Self test. Run with  python -m shared.embeddings
    sentences = [
        "The cat sat on the sofa.",
        "A kitten is resting on the couch.",
        "The stock market fell sharply today.",
    ]
    vectors = embed_texts(sentences)

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    print("sentences embedded  ", len(vectors))
    print("numbers per sentence", len(vectors[0]), "(config expects", config.EMBEDDING_DIMENSIONS, ")")
    print("cat vs kitten       ", round(dot(vectors[0], vectors[1]), 3))
    print("cat vs stocks       ", round(dot(vectors[0], vectors[2]), 3))
    print("word pieces in the code ITM-20417 ->", count_tokens("ITM-20417"))