# indexing/build_index.py
# PHASE 1, the script you run. It chains every Phase 1 step.
#   1. load_documents()         read and clean the files in data/raw
#   2. chunk_documents()        cut them into chunks, save data/chunks.jsonl
#   3. embed_texts()            turn every chunk into 384 numbers (the slow step)
#   4. Chroma                   store text, numbers and labels in data/chroma
# Then it asks the finished database three sample questions as a smoke test.
#
# Run from the project root with  python -m indexing.build_index
# It always rebuilds from scratch, so you can run it again after any change
# to the documents, to config.py, or to the chunking code.

import json
import time

import config
from indexing.chunking import chunk_documents
from indexing.loaders import load_documents
from shared import store
from shared.embeddings import embed_texts

ADD_BATCH_SIZE = 200

SAMPLE_QUESTIONS = [
    "How many days of annual leave do I get?",
    "What is the price of item ITM-20417?",
    "What does VPN error 612 mean?",
]


def write_chunks_file(chunks):
    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def build():
    started = time.time()

    print("Step 1 of 4, loading documents")
    pages = load_documents()

    print("\nStep 2 of 4, chunking")
    chunks = chunk_documents(pages)
    write_chunks_file(chunks)
    print(f"Saved {len(chunks)} chunks to {config.CHUNKS_PATH}")

    print("\nStep 3 of 4, embedding every chunk (the slow step)")
    vectors = embed_texts([chunk["text"] for chunk in chunks], show_progress=True)

    print("\nStep 4 of 4, storing in Chroma")
    collection = store.create_fresh_collection()
    for start in range(0, len(chunks), ADD_BATCH_SIZE):
        batch = chunks[start:start + ADD_BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors[start:start + ADD_BATCH_SIZE],
            documents=[c["text"] for c in batch],
            # Chroma accepts only text, whole numbers, decimals and true/false here.
            metadatas=[{"source": c["source"], "page": int(c["page"]), "section": c["section"]}
                       for c in batch],
        )

    # If this script and the app ever run in one process, make them see the new data.
    store.get_collection.cache_clear()
    store.load_chunks.cache_clear()

    stored = store.get_collection().count()
    if stored != len(chunks):
        raise RuntimeError(f"Chroma holds {stored} vectors but there are {len(chunks)} chunks.")
    print(f"Chroma now holds {stored} vectors. Finished in {time.time() - started:.0f} seconds.")

    print("\nSmoke test, the three closest chunks for each sample question")
    for question in SAMPLE_QUESTIONS:
        print(f"\nQuestion  {question}")
        for hit in store.vector_search(embed_texts(question)[0], 3):
            preview = hit["text"].replace("\n", " ")[:100]
            print(f"  {hit['vector_similarity']:.3f}  {hit['source']}  |  {preview}")


if __name__ == "__main__":
    build()