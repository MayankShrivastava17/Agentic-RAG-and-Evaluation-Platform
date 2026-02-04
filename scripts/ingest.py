import argparse
import json
from src.common.config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL
from src.retrieval.embedder import Embedder
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.chunking import simple_chunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to jsonl with {source, text}")
    args = ap.parse_args()

    embedder = Embedder(EMBEDDING_MODEL)
    vector_size = len(embedder.embed(["ping"])[0])
    store = QdrantStore(url=QDRANT_URL, collection=QDRANT_COLLECTION, vector_size=vector_size)
    store.ensure_collection()

    ids, texts, payloads = [], [], []

    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            source = obj["source"]
            text = obj["text"]
            chunks = simple_chunk(text, source=source)
            for c in chunks:
                ids.append(c.chunk_id)
                texts.append(c.text)
                payloads.append({"text": c.text, "source": c.source})

    vecs = embedder.embed(texts)
    store.upsert(ids=ids, vectors=vecs, payloads=payloads)
    print(f"Ingested {len(ids)} chunks into collection={QDRANT_COLLECTION}")

if __name__ == "__main__":
    main()
