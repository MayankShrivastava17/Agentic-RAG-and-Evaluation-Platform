from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

class QdrantStore:
    def __init__(self, url: str, collection: str, vector_size: int):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=self.vector_size,
                distance=qm.Distance.COSINE,
            ),
        )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        points = [
            qm.PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        results = []
        for h in hits:
            results.append(
                {
                    "id": str(h.id),
                    "score": float(h.score),
                    "text": h.payload.get("text", ""),
                    "source": h.payload.get("source", ""),
                }
            )
        return results
