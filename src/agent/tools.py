from src.common.config import TOP_K
from src.retrieval.embedder import Embedder
from src.retrieval.qdrant_store import QdrantStore

class RetrievalTool:
    def __init__(self, embedder: Embedder, store: QdrantStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = TOP_K):
        qv = self.embedder.embed([query])[0]
        return self.store.search(qv, top_k=top_k)
