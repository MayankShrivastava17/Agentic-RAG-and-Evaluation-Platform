from fastapi import FastAPI
from src.common.logging import setup_logging
from src.common.config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL
from src.retrieval.embedder import Embedder
from src.retrieval.qdrant_store import QdrantStore
from src.agent.tools import RetrievalTool
from src.agent.graph import build_graph
from src.api.schemas import AskRequest, AskResponse

setup_logging()
app = FastAPI(title="Agentic RAG (LangGraph)")

embedder = Embedder(EMBEDDING_MODEL)
vector_size = len(embedder.embed(["ping"])[0])
store = QdrantStore(url=QDRANT_URL, collection=QDRANT_COLLECTION, vector_size=vector_size)
store.ensure_collection()
retrieval_tool = RetrievalTool(embedder, store)

agent_app = build_graph(retrieval_tool)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    out = agent_app.invoke({"question": req.question})
    return AskResponse(
        question=req.question,
        plan=out.get("plan"),
        retrieved=out.get("retrieved", []),
        verification=out.get("verification", {}),
        answer=out.get("answer", ""),
    )
