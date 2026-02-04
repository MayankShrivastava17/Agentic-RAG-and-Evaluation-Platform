import argparse
from src.common.config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL
from src.retrieval.embedder import Embedder
from src.retrieval.qdrant_store import QdrantStore
from src.agent.tools import RetrievalTool
from src.agent.graph import build_graph
from src.eval.runner import run_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Eval jsonl with {question, gold_chunk_ids[]}")
    args = ap.parse_args()

    embedder = Embedder(EMBEDDING_MODEL)
    vector_size = len(embedder.embed(["ping"])[0])
    store = QdrantStore(url=QDRANT_URL, collection=QDRANT_COLLECTION, vector_size=vector_size)
    store.ensure_collection()
    retrieval_tool = RetrievalTool(embedder, store)
    agent_app = build_graph(retrieval_tool)

    res = run_eval(args.path, retrieval_tool, agent_app)
    print("EVAL RESULTS")
    print(f"n={res.n}")
    print(f"Recall@3={res.recall_at_3:.3f}  Recall@6={res.recall_at_6:.3f}")
    print(f"MRR={res.mrr:.3f}  nDCG@6={res.ndcg_at_6:.3f}")
    print(f"AgentSuccess={res.agent_success:.3f}")

if __name__ == "__main__":
    main()
