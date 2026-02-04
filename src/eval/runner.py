import json
from dataclasses import dataclass
from src.eval.metrics import recall_at_k, mrr, ndcg_at_k

@dataclass
class EvalResult:
    n: int
    recall_at_3: float
    recall_at_6: float
    mrr: float
    ndcg_at_6: float
    agent_success: float

def run_eval(eval_path: str, retrieval_tool, agent_app) -> EvalResult:
    rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    r3 = r6 = m = n6 = success = 0.0
    for ex in rows:
        q = ex["question"]
        gold = set(ex.get("gold_chunk_ids", []))

        retrieved = retrieval_tool.retrieve(q, top_k=6)
        retrieved_ids = [r["id"] for r in retrieved]

        r3 += recall_at_k(gold, retrieved_ids, 3)
        r6 += recall_at_k(gold, retrieved_ids, 6)
        m += mrr(gold, retrieved_ids)
        n6 += ndcg_at_k(gold, retrieved_ids, 6)

        # Agent success = did it produce a non-empty answer and pass basic sufficiency?
        out = agent_app.invoke({"question": q})
        ans_ok = bool(out.get("answer", "").strip())
        ver = out.get("verification", {})
        success += 1.0 if (ans_ok and (ver.get("sufficient") is True or len(retrieved) > 0)) else 0.0

    n = max(1, len(rows))
    return EvalResult(
        n=n,
        recall_at_3=r3 / n,
        recall_at_6=r6 / n,
        mrr=m / n,
        ndcg_at_6=n6 / n,
        agent_success=success / n,
    )
