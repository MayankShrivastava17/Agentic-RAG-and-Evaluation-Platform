import math

def recall_at_k(gold_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    hit = len(gold_ids.intersection(set(retrieved_ids[:k])))
    return hit / len(gold_ids)

def mrr(gold_ids: set[str], retrieved_ids: list[str]) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / i
    return 0.0

def ndcg_at_k(gold_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    def dcg(ids):
        s = 0.0
        for i, rid in enumerate(ids[:k], start=1):
            rel = 1.0 if rid in gold_ids else 0.0
            s += rel / math.log2(i + 1)
        return s

    ideal = list(gold_ids) + ["__pad__"] * 100
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(retrieved_ids) / denom
