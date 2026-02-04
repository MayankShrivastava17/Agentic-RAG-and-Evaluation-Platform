import json
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.prompts import PLAN_PROMPT, VERIFY_PROMPT, ANSWER_PROMPT
from src.common.config import LLM_MODE, OPENAI_API_KEY, OPENAI_MODEL
import httpx

def _format_context(retrieved: list[dict], max_items: int = 6) -> str:
    blocks = []
    for r in retrieved[:max_items]:
        blocks.append(f"[source={r.get('source','')}, score={r.get('score',0):.3f}]\n{r.get('text','')}")
    return "\n\n---\n\n".join(blocks)

async def _call_openai(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

async def llm(prompt: str) -> str:
    if LLM_MODE == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_MODE=openai")
        return await _call_openai(prompt)
    return "STUB_RESPONSE: " + prompt[:180].replace("\n", " ") + "..."

def build_graph(retrieval_tool):
    g = StateGraph(AgentState)

    async def planner(state: AgentState) -> AgentState:
        plan = await llm(PLAN_PROMPT.format(question=state["question"]))
        return {"plan": plan}

    async def retrieve(state: AgentState) -> AgentState:
        docs = retrieval_tool.retrieve(state["question"])
        logs = state.get("tool_logs", [])
        logs.append({"tool": "retrieve", "k": len(docs)})
        return {"retrieved": docs, "tool_logs": logs}

    async def verify(state: AgentState) -> AgentState:
        ctx = _format_context(state.get("retrieved", []))
        raw = await llm(VERIFY_PROMPT.format(question=state["question"], context=ctx))
        try:
            v = json.loads(raw)
            sufficient = bool(v.get("sufficient", False))
            reason = str(v.get("reason", ""))
            missing = str(v.get("missing", ""))
        except Exception:
            sufficient = len(state.get("retrieved", [])) > 0
            reason = "Fallback verification (non-JSON verifier output)."
            missing = ""
        return {"verification": {"sufficient": sufficient, "reason": reason, "missing": missing}}

    def route_after_verify(state: AgentState) -> str:
        v = state.get("verification", {})
        if v.get("sufficient") is True:
            return "answer"
        retry_count = sum(1 for l in state.get("tool_logs", []) if l.get("tool") == "retrieve")
        return "retrieve" if retry_count < 2 else "answer"

    async def answer(state: AgentState) -> AgentState:
        ctx = _format_context(state.get("retrieved", []))
        ans = await llm(ANSWER_PROMPT.format(question=state["question"], context=ctx))
        return {"answer": ans}

    g.add_node("planner", planner)
    g.add_node("retrieve", retrieve)
    g.add_node("verify", verify)
    g.add_node("answer", answer)

    g.set_entry_point("planner")
    g.add_edge("planner", "retrieve")
    g.add_edge("retrieve", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"retrieve": "retrieve", "answer": "answer"})
    g.add_edge("answer", END)

    return g.compile()
