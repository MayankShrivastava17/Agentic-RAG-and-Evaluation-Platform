from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict, total=False):
    question: str
    plan: str
    retrieved: List[Dict[str, Any]]
    verification: Dict[str, Any]
    answer: str
    tool_logs: List[Dict[str, Any]]
