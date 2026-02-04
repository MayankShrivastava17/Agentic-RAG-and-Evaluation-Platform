from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    plan: Optional[str] = None
    retrieved: List[Dict[str, Any]] = []
    verification: Dict[str, Any] = {}
    answer: str
