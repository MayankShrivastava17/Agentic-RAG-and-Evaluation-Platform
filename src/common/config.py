import os
from dotenv import load_dotenv

load_dotenv()

def env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"Missing required env var: {key}")
    return val

LLM_MODE = os.getenv("LLM_MODE", "stub").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

QDRANT_URL = env("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = env("QDRANT_COLLECTION", "docs")

TOP_K = int(os.getenv("TOP_K", "6"))
EMBEDDING_MODEL = env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
