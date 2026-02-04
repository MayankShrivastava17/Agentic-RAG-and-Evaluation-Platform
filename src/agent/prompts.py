PLAN_PROMPT = """You are a planner. Create a short plan (2-5 steps) to answer the user question.
Question: {question}
Return only the plan."""

VERIFY_PROMPT = """You are a verifier. Decide if the retrieved context is sufficient and relevant to answer.
Return JSON with keys: sufficient (true/false), reason (string), missing (string).
Question: {question}
Context:
{context}"""

ANSWER_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context.
If context is insufficient, say what is missing briefly.
Question: {question}
Context:
{context}
Answer:"""
