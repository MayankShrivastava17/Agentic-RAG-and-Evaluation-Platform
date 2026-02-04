from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str

def simple_chunk(text: str, source: str, max_chars: int = 900, overlap: int = 120):
    text = text.strip()
    chunks = []
    start = 0
    i = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk_text = text[start:end]
        chunks.append(Chunk(chunk_id=f"{source}:{i}", text=chunk_text, source=source))
        i += 1
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks
