from __future__ import annotations

from typing import Iterable

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # optional dependency at runtime
    SentenceTransformer = None


def build_report_vectors(reports: Iterable[dict], model_name: str = "all-MiniLM-L6-v2") -> list[dict]:
    """Create Pinecone-ready vectors. Requires sentence-transformers for real embeddings."""
    reports = list(reports)
    if SentenceTransformer is None:
        raise RuntimeError("Install sentence-transformers to create transformer embeddings.")
    model = SentenceTransformer(model_name)
    texts = [r["report_text"] for r in reports]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [
        {
            "id": str(r["report_id"]),
            "values": emb.tolist(),
            "metadata": {"target_upi": r["target_upi"], "category": r["category"], "text": r["report_text"]},
        }
        for r, emb in zip(reports, embeddings)
    ]


def upsert_to_pinecone(index, vectors: list[dict], batch_size: int = 100) -> int:
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])
    return len(vectors)
