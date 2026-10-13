from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


@dataclass
class ReportMatch:
    score: float
    report_id: str
    text: str
    category: str


class LocalSemanticReportStore:
    """Local fallback for the Pinecone layer; no network/API key required."""
    def __init__(self, reports) -> None:
        self.reports = reports.reset_index(drop=True)
        self.model = SentenceTransformer("all-MiniLM-L6-v2") if SentenceTransformer else None
        self.matrix = None
        if self.model is not None and len(self.reports):
            self.matrix = self.model.encode(self.reports["report_text"].tolist(), normalize_embeddings=True)

    def search(self, query: str, target_upi: str | None = None, top_k: int = 5) -> list[ReportMatch]:
        candidates = self.reports
        if target_upi:
            target = candidates[candidates["target_upi"].eq(target_upi)]
            if not target.empty:
                candidates = target
        if candidates.empty:
            return []
        if self.model is None or self.matrix is None:
            return [ReportMatch(0.5, str(r.report_id), r.report_text, r.category) for _, r in candidates.head(top_k).iterrows()]
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        # Matrix rows are aligned to original reports; recalculate for filtered candidate indices.
        scores = self.matrix[candidates.index] @ query_vec
        order = scores.argsort()[::-1][:top_k]
        return [
            ReportMatch(float(scores[i]), str(candidates.iloc[pos].report_id), candidates.iloc[pos].report_text, candidates.iloc[pos].category)
            for i, pos in enumerate(order)
        ]


class PineconeReportStore:
    def __init__(self, index: Any):
        self.index = index

    def search(self, query_vector: list[float], top_k: int = 5):
        return self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
