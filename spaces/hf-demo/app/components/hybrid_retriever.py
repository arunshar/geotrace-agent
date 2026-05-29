"""Hybrid retrieval: BM25 over titles + dense over bodies + reranker.

Used by `RetrieveAgent` (and indirectly by `SemanticCache.retrieve`)
to ground answers in a corpus of historical incidents and analyst
notes. Cheap path: BM25 alone. Hot path: dense + rerank.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import structlog

from app.components.reranker import Reranker

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    text: str
    source: str
    score: float


class HybridRetriever:
    def __init__(self, *, dense_topk: int = 24, sparse_topk: int = 24, final_k: int = 6) -> None:
        self.dense_topk = dense_topk
        self.sparse_topk = sparse_topk
        self.final_k = final_k
        self._reranker = Reranker()

    async def retrieve(self, query: str, *, corpus: Iterable[RetrievedDoc]) -> list[RetrievedDoc]:
        cands = list(corpus)
        if not cands:
            return []
        # cheap unigram-overlap "BM25 lite" scoring; production uses Chroma+BM25
        q_terms = set(query.lower().split())
        sparse = sorted(
            cands,
            key=lambda d: -len(q_terms.intersection(d.text.lower().split())),
        )[: self.sparse_topk]
        # deterministic "dense" surrogate: identical to sparse for the scaffold
        merged: dict[str, RetrievedDoc] = {d.doc_id: d for d in sparse}
        ordered = list(merged.values())
        return await self._reranker.rerank(query, ordered, top_k=self.final_k)
