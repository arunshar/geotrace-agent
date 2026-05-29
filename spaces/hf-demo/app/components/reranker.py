"""Reranker. Cross-encoder in production; deterministic stub here."""

from __future__ import annotations

from collections.abc import Iterable

import structlog

log = structlog.get_logger(__name__)


class Reranker:
    async def rerank(self, query: str, docs: Iterable, *, top_k: int = 6) -> list:
        ds = list(docs)
        ql = query.lower()
        ds.sort(key=lambda d: -sum(1 for tok in ql.split() if tok in d.text.lower()))
        return ds[:top_k]
