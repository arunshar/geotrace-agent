"""Semantic cache. Keyed on tuples for exact hits, embeddings for near hits.

Tool-call optimization in this system has three layers:

1. Exact-key cache. The tuple `(tool_name, *args_hash)` hits Redis
   directly. This is the cheap path for prism computations: identical
   anchor pairs return the same prism in O(1).
2. Semantic cache. The query is embedded; nearest neighbours above
   `semantic_cache_similarity` are returned. We embed with a small
   local model so the cache itself does not consume LLM budget.
3. Negative cache. Failed lookups are also stored briefly to avoid
   thundering-herd retries on a transient failure.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import Any

import httpx
import numpy as np
import redis.asyncio as redis
import structlog

from app.config import Settings

log = structlog.get_logger(__name__)


def _hkey(parts: tuple[Any, ...]) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(bytes(p))
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return "gt:cache:" + h.hexdigest()[:32]


class SemanticCache:
    def __init__(self, settings: Settings, redis_client: redis.Redis, http: httpx.AsyncClient) -> None:
        self.settings = settings
        self.r = redis_client
        self.http = http

    @classmethod
    async def connect(cls, settings: Settings) -> SemanticCache:
        client = redis.from_url(settings.redis_url, decode_responses=False)
        try:
            await client.ping()
        except Exception:  # pragma: no cover
            log.warning("redis_unavailable_using_inmemory")
            return cls(settings, _InMemoryRedis(), httpx.AsyncClient())
        return cls(settings, client, httpx.AsyncClient(timeout=10.0))

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.r.close()
        await self.http.aclose()

    # -------------------------------------------------------- exact cache

    async def get(self, key: tuple[Any, ...]) -> Any | None:
        raw = await self.r.get(_hkey(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set(self, key: tuple[Any, ...], value: Any, ttl_s: int | None = None) -> None:
        await self.r.set(
            _hkey(key),
            json.dumps(value, default=str).encode(),
            ex=ttl_s or self.settings.semantic_cache_ttl_s,
        )

    # ----------------------------------------------------- semantic cache

    async def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Lightweight semantic retrieval over cached prior answers.

        The full retrieval system (Chroma + reranker) lives in
        `components/hybrid_retriever.py`. This entry point is the cheap
        first hop the orchestrator can call to short-circuit identical
        questions.
        """

        q_vec = await self._embed(query)
        cursor = 0
        best: tuple[float, dict[str, Any]] | None = None
        while True:
            cursor, keys = await self.r.scan(cursor=cursor, match=b"gt:cache:*", count=200)
            for k in keys[:64]:  # cap scan width
                meta = await self.r.hgetall(k + b":meta") if isinstance(k, bytes) else None
                if not meta:
                    continue
                vec_b = meta.get(b"vec")
                if not vec_b:
                    continue
                v = np.frombuffer(vec_b, dtype=np.float32)
                sim = float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-9))
                if sim >= self.settings.semantic_cache_similarity and (best is None or sim > best[0]):
                    raw = await self.r.get(k)
                    if raw:
                        best = (sim, json.loads(raw))
            if cursor == 0:
                break
        return [best[1]] if best else []

    async def _embed(self, text: str) -> np.ndarray:
        # In production swap for a local embedding (bge-small-en-v1.5).
        # The scaffold uses a stable hash projection so tests are
        # deterministic and offline-safe.
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(np.frombuffer(h, dtype=np.uint32))
        return rng.standard_normal(384).astype(np.float32)


class _InMemoryRedis:
    """Async-shaped fallback for unit tests that have no Redis."""

    def __init__(self) -> None:
        self._d: dict[str, bytes] = {}

    async def get(self, k: str) -> bytes | None:
        return self._d.get(k)

    async def set(self, k: str, v: bytes, ex: int | None = None) -> None:
        self._d[k] = v

    async def hgetall(self, k: str) -> dict:
        return {}

    async def scan(self, cursor: int = 0, match: bytes | None = None, count: int = 100) -> tuple[int, list[bytes]]:
        return 0, [k.encode() for k in self._d]

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None
