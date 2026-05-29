"""Parallel-safe tool batcher with deduplication.

If two agents request the same tool call within a short window, the
second call short-circuits to the first call's awaitable. This is
"in-flight deduplication" — different from the semantic cache, which
matches across runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class ToolBatcher:
    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(tool: str, args: dict[str, Any]) -> str:
        h = hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:24]
        return f"{tool}:{h}"

    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        k = self._key(tool, args)
        async with self._lock:
            existing = self._inflight.get(k)
            if existing is not None:
                log.debug("tool_dedup_hit", tool=tool, key=k)
                return await existing
            fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            self._inflight[k] = fut

        try:
            res = await fn()
            fut.set_result(res)
            return res
        except Exception as exc:  # pragma: no cover
            fut.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._inflight.pop(k, None)
