"""Vector search tool. Wraps Chroma in production; deterministic stub here."""

from __future__ import annotations

from typing import Any


async def run(args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query", "")
    return {"query": query, "hits": []}
