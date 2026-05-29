"""Web search tool. SerpAPI / Tavily in production; stub here."""

from __future__ import annotations

from typing import Any


async def run(args: dict[str, Any]) -> dict[str, Any]:
    return {"query": args.get("query", ""), "results": []}
