"""Tool registry. Every tool exposed to agents is registered here.

Each tool ships with a JSON Schema describing its inputs and outputs so
the planner can reason about which tools are appropriate without a
free-form lookup. New tools are added in three steps:

1. Implement the tool in this package.
2. Register it in `REGISTRY`.
3. Write its capability card under `app/a2a/cards/tools/`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from . import ais_history, code_search, road_network, vector_search, weather, web_search

REGISTRY: dict[str, dict[str, Any]] = {
    "vector.search": {
        "description": "Dense+BM25 hybrid retrieval over the historical incident corpus",
        "fn": vector_search.run,
    },
    "web.search": {
        "description": "OSINT search for vessel registries, incident reports",
        "fn": web_search.run,
    },
    "code.search": {
        "description": "Grep over the analyst playbook repository",
        "fn": code_search.run,
    },
    "ais.history": {
        "description": "AIS position history for a vessel ID over a time window",
        "fn": ais_history.run,
    },
    "road.network": {
        "description": "OSM road network within a bounding box, with travel-time bounds",
        "fn": road_network.run,
    },
    "weather.fetch": {
        "description": "Copernicus CDS sea-state and wind for a bbox+time window",
        "fn": weather.run,
    },
}


async def call(tool_name: str, args: dict[str, Any]) -> Any:
    spec = REGISTRY.get(tool_name)
    if spec is None:
        raise KeyError(f"unknown tool: {tool_name}")
    fn: Callable[[dict[str, Any]], Awaitable[Any]] = spec["fn"]
    return await fn(args)
