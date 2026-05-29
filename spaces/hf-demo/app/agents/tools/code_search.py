"""Grep over the analyst playbook repo. Stub uses an in-memory dict."""

from __future__ import annotations

from typing import Any

_PLAYBOOK = {
    "rendezvous_protocol": "When two prisms intersect over open water, escalate if AGM > 0.7.",
}


async def run(args: dict[str, Any]) -> dict[str, Any]:
    q = (args.get("query") or "").lower()
    return {"hits": [{"id": k, "text": v} for k, v in _PLAYBOOK.items() if q in v.lower()]}
