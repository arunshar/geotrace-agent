"""Content filter. Second layer.

Strips PII-adjacent fields (raw MMSI, owner names) from any object that
crosses the security boundary on its way out of the orchestrator.
"""

from __future__ import annotations

from typing import Any

_REDACTABLE_KEYS = {"mmsi", "imo", "owner", "operator", "phone", "email"}


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k in _REDACTABLE_KEYS else scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value
