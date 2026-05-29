"""Versioned, hot-swappable prompt registry.

Every prompt has a stable name like `planner.v3`. Old versions are
preserved so we can replay historical traces. Live edits go to a new
version; the live system pin lives in `prompts.yaml` (read by tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Any

from app.prompts import templates as T


@dataclass(frozen=True)
class _Prompt:
    name: str
    template: str

    def render(self, **kwargs: Any) -> str:
        return Template(self.template).safe_substitute(**{k: _coerce(v) for k, v in kwargs.items()})


def _coerce(v: Any) -> str:
    if isinstance(v, str):
        return v
    return repr(v)


_REGISTRY: dict[str, _Prompt] = {
    "planner.v1": _Prompt("planner.v1", T.PLANNER_V1),
    "planner.v2": _Prompt("planner.v2", T.PLANNER_V2),
    "planner.v3": _Prompt("planner.v3", T.PLANNER_V3),
    "summarize.v1": _Prompt("summarize.v1", T.SUMMARIZE_V1),
    "summarize.v2": _Prompt("summarize.v2", T.SUMMARIZE_V2),
    "gap_score.v1": _Prompt("gap_score.v1", T.GAP_SCORE_V1),
}


def get_prompt(name: str) -> _Prompt:
    return _REGISTRY[name]


def list_prompts() -> list[str]:
    return sorted(_REGISTRY.keys())
