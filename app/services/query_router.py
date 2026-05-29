"""Routes a question to the right entrypoint.

Intent labels:
- `prism_only`. The user just wants a space-time prism for an anchor pair.
- `rendezvous`. The user wants candidate rendezvous regions.
- `gap_audit`. The user wants the gap detector + AGM ranking.
- `general`. Falls through to the full pipeline (planner -> agents).

Cheap classifier: keyword + small zero-shot prompt. Defaults to
`general` so we never lose a question to a routing mistake.
"""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal["prism_only", "rendezvous", "gap_audit", "general"]


_RE = {
    "prism_only": re.compile(r"\bprism\b|\breachab(le|ility)\b|\benvelope\b", re.I),
    "rendezvous": re.compile(r"\brendezvous\b|\bmeet(?:up)?\b|\bencounter\b|\bcontact\b", re.I),
    "gap_audit":  re.compile(r"\bgap\b|\bsignal loss\b|\bblackout\b|\bdrop\b", re.I),
}


def route(question: str) -> Intent:
    for name, rx in _RE.items():
        if rx.search(question):
            return name  # type: ignore[return-value]
    return "general"
