"""Query rewriter. Canonicalizes ambiguous spatial / temporal phrases.

Reduces token consumption downstream by replacing colloquial phrases
("around noon", "near the strait") with structured anchors and bounded
intervals so the planner does not need to reason about them.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

_NEAR_PATTERNS = re.compile(r"near (?P<lat>-?\d+(?:\.\d+)?)[ ,]+(?P<lon>-?\d+(?:\.\d+)?)")
_AROUND_NOON = re.compile(r"around noon", re.I)


def rewrite(question: str, *, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    out = question
    out = _NEAR_PATTERNS.sub(lambda m: f"at ({m.group('lat')}, {m.group('lon')})", out)
    if _AROUND_NOON.search(out):
        target = now.replace(hour=12, minute=0, second=0, microsecond=0)
        out = _AROUND_NOON.sub(target.isoformat(), out)
    return out
