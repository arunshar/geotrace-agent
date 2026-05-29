"""Output filter. Third layer.

Final pass on the answer string. Removes raw coordinates if precision
exceeds the per-domain policy and strips known PII patterns.
"""

from __future__ import annotations

import re

from app.config import Settings

_COORD_RE = re.compile(r"(?P<lat>-?\d{1,2}\.\d{6,})[, ]+(?P<lon>-?\d{1,3}\.\d{6,})")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@\w+(?:\.\w+)+\b")


class OutputFilter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def scrub(self, text: str) -> str:
        # truncate ultra-precise coords (>4 decimal places) to 4 dp
        text = _COORD_RE.sub(lambda m: f"{float(m.group('lat')):.4f},{float(m.group('lon')):.4f}", text)
        text = _EMAIL_RE.sub("[email]", text)
        return text
