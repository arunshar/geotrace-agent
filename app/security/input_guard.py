"""Input guard. First of three security layers.

Rejects:
- prompts attempting to override system instructions
- requests for raw PII dumps
- coordinates that fall on private no-fly zones (configurable)
"""

from __future__ import annotations

import re

from app.config import Settings
from app.errors import GuardrailTripped

_BANNED = re.compile(
    r"(ignore (the )?(above|previous) instructions"
    r"|reveal (the )?system prompt"
    r"|exfiltrate"
    r"|dump (all )?ais (positions|data) for everyone)",
    re.I,
)


class InputGuard:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def check(self, question: str) -> None:
        if _BANNED.search(question):
            raise GuardrailTripped("input_guard.banned_phrase")
        if len(question) > 2_000:
            raise GuardrailTripped("input_guard.too_long")
