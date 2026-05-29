"""Conversation state. Bounded history; rolling summary keeps token bills flat."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.models import ConversationState


class ConversationStore:
    def __init__(self) -> None:
        self._db: dict[UUID, ConversationState] = {}

    async def get(self, cid: UUID | None) -> ConversationState:
        if cid is None:
            cs = ConversationState()
            self._db[cs.conversation_id] = cs
            return cs
        return self._db.setdefault(cid, ConversationState(conversation_id=cid))

    async def append_turn(self, cid: UUID, turn: dict[str, Any]) -> ConversationState:
        cs = await self.get(cid)
        cs.history.append(turn)
        if len(cs.history) > 16:
            head = cs.history[:8]
            cs.summary = (cs.summary + "\n" + " | ".join(t.get("text", "")[:80] for t in head)).strip()[:1200]
            cs.history = cs.history[8:]
        return cs
