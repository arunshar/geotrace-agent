"""SemanticCache exact-key path."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services.semantic_cache import SemanticCache


@pytest.mark.asyncio
async def test_set_and_get_round_trip() -> None:
    s = get_settings()
    cache = await SemanticCache.connect(s)
    try:
        await cache.set(("test", "k"), {"hello": "world"})
        v = await cache.get(("test", "k"))
        assert v == {"hello": "world"}
    finally:
        await cache.close()
