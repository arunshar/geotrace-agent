"""Smoke test the running service from outside Docker."""

from __future__ import annotations

import asyncio
import sys

import httpx


async def main(url: str = "http://localhost:8000") -> int:
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(f"{url}/healthz")
        if r.status_code != 200:
            print(f"healthz failed: {r.status_code}")
            return 1
        print(r.json())
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")))
