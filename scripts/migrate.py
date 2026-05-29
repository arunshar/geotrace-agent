"""Forward-only migrations. Each migration is an SQL file in scripts/migrations/.

Production uses Alembic; this scaffold ships a minimal runner so the
docker-compose flow has zero external dependencies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from app.config import get_settings


async def main() -> None:
    s = get_settings()
    conn = await asyncpg.connect(s.pg_dsn.replace("+asyncpg", ""))
    try:
        await conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (id TEXT PRIMARY KEY)")
        applied = {r["id"] for r in await conn.fetch("SELECT id FROM schema_migrations")}
        for path in sorted(Path("scripts/migrations").glob("*.sql")):
            mid = path.name
            if mid in applied:
                continue
            await conn.execute(path.read_text())
            await conn.execute("INSERT INTO schema_migrations (id) VALUES ($1)", mid)
    finally:
        await conn.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
