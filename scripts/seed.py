"""Seed Postgres with minimal AIS schema + a small sample track."""

from __future__ import annotations

import asyncio

import asyncpg

from app.config import get_settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS ais_positions (
  id        BIGSERIAL PRIMARY KEY,
  mmsi      INTEGER NOT NULL,
  lat       DOUBLE PRECISION NOT NULL,
  lon       DOUBLE PRECISION NOT NULL,
  t         TIMESTAMPTZ NOT NULL,
  sog_kts   DOUBLE PRECISION,
  cog_deg   DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ais_mmsi_t_idx ON ais_positions (mmsi, t);
CREATE INDEX IF NOT EXISTS ais_geog_idx
  ON ais_positions USING GIST (ST_MakePoint(lon, lat));

CREATE TABLE IF NOT EXISTS hitl_queue (
  id          BIGSERIAL PRIMARY KEY,
  trace_id    TEXT NOT NULL,
  payload     JSONB NOT NULL,
  status      TEXT NOT NULL DEFAULT 'pending',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cost_ledger (
  id          BIGSERIAL PRIMARY KEY,
  trace_id    TEXT NOT NULL,
  stage       TEXT NOT NULL,
  tokens_in   INT NOT NULL,
  tokens_out  INT NOT NULL,
  cost_usd    NUMERIC(10, 6) NOT NULL,
  cache_hit   BOOLEAN NOT NULL,
  ts          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def main() -> None:
    s = get_settings()
    conn = await asyncpg.connect(s.pg_dsn.replace("+asyncpg", ""))
    try:
        await conn.execute(SCHEMA)
        await conn.execute(
            "INSERT INTO ais_positions (mmsi, lat, lon, t, sog_kts, cog_deg) VALUES "
            "(1234, 56.10, -162.05, NOW() - INTERVAL '6 hours', 12.0, 90.0),"
            "(1234, 56.30, -162.40, NOW() - INTERVAL '0 hours', 10.0, 95.0)"
        )
    finally:
        await conn.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
