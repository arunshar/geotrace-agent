"""12-factor settings. Read once, immutable thereafter."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GT_", extra="ignore")

    env: Literal["dev", "staging", "prod"] = "dev"
    version: str = "0.1.0"

    # storage
    pg_dsn: str = "postgresql+asyncpg://geotrace:geotrace@postgres:5432/geotrace"
    redis_url: str = "redis://redis:6379/0"
    chroma_url: str = "http://chroma:8000"

    # llms
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    primary_model: str = "claude-sonnet-4-6"
    fallback_model: str = "gpt-4.1-mini"

    # observability
    otel_endpoint: str = "http://otel-collector:4318"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "http://langfuse:3000"

    # budgets (defaults; overridable per-request)
    default_max_tokens: int = 12_000
    default_max_tools: int = 8
    default_max_seconds: float = 30.0

    # kinematics (hard physical bounds)
    vessel_v_max_kts: float = Field(25.0, gt=0)
    vehicle_v_max_kmh: float = Field(130.0, gt=0)

    # cache
    semantic_cache_enabled: bool = True
    semantic_cache_ttl_s: int = 3600
    semantic_cache_similarity: float = 0.92

    # HITL
    hitl_confidence_threshold: float = 0.7


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
