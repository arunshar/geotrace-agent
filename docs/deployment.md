# Deployment

## Local

```bash
docker compose up --build
python scripts/seed.py
python scripts/healthcheck.py
```

## Kubernetes (production sketch)

- API: 3 replicas behind a Service with HPA on RPS.
- Postgres: managed (Aurora / CloudSQL).
- Redis: managed (ElastiCache / Memorystore).
- Chroma: stateful set with persistent volumes; alternatively swap for
  Pinecone / Weaviate.
- OTEL collector: DaemonSet. Tempo + Grafana hosted.

## Configuration surface

All settings come from env vars prefixed `GT_`. See `app/config.py`.
Production must set:
- `GT_PG_DSN`
- `GT_REDIS_URL`
- `GT_CHROMA_URL`
- `GT_ANTHROPIC_API_KEY` and / or `GT_OPENAI_API_KEY`
- `GT_OTEL_ENDPOINT`
- `GT_LANGFUSE_PUBLIC_KEY` and `GT_LANGFUSE_SECRET_KEY` (optional)

## Rollouts

- Prompts. Version bump (`planner.v3` → `planner.v4`) gated by an
  online evaluator; rollback is `pin: planner.v3` in `prompts.yaml`.
- Models. The TokenOptimizer routes by `settings.primary_model` with
  fallback to `settings.fallback_model`; both are env-driven.
