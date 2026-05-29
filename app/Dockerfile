FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libgeos-dev libproj-dev \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install hatchling && pip install .

COPY app /srv/app
COPY observability /srv/observability
COPY evaluation /srv/evaluation

ENV PYTHONPATH=/srv
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
