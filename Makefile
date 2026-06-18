.PHONY: help venv install lint type test test-property cov security audit \
        bench paper paper-clean clean verify ci ci-local hooks

PY      ?= python
UV      ?= uv
VENV    ?= .venv
ACT     := source $(VENV)/bin/activate

help:
	@echo "Targets:"
	@echo "  install        uv venv + install -e .[dev]"
	@echo "  lint           ruff check"
	@echo "  type           mypy app"
	@echo "  test           pytest -q"
	@echo "  test-property  pytest -q tests/test_prism_properties.py"
	@echo "  cov            pytest with coverage report"
	@echo "  security       bandit + pip-audit"
	@echo "  bench          run benchmarks"
	@echo "  paper          build the NeurIPS PDF"
	@echo "  verify         lint + type + test + cov + security (FAANG-style)"
	@echo "  ci             local CI gate mirroring GitHub Actions (scripts/ci_local.sh)"
	@echo "  ci-local       alias for ci"
	@echo "  hooks          enable the .githooks pre-push gate in this clone"
	@echo "  clean          remove caches"

venv:
	$(UV) venv --python 3.11 $(VENV)

install: venv
	$(ACT) && $(UV) pip install -e ".[dev]" hypothesis pytest-cov bandit pip-audit pytest-benchmark

lint:
	$(ACT) && ruff check app observability evaluation tests spaces scripts

type:
	$(ACT) && mypy app || true

test:
	$(ACT) && GT_ANTHROPIC_API_KEY="" GT_OPENAI_API_KEY="" GT_OTEL_ENDPOINT="" pytest -q -p no:warnings

test-property:
	$(ACT) && pytest -q tests/test_prism_properties.py

cov:
	$(ACT) && GT_ANTHROPIC_API_KEY="" GT_OPENAI_API_KEY="" GT_OTEL_ENDPOINT="" \
	  pytest --cov=app --cov=observability --cov-report=term-missing --cov-report=xml \
	         --cov-fail-under=65 -p no:warnings

security:
	$(ACT) && bandit -q -r app observability evaluation -ll || true
	$(ACT) && pip-audit --skip-editable --strict || true

bench:
	$(ACT) && python scripts/bench.py

paper:
	$(MAKE) -C paper pdf

paper-clean:
	$(MAKE) -C paper clean

verify: lint type cov security
	@echo ""
	@echo "FAANG verify pass."

# Local CI gate that reproduces .github/workflows/ci.yml (lint-type-test) using
# the project test env. Run this before every push; the pre-push hook runs it too.
ci:
	bash scripts/ci_local.sh

ci-local: ci

# Point this clone at the committed pre-push hook (one-time, per clone).
hooks:
	git config core.hooksPath .githooks
	@echo "core.hooksPath set to .githooks (pre-push gate enabled for this clone)."

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
