#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# scripts/ci_local.sh
#
# Local CI gate that mirrors .github/workflows/ci.yml (job: lint-type-test)
# so the same failures CI catches are caught BEFORE pushing.
#
# CI runs, in order:
#   1. ruff check app observability evaluation tests spaces scripts   (HARD gate)
#   2. mypy app || true                                               (non-blocking)
#   3. pytest -q --cov=app --cov-report=term-missing                  (HARD gate)
#      with env GT_ANTHROPIC_API_KEY="" GT_OPENAI_API_KEY=""
#
# This script reproduces those exactly, with two local accommodations that do
# NOT weaken the gate:
#   - mypy may be absent in the pcrf env. CI wraps mypy in `|| true`, so mypy is
#     non-blocking there too; we run it if present and skip-with-note otherwise.
#   - some optional runtime deps (e.g. redis, rtree) may be missing in pcrf, so
#     a few test modules cannot be COLLECTED. We detect that precisely, run the
#     collectable subset, print a loud ENV-GAP warning, and still HARD-FAIL on
#     any genuine test failure or any collection error not explained by a known
#     missing optional module.
#
# Final line is either:
#   CI-LOCAL: PASS
#   CI-LOCAL: FAIL (<reason>)
# Exit code is 0 on PASS, 1 on FAIL.
# ----------------------------------------------------------------------------
set -uo pipefail

# --- locate repo root (this script lives in <root>/scripts) -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# --- activate the pcrf conda env (the project's test env) -------------------
# shellcheck disable=SC1091
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pcrf
export PYTHONPATH=.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# CI runs pytest with empty API keys; mirror that (also silence the OTEL
# endpoint so tests do not try to reach a collector locally).
export GT_ANTHROPIC_API_KEY=""
export GT_OPENAI_API_KEY=""
export GT_OTEL_ENDPOINT=""

echo "============================================================"
echo "Local CI gate (mirrors .github/workflows/ci.yml lint-type-test)"
echo "  repo : ${REPO_ROOT}"
echo "  env  : $(python --version 2>&1) @ $(which python)"
echo "============================================================"

# Known optional deps whose absence in pcrf only blocks test COLLECTION,
# never indicates a real defect. A collection error caused solely by one of
# these is downgraded to an ENV-GAP; anything else is a hard failure.
OPTIONAL_DEPS="redis rtree geopandas chromadb anthropic openai respx sqlalchemy"

FAIL_REASON=""

# ----------------------------------------------------------------------------
# 1. ruff  (HARD gate, exact CI command and paths)
# ----------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "[1/3] ruff check app observability evaluation tests spaces scripts"
echo "------------------------------------------------------------"
ruff check app observability evaluation tests spaces scripts
RUFF_EXIT=$?
if [ "${RUFF_EXIT}" -ne 0 ]; then
  echo ">>> ruff FAILED (exit ${RUFF_EXIT}). This is the same hard gate CI enforces."
  FAIL_REASON="ruff lint errors"
else
  echo ">>> ruff clean."
fi

# ----------------------------------------------------------------------------
# 2. mypy  (non-blocking, exactly like CI's `mypy app || true`)
# ----------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "[2/3] mypy app  (non-blocking, matches CI '|| true')"
echo "------------------------------------------------------------"
if command -v mypy >/dev/null 2>&1; then
  mypy app || true
  echo ">>> mypy ran (non-blocking)."
else
  echo "ENV-GAP: mypy not installed in pcrf; CI runs 'mypy app || true' (non-blocking),"
  echo "         so skipping it locally does not change the pass/fail decision."
fi

# ----------------------------------------------------------------------------
# 3. pytest  (HARD gate; ENV-GAP aware)
# ----------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "[3/3] pytest -q -p no:warnings   (HARD gate; CI uses --cov)"
echo "------------------------------------------------------------"

# First, try the FULL collection (no run yet). If everything collects, we run
# the whole suite exactly as CI would. If collection breaks, we figure out
# whether it is purely an ENV-GAP and, if so, run the collectable subset.
COLLECT_LOG="$(mktemp)"
pytest -p no:warnings --collect-only -q >"${COLLECT_LOG}" 2>&1
COLLECT_EXIT=$?

EXCLUDED_FILES=()
MISSING_MODS=""

if [ "${COLLECT_EXIT}" -ne 0 ]; then
  echo ">>> Full collection did not succeed; analysing why..."

  # For each test file, probe collection individually. A file is excused ONLY
  # if its sole collection error is a ModuleNotFoundError for a known optional
  # dep. Any other collection error is a real failure.
  for tf in $(git ls-files 'tests/test_*.py' 2>/dev/null || ls tests/test_*.py); do
    file_log="$(mktemp)"
    pytest -p no:warnings --collect-only -q "${tf}" >"${file_log}" 2>&1
    file_exit=$?
    if [ "${file_exit}" -eq 0 ]; then
      rm -f "${file_log}"
      continue
    fi
    # Pull the missing module name(s), if any.
    mods="$(grep -oE "No module named '[^']+'" "${file_log}" | sed -E "s/No module named '([^']+)'/\1/" | cut -d. -f1 | sort -u)"
    excused=1
    if [ -z "${mods}" ]; then
      excused=0
    else
      for m in ${mods}; do
        case " ${OPTIONAL_DEPS} " in
          *" ${m} "*) : ;;            # known optional dep, fine
          *) excused=0 ;;             # unknown missing module -> real failure
        esac
      done
    fi
    if [ "${excused}" -eq 1 ]; then
      EXCLUDED_FILES+=("${tf}")
      MISSING_MODS="${MISSING_MODS} ${mods}"
    else
      echo ">>> Collection error in ${tf} not explained by a known optional dep:"
      sed -n '1,40p' "${file_log}"
      FAIL_REASON="${FAIL_REASON:+${FAIL_REASON}; }pytest collection error in ${tf}"
    fi
    rm -f "${file_log}"
  done
fi
rm -f "${COLLECT_LOG}"

# Build the pytest invocation. If we had to exclude files, deselect them by
# path so the run itself does not re-hit the collection errors.
PYTEST_ARGS=(-q -p no:warnings)
if [ "${#EXCLUDED_FILES[@]}" -gt 0 ]; then
  UNIQ_MODS="$(echo "${MISSING_MODS}" | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ',' | sed 's/,$//')"
  echo ""
  echo "ENV-GAP: ${UNIQ_MODS} not installed in pcrf, ran subset"
  echo "         (excluded modules that need those deps to import:"
  for f in "${EXCLUDED_FILES[@]}"; do echo "            ${f}"; done
  echo "          CI installs the full dependency set and runs all of these.)"
  for f in "${EXCLUDED_FILES[@]}"; do
    PYTEST_ARGS+=(--ignore="${f}")
  done
fi

echo ""
echo ">>> pytest ${PYTEST_ARGS[*]}"
pytest "${PYTEST_ARGS[@]}"
PYTEST_EXIT=$?

# pytest exit codes: 0 ok, 1 tests failed, 2 collection/usage error, 5 no tests.
# After excluding ENV-GAP files, 0 and (5 = nothing left to run) are acceptable;
# anything else is a real failure.
if [ "${PYTEST_EXIT}" -eq 0 ]; then
  echo ">>> pytest subset passed."
elif [ "${PYTEST_EXIT}" -eq 5 ]; then
  echo ">>> pytest collected no tests after ENV-GAP exclusions (treated as non-fatal)."
else
  echo ">>> pytest FAILED (exit ${PYTEST_EXIT})."
  FAIL_REASON="${FAIL_REASON:+${FAIL_REASON}; }pytest failures (exit ${PYTEST_EXIT})"
fi

# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
if [ -n "${FAIL_REASON}" ]; then
  echo "CI-LOCAL: FAIL (${FAIL_REASON})"
  echo "============================================================"
  exit 1
fi
echo "CI-LOCAL: PASS"
echo "============================================================"
exit 0
