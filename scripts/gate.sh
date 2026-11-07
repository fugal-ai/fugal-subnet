#!/usr/bin/env bash
#
# Run the gate the way CI runs it, in every environment CI uses.
#
# WHY THIS EXISTS. A pre-built .venv that happened to carry an OPTIONAL extra
# once hid a genuine CI failure through nine consecutive "green" local runs:
# `dcap-qvl` is in `--extra tee`, the tests imported it unconditionally, and
# the local venv had it. The gate was green locally and red on main.
#
# Two things caused that, and this script removes both:
#
#   1. The environment was hand-rolled instead of `uv sync --locked`, so it
#      drifted from the lockfile and from CI's extras.
#   2. The steps were a hand-copied SUBSET of CI's, in fewer environments.
#
# So the steps of the `test`, `tee-verifier` and `quality` jobs are not
# written here at all -- they are READ OUT OF .github/workflows/ci.yml (every
# `uv run` / `uv build` line). A step added to those jobs is picked up on the
# next run with no edit to this file.
#
# What it does NOT run, on purpose:
#   - success-contract: CI runs it with plain `python` and a 1.2 GB backbone
#     download. The two cheap steps (vendor provenance, numeric fixtures) run
#     here, written out below; the backbone steps
#     (verify_success_contract.py, check_success_embeddings.py) do not.
#   - container: `docker build` of the image.
# A change to the backbone path, the vendored contract or the Dockerfile still
# needs CI (or those commands by hand) before it is known green.
#
# Usage:
#   scripts/gate.sh          # every job below
#   scripts/gate.sh test     # just CI's `test` job (no DCAP verifier)
#   scripts/gate.sh tee      # just CI's `tee-verifier` job
#   scripts/gate.sh quality  # just CI's `quality` job (ruff, pip-audit, build)
#   scripts/gate.sh success  # just the two cheap success-contract checks
#
# Exit code is 0 only if every step of every selected job passed.
#
# NOTE ON PIPES. Never pipe a step into `tail` to shorten its output: the pipe
# replaces the command's exit status with tail's, and a FAIL then reads as
# green. That happened during this work too. Output is captured, not piped.
set -uo pipefail
cd "$(dirname "$0")/.."

WHICH="${1:-all}"
FAILED=0

steps_for() {
  # Emit one `uv run ...` / `uv build ...` command per line for the named job in ci.yml.
  uv run --quiet python - "$1" <<'PY'
import sys, pathlib
try:
    import yaml
except ModuleNotFoundError:                       # pragma: no cover
    # Transitive via bittensor/torch rather than declared, so say so plainly
    # instead of failing with a traceback that looks like a repo problem.
    sys.exit("gate.sh needs PyYAML to read ci.yml: uv pip install pyyaml")
job = sys.argv[1]
ci = yaml.safe_load(pathlib.Path(".github/workflows/ci.yml").read_text())
for step in ci["jobs"][job].get("steps", []):
    run = step.get("run")
    if not run:
        continue
    for line in run.strip().splitlines():
        line = line.strip()
        # Only the project's own commands. `uv sync` is the environment and is
        # applied separately; anything else is CI plumbing.
        if line.startswith(("uv run", "uv build")) and "uv sync" not in line:
            print(line)
PY
}

run_steps() {         # run_steps <label> <newline-separated steps> <uv sync args...>
  local label="$1" steps="$2"; shift 2
  echo ""
  echo "================================================================"
  echo "  $label"
  echo "  environment: uv sync --locked $*"
  echo "================================================================"
  if ! uv sync --locked "$@" >/dev/null 2>&1; then
    echo "FAIL  could not create the environment"; FAILED=1; return
  fi
  # Zero steps is a failure, never a pass: a renamed CI job or a broken
  # reader would otherwise print GATE GREEN having run nothing.
  if [ -z "${steps//[[:space:]]/}" ]; then
    echo "FAIL  no steps found"; FAILED=1; return
  fi
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    local out rc
    out=$(eval "$cmd" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then
      printf 'PASS  %s\n' "$cmd"
    else
      printf 'FAIL(%d)  %s\n' "$rc" "$cmd"
      echo "$out" | tail -20 | sed 's/^/        /'
      FAILED=1
    fi
  done <<< "$steps"
}

run_job() {           # run_job <ci job name> <label> <uv sync args...>
  local job="$1" label="$2" steps; shift 2
  if ! steps=$(steps_for "$job"); then
    echo ""; echo "FAIL  could not read the steps of CI job '$job' from ci.yml"; FAILED=1; return
  fi
  run_steps "$label" "$steps" "$@"
}

case "$WHICH" in
  all|test|tee|quality|success) ;;
  *) echo "unknown job '$WHICH' (use: all, test, tee, quality, success)" >&2; exit 2 ;;
esac

if [ "$WHICH" = "all" ] || [ "$WHICH" = "quality" ]; then
  run_job quality "CI job: quality (ruff, pip-audit, build)" --extra dev
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "test" ]; then
  # CI's `test` job. Deliberately WITHOUT --extra tee: verify_dcap takes a
  # different path when the verifier is absent, and this is the environment
  # that proves the absent path still works.
  run_job test "CI job: test (no DCAP verifier)" --extra dev
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "tee" ]; then
  run_job tee-verifier "CI job: tee-verifier (DCAP verifier installed)" --extra dev --extra tee
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "success" ]; then
  # CI's `success-contract` job runs with plain `python`, so these are written
  # out rather than read. Without --materialize-core: that flag git-inits
  # tests/core_snapshot, which only CI's throwaway checkout should do.
  run_steps "CI job: success-contract (cheap checks; no backbone)" \
    "uv run python scripts/check_success_vendor.py
uv run python scripts/check_success_fixtures.py" --extra dev
fi

# Leave the developer with the fuller environment regardless of what ran, so a
# later bare `pytest` is not silently missing the verifier.
uv sync --locked --extra dev --extra tee >/dev/null 2>&1 || true

echo ""
echo "================================================================"
if [ $FAILED -eq 0 ]; then echo "  GATE GREEN"; else echo "  GATE RED"; fi
echo "================================================================"
exit $FAILED
