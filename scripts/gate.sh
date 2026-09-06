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
#   2. The steps were a hand-copied SUBSET. CONTRIBUTING.md still lists three
#      commands; CI runs eight, in two different environments.
#
# So the steps are not written here at all -- they are READ OUT OF
# .github/workflows/ci.yml. A step added to CI is picked up on the next run
# with no edit to this file, which is the only version of "keep them in sync"
# that survives contact with a repo three people are editing.
#
# Usage:
#   scripts/gate.sh          # both environments, every step
#   scripts/gate.sh test     # just CI's `test` job (no DCAP verifier)
#   scripts/gate.sh tee      # just CI's `tee-verifier` job
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
  # Emit one `uv run ...` command per line for the named job in ci.yml.
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
        if line.startswith("uv run") and "uv sync" not in line:
            print(line)
PY
}

run_job() {           # run_job <ci job name> <label> <uv sync args...>
  local job="$1" label="$2"; shift 2
  echo ""
  echo "================================================================"
  echo "  $label"
  echo "  environment: uv sync --locked $*"
  echo "================================================================"
  if ! uv sync --locked "$@" >/dev/null 2>&1; then
    echo "FAIL  could not create the environment"; FAILED=1; return
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
  done < <(steps_for "$job")
}

if [ "$WHICH" = "all" ] || [ "$WHICH" = "test" ]; then
  # CI's `test` job. Deliberately WITHOUT --extra tee: verify_dcap takes a
  # different path when the verifier is absent, and this is the environment
  # that proves the absent path still works.
  run_job test "CI job: test (no DCAP verifier)" --extra dev
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "tee" ]; then
  run_job tee-verifier "CI job: tee-verifier (DCAP verifier installed)" --extra dev --extra tee
fi

# Leave the developer with the fuller environment regardless of what ran, so a
# later bare `pytest` is not silently missing the verifier.
uv sync --locked --extra dev --extra tee >/dev/null 2>&1 || true

echo ""
echo "================================================================"
if [ $FAILED -eq 0 ]; then echo "  GATE GREEN"; else echo "  GATE RED"; fi
echo "================================================================"
exit $FAILED
