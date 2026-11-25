"""Split the test files across CI jobs, deterministically and without gaps.

CI runs the suite as several shard jobs per Python version (see
.github/workflows/ci.yml). Each job asks this script which files are its own:

    python scripts/ci_test_shards.py 2 4     # files of shard 2 of 4, one per line

Every file is in exactly one shard. The split is greedy longest-first over the
per-file seconds in tests/ci_durations.json (recorded from CI's own JUnit
reports, so a shard's share is wall time, not file count). A file with no
recorded time weighs the median, so a new test file lands somewhere sensible
before anyone re-records. Nothing here is random: the same files and the same
table give the same split on every runner.

Guards, because a sharded suite fails quietly if it fails at all:

* a test file pytest would collect but this script would not (one in a
  subdirectory, or named *_test.py) is an error, not a skipped file;
* an empty shard is an error: `pytest` with no file arguments runs the whole
  suite, which would hide a broken split behind a slow green job.

Re-record the table from a CI run's JUnit artifacts (one XML per shard):

    python scripts/ci_test_shards.py --record junit-*.xml
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
DURATIONS = TESTS / "ci_durations.json"


def test_files() -> list[str]:
    """Every file pytest collects from tests/, as repo-relative posix paths."""
    top = sorted(p.relative_to(ROOT).as_posix() for p in TESTS.glob("test_*.py"))
    everywhere = {p.relative_to(ROOT).as_posix()
                  for pattern in ("test_*.py", "*_test.py") for p in TESTS.rglob(pattern)}
    stray = sorted(everywhere - set(top))
    if stray:
        sys.exit("ci_test_shards: test files outside tests/test_*.py would never run "
                 f"in a shard: {stray}. Teach this script about them.")
    if not top:
        sys.exit("ci_test_shards: found no test files")
    return top


def load_weights() -> dict[str, float]:
    if not DURATIONS.exists():
        return {}
    return {k: float(v) for k, v in json.loads(DURATIONS.read_text()).items()}


def split(files: list[str], weights: dict[str, float], total: int) -> list[list[str]]:
    known = [weights[f] for f in files if f in weights]
    default = statistics.median(known) if known else 1.0
    cost = {f: weights.get(f, default) for f in files}
    shards: list[list[str]] = [[] for _ in range(total)]
    load = [0.0] * total
    # Longest first; ties broken by name so the order never depends on the table's order.
    for f in sorted(files, key=lambda f: (-cost[f], f)):
        i = min(range(total), key=lambda i: (load[i], i))
        shards[i].append(f)
        load[i] += cost[f]
    return [sorted(s) for s in shards]


def record(reports: list[Path]) -> None:
    seconds: dict[str, float] = defaultdict(float)
    for report in reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            # classname is the dotted module path plus any class: tests.test_x.TestY
            parts = (case.get("classname") or "").split(".")
            if len(parts) < 2 or parts[0] != "tests":
                continue
            seconds[f"tests/{parts[1]}.py"] += float(case.get("time") or 0.0)
    if not seconds:
        sys.exit("ci_test_shards: no test cases in the given reports")
    files = set(test_files())
    table = {f: round(s, 1) for f, s in sorted(seconds.items()) if f in files}
    DURATIONS.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(f"recorded {len(table)} files, {sum(table.values()) / 60:.1f} test-minutes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shard", nargs="?", type=int, help="1-based shard number")
    ap.add_argument("total", nargs="?", type=int, help="number of shards")
    ap.add_argument("--summary", action="store_true", help="print each shard's estimated load")
    ap.add_argument("--record", nargs="+", type=Path, metavar="JUNIT_XML",
                    help="rewrite tests/ci_durations.json from JUnit reports")
    args = ap.parse_args()

    if args.record:
        record(args.record)
        return
    if args.total is None or args.shard is None or not 1 <= args.shard <= args.total:
        ap.error("give SHARD TOTAL with 1 <= SHARD <= TOTAL")

    weights = load_weights()
    files = test_files()
    shards = split(files, weights, args.total)
    assigned = [f for s in shards for f in s]
    assert sorted(assigned) == files and len(assigned) == len(set(assigned)), "split lost a file"
    if any(not s for s in shards):
        sys.exit(f"ci_test_shards: an empty shard among {args.total}; use fewer shards")

    if args.summary:
        default = statistics.median(weights.values()) if weights else 1.0
        for i, s in enumerate(shards, 1):
            est = sum(weights.get(f, default) for f in s)
            print(f"shard {i}/{args.total}: {len(s)} files, ~{est / 60:.1f} test-minutes",
                  file=sys.stderr)
    print("\n".join(shards[args.shard - 1]))


if __name__ == "__main__":
    main()
