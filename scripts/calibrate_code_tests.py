#!/usr/bin/env python3
"""Calibrate v3 code questions: keep only tests a reference solution passes with
a wide CPU margin, in the SAME sandbox the grader uses (graders_code.run_program).

Why: grades near a CPU limit can differ between miner TDs. A test the reference
solution passes in <= MARGIN of the limit, twice, leaves an efficient answer a
wide margin on any TD; slower answers can still differ, and shared grades (the
median across coldkeys) absorb that.

Sources (downloaded beforehand into their own directories; UNTRUSTED data):
  --leetcode     newfacade/LeetCodeDataset jsonl (exec_func; reference = `completion`)
  --codecontests deepmind/code_contests parquet (exec_stdio; references = the
                 problem's Python 3 solutions)

Dataset code is executed ONLY through graders_code.run_program (seccomp,
rlimits, no network, no writes). Run it isolated, with paths as arguments:

    uv run python -I scripts/calibrate_code_tests.py \\
        --leetcode /data/lc/lc_test.jsonl --codecontests /data/cc/cc_test.parquet \\
        --limit 200 --out results/code_keep.json

Output: one row per kept problem with the kept test indices and the measured
reference CPU per test. Rows go to --out (never a checked-in file); promoting a
keep-list into the v3 pool manifest is a reviewed step.
"""
import argparse
import ast
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from fugal_subnet import graders_code as gc

FLOATISH = re.compile(rb"^[+-]?\d*\.\d+([eE][+-]?\d+)?$")


# ---------- LeetCodeDataset (exec_func) ----------
def _lc_kwargs(text: str):
    """'n = 7, queries = [[0,5]]' -> {'n': 7, 'queries': [[0, 5]]}, JSON-native only."""
    call = ast.parse(f"f({text})", mode="eval").body
    if not isinstance(call, ast.Call) or call.args:
        raise ValueError("positional or malformed input")
    out = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    if any(k is None for k in out):
        raise ValueError("**kwargs input")
    return out


def _json_native(x) -> bool:
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return True
    if isinstance(x, float):
        return False                      # floats need a tolerance; left out here
    if isinstance(x, (list, tuple)):
        return all(_json_native(v) for v in x)
    if isinstance(x, dict):
        return all(isinstance(k, str) and _json_native(v) for k, v in x.items())
    return False


def leetcode_problems(path: str, limit: int):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit <= 0:
                return
            row = json.loads(line)
            tests = []
            for io in row.get("input_output") or []:
                try:
                    kw = _lc_kwargs(io["input"])
                    want = ast.literal_eval(io["output"])
                except Exception:              # noqa: BLE001 - unparseable test
                    continue
                if _json_native(kw) and _json_native(want):
                    tests.append((kw, json.loads(json.dumps(want))))
            if not tests or not gc.FUNC_NAME.match(row.get("entry_point") or ""):
                continue
            limit -= 1
            yield {"source": "leetcode", "id": row["task_id"], "kind": "exec_func",
                   "func": row["entry_point"], "tests": tests,
                   "references": [row["completion"]]}


# ---------- CodeContests (exec_stdio) ----------
def codecontests_problems(path: str, limit: int):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    for row in table.to_pylist():
        if limit <= 0:
            return
        tests = []
        for group in ("public_tests", "private_tests", "generated_tests"):
            g = row.get(group) or {}
            tests += list(zip(g.get("input") or [], g.get("output") or []))
        sols = row.get("solutions") or {}
        refs = [s for lang, s in zip(sols.get("language") or [], sols.get("solution") or [])
                if lang == 3][:3]                       # 3 = PYTHON3
        if not tests or not refs or row.get("input_file") or row.get("output_file"):
            continue
        if any(FLOATISH.match(tok) for _, o in tests for tok in o.encode().split()):
            continue                                    # float answers: tolerance unknown
        limit -= 1
        yield {"source": "codecontests", "id": row["name"], "kind": "exec_stdio",
               "tests": tests, "references": refs}


# ---------- measurement ----------
def _stdio_pass(results, tests):
    return [r.clean and gc._tokens_match(r.stdout, o.encode(), None)
            for r, (_, o) in zip(results, tests)]


def calibrate_stdio(p, args):
    limit = gc.STDIO_CPU_SECS * args.margin
    tests = p["tests"][: args.scan_tests]
    stdins = [i for i, _ in tests]
    passing_refs, per_test = 0, None
    for ref in p["references"]:
        # Run every test once with the CPU cap at the margin (anything slower
        # is dropped anyway, so there is no point waiting for the full limit),
        # then repeat only the tests that passed.
        good = [True] * len(tests)
        cpu = [0.0] * len(tests)
        for _ in range(args.runs):
            idx = [i for i in range(len(tests)) if good[i]]
            if not idx:
                break
            rs = gc.run_program(ref, [stdins[i] for i in idx], limit)
            for i, r, ok in zip(idx, rs, _stdio_pass(rs, [tests[i] for i in idx])):
                cpu[i] = max(cpu[i], r.cpu_secs)
                good[i] = ok and r.cpu_secs <= limit
        if per_test is None:
            per_test = (good, cpu)
        if sum(good) >= args.min_tests:
            passing_refs += 1
    good, cpu = per_test
    keep = [i for i, g in enumerate(good) if g][: args.max_tests]
    # Two independent references agreeing with the expected outputs is the
    # check that the answer is unique (CodeContests does not flag multi-answer
    # problems). A problem with a single Python reference is kept only if its
    # outputs are a single token per test (yes/no, a count, an index).
    unique = passing_refs >= 2 or all(len(tests[i][1].split()) == 1 for i in keep)
    return keep, [round(cpu[i], 4) for i in keep], unique


def calibrate_func(p, args):
    limit = gc.FUNC_CPU_SECS * args.margin
    tests = p["tests"][: args.scan_tests]
    ref = p["references"][0]
    try:
        program = gc.build_func_program(ref, p["func"])
    except ValueError:
        return [], [], False
    base = min(r.cpu_secs for r in gc.run_program("pass", ["", ""], gc.FUNC_CPU_SECS))
    per_test = []
    for kw, want in tests:
        rs = gc.run_program(program, [json.dumps([kw])] * args.runs, gc.FUNC_CPU_SECS)
        ok = all(r.clean and _func_ok(r, [want]) for r in rs) and len(rs) == args.runs
        per_test.append((ok, max(r.cpu_secs for r in rs) if rs else float("inf")))
    keep, est = [], base
    for i, (ok, c) in enumerate(per_test):
        if ok and len(keep) < args.max_tests and est + max(0.0, c - base) <= limit:
            keep.append(i)
            est += max(0.0, c - base)
    if len(keep) < args.min_tests:
        return keep, [], True
    # confirm the kept set as ONE run, the way exec_func grades it
    inputs = json.dumps([tests[i][0] for i in keep])
    wants = [tests[i][1] for i in keep]
    rs = gc.run_program(program, [inputs] * args.runs, gc.FUNC_CPU_SECS)
    if not all(r.clean and r.cpu_secs <= limit and _func_ok(r, wants) for r in rs):
        return [], [], True
    return keep, [round(per_test[i][1], 4) for i in keep], True


def _func_ok(r, wants):
    try:
        return gc._json_match(gc.parse_func_result(r.stdout), wants, None)
    except Exception:                     # noqa: BLE001
        return False


def one(p, args):
    t0 = time.monotonic()
    if p["kind"] == "exec_stdio":
        keep, cpu, unique = calibrate_stdio(p, args)
    else:
        keep, cpu, unique = calibrate_func(p, args)
    kept = len(keep) >= args.min_tests and unique
    return {"source": p["source"], "id": p["id"], "kind": p["kind"], "kept": kept,
            "tests_scanned": min(len(p["tests"]), args.scan_tests), "tests": keep,
            "reference_cpu": cpu, "unique_answer": unique,
            "wall_secs": round(time.monotonic() - t0, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leetcode", default="")
    ap.add_argument("--codecontests", default="")
    ap.add_argument("--limit", type=int, default=100, help="problems per source")
    ap.add_argument("--margin", type=float, default=0.25, help="max share of the CPU limit")
    ap.add_argument("--runs", type=int, default=2, help="runs per test; all must pass")
    ap.add_argument("--max-tests", type=int, default=16)
    ap.add_argument("--min-tests", type=int, default=5)
    ap.add_argument("--scan-tests", type=int, default=40, help="tests examined per problem")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ok, why = gc.sandbox_selftest()
    if not ok:
        print(f"refusing to run dataset code: sandbox self-test failed ({why})", file=sys.stderr)
        return 2
    problems = []
    if args.leetcode:
        problems += list(leetcode_problems(args.leetcode, args.limit))
    if args.codecontests:
        problems += list(codecontests_problems(args.codecontests, args.limit))
    t0 = time.monotonic()
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(lambda p: one(p, args), problems))
    wall = time.monotonic() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"grader": gc.grader_hash(), "margin": args.margin, "runs": args.runs,
                   "max_tests": args.max_tests, "min_tests": args.min_tests,
                   "workers": args.workers, "wall_secs": round(wall, 1), "rows": rows}, f, indent=1)
    for src in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == src]
        kept = [r for r in rs if r["kept"]]
        per = sum(r["wall_secs"] for r in rs) / max(1, len(rs))
        print(f"{src:13s} {len(kept):4d} kept of {len(rs):4d}  "
              f"median kept tests {sorted(len(r['tests']) for r in kept)[len(kept) // 2] if kept else 0}  "
              f"mean {per:.1f} s/problem (one worker)")
    print(f"wall {wall:.0f} s with {args.workers} workers -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
