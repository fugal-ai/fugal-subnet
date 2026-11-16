#!/usr/bin/env python3
"""Build the v3 code keep-list (fugal_subnet/benchmarks/code_keep_v3.json).

Candidates come from fugal_subnet.benchmarks.code_sets, the SAME functions the pool loader
uses, so a kept test index means the same test at load time. Each candidate's reference
solutions run in the grader's own sandbox (scripts/calibrate_code_tests.py: twice, within a
quarter of the CPU limit; stdin programs also need a unique answer). code_contests is
calibrated in sha256 order of the question id until CC_SAMPLE problems are kept.

Inputs are local copies of the pinned files (downloaded beforehand into their own
directories; UNTRUSTED). Dataset code runs ONLY inside graders_code.run_program.

    uv run python -I scripts/build_code_keep_v3.py \\
        --leetcode .../LeetCodeDataset-train.jsonl --mbppplus .../mbppplus.parquet \\
        --humanevalplus .../humanevalplus.parquet --cc-dir .../code_contests_train \\
        --work results/code_keep_work.jsonl --out results/code_keep_v3.json

Re-runs resume from --work. Promoting --out into the package is a reviewed step (and the
sha256 pin in scripts/check_safety_invariants.py moves with it).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibrate_code_tests import one  # noqa: E402

from fugal_subnet import graders_code as gc  # noqa: E402
from fugal_subnet.benchmarks import code_sets as cs  # noqa: E402


def _rows_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _rows_parquet(paths, columns=None):
    import pyarrow.parquet as pq
    rows = []
    for p in paths:
        rows.extend(pq.read_table(p, columns=columns).to_pylist())
    return rows


def _order(cands):
    return sorted(cands, key=lambda c: hashlib.sha256(cs.question_id(c).encode()).hexdigest())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leetcode", required=True)
    ap.add_argument("--mbppplus", required=True)
    ap.add_argument("--humanevalplus", required=True)
    ap.add_argument("--cc-dir", default="", help="directory holding the pinned train shards (empty: skip code_contests)")
    ap.add_argument("--cc-target", type=int, default=cs.CC_SAMPLE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--work", required=True, help="per-problem results (jsonl, resumable)")
    ap.add_argument("--extra-work", nargs="*", default=[], help="more result files to resume from (read only)")
    ap.add_argument("--only", default="", help="calibrate only these benchmarks (comma list); still writes --out from all results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ok, why = gc.sandbox_selftest()
    if not ok:
        print(f"refusing to run dataset code: sandbox self-test failed ({why})", file=sys.stderr)
        return 2
    cal = argparse.Namespace(margin=0.25, runs=2, max_tests=16, min_tests=cs.MIN_TESTS, scan_tests=40)

    done = {}
    for path in [*args.extra_work, args.work]:
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                done[(r["benchmark"], r["source_id"])] = r
    only = set(filter(None, args.only.split(",")))

    stats = {}
    sources = {
        "code_leetcode": cs.leetcode_candidates(_rows_jsonl(args.leetcode), stats.setdefault("code_leetcode", __import__("collections").Counter())),
        "code_mbppplus": cs.mbppplus_candidates(_rows_parquet([args.mbppplus]), stats.setdefault("code_mbppplus", __import__("collections").Counter())),
        "code_humanevalplus": cs.humanevalplus_candidates(_rows_parquet([args.humanevalplus]), stats.setdefault("code_humanevalplus", __import__("collections").Counter())),
    }
    sources["code_contests"] = []
    stats["code_contests"] = __import__("collections").Counter()
    if args.cc_dir:
        shard_paths = [os.path.join(args.cc_dir, os.path.basename(s)) for s in cs.CC_TRAIN_SHARDS]
        missing = [p for p in shard_paths if not os.path.exists(p)]
        if missing:
            print(f"missing code_contests shards: {missing}", file=sys.stderr)
            return 2
        sources["code_contests"] = cs.codecontests_candidates(
            cs.iter_parquet_rows(shard_paths, list(cs.CC_COLUMNS)), stats["code_contests"])
    for b, cands in sources.items():
        print(f"{b:20s} {len(cands):5d} candidates  {dict(stats[b])}", flush=True)

    t0 = time.monotonic()
    work = open(args.work, "a", encoding="utf-8")

    def run(c):
        key = (c["benchmark"], c["source_id"])
        if key in done:
            return done[key]
        p = {"source": c["benchmark"], "id": c["source_id"], "kind": c["kind"],
             "func": c.get("func"), "tests": c["tests"], "references": c["references"]}
        r = one(p, cal)
        row = {"benchmark": c["benchmark"], "source_id": c["source_id"], "kept": r["kept"],
               "tests": r["tests"], "reference_cpu": r["reference_cpu"],
               "unique_answer": r["unique_answer"], "wall_secs": r["wall_secs"]}
        work.write(json.dumps(row) + "\n")
        work.flush()
        done[key] = row
        return row

    with ThreadPoolExecutor(args.workers) as ex:
        for b in ("code_leetcode", "code_mbppplus", "code_humanevalplus"):
            if only and b not in only:
                continue
            list(ex.map(run, sources[b]))
            print(f"{b} calibrated ({time.monotonic() - t0:.0f} s)", flush=True)
        # code_contests: in sha256 order of the id, batch by batch, until the target is kept.
        order = _order(sources["code_contests"]) if not only or "code_contests" in only else []
        kept_cc, i, batch = 0, 0, args.workers * 8
        while kept_cc < args.cc_target and i < len(order):
            rows = list(ex.map(run, order[i:i + batch]))
            i += batch
            kept_cc = sum(1 for c in order[:i] if done[(c["benchmark"], c["source_id"])]["kept"])
            print(f"code_contests {kept_cc} kept of {i} tried ({time.monotonic() - t0:.0f} s)", flush=True)
    work.close()

    rows = []
    for b in ("code_leetcode", "code_mbppplus", "code_humanevalplus"):
        rows += [done[(c["benchmark"], c["source_id"])] for c in sources[b]
                 if done.get((c["benchmark"], c["source_id"]), {}).get("kept")]
    missing = [b for b in sources if sources[b] and not all((c["benchmark"], c["source_id"]) in done
                                                            for c in sources[b][:1])]
    if missing:
        print(f"NOTE: no results yet for {missing}; --out covers only what is calibrated", flush=True)
    cc_kept = [c for c in _order(sources["code_contests"])
               if (c["benchmark"], c["source_id"]) in done and done[(c["benchmark"], c["source_id"])]["kept"]]
    rows += [done[(c["benchmark"], c["source_id"])] for c in cc_kept[:args.cc_target]]
    out = {"grader": gc.grader_hash(), "margin": cal.margin, "runs": cal.runs,
           "max_tests": cal.max_tests, "min_tests": cal.min_tests,
           "revisions": cs.DATASET_REVISIONS_CODE, "cc_shards": list(cs.CC_TRAIN_SHARDS),
           "rows": sorted(({"benchmark": r["benchmark"], "source_id": r["source_id"], "tests": r["tests"],
                            "reference_cpu": r["reference_cpu"]} for r in rows),
                          key=lambda r: (r["benchmark"], r["source_id"]))}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0, sort_keys=True)
        f.write("\n")
    by = {}
    for r in out["rows"]:
        by[r["benchmark"]] = by.get(r["benchmark"], 0) + 1
    print("kept:", by, "total", len(out["rows"]), f"wall {time.monotonic() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
