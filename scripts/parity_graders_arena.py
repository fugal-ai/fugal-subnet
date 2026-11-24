#!/usr/bin/env python3
"""Parity of fugal_subnet/graders_arena.py with RouterArena's own grading.

Two passes, because RouterArena's evaluation needs its own environment:

1. reference (run with RouterArena's venv, from anywhere):
     <ra>/.venv/bin/python -I scripts/parity_graders_arena.py reference \
         --ra-dir <ra> --out <work>/reference.jsonl [--files a.json,b.json] [--limit N]
   For every graded row of the chosen prediction files (non-LiveCodeBench), it grades the
   reply with RouterArena's code exactly as llm_evaluation/run.py does (dataset from the
   global index, metric from config/eval_config, ground truth from the local RouterArena
   dataset) and writes {gid, dataset, gold, reply, ref, stored}. `stored` is the grade the
   prediction file already carries (from whichever RouterArena version graded it).

2. compare (run with this repo's venv):
     uv run python scripts/parity_graders_arena.py compare --ref <work>/reference.jsonl \
         [--out <work>/parity.json]
   Grades every row with graders_arena.score_text and reports agreement per dataset with the
   RouterArena reference (exact for 0/1 metrics, |diff| < 1e-9 for METEOR), plus every
   mismatch class with examples.

RouterArena's data never enters this repo: both passes read it from --ra-dir.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter, defaultdict

DEFAULT_FILES = (
    "divyam-router.json", "BizRouter.json", "auto_router.json", "chayan.json",
    "azure-model-router.json", "fugal-uid132-eco-full.json",
)


def _reply(row):
    g = row.get("generated_result")
    if isinstance(g, str):
        try:
            g = ast.literal_eval(g)
        except Exception:
            return None
    if not isinstance(g, dict) or not g.get("success", True):
        return None
    a = g.get("generated_answer")
    return a if isinstance(a, str) else None


def reference(args) -> int:
    ra = os.path.abspath(args.ra_dir)
    os.chdir(os.path.join(ra, "llm_evaluation"))
    sys.path.insert(0, os.path.join(ra, "llm_evaluation"))
    sys.path.insert(0, ra)
    from datasets import load_from_disk
    from eval_reasoning import get_scorers_for_dataset
    from evaluate_models import ModelEvaluator, load_eval_config_for_dataset

    ds = load_from_disk(os.path.join(ra, "dataset", "routerarena"))
    gold = {r["Global Index"]: r["Answer"] for r in ds}
    ev = ModelEvaluator.__new__(ModelEvaluator)        # only the dataset-name helper is used
    files = args.files.split(",") if args.files else list(DEFAULT_FILES)
    pdir = os.path.join(ra, "router_inference", "predictions")
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for fname in files:
            rows = json.load(open(os.path.join(pdir, fname), encoding="utf-8"))
            if isinstance(rows, dict):
                rows = rows.get("predictions", [])
            seen = 0
            for row in rows:
                gid = row.get("global index") or row.get("global_index")
                if not gid or gid not in gold or str(gid).startswith("LiveCodeBench"):
                    continue
                reply = _reply(row)
                if reply is None:
                    continue
                dataset = ev.determine_dataset_from_global_index(gid)
                scorers = get_scorers_for_dataset(dataset, load_eval_config_for_dataset(dataset))
                scorer, _ = scorers[0]
                try:
                    res = scorer(reply, gold[gid])
                    ref = res[0] if isinstance(res, tuple) else res
                except Exception:
                    ref = 0.0
                stored = row.get("accuracy")
                out.write(json.dumps({"file": fname, "gid": gid, "dataset": gid.split("_")[0],
                                      "ra_metric": scorer.__name__, "gold": gold[gid],
                                      "reply": reply, "ref": float(ref),
                                      "stored": None if stored is None else float(stored)},
                                     ensure_ascii=False) + "\n")
                n += 1
                seen += 1
                if args.limit and seen >= args.limit:
                    break
    print(f"wrote {n} reference rows to {args.out}")
    return 0


def compare(args) -> int:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from fugal_subnet import graders_arena

    per = defaultdict(Counter)
    classes = defaultdict(list)
    stored_agree = Counter()
    metric_of = {}
    with open(args.ref, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ds = r["dataset"]
            metric_of[ds] = r["ra_metric"]
            ours = graders_arena.score_text(ds, r["gold"], r["reply"])
            ok = abs(ours - r["ref"]) < 1e-9
            per[ds]["n"] += 1
            per[ds]["agree"] += ok
            if r["stored"] is not None:
                stored_agree["n"] += 1
                stored_agree["agree"] += abs(r["stored"] - r["ref"]) < 1e-9
            if not ok:
                key = f"{ds}: ref {r['ref']:.3g} ours {ours:.3g}"
                if len(classes[key]) < 3:
                    classes[key].append({"gid": r["gid"], "gold": r["gold"],
                                         "reply_tail": r["reply"][-300:]})
                else:
                    classes[key].append(None)
    table = {ds: {"metric": metric_of[ds], "n": c["n"], "agree": c["agree"],
                  "parity_pct": round(100 * c["agree"] / c["n"], 3)}
             for ds, c in sorted(per.items())}
    total = sum(c["n"] for c in per.values())
    agree = sum(c["agree"] for c in per.values())
    report = {"rows": total, "agree": agree,
              "parity_pct": round(100 * agree / max(total, 1), 4),
              "stored_vs_local_reference_pct": round(
                  100 * stored_agree["agree"] / max(stored_agree["n"], 1), 4),
              "per_dataset": table,
              "mismatch_classes": {k: {"count": len(v), "examples": [e for e in v if e]}
                                   for k, v in sorted(classes.items(), key=lambda kv: -len(kv[1]))}}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"parity {report['parity_pct']}% on {total} rows "
          f"(stored file grades vs local RouterArena: {report['stored_vs_local_reference_pct']}%)")
    for ds, t in table.items():
        print(f"  {ds:28s} {t['metric']:22s} n={t['n']:6d} parity={t['parity_pct']}%")
    for k, v in list(report["mismatch_classes"].items())[:25]:
        print(f"  MISMATCH {k}: {v['count']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--ra-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--files", default="")
    r.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--out", default="")
    args = ap.parse_args()
    return reference(args) if args.cmd == "reference" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
