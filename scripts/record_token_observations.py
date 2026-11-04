#!/usr/bin/env python3
"""Record real per-model token counts for the success-contract manifest.

The token manifest (`data/benchmark_tokens_v1.json`) is the cost side of every
miner's routing decision: estimated dollars per query = measured token means x
the pinned prices. The shipped candidate rests on 945 calls from three old proof
bundles, with THREE samples for most models. This script records fresh
observations through the same path the benchmark uses — the MeteringProxy and
the harness's `_call_model`, same `max_tokens`, same prompt shape — on a
deterministic, stratified sample of the pinned pool, and writes rows that
`scripts/build_token_manifest.py` accepts. No response text is kept.

    # $0 dry run against the stub (the default is --mock, which REFUSES to
    # talk to the real upstream):
    python scripts/stub_upstream.py --pool pool.json --port 8799 &
    FUGAL_OPENROUTER_BASE=http://127.0.0.1:8799 \\
      python scripts/record_token_observations.py --per-model 2

    # Real calls [PAID]: the key comes from a 0600 file, never argv or the
    # shell history, and the run stops at --max-usd.
    python scripts/record_token_observations.py --live \\
      --api-key-file ~/.fugal/openrouter.key --per-model 40 --max-usd 12

Re-runs resume: (model, question) pairs already in --out are skipped, so a run
cut short by the budget can be topped up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_OUT = os.path.join(ROOT, "data", "observations", "benchmark_tokens_v2.jsonl")
DEFAULT_SEED = "fugal-token-observations-2026-11"


def _source_hash() -> str:
    import fugal_subnet
    src_dir = os.path.dirname(os.path.abspath(fugal_subnet.__file__))
    h = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(src_dir)):
        for fname in sorted(files):
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                h.update(fpath[len(src_dir):].encode())
                h.update(open(fpath, "rb").read())
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort outside a checkout
        return ""


def plan(pool: list[dict], models: list[str], per_model: int, seed: str,
         categories: dict[str, str] | None = None,
         targets: dict[str, float] | None = None) -> list[tuple[str, dict]]:
    """A deterministic, benchmark-stratified sample per model.

    Same nonce machinery as an epoch slice, so the sample has the pool's shape
    (MMLU-heavy, capped by SLICE_SHARE_FACTOR) rather than whatever the first
    K rows happen to be. Different models get different questions, which is
    what the live benchmark does too.
    """
    from fugal_subnet.benchmarks.slicer import derive_nonce, select_slice, select_slice_v2

    seed_hash = "0x" + hashlib.sha256(seed.encode()).hexdigest()
    out = []
    for model in models:
        nonce = derive_nonce(f"obs:{model}", seed_hash)
        # Protocol v2: the sample has the v2 slice's category mix (category-first,
        # largest remainder), so the token means describe what a v2 epoch asks.
        sample = (select_slice_v2(nonce, pool, per_model, targets, categories)
                  if categories else select_slice(nonce, pool, per_model))
        for q in sample:
            out.append((model, q))
    return out


def already_recorded(path: str) -> set[tuple[str, str]]:
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                seen.add((row.get("model", ""), row.get("question_id", "")))
    return seen


def record(proxy, scheduled, out_path: str, max_usd: float, concurrency: int,
           provenance: dict, reasoning: dict | None = None) -> dict:
    """Make the calls through the proxy, append one JSONL row per completed call."""
    import threading

    from fugal_subnet.tee.harness import _call_model, _request_ctx

    totals = {"rows": 0, "failed": 0, "cost_usd": 0.0, "provider_cost_usd": 0.0}
    stop = {"budget": False}
    gate = threading.Lock()

    def one(item):
        model, q = item
        # The budget is enforced where the money is spent: by the calling
        # thread, against the proxy's own priced records, before each call.
        # A check in the consumer loop ran behind the workers and let a whole
        # batch through after the cap (caught by the test).
        with gate:
            if stop["budget"]:
                return None
            spent = sum(r.cost_usd for r in proxy.records)
            if spent >= max_usd:
                stop["budget"] = True
                print(f"budget reached: ${spent:.2f} >= ${max_usd:.2f}; no further calls",
                      flush=True)
                return None
        rid = f"obs:{model}:{q['question_id']}"
        _request_ctx.request_id = rid
        try:
            text = _call_model(proxy, model, q, reasoning=(reasoning or {}).get(model))
        finally:
            _request_ctx.request_id = ""
        mine = [r for r in proxy.records if r.request_id == rid]
        if text is None or not mine:
            return (model, q["question_id"], None)
        r = mine[-1]
        return (model, q["question_id"], r)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=concurrency) as ex:
        for res in ex.map(one, scheduled):
            if res is None:
                continue
            model, qid, r = res
            if r is None:
                totals["failed"] += 1
                continue
            row = {
                "model": model,
                "prompt_tokens": int(r.prompt_tokens),
                "completion_tokens": int(r.completion_tokens),
                "cost_usd": float(r.cost_usd),
                "provider_cost_usd": float(r.provider_cost_usd),
                "question_id": qid,
                "response_hash": r.response_hash,
                "recorded_at": int(time.time()),
                **provenance,
            }
            out.write(json.dumps(row, sort_keys=True) + "\n")
            out.flush()
            totals["rows"] += 1
            totals["cost_usd"] += row["cost_usd"]
            totals["provider_cost_usd"] += row["provider_cost_usd"]
    return totals


def summarize(path: str) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    print(f"\n{'model':<40}{'n':>5}{'in':>8}{'out':>8}{'$table':>9}{'$provider':>11}")
    print("-" * 81)
    for m in sorted(by):
        rs = by[m]
        n = len(rs)
        print(f"{m:<40}{n:>5}{sum(r['prompt_tokens'] for r in rs)/n:>8.0f}"
              f"{sum(r['completion_tokens'] for r in rs)/n:>8.0f}"
              f"{sum(r['cost_usd'] for r in rs):>9.3f}{sum(r['provider_cost_usd'] for r in rs):>11.3f}")
    print("-" * 81)
    print(f"{len(rows)} rows, table cost ${sum(r['cost_usd'] for r in rows):.3f}, "
          f"provider-reported ${sum(r['provider_cost_usd'] for r in rows):.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-model", type=int, default=40)
    ap.add_argument("--models", default=os.path.join(ROOT, "data", "models.json"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--max-usd", type=float, default=12.0,
                    help="Stop scheduling calls once the table-priced total reaches this")
    ap.add_argument("--api-key-file", default="",
                    help="0600 file holding the OpenRouter key (live only); never argv")
    ap.add_argument("--live", action="store_true",
                    help="Call the real upstream [PAID]. Default is --mock, which requires "
                         "FUGAL_OPENROUTER_BASE to name a stub and refuses the real endpoint.")
    ap.add_argument("--mock", action="store_true", help="(default)")
    ap.add_argument("--protocol", type=int, default=1, choices=(1, 2),
                    help="2: send each model's pinned `reasoning` setting (from --models) and price "
                         "with that table, as a v2 miner does")
    ap.add_argument("--pool-file", default="",
                    help="JSON list of questions to sample from instead of the pinned pool")
    ap.add_argument("--categories", default="",
                    help="JSON {benchmark: category}; sample each model at the v2 category targets")
    args = ap.parse_args(argv)

    if args.live:
        if not args.api_key_file:
            print("--live needs --api-key-file (a 0600 file); the key is never taken from argv",
                  file=sys.stderr)
            return 2
        with open(args.api_key_file, encoding="utf-8") as f:
            key = f.read().strip()
        if not key:
            print("empty key file", file=sys.stderr)
            return 2
        os.environ["OPENROUTER_API_KEY"] = key   # the proxy reads it; nothing prints it
        del key
    else:
        base = os.environ.get("FUGAL_OPENROUTER_BASE", "")
        if not base or "openrouter.ai" in base:
            print("mock mode refuses the real upstream: set FUGAL_OPENROUTER_BASE to a stub "
                  "(scripts/stub_upstream.py) or pass --live with --api-key-file",
                  file=sys.stderr)
            return 2
        os.environ.pop("OPENROUTER_API_KEY", None)

    # Imports after the environment is settled: runtime.py reads the upstream
    # base at import time.
    from fugal_subnet.benchmarks.loader import load_all, pool_hash
    from fugal_subnet.config import HARNESS_CONCURRENCY
    from fugal_subnet.tee.runtime import MeteringProxy

    table = json.load(open(args.models, encoding="utf-8"))
    models = [r["id"] for r in table]
    reasoning = {r["id"]: r["reasoning"] for r in table if r.get("reasoning")} if args.protocol == 2 else None
    pool = (json.load(open(args.pool_file, encoding="utf-8")) if args.pool_file else load_all(strict=True))
    categories = json.load(open(args.categories, encoding="utf-8")) if args.categories else None
    from fugal_subnet.config import CATEGORY_TARGETS_V2
    scheduled = plan(pool, models, args.per_model, args.seed,
                     categories=categories, targets=CATEGORY_TARGETS_V2 if categories else None)
    seen = already_recorded(args.out)
    scheduled = [(m, q) for m, q in scheduled if (m, q["question_id"]) not in seen]
    print(f"{len(models)} models x {args.per_model} questions = {len(models) * args.per_model} "
          f"calls planned, {len(seen)} already recorded, {len(scheduled)} to make; "
          f"budget ${args.max_usd:.2f} ({'LIVE' if args.live else 'mock'})", flush=True)
    if not scheduled:
        summarize(args.out)
        return 0

    provenance = {
        "pool_hash": pool_hash(pool),
        "source_runtime": _source_hash(),
        "script_commit": _git_commit(),
        "upstream": "openrouter" if args.live else os.environ.get("FUGAL_OPENROUTER_BASE", ""),
        "seed": args.seed,
        "protocol": args.protocol,
    }
    proxy = MeteringProxy()
    from fugal_subnet.api import load_prices
    proxy.prices = load_prices(args.models)     # price with the table being measured, not the v1 default
    proxy.start()
    try:
        totals = record(proxy, scheduled, args.out, args.max_usd, HARNESS_CONCURRENCY, provenance,
                        reasoning=reasoning)
    finally:
        proxy.stop()
    print(f"recorded {totals['rows']} rows ({totals['failed']} failed calls), table cost "
          f"${totals['cost_usd']:.3f}, provider-reported ${totals['provider_cost_usd']:.3f}")
    summarize(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
