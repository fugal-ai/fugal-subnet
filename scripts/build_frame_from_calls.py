#!/usr/bin/env python3
"""Build the bootstrap reference frame from direct, logged, re-gradeable model calls.

WHY THIS EXISTS. The frame is per-model evidence (accuracy and token counts
over the pinned pool) and the frontier pays miners in full only once every
hull model has FRONTIER_MIN_TRIALS distinct observations. On the live path
those observations are the forced exploration calls miners make under
attestation, pooled by `aggregate_observations` and folded in by
`accumulate_exploration`, fifteen per epoch at the shipped quota: weeks from a
cold start, days at a raised testnet quota. This script makes the same calls
directly, through the same MeteringProxy, with the same prompts, the same
graders and the same pricing, and folds them in with the same functions, so a
subnet can launch with a warm frame in hours.

WHAT IT IS NOT. Not attested. A live reveal carries a TDX quote; these rows
carry a response hash, the token counts the proxy metered, the provider's
cost, the pool hash, the source hash and the commit. Anyone with the pool and
an API key can re-issue the same (question, model) calls and re-grade; nobody
can prove these particular calls happened inside a TEE. That is the trade:
the shipped bootstrap frame has always been a reviewed, hash-pinned artifact
(check_safety_invariants pins its sha256), and this makes its provenance a
committed JSONL log instead of a directory of someone's reveals. Say so in
the PR that pins the frame. After launch the live exploration keeps folding
in and the bootstrap decays out at FRAME_HALF_LIFE.

WHAT IT MIRRORS, exactly:
  - questions: `select_slice` with a published seed, per model, so the sample
    has the pool's benchmark mix (the observation pass does the same);
  - calls: `fugal_subnet.tee.harness._call_model` through `MeteringProxy`,
    i.e. the miner's own prompt, max_tokens and upstream path;
  - grading: `graders.grade(build_grader_task(q), text, allow_exec=
    HARNESS_ALLOW_EXEC)`, the harness's own line;
  - aggregation: `aggregate_observations` (one voter here) and
    `accumulate_exploration`. By default every row folds in ONE undecayed
    batch, which is how the shipped data/reference_frame.json was built
    (8fa02e2) and what tests/test_frame_from_calls.py proves reproduces its
    pin. `--batch N` folds N rows per simulated epoch instead, so decay
    matches a live run of that length; that is a different frame.

    python scripts/build_frame_from_calls.py --live --api-key-file ~/.fugal/openrouter.key \
        --per-model 300 --max-usd 40                     # [PAID ~$1-2 per model]
    python scripts/build_frame_from_calls.py --build-only  # fold the committed log: the pinned frame

Mock mode needs FUGAL_OPENROUTER_BASE pointing at scripts/stub_upstream.py.
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

DEFAULT_LOG = os.path.join(ROOT, "data", "observations", "bootstrap_frame_calls_v1.jsonl")
DEFAULT_FRAME = os.path.join(ROOT, "data", "reference_frame.json")
DEFAULT_SEED = "fugal-bootstrap-frame-2026-11"
VOTER = "bootstrap-direct-calls"


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
         question_ids: list[str] | None = None) -> list[tuple[str, dict]]:
    """Deterministic, benchmark-stratified sample per model (same machinery as a slice).

    With `question_ids`, every model is called on that same fixed list instead:
    a dense (question x model) block, which is what a router head needs to
    learn from (the frame itself does not care either way).
    """
    from fugal_subnet.benchmarks.slicer import derive_nonce, select_slice

    if question_ids:
        by_id = {q["question_id"]: q for q in pool}
        chosen = [by_id[i] for i in question_ids if i in by_id]
        return [(m, q) for m in models for q in chosen]
    seed_hash = "0x" + hashlib.sha256(seed.encode()).hexdigest()
    out = []
    for model in models:
        nonce = derive_nonce(f"frame:{model}", seed_hash)
        for q in select_slice(nonce, pool, per_model):
            out.append((model, q))
    return out


def already_recorded(path: str) -> set[tuple[str, str]]:
    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    seen.add((row.get("model", ""), row.get("question_id", "")))
    return seen


def record(proxy, scheduled, log_path: str, max_usd: float, concurrency: int,
           provenance: dict) -> dict:
    """Call, grade, append one JSONL row per completed call. Budget enforced per call."""
    import threading

    from fugal_subnet.config import HARNESS_ALLOW_EXEC
    from fugal_subnet.graders import grade
    from fugal_subnet.grading_task import build_grader_task
    from fugal_subnet.tee.harness import _call_model, _request_ctx

    totals = {"rows": 0, "failed": 0, "correct": 0, "cost_usd": 0.0, "provider_cost_usd": 0.0}
    stop = {"budget": False}
    gate = threading.Lock()

    def one(item):
        model, q = item
        with gate:
            if stop["budget"]:
                return None
            spent = sum(r.cost_usd for r in proxy.records)
            if spent >= max_usd:
                stop["budget"] = True
                print(f"budget reached: ${spent:.2f} >= ${max_usd:.2f}; no further calls", flush=True)
                return None
        rid = f"frame:{model}:{q['question_id']}"
        _request_ctx.request_id = rid
        try:
            text = _call_model(proxy, model, q)
        finally:
            _request_ctx.request_id = ""
        mine = [r for r in proxy.records if r.request_id == rid]
        if text is None or not mine:
            return (model, q, None, None)
        # The harness's own grading line: a completed call with no text is a
        # wrong answer; a call that did not complete is dropped (measures the
        # network, not the model), exactly as the frame treats `failed`.
        correct = bool(grade(build_grader_task(q), text, allow_exec=HARNESS_ALLOW_EXEC))
        return (model, q, mine[-1], correct)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=concurrency) as ex:
        for res in ex.map(one, scheduled):
            if res is None:
                continue
            model, q, r, correct = res
            if r is None:
                totals["failed"] += 1
                continue
            row = {
                "model": model,
                "question_id": q["question_id"],
                "benchmark": q.get("benchmark", ""),
                "correct": bool(correct),
                "prompt_tokens": int(r.prompt_tokens),
                "completion_tokens": int(r.completion_tokens),
                "cost_usd": float(r.cost_usd),
                "provider_cost_usd": float(r.provider_cost_usd),
                "response_hash": r.response_hash,
                "recorded_at": int(time.time()),
                **provenance,
            }
            out.write(json.dumps(row, sort_keys=True) + "\n")
            out.flush()
            totals["rows"] += 1
            totals["correct"] += int(bool(correct))
            totals["cost_usd"] += row["cost_usd"]
            totals["provider_cost_usd"] += row["provider_cost_usd"]
    return totals


def build_frame(log_path: str, batch: int = 0):
    """Fold the log into a ReferenceFrame, reproducibly from the committed log.

    Rows are sorted (model, question). `batch` <= 0 folds them all as one
    undecayed batch (the shipped frame's recipe); a positive `batch` groups
    them into epoch-sized batches, each through aggregate_observations (one
    voter) and accumulate_exploration, which applies the per-epoch decay. Rows
    from a different protocol identity are refused: the frame carries the
    identity and a validator refuses a mismatch.
    """
    from fugal_subnet.reference_frame import (
        Observation,
        accumulate_exploration,
        aggregate_observations,
    )
    from fugal_subnet.routing_protocol import identity

    rows = [json.loads(line) for line in open(log_path, encoding="utf-8") if line.strip()]
    ident = identity()
    bad = [r for r in rows if r.get("routing_protocol") and r["routing_protocol"] != ident]
    if bad:
        raise SystemExit(f"{len(bad)} rows were recorded under a different protocol identity; "
                         "the frame must be rebuilt from calls made under the current one")
    rows.sort(key=lambda r: (r["model"], r["question_id"]))
    if batch <= 0:
        batch = max(1, len(rows))
    frame = None
    for i in range(0, len(rows), batch):
        obs = [Observation(question_id=r["question_id"], model=r["model"], correct=bool(r["correct"]),
                           prompt_tokens=int(r["prompt_tokens"]), completion_tokens=int(r["completion_tokens"]),
                           failed=False, coldkey=VOTER) for r in rows[i:i + batch]]
        frame = accumulate_exploration(frame, aggregate_observations(obs))
    return frame, rows


def summarize(frame, rows) -> None:
    from fugal_subnet.config import FRONTIER_MIN_TRIALS

    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    print(f"\n{'model':<40}{'n':>5}{'acc':>7}{'trials':>8}{'in':>7}{'out':>7}{'$':>8}")
    print("-" * 82)
    for m in sorted(by):
        rs = by[m]
        n = len(rs)
        print(f"{m:<40}{n:>5}{sum(r['correct'] for r in rs)/n:>7.3f}{frame.trials.get(m, 0):>8.1f}"
              f"{sum(r['prompt_tokens'] for r in rs)/n:>7.0f}{sum(r['completion_tokens'] for r in rs)/n:>7.0f}"
              f"{sum(r['provider_cost_usd'] for r in rs):>8.3f}")
    print("-" * 82)
    least = min(frame.trials.values()) if frame.trials else 0.0
    print(f"{len(rows)} rows, provider-reported ${sum(r['provider_cost_usd'] for r in rows):.3f}; "
          f"frame epochs {frame.epochs}, least trials {least:.1f} / {FRONTIER_MIN_TRIALS:.0f} "
          f"({'FULL confidence' if least >= FRONTIER_MIN_TRIALS else 'below the floor'})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-model", type=int, default=300)
    ap.add_argument("--models", default=os.path.join(ROOT, "data", "models.json"))
    ap.add_argument("--log", default=DEFAULT_LOG, help="append-only JSONL of calls (committed with the frame)")
    ap.add_argument("--frame-out", default=DEFAULT_FRAME)
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--question-ids", default="",
                    help="file with one question_id per line: call EVERY model on these (dense block)")
    ap.add_argument("--batch", type=int, default=0,
                    help="rows per simulated epoch when folding; 0 (default) folds every row in one "
                         "undecayed batch, the shipped frame's recipe. The default used to be 150, "
                         "which builds a different, decayed frame than the pinned one")
    ap.add_argument("--max-usd", type=float, default=40.0, help="hard cap on this run's priced spend")
    ap.add_argument("--api-key-file", default="", help="0600 file holding the OpenRouter key (--live)")
    ap.add_argument("--live", action="store_true", help="real upstream, real spend [PAID]")
    ap.add_argument("--mock", action="store_true", help="(default) stub upstream via FUGAL_OPENROUTER_BASE")
    ap.add_argument("--build-only", action="store_true", help="no calls; fold the existing log into a frame")
    args = ap.parse_args(argv)

    if not args.build_only:
        if args.live:
            if not args.api_key_file:
                print("--live needs --api-key-file (a 0600 file); the key is never taken from argv", file=sys.stderr)
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
                      "(scripts/stub_upstream.py) or pass --live with --api-key-file", file=sys.stderr)
                return 2
            os.environ.pop("OPENROUTER_API_KEY", None)

        from fugal_subnet.benchmarks.loader import load_all, pool_hash
        from fugal_subnet.config import HARNESS_CONCURRENCY
        from fugal_subnet.routing_protocol import identity
        from fugal_subnet.tee.runtime import MeteringProxy

        models = [r["id"] for r in json.load(open(args.models, encoding="utf-8"))]
        pool = load_all(strict=True)
        qids = [l.strip() for l in open(args.question_ids, encoding="utf-8") if l.strip()] if args.question_ids else None
        scheduled = plan(pool, models, args.per_model, args.seed, qids)
        seen = already_recorded(args.log)
        scheduled = [(m, q) for m, q in scheduled if (m, q["question_id"]) not in seen]
        print(f"{len(models)} models x {args.per_model} questions = {len(models) * args.per_model} calls planned, "
              f"{len(seen)} already recorded, {len(scheduled)} to make; budget ${args.max_usd:.2f} "
              f"({'LIVE' if args.live else 'mock'})", flush=True)
        provenance = {
            "seed": args.seed,
            "pool_hash": pool_hash(pool),
            "routing_protocol": identity(),
            "source_runtime": _source_hash(),
            "script_commit": _git_commit(),
            "upstream": "openrouter" if args.live else os.environ.get("FUGAL_OPENROUTER_BASE", ""),
            "voter": VOTER,
        }
        proxy = MeteringProxy()
        proxy.start()
        try:
            totals = record(proxy, scheduled, args.log, args.max_usd, HARNESS_CONCURRENCY, provenance)
        finally:
            proxy.stop()
        print(f"calls: {totals['rows']} recorded, {totals['failed']} failed (dropped), "
              f"{totals['correct']} correct; table ${totals['cost_usd']:.3f}, "
              f"provider ${totals['provider_cost_usd']:.3f}", flush=True)

    frame, rows = build_frame(args.log, args.batch)
    if frame is None:
        print("no rows to fold", file=sys.stderr)
        return 1
    summarize(frame, rows)
    payload = json.dumps(frame.to_dict(), indent=2, sort_keys=True) + "\n"
    os.makedirs(os.path.dirname(args.frame_out) or ".", exist_ok=True)
    with open(args.frame_out, "w", encoding="utf-8") as f:
        f.write(payload)
    print(f"\nwrote {args.frame_out}\nsha256 {hashlib.sha256(payload.encode()).hexdigest()}  "
          "<- PINNED_BOOTSTRAP_FRAME_SHA256 in scripts/check_safety_invariants.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
