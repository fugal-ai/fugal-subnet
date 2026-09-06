#!/usr/bin/env python3
"""A deterministic OpenAI-compatible endpoint, for running miners without spend.

SECURITY NOTE, READ BEFORE USING THIS ANYWHERE REAL. This script is also an
exploit. It substitutes the model layer by setting FUGAL_OPENROUTER_BASE, and a
miner can do exactly the same thing in production: point it at a server that
returns each question's own gold answer with two tokens of usage, and collect
perfect accuracy at near-zero attested cost with every hash binding, the DCAP
signature and the approved measurement still checking out.

That is not a flaw in this file — it is a flaw in the fact that the upstream is
a runtime environment variable rather than part of what the attestation measures.
See docs/INVARIANTS.md, "The model upstream is miner-controlled". Until the
upstream is inside the measured image, a --live subnet is not protected against
the thing this script does for convenience.

Point a miner's metering proxy here with FUGAL_OPENROUTER_BASE and every
production code path still runs — the harness, the real MeteringProxy, real
HTTP, real token accounting, real pricing against the pinned table, real
grading. The only substitution is where the request lands. No API key exists
anywhere in a run that uses this.

    python scripts/stub_upstream.py --pool data/rehearsal/pool_2k.json --port 8799
    FUGAL_OPENROUTER_BASE=http://127.0.0.1:8799 python neurons/miner.py ...

Replies are a pure function of (prompt, model), so two miners asked the same
question of the same model see byte-identical answers, and so do two runs. That
is what makes cross-validator agreement assertions mean something rather than
being a coincidence of timing.

MODEL SKILL IS NOT UNIFORM, and that is the point. A stub where every model
answers equally well makes routing unmeasurable: thrift would reward always
choosing the cheapest model and quality would never object, so a degenerate
head would score top. Here skill rises with price and the *gap* is widest on
hard benchmarks — cheap models keep up on gsm8k and fall behind on aime. That
is the structure a router has to discover, so a head that learns it genuinely
outscores one that does not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("fugal.stub")

# How hard each benchmark is, and how much price buys you on it. Frontier
# models pull away on reasoning-heavy sets and barely help on arithmetic.
BENCH_DIFFICULTY = {
    "gsm8k": 0.15, "mmlu": 0.30, "ifeval": 0.25,
    "humaneval": 0.45, "math": 0.55, "aime": 0.80, "gpqa": 0.70,
}
DEFAULT_DIFFICULTY = 0.4


def build_skill(prices: dict) -> dict:
    """P(correct) per (model, benchmark), from the model's price rank.

    Rank rather than absolute price: the table spans four orders of magnitude,
    so a linear map would put every model but the top two at the floor.
    """
    ranked = sorted(prices, key=lambda m: prices[m][0] + prices[m][1])
    n = max(1, len(ranked) - 1)
    tier = {m: i / n for i, m in enumerate(ranked)}  # 0.0 cheapest .. 1.0 dearest

    skill: dict[str, dict[str, float]] = {}
    for m in ranked:
        skill[m] = {}
        for bench, diff in list(BENCH_DIFFICULTY.items()) + [("", DEFAULT_DIFFICULTY)]:
            # Easy questions: everyone near ceiling. Hard ones: tier decides.
            ceiling = 0.97 - 0.10 * diff
            floor = ceiling - diff * 0.75
            skill[m][bench] = floor + (ceiling - floor) * tier[m]
    return skill


class Handler(BaseHTTPRequestHandler):
    gold: dict = {}
    bench: dict = {}
    skill: dict = {}
    calls = 0

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            req = json.loads(body)
            model = req.get("model", "")
            prompt = req["messages"][0]["content"]
        except Exception:
            self.send_response(400); self.end_headers()
            self.wfile.write(b'{"error":"bad request"}')
            return

        digest = hashlib.sha256(f"{prompt}|{model}".encode("utf-8")).digest()
        bench = Handler.bench.get(prompt, "")
        p_correct = Handler.skill.get(model, {}).get(bench, 0.5)
        correct = (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF) < p_correct
        text = Handler.gold.get(prompt, "0") if correct else "0"
        Handler.calls += 1

        payload = json.dumps({
            "id": "stub", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            # Varying token counts so cost is not a constant and thrift has
            # something real to measure.
            "usage": {
                "prompt_tokens": 400 + (digest[1] % 200),
                "completion_tokens": 100 + (digest[2] % 100),
                "total_tokens": 500 + (digest[1] % 200) + (digest[2] % 100),
            },
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="Benchmark pool JSON the miners use")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from fugal_subnet.api import load_prices

    with open(args.pool, encoding="utf-8") as f:
        pool = json.load(f)

    # The prompt the model receives is what the harness sends, which is the
    # question's `prompt` field. Key on it directly so no reconstruction can
    # drift from what the harness actually does.
    Handler.gold = {q["prompt"]: str(q.get("gold", "")) for q in pool}
    Handler.bench = {q["prompt"]: q.get("benchmark", "") for q in pool}
    Handler.skill = build_skill(load_prices())

    logger.info("Pool: %d questions | models: %d", len(pool), len(Handler.skill))
    cheap = min(Handler.skill, key=lambda m: Handler.skill[m]["aime"])
    dear = max(Handler.skill, key=lambda m: Handler.skill[m]["aime"])
    logger.info("Skill spread on aime: %s %.2f .. %s %.2f",
                cheap, Handler.skill[cheap]["aime"], dear, Handler.skill[dear]["aime"])
    logger.info("Skill spread on gsm8k: %s %.2f .. %s %.2f",
                cheap, Handler.skill[cheap]["gsm8k"], dear, Handler.skill[dear]["gsm8k"])

    srv = HTTPServer((args.host, args.port), Handler)
    logger.info("Stub upstream on http://%s:%d — no API key, no spend",
                args.host, args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping after %d calls", Handler.calls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
