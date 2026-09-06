#!/usr/bin/env python3
"""Build cheap, distinct rehearsal heads that genuinely route.

A dress rehearsal needs miners that (a) call only cheap models, so real API
spend stays in cents, (b) route DIFFERENTLY from each other, so behavioural
dedup is exercised without disqualifying anyone, and (c) actually spread their
routing across their models rather than collapsing onto one.

(c) is the part a synthetic-trained head fails. Measured: two heads from
`train_head.py --synthetic` put 99.5% and 100% of real Qwen3 embeddings on the
same single model — the synthetic hidden states share nothing with the real
ones, so the trained bias dominates and both heads become the same constant
policy. Dedup would then cluster them, correctly.

So this builds heads the way `scripts/dress_rehearsal.py::write_head` does — a
random Gaussian W, which routes by embedding direction — and then balances the
bias against REAL embeddings so each model gets roughly an equal share. Two
seeds agree on ~31% of questions, which is chance for three models.

    python scripts/make_rehearsal_heads.py \
        --embeddings data/experiments/pool_hidden.npy \
        --out-dir data/rehearsal

Writes head_cheap_a.npz and head_cheap_b.npz. The models are the three cheapest
in data/models.json; the per-epoch spend at a 300-question slice is about
$0.014 of routing plus the unavoidable ~$0.027 exploration quota.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

CHEAP_MODELS = [
    "deepseek/deepseek-v4-flash",
    "openai/gpt-5.4-nano",
    "meta-llama/llama-4-maverick",
]


def balanced_head(seed: int, hidden: np.ndarray | None, models: list[str],
                  hidden_dim: int, std: float = 0.02):
    rng = np.random.RandomState(seed)
    W = (rng.randn(len(models), hidden_dim) * std).astype(np.float32)
    b = np.zeros(len(models), dtype=np.float32)
    if hidden is None:
        return W, b, None
    logits = hidden @ W.T
    scale = float(np.abs(logits).mean()) or 1.0
    target = 1.0 / len(models)
    for _ in range(300):
        share = np.bincount((logits + b).argmax(1), minlength=len(models)) / len(hidden)
        b -= (0.5 * (share - target) * scale).astype(np.float32)
    return W, b, (logits + b).argmax(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", default="",
                    help="Real backbone hidden states (.npy, N x 1024) to balance against")
    ap.add_argument("--out-dir", default="data/rehearsal")
    ap.add_argument("--models", nargs="+", default=CHEAP_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 202])
    ap.add_argument("--names", nargs="+", default=["a", "b"])
    args = ap.parse_args()

    from fugal_subnet.api import load_prices
    from fugal_subnet.config import HEAD_HIDDEN_DIM
    from fugal_subnet.head_eval import load_head_from_npz

    prices = load_prices()
    unknown = [m for m in args.models if m not in prices]
    if unknown:
        raise SystemExit(f"not in the pinned price table: {unknown}")

    hidden = None
    if args.embeddings:
        hidden = np.load(args.embeddings, allow_pickle=False).astype(np.float32)
        if hidden.shape[1] != HEAD_HIDDEN_DIM:
            raise SystemExit(f"embeddings are {hidden.shape[1]}-d, head needs {HEAD_HIDDEN_DIM}")
        print(f"balancing against {hidden.shape[0]} real embeddings")

    os.makedirs(args.out_dir, exist_ok=True)
    routes = {}
    for name, seed in zip(args.names, args.seeds):
        W, b, idx = balanced_head(seed, hidden, args.models, HEAD_HIDDEN_DIM)
        path = os.path.join(args.out_dir, f"head_cheap_{name}.npz")
        np.savez(path, W=W, b=b, models=np.array(args.models, dtype="U100"))
        data = open(path, "rb").read()
        load_head_from_npz(data)  # the validator's own loader must accept it
        digest = hashlib.sha256(data).hexdigest()
        line = f"{path}: {len(data)} bytes sha256={digest}"
        if idx is not None:
            share = np.bincount(idx, minlength=len(args.models)) / len(idx)
            line += " shares=" + " ".join(f"{m.split('/')[-1]}={s:.3f}"
                                          for m, s in zip(args.models, share))
            routes[name] = idx
        print(line)
    names = list(routes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            agree = float((routes[names[i]] == routes[names[j]]).mean())
            print(f"routing agreement {names[i]} vs {names[j]}: {agree:.3f} "
                  f"(chance for {len(args.models)} models is {1/len(args.models):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
