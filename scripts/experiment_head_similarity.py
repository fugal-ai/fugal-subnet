#!/usr/bin/env python3
"""How far apart are independently trained heads, and how close is a nudged copy?

Diligence 2026-10, finding 5. Dedup compared routing decisions at cosine
>= 0.99, and a copier passed by nudging a bias term until three or four of
300 decisions changed. The fix adds a PARAMETER clause: heads whose weights
sit within DEDUP_PARAM_DISTANCE relative Frobenius distance are one head. That
constant has to come from a measurement, not a guess, and this script is the
measurement:

  1. train K heads on the cached SPROUT routing data, each on its own random
     subset of questions with its own epoch count (the repository's own
     trainer, fugal_subnet.success_training.fit -- which is deterministic from
     a zero start, so seeds alone produce identical heads; see the note in
     main());
  2. report every pair's relative distance and routing agreement on a held-out
     slice -- the "independent" band;
  3. take one head and make copies with multiplicative weight noise at several
     scales, reporting the same two numbers -- the "copy" band, with how many
     of the 300 decisions each nudge changed.

The threshold goes at the geometric midpoint of the two bands. Fully offline;
no API spend. Needs data/experiments/sprout_* from
scripts/experiment_routing_dataset.py.

    python scripts/experiment_head_similarity.py --heads 8 --json results/head_similarity.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

from fugal_subnet.head_distance import relative_distance  # noqa: E402
from fugal_subnet.success_training import fit  # noqa: E402
from fugal_subnet.vendor import success_contract as contract  # noqa: E402


def _load():
    from experiment_routing_dataset import load_experiment_data

    hidden, matrix, cost, models, _sources, _raw = load_experiment_data(1.0)
    return hidden.astype(np.float32), matrix.astype(np.float32), cost, list(models)


def _decisions(W, b, hidden, costs):
    p = contract.predictions(W, b, hidden)
    return contract.rank(p, costs, 1.0)[:, 0]


def _agreement(a, b) -> float:
    return float(np.mean(a == b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--slice", type=int, default=300, help="held-out questions for agreement")
    ap.add_argument("--nudges", default="1e-5,1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--seed", type=int, default=1105)
    ap.add_argument("--train-frac", type=float, default=0.6,
                    help="share of the non-held-out questions each head trains on")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    hidden, matrix, cost, models = _load()
    n = len(hidden)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    held = np.sort(perm[:args.slice])
    rest = perm[args.slice:]
    costs = cost.mean(axis=0) if cost.ndim == 2 else np.asarray(cost)
    costs = costs / max(float(costs.max()), 1e-12)   # rank() wants finite nonneg costs

    # FOUND ON THE FIRST RUN: the shipped trainer starts W and b at zero and
    # runs deterministic full-batch descent, so the seed changes nothing and
    # eight heads trained on the same data were byte-identical (distance 0,
    # agreement 1.000). Independence between honest miners therefore comes
    # from DATA -- each miner's own observations of which model answers what
    # -- and from training choices, not from randomness. Each head here trains
    # on its own random `--train-frac` of the questions with its own
    # validation split and its own epoch count, which is the honest-miner
    # shape. Two miners who train the shipped trainer on the same public data
    # really do produce one head, and dedup will treat them as such.
    heads = []
    for k in range(args.heads):
        r = np.random.default_rng(args.seed + k)
        mine = np.sort(r.permutation(rest)[: int(len(rest) * args.train_frac)])
        n_v = max(1, len(mine) // 10)
        v, t = np.sort(mine[:n_v]), np.sort(mine[n_v:])
        epochs = int(args.epochs * (0.5 + r.random()))      # 50%..150% of --epochs
        W, b, _epoch, _hist = fit(hidden, matrix, t, v, epochs=epochs, seed=args.seed + k)
        heads.append({"W": W.astype(np.float32), "b": b.astype(np.float32), "models": models})
        print(f"trained head {k}: {len(t)} questions, {epochs} epochs", flush=True)

    dec = [_decisions(h["W"], h["b"], hidden[held], costs) for h in heads]
    independent = []
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            independent.append({
                "pair": [i, j],
                "distance": relative_distance(heads[i], heads[j]),
                "agreement": _agreement(dec[i], dec[j]),
            })

    copies = []
    base = heads[0]
    for scale in [float(x) for x in args.nudges.split(",")]:
        for rep in range(3):
            noise = rng.standard_normal(base["W"].shape).astype(np.float32)
            copy = {"W": base["W"] * (1 + scale * noise), "b": base["b"].copy(), "models": models}
            d = _decisions(copy["W"], copy["b"], hidden[held], costs)
            copies.append({
                "scale": scale, "rep": rep,
                "distance": relative_distance(base, copy),
                "agreement": _agreement(dec[0], d),
                "decisions_changed": int(np.sum(dec[0] != d)),
            })

    ind_d = [p["distance"] for p in independent]
    ind_a = [p["agreement"] for p in independent]
    print(f"\nindependent heads ({len(independent)} pairs): distance "
          f"min {min(ind_d):.3f} median {np.median(ind_d):.3f} max {max(ind_d):.3f}; "
          f"agreement min {min(ind_a):.3f} median {np.median(ind_a):.3f} max {max(ind_a):.3f}")
    print(f"\n{'nudge':>8}{'distance':>11}{'agreement':>11}{'changed/300':>13}")
    for c in copies:
        print(f"{c['scale']:>8.0e}{c['distance']:>11.4f}{c['agreement']:>11.3f}"
              f"{c['decisions_changed']:>13d}")
    # The gap: the largest copy distance that still agrees >= 0.95 (a copy
    # anyone would bother with) against the smallest independent distance.
    useful = [c["distance"] for c in copies if c["agreement"] >= 0.95]
    if useful:
        lo, hi = max(useful), min(ind_d)
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 10
        print(f"\ncopy band tops out at {lo:.4f} (agreement >= 0.95); independent band starts "
              f"at {hi:.4f}; geometric midpoint {mid:.4f}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"independent": independent, "copies": copies,
                       "heads": args.heads, "epochs": args.epochs, "slice": args.slice},
                      f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
