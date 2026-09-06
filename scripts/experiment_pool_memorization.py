#!/usr/bin/env python3
"""Can a miner satisfy the scoring function by memorising the public pool?

The benchmark pool is public, finite (21,717 questions) and shipped in the
repository. Every question a miner is ever scored on is drawn from it --
`slicer.select_slice` and `exploration.select_explore_set` both sample the
pool and nothing else. So a head that has learned the pool's routing labels by
rote, with no generalisable structure whatsoever, would be scored exactly as
highly as one that learned to route. If that is achievable, the subnet pays
for lookup tables and the incentive measures nothing.

Two things decide it, and they are separate questions:

(1) CAPACITY -- is rote memorisation even representable? The head is a fixed
    linear map: W is (L, 1024) over a frozen Qwen3-0.6B embedding, and the
    routing rule is an argmax. Fitting arbitrary labels to N questions with a
    linear map is possible only while N stays within what 1024 dimensions can
    separate. `capacity` measures where that ceiling actually is, by fitting
    UNIFORMLY RANDOM labels -- which have no structure to generalise, so any
    training accuracy above chance is memorisation and nothing else.

(2) GENERALISATION GAP -- on real routing labels, how much of what a head
    learns is pool-specific? `gap` trains on K questions and reports accuracy
    on those same K (in-pool) against held-out questions (out-of-pool). This
    needs the SPROUT ground truth from
    `scripts/experiment_routing_dataset.py`; `capacity` does not, and runs on
    the real pool's own embeddings.

Both are fully offline. No paid API call happens anywhere in this file.

Usage
-----
    # embed the real pool once (GPU strongly advised)
    python scripts/experiment_pool_memorization.py embed-pool --device cuda

    # (1) memorisation capacity, on the real pool
    python scripts/experiment_pool_memorization.py capacity --json results/pool_capacity.json

    # (2) in-pool vs out-of-pool, on SPROUT routing labels
    python scripts/experiment_pool_memorization.py gap --json results/pool_gap.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# isort: off
# Order is load-bearing: numpy and torch read the CPU-dispatch env vars once,
# at import, so determinism has to come first.
import fugal_subnet.determinism  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fugal_subnet.config import HEAD_HIDDEN_DIM  # noqa: E402

logger = logging.getLogger("fugal.experiment.memorization")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "experiments")
POOL_HIDDEN = os.path.join(DATA_DIR, "pool_hidden.npy")
POOL_META = os.path.join(DATA_DIR, "pool_hidden_meta.json")
DEFAULT_POOL = os.path.join(os.path.dirname(__file__), "..", "data", "rehearsal", "pool_full.json")


# -- shared: fit a linear head to hard labels --

def fit_linear_head(
    hidden: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit W, b by cross-entropy -- the most favourable case for memorisation.

    Deliberately harder-working than `train_head.train_sft`: full-batch AdamW,
    no weight decay by default, no early stop. The question is what the
    ARCHITECTURE can represent, so the optimiser must not be the binding
    constraint. Anything it fails to fit here, a miner's trainer cannot fit
    either.
    """
    H = torch.tensor(hidden, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    W = torch.zeros(n_classes, hidden.shape[1], device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(H @ W.T + b, y)
        loss.backward()
        opt.step()
        sched.step()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def _argmax_acc(hidden, W, b, labels) -> float:
    pred = np.argmax(hidden @ W.T + b, axis=1)
    return float((pred == labels).mean())


def fit_multirow_head(
    hidden: np.ndarray,
    labels: np.ndarray,
    n_rows: int,
    n_models: int,
    epochs: int,
    lr: float,
    device: str,
) -> float:
    """Fit a head with MORE ROWS THAN MODELS, and return train accuracy.

    This is the adversary's best head shape, and it is available today.
    `load_head_from_npz` caps rows at HEAD_MAX_MODELS = 64 but never checks
    that the model names are distinct -- verified: a head with 64 rows all
    naming `openai/gpt-4o` loads without complaint. Since the routing rule is
    `argmax` over rows and the row's name is what gets called, R rows over L
    models is a piecewise-linear partition with R regions instead of L, which
    is strictly more memorisation capacity for the same L models.

    Trained through a log-sum-exp over each model's rows, the smooth version of
    "the best of this model's rows wins".
    """
    H = torch.tensor(hidden, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    group = torch.arange(n_rows, device=device) % n_models
    W = torch.zeros(n_rows, hidden.shape[1], device=device, requires_grad=True)
    b = torch.zeros(n_rows, device=device, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        logits = H @ W.T + b
        grouped = torch.stack(
            [torch.logsumexp(logits[:, group == k], dim=1) for k in range(n_models)],
            dim=1,
        )
        F.cross_entropy(grouped, y).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pred = group[(H @ W.T + b).argmax(dim=1)].cpu().numpy()
    return float((pred == labels.astype(pred.dtype)).mean())


def gaussian_control(n: int, dim: int, seed: int) -> np.ndarray:
    """Isotropic unit-norm vectors: the easiest geometry a head could face.

    The upper bound on memorisation. Real embeddings of real questions cluster
    -- every MMLU item looks like every other MMLU item -- and clustered points
    are harder to separate arbitrarily, so whatever the pool measures must sit
    at or below this.
    """
    rng = np.random.RandomState(seed)
    h = rng.randn(n, dim).astype(np.float32)
    return h / np.linalg.norm(h, axis=1, keepdims=True)


# -- embed-pool --

def embed_pool(args) -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from experiment_routing_dataset import embed_in_length_order, prewarm_backbone
    prewarm_backbone(args.device, args.dtype)

    with open(args.pool) as fh:
        pool = json.load(fh)

    # `--limit` takes a deterministic RANDOM sample, not a prefix. The pool
    # arrives grouped by benchmark -- aime first, then mmlu -- so a prefix
    # would hand the capacity experiment one domain's questions, which are far
    # more alike than the pool is and would understate how many distinct
    # questions a head can separate.
    idx = list(range(len(pool)))
    if args.limit and args.limit < len(pool):
        idx = sorted(np.random.RandomState(args.seed).choice(
            len(pool), size=args.limit, replace=False).tolist())
    prompts = [pool[i]["prompt"] for i in idx]

    logger.info("embedding %d of %d pool prompts on %s",
                len(prompts), len(pool), args.device)
    t0 = time.time()
    hidden = embed_in_length_order(
        prompts, args.device, args.batch_size, args.max_length,
    )
    logger.info("done in %.1fs -> %s", time.time() - t0, hidden.shape)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(POOL_HIDDEN, hidden.astype(np.float32))
    with open(POOL_META, "w") as fh:
        json.dump({
            "pool": os.path.abspath(args.pool),
            "pool_size": len(pool),
            "n": len(prompts),
            "index": idx,
            "device": args.device,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "benchmarks": [pool[i]["benchmark"] for i in idx],
        }, fh)
    logger.info("wrote %s", POOL_HIDDEN)


# -- capacity --

def capacity(args) -> dict:
    if args.gaussian:
        n_total, d = max(args.sizes), HEAD_HIDDEN_DIM
        hidden = gaussian_control(n_total, d, args.seed)
        meta = {"pool": "isotropic gaussian control (upper bound)"}
        logger.info("gaussian control: %d x %d", n_total, d)
    else:
        hidden = np.load(POOL_HIDDEN, allow_pickle=False)
        with open(POOL_META) as fh:
            meta = json.load(fh)
        n_total, d = hidden.shape
        assert d == HEAD_HIDDEN_DIM
        logger.info("pool embeddings: %d x %d", n_total, d)

    rows = []
    for n in args.sizes:
        if n > n_total:
            logger.info("skipping N=%d (only %d embedded)", n, n_total)
            continue
        for L in args.widths:
            rng = np.random.RandomState(args.seed + n + L)
            idx = rng.choice(n_total, size=n, replace=False)
            H = hidden[idx]
            y = rng.randint(0, L, size=n)
            t0 = time.time()
            W, b = fit_linear_head(H, y, L, args.epochs, args.lr, args.device)
            acc = _argmax_acc(H, W, b, y)
            rows.append({
                "n_questions": n, "n_models": L, "n_rows": L,
                "train_accuracy_on_random_labels": acc,
                "chance": 1.0 / L,
                "memorised_fraction": (acc - 1.0 / L) / (1.0 - 1.0 / L),
                "seconds": round(time.time() - t0, 1),
            })
            logger.info("N=%-6d L=%-3d rows=%-3d  train acc on random labels "
                        "%.3f (chance %.3f)  [%.1fs]", n, L, L, acc, 1.0 / L,
                        rows[-1]["seconds"])

            # The duplicate-row variant: same L models, more rows. Capacity is
            # monotone in rows by construction (extra rows can be zeroed), so a
            # DROP as rows grow is the optimiser giving up, not the ceiling --
            # read such a row as a lower bound.
            for R in args.rows:
                if R <= L:
                    continue
                t0 = time.time()
                acc_r = fit_multirow_head(H, y, R, L, args.epochs, args.lr,
                                          args.device)
                rows.append({
                    "n_questions": n, "n_models": L, "n_rows": R,
                    "train_accuracy_on_random_labels": acc_r,
                    "chance": 1.0 / L,
                    "memorised_fraction": (acc_r - 1.0 / L) / (1.0 - 1.0 / L),
                    "seconds": round(time.time() - t0, 1),
                })
                logger.info("N=%-6d L=%-3d rows=%-3d  train acc on random "
                            "labels %.3f (chance %.3f)  [%.1fs]", n, L, R,
                            acc_r, 1.0 / L, rows[-1]["seconds"])

    return {"meta": {"pool": meta.get("pool"), "n_embedded": n_total,
                     "hidden_dim": d, "epochs": args.epochs,
                     "lr": args.lr, "gaussian": bool(args.gaussian)},
            "rows": rows}


# -- gap --

def gap(args) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from experiment_routing_dataset import load_experiment_data

    hidden, matrix, cost, models, _sources, _raw = load_experiment_data(args.threshold)
    n, m = matrix.shape
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_hold = min(args.holdout, n // 3)
    hold = perm[:n_hold]
    rest = perm[n_hold:]

    # Label = the cheapest model that answered correctly; questions no model
    # answered are dropped, exactly as the scoring path drops them.
    def labels_for(idx):
        keep, lab = [], []
        for i in idx:
            ok = np.where(matrix[i] == 1)[0]
            if len(ok):
                keep.append(i)
                lab.append(int(ok[np.argmin(cost[i, ok])]))
        return np.array(keep), np.array(lab)

    hold_idx, hold_lab = labels_for(hold)

    # Two very different questions, and reporting only the first would mislead.
    #
    # LABEL MATCH asks "did the head name the same model the oracle would
    # have". That is a 13-way classification whose majority class is already
    # 0.27, and getting it wrong is not necessarily a routing mistake.
    #
    # ROUTED CORRECT asks "did the model the head chose answer the question".
    # That is what the subnet scores -- `proof.accuracy` counts a route correct
    # whether or not it was the CHEAPEST correct one. A head can miss the label
    # on almost every question and still route well.
    def routed_correct(idx, W, b) -> float:
        pred = np.argmax(hidden[idx] @ W.T + b, axis=1)
        return float(matrix[idx, pred].mean())

    majority = float(np.bincount(hold_lab, minlength=m).max() / len(hold_lab))
    logger.info("holdout: %d questions; majority-class label baseline %.3f, "
                "uniform chance %.3f", len(hold_idx), majority, 1.0 / m)

    rows = []
    for k in args.sizes:
        take = rest[:k]
        tr_idx, tr_lab = labels_for(take)
        if len(tr_idx) < 10:
            continue
        W, b = fit_linear_head(hidden[tr_idx], tr_lab, m, args.epochs, args.lr,
                               args.device, weight_decay=args.weight_decay)
        in_acc = _argmax_acc(hidden[tr_idx], W, b, tr_lab)
        out_acc = _argmax_acc(hidden[hold_idx], W, b, hold_lab)
        in_routed = routed_correct(tr_idx, W, b)
        out_routed = routed_correct(hold_idx, W, b)
        rows.append({
            "n_train": int(len(tr_idx)),
            "in_pool_label_match": in_acc,
            "out_of_pool_label_match": out_acc,
            "label_match_gap": in_acc - out_acc,
            "in_pool_routed_correct": in_routed,
            "out_of_pool_routed_correct": out_routed,
            "routed_correct_gap": in_routed - out_routed,
        })
        logger.info("K=%-6d  label match  in %.3f / out %.3f (gap %+.3f)   "
                    "routed correct  in %.3f / out %.3f (gap %+.3f)",
                    len(tr_idx), in_acc, out_acc, in_acc - out_acc,
                    in_routed, out_routed, in_routed - out_routed)
    return {"meta": {"n_questions": n, "n_models": m,
                     "n_holdout": int(len(hold_idx)), "epochs": args.epochs,
                     "weight_decay": args.weight_decay,
                     "majority_class_label_baseline": majority,
                     "uniform_chance": 1.0 / m},
            "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed-pool")
    e.add_argument("--pool", type=str, default=DEFAULT_POOL)
    e.add_argument("--device", type=str, default="cpu")
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--max-length", type=int, default=512)
    e.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--seed", type=int, default=1105)
    e.set_defaults(func=embed_pool)

    c = sub.add_parser("capacity")
    c.add_argument("--sizes", type=int, nargs="+",
                   default=[250, 500, 1000, 2000, 4000, 8000, 16000, 21717])
    c.add_argument("--widths", type=int, nargs="+", default=[8],
                   help="distinct models the head declares")
    c.add_argument("--rows", type=int, nargs="+", default=[16, 32, 64],
                   help="head rows > models, i.e. duplicate model names; "
                        "HEAD_MAX_MODELS caps this at 64")
    c.add_argument("--gaussian", action="store_true",
                   help="run against isotropic gaussian vectors instead of the "
                        "pool -- the upper bound on what any embedding allows")
    c.add_argument("--epochs", type=int, default=3000)
    c.add_argument("--lr", type=float, default=0.05)
    c.add_argument("--device", type=str, default="cpu")
    c.add_argument("--seed", type=int, default=1105)
    c.add_argument("--json", type=str, default="")
    c.set_defaults(func=capacity)

    g = sub.add_parser("gap")
    g.add_argument("--sizes", type=int, nargs="+",
                   default=[100, 250, 500, 1000, 2000, 4000, 8000])
    g.add_argument("--holdout", type=int, default=3000)
    g.add_argument("--threshold", type=float, default=1.0)
    g.add_argument("--epochs", type=int, default=3000)
    g.add_argument("--lr", type=float, default=0.05)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--device", type=str, default="cpu")
    g.add_argument("--seed", type=int, default=1105)
    g.add_argument("--json", type=str, default="")
    g.set_defaults(func=gap)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.func(args)
    if out and getattr(args, "json", ""):
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        logger.info("wrote %s", args.json)


if __name__ == "__main__":
    main()
