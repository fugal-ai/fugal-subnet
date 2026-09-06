#!/usr/bin/env python3
"""Build the offline routing ground truth this repo does not otherwise have.

Why this exists
---------------
The subnet's premise is that a learned routing head sends a question to a
better model than chance does. Nothing in this repository can test that: a
proof records only the *one* model a miner routed to, so there is no
question-by-model correctness matrix anywhere in `results/`, and producing one
would mean calling every model on every question -- real money, which the
project's rules forbid without explicit approval.

So the matrix comes from published data instead. `CARROT-LLM-Routing/SPROUT`
(HuggingFace, parquet, free) scored 13 hosted LLMs on ~44k prompts drawn from
MMLU-Pro, MATH, RAGBench and OpenHermes, and records per model a correctness
score in [0, 1] plus input and output token counts. That is exactly the shape
`head_eval.evaluate_head` wants, and the prompts are real.

What this buys and what it does not:

- It IS real per-model heterogeneity on real questions, which is the only
  thing that can settle "is routing learnable at all".
- It is NOT this subnet's pool, its models, or its graders. SPROUT's labels
  come from an LLM judge; `graders.py` is mechanical. Read every number this
  produces as evidence about the *mechanism*, not as this subnet's own score.

No paid API call happens here. Downloads are free HuggingFace reads, and only
the columns needed are fetched (parquet column pruning over HTTP range
requests) -- the response and judge-rationale text, which is ~95% of the
580 MB, is never transferred.

Usage
-----
    # fetch the matrix (free, ~5 min)
    python scripts/experiment_routing_dataset.py fetch

    # embed prompts with the subnet's own backbone (GPU strongly advised)
    python scripts/experiment_routing_dataset.py embed --limit 20000 --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

logger = logging.getLogger("fugal.experiment.dataset")

HF_REPO = "CARROT-LLM-Routing/SPROUT"
HF_FILES = [
    "data/train-00000-of-00003.parquet",
    "data/train-00001-of-00003.parquet",
    "data/train-00002-of-00003.parquet",
    "data/validation-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
]

MODELS = [
    "aws-claude-3-5-sonnet-v1",
    "aws-titan-text-premier-v1",
    "openai-gpt-4o",
    "openai-gpt-4o-mini",
    "wxai-granite-3-2b-instruct-8k-max-tokens",
    "wxai-granite-3-8b-instruct-8k-max-tokens",
    "wxai-llama-3-1-70b-instruct",
    "wxai-llama-3-1-8b-instruct",
    "wxai-llama-3-2-1b-instruct",
    "wxai-llama-3-2-3b-instruct",
    "wxai-llama-3-3-70b-instruct",
    "wxai-llama-3-405b-instruct",
    "wxai-mixtral-8x7b-instruct-v01",
]

# ASSUMED, not measured. Public list prices in USD per 1M tokens as of
# 2026-09, (input, output). SPROUT does not ship prices, and the point of the
# cost axis here is that models differ by ~100x -- the exact figures move the
# thrift numbers a little and the ranking not at all. Every experiment that
# uses these also reports a price-sensitivity sweep, and the accuracy results
# do not depend on them at all.
PRICES_USD_PER_MTOK = {
    "aws-claude-3-5-sonnet-v1": (3.00, 15.00),
    "aws-titan-text-premier-v1": (0.50, 1.50),
    "openai-gpt-4o": (2.50, 10.00),
    "openai-gpt-4o-mini": (0.15, 0.60),
    "wxai-granite-3-2b-instruct-8k-max-tokens": (0.10, 0.10),
    "wxai-granite-3-8b-instruct-8k-max-tokens": (0.20, 0.20),
    "wxai-llama-3-1-70b-instruct": (0.60, 0.60),
    "wxai-llama-3-1-8b-instruct": (0.06, 0.06),
    "wxai-llama-3-2-1b-instruct": (0.02, 0.02),
    "wxai-llama-3-2-3b-instruct": (0.03, 0.03),
    "wxai-llama-3-3-70b-instruct": (0.60, 0.60),
    "wxai-llama-3-405b-instruct": (3.50, 3.50),
    "wxai-mixtral-8x7b-instruct-v01": (0.30, 0.30),
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "experiments")
MATRIX_PATH = os.path.join(DATA_DIR, "sprout_matrix.npz")
PROMPTS_PATH = os.path.join(DATA_DIR, "sprout_prompts.json")
HIDDEN_PATH = os.path.join(DATA_DIR, "sprout_hidden.npy")
INDEX_PATH = os.path.join(DATA_DIR, "sprout_embed_index.json")


# -- fetch --

def fetch(args) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    cols = ["key", "dataset", "dataset_level", "prompt"]
    for m in MODELS:
        cols += [f"{m}.score", f"{m}.num_input_tokens", f"{m}.num_output_tokens"]

    fs = HfFileSystem()
    keys: list[str] = []
    datasets: list[str] = []
    prompts: list[str] = []
    scores: list[list[float]] = []
    in_tok: list[list[float]] = []
    out_tok: list[list[float]] = []

    for rel in HF_FILES:
        path = f"datasets/{HF_REPO}/{rel}"
        t0 = time.time()
        with fs.open(path, "rb") as fh:
            tb = pq.read_table(fh, columns=cols)
        logger.info("read %s: %d rows in %.1fs", rel, tb.num_rows, time.time() - t0)

        # Nested selection flattens to positional columns: the four scalars
        # first, then (score, num_input_tokens, num_output_tokens) per model in
        # MODELS order. Index positionally -- the flattened names collide.
        base = 4
        keys += tb.column(0).to_pylist()
        datasets += tb.column(1).to_pylist()
        prompts += tb.column(3).to_pylist()

        per_score = [tb.column(base + 3 * i + 0).to_pylist() for i in range(len(MODELS))]
        per_in = [tb.column(base + 3 * i + 1).to_pylist() for i in range(len(MODELS))]
        per_out = [tb.column(base + 3 * i + 2).to_pylist() for i in range(len(MODELS))]
        for r in range(tb.num_rows):
            scores.append([per_score[i][r] for i in range(len(MODELS))])
            in_tok.append([per_in[i][r] for i in range(len(MODELS))])
            out_tok.append([per_out[i][r] for i in range(len(MODELS))])

    def _f(x, default):
        return default if x is None else float(x)

    S = np.array([[_f(v, np.nan) for v in row] for row in scores], dtype=np.float32)
    IT = np.array([[_f(v, 0.0) for v in row] for row in in_tok], dtype=np.float32)
    OT = np.array([[_f(v, 0.0) for v in row] for row in out_tok], dtype=np.float32)

    # Cost per (question, model) in USD, from the token counts SPROUT recorded
    # and the assumed price table above.
    p_in = np.array([PRICES_USD_PER_MTOK[m][0] for m in MODELS], dtype=np.float64)
    p_out = np.array([PRICES_USD_PER_MTOK[m][1] for m in MODELS], dtype=np.float64)
    cost = (IT * p_in + OT * p_out) / 1e6

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(
        MATRIX_PATH,
        scores=S,
        cost_usd=cost.astype(np.float32),
        in_tokens=IT,
        out_tokens=OT,
        models=np.array(MODELS, dtype="U80"),
        datasets=np.array(datasets, dtype="U64"),
    )
    with open(PROMPTS_PATH, "w") as fh:
        json.dump({"keys": keys, "prompts": prompts}, fh)

    logger.info("wrote %s  (%d questions x %d models)", MATRIX_PATH, S.shape[0], S.shape[1])
    logger.info("nan score cells: %d of %d", int(np.isnan(S).sum()), S.size)
    for i, m in enumerate(MODELS):
        logger.info("  %-45s mean score %.3f   mean $%.6f",
                    m, float(np.nanmean(S[:, i])), float(cost[:, i].mean()))


# -- embed --

def prewarm_backbone(device: str, dtype: str) -> None:
    """Seed the backbone cache with a chosen dtype before embedding.

    `compute_hidden_states` asks `get_backbone` for float16 whenever the device
    is CUDA, and `_model_cache` is keyed on (model, device) alone -- so loading
    the model here first decides the dtype for the whole run without touching
    `backbone.py`.

    Worth having because the default is wrong on some hardware. Measured on a
    GTX 1660 Ti (Turing TU116, no tensor cores): 0.60 TFLOPS in float16
    against 2.66 in float32, so the "fast" dtype is 4.4x slower. float32 is
    also the dtype the validator uses on CPU, so this is closer to the
    reference path, not further from it.
    """
    import torch

    from fugal_subnet.backbone import get_backbone
    from fugal_subnet.config import BACKBONE_MODEL

    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    get_backbone(BACKBONE_MODEL, device, torch_dtype)


def embed_in_length_order(
    prompts: list[str], device: str, batch_size: int, max_length: int,
) -> np.ndarray:
    """Embed with `compute_hidden_states`, batching similar lengths together.

    The tokenizer pads to the longest prompt in each batch, so a batch holding
    one 500-token RAG context and fifteen 40-token maths questions does 12x the
    work it needs to. Sorting by length before batching and undoing the sort
    afterwards removes most of that. Measured on this pool: mean prompt 766
    characters against a 512-token cap, so most of every batch was padding.

    Only the grouping changes. The function doing the work is the repository's
    own `compute_hidden_states`, and attention masks exclude padding, so a
    prompt's embedding does not depend on what shares its batch.
    """
    from fugal_subnet.backbone import compute_hidden_states

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    packed = compute_hidden_states(
        [prompts[i] for i in order], device=device,
        batch_size=batch_size, max_length=max_length,
    )
    out = np.empty_like(packed)
    out[np.asarray(order)] = packed
    return out


def embed(args) -> None:
    prewarm_backbone(args.device, args.dtype)

    with open(PROMPTS_PATH) as fh:
        prompts = json.load(fh)["prompts"]
    n_total = len(prompts)

    with np.load(MATRIX_PATH, allow_pickle=False) as npz:
        sources = [str(s) for s in npz["datasets"]]

    # Drop prompts longer than the real pool's longest before sampling.
    #
    # Two reasons, and the first is the important one. SPROUT carries RAG
    # contexts up to 87,000 characters; the real benchmark pool tops out at
    # 4,891, averaging 396. Keeping the monsters would measure routing on a
    # question population this subnet does not have, and the backbone truncates
    # at 512 tokens anyway, so the head would be reading the first 2% of them.
    # Second, they cost almost all of the compute: 2% of SPROUT is over 5,000
    # characters and every one of those saturates the token cap.
    eligible = [i for i in range(n_total) if len(prompts[i]) <= args.max_chars]
    logger.info("%d of %d prompts are within %d characters (the real pool's max)",
                len(eligible), n_total, args.max_chars)

    # Stratified-by-source subsample, deterministic.
    limit = min(args.limit, len(eligible))
    rng = np.random.RandomState(args.seed)
    by_source: dict[str, list[int]] = {}
    for i in eligible:
        by_source.setdefault(sources[i], []).append(i)
    picked: list[int] = []
    per = max(1, limit // max(len(by_source), 1))
    for s in sorted(by_source):
        idx = np.array(by_source[s])
        picked += list(rng.choice(idx, size=min(per, len(idx)), replace=False))
    remaining = sorted(set(eligible) - set(int(i) for i in picked))
    if len(picked) < limit and remaining:
        picked += list(rng.choice(np.array(remaining),
                                  size=min(limit - len(picked), len(remaining)),
                                  replace=False))
    picked = sorted(int(i) for i in picked)

    logger.info("embedding %d of %d prompts on %s", len(picked), n_total, args.device)
    t0 = time.time()
    hidden = embed_in_length_order(
        [prompts[i] for i in picked], args.device, args.batch_size, args.max_length,
    )
    logger.info("embedded in %.1fs -> %s", time.time() - t0, hidden.shape)

    np.save(HIDDEN_PATH, hidden.astype(np.float32))
    with open(INDEX_PATH, "w") as fh:
        json.dump({
            "index": picked,
            "device": args.device,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "max_chars": args.max_chars,
            "seed": args.seed,
            "prompt_sha256": hashlib.sha256(
                "|".join(prompts[i] for i in picked).encode()
            ).hexdigest(),
        }, fh)
    logger.info("wrote %s and %s", HIDDEN_PATH, INDEX_PATH)


def load_experiment_data(threshold: float = 1.0):
    """Load (hidden, binary matrix, cost, models, sources, raw scores).

    `threshold` binarises SPROUT's [0, 1] judge score. 1.0 means "fully
    correct", which is the closest analogue to this subnet's mechanical
    graders, whose verdict is binary by construction.
    """
    hidden = np.load(HIDDEN_PATH, allow_pickle=False)
    with open(INDEX_PATH) as fh:
        idx = json.load(fh)["index"]
    with np.load(MATRIX_PATH, allow_pickle=False) as npz:
        S = npz["scores"][idx]
        cost = npz["cost_usd"][idx].astype(np.float64)
        models = [str(m) for m in npz["models"]]
        sources = [str(s) for s in npz["datasets"][idx]]
    S = np.nan_to_num(S, nan=0.0)
    matrix = (S >= threshold).astype(np.int8)
    return hidden, matrix, cost, models, sources, S


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download the per-model correctness matrix (free)")
    f.set_defaults(func=fetch)

    e = sub.add_parser("embed", help="compute backbone hidden states for the prompts")
    e.add_argument("--limit", type=int, default=20000)
    e.add_argument("--device", type=str, default="cpu")
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--max-length", type=int, default=512)
    e.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    e.add_argument("--max-chars", type=int, default=5000,
                   help="drop prompts longer than this before sampling; the "
                        "default is just above the real pool's longest (4,891)")
    e.add_argument("--seed", type=int, default=1105)
    e.set_defaults(func=embed)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
