#!/usr/bin/env python3
"""Reference head trainer: SFT on soft distributions + sep-CMA-ES refinement.

Two-stage training pipeline:
  Stage 1 — SFT: KL divergence loss against soft target distributions, AdamW.
  Stage 2 — sep-CMA-ES: derivative-free optimization on actual routing accuracy.

Usage:
    # With a saved matrix:
    python scripts/train_head.py --matrix data/matrix.npz --models model_a model_b model_c

    # Quick synthetic test (no data needed):
    python scripts/train_head.py --synthetic --n-questions 200 --models openai/gpt-4o-mini google/gemini-2.0-flash-001 anthropic/claude-3.5-haiku

    # Full pipeline with backbone on GPU:
    python scripts/train_head.py --matrix data/matrix.npz --models model_a model_b --device cuda --sft-epochs 50
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# isort: off
# Order is load-bearing — do not let an import sorter reflow this block.
# fugal_subnet.backbone pins the CPU kernel dispatch env vars
# (ATEN_CPU_CAPABILITY, MKL_CBWR, DNNL_MAX_CPU_ISA) that torch reads once, at
# import. Importing torch first makes those pins a no-op and yields embeddings
# that do not match the validator's.
from fugal_subnet.backbone import compute_hidden_states  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
# isort: on

logger = logging.getLogger("fugal.trainer")

from fugal_subnet.config import (
    BACKBONE_MODEL,
    HEAD_HIDDEN_DIM,
    ROUTING_LAMBDA,
    SOFT_TARGET_TAU,
)
from fugal_subnet.soft_targets import compute_soft_targets


def parse_args():
    p = argparse.ArgumentParser(description="Train a Fugal router head")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--matrix", type=str, help="Path to matrix .npz (keys: matrix, questions, models)")
    src.add_argument("--synthetic", action="store_true", help="Generate synthetic data for testing")

    p.add_argument("--models", nargs="+", required=True, help="Model IDs for the head")
    p.add_argument("--output", type=str, default="data/head.npz", help="Output .npz path")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--n-questions", type=int, default=300, help="Number of questions (synthetic mode)")
    p.add_argument("--hidden-dim", type=int, default=HEAD_HIDDEN_DIM)
    p.add_argument("--tau", type=float, default=SOFT_TARGET_TAU)
    p.add_argument("--lam", type=float, default=ROUTING_LAMBDA)

    sft = p.add_argument_group("SFT (Stage 1)")
    sft.add_argument("--sft-epochs", type=int, default=100)
    sft.add_argument("--sft-lr", type=float, default=1e-3)
    sft.add_argument("--sft-batch-size", type=int, default=64)

    cma = p.add_argument_group("sep-CMA-ES (Stage 2)")
    cma.add_argument("--cma-generations", type=int, default=50)
    cma.add_argument("--cma-popsize", type=int, default=32)
    cma.add_argument("--cma-sigma", type=float, default=0.1)
    cma.add_argument("--skip-cma", action="store_true", help="Skip CMA-ES stage")

    bb = p.add_argument_group("Backbone")
    bb.add_argument("--backbone", type=str, default=BACKBONE_MODEL)
    bb.add_argument("--use-backbone", action="store_true",
                    help="Use real backbone for hidden states (downloads model)")
    bb.add_argument("--hidden-states", type=str,
                    help="Path to pre-computed hidden states .npy file")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


def generate_synthetic_data(n_questions: int, n_models: int, hidden_dim: int,
                            seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    matrix = (rng.rand(n_questions, n_models) > 0.4).astype(np.int8)
    for i in range(n_questions):
        if matrix[i].sum() == 0:
            matrix[i, rng.randint(n_models)] = 1
    hidden = rng.randn(n_questions, hidden_dim).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    return matrix, hidden


# ── Stage 1: SFT ──

def train_sft(
    hidden_states: np.ndarray,
    soft_targets: np.ndarray,
    n_models: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """SFT on soft distributions using KL divergence loss."""
    N = hidden_states.shape[0]

    H = torch.tensor(hidden_states, dtype=torch.float32, device=device)
    T = torch.tensor(soft_targets, dtype=torch.float32, device=device)

    W = torch.randn(n_models, hidden_dim, device=device) * 0.01
    b = torch.zeros(n_models, device=device)
    W.requires_grad_(True)
    b.requires_grad_(True)

    optimizer = torch.optim.AdamW([W, b], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_W, best_b = W.detach().clone(), b.detach().clone()

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            h_batch = H[idx]
            t_batch = T[idx]

            logits = h_batch @ W.T + b
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, t_batch, reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W, b], max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_W = W.detach().clone()
            best_b = b.detach().clone()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("  SFT epoch %3d/%d  loss=%.6f  best=%.6f  lr=%.2e",
                        epoch + 1, epochs, avg_loss, best_loss,
                        scheduler.get_last_lr()[0])

    return best_W.cpu().numpy(), best_b.cpu().numpy()


# ── Stage 2: sep-CMA-ES ──

def evaluate_routing(
    W: np.ndarray,
    b: np.ndarray,
    hidden_states: np.ndarray,
    matrix: np.ndarray,
    model_costs: np.ndarray,
    lam: float,
) -> float:
    """Evaluate routing fitness: accuracy - lam * normalized_cost."""
    N = hidden_states.shape[0]
    logits = hidden_states @ W.T + b
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    n_head_models = W.shape[0]
    n_matrix_models = matrix.shape[1]
    use_models = min(n_head_models, n_matrix_models)

    utility = probs[:, :use_models] - lam * model_costs[:use_models]
    selected = np.argmax(utility, axis=1)

    correct = 0
    total_cost = 0.0
    for q in range(N):
        sel = selected[q]
        if sel < n_matrix_models and matrix[q, sel] == 1:
            correct += 1
        if sel < len(model_costs):
            total_cost += model_costs[sel]

    accuracy = correct / max(N, 1)
    avg_cost = total_cost / max(N, 1)
    max_cost = model_costs.max() if len(model_costs) > 0 else 1.0
    normalized_cost = avg_cost / max(max_cost, 1e-10)

    return accuracy - 0.1 * normalized_cost


def train_cma(
    W_init: np.ndarray,
    b_init: np.ndarray,
    hidden_states: np.ndarray,
    matrix: np.ndarray,
    model_costs: np.ndarray,
    lam: float,
    generations: int,
    popsize: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """sep-CMA-ES refinement of head weights."""
    from cmaes import SepCMA

    n_models, hidden_dim = W_init.shape
    theta = np.concatenate([W_init.flatten(), b_init.flatten()])
    dim = len(theta)

    optimizer = SepCMA(mean=theta, sigma=sigma, population_size=popsize)

    best_fitness = -float("inf")
    best_theta = theta.copy()

    for gen in range(generations):
        solutions = []
        for _ in range(popsize):
            candidate = optimizer.ask()
            W_c = candidate[:n_models * hidden_dim].reshape(n_models, hidden_dim).astype(np.float32)
            b_c = candidate[n_models * hidden_dim:].astype(np.float32)
            fitness = evaluate_routing(W_c, b_c, hidden_states, matrix, model_costs, lam)
            solutions.append((candidate, -fitness))  # CMA minimizes

        optimizer.tell(solutions)

        gen_best = -min(s[1] for s in solutions)
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_idx = np.argmin([s[1] for s in solutions])
            best_theta = solutions[best_idx][0].copy()

        if (gen + 1) % 10 == 0 or gen == 0:
            logger.info("  CMA gen %3d/%d  best_fitness=%.4f  gen_best=%.4f",
                        gen + 1, generations, best_fitness, gen_best)

    W_out = best_theta[:n_models * hidden_dim].reshape(n_models, hidden_dim).astype(np.float32)
    b_out = best_theta[n_models * hidden_dim:].astype(np.float32)
    return W_out, b_out


# ── Save ──

def save_head(W: np.ndarray, b: np.ndarray, models: list[str], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    models_arr = np.array(models, dtype="U100")
    np.savez(path, W=W.astype(np.float32), b=b.astype(np.float32), models=models_arr)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    logger.info("Saved head: %s (%d bytes, hash=%s)", path, size, h[:16])
    return h


# ── Main ──

def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_models = len(args.models)
    logger.info("Models (%d): %s", n_models, args.models)
    logger.info("Device: %s", args.device)

    # ── Load or generate data ──
    if args.synthetic:
        logger.info("Generating synthetic data: %d questions × %d models",
                     args.n_questions, n_models)
        matrix, hidden_states = generate_synthetic_data(
            args.n_questions, n_models, args.hidden_dim, args.seed,
        )
    else:
        logger.info("Loading matrix from %s", args.matrix)
        data = np.load(args.matrix, allow_pickle=False)
        matrix = data["matrix"]

        if args.hidden_states:
            logger.info("Loading pre-computed hidden states from %s", args.hidden_states)
            hidden_states = np.load(args.hidden_states, allow_pickle=False)
        elif args.use_backbone:
            prompts = [str(q) for q in data.get("prompts", data.get("questions", []))]
            if not prompts:
                logger.error("Matrix has no 'prompts' or 'questions' array for backbone")
                sys.exit(1)
            # Always CPU, regardless of --device: the validator embeds on CPU in
            # float32, and CUDA would use float16. Training against embeddings
            # the validator will never reproduce yields a head that scores far
            # below what local fitness suggests. --device still drives SFT/CMA.
            hidden_states = compute_hidden_states(
                prompts, model_name=args.backbone, device="cpu",
            )
        else:
            logger.info("No hidden states source — generating random (for testing)")
            hidden_states = np.random.randn(matrix.shape[0], args.hidden_dim).astype(np.float32)
            hidden_states /= np.linalg.norm(hidden_states, axis=1, keepdims=True)

    N, M = matrix.shape
    assert M == n_models, f"Matrix has {M} models but {n_models} model IDs provided"
    assert hidden_states.shape == (N, args.hidden_dim), \
        f"Hidden states shape {hidden_states.shape} != expected ({N}, {args.hidden_dim})"

    logger.info("Data: %d questions, %d models, hidden_dim=%d", N, M, args.hidden_dim)

    # ── Soft targets ──
    soft = compute_soft_targets(matrix, tau=args.tau)
    logger.info("Soft targets computed (tau=%.2f)", args.tau)

    # ── Stage 1: SFT ──
    logger.info("=" * 50)
    logger.info("Stage 1: SFT (KL divergence, %d epochs)", args.sft_epochs)
    logger.info("=" * 50)
    t0 = time.time()

    W, b = train_sft(
        hidden_states, soft, n_models, args.hidden_dim,
        epochs=args.sft_epochs, lr=args.sft_lr,
        batch_size=args.sft_batch_size, device=args.device,
    )

    sft_time = time.time() - t0
    logger.info("SFT complete in %.1fs", sft_time)

    model_costs = np.array([(i + 1) * 0.005 for i in range(n_models)], dtype=np.float64)
    sft_fitness = evaluate_routing(W, b, hidden_states, matrix, model_costs, args.lam)
    logger.info("SFT fitness: %.4f", sft_fitness)

    # ── Stage 2: sep-CMA-ES ──
    if not args.skip_cma:
        logger.info("=" * 50)
        logger.info("Stage 2: sep-CMA-ES (%d generations, pop=%d)",
                     args.cma_generations, args.cma_popsize)
        logger.info("=" * 50)
        t1 = time.time()

        W, b = train_cma(
            W, b, hidden_states, matrix, model_costs, args.lam,
            generations=args.cma_generations, popsize=args.cma_popsize,
            sigma=args.cma_sigma,
        )

        cma_time = time.time() - t1
        cma_fitness = evaluate_routing(W, b, hidden_states, matrix, model_costs, args.lam)
        logger.info("CMA complete in %.1fs", cma_time)
        logger.info("CMA fitness: %.4f (was %.4f after SFT)", cma_fitness, sft_fitness)
    else:
        logger.info("Skipping CMA-ES stage (--skip-cma)")

    # ── Save ──
    head_hash = save_head(W, b, args.models, args.output)
    logger.info("Training complete. Head ready for miner submission.")

    # ── Validation ──
    from fugal_subnet.head_eval import load_head_from_npz
    with open(args.output, "rb") as f:
        head = load_head_from_npz(f.read())
    logger.info("Validation: head loads OK (W=%s, b=%s, models=%d)",
                head.W.shape, head.b.shape, len(head.models))


if __name__ == "__main__":
    main()
