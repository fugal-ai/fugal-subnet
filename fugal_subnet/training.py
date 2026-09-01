"""Installed reveal-native trainer for Fugal v2 router heads."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fugal_subnet.head_eval import HeadArtifact
from fugal_subnet.sandbox.client import GradingClient
from fugal_subnet.v2.backbone import compute_hidden_states, configure_determinism
from fugal_subnet.v2.head_eval import V2HeadScore, evaluate_head
from fugal_subnet.v2.reveal import MAX_REVEAL_BYTES, VerifiedReveal, verify_reveal
from fugal_subnet.v2.soft_targets import compute_soft_targets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingData:
    questions: list[str]
    matrix: np.ndarray
    hidden_states: np.ndarray
    model_ids: list[str]
    canonical_costs: dict[str, float]


def generate_synthetic_data(
    model_ids: list[str],
    *,
    n_questions: int = 300,
    hidden_dim: int = 1024,
    seed: int = 42,
) -> TrainingData:
    """Create deterministic local/mock-only data without API or chain access."""
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise ValueError("synthetic model IDs must be unique and non-empty")
    if n_questions <= 0 or hidden_dim <= 0:
        raise ValueError("synthetic dimensions must be positive")
    random = np.random.default_rng(seed)
    hidden = random.standard_normal((n_questions, hidden_dim)).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True).clip(min=1e-12)
    matrix = random.binomial(1, 0.6, size=(n_questions, len(model_ids))).astype(np.int8)
    empty = np.flatnonzero(matrix.sum(axis=1) == 0)
    matrix[empty, empty % len(model_ids)] = 1
    return TrainingData(
        questions=[f"mock question {index}" for index in range(n_questions)],
        matrix=matrix,
        hidden_states=hidden,
        model_ids=model_ids,
        canonical_costs={model: (index + 1) * 0.001 for index, model in enumerate(model_ids)},
    )


def _historical_resolver(network: str, netuid: int):
    import bittensor as bt

    subtensor = bt.Subtensor(network=network)

    def resolve(receipt):
        if receipt.netuid != netuid:
            raise RuntimeError("reveal commitment receipt netuid differs")
        actual_block_hash = str(subtensor.get_block_hash(receipt.block)).removeprefix("0x")
        expected_block_hash = receipt.block_hash.removeprefix("0x")
        if actual_block_hash != expected_block_hash:
            raise RuntimeError("reveal commitment receipt block hash differs")
        return subtensor.get_commitment(netuid, receipt.uid, block=receipt.block)

    return resolve


def load_verified_reveal(
    path: str | Path,
    *,
    grader_socket: str | None,
    network: str | None,
    netuid: int | None,
    allow_unverified_chain: bool = False,
) -> tuple[TrainingData, VerifiedReveal]:
    payload = Path(path).read_bytes()
    if not payload or len(payload) > MAX_REVEAL_BYTES:
        raise ValueError("reveal artifact size is invalid")
    raw = json.loads(payload.decode("utf-8"))
    prompts = [question["prompt"] for question in raw["questions"]]
    hidden_cache: list[np.ndarray] = []

    def embedder(values: list[str]) -> np.ndarray:
        if values != prompts:
            raise RuntimeError("reveal verifier prompt order changed")
        if not hidden_cache:
            hidden_cache.append(compute_hidden_states(values))
        return hidden_cache[0]

    client = GradingClient(grader_socket) if grader_socket else None
    if network is not None:
        if netuid is None:
            raise ValueError("--netuid is required with --network")
        resolver = _historical_resolver(network, netuid)
    elif allow_unverified_chain:
        resolver = None
    else:
        raise ValueError(
            "reveal training requires --network/--netuid historical checks or "
            "explicit --allow-unverified-chain"
        )
    verified = verify_reveal(
        payload,
        grading_client=client,
        embedder=embedder,
        chain_resolver=resolver,
    )
    return TrainingData(
        questions=prompts,
        matrix=verified.matrix,
        hidden_states=hidden_cache[0],
        model_ids=list(verified.model_ids),
        canonical_costs=verified.canonical_costs,
    ), verified


def load_legacy_npz(path: str | Path) -> TrainingData:
    """Load the explicitly opted-in historical matrix interchange format."""
    with np.load(path, allow_pickle=False) as data:
        required = {"matrix", "models", "hidden_states"}
        if not required <= set(data.files):
            raise ValueError("legacy NPZ requires matrix, models, and hidden_states")
        matrix = np.asarray(data["matrix"], dtype=np.int8)
        models = [str(value) for value in data["models"]]
        hidden = np.asarray(data["hidden_states"], dtype=np.float32)
        if "model_costs" not in data.files:
            raise ValueError("legacy NPZ requires explicit model_costs")
        costs_array = np.asarray(data["model_costs"], dtype=np.float64)
        questions = [str(value) for value in data["prompts"]] if "prompts" in data.files else []
    if matrix.ndim != 2 or matrix.shape[1] != len(models):
        raise ValueError("legacy matrix/model shape differs")
    if hidden.ndim != 2 or hidden.shape[0] != matrix.shape[0]:
        raise ValueError("legacy hidden-state shape differs")
    if costs_array.shape != (len(models),) or not np.all(np.isfinite(costs_array)):
        raise ValueError("legacy model costs are invalid")
    return TrainingData(
        questions=questions,
        matrix=matrix,
        hidden_states=hidden,
        model_ids=models,
        canonical_costs={model: float(costs_array[index]) for index, model in enumerate(models)},
    )


def train_head(
    data: TrainingData,
    *,
    selected_models: list[str] | None = None,
    epochs: int = 200,
    learning_rate: float = 0.01,
    seed: int = 42,
    routing_lambda: float = 2.0,
) -> tuple[HeadArtifact, V2HeadScore]:
    """Train on CPU, then evaluate with the exact v2 routing implementation."""
    configure_determinism()
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive")
    models = selected_models or list(data.model_ids)
    if not models or len(models) != len(set(models)) or not set(models) <= set(data.model_ids):
        raise ValueError("selected models must be a unique non-empty registry subset")
    canonical_index = {model: index for index, model in enumerate(data.model_ids)}
    columns = [canonical_index[model] for model in models]
    subset_matrix = data.matrix[:, columns]
    targets = compute_soft_targets(subset_matrix)
    hidden = np.asarray(data.hidden_states, dtype=np.float32)
    if hidden.shape[0] != subset_matrix.shape[0] or not np.all(np.isfinite(hidden)):
        raise ValueError("training hidden states are invalid")

    torch.manual_seed(seed)
    features = torch.from_numpy(hidden)
    target_tensor = torch.from_numpy(targets.astype(np.float32))
    costs = torch.tensor(
        [data.canonical_costs[model] for model in models], dtype=torch.float32
    )
    weights = torch.nn.Parameter(torch.zeros((len(models), hidden.shape[1]), dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(len(models), dtype=torch.float32))
    optimizer = torch.optim.Adam([weights, bias], lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        probabilities = torch.softmax(features @ weights.T + bias, dim=1)
        kl_loss = torch.nn.functional.kl_div(
            torch.log(probabilities.clamp_min(1e-12)),
            target_tensor,
            reduction="batchmean",
        )
        cost_loss = (probabilities * costs).sum(dim=1).mean()
        loss = kl_loss + routing_lambda * cost_loss
        loss.backward()
        optimizer.step()

    artifact = HeadArtifact(
        W=weights.detach().cpu().numpy().astype(np.float32),
        b=bias.detach().cpu().numpy().astype(np.float32),
        models=models,
        commit_hash="",
    )
    full_targets = compute_soft_targets(data.matrix)
    evaluation = evaluate_head(
        artifact,
        hidden,
        data.matrix,
        data.model_ids,
        full_targets,
        data.canonical_costs,
        wire_model_pool=models,
        routing_lambda=routing_lambda,
    )
    return artifact, evaluation


def save_head(path: str | Path, head: HeadArtifact) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        W=head.W.astype(np.float32),
        b=head.b.astype(np.float32),
        models=np.asarray(head.models),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Fugal router head from a verified v2 reveal")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reveal", help="Canonical v2 reveal.json")
    inputs.add_argument("--legacy-npz", help="Historical NPZ matrix (explicit compatibility mode)")
    inputs.add_argument("--synthetic", action="store_true", help="Local/mock-only synthetic smoke data")
    parser.add_argument("--allow-legacy-npz", action="store_true")
    parser.add_argument("--allow-unverified-chain", action="store_true")
    parser.add_argument("--network", help="Bittensor network used for exact-block receipt checks")
    parser.add_argument("--netuid", type=int)
    parser.add_argument("--grader-socket", help="Required for code/symbolic response regrading")
    parser.add_argument("--models", help="Comma-separated active registry subset")
    parser.add_argument("--n-questions", type=int, default=300)
    parser.add_argument("--output", default="head.npz")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--routing-lambda", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.legacy_npz:
        if not args.allow_legacy_npz:
            raise SystemExit("--legacy-npz requires explicit --allow-legacy-npz")
        data = load_legacy_npz(args.legacy_npz)
    elif args.reveal:
        data, verified = load_verified_reveal(
            args.reveal,
            grader_socket=args.grader_socket,
            network=args.network,
            netuid=args.netuid,
            allow_unverified_chain=args.allow_unverified_chain,
        )
        logger.info(
            "Verified v2 reveal %s (historical chain=%s)",
            verified.epoch_id,
            "verified" if verified.chain_receipts_verified else "explicitly skipped",
        )
    else:
        if not args.models:
            raise SystemExit("--synthetic requires --models model/a,model/b")
        data = generate_synthetic_data(
            args.models.split(","), n_questions=args.n_questions, seed=args.seed,
        )
    selected = args.models.split(",") if args.models and not args.synthetic else None
    head, evaluation = train_head(
        data,
        selected_models=selected,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        routing_lambda=args.routing_lambda,
    )
    save_head(args.output, head)
    logger.info(
        "Saved %s: accuracy=%.6f cost_efficiency=%.6f kl=%.6f",
        args.output,
        evaluation.accuracy,
        evaluation.cost_efficiency,
        evaluation.kl_score,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
