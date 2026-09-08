"""Head evaluation against the ground truth matrix.

Loads a miner's .npz head, evaluates routing decisions against the matrix,
computes accuracy, cost efficiency, and KL divergence.
"""
from __future__ import annotations

import base64
import io
import logging
import zipfile
from dataclasses import dataclass

import numpy as np

from fugal_subnet.config import (
    HEAD_HIDDEN_DIM,
    HEAD_MAX_BYTES,
    HEAD_MAX_DECOMPRESSED_BYTES,
    HEAD_MAX_MODEL_ID_LEN,
    HEAD_MAX_MODELS,
    ROUTING_DECISION_QUANTUM,
)
from fugal_subnet.routing_protocol import BENCHMARK_LAMBDA
from fugal_subnet.vendor import success_contract as contract

logger = logging.getLogger(__name__)



def quantize_utility(utility: np.ndarray) -> np.ndarray:
    """Snap routing utilities to a fixed grid so validators agree on argmax.

    Computed in float64 and rounded to ROUTING_DECISION_QUANTUM. Both matter:
    float64 keeps the rounding itself from being the thing that differs, and a
    fixed grid absorbs the small float differences that separate two honest
    validators running different BLAS kernels or CPU generations.
    """
    scaled = np.asarray(utility, dtype=np.float64) / ROUTING_DECISION_QUANTUM
    return np.round(scaled) * ROUTING_DECISION_QUANTUM

@dataclass
class HeadArtifact:
    W: np.ndarray           # (L, d) float32
    b: np.ndarray           # (L,) float32
    models: list[str]       # L model IDs
    commit_hash: str        # SHA256 for dedup seniority
    success: dict | None = None


@dataclass
class HeadScore:
    accuracy: float         # fraction of scoreable questions where selected model was correct
    cost_efficiency: float  # oracle_cost / head_cost, capped at 1.0 (1.0 = routed as cheap as the oracle)
    kl_score: float         # -mean KL divergence (higher = better distribution match)
    routing_decisions: np.ndarray  # (N,) model indices chosen by head
    correct_mask: np.ndarray       # (N,) bool — did the chosen model get it right?
    coverage: float = 1.0  # fraction of pool models this head covers (intersection / pool size)
    n_correct: int = 0
    n_scored: int = 0
    total_head_cost: float = 0.0
    total_oracle_cost: float = 0.0
    total_kl: float = 0.0


def load_head_from_npz(data: bytes) -> HeadArtifact:
    """Load a head artifact from raw .npz bytes.

    Validates: size cap, required arrays, shape consistency, model strings.
    """
    if len(data) > HEAD_MAX_BYTES:
        raise ValueError(f"Head too large: {len(data)} bytes (max {HEAD_MAX_BYTES})")

    # .npz is a zip archive; check declared decompressed sizes BEFORE np.load
    # allocates anything. Blocks zip-bomb DoS (1 MB zip -> multi-GB array).
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total_decompressed = sum(info.file_size for info in zf.infolist())
    except zipfile.BadZipFile as e:
        raise ValueError(f"Head is not a valid .npz archive: {e}")
    if total_decompressed > HEAD_MAX_DECOMPRESSED_BYTES:
        raise ValueError(
            f"Head decompresses to {total_decompressed} bytes "
            f"(max {HEAD_MAX_DECOMPRESSED_BYTES})"
        )

    arrays = contract.read_archive(data)
    if "contract" in arrays:
        z = contract.validate_head(arrays)
        return HeadArtifact(z["W"], z["b"], z["models"].tolist(), "", z)

    buf = io.BytesIO(data)
    with np.load(buf, allow_pickle=False) as npz:
        if "W" not in npz or "b" not in npz or "models" not in npz:
            raise ValueError("Head missing required arrays: W, b, models")

        W = npz["W"].astype(np.float32)
        b = npz["b"].astype(np.float32)
        models = [str(m) for m in npz["models"]]

    if W.ndim != 2:
        raise ValueError(f"Head W must be 2-D, got shape {W.shape}")
    L, d = W.shape
    if L == 0:
        # A head declaring no models routes nothing. It survives every other
        # check (shapes agree, no non-finite values) and scores 0.0, so it is
        # not a crash — but it still occupies a slot in scoring, dedup, and the
        # published reveal. Degenerate input dies at the boundary.
        raise ValueError("Head declares no models")
    if L > HEAD_MAX_MODELS:
        raise ValueError(f"Head has {L} model rows (max {HEAD_MAX_MODELS})")
    if d != HEAD_HIDDEN_DIM:
        raise ValueError(f"Head hidden dim {d} != expected {HEAD_HIDDEN_DIM}")
    if b.shape != (L,):
        raise ValueError(f"Head bias shape {b.shape} != expected ({L},)")
    if len(models) != L:
        raise ValueError(f"Head has {L} rows but {len(models)} model names")
    for name in models:
        if len(name) > HEAD_MAX_MODEL_ID_LEN:
            raise ValueError(
                f"Head model ID is {len(name)} chars (max {HEAD_MAX_MODEL_ID_LEN})"
            )
    # Distinct names, because HEAD_MAX_MODELS is a CAPACITY bound and duplicates
    # spend it on something else. The cap exists to stop a head large enough to
    # memorise the question pool rather than learn to route; a row is a
    # parameter vector, and 64 rows all naming one model is 64 rows of capacity
    # wearing the costume of a one-model head.
    #
    # The justification is the NAMING GAP, not a measured exploit, and that
    # distinction was itself a correction. On synthetic isotropic embeddings,
    # 32 rows over 8 names lifted random-label fit from 0.269 to 0.441, which
    # looked like real capacity bought cheaply. Re-measured on actual pool
    # embeddings it does not reproduce: three of five differences are zero or
    # negative and the largest gain is +0.036, against the +0.172 the synthetic
    # case suggested. Clustered real embeddings span fewer effective dimensions
    # than Gaussian ones, so the synthetic figure was an upper bound behaving
    # like a finding.
    #
    # So this is not closing a demonstrated memorisation route. It is closing
    # the gap between what HEAD_MAX_MODELS is NAMED and what it bounds, which
    # stands on its own.
    #
    # It is also meaningless on its own terms — two rows naming the same model
    # are two ways to say the same route, and softmax over them is a
    # reparameterisation with no routing content. Nothing honest produces it.
    if len(set(models)) != L:
        dupes = sorted({m for m in models if models.count(m) > 1})
        raise ValueError(
            f"Head names {len(set(models))} distinct models across {L} rows — "
            f"duplicates: {dupes[:3]}. Rows must name distinct models: the row "
            "cap bounds head capacity, and duplicate names spend that capacity "
            "without declaring it."
        )

    if not np.all(np.isfinite(W)):
        raise ValueError("Head W contains non-finite values")
    if not np.all(np.isfinite(b)):
        raise ValueError("Head b contains non-finite values")

    return HeadArtifact(W=W, b=b, models=models, commit_hash="")


def load_head_from_b64(b64_str: str) -> HeadArtifact:
    """Load a head from base64-encoded .npz bytes."""
    data = base64.b64decode(b64_str)
    return load_head_from_npz(data)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def evaluate_head(
    head: HeadArtifact,
    hidden_states: np.ndarray,
    matrix: np.ndarray,
    models_in_matrix: list[str],
    soft_targets: np.ndarray,
    model_costs: dict[str, float],
) -> HeadScore:
    """Evaluate a head's routing decisions against the ground truth matrix.

    Args:
        head: The miner's head artifact.
        hidden_states: (N, d) backbone hidden states for all questions.
        matrix: (N, M_full) ground truth binary matrix.
        models_in_matrix: M_full model IDs matching matrix columns.
        soft_targets: (N, M_full) oracle soft target distributions.
        model_costs: {model_id: cost_per_query}.

    Returns:
        HeadScore with accuracy, cost efficiency, KL divergence, and decisions.
    """
    if head.success is not None:
        return evaluate_success_head(head, hidden_states, matrix, models_in_matrix, model_costs)
    N = hidden_states.shape[0]
    M_pool = len(models_in_matrix)
    head_model_to_matrix_idx = {}
    for i, m in enumerate(head.models):
        if m in models_in_matrix:
            head_model_to_matrix_idx[i] = models_in_matrix.index(m)

    coverage = len(head_model_to_matrix_idx) / max(M_pool, 1)

    if not head_model_to_matrix_idx:
        return HeadScore(
            accuracy=0.0, cost_efficiency=0.0, kl_score=-100.0,
            routing_decisions=np.full(N, -1, dtype=np.int32),
            correct_mask=np.zeros(N, dtype=bool),
            coverage=0.0,
        )

    routing_decisions = np.zeros(N, dtype=np.int32)
    correct_mask = np.zeros(N, dtype=bool)
    total_kl = 0.0
    total_head_cost = 0.0
    total_oracle_cost = 0.0
    n_scored = 0

    for q_idx in range(N):
        h = hidden_states[q_idx]
        logits = head.W @ h + head.b
        p = _softmax(logits)

        # No cost term: the routing rule is argmax over the head's own
        # preference. See config.TRAINING_COST_LAMBDA for why the subnet no
        # longer dictates a quality/cost exchange rate.
        #
        # Quantize before the argmax. See config.ROUTING_DECISION_QUANTUM:
        # raw argmax is a discontinuity, so any cross-validator float
        # difference flips the routing decision on a near-tie. Rounding to a
        # fixed step makes the decision agree unless validators differ by a
        # whole quantum, and exact ties fall to the lowest index.
        utility = quantize_utility(p)
        selected_head_idx = int(np.argmax(utility))
        routing_decisions[q_idx] = selected_head_idx

        # Questions no model in the pool answered correctly carry no routing
        # signal — exclude them from all score components (accuracy, cost, KL)
        # so every miner is graded on the same scoreable set.
        correct_in_row = np.where(matrix[q_idx] == 1)[0]
        if len(correct_in_row) == 0:
            continue
        n_scored += 1

        selected_model = head.models[selected_head_idx]
        total_head_cost += model_costs.get(selected_model, 0.01)

        if selected_head_idx in head_model_to_matrix_idx:
            matrix_idx = head_model_to_matrix_idx[selected_head_idx]
            correct_mask[q_idx] = matrix[q_idx, matrix_idx] == 1

        oracle_dist = soft_targets[q_idx]
        head_dist_aligned = _align_distribution(
            p, head.models, models_in_matrix, len(models_in_matrix)
        )
        total_kl += _kl_divergence(oracle_dist, head_dist_aligned)

        oracle_costs = [model_costs.get(models_in_matrix[i], 0.01) for i in correct_in_row]
        total_oracle_cost += min(oracle_costs)

    if n_scored == 0:
        return HeadScore(
            accuracy=0.0, cost_efficiency=0.0, kl_score=-100.0,
            routing_decisions=routing_decisions,
            correct_mask=correct_mask,
            coverage=coverage,
            n_correct=0, n_scored=0,
            total_head_cost=0.0, total_oracle_cost=0.0, total_kl=0.0,
        )

    accuracy = float(correct_mask.sum()) / n_scored
    # Capped at 1.0: the oracle (cheapest-correct routing) is the ceiling.
    # Uncapped, an always-route-to-cheapest head could earn a cost ratio >> 1
    # and dominate the composite score without routing at all.
    cost_efficiency = min(1.0, total_oracle_cost / max(total_head_cost, 1e-10))
    kl_score = -total_kl / n_scored

    return HeadScore(
        accuracy=accuracy,
        cost_efficiency=cost_efficiency,
        kl_score=kl_score,
        routing_decisions=routing_decisions,
        correct_mask=correct_mask,
        coverage=coverage,
        n_correct=int(correct_mask.sum()),
        n_scored=n_scored,
        total_head_cost=total_head_cost,
        total_oracle_cost=total_oracle_cost,
        total_kl=total_kl,
    )


def _align_distribution(
    head_dist: np.ndarray,
    head_models: list[str],
    matrix_models: list[str],
    M: int,
) -> np.ndarray:
    """Align a head's distribution to the full model pool ordering."""
    aligned = np.zeros(M, dtype=np.float64)
    for i, m in enumerate(head_models):
        if m in matrix_models:
            j = matrix_models.index(m)
            aligned[j] = head_dist[i]
    s = aligned.sum()
    if s > 0:
        aligned /= s
    else:
        aligned[:] = 1.0 / M
    return aligned


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL(p || q) — how well q approximates the oracle distribution p."""
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def evaluate_success_head(head, hidden_states, matrix, models_in_matrix, model_costs):
    """Observed failures count; unavailable routed labels stay unscored."""
    if len(set(models_in_matrix)) != len(models_in_matrix):
        raise ValueError("duplicate matrix model columns")
    indices = [models_in_matrix.index(m) for m in head.models]
    costs = np.array([model_costs[m] for m in head.models])
    p = contract.predictions(head.W, head.b, hidden_states)
    decisions = contract.rank(p, costs, BENCHMARK_LAMBDA)[:, 0]
    labels = matrix[:, indices]
    chosen = labels[np.arange(len(labels)), decisions]
    observed = np.isfinite(chosen) & (chosen >= 0)
    correct = (chosen == 1) & observed
    n = int(observed.sum())
    return HeadScore(accuracy=float(correct.sum()/n) if n else 0., cost_efficiency=0.,
                     kl_score=0., routing_decisions=decisions, correct_mask=correct,
                     coverage=len(indices)/len(models_in_matrix), n_correct=int(correct.sum()),
                     n_scored=n, total_head_cost=float(costs[decisions[observed]].sum()))
