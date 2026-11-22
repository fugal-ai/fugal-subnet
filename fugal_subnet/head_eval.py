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
    HEAD_MAX_BYTES,
    HEAD_MAX_DECOMPRESSED_BYTES,
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
    z = contract.validate_head(_read_success_archive(data))
    return HeadArtifact(z["W"], z["b"], z["models"].tolist(), "", z)


def _read_success_archive(data: bytes) -> dict:
    """The arrays of a success-contract head archive, read under every bound
    that applies before any array is parsed. Shared by the v1-v3 loader and
    the v4 one, so both refuse the same bytes for the same reasons."""
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
    if "contract" not in arrays:
        # ONLY success-contract heads. A head without the contract carries no
        # cost profile, so it can never be priced or scored (benchmark_costs
        # refuses it) under either protocol; parsing one did nothing but give
        # miner bytes a second, older code path to reach before any hardware
        # check. Refused here, before anything in it is read.
        raise ValueError(
            "Head is not a success-contract head (no `contract` member): build it with "
            "scripts/train_head.py; heads without a cost profile cannot be scored"
        )
    return arrays


# --- protocol v4 heads (docs/decisions/LIVE_MENU_V4.md, section 11) ------------
#
# A v4 head is today's success-head artifact: a linear layer W (rows x 1024)
# and a bias b on the pinned question embedding, under today's contract,
# profile and backbone. Its rows are keyed by menu entry id. It needs no
# token means and no cost profile: a v4 pick is priced at the epoch's
# MenuState (head_odds.picks_and_odds_v4), so those fields, when a head
# carries them, are checked as today and never read.
#
# The vendored contract (vendor/success_contract.py) requires the cost fields
# and caps a head at 64 rows, and it is byte-compared with its upstream, so
# this acceptance sits beside it rather than in it. What the upstream
# contract must add for the product to read a v4 head is listed in
# LIVE_MENU_V4.md 11.2.

V4_REQUIRED_FIELDS = frozenset({"W", "b", "models", "contract", "profile_id",
                                "backbone_revision", "provenance", "lam"})
# Optional, and never read under v4.
V4_IGNORED_FIELDS = frozenset({"mean_in_tokens", "mean_out_tokens", "cost_profile_id"})
# Rows a v4 head may carry: the menu's cap plus as many entries as may be
# provisional at once (menu_state.MENU_CAP 64 + MENU_MAX_PROVISIONAL 8;
# tests/test_head_v4.py pins the sum), so a head can hold a row for an entry
# before it joins. A literal here so that nothing on the
# v1-v3 path imports the menu module.
HEAD_MAX_ROWS_V4 = 72
_V4_DIM = 1024                 # the pinned embedding's width (contract.embed)
# The dtypes a v4 head's W and b may have: little-endian float32 or float64.
V4_PARAM_DTYPES = ("<f4", "<f8")
_V4_MAX_ID_LEN = 256           # as the contract bounds a row id today


def validate_head_v4(z: dict) -> dict:
    """The v4 acceptance of a head's arrays, or ValueError.

    Required: every field of V4_REQUIRED_FIELDS, each as the contract checks
    it today (`contract`, `profile_id` and `backbone_revision` equal to the
    pinned values; `provenance` a non-empty string; `lam` a finite
    non-negative number, which no pick reads). `models`: 1 to
    HEAD_MAX_ROWS_V4 unique, non-empty ids of at most 256 characters. `W`:
    (rows, 1024) and `b`: (rows,), numeric, finite, every value within
    head_odds.HEAD_MAX_ABS_PARAM. Optional: any of V4_IGNORED_FIELDS, held to
    today's checks when present. Any other field is refused.
    """
    from fugal_subnet.head_odds import check_head_bounds

    fields = set(z)
    if not V4_REQUIRED_FIELDS <= fields or fields - V4_REQUIRED_FIELDS - V4_IGNORED_FIELDS:
        raise ValueError("unsupported head fields: "
                         f"{sorted((V4_REQUIRED_FIELDS - fields) | (fields - V4_REQUIRED_FIELDS - V4_IGNORED_FIELDS))}")
    for key, expected in (("contract", contract.CONTRACT), ("profile_id", contract.PROFILE_ID),
                          ("backbone_revision", contract.REVISION)):
        if contract.scalar(z, key) != expected:
            raise ValueError(f"unsupported {key}")
    if not contract.scalar(z, "provenance").strip():
        raise ValueError("missing provenance")
    if "cost_profile_id" in z:
        cpid = contract.scalar(z, "cost_profile_id")
        if len(cpid) != 64 or any(ch not in "0123456789abcdef" for ch in cpid):
            raise ValueError("invalid cost profile identity")
    models = z["models"]
    if models.ndim != 1 or models.dtype.kind != "U" or not 1 <= len(models) <= HEAD_MAX_ROWS_V4:
        raise ValueError(f"a v4 head carries 1 to {HEAD_MAX_ROWS_V4} rows, keyed by entry id")
    ids = models.tolist()
    if len(set(ids)) != len(ids) or any(not m.strip() or len(m) > _V4_MAX_ID_LEN for m in ids):
        raise ValueError("row ids must be unique, nonempty and bounded")
    n = len(ids)
    # W and b are read by exact arithmetic on every node: only the two widths
    # every platform decodes alike, little-endian, before any value is read.
    for key in ("W", "b"):
        if np.dtype(z[key].dtype).str not in V4_PARAM_DTYPES:
            raise ValueError(f"{key} must be little-endian float32 or float64, not "
                             f"{np.dtype(z[key].dtype).str}")
    shapes = {"W": (n, _V4_DIM), "b": (n,), "lam": (),
              "mean_in_tokens": (n,), "mean_out_tokens": (n,)}
    for key, shape in shapes.items():
        if key not in z:
            continue
        a = z[key]
        if a.shape != shape or a.dtype.kind not in "fi" or not np.isfinite(a).all():
            raise ValueError(f"invalid {key}")
        if key not in ("W", "b") and (a < 0).any():
            raise ValueError(f"negative {key}")
    check_head_bounds(z["W"], z["b"])
    return z


def load_head_v4(data: bytes) -> HeadArtifact:
    """Load a protocol v4 head from raw .npz bytes: the same byte, archive and
    no-pickle bounds as `load_head_from_npz`, then `validate_head_v4`."""
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("a head is bytes")
    try:
        arrays = _read_success_archive(bytes(data))
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 - miner bytes: a refusal with its reason
        # zipfile raises NotImplementedError for a compression method it
        # cannot unpack (and other errors for other damage): a v4 head the
        # reader cannot unpack is refused like any other unreadable head.
        raise ValueError(f"Head is not a readable .npz archive ({type(e).__name__}: {e})") from e
    z = validate_head_v4(arrays)
    return HeadArtifact(z["W"], z["b"], z["models"].tolist(), "", z)


def v4_head_verdict(head_bytes: bytes | None, state) -> str | None:
    """None, or why a head cannot run a v4 epoch whose menu is `state`
    (a menu_state.MenuState): it does not load as a v4 head, or it has no
    row for any entry on the menu. Rows for other ids are never read, and an
    entry with no row is the scoring rule's to price (grace, 9.1)."""
    try:
        if head_bytes is None:
            raise ValueError("missing success head bytes")
        head = load_head_v4(head_bytes)
    except ValueError as e:
        return f"Incompatible success head: {e}"
    ids = set(state.ids())
    if not any(m in ids for m in head.models):
        return "Incompatible success head: no row for any entry on this epoch's menu"
    return None


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
