"""Property-based tests for the head loader — invariant I2.

Example tests find the bugs you thought of. The miner interface is adversarial:
a registered miner can serve any bytes it likes, and the ones worth worrying
about are the ones nobody enumerated. These state the *properties* the loader
must satisfy for every input and let Hypothesis search for counterexamples.

The properties, not the examples, are the specification:

  P1  Loading arbitrary bytes never raises anything but ValueError. Any other
      exception type escapes the validator's per-miner try/except and takes
      down the epoch — one miner halting everyone's emissions (I6).
  P2  A head that loads is always structurally usable: 2-D W, matching b and
      models, correct hidden dim, within caps, all values finite. Downstream
      scoring math assumes every one of these.
  P3  Loading is a pure function. Same bytes, same result — a loader with
      hidden state would make two validators disagree (I1).
  P4  Any head that loads can be scored without raising. Passing validation but
      exploding in evaluate_head is the same liveness failure as P1.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fugal_subnet.config import (
    HEAD_HIDDEN_DIM,
    HEAD_MAX_MODEL_ID_LEN,
    HEAD_MAX_MODELS,
)
from fugal_subnet.head_eval import (
    HeadArtifact,
    evaluate_head,
    load_head_from_b64,
    load_head_from_npz,
)
from fugal_subnet.soft_targets import compute_soft_targets

SLOW = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def _npz(W, b, models, models_dtype="U100") -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, W=W, b=b, models=np.array(models, dtype=models_dtype))
    return buf.getvalue()


# ── P1: arbitrary bytes never escape as a non-ValueError ───────────────────

@SLOW
@given(payload=st.binary(min_size=0, max_size=4096))
def test_arbitrary_bytes_raise_only_value_error(payload):
    """A miner serving random bytes must not crash the epoch loop."""
    try:
        load_head_from_npz(payload)
    except ValueError:
        pass
    except Exception as e:  # noqa: BLE001 — the point is to catch the type
        pytest.fail(
            f"{type(e).__name__} escaped the loader for {len(payload)} bytes: {e}. "
            "Only ValueError is caught per-miner; anything else halts the epoch."
        )


@SLOW
@given(text=st.text(max_size=2048))
def test_arbitrary_base64_raises_only_value_error(text):
    """The axon hands over a base64 string, so decoding is attack surface too."""
    try:
        load_head_from_b64(text)
    except ValueError:
        pass
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"{type(e).__name__} escaped load_head_from_b64: {e}")


# ── P2: anything accepted is structurally usable ───────────────────────────

@SLOW
@given(
    n_models=st.integers(min_value=0, max_value=HEAD_MAX_MODELS + 4),
    hidden_dim=st.sampled_from([1, 64, HEAD_HIDDEN_DIM - 1, HEAD_HIDDEN_DIM,
                                HEAD_HIDDEN_DIM + 1]),
    b_len_delta=st.integers(min_value=-2, max_value=2),
    fill=st.sampled_from([0.0, 1e-8, 1.0, -1.0, 1e30, np.nan, np.inf, -np.inf]),
)
def test_accepted_heads_are_structurally_valid(n_models, hidden_dim, b_len_delta, fill):
    """Whatever the loader accepts must satisfy every downstream assumption."""
    W = np.full((n_models, hidden_dim), fill, dtype=np.float32)
    b_len = max(0, n_models + b_len_delta)
    b = np.full(b_len, fill, dtype=np.float32)
    models = [f"vendor/m{i}" for i in range(n_models)]

    try:
        head = load_head_from_npz(_npz(W, b, models))
    except ValueError:
        return  # Rejection is always a valid outcome.

    assert head.W.ndim == 2
    L, d = head.W.shape
    assert d == HEAD_HIDDEN_DIM
    assert 0 < L <= HEAD_MAX_MODELS
    assert head.b.shape == (L,)
    assert len(head.models) == L
    assert np.all(np.isfinite(head.W)), "non-finite W passed validation"
    assert np.all(np.isfinite(head.b)), "non-finite b passed validation"
    assert all(len(m) <= HEAD_MAX_MODEL_ID_LEN for m in head.models)


@SLOW
@given(id_len=st.integers(min_value=1, max_value=HEAD_MAX_MODEL_ID_LEN * 3))
def test_model_id_length_cap_is_enforced(id_len):
    """Oversized IDs are amplification into the published reveal artifact."""
    W = np.zeros((2, HEAD_HIDDEN_DIM), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    models = ["v/" + "a" * id_len] * 2
    try:
        head = load_head_from_npz(_npz(W, b, models, models_dtype=f"U{id_len + 8}"))
    except ValueError:
        return
    assert all(len(m) <= HEAD_MAX_MODEL_ID_LEN for m in head.models)


# ── P3: loading is pure ────────────────────────────────────────────────────

@SLOW
@given(
    n_models=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_loading_is_deterministic(n_models, seed):
    """Same bytes, same head — a stateful loader would break I1."""
    rng = np.random.RandomState(seed)
    W = (rng.randn(n_models, HEAD_HIDDEN_DIM) * 0.01).astype(np.float32)
    b = (rng.randn(n_models) * 0.01).astype(np.float32)
    payload = _npz(W, b, [f"vendor/m{i}" for i in range(n_models)])

    first = load_head_from_npz(payload)
    second = load_head_from_npz(payload)

    assert np.array_equal(first.W, second.W)
    assert np.array_equal(first.b, second.b)
    assert first.models == second.models


# ── P4: anything accepted can be scored ────────────────────────────────────

@SLOW
@given(
    n_models=st.integers(min_value=1, max_value=6),
    scale=st.sampled_from([1e-12, 1e-3, 1.0, 1e3, 1e20]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_accepted_heads_can_always_be_scored(n_models, scale, seed):
    """Passing validation then exploding in scoring is the same liveness bug.

    Extreme magnitudes are the interesting part: softmax overflow or a division
    by a zero cost would surface here rather than in production.
    """
    rng = np.random.RandomState(seed)
    W = (rng.randn(n_models, HEAD_HIDDEN_DIM) * scale).astype(np.float32)
    b = (rng.randn(n_models) * scale).astype(np.float32)
    models = [f"vendor/m{i}" for i in range(n_models)]
    try:
        head = load_head_from_npz(_npz(W, b, models))
    except ValueError:
        return

    n_q = 12
    hidden = rng.randn(n_q, HEAD_HIDDEN_DIM).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    matrix = (rng.rand(n_q, n_models) > 0.5).astype(np.int8)
    soft = compute_soft_targets(matrix)
    costs = {m: 0.001 * (i + 1) for i, m in enumerate(models)}

    score = evaluate_head(head, hidden, matrix, models, soft, costs)

    assert np.isfinite(score.accuracy)
    assert np.isfinite(score.cost_efficiency)
    assert np.isfinite(score.kl_score)
    assert 0.0 <= score.accuracy <= 1.0
    assert 0.0 <= score.cost_efficiency <= 1.0
    assert score.routing_decisions.shape == (n_q,)
    assert np.all(score.routing_decisions >= 0)
    assert np.all(score.routing_decisions < n_models)


@SLOW
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_scoring_is_deterministic_for_same_inputs(seed):
    """Scoring the same head twice must agree exactly — I1 at the unit level."""
    rng = np.random.RandomState(seed)
    n_models, n_q = 5, 16
    models = [f"vendor/m{i}" for i in range(n_models)]
    W = (rng.randn(n_models, HEAD_HIDDEN_DIM) * 0.01).astype(np.float32)
    # Near-identical biases put utilities on top of each other, which is where
    # an unquantized argmax would go nondeterministic.
    b = np.full(n_models, 1e-9, dtype=np.float32)
    head = HeadArtifact(W=W, b=b, models=models, commit_hash="x")

    hidden = rng.randn(n_q, HEAD_HIDDEN_DIM).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    matrix = (rng.rand(n_q, n_models) > 0.5).astype(np.int8)
    soft = compute_soft_targets(matrix)
    costs = {m: 0.001 for m in models}

    a = evaluate_head(head, hidden, matrix, models, soft, costs)
    c = evaluate_head(head, hidden, matrix, models, soft, costs)

    assert a.accuracy == c.accuracy
    assert a.cost_efficiency == c.cost_efficiency
    assert a.kl_score == c.kl_score
    assert np.array_equal(a.routing_decisions, c.routing_decisions)
