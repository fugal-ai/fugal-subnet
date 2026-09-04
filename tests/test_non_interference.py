"""Invariant I4 — non-interference between miners.

A miner must not be able to lower another miner's score or prevent them from
being scored. This is the least-audited invariant in the system and the one an
adversary profits from most directly: degrading a rival raises your own relative
weight, so griefing pays even when it wins you nothing on merit.
"""
from __future__ import annotations

import numpy as np

from fugal_subnet.dedup import find_duplicates
from fugal_subnet.head_eval import HeadArtifact, evaluate_head


def _decisions(pattern: list[int]) -> np.ndarray:
    return np.array(pattern, dtype=np.int32)


def test_copier_cannot_outrank_author_after_the_author_retrains():
    """The copy-and-outrank attack. Regression guard.

    Seniority used to come from a hotkey's *current* commitment block, which
    resets every time a miner updates their head:

      1. Author commits h1 at block 1000.
      2. Copier fetches h1 from the author's axon and commits it at 1100 —
         correctly deduped as junior.
      3. Author retrains to h2 and commits at 2000. h2 is an incremental
         improvement, so it still clusters with h1's behavior.
      4. Cluster survivor is the lowest block — the copier, at 1100.

    The author is disqualified for improving, which is the exact behavior the
    subnet pays for. Seniority must track when a hotkey was FIRST seen
    committing, so step 3 cannot hand the cluster to the copier.
    """
    author_uid, copier_uid = 1, 2
    behavior = _decisions([0, 1, 2, 1, 0, 2, 1, 0] * 4)

    # Both heads route near-identically — the copier copied.
    head_outputs = {author_uid: behavior, copier_uid: behavior.copy()}

    # Seniority as the fixed code computes it: first-ever commitment per hotkey.
    # The author was committing from block 1000, before the copier existed.
    first_seen = {author_uid: 1000, copier_uid: 1100}

    disqualified = find_duplicates(head_outputs, first_seen)

    assert copier_uid in disqualified, "the copier should lose the cluster"
    assert author_uid not in disqualified, (
        "the author was disqualified in favor of their own copier — "
        "seniority regressed to the current commitment block"
    )


def test_broken_seniority_would_disqualify_the_author():
    """Demonstrates the bug the fix prevents, so the guard above has teeth.

    Feeding find_duplicates the *current* commitment blocks — what the old code
    passed — reproduces the inversion. If this ever stops holding, the two tests
    are no longer testing different things.
    """
    author_uid, copier_uid = 1, 2
    behavior = _decisions([0, 1, 2, 1, 0, 2, 1, 0] * 4)
    head_outputs = {author_uid: behavior, copier_uid: behavior.copy()}

    # Author retrained (block 2000); the copier never updated (still 1100).
    current_blocks = {author_uid: 2000, copier_uid: 1100}

    disqualified = find_duplicates(head_outputs, current_blocks)

    assert author_uid in disqualified
    assert copier_uid not in disqualified


def test_distinct_behavior_is_never_clustered():
    """Non-interference floor: an honest miner routing differently is untouched.

    If dedup were loose enough to cluster genuinely different heads, a miner
    could grief a rival by mimicking them just closely enough.
    """
    a = _decisions([0, 1, 2, 3] * 8)
    b = _decisions([3, 2, 1, 0] * 8)

    disqualified = find_duplicates({1: a, 2: b}, {1: 1000, 2: 1100})

    assert disqualified == set(), "independent heads must not be clustered"


def test_a_single_miner_is_never_self_disqualified():
    """One miner alone can never be a duplicate of anything."""
    assert find_duplicates({1: _decisions([0, 1, 2] * 8)}, {1: 1000}) == set()


def test_uncommitted_heads_are_always_dedup_junior():
    """A head with no valid commitment must never outrank a committed one.

    Commitment-optional mode assigns math.inf, so an uncommitted head can never
    win a cluster and take an honest miner's slot.
    """
    behavior = _decisions([0, 1, 2, 1] * 8)
    disqualified = find_duplicates(
        {1: behavior, 2: behavior.copy()},
        {1: float("inf"), 2: 1100},
    )
    assert 1 in disqualified
    assert 2 not in disqualified


def test_sybil_pool_stuffing_does_not_zero_victim():
    """The pool-eviction attack must not work.

    Two sybil heads declare 5 junk models each. The victim declares 5 real
    models. Under the old fixed-cap pool (cap=5, declare-count priority), the
    sybils' models would dominate and the victim would score 0.000. Under the
    routed-model pool, the victim's models are always in the matrix and the
    victim scores normally.
    """
    d = 8
    n_questions = 20
    real_models = [f"real/model-{i}" for i in range(5)]
    junk_models = [f"junk/model-{i}" for i in range(5)]

    np.random.seed(42)

    victim_head = HeadArtifact(
        W=np.random.randn(5, d).astype(np.float32),
        b=np.zeros(5, dtype=np.float32),
        models=real_models,
        commit_hash="victim",
    )
    sybil_head = HeadArtifact(
        W=np.random.randn(5, d).astype(np.float32),
        b=np.zeros(5, dtype=np.float32),
        models=junk_models,
        commit_hash="sybil",
    )

    # The pool is the union of all routed models — no fixed cap, no eviction.
    all_models = sorted(set(real_models + junk_models))
    M = len(all_models)

    hidden = np.random.randn(n_questions, d).astype(np.float32)
    # Real models answer correctly; junk models answer incorrectly.
    matrix = np.zeros((n_questions, M), dtype=np.int8)
    for i, m in enumerate(all_models):
        if m.startswith("real/"):
            matrix[:, i] = 1

    soft = np.ones((n_questions, M), dtype=np.float64) / M
    costs = {m: 0.01 for m in all_models}

    victim_score = evaluate_head(
        victim_head, hidden, matrix, all_models, soft, costs,
    )
    sybil_score = evaluate_head(
        sybil_head, hidden, matrix, all_models, soft, costs,
    )

    assert victim_score.accuracy > 0, (
        "victim scores zero — pool eviction attack still works"
    )
    assert victim_score.accuracy > sybil_score.accuracy, (
        "sybil routing to junk models should not outscore a victim routing to correct models"
    )
    assert victim_score.coverage == 5 / M
    assert sybil_score.coverage == 5 / M


_MODELS = [f"model/{i}" for i in range(10)]
# Fixed true accuracies — facts about the world, not about the miner field.
_TRUE_ACC = {m: 0.3 + 0.05 * i for i, m in enumerate(_MODELS)}
_TRUE_BEST = max(_TRUE_ACC.values())


def _frame_from_field(n_miners, epochs, samples_per_miner=15, seed=0):
    """Build a reference frame from `n_miners` miners' exploration samples."""
    import random

    from fugal_subnet.reference_frame import ReferenceFrame, accumulate_exploration

    rng = random.Random(seed)
    frame = ReferenceFrame()
    for _ in range(epochs):
        samples = []
        for _ in range(n_miners):
            for _ in range(samples_per_miner):
                m = rng.choice(_MODELS)
                samples.append((m, rng.random() < _TRUE_ACC[m], 500, 300))
        frame = accumulate_exploration(frame, samples)
    return frame


def _ceiling(frame):
    from fugal_subnet.reference_frame import best_model

    prices = {m: (1e-6, 2e-6) for m in _MODELS}
    return best_model(frame, prices)[1]


def test_reference_frame_converges_to_the_same_place_at_any_field_size():
    """I4, restated for the TEE architecture.

    A miner's score must depend on its own behaviour and on facts about the
    world — never on how many other miners are online. The frame therefore
    pools exploration over TIME rather than over MINERS: field size changes how
    fast the estimate converges, not what it converges to.

    Some early sensitivity is unavoidable — a small field simply gathers
    evidence more slowly — so the property to hold is convergence, not instant
    equality. That is also why the prior is weak (see FRAME_PRIOR_STRENGTH):
    a strong neutral prior biases the ceiling low for longer in a thin field,
    which is precisely field-size dependence.
    """
    ceilings = [_ceiling(_frame_from_field(n, epochs=300)) for n in (3, 20, 50)]

    for c in ceilings:
        assert abs(c - _TRUE_BEST) < 0.05, (
            f"ceiling {c:.3f} did not converge to the true best accuracy "
            f"{_TRUE_BEST:.3f}"
        )
    spread = max(ceilings) - min(ceilings)
    assert spread < 0.05, (
        f"ceiling moved by {spread:.4f} purely because the number of miners "
        f"changed (3 -> 20 -> 50 gave {[round(c, 3) for c in ceilings]}) — "
        "that is I4 non-interference broken"
    )


def test_one_miner_cannot_materially_move_the_frame():
    """The sharpest form of non-interference: no single actor steers the frame.

    Even a miner feeding deliberately unrepresentative results — every
    exploration answer wrong — must not meaningfully shift the reference the
    rest of the field is scored against.
    """
    from fugal_subnet.reference_frame import accumulate_exploration

    honest = _frame_from_field(20, epochs=300)
    baseline = _ceiling(honest)

    poisoned = accumulate_exploration(
        honest, [(m, False, 500, 300) for m in _MODELS for _ in range(15)],
    )
    assert abs(_ceiling(poisoned) - baseline) < 0.02, (
        "a single miner's exploration samples moved the shared reference"
    )


def test_exploration_targets_are_not_miner_chosen():
    """A miner must not be able to steer what the frame observes.

    Targets derive from the epoch nonce and the global model list, so two
    miners in the same epoch are assigned identical exploration work, and no
    miner can bias the sample toward a model that flatters it.
    """
    from fugal_subnet.exploration import expected_exploration

    pool = [
        {"question_id": f"q{i}", "benchmark": "mmlu", "prompt": "x"}
        for i in range(500)
    ]
    models = [f"model/{i}" for i in range(10)]
    nonce = b"\x11" * 32
    scored = {f"q{i}" for i in range(100)}

    a = expected_exploration(nonce, pool, scored, models, 15)
    b = expected_exploration(nonce, pool, scored, models, 15)
    assert a == b and len(a) == 15

    # Disjoint from the scored slice: never graded on a forced misroute.
    assert not (set(a) & scored)

    # A different epoch assigns different work.
    other = expected_exploration(b"\x22" * 32, pool, scored, models, 15)
    assert set(other) != set(a)

    # Targets spread across the pool rather than collapsing onto one model.
    assert len(set(a.values())) > 1
