"""Invariant I4 — non-interference between miners.

A miner must not be able to lower another miner's score or prevent them from
being scored. This is the least-audited invariant in the system and the one an
adversary profits from most directly: degrading a rival raises your own relative
weight, so griefing pays even when it wins you nothing on merit.
"""
from __future__ import annotations

import numpy as np

from fugal_subnet.dedup import find_duplicates


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
