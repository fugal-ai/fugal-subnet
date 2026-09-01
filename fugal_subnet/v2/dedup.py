"""V2 behavioral deduplication over canonical decisions and distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DECISION_THRESHOLD = 0.99
DISTRIBUTION_THRESHOLD = 0.999


@dataclass(frozen=True)
class DedupResult:
    disqualified: frozenset[int]
    clusters: tuple[tuple[int, ...], ...]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return 0.0
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm <= 1e-15:
        return 0.0
    return float(np.dot(a, b) / norm)


def _decision_vector(decisions: np.ndarray, n_models: int) -> np.ndarray:
    decisions = np.asarray(decisions)
    one_hot: np.ndarray = np.zeros((len(decisions), n_models), dtype=np.float64)
    for row, decision in enumerate(decisions):
        index = int(decision)
        if 0 <= index < n_models:
            one_hot[row, index] = 1.0
    return one_hot.reshape(-1)


def find_duplicates(
    decisions: dict[int, np.ndarray],
    distributions: dict[int, np.ndarray],
    commit_blocks: dict[int, int | float],
    hotkeys: dict[int, str],
    *,
    decision_threshold: float = DECISION_THRESHOLD,
    distribution_threshold: float = DISTRIBUTION_THRESHOLD,
) -> DedupResult:
    """Cluster canonical behavior; earliest block/hotkey/UID survives."""
    uids = sorted(decisions)
    if set(uids) != set(distributions):
        raise ValueError("decisions and distributions must contain the same UIDs")
    if not uids:
        return DedupResult(frozenset(), ())
    if not 0 <= decision_threshold <= 1 or not 0 <= distribution_threshold <= 1:
        raise ValueError("dedup thresholds must be between zero and one")

    first_shape = np.asarray(distributions[uids[0]]).shape
    if len(first_shape) != 2 or first_shape[1] == 0:
        raise ValueError("routing distributions must have shape (N, M)")
    n_questions, n_models = first_shape
    decision_vectors: dict[int, np.ndarray] = {}
    distribution_vectors: dict[int, np.ndarray] = {}
    for uid in uids:
        uid_decisions = np.asarray(decisions[uid])
        uid_distributions = np.asarray(distributions[uid], dtype=np.float64)
        if uid_decisions.shape != (n_questions,) or uid_distributions.shape != first_shape:
            raise ValueError("all routing outputs must share canonical shapes")
        if not np.all(np.isfinite(uid_distributions)):
            raise ValueError("routing distributions must be finite")
        decision_vectors[uid] = _decision_vector(uid_decisions, n_models)
        distribution_vectors[uid] = uid_distributions.reshape(-1)

    parent = {uid: uid for uid in uids}

    def find(uid: int) -> int:
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for offset, left in enumerate(uids):
        for right in uids[offset + 1:]:
            decision_similarity = _cosine(
                decision_vectors[left], decision_vectors[right]
            )
            distribution_similarity = _cosine(
                distribution_vectors[left], distribution_vectors[right]
            )
            if (
                decision_similarity >= decision_threshold
                and distribution_similarity >= distribution_threshold
            ):
                union(left, right)

    components: dict[int, list[int]] = {}
    for uid in uids:
        components.setdefault(find(uid), []).append(uid)

    clusters: list[tuple[int, ...]] = []
    disqualified: set[int] = set()
    for members in components.values():
        if len(members) < 2:
            continue

        def seniority(uid: int) -> tuple[float, str, int]:
            block = float(commit_blocks.get(uid, math.inf))
            if not math.isfinite(block):
                block = math.inf
            return block, hotkeys.get(uid, ""), uid

        ordered = tuple(sorted(members, key=seniority))
        clusters.append(ordered)
        disqualified.update(ordered[1:])

    clusters.sort(key=lambda cluster: cluster[0])
    return DedupResult(frozenset(disqualified), tuple(clusters))
