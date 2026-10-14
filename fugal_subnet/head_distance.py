"""Parameter-space distance between two router heads.

Behavioural dedup compares routing DECISIONS: two heads that pick the same
model on 99% of a slice are clones. That threshold is a lever a copier can
pull by hand -- nudge one bias term until three or four of 300 decisions
change, and the copy passes (diligence 2026-10, finding 5; measured with
`dedup.find_duplicates`: copies differing on 0-2 decisions caught, 3 or more
not). A nudged copy is still the same head in the space that matters: its
weights sit at a relative distance of 1e-4 from the original, where two
independently trained heads sit near sqrt(2). This module measures that.

`relative_distance` is the Frobenius norm of the difference of the two
`[W | b]` matrices, rows aligned by model name, divided by the larger of the
two norms. It is scale-free, symmetric, and 0 for an exact copy. Rows are
aligned by model id because a copier could reorder rows for free; if the two
heads share too few models to compare (`min_overlap`), they are unrelated and
the distance is infinite. Renaming rows to dodge alignment is not a free move
either: every model id must be in the pinned price table or the head is
rejected at load (`routing_protocol.benchmark_costs`).
"""
from __future__ import annotations

import math

import numpy as np


def _parts(head) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if isinstance(head, dict):
        return np.asarray(head["W"]), np.asarray(head["b"]), list(head["models"])
    return np.asarray(head.W), np.asarray(head.b), list(head.models)


def relative_distance(head_a, head_b, min_overlap: float = 0.8) -> float:
    """||[W|b]_a - [W|b]_b||_F / max(||[W|b]_a||_F, ||[W|b]_b||_F), rows aligned
    by model id. `inf` when the heads share fewer than `min_overlap` of the
    larger head's models, or have different hidden widths.

    Accepts `HeadArtifact`s or `{"W", "b", "models"}` dicts.
    """
    W_a, b_a, m_a = _parts(head_a)
    W_b, b_b, m_b = _parts(head_b)
    if W_a.ndim != 2 or W_b.ndim != 2 or W_a.shape[1] != W_b.shape[1]:
        return math.inf
    shared = sorted(set(m_a) & set(m_b))
    if not shared or len(shared) < min_overlap * max(len(m_a), len(m_b)):
        return math.inf
    ia = [m_a.index(m) for m in shared]
    ib = [m_b.index(m) for m in shared]
    A = np.concatenate([W_a[ia], b_a[ia][:, None]], axis=1).astype(np.float64)
    B = np.concatenate([W_b[ib], b_b[ib][:, None]], axis=1).astype(np.float64)
    scale = max(float(np.linalg.norm(A)), float(np.linalg.norm(B)))
    if scale < 1e-12:
        return 0.0
    return float(np.linalg.norm(A - B) / scale)
