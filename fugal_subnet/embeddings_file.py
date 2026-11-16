"""The pinned embeddings file: one row per pool question (protocol v3).

Both neurons read it. The miner's TD routes from it instead of embedding the
whole pool at boot (hours of CPU on every new TD), and every validator
computes heads' odds from it (fugal_subnet/head_odds.py). It is therefore a
consensus input: its sha256 is pinned (config.EMBEDDINGS_SHA256) and a file
that does not match is refused.
"""
from __future__ import annotations

import numpy as np


class Embeddings:
    """An .npz with `question_ids` (unicode) and `H` (float32, questions x
    HEAD_HIDDEN_DIM), loaded with allow_pickle=False and checked against the
    pinned hash and the bounds exact odds are computed under."""

    def __init__(self, question_ids, hidden):
        from fugal_subnet.head_odds import check_embedding_bounds

        if hidden.ndim != 2 or hidden.shape[0] != len(question_ids):
            raise ValueError("embeddings: one row per question id is required")
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("embeddings: a question id is repeated")
        check_embedding_bounds(hidden)
        self.hidden = hidden
        self.row = {q: i for i, q in enumerate(question_ids)}

    @classmethod
    def load(cls, path: str, expected_sha256: str | None) -> "Embeddings":
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if expected_sha256 and h.hexdigest() != expected_sha256:
            raise ValueError(f"embeddings file {path} has sha256 {h.hexdigest()}, the pinned "
                             f"one is {expected_sha256}")
        with np.load(path, allow_pickle=False) as z:
            ids = [str(x) for x in z["question_ids"].tolist()]
            hidden = np.array(z["H"], dtype=np.float32)
        return cls(ids, hidden)

    def of(self, question_id: str):
        return self.hidden[self.row[question_id]]

    def covers(self, question_ids) -> list[str]:
        """The ids with no row (empty when the file covers the pool)."""
        return [q for q in question_ids if q not in self.row]

    def aligned(self, question_ids) -> np.ndarray:
        """The rows for `question_ids`, in that order (what the miner's harness
        indexes by pool position). Raises by name if one has no row."""
        missing = self.covers(question_ids)
        if missing:
            raise ValueError(f"the embeddings file has no row for {len(missing)} question(s), "
                             f"e.g. {missing[0]}")
        return self.hidden[[self.row[q] for q in question_ids]]
