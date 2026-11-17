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


DEFAULT_PATH = "data/embeddings/pool_v3.npz"


def embeddings_pin() -> str | None:
    """The sha256 a loaded embeddings file must have: config.EMBEDDINGS_SHA256,
    or None when the pool is declared unpinned (FUGAL_POOL_UNPINNED).

    The file holds one row per pool question, so a pool that is not the pinned
    one (a local testnet or rehearsal fixture) cannot have the pinned file. The
    waiver is the pool's own: FUGAL_POOL_UNPINNED is a consensus variable, so
    the consensus-environment check refuses it on finney, and elsewhere it is
    in the digest every neuron must share. Finney therefore always checks the
    compiled pin."""
    from fugal_subnet import config
    from fugal_subnet.benchmarks.loader import _unpinned_pool_declared

    return None if _unpinned_pool_declared() else config.EMBEDDINGS_SHA256


def fetch(url: str, expected_sha256: str, dest: str, *, timeout: float = 600.0) -> str:
    """Download the pinned embeddings file to `dest` and return its path.

    The pinned hash is the only thing trusted: the bytes are streamed to a
    temporary file beside `dest`, hashed, and moved into place only when the
    hash matches, so a wrong or truncated download never replaces a good file.
    A `dest` that already has the pinned hash is kept without a download.
    """
    import hashlib
    import os
    import tempfile
    from urllib.request import urlopen

    if not expected_sha256:
        raise ValueError("no pinned embeddings hash: nothing to fetch against")

    def sha_of(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    if os.path.exists(dest) and sha_of(dest) == expected_sha256:
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".embeddings-", dir=os.path.dirname(os.path.abspath(dest)))
    try:
        h = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, urlopen(url, timeout=timeout) as resp:  # noqa: S310
            for block in iter(lambda: resp.read(1 << 20), b""):
                h.update(block)
                out.write(block)
        if h.hexdigest() != expected_sha256:
            raise ValueError(f"downloaded embeddings have sha256 {h.hexdigest()}, the pinned "
                             f"one is {expected_sha256}; nothing was written to {dest}")
        os.replace(tmp, dest)
        tmp = ""
        return dest
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def main(argv=None) -> int:
    """`python -m fugal_subnet.embeddings_file fetch [--dest PATH]`: fetch the
    file pinned in config (EMBEDDINGS_URL, EMBEDDINGS_SHA256). With no pin yet
    (before the v3 release) it says so and exits 0, so an image built between
    releases still builds; `check_safety_invariants --release` requires both."""
    import argparse
    import os

    from fugal_subnet import config

    ap = argparse.ArgumentParser(prog="python -m fugal_subnet.embeddings_file")
    ap.add_argument("command", choices=["fetch"])
    ap.add_argument("--dest", default=os.getenv("FUGAL_EMBEDDINGS_FILE", DEFAULT_PATH))
    args = ap.parse_args(argv)
    if not (config.EMBEDDINGS_URL and config.EMBEDDINGS_SHA256):
        print("embeddings: no EMBEDDINGS_URL/EMBEDDINGS_SHA256 pinned in this release; "
              "nothing to fetch")
        return 0
    path = fetch(config.EMBEDDINGS_URL, config.EMBEDDINGS_SHA256, args.dest)
    print(f"embeddings: {path} matches the pinned sha256 {config.EMBEDDINGS_SHA256[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
