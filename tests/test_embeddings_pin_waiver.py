"""The embeddings pin is waived exactly where the pool pin is: a pool declared
unpinned (FUGAL_POOL_UNPINNED, refused on finney by the consensus-environment
check and recorded in the digest elsewhere) has no pinned embeddings, because
the file holds one row per pool question.

Found by the v3 local-chain rehearsal on the release candidate (2026-11-16):
once config.EMBEDDINGS_SHA256 was pinned, the rehearsal's fixture embeddings
(one row per fixture question) were refused by every miner and validator
("embeddings file ... has sha256 eb4e..., the pinned one is f64c..."), so no
v3 epoch could run off finney at all.
"""
import numpy as np
import pytest

from fugal_subnet import config
from fugal_subnet.embeddings_file import Embeddings, embeddings_pin


def _write(path, ids):
    from fugal_subnet.config import HEAD_HIDDEN_DIM

    np.savez(path, question_ids=np.array(ids),
             H=np.zeros((len(ids), HEAD_HIDDEN_DIM), dtype=np.float32))
    return str(path)


def test_the_pin_is_the_compiled_one_unless_the_pool_is_declared_unpinned(monkeypatch):
    monkeypatch.delenv("FUGAL_POOL_UNPINNED", raising=False)
    assert embeddings_pin() == config.EMBEDDINGS_SHA256
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "0")
    assert embeddings_pin() == config.EMBEDDINGS_SHA256
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "1")
    assert embeddings_pin() is None


def test_a_fixture_file_loads_only_with_an_unpinned_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDINGS_SHA256", "f" * 64)
    path = _write(tmp_path / "fixture.npz", ["q1", "q2"])
    monkeypatch.delenv("FUGAL_POOL_UNPINNED", raising=False)
    with pytest.raises(ValueError, match="pinned one"):
        Embeddings.load(path, embeddings_pin())
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "1")
    assert Embeddings.load(path, embeddings_pin()).covers(["q1", "q2"]) == []


def test_the_waiver_is_refused_on_finney():
    """The waiver rides on FUGAL_POOL_UNPINNED, which is a consensus variable:
    the consensus-environment check refuses it on finney."""
    assert "FUGAL_POOL_UNPINNED" in config.CONSENSUS_ENV_VARS


def test_both_neurons_load_embeddings_through_the_pin_helper():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for f in ("neurons/miner.py", "neurons/validator.py"):
        src = (root / f).read_text(encoding="utf-8")
        assert "embeddings_pin()" in src, f"{f} must take the pin from embeddings_pin()"
        assert "Embeddings.load(" in src
