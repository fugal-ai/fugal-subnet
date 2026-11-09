"""The commit and the reveal never take an epoch's weights down (I6).

reveal_epoch runs BEFORE set_weights in the validator's epoch, so an exception
from it (a full disk, a value JSON cannot encode such as a numpy float32 in a
score) used to end the epoch with no weights set. commit_epoch runs at the
start of the epoch, with the same effect. Both now log, return a failed
reveal, and leave no partial file (found 2026-11-08, theme-4 review).
"""
import json
import os

import numpy as np
import pytest

from fugal_subnet import commit_reveal as cr

QUESTIONS = [{"question_id": "q1", "prompt": "1+1?", "gold": "2"}]


@pytest.fixture
def epoch_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "EPOCH_DIR", str(tmp_path))
    return tmp_path


def _reveal(scores=None):
    return cr.reveal_epoch("e1", QUESTIONS, np.array([[1]]), ["m/a"], {"m/a": 0.01},
                           {1: "h" * 64}, {1: [0]}, scores or {1: {"accuracy": 0.5}}, {1: 1.0})


def _leftovers(d):
    return sorted(p.name for p in (d / "e1").iterdir())


def test_a_normal_commit_and_reveal(epoch_dir):
    cr.commit_epoch("e1", QUESTIONS, "0xabc")
    assert _reveal() is True
    assert json.loads((epoch_dir / "e1" / "reveal.json").read_text())["epoch_id"] == "e1"
    assert _leftovers(epoch_dir) == ["commit.json", "reveal.json"]


def test_a_value_json_cannot_encode_is_a_failed_reveal_not_an_exception(epoch_dir):
    cr.commit_epoch("e1", QUESTIONS, "0xabc")
    assert _reveal(scores={1: {"accuracy": np.float32(0.5)}}) is False
    assert _leftovers(epoch_dir) == ["commit.json"]          # no reveal, no temp file


def test_a_write_failure_is_a_failed_reveal(epoch_dir, monkeypatch):
    cr.commit_epoch("e1", QUESTIONS, "0xabc")

    def full_disk(path, obj):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cr, "write_json_atomic", full_disk)
    assert _reveal() is False


@pytest.mark.parametrize("content", ["{not json", json.dumps({"commit_hash": "x"}), "[]"])
def test_an_unreadable_commitment_is_a_failed_reveal(epoch_dir, content):
    (epoch_dir / "e1").mkdir()
    (epoch_dir / "e1" / "commit.json").write_text(content)
    assert _reveal() is False


def test_a_commit_that_cannot_be_written_still_returns_its_commitment(epoch_dir, monkeypatch, caplog):
    def full_disk(path, obj):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cr, "write_json_atomic", full_disk)
    c = cr.commit_epoch("e1", QUESTIONS, "0xabc")
    assert c.commit_hash == cr.compute_commit_hash(QUESTIONS, c.grader_hash)
    assert any("could not be written" in r.getMessage() for r in caplog.records)
    monkeypatch.undo()
    monkeypatch.setattr(cr, "EPOCH_DIR", str(epoch_dir))
    assert _reveal() is False                               # no commitment on disk to reveal against


def test_an_atomic_write_that_fails_leaves_nothing_behind(tmp_path):
    target = tmp_path / "x.json"
    with pytest.raises(TypeError):
        cr.write_json_atomic(str(target), {"v": np.float32(1.0)})
    assert os.listdir(tmp_path) == []
