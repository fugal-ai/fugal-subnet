"""Interrupted writes retain the last complete, profile-bound embedding checkpoint."""
from pathlib import Path

import pytest

from scripts.evaluate_success_contract import load_progress, save_progress


def test_interrupted_checkpoint_preserves_previous_progress(tmp_path, monkeypatch):
    path = tmp_path / "progress.npz"
    questions = ["one", "two"]
    hidden, completed, seconds, peak = load_progress(path, questions, "profile")
    assert completed == seconds == peak == 0
    hidden[0, 0] = 1
    save_progress(path, questions, "profile", hidden, 1, 3., 42)
    hidden[1, 1] = 1

    def interrupted(*args):
        raise OSError("simulated interrupted atomic replace")

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        save_progress(path, questions, "profile", hidden, 2, 6., 50)
    restored, completed, seconds, peak = load_progress(path, questions, "profile")
    assert (completed, seconds, peak) == (1, 3., 42)
    assert restored[0, 0] == 1 and not restored[1].any()
    with pytest.raises(ValueError, match="profile/input mismatch"):
        load_progress(path, questions, "different-profile")
    with pytest.raises(ValueError, match="profile/input mismatch"):
        load_progress(path, questions[::-1], "profile")
