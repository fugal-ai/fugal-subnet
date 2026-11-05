"""Reveals are published after the commit-reveal window, hash on chain.

Diligence 2026-10, additional finding A1: the reveal, the subnet's stated
product, was written to the validator's disk and never left it.
"""
from __future__ import annotations

import json

from fugal_subnet.reveal_publisher import (
    commitment_for,
    delay_epochs_for,
    due_reveals,
    publish_due_reveals,
    reveal_digest,
)


def _epochs(tmp_path, ids):
    for e in ids:
        d = tmp_path / e
        d.mkdir()
        (d / "reveal.json").write_text(json.dumps({"epoch_id": e}), encoding="utf-8")
    return str(tmp_path)


class Uploader:
    def __init__(self, fail_on=()):
        self.sent, self.fail_on = [], set(fail_on)

    def upload(self, path, key):
        if any(e in key for e in self.fail_on):
            raise OSError("bucket unreachable")
        self.sent.append(key)
        return f"gs://b/{key}"


def test_only_epochs_past_the_hold_are_due_and_each_is_published_once(tmp_path):
    root = _epochs(tmp_path, ["e00000001", "e00000002", "e00000003"])
    assert [p.split("/")[-2] for p in due_reveals(root, 3, 1)] == ["e00000001", "e00000002"]
    up, commits = Uploader(), []
    out = publish_due_reveals(root, 3, 1, 7, up, commit=lambda d: commits.append(d) or True)
    assert [r["epoch_id"] for r in out] == ["e00000001", "e00000002"]
    assert up.sent == ["7/e00000001/reveal.json", "7/e00000002/reveal.json"]
    digest = reveal_digest((tmp_path / "e00000002" / "reveal.json").read_bytes())
    assert commits == [commitment_for("e00000002", digest)]
    assert out[-1]["committed"] is True
    # Next epoch: nothing is re-sent, and the newest epoch becomes due.
    out2 = publish_due_reveals(root, 4, 1, 7, up, commit=lambda d: True)
    assert [r["epoch_id"] for r in out2] == ["e00000003"]


def test_a_failed_upload_is_not_marked_and_blocks_newer_epochs(tmp_path):
    root = _epochs(tmp_path, ["e00000001", "e00000002"])
    up = Uploader(fail_on=["e00000001"])
    assert publish_due_reveals(root, 5, 1, 7, up) == []
    assert not (tmp_path / "e00000001" / ".published").exists()
    assert up.sent == []


def test_the_hold_is_the_chain_commit_reveal_period_when_enabled():
    class HP:
        commit_reveal_weights_enabled = True
        commit_reveal_period = 3

    class Sub:
        def __init__(self, hp):
            self.hp = hp

        def get_subnet_hyperparameters(self, netuid):
            if self.hp is None:
                raise ConnectionError("no chain")
            return self.hp

    assert delay_epochs_for(Sub(HP()), 1) == 3
    HP.commit_reveal_weights_enabled = False
    assert delay_epochs_for(Sub(HP()), 1) == 1
    assert delay_epochs_for(Sub(None), 1) == 1


def test_a_reveal_that_does_not_parse_is_never_uploaded_or_committed(tmp_path):
    """A truncated reveal (a crash mid-write) used to be uploaded as-is and its
    hash committed on chain, with the marker stopping any later replacement."""
    root = _epochs(tmp_path, ["e00000001", "e00000002"])
    (tmp_path / "e00000001" / "reveal.json").write_text('{"epoch_id": "e0000', encoding="utf-8")
    up, commits = Uploader(), []
    out = publish_due_reveals(root, 5, 1, 7, up, commit=lambda d: commits.append(d) or True)
    assert out == [] and up.sent == [] and commits == []
    assert not (tmp_path / "e00000001" / ".published").exists()
    # A reveal naming another epoch is refused the same way.
    (tmp_path / "e00000001" / "reveal.json").write_text(json.dumps({"epoch_id": "e00000009"}),
                                                        encoding="utf-8")
    assert publish_due_reveals(root, 5, 1, 7, up) == [] and up.sent == []


def test_the_writer_leaves_no_partial_file_behind(tmp_path):
    from fugal_subnet.commit_reveal import write_json_atomic

    target = tmp_path / "reveal.json"
    write_json_atomic(str(target), {"epoch_id": "e1", "n": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"epoch_id": "e1", "n": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["reveal.json"]      # no .tmp left
