"""The proof collection point is consensus state (I1, I6, I9).

A miner cannot benchmark before its epoch's boundary block exists, so there is
an unavoidable gap between an epoch starting and any proof existing for it.
The validator therefore waits — and *when* it stops waiting decides which
miners it scores, which makes the offset a consensus parameter rather than a
tuning knob. These tests pin the three properties that follow from that.
"""
import ast
import pathlib

from fugal_subnet.benchmarks.slicer import collect_block_for_epoch

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_collection_block_is_deterministic():
    a = collect_block_for_epoch(1234, 360, 0.5)
    b = collect_block_for_epoch(1234, 360, 0.5)
    assert a == b == 1234 * 360 + 180


def test_collection_block_stays_inside_its_epoch():
    """A misconfigured fraction must not collect before the epoch exists or
    after it has ended — either would silently score the wrong epoch."""
    for bpe in (2, 10, 150, 360, 3600):
        boundary = 7 * bpe
        for fraction in (-1.0, 0.0, 0.5, 1.0, 99.0):
            block = collect_block_for_epoch(7, bpe, fraction)
            assert boundary < block < boundary + bpe, (bpe, fraction, block)


def test_collection_fraction_is_in_the_consensus_digest():
    """Two validators collecting at different offsets see different fields and
    publish different weights for honest reasons. That must be diagnosable from
    the published reveal, which means it belongs in the digest."""
    from fugal_subnet.fingerprint import environment_fingerprint

    assert "collect_fraction" in environment_fingerprint()["epoch"]


def test_digest_changes_when_the_collection_fraction_changes(monkeypatch):
    from fugal_subnet import config, fingerprint

    before = fingerprint.consensus_digest()
    # environment_fingerprint imports from config at call time, so patching
    # the constant there is what a differently-configured validator looks like.
    monkeypatch.setattr(config, "EPOCH_COLLECT_FRACTION", 0.75)
    assert fingerprint.consensus_digest() != before


def test_only_the_slicer_derives_the_collection_block():
    """Structural, not behavioural. A second place that computes boundary +
    offset is a consensus break, and a test that derives the value once and
    hands it to both sides cannot catch that — which is exactly how the epoch
    id came to be formatted two different ways and stopped the subnet setting
    weights at all."""
    offenders = []
    for path in (REPO / "neurons").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # boundary_block + <anything> outside the slicer is the shape of a
            # hand-rolled collection point.
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
                    and isinstance(node.left, ast.Name)
                    and node.left.id in ("boundary_block", "boundary")):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "collection block derived outside slicer.collect_block_for_epoch: "
        + ", ".join(offenders)
    )


def test_accuracy_anomalies_read_the_key_the_validator_writes():
    """A warning that is always wrong teaches operators to skip the line.

    detect_anomalies read `acc` while the validator writes `accuracy`, so the
    low-accuracy anomaly fired on every epoch and the high-accuracy one could
    never fire. Caught on a live epoch reporting "all miners <10%" while the
    miners had scored 62% and 46%.
    """
    from fugal_subnet.epoch_logger import detect_anomalies

    weights = {2: 0.45, 3: 0.55}
    healthy = {2: {"accuracy": 0.62}, 3: {"accuracy": 0.46}}
    assert not [a for a in detect_anomalies(healthy, weights, 3, 3)
                if "accuracy" in a]

    genuinely_low = {2: {"accuracy": 0.02}, 3: {"accuracy": 0.05}}
    assert any("universally_low_accuracy" in a
               for a in detect_anomalies(genuinely_low, weights, 3, 3))

    genuinely_high = {2: {"accuracy": 0.99}, 3: {"accuracy": 0.98}}
    assert any("suspiciously_high_accuracy" in a
               for a in detect_anomalies(genuinely_high, weights, 3, 3))


def test_reveal_carries_exploration_observations(tmp_path, monkeypatch):
    """The published artifact must contain the samples a router can learn from.

    `matrix` only ever holds models a miner CHOSE, so on a field of similar
    heads it collapses to a couple of columns — measured on netuid 552, 2 of 17
    models observed, which is untrainable. Exploration answers use a
    nonce-chosen model the miner does not pick, so they are the only coverage of
    the rest of the pool and the only unbiased samples in the artifact.
    """
    import json

    import numpy as np

    from fugal_subnet import commit_reveal

    monkeypatch.setattr(commit_reveal, "EPOCH_DIR", str(tmp_path))
    questions = [{"question_id": "q1", "prompt": "p", "gold": "1", "grader_id": "x"}]
    commit_reveal.commit_epoch("e1", questions, "0xabc")

    exploration = [
        {"question_id": "z9", "model": "b", "correct": True,
         "prompt_tokens": 1, "completion_tokens": 1},
        {"question_id": "z9", "model": "a", "correct": False,
         "prompt_tokens": 1, "completion_tokens": 1},
    ]
    assert commit_reveal.reveal_epoch(
        "e1", questions, np.array([[1]]), ["a"], {"a": 0.1},
        {}, {}, {}, {}, exploration=exploration,
    )

    revealed = json.loads((tmp_path / "e1" / "reveal.json").read_text())
    assert len(revealed["exploration"]) == 2
    # Deterministically ordered, or two validators publish different bytes.
    assert [e["model"] for e in revealed["exploration"]] == ["a", "b"]


def test_reveal_without_exploration_still_verifies():
    """Backward compatible: the argument is optional and an absent list is
    an empty one, not a crash."""
    import inspect

    from fugal_subnet.commit_reveal import reveal_epoch

    sig = inspect.signature(reveal_epoch)
    assert sig.parameters["exploration"].default is None
