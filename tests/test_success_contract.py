"""Success training and fresh evidence boundaries, without paid inference."""
import io
import json
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest
import torch

from fugal_subnet import routing_protocol
from fugal_subnet import success_test_fixtures as fixtures
from fugal_subnet.head_eval import evaluate_success_head, load_head_from_npz
from fugal_subnet.reference_frame import ReferenceFrame, rebuild_from_reveals
from fugal_subnet.success_training import fit, grouped_split, masked_bce
from fugal_subnet.tee.harness import _route_question
from fugal_subnet.vendor import success_contract as c


def test_masked_bce_keeps_all_failed_questions_and_ignores_missing():
    logits = torch.tensor([[0., 1.], [2., 3.]], requires_grad=True)
    y = torch.tensor([[0., 0.], [1., float("nan")]])
    loss = masked_bce(logits, y)
    loss.backward()
    assert logits.grad[0].gt(0).all()  # observed failures push predictions down
    assert logits.grad[1, 0] < 0
    assert logits.grad[1, 1] == 0
    with pytest.raises(ValueError):
        masked_bce(logits, torch.full((2, 2), float("nan")))


def test_duplicate_groups_are_disjoint_and_checkpoint_uses_validation():
    questions = [f"q{i}" for i in range(50)] * 2
    tr, va, te = grouped_split(questions)
    assert [len(x) for x in (tr, va, te)] == [60, 20, 20]
    assert not set(np.array(questions)[tr]) & set(np.array(questions)[va])
    assert not set(np.array(questions)[te]) & set(np.array(questions)[tr])
    h = np.zeros((100, 1024), np.float32)
    y = np.ones((100, 2))
    y[va] = 0
    _, _, epoch, history = fit(h, y, tr, va, epochs=5)
    assert epoch == 1
    assert history[0]["validation_bce"] == min(r["validation_bce"] for r in history)


def test_lambda_environment_cannot_change_benchmark_route():
    head = load_head_from_npz(fixtures.head_bytes(np.zeros((2, 1024)), np.array([0., .1]), ["a", "b"]))
    costs = np.array([0., .1])
    for value in ("0", "10000", "nan"):
        with mock.patch.dict("os.environ", {"FUGAL_LAMBDA": value, "FUGAL_TRAINING_COST_LAMBDA": value}):
            assert _route_question(head, np.zeros(1024), costs) == 0
    assert c.rank(c.predictions(head.W, head.b, np.zeros(1024)), costs, 0)[0] == 1


def test_old_head_cannot_enter_benchmark_and_candidates_cannot_deploy():
    buf = io.BytesIO()
    np.savez(buf, W=np.zeros((1, 1024)), b=np.zeros(1), models=np.array(["a"]))
    with pytest.raises(ValueError, match="legacy"):
        routing_protocol.benchmark_costs(load_head_from_npz(buf.getvalue()))
    manifest = json.loads(routing_protocol.MANIFEST_PATH.read_text())
    with pytest.raises(ValueError, match="reviewed"):
        c.validate_manifest(manifest, deployable=True)


def test_offline_evaluation_keeps_observed_failures_and_aligns_columns():
    head = load_head_from_npz(fixtures.head_bytes(np.zeros((2, 1024)), np.zeros(2), ["b", "a"]))
    y = np.array([[1., 0.], [0., 0.], [0., np.nan]])
    score = evaluate_success_head(head, np.zeros((3, 1024)), y, ["a", "b"], {"a": 0., "b": 0.})
    assert score.n_scored == 2 and score.n_correct == 0


def test_protocol_is_hashed_into_proofs_and_old_proofs_are_rejected():
    from fugal_subnet.attacks.run_tee_attacks import _proof, _verify
    proof = _proof([])
    old = replace(proof, routing_protocol="")
    assert old.content_hash() != proof.content_hash()
    verdict = _verify(old)
    assert not verdict.valid and "protocol" in verdict.reason.lower()
    serialized = proof.to_dict()
    serialized.pop("routing_protocol")
    assert type(proof).from_dict(serialized).routing_protocol == ""


def test_old_reference_frame_and_reveals_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="namespace"):
        ReferenceFrame.from_dict({"successes": {"a": 3}})
    p = tmp_path / "e1"
    p.mkdir()
    (p / "reveal.json").write_text(json.dumps({"epoch_id": "e1", "exploration": []}))
    with pytest.raises(ValueError, match="namespace"):
        rebuild_from_reveals(tmp_path)


def test_incompatible_validator_state_never_silently_resumes(tmp_path):
    import sys

    from tests import bt_mock
    with mock.patch.dict(sys.modules, {"bittensor": bt_mock}):
        from neurons import validator
    from fugal_subnet.rewards import MinerRecord
    path = tmp_path / "state.json"
    path.write_text('{"records": {}}')
    with mock.patch.object(validator, "STATE_PATH", str(path)), pytest.raises(ValueError, match="namespace"):
        validator.load_state(MinerRecord)


def test_miner_statistics_must_match_manifest_even_with_correct_identity():
    z = fixtures.arrays(np.zeros((1, 1024)), np.zeros(1), ["a"])
    z["mean_in_tokens"][0] += 1
    with pytest.raises(ValueError, match="disagree"):
        c.check_manifest(z, fixtures.manifest(["a"]))
