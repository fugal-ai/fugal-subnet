from __future__ import annotations

import numpy as np
import pytest

import fugal_subnet.training as training_module
from fugal_subnet.head_eval import load_head_from_npz
from fugal_subnet.training import (
    TrainingData,
    generate_synthetic_data,
    load_legacy_npz,
    save_head,
    train_head,
)


def _data() -> TrainingData:
    hidden = np.zeros((4, 1024), dtype=np.float32)
    hidden[:, 0] = [1, -1, 1, -1]
    return TrainingData(
        questions=["q1", "q2", "q3", "q4"],
        matrix=np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int8),
        hidden_states=hidden,
        model_ids=["provider/a", "provider/b"],
        canonical_costs={"provider/a": 0.001, "provider/b": 0.002},
    )


def test_installed_trainer_uses_v2_evaluation_and_writes_safe_head(tmp_path):
    head, evaluation = train_head(_data(), epochs=20, learning_rate=0.05)
    assert head.models == ["provider/a", "provider/b"]
    assert evaluation.routing_decisions.shape == (4,)
    path = tmp_path / "head.npz"
    save_head(path, head)
    loaded = load_head_from_npz(path.read_bytes())
    np.testing.assert_allclose(loaded.W, head.W)
    assert loaded.models == head.models


def test_trainer_rejects_non_registry_or_duplicate_subsets():
    with pytest.raises(ValueError, match="registry subset"):
        train_head(_data(), selected_models=["provider/nope"], epochs=1)
    with pytest.raises(ValueError, match="registry subset"):
        train_head(_data(), selected_models=["provider/a", "provider/a"], epochs=1)


def test_legacy_import_is_pickle_disabled_and_requires_real_costs(tmp_path):
    valid = tmp_path / "legacy.npz"
    data = _data()
    np.savez(
        valid,
        matrix=data.matrix,
        models=np.asarray(data.model_ids),
        hidden_states=data.hidden_states,
        model_costs=np.asarray([0.001, 0.002]),
        prompts=np.asarray(data.questions),
    )
    loaded = load_legacy_npz(valid)
    assert loaded.canonical_costs["provider/b"] == 0.002

    missing = tmp_path / "missing.npz"
    np.savez(missing, matrix=data.matrix, models=np.asarray(data.model_ids), hidden_states=data.hidden_states)
    with pytest.raises(ValueError, match="model_costs"):
        load_legacy_npz(missing)


def test_synthetic_mode_is_deterministic_and_mock_only_data():
    first = generate_synthetic_data(["provider/a", "provider/b"], n_questions=5)
    second = generate_synthetic_data(["provider/a", "provider/b"], n_questions=5)
    np.testing.assert_array_equal(first.matrix, second.matrix)
    np.testing.assert_array_equal(first.hidden_states, second.hidden_states)


def test_historical_resolver_checks_exact_block_hash(monkeypatch):
    class FakeSubtensor:
        def get_block_hash(self, block):
            assert block == 10
            return "0x" + "a" * 64

        def get_commitment(self, netuid, uid, *, block):
            assert (netuid, uid, block) == (11, 7, 10)
            return "payload"

    class FakeBt:
        @staticmethod
        def Subtensor(*, network):
            assert network == "test"
            return FakeSubtensor()

    monkeypatch.setitem(__import__("sys").modules, "bittensor", FakeBt)
    resolver = training_module._historical_resolver("test", 11)
    receipt = type("Receipt", (), {
        "netuid": 11,
        "uid": 7,
        "block": 10,
        "block_hash": "a" * 64,
    })()
    assert resolver(receipt) == "payload"
    receipt.block_hash = "b" * 64
    with pytest.raises(RuntimeError, match="block hash"):
        resolver(receipt)
