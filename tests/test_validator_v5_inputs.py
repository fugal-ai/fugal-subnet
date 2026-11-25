"""The validator's protocol v5 inputs (neurons/validator._v5_load_inputs): the v5 pool, its
pinned routing embeddings, the exam part by RouterArena dataset and the mix. No network."""
from __future__ import annotations

import types

import pytest

from fugal_subnet.benchmarks import arena_sets


def _pool(n=200):
    sets = ("MMLUPro", "OpenTDB", "QANTA")
    return [{"question_id": f"arena_{sets[i % 3]}_{i:04d}", "benchmark": f"arena_{sets[i % 3]}",
             "prompt": "p", "route_text": "r", "gold": "A", "grader_id": "arena",
             "metadata": {"ra_dataset": sets[i % 3]}} for i in range(n)]


class _Emb:
    def __init__(self, ids):
        self.row = {q: i for i, q in enumerate(ids)}

    def covers(self, ids):
        return [q for q in ids if q not in self.row]


@pytest.fixture
def v(monkeypatch):
    from fugal_subnet import validator_setup
    from fugal_subnet.benchmarks import loader
    from neurons import validator
    pool = _pool()
    monkeypatch.setattr(loader, "load_pool", lambda version, strict=True: pool if version == 5
                        else pytest.fail("only the v5 pool is loaded here"))
    monkeypatch.setattr(validator_setup, "ensure_embeddings", lambda path, pin, url: "ok")
    monkeypatch.setattr(validator.Embeddings, "load",
                        classmethod(lambda cls, path, pin: _Emb([q["question_id"] for q in pool])))
    return validator, pool


@pytest.mark.parametrize("pin", ["TBD", None, ""])
def test_v5_refuses_to_load_without_an_embeddings_pin(v, pin):
    validator, _pool_ = v
    with pytest.raises(ValueError, match="not pinned"):
        validator._v5_load_inputs(types.SimpleNamespace(EMBEDDINGS_V5_SHA256=pin), live=True)


def test_v5_inputs_are_the_v5_pool_its_exam_part_and_routerarenas_mix(v):
    validator, pool = v
    got = validator._v5_load_inputs(types.SimpleNamespace(EMBEDDINGS_V5_SHA256="ab" * 32,
                                                          EMBEDDINGS_V5_URL=None), live=True)
    assert got.pool is pool and set(got.gold) == {q["question_id"] for q in pool}
    exam_ids = {q for ids in got.exam_by_dataset.values() for q in ids}
    assert exam_ids and all(not arena_sets.in_coverage(q) for q in exam_ids)
    assert exam_ids | {q["question_id"] for q in pool if arena_sets.in_coverage(q["question_id"])} \
        == {q["question_id"] for q in pool}
    assert got.weights_ppm == arena_sets.weights_ppm()


def test_v5_refuses_embeddings_that_miss_a_pool_question(v, monkeypatch):
    validator, pool = v
    monkeypatch.setattr(validator.Embeddings, "load",
                        classmethod(lambda cls, path, pin: _Emb([q["question_id"] for q in pool][1:])))
    with pytest.raises(ValueError, match="no row"):
        validator._v5_load_inputs(types.SimpleNamespace(EMBEDDINGS_V5_SHA256="ab" * 32,
                                                        EMBEDDINGS_V5_URL=None), live=True)
