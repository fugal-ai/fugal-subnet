"""Protocol v5's proof check (tee/verify._v5_exam_check, verify_proof's v5
argument): a v5 proof answers exactly its assigned exam cells (I8)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fugal_subnet.tee import verify


def _r(q, e, exam=True):
    return SimpleNamespace(question_id=q, routed_model=e, exam=exam)


def _proof(results, spec_hash="a" * 64, scored=False, challenge=None, held=()):
    return SimpleNamespace(results=results, spec_hash=spec_hash, scored=scored,
                           challenge=challenge, holder_results=list(held))


def _exp(cells, spec_hash="a" * 64):
    return verify.V5Expected(v4=None, spec_hash=spec_hash, exam_cells=frozenset(cells))


CELLS = {("q1", "m/a"), ("q2", "m/b")}


def test_exactly_the_assigned_cells_pass():
    p = _proof([_r("q1", "m/a"), _r("q2", "m/b"), _r("c1", "m/c", exam=False)])
    assert verify._v5_exam_check(p, _exp(CELLS)) is None


@pytest.mark.parametrize("results,why", [
    ([_r("q1", "m/a")], "1 missing"),
    ([_r("q1", "m/a"), _r("q2", "m/b"), _r("q3", "m/a")], "1 not assigned"),
    ([_r("q1", "m/a"), _r("q2", "m/a")], "not this hotkey's assigned cells"),
    ([_r("q1", "m/a"), _r("q1", "m/a"), _r("q2", "m/b")], "answered twice"),
])
def test_anything_else_is_refused(results, why):
    assert why in verify._v5_exam_check(_proof(results), _exp(CELLS))


def test_the_spec_hash_is_bound():
    p = _proof([_r("q1", "m/a"), _r("q2", "m/b")], spec_hash="b" * 64)
    assert "not this epoch's" in verify._v5_exam_check(p, _exp(CELLS))


def test_no_challenge_and_no_holder_side_call():
    ok = [_r("q1", "m/a"), _r("q2", "m/b")]
    assert "no challenge" in verify._v5_exam_check(_proof(ok, scored=True), _exp(CELLS))
    assert "holder-side" in verify._v5_exam_check(_proof(ok, held=[object()]), _exp(CELLS))


def test_a_coverage_only_epoch_carries_no_exam():
    assert verify._v5_exam_check(_proof([_r("c", "m/a", exam=False)], spec_hash=None),
                                 _exp((), spec_hash=None)) is None
    assert "coverage only" in verify._v5_exam_check(_proof([_r("q1", "m/a")], spec_hash=None),
                                                    _exp((), spec_hash=None))


def test_verify_proof_needs_a_v5_expectation_under_v5_and_only_then():
    with pytest.raises(ValueError, match="V5Expected"):
        verify.verify_proof(SimpleNamespace(), set(), "", "", {}, protocol_version=5)
    with pytest.raises(ValueError, match="protocol v5 epoch"):
        verify.verify_proof(SimpleNamespace(), set(), "", "", {}, protocol_version=4,
                            v5=_exp(CELLS))
