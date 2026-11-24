"""Protocol v5 in the TD (~/projects/fugal-v2-handoff/v5/DESIGN.md sections 5 and 7).

The TD labels the exam cells the root's spec (in its reveal of E - SPEC_LAG)
assigns to its hotkey, plus coverage cells; it makes no challenge; every call
is held to the v5 deadline and none starts after the v5 budget, so the proof
is ready before collection; each result carries `exam` and its label `y`, and
the proof the spec's hash. No network, no spend.

The exam spec here is a TEST SHIM with the interface lane V1's
fugal_subnet/exam_v5.py provides (INTERFACES.md): the coordinator runs these
tests against the real module at integration.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import types

import numpy as np
import pytest

from fugal_subnet.tee import harness, td_v4, td_v5
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from tests import menu_fixtures as mf
from tests.test_harness_v4 import FIXTURE_IDS, HOTKEY, NONCE, Provider, _inputs, _Proxy
from tests.test_tee_e2e import _head, _pool

OTHER = "5OtherHotkeyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    mf.allow_small_menus(monkeypatch)


# --- TEST SHIM for lane V1's exam_v5 (ExamSpec, assigned_cells) ------------------------


@dataclasses.dataclass(frozen=True)
class _Spec:
    epoch: int
    questions: tuple
    cells: tuple
    assignment: dict

    def to_json(self) -> str:
        return json.dumps({"epoch": self.epoch, "questions": list(self.questions),
                           "cells": [list(c) for c in self.cells],
                           "assignment": [[list(c), list(h)] for c, h in
                                          sorted(self.assignment.items())]},
                          sort_keys=True, separators=(",", ":"))

    def spec_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @classmethod
    def from_json(cls, s: str) -> _Spec:
        d = json.loads(s)
        return cls(epoch=d["epoch"], questions=tuple(d["questions"]),
                   cells=tuple(tuple(c) for c in d["cells"]),
                   assignment={tuple(c): tuple(h) for c, h in d["assignment"]})


_SHIM = types.SimpleNamespace(
    ExamSpec=_Spec,
    assigned_cells=lambda spec, hotkey: [c for c, hks in spec.assignment.items() if hotkey in hks])


@pytest.fixture(autouse=True)
def _exam_shim(monkeypatch):
    monkeypatch.setattr(td_v5, "_exam_v5", lambda: _SHIM)


def _spec(epoch=104, *, mine=(("q1", FIXTURE_IDS[0]), ("q2", FIXTURE_IDS[1]), ("q2", FIXTURE_IDS[2])),
          theirs=(("q3", FIXTURE_IDS[0]),)) -> _Spec:
    assignment = {c: (HOTKEY, OTHER) for c in mine}
    assignment.update({c: (OTHER,) for c in theirs})
    cells = tuple(sorted(assignment))
    return _Spec(epoch=epoch, questions=tuple(sorted({c[0] for c in cells})), cells=cells,
                 assignment=assignment)


def _reveal(epoch_id: str, spec: _Spec | None, **field_over) -> tuple[bytes, str]:
    doc = {"epoch_id": epoch_id, "champion": {"champion": None}}
    if spec is not None:
        field = {"epoch": spec.epoch, "spec": spec.to_json(), "spec_hash": spec.spec_hash()}
        field.update(field_over)
        doc[td_v5.REVEAL_SPEC_KEY] = field
    raw = json.dumps(doc).encode()
    return raw, hashlib.sha256(raw).hexdigest()


# --- reading the spec from the root's reveal ----------------------------------------------


def test_spec_is_read_from_the_committed_reveal():
    spec = _spec()
    raw, sha = _reveal("e00000102", spec)
    got = td_v5.exam_spec_from_reveal(raw, "e00000102", sha)
    assert got == spec and got.spec_hash() == spec.spec_hash()


def test_a_reveal_without_a_spec_names_none():
    raw, sha = _reveal("e00000102", None)
    assert td_v5.exam_spec_from_reveal(raw, "e00000102", sha) is None


@pytest.mark.parametrize("case", ["sha", "epoch_id", "hash", "epoch", "shape"])
def test_a_spec_that_does_not_check_is_refused(case):
    spec = _spec()
    over = {"hash": {"spec_hash": "0" * 64}, "epoch": {"epoch": 999},
            "shape": {"spec_hash": "not-hex"}}.get(case, {})
    raw, sha = _reveal("e00000102", spec, **over)
    if case == "sha":
        sha = "f" * 64
    with pytest.raises(ValueError):
        td_v5.exam_spec_from_reveal(raw, "e00000103" if case == "epoch_id" else "e00000102",
                                    sha)


# --- the exam policy ---------------------------------------------------------------------


def _policy(read, first=102):
    return td_v5.ExamPolicy(first_epoch=first, hotkey=HOTKEY, read_spec=read)


def test_the_first_epochs_after_activation_are_coverage_only():
    calls = []
    pol = _policy(lambda e: calls.append(e))
    assert [pol(e) for e in range(102, 102 + td_v5.SPEC_LAG)] == [None] * td_v5.SPEC_LAG
    assert calls == []
    with pytest.raises(ValueError):
        pol(101)


def test_the_exam_is_this_hotkeys_cells_of_the_spec_two_epochs_back():
    spec = _spec(104)
    task = _policy(lambda e: (e - td_v5.SPEC_LAG, spec))(104)
    assert task.epoch == 104 and task.spec_hash == spec.spec_hash()
    assert task.cells == (("q1", FIXTURE_IDS[0]), ("q2", FIXTURE_IDS[1]), ("q2", FIXTURE_IDS[2]))


@pytest.mark.parametrize("read", [
    lambda e: (_ for _ in ()).throw(LookupError("no reveal")),   # unreadable
    lambda e: None,                                              # no commitment
    lambda e: (e - 1, _spec(e)),                                 # newest reveal not E - lag
    lambda e: (e - 2, None),                                     # reveal without a spec
    lambda e: (e - 2, _spec(e + 1)),                             # spec for another epoch
])
def test_an_unusable_spec_means_coverage_only(read):
    assert _policy(read)(104) is None


def test_the_challenge_is_gated_off_from_the_v5_epoch():
    seen = []

    class Policy:
        size = 500

        def __call__(self, epoch):
            seen.append(epoch)
            return "challenge"

    gated = td_v5.gate_challenge(Policy(), 104)
    assert gated(103) == "challenge" and gated(104) is None and gated(105) is None
    assert seen == [103]
    assert td_v5.gate_challenge(None, 104)(103) is None


def test_the_v5_session_never_hands_a_challenge_to_the_harness():
    pool = _pool()

    class V4:
        def inputs_for(self, epoch):
            return _inputs(pool, challenge=td_v4.ChallengeSpec(12, None))

        def ran(self, inputs, proof):
            return "ran"

    session = td_v5.V5Session(v4_session=V4(), exam=_policy(lambda e: (e - 2, _spec(e))),
                              first_epoch=102)
    got = session.inputs_for(104)
    assert got.v4.challenge is None and got.exam.epoch == 104
    assert session.ran(got, None) == "ran"
    with pytest.raises(ValueError):
        session.inputs_for(101)


# --- the harness -------------------------------------------------------------------------

COVERAGE = {"q10": FIXTURE_IDS[2], "q11": FIXTURE_IDS[0]}


def _run_v5(monkeypatch, provider=None, *, exam=True, coverage=COVERAGE, budget_s=None,
            clock=None):
    provider = provider or Provider()
    pool = _pool()
    monkeypatch.setattr(harness, "_call_model", provider.call)
    seen_excluded = []

    def fake_coverage(nonce, p, excluded, state, hotkey, index=None):
        seen_excluded.append(set(excluded))
        return dict(coverage)
    monkeypatch.setattr(td_v5, "coverage_v5", fake_coverage)
    if clock is not None:
        monkeypatch.setattr(harness, "_clock", clock)
    spec = _spec(104)
    task = (td_v5.ExamTask(epoch=104, spec=spec, spec_hash=spec.spec_hash(),
                           cells=tuple(sorted(_SHIM.assigned_cells(spec, HOTKEY))))
            if exam else None)
    v4_inputs = _inputs(pool, challenge=None, fetch_listing=provider.fetch)
    inputs = td_v5.V5EpochInputs(epoch=104, v4=v4_inputs, exam=task,
                                 **({"budget_s": budget_s} if budget_s is not None else {}))
    proof = harness.run_benchmark(
        nonce=NONCE, head_bytes=_head(FIXTURE_IDS), benchmark_pool=pool, proxy=_Proxy(port=0),
        hidden_states=np.zeros((len(pool), 1024), np.float32), slice_size=12,
        epoch_id="e00000104", source_hash="img", hotkey=HOTKEY, protocol_version=5, v5=inputs)
    return proof, provider, spec, seen_excluded


def test_the_td_calls_exactly_its_cells_then_coverage(monkeypatch):
    proof, provider, spec, excluded = _run_v5(monkeypatch)
    called = [(qid, m) for m, _req, qid in provider.calls]
    mine = [("q1", FIXTURE_IDS[0]), ("q2", FIXTURE_IDS[1]), ("q2", FIXTURE_IDS[2])]
    assert sorted(called) == sorted(mine + [(q, e) for q, e in COVERAGE.items()])
    assert ("q3", FIXTURE_IDS[0]) not in called             # another hotkey's cell
    assert [(r.question_id, r.routed_model) for r in proof.results if r.exam] == mine
    assert all(r.is_exploration and r.exam is False for r in proof.results if not r.exam)
    # Coverage never draws the spec's questions.
    assert excluded == [set(spec.questions)]
    assert proof.spec_hash == spec.spec_hash() and proof.scored is False
    assert proof.challenge is None


def test_every_result_is_labelled_and_correct_means_a_full_label(monkeypatch):
    proof, *_ = _run_v5(monkeypatch, Provider(fail_ids={"q2"}))
    for r in proof.results:
        assert r.y in range(td_v5.Y_SCALE + 1)
        assert r.correct == (r.y == td_v5.Y_SCALE)
        if r.failed:
            assert r.y == 0
    assert any(r.failed for r in proof.results)


def test_the_proof_round_trips_and_binds_the_v5_fields(monkeypatch):
    proof, *_ = _run_v5(monkeypatch)
    back = BenchmarkProof.from_dict(json.loads(json.dumps(proof.to_dict())))
    assert back.content_hash() == proof.content_hash()
    assert back.spec_hash == proof.spec_hash
    tampered = dataclasses.replace(back, spec_hash="0" * 64)
    assert tampered.content_hash() != proof.content_hash()


def test_a_coverage_only_epoch_has_no_spec_hash(monkeypatch):
    proof, provider, *_ = _run_v5(monkeypatch, exam=False)
    assert proof.spec_hash is None and all(not r.exam for r in proof.results)
    assert len(provider.calls) == len(COVERAGE)
    assert "spec_hash" not in proof.to_dict()


def test_every_call_is_held_to_the_v5_deadline(monkeypatch):
    deadlines = []
    provider = Provider()
    inner = provider.call

    def call(proxy, model_id, question, request=None, reasoning=None, deadline_s=None):
        deadlines.append(deadline_s)
        return inner(proxy, model_id, question, request=request, reasoning=reasoning,
                     deadline_s=deadline_s)
    provider.call = call
    _run_v5(monkeypatch, provider)
    assert deadlines and all(d is not None and d <= td_v5.EXAM_CALL_DEADLINE_S
                             for d in deadlines)


def test_no_call_starts_after_the_budget_and_those_are_reported_not_started(monkeypatch):
    ticks = iter(range(0, 10_000, 100))
    proof, *_ = _run_v5(monkeypatch, budget_s=150, clock=lambda: next(ticks))
    cut = [r for r in proof.results if r.not_started]
    assert cut, "the budget cut nothing"
    for r in cut:
        assert r.failed and r.y == 0 and not r.correct and r.cost_usd == 0
        assert r.prompt_tokens == 0 and r.completion_tokens == 0 and r.provider_cost is None
    # Exam cells are started first: once one call is cut, every later one is.
    flags = [bool(r.not_started) for r in proof.results]
    assert flags == sorted(flags)
    back = BenchmarkProof.from_dict(json.loads(json.dumps(proof.to_dict())))
    assert back.content_hash() == proof.content_hash()


def test_the_timing_fits_the_collection_window():
    # The last call starts by EXAM_BUDGET_S and ends by its deadline (plus the
    # harness's wait margin): inside the 36 minutes to collection.
    from fugal_subnet import config
    window_s = 360 * 12 * config.EPOCH_COLLECT_FRACTION
    assert td_v5.EXAM_BUDGET_S + td_v5.EXAM_CALL_DEADLINE_S + config.HARNESS_WAIT_MARGIN_S < (
        window_s - 120)


# --- the v5 result fields ----------------------------------------------------------------


def _result(**over) -> dict:
    d = {"question_id": "q1", "routed_model": "m", "correct": True, "cost_usd": 0.001,
         "response_hash": "h", "prompt_tokens": 5, "completion_tokens": 7,
         "is_exploration": False, "exam": True, "y": td_v5.Y_SCALE}
    d.update(over)
    return d


def test_a_v5_result_reads_and_hashes_its_fields():
    r = QuestionResult.from_dict(_result())
    assert r.exam is True and r.y == td_v5.Y_SCALE
    assert {"exam", "y"} <= set(r.hash_payload())
    assert QuestionResult.from_dict(_result(exam=False, is_exploration=True)).exam is False


def test_an_earlier_result_hashes_as_before():
    d = _result()
    del d["exam"], d["y"]
    r = QuestionResult.from_dict(d)
    assert "exam" not in r.hash_payload() and "y" not in r.hash_payload()


@pytest.mark.parametrize("over", [
    {"y": None},                                    # exam without y (key present as null)
    {"y": 1.0}, {"y": True}, {"y": td_v5.Y_SCALE + 1}, {"y": -1},
    {"exam": 1},
    {"correct": False},                             # a full label must be correct
    {"y": 5000},                                    # a partial label is not correct
    {"failed": True, "correct": False, "y": 3},     # a failed call is labelled 0
    {"is_exploration": True},                       # an exam cell is not exploration
    {"holder_pick": True},                          # no holder-side call under v5
])
def test_an_inconsistent_v5_result_is_refused(over):
    d = _result(**over)
    if d.get("y") is None:
        d.pop("y")
    with pytest.raises((ValueError, TypeError)):
        QuestionResult.from_dict(d)


def test_v5_keys_are_not_read_in_an_earlier_namespace():
    r = QuestionResult.from_dict(_result(y="garbage"), v4_fields=False)
    assert r.exam is None and r.y is None


def test_grade_y_falls_back_to_the_v4_grader_and_reads_arena_grades(monkeypatch):
    from fractions import Fraction
    q = {"question_id": "q1", "prompt": "?", "gold": "4", "grader_id": "numeric_final",
         "benchmark": "gsm8k", "metadata": {}}
    assert td_v5.grade_y(q, "4") == td_v5.Y_SCALE and td_v5.grade_y(q, "5") == 0
    fake = types.ModuleType("fugal_subnet.graders_arena")
    import sys
    monkeypatch.setitem(sys.modules, "fugal_subnet.graders_arena", fake)
    for given, want in [(Fraction(1, 2), td_v5.Y_SCALE // 2), (True, td_v5.Y_SCALE), (7, 7)]:
        fake.grade = lambda q, t, g=given: g
        assert td_v5.grade_y(q, "x") == want
    for bad in (0.5, Fraction(3, 2), td_v5.Y_SCALE + 1):
        fake.grade = lambda q, t, g=bad: g
        with pytest.raises(ValueError):
            td_v5.grade_y(q, "x")


# --- the miner's v5 session ----------------------------------------------------------------


def test_the_miner_builds_a_v5_session_on_its_v4_session():
    from neurons import miner

    class V4:
        challenge = None

    v4 = V4()
    session = miner._v5_session(None, netuid=70, v4_session=v4, first_epoch=104,
                                blocks_per_epoch=360, hotkey=HOTKEY, genesis=None,
                                read_spec=lambda e: (e - 2, _spec(e)))
    assert session.is_v5(104) and not session.is_v5(103)
    assert v4.challenge(105) is None
