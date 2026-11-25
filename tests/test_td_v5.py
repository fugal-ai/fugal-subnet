"""Protocol v5 in the TD (~/projects/fugal-v2-handoff/v5/DESIGN.md sections 5 and 7).

The TD labels the exam cells the root's spec (in its reveal of E - SPEC_LAG)
assigns to its hotkey, plus coverage cells; it makes no challenge; every call
is held to the v5 deadline and none starts after the v5 budget, so the proof
is ready before collection; each result carries `exam` and its label `y`, and
the proof the spec's hash. No network, no spend.

The exam spec is the real fugal_subnet.exam_v5.ExamSpec.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from fugal_subnet import exam_v5
from fugal_subnet.exam_v5 import ExamSpec
from fugal_subnet.tee import harness, td_v4, td_v5
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from tests import menu_fixtures as mf
from tests.test_harness_v4 import FIXTURE_IDS, HOTKEY, NONCE, Provider, _inputs, _Proxy
from tests.test_tee_e2e import _head, _pool

OTHER = "5OtherHotkeyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    mf.allow_small_menus(monkeypatch)


# --- specs: the real exam_v5.ExamSpec ----------------------------------------------------


def _spec(epoch=104, *, mine=(("q1", FIXTURE_IDS[0]), ("q2", FIXTURE_IDS[1]), ("q2", FIXTURE_IDS[2])),
          theirs=(("q3", FIXTURE_IDS[0]),)) -> ExamSpec:
    assignment = {c: (HOTKEY, OTHER) for c in mine}
    assignment.update({c: (OTHER,) for c in theirs})
    cells = tuple(sorted(assignment))
    return ExamSpec(epoch=epoch, questions=tuple(sorted({c[0] for c in cells})), picks={},
                    baseline=FIXTURE_IDS[0], baseline_questions=(), cells=cells,
                    ring=tuple(sorted([(HOTKEY, "ck-mine"), (OTHER, "ck-other")])),
                    assignment=assignment)


def _reveal(epoch_id: str, spec: ExamSpec | None, **field_over) -> tuple[bytes, str]:
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
                           cells=tuple(sorted(exam_v5.assigned_cells(spec, HOTKEY))))
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


def _arena_item(qid="arena_MMLUPro_x", gold="B", dataset="MMLUPro", **meta):
    return {"question_id": qid, "benchmark": f"arena_{dataset}", "prompt": "?",
            "route_text": "?", "gold": gold, "grader_id": "arena",
            "metadata": {"ra_dataset": dataset, **meta}}


def test_grade_y_is_the_arena_grade_on_its_scale():
    q = _arena_item()
    assert td_v5.grade_y(q, "so \\boxed{B}") == td_v5.Y_SCALE
    assert td_v5.grade_y(q, "so \\boxed{C}") == 0


def test_grade_y_grades_code_in_the_sandbox_only(monkeypatch):
    from fugal_subnet import graders_arena
    from fugal_subnet.tee import harness
    calls = []
    monkeypatch.setattr(harness, "require_code_sandbox", lambda: calls.append("sandbox"))
    seen = {}

    def fake_grade(task, reply, allow_exec=False):
        seen["allow_exec"] = allow_exec
        return td_v5.Y_SCALE
    monkeypatch.setattr(graders_arena, "grade", fake_grade)
    code = _arena_item(dataset="LiveCodeBench", gold="x", checker={"id": "stdin_stdout"})
    assert td_v5.grade_y(code, "print(1)") == td_v5.Y_SCALE
    assert calls == ["sandbox"] and seen["allow_exec"] is True
    seen.clear()
    td_v5.grade_y(_arena_item(), "x")
    assert calls == ["sandbox"] and seen["allow_exec"] is False


@pytest.mark.parametrize("bad", [0.5, True, td_v5.Y_SCALE + 1, -1])
def test_grade_y_refuses_a_grade_off_its_integer_scale(monkeypatch, bad):
    from fugal_subnet import graders_arena
    monkeypatch.setattr(graders_arena, "grade", lambda task, reply, allow_exec=False: bad)
    with pytest.raises(ValueError):
        td_v5.grade_y(_arena_item(), "x")


def _arena_pool(n=400):
    datasets = ("MMLUPro", "OpenTDB", "QANTA", "PubMedQA")
    return [_arena_item(qid=f"arena_{datasets[i % 4]}_{i:04d}", dataset=datasets[i % 4])
            for i in range(n)]


def test_coverage_draws_only_the_pools_coverage_part_and_never_an_excluded_question():
    from fugal_subnet.benchmarks import arena_sets
    pool = _arena_pool(2000)      # every dataset's coverage part covers its share of the draw
    exam_part, cover_part = arena_sets.partition(pool)
    cover_ids = {q for ids in cover_part.values() for q in ids}
    exam_ids = {q for ids in exam_part.values() for q in ids}
    assert cover_ids and exam_ids and not cover_ids & exam_ids
    state = mf.state()
    excluded = set(sorted(cover_ids)[:5])
    got = td_v5.coverage_v5(bytes.fromhex(NONCE), pool, excluded, state, HOTKEY)
    assert got and set(got) <= cover_ids and not set(got) & excluded
    assert len(got) == min(td_v5.COVERAGE_CALLS, len(cover_ids) - len(excluded))
    assert set(got.values()) <= set(state.ids())
    # deterministic per (nonce, hotkey), different for another hotkey
    assert td_v5.coverage_v5(bytes.fromhex(NONCE), pool, excluded, state, HOTKEY) == got
    assert td_v5.coverage_v5(bytes.fromhex(NONCE), pool, excluded, state, HOTKEY + "x") != got


# --- the miner's v5 session ----------------------------------------------------------------


def test_the_session_reads_the_protocol_from_the_one_source():
    # is_v5 follows protocol_of(boundary) when given (the miner binds it to
    # slicer.protocol_version_for_boundary), not first_epoch
    session = td_v5.V5Session(v4_session=None, exam=_policy(lambda e: None), first_epoch=0,
                              protocol_of=lambda b: 5 if b >= 104 * 360 else 4,
                              blocks_per_epoch=360)
    assert session.is_v5(104) and not session.is_v5(103)


def test_the_miners_protocol_source_is_the_slicer_on_the_chain_blocks(monkeypatch):
    from fugal_subnet import config as cfg
    from neurons import miner

    monkeypatch.setattr(cfg, "V2_ACTIVATION_BLOCK", 1000)
    monkeypatch.setattr(cfg, "v3_activation_block", lambda g: 2000)
    monkeypatch.setattr(cfg, "v4_activation_block", lambda g: 3000)
    monkeypatch.setattr(cfg, "v5_activation_block", lambda g: 104 * 360)
    of = miner._protocol_of_boundary(None)
    assert of(104 * 360) == 5 and of(103 * 360) == 4 and of(2500) == 3


def test_the_miner_builds_a_v5_session_on_its_v4_session(monkeypatch):
    from neurons import miner

    class V4:
        challenge = None

    monkeypatch.setattr(miner, "_protocol_of_boundary",
                        lambda g: (lambda b: 5 if b >= 104 * 360 else 4))
    v4 = V4()
    session = miner._v5_session(None, netuid=70, v4_session=v4, first_epoch=104,
                                blocks_per_epoch=360, hotkey=HOTKEY, genesis=None,
                                read_spec=lambda e: (e - 2, _spec(e)), pool=_arena_pool(8))
    assert session.is_v5(104) and not session.is_v5(103)
    assert v4.challenge(105) is None
