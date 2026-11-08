"""This epoch's contribution to the reference frame, and who is kept out of it.

This lived inline in neurons/validator.py's epoch loop, where every planted
bug survived the whole suite (2026-11-08), among them: a flagged operator's
exploration still feeding the frame, either detector ignored, observations
voted per UID instead of per coldkey, failed calls fed in as wrong answers.
The detectors and the aggregation are tested in their own modules; these
tests check that the validator wires them together the way the docs say.
"""
import ast
import pathlib
from types import SimpleNamespace

import pytest

from fugal_subnet import reference_frame
from fugal_subnet.reference_frame import ReferenceFrame
from neurons import validator
from neurons.validator import _frame_input


def _r(qid="q1", model="m/a", correct=True, failed=False):
    return SimpleNamespace(question_id=qid, routed_model=model, correct=correct,
                           prompt_tokens=10, completion_tokens=5, failed=failed)


def _proofs(answers):
    """{uid: [result, ...]} -> {uid: proof}"""
    return {uid: SimpleNamespace(exploration_results=rs) for uid, rs in answers.items()}


def _run(proofs, coldkeys, version=1, questions=(), explore_map=None, gold=None):
    return _frame_input(ReferenceFrame(), proofs, coldkeys, version, list(questions),
                        explore_map or {}, gold or {})


@pytest.fixture
def quiet_detectors(monkeypatch):
    """Both detectors silent, recording what they were shown."""
    seen = {}

    def implausible(frame, per_miner):
        seen["implausible"] = per_miner
        return {}

    def disagreeing(per_miner, coldkey_of):
        seen["disagreeing"] = (per_miner, coldkey_of)
        return {}

    monkeypatch.setattr(reference_frame, "implausible_exploration", implausible)
    monkeypatch.setattr(reference_frame, "disagreeing_exploration", disagreeing)
    return seen


# ------------------------------------------------------------ one vote each --

def test_one_vote_per_coldkey_not_per_uid(quiet_detectors):
    """Three hotkeys under one coldkey say True, two other operators say
    False: per operator that is 1 against 2, so the sample is wrong."""
    proofs = _proofs({1: [_r(correct=True)], 2: [_r(correct=True)], 3: [_r(correct=True)],
                      4: [_r(correct=False)], 5: [_r(correct=False)]})
    coldkeys = ["c-0", "A", "A", "A", "B", "C"]
    out = _run(proofs, coldkeys)
    assert out.samples == [("m/a", 0.0, 10.0, 5.0)]
    assert {o.coldkey for o in out.observations} == {"A", "B", "C"}


def test_a_uid_beyond_the_coldkey_list_votes_alone(quiet_detectors):
    out = _run(_proofs({7: [_r()]}), coldkeys=[])
    assert out.coldkey_of == {7: "uid-7"} and out.observations[0].coldkey == "uid-7"


def test_a_failed_call_is_not_a_wrong_answer(quiet_detectors):
    proofs = _proofs({1: [_r(correct=False, failed=True)], 2: [_r(correct=True)]})
    out = _run(proofs, ["x", "A", "B"])
    assert out.samples == [("m/a", 1.0, 10.0, 5.0)]
    assert [o.failed for o in out.observations] == [True, False]


# ----------------------------------------------------------- the detectors --

def test_both_detectors_see_the_epoch_and_failed_calls_are_not_disagreement(quiet_detectors):
    proofs = _proofs({1: [_r("q1"), _r("q2", failed=True)], 2: [_r("q1", correct=False)]})
    _run(proofs, ["x", "A", "B"])
    assert quiet_detectors["implausible"] == {1: [("m/a", True), ("m/a", True)],
                                              2: [("m/a", False)]}
    per_miner, coldkey_of = quiet_detectors["disagreeing"]
    assert per_miner == {1: [("q1", "m/a", True)], 2: [("q1", "m/a", False)]}
    assert coldkey_of == {1: "A", 2: "B"}


@pytest.mark.parametrize("detector", ["implausible_exploration", "disagreeing_exploration"])
def test_a_flagged_operator_is_kept_out_of_the_frame(quiet_detectors, monkeypatch, detector):
    monkeypatch.setattr(reference_frame, detector, lambda *a, **k: {2: "flagged"})
    proofs = _proofs({1: [_r(correct=True)], 2: [_r(correct=False)]})
    out = _run(proofs, ["x", "A", "B"])
    assert out.suspicious == {2: "flagged"}
    assert out.samples == [("m/a", 1.0, 10.0, 5.0)]
    # Still in the reveal's observations: excluded from the frame, not hidden.
    assert {o.coldkey for o in out.observations} == {"A", "B"}


def test_flags_from_both_detectors_are_combined(quiet_detectors, monkeypatch):
    monkeypatch.setattr(reference_frame, "implausible_exploration", lambda *a: {1: "too good"})
    monkeypatch.setattr(reference_frame, "disagreeing_exploration", lambda *a: {2: "disagrees"})
    proofs = _proofs({1: [_r()], 2: [_r()], 3: [_r(correct=False)]})
    out = _run(proofs, ["x", "A", "B", "C"])
    assert out.suspicious == {1: "too good", 2: "disagrees"}
    assert out.samples == [("m/a", 0.0, 10.0, 5.0)]


# -------------------------------------------------------------- v2 weights --

def test_v1_samples_are_unweighted(quiet_detectors):
    out = _run(_proofs({1: [_r()]}), ["x", "A"], version=1,
               questions=[{"question_id": "s1", "benchmark": "math"}],
               explore_map={"q1": "m/a"}, gold={"q1": {"question_id": "q1", "benchmark": "mmlu"}})
    assert out.sample_weights is None and out.mix == {} and out.bench_of == {}


def test_v2_samples_are_weighted_by_the_scored_mix(quiet_detectors):
    from fugal_subnet.benchmarks.slicer import mix_weights
    scored = [{"question_id": f"s{i}", "benchmark": "math" if i < 3 else "mmlu"} for i in range(4)]
    gold = {"q1": {"question_id": "q1", "benchmark": "math"},
            "q2": {"question_id": "q2", "benchmark": "mmlu"}}
    proofs = _proofs({1: [_r("q1"), _r("q2")]})
    out = _run(proofs, ["x", "A"], version=2, questions=scored,
               explore_map={"q1": "m/a", "q2": "m/a"}, gold=gold)
    mix = mix_weights(scored, [gold["q1"], gold["q2"]])
    assert out.mix == mix and out.bench_of == {"q1": "math", "q2": "mmlu"}
    assert out.sample_weights == [mix["math"], mix["mmlu"]]
    assert mix["math"] != mix["mmlu"]


# ------------------------------------------ inline wiring, on the syntax tree --

def test_the_epoch_frame_is_built_from_this_epochs_input_and_used_downstream():
    tree = ast.parse(pathlib.Path(validator.__file__).read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = {getattr(n.func, "id", ""): n for n in ast.walk(main) if isinstance(n, ast.Call)}
    acc = calls["accumulate_exploration"]
    assert [ast.unparse(a) for a in acc.args] == ["frame", "samples"]
    assert {k.arg: ast.unparse(k.value) for k in acc.keywords} == {"weights": "sample_weights"}
    assert ast.unparse(calls["best_model"].args[0]) == "epoch_frame"
    assert ast.unparse(calls["build_frontier"].args[0]) == "epoch_frame"
    fi = calls["_frame_input"]
    assert [ast.unparse(a) for a in fi.args] == [
        "frame", "verified_proofs", "coldkeys", "protocol_version", "questions",
        "explore_map", "gold_answers"]
