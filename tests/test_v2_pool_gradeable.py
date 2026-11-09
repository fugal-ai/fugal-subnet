"""The v2 pool leaves out items the pinned grader cannot judge (2026-11-08).

Measured on the real pool: 48 IFEval items whose bullet-list or JSON checks
fail compliant answers, one AIME gold "080 or 081 (both were accepted)", one
Omni-MATH gold beyond 2**53. Every honest miner failed all of them. graders.py
is immutable, so the v2 loaders filter them; the v1 pool is pinned and keeps
them. These tests pin the premise (what the grader does) and the filters.
"""
import json

import pytest

from fugal_subnet.benchmarks import v2_sets
from fugal_subnet.graders import grade
from fugal_subnet.grading_task import build_grader_task


def _ifeval(qid, *constraints):
    return {"question_id": qid, "benchmark": "ifeval", "prompt": "p", "gold": "",
            "grader_id": "constraint_set",
            "metadata": {"checker": {"id": "constraint_set",
                                     "params": {"constraints": [dict(c) for c in constraints]}}}}


def test_the_premise_compliant_answers_grade_zero_on_these_checks():
    bullets = _ifeval("b", {"kind": "bullet_lines"})
    assert grade(build_grader_task(bullets), "* one\n* two\n* three") == 0      # what IFEval asks for
    assert grade(build_grader_task(bullets), "- one\n- two\n- three") == 1
    js = _ifeval("j", {"kind": "json_object", "required": {}})
    assert grade(build_grader_task(js), json.dumps({"title": "Ad", "body": "Buy"})) == 0
    assert grade(build_grader_task(js), "{}") == 1


def test_ifeval_items_with_an_ungradeable_check_are_left_out(monkeypatch):
    from fugal_subnet.benchmarks import ifeval
    items = [_ifeval("keep_words", {"kind": "word_count_range", "min": 1, "max": 9}),
             _ifeval("drop_bullets", {"kind": "bullet_lines"}),
             _ifeval("drop_json", {"kind": "lowercase_only"}, {"kind": "json_object", "required": {}}),
             _ifeval("keep_none", )]
    monkeypatch.setattr(ifeval, "load", lambda: items)
    assert [q["question_id"] for q in v2_sets.load_ifeval()] == ["keep_words", "keep_none"]


def test_aime_golds_that_no_reply_can_match_are_left_out(monkeypatch):
    from fugal_subnet.benchmarks import aime
    items = [{"question_id": q, "gold": g} for q, g in
             (("a", "60"), ("b", "080 or 081 (both were accepted)"), ("c", "080"), ("d", ""))]
    monkeypatch.setattr(aime, "load", lambda: items)
    assert [q["question_id"] for q in v2_sets.load_aime_hist()] == ["a", "c"]


@pytest.mark.parametrize("answer,kept", [(str(2 ** 53), True), (str(2 ** 53 + 1), False),
                                         ("66071772829247409", False), ("-12", True)])
def test_omnimath_golds_beyond_float_range_are_left_out(monkeypatch, answer, kept):
    rows = [{"problem": "p", "answer": answer, "difficulty": 7.0, "domain": "d"}]
    monkeypatch.setattr(v2_sets, "load_dataset", lambda *a, **k: rows)
    assert bool(v2_sets._omni_integer()) is kept


def test_the_pinned_v2_manifest_counts():
    from fugal_subnet.benchmarks.loader import manifest_path_v2
    m = json.loads(open(manifest_path_v2(), encoding="utf-8").read())
    assert m["n_questions"] == 28_986
    counts = {k: v["n"] for k, v in m["per_benchmark"].items()}
    assert (counts["ifeval"], counts["aime"], counts["omnimath_hard"]) == (211, 932, 845)
