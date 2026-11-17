"""Protocol v3 pool: categories, exclusion steps, the manifest hash, code questions, harness wiring."""
import json
import os
from collections import Counter

import pytest

from fugal_subnet import graders_code
from fugal_subnet.benchmarks import code_sets as cs
from fugal_subnet.benchmarks import loader
from fugal_subnet.config import BENCHMARK_CATEGORY_V3, BENCHMARK_WEIGHT_V3, CATEGORY_TARGETS_V3

SANDBOX = graders_code.sandbox_selftest()[0]
needs_sandbox = pytest.mark.skipif(not SANDBOX, reason="code sandbox unavailable on this machine")
MANIFEST_PATH = loader.manifest_path_v3()
HAVE_MANIFEST = os.path.exists(MANIFEST_PATH)


# --- categories ----------------------------------------------------------------

def test_every_v3_benchmark_has_a_category_and_the_targets_sum_to_one():
    from fugal_subnet.benchmarks.v2_sets import LOADERS_V2
    from fugal_subnet.benchmarks.v3_sets import LOADERS_V3

    benches = (set(LOADERS_V2) - {"ifeval"}) | set(LOADERS_V3) | set(cs.LOADERS_CODE)
    assert benches == set(BENCHMARK_CATEGORY_V3)
    assert set(BENCHMARK_CATEGORY_V3.values()) == set(CATEGORY_TARGETS_V3)
    assert abs(sum(CATEGORY_TARGETS_V3.values()) - 1.0) < 1e-9
    assert BENCHMARK_WEIGHT_V3 == {}                      # every category splits by pool share


def test_the_v3_targets_are_the_decided_ones():
    assert CATEGORY_TARGETS_V3 == {"knowledge": 0.20, "hard": 0.20, "math": 0.10, "instruction": 0.15,
                                   "everyday": 0.15, "easy_math": 0.05, "code": 0.15}


def test_load_pool_dispatches_three_and_refuses_an_unknown_version():
    with pytest.raises(ValueError):
        loader.load_pool(4)


# --- exclusion steps -------------------------------------------------------------

EXCL = {"routerarena": {"question_ids": ["mmlupro_1"]},
        "routerarena_code": {"leetcode_slugs": ["two-sum"]},
        "all_fail": {"question_ids": ["supergpqa_hard_9"]}}


def _q(qid, bench, **meta):
    return {"question_id": qid, "benchmark": bench, "prompt": qid, "gold": "A", "grader_id": "letter_mcq",
            "metadata": meta}


def test_the_routerarena_step_drops_its_questions_and_its_leetcode_slugs():
    pool = [_q("mmlupro_1", "mmlupro"), _q("mmlupro_2", "mmlupro"),
            _q("code_lc_two-sum", "code_leetcode", source_id="two-sum"),
            _q("code_lc_three-sum", "code_leetcode", source_id="three-sum")]
    kept, dropped = loader.exclude_routerarena(pool, EXCL)
    assert [q["question_id"] for q in kept] == ["mmlupro_2", "code_lc_three-sum"]
    assert dropped == {"mmlupro": 1, "code_leetcode": 1}


def test_the_all_fail_step_drops_only_its_questions():
    pool = [_q("supergpqa_hard_9", "supergpqa_hard"), _q("supergpqa_hard_8", "supergpqa_hard")]
    kept, dropped = loader.exclude_all_fail(pool, EXCL)
    assert [q["question_id"] for q in kept] == ["supergpqa_hard_8"] and dropped == {"supergpqa_hard": 1}


def test_apply_exclusions_reports_counts_before_and_after():
    pool = [_q("mmlupro_1", "mmlupro"), _q("mmlupro_2", "mmlupro"), _q("supergpqa_hard_9", "supergpqa_hard")]
    report: dict = {}
    out = loader.apply_exclusions_v3(pool, EXCL, report)
    assert [q["question_id"] for q in out] == ["mmlupro_2"]
    assert report == {"loaded": {"mmlupro": 2, "supergpqa_hard": 1}, "routerarena": {"mmlupro": 1},
                      "all_fail": {"supergpqa_hard": 1}, "kept": {"mmlupro": 1}}


def test_the_shipped_exclusion_file_has_the_measured_counts():
    excl = loader.read_exclusions_v3()
    assert len(excl["routerarena"]["question_ids"]) == 2208
    assert len(excl["routerarena_code"]["leetcode_slugs"]) == 183
    assert len(excl["all_fail"]["question_ids"]) == 237


# --- the manifest hash covers every field a grader reads ------------------------------

def _constraint_q(limit):
    return {"question_id": "ifeval_key_1", "benchmark": "ifeval", "prompt": "Write.", "gold": "",
            "grader_id": "constraint_set",
            "metadata": {"checker": {"id": "constraint_set", "params": {"constraints": [
                {"kind": "word_count_range", "min": 0, "max": limit}]}}}}


def test_the_v3_manifest_hashes_every_field_a_grader_reads():
    a, b = _constraint_q(100), _constraint_q(5)
    assert loader.content_hash_v3([a]) != loader.content_hash_v3([b])
    code = cs.to_question(_func_cand(), [0, 1, 2, 3, 4])
    other = json.loads(json.dumps(code))
    other["metadata"]["checker"]["params"]["inputs"][0] = [99]
    assert loader.content_hash_v3([code]) != loader.content_hash_v3([other])
    moved = dict(code, benchmark="code_contests")
    assert loader.content_hash_v3([code]) != loader.content_hash_v3([moved])


def test_metadata_a_grader_does_not_read_leaves_the_v3_hash_alone():
    a = _constraint_q(100)
    b = json.loads(json.dumps(a))
    b["metadata"]["row_index"] = 7
    assert loader.content_hash_v3([a]) == loader.content_hash_v3([b])


# --- code candidates ------------------------------------------------------------

def _lc_row(**over):
    row = {"task_id": "add-two", "question_id": 9001, "difficulty": "Easy",
           "problem_description": "Return a + b.", "starter_code": "class Solution:\n    def add(self, a: int, b: int) -> int:\n        ",
           "estimated_date": "2024-01-01", "entry_point": "Solution().add", "test": "",
           "completion": "class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a + b\n",
           "input_output": [{"input": f"a = {i}, b = {i + 1}", "output": str(2 * i + 1)} for i in range(8)]}
    row.update(over)
    return row


def _func_cand():
    return cs.leetcode_candidates([_lc_row()])[0]


def test_leetcode_candidates_parse_keyword_tests_and_skip_what_exact_compare_cannot_judge():
    stats = Counter()
    rows = [_lc_row(),
            _lc_row(task_id="any", problem_description="Return the pairs in any order."),
            _lc_row(task_id="tree", test="tree_node(root)"),
            _lc_row(task_id="void", starter_code="class Solution:\n    def f(self, a: int) -> None:\n"),
            _lc_row(task_id="few", input_output=[{"input": "a = 1, b = 2", "output": "3"}]),
            _lc_row(task_id="floats", input_output=[{"input": f"a = {i}, b = 1", "output": f"{i}.5"} for i in range(8)])]
    out = cs.leetcode_candidates(rows, stats)
    assert [c["source_id"] for c in out] == ["add-two"]
    assert out[0]["tests"][0] == ({"a": 0, "b": 1}, 1) and out[0]["func"] == "Solution().add"
    assert stats["multi_answer"] == 1 and stats["linked_list_or_tree"] == 1 and stats["in_place"] == 1
    assert stats["too_few_tests"] == 2
    assert out[0]["prompt"].endswith(cs.FUNC_SUFFIX)


def test_evalplus_tests_are_read_from_literals_only():
    src = ("import numpy as np\ndef check(candidate):\n    inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]\n"
           "    results = [3, 7, 11, 15, 19]\n    for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
           "        assertion(candidate(*inp), exp, 0)\n")
    inputs, results, atol, name = cs._evalplus_io(src)
    assert inputs[0] == [1, 2] and results[-1] == 19 and atol == 0.0 and name == "candidate"
    row = {"task_id": "HumanEval/0", "prompt": "def add(a, b):\n    ", "canonical_solution": "return a + b\n",
           "entry_point": "add", "test": src}
    c = cs.humanevalplus_candidates([row])[0]
    assert c["func"] == "add" and c["float_tol"] is None and c["references"] == ["def add(a, b):\n    return a + b\n"]
    with pytest.raises(Exception):
        cs._evalplus_io("inputs = make_inputs()\nresults = []\n")


def test_codecontests_candidates_need_a_python3_reference_and_exact_outputs():
    def row(name, lang=3, out="3", infile=""):
        tests = {"input": [f"{i}\n" for i in range(6)], "output": [out] * 6}
        return {"name": name, "description": "Print 3.", "public_tests": tests, "private_tests": {},
                "generated_tests": {}, "solutions": {"language": [lang], "solution": ["print(3)"]},
                "input_file": infile, "output_file": ""}
    stats = Counter()
    out = cs.codecontests_candidates([row("ok"), row("cpp", lang=2), row("float", out="0.5"),
                                      row("file", infile="in.txt")], stats)
    assert [c["source_id"] for c in out] == ["ok"]
    assert stats["no_python3_solution"] == 1 and stats["float_output"] == 1 and stats["file_io"] == 1
    assert out[0]["prompt"].endswith(cs.STDIO_SUFFIX)


def test_the_keep_list_must_agree_with_the_loader():
    c = _func_cand()
    with pytest.raises(ValueError):
        cs.apply_keep([c], {"add-two": [0, 99]})
    assert cs.apply_keep([c], {}) == []
    q = cs.apply_keep([c], {"add-two": [1, 3, 5, 6, 7]})[0]
    assert q["question_id"] == "code_lc_add-two" and q["grader_id"] == "exec_func"
    assert q["metadata"]["checker"]["params"]["inputs"] == [{"a": 1, "b": 2}, {"a": 3, "b": 4},
                                                           {"a": 5, "b": 6}, {"a": 6, "b": 7}, {"a": 7, "b": 8}]
    assert q["gold"] == [3, 7, 11, 13, 15]


@needs_sandbox
def test_a_code_question_grades_its_reference_right_and_a_wrong_answer_wrong():
    from fugal_subnet.grading_task import build_grader_task

    c = _func_cand()
    q = cs.to_question(c, [0, 1, 2, 3, 4])
    task = build_grader_task(q)
    good = "```python\n" + c["references"][0] + "```"
    bad = good.replace("return a + b", "return a - b")
    assert graders_code.grade(task, good, allow_exec=True) == 1
    assert graders_code.grade(task, bad, allow_exec=True) == 0


@needs_sandbox
def test_a_stdin_code_question_grades_through_the_same_task_builder():
    from fugal_subnet.grading_task import build_grader_task

    tests = {"input": [f"{i} {i}\n" for i in range(6)], "output": [str(2 * i) for i in range(6)]}
    row = {"name": "double", "description": "Print a+b.", "public_tests": tests, "private_tests": {},
           "generated_tests": {}, "solutions": {"language": [3], "solution": ["a,b=map(int,input().split())\nprint(a+b)"]},
           "input_file": "", "output_file": ""}
    q = cs.to_question(cs.codecontests_candidates([row])[0], [0, 1, 2, 3, 4])
    assert q["grader_id"] == "exec_stdio" and q["gold"] == ""
    task = build_grader_task(q)
    assert graders_code.grade(task, "```python\na,b=map(int,input().split())\nprint(a+b)\n```", True) == 1
    assert graders_code.grade(task, "```python\nprint(0)\n```", True) == 0


# --- harness wiring --------------------------------------------------------------

def test_the_harness_grades_code_with_the_code_grader_and_refuses_without_its_sandbox(monkeypatch):
    from fugal_subnet.tee import harness

    q = cs.to_question(_func_cand(), [0, 1, 2, 3, 4])
    calls = []
    monkeypatch.setattr(harness.graders_code, "grade", lambda task, reply, allow_exec=False: calls.append(allow_exec) or 1)
    monkeypatch.setattr(harness, "_CODE_SANDBOX", (True, "ok"))
    assert harness.grade_question(q, "x") is True and calls == [True]
    monkeypatch.setattr(harness, "_CODE_SANDBOX", (False, "no seccomp"))
    with pytest.raises(harness.CodeSandboxUnavailable):
        harness.grade_question(q, "x")


def test_a_non_code_question_never_reaches_the_code_grader(monkeypatch):
    from fugal_subnet.tee import harness

    monkeypatch.setattr(harness, "_CODE_SANDBOX", (False, "no seccomp"))
    q = {"question_id": "mmlu_1", "benchmark": "mmlu", "prompt": "?", "gold": "A", "grader_id": "letter_mcq",
         "metadata": {}}
    assert harness.grade_question(q, "Answer: A") is True


def test_the_sandboxed_code_checkers_are_exactly_the_code_graders_and_never_need_allow_exec():
    from fugal_subnet.config import EXECUTION_CHECKERS, SANDBOXED_CODE_CHECKERS

    assert SANDBOXED_CODE_CHECKERS == set(graders_code.CHECKERS)
    assert not SANDBOXED_CODE_CHECKERS & EXECUTION_CHECKERS
    q = cs.to_question(_func_cand(), [0, 1, 2, 3, 4])
    assert loader._drop_unscoreable([q]) == [q]               # kept with HARNESS_ALLOW_EXEC off
    for cid in SANDBOXED_CODE_CHECKERS:                       # and inert unless the sandbox is allowed
        task = {"checker": {"id": cid, "params": {"func": "f", "inputs": [[1]], "tests": [{"input": "", "output": ""}]}},
                "gold": [1]}
        assert graders_code.grade(task, "```python\ndef f(x): return x\nprint('')\n```", allow_exec=False) == 0


def test_the_harness_refuses_a_code_epoch_before_any_call_when_the_sandbox_failed(monkeypatch):
    import numpy as np

    from fugal_subnet.tee import harness

    q = cs.to_question(_func_cand(), [0, 1, 2, 3, 4])
    monkeypatch.setattr(harness, "_CODE_SANDBOX", (False, "no seccomp"))
    monkeypatch.setattr(harness, "load_head_from_npz", lambda b: type("H", (), {"models": ["m"]})())
    monkeypatch.setattr(harness, "benchmark_costs_for", lambda head, v: None)
    monkeypatch.setattr(harness, "_route_question", lambda head, h, costs: 0)
    monkeypatch.setattr(harness.epoch_plan, "scored_slice", lambda v, n, pool, size: pool)
    called = []
    monkeypatch.setattr(harness, "_call_all", lambda *a, **k: called.append(1) or iter(()))
    with pytest.raises(harness.CodeSandboxUnavailable):
        harness.run_benchmark("00" * 32, b"", [q], type("P", (), {"clear": lambda self: None})(),
                              np.zeros((1, 4)), protocol_version=3)
    assert called == []


def test_the_grading_identity_covers_both_grader_files():
    from fugal_subnet import fingerprint, graders

    ident = fingerprint.grader_hash()
    assert graders.grader_hash() in ident and graders_code.grader_hash() in ident


# --- the pinned manifest (once built) --------------------------------------------

@pytest.mark.skipif(not HAVE_MANIFEST, reason="pool_manifest_v3.json not built yet")
def test_the_v3_manifest_pins_the_code_and_exclusion_files_and_the_categories():
    import hashlib

    m = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    assert m["protocol"] == 3 and m["category_targets"] == CATEGORY_TARGETS_V3
    for b, row in m["per_benchmark"].items():
        assert row["category"] == BENCHMARK_CATEGORY_V3[b]
    assert m["n_questions"] == sum(r["n"] for r in m["per_benchmark"].values()) >= 20000
    assert m["code_keep_sha256"] == hashlib.sha256(open(cs.keep_path(), "rb").read()).hexdigest()
    assert m["exclusions_sha256"] == hashlib.sha256(open(loader.exclusions_path_v3(), "rb").read()).hexdigest()


@pytest.mark.skipif(not HAVE_MANIFEST, reason="pool_manifest_v3.json not built yet")
def test_the_best_questions_to_memorise_carry_at_most_a_seventh_of_the_net_value():
    """A head memorises about 2,000-2,500 questions (INVARIANTS, pool size) and every cell's
    grade is published. The 2,500 questions with the highest per-question share of the seat
    test's net value (draw rate x the hard weight) must carry at most 15% of it, measured on
    the shipped slicer over the manifest's sizes."""
    import hashlib

    from fugal_subnet.benchmarks.slicer import select_slice_v2
    from fugal_subnet.config import CHAMPION_HARD_WEIGHT, SLICE_SIZE

    m = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    pool = [{"question_id": f"{b}_{i:05d}", "benchmark": b}
            for b, r in m["per_benchmark"].items() for i in range(r["n"])]
    weight = Counter()
    epochs = 200
    for e in range(epochs):
        for q in select_slice_v2(hashlib.sha256(b"e%d" % e).digest(), pool, SLICE_SIZE, CATEGORY_TARGETS_V3,
                                 BENCHMARK_CATEGORY_V3, bench_weight=BENCHMARK_WEIGHT_V3):
            weight[q["benchmark"]] += CHAMPION_HARD_WEIGHT if BENCHMARK_CATEGORY_V3[q["benchmark"]] == "hard" else 1.0
    total = sum(weight.values())
    per_q = sorted(((weight[b] / r["n"], r["n"]) for b, r in m["per_benchmark"].items()), reverse=True)
    left, mem = 2500, 0.0
    for w, n in per_q:
        take = min(left, n)
        mem += w * take
        left -= take
        if not left:
            break
    assert mem / total <= 0.15, mem / total
