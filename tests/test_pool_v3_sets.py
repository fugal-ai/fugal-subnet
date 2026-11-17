"""Protocol v3 benchmark loaders (`benchmarks/v3_sets.py`): every transform, on inline fixtures.

What these hold, per loader: the gold a row produces, the rows that are left out and why, the
exact prompt suffix (consensus text), ids that do not depend on row order and never share a
form with a v1 or v2 id, and gradeability under the pinned checkers: the gold answer written
the way a model writes it passes, a wrong answer fails.

The checks at the bottom load the real datasets at their pinned revisions. They are skipped
unless the dataset is already in the local Hugging Face cache or FUGAL_TEST_NETWORK=1 is set,
so a run without network stays green.
"""
import ast
import inspect
import json
import os
import re
from collections import Counter

import pytest

from fugal_subnet.benchmarks import gsm8k, ifeval, math_, v2_sets, v3_sets
from fugal_subnet.graders import CHECKERS, grade
from fugal_subnet.grading_task import build_grader_task

NUMERIC_SUFFIX = "End your response with the final answer as a plain number."
MCQ_SUFFIX = "Answer with the letter only."
# Every id form a v1 or v2 loader produces for the sources v3 reads again.
OLD_ID_FORMS = (r"gsm8k_\d{4}", r"math_(?!train_)[a-z_]+_\d{4}", r"ifeval_\d{4}", r"mmlu_\d{5}")
QUESTION_KEYS = {"prompt", "gold", "grader_id", "benchmark", "question_id", "metadata"}


def _graded(q, reply):
    return grade(build_grader_task(q), reply)


def _well_formed(q, benchmark):
    assert set(q) == QUESTION_KEYS, set(q) ^ QUESTION_KEYS
    assert q["benchmark"] == benchmark and q["grader_id"] in CHECKERS
    assert isinstance(q["prompt"], str) and q["prompt"].strip()
    assert isinstance(q["gold"], str) and isinstance(q["metadata"], dict)
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", q["question_id"]), q["question_id"]
    assert not any(re.fullmatch(form, q["question_id"]) for form in OLD_ID_FORMS), q["question_id"]
    assert json.loads(json.dumps(q)) == q                        # a pool file round-trips it


def _constraints(q):
    return q["metadata"]["checker"]["params"]["constraints"]


# --- the module's rules -----------------------------------------------------------

def test_the_registry_holds_the_new_sets_and_keeps_the_held_back_ones_out():
    assert set(v3_sets.LOADERS_V3) == {"gsm8k_train", "math_l12", "rlvr_ifeval", "arc", "ifeval"}
    assert set(v3_sets.LOADERS_HELD_BACK) == set(v3_sets.HELD_BACK_REVIEW) == {"bbh", "rlvr_ifeval_repeat"}
    assert not set(v3_sets.LOADERS_V3) & set(v3_sets.LOADERS_HELD_BACK)
    assert set(v3_sets.V3_LICENCES) == set(v3_sets.LOADERS_V3)
    # Only `ifeval` shares a name with a v2 set: it replaces it. The others are new names.
    assert set(v3_sets.LOADERS_V3) & set(v2_sets.LOADERS_V2) == {"ifeval"}
    assert not (set(v3_sets.LOADERS_HELD_BACK) & set(v2_sets.LOADERS_V2))


def test_every_source_is_pinned_to_a_commit_and_the_old_pins_are_not_moved():
    from fugal_subnet.benchmarks.loader import DATASET_REVISIONS
    for repo, rev in v3_sets.DATASET_REVISIONS_V3.items():
        assert re.fullmatch(r"[0-9a-f]{40}", rev), (repo, rev)
        if repo in DATASET_REVISIONS:
            assert rev == DATASET_REVISIONS[repo], repo


def test_only_train_splits_are_loaded_and_every_load_is_pinned():
    """The owner's rule: test and validation splits are held back. Read from the source, so a
    new loader that reaches for another split fails here before it reaches a pool."""
    calls = {name: [n for n in ast.walk(ast.parse(inspect.getsource(v3_sets)))
                    if isinstance(n, ast.Call) and getattr(n.func, "id", "") == name]
             for name in ("load_dataset", "hf_hub_download")}
    assert (len(calls["load_dataset"]), len(calls["hf_hub_download"])) == (5, 1)
    for call in calls["load_dataset"]:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw["split"], ast.Constant) and kw["split"].value == "train", ast.dump(call)
        assert "DATASET_REVISIONS_V3" in ast.unparse(kw["revision"]), ast.dump(call)
    for call in calls["hf_hub_download"]:                      # a parquet file fetched by name
        kw = {k.arg: k.value for k in call.keywords}
        assert "/train/" in ast.unparse(call.args[1]), ast.dump(call)
        assert "DATASET_REVISIONS_V3" in ast.unparse(kw["revision"]), ast.dump(call)


def test_the_prompt_suffixes_are_the_agreed_consensus_text():
    assert v3_sets.NUMERIC_SUFFIX == NUMERIC_SUFFIX
    assert v3_sets._numeric_prompt("How many?") == "How many?\n\n" + NUMERIC_SUFFIX
    assert v2_sets._mcq_prompt("Q", ["x", "y"]) == "Q\n\nA. x\nB. y\n\n" + MCQ_SUFFIX


# --- shared helpers ---------------------------------------------------------------

def _q(qid, prompt, gold="1", checker=None):
    return {"question_id": qid, "prompt": prompt, "gold": gold, "grader_id": "numeric_final",
            "benchmark": "b", "metadata": {"checker": checker} if checker else {}}


def test_unique_keeps_a_repeated_question_once_and_does_not_depend_on_row_order():
    items = [_q("b", "two"), _q("a", "one"), _q("a", "one"), _q("c", "one")]
    stats = Counter()
    kept = v3_sets._unique(items, stats)
    assert [q["question_id"] for q in kept] == ["a", "b"]         # "one" kept under its smallest id
    assert stats["duplicate"] == 2
    assert v3_sets._unique(list(reversed(items))) == kept


def test_unique_drops_every_row_of_a_prompt_the_source_grades_two_ways():
    stats = Counter()
    kept = v3_sets._unique([_q("a", "same", gold="1"), _q("b", "same", gold="2"), _q("c", "other")], stats)
    assert [q["question_id"] for q in kept] == ["c"]
    assert stats["conflicting_duplicate"] == 2
    two_checkers = [_q("a", "same", checker={"id": "x", "params": {"constraints": [{"kind": "no_digits"}]}}),
                    _q("b", "same", checker={"id": "x", "params": {"constraints": [{"kind": "lowercase_only"}]}})]
    assert v3_sets._unique(two_checkers) == []


def test_unique_refuses_one_id_on_two_prompts():
    with pytest.raises(ValueError, match="two different prompts"):
        v3_sets._unique([_q("a", "one"), _q("a", "two")])


def test_subset_is_chosen_by_hash_of_the_id_not_by_load_order():
    items = [_q(f"q{i:03d}", f"p{i}") for i in range(50)]
    chosen = v3_sets.subset(items, 10)
    assert len(chosen) == 10 and chosen == v3_sets.subset(list(reversed(items)), 10)
    assert [q["question_id"] for q in chosen] != [f"q{i:03d}" for i in range(10)]
    assert {q["question_id"] for q in v3_sets.subset(items, 5)} <= {q["question_id"] for q in chosen}


# --- gsm8k_train --------------------------------------------------------------------

GSM_ROWS = [
    {"question": "Natalia sold 48 clips in April and half as many in May. How many in all?",
     "answer": "48/2 = <<48/2=24>>24 in May.\n48+24 = <<48+24=72>>72 altogether.\n#### 72"},
    {"question": "A stadium seats 1,250 in each of 2 stands. How many seats?", "answer": "2*1250\n#### 2,500"},
    {"question": "The temperature fell 12 degrees from 5. What is it now?", "answer": "5-12\n#### -7"},
]


def test_gsm8k_train_gold_is_the_integer_after_the_marker_with_commas_removed():
    qs = v3_sets._gsm8k_train_questions(GSM_ROWS)
    assert sorted(q["gold"] for q in qs) == ["-7", "2500", "72"]
    for q in qs:
        _well_formed(q, "gsm8k_train")
        assert q["grader_id"] == "numeric_final"
        assert q["prompt"].endswith("\n\n" + NUMERIC_SUFFIX)
        assert q["prompt"][:-len("\n\n" + NUMERIC_SUFFIX)] in {r["question"] for r in GSM_ROWS}
        assert re.fullmatch(r"gsm8k_train_[0-9a-f]{16}", q["question_id"])


@pytest.mark.parametrize("answer", ["half\n#### 3.5", "#### 1/2", "no marker at all", "#### 12 apples",
                                    f"#### {2 ** 53 + 1}", "#### "])
def test_gsm8k_train_leaves_out_a_gold_that_is_not_one_gradeable_integer(answer):
    stats = Counter()
    assert v3_sets._gsm8k_train_questions([{"question": "q?", "answer": answer}], stats) == []
    assert stats["non_integer_gold"] == 1 and stats["rows"] == 1


def test_gsm8k_train_ids_follow_the_question_text_not_the_row_order():
    forward = v3_sets._gsm8k_train_questions(GSM_ROWS)
    assert v3_sets._gsm8k_train_questions(list(reversed(GSM_ROWS))) == forward
    alone = v3_sets._gsm8k_train_questions(GSM_ROWS[1:2])[0]
    assert alone in forward
    assert [q["question_id"] for q in forward] == sorted(q["question_id"] for q in forward)


def test_gsm8k_train_grades_the_way_a_model_answers():
    q = next(q for q in v3_sets._gsm8k_train_questions(GSM_ROWS) if q["gold"] == "2500")
    for reply in ("2500", "2 * 1,250 = 2,500 seats.\n\n**2500**", "The stadium has $2,500.00 worth... 2500.",
                  "Step 1: 2 stands. Step 2: 1250 each.\n\nFinal answer: 2500"):
        assert _graded(q, reply) == 1, reply
    for reply in ("2501", "", "twenty-five hundred", "2500 seats in 2 stands", "2500.5"):
        assert _graded(q, reply) == 0, reply
    negative = next(q for q in v3_sets._gsm8k_train_questions(GSM_ROWS) if q["gold"] == "-7")
    assert _graded(negative, "It is now -7") == 1 and _graded(negative, "It is now 7") == 0


def test_numeric_final_is_the_checker_because_integer_exact_truncates():
    """Why integer word problems use numeric_final: integer_exact passes 72.9 against 72."""
    q = v3_sets._gsm8k_train_questions(GSM_ROWS[:1])[0]
    assert _graded(q, "72.9") == 0
    assert _graded({**q, "grader_id": "integer_exact"}, "72.9") == 1


# --- math_l12 -----------------------------------------------------------------------

MATH_ROWS = [
    {"problem": "What is $\\frac{1}{4}+\\frac{1}{4}$?", "level": "Level 1", "type": "Prealgebra",
     "solution": "We get $\\boxed{3}$ wrong at first, but the sum is $\\boxed{\\frac{1}{2}}$."},
    {"problem": "Solve $x+2=5$.", "level": "Level 2", "type": "Algebra", "solution": "$x=\\boxed{3}$"},
    {"problem": "A hard one.", "level": "Level 3", "type": "Algebra", "solution": "$\\boxed{7}$"},
    {"problem": "Unrated.", "level": "Level ?", "type": "Geometry", "solution": "$\\boxed{7}$"},
    {"problem": "No box here.", "level": "Level 1", "type": "Algebra", "solution": "The answer is 4."},
]


def test_math_l12_keeps_levels_one_and_two_with_the_last_boxed_expression_as_gold():
    stats = Counter()
    qs = v3_sets._math_l12_questions("algebra", MATH_ROWS, stats)
    assert [(q["prompt"], q["gold"]) for q in qs] == [
        ("What is $\\frac{1}{4}+\\frac{1}{4}$?", "\\frac{1}{2}"), ("Solve $x+2=5$.", "3")]
    assert (stats["rows"], stats["other_level"], stats["no_boxed_gold"]) == (5, 2, 1)
    for q in qs:
        _well_formed(q, "math_l12")
        assert q["grader_id"] == "boxed_math" and q["metadata"]["split"] == "train"
        assert re.fullmatch(r"math_train_algebra_[0-9a-f]{16}", q["question_id"])


def test_math_l12_ids_follow_the_problem_text():
    a = v3_sets._math_l12_questions("algebra", MATH_ROWS)
    b = v3_sets._math_l12_questions("algebra", list(reversed(MATH_ROWS)))
    assert {q["question_id"]: q for q in a} == {q["question_id"]: q for q in b}


def test_math_l12_grades_a_boxed_answer():
    half, three = v3_sets._math_l12_questions("algebra", MATH_ROWS)
    assert _graded(half, "Adding the quarters gives $\\boxed{\\frac{1}{2}}$.") == 1
    assert _graded(half, "$\\boxed{\\frac{1}{3}}$") == 0
    assert _graded(three, "So $x = \\boxed{3}$") == 1
    assert _graded(three, "So $x = \\boxed{4}$") == 0


# --- rlvr_ifeval --------------------------------------------------------------------

_GT_FIELDS = ("N", "quantifier", "end_phrase", "keyword_list", "word", "forbidden_words", "letter", "i",
              "first_word", "postscript_marker", "options", "section_splitter", "original_prompt")


def _gt(func_name, **fields):
    return {"func_name": func_name, **{k: None for k in _GT_FIELDS}, **fields}


def _rlvr_row(content, func_name, constraint="", **fields):
    return {"messages": [{"content": content, "role": "user"}], "dataset": "ifeval",
            "ground_truth": json.dumps(_gt(func_name, **fields)), "constraint_type": "t", "constraint": constraint}


RLVR_EXACT = [
    (_gt("validate_lowercase"), [{"kind": "lowercase_only"}],
     "all lower case here", "One capital"),
    (_gt("validate_no_commas"), [{"kind": "must_exclude", "s": ","}],
     "no commas at all", "one, comma"),
    (_gt("validate_end", end_phrase="That's all."), [{"kind": "ends_with", "s": "That's all."}],
     "Here it is. That's all.", "That's all. And more."),
    (_gt("validate_forbidden_words", forbidden_words=["magneto", "secretly", "dervishism"]),
     [{"kind": "must_exclude", "s": "magneto"}, {"kind": "must_exclude", "s": "secretly"},
      {"kind": "must_exclude", "s": "dervishism"}],
     "A robot from the moon.", "A robot who secretly came from the moon."),      # the SECOND word
    (_gt("verify_keywords", keyword_list=["unreassuring", "underheaven"]),
     [{"kind": "must_include", "s": "unreassuring", "k": 1}, {"kind": "must_include", "s": "underheaven", "k": 1}],
     "An unreassuring day underheaven.", "An unreassuring day."),                # the second is missing
    (_gt("validate_word_constraint", N=5, quantifier="at least"),
     [{"kind": "word_count_range", "min": 5, "max": 999999}], "one two three four five", "one two three four"),
    (_gt("validate_word_constraint", N=5, quantifier="at most"),
     [{"kind": "word_count_range", "min": 0, "max": 5}], "one two three four five", "one two three four five six"),
    (_gt("validate_quotation"), [{"kind": "starts_with", "s": '"'}, {"kind": "ends_with", "s": '"'}],
     '"A quoted reply."', 'Sure: "a quoted reply."'),
]


@pytest.mark.parametrize("gt,want,compliant,violating", RLVR_EXACT, ids=lambda v: v.get("func_name") if isinstance(v, dict) else None)
def test_rlvr_exact_kinds_map_to_constraints_the_pinned_checker_implements(gt, want, compliant, violating):
    constraints, reason = v3_sets._rlvr_constraints(gt)
    assert (constraints, reason) == (want, "")
    q = v3_sets._rlvr_questions([{"messages": [{"content": "Do the task.", "role": "user"}],
                                  "ground_truth": json.dumps(gt), "constraint": "c"}])[0]
    _well_formed(q, "rlvr_ifeval")
    assert _constraints(q) == want and q["grader_id"] == "constraint_set" and q["gold"] == ""
    assert q["prompt"] == "Do the task."                             # the message as sent, no suffix
    # An unknown kind or a misnamed field never passes, so a passing reply proves both.
    assert _graded(q, compliant) == 1
    assert _graded(q, violating) == 0
    assert _graded(q, "") == 0


@pytest.mark.parametrize("gt,reason", [
    (_gt("validate_word_constraint", N=36, quantifier="around"), "word_count_around"),
    (_gt("verify_keyword_frequency", N=3, word="apple"), "no_exact_kind:verify_keyword_frequency"),
    (_gt("validate_json_format"), "no_exact_kind:validate_json_format"),
    (_gt("verify_bullet_points", N=4), "no_exact_kind:verify_bullet_points"),
    (_gt("verify_paragraph_count", N=3), "no_exact_kind:verify_paragraph_count"),
    (_gt("validate_paragraphs", N=3, i=1, first_word="so"), "no_exact_kind:validate_paragraphs"),
    (_gt("validate_uppercase"), "no_exact_kind:validate_uppercase"),
    (_gt("validate_placeholders", N=2), "no_exact_kind:validate_placeholders"),
    (_gt("validate_highlighted_sections", N=2), "no_exact_kind:validate_highlighted_sections"),
    (_gt("validate_sections", N=2, section_splitter="Part"), "no_exact_kind:validate_sections"),
    (_gt("validate_title"), "no_exact_kind:validate_title"),
    (_gt("validate_two_responses"), "no_exact_kind:validate_two_responses"),
    (_gt("validate_choice", options="a), b)"), "no_exact_kind:validate_choice"),
    (_gt("verify_letter_frequency", N=9, letter="d"), "no_exact_kind:verify_letter_frequency"),
    (_gt("validate_frequency_capital_words", N=8, quantifier="at least"), "no_exact_kind:validate_frequency_capital_words"),
    (_gt("verify_sentence_constraint", N=4, quantifier="at most"), "no_exact_kind:verify_sentence_constraint"),
    (_gt("verify_postscript", postscript_marker="PS:"), "no_exact_kind:verify_postscript"),
    (_gt("validate_forbidden_words", forbidden_words=["ate", "dinner", "plate"]), "forbidden_word_under_4_chars"),
    (_gt("validate_forbidden_words", forbidden_words=[]), "malformed_ground_truth"),
    (_gt("validate_forbidden_words", forbidden_words=["fine", ""]), "malformed_ground_truth"),
    (_gt("verify_keywords", keyword_list=[""]), "malformed_ground_truth"),
    (_gt("verify_keywords"), "malformed_ground_truth"),
    (_gt("validate_end", end_phrase="Bye. "), "malformed_ground_truth"),
    (_gt("validate_end"), "malformed_ground_truth"),
    (_gt("validate_word_constraint", N=0, quantifier="at most"), "malformed_ground_truth"),
    (_gt("validate_word_constraint", N=True, quantifier="at least"), "malformed_ground_truth"),
])
def test_rlvr_approximate_and_unmappable_kinds_are_left_out(gt, reason):
    assert v3_sets._rlvr_constraints(gt) == (None, reason)
    stats = Counter()
    row = {"messages": [{"content": "Do the task.", "role": "user"}], "ground_truth": json.dumps(gt), "constraint": ""}
    assert v3_sets._rlvr_questions([row], stats) == []
    assert stats[reason] == 1


def test_rlvr_every_emitted_kind_is_one_the_v2_pool_also_trusts():
    emitted = {c["kind"] for _gt_, want, _c, _v in RLVR_EXACT for c in want}
    assert emitted == {"lowercase_only", "must_exclude", "ends_with", "must_include", "word_count_range", "starts_with"}
    assert not emitted & v2_sets.UNGRADEABLE_IFEVAL_KINDS


def test_rlvr_a_short_forbidden_word_would_fail_honest_replies_which_is_why_it_is_left_out():
    """The premise of the 4-character rule: must_exclude is a substring test."""
    q = v3_sets._constraint_question("p", [{"kind": "must_exclude", "s": "ate"}], "rlvr_ifeval", "x")
    assert _graded(q, "We will see you later.") == 0


def test_rlvr_rows_that_are_not_one_user_message_or_not_json_are_left_out():
    good = _rlvr_row("Say hi.", "validate_lowercase")
    two = {**good, "messages": good["messages"] + [{"content": "hello", "role": "assistant"}]}
    system = {**good, "messages": [{"content": "Say hi.", "role": "system"}]}
    broken = {**good, "ground_truth": "{not json"}
    listed = {**good, "ground_truth": "[1, 2]"}
    stats = Counter()
    assert len(v3_sets._rlvr_questions([good, two, system, broken, listed], stats)) == 1
    assert (stats["not_one_user_message"], stats["malformed_ground_truth"]) == (2, 2)


def test_rlvr_ids_follow_the_prompt_and_a_repeated_row_is_kept_once():
    rows = [_rlvr_row("Write a poem. No commas.", "validate_no_commas"),
            _rlvr_row("Describe IPv6. all lowercase.", "validate_lowercase"),
            _rlvr_row("Write a poem. No commas.", "validate_no_commas")]
    stats = Counter()
    qs = v3_sets._rlvr_questions(rows, stats)
    assert len(qs) == 2 and stats["duplicate"] == 1
    assert v3_sets._rlvr_questions(list(reversed(rows))) == qs
    for q in qs:
        assert re.fullmatch(r"rlvr_ifeval_[0-9a-f]{16}", q["question_id"])
        assert q["metadata"]["source_check"] in {"validate_no_commas", "validate_lowercase"}


REPEAT = ("First, repeat the request without change, then give your answer (do not say anything before "
          "repeating the request; the request you need to repeat does not include this sentence)")


def test_the_premise_the_source_gold_for_repeat_the_request_fails_a_compliant_reply():
    """In the source, original_prompt is the WHOLE message, instruction sentence included,
    although that sentence says it is not part of the request. Graded against it, a reply that
    does exactly what was asked scores 0."""
    request = "Find the median of 4, 8, 10."
    whole = f"{request} {REPEAT}"
    source_gold = v3_sets._constraint_question(whole, [{"kind": "starts_with", "s": whole}], "x", "x")
    assert _graded(source_gold, f"{request}\n\nThe median is 8.") == 0


@pytest.mark.parametrize("content", ["Find the median of 4, 8, 10. " + REPEAT, REPEAT + " Find the median of 4, 8, 10."])
def test_rlvr_repeat_grades_against_the_request_without_the_instruction_sentence(content):
    request = "Find the median of 4, 8, 10."
    assert v3_sets._rlvr_repeat_request(content, REPEAT) == request
    row = _rlvr_row(content, "validate_repeat_prompt", constraint=REPEAT, original_prompt=content)
    q = v3_sets._rlvr_questions([row], repeat=True)[0]
    _well_formed(q, "rlvr_ifeval_repeat")
    assert _constraints(q) == [{"kind": "starts_with", "s": request}] and q["prompt"] == content
    assert _graded(q, f"{request}\n\nThe median is 8.") == 1
    assert _graded(q, "The median is 8.") == 0
    assert _graded(q, "Sure! " + request + " The median is 8.") == 0


def test_rlvr_repeat_needs_the_instruction_at_one_end_of_the_message():
    assert v3_sets._rlvr_repeat_request("Find it. " + REPEAT + " Thanks.", REPEAT) is None
    assert v3_sets._rlvr_repeat_request(REPEAT, REPEAT) is None            # nothing left to repeat
    assert v3_sets._rlvr_repeat_request("Find it.", "") is None
    stats = Counter()
    row = _rlvr_row("Find it. " + REPEAT + " Thanks.", "validate_repeat_prompt", constraint=REPEAT)
    assert v3_sets._rlvr_questions([row], stats, repeat=True) == []
    assert stats["instruction_not_separable"] == 1


def test_rlvr_repeat_rows_are_held_back_from_the_default_loader():
    rows = [_rlvr_row("Find it. " + REPEAT, "validate_repeat_prompt", constraint=REPEAT),
            _rlvr_row("Say hi. all lowercase.", "validate_lowercase")]
    stats = Counter()
    default = v3_sets._rlvr_questions(rows, stats)
    assert [q["metadata"]["source_check"] for q in default] == ["validate_lowercase"]
    assert stats["held_back_repeat_prompt"] == 1
    held = v3_sets._rlvr_questions(rows, repeat=True)
    assert [q["metadata"]["source_check"] for q in held] == ["validate_repeat_prompt"]
    assert {q["benchmark"] for q in default + held} == {"rlvr_ifeval", "rlvr_ifeval_repeat"}


# --- the witness: a reply that passes ------------------------------------------------

@pytest.mark.parametrize("constraints", [
    [{"kind": "lowercase_only"}, {"kind": "must_exclude", "s": ","}],
    [{"kind": "word_count_range", "min": 40, "max": 999999}, {"kind": "ends_with", "s": "Is there anything else?"}],
    [{"kind": "word_count_range", "min": 0, "max": 3}, {"kind": "must_include", "s": "tea", "k": 1}],
    [{"kind": "starts_with", "s": '"'}, {"kind": "ends_with", "s": '"'}],
    [{"kind": "must_exclude", "s": "ok"}, {"kind": "word_count_range", "min": 5, "max": 9}],
    [{"kind": "must_include", "s": "kiwi", "k": 3}, {"kind": "no_digits"}],
])
def test_witness_builds_a_reply_the_pinned_checker_accepts(constraints):
    reply = v3_sets._witness(constraints)
    q = v3_sets._constraint_question("p", constraints, "b", "x")
    assert reply is not None and _graded(q, reply) == 1


@pytest.mark.parametrize("constraints", [
    [{"kind": "lowercase_only"}, {"kind": "ends_with", "s": "Any other questions?"}],   # a capital is required
    [{"kind": "must_exclude", "s": ","}, {"kind": "must_include", "s": "red, white and blue", "k": 1}],
    [{"kind": "word_count_range", "min": 0, "max": 3}, {"kind": "ends_with", "s": "Is there anything else?"}],
    [{"kind": "ends_with", "s": "Bye."}, {"kind": "ends_with", "s": "Ciao."}],
    [{"kind": "an_unknown_kind"}],
    [],
])
def test_witness_finds_none_for_constraints_that_contradict_each_other(constraints):
    assert v3_sets._witness(constraints) is None


# --- arc ----------------------------------------------------------------------------

def _arc_row(rid, question, texts, labels, key):
    return {"id": rid, "question": question, "choices": {"text": texts, "label": labels}, "answerKey": key}


ARC_ROWS = [
    _arc_row("Mercury_7220990", "Which factor most likely causes a fever?",
             ["a relaxing muscle", "bacteria in the blood", "viruses on the skin", "digestion"], ["A", "B", "C", "D"], "B"),
    _arc_row("NAEP_2005_8_S11+3", "Which is a mammal?", ["frog", "trout", "whale", "hawk"], ["1", "2", "3", "4"], "3"),
    _arc_row("AKDE&ED_2008_4_21", "Which is hottest?", ["ice", "steam", "water"], ["A", "B", "C"], "B"),
    _arc_row("MCAS_1", "Pick one.", ["v", "w", "x", "y", "z"], ["A", "B", "C", "D", "E"], "E"),
]


def test_arc_choices_are_relettered_by_position_and_the_key_is_mapped_through_the_labels():
    qs = {q["metadata"]["source_id"]: q for q in v3_sets._arc_questions("easy", ARC_ROWS)}
    whale = qs["NAEP_2005_8_S11+3"]
    assert whale["prompt"] == "Which is a mammal?\n\nA. frog\nB. trout\nC. whale\nD. hawk\n\n" + MCQ_SUFFIX
    assert whale["gold"] == "C"                                    # answerKey "3" is the third choice
    assert [qs[r["id"]]["gold"] for r in ARC_ROWS] == ["B", "C", "B", "E"]
    for q in qs.values():
        _well_formed(q, "arc")
        assert q["grader_id"] == "letter_mcq" and q["prompt"].endswith("\n\n" + MCQ_SUFFIX)
        assert q["metadata"] == {"config": "easy", "source_id": q["metadata"]["source_id"], "split": "train"}


def test_arc_ids_come_from_the_source_id_and_the_config():
    easy = v3_sets._arc_questions("easy", ARC_ROWS)
    assert sorted(q["question_id"] for q in easy) == [
        "arc_easy_AKDE_ED_2008_4_21", "arc_easy_MCAS_1", "arc_easy_Mercury_7220990", "arc_easy_NAEP_2005_8_S11_3"]
    challenge = v3_sets._arc_questions("challenge", list(reversed(ARC_ROWS)))
    assert {q["question_id"] for q in challenge} == {q["question_id"].replace("arc_easy_", "arc_challenge_") for q in easy}


@pytest.mark.parametrize("row,reason", [
    (_arc_row("a", "q", ["x", "y"], ["A", "B"], "C"), "answer_key_not_a_label"),
    (_arc_row("a", "q", ["x", "y"], ["A", "B"], ""), "answer_key_not_a_label"),
    (_arc_row("a", "q", ["x", "y"], ["A", "A"], "A"), "malformed_choices"),
    (_arc_row("a", "q", ["x"], ["A"], "A"), "malformed_choices"),
    (_arc_row("a", "q", ["x", "y", "z"], ["A", "B"], "A"), "malformed_choices"),
    (_arc_row("a", "q", ["x", " "], ["A", "B"], "A"), "malformed_choices"),
    (_arc_row("a", "q", [str(i) for i in range(11)], list("ABCDEFGHIJK"), "A"), "malformed_choices"),
])
def test_arc_rows_the_checker_cannot_grade_are_left_out(row, reason):
    stats = Counter()
    assert v3_sets._arc_questions("easy", [row], stats) == []
    assert stats[reason] == 1


def test_arc_grades_the_way_a_model_answers():
    q = next(q for q in v3_sets._arc_questions("easy", ARC_ROWS) if q["gold"] == "C")
    for reply in ("C", "C.", "(C)", "**C**", "The answer is C.", "Whales nurse their young.\n\nAnswer: C", "C. whale"):
        assert _graded(q, reply) == 1, reply
    for reply in ("A", "The answer is D.", "whale", "", "3"):
        assert _graded(q, reply) == 0, reply


def test_arc_two_ids_on_one_prompt_are_kept_once_at_load_time():
    twin = _arc_row("Mercury_9999999", ARC_ROWS[0]["question"], ARC_ROWS[0]["choices"]["text"], ["A", "B", "C", "D"], "B")
    stats = Counter()
    kept = v3_sets._unique(v3_sets._arc_questions("easy", [*ARC_ROWS, twin]), stats)
    assert len(kept) == 4 and stats["duplicate"] == 1
    assert "arc_easy_Mercury_9999999" not in {q["question_id"] for q in kept}     # the smaller id stays


# --- ifeval: the v3 mapping, and v2 left alone ------------------------------------------

def _ifeval_row(key, prompt, *instructions):
    return {"key": key, "prompt": prompt, "instruction_id_list": [i for i, _ in instructions],
            "kwargs": [k for _, k in instructions]}


LESS_THAN = ("length_constraints:number_words", {"relation": "less than", "num_words": 50})
AT_LEAST = ("length_constraints:number_words", {"relation": "at least", "num_words": 300})
KEYWORDS = ("keywords:existence", {"keywords": ["talented", "tianjin", "talented"]})
FORBIDDEN = ("keywords:forbidden_words", {"forbidden_words": ["reschedule", "free"]})


def test_ifeval_v3_maps_less_than_to_a_limit_one_below():
    assert v3_sets._ifeval_constraints_v3(*LESS_THAN) == [{"kind": "word_count_range", "min": 0, "max": 49}]
    assert v3_sets._ifeval_constraints_v3(*AT_LEAST) == [{"kind": "word_count_range", "min": 300, "max": 999999}]
    assert v3_sets._ifeval_constraints_v3("length_constraints:number_words", {"relation": "at most", "num_words": 9}) == [
        {"kind": "word_count_range", "min": 0, "max": 9}]
    assert v3_sets._ifeval_constraints_v3("length_constraints:number_words", {"relation": "less than", "num_words": None}) == []
    assert v3_sets._ifeval_constraints_v3("length_constraints:number_words", {"relation": "exactly", "num_words": 9}) == []
    q = v3_sets._ifeval_v3_questions([_ifeval_row(1001, "Write a short note.", LESS_THAN)])[0]
    assert _graded(q, " ".join(["word"] * 49)) == 1
    assert _graded(q, " ".join(["word"] * 50)) == 0                # "less than 50" excludes 50


def test_ifeval_v3_emits_one_constraint_for_every_keyword_and_every_forbidden_word():
    assert v3_sets._ifeval_constraints_v3(*KEYWORDS) == [
        {"kind": "must_include", "s": "talented", "k": 1}, {"kind": "must_include", "s": "tianjin", "k": 1}]
    assert v3_sets._ifeval_constraints_v3(*FORBIDDEN) == [
        {"kind": "must_exclude", "s": "reschedule"}, {"kind": "must_exclude", "s": "free"}]
    q = v3_sets._ifeval_v3_questions([_ifeval_row(1002, "Write a blurb.", KEYWORDS, FORBIDDEN)])[0]
    assert _graded(q, "A talented chef from tianjin.") == 1
    assert _graded(q, "A talented chef.") == 0                     # the second keyword is missing
    assert _graded(q, "A talented chef from tianjin, free tonight.") == 0   # the second forbidden word


def test_ifeval_v3_delegates_every_other_instruction_to_the_v2_mapping():
    others = [("change_case:english_lowercase", {}), ("startend:end_checker", {"end_phrase": "Bye."}),
              ("punctuation:no_comma", {}), ("detectable_format:number_bullet_lists", {"num_bullets": 3}),
              ("detectable_format:json_format", {}), ("detectable_format:title", {}),
              ("keywords:frequency", {"keyword": "x", "frequency": 2, "relation": "at least"}),
              ("startend:end_checker", {"end_phrase": ""})]
    for inst, kwargs in others:
        v2 = ifeval._ifeval_to_constraint(inst, kwargs)
        assert v3_sets._ifeval_constraints_v3(inst, kwargs) == ([v2] if v2 else []), inst


def test_ifeval_v3_questions_use_the_source_key_and_leave_out_what_cannot_be_graded():
    rows = [
        _ifeval_row(1000, "Write 300+ words.", AT_LEAST),
        _ifeval_row(1005, "Only a title.", ("detectable_format:title", {})),
        _ifeval_row(1007, "Bullets.", ("detectable_format:number_bullet_lists", {"num_bullets": 3}), AT_LEAST),
        _ifeval_row(1010, "No yes or no.", ("keywords:forbidden_words", {"forbidden_words": ["maybe", "no"]})),
        _ifeval_row(1012, "Lowercase, formal ending.", ("change_case:english_lowercase", {}),
                    ("startend:end_checker", {"end_phrase": "Any other questions?"})),
        _ifeval_row(1019, "No commas please.", ("punctuation:no_comma", {})),
    ]
    stats = Counter()
    qs = v3_sets._ifeval_v3_questions(rows, stats)
    assert [q["question_id"] for q in qs] == ["ifeval_key_1000", "ifeval_key_1019"]
    assert {k: v for k, v in stats.items() if v} == {
        "rows": 6, "no_mapped_instruction": 1, "ungradeable_kind": 1,
        "forbidden_word_under_4_chars": 1, "no_reply_can_pass": 1}
    for q in qs:
        _well_formed(q, "ifeval")
    assert qs[0]["metadata"]["row_index"] == 0 and qs[1]["metadata"]["row_index"] == 5
    assert _constraints(qs[1]) == [{"kind": "must_exclude", "s": ","}]         # the comma rule is not "a short word"


def test_ifeval_v3_records_how_many_of_a_questions_instructions_it_grades():
    """As in v2, an instruction with no exact constraint is skipped and the rest are graded."""
    row = _ifeval_row(1020, "A titled note.", ("detectable_format:title", {}), ("punctuation:no_comma", {}), KEYWORDS)
    q = v3_sets._ifeval_v3_questions([row])[0]
    assert (q["metadata"]["n_instructions"], q["metadata"]["n_mapped_instructions"]) == (3, 2)
    assert len(_constraints(q)) == 3                                # the comma rule and two keywords


def test_ifeval_v3_reads_kwargs_stored_as_a_json_string_or_with_empty_slots():
    as_text = {"key": 7, "prompt": "p", "instruction_id_list": [LESS_THAN[0]], "kwargs": json.dumps([LESS_THAN[1]])}
    assert _constraints(v3_sets._ifeval_v3_questions([as_text])[0]) == [{"kind": "word_count_range", "min": 0, "max": 49}]
    with_none = {"key": 8, "prompt": "p", "instruction_id_list": ["punctuation:no_comma", LESS_THAN[0]], "kwargs": [None, LESS_THAN[1]]}
    assert len(_constraints(v3_sets._ifeval_v3_questions([with_none])[0])) == 2


def test_the_v2_ifeval_mapping_is_not_changed_by_any_of_this():
    """v2 is live and its pool is pinned: `ifeval._ifeval_to_constraint` still drops "less
    than", still keeps only the first keyword and the first forbidden word, and v2's loader
    still emits `ifeval_NNNN` ids from it."""
    assert ifeval._ifeval_to_constraint(*LESS_THAN) is None
    assert ifeval._ifeval_to_constraint(*AT_LEAST) == {"kind": "word_count_range", "min": 300, "max": 999999}
    assert ifeval._ifeval_to_constraint(*KEYWORDS) == {"kind": "must_include", "s": "talented", "k": 1}
    assert ifeval._ifeval_to_constraint(*FORBIDDEN) == {"kind": "must_exclude", "s": "reschedule"}
    assert ifeval._build_constraints(_ifeval_row(1, "p", KEYWORDS, FORBIDDEN, LESS_THAN, ("punctuation:no_comma", {}))) == [
        {"kind": "must_include", "s": "talented", "k": 1}, {"kind": "must_exclude", "s": "reschedule"},
        {"kind": "must_exclude", "s": ","}]
    assert v2_sets.LOADERS_V2["ifeval"] is v2_sets.load_ifeval


def test_the_v2_ifeval_loader_output_on_a_fixture_is_unchanged(monkeypatch):
    rows = [_ifeval_row(1000, "Blurb.", KEYWORDS, FORBIDDEN), _ifeval_row(1001, "Short.", LESS_THAN),
            _ifeval_row(1002, "Bullets.", ("detectable_format:number_bullet_lists", {"num_bullets": 2})),
            _ifeval_row(1003, "Long.", AT_LEAST)]
    monkeypatch.setattr(ifeval, "load_dataset", lambda *a, **k: rows)
    got = v2_sets.load_ifeval()
    assert [(q["question_id"], q["prompt"], _constraints(q)) for q in got] == [
        ("ifeval_0000", "Blurb.", [{"kind": "must_include", "s": "talented", "k": 1},
                                   {"kind": "must_exclude", "s": "reschedule"}]),
        ("ifeval_0003", "Long.", [{"kind": "word_count_range", "min": 300, "max": 999999}])]
    assert all(q["benchmark"] == "ifeval" and q["gold"] == "" and q["grader_id"] == "constraint_set" for q in got)


# --- bbh (held back) ------------------------------------------------------------------

def _bbh(text, target):
    return {"input": text, "target": target}


def test_bbh_is_held_back_because_its_only_split_is_the_test_set():
    assert "bbh" in v3_sets.HELD_BACK_REVIEW and "bbh" not in v3_sets.LOADERS_V3
    assert "test set" in v3_sets.HELD_BACK_REVIEW["bbh"]
    assert set(v3_sets.BBH_DROPPED_TASKS) == {"dyck_languages", "word_sorting", "geometric_shapes"}
    assert not set(v3_sets.BBH_TASKS) & set(v3_sets.BBH_DROPPED_TASKS)
    assert len(v3_sets.BBH_TASKS) == 24 == len(set(v3_sets.BBH_TASKS))
    for task in v3_sets.BBH_DROPPED_TASKS:
        with pytest.raises(ValueError):
            v3_sets._bbh_questions(task, [])


def test_bbh_letter_tasks_lose_the_parentheses_and_gain_the_letter_suffix():
    row = _bbh("Today is Christmas Eve of 1937. What is the date tomorrow?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937\n(C) 01/04/1938", "(B)")
    q = v3_sets._bbh_questions("date_understanding", [row])[0]
    _well_formed(q, "bbh")
    assert q["prompt"] == ("Today is Christmas Eve of 1937. What is the date tomorrow?\n\n"
                           "A. 12/11/1937\nB. 12/25/1937\nC. 01/04/1938\n\n" + MCQ_SUFFIX)
    assert (q["gold"], q["grader_id"], q["metadata"]) == ("B", "letter_mcq", {"task": "date_understanding"})
    assert re.fullmatch(r"bbh_date_understanding_[0-9a-f]{16}", q["question_id"])
    assert _graded(q, "B") == 1 and _graded(q, "(B)") == 1 and _graded(q, "The answer is A.") == 0


@pytest.mark.parametrize("text,target,reason", [
    ("Which?\nOptions:\n(A) x\n(B) y", "dearth, wind, & fire", "non_letter_target"),
    ("Which?\nOptions:\n(A) x\n(B) y", "(C)", "target_not_an_option"),
    ("Which?\nOptions:\n(A) only one", "(A)", "unparseable_options"),
    ("Which?\nOptions:\n(A) x\n(C) y", "(A)", "unparseable_options"),
    ("Which?\nOptions:\n(A) x\nspills onto a second line\n(B) y", "(A)", "unparseable_options"),
    ("Which? (A) x (B) y", "(A)", "no_options_block"),
    ("Count.\nOptions:\n" + "\n".join(f"({c}) {i}" for i, c in enumerate("ABCDEFGHIJKLM")), "(K)", "gold_past_J"),
])
def test_bbh_letter_rows_the_checker_cannot_grade_are_left_out(text, target, reason):
    stats = Counter()
    assert v3_sets._bbh_questions("ruin_names", [_bbh(text, target)], stats) == []
    assert stats[reason] == 1


def test_bbh_more_than_ten_options_with_a_gold_in_the_first_ten_shows_the_first_ten():
    text = "How many remain?\nOptions:\n" + "\n".join(f"({c}) {i}" for i, c in enumerate("ABCDEFGHIJKLMNOPQ"))
    stats = Counter()
    q = v3_sets._bbh_questions("reasoning_about_colored_objects", [_bbh(text, "(D)")], stats)[0]
    assert q["gold"] == "D" and q["metadata"]["options_truncated"] is True
    assert "\nJ. 9\n\n" + MCQ_SUFFIX in q["prompt"] and "K." not in q["prompt"]
    assert stats["kept_with_options_cut_to_ten"] == 1


@pytest.mark.parametrize("task,text,target,stem,options,gold", [
    ("navigate", "Do you return to the start? Take 1 step.\nOptions:\n- Yes\n- No", "No",
     "Do you return to the start? Take 1 step.", ("Yes", "No"), "B"),
    ("causal_judgement", "Did the wire cause it?\nOptions:\n- Yes\n- No", "Yes", "Did the wire cause it?", ("Yes", "No"), "A"),
    ("formal_fallacies", "Is the argument valid or invalid?\nOptions:\n- valid \n- invalid", "invalid",
     "Is the argument valid or invalid?", ("valid", "invalid"), "B"),
    ("web_of_lies", "Question: Ann lies. Does Ann tell the truth?", "No", "Question: Ann lies. Does Ann tell the truth?", ("Yes", "No"), "B"),
    ("sports_understanding", 'Is the following sentence plausible? "He beat the buzzer."', "yes",
     'Is the following sentence plausible? "He beat the buzzer."', ("Yes", "No"), "A"),
    ("boolean_expressions", "not ( True ) and ( True ) is", "False", "not ( True ) and ( True ) is", ("True", "False"), "B"),
])
def test_bbh_two_way_tasks_are_recast_as_options_a_and_b(task, text, target, stem, options, gold):
    q = v3_sets._bbh_questions(task, [_bbh(text, target)])[0]
    _well_formed(q, "bbh")
    assert q["prompt"] == f"{stem}\n\nA. {options[0]}\nB. {options[1]}\n\n" + MCQ_SUFFIX
    assert (q["gold"], q["grader_id"]) == (gold, "letter_mcq")
    wrong = "A" if gold == "B" else "B"
    assert _graded(q, gold) == 1 and _graded(q, f"{gold}. {options['AB'.index(gold)]}") == 1
    assert _graded(q, wrong) == 0
    assert _graded(q, target) == 0          # the bare word is not a letter: the prompt asks for the letter


@pytest.mark.parametrize("task,text,target,reason", [
    ("navigate", "Do you return?\nOptions:\n- Yes\n- No", "Maybe", "unexpected_target"),
    ("navigate", "Do you return?\nOptions:\n- No\n- Yes", "No", "unexpected_options"),
    ("navigate", "Do you return?\nOptions:\n- Yes\n- No\n- Unknown", "No", "unexpected_options"),
    ("web_of_lies", "   ", "No", "empty_question"),
])
def test_bbh_two_way_rows_that_do_not_fit_the_recast_are_left_out(task, text, target, reason):
    stats = Counter()
    assert v3_sets._bbh_questions(task, [_bbh(text, target)], stats) == []
    assert stats[reason] == 1


def test_bbh_integer_tasks_use_numeric_final_with_the_number_suffix():
    rows = [_bbh("((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) =", "24"), _bbh("((1 - 9)) =", "-8")]
    qs = sorted(v3_sets._bbh_questions("multistep_arithmetic_two", rows), key=lambda q: int(q["gold"]))
    assert [q["gold"] for q in qs] == ["-8", "24"]
    q = qs[1]
    _well_formed(q, "bbh")
    assert q["prompt"] == "((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) =\n\n" + NUMERIC_SUFFIX
    assert q["grader_id"] == "numeric_final"
    assert _graded(q, "46 - 22 = 24") == 1 and _graded(q, "46 - 22 = 23") == 0
    assert _graded(qs[0], "The result is -8.") == 1 and _graded(qs[0], "The result is 8.") == 0
    stats = Counter()
    assert v3_sets._bbh_questions("object_counting", [_bbh("How many?", "several")], stats) == []
    assert stats["non_integer_gold"] == 1


def test_bbh_a_question_the_source_answers_two_ways_is_dropped_whole():
    rows = [_bbh("Did it cause it?\nOptions:\n- Yes\n- No", "Yes"), _bbh("Did it cause it?\nOptions:\n- Yes\n- No", "No"),
            _bbh("Was it plausible?\nOptions:\n- Yes\n- No", "No"), _bbh("Was it plausible?\nOptions:\n- Yes\n- No", "No")]
    stats = Counter()
    kept = v3_sets._unique(v3_sets._bbh_questions("causal_judgement", rows), stats)
    assert [q["gold"] for q in kept] == ["B"]
    assert (stats["conflicting_duplicate"], stats["duplicate"]) == (2, 1)


# --- the real datasets, at their pinned revisions ---------------------------------------
# Skipped unless the dataset is in the local Hugging Face cache or FUGAL_TEST_NETWORK=1.

def _cached(repo):
    from huggingface_hub.constants import HF_HUB_CACHE
    snapshot = os.path.join(HF_HUB_CACHE, "datasets--" + repo.replace("/", "--"), "snapshots",
                            v3_sets.DATASET_REVISIONS_V3[repo])
    return os.path.isdir(snapshot)


def _load(name, repo):
    """(questions, drop counts) from the real loader, or skip when the data is not reachable."""
    wanted = os.getenv("FUGAL_TEST_NETWORK") == "1"
    if not wanted and not _cached(repo):
        pytest.skip(f"{repo} is not in the local Hugging Face cache and FUGAL_TEST_NETWORK is not 1")
    stats = Counter()
    loader = {**v3_sets.LOADERS_V3, **v3_sets.LOADERS_HELD_BACK}[name]
    try:
        questions = loader(stats)
    except OSError as e:                      # a partial cache and no network
        if wanted:
            raise
        pytest.skip(f"{repo} could not be read from the local cache: {str(e)[:120]}")
    return questions, {k: v for k, v in stats.items() if v}


def _model_answer(q):
    """(the gold answer the way a model writes it, a wrong answer)."""
    if q["grader_id"] == "numeric_final":
        return f"Adding it up, the total comes to {q['gold']}.", f"The total comes to {int(q['gold']) + 1}."
    if q["grader_id"] == "boxed_math":
        return f"So the answer is $\\boxed{{{q['gold']}}}$.", "So the answer is $\\boxed{987654.321}$."
    if q["grader_id"] == "letter_mcq":
        return f"The answer is {q['gold']}.", "B" if q["gold"] == "A" else "A"
    return v3_sets._witness(_constraints(q)), ""


def _check_real_set(questions, benchmark):
    ids = [q["question_id"] for q in questions]
    assert ids == sorted(set(ids))
    assert len({q["prompt"] for q in questions}) == len(questions)
    not_gradeable = []
    for q in questions:
        _well_formed(q, benchmark)
        right, wrong = _model_answer(q)
        if right is None or _graded(q, right) != 1 or _graded(q, wrong) != 0:
            not_gradeable.append(q["question_id"])
        if q["grader_id"] == "letter_mcq" and _graded(q, q["gold"]) != 1:
            not_gradeable.append(q["question_id"])
    assert not not_gradeable, not_gradeable[:10]


def _shares_nothing_with(questions, older, strip_suffix=""):
    """No id and no question text in common with a set an earlier pool already holds."""
    assert not {q["question_id"] for q in questions} & {q["question_id"] for q in older}
    texts = {q["prompt"].removesuffix(strip_suffix) for q in questions}
    assert len(texts) == len(questions)
    assert not texts & {q["prompt"] for q in older}


def test_real_gsm8k_train():
    questions, stats = _load("gsm8k_train", "openai/gsm8k")
    assert (len(questions), stats) == (7473, {"rows": 7473})
    assert all(q["prompt"].endswith("\n\n" + NUMERIC_SUFFIX) for q in questions)
    _check_real_set(questions, "gsm8k_train")
    _shares_nothing_with(questions, gsm8k.load(), strip_suffix="\n\n" + NUMERIC_SUFFIX)    # v1's set: the test split


def test_real_math_l12():
    questions, stats = _load("math_l12", "EleutherAI/hendrycks_math")
    assert (len(questions), stats) == (1912, {"rows": 7500, "other_level": 5588})
    assert Counter(q["metadata"]["level"] for q in questions) == {"Level 1": 564, "Level 2": 1348}
    _check_real_set(questions, "math_l12")
    _shares_nothing_with(questions, math_.load())          # v1 `math` and v2 `math_l35`: the test split


def test_real_rlvr_ifeval():
    questions, stats = _load("rlvr_ifeval", "allenai/RLVR-IFeval")
    assert len(questions) == 4143 and stats["rows"] == 14973
    assert Counter(q["metadata"]["source_check"] for q in questions) == {
        "validate_quotation": 646, "validate_no_commas": 628, "validate_lowercase": 622, "validate_end": 618,
        "validate_forbidden_words": 610, "verify_keywords": 610, "validate_word_constraint": 409}
    assert (stats["held_back_repeat_prompt"], stats["word_count_around"],
            stats["forbidden_word_under_4_chars"], stats["duplicate"]) == (604, 209, 7, 1)
    assert sum(v for k, v in stats.items() if k.startswith("no_exact_kind:")) == 10009
    assert sum(v for k, v in stats.items() if k != "rows") + len(questions) == stats["rows"]
    assert {c["kind"] for q in questions for c in _constraints(q)} == {
        "lowercase_only", "must_exclude", "ends_with", "must_include", "word_count_range", "starts_with"}
    _check_real_set(questions, "rlvr_ifeval")


def test_real_rlvr_ifeval_repeat():
    questions, stats = _load("rlvr_ifeval_repeat", "allenai/RLVR-IFeval")
    assert (len(questions), stats) == (604, {"rows": 14973, "not_repeat_prompt": 14369})
    for q in questions:
        request = _constraints(q)[0]["s"]
        assert request in q["prompt"] and REPEAT not in request and request == request.strip()
    _check_real_set(questions, "rlvr_ifeval_repeat")


def test_real_arc():
    questions, stats = _load("arc", "allenai/ai2_arc")
    assert (len(questions), stats) == (3369, {"rows": 3370, "duplicate": 1})
    assert Counter(q["metadata"]["config"] for q in questions) == {"easy": 2251, "challenge": 1118}
    assert all(q["prompt"].endswith("\n\n" + MCQ_SUFFIX) for q in questions)
    _check_real_set(questions, "arc")


def test_real_ifeval_v3():
    questions, stats = _load("ifeval", "google/IFEval")
    assert (len(questions), stats) == (215, {"rows": 541, "no_mapped_instruction": 268, "ungradeable_kind": 48,
                                             "forbidden_word_under_4_chars": 10})
    _check_real_set(questions, "ifeval")
    v2 = v2_sets.LOADERS_V2["ifeval"]()
    assert not {q["question_id"] for q in questions} & {q["question_id"] for q in v2}
    # Same prompts as v2 where both keep the row, never a looser grading: v3 only adds constraints.
    old = {q["prompt"]: _constraints(q) for q in v2}
    both = [q for q in questions if q["prompt"] in old]
    assert (len(both), len(questions) - len(both), len(old) - len(both)) == (201, 14, 10)
    for q in both:
        assert all(c in _constraints(q) for c in old[q["prompt"]]), q["question_id"]
    assert sum(1 for q in both if _constraints(q) != old[q["prompt"]]) == 53


def test_real_v2_ifeval_questions_are_byte_for_byte_what_they_were():
    """The whole v2 `ifeval` set as `load_pool(2)` builds it, checker parameters included,
    hashed on 2026-11-13 before the v3 loaders were added."""
    import hashlib

    from fugal_subnet.benchmarks.loader import manifest_path_v2

    if os.getenv("FUGAL_TEST_NETWORK") != "1" and not _cached("google/IFEval"):
        pytest.skip("google/IFEval is not in the local Hugging Face cache and FUGAL_TEST_NETWORK is not 1")
    got = v2_sets.LOADERS_V2["ifeval"]()
    blob = json.dumps(got, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    assert hashlib.sha256(blob).hexdigest() == "8a964c76c88ddab51c323138cc48ad5e6d6e3e2b2bd532f976eed0f3b68bf70c"
    with open(manifest_path_v2(), encoding="utf-8") as f:
        assert len(got) == json.load(f)["per_benchmark"]["ifeval"]["n"] == 211


def test_real_bbh():
    questions, stats = _load("bbh", "maveriq/bigbenchhard")
    assert (len(questions), stats) == (5692, {
        "rows": 5761, "gold_past_J": 59, "kept_with_options_cut_to_ten": 116, "conflicting_duplicate": 4,
        "non_letter_target": 3, "duplicate": 2, "unparseable_options": 1})
    assert Counter(q["grader_id"] for q in questions) == {"letter_mcq": 5192, "numeric_final": 500}
    assert {q["metadata"]["task"] for q in questions} == set(v3_sets.BBH_TASKS)
    _check_real_set(questions, "bbh")
