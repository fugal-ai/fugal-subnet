"""graders_arena: RouterArena's scoring, ported. Fixtures are synthetic (no RouterArena items);
the full parity run against RouterArena's own code is scripts/parity_graders_arena.py
(99.98% of 38,161 RouterArena-graded replies, all non-math datasets 100%)."""
import json
import time

import pytest

from fugal_subnet import graders_arena as g
from fugal_subnet.benchmarks import arena_sets

Y = g.Y_SCALE


def item(dataset, gold, **meta):
    return {"gold": gold, "metadata": {"ra_dataset": dataset, **meta}}


# --- (dataset, gold, reply, expected score) — each pins one RouterArena behaviour -------------
CASES = [
    # mcq_accuracy: gold is an index, a letter or a lower-case letter
    ("MMLUPro", "2", "Reasoning...\n\\boxed{C}", 1.0),
    ("MMLUPro", "2", "\\boxed{B}", 0.0),
    ("MathQA", "b", "The answer is \\boxed{B}.", 1.0),
    ("OpenTDB", "1", "Answer: B", 1.0),                       # "Answer: X" pattern
    ("MedMCQA", "0", "I think option A is right", 1.0),       # "option X"
    ("MedMCQA", "0", "", 0.0),                                # empty reply
    ("MMLUPro", "26", "\\boxed{A}", 0.0),                     # out-of-range gold scores 0
    ("PubMedQA", "1", "\\boxed{\\text{B}}", 1.0),             # \text{ is stripped
    ("Ethics", "1", "It is acceptable, so B", 1.0),           # last standalone capital
    # math_metric
    ("GSM8K", "42", "... so \\boxed{42}", 1.0),
    ("GSM8K", "42", "\\boxed{42.0}", 1.0),
    ("GSM8K", "0.5", "\\boxed{50\\%}", 1.0),                  # percentage forms
    ("MATH", "\\frac{1}{2}", "\\boxed{\\frac{1}{2}}", 1.0),
    ("MATH", "\\frac{1}{2}", "\\boxed{0.5}", 1.0),            # symbolic step (math_verify)
    ("MATH", "x=3", "\\boxed{3}", 1.0),
    ("AsDiv", "Harry", "\\boxed{\\text{Harry}}", 0.0),       # RouterArena rejects text answers
    ("GSM8K", "95", "\\boxed{95 \\text{ minutes}}", 0.0),     # ... and units
    ("AIME", "004", "\\boxed{{4}}", 0.0),                     # ... and doubled braces
    ("FinQA", "150", "the answer is 150", 1.0),               # enhanced extraction, no box
    # exact_match (QANTA, GeoGraphyData)
    ("QANTA", "Carl Nielsen", "\\boxed{Carl_Nielsen}", 1.0),
    ("QANTA", "Carl Nielsen", "\\boxed{carl   nielsen}", 1.0),
    ("GeoGraphyData", "Paris", "\\boxed{Lyon}", 0.0),
    # chess
    ("ChessInstruct", "e2e4", "Best is \\boxed{E2-E4}", 1.0),
    ("ChessInstruct", "e2e4", "I would play e2e4 here", 1.0),
    ("ChessInstruct", "1", "\\boxed{B}", 1.0),                # mcq kinds use chess_accuracy
    # superglue
    ("SuperGLUE-Wic", "No", "\\boxed{No}", 1.0),
    ("SuperGLUE-Wic", "No", "\\boxed{B}", 1.0),               # B -> No
    ("SuperGLUE-QA", "1.0", "\\boxed{True}", 1.0),
    ("SuperGLUE-Entailment", "0.0", "\\boxed{1}", 0.0),
    ("SuperGLUE-ClozeTest", "OECD", "\\boxed{OECD}", 1.0),
    ("SuperGLUE-ClozeTest", "OECD", "OECD is the answer", 0.0),   # no box: first character "O"
    ("SuperGLUE-ClozeTest", "1", "b, clearly", 1.0),          # gold index 1 -> "B"
    # meteor
    # RouterArena's METEOR: even an exact 3-word match pays the chunk penalty 0.5*(1/3)^2
    ("NarrativeQA", "Faster Than Light", "\\boxed{faster than light}", 1 - 0.5 / 9),
    ("WMT19-de-en", "Guten Morgen", "", 0.0),
]


@pytest.mark.parametrize("dataset,gold,reply,expected", CASES)
def test_routerarena_behaviour(dataset, gold, reply, expected):
    assert g.score_text(dataset, gold, reply) == pytest.approx(expected)


def test_meteor_is_fractional_and_bounded():
    s = g.score_text("WMT19-de-en", "das ist ein kleines Haus", "\\boxed{das Haus ist klein}")
    assert 0.0 < s < 1.0
    assert g.single_meteor("a a a a", "a") <= 1.0             # RouterArena issue #210 cap


def test_meteor_chinese_is_per_character():
    assert g.score_text("WMT19-zh-en", "我爱你", "\\boxed{我爱你}") == pytest.approx(1 - 0.5 / 9)


def test_labels_are_integers_in_range():
    assert g.grade(item("MMLUPro", "0"), "\\boxed{A}") == Y
    assert g.grade(item("MMLUPro", "0"), "\\boxed{B}") == 0
    lab = g.grade(item("NarrativeQA", "the red house"), "\\boxed{red house}")
    assert isinstance(lab, int) and 0 < lab < Y
    assert g.to_units(0.5) == 5000 and g.to_units(1.0) == Y and g.to_units(2.0) == 0


@pytest.mark.parametrize("reply", [None, 7, b"\\boxed{A}", ["A"], "\x00" * 10, "{" * 5000,
                                   "\\boxed{" * 2000, "A" * 200_000])
def test_hostile_replies_never_raise(reply):
    for dataset in g.METRIC_BY_DATASET:
        if g.METRIC_BY_DATASET[dataset] == g.CODE:
            continue
        s = g.score_text(dataset, "1", reply)
        assert 0.0 <= s <= 1.0
        assert isinstance(g.grade(item(dataset, "1"), reply), int)


def test_hostile_math_is_bounded_in_time():
    start = time.monotonic()
    for reply in ("\\boxed{" + "x^" * 150 + "2}", "\\boxed{" + "(" * 300 + "1" + ")" * 300 + "}",
                  "\\boxed{__import__('os').system('true')}"):
        g.score_text("MATH", "1", reply)
    assert time.monotonic() - start < 30


def test_symbolic_step_never_evals(monkeypatch):
    import builtins
    called = []
    real_eval = builtins.eval

    def spy(*a, **k):
        called.append(a[0] if a else None)
        return real_eval(*a, **k)
    monkeypatch.setattr(builtins, "eval", spy)
    g.score_text("MATH", "2", "\\boxed{__import__('os').getcwd()}")
    assert not any("__import__" in str(c) for c in called)


def test_unknown_dataset_or_missing_metadata_scores_zero():
    assert g.score_text("NoSuchSet", "A", "\\boxed{A}") == 0.0
    assert g.grade({"gold": "A"}, "\\boxed{A}") == 0
    assert g.grade(None, "\\boxed{A}") == 0


def test_code_items_go_to_the_code_sandbox(monkeypatch):
    from fugal_subnet import graders_code
    seen = {}

    def fake(task, reply, allow_exec=False):
        seen.update(task=task, allow=allow_exec)
        return 1
    monkeypatch.setattr(graders_code, "grade", fake)
    checker = {"id": "exec_func", "params": {"func": "f", "inputs": []}}
    lab = g.grade({"gold": "[]", "metadata": {"ra_dataset": "LiveCodeBench", "checker": checker}},
                  "```python\ndef f(): pass\n```", allow_exec=True)
    assert lab == Y and seen["task"] == {"gold": "[]", "checker": checker} and seen["allow"]
    assert g.score_text("LiveCodeBench", "x", "y") == 0.0     # never graded as text


def test_code_without_exec_scores_zero():
    checker = {"id": "exec_func", "params": {"func": "f", "inputs": [[]], "outputs": [1]}}
    assert g.grade({"gold": "1", "metadata": {"ra_dataset": "LiveCodeBench", "checker": checker}},
                   "```python\ndef f(): return 1\n```", allow_exec=False) == 0


def test_every_pool_dataset_has_a_metric_and_a_template():
    with open(arena_sets.files("fugal_subnet.benchmarks") / "arena_prompts.json") as f:
        templates = json.load(f)["prompts"]
    for key in g.METRIC_BY_DATASET:
        if key in ("LiveCodeBench", "GPQA", "Ethics", "ChessInstruct"):
            continue
        assert key in templates, key
    for key in ("Ethics_commonsense", "Ethics_justice", "Ethics_deontology", "Ethics_virtue",
                "ChessInstruct", "ChessInstruct_mcq"):
        assert key in templates


def test_prompt_formatting_matches_routerarena_rules():
    row = arena_sets._row("MMLUPro_law", "s", 1, "What is {x}?", 1, options=["a", "b"])
    p = arena_sets.format_prompt(row)
    # RouterArena escapes braces in the VALUES before str.format, so they come out doubled
    # (1,400 of its 8,400 prompts carry "{{"); the port keeps that.
    assert "Context: None" in p and "A. a\nB. b\n" in p and "What is {{x}}?" in p
    rc = arena_sets._row("SuperGLUE-RC", "s", 2, "Paragraph: P\nQuestion: Q", "1.0",
                         options=["zebra-quokka"])
    assert "zebra-quokka" not in arena_sets.format_prompt(rc)   # RouterArena drops the RC answer
    long = arena_sets._row("NarrativeQA", "s", 3, "Q", "A", context="x" * 20000)
    p = arena_sets.format_prompt(long)
    assert len(p) == 10003 and "..." in p


def _template_literals(template: str) -> list[str]:
    """The template's own sentences: its literal text between placeholders, in pieces of 12 or
    more characters (labels like "Context:" and every instruction sentence)."""
    import string
    out = []
    for lit, _f, _s, _c in string.Formatter().parse(template):
        for piece in lit.replace("\\n", "\n").split("\n"):
            piece = piece.strip()
            if len(piece) >= 12:
                out.append(piece)
    return out


@pytest.mark.parametrize("key", sorted(json.load(
    open(arena_sets.files("fugal_subnet.benchmarks") / "arena_prompts.json"))["prompts"]))
def test_route_text_carries_the_rows_content_and_none_of_its_template(key):
    tmpl = arena_sets.prompts()[key].get("prompt")
    if tmpl is None:
        pytest.skip("no template (LiveCodeBench: code items route on their own prompt)")
    name = {"ChessInstruct_mcq": "ChessInstruct_mcq"}.get(key, key)
    has_options = "{Options}" in tmpl
    row = arena_sets._row(name, "s", 1, "QUESTION-BODY-7f3", "A",
                          context="CONTEXT-BODY-9c1" if "{Context}" in tmpl else "",
                          options=["OPTION-ONE", "OPTION-TWO"] if has_options else ())
    rt = arena_sets.route_text(row)
    assert "QUESTION-BODY-7f3" in rt
    if "{Context}" in tmpl:
        assert "CONTEXT-BODY-9c1" in rt
    if has_options:
        assert "A. OPTION-ONE\nB. OPTION-TWO" in rt
    for literal in _template_literals(tmpl):
        assert literal not in rt, (key, literal)
    # the model still gets RouterArena's full templated prompt
    assert arena_sets.format_prompt(row) != rt and "QUESTION-BODY-7f3" in arena_sets.format_prompt(row)


def test_route_text_drops_the_empty_context_filler_and_cuts_like_the_prompt():
    row = arena_sets._row("QANTA_History", "s", 1, "Who?", "X")
    assert arena_sets.route_text(row) == "Who?"          # no "Context: None"
    assert "Context: None" in arena_sets.format_prompt(row)
    long = arena_sets._row("NarrativeQA", "s", 3, "Q", "A", context="x" * 20000)
    rt = arena_sets.route_text(long)
    assert len(rt) == 10003 and rt.startswith("x" * 5000)


def test_code_items_route_on_their_own_prompt():
    v3 = [{"question_id": "c1", "benchmark": "code_contests", "prompt": "Write a program.",
           "gold": "", "grader_id": "code_stdio", "metadata": {"checker": {"id": "x"}}}]
    (item,) = arena_sets.code_items(v3)
    assert item["route_text"] == item["prompt"] == "Write a program."


def test_the_coverage_partition_is_a_fixed_fifth():
    ids = [f"arena_MMLUPro_{i:06d}" for i in range(20000)]
    share = sum(arena_sets.in_coverage(q) for q in ids) / len(ids)
    assert 0.19 < share < 0.21
    assert [arena_sets.in_coverage(q) for q in ids[:50]] == [arena_sets.in_coverage(q)
                                                              for q in ids[:50]]


def test_the_v5_pool_loader_refuses_a_pool_that_is_not_the_pinned_one(tmp_path, monkeypatch):
    import gzip

    from fugal_subnet.benchmarks import loader
    path = tmp_path / "pool_v5.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump([{"question_id": "q", "benchmark": "arena_QANTA", "prompt": "p",
                    "route_text": "p", "gold": "g", "grader_id": "arena",
                    "metadata": {"ra_dataset": "QANTA"}}], f)
    monkeypatch.setenv("FUGAL_BENCHMARK_POOL_V5", str(path))
    with pytest.raises(ValueError, match="not the pinned one"):
        loader.load_pool(5)


def test_grading_key_and_template_key():
    assert arena_sets.ra_dataset_of("MMLUPro_computer science") == "MMLUPro"
    assert arena_sets.ra_dataset_of("Ethics_virtue") == "Ethics"
    assert arena_sets.ra_dataset_of("GeoGraphyData_100k") == "GeoGraphyData"
    assert arena_sets.ra_dataset_of("ChessInstruct_mcq") == "ChessInstruct"
    assert arena_sets._template_key("ChessInstruct_mcq", True) == "ChessInstruct_mcq"
    assert arena_sets._template_key("Ethics_virtue", True) == "Ethics_virtue"


def test_exclusion_by_text_and_shingle():
    ra = [{"Global Index": "MMLUPro_law_1", "Dataset name": "MMLUPro_law",
           "Question": "one two three four five six seven eight nine ten eleven twelve thirteen"},
          {"Global Index": "GeoGraphyData_100k_0", "Dataset name": "GeoGraphyData_100k",
           "Question": "What is the capital of France?"}]
    ex = arena_sets.Exclusion(ra)
    hit = arena_sets._row("MMLUPro_law", "s", 1, "zero one two three four five six seven eight "
                          "nine ten eleven twelve", 0, options=["a"])
    near = arena_sets._row("MMLUPro_law", "s", 2, "zero one two three four five six seven eight "
                           "nine ten eleven", 0, options=["a"])
    same = arena_sets._row("GeoGraphyData_100k", "s", 3, "what is the capital of  France", "Paris")
    other_set = arena_sets._row("QANTA_History", "s", 4, "one two three four five six seven "
                                "eight nine ten eleven twelve", "x")
    assert ex.hit(hit) == "shingle" and ex.hit(near) is None and ex.hit(same) == "text"
    assert ex.hit(other_set) is None          # shingles compare within the same dataset only


def test_shares_drop_missing_sets_and_sum_to_a_million():
    ra = ([{"Global Index": "MMLUPro_x_1"}] * 3 + [{"Global Index": "ArcMMLU_1"}]
          + [{"Global Index": "QANTA_Lit_1"}])
    w = arena_sets.shares(ra)
    assert "ArcMMLU" not in w and sum(w.values()) == 1_000_000 and w["MMLUPro"] == 750_000


def test_grader_hash_is_stable():
    assert g.grader_hash() == g.grader_hash() and g.grader_hash().startswith("sha256:")
