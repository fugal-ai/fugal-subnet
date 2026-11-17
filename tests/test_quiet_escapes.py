"""Code-question parsing raises no "invalid escape sequence" warnings, and the
values are exactly what a plain parse gives (fugal_subnet/benchmarks/code_sets.py)."""
import ast
import warnings

from fugal_subnet.benchmarks import code_sets, humaneval

LC_INPUT = 's = "a\\pb\\/c", k = 2'          # dataset text: backslashes in a string literal
EVALPLUS = 'inputs = [["x\\py"]]\nresults = ["x\\py"]\nassertion(f(*inp), exp, 0)\n'
HUMANEVAL = 'def check(candidate):\n    assert candidate("a\\ b") == "a\\ b"\n'


def _plain(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args)


def test_leetcode_inputs_parse_silently_to_the_same_values():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = code_sets._lc_kwargs(LC_INPUT)
    assert got == _plain(lambda t: {k.arg: ast.literal_eval(k.value) for k in
                                    ast.parse(f"f({t})", mode="eval").body.keywords}, LC_INPUT)
    assert got["s"] == "a\\pb\\/c"


def test_evalplus_tests_parse_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inputs, results, atol, name = code_sets._evalplus_io(EVALPLUS)
    assert inputs == [["x\\py"]] and results == ["x\\py"] and atol == 0.0 and name == "f"


def test_humaneval_tests_parse_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ins, outs = humaneval._extract_io_from_tests(HUMANEVAL, "candidate", "")
    assert ins == [["a\\ b"]] and outs == ["a\\ b"]


def test_the_filter_does_not_leak_out_of_the_parse():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with code_sets.quiet_escapes():
            pass
        warnings.warn("still reported", UserWarning)
    assert any("still reported" in str(w.message) for w in seen)
