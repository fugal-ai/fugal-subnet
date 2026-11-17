"""Protocol v3 code questions, graded by the pinned code grader (fugal_subnet/graders_code.py).

Two stages, so the pool is a pure function of pinned inputs:

1. CANDIDATES. Each source's rows (pinned revision) become candidate problems: a prompt, a
   checker kind, every test the checker can judge exactly, and the source's reference
   solutions. Pure functions of the rows; no code runs here.
2. THE KEEP-LIST. `scripts/calibrate_code_tests.py` runs the reference solutions in the
   grader's own sandbox and keeps a problem only when a reference passes its tests twice within
   a quarter of the CPU limit (and, for stdin programs, when the answer is unique). The result,
   `code_keep_v3.json` (problem id -> kept test indices), is pinned data: its sha256 is
   checked by scripts/check_safety_invariants.py, and the pool manifest's content hash covers
   every test it selects. Loading applies it; nothing is executed at load time.

Sources (licences accepted by the owner, 2026-11-16):
- code_leetcode    newfacade/LeetCodeDataset, TRAIN file only (`exec_func`, method of Solution)
- code_mbppplus    evalplus/mbppplus (`exec_func`)
- code_humanevalplus evalplus/humanevalplus (`exec_func`)
- code_contests    deepmind/code_contests, the first CC_TRAIN_SHARDS train shards (`exec_stdio`)
LiveCodeBench is not used: it has no reference solutions to calibrate against, its private
tests are pickled, and the problems RouterArena evaluates on (its release_v2) are excluded from
training anyway.

Prompt suffixes are consensus text (FUNC_SUFFIX, STDIO_SUFFIX).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import warnings
from collections import Counter
from contextlib import contextmanager

from fugal_subnet import graders_code

DATASET_REVISIONS_CODE: dict[str, str] = {
    "newfacade/LeetCodeDataset": "215604aeed660029df7de2fea5a4d7b6ed476a08",
    "evalplus/mbppplus": "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
    "evalplus/humanevalplus": "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
    "deepmind/code_contests": "802411c3010cb00d1b05bad57ca77365a3c699d6",
}
LEETCODE_TRAIN_FILE = "LeetCodeDataset-train.jsonl"
# The first 16 of the 39 train shards (~3 GB): enough problems for the 2,500 the pool takes,
# without every operator downloading 7.6 GB.
CC_TRAIN_SHARDS = (
    "data/train-00000-of-00039-e991a271dbfa9925.parquet",
    "data/train-00001-of-00039-e092fe56fda18715.parquet",
    "data/train-00002-of-00039-9cea23812e920e41.parquet",
    "data/train-00003-of-00039-e3822fccad6e083a.parquet",
    "data/train-00004-of-00039-cefe355b4667b27e.parquet",
    "data/train-00005-of-00039-b7580d2d846c2136.parquet",
    "data/train-00006-of-00039-65184bb9f7d61fde.parquet",
    "data/train-00007-of-00039-05785de21e8b8429.parquet",
    "data/train-00008-of-00039-7246e6b7423b404f.parquet",
    "data/train-00009-of-00039-b8c920f6629b57b2.parquet",
    "data/train-00010-of-00039-6de28ba20654f69b.parquet",
    "data/train-00011-of-00039-5de236be5188959d.parquet",
    "data/train-00012-of-00039-da9476a39a1bdbb7.parquet",
    "data/train-00013-of-00039-30b8c3829ee3b962.parquet",
    "data/train-00014-of-00039-dc3ebb07a3cba8e4.parquet",
    "data/train-00015-of-00039-19ccd7331d695677.parquet",
)
CC_COLUMNS = ("name", "description", "public_tests", "private_tests", "generated_tests",
              "solutions", "input_file", "output_file")
CC_SAMPLE = 2500                   # code_contests problems in the pool (sha256 order of the id)

CODE_LICENCES = {
    "code_leetcode": "Apache-2.0 (newfacade/LeetCodeDataset; statements are LeetCode's)",
    "code_mbppplus": "Apache-2.0 (evalplus/mbppplus)",
    "code_humanevalplus": "Apache-2.0 (evalplus/humanevalplus)",
    "code_contests": "CC-BY-4.0 (deepmind/code_contests)",
}

FUNC_SUFFIX = ("Reply with your complete solution in one fenced ```python code block. Use the "
               "Python 3 standard library only, and do not read input or print.")
STDIO_SUFFIX = ("Write a complete Python 3 program that reads the input from standard input and "
                "writes the answer to standard output. Use the standard library only. Reply with "
                "the program in one fenced ```python code block.")

MAX_TEST_CHARS = 2000              # one test's input + expected output, as JSON text
MIN_TESTS = 5                      # fewer usable tests and a candidate is not offered
MAX_CANDIDATE_TESTS = 40            # the first usable tests only: calibration examines no more
ID_HASH_CHARS = 16
# Problems whose statement allows more than one right answer: exact comparison would fail
# correct replies (as found in the code-set research, 2026-11-16).
MULTI_ANSWER = re.compile(r"any order|return any|any of them|any valid|any one of|"
                          r"multiple (valid |possible )?(answers|solutions)|answers within 10", re.I)
FLOATISH = re.compile(r"^[+-]?\d*\.\d+([eE][+-]?\d+)?$")
PYTHON3 = 3                        # code_contests solutions.language


def _note(stats: Counter | None, reason: str, n: int = 1) -> None:
    if stats is not None:
        stats[reason] += n


def _hid(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:ID_HASH_CHARS]


def _json_native(x, floats: bool = True) -> bool:
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return True
    if isinstance(x, float):
        return floats and x == x and x not in (float("inf"), float("-inf"))
    if isinstance(x, (list, tuple)):
        return all(_json_native(v, floats) for v in x)
    if isinstance(x, dict):
        return all(isinstance(k, str) and _json_native(v, floats) for k, v in x.items())
    return False


def _canon(x):
    """JSON round trip: tuples become lists, as they do on the wire to the grader."""
    return json.loads(json.dumps(x))


def _small(*parts) -> bool:
    return sum(len(json.dumps(p)) for p in parts) <= MAX_TEST_CHARS


@contextmanager
def quiet_escapes():
    """Parse dataset text without Python's "invalid escape sequence" warnings.

    Test cases quote raw text with backslashes (LaTeX, regexes); Python 3.12
    reports each as a SyntaxWarning while parsing (3.11 as a silent
    DeprecationWarning). The parsed values are the same either way; the
    warnings only filled operators' logs (v0.5.0, Yuma)."""
    with warnings.catch_warnings():
        for category in (SyntaxWarning, DeprecationWarning):
            warnings.filterwarnings("ignore", message=r".*invalid escape sequence",
                                    category=category)
        yield


# --- LeetCodeDataset (exec_func) ------------------------------------------------

def _lc_kwargs(text: str) -> dict:
    """'n = 7, queries = [[0,5]]' -> {'n': 7, 'queries': [[0, 5]]}; keyword literals only."""
    with quiet_escapes():
        call = ast.parse(f"f({text})", mode="eval").body
    if not isinstance(call, ast.Call) or call.args:
        raise ValueError("positional or malformed input")
    out = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    if any(k is None for k in out):
        raise ValueError("**kwargs input")
    return out


def leetcode_candidates(rows, stats: Counter | None = None) -> list[dict]:
    out = []
    for r in rows:
        _note(stats, "rows")
        starter, test = r.get("starter_code") or "", r.get("test") or ""
        entry = r.get("entry_point") or ""
        if "list_node" in test or "tree_node" in test or "Node" in starter:
            _note(stats, "linked_list_or_tree")
            continue
        if not entry.startswith("Solution().") or not graders_code.FUNC_NAME.match(entry):
            _note(stats, "not_a_solution_method")
            continue
        if "-> None" in starter:
            _note(stats, "in_place")
            continue
        if MULTI_ANSWER.search(r.get("problem_description") or ""):
            _note(stats, "multi_answer")
            continue
        tests = []
        for io in r.get("input_output") or []:
            try:
                with quiet_escapes():
                    kw, want = _lc_kwargs(io["input"]), ast.literal_eval(io["output"])
            except Exception:                      # noqa: BLE001 - an unparseable test
                continue
            if _json_native(kw) and _json_native(want, floats=False) and _small(kw, want):
                tests.append((_canon(kw), _canon(want)))
        tests = tests[:MAX_CANDIDATE_TESTS]
        if len(tests) < MIN_TESTS:
            _note(stats, "too_few_tests")
            continue
        prompt = (f"{r['problem_description'].strip()}\n\nComplete this code:\n```python\n"
                  f"{starter.rstrip()}\n```\n\n{FUNC_SUFFIX}")
        out.append({"benchmark": "code_leetcode", "source_id": r["task_id"], "kind": "exec_func",
                    "func": entry, "prompt": prompt, "tests": tests, "float_tol": None,
                    "references": [r.get("completion") or ""],
                    "meta": {"difficulty": r.get("difficulty"), "date": r.get("estimated_date"),
                             "leetcode_id": r.get("question_id")}})
    return sorted(out, key=lambda c: c["source_id"])


# --- EvalPlus (exec_func) -------------------------------------------------------

def _evalplus_io(test_src: str):
    """(inputs, results, atol, called name) from an EvalPlus test: the literal `inputs` and
    `results` assignments, and the first `assertion(<f>(*inp), exp, atol)` call."""
    with quiet_escapes():
        tree = ast.parse(test_src)
    vals, atol, name = {}, None, None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("inputs", "results")):
            vals[node.targets[0].id] = ast.literal_eval(node.value)
        if (atol is None and isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "assertion" and len(node.args) == 3):
            atol = ast.literal_eval(node.args[2])
            inner = node.args[0]
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                name = inner.func.id
    if "inputs" not in vals or "results" not in vals or atol is None:
        raise ValueError("no literal inputs/results/assertion")
    return vals["inputs"], vals["results"], float(atol), name


def _evalplus_candidates(rows, bench: str, stats: Counter | None, *, mbpp: bool) -> list[dict]:
    out = []
    for r in rows:
        _note(stats, "rows")
        try:
            inputs, results, atol, called = _evalplus_io(r["test"])
        except Exception:                          # noqa: BLE001 - not a literal test
            _note(stats, "no_literal_tests")
            continue
        func = called if mbpp else r["entry_point"]
        if not func or not graders_code.FUNC_NAME.match(func):
            _note(stats, "no_function_name")
            continue
        tests = [(_canon(list(i)), _canon(w)) for i, w in zip(inputs, results)
                 if isinstance(i, (list, tuple)) and _json_native(i) and _json_native(w) and _small(i, w)]
        tests = tests[:MAX_CANDIDATE_TESTS]
        if len(tests) < MIN_TESTS:
            _note(stats, "too_few_tests")
            continue
        if mbpp:
            example = (r.get("test_list") or [""])[0]
            prompt = (f"{r['prompt'].strip()}\n\nYour function must pass this test:\n```python\n"
                      f"{example}\n```\n\n{FUNC_SUFFIX}")
            reference, sid = r["code"], f"mbpp_{r['task_id']}"
        else:
            prompt = (f"Complete this Python function:\n```python\n{r['prompt'].rstrip()}\n```\n\n"
                      f"{FUNC_SUFFIX}")
            reference, sid = r["prompt"] + r["canonical_solution"], str(r["task_id"])
        out.append({"benchmark": bench, "source_id": sid, "kind": "exec_func", "func": func,
                    "prompt": prompt, "tests": tests, "float_tol": atol if atol > 0 else None,
                    "references": [reference], "meta": {}})
    return sorted(out, key=lambda c: c["source_id"])


def mbppplus_candidates(rows, stats: Counter | None = None) -> list[dict]:
    return _evalplus_candidates(rows, "code_mbppplus", stats, mbpp=True)


def humanevalplus_candidates(rows, stats: Counter | None = None) -> list[dict]:
    return _evalplus_candidates(rows, "code_humanevalplus", stats, mbpp=False)


# --- CodeContests (exec_stdio) --------------------------------------------------

def codecontests_candidates(rows, stats: Counter | None = None) -> list[dict]:
    out = []
    for r in rows:
        _note(stats, "rows")
        if r.get("input_file") or r.get("output_file"):
            _note(stats, "file_io")
            continue
        sols = r.get("solutions") or {}
        refs = [s for lang, s in zip(sols.get("language") or [], sols.get("solution") or [])
                if lang == PYTHON3][:3]
        if not refs:
            _note(stats, "no_python3_solution")
            continue
        desc = (r.get("description") or "").strip()
        if not desc or "interactive" in desc.lower():
            _note(stats, "interactive_or_empty")
            continue
        tests = []
        for group in ("public_tests", "private_tests", "generated_tests"):
            g = r.get(group) or {}
            for i, o in zip(g.get("input") or [], g.get("output") or []):
                if _small(i, o):
                    tests.append((i, o))
        if any(FLOATISH.match(tok) for _, o in tests for tok in o.split()):
            _note(stats, "float_output")
            continue
        tests = tests[:MAX_CANDIDATE_TESTS]
        if len(tests) < MIN_TESTS:
            _note(stats, "too_few_tests")
            continue
        out.append({"benchmark": "code_contests", "source_id": r["name"], "kind": "exec_stdio",
                    "prompt": f"{desc}\n\n{STDIO_SUFFIX}", "tests": tests, "float_tol": None,
                    "references": refs, "meta": {}})
    return sorted(out, key=lambda c: c["source_id"])


# --- candidates -> questions ----------------------------------------------------

def question_id(c: dict) -> str:
    prefix = {"code_leetcode": "code_lc_", "code_mbppplus": "code_mbpp_",
              "code_humanevalplus": "code_he_", "code_contests": "code_cc_"}[c["benchmark"]]
    sid = c["source_id"]
    tail = sid if c["benchmark"] == "code_leetcode" else _hid(sid)
    return prefix + tail


def to_question(c: dict, kept: list[int]) -> dict:
    """The pool question for candidate `c` with the calibrated tests `kept` (indices)."""
    tests = [c["tests"][i] for i in kept]
    if c["kind"] == "exec_func":
        params = {"func": c["func"], "inputs": [t[0] for t in tests]}
        gold = [t[1] for t in tests]
    else:
        params = {"tests": [{"input": t[0], "output": t[1]} for t in tests]}
        gold = ""
    if c["float_tol"] is not None:
        params["float_tol"] = c["float_tol"]
    return {"question_id": question_id(c), "benchmark": c["benchmark"], "prompt": c["prompt"],
            "gold": gold, "grader_id": c["kind"],
            "metadata": {"checker": {"id": c["kind"], "params": params},
                         "source_id": c["source_id"], **{k: v for k, v in c["meta"].items() if v is not None}}}


def keep_path() -> str:
    from importlib.resources import files
    return str(files("fugal_subnet.benchmarks") / "code_keep_v3.json")


def read_keep(path: str | None = None) -> dict[str, dict[str, list[int]]]:
    """{benchmark: {source_id: kept test indices}} from the pinned keep-list."""
    with open(path or keep_path(), encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, dict[str, list[int]]] = {}
    for row in data["rows"]:
        out.setdefault(row["benchmark"], {})[row["source_id"]] = list(row["tests"])
    return out


def apply_keep(cands: list[dict], keep: dict[str, list[int]], stats: Counter | None = None) -> list[dict]:
    out = []
    for c in cands:
        kept = keep.get(c["source_id"])
        if not kept:
            _note(stats, "not_calibrated")
            continue
        if max(kept) >= len(c["tests"]):
            raise ValueError(f"keep-list names test {max(kept)} of {c['source_id']!r}, which has "
                             f"{len(c['tests'])}: the keep-list and the loader disagree")
        out.append(to_question(c, kept))
    return sorted(out, key=lambda q: q["question_id"])


# --- loading from the pinned revisions ------------------------------------------

def _hf_file(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset", revision=DATASET_REVISIONS_CODE[repo])


def leetcode_rows():
    with open(_hf_file("newfacade/LeetCodeDataset", LEETCODE_TRAIN_FILE), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _parquet_rows(repo: str, path_glob: str, columns=None):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset",
                                                     revision=DATASET_REVISIONS_CODE[repo])
                   if re.fullmatch(path_glob, f))
    rows = []
    for f in files:
        rows.extend(pq.read_table(_hf_file(repo, f), columns=columns).to_pylist())
    return rows


def mbppplus_rows():
    return _parquet_rows("evalplus/mbppplus", r"data/test-.*\.parquet")


def humanevalplus_rows():
    return _parquet_rows("evalplus/humanevalplus", r"data/test-.*\.parquet")


def iter_parquet_rows(paths, columns=None, batch_size: int = 64):
    """Rows of each file in order, a batch at a time: the code_contests shards hold every
    language's solutions, and reading a shard whole takes gigabytes."""
    import pyarrow.parquet as pq
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def codecontests_rows():
    return iter_parquet_rows([_hf_file("deepmind/code_contests", s) for s in CC_TRAIN_SHARDS],
                             columns=list(CC_COLUMNS))


CANDIDATES = {
    "code_leetcode": (leetcode_rows, leetcode_candidates),
    "code_mbppplus": (mbppplus_rows, mbppplus_candidates),
    "code_humanevalplus": (humanevalplus_rows, humanevalplus_candidates),
    "code_contests": (codecontests_rows, codecontests_candidates),
}


def load_code(bench: str, stats: Counter | None = None, keep: dict | None = None) -> list[dict]:
    rows_fn, cand_fn = CANDIDATES[bench]
    keep = keep if keep is not None else read_keep()
    return apply_keep(cand_fn(rows_fn(), stats), keep.get(bench, {}), stats)


LOADERS_CODE = {b: (lambda b=b: load_code(b)) for b in CANDIDATES}


def code_sandbox_required(pool: list[dict]) -> bool:
    from fugal_subnet.config import SANDBOXED_CODE_CHECKERS
    return any(q.get("grader_id") in SANDBOXED_CODE_CHECKERS for q in pool)


__all__ = ["CANDIDATES", "CODE_LICENCES", "LOADERS_CODE", "apply_keep", "read_keep", "to_question",
           "question_id", "keep_path"]
