"""The sharded CI keeps the check names branch protection requires, and the
shard split covers every test file exactly once (scripts/ci_test_shards.py,
.github/workflows/ci.yml).

A required check that never reports blocks every pull request; a test file no
shard runs is a test that silently stopped running. Both are easy to cause by
editing the workflow, and neither shows up as a red job.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

yaml = pytest.importorskip("yaml")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
SCRIPT = ROOT / "scripts" / "ci_test_shards.py"

# Branch protection on main requires exactly these (codeql.yml supplies
# "Analyze Python").
REQUIRED = {
    "Python 3.11", "Python 3.12", "Lint, audit, and package",
    "Attack suites with the DCAP verifier installed", "Docker build smoke test",
    "Shared success contract conformance",
}


def _shards_module():
    spec = importlib.util.spec_from_file_location("ci_test_shards", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jobs():
    return yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]


def _names(job):
    """Every check name a job reports, with its matrix expanded."""
    name = job.get("name", "")
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    names = [name]
    for key, values in matrix.items():
        token = "${{ matrix.%s }}" % key
        if token in name:
            names = [n.replace(token, str(v)) for n in names for v in values]
    return names


def test_every_required_check_name_is_reported():
    reported = {n for job in _jobs().values() for n in _names(job)}
    assert REQUIRED <= reported, REQUIRED - reported


def test_a_required_matrix_job_is_never_skipped():
    # A skipped matrix job reports its name unexpanded ("Python ${{ ... }}"),
    # so the required "Python 3.x" check would never arrive.
    for job_id, job in _jobs().items():
        if (job.get("strategy") or {}).get("matrix") and set(_names(job)) & REQUIRED:
            assert job.get("if") == "${{ !cancelled() }}", job_id


def test_the_python_checks_stand_for_every_shard_and_the_checks_job():
    jobs = _jobs()
    gate = next(j for j in jobs.values() if "Python 3.11" in _names(j))
    assert set(gate["needs"]) >= {"changes", "tests", "checks"}
    shards = jobs["tests"]["strategy"]["matrix"]["shard"]
    assert shards == list(range(1, len(shards) + 1))
    step = next(s["run"] for s in jobs["tests"]["steps"] if "ci_test_shards.py" in s.get("run", ""))
    assert f"ci_test_shards.py ${{{{ matrix.shard }}}} {len(shards)} " in step
    assert f"/{len(shards)}" in jobs["tests"]["name"]
    assert set(jobs["tests"]["strategy"]["matrix"]["python-version"]) == {"3.11", "3.12"}


@pytest.mark.parametrize("total", [1, 2, 3, 4, 7])
def test_the_split_covers_every_test_file_exactly_once(total):
    mod = _shards_module()
    files = mod.test_files()
    shards = mod.split(files, mod.load_weights(), total)
    flat = [f for s in shards for f in s]
    assert sorted(flat) == files and len(flat) == len(set(flat))
    assert pathlib.Path(__file__).relative_to(ROOT).as_posix() in flat
    assert shards == mod.split(list(reversed(files)), mod.load_weights(), total)


def test_the_cli_prints_a_shard_and_refuses_a_bad_one():
    out = subprocess.run([sys.executable, str(SCRIPT), "1", "4"], capture_output=True,
                         text=True, check=True).stdout.split()
    assert out and all(f.startswith("tests/test_") for f in out)
    bad = subprocess.run([sys.executable, str(SCRIPT), "5", "4"], capture_output=True, text=True)
    assert bad.returncode != 0


def test_an_empty_shard_is_an_error_not_the_whole_suite():
    mod = _shards_module()
    n = len(mod.test_files())
    res = subprocess.run([sys.executable, str(SCRIPT), "1", str(n + 1)],
                         capture_output=True, text=True)
    assert res.returncode != 0 and not res.stdout.strip()


def test_the_capacity_tests_leave_the_shards_for_their_own_job():
    jobs = _jobs()
    step = next(s["run"] for s in jobs["tests"]["steps"] if "ci_test_shards.py" in s.get("run", ""))
    assert '-m "not capacity"' in step
    runs = [s.get("run", "") for s in jobs["capacity"]["steps"]]
    assert any("-m capacity" in r for r in runs)
    assert "capacity" in jobs["test"]["needs"]
    assert "release/" in jobs["capacity"]["if"] and "push" in jobs["capacity"]["if"]
    nightly = yaml.safe_load((CI.parent / "nightly.yml").read_text(encoding="utf-8"))["jobs"]
    assert any("-m capacity" in s.get("run", "")
               for j in nightly.values() for s in j.get("steps", []))
