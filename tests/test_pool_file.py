"""A materialised pool file is preferred over Hugging Face, and checked.

Diligence 2026-10, finding 8. `load_all()` rebuilt the pool from Hugging Face
on every neuron; a removed pinned revision would stop new operators starting.
scripts/export_pool.py writes one gzipped file and the loader prefers it when
present, running the same manifest and size checks as an explicit override.
"""
import gzip
import json

import pytest

from fugal_subnet.benchmarks import loader
from tests.test_pool_override_verified import _tiny_pool


@pytest.fixture
def pool_file(tmp_path, monkeypatch):
    path = tmp_path / "pool.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(_tiny_pool(), f)
    monkeypatch.delenv("FUGAL_BENCHMARK_POOL", raising=False)
    monkeypatch.delenv("FUGAL_SKIP_BENCHMARKS", raising=False)
    monkeypatch.setenv("FUGAL_POOL_FILE", str(path))
    return path


def test_the_pool_file_is_loaded_without_hugging_face(pool_file, monkeypatch):
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "1")   # a 150-question fixture, declared
    pool = loader.load_all(strict=True)
    assert len(pool) == 150 and pool[0]["question_id"] == "t_0000"


def test_the_pool_file_is_checked_like_an_override(pool_file, monkeypatch):
    monkeypatch.delenv("FUGAL_POOL_UNPINNED", raising=False)
    with pytest.raises(RuntimeError):
        loader.load_all(strict=True)


def test_an_explicit_override_wins_over_the_default_file(pool_file, tmp_path, monkeypatch):
    other = tmp_path / "other.json"
    other.write_text(json.dumps(_tiny_pool(10)), encoding="utf-8")
    monkeypatch.setenv("FUGAL_BENCHMARK_POOL", str(other))
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "1")
    assert len(loader.load_all(strict=True)) == 10


def test_a_pool_file_path_that_is_not_a_regular_file_is_ignored(monkeypatch):
    """scripts/export_pool.py could not run on Linux (found 2026-11-02).

    It set FUGAL_POOL_FILE=os.devnull to mean "no file", the loader tested the
    path with os.path.exists — true for /dev/null — and handed a device node to
    the JSON parser. The documented way of building the consensus pool failed
    with "Expecting value: line 1 column 1" before touching Hugging Face. The
    loader must treat anything that is not a regular file as absent and fall
    through to the pinned revisions.
    """
    import os

    monkeypatch.delenv("FUGAL_BENCHMARK_POOL", raising=False)
    monkeypatch.delenv("FUGAL_POOL_UNPINNED", raising=False)
    monkeypatch.setenv("FUGAL_POOL_FILE", os.devnull)
    # Skip every benchmark so the fall-through path needs no network; the
    # assertion is only that it is the path taken, not what it loads.
    monkeypatch.setenv("FUGAL_SKIP_BENCHMARKS", ",".join(sorted(loader._BENCHMARKS)))
    pool = loader.load_all(strict=False)
    assert pool == []


def test_export_pool_never_points_the_loader_at_an_existing_path():
    """The script's own "no file" sentinel must not resolve to a regular file."""
    import importlib.util
    import os
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "export_pool.py"
    spec = importlib.util.spec_from_file_location("export_pool", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.NO_POOL_FILE != os.devnull
    assert not os.path.exists(mod.NO_POOL_FILE) or not os.path.isfile(mod.NO_POOL_FILE)


def test_export_pool_writes_a_loadable_byte_reproducible_file(tmp_path, monkeypatch):
    """scripts/export_pool.py end to end, against a stubbed loader.

    Two defects hid behind each other until 2026-11-02: the os.devnull sentinel
    above, and `gzip.open(..., mtime=0)`, which raises TypeError (only
    GzipFile takes mtime). Neither was reachable by any test because nothing
    ran the script. This does, and asserts the properties it promises: the
    file loads back through the same loader, and two builds are byte-identical.
    """
    import gzip
    import hashlib
    import importlib.util
    import json
    import pathlib
    import sys

    src = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "export_pool.py"
    spec = importlib.util.spec_from_file_location("export_pool", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fixture = _tiny_pool(12)
    monkeypatch.setattr(loader, "load_all", lambda strict=True: list(reversed(fixture)))

    def build(name):
        out = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["export_pool.py", "--out", str(out)])
        assert mod.main() == 0
        return out.read_bytes()

    first, second = build("a.json.gz"), build("b.json.gz")
    assert first == second, "mtime=0 must make the archive reproducible"
    with gzip.open(tmp_path / "a.json.gz", "rt", encoding="utf-8") as f:
        pool = json.load(f)
    assert [q["question_id"] for q in pool] == sorted(q["question_id"] for q in fixture)
    assert hashlib.sha256(first).hexdigest()  # a real file came out
