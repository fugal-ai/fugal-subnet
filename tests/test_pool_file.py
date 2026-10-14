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
