"""The v2 miner caches its manifest-verified pool on the results volume so a restart does
not need Hugging Face. The cache is untrusted: verified on every read, deleted and rebuilt
when it fails, and never trusted because of FUGAL_POOL_UNPINNED or a lax caller."""
import gzip
import json

import pytest

from fugal_subnet import config
from fugal_subnet.benchmarks import loader


def _pool(n=40):
    # Deliberately NOT sorted by id: the cache must keep load order (embeddings are positional).
    return [{"question_id": f"mmlupro_{(i * 7) % n:05d}", "benchmark": "mmlupro", "prompt": f"Q{i}?",
             "gold": "A", "grader_id": "mcq"} for i in range(n)]


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """A manifest pinned to _pool(), a builder that counts its (Hugging Face) builds."""
    pool = _pool()
    manifest = tmp_path / "pool_manifest_v2.json"
    manifest.write_text(json.dumps({"n_questions": len(pool), "pool_hash": loader.pool_hash(pool),
                                    "content_hash": loader.content_hash(pool)}))
    monkeypatch.setattr(loader, "manifest_path_v2", lambda: str(manifest))
    monkeypatch.setattr(config, "BENCHMARK_POOL_MIN_SIZE", 10)
    builds = {"n": 0}

    def build(strict=True):
        builds["n"] += 1
        return _pool()

    monkeypatch.setattr(loader, "build_pool_v2", build)
    for var in ("FUGAL_BENCHMARK_POOL_V2", "FUGAL_POOL_UNPINNED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FUGAL_POOL_FILE_V2", str(tmp_path / "absent.json.gz"))
    cache = tmp_path / "results" / "pool_v2.json.gz"
    monkeypatch.setenv("FUGAL_POOL_CACHE_V2", str(cache))
    return cache, builds


def _tamper(cache, fn):
    with gzip.open(cache, "rt", encoding="utf-8") as f:
        pool = json.load(f)
    fn(pool)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump(pool, f)


def test_the_first_load_builds_and_caches_and_the_second_reads_the_cache(pinned):
    cache, builds = pinned
    first = loader.load_pool(2)
    assert builds["n"] == 1 and cache.is_file()
    second = loader.load_pool(2)
    assert builds["n"] == 1                       # no Hugging Face on the restart
    assert second == first                        # same questions, same ORDER
    assert not list(cache.parent.glob("*.tmp.*"))  # the write was atomic


@pytest.mark.parametrize("tamper", [
    lambda p: p[0].update(gold="B"),            # a changed gold answer
    lambda p: p.pop(),                            # a dropped question
    lambda p: p.append(dict(p[0], question_id="zzz")),
])
def test_a_tampered_cache_is_deleted_and_rebuilt_from_the_pinned_revisions(pinned, tamper):
    cache, builds = pinned
    loader.load_pool(2)
    _tamper(cache, tamper)
    pool = loader.load_pool(2)
    assert builds["n"] == 2 and pool == _pool()
    assert loader.content_hash(loader._read_pool_file(str(cache))) == loader.content_hash(_pool())


def test_a_corrupt_cache_file_is_rebuilt(pinned):
    cache, builds = pinned
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"not gzip")
    assert loader.load_pool(2) == _pool() and builds["n"] == 1


def test_unpinned_and_lax_callers_still_get_a_verified_cache(pinned, monkeypatch):
    cache, builds = pinned
    loader.load_pool(2)
    _tamper(cache, lambda p: p[0].update(gold="B"))
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "1")
    assert loader.load_all_v2(strict=False) == _pool() and builds["n"] == 2


def test_a_pool_that_fails_the_manifest_is_never_cached(pinned, monkeypatch):
    cache, builds = pinned
    monkeypatch.setattr(loader, "build_pool_v2", lambda strict=True: _pool()[:-1])
    with pytest.raises(RuntimeError, match="does not match the pinned manifest"):
        loader.load_pool(2)
    assert not cache.exists()


def test_no_cache_without_the_setting(pinned, monkeypatch):
    cache, builds = pinned
    monkeypatch.delenv("FUGAL_POOL_CACHE_V2")
    loader.load_pool(2)
    loader.load_pool(2)
    assert builds["n"] == 2 and not cache.exists()
