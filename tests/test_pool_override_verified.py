"""FUGAL_BENCHMARK_POOL is a file, not a reason to trust its contents.

The override is the documented production path: MAINNET_LAUNCH tells every
operator to materialise the pinned pool and point every neuron at it. It was
also the one path on which neither consensus guard ran — `load_all` returned
before `_verify_against_manifest` and before the size floor. Measured: a
150-question pool served 40+ live epochs on netuid 552 and nothing objected.

So the override must be checked exactly like the loader path, and a pool that
is deliberately NOT the pinned one has to declare it with FUGAL_POOL_UNPINNED=1
— a loud, greppable statement instead of a silent bypass.
"""
import json

import pytest

from fugal_subnet.benchmarks import loader


def _tiny_pool(n: int = 150) -> list[dict]:
    return [{
        "question_id": f"t_{i:04d}", "prompt": f"{i}+{i}?", "gold": str(2 * i),
        "grader_id": "numeric_final", "benchmark": "gsm8k", "metadata": {},
    } for i in range(n)]


@pytest.fixture
def tiny_override(tmp_path, monkeypatch):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(_tiny_pool()), encoding="utf-8")
    monkeypatch.setenv("FUGAL_BENCHMARK_POOL", str(path))
    monkeypatch.delenv("FUGAL_POOL_UNPINNED", raising=False)
    # Unset, so the pinned manifest's own skip list applies and the content
    # check actually runs rather than being disabled by a skip mismatch.
    monkeypatch.delenv("FUGAL_SKIP_BENCHMARKS", raising=False)
    return path


def test_an_undeclared_tiny_override_is_refused(tiny_override):
    with pytest.raises(RuntimeError) as e:
        loader.load_all(strict=True)
    msg = str(e.value)
    # Either guard may fire first; both name the cause rather than the symptom.
    assert "manifest" in msg or "floor" in msg


def test_the_size_floor_applies_to_the_override(tiny_override, monkeypatch):
    """Even when the manifest check is neutralised, the floor still holds."""
    monkeypatch.setattr(loader, "_verify_against_manifest", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="floor"):
        loader.load_all(strict=True)


def test_a_declared_unpinned_override_loads(tiny_override, monkeypatch, caplog):
    monkeypatch.setenv("FUGAL_POOL_UNPINNED", "1")
    with caplog.at_level("WARNING"):
        pool = loader.load_all(strict=True)
    assert len(pool) == 150
    assert any("FUGAL_POOL_UNPINNED" in r.getMessage() for r in caplog.records), (
        "declaring a pool unpinned must be loud in the log"
    )


def test_non_strict_only_warns(tiny_override, caplog):
    with caplog.at_level("WARNING"):
        pool = loader.load_all(strict=False)
    assert len(pool) == 150
    assert any("floor" in r.getMessage() or "manifest" in r.getMessage()
               for r in caplog.records)
