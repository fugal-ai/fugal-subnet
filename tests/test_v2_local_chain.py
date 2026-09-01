from types import SimpleNamespace

from scripts import test_v2_local_chain as harness


def test_local_backbone_prewarm_creates_cache_and_forces_parent_offline(
    monkeypatch,
    tmp_path,
):
    import fugal_subnet.benchmarks.slicer as slicer
    import fugal_subnet.v2.benchmarks as benchmarks

    observed = {}
    pool = [{"prompt": f"prompt-{index}"} for index in range(harness.SLICE_SIZE)]

    def load_pool():
        observed["offline"] = (
            harness.os.environ.get("HF_DATASETS_OFFLINE"),
            harness.os.environ.get("HF_HUB_OFFLINE"),
            harness.os.environ.get("TRANSFORMERS_OFFLINE"),
        )
        return pool

    monkeypatch.setattr(benchmarks, "load_pool", load_pool)
    monkeypatch.setattr(slicer, "derive_nonce", lambda *_args: b"nonce")
    monkeypatch.setattr(slicer, "select_slice", lambda *_args: pool)
    calls = []
    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command)
        or SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    harness._prewarm_local_backbone(
        tmp_path,
        {},
        boundary_hash="a" * 64,
        lock_path=tmp_path / "backbone.lock",
    )

    assert observed["offline"] == ("1", "1", "1")
    assert not any(key in harness.os.environ for key in (
        "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"
    ))
    assert (tmp_path / "backbone-cache").is_dir()
    assert len(calls) == 2
