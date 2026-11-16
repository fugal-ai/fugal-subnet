"""Fetching the pinned v3 embeddings file: the hash is the only thing trusted."""
import hashlib

import numpy as np
import pytest

from fugal_subnet import config
from fugal_subnet import embeddings_file as ef


def make(path, seed=0):
    H = np.random.default_rng(seed).standard_normal((3, 8)).astype(np.float32)
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    np.savez(path, question_ids=np.array(["a", "b", "c"]), H=H)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_file_with_the_pinned_hash_is_fetched_and_loads(tmp_path):
    src = tmp_path / "src.npz"
    sha = make(src)
    dest = tmp_path / "out" / "pool_v3.npz"
    assert ef.fetch(src.as_uri(), sha, str(dest)) == str(dest)
    assert ef.Embeddings.load(str(dest), sha).covers(["a", "b", "c"]) == []


def test_a_download_with_another_hash_never_replaces_the_file(tmp_path):
    good, bad = tmp_path / "good.npz", tmp_path / "bad.npz"
    sha = make(good, 0)
    make(bad, 1)
    dest = tmp_path / "pool_v3.npz"
    dest.write_bytes(good.read_bytes())
    dest.write_bytes(b"stale bytes a previous run left")     # not the pinned file
    with pytest.raises(ValueError, match="pinned"):
        ef.fetch(bad.as_uri(), sha, str(dest))
    assert dest.read_bytes() == b"stale bytes a previous run left"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".embeddings-")] == []


def test_a_file_already_in_place_is_kept_without_a_download(tmp_path):
    src = tmp_path / "src.npz"
    sha = make(src)
    dest = tmp_path / "pool_v3.npz"
    dest.write_bytes(src.read_bytes())
    assert ef.fetch("file:///nonexistent/never-read.npz", sha, str(dest)) == str(dest)


def test_no_pin_means_nothing_to_fetch(monkeypatch, capsys):
    monkeypatch.setattr(config, "EMBEDDINGS_URL", None)
    monkeypatch.setattr(config, "EMBEDDINGS_SHA256", None)
    assert ef.main(["fetch"]) == 0
    assert "nothing to fetch" in capsys.readouterr().out
    with pytest.raises(ValueError, match="no pinned"):
        ef.fetch("file:///x", "", "/tmp/never")


def test_the_command_fetches_what_config_pins(tmp_path, monkeypatch):
    src = tmp_path / "src.npz"
    sha = make(src)
    monkeypatch.setattr(config, "EMBEDDINGS_URL", src.as_uri())
    monkeypatch.setattr(config, "EMBEDDINGS_SHA256", sha)
    dest = tmp_path / "d" / "pool_v3.npz"
    assert ef.main(["fetch", "--dest", str(dest)]) == 0
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == sha
