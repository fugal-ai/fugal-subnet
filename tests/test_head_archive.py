"""The validator's private head archive (mainnet, 2026-11-04).

Verified heads used to be discarded after their sha256 checks. With
FUGAL_HEAD_ARCHIVE_DIR set they are kept: raw bytes named by sha256, plus an
index row per (epoch, uid). Diagnostics only. These tests pin the file layout,
the privacy modes, that a hash the bytes don't match never becomes a path, that
the archive never parses a head, that a disk error cannot cost the epoch, and
that nothing it produces reaches scoring, the reveal or the fingerprint.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from types import SimpleNamespace

import fugal_subnet.config as config
from fugal_subnet.head_archive import archive_verified_heads
from neurons.validator import _archive_heads

ROOT = Path(__file__).resolve().parents[1]
HEAD_A = b"PK\x03\x04 head A bytes"
HEAD_B = b"PK\x03\x04 head B bytes"
SHA_A = hashlib.sha256(HEAD_A).hexdigest()
SHA_B = hashlib.sha256(HEAD_B).hexdigest()


def _rows(d):
    return [json.loads(x) for x in (Path(d) / "index.jsonl").read_text().splitlines()]


def test_writes_bytes_once_per_hash_with_private_modes(tmp_path):
    d = tmp_path / "heads"
    c = archive_verified_heads(
        str(d), "e1", {214: HEAD_A, 228: HEAD_B}, {214: SHA_A, 228: SHA_B},
        hotkeys={214: "5hotA", 228: "5hotB"}, coldkeys={214: "5coldA", 228: "5coldB"},
    )
    assert c == {"first_seen": 2, "repeat": 0, "skipped": 0}
    assert (d / f"{SHA_A}.npz").read_bytes() == HEAD_A
    assert (d / f"{SHA_B}.npz").read_bytes() == HEAD_B
    assert stat.S_IMODE(os.stat(d).st_mode) & 0o077 == 0
    assert stat.S_IMODE(os.stat(d / f"{SHA_A}.npz").st_mode) & 0o077 == 0
    assert stat.S_IMODE(os.stat(d / "index.jsonl").st_mode) & 0o077 == 0
    rows = _rows(d)
    assert [(r["uid"], r["hotkey"], r["coldkey"], r["weights_hash"], r["status"]) for r in rows] == [
        (214, "5hotA", "5coldA", SHA_A, "first_seen"),
        (228, "5hotB", "5coldB", SHA_B, "first_seen"),
    ]
    assert all(r["epoch_id"] == "e1" for r in rows)


def test_a_second_epoch_with_the_same_head_is_a_repeat(tmp_path):
    d = str(tmp_path)
    archive_verified_heads(d, "e1", {214: HEAD_A}, {214: SHA_A}, {}, {})
    mtime = os.stat(Path(d) / f"{SHA_A}.npz").st_mtime_ns
    c = archive_verified_heads(d, "e2", {214: HEAD_A, 203: HEAD_A}, {214: SHA_A, 203: SHA_A}, {}, {})
    assert c == {"first_seen": 0, "repeat": 2, "skipped": 0}
    assert os.stat(Path(d) / f"{SHA_A}.npz").st_mtime_ns == mtime
    assert [r["status"] for r in _rows(d)] == ["first_seen", "repeat", "repeat"]


def test_a_hash_the_bytes_do_not_match_is_skipped_and_never_a_path(tmp_path):
    d = tmp_path / "heads"
    c = archive_verified_heads(
        str(d), "e1", {1: HEAD_A, 2: HEAD_B}, {1: "../../etc/passwd", 2: SHA_A}, {}, {},
    )
    assert c == {"first_seen": 0, "repeat": 0, "skipped": 2}
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["heads"]


def test_unset_dir_does_nothing(tmp_path):
    assert archive_verified_heads("", "e1", {1: HEAD_A}, {1: SHA_A}, {}, {}) == {
        "first_seen": 0, "repeat": 0, "skipped": 0}
    assert list(tmp_path.iterdir()) == []


def test_the_archive_never_parses_a_head():
    tree = ast.parse((ROOT / "fugal_subnet" / "head_archive.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    # Stdlib for hashing, paths and JSON only: nothing that can open an npz.
    assert imported <= {"__future__", "hashlib", "json", "logging", "os", "re", "time"}


def test_a_disk_error_does_not_cost_the_epoch(tmp_path, monkeypatch):
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")
    monkeypatch.setattr(config, "HEAD_ARCHIVE_DIR", str(not_a_dir / "heads"))
    mg = SimpleNamespace(hotkeys=["h"] * 300, coldkeys=["c"] * 300)
    _archive_heads("e1", {214: HEAD_A}, {214: SHA_A}, mg)   # must not raise


def test_archive_is_wired_after_verification_and_feeds_nothing_back():
    src = (ROOT / "neurons" / "validator.py").read_text()
    calls = re.findall(r"^\s*(.*)_archive_heads\((.*)\)\s*$", src, re.M)
    assert calls == [("", "epoch_id, verified_heads, head_hashes, metagraph")]
    # Its result is discarded (no assignment), so no score, hash, reveal or
    # fingerprint input can depend on it.
    m = re.search(r"epoch_score_dicts = \{(.*?)\n            \}\n", src, re.S)
    assert m and "archive" not in m.group(1)
    for name in ("reveal_epoch(", "find_duplicates(", "update_scores(", "compute_weights("):
        i = src.index(name)
        assert "archive" not in src[i:src.index(")", i)]
