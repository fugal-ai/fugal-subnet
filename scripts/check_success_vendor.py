#!/usr/bin/env python3
"""Verify exact core Git provenance without cross-private-repository credentials.

The commit object and required tree objects form a Git Merkle proof. Recompute
Git object IDs through commit -> directory trees -> file blobs. This verifies
both runtime vendor files and the small test-only serving snapshot at SOURCE's
exact commit, without trusting a second copy of a SHA256 list. --core additionally
checks an available local checkout against that same immutable commit.
"""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def object_id(kind, data):
    return hashlib.sha1(f"{kind} {len(data)}\0".encode() + data).hexdigest()


def tree_entries(data):
    result = {}
    while data:
        header, data = data.split(b"\0", 1)
        _, name = header.split(b" ", 1)
        result[name.decode()] = data[:20].hex()
        data = data[20:]
    return result


def verify_source(source, proof, file_data):
    commit = base64.b64decode(proof["commit"], validate=True)
    assert object_id("commit", commit) == source["commit"], "core commit object mismatch"
    trees = {name: base64.b64decode(value, validate=True) for name, value in proof["trees"].items()}
    root_oid = commit.splitlines()[0].decode().removeprefix("tree ")
    assert object_id("tree", trees[""]) == root_oid, "core root tree mismatch"
    for name, data in trees.items():
        if not name:
            continue
        path = PurePosixPath(name)
        parent = "" if str(path.parent) == "." else str(path.parent)
        expected = tree_entries(trees[parent])[path.name]
        assert object_id("tree", data) == expected, f"core directory tree mismatch: {name}"
    for name, data in file_data.items():
        path = PurePosixPath(name)
        parent = "" if str(path.parent) == "." else str(path.parent)
        expected = tree_entries(trees[parent])[path.name]
        assert object_id("blob", data) == expected, f"core source blob mismatch: {name}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", help="Optional real core checkout containing the pinned commit")
    p.add_argument("--materialize-core", action="store_true", help="CI only: materialize proven Git objects for exact commit reporting")
    args = p.parse_args()
    vendor = ROOT / "fugal_subnet/vendor"
    source = json.loads((vendor / "SOURCE.json").read_text())
    data = {}
    for name, spec in source["files"].items():
        data[name] = (vendor / spec["vendored"]).read_bytes()
        assert hashlib.sha256(data[name]).hexdigest() == spec["sha256"], name
    for name, expected in source["test_snapshot"].items():
        snapshot = (ROOT / "tests/core_snapshot" / name).read_bytes()
        assert hashlib.sha256(snapshot).hexdigest() == expected, name
        if name in data:
            assert data[name] == snapshot, name
        data[name] = snapshot
    proof = json.loads((vendor / "source_proof.json").read_text())
    verify_source(source, proof, data)
    if args.materialize_core:
        snapshot = ROOT / "tests/core_snapshot"
        subprocess.run(["git", "init", "--quiet", str(snapshot)], check=True)
        objects = [("commit", base64.b64decode(proof["commit"]))]
        objects += [("tree", base64.b64decode(value)) for value in proof["trees"].values()]
        objects += [("blob", content) for content in data.values()]
        for kind, content in objects:
            subprocess.run(["git", "hash-object", "-w", "-t", kind, "--stdin"],
                           cwd=snapshot, input=content, stdout=subprocess.DEVNULL, check=True)
        subprocess.run(["git", "update-ref", "--no-deref", "HEAD", source["commit"]], cwd=snapshot, check=True)
    if args.core:
        for name, content in data.items():
            expected = subprocess.check_output(["git", "show", source["commit"] + ":" + name], cwd=args.core)
            assert content == expected, name
    for name in ("models.json", "benchmark_tokens_v1.json"):
        assert (ROOT / "data" / name).read_bytes() == (ROOT / "fugal_subnet/routing_data" / name).read_bytes(), name
    print(f"Success contract and serving test snapshot proven at core {source['commit']}")


if __name__ == "__main__":
    main()
