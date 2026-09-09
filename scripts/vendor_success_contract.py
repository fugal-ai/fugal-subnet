#!/usr/bin/env python3
"""Update the minimal runtime vendor and immutable test-only core projection.

Run only for a reviewed exact core commit, then inspect the diff and run CI.
This reads Git objects from a local checkout; it never needs a CI access token.
"""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PATHS = ["fugal/__init__.py", "fugal/router.py", "fugal/success_contract.py",
         "data/success_conformance.json", "verify/verify_success_contract.py",
         "scripts/fetch_backbone.py", "LICENSE", "NOTICE"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", required=True)
    p.add_argument("--commit", required=True)
    args = p.parse_args()

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=args.core)

    commit = git("rev-parse", args.commit + "^{commit}").decode().strip()
    if commit != args.commit:
        p.error("--commit must be an exact full commit ID")
    vendor = ROOT / "fugal_subnet/vendor"
    source = {"repository": "https://github.com/fugal-ai/fugal-core", "commit": commit,
              "files": {}, "test_snapshot": {}}
    content = {}
    for name in PATHS:
        data = git("show", commit + ":" + name)
        content[name] = data
        destination = ROOT / "tests/core_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        source["test_snapshot"][name] = hashlib.sha256(data).hexdigest()
    for name, target in (("fugal/success_contract.py", "success_contract.py"),
                         ("data/success_conformance.json", "conformance.json")):
        (vendor / target).write_bytes(content[name])
        source["files"][name] = {"vendored": target, "sha256": hashlib.sha256(content[name]).hexdigest()}
    (vendor / "LICENSE").write_bytes(content["LICENSE"])
    parents = {""}
    for name in PATHS:
        parent = PurePosixPath(name).parent
        while str(parent) != ".":
            parents.add(str(parent))
            parent = parent.parent
    proof = {"commit": base64.b64encode(git("cat-file", "commit", commit)).decode(),
             "trees": {name: base64.b64encode(git("cat-file", "tree", commit + (":" + name if name else "^{tree}"))).decode()
                       for name in sorted(parents)}}
    (vendor / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    (vendor / "source_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(f"Vendored core {commit}; run scripts/check_success_vendor.py --core {args.core}")


if __name__ == "__main__":
    main()
