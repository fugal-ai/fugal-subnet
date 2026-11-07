#!/usr/bin/env python3
"""Verify a signed release tag and print the approved entry it carries for one network.

The validator side of docs/APPROVED_ENTRY_ROTATION.md's publication channel
(diligence 2026-10, finding 7). An approved entry is only evidence if the
operator can trace it to a signed statement by the team: this fetches the
tag, verifies its signature against the allowed signers in deploy/keys/
(SSH or GPG public keys committed to the repository, so the trust root is in
the same reviewed history as the code), and reads the entry from the SIGNED
TAG MESSAGE (`approved-entry <network>: <entry>` lines, written by
scripts/publish_approved_entry.py). The release note is a mirror for people:
if it names a different entry for the network, this refuses. A tag whose
message carries no entry lines (image-v2026.11.03b and 03c, published before this format) is
refused: its entry is not part of what was signed.

It never writes to the validator's environment: that step is the operator's,
deliberately.

    python scripts/fetch_approved_entries.py --tag image-v<date> --network finney
    # -> FUGAL_TEE_MEASUREMENTS=<base>:<compose_hash>

Requires `git` and `gh` (the GitHub CLI) and network access; no API spend.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = os.path.join(ROOT, "deploy", "keys")
_ENTRY = r"[0-9a-f]{64}:[0-9a-f]{64}"
# Mirrors publish_approved_entry.TAG_LINE.
TAG_LINE_RE = re.compile(rf"^approved-entry (finney|test): ({_ENTRY})$", re.M)
NOTE_ROW_RE = re.compile(rf"^\| (finney|test) \| `({_ENTRY})` \|", re.M)


def run(*cmd, **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120, **kw).stdout


def verify_tag(tag: str) -> None:
    """git tag -v, with the committed allowed-signers as the only trust root."""
    signers = os.path.join(KEYS, "allowed_signers")
    if not os.path.exists(signers):
        raise SystemExit(f"{signers} is missing: no trust root to verify {tag} against")
    run("git", "-c", f"gpg.ssh.allowedSignersFile={signers}", "tag", "-v", tag, cwd=ROOT)


def entries(text: str, pattern: re.Pattern) -> dict[str, str]:
    """{network: entry}; a network named twice with different entries is an error."""
    out: dict[str, str] = {}
    for network, entry in pattern.findall(text):
        if out.get(network, entry) != entry:
            raise SystemExit(f"two different entries for {network}: {out[network]} and {entry}")
        out[network] = entry
    return out


def select(tag: str, network: str, tag_message: str, notes: str) -> str:
    """The entry for `network` from the signed tag message, cross-checked against the note."""
    signed = entries(tag_message, TAG_LINE_RE)
    if not signed:
        raise SystemExit(
            f"{tag}'s signed tag message carries no approved-entry lines: its entry is only in "
            "the release note, which the signature does not cover. Not approvable from this "
            "tool; keep the entry you already run, or use a release from "
            "scripts/publish_approved_entry.py (docs/APPROVED_ENTRY_ROTATION.md)."
        )
    if network not in signed:
        raise SystemExit(f"{tag} signs no entry for {network} (it has: {', '.join(sorted(signed))})")
    mirrored = entries(notes, NOTE_ROW_RE).get(network)
    if mirrored is not None and mirrored != signed[network]:
        raise SystemExit(f"{tag}: the release note's {network} entry {mirrored} differs from the "
                         f"signed one {signed[network]}; refusing both")
    return signed[network]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--network", required=True, choices=("finney", "test"),
                    help="which network's entry to print (finney/70 or test/552)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the signature check (reading only; never for approval)")
    args = ap.parse_args()

    run("git", "fetch", "--tags", "origin", args.tag, cwd=ROOT)
    if args.no_verify:
        print(f"# tag {args.tag}: signature NOT verified (--no-verify); do not approve from this",
              file=sys.stderr)
    else:
        verify_tag(args.tag)
        print(f"# tag {args.tag}: signature verified against deploy/keys/allowed_signers",
              file=sys.stderr)
    message = run("git", "tag", "-l", "--format=%(contents)", args.tag, cwd=ROOT)
    notes = run("gh", "release", "view", args.tag, "--json", "body", "--jq", ".body", cwd=ROOT)
    print(f"FUGAL_TEE_MEASUREMENTS={select(args.tag, args.network, message, notes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
