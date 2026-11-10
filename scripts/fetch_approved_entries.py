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

Requires `git` and network access; no API spend. The release note (a
mirror, cross-checked) is read with the GitHub CLI when it is installed and
from GitHub's public API otherwise; if neither answers, the signed message
alone decides, with a warning.
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


_TAG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def fetch_tag(tag: str, cwd: str = ROOT) -> None:
    """Fetch `tag` from origin, replacing a local copy that differs.

    A tag the team moved (a build tag later signed under the same name) made a
    plain `git fetch --tags` refuse with `would clobber existing tag`, so a
    validator that fetched early could not approve (Fable review, item 8).
    Taking the remote's tag is safe: its signature is verified after this,
    against the committed signers, and that is what is trusted."""
    if not _TAG_NAME.match(tag) or ".." in tag:
        raise SystemExit(f"{tag!r} is not a release tag name")
    run("git", "fetch", "--no-tags", "origin", f"+refs/tags/{tag}:refs/tags/{tag}", cwd=cwd)


def _api_release_body(owner: str, repo: str, tag: str) -> str:
    import json
    import urllib.request

    url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "fugal-fetch-approved-entries"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - fixed https URL
        return json.loads(r.read(1 << 20).decode("utf-8")).get("body") or ""


def release_notes(tag: str, cwd: str = ROOT) -> str:
    """The release note's body: through `gh` if installed, else the public
    API; "" (with a warning) if neither answers. The note is only a mirror."""
    try:
        return run("gh", "release", "view", tag, "--json", "body", "--jq", ".body", cwd=cwd)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        url = run("git", "remote", "get-url", "origin", cwd=cwd).strip()
        m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
        if not m:
            raise ValueError(f"origin {url!r} is not a GitHub repository")
        return _api_release_body(m.group(1), m.group(2), tag)
    except Exception as e:  # noqa: BLE001 - the note is a mirror; the signature decides
        print(f"# release note for {tag} unreadable ({e}); using the signed message alone",
              file=sys.stderr)
        return ""


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

    fetch_tag(args.tag)
    if args.no_verify:
        print(f"# tag {args.tag}: signature NOT verified (--no-verify); do not approve from this",
              file=sys.stderr)
    else:
        verify_tag(args.tag)
        print(f"# tag {args.tag}: signature verified against deploy/keys/allowed_signers",
              file=sys.stderr)
    message = run("git", "tag", "-l", "--format=%(contents)", args.tag, cwd=ROOT)
    notes = release_notes(args.tag)
    print(f"FUGAL_TEE_MEASUREMENTS={select(args.tag, args.network, message, notes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
