#!/usr/bin/env python3
"""Write the signed tag message and the release note for an image's approved entries.

An approved entry is a consensus parameter: `<base>:<compose_hash>` is what
every --live validator accepts, one per network (finney/70 and test/552 freeze
different composes). docs/APPROVED_ENTRY_ROTATION.md publishes them as a
signed `image-v*` release. The ENTRIES LIVE IN THE SIGNED TAG MESSAGE, one
`approved-entry <network>: <entry>` line each; that is what
scripts/fetch_approved_entries.py reads after verifying the signature. The
release note mirrors them for people, and fetch refuses a note that disagrees.
This script writes both; the operator signs and publishes:

    python scripts/publish_approved_entry.py --digest ghcr.io/...@sha256:... \\
        --commit <sha> --base <base> --tag image-v<date> \\
        --compose finney=deploy/dstack/app-compose.frozen-<tag>-finney.json \\
        --compose test=deploy/dstack/app-compose.frozen-<tag>-test.json \\
        --tag-message tag-message.txt > notes.md
    git tag -s image-v<date> -F tag-message.txt
    git push origin image-v<date>
    gh release create image-v<date> --notes-file notes.md --verify-tag

A `--compose` value is `<network>=<path>`, or a bare path whose file name says
the network (`-test` / `.test.` means test, anything else finney). Each frozen
compose must embed a live deploy/dstack/docker-compose*.yaml (review F-074).

The CI image workflow drafts a note from the docker-compose.yaml without
--base (it does not know the platform's measurement): no entries, nothing to
sign. The operator completes the release after `dstack-cloud prepare` has
produced the frozen app-compose.json files, whose RAW bytes are the hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NETWORKS = ("finney", "test")
# One line per network in the signed tag message. fetch_approved_entries parses
# exactly this; change both together (tests/test_approved_entry_channel.py).
TAG_LINE = "approved-entry {network}: {entry}"


def network_of(spec: str) -> tuple[str, str]:
    """`<network>=<path>`, or a path whose file name names the network."""
    if "=" in spec:
        network, path = spec.split("=", 1)
    else:
        path = spec
        network = "test" if re.search(r"[-.]test[-.]", os.path.basename(path)) else "finney"
    if network not in NETWORKS:
        raise SystemExit(f"unknown network {network!r} in --compose {spec!r} (use {'/'.join(NETWORKS)})")
    return network, path


def frozen_compose_hash(path: str, allow_drift: bool) -> str:
    """sha256 of a frozen app-compose.json's raw bytes; '' for a docker-compose.yaml."""
    with open(path, "rb") as f:
        raw = f.read()
    if not path.endswith(".json"):
        return ""
    if not allow_drift:
        # The frozen file must embed one of the live composes, or the entry
        # names a compose nobody deploys from this checkout (review F-074).
        embedded = json.loads(raw.decode("utf-8")).get("docker_compose_file", "")
        live_dir = os.path.join(ROOT, "deploy", "dstack")
        live = []
        for f in os.listdir(live_dir):
            if f.startswith("docker-compose") and f.endswith(".yaml"):
                with open(os.path.join(live_dir, f), encoding="utf-8") as fh:
                    live.append(fh.read())
        if embedded not in live:
            raise SystemExit(f"{path} does not embed any live deploy/dstack/docker-compose*.yaml; "
                             "freeze the compose you deploy (or pass --allow-drift to publish anyway)")
    return hashlib.sha256(raw).hexdigest()


def tag_message(tag: str, entries: dict[str, str], digest: str, commit: str) -> str:
    lines = [f"Miner image {tag}".rstrip(), ""]
    lines += [TAG_LINE.format(network=n, entry=entries[n]) for n in NETWORKS if n in entries]
    lines += ["", f"image: {digest}", f"commit: {commit}", ""]
    return "\n".join(lines)


def release_note(tag: str, entries: dict[str, str], files: dict[str, str], base: str,
                 digest: str, commit: str) -> str:
    out = [f"# Approved entries {tag}".rstrip(), "",
           "| network | entry | compose file |", "|---|---|---|"]
    for n in NETWORKS:
        if n in files:
            entry = f"`{entries[n]}`" if n in entries else "(complete after prepare)"
            out.append(f"| {n} | {entry} | `{os.path.basename(files[n])}` |")
    out += ["", "| field | value |", "|---|---|",
            f"| base measurement | `{base or 'to be filled'}` |",
            f"| image digest | `{digest}` |",
            f"| commit | `{commit}` |",
            f"| published | {date.today().isoformat()} |", "",
            "The entries are the `approved-entry` lines of the signed tag message; this "
            "table mirrors them. Validators: `python scripts/fetch_approved_entries.py "
            "--tag <tag> --network <finney|test>` verifies the signature and prints your "
            "network's line. Add it to `FUGAL_TEE_MEASUREMENTS` alongside the current "
            "entry, restart, and retire the old one per docs/APPROVED_ENTRY_ROTATION.md."]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--digest", required=True, help="image reference by digest")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--compose", action="append", required=True,
                    help="[<network>=]<frozen app-compose.json> (hashed raw), or a "
                         "docker-compose.yaml (listed only); repeat per network")
    ap.add_argument("--base", default="", help="base measurement of the dstack image on the platform")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tag-message", default="",
                    help="write the message to sign (`git tag -s <tag> -F <file>`) here")
    ap.add_argument("--allow-drift", action="store_true",
                    help="publish a frozen compose that no live docker-compose*.yaml matches")
    args = ap.parse_args()

    files: dict[str, str] = {}
    for spec in args.compose:
        network, path = network_of(spec)
        if network in files:
            raise SystemExit(f"two --compose files for {network}: {files[network]} and {path}")
        files[network] = path
    entries = {}
    for network, path in files.items():
        compose_hash = frozen_compose_hash(path, args.allow_drift)
        if args.base and compose_hash:
            entries[network] = f"{args.base}:{compose_hash}"

    if args.tag_message:
        if not entries:
            raise SystemExit("--tag-message needs --base and at least one frozen app-compose.json: "
                             "a tag without approved-entry lines approves nothing")
        with open(args.tag_message, "w", encoding="utf-8") as f:
            f.write(tag_message(args.tag, entries, args.digest, args.commit))
        print(f"tag message written to {args.tag_message}", file=sys.stderr)
    sys.stdout.write(release_note(args.tag, entries, files, args.base, args.digest, args.commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
