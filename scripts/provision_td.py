#!/usr/bin/env python3
"""Provision a running TD miner over the attested channel — the operator side.

    python scripts/provision_td.py --inspect --address http://<td-ip>:8092

    python scripts/provision_td.py --address http://<td-ip>:8092 \
        --approved <base_measurement>:<compose_hash> --instance-id <hex> \
        --head data/rehearsal/head_cheap_a.npz \
        --wallet fugal_owner2 --hotkey td1 \
        --api-key-file ~/.fugal/openrouter.key

`--inspect` asks the TD for a fresh quote over a nonce and prints what it IS —
base measurement, compose hash, instance id, every event in its RTMR3 log — and
pushes nothing. It is how the operator learns the instance id to pin, and it is
safe to run against anything: a quote is Intel-signed public evidence, not a
secret.

The push refuses before sending unless the TD's quote is over OUR nonce, its
base measurement is the approved one, its event log replays to the signed
RTMR3, and the log names the approved compose hash and the instance WE created
(`fugal_subnet.tee.provision.push`, in that order). Only then do the head, the
hotkey keyfile, coldkeypub and the API key cross.

SECRETS NEVER APPEAR ON THE COMMAND LINE. The API key is read from a file (mode
0600, or the run refuses), the wallet from the bittensor wallet directory, and
nothing this script prints contains either — field names and byte counts only.
An encrypted hotkey is refused: the TD cannot answer a password prompt, and a
miner that hangs at "Enter your password" inside a TD looks exactly like a
healthy miner that has not been provisioned yet.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _read_private(path: str, what: str) -> bytes:
    path = os.path.expanduser(path)
    st = os.stat(path)
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise SystemExit(
            f"{what} at {path} is readable by group/other "
            f"(mode {oct(stat.S_IMODE(st.st_mode))}); chmod 600 it first"
        )
    with open(path, "rb") as f:
        return f.read()


def _hotkey_files(wallet_root: str, wallet: str, hotkey: str) -> tuple[bytes, bytes, str]:
    root = os.path.expanduser(wallet_root or "~/.bittensor/wallets")
    hot_path = os.path.join(root, wallet, "hotkeys", hotkey)
    pub_path = os.path.join(root, wallet, "coldkeypub.txt")
    hot = _read_private(hot_path, "hotkey keyfile")
    if hot.startswith(b"$"):
        raise SystemExit(
            f"hotkey {wallet}/{hotkey} is encrypted; the TD cannot prompt for a "
            "password. Use an unencrypted hotkey (hotkeys are routinely stored "
            "plaintext; the coldkey is what must stay encrypted and off every host)."
        )
    try:
        ss58 = json.loads(hot)["ss58Address"]
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"hotkey keyfile is not a bittensor keyfile JSON: {type(e).__name__}") from e
    if not os.path.exists(pub_path):
        raise SystemExit(f"{pub_path} missing — the SDK needs coldkeypub.txt to serve an axon")
    with open(pub_path, "rb") as f:
        pub = f.read()
    return hot, pub, ss58


def inspect(address: str, timeout: int) -> int:
    import urllib.request

    from fugal_subnet.tee.attestation import (
        measurement_id,
        parse_quote,
        replay_event_log,
    )
    from fugal_subnet.tee.provision import new_nonce
    from fugal_subnet.tee.verify import unwrap_attestation

    nonce = new_nonce()
    req = urllib.request.Request(
        f"{address.rstrip('/')}/provision/attest",
        data=json.dumps({"nonce": nonce}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        blob = bytes.fromhex(json.loads(r.read())["attestation"])
    quote_bytes, event_log = unwrap_attestation(blob)
    quote = parse_quote(quote_bytes)
    # The guest right-pads report_data with zeros (measured; see INVARIANTS I8).
    expected_rd = bytes.fromhex(nonce).ljust(64, b"\0").hex()
    print(f"attestation        {len(blob)} bytes, quote {len(quote_bytes)} bytes")
    print(f"nonce honoured     {quote.report_data == expected_rd}")
    print(f"base_measurement   {measurement_id(quote)}")
    print(f"mrtd/rtmr0-3       {quote.mrtd[:16]}… {quote.rtmr0[:16]}… "
          f"{quote.rtmr1[:16]}… {quote.rtmr2[:16]}… {quote.rtmr3[:16]}…")
    if not event_log:
        print("event log          (none) — a bare quote; no compose hash to pin")
        return 0
    replayed, events = replay_event_log(event_log)
    print(f"log replays RTMR3  {replayed == quote.rtmr3}")
    for name, payload in events.items():
        shown = payload.hex() if len(payload) <= 64 else payload[:48].hex() + "…"
        print(f"  event {name:<20} {shown}")
    print(f"compose_hash       {events.get('compose-hash', b'').hex() or '(none)'}")
    print(f"instance_id        {events.get('instance-id', b'').hex() or '(none)'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", required=True, help="http://<td-ip>:8092")
    ap.add_argument("--inspect", action="store_true",
                    help="Print what the TD attests to and push nothing")
    ap.add_argument("--approved", default="",
                    help="<base_measurement>:<compose_hash> the TD must prove")
    ap.add_argument("--instance-id", default="",
                    help="The instance-id WE created (from --inspect or dstack-cloud)")
    ap.add_argument("--allow-any-instance", action="store_true",
                    help="Skip the instance-id check (a correctly imaged TD someone "
                         "else booted could then be handed the key — say so deliberately)")
    ap.add_argument("--head", help="Path to the .npz head to push")
    ap.add_argument("--wallet", help="Wallet (coldkey) name whose hotkey to push")
    ap.add_argument("--hotkey", default="default")
    ap.add_argument("--wallet-path", default="", help="Bittensor wallet root")
    ap.add_argument("--api-key-file", help="0600 file holding the OpenRouter key")
    ap.add_argument("--no-api-key", action="store_true",
                    help="Push without an API key (stub-upstream rehearsals only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate everything locally and print field names; contact nothing")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    from fugal_subnet.logging_setup import configure_logging
    configure_logging("INFO")

    if args.inspect:
        return inspect(args.address, args.timeout)

    missing = [n for n, v in (("--approved", args.approved), ("--head", args.head),
                              ("--wallet", args.wallet)) if not v]
    if missing:
        raise SystemExit(f"required for a push: {', '.join(missing)}")
    if not args.instance_id and not args.allow_any_instance:
        raise SystemExit("--instance-id is required (get it from --inspect), or pass "
                         "--allow-any-instance and mean it")
    base, sep, app = args.approved.partition(":")
    if not sep or not app or len(app) != 64:
        raise SystemExit("--approved must be <base_measurement>:<compose_hash> with a "
                         "64-hex compose hash; a bare base would push to ANY app on the image")

    from fugal_subnet.head_eval import load_head_from_npz
    from fugal_subnet.tee.provision import push, validate_payload

    with open(args.head, "rb") as f:
        head = f.read()
    load_head_from_npz(head)  # refuse to push a head the validator would reject

    hot, pub, ss58 = _hotkey_files(args.wallet_path, args.wallet, args.hotkey)

    payload = {
        "head_b64": base64.b64encode(head).decode(),
        "hotkey_ss58": ss58,
        "hotkey_keyfile_b64": base64.b64encode(hot).decode(),
        "coldkeypub_b64": base64.b64encode(pub).decode(),
    }
    if not args.no_api_key:
        if not args.api_key_file:
            raise SystemExit("--api-key-file is required (or --no-api-key)")
        key = _read_private(args.api_key_file, "API key file").decode().strip()
        if not key:
            raise SystemExit("API key file is empty")
        payload["openrouter_api_key"] = key
    validate_payload(payload)

    print(f"head               {len(head)} bytes sha256={hashlib.sha256(head).hexdigest()}")
    print(f"hotkey             {args.wallet}/{args.hotkey} = {ss58}")
    print(f"approved entry     {base[:16]}…:{app[:16]}…")
    print(f"instance_id        {args.instance_id or '(ANY — --allow-any-instance)'}")
    print(f"fields             {sorted(payload)}")
    if args.dry_run:
        print("dry run: nothing sent")
        return 0

    push(
        args.address, payload,
        approved_measurements={base},
        expected_app_identity=app,
        expected_instance_id=args.instance_id,
        timeout=args.timeout,
    )
    print(f"PUSHED to {args.address}: {sorted(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
