"""The approved-entry channel: publish writes it, fetch reads back exactly that.

The entries travel in the signed tag message, one `approved-entry <network>:`
line per network; the release note mirrors them. These tests pin the contract
between scripts/publish_approved_entry.py and scripts/fetch_approved_entries.py
without git or GitHub: the formats round-trip for both networks, fetch never
takes an entry the tag does not sign, and a note that disagrees is refused.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "f4be9383e31c6cc87707b2b01ace9114ed958435b524425677e63d65065aa427"
FINNEY = "9ab870cfe60d0c61f5764966a177d9146276979e65df0dbaa8370dcd3a75f166"
TEST = "c9b5c43b57d540d0e8a321efb673b6c42d1effd011dd716934f5df2a6ff0dad8"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pub, fetch = _load("publish_approved_entry"), _load("fetch_approved_entries")


def _published(entries):
    files = {n: f"app-compose.frozen-x-{n}.json" for n in entries}
    return (pub.tag_message("image-vX", entries, "ghcr.io/x@sha256:1", "abc"),
            pub.release_note("image-vX", entries, files, BASE, "ghcr.io/x@sha256:1", "abc"))


def test_both_networks_round_trip_through_the_signed_message():
    entries = {"finney": f"{BASE}:{FINNEY}", "test": f"{BASE}:{TEST}"}
    message, note = _published(entries)
    for network, entry in entries.items():
        assert fetch.select("image-vX", network, message, note) == entry


def test_an_edited_release_note_is_refused():
    """The note is not signed; a note that disagrees with the tag is refused."""
    message, note = _published({"finney": f"{BASE}:{FINNEY}"})
    forged = note.replace(FINNEY, "0" * 64)
    with pytest.raises(SystemExit, match="differs from the signed one"):
        fetch.select("image-vX", "finney", message, forged)


def test_a_tag_that_signs_no_entry_is_refused():
    """image-v2026.11.03c's signed message, verbatim apart from the signature:
    its entry existed only in the release note."""
    message = ("Miner image 2026-11-03c: proxy retries on 429/5xx, operator errors named, "
               "canonical result types\n-----BEGIN SSH SIGNATURE-----\n...\n")
    note = f"| entry | `{BASE}:{FINNEY}` |\n"
    with pytest.raises(SystemExit, match="carries no approved-entry lines"):
        fetch.select("image-v2026.11.03c", "finney", message, note)


def test_a_network_the_tag_does_not_sign_is_refused():
    message, note = _published({"finney": f"{BASE}:{FINNEY}"})
    with pytest.raises(SystemExit, match="signs no entry for test"):
        fetch.select("image-vX", "test", message, note)


def test_the_network_comes_from_the_flag_or_the_file_name():
    assert pub.network_of("test=a.json") == ("test", "a.json")
    assert pub.network_of("deploy/dstack/app-compose.frozen-2026-11-03c-test.json")[0] == "test"
    assert pub.network_of("deploy/dstack/app-compose.frozen-2026-11-03c-finney.json")[0] == "finney"
    assert pub.network_of("out/docker-compose.test.yaml")[0] == "test"
    assert pub.network_of("out/docker-compose.yaml")[0] == "finney"
    with pytest.raises(SystemExit):
        pub.network_of("mainnet=a.json")


def test_the_real_03c_files_publish_the_entries_validators_run(tmp_path, monkeypatch, capsys):
    """End to end on the committed frozen files: the signed lines are the
    published 03c entries (finney 9ab870cf…, test c9b5c43b…)."""
    msg = tmp_path / "tag.txt"
    monkeypatch.setattr(sys, "argv", [
        "x", "--digest", "ghcr.io/x@sha256:1", "--commit", "abc", "--base", BASE,
        "--tag", "image-vX", "--allow-drift", "--tag-message", str(msg),
        "--compose", str(ROOT / "deploy/dstack/app-compose.frozen-2026-11-03c-finney.json"),
        "--compose", str(ROOT / "deploy/dstack/app-compose.frozen-2026-11-03c-test.json")])
    assert pub.main() == 0
    note = capsys.readouterr().out
    message = msg.read_text(encoding="utf-8")
    assert fetch.select("image-vX", "finney", message, note) == f"{BASE}:{FINNEY}"
    assert fetch.select("image-vX", "test", message, note) == f"{BASE}:{TEST}"
