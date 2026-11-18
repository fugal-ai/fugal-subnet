"""Validator upgrades are "update + restart" (fugal_subnet/validator_setup.py)."""
import ast
import hashlib
import os
from pathlib import Path

import pytest

from fugal_subnet import validator_setup as vs

ROOT = Path(__file__).resolve().parents[1]
BASE = "f4be9383e31c6cc87707b2b01ace9114ed958435b524425677e63d65065aa427"


def _hash(name: str) -> str:
    return hashlib.sha256((ROOT / "deploy" / "dstack" / name).read_bytes()).hexdigest()


def test_finney_approves_every_current_frozen_compose_from_the_code():
    entries = dict(vs.frozen_compose_entries("finney"))
    assert f"{BASE}:{_hash('app-compose.frozen-2026-11-17b-finney.json')}" in entries
    assert all(name.endswith("-finney.json") for name in entries.values())


def test_retired_protocol_images_are_not_approved_from_the_code():
    """v1 (2026-11-03c) and v2 (2026-11-10b) images cannot prove the protocol in
    force, so the approved set holds only images that can score."""
    entries = dict(vs.frozen_compose_entries("finney"))
    for old in ("2026-11-03c", "2026-11-10b"):
        for net in ("finney", "test"):
            h = _hash(f"app-compose.frozen-{old}-{net}.json")
            assert h in vs.SUPERSEDED_COMPOSES
            assert f"{BASE}:{h}" not in entries


def test_a_superseded_compose_is_never_approved_from_the_code():
    bad = _hash("app-compose.frozen-2026-11-17-finney.json")
    assert bad in vs.SUPERSEDED_COMPOSES
    assert not any(e.endswith(bad) for e, _ in vs.frozen_compose_entries("finney"))
    bad_test = _hash("app-compose.frozen-2026-11-17-test.json")
    assert bad_test in vs.SUPERSEDED_COMPOSES


def test_the_newest_finney_compose_is_v060_and_the_v050_one_stays_approved():
    names = [n for _, n in vs.frozen_compose_entries("finney")]
    assert names[-1] == "app-compose.frozen-2026-11-17c-finney.json"
    # v0.5.0 miners (image-v2026.11.17b) stay approved until they re-provision.
    assert "app-compose.frozen-2026-11-17b-finney.json" in names


def test_testnet_uses_the_test_composes_and_a_local_chain_none():
    assert all(n.endswith("-test.json") for _, n in vs.frozen_compose_entries("test"))
    assert vs.frozen_compose_entries("ws://127.0.0.1:9944") == []
    assert vs.frozen_compose_entries("local") == []


def test_env_entries_are_kept_and_the_code_adds_what_is_missing():
    old_v2 = f"{BASE}:{_hash('app-compose.frozen-2026-11-10b-finney.json')}"
    extra = f"{BASE}:{'ab' * 32}"
    entries, added = vs.resolve_approved([old_v2, extra], "finney", from_code=True)
    assert entries[:2] == [old_v2, extra]
    assert f"{BASE}:{_hash('app-compose.frozen-2026-11-17b-finney.json')}" in entries
    assert old_v2 not in [e for e, _ in added]
    assert len(entries) == len(set(entries))


def test_opting_out_uses_the_env_alone(monkeypatch):
    monkeypatch.setenv("FUGAL_APPROVED_FROM_CODE", "0")
    entries, added = vs.resolve_approved(["x:y"], "finney")
    assert entries == ["x:y"] and added == []


def test_every_code_entry_parses_as_a_live_entry():
    from fugal_subnet.tee.verify import parse_approved

    entries = [e for e, _ in vs.frozen_compose_entries("finney")]
    parsed = parse_approved(entries, require_app=True)
    assert set(parsed) == {BASE}


def test_finney_defaults_the_archive_and_fallback_endpoints():
    a, f, changed = vs.default_endpoints("finney", [], [])
    assert a == [vs.FINNEY_DEFAULT_ARCHIVE] and f == [vs.FINNEY_DEFAULT_FALLBACK] and changed
    a, f, changed = vs.default_endpoints("finney", ["wss://mine"], ["wss://fb"])
    assert a == ["wss://mine"] and f == ["wss://fb"] and not changed
    a, f, changed = vs.default_endpoints("test", [], [])
    assert a == [] and f == [] and not changed


def test_prepare_updates_the_running_config_in_place(monkeypatch):
    from fugal_subnet import config

    monkeypatch.setattr(config, "TEE_APPROVED_MEASUREMENTS", [])
    monkeypatch.setattr(config, "ARCHIVE_ENDPOINTS", [])
    monkeypatch.setattr(config, "FALLBACK_ENDPOINTS", [])
    same_list = config.TEE_APPROVED_MEASUREMENTS
    summary = vs.prepare("finney")
    assert same_list is config.TEE_APPROVED_MEASUREMENTS and same_list
    assert config.ARCHIVE_ENDPOINTS == [vs.FINNEY_DEFAULT_ARCHIVE]
    assert summary["newest"] == "app-compose.frozen-2026-11-17c-finney.json"
    assert summary["entries"] == len(same_list) and summary["archive"]


def test_prepare_leaves_the_consensus_digest_unchanged(monkeypatch):
    from fugal_subnet import config
    from fugal_subnet.fingerprint import consensus_digest

    monkeypatch.setattr(config, "TEE_APPROVED_MEASUREMENTS", [])
    monkeypatch.setattr(config, "ARCHIVE_ENDPOINTS", [])
    monkeypatch.setattr(config, "FALLBACK_ENDPOINTS", [])
    before = consensus_digest()
    vs.prepare("finney")
    assert consensus_digest() == before


def test_embeddings_are_kept_when_they_match_and_fetched_when_missing(tmp_path):
    good = tmp_path / "e.npz"
    good.write_bytes(b"pinned bytes")
    pin = hashlib.sha256(b"pinned bytes").hexdigest()
    calls = []
    note = vs.ensure_embeddings(str(good), pin, "https://x", fetch=lambda *a: calls.append(a))
    assert calls == [] and "matches" in note
    missing = tmp_path / "sub" / "m.npz"
    vs.ensure_embeddings(str(missing), pin, "https://x", fetch=lambda *a: calls.append(a))
    assert calls == [("https://x", pin, str(missing))]
    wrong = tmp_path / "w.npz"
    wrong.write_bytes(b"other")
    vs.ensure_embeddings(str(wrong), pin, "https://x", fetch=lambda *a: calls.append(a))
    assert len(calls) == 2


def test_embeddings_without_a_url_fail_clearly_and_an_unpinned_pool_skips(tmp_path):
    with pytest.raises(ValueError, match="EMBEDDINGS_URL"):
        vs.ensure_embeddings(str(tmp_path / "x.npz"), "ab" * 32, None)
    assert "unpinned" in vs.ensure_embeddings(str(tmp_path / "x.npz"), None, None)


def test_the_ready_line_says_ready_or_names_each_fix():
    ok, line = vs.ready_line(entries=3, newest="app-compose.frozen-2026-11-17b-finney.json",
                             v3="ok:48320", archive=True, collateral=True, role="ROOT")
    assert ok and line.startswith("READY: approved entries 3 (newest app-compose.frozen-2026-11-17b")
    assert "v3 inputs ok (pool 48320, embeddings ok)" in line and "role ROOT" in line
    ok, line = vs.ready_line(entries=0, newest="", v3="failed:embeddings missing",
                             archive=False, collateral=False, role="follower")
    assert not ok and line.startswith("NOT READY: ")
    for fix in ("git pull", "embeddings_file fetch", "FUGAL_ARCHIVE_ENDPOINTS", "FUGAL_PCCS_URL"):
        assert fix in line


def _main_calls() -> list[str]:
    src = (ROOT / "neurons" / "validator.py").read_text()
    main = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    names = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            f = node.func
            names.append(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return names


def test_the_validator_prepares_before_checking_and_logs_ready():
    src = (ROOT / "neurons" / "validator.py").read_text()
    assert src.index("validator_setup.prepare(") < src.index("_check_live_configuration(\n        live")
    calls = _main_calls()
    assert "prepare" in calls and "ensure_embeddings" in calls and "ready_line" in calls


def test_the_checklist_and_guide_say_update_and_restart():
    guide = (ROOT / "docs" / "VALIDATOR_GUIDE.md").read_text()
    assert "READY:" in guide
    assert os.path.exists(ROOT / "fugal_subnet" / "validator_setup.py")


# --- keyed on the chain's genesis, not the --network string (v0.6.0 review, finding 3) ---

def test_a_finney_validator_on_its_own_node_gets_the_release_entries(monkeypatch):
    from fugal_subnet import config
    url = "ws://127.0.0.1:9944"
    assert vs.network_for_genesis(config.FINNEY_GENESIS_HASH, url) == "finney"
    monkeypatch.setattr(config, "TEE_APPROVED_MEASUREMENTS", [])
    monkeypatch.setattr(config, "ARCHIVE_ENDPOINTS", [])
    monkeypatch.setattr(config, "FALLBACK_ENDPOINTS", [])
    out = vs.apply_for_chain(config.FINNEY_GENESIS_HASH, url)
    want = f"{BASE}:{_hash('app-compose.frozen-2026-11-17b-finney.json')}"
    assert want in config.TEE_APPROVED_MEASUREMENTS
    assert out["newest"] == "app-compose.frozen-2026-11-17c-finney.json"
    assert out["endpoints_changed"]
    assert config.ARCHIVE_ENDPOINTS == [vs.FINNEY_DEFAULT_ARCHIVE]


def test_a_local_chain_gets_no_entries_from_the_code(monkeypatch):
    from fugal_subnet import config
    local = "0x" + "11" * 32
    monkeypatch.setattr(config, "TEE_APPROVED_MEASUREMENTS", [])
    monkeypatch.setattr(config, "ARCHIVE_ENDPOINTS", [])
    out = vs.apply_for_chain(local, "ws://127.0.0.1:9944")
    assert vs.network_for_genesis(local, "ws://127.0.0.1:9944") is None
    assert config.TEE_APPROVED_MEASUREMENTS == [] and out["added"] == []
    assert not out["endpoints_changed"]


def test_a_named_network_is_not_resolved_twice(monkeypatch):
    from fugal_subnet import config
    monkeypatch.setattr(config, "TEE_APPROVED_MEASUREMENTS", ["x:y"])
    out = vs.apply_for_chain(config.FINNEY_GENESIS_HASH, "finney")
    assert out["added"] == [] and config.TEE_APPROVED_MEASUREMENTS == ["x:y"]


def test_the_validator_applies_the_chain_setup_after_reading_the_genesis():
    src = (ROOT / "neurons" / "validator.py").read_text()
    g = src.index("genesis = chain_genesis_hash(subtensor)")
    a = src.index("validator_setup.apply_for_chain(genesis, network)")
    c = src.index("confirm_block = _check_chain_configuration(")
    assert g < a < c
