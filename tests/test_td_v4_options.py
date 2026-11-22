"""Protocol v4: a miner in a TD sets its own options (the proposal it files,
its measurement spend ceiling, the size of the challenge it answers) over
the attested provisioning channel, and the TD checks each value strictly
before it keeps anything (fugal_subnet/tee/v4_options.py, tee/provision.py,
neurons/miner._v4_pushed_options, scripts/provision_td.py).

The channel before protocol v4 is unchanged both ways: an operator that
sends only the fields it always sent is accepted by the new receiver and
changes nothing in the miner, and the new operator's push without v4
options is exactly the push an old receiver accepts."""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import types

import pytest
from cryptography.hazmat.primitives.asymmetric import x25519

from fugal_subnet.tee import provision, v4_options
from fugal_subnet.tee.provision import ProvisionError, ProvisionStore, seal, unseal
from tests import menu_fixtures as fx
from tests.test_provision_schema import EXPECTED_BEFORE_V4

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROPOSAL = {"model": "lab-c/model-new", "reasoning": ["effort", "high"],
            "provider_order": ["host-a/fp8"], "quantization": "fp8"}
PROPOSAL_TEXT = json.dumps(PROPOSAL)
BEFORE_V4 = {"head_b64": base64.b64encode(b"head").decode(), "hotkey_ss58": "5Fk",
             "openrouter_api_key": "sk-or-v1-x", "hotkey_keyfile_b64": "a2V5",
             "coldkeypub_b64": "cHVi"}
V4 = {"v4_proposal": PROPOSAL_TEXT, "v4_challenge": "1000"}


def _td():
    key = x25519.X25519PrivateKey.generate()
    return key, key.public_key().public_bytes_raw()


def _sealed_raw(plaintext: bytes, td_pub: bytes, nonce: str) -> dict:
    """An envelope around any bytes: what a hostile operator can send, past
    the operator-side check that `seal` runs."""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    eph = x25519.X25519PrivateKey.generate()
    shared = eph.exchange(x25519.X25519PublicKey.from_public_bytes(td_pub))
    k = provision._derive(shared, bytes.fromhex(nonce), provision._HKDF_INFO_KEY)
    iv = os.urandom(12)
    ct = ChaCha20Poly1305(k).encrypt(iv, plaintext, bytes.fromhex(nonce))
    return {"nonce": nonce, "epk": eph.public_key().public_bytes_raw().hex(), "iv": iv.hex(),
            "ct": ct.hex()}


def _receive(payload_bytes: bytes) -> ProvisionStore:
    """The TD side of one push: unseal, then the store's accept."""
    priv, pub = _td()
    nonce = provision.new_nonce()
    values, key = unseal(priv, _sealed_raw(payload_bytes, pub, nonce))
    store = ProvisionStore()
    store.accept(values, key)
    return store


# --- each value is checked strictly -------------------------------------------------------


def test_the_two_options_parse_to_what_the_miner_takes():
    spec = v4_options.parse_proposal(PROPOSAL_TEXT)
    assert spec.id == "lab-c/model-new@high" and spec.provider_order == ("host-a/fp8",)
    assert v4_options.listed_refusal(spec, fx.ALLOW) is None
    assert not hasattr(v4_options, "parse_measure_cap")    # nothing is measured or spent
    assert [v4_options.parse_challenge(w) for w in ("0", "500", "1000")] == [0, 500, 1000]
    assert v4_options.check_pushed({**BEFORE_V4, **V4}) == {
        "v4_proposal": spec, "v4_challenge": 1000}


@pytest.mark.parametrize("value", [
    None, 7, ["on"], b"on", "", "x" * 2000, PROPOSAL_TEXT + " " * 1024,
    "{", "[]", "null", json.dumps({**PROPOSAL, "extra": 1}),
    json.dumps({k: v for k, v in PROPOSAL.items() if k != "quantization"}),
    '{"model": "lab-c/model-new", "model": "lab-a/x", "reasoning": ["effort", "high"], '
    '"provider_order": ["host-a/fp8"], "quantization": "fp8"}',
    PROPOSAL_TEXT.replace('"fp8"}', 'NaN}'),
    PROPOSAL_TEXT.replace("model-new", "model-new\x00"),
    PROPOSAL_TEXT.replace("model-new", "model-new\n"),
    PROPOSAL_TEXT.replace("model-new", "model-néw"),
    PROPOSAL_TEXT.replace("model-new", "model-new\x1b[2J"),
    json.dumps({**PROPOSAL, "model": "Lab-C/Model"}),
    json.dumps({**PROPOSAL, "model": "lab-c/model-new:free"}),
    json.dumps({**PROPOSAL, "reasoning": ["effort", "ultra"]}),
    json.dumps({**PROPOSAL, "reasoning": ["enabled", False]}),
    json.dumps({**PROPOSAL, "reasoning": ["effort"]}),
    json.dumps({**PROPOSAL, "provider_order": []}),
    json.dumps({**PROPOSAL, "provider_order": ["host-a/fp8", "host-b/fp8", "host-c/fp8"]}),
    json.dumps({**PROPOSAL, "provider_order": ["host-a/fp8", "host-a/fp8"]}),
    json.dumps({**PROPOSAL, "provider_order": ["host-a/flex/fp8"]}),
    json.dumps({**PROPOSAL, "provider_order": ["host-a/int8"]}),
    json.dumps({**PROPOSAL, "quantization": "fp7"}),
    json.dumps({**PROPOSAL, "provider_order": [7]}),
])
def test_a_proposal_that_is_not_one_is_refused(value):
    with pytest.raises(v4_options.OptionRefused):
        v4_options.parse_proposal(value)


@pytest.mark.parametrize("value, rule", [
    ({**PROPOSAL, "model": "lab-z/model-new"}, "lab_not_allowed"),
    ({**PROPOSAL, "provider_order": ["host-z/fp8"]}, "host_not_allowed"),
    ({**PROPOSAL, "provider_order": ["host-a"], "quantization": "unknown"},
     "unknown_quantization"),
])
def test_a_proposal_a_list_does_not_allow_passes_provisioning_and_the_rule_is_named(value, rule):
    """No signed list is read at provisioning, on either side: a list in
    force at a later filing may allow what the image's list does not. Which
    rule a given list refuses it under is still said (READY line, push
    script); the filing decides (test_a_td_files_a_proposal_only_under_a_list_that_allows_it)."""
    text = json.dumps(value)
    spec = v4_options.parse_proposal(text)                  # its own rules pass
    assert v4_options.check_pushed({**BEFORE_V4, "v4_proposal": text}) == {"v4_proposal": spec}
    assert provision.validate_payload({**BEFORE_V4, "v4_proposal": text})["v4_proposal"] == text
    assert rule in v4_options.listed_refusal(spec, fx.ALLOW)
    assert v4_options.listed_refusal(spec, None) == "no signed lab and host list"


def test_a_td_keeps_a_proposal_its_images_list_does_not_allow():
    off = json.dumps({**PROPOSAL, "model": "lab-z/model-new"})
    store = _receive(json.dumps({**BEFORE_V4, **V4, "v4_proposal": off}).encode())
    assert store.get("v4_proposal") == off


def _reads(spec):
    from fugal_subnet.tee import endpoint_reads as er

    return [er.EndpointRead(spec.model, tuple(er.HostListing(h, True, 1, 1, 65536)
                                              for h in spec.provider_order))]
@pytest.mark.parametrize("value", [None, True, 500, "", "on", "auto", " 500", "500 ", "0500",
                                   "+500", "750", "1e3", "1000\x00", "10000"])
def test_the_challenge_is_exactly_0_500_or_1000(value):
    with pytest.raises(v4_options.OptionRefused):
        v4_options.parse_challenge(value)


def test_a_refusal_names_the_option_and_the_rule_never_the_value():
    secret = "sk-or-v1-" + "f" * 40
    for field in V4:
        with pytest.raises(ProvisionError) as e:
            provision.validate_payload({**BEFORE_V4, field: secret})
        assert secret not in str(e.value) and "f" * 10 not in str(e.value)


# --- the receiver: what a push may carry -----------------------------------------------------


def test_the_td_keeps_a_push_with_valid_v4_options_and_the_miner_reads_them():
    store = _receive(json.dumps({**BEFORE_V4, **V4}).encode())
    assert {f: store.get(f) for f in V4} == V4


@pytest.mark.parametrize("payload, why", [
    ({**BEFORE_V4, **V4, "v4_measure_cap_usd": "25"}, "removed provisioning field"),
    ({**BEFORE_V4, **V4, "v4_scored": "on"}, "removed provisioning field"),
    ({**BEFORE_V4, **V4, "v4_challenge": "750"}, "one of"),
    ({**BEFORE_V4, "v4_anchor": "menu.json"}, "unknown provisioning field"),
    ({**BEFORE_V4, "v4_state_dir": "/tmp"}, "unknown provisioning field"),
    ({**BEFORE_V4, "v4_challenge": 25}, "must be a string"),
])
def test_a_push_with_a_v4_value_that_fails_is_refused_whole(payload, why):
    with pytest.raises(ProvisionError, match=why):
        _receive(json.dumps(payload).encode())


def test_a_field_sent_twice_is_refused_for_every_field():
    for name, a, b in (("v4_challenge", "0", "1000"), ("v4_proposal", "{}", "[]"),
                       ("openrouter_api_key", "sk-a", "sk-b")):
        body = json.dumps(BEFORE_V4)[:-1] + f', "{name}": "{a}", "{name}": "{b}"}}'
        with pytest.raises(ProvisionError, match="sent twice"):
            _receive(body.encode())


# --- the miner applies them ---------------------------------------------------------------------


def test_the_miner_takes_the_pushed_options_and_refuses_them_set_twice():
    from neurons import miner

    unset = (None, "0")
    spec, challenge, came = miner._v4_pushed_options(V4, 4, env=unset, v4_from="provisioning")
    assert spec == v4_options.parse_proposal(PROPOSAL_TEXT)
    assert challenge == 1000 and came == "provisioning"
    # Only some of them: the others keep their defaults.
    got = miner._v4_pushed_options({"v4_challenge": "500"}, 4, env=unset, v4_from="provisioning")
    assert got == (None, 500, "provisioning")
    # Nothing pushed: what the miner had.
    assert miner._v4_pushed_options({}, 4, env=unset, v4_from="provisioning") == (
        None, "0", "provisioning")
    # One operator's choice has one source.
    for env in ((v4_options.parse_proposal(PROPOSAL_TEXT), "0"), (None, "500")):
        with pytest.raises(ValueError, match="both in this miner's options and in the"):
            miner._v4_pushed_options(V4, 4, env=env, v4_from="provisioning")
    # An image that serves protocol 3 ignores them.
    assert miner._v4_pushed_options(V4, 3, env=unset, v4_from="provisioning") == (
        None, "0", "provisioning")


def test_the_ready_line_names_the_options_in_force_and_where_they_came_from(monkeypatch):
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch
    from fugal_subnet.config import EPOCH_INTERVAL
    from neurons import miner

    block = 100 * blocks_per_epoch(EPOCH_INTERVAL)
    spec, challenge, came = miner._v4_pushed_options(
        V4, 4, env=(None, "0"), v4_from="provisioning")
    ok, line = miner._miner_ready_line(4, block, v4_spec=spec, v4_challenge=challenge,
                                       pool_size=10, root_hotkeys=((0, "5Froot"),), v4_from=came)
    assert ok, line
    assert "proposal lab-c/model-new@high" in line and "ceiling" not in line
    assert "challenge 1000" in line and "options from provisioning" in line
    ok, line = miner._miner_ready_line(4, block, pool_size=10, root_hotkeys=((0, "5Froot"),),
                                       v4_from="pushed at provisioning")
    assert "pushed at provisioning" in line and "proposal none" not in line


def test_the_ready_line_names_a_proposal_the_shipped_list_does_not_allow(monkeypatch):
    """Where the image ships a list, the READY line (and so --check outside a
    TD, and the log after a push) says whether it allows the proposal, and
    why not; the miner is still ready, and the filing decides each epoch."""
    from fugal_subnet import menu_rules as mr
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch
    from fugal_subnet.config import EPOCH_INTERVAL
    from neurons import miner

    monkeypatch.setattr(mr, "list_signers", lambda: b"")
    monkeypatch.setattr(mr, "shipped_list", lambda kind: fx.ALLOW)
    block = 100 * blocks_per_epoch(EPOCH_INTERVAL)
    kw = dict(v4_challenge="0", pool_size=10, root_hotkeys=((0, "5Froot"),),
              v4_from="provisioning")
    off = v4_options.parse_proposal(json.dumps({**PROPOSAL, "model": "lab-z/model-new"}))
    ok, line = miner._miner_ready_line(4, block, v4_spec=off, **kw)
    assert ok, line
    assert (f"proposal not allowed by the shipped list serial {fx.ALLOW.serial} (the compiled "
            "entry rule lab_not_allowed): admitted only under a list in force that allows "
            "it") in line
    ok, line = miner._miner_ready_line(4, block, v4_spec=v4_options.parse_proposal(PROPOSAL_TEXT),
                                       **kw)
    assert ok and "not allowed" not in line


def test_the_push_script_notes_whether_its_shipped_list_allows_the_proposal(monkeypatch):
    from fugal_subnet import menu_rules as mr

    spec = importlib.util.spec_from_file_location("provision_td", ROOT / "scripts" / "provision_td.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mr, "shipped_list", lambda kind: fx.ALLOW)
    assert "allows it" in mod._proposal_list_note(PROPOSAL_TEXT)
    note = mod._proposal_list_note(json.dumps({**PROPOSAL, "model": "lab-z/model-new"}))
    assert "does not allow it (the compiled entry rule lab_not_allowed)" in note
    assert "admits it only under a list in force that allows it" in note

    def none(kind):
        raise FileNotFoundError(kind)

    monkeypatch.setattr(mr, "shipped_list", none)
    assert "ships no signed lab and host list" in mod._proposal_list_note(PROPOSAL_TEXT)


def test_the_miner_hands_the_pushed_options_on_after_a_real_push(monkeypatch):
    """The miner's own receiver, a sealed push through it, and the values the
    miner then reads."""
    import socket

    from neurons import miner
    from tests.test_provision_transport import _post

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr("fugal_subnet.tee.dstack_client.attest", lambda rd, path="": b"\x00" * 8)
    monkeypatch.setattr("fugal_subnet.tee.provision.PROVISION_PORT", port)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result, pushed = {}, {}

    def run():
        result["head"], _ = miner._await_provisioning(coldkey="c", hotkey="h",
                                                      wallet_path="/w", v4_options=pushed)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.5)
    base = f"http://127.0.0.1:{port}"
    nonce = provision.new_nonce()
    ans = _post(f"{base}/provision/attest", {"nonce": nonce})
    env, _ = seal({"head_b64": base64.b64encode(b"h").decode(), **V4},
                  bytes.fromhex(ans["pubkey"]), nonce)
    _post(f"{base}/provision", env)
    t.join(timeout=10)
    assert result.get("head") == b"h" and pushed == V4


# --- compatibility with the channel before protocol v4, both ways ----------------------------------


def test_a_push_from_before_v4_is_accepted_and_changes_nothing_in_the_miner():
    from neurons import miner

    assert set(BEFORE_V4) == EXPECTED_BEFORE_V4
    priv, pub = _td()
    nonce = provision.new_nonce()
    envelope, key = seal(dict(BEFORE_V4), pub, nonce)       # the envelope is unchanged
    assert set(envelope) == {"nonce", "epk", "iv", "ct"}
    values, _ = unseal(priv, envelope)
    assert values == BEFORE_V4
    store = ProvisionStore()
    store.accept(values, key)
    pushed = {f: store.get(f) for f in V4 if store.get(f)}
    assert pushed == {}
    assert miner._v4_pushed_options(pushed, 4, env=(None, "auto"),
                                    v4_from="provisioning") == (None, "auto", "provisioning")


def test_the_operators_push_without_v4_options_is_the_push_before_v4():
    spec = importlib.util.spec_from_file_location("provision_td", ROOT / "scripts" / "provision_td.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.v4_option_fields(None, None) == {}
    assert mod.v4_option_fields(PROPOSAL_TEXT, "1000") == V4


def _old_provision_module():
    """tee/provision.py as the v0.6.0 image ships it (signed tag
    image-v2026.11.17c), from this checkout's history; skipped where the
    history is not there (a shallow CI checkout)."""
    try:
        source = subprocess.run(["git", "show", "image-v2026.11.17c:fugal_subnet/tee/provision.py"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30, check=True
                                ).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("the v0.6.0 tag is not in this checkout")
    mod = types.ModuleType("provision_v060")
    exec(compile(source, "provision_v060.py", "exec"), mod.__dict__)
    return mod


def test_against_the_v060_module_both_ways():
    old = _old_provision_module()
    assert old.ALLOWED_FIELDS == frozenset(EXPECTED_BEFORE_V4)
    # A v0.6.0 operator pushing to the new image.
    priv, pub = _td()
    nonce = provision.new_nonce()
    envelope, _ = old.seal(dict(BEFORE_V4), pub, nonce)
    assert unseal(priv, envelope)[0] == BEFORE_V4
    # The new operator, no v4 option, pushing to a v0.6.0 image: accepted.
    old_priv = x25519.X25519PrivateKey.generate()
    old_pub = old_priv.public_key().public_bytes_raw()
    envelope, _ = seal(dict(BEFORE_V4), old_pub, nonce)
    assert old.unseal(old_priv, envelope)[0] == BEFORE_V4
    # With a v4 option the v0.6.0 receiver refuses the whole push by name,
    # as it refuses any field it does not know; the TD stays unprovisioned.
    envelope, _ = seal({**BEFORE_V4, "v4_challenge": "1000"}, old_pub, nonce)
    with pytest.raises(old.ProvisionError, match="unknown provisioning field"):
        old.unseal(old_priv, envelope)


def test_the_measured_identity_does_not_name_the_options():
    """The compose a TD is approved by carries none of them and allows no
    environment of its own: the values reach the TD only by the push."""
    composes = sorted((ROOT / "deploy" / "dstack").glob("*compose*"))
    assert composes
    for path in composes:
        text = path.read_text(encoding="utf-8")
        for var in v4_options.FIELDS:
            assert var not in text, (path.name, var)
        if path.suffix == ".json":
            assert json.loads(text)["allowed_envs"] == []
    assert sys.modules["fugal_subnet.tee.v4_options"] is v4_options


def test_the_tee_attack_suite_blocks_every_hostile_option_and_keeps_the_honest_push():
    from fugal_subnet.attacks import run_tee_attacks as suite

    for name, _inv, fn in suite.V4_OPTION_ATTACKS:
        got = fn()
        assert not got.valid, (name, got.reason)
    assert suite._v4_option_control()["verdict"] == "CONTROL"


@pytest.mark.parametrize("field", sorted(provision.REMOVED_FIELDS))
def test_a_push_carrying_a_removed_field_is_refused_with_its_replacement(field):
    """An operator's tooling from an earlier v4 build pushes a field the
    final build no longer takes: the push is refused naming the field and
    what replaced it, and the value is never echoed."""
    with pytest.raises(ProvisionError) as e:
        provision.validate_payload({"head_b64": "AA==", field: "secret-looking-value"})
    msg = str(e.value)
    assert field in msg and "no longer an option" in msg
    assert "secret-looking-value" not in msg
    assert not set(provision.REMOVED_FIELDS) & provision.ALLOWED_FIELDS
