"""Protocol v4 proof verification (docs/decisions/LIVE_MENU_V4.md, 4.1, 5, 8, 12).

A v4 proof is checked against the epoch's MenuState: its menu hash and
namespace, its scored routes and its exploration (the shared set and both
private parts). No proof measures a proposal (FAST_CROWN_V4.md 18): a new
entry is measured by every miner's coverage in its join epoch, and a proof
that carries a proposal or a measurement is refused as it is read. Every
attack case is refused, and each is shown
to verify once the one rule that refuses it is taken out: the rule, not some
other check, is what holds.

DCAP is stubbed to True, as in the TEE attack suite: the attacker owns real
hardware, so a case is refused by a binding, never by a signature.
"""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from fugal_subnet import epoch_plan
from fugal_subnet import menu_state as ms
from fugal_subnet.attacks import run_tee_attacks as suite
from fugal_subnet.tee import td_v4
from fugal_subnet.tee import verify as verify_mod


@pytest.fixture(autouse=True)
def _real_hardware(monkeypatch):
    monkeypatch.setattr(verify_mod, "verify_dcap", lambda q, **_kw: True)


def test_an_honest_v4_proof_verifies():
    r = suite._v4_verify(suite._v4_proof())
    assert r.valid, r.reason


def test_an_honest_v4_proof_verifies_in_mock_mode_too():
    assert suite._v4_verify(suite._v4_proof(), mock=True).valid


def test_the_expectation_is_the_tds_own_draw():
    """The validator's expectation is what the TD runs: td_v4.exploration_v4
    on the epoch's menu, and nothing else (no measurement)."""
    w = suite._v4_world()
    x = w["expected"]
    assert x.exploration == td_v4.exploration_v4(w["nonce"], w["pool"], set(w["slice"]),
                                                 w["state"], suite.HONEST_HOTKEY)
    assert set(x.shared) | x.private_ids == set(x.exploration)
    assert not set(x.shared) & x.private_ids
    assert not hasattr(x, "measurement") and not hasattr(x, "measure_entry")
    assert x.menu_hash == ms.menu_hash(w["state"])


def test_in_its_join_epoch_every_miners_coverage_is_aimed_first_at_a_provisional_entry():
    """The entry a join delta put on the epoch's menu draws coverage pairs
    from every miner in that epoch, as the TD draws them; in the next epoch
    (still provisional) it draws only its ordinary share."""
    from tests import menu_fixtures as fx

    w = suite._v4_world()
    e = w["state"].entries[0]
    doc = fx.document(f"{e.model.split('/')[0]}/model-new", "high",
                      hosts=e.provider_order, quantization=e.quantization)[0]
    joined = ms.apply_joins(w["state"], [ms.NewEntry(
        request=doc.request(e.max_tokens), price_in=e.price_in, price_out=e.price_out,
        proposal="ab" * 32, proposer=fx.HOTKEY, coldkey=fx.COLDKEY)])
    new = next(x.id for x in joined.entries if x.provisional)

    def pairs(state, hotkey):
        x = verify_mod.expected_v4(state, epoch=state.epoch, nonce=w["nonce"], pool=w["pool"],
                                   scored_ids=w["slice"], hotkey=hotkey, plan=w["plan"])
        assert x.exploration == td_v4.exploration_v4(w["nonce"], w["pool"], set(w["slice"]),
                                                     state, hotkey)
        return sum(1 for eid in x.exploration.values() if eid == new)

    hotkeys = (suite.HONEST_HOTKEY, "5Fother-a", "5Fother-b")
    first = [pairs(joined, hk) for hk in hotkeys]
    later = [pairs(dataclasses.replace(joined, epoch=joined.epoch + 1), hk) for hk in hotkeys]
    assert all(first) and sum(first) > sum(later)


def test_v4_verification_needs_the_epochs_menu():
    with pytest.raises(ValueError, match="V4Expected"):
        suite._v4_verify(suite._v4_proof(), v4=None)


def test_an_expectation_for_another_hotkey_is_a_caller_error():
    w = suite._v4_world()
    other = dataclasses.replace(w["expected"], hotkey="5Fsomeone-else")
    with pytest.raises(ValueError, match="another hotkey"):
        suite._v4_verify(suite._v4_proof(), v4=other)


def test_a_second_exploration_expectation_must_agree():
    w = suite._v4_world()
    same = dict(w["expected"].exploration)
    assert suite._v4_verify(suite._v4_proof(), expected_exploration=same).valid
    other = dict(same)
    other[next(iter(other))] = "lab-new/elsewhere"
    with pytest.raises(ValueError, match="differs"):
        suite._v4_verify(suite._v4_proof(), expected_exploration=other)


def test_the_expectation_is_for_the_epoch_of_its_menu():
    w = suite._v4_world()
    with pytest.raises(ValueError, match="menu of epoch"):
        verify_mod.expected_v4(w["state"], epoch=suite.V4_EPOCH + 1, nonce=w["nonce"],
                               pool=w["pool"], scored_ids=w["slice"],
                               hotkey=suite.HONEST_HOTKEY, plan=w["plan"])


def test_a_v4_proof_in_a_v3_epoch_and_a_v3_proof_in_a_v4_epoch_are_refused():
    from fugal_subnet.routing_protocol import identity_v3

    r = suite._v4_verify(suite._v4_proof(), protocol_version=3, v4=None)
    assert not r.valid and "namespace" in r.reason
    v3 = suite._v4_proof(namespace=identity_v3())
    v3 = dataclasses.replace(v3, menu_hash=None, endpoint_reads=None)
    v3.attestation_quote = suite._quote(bytes.fromhex(v3.content_hash()), suite.HONEST_MRTD,
                                        suite.HONEST_RTMR3)
    r = suite._v4_verify(v3)
    assert not r.valid and "menu" in r.reason


def test_a_v4_proof_without_its_endpoint_reads_is_refused():
    p = dataclasses.replace(suite._v4_proof(), endpoint_reads=None)
    p.attestation_quote = suite._quote(bytes.fromhex(p.content_hash()), suite.HONEST_MRTD,
                                       suite.HONEST_RTMR3)
    r = suite._v4_verify(p)
    assert not r.valid and "endpoint reads" in r.reason


def test_a_head_with_no_row_on_the_menu_is_refused():
    import io
    import json

    import numpy as np

    from fugal_subnet.routing_protocol import MANIFEST_PATH
    from fugal_subnet.vendor import success_contract as contract

    rows = ["lab-new/off-menu@high"]
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["models"] = [{"id": rows[0], "samples": 1, "mean_in_tokens": 1.0,
                           "mean_out_tokens": 1.0}]
    buf = io.BytesIO()
    np.savez(buf, **contract.make_head(np.zeros((1, 1024)), np.zeros(1), rows, manifest, "t"))
    head = buf.getvalue()
    w = dict(suite._v4_world(), head=head, head_hash=hashlib.sha256(head).hexdigest())
    r = suite._v4_verify(suite._v4_proof(world=w), world=w)
    assert not r.valid and "no row" in r.reason


# --- each attack is refused, and by its own rule ---------------------------------------

CASES = [
    (suite.a_v4_stale_menu, "_v4_menu_check"),
    (suite.a_v4_foreign_menu, "_v4_menu_check"),
    (suite.a_v4_namespace_of_another_menu, "_v4_menu_check"),
    (suite.a_v4_route_off_the_menu, "_v4_route_check"),
]


@pytest.mark.parametrize("attack, rule", CASES, ids=[c[0].__name__ for c in CASES])
def test_each_v4_attack_is_refused_by_its_rule(attack, rule, monkeypatch):
    assert not attack().valid
    monkeypatch.setattr(verify_mod, rule, lambda *_a, **_k: None)
    r = attack()
    assert r.valid, f"{attack.__name__} is refused by something other than {rule}: {r.reason}"


READ_CASES = [suite.a_v4_proof_carrying_a_proposal, suite.a_v4_result_claiming_a_measurement]


@pytest.mark.parametrize("attack", READ_CASES, ids=[c.__name__ for c in READ_CASES])
def test_a_proof_with_a_proposal_or_a_measurement_is_refused_as_it_is_read(attack):
    r = attack()
    assert not r.valid and r.reason.startswith("refused as read"), r.reason


def test_every_v4_attack_is_in_the_suite():
    names = {fn for _n, _i, fn in suite.ATTACKS}
    assert {c[0] for c in CASES} | set(READ_CASES) <= names


# --- the epoch's shares, computed once (integration decision 2c) ---------------------------


def test_the_expectation_takes_the_epochs_shares_and_computes_none_itself(monkeypatch):
    from fugal_subnet.explore_weights import SharesV4
    from neurons import validator

    w = suite._v4_world()
    shares = epoch_plan.explore_shares_v4(w["state"])
    assert isinstance(shares, SharesV4)
    calls = []
    real = epoch_plan.explore_shares_v4
    monkeypatch.setattr(epoch_plan, "explore_shares_v4",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    for hk in (suite.HONEST_HOTKEY, "5Fother-a", "5Fother-b"):
        got = validator._v4_expected(w["state"], epoch_index=suite.V4_EPOCH, nonce=w["nonce"],
                                     pool=w["pool"], scored_ids=w["slice"], hotkey=hk,
                                     plan=w["plan"],
                                     shares=shares)
        assert got == verify_mod.expected_v4(
            w["state"], epoch=suite.V4_EPOCH, nonce=w["nonce"], pool=w["pool"],
            scored_ids=w["slice"], hotkey=hk, plan=w["plan"], shares=real(w["state"]))
    assert calls == []
    # Shares of another menu are refused, not used.
    other = dataclasses.replace(w["state"], entries=w["state"].entries[1:])
    with pytest.raises(ValueError, match="shares"):
        verify_mod.expected_v4(w["state"], epoch=suite.V4_EPOCH, nonce=w["nonce"],
                               pool=w["pool"], scored_ids=w["slice"], hotkey="5Fx",
                               plan=w["plan"],
                               shares=real(other))


# That the loop draws the epoch's shares once and hands the same shares to every
# hotkey's expectation is shown by running the loop (tests/test_v4_loop.py),
# which replaced a search of the loop's source text.


# --- the v4 head check is W7's verdict ------------------------------------------------------


def _head_without_token_means(rows):
    import io

    import numpy as np

    from fugal_subnet import success_test_fixtures as sfx

    z = dict(sfx.arrays(np.zeros((len(rows), 1024), np.float32), np.zeros(len(rows), np.float32),
                        list(rows)))
    for key in ("mean_in_tokens", "mean_out_tokens", "cost_profile_id"):
        z.pop(key, None)
    buf = io.BytesIO()
    np.savez(buf, **z)
    return buf.getvalue()


def test_a_proof_from_a_head_with_no_token_means_verifies_under_v4():
    from fugal_subnet import head_eval

    w = suite._v4_world()
    head = _head_without_token_means(w["rows"])
    assert head_eval.v4_head_verdict(head, w["state"]) is None
    with pytest.raises(ValueError):
        head_eval.load_head_from_npz(head)                  # the v3 loader refuses it
    proof = suite._v4_proof()
    proof.weights_hash = hashlib.sha256(head).hexdigest()
    proof.attestation_quote = suite._quote(bytes.fromhex(proof.content_hash()),
                                           suite.HONEST_MRTD, suite.HONEST_RTMR3)
    r = suite._v4_verify(proof, head_bytes=head, expected_weights_hash=proof.weights_hash)
    assert r.valid, r.reason
    # A head with no row on the epoch's menu is still refused.
    other = _head_without_token_means(["lab-z/not-on-the-menu"])
    proof.weights_hash = hashlib.sha256(other).hexdigest()
    proof.attestation_quote = suite._quote(bytes.fromhex(proof.content_hash()),
                                           suite.HONEST_MRTD, suite.HONEST_RTMR3)
    r = suite._v4_verify(proof, head_bytes=other, expected_weights_hash=proof.weights_hash)
    assert not r.valid and "no row" in r.reason
