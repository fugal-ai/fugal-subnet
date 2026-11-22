"""Protocol v4: the scored questions are optional (docs/decisions/SEAT_MINIMAL_V4.md
3.4, 5.1; package S3).

Every v4 proof carries its exploration; a miner chooses each epoch whether its
TD also answers the epoch's scored questions, all of them or none. The proof
says which (`BenchmarkProof.scored`, hashed and attested). A proof with no
scored questions verifies; one with some of them is refused whatever it says;
one that says it answered none carries none. The miner's choice is
`--v4-scored` (auto, on, off; `td_v4.ScoredPolicy`), "auto" by default:
answer only while the newest root reveal names this head as the seat holder.
v1-v3 proofs keep their bytes and hashes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from fugal_subnet.attacks import run_tee_attacks as suite
from fugal_subnet.tee import td_v4
from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.proof import BenchmarkProof


@pytest.fixture(autouse=True)
def _real_hardware(monkeypatch):
    monkeypatch.setattr(verify_mod, "verify_dcap", lambda q, **_kw: True)


def _exploration_only(world=None, *, scored=False, extra=()):
    w = world or suite._v4_world()
    results = [r for r in suite._v4_results(w) if r.is_exploration] + list(extra)
    return suite._v4_proof(results, world=w, scored=scored)


# --- the field ----------------------------------------------------------------------------


def test_the_scored_field_is_hashed_serialised_and_read_strictly():
    full = suite._v4_proof()
    assert full.scored is True and full.to_dict()["scored"] is True
    back = BenchmarkProof.from_dict(json.loads(json.dumps(full.to_dict())))
    assert back.scored is True and back.content_hash() == full.content_hash()
    flipped = dataclasses.replace(full, scored=False)
    assert flipped.content_hash() != full.content_hash()
    assert "scored" not in dataclasses.replace(full, scored=None).to_dict()
    for bad in (1, 0, "true", None, []):
        d = full.to_dict()
        d["scored"] = bad
        with pytest.raises(ValueError):
            BenchmarkProof.from_dict(d)


def test_an_earlier_protocols_proof_never_reads_the_field():
    """A v1-v3 proof is read as before the field existed: a `scored` key on
    one is neither hashed nor refused (tests/test_proof_v4_fields.py pins
    their bytes and hashes)."""
    from tests.test_proof_v4_fields import PINNED, _legacy_proofs, _wire

    for name, p in _legacy_proofs().items():
        for junk in (True, False, "nonsense"):
            d = json.loads(_wire(p))
            d["scored"] = junk
            back = BenchmarkProof.from_dict(d)
            assert back.scored is None and back.content_hash() == PINNED[name][0]


# --- verification -------------------------------------------------------------------------


def test_a_full_proof_and_an_exploration_only_proof_both_verify():
    full = suite._v4_verify(suite._v4_proof())
    assert full.valid, full.reason
    p = _exploration_only()
    assert p.scored is False and p.scored_results == [] and p.exploration_results
    r = suite._v4_verify(p)
    assert r.valid, r.reason


def test_an_exploration_only_proof_still_carries_its_exploration():
    w = suite._v4_world()
    p = _exploration_only(w)
    assert {r.question_id for r in p.exploration_results} == set(w["expected"].exploration)
    # Exploration is still the whole assigned set: one dropped is refused as before.
    short = _exploration_only(w)
    short.results = short.results[1:]
    short.attestation_quote = suite._quote(bytes.fromhex(short.content_hash()),
                                           suite.HONEST_MRTD, suite.HONEST_RTMR3)
    assert not suite._v4_verify(short).valid


@pytest.mark.parametrize("keep", [1, 19])
def test_some_of_the_scored_questions_are_refused_whatever_the_proof_says(keep):
    w = suite._v4_world()
    scored = [r for r in suite._v4_results(w) if not r.is_exploration]
    for flag in (True, False):
        p = _exploration_only(w, scored=flag, extra=scored[:keep])
        r = suite._v4_verify(p)
        assert not r.valid and "scored" in r.reason


def test_a_proof_that_does_not_say_is_refused():
    p = suite._v4_proof(scored=None)
    r = suite._v4_verify(p)
    assert not r.valid and "`scored`" in r.reason


def test_a_repeated_scored_question_is_refused():
    w = suite._v4_world()
    results = suite._v4_results(w)
    p = suite._v4_proof(results[:1] + results, world=w)
    assert not suite._v4_verify(p).valid


# --- the TD ---------------------------------------------------------------------------------


def test_the_td_makes_an_exploration_only_proof_when_told_and_calls_nothing_scored(monkeypatch):
    from tests import test_harness_v4 as h

    full, provider_full, _ = h._run(monkeypatch)
    off, provider_off, _ = h._run(monkeypatch, challenge=None)
    assert full.scored is True and len(full.scored_results) == h.SLICE
    assert off.scored is False and off.scored_results == [] and off.challenge is None
    # The same slice drawn (its hash is in the proof), the same exploration, the same reads.
    assert off.questions_hash == full.questions_hash
    assert ([(r.question_id, r.routed_model) for r in off.exploration_results]
            == [(r.question_id, r.routed_model) for r in full.exploration_results])
    scored_ids = {r.question_id for r in full.scored_results}
    assert not {q for _m, _r, q in provider_off.calls} & scored_ids
    assert len(provider_off.calls) == len(full.exploration_results)
    assert off.total_cost_usd < full.total_cost_usd


def test_the_session_asks_the_policy_every_epoch():
    asked = []

    def policy(epoch):
        asked.append(epoch)
        return td_v4.ChallengeSpec(500, None) if epoch % 2 == 0 else None

    from tests import menu_fixtures as mf
    s = td_v4.V4Session(first_epoch=100, fetch_listing=lambda m: b"", read_commitment=lambda e: None,
                        menu_bytes=lambda h: None, list_bytes=lambda s, h: None,
                        plan=object(), anchor=mf.state(epoch=99), challenge=policy)
    s.previous = mf.state(epoch=101)
    s._recent[101] = s.previous
    assert s.inputs_for(102).challenge == td_v4.ChallengeSpec(500, None)
    s._recent[102] = mf.state(epoch=102)
    assert s.inputs_for(103).challenge is None
    assert asked == [102, 103]
    plain = td_v4.V4EpochInputs(epoch=1, previous=None, committed_hash=None, published=None,
                                allowlist=None, plan=None, fetch_listing=None)
    assert plain.challenge is None


# --- the miner's choice ---------------------------------------------------------------------

HEAD = "ab" * 32
OTHER = "cd" * 32


def test_a_size_of_0_answers_nothing_and_500_or_1000_every_epoch():
    def read(epoch):
        return epoch - 2, None
    none = td_v4.ChallengePolicy(0, first_epoch=0)
    assert all(none(e) is None for e in range(5))
    for size in (500, 1000):
        pol = td_v4.ChallengePolicy(size, first_epoch=0, read_holder=read,
                                    holder_bytes=lambda h: None)
        assert all(pol(e) == td_v4.ChallengeSpec(size, None) for e in range(2, 7))
    for bad in (200, 750, "500", True):
        with pytest.raises(ValueError):
            td_v4.ChallengePolicy(bad, first_epoch=0, read_holder=read,
                                  holder_bytes=lambda h: None)
    with pytest.raises(ValueError):
        td_v4.ChallengePolicy(500, first_epoch=0)              # it needs both readers


def test_a_failed_holder_read_makes_no_challenge_that_epoch():
    def read(epoch):
        if epoch == 5:
            raise ConnectionError("the chain could not be read")
        return epoch - 2, None

    pol = td_v4.ChallengePolicy(500, first_epoch=0, read_holder=read, holder_bytes=lambda h: None)
    assert [pol(e) is None for e in range(3, 8)] == [False, False, True, False, False]


def test_the_holder_is_read_from_the_reveal_the_root_committed():
    doc = {"epoch_id": "e00000050", "champion": {"champion": HEAD, "event": "hold"}}
    raw = json.dumps(doc).encode()
    sha = hashlib.sha256(raw).hexdigest()
    assert td_v4.seat_holder_from_reveal(raw, "e00000050", sha) == HEAD
    empty = json.dumps({"epoch_id": "e00000050", "champion": {"champion": None}}).encode()
    assert td_v4.seat_holder_from_reveal(empty, "e00000050",
                                         hashlib.sha256(empty).hexdigest()) is None
    with pytest.raises(ValueError):
        td_v4.seat_holder_from_reveal(raw, "e00000050", "00" * 32)       # not the committed bytes
    with pytest.raises(ValueError):
        td_v4.seat_holder_from_reveal(raw, "e00000051", sha)             # another epoch's
    bad = json.dumps({"epoch_id": "e00000050", "champion": {"champion": "x"}}).encode()
    with pytest.raises(ValueError):
        td_v4.seat_holder_from_reveal(bad, "e00000050", hashlib.sha256(bad).hexdigest())


def test_the_miner_flag_defaults_to_0_and_reaches_the_session(monkeypatch):
    import click

    from fugal_subnet import config
    from neurons import miner

    opt = next(p for p in miner.main.params if p.name == "v4_challenge")
    monkeypatch.delenv("FUGAL_V4_CHALLENGE", raising=False)
    assert opt.get_default(click.Context(miner.main)) == "0"
    assert list(opt.type.choices) == ["0", "500", "1000"]
    assert "FUGAL_V4_CHALLENGE" in config.OPERATIONAL_ENV_VARS
    import inspect
    assert inspect.signature(miner.main.callback).parameters["v4_challenge"].default == "0"
    monkeypatch.setattr(config, "ROOT_HOTKEY_ENV", "5Froot")
    from tests.test_v4_loop import _pool

    s = miner._v4_session(None, netuid=7, pool=_pool(), first_epoch=10, blocks_per_epoch=10,
                          coldkey=None, genesis="0x" + "4c" * 32, challenge_size=1000,
                          read_holder=lambda e: (e - 2, None), holder_bytes=lambda h: None)
    assert isinstance(s.challenge, td_v4.ChallengePolicy) and s.challenge.size == 1000
    assert s.challenge(13) == td_v4.ChallengeSpec(1000, None)
    off = miner._v4_session(None, netuid=7, pool=_pool(), first_epoch=10, blocks_per_epoch=10,
                            coldkey=None, genesis="0x" + "4c" * 32)
    assert off.challenge(13) is None
