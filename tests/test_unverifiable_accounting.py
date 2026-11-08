"""An outage must not be recorded as fraud.

`verify_proof` can now say "I could not check this" instead of "this is bad",
but that distinction is worthless if the validator folds both into one counter
before anyone sees it. These tests cover the accounting rather than the
classification -- `test_unverifiable.py` covers the classification.

They are deliberately about what gets RECORDED. The behaviour is unchanged:
an unverifiable miner is skipped exactly as it was, and falls through to
`apply_miss` like any absent miner. Skipping is also the only I6-compatible
option -- abandoning the epoch instead would let one miner with an unknown
FMSPC stop every validator publishing simultaneously.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from fugal_subnet.epoch_logger import EpochLog


def _log(**kw) -> EpochLog:
    base = dict(
        epoch_id="e-1", block_hash="0x" + "ab" * 32, timestamp=0.0,
        n_questions=300, n_miners_queried=256, n_heads_valid=0, n_heads_invalid=0,
    )
    base.update(kw)
    return EpochLog(**base)


def test_unverifiable_defaults_to_zero_so_existing_logs_keep_their_meaning():
    assert _log().n_heads_unverifiable == 0


def test_unverifiable_is_a_separate_field_from_invalid():
    log = _log(n_heads_invalid=3, n_heads_unverifiable=247)
    assert log.n_heads_invalid == 3
    assert log.n_heads_unverifiable == 247


def test_the_field_survives_serialisation():
    """The log is the artifact an operator reads (I7); an unserialised field
    records nothing."""
    round_tripped = json.loads(json.dumps(asdict(_log(n_heads_unverifiable=12))))
    assert round_tripped["n_heads_unverifiable"] == 12


def _check_with(monkeypatch, result):
    from types import SimpleNamespace

    from fugal_subnet.tee import verify as verify_mod
    from neurons import validator

    monkeypatch.setattr(verify_mod, "verify_proof", lambda proof, **kw: result)
    monkeypatch.setattr(validator, "_get_bundle_for_uid", lambda uid, resp: (object(), b"h"))
    resp = SimpleNamespace(proof_hash="p", weights_hash="w")
    checks = validator._EpochChecks(
        boundary_block=10, require_commitment=False, approved_measurements=set(),
        questions_hash="", nonce_hex="", slice_gold={}, question_ids=set(), explore_map={},
        mock=False, collateral_budget=None, protocol_version=1)
    return validator._check_response(1, resp, "5Hotkey", ("", 0), 0, checks)


def test_validator_counts_unverifiable_before_calling_anything_invalid(monkeypatch):
    """The regression this guards is a future refactor collapsing the branch.

    Re-merging the counters would restore the exact failure the split exists to
    remove -- 247 honest miners logged as invalid during an endpoint outage,
    sending an operator hunting an attack that never happened. The epoch loop
    counts each status apart (tests/test_validator_verify.py checks the loop).
    """
    from types import SimpleNamespace
    unverifiable = SimpleNamespace(valid=False, unverifiable=True, reason="pccs down", warnings=[])
    rejected = SimpleNamespace(valid=False, unverifiable=False, reason="bad quote", warnings=[])
    assert _check_with(monkeypatch, unverifiable).status == "unverifiable"
    assert _check_with(monkeypatch, rejected).status == "invalid"


def test_an_unverifiable_proof_never_stops_the_epoch(monkeypatch):
    """I6: no miner behaviour may stop a validator setting weights.

    The unverifiable outcome must come back like any other unscored miner's. A
    raise there would hand every registered miner a subnet kill switch, because
    the crafted-FMSPC path into `unverifiable` is miner-reachable.
    """
    from types import SimpleNamespace
    out = _check_with(monkeypatch, SimpleNamespace(valid=False, unverifiable=True,
                                                   reason="unknown FMSPC", warnings=[]))
    assert out.status == "unverifiable" and out.proof is None
