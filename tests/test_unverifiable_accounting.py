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
import re
from dataclasses import asdict
from pathlib import Path

from fugal_subnet.epoch_logger import EpochLog

ROOT = Path(__file__).resolve().parents[1]


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


def test_validator_counts_unverifiable_before_calling_anything_invalid():
    """The regression this guards is a future refactor collapsing the branch.

    Re-merging the counters would restore the exact failure the split exists to
    remove -- 247 honest miners logged as invalid during an endpoint outage,
    sending an operator hunting an attack that never happened.
    """
    src = (ROOT / "neurons" / "validator.py").read_text(encoding="utf-8")

    branch = re.search(
        r"if not result\.valid:(.*?)\n                    continue",
        src, re.DOTALL,
    )
    assert branch, "the invalid branch in the verify loop moved or was renamed"
    body = branch.group(1)

    assert "unverifiable" in body, (
        "the verify loop no longer distinguishes unverifiable proofs — an "
        "outage would again be recorded as mass fraud"
    )
    assert "n_unverifiable += 1" in body
    assert "n_invalid += 1" in body


def test_an_unverifiable_proof_never_stops_the_epoch():
    """I6: no miner behaviour may stop a validator setting weights.

    The unverifiable branch must fall through to the same `continue` as any
    other unscored miner. A `raise`, `break` or early return there would hand
    every registered miner a subnet kill switch, because the crafted-FMSPC path
    into `unverifiable` is miner-reachable.
    """
    src = (ROOT / "neurons" / "validator.py").read_text(encoding="utf-8")
    branch = re.search(
        r"if not result\.valid:(.*?)\n                    continue",
        src, re.DOTALL,
    ).group(1)

    for forbidden in ("raise", "break", "return"):
        assert not re.search(rf"\b{forbidden}\b", branch), (
            f"'{forbidden}' in the unverifiable branch would let one miner stop "
            "the epoch — see I6 and the kill-switch retraction in INVARIANTS"
        )
