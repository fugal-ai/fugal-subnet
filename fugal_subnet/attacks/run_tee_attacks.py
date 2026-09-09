#!/usr/bin/env python3
"""Adversarial suite for the TEE verification path (I8, I3).

Every case here succeeded against production code before the attestation was
bound to what it claims to prove. They are kept as executable regressions
because each one is a *silent* failure: the proof verifies, the miner is paid,
and nothing in a log says otherwise.

The threat model is deliberately generous to the attacker: they own genuine
Intel TDX hardware, so `verify_dcap` really does pass and their quote really is
Intel-signed. DCAP proves the *hardware* is real. It says nothing about whether
the code running inside it is the published code — only the measurement
registers do. So DCAP is stubbed to True throughout; if a case is blocked here,
it is blocked by a binding, not by a signature.

    python -m fugal_subnet.attacks.run_tee_attacks
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
import time

import numpy as np

from fugal_subnet.routing_protocol import MANIFEST_PATH
from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.attestation import measurement_id, parse_quote
from fugal_subnet.tee.proof import (
    BenchmarkProof,
    QuestionResult,
    compute_questions_hash,
)
from fugal_subnet.vendor import success_contract as contract

HONEST_MRTD = bytes.fromhex("11" * 48)
ATTACKER_MRTD = bytes.fromhex("de" * 48)
SLICE = [f"q{i}" for i in range(10)]
POOL_GOLD = {f"q{i}": {"question_id": f"q{i}"} for i in range(1000)}
# An honest control must pass actual success-head admission, not just a byte hash.

_head_buffer = io.BytesIO()
np.savez(_head_buffer, **contract.make_head(np.zeros((1, 1024)), np.zeros(1),
    ["deepseek/deepseek-v4-flash"], json.loads(MANIFEST_PATH.read_text()), "synthetic-test-only attack control"))
HEAD = _head_buffer.getvalue()
HEAD_HASH = hashlib.sha256(HEAD).hexdigest()


def _quote(report_data: bytes, mrtd: bytes, rtmr3: bytes = b"\x00" * 48) -> bytes:
    q = bytearray(632)
    struct.pack_into("<H", q, 0, 4)        # version 4
    struct.pack_into("<I", q, 4, 0x81)     # tee_type TDX
    q[184:232] = mrtd
    q[520:568] = rtmr3
    q[568:632] = report_data.ljust(64, b"\x00")[:64]
    return bytes(q)


APPROVED = {measurement_id(parse_quote(_quote(b"\x00" * 64, HONEST_MRTD)))}


def _qr(qid, model="deepseek/deepseek-v4-flash", correct=True, cost=0.001, explore=False):
    return QuestionResult(
        qid, model, correct, cost, hashlib.sha256(qid.encode()).hexdigest(),
        prompt_tokens=500, completion_tokens=300, is_exploration=explore,
    )


HONEST_HOTKEY = "5HonestMinerHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
ATTACKER_HOTKEY = "5AttackerHotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"


def _proof(results, mrtd=HONEST_MRTD, weights_hash=HEAD_HASH,
           nonce="n" * 64, qhash=None, total=None, rtmr3=b"\x00" * 48,
           hotkey=HONEST_HOTKEY):
    per_model: dict[str, float] = {}
    for r in results:
        per_model[r.routed_model] = per_model.get(r.routed_model, 0.0) + r.cost_usd
    p = BenchmarkProof(
        epoch_id="e00000001", nonce=nonce,
        questions_hash=qhash if qhash is not None else compute_questions_hash(SLICE),
        weights_hash=weights_hash, source_hash="a" * 64, results=results,
        total_cost_usd=total if total is not None else sum(r.cost_usd for r in results),
        per_model_costs=per_model if total is None else {"deepseek/deepseek-v4-flash": total},
        attestation_quote=b"", timestamp=1.0, hotkey=hotkey,
    )
    p.attestation_quote = _quote(bytes.fromhex(p.content_hash()), mrtd, rtmr3)
    return p


def _verify(proof, **over):
    kw = dict(
        approved_measurements=APPROVED,
        expected_questions_hash=compute_questions_hash(SLICE),
        expected_nonce="n" * 64,
        gold_answers={k: POOL_GOLD[k] for k in SLICE},
        expected_question_ids=set(SLICE),
        expected_weights_hash=HEAD_HASH,
        expected_hotkey=HONEST_HOTKEY,
        head_bytes=HEAD,
        mock=False,
    )
    kw.update(over)
    kw.setdefault("expected_proof_hash", proof.content_hash())
    return verify_mod.verify_proof(proof, **kw)


# --- the attacks -------------------------------------------------------------

def a_relayed_proof():
    """Serve an honest miner's proof as your own.

    The proof is genuine: real hardware, real epoch, real slice, and its
    weights_hash matches a commitment the attacker is free to make by copying
    it off chain. The head is public — it is served inline by the honest miner
    to every validator that asks. Nothing here is forged.

    Before the hotkey was bound into content_hash this PASSED verification, and
    what stopped it was dedup plus commit-block seniority: a cosine threshold
    and a block ordering. This is the case that made that a cryptographic
    property instead.
    """
    honest = _proof([_qr(q) for q in SLICE], hotkey=HONEST_HOTKEY)
    return _verify(honest, expected_hotkey=ATTACKER_HOTKEY)


def a_rewritten_hotkey():
    """Relay, then rewrite the field to your own hotkey.

    The obvious follow-up to the above, and it must fail differently: the quote
    was signed over the original content_hash, so changing the hotkey breaks
    report_data. If this ever reports EXPLOITED, the hotkey has fallen out of
    content_hash and the binding is decorative.
    """
    honest = _proof([_qr(q) for q in SLICE], hotkey=HONEST_HOTKEY)
    honest.hotkey = ATTACKER_HOTKEY
    return _verify(honest, expected_hotkey=ATTACKER_HOTKEY)


def a_modified_image():
    """Run a tampered harness inside a genuine TDX VM."""
    return _verify(_proof([_qr(q) for q in SLICE], mrtd=ATTACKER_MRTD))


def a_substituted_questions():
    """Grade 10 easy questions while claiming the assigned slice."""
    easy = [_qr(f"q{900 + i}") for i in range(10)]
    gold = {k: POOL_GOLD[k] for k in SLICE}
    gold.update({f"q{900 + i}": {} for i in range(10)})
    return _verify(_proof(easy), gold_answers=gold)


def a_understated_cost():
    """Spend $5, report $0.0001 — understating cost raises the thrift term."""
    return _verify(_proof([_qr(q, "exp", True, 0.50) for q in SLICE], total=0.0001))


def a_head_swap():
    """Commit head A on-chain, actually run head B."""
    return _verify(_proof([_qr(q) for q in SLICE], weights_hash="ff" * 32))


def a_bundle_swap():
    """Advertise one proof over the axon, serve another in the bundle."""
    return _verify(_proof([_qr(q) for q in SLICE]), expected_proof_hash="0" * 64)


def a_head_not_in_bundle():
    """Ship a head that is not the one the proof attests to."""
    return _verify(_proof([_qr(q) for q in SLICE]), head_bytes=b"a different head")


def a_post_attestation_tamper():
    """Edit an attested field after the quote is generated."""
    p = _proof([_qr(q) for q in SLICE])
    p.results[0].correct = True
    p.results[0].cost_usd = 0.0
    return _verify(p)


def a_replayed_proof():
    """Resubmit last epoch's proof."""
    return _verify(_proof([_qr(q) for q in SLICE], nonce="old" + "0" * 61))


def a_skipped_exploration():
    """Omit the exploration quota to save ~5% of inference cost."""
    explore = {"q900": "deepseek/deepseek-v4-flash", "q901": "deepseek/deepseek-v4-flash"}
    gold = {k: POOL_GOLD[k] for k in SLICE}
    gold.update({"q900": {}, "q901": {}})
    return _verify(_proof([_qr(q) for q in SLICE]),
                   expected_exploration=explore, gold_answers=gold)


def a_redirected_exploration():
    """Explore, but route to a model of the miner's choosing."""
    explore = {"q900": "b/expected", "q901": "b/expected"}
    results = [_qr(q) for q in SLICE]
    results += [_qr("q900", "a/chosen-by-me", explore=True),
                _qr("q901", "a/chosen-by-me", explore=True)]
    gold = {k: POOL_GOLD[k] for k in SLICE}
    gold.update({"q900": {}, "q901": {}})
    return _verify(_proof(results), expected_exploration=explore, gold_answers=gold)


def a_duplicate_results():
    """Pad the result list to inflate the scored count."""
    return _verify(_proof([_qr(q) for q in SLICE] + [_qr(SLICE[0])]))


# --- the surface added with the runtime-identity work ------------------------
#
# Event logs, approved-entry parsing and RTMR3 replay all take untrusted miner
# input and were shipped with unit tests and no adversarial cases. These are the
# cases that matter: a log is believable ONLY because replaying it reproduces a
# register the CPU signed, so every attack here is a way of trying to be
# believed without reproducing it.

from fugal_subnet.tee.attestation import (  # noqa: E402
    dstack_event_digest,
    replay_event_log,
    runtime_identity,
)

APP_ID = runtime_identity("src", "pool", "grader", "https://openrouter.ai/api/v1")
_HONEST_LOG = [
    {"imr": 3, "event": "compose-hash", "event_payload": "abcd",
     "digest": dstack_event_digest("compose-hash", bytes.fromhex("abcd"))},
    {"imr": 3, "event": "instance-id", "event_payload": "beef",
     "digest": dstack_event_digest("instance-id", bytes.fromhex("beef"))},
]
_HONEST_RTMR3 = bytes.fromhex(replay_event_log(_HONEST_LOG)[0])
BASE = next(iter(APPROVED))
APPROVED_APP = {f"{BASE}:abcd"}


def _verify_log(log, rtmr3=_HONEST_RTMR3, approved=None):
    proof = _proof([_qr(q) for q in SLICE], rtmr3=rtmr3)
    return _verify(proof, approved_measurements=approved or APPROVED_APP,
                   event_log=log)


def a_forged_event_log():
    """Claim an approved compose hash in a log that does not reproduce RTMR3."""
    forged = [dict(_HONEST_LOG[0]),
              {"imr": 3, "event": "instance-id", "event_payload": "0bad",
               "digest": dstack_event_digest("instance-id", bytes.fromhex("0bad"))}]
    return _verify_log(forged)


def a_reordered_event_log():
    """Extends are not commutative. Reorder to try to reach the target value."""
    return _verify_log(list(reversed(_HONEST_LOG)))


def a_truncated_event_log():
    """Drop the trailing entries so the chain stops on a chosen register."""
    return _verify_log(_HONEST_LOG[:1])


def a_lying_preimage():
    """A v2 preimage that does not hash to the digest it accompanies."""
    log = [dict(_HONEST_LOG[0], preimage="00" * 16), dict(_HONEST_LOG[1])]
    return _verify_log(log)


def a_wrong_width_digest():
    """A SHA256 digest where the register needs SHA384."""
    log = [dict(_HONEST_LOG[0], digest="ab" * 32), dict(_HONEST_LOG[1])]
    return _verify_log(log)


def a_unapproved_app_in_a_valid_log():
    """A log that replays correctly but names an app nobody approved.

    The one that matters most: the chain is honest, the CPU signed it, and the
    application is still not ours. This is the check the whole locked-image
    argument exists to make.
    """
    other = [
        {"imr": 3, "event": "compose-hash", "event_payload": "dead",
         "digest": dstack_event_digest("compose-hash", bytes.fromhex("dead"))},
        dict(_HONEST_LOG[1]),
    ]
    rtmr3 = bytes.fromhex(replay_event_log(other)[0])
    return _verify_log(other, rtmr3=rtmr3)


def a_smuggled_colon_in_the_approved_entry():
    """A trailing colon must not silently degrade to an image-only entry.

    Not a miner attack — the operator owns the approved list. It is a
    CONFIGURATION failure that silently disables a security check: an unexpanded
    shell variable or a stray copy-paste leaves '<base>:' and the app binding
    quietly stops being enforced while the operator believes it is on. Rejecting
    the whole verification is the correct outcome, so a raise counts as blocked.
    """
    proof = _proof([_qr(q) for q in SLICE], rtmr3=b"\xff" * 48)
    try:
        return _verify(proof, approved_measurements={f"{BASE}:"}, event_log=None)
    except ValueError as e:
        return verify_mod.VerifyResult(False, f"rejected malformed entry: {e}")


def a_oversized_event_log():
    """Allocation through the log (I2). Bounded at the protocol edge, but the
    replay must not be quadratic or a crash on a large one."""
    big = [dict(_HONEST_LOG[0]) for _ in range(20_000)]
    return _verify_log(big)


def a_single_extend_identity_mismatch():
    """No log supplied: the register must still replay from an approved identity."""
    proof = _proof([_qr(q) for q in SLICE], rtmr3=b"\x11" * 48)
    return _verify(proof, approved_measurements={f"{BASE}:{APP_ID}"}, event_log=None)


ATTACKS = [
    ("modified harness in a real TDX VM", "I8", a_modified_image),
    ("answer easy questions, claim the slice", "I3", a_substituted_questions),
    ("spend $5, report $0.0001", "I3", a_understated_cost),
    ("commit head A, run head B", "I3", a_head_swap),
    ("serve a bundle other than the one advertised", "I8", a_bundle_swap),
    ("bundle a head the proof does not attest", "I8", a_head_not_in_bundle),
    ("edit the proof after attestation", "I8", a_post_attestation_tamper),
    ("relay an honest miner's proof as your own", "I8", a_relayed_proof),
    ("relay a proof with the hotkey rewritten", "I8", a_rewritten_hotkey),
    ("replay a previous epoch's proof", "I3", a_replayed_proof),
    ("skip the exploration quota", "I3", a_skipped_exploration),
    ("redirect exploration to a chosen model", "I3", a_redirected_exploration),
    ("pad the result list with duplicates", "I3", a_duplicate_results),
    ("forge an event log for an approved app", "I8", a_forged_event_log),
    ("reorder the event log to hit a target", "I8", a_reordered_event_log),
    ("truncate the event log", "I8", a_truncated_event_log),
    ("v2 preimage that lies about its digest", "I8", a_lying_preimage),
    ("SHA256 digest where SHA384 is required", "I8", a_wrong_width_digest),
    ("valid log, unapproved application", "I8", a_unapproved_app_in_a_valid_log),
    ("smuggle a colon to drop the app requirement", "I8",
     a_smuggled_colon_in_the_approved_entry),
    ("20k-entry event log", "I2", a_oversized_event_log),
    ("single-extend identity mismatch", "I8", a_single_extend_identity_mismatch),
]


def _control():
    """The honest path must still verify — a verifier that rejects everything
    blocks every attack and is worthless."""
    r = _verify(_proof([_qr(q) for q in SLICE]))
    return {
        "name": "control: honest proof verifies", "invariant": "—",
        "verdict": "CONTROL" if r.valid else "BROKEN",
        "detail": r.reason, "secs": 0.0,
    }


def main() -> int:
    # The attacker owns real TDX hardware: their quote is genuinely Intel-signed.
    verify_mod.verify_dcap = lambda q, **_kw: True  # **_kw: keeps matching
    # verify_dcap's real signature as it grows (it gained `budget` for the
    # epoch collateral ceiling). Without it the stub raises TypeError, which
    # verify_proof catches and reports as a failed proof -- so the CONTROL
    # breaks while all 22 attacks still show BLOCKED, which is precisely the
    # shape of a suite that has stopped testing anything.

    results = []
    for name, inv, fn in ATTACKS:
        t0 = time.time()
        try:
            r = fn()
            verdict = "EXPLOITED" if r.valid else "BLOCKED"
            detail = r.reason
        except Exception as e:  # a crash is not a defence
            verdict, detail = "EXPLOITED", f"raised {type(e).__name__}: {e}"
        results.append({"name": name, "invariant": inv, "verdict": verdict,
                        "detail": detail, "secs": time.time() - t0})
    results.append(_control())

    print()
    print(f"{'TEE ATTACK SUITE':<50}{'INV':<5}{'VERDICT':<11}{'SECS':>6}")
    print("-" * 78)
    for r in results:
        flag = "OK" if r["verdict"] in ("BLOCKED", "CONTROL") else "!!"
        print(f"{flag} {r['name'][:47]:<47}{r['invariant']:<5}"
              f"{r['verdict']:<11}{r['secs']:>6.2f}")
        if r["verdict"] == "EXPLOITED":
            print(f"     ↳ verified anyway: {r['detail']}")
    print("-" * 78)

    blocked = sum(1 for r in results if r["verdict"] == "BLOCKED")
    exploited = [r for r in results if r["verdict"] == "EXPLOITED"]
    broken_control = [r for r in results if r["verdict"] == "BROKEN"]
    print(f"{blocked} blocked, {len(exploited)} EXPLOITED, "
          f"{len(broken_control)} broken controls")

    if exploited or broken_control:
        print("\nFAIL — the cases marked !! above verify when they must not.")
        return 1
    print("PASS — every attested claim is bound; the honest path still verifies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
