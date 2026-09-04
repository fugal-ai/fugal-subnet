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
import struct
import time

from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.attestation import measurement_id, parse_quote
from fugal_subnet.tee.proof import (
    BenchmarkProof,
    QuestionResult,
    compute_questions_hash,
)

HONEST_MRTD = bytes.fromhex("11" * 48)
ATTACKER_MRTD = bytes.fromhex("de" * 48)
SLICE = [f"q{i}" for i in range(10)]
POOL_GOLD = {f"q{i}": {"question_id": f"q{i}"} for i in range(1000)}
HEAD = b"the head that was committed on chain"
HEAD_HASH = hashlib.sha256(HEAD).hexdigest()


def _quote(report_data: bytes, mrtd: bytes) -> bytes:
    q = bytearray(632)
    struct.pack_into("<H", q, 0, 4)        # version 4
    struct.pack_into("<I", q, 4, 0x81)     # tee_type TDX
    q[184:232] = mrtd
    q[568:632] = report_data.ljust(64, b"\x00")[:64]
    return bytes(q)


APPROVED = {measurement_id(parse_quote(_quote(b"\x00" * 64, HONEST_MRTD)))}


def _qr(qid, model="cheap", correct=True, cost=0.001, explore=False):
    return QuestionResult(
        qid, model, correct, cost, hashlib.sha256(qid.encode()).hexdigest(),
        prompt_tokens=500, completion_tokens=300, is_exploration=explore,
    )


def _proof(results, mrtd=HONEST_MRTD, weights_hash=HEAD_HASH,
           nonce="n" * 64, qhash=None, total=None):
    per_model: dict[str, float] = {}
    for r in results:
        per_model[r.routed_model] = per_model.get(r.routed_model, 0.0) + r.cost_usd
    p = BenchmarkProof(
        epoch_id="e00000001", nonce=nonce,
        questions_hash=qhash if qhash is not None else compute_questions_hash(SLICE),
        weights_hash=weights_hash, source_hash="a" * 64, results=results,
        total_cost_usd=total if total is not None else sum(r.cost_usd for r in results),
        per_model_costs=per_model if total is None else {"cheap": total},
        attestation_quote=b"", timestamp=1.0,
    )
    p.attestation_quote = _quote(bytes.fromhex(p.content_hash()), mrtd)
    return p


def _verify(proof, **over):
    kw = dict(
        approved_measurements=APPROVED,
        expected_questions_hash=compute_questions_hash(SLICE),
        expected_nonce="n" * 64,
        gold_answers={k: POOL_GOLD[k] for k in SLICE},
        expected_question_ids=set(SLICE),
        expected_weights_hash=HEAD_HASH,
        head_bytes=HEAD,
        mock=False,
    )
    kw.update(over)
    kw.setdefault("expected_proof_hash", proof.content_hash())
    return verify_mod.verify_proof(proof, **kw)


# --- the attacks -------------------------------------------------------------

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
    explore = {"q900": "cheap", "q901": "cheap"}
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


ATTACKS = [
    ("modified harness in a real TDX VM", "I8", a_modified_image),
    ("answer easy questions, claim the slice", "I3", a_substituted_questions),
    ("spend $5, report $0.0001", "I3", a_understated_cost),
    ("commit head A, run head B", "I3", a_head_swap),
    ("serve a bundle other than the one advertised", "I8", a_bundle_swap),
    ("bundle a head the proof does not attest", "I8", a_head_not_in_bundle),
    ("edit the proof after attestation", "I8", a_post_attestation_tamper),
    ("replay a previous epoch's proof", "I3", a_replayed_proof),
    ("skip the exploration quota", "I3", a_skipped_exploration),
    ("redirect exploration to a chosen model", "I3", a_redirected_exploration),
    ("pad the result list with duplicates", "I3", a_duplicate_results),
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
    verify_mod.verify_dcap = lambda q: True

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
