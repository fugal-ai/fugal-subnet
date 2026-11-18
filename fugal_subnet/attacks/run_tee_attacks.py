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
from fugal_subnet.tee.attestation import (
    dstack_event_digest,
    measurement_id,
    parse_quote,
    replay_event_log,
    replay_rtmr,
)
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


BASE = measurement_id(parse_quote(_quote(b"\x00" * 64, HONEST_MRTD)))
# The live shape, and the only one: the approved entry is <base>:<compose
# hash>, and the proof carries the RTMR3 event log that names the compose hash
# and replays to the register in the quote. The control and every attack run
# through it, so the suite tests the verifier the way a live validator runs it.
COMPOSE = hashlib.sha256(b"the approved app-compose.json").hexdigest()
OTHER_COMPOSE = hashlib.sha256(b"a compose the miner wrote").hexdigest()


def _ev(name, payload_hex, version=2):
    return {"imr": 3, "event": name, "event_payload": payload_hex,
            "digest": dstack_event_digest(name, bytes.fromhex(payload_hex), version)}


_HONEST_LOG = [_ev("compose-hash", COMPOSE), _ev("instance-id", "beef")]
HONEST_RTMR3 = bytes.fromhex(replay_event_log(_HONEST_LOG)[0])
APPROVED = {f"{BASE}:{COMPOSE}"}


def _qr(qid, model="deepseek/deepseek-v4-flash", correct=True, cost=0.001, explore=False):
    return QuestionResult(
        qid, model, correct, cost, hashlib.sha256(qid.encode()).hexdigest(),
        prompt_tokens=500, completion_tokens=300, is_exploration=explore,
    )


HONEST_HOTKEY = "5HonestMinerHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
ATTACKER_HOTKEY = "5AttackerHotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"


def _proof(results, mrtd=HONEST_MRTD, weights_hash=HEAD_HASH,
           nonce="n" * 64, qhash=None, total=None, rtmr3=HONEST_RTMR3,
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
        event_log=_HONEST_LOG,
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


def a_blocked_exploration():
    """Drop the network during exploration only: every exploration call fails,
    the scored pass is untouched. Failed calls no longer count against the
    model in the reference frame, so this would otherwise be 5% of the
    inference bill saved for free (diligence 2026-10, finding 2)."""
    explore = {f"q{i}": "deepseek/deepseek-v4-flash" for i in range(900, 915)}
    results = [_qr(q) for q in SLICE]
    results += [
        QuestionResult(q, m, False, 0.0, hashlib.sha256(b"").hexdigest(),
                       prompt_tokens=0, completion_tokens=0, is_exploration=True,
                       failed=True)
        for q, m in explore.items()
    ]
    gold = {k: POOL_GOLD[k] for k in SLICE}
    gold.update({q: {} for q in explore})
    return _verify(_proof(results), expected_exploration=explore, gold_answers=gold)


def a_failed_but_correct():
    """Mark a scored answer failed AND correct: graded, but invisible to any
    failure accounting. The flag must mean one thing."""
    results = [_qr(q) for q in SLICE]
    results[0] = QuestionResult(SLICE[0], "deepseek/deepseek-v4-flash", True, 0.001,
                                hashlib.sha256(SLICE[0].encode()).hexdigest(),
                                prompt_tokens=500, completion_tokens=300, failed=True)
    return _verify(_proof(results))


def a_huge_token_count():
    """Carry a token count no call can produce: attested or not, it is refused
    at verification, before anything downstream computes with it (I2)."""
    results = [_qr(q) for q in SLICE]
    results[0] = QuestionResult(SLICE[0], "deepseek/deepseek-v4-flash", True, 0.001,
                                hashlib.sha256(SLICE[0].encode()).hexdigest(),
                                prompt_tokens=500, completion_tokens=10**400)
    return _verify(_proof(results))


def a_negative_token_count():
    results = [_qr(q) for q in SLICE]
    results[1] = QuestionResult(SLICE[1], "deepseek/deepseek-v4-flash", True, 0.001,
                                hashlib.sha256(SLICE[1].encode()).hexdigest(),
                                prompt_tokens=-5, completion_tokens=300)
    return _verify(_proof(results))


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

def _verify_log(log, rtmr3=HONEST_RTMR3, approved=None):
    proof = _proof([_qr(q) for q in SLICE], rtmr3=rtmr3)
    return _verify(proof, approved_measurements=approved or APPROVED,
                   event_log=log)


def a_forged_event_log():
    """Claim an approved compose hash in a log that does not reproduce RTMR3."""
    forged = [dict(_HONEST_LOG[0]),
              _ev("instance-id", "0bad")]
    return _verify_log(forged)


def a_relabelled_compose_payload():
    """Keep every genuine digest; rewrite only the compose-hash PAYLOAD to the
    approved value. The chain of digests still reproduces RTMR3, so a replay
    that checks only the register reads the relabelled field and approves an
    application nobody approved. This VERIFIED on the real fixtures until the
    replay bound each digest to its (name, payload) (2026-11-05)."""
    other = [_ev("compose-hash", OTHER_COMPOSE), dict(_HONEST_LOG[1])]
    rtmr3 = bytes.fromhex(replay_event_log(other)[0])
    forged = [dict(other[0], event_payload=COMPOSE), other[1]]   # approved value, genuine digest
    return _verify_log(forged, rtmr3=rtmr3)


def a_relabelled_compose_payload_v1_encoding():
    """The same relabel against a log in dstack's v1 event encoding, which is
    what the real GCP fixtures use (tests/fixtures/attestation_*.bin)."""
    other = [_ev("compose-hash", OTHER_COMPOSE, 1), _ev("instance-id", "beef", 1)]
    rtmr3 = bytes.fromhex(replay_event_log(other)[0])
    forged = [dict(other[0], event_payload=COMPOSE), other[1]]
    return _verify_log(forged, rtmr3=rtmr3)


def a_second_compose_hash_event():
    """Extend a second `compose-hash` event, with the approved value, AFTER the
    initrd's genuine one (the guest agent's EmitEvent on a newer base). Every
    digest is genuine and the log replays; a last-wins read would believe the
    application's own claim. A name that appears twice is refused."""
    log = [_ev("compose-hash", OTHER_COMPOSE), dict(_HONEST_LOG[1]),
           _ev("compose-hash", COMPOSE)]
    rtmr3 = bytes.fromhex(replay_rtmr([e["digest"] for e in log]))
    return _verify_log(log, rtmr3=rtmr3)


def a_resplit_compose_hash_name():
    """The v1 encoding joins type, name and payload with ':'. A miner varies its
    own compose until the hash starts with ':', then presents the genuine
    compose-hash digest as the name 'compose-hash:' over the remaining bytes,
    so the approved value it extended later is the only 'compose-hash'. Same
    digests, same register; only the name rule refuses it."""
    i = 0
    while (mine := hashlib.sha256(b'{"compose": %d}' % i).digest())[0] != ord(":"):
        i += 1
    genuine = [_ev("compose-hash", mine.hex(), 1), _ev("instance-id", "beef", 1),
               _ev("compose-hash", COMPOSE, 1)]
    rtmr3 = bytes.fromhex(replay_rtmr([e["digest"] for e in genuine]))
    forged = [dict(genuine[0], event="compose-hash:", event_payload=mine[1:].hex()),
              genuine[1], genuine[2]]
    return _verify_log(forged, rtmr3=rtmr3)


def a_reordered_event_log():
    """Extends are not commutative. Reorder to try to reach the target value."""
    return _verify_log(list(reversed(_HONEST_LOG)))


def a_truncated_event_log():
    """Drop the trailing entries so the chain stops on a chosen register."""
    return _verify_log(_HONEST_LOG[:1])


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
    other = [_ev("compose-hash", OTHER_COMPOSE), dict(_HONEST_LOG[1])]
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
    big = [_ev(f"e{i}", "00") for i in range(20_000)]
    return _verify_log(big)


def a_runtime_identity_entry():
    """An approved entry whose app half is a 96-hex runtime identity, and a
    bare quote whose RTMR3 is one extend of exactly that value. Nothing in the
    image measures who wrote the extend, so the miner's own code writes the
    expected value. Only a compose hash replayed out of the log is evidence."""
    identity = hashlib.sha384(b"src|pool|grader|upstream").hexdigest()
    proof = _proof([_qr(q) for q in SLICE], rtmr3=bytes.fromhex(replay_rtmr([identity])))
    return _verify(proof, approved_measurements={f"{BASE}:{identity}"}, event_log=None)


def a_proof_without_an_event_log():
    """A bare quote with the honest register and no log to replay it from."""
    proof = _proof([_qr(q) for q in SLICE])
    return _verify(proof, event_log=None)


def a_bare_base_approves_any_app():
    """Run any application on an approved image under a bare base entry.

    The application is what fixes the model upstream, so a bare base lets the
    miner choose it: perfect accuracy at two tokens of attested cost with the
    DCAP signature, the image measurement, the question hash, the nonce and
    the head hash all intact (diligence 2026-10, finding 1). This case
    VERIFIED until bare bases were refused outside mock.
    """
    proof = _proof([_qr(q) for q in SLICE], rtmr3=b"\xde" * 48)
    return _verify(proof, approved_measurements={BASE}, event_log=None)



def _rogue_chain(ca_name):
    """A CA the attacker made, named however they like, and an AK leaf it signed."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    now = datetime.datetime.now(datetime.timezone.utc)

    def cert(subject, issuer, key, signer, ca):
        return (x509.CertificateBuilder()
                .subject_name(subject).issuer_name(issuer)
                .public_key(key.public_key()).serial_number(1)
                .not_valid_before(now - datetime.timedelta(days=1))
                .not_valid_after(now + datetime.timedelta(days=365))
                .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
                .sign(signer, hashes.SHA256())
                .public_bytes(serialization.Encoding.DER))

    ca_key, ak_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    leaf_name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "attacker AK")])
    return (cert(leaf_name, ca_name, ak_key, ca_key, ca=False),
            cert(ca_name, ca_name, ca_key, ca_key, ca=True))


def _chain_verdict(leaf, intermediates):
    """What a proof carrying these intermediates gets from the chain check that
    verify_proof runs inside unwrap_attestation (tpm.verify_ak_chain)."""
    from fugal_subnet.tee.tpm import TpmError, verify_ak_chain

    try:
        ok = verify_ak_chain(leaf, extra_certs=intermediates)
    except TpmError as e:
        return verify_mod.VerifyResult(False, f"TpmError: {e}")
    return verify_mod.VerifyResult(ok, "chained to the pinned root" if ok
                                   else "did not chain to the pinned root")


def a_rogue_tpm_intermediate():
    """Carry your own CA in the proof's tpm_intermediates, named exactly like
    Google's EK/AK intermediate, and an AK certificate it signed.

    Proof-carried intermediates exist so an unvendored region verifies; they
    must not become a way to bring your own trust anchor. The chain walk has to
    reach the pinned root by signature, not by name.
    """
    from cryptography import x509

    from fugal_subnet.tee.tpm import _load_pinned

    _root, vendored = _load_pinned()
    name = vendored[0].subject if vendored else x509.Name(
        [x509.NameAttribute(x509.NameOID.COMMON_NAME, "EK/AK CA Intermediate")])
    leaf, rogue_ca = _rogue_chain(name)
    return _chain_verdict(leaf, [rogue_ca])


def a_supplied_cert_named_as_the_root():
    """Carry a self-signed CA whose subject is the pinned ROOT's name. A walker
    that resolved issuers by name alone would stop at it and call it the root."""
    from fugal_subnet.tee.tpm import _load_pinned

    root, _ = _load_pinned()
    leaf, rogue_root = _rogue_chain(root.subject)
    return _chain_verdict(leaf, [rogue_root])


def _throwaway_chain(*, intermediate_ca=True, intermediate_expired=False):
    """root -> intermediate -> AK, all the attacker's, every signature genuine.
    Returns (root certificate, intermediate DER, leaf DER). Nobody holds a
    Google key, so the rules beyond "signed by the root" can only be broken
    one at a time under a root that stands in for the pinned one."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    now = datetime.datetime.now(datetime.timezone.utc)
    day = datetime.timedelta(days=1)

    def cn(name):
        return x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, name)])

    def cert(subject, issuer, key, signer, ca, until):
        return (x509.CertificateBuilder()
                .subject_name(cn(subject)).issuer_name(cn(issuer))
                .public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - 30 * day).not_valid_after(until)
                .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
                .sign(signer, hashes.SHA256()))

    root_key, mid_key, ak_key = (ec.generate_private_key(ec.SECP256R1()) for _ in range(3))
    root = cert("stand-in root", "stand-in root", root_key, root_key, True, now + 365 * day)
    mid = cert("EK/AK CA Intermediate", "stand-in root", mid_key, root_key, intermediate_ca,
               now - day if intermediate_expired else now + 365 * day)
    leaf = cert("AK", "EK/AK CA Intermediate", ak_key, mid_key, False, now + 365 * day)
    der = serialization.Encoding.DER
    return root, mid.public_bytes(der), leaf.public_bytes(der)


def _under_stand_in_root(**broken):
    """The chain verdict for a correctly signed chain that breaks one rule,
    with the attacker's root standing in for the pinned one. The same chain
    with nothing broken must verify first, or the case shows nothing."""
    from fugal_subnet.tee import tpm

    real = tpm._load_pinned
    try:
        root, mid, leaf = _throwaway_chain()
        tpm._load_pinned = lambda *a, **k: (root, [])
        if not tpm.verify_ak_chain(leaf, extra_certs=[mid]):
            return verify_mod.VerifyResult(True, "the case did not run: its unbroken chain is refused")
        root, mid, leaf = _throwaway_chain(**broken)
        tpm._load_pinned = lambda *a, **k: (root, [])
        return _chain_verdict(leaf, [mid])
    finally:
        tpm._load_pinned = real


def a_non_ca_certificate_as_intermediate():
    """Carry, as the intermediate, a certificate the root really signed but
    that may not issue (a leaf). Its signature on the attestation key is
    genuine; only the CA rule refuses it. One honest box would otherwise let
    its owner mint attestation keys for machines they do not have."""
    return _under_stand_in_root(intermediate_ca=False)


def a_expired_intermediate():
    """Carry an intermediate whose signatures all verify and whose validity
    has ended (a retired CA key is the one most likely to have leaked)."""
    return _under_stand_in_root(intermediate_expired=True)


def _leaf_with_issuer_address(url):
    """An attestation-key certificate under the attacker's CA, named like
    Google's intermediate, whose issuer address (AIA caIssuers) is `url`.
    Returns (leaf DER, the attacker's CA DER)."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import AuthorityInformationAccessOID

    from fugal_subnet.tee.tpm import _load_pinned

    name = _load_pinned()[1][0].subject
    ca_key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.timezone.utc)
    ca = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
          .public_key(ca_key.public_key()).serial_number(2)
          .not_valid_before(now - datetime.timedelta(days=1))
          .not_valid_after(now + datetime.timedelta(days=365))
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
          .sign(ca_key, hashes.SHA256()))
    leaf = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "attacker AK")]))
            .issuer_name(name).public_key(ec.generate_private_key(ec.SECP256R1()).public_key())
            .serial_number(3)
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.AuthorityInformationAccess([x509.AccessDescription(
                AuthorityInformationAccessOID.CA_ISSUERS,
                x509.UniformResourceIdentifier(url))]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    der = serialization.Encoding.DER
    return leaf.public_bytes(der), ca.public_bytes(der)


_A_GOOGLE_CA_ADDRESS = ("http://privateca-content-00000000-0000-0000-0000-000000000000"
                        ".storage.googleapis.com/0000/ca.crt")


def a_own_ca_published_at_a_google_ca_address():
    """Stand up a CA on Google's own CA service, name it like Google's
    intermediate, and point the attestation key's issuer address at it. The
    chain builders (the miner's at startup, the operator's before it hands a
    TD its key) fetch the issuer from that address: fetched bytes must be
    believed only if they chain to the pinned root."""
    from fugal_subnet.tee.tpm import intermediates_for

    leaf, own_ca = _leaf_with_issuer_address(_A_GOOGLE_CA_ADDRESS)
    asked = []
    chain = intermediates_for(leaf, fetch=lambda url: asked.append(url) or own_ca)
    if asked != [_A_GOOGLE_CA_ADDRESS]:
        return verify_mod.VerifyResult(True, f"the case did not run: fetched {asked}")
    if chain:
        return verify_mod.VerifyResult(True, "the fetched CA was carried as an intermediate")
    return _chain_verdict(leaf, [own_ca])


def a_issuer_address_aimed_at_the_operator():
    """Answer the operator's check with a certificate whose issuer address is
    a cloud metadata service, a local file, or a port on the operator's own
    machine, and let its chain builder make the request for you."""
    import urllib.request

    from fugal_subnet.tee import tpm

    opened = []

    def refuse(*a, **k):
        opened.append(a)
        raise OSError("refused by the attack suite")

    real = urllib.request.urlopen, urllib.request.OpenerDirector.open
    urllib.request.urlopen, urllib.request.OpenerDirector.open = refuse, refuse
    try:
        for url in ("http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
                    "file:///etc/passwd", "http://127.0.0.1:8092/provision/logs",
                    "http://privateca-content-00000000-0000-0000-0000-000000000000"
                    ".storage.googleapis.com.attacker.example/ca.crt"):
            leaf, _ca = _leaf_with_issuer_address(url)
            if tpm.intermediates_for(leaf, fetch=tpm.fetch_google_ca_issuer):
                return verify_mod.VerifyResult(True, f"a chain was built from {url}")
    finally:
        urllib.request.urlopen, urllib.request.OpenerDirector.open = real
    if opened:
        return verify_mod.VerifyResult(True, f"the operator's machine was sent to {opened[0]!r}")
    return verify_mod.VerifyResult(False, "no address outside Google's CA content was asked")


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
    ("block the network during exploration only", "I9", a_blocked_exploration),
    ("a failed call that is also correct", "I9", a_failed_but_correct),
    ("forge an event log for an approved app", "I8", a_forged_event_log),
    ("relabel the compose-hash payload, keep the digests", "I8", a_relabelled_compose_payload),
    ("relabel the payload in the v1 event encoding", "I8", a_relabelled_compose_payload_v1_encoding),
    ("extend a second compose-hash event after boot", "I8", a_second_compose_hash_event),
    ("re-split the compose-hash digest into another name", "I8", a_resplit_compose_hash_name),
    ("reorder the event log to hit a target", "I8", a_reordered_event_log),
    ("truncate the event log", "I8", a_truncated_event_log),
    ("SHA256 digest where SHA384 is required", "I8", a_wrong_width_digest),
    ("valid log, unapproved application", "I8", a_unapproved_app_in_a_valid_log),
    ("smuggle a colon to drop the app requirement", "I8",
     a_smuggled_colon_in_the_approved_entry),
    ("20k-entry event log", "I2", a_oversized_event_log),
    ("a runtime-identity app half, extended by the miner", "I8", a_runtime_identity_entry),
    ("a bare quote with no event log", "I8", a_proof_without_an_event_log),
    ("bare base entry, miner-chosen application", "I8", a_bare_base_approves_any_app),
    ("rogue TPM intermediate carried in the proof", "I8", a_rogue_tpm_intermediate),
    ("supplied cert named as the pinned root", "I8", a_supplied_cert_named_as_the_root),
    ("a certificate that may not issue, as intermediate", "I8", a_non_ca_certificate_as_intermediate),
    ("an expired intermediate carried in the proof", "I8", a_expired_intermediate),
    ("own CA published at a Google CA address", "I8", a_own_ca_published_at_a_google_ca_address),
    ("issuer address aimed at the operator's machine", "I8", a_issuer_address_aimed_at_the_operator),
    ("a token count no call can produce", "I2", a_huge_token_count),
    ("a negative token count", "I2", a_negative_token_count),
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
