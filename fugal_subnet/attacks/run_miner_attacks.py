#!/usr/bin/env python3
"""Adversarial miner harness — invariants I2 and I6.

The grader attack suite (`run_attacks.py`) covers hostile *model output*. This
covers hostile *miner input*, which is the surface a registered miner actually
controls: the bytes returned from its axon, the model IDs it declares, and how
long it takes to answer.

Two invariants:

  I2 — bounded ingestion. No miner-supplied bytes reach deserialization,
       allocation, or execution without size, shape, and value bounds. A
       violation is remote code execution or an OOM.
  I6 — liveness. No miner behavior stops the validator completing an epoch and
       setting weights. A violation means one miner can halt everyone's
       emissions, which is cheap griefing with a large payoff.

Each case states what a miner would gain, so a failure reads as an exploit
rather than a failed assertion. Verdicts:

  BLOCKED — the attack was rejected (the expected result)
  SURVIVED — the attack was tolerated safely without rejection, and is benign
  BROKEN — the validator would accept or hang. A real finding.

Run: python -m fugal_subnet.attacks.run_miner_attacks
"""
from __future__ import annotations

import base64
import io
import time
import zipfile

import numpy as np

from fugal_subnet.config import HEAD_HIDDEN_DIM
from fugal_subnet.head_eval import load_head_from_b64, load_head_from_npz

# A head large enough to be realistic but cheap to build.
_DIM = HEAD_HIDDEN_DIM


def _npz_bytes(W, b, models, models_dtype="U100") -> bytes:
    """Serialize a bare three-array (W, b, models) archive.

    Heads are saved by scripts/train_head.py's main(), as
    `np.savez(path, **success_contract.make_head(...))`, and only such
    success-contract heads load: head_eval.load_head_from_npz refuses any
    archive without the `contract` member before reading anything in it.
    These payloads omit it, so each is refused at that gate; they stay as a
    regression that no miner-controlled array in them is ever parsed (I2).

    The dtype matters: an object-dtype models array is rejected outright by
    allow_pickle=False, which would block every payload here for a reason that
    has nothing to do with the defense under test — a suite that passes while
    proving nothing. Only atk_object_array uses object dtype, deliberately.
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, W=W, b=b, models=np.array(models, dtype=models_dtype))
    return buf.getvalue()


def _valid_head(n_models: int = 4) -> bytes:
    """An honest success-contract head: the control every payload differs from."""
    from fugal_subnet import success_test_fixtures as fx

    rng = np.random.RandomState(0)
    W = (rng.randn(n_models, _DIM) * 0.01).astype(np.float32)
    b = np.zeros(n_models, dtype=np.float32)
    return fx.head_bytes(W, b, [f"vendor/m{i}" for i in range(n_models)])


# ── Attack constructors ────────────────────────────────────────────────────
# Each returns raw bytes a hostile miner could serve from its axon.

def atk_zip_bomb() -> bytes:
    """Tiny compressed payload that expands to gigabytes — OOM the validator."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("W.npy", b"\0" * (200 * 1024 * 1024))
    return buf.getvalue()


def atk_nan_weights() -> bytes:
    """NaN in W. Poisons argmax and can propagate into every miner's score."""
    W = np.full((4, _DIM), np.nan, dtype=np.float32)
    return _npz_bytes(W, np.zeros(4, dtype=np.float32), [f"v/m{i}" for i in range(4)])


def atk_inf_bias() -> bytes:
    """+inf bias — force selection of one model regardless of the question."""
    W = np.zeros((4, _DIM), dtype=np.float32)
    b = np.array([np.inf, 0, 0, 0], dtype=np.float32)
    return _npz_bytes(W, b, [f"v/m{i}" for i in range(4)])


def atk_too_many_models() -> bytes:
    """Declare far past the model cap to blow up evaluation cost."""
    n = 5000
    W = np.zeros((n, _DIM), dtype=np.float32)
    return _npz_bytes(W, np.zeros(n, dtype=np.float32), [f"v/m{i}" for i in range(n)])


def atk_wrong_hidden_dim() -> bytes:
    """Mismatched hidden dim — shape confusion in the matmul."""
    W = np.zeros((4, _DIM * 2), dtype=np.float32)
    return _npz_bytes(W, np.zeros(4, dtype=np.float32), [f"v/m{i}" for i in range(4)])


def atk_shape_mismatch() -> bytes:
    """W and b disagree on model count."""
    W = np.zeros((4, _DIM), dtype=np.float32)
    return _npz_bytes(W, np.zeros(9, dtype=np.float32), [f"v/m{i}" for i in range(4)])


def atk_1d_weights() -> bytes:
    """W is 1-D rather than a matrix."""
    W = np.zeros(_DIM, dtype=np.float32)
    return _npz_bytes(W, np.zeros(1, dtype=np.float32), ["v/m0"])


def atk_object_array() -> bytes:
    """Object-dtype array — the classic pickle deserialization vector."""
    buf = io.BytesIO()
    obj = np.array([{"exploit": "payload"}], dtype=object)
    np.savez_compressed(buf, W=obj, b=obj, models=np.array(["v/m0"], dtype=object))
    return buf.getvalue()


def atk_garbage() -> bytes:
    """Not an npz at all."""
    return b"\x00\x01\x02not-a-zip-archive" * 500


def atk_empty() -> bytes:
    """Zero bytes."""
    return b""


def atk_missing_arrays() -> bytes:
    """Valid npz, none of the required keys."""
    buf = io.BytesIO()
    np.savez_compressed(buf, unrelated=np.zeros(4, dtype=np.float32))
    return buf.getvalue()


def atk_huge_model_ids() -> bytes:
    """Megabyte-long model ID strings — memory amplification via metadata."""
    W = np.zeros((2, _DIM), dtype=np.float32)
    return _npz_bytes(W, np.zeros(2, dtype=np.float32),
                      ["v/" + "a" * 1_000_000] * 2, models_dtype="U1000001")


def atk_oversize_head() -> bytes:
    """Past the raw byte cap, incompressible so the cap is what must catch it."""
    rng = np.random.RandomState(1)
    W = rng.randn(600, _DIM).astype(np.float32)
    return _npz_bytes(W, np.zeros(600, dtype=np.float32), [f"v/m{i}" for i in range(600)])


ATTACKS = [
    ("zip bomb (200MB expansion)", atk_zip_bomb, "OOM the validator", "I2"),
    ("object-dtype array (pickle vector)", atk_object_array, "RCE on the validator", "I2"),
    ("NaN weights", atk_nan_weights, "poison scoring for everyone", "I2"),
    ("+inf bias", atk_inf_bias, "force a fixed routing choice", "I2"),
    ("5000 declared models", atk_too_many_models, "blow up evaluation cost", "I2"),
    ("wrong hidden dim", atk_wrong_hidden_dim, "shape confusion in matmul", "I2"),
    ("W/b shape mismatch", atk_shape_mismatch, "index out of bounds", "I2"),
    ("1-D W", atk_1d_weights, "shape confusion", "I2"),
    ("garbage bytes", atk_garbage, "parser crash", "I2"),
    ("empty payload", atk_empty, "parser crash", "I2"),
    ("npz missing required arrays", atk_missing_arrays, "KeyError crash", "I2"),
    ("1MB model ID strings", atk_huge_model_ids, "memory amplification", "I2"),
    ("oversize head past byte cap", atk_oversize_head, "OOM the validator", "I2"),
]


# ── Protocol v4 heads (docs/decisions/LIVE_MENU_V4.md, section 11) ──────────
# A v4 head is today's format with rows keyed by entry id and the cost fields
# optional (head_eval.load_head_v4). Every head above is run through the v4
# loader too; these are the v4 cases on top: the rows a v4 head may carry,
# and a head without the cost fields whose arrays are hostile.


def _v4_head(n_rows: int = 4, *, ids=None, W=None, b=None) -> bytes:
    """A v4 head without token means or a cost profile, with the given rows."""
    from fugal_subnet.vendor import success_contract as c

    ids = ids if ids is not None else [f"lab/e{i:03d}" for i in range(n_rows)]
    n = len(ids)
    z = dict(W=np.zeros((n, _DIM), dtype=np.float32) if W is None else W,
             b=np.zeros(n, dtype=np.float32) if b is None else b,
             models=np.asarray(ids), contract=np.asarray(c.CONTRACT),
             profile_id=np.asarray(c.PROFILE_ID), backbone_revision=np.asarray(c.REVISION),
             provenance=np.asarray("attack"), lam=np.asarray(1.0))
    buf = io.BytesIO()
    np.savez(buf, **z)
    return buf.getvalue()


def atk_v4_rows_past_the_bound() -> bytes:
    """One row more than a v4 head may carry: more odds to compute per question."""
    from fugal_subnet.head_eval import HEAD_MAX_ROWS_V4
    return _v4_head(HEAD_MAX_ROWS_V4 + 1)


def atk_v4_oversize() -> bytes:
    """Rows within the bound, and a string field that carries the archive
    past the byte cap (the rows alone cannot: 72 float64 rows are 590 KB)."""
    z = dict(np.load(io.BytesIO(_v4_head(4)), allow_pickle=False))
    z["provenance"] = np.asarray("p" * 300_000)
    buf = io.BytesIO()
    np.savez(buf, **z)
    return buf.getvalue()


def atk_v4_nan_row() -> bytes:
    """A NaN in one row: the exact odds of that entry are undefined."""
    W = np.zeros((4, _DIM), dtype=np.float32)
    W[2, 7] = np.nan
    return _v4_head(4, W=W)


def atk_v4_inf_bias() -> bytes:
    """+inf bias on one entry: certain odds whatever the question."""
    return _v4_head(4, b=np.array([0, np.inf, 0, 0], dtype=np.float32))


def atk_v4_param_past_bound() -> bytes:
    """A weight past HEAD_MAX_ABS_PARAM: logits beyond the exact-sum bound."""
    W = np.zeros((4, _DIM), dtype=np.float32)
    W[0, 0] = 5e6
    return _v4_head(4, W=W)


def atk_v4_duplicate_rows() -> bytes:
    """Two rows for one entry id: which one is read would be a choice."""
    return _v4_head(ids=["lab/e000", "lab/e001", "lab/e000"])


def atk_v4_unknown_field() -> bytes:
    """A field the format does not have (say, a per-question cost head)."""
    z = dict(np.load(io.BytesIO(_v4_head(3)), allow_pickle=False))
    z["cost_head"] = np.zeros((3, _DIM), dtype=np.float32)
    buf = io.BytesIO()
    np.savez(buf, **z)
    return buf.getvalue()


V4_HEAD_ATTACKS = [
    ("v4: rows past HEAD_MAX_ROWS_V4", atk_v4_rows_past_the_bound,
     "raise validator compute per question", "I2"),
    ("v4: archive past the byte cap", atk_v4_oversize, "OOM the validator", "I2"),
    ("v4: NaN in one row", atk_v4_nan_row, "poison exact odds", "I2"),
    ("v4: +inf bias on one entry", atk_v4_inf_bias, "certain odds on one entry", "I2"),
    ("v4: weight past HEAD_MAX_ABS_PARAM", atk_v4_param_past_bound,
     "logits outside the exact-sum bound", "I1"),
    ("v4: duplicate row ids", atk_v4_duplicate_rows, "two odds for one entry", "I1"),
    ("v4: field outside the format", atk_v4_unknown_field, "unbounded extra arrays", "I2"),
]


def _v4_head_control() -> dict:
    """A v4 head with no token means loads under v4 (and not under v3)."""
    from fugal_subnet.head_eval import load_head_v4

    start = time.time()
    try:
        head = load_head_v4(_v4_head(4))
        refused_v3 = False
        try:
            load_head_from_npz(_v4_head(4))
        except ValueError:
            refused_v3 = True
        ok = head.W.shape == (4, _DIM) and refused_v3
        detail = "loaded under v4, refused under v3" if ok else "unexpected verdicts"
    except Exception as e:
        ok, detail = False, f"valid v4 head rejected: {e}"
    return {"name": "control: a v4 head without token means loads", "gain": "n/a",
            "invariant": "—", "verdict": "CONTROL" if ok else "BROKEN", "detail": detail,
            "secs": time.time() - start}


# ── Protocol v4 (docs/decisions/LIVE_MENU_V4.md, section 12 and 14) ─────────
# Two more surfaces. A v4 proof carries new miner-supplied fields, which the
# validator parses (I2). And the miner's TD reads two things it does not
# control: OpenRouter's endpoints listing, and menu bytes from any source.
# Neither may make the TD attest a price it did not read or call an entry the
# compiled rules forbid.


def _v4_proof_dict(**over) -> dict:
    """A well-formed v4 proof as JSON, with `over` replacing top-level fields."""
    from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
    r = QuestionResult("q1", "lab/m@high", True, 0.001, "ab" * 32, 10, 20, latency_ms=5,
                       provider_cost=1000, served_by="Host")
    d = BenchmarkProof(
        hotkey="5F", routing_protocol="ff" * 32, epoch_id="e1", nonce="0f" * 32,
        questions_hash="1e" * 32, weights_hash="2d" * 32, source_hash="3c" * 32, results=[r],
        total_cost_usd=0.001, per_model_costs={"lab/m@high": 0.001}, attestation_quote=b"",
        timestamp=1.0, menu_hash="cd" * 32, endpoint_reads=()).to_dict()
    d.update(over)
    return d


def _host(i: int) -> dict:
    return {"host": f"h{i}", "listed": False, "price_in": 0, "price_out": 0, "limit": 0,
            "quantization": "", "status": 0}


def _result_with(**fields) -> list:
    r = _v4_proof_dict()["results"][0]
    r.update(fields)
    return [r]


PROOF_FIELD_ATTACKS = [
    ("100k endpoint reads", lambda: _v4_proof_dict(endpoint_reads=[
        {"model": f"lab/m{i}", "hosts": [_host(0)]} for i in range(100_000)]),
     "OOM the validator", "I2"),
    ("10k hosts in one read", lambda: _v4_proof_dict(endpoint_reads=[
        {"model": "lab/m", "hosts": [_host(i) for i in range(10_000)]}]),
     "OOM the validator", "I2"),
    ("a 400-digit read price", lambda: _v4_proof_dict(endpoint_reads=[
        {"model": "lab/m", "hosts": [dict(_host(0), listed=True, quantization="fp8",
                                          price_in=10**400)]}]),
     "move a voted price", "I2"),
    ("a negative billed cost", lambda: _v4_proof_dict(results=_result_with(provider_cost=-1)),
     "drift the billed-cost check", "I2"),
    ("a 1MB serving host", lambda: _v4_proof_dict(results=_result_with(served_by="x" * 2**20)),
     "memory amplification", "I2"),
    ("a scored measurement result", lambda: _v4_proof_dict(results=_result_with(
        measure_for="ab" * 32, is_exploration=False)),
     "score on pairs the proposer prepared for (A9)", "I3"),
    ("a proposal pinning 10k hosts", lambda: _v4_proof_dict(proposal={
        "id": "lab/m@high", "model": "lab/m", "reasoning": {"effort": "high"},
        "provider": {"order": [f"h{i}" for i in range(10_000)], "allow_fallbacks": False},
        "quantization": "fp8"}),
     "blow up rule checking", "I2"),
    ("a proposal allowing fallbacks", lambda: _v4_proof_dict(proposal={
        "id": "lab/m@high", "model": "lab/m", "reasoning": {"effort": "high"},
        "provider": {"order": ["h"], "allow_fallbacks": True}, "quantization": "fp8"}),
     "measure one host, serve another", "I8"),
    ("a read price as a float", lambda: _v4_proof_dict(endpoint_reads=[
        {"model": "lab/m", "hosts": [dict(_host(0), listed=True, quantization="fp8",
                                          price_in=1.5)]}]),
     "a price that is not whole picodollars (A4)", "I1"),
    ("a read price as decimal text", lambda: _v4_proof_dict(endpoint_reads=[
        {"model": "lab/m", "hosts": [dict(_host(0), listed=True, quantization="fp8",
                                          price_in="0.000001")]}]),
     "a price read two ways (A4)", "I1"),
    ("a host listed twice in one read", lambda: _v4_proof_dict(endpoint_reads=[
        {"model": "lab/m", "hosts": [dict(_host(0), listed=True, quantization="fp8", price_in=1),
                                     dict(_host(0), listed=True, quantization="fp8",
                                          price_in=9)]}]),
     "two votes from one read (A4)", "I4"),
    ("a menu hash that is not a hash", lambda: _v4_proof_dict(menu_hash="../" * 40),
     "confuse the namespace check (A7)", "I2"),
]


def _run_proof_field(name, build, gain, inv) -> dict:
    from fugal_subnet.tee.proof import BenchmarkProof
    start = time.time()
    try:
        BenchmarkProof.from_dict(build())
    except Exception as e:
        return {"name": name, "verdict": "BLOCKED", "detail": type(e).__name__,
                "gain": gain, "invariant": inv, "secs": time.time() - start}
    return {"name": name, "verdict": "BROKEN", "detail": "accepted", "gain": gain,
            "invariant": inv, "secs": time.time() - start}


def _page(endpoints, model="lab/m") -> bytes:
    import json
    return json.dumps({"data": {"id": model, "endpoints": endpoints}}).encode()


def _ep(tag="host-a/fp8", prompt="0.000001", completion="0.000004", **over) -> dict:
    d = {"tag": tag, "pricing": {"prompt": prompt, "completion": completion},
         "max_completion_tokens": 65536, "quantization": "fp8", "status": 0}
    d.update(over)
    return d


LISTING_ATTACKS = [
    ("an oversized listing page", lambda: _page([_ep()]) + b" " * (1 << 20),
     "OOM the TD, stall the epoch", "I2"),
    ("a pinned host listed twice", lambda: _page([_ep(), _ep(prompt="0")]),
     "pick the cheaper of two prices", "I4"),
    ("a negative price", lambda: _page([_ep(prompt="-0.000001")]),
     "a negative cost for an entry", "I4"),
    ("an absurd price", lambda: _page([_ep(completion="99999")]),
     "an entry nobody can be routed to", "I4"),
    ("a price as a float", lambda: _page([_ep(prompt=1e-6)]),
     "a price that differs by host", "I1"),
    ("a price below a picodollar", lambda: _page([_ep(prompt="0.0000000000001")]),
     "a price that rounds differently by node", "I1"),
    ("an answer limit as a string", lambda: _page([_ep(max_completion_tokens="1e9")]),
     "an unbounded answer budget", "I2"),
    ("another model's page", lambda: _page([_ep()], model="lab/other"),
     "attest one model's prices for another", "I4"),
    ("10k endpoints", lambda: _page([_ep()] + [_ep(tag=f"h{i}") for i in range(10_000)]),
     "OOM the TD", "I2"),
]


def _run_listing(name, build, gain, inv) -> dict:
    from fugal_subnet.tee.endpoint_reads import ListingRefused, parse_listing
    start = time.time()
    try:
        parse_listing("lab/m", build(), ["host-a/fp8"])
    except ListingRefused as e:
        return {"name": name, "verdict": "BLOCKED", "detail": str(e)[:60], "gain": gain,
                "invariant": inv, "secs": time.time() - start}
    except Exception as e:
        return {"name": name, "verdict": "BROKEN", "detail": f"crashed: {type(e).__name__}",
                "gain": gain, "invariant": inv, "secs": time.time() - start}
    return {"name": name, "verdict": "BROKEN", "detail": "accepted", "gain": gain,
            "invariant": inv, "secs": time.time() - start}


def _menu_world():
    """A menu in force under a lab and host list (built in process: no key is
    involved), and an entry the list refuses."""
    import dataclasses
    import hashlib

    from fugal_subnet import menu_rules as mr
    from fugal_subnet import menu_state as ms
    seed = ms.seed_state(100)
    labs = sorted({e.model.split("/")[0] for e in seed.entries})
    hosts = sorted({mr.host_identity(h) for e in seed.entries for h in e.provider_order})
    fp = _seed_first_party(seed)
    raw = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, 1, labs=labs, hosts=hosts,
                                  first_party=fp)
    allow = mr.MenuAllowlist(serial=1, labs=frozenset(labs), hosts=frozenset(hosts),
                             first_party=mr.first_party_map(fp),
                             sha256=hashlib.sha256(raw).hexdigest(), signer="attack-suite")
    prev = dataclasses.replace(seed, serial=1, labs=ms.AllowlistRef(1, allow.sha256))
    rogue = dataclasses.replace(seed.entries[0], id="rogue-lab/m@high", model="rogue-lab/m")
    return ms, prev, allow, rogue


def _seed_first_party(seed) -> dict:
    """{lab: hosts} a seed's entries need marked first-party: the hosts a lab's
    entry pins with no quantization that are not the lab's own name."""
    from fugal_subnet import menu_rules as mr

    out: dict = {}
    for e in seed.entries:
        lab = e.model.split("/")[0]
        for h in e.provider_order:
            if e.quantization == mr.UNKNOWN_QUANTIZATION and mr.host_identity(h) != lab:
                out.setdefault(lab, set()).add(mr.host_identity(h))
    return out


def _hostile_menu(kind: str) -> bytes:
    import dataclasses
    ms, prev, _allow, rogue = _menu_world()
    nxt = dataclasses.replace(ms.carry(prev), serial=prev.serial + 1)
    if kind == "forbidden entry":
        return ms.canonical_json(dataclasses.replace(
            nxt, entries=tuple(sorted(nxt.entries + (rogue,), key=lambda e: e.id))))
    if kind == "a rolled-back list":
        return ms.canonical_json(dataclasses.replace(nxt, labs=ms.AllowlistRef(0, "")))
    if kind == "not canonical":
        import json
        return json.dumps(ms.to_dict(nxt), indent=2).encode()
    if kind == "out of sequence":
        return ms.canonical_json(dataclasses.replace(nxt, serial=prev.serial + 9))
    if kind == "a list nobody holds":
        return ms.canonical_json(dataclasses.replace(nxt, serial=prev.serial + 9,
                                                     labs=ms.AllowlistRef(99, "ab" * 32)))
    return b"{" * 100_000


MENU_ATTACKS = [
    ("committed menu with a forbidden entry", "forbidden entry",
     "a lab fitted to the pool on the menu (A1, A8)", "I8"),
    ("committed menu rolling back the list", "a rolled-back list",
     "re-admit a removed lab (A11)", "I8"),
    ("committed menu, non-canonical bytes", "not canonical", "two states, one hash", "I1"),
    ("committed menu out of sequence", "out of sequence", "skip a step's changes", "I1"),
    ("refused menu naming a list nobody holds", "a list nobody holds",
     "stop every TD while validators carry on", "I1"),
    ("committed menu, nested garbage", "garbage", "crash the TD", "I2"),
]


def _sshsig(key, payload: bytes) -> bytes:
    """An armored OpenSSH signature (PROTOCOL.sshsig) of `payload` in the
    signed lists' namespace, by an Ed25519 key made for this run."""
    import base64
    import hashlib

    from cryptography.hazmat.primitives import serialization

    from fugal_subnet import menu_rules as mr

    def string(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big") + b

    ns = mr.LIST_NAMESPACE.encode()
    signed = (b"SSHSIG" + string(ns) + string(b"") + string(b"sha512")
              + string(hashlib.sha512(payload).digest()))
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    blob = (b"SSHSIG" + (1).to_bytes(4, "big")
            + string(string(b"ssh-ed25519") + string(raw)) + string(ns) + string(b"")
            + string(b"sha512") + string(string(b"ssh-ed25519") + string(key.sign(signed))))
    b64 = base64.b64encode(blob).decode()
    return ("-----BEGIN SSH SIGNATURE-----\n"
            + "\n".join(b64[i:i + 70] for i in range(0, len(b64), 70))
            + "\n-----END SSH SIGNATURE-----\n").encode()


def _menu_session(prev, raw: bytes | None):
    """A miner's TD session holding `prev` (the menu of epoch 100), with
    `raw` committed for epoch 101 and offered, and the list in force offered
    signed by a key made for this run (the suite's allowed signers). The TD
    picks every list itself, as in a real epoch."""
    import hashlib

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from fugal_subnet import menu_rules as mr
    from fugal_subnet.tee import td_v4
    _ms, _prev, allow, _rogue = _menu_world()
    key = Ed25519PrivateKey.generate()
    line = key.public_key().public_bytes(serialization.Encoding.OpenSSH,
                                         serialization.PublicFormat.OpenSSH).decode()
    signers = f'attack@example.test namespaces="{mr.LIST_NAMESPACE}" {line}\n'.encode()
    seed = _ms.load_seed()
    labs = sorted({e.model.split("/")[0] for e in seed.entries})
    hosts = sorted({mr.host_identity(h) for e in seed.entries for h in e.provider_order})
    list_raw = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, 1, labs=labs, hosts=hosts,
                                       first_party=_seed_first_party(seed))
    assert hashlib.sha256(list_raw).hexdigest() == allow.sha256
    h = None if raw is None else hashlib.sha256(raw).hexdigest()
    return td_v4.V4Session(
        first_epoch=100, plan=object(), fetch_listing=lambda model: b"",
        read_commitment=lambda epoch: h if epoch == 101 else None,
        menu_bytes=lambda wanted: raw if wanted == h else None,
        list_bytes=lambda serial, sha: (list_raw, _sshsig(key, list_raw))
        if sha == allow.sha256 else None,
        anchor=prev, signers=signers)


def _run_menu(name, kind, gain, inv) -> dict:
    """The TD runs the carried menu whatever bytes carry the committed hash,
    and never one with an entry the rules refuse. Through the miner's own
    session: the TD chooses the list each menu is checked under, so bytes
    naming another list (or none) cannot change which list that is."""
    ms, prev, _allow, rogue = _menu_world()
    start = time.time()
    raw = _hostile_menu(kind)
    try:
        state = _menu_session(prev, raw).inputs_for(101).menu()
    except Exception as e:
        return {"name": name, "verdict": "BROKEN", "detail": f"crashed: {type(e).__name__}",
                "gain": gain, "invariant": inv, "secs": time.time() - start}
    if state != ms.carry(prev) or rogue.id in state.ids():
        return {"name": name, "verdict": "BROKEN", "detail": "ran on the hostile menu",
                "gain": gain, "invariant": inv, "secs": time.time() - start}
    return {"name": name, "verdict": "BLOCKED", "detail": "carried the previous menu",
            "gain": gain, "invariant": inv, "secs": time.time() - start}


# ── Protocol v4: what the root's committed menu cannot do (design 4.3) ───────
# Each case is a menu a hostile root could commit, read through the miner's
# own TD session. The TD must carry the previous menu. Each case also offers
# the same menu without the violation, which must run: otherwise the case
# would pass on some other rule and prove nothing about its own.


def _root_menu_case(kind: str):
    """(previous menu, the hostile committed menu, its honest twin)."""
    import dataclasses

    ms, prev, _allow, _rogue = _menu_world()
    if kind == "a running cooldown dropped":
        prev = dataclasses.replace(prev, retired=(
            ms.Retired("rogue-lab/benched@high", 90, ms.REASON_GATE_USEFUL, 140),))
    nxt = dataclasses.replace(ms.carry(prev), serial=prev.serial + 1)
    first = nxt.entries[0]
    if kind == "an entry's scored epoch moved":
        bad = dataclasses.replace(first, joined_epoch=nxt.epoch, scored_from=nxt.epoch + 3)
        return prev, dataclasses.replace(nxt, entries=(bad,) + nxt.entries[1:]), nxt
    if kind == "a price moved outside a refresh epoch":
        bad = dataclasses.replace(first, price_in=first.price_in + 1)
        return prev, dataclasses.replace(nxt, entries=(bad,) + nxt.entries[1:]), nxt
    if kind == "a running cooldown dropped":
        return prev, dataclasses.replace(nxt, retired=()), nxt
    if kind == "a pick-cost table moved outside a refresh epoch":
        # The cheapest pick on the menu made to look dear: the cells it
        # would take, at the answer cap, shrunk toward its call means.
        cap = ms.menu_rules.MAX_ANSWER_TOKENS
        dear = ms.pick_table((ms.PickCell("code", 500, 500 * cap, 500 * cap),), first,
                             ms.READER_PARAMS)
        bad = dataclasses.replace(first, pick=dear)
        return prev, dataclasses.replace(nxt, entries=(bad,) + nxt.entries[1:]), nxt

    # "an entry its step adds": entries join only through a committed delta
    # at the boundary (chain_proposals.check_delta), never in the step.
    from fugal_subnet import chain_proposals as cp

    e = nxt.entries[0]
    doc = cp.ProposalDoc(proposer="5Fattack-hk", model=f"{e.model.split('/')[0]}/attack-new",
                         reasoning=e.reasoning, provider_order=e.provider_order,
                         quantization=e.quantization)
    added = ms.apply_joins(nxt, (ms.NewEntry(
        request=doc.request(e.max_tokens), price_in=e.price_in, price_out=e.price_out,
        proposal="ab" * 32, proposer=doc.proposer, coldkey="5Fattack-ck"),))
    return prev, added, nxt


ROOT_MENU_ATTACKS = [
    ("root moves an entry's scored epoch", "an entry's scored epoch moved",
     "make a full entry provisional again, or score one early", "I3"),
    ("root reprices outside a refresh epoch", "a price moved outside a refresh epoch",
     "move prices whenever it likes, a new menu hash each epoch", "I1"),
    ("root drops a running cooldown", "a running cooldown dropped",
     "re-file a failed or lapsed id at once", "I4"),
    ("root moves pick costs outside a refresh", "a pick-cost table moved outside a refresh epoch",
     "steer every head's picks away from an entry whenever it likes (W13)", "I1"),
    ("root adds an entry in its step", "an entry its step adds",
     "put an entry on the menu that no admitted proposal named (A2)", "I4"),
]


def _run_root_menu(name, kind, gain, inv) -> dict:
    ms, *_ = _menu_world()
    start = time.time()

    def run(prev, state):
        return _menu_session(prev, ms.canonical_json(state)).inputs_for(prev.epoch + 1).menu()

    def verdict(v, detail):
        return {"name": name, "verdict": v, "detail": detail, "gain": gain, "invariant": inv,
                "secs": time.time() - start}
    try:
        prev, hostile, honest = _root_menu_case(kind)
        if run(prev, honest) != honest:
            return verdict("BROKEN", "the honest twin does not run: the case proves nothing")
        got = run(prev, hostile)
    except Exception as e:
        return verdict("BROKEN", f"crashed: {type(e).__name__}: {e}")
    if got != ms.carry(prev):
        return verdict("BROKEN", "the TD ran on the hostile menu")
    return verdict("BLOCKED", "carried the previous menu")


def _run_pick_table_beyond_its_cells() -> dict:
    """At a refresh epoch a root may publish new pick-cost tables, but only
    ones its graded cells could give: every cell at most the answer cap, the
    prior's share of the call means at least. A table past that range (an
    entry priced as if every cell cost ten times the cap, or as if its cells
    were free and the prior ignored) is refused by every reader, which runs
    the epoch on the carried menu; the same step with a table its cells
    could give is accepted, or the case proves nothing."""
    import dataclasses

    ms, prev, allow, _rogue = _menu_world()
    name = "root prices picks past what cells can give"
    gain = "make an entry unpickable, or free, at a refresh (W13)"
    start = time.time()
    prev = dataclasses.replace(prev, epoch=ms.MENU_REFRESH_EPOCHS * 6 - 1)
    nxt = dataclasses.replace(ms.carry(prev), serial=prev.serial + 1)
    first = nxt.entries[0]
    cap = ms.menu_rules.MAX_ANSWER_TOKENS
    honest = ms.pick_table((ms.PickCell("code", 400, 400 * 500, 400 * cap),), first,
                           ms.READER_PARAMS)

    def with_table(table):
        return dataclasses.replace(nxt, entries=(dataclasses.replace(first, pick=table),)
                                   + nxt.entries[1:])

    def verdict(v, detail):
        return {"name": name, "verdict": v, "detail": detail, "gain": gain, "invariant": "I1",
                "secs": time.time() - start}

    def read(state):
        return ms.menu_for_epoch(prev, ms.menu_hash(state), ms.canonical_json(state),
                                 allowlist=allow)
    try:
        if read(with_table(honest)) != with_table(honest):
            return verdict("BROKEN", "the honest table was not accepted: the case proves nothing")
        for bad in (dataclasses.replace(honest, mean_out=honest.mean_out * 10),
                    dataclasses.replace(honest, mean_in=0, mean_out=0)):
            try:
                read(with_table(bad))
            except ms.MenuRefused as e:
                if e.carried != ms.carry(prev):
                    return verdict("BROKEN", "refused, but not with the carried menu")
                continue
            return verdict("BROKEN", "a reader accepted a table no cells could give")
    except Exception as e:
        return verdict("BROKEN", f"crashed: {type(e).__name__}: {e}")
    return verdict("BLOCKED", "refused; the carried menu runs")


def _run_oversized_menu() -> dict:
    """Committed bytes over MENU_MAX_BYTES: the TD carries the menu without
    parsing them (no node parses more than the limit from the root)."""
    import hashlib
    import json

    ms, prev, _allow, _rogue = _menu_world()
    name, gain = "root commits a menu over the byte limit", "make every TD parse unbounded bytes"
    start = time.time()
    # A document over the limit, well-formed JSON all the way through.
    raw = json.dumps({"pad": "x" * ms.MENU_MAX_BYTES}).encode()
    parsed = []
    real = ms.from_json

    def spy(data):
        parsed.append(len(data))
        return real(data)
    ms.from_json = spy
    try:
        session = _menu_session(prev, raw)
        assert hashlib.sha256(raw).hexdigest() == session.read_commitment(101)
        got = session.inputs_for(101).menu()
    except Exception as e:
        got, parsed = e, parsed
    finally:
        ms.from_json = real
    big = [n for n in parsed if n > ms.MENU_MAX_BYTES]
    ok = got == ms.carry(prev) and not big
    return {"name": name, "verdict": "BLOCKED" if ok else "BROKEN",
            "detail": "carried, unread" if ok else f"parsed {big} bytes / got {type(got).__name__}",
            "gain": gain, "invariant": "I2", "secs": time.time() - start}


def _run_tier_host() -> dict:
    """A service tier is never a pinned host, by the entry rule itself, even
    under a list that names one (no signed list can: its parser refuses it;
    the entry rule holds on its own)."""
    import hashlib

    from fugal_subnet import menu_rules as mr
    name, gain = "a service-tier host under a list naming it", "a cheaper tier billed at list price"
    start = time.time()
    hosts = ("deepinfra", "deepinfra/flex")
    raw = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, 1, labs=("lab",), hosts=hosts,
                                  first_party={})
    allow = mr.MenuAllowlist(serial=1, labs=frozenset({"lab"}), hosts=frozenset(hosts),
                             sha256=hashlib.sha256(raw).hexdigest(), signer="attack-suite")
    refused = []
    for tag in ("deepinfra/flex/fp8", "deepinfra/fp8/flex", "deepinfra/flex"):
        req = {"model": "lab/m", "reasoning": {"effort": "high"},
               "provider": {"order": [tag], "allow_fallbacks": False},
               "quantization": "fp8" if "fp8" in tag else "unknown", "max_tokens": 1000}
        r = mr.check_request(req, allow)
        refused.append(r is not None and r.code == "host_tier")
    ok = all(refused)
    return {"name": name, "verdict": "BLOCKED" if ok else "BROKEN",
            "detail": "refused: host_tier" if ok else f"tier host passed: {refused}",
            "gain": gain, "invariant": "I8", "secs": time.time() - start}


def _run_provisional_budget() -> dict:
    """A new entry whose host never answers: each call runs to its own
    deadline and fails. The TD stops starting its calls once its budget for
    them has passed, never calls an entry it skips, and makes every call to
    the entries in full standing (harness._call_all_v4)."""
    from fugal_subnet.tee import harness

    name, gain = "a provisional entry whose host never answers", \
        "hold every TD's proof past its deadline with one join"
    start = time.time()
    now = [0.0]
    made = []

    class Proxy:
        records = []

    def call(proxy, key, q, request=None, deadline_s=None):
        made.append((request["model"], deadline_s))
        if request["model"] == "new":
            now[0] += harness.PROVISIONAL_CALL_DEADLINE_S        # ran to its deadline
            return None
        now[0] += 1
        return "A"

    items = ([({"question_id": f"n{i}"}, "new", "explore", {"model": "new"}, "k")
              for i in range(40)]
             + [({"question_id": f"s{i}"}, "skip", "explore", {"model": "skip"}, "k")
                for i in range(5)]
             + [({"question_id": f"o{i}"}, "old", "explore", {"model": "old"}, "k")
                for i in range(20)])
    saved = (harness._call_model, harness.HARNESS_CONCURRENCY)
    harness._call_model, harness.HARNESS_CONCURRENCY = call, 1
    real_check = None
    try:
        from fugal_subnet.tee import td_v4
        real_check = td_v4.check_wire_request
        td_v4.check_wire_request = lambda r: None
        harness._call_all_v4(Proxy(), items, "e1", provisional=frozenset({"new", "skip"}),
                             skipped=frozenset({"skip"}), clock=lambda: now[0])
    except Exception as e:
        return {"name": name, "verdict": "BROKEN", "detail": f"crashed: {type(e).__name__}: {e}",
                "gain": gain, "invariant": "I6", "secs": time.time() - start}
    finally:
        harness._call_model, harness.HARNESS_CONCURRENCY = saved
        if real_check is not None:
            td_v4.check_wire_request = real_check
    new = [d for m, d in made if m == "new"]
    bound = int(harness.PROVISIONAL_BUDGET_S // harness.PROVISIONAL_CALL_DEADLINE_S) + 1
    ok = (len(new) <= bound and all(d == harness.PROVISIONAL_CALL_DEADLINE_S for d in new)
          and not any(m == "skip" for m, _ in made)
          and sum(1 for m, _ in made if m == "old") == 20)
    return {"name": name, "verdict": "BLOCKED" if ok else "BROKEN",
            "detail": f"{len(new)} of 80 attempts made (bound {bound}), 20 of 20 others",
            "gain": gain, "invariant": "I6", "secs": time.time() - start}


def _run_challenge_spend_ceiling() -> dict:
    """A holder whose picks are answered far above their pick cost (every
    answer at its limit) hopes a challenger's TD keeps paying: the challenge
    ceiling bounds realised spend, counting calls in flight at their
    estimate. BLOCKED when, with 40 calls in flight at once, the challenge's
    realised spend stays within the ceiling plus the calls in flight when it
    was reached (each at most one answer's cost), exploration is never cut,
    and what is not started is the end of the call order (harness._call_all_v4)."""
    import threading
    from fractions import Fraction

    from fugal_subnet.tee import harness, td_v4
    from fugal_subnet.tee.runtime import APICallRecord

    name, gain = "a challenge answered far above its pick cost", \
        "drain a challenger's provider key past its ceiling"
    start = time.time()
    per_call = 5.0                                  # $5 realised against $0.01 estimated
    lock = threading.Lock()

    class Proxy:
        records = []

    def call(proxy, key, q, request=None, deadline_s=None):
        time.sleep(0.002)
        with lock:
            Proxy.records.append(APICallRecord(
                model_id=key, prompt_tokens=1, completion_tokens=1, cost_usd=per_call,
                timestamp=0.0, response_hash="", request_id=harness.current_request_id()))
        return "A"

    items = ([({"question_id": f"c{i}"}, "dear", "scored", {"model": "dear"}, "dear")
              for i in range(1000)]
             + [({"question_id": f"x{i}"}, "dear", "explore", {"model": "dear"}, "dear")
                for i in range(50)])
    saved = (harness._call_model, harness.HARNESS_CONCURRENCY, td_v4.check_wire_request)
    harness._call_model, harness.HARNESS_CONCURRENCY = call, 40
    td_v4.check_wire_request = lambda r: None
    try:
        out = harness._call_all_v4(Proxy(), items, "e1", spend_cap=100,
                                   call_estimate={"dear": Fraction(1, 100)})
    except Exception as e:
        return {"name": name, "verdict": "BROKEN", "detail": f"crashed: {type(e).__name__}: {e}",
                "gain": gain, "invariant": "I6", "secs": time.time() - start}
    finally:
        harness._call_model, harness.HARNESS_CONCURRENCY, td_v4.check_wire_request = saved
    made = [not o.not_started for o in out[:1000]]
    spent = per_call * sum(made)
    suffix = made == sorted(made, reverse=True)
    ok = (spent <= 100 + 40 * per_call and suffix
          and all(not o.not_started and not o.failed for o in out[1000:]))
    return {"name": name, "verdict": "BLOCKED" if ok else "BROKEN",
            "detail": f"{sum(made)} of 1000 challenge calls started, ${spent:.0f} realised "
                      f"against a $100 ceiling (bound ${100 + 40 * per_call:.0f}), 50 of 50 "
                      "exploration calls",
            "gain": gain, "invariant": "I6", "secs": time.time() - start}


def _run_held_menu() -> dict:
    """A menu the operator hands the TD (no commitment) with a forbidden entry."""
    import dataclasses

    from fugal_subnet.tee import td_v4
    ms, prev, allow, rogue = _menu_world()
    held = dataclasses.replace(prev, entries=tuple(sorted(prev.entries + (rogue,),
                                                          key=lambda e: e.id)))
    name, gain = "operator-held menu with a forbidden entry", "call an entry the rules forbid"
    start = time.time()
    try:
        td_v4.epoch_menu(101, previous=held, committed_hash=None, published=None,
                         allowlist=allow)
    except td_v4.MenuRejected:
        return {"name": name, "verdict": "BLOCKED", "detail": "no proof", "gain": gain,
                "invariant": "I8", "secs": time.time() - start}
    except Exception as e:
        return {"name": name, "verdict": "BROKEN", "detail": f"crashed: {type(e).__name__}",
                "gain": gain, "invariant": "I8", "secs": time.time() - start}
    return {"name": name, "verdict": "BROKEN", "detail": "ran on it", "gain": gain,
            "invariant": "I8", "secs": time.time() - start}


# ── Protocol v4: a join delta the root could commit (FAST_CROWN_V4.md 18) ─────
# Every node (TDs included) re-runs the root's admission on the delta's own
# contents before it applies it (chain_proposals.check_delta), and a refused
# delta is applied as an empty one: the base menu runs. Each case also checks
# the honest delta it was made from, which must pass.


def _root_delta_case(kind: str):
    """(base menu, list, hostile delta bytes, honest delta bytes)."""
    import dataclasses

    many = [f"5Fs{i}" for i in range(10)]
    if kind == "two joins of one coldkey":
        cp, base, allow, d = _admit([("5Fhk-a", "5Fck", "attack-a", None),
                                     ("5Fhk-b", "5Fck", "attack-b", None)], serving=many)
        flip = {cp.COLDKEY_ONCE: cp.JOINED}
    elif kind == "joins past the epoch's limit":
        cp, base, allow, d = _admit([(f"5Fhk{i}", f"5Fck{i}", f"attack-{i}", None)
                                     for i in range(3)])          # under 10 serving: one join
        flip = {cp.NO_PLACE: cp.JOINED}
    elif kind == "an entry the list refuses":
        cp, base, allow, d = _admit([("5Fhk", "5Fck", "attack-a", "rogue-lab")])
        flip = {c.verdict: cp.JOINED for c in d.considered}
    elif kind == "the admission order changed":
        cp, base, allow, d = _admit([(f"5Fhk{i}", f"5Fck{i}", f"attack-{i}", None)
                                     for i in range(2)])
        flip = {cp.JOINED: cp.NO_PLACE, cp.NO_PLACE: cp.JOINED}
    else:                                                         # another base menu
        cp, base, allow, d = _admit([("5Fhk", "5Fck", "attack-a", None)])
        return cp, base, allow, cp.delta_bytes(dataclasses.replace(d, base="cd" * 32)), \
            cp.delta_bytes(d)
    bad = dataclasses.replace(d, considered=tuple(
        dataclasses.replace(c, verdict=flip.get(c.verdict, c.verdict)) for c in d.considered))
    return cp, base, allow, cp.delta_bytes(bad), cp.delta_bytes(d)


ROOT_DELTA_ATTACKS = [
    ("root admits two of one coldkey", "two joins of one coldkey",
     "let one operator take an epoch's joins (A2)", "I4"),
    ("root admits past the epoch's limit", "joins past the epoch's limit",
     "flood a small field's menu with new entries", "I4"),
    ("root admits an entry the list refuses", "an entry the list refuses",
     "call a lab or host the signed list forbids", "I8"),
    ("root changes the admission order", "the admission order changed",
     "choose which proposal takes the epoch's place", "I4"),
    ("root's delta on another base menu", "another base menu",
     "apply joins to a menu the epoch does not run", "I1"),
]


def _run_root_delta(name, kind, gain, inv) -> dict:
    start = time.time()

    def verdict(v, detail):
        return {"name": name, "verdict": v, "detail": detail, "gain": gain, "invariant": inv,
                "secs": time.time() - start}
    try:
        cp, base, allow, hostile, honest = _root_delta_case(kind)
        cp.check_delta(base, honest, allow)
    except Exception as e:
        return verdict("BROKEN", f"the honest delta does not pass: {type(e).__name__}: {e}")
    try:
        cp.check_delta(base, hostile, allow)
    except (cp.DeltaRefused, cp.DeltaStale) as e:
        return verdict("BLOCKED", f"refused, applied as empty: {str(e)[:60]}")
    except Exception as e:
        return verdict("BROKEN", f"crashed: {type(e).__name__}: {e}")
    return verdict("BROKEN", "the hostile delta was applied")


# ── Protocol v4: the menu step's inputs (fugal_subnet/menu_inputs.py) ─────────
# What verified proofs hand the derivation step is a vote of operators: one
# coldkey is one voter however many hotkeys it runs. Proposals are not in the
# proofs: they are filed on chain and admitted by the root's join step
# (chain_proposals.admit), which the next cases run. Each case is an operator using only
# proofs that verify.


def _vp(hotkey, coldkey, results=(), reads=()):
    from fugal_subnet.menu_inputs import VerifiedProof
    from fugal_subnet.tee.proof import BenchmarkProof
    return VerifiedProof(BenchmarkProof(
        hotkey=hotkey, routing_protocol="x", epoch_id="e1", nonce="00" * 32, questions_hash="",
        weights_hash="1f" * 32, source_hash="", results=list(results), total_cost_usd=0.0,
        per_model_costs={}, attestation_quote=b"", timestamp=0.0, menu_hash="cd" * 32,
        endpoint_reads=tuple(reads)), coldkey)


def _one_read(model, host, price, listed=True):
    from fugal_subnet.tee.endpoint_reads import EndpointRead, HostListing
    h = HostListing(host, True, price, price, 65536, "fp8", 0) if listed else HostListing(host, False)
    return EndpointRead(model, (h,))


def atk_many_hotkeys_one_price():
    """One operator, twenty hotkeys, all reading a price 100x the listing's,
    against three honest operators. The vote must stay the listing's."""
    from fugal_subnet.menu_inputs import endpoint_vote
    proofs = [_vp(f"5Fh{i}", f"ck-honest-{i}", reads=[_one_read("lab/m", "host-a/fp8", 1000)])
              for i in range(3)]
    proofs += [_vp(f"5Fa{i:02d}", "ck-attacker", reads=[_one_read("lab/m", "host-a/fp8", 100_000)])
               for i in range(20)]
    (r,) = endpoint_vote(proofs)
    return r.price_in == 1000 and r.voters == 4, f"voted price {r.price_in} on {r.voters} voters"


def atk_many_hotkeys_unlist_a_host():
    """One operator's twenty hotkeys read a listed host as gone."""
    from fugal_subnet.menu_inputs import endpoint_vote
    proofs = [_vp(f"5Fh{i}", f"ck-honest-{i}", reads=[_one_read("lab/m", "host-a/fp8", 1000)])
              for i in range(3)]
    proofs += [_vp(f"5Fa{i:02d}", "ck-attacker",
                   reads=[_one_read("lab/m", "host-a/fp8", 0, listed=False)]) for i in range(20)]
    (r,) = endpoint_vote(proofs)
    return r.listed, f"listed={r.listed}"


def atk_forged_read_in_vote():
    """A read whose host appears twice and whose price is not whole
    picodollars, built past the proof parser: no vote."""
    from fugal_subnet.menu_inputs import endpoint_vote
    from fugal_subnet.tee.endpoint_reads import EndpointRead, HostListing
    forged = EndpointRead("lab/m", (HostListing("host-a/fp8", True, 0.5, 1, 1, "fp8", 0),
                                    HostListing("host-a/fp8", True, 1, 1, 1, "fp8", 0)))
    proofs = [_vp(f"5Ff{i}", f"ck-f{i}", reads=[forged]) for i in range(5)]
    proofs += [_vp(f"5Fh{i}", f"ck-honest-{i}", reads=[_one_read("lab/m", "host-a/fp8", 1000)])
               for i in range(3)]
    (r,) = endpoint_vote(proofs)
    return r.price_in == 1000 and r.voters == 3, f"voted price {r.price_in} on {r.voters} voters"


def atk_one_coldkey_fails_an_entry():
    """One operator drops every call to a rival's favourite entry for twelve
    epochs, from ten hotkeys; four other operators answer it."""
    from fugal_subnet import menu_inputs as mi
    from fugal_subnet import menu_state as ms
    _ms, prev, _allow, _rogue = _menu_world()
    target, other = prev.entries[0], prev.entries[1]
    window = []
    for k in range(88, 100):
        calls = [mi.CallCount(target.id, target.rev, "ck-attacker", 10, 10)]
        calls += [mi.CallCount(target.id, target.rev, f"ck-honest-{i}", 1, 0) for i in range(4)]
        calls += [mi.CallCount(other.id, other.rev, ck, 40, 0)
                  for ck in ["ck-attacker"] + [f"ck-honest-{i}" for i in range(4)]]
        window.append(mi.EpochSummary(k, (), tuple(sorted(
            calls, key=lambda c: (c.entry_id, c.rev, c.coldkey))), (), ()))
    h = {s.entry_id: s for s in mi.health_stats(prev, window, through=99)}[target.id]
    params = ms.MenuParams(mix=(("any", 1),), min_category_accuracy_bp=0)
    return ms.health_retires(h, params) is None, f"eligible {h.eligible}, failing {h.failing}"


def atk_every_call_fails_for_three_epochs():
    """Three epochs in which every call of every operator fails (an outage),
    on five dear entries each operator calls only now and then: two good
    calls earlier in the window, then one call in each failing epoch."""
    from fugal_subnet import menu_inputs as mi
    from fugal_subnet import menu_state as ms
    _ms, prev, _allow, _rogue = _menu_world()
    dear, cheap = prev.entries[:5], prev.entries[5]
    window = []
    for k in range(88, 100):
        bad = k >= 97
        calls = [mi.CallCount(e.id, e.rev, f"ck{i}", 1, 1 if bad else 0)
                 for e in dear for i in range(5) if bad or k in (88, 93)]
        calls += [mi.CallCount(cheap.id, cheap.rev, f"ck{i}", 10, 10 if bad else 0)
                  for i in range(5)]
        window.append(mi.EpochSummary(k, (), tuple(sorted(
            calls, key=lambda c: (c.entry_id, c.rev, c.coldkey))), (), ()))
    params = ms.MenuParams(mix=(("any", 1),), min_category_accuracy_bp=0)
    retired = [h.entry_id for h in mi.health_stats(prev, window, through=99)
               if ms.health_retires(h, params)]
    return not retired, f"retired {retired}"


def _join_world():
    """A base menu under its list, and a maker of proposals on its first
    single-host entry's lab and host: (cp, ms, base, allow, propose) where
    propose(name, hotkey, lab=None) gives (document bytes, its hash, the
    root's read of its listing)."""
    import dataclasses

    from fugal_subnet import chain_proposals as cp
    ms, prev, allow, _rogue = _menu_world()
    base = dataclasses.replace(ms.carry(prev), epoch=prev.epoch + 1)
    e = next(x for x in base.entries if len(x.provider_order) == 1)

    def propose(name, hotkey, lab=None):
        raw = cp.document_bytes(cp.ProposalDoc(
            proposer=hotkey, model=f"{lab or e.model.split('/')[0]}/{name}",
            reasoning=e.reasoning, provider_order=e.provider_order,
            quantization=e.quantization))
        listing = tuple(cp.HostListing(h, True, e.price_in, e.price_out, e.max_tokens, True)
                        for h in sorted(e.provider_order))
        return raw, cp.proposal_hash(raw), listing

    return cp, ms, base, allow, propose


def _admit(filed, serving=()):
    """The root's admission at a boundary of `filed`: (hotkey, coldkey, name,
    lab) each committing a proposal from a serving hotkey; `serving` adds
    hotkeys that serve and file nothing."""
    cp, ms, base, allow, propose = _join_world()
    docs, listings, proposals, coldkey_of = {}, {}, [], {}
    for hk, ck, name, lab in filed:
        raw, h, listing = propose(name, hk, lab)
        docs[h], listings[h] = raw, listing
        proposals.append((hk, h))
        coldkey_of[hk] = ck
    read = cp.ChainRead(epoch=base.epoch, block=1000, block_hash="ab" * 32,
                        proposals=tuple(sorted(proposals)), coldkey_of=coldkey_of,
                        serving=frozenset({hk for hk, *_ in filed} | set(serving)),
                        reveal=None)
    return cp, base, allow, cp.admit(base, read, docs, listings, allow)


def atk_file_an_entry_on_the_menu():
    """Propose an entry already on the menu: churn its rev and restart its
    exploration count (A2, A12)."""
    cp, ms, base, allow, propose = _join_world()
    e = next(x for x in base.entries if len(x.provider_order) == 1)
    lab, name = e.model.split("/")
    _c, _b, _a, delta = _admit([("5Fhk", "5Fck", name, lab)])
    (c,) = delta.considered
    return c.verdict == cp.REFUSED + "already_on_menu", c.verdict


def atk_two_miners_propose_one_entry():
    """Two miners propose the same new entry in one epoch: the second would
    take over the first one's entry."""
    cp, base, allow, delta = _admit([("5Fhk-a", "5Fck-a", "attack-new", None),
                                     ("5Fhk-b", "5Fck-b", "attack-new", None)],
                                    serving=[f"5Fs{i}" for i in range(10)])
    verdicts = sorted(c.verdict for c in delta.considered)
    return verdicts == [cp.JOINED, cp.REFUSED + "already_on_menu"], str(verdicts)


def atk_one_coldkey_fills_the_epochs_joins():
    """One operator's 6 hotkeys each propose an entry, to take every place
    of the epoch (A2)."""
    cp, base, allow, delta = _admit([(f"5Fhk{i}", "5Fck-spam", f"attack-{i}", None)
                                     for i in range(6)], serving=[f"5Fs{i}" for i in range(10)])
    joined = [c for c in delta.considered if c.verdict == cp.JOINED]
    return len(joined) == 1, f"{len(joined)} admitted of 6"


def atk_junk_proposal_document():
    """Commit the hash of bytes that are not a proposal document (or one of
    another hotkey), to make every node parse junk."""
    cp, ms, base, allow, propose = _join_world()
    junk = b'{"not": "a proposal"}'
    other, _h, listing = propose("attack-new", "5Fsomeone-else")
    proposals = (("5Fhk-a", cp.proposal_hash(junk)), ("5Fhk-b", cp.proposal_hash(other)))
    read = cp.ChainRead(epoch=base.epoch, block=1000, block_hash="ab" * 32,
                        proposals=proposals, coldkey_of={"5Fhk-a": "a", "5Fhk-b": "b"},
                        serving=frozenset({"5Fhk-a", "5Fhk-b"}), reveal=None)
    delta = cp.admit(base, read, {cp.proposal_hash(junk): junk, cp.proposal_hash(other): other},
                     {cp.proposal_hash(other): listing}, allow)
    verdicts = {c.hotkey: c.verdict for c in delta.considered}
    return set(verdicts.values()) == {cp.MALFORMED}, str(verdicts)


def atk_proposal_from_a_hotkey_that_served_nothing():
    """Propose from a fresh hotkey with no verified proof in the epoch the
    root's reveal names (filing costs nothing else)."""
    cp, ms, base, allow, propose = _join_world()
    raw, h, listing = propose("attack-new", "5Ffresh")
    read = cp.ChainRead(epoch=base.epoch, block=1000, block_hash="ab" * 32,
                        proposals=(("5Ffresh", h),), coldkey_of={"5Ffresh": "ck"},
                        serving=frozenset({"5Fother"}), reveal=None)
    (c,) = cp.admit(base, read, {h: raw}, {h: listing}, allow).considered
    return c.verdict == cp.NOT_SERVED, c.verdict


def _scored(qid, entry, p_tok, c_tok):
    from fugal_subnet.tee.proof import QuestionResult
    return QuestionResult(qid, entry, True, 0.0, "ab" * 32, p_tok, c_tok)


def atk_many_hotkeys_inflate_graded_cells():
    """One operator, twenty hotkeys, answering every scored question three
    honest operators also routed to an entry, each reporting the longest
    answer and prompt there can be: the cells' tokens, and so the entry's
    pick-cost table, are the honest operators' median (W13)."""
    from fugal_subnet import menu_inputs as mi
    from fugal_subnet import menu_state as ms
    _ms, prev, _allow, _rogue = _menu_world()
    target = prev.entries[0]
    qids = [f"pc{i:03d}" for i in range(20)]
    cats = {q: "code" for q in qids}
    honest = [_vp(f"5Fh{c}", f"ck-honest-{c}", [_scored(q, target.id, 300, 900) for q in qids])
              for c in range(3)]
    attacker = [_vp(f"5Fa{j:02d}", "ck-attacker",
                    [_scored(q, target.id, 10_000_000, 10_000_000) for q in qids])
                for j in range(20)]
    alone = mi.graded_cells(honest, prev, cats)
    attacked = mi.graded_cells(honest + attacker, prev, cats)
    params = ms.READER_PARAMS
    t0 = ms.pick_table(ms.PickStat(target.id, target.rev, tuple(
        ms.PickCell(c.category, c.cells, c.tokens_in, c.tokens_out) for c in alone)).cells,
        target, params)
    t1 = ms.pick_table(tuple(ms.PickCell(c.category, c.cells, c.tokens_in, c.tokens_out)
                             for c in attacked), target, params)
    return attacked == alone and t0 == t1, f"pick means {t1.mean_in}/{t1.mean_out}"


def atk_absurd_tokens_in_one_cell():
    """One operator alone on one scored cell, reporting the most tokens a
    verified proof may carry (PROOF_MAX_TOKENS) for prompt and answer: the
    cell counts at most the entry's answer limit and the answer cap, so the
    entry's pick means move by at most one capped cell's share (W13)."""
    from fugal_subnet import menu_inputs as mi
    from fugal_subnet import menu_state as ms
    from fugal_subnet.config import PROOF_MAX_TOKENS
    _ms, prev, _allow, _rogue = _menu_world()
    target = prev.entries[0]
    cap = ms.menu_rules.MAX_ANSWER_TOKENS
    proofs = [_vp("5Fa", "ck-attacker", [_scored("pc000", target.id, PROOF_MAX_TOKENS,
                                                 PROOF_MAX_TOKENS)])]
    (cell,) = mi.graded_cells(proofs, prev, {"pc000": "code"})
    table = ms.pick_table((ms.PickCell(cell.category, cell.cells, cell.tokens_in,
                                       cell.tokens_out),), target, ms.READER_PARAMS)
    k = ms.READER_PARAMS.pick_prior_cells_entry
    bound_out = (ms.MILLI * min(cap, target.max_tokens) + k * target.mean_out) / (1 + k) + 1
    ok = (cell.tokens_in <= cap and cell.tokens_out <= min(cap, target.max_tokens)
          and table.mean_out <= bound_out)
    return ok, f"cell {cell.tokens_in}/{cell.tokens_out} tokens, pick out {table.mean_out}"


V4_INPUT_ATTACKS = [
    ("one operator's 20 hotkeys vote a price", atk_many_hotkeys_one_price, "move a price (A4)",
     "I4"),
    ("one operator's 20 hotkeys unlist a host", atk_many_hotkeys_unlist_a_host,
     "retire an entry as gone (A4)", "I4"),
    ("a forged read reaches the vote", atk_forged_read_in_vote, "move a price (A4)", "I4"),
    ("one coldkey fails every call to an entry", atk_one_coldkey_fails_an_entry,
     "retire a rival's entry (A5)", "I4"),
    ("every call fails for three epochs", atk_every_call_fails_for_three_epochs,
     "retire healthy entries (A5, A10)", "I4"),
    ("propose an entry already on the menu", atk_file_an_entry_on_the_menu,
     "churn an entry's revision (A2)", "I4"),
    ("two miners propose one entry", atk_two_miners_propose_one_entry,
     "take over another miner's new entry (A2)", "I4"),
    ("one coldkey's 6 hotkeys propose", atk_one_coldkey_fills_the_epochs_joins,
     "take every join place of an epoch (A2)", "I4"),
    ("a junk proposal document", atk_junk_proposal_document,
     "make every node admit bytes that are not a proposal", "I2"),
    ("a proposal from a hotkey that served nothing", atk_proposal_from_a_hotkey_that_served_nothing,
     "file proposals from free hotkeys (A2)", "I4"),
    ("one coldkey's 20 hotkeys inflate cells", atk_many_hotkeys_inflate_graded_cells,
     "price a rival's favourite entry out of every pick (W13)", "I4"),
    ("absurd tokens in one graded cell", atk_absurd_tokens_in_one_cell,
     "price an entry out of every pick from one cell (W13)", "I2"),
]


# --- protocol v4: the optional scored questions and rule B (package S3) -------------------
#
# docs/decisions/SEAT_MINIMAL_V4.md 3.4, 3.7, 5.4. The scored questions are
# optional, all of them or none; rule B reads only a head's unbroken run of
# answered epochs; a proof with more than 34% of its exploration failed is not
# counted outside an outage epoch. Each case returns (held, detail), and
# tests/test_picks_v4_attacks.py breaks each once its rule is taken out.


def _picks_proof(results, *, scored):
    """A v4 proof of the TEE suite's honest world with these results, checked
    in mock mode: the scored-question rule is a property of the proof's
    content, enforced in every mode."""
    from fugal_subnet.attacks import run_tee_attacks as tee

    p = tee._v4_proof(results, scored=scored)
    return tee._v4_verify(p, mock=True)


def atk_partial_scored_set():
    """Answer the scored questions the head does well on and leave the rest
    out: the proof says it answered them, and carries half the slice."""
    from fugal_subnet.attacks import run_tee_attacks as tee

    results = tee._v4_results()
    scored = [r for r in results if not r.is_exploration]
    kept = scored[: len(scored) // 2] + [r for r in results if r.is_exploration]
    r = _picks_proof(kept, scored=True)
    return not r.valid, r.reason[:90]


def atk_exploration_only_with_foreign_scored_results():
    """An exploration-only proof (it says it answered no scored question)
    that carries scored results for another epoch's slice, answered when they
    suited it, to be read as this epoch's picks."""
    from fugal_subnet.attacks import run_tee_attacks as tee
    from fugal_subnet.epoch_plan import scored_slice

    w = tee._v4_world()
    other = [q["question_id"] for q in scored_slice(3, b"\x01" * 32, w["pool"], tee.V4_SLICE_SIZE)
             if q["question_id"] not in set(w["slice"]) | set(w["expected"].exploration)]
    explore = [r for r in tee._v4_results(w) if r.is_exploration]
    foreign = [tee._qr(q, w["rows"][0]) for q in other]
    r = _picks_proof(explore + foreign, scored=False)
    return not r.valid, r.reason[:90]


def _picks_records(epochs, values):
    """Challenge records paired with "holder": `values(e)` maps each head
    with a counted challenge in e to its items (500 a challenge)."""
    from fugal_subnet.picks_v4 import PicksEpoch
    return {e: PicksEpoch(e, ("hard",) * 500, {h: tuple(v) for h, v in values(e).items()},
                          holder="holder")
            for e in epochs}


def atk_holder_stops_answering():
    """A challenger with challenges in epochs 1-3 and exploration-only
    proofs from epoch 4 hopes the epochs it did not challenge still count:
    each reads as an epoch without a counted challenge, and its run is
    empty."""
    from fugal_subnet import picks_v4 as pv

    recs = _picks_records(range(1, 7), lambda e: {"rival": (0,) * 500} if e < 4 else {})
    missed = [e for e in range(1, 7) if not pv.answered_in(recs, "rival", e)]
    run = pv.answered_run(recs, "rival", 6)
    return missed == [4, 5, 6] and run == (), f"missed {missed}, run {run}"


def atk_answer_only_the_epochs_that_suit():
    """A head challenges only in the epochs its challenge goes well (a
    point better than the holder, SD 20) and serves exploration-only proofs
    in the others, hoping its evidence is its good epochs together. The run
    restarts at every miss, so each attempt is judged with no hidden one
    beside it, and one attempt of 500 at a point better does not pass."""
    from fugal_subnet import picks_v4 as pv

    unit = pv.UNITS_PER_POINT
    good = {1, 3, 5, 7, 9, 11}
    items = tuple((21 if i % 2 else -19) * unit for i in range(500))
    recs = _picks_records(range(1, 12), lambda e: {"x": items} if e in good else {})
    ev = pv.picks_check(recs, "x", "holder", through=11)
    return not ev.passes, f"run {ev.run}, {ev.paired} paired, passes {ev.passes}"


def atk_junk_exploration_past_the_bound():
    """Pay for the epoch's exploration and fail most of it on purpose (the
    miner controls its own network), in an epoch the rest of the field
    answers: the proof is not counted."""
    from fugal_subnet import menu_inputs as mi
    from fugal_subnet import menu_state as ms
    from fugal_subnet import picks_v4 as pv
    from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult

    state = ms.seed_state(1000)
    entry = state.ids()[0]

    def proof(hotkey, failed):
        results = [QuestionResult(f"x{i}", entry, False, 0.0, "ab" * 32, is_exploration=True,
                                  failed=i < failed) for i in range(112)]
        return BenchmarkProof(epoch_id="e00001000", nonce="0f" * 32, questions_hash="1e" * 32,
                              weights_hash="2d" * 32, source_hash="3c" * 32, results=results,
                              total_cost_usd=0.0, per_model_costs={}, attestation_quote=b"",
                              timestamp=1.0, hotkey=hotkey, routing_protocol="ff" * 32,
                              menu_hash="cd" * 32, endpoint_reads=(), scored=False)

    proofs = [mi.VerifiedProof(proof(f"5Fh{k}", 3), f"ck-honest-{k}") for k in range(4)]
    proofs.append(mi.VerifiedProof(proof("5Fjunk", 60), "ck-junk"))
    v = pv.failure_refusals(proofs, state)
    return "5Fjunk" in v.refused and not v.outage, f"refused {sorted(v.refused)}"


PICKS_ATTACKS = [
    ("a proof with a partial scored set", atk_partial_scored_set,
     "keep only the scored questions it does well on (3.4)", "I3"),
    ("exploration-only proof, foreign scored results", atk_exploration_only_with_foreign_scored_results,
     "have another epoch's picks read as this one's (3.4)", "I3"),
    ("a challenge counted in an epoch without one", atk_holder_stops_answering,
     "count epochs it did not challenge in its run (FAST_CROWN_V4.md 4)", "I3"),
    ("challenge only when it suits", atk_answer_only_the_epochs_that_suit,
     "pass rule B on its good attempts together (FAST_CROWN_V4.md 4)", "I3"),
    ("junk exploration past the failure bound", atk_junk_exploration_past_the_bound,
     "buy a proof's standing without its labels (5.4)", "I4"),
]


# --- protocol v4: the seat (package S4) ------------------------------------------------------
#
# docs/decisions/SEAT_MINIMAL_V4.md 3.5 to 3.7, 4.4. On the seat's synthetic
# world (fugal_subnet/attacks/seat_v4_world.py), at the design's constants.
# Each case returns (held, detail); tests/test_v4_seat_attacks.py breaks each
# once its rule is taken out (rule C, the standing window, the byte-identical
# rows of succession, a successor reading its predecessors' odds).


def _seat_world():
    """A holder ("h") the field has not beaten, and an original head
    ("orig", committed at block 7) that is a much better predictor and starts
    answering the scored questions at epoch 9; its near-copy ("copy", another
    operator, committed at block 8, odds perturbed by a unit on some labels)
    answers from epoch 6, so it has its 1,000 questions first."""
    from fugal_subnet.attacks.seat_v4_world import World

    w = World()
    w.add("h", skill=0.2)
    w.add("orig", skill=0.6, block=7)
    w.add("copy", skill=0.6, block=8, noise=5, owner="ck-other")
    for e in range(1, 6):
        w.epoch(e, served=["h"], answered=["h"])
    return w


def atk_seat_near_copy_of_an_earlier_head():
    """A near-copy of a better head, committed after it and answering the
    scored questions sooner, tries to take the seat before the original can."""
    w = _seat_world()
    copy_held = []
    for e in range(6, 12):
        w.epoch(e, served=["h", "orig", "copy"],
                answered=["h", "copy"] + (["orig"] if e >= 9 else []))
        copy_held.append(w.holder() == "copy")
    return not any(copy_held), f"holders through 11: {w.holder()}; copy held {copy_held}"


def atk_seat_near_copy_while_the_original_misses():
    """The same near-copy, in the epoch the original misses (not served):
    the original still stands and still bars it."""
    w = _seat_world()
    held = []
    for e in range(6, 18):
        missed = e == 11
        w.epoch(e, served=["h", "copy"] + ([] if missed else ["orig"]),
                answered=["h", "copy"] + (["orig"] if e >= 9 and not missed else []))
        held.append((e, w.holder()))
    copy_epochs = [e for e, x in held if x == "copy"]
    return not copy_epochs and held[-1][1] == "orig", f"copy held in {copy_epochs}; last {held[-1]}"


def atk_seat_borrow_an_earlier_commit():
    """An operator with an old, weak head commits, on the same hotkey, a
    copy of a later operator's better head (its rows changed, none kept),
    to take the old head's place in the precedence order ahead of the head
    it copied."""
    from fugal_subnet.attacks.seat_v4_world import World

    w = World()
    w.add("h", skill=0.2)
    w.add("old", skill=0.0, block=3, hotkey="hk-x", owner="ck-x")
    w.add("good", skill=0.6, block=20)
    for e in range(1, 8):
        w.epoch(e, served=["h", "old"] + (["good"] if e >= 6 else []),
                answered=["h", "old"] + (["good"] if e >= 6 else []))
    w.add("steal", skill=0.6, block=75, hotkey="hk-x", owner="ck-x", noise=9,
          digest_salt="other bytes")
    for e in range(8, 16):
        w.epoch(e, served=["h", "steal", "good"], answered=["h", "steal", "good"])
    hs = w.state.head(w.heads["steal"].hash)
    return (w.holder() == "good" and hs.origin == w.heads["steal"].hash,
            f"holder {w.holder()}, steal's origin is itself: {hs.origin == hs.head}")


def recommit_take_epoch(recommit: bool, *, helpers: int = 31, last: int = 40
                        ) -> tuple[int | None, dict[int, int]]:
    """The epoch a clearly better challenger takes the seat from a holder
    whose hotkey commits a successor of its head every epoch from 12 (its
    rows byte for byte, new bytes) or keeps its head, with `helpers` other
    hotkeys committing a successor of theirs every epoch, so that 32 heads
    share the late-odds bound (32 head-epochs an epoch, the live schedule);
    and the labels its comparison with the bar read each epoch."""
    from fugal_subnet.attacks.seat_v4_world import World, hash_of

    def helper_heads(w, e):
        out = []
        for i in range(helpers):
            w.add(f"helper{i}-{e}", skill=0.3, hotkey=f"hk-helper{i}", owner="ck-helpers",
                  block=e * 10 - 1)
            out.append(f"helper{i}-{e}")
        return out

    w = World(late_slots=32, labels=80, questions=500)
    w.add("a0", skill=0.5, hotkey="hk-a", owner="ck-a", block=5)
    for e in range(1, 12):
        w.epoch(e, served=["a0"], answered=["a0"])
    w.add("c", skill=0.8, block=111)
    others, a, seen = helper_heads(w, 11), "a0", {}
    for e in range(12, last + 1):
        if recommit:
            a = f"a{e}"
            w.add(a, skill=0.5, hotkey="hk-a", owner="ck-a", block=e * 10 - 1)
        others = helper_heads(w, e)
        d = w.epoch(e, served=[a, "c", *others], answered=[a, "c"])
        doc = d.evidence["heads"].get(hash_of("c"))
        if doc and doc["vs_bar"]:
            seen[e] = doc["vs_bar"]["n"]
        if w.holder() == "c":
            return e, seen
    return None, seen


def atk_seat_recommitted_head_keeps_its_evidence():
    """A holder whose hotkey re-commits its head every epoch, with 31 other
    hotkeys doing the same so that the late-odds bound is shared by 32
    heads: a successor's odds before its commit are its predecessors'
    (4.4), and late-odds rounds serve the seat's heads first
    (seat_v4.late_odds_first), so the comparisons against it read the same
    labels as against a holder that keeps its head, and the better
    challenger takes the seat in the same epoch."""
    static, _ = recommit_take_epoch(False)
    took, seen = recommit_take_epoch(True)
    return (static is not None and took == static,
            f"takes at {took} (at {static} from a static holder); labels read {seen}")


SEAT_ATTACKS = [
    ("v4 seat: a near-copy of an earlier head", atk_seat_near_copy_of_an_earlier_head,
     "take the seat with another operator's head (3.5)", "I3"),
    ("v4 seat: a near-copy while the original misses", atk_seat_near_copy_while_the_original_misses,
     "take the seat in an outage of the head it copied (3.5, 3.7)", "I3"),
    ("v4 seat: borrow an earlier commit's precedence", atk_seat_borrow_an_earlier_commit,
     "jump the precedence order with a changed head (4.4)", "I3"),
    ("v4 seat: a head that is re-committed each epoch keeps its evidence",
     atk_seat_recommitted_head_keeps_its_evidence,
     "keep a better head out of the seat (4.4, late odds)", "I3"),
]


# --- protocol v4: the published set's snapshot (fugal_subnet/memo_sync.py) ------------------
#
# A TD, and a validator without history, read the published set from bytes
# any source serves (the root's store, a mirror, whatever an operator puts in
# front of them), named by the digest a verified root reveal carries
# (memo_v4.published_digest). Bytes that are not that set must never become
# the plan a proof is drawn on, nor the set a validator judges heads by. And
# a root reveal naming a set that leaves out published questions (so their
# outcomes could be drawn as "fresh") must be caught by any node with history.


def _published_world():
    """A pool over the v4 mix's sets, a seed observed through epoch 8 with
    three questions of each set unpublished, and the set as published
    through 12 (one reserve question an epoch)."""
    from fugal_subnet import epoch_plan, memo_v4

    sets = sorted(epoch_plan.mix_v4().bench_category)
    pool = [{"question_id": f"{b}-{i:02d}", "benchmark": b} for b in sets for i in range(12)]
    seed = {"format": memo_v4.SEED_FORMAT, "through": "e00000008",
            "pool": {"pool_hash": "attack", "n_questions": len(pool),
                     "per_set": {b: 12 for b in sets}},
            "observations_sha256": "0" * 64,
            "unpublished": {b: [f"{b}-{i:02d}" for i in (9, 10, 11)] for b in sets}}
    pub = memo_v4.seed_published(pool, seed)
    reserve = sorted(q for ids in seed["unpublished"].values() for q in ids)
    for k in range(9, 13):
        pub = memo_v4.fold(pub, k, [reserve[k - 9]])
    return pool, pub, seed


def _snapshot_refused(raw: bytes, digest: str, *, through: int = 12):
    """(held, detail): the TD's and a validator's reading of `raw` as the
    snapshot named `digest`: refused, and nothing comes back to draw on."""
    from fugal_subnet import memo_sync, memo_v4

    pool, _pub, _seed = _published_world()
    ids = {q["question_id"] for q in pool}
    try:
        memo_v4.published_from_snapshot(raw, digest, pool_ids=ids, through=through)
    except ValueError as e:
        refused = str(e)
    else:
        return False, "the bytes were read as the published set"
    got = memo_sync.read_snapshot(raw, digest, through=through, pool_ids=ids)
    return got is None, f"refused: {refused[:90]}"


def _canonical(doc) -> bytes:
    import json

    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def atk_published_oversized():
    import hashlib

    from fugal_subnet import memo_v4

    raw = b" " * (memo_v4.MAX_SNAPSHOT_BYTES + 1)
    return _snapshot_refused(raw, hashlib.sha256(raw).hexdigest())


def atk_published_wrong_digest():
    """The bytes of another set (one question dropped) under the digest the
    reveal names."""
    from fugal_subnet import memo_v4

    _pool, pub, _seed = _published_world()
    other = memo_v4.Published({q: e for q, e in pub.first.items() if q != min(pub.first)}, 12)
    return _snapshot_refused(memo_v4.snapshot_bytes(other), pub.digest)


def atk_published_malformed():
    import hashlib

    for raw in (b"{", b"[]", _canonical({"format": "published-v4-state-v1", "through": 12}),
                _canonical({"format": "published-v4-state-v1", "through": 12,
                            "first": {"12": "a-00"}})):
        held, detail = _snapshot_refused(raw, hashlib.sha256(raw).hexdigest())
        if not held:
            return held, detail
    return True, "every malformed document refused"


def atk_published_ids_outside_the_pool():
    from fugal_subnet import memo_v4

    _pool, pub, _seed = _published_world()
    alien = memo_v4.Published({**pub.first, "zz-not-a-question": 12}, 12)
    return _snapshot_refused(memo_v4.snapshot_bytes(alien), alien.digest)


def atk_published_duplicate_ids():
    import hashlib

    _pool, pub, _seed = _published_world()
    doc = pub.to_doc()
    q = doc["first"]["8"][0]
    doc["first"]["12"] = sorted(doc["first"]["12"] + [q])      # once at 8, again at 12
    raw = _canonical(doc)
    return _snapshot_refused(raw, hashlib.sha256(raw).hexdigest())


def atk_published_reveal_omits_published_questions():
    """A root reveal naming the digest of a set that leaves out questions
    already published (their outcomes would be drawn as fresh): a validator
    finds it against its own state, and a TD refuses to draw on it."""
    import hashlib
    import json

    from fugal_subnet import memo_sync, memo_v4
    from fugal_subnet.reveal_publisher import commitment_for

    pool, pub, seed = _published_world()
    short = memo_v4.Published({q: e for q, e in pub.first.items() if e < 12 or q != max(
        q2 for q2, e2 in pub.first.items() if e2 == 12)}, 12)
    reveal = {"epoch_id": "e00000014", "questions": [],
              "memo_v4": {"through": 12, "published_digest": short.digest}}
    finding = memo_sync.check_named(pub, reveal)
    raw = json.dumps(reveal).encode()
    sha = hashlib.sha256(raw).hexdigest()
    td = memo_sync.ChainSet(
        pool=pool, read_at=lambda b: commitment_for("e00000014", sha) if b >= 16 else None,
        reveal_bytes=lambda eid, s: raw, refuse=True, seed_doc=seed)
    td.held, td.named, td.at = pub, 12, 14
    try:
        td.advance_to(16)
    except memo_sync.PublishedUnavailable as e:
        refused = str(e)
    else:
        refused = None
    return (finding is not None and refused is not None,
            f"follower: {(finding or 'none')[:60]}; TD: {(refused or 'drew on it')[:60]}")


PUBLISHED_ATTACKS = [
    ("v4 published set: snapshot past the byte cap", atk_published_oversized,
     "OOM a TD or a validator", "I2"),
    ("v4 published set: another set's bytes", atk_published_wrong_digest,
     "draw a plan on a set the root never named", "I1"),
    ("v4 published set: malformed snapshot", atk_published_malformed,
     "crash or confuse the reader", "I2"),
    ("v4 published set: ids outside the pool", atk_published_ids_outside_the_pool,
     "a set no fold of this pool's reveals gives", "I1"),
    ("v4 published set: a question twice", atk_published_duplicate_ids,
     "two first epochs for one question", "I1"),
    ("v4 published set: reveal omits published qs", atk_published_reveal_omits_published_questions,
     "have published outcomes drawn as fresh (6.3)", "I3"),
]


def _run_input(name, fn, gain, inv) -> dict:
    start = time.time()
    try:
        held, detail = fn()
    except Exception as e:
        return {"name": name, "verdict": "BROKEN", "detail": f"crashed: {type(e).__name__}: {e}",
                "gain": gain, "invariant": inv, "secs": time.time() - start}
    return {"name": name, "verdict": "BLOCKED" if held else "BROKEN", "detail": detail,
            "gain": gain, "invariant": inv, "secs": time.time() - start}


def _v4_controls() -> list[dict]:
    """The well-formed v4 inputs must still pass, or the v4 cases prove nothing:
    a v4 proof parses, a good listing page gives its price, and a committed
    menu that may follow the previous one is the menu the TD runs."""
    import dataclasses

    from fugal_subnet.tee.endpoint_reads import parse_listing
    from fugal_subnet.tee.proof import BenchmarkProof

    def control(name, check):
        start = time.time()
        try:
            ok, detail = check()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        return {"name": name, "gain": "n/a", "invariant": "—",
                "verdict": "CONTROL" if ok else "BROKEN", "detail": detail,
                "secs": time.time() - start}

    def proof():
        p = BenchmarkProof.from_dict(_v4_proof_dict())
        return p.menu_hash == "cd" * 32, "parsed"

    def listing():
        read = parse_listing("lab/m", _page([_ep()]), ["host-a/fp8"])
        return read.hosts[0].price_in == 1_000_000, "parsed"

    def menu():
        ms, prev, _allow, _rogue = _menu_world()
        # An entry retires, leaving the record a retirement writes.
        gone = prev.entries[0]
        nxt = dataclasses.replace(ms.carry(prev), serial=prev.serial + 1,
                                  entries=prev.entries[1:],
                                  retired=(ms.Retired(gone.id, 101, ms.REASON_GONE,
                                                      101 + ms.PROPOSAL_COOLDOWN_EPOCHS),))
        got = _menu_session(prev, ms.canonical_json(nxt)).inputs_for(101).menu()
        return got == nxt, "the committed menu runs"

    def exploration_only():
        from fugal_subnet.attacks import run_tee_attacks as tee
        r = _picks_proof([x for x in tee._v4_results() if x.is_exploration], scored=False)
        full = _picks_proof(tee._v4_results(), scored=True)
        return r.valid and full.valid, r.reason or full.reason

    def snapshot():
        from fugal_subnet import memo_v4

        pool, pub, _seed = _published_world()
        got = memo_v4.published_from_snapshot(memo_v4.snapshot_bytes(pub), pub.digest,
                                              pool_ids={q["question_id"] for q in pool},
                                              through=12)
        return got.digest == pub.digest, "read"

    return [control("control: a v4 proof parses", proof),
            control("control: a good listing page parses", listing),
            control("control: a valid committed menu runs", menu),
            control("control: exploration-only and full v4 proofs verify", exploration_only),
            control("control: the named snapshot reads", snapshot)]


def _run_one(name: str, build, gain: str, inv: str, budget_s: float,
             loader=load_head_from_npz) -> dict:
    """Load one hostile payload, bounding both time and outcome."""
    try:
        payload = build()
    except MemoryError:
        return {"name": name, "verdict": "BLOCKED", "detail": "payload too large to build",
                "gain": gain, "invariant": inv, "secs": 0.0}

    start = time.time()
    try:
        head = loader(payload)
    except Exception as e:
        return {"name": name, "verdict": "BLOCKED", "detail": type(e).__name__,
                "gain": gain, "invariant": inv, "secs": time.time() - start}
    elapsed = time.time() - start

    # Accepted. That is only safe if the values cannot poison downstream math.
    finite = bool(np.all(np.isfinite(head.W)) and np.all(np.isfinite(head.b)))
    if not finite:
        return {"name": name, "verdict": "BROKEN", "detail": "accepted non-finite weights",
                "gain": gain, "invariant": inv, "secs": elapsed}
    if elapsed > budget_s:
        return {"name": name, "verdict": "BROKEN", "detail": f"took {elapsed:.1f}s (I6)",
                "gain": gain, "invariant": inv, "secs": elapsed}
    return {"name": name, "verdict": "SURVIVED", "detail": "accepted, values finite",
            "gain": gain, "invariant": inv, "secs": elapsed}


def _control() -> dict:
    """A well-formed head must still load — otherwise the suite proves nothing."""
    start = time.time()
    try:
        head = load_head_from_b64(base64.b64encode(_valid_head()).decode("ascii"))
        ok = head.W.shape == (4, _DIM)
        return {"name": "control: valid head loads", "gain": "n/a", "invariant": "—",
                "verdict": "CONTROL" if ok else "BROKEN",
                "detail": "loaded" if ok else "valid head rejected",
                "secs": time.time() - start}
    except Exception as e:
        return {"name": "control: valid head loads", "gain": "n/a", "invariant": "—",
                "verdict": "BROKEN", "detail": f"valid head rejected: {e}",
                "secs": time.time() - start}


def _b64_control() -> dict:
    """Malformed base64 from the axon must be rejected, not crash the loop."""
    start = time.time()
    try:
        load_head_from_b64("!!!not-valid-base64!!!")
    except Exception as e:
        return {"name": "malformed base64 from axon", "gain": "crash the epoch loop",
                "invariant": "I6", "verdict": "BLOCKED", "detail": type(e).__name__,
                "secs": time.time() - start}
    return {"name": "malformed base64 from axon", "gain": "crash the epoch loop",
            "invariant": "I6", "verdict": "BROKEN", "detail": "accepted junk base64",
            "secs": time.time() - start}


def main() -> int:
    budget_s = 10.0
    from fugal_subnet.head_eval import load_head_v4

    results = [_run_one(n, f, g, i, budget_s) for n, f, g, i in ATTACKS]
    # Every head payload through the v4 loader too, then the v4-only cases.
    results += [_run_one(f"{n} (v4 loader)", f, g, i, budget_s, loader=load_head_v4)
                for n, f, g, i in ATTACKS + V4_HEAD_ATTACKS]
    # The TD logs each refused menu as an alarm; here every one is expected.
    import logging
    logging.getLogger("fugal_subnet.tee").setLevel(logging.CRITICAL)
    results += [_run_proof_field(n, f, g, i) for n, f, g, i in PROOF_FIELD_ATTACKS]
    results += [_run_listing(n, f, g, i) for n, f, g, i in LISTING_ATTACKS]
    results += [_run_menu(n, k, g, i) for n, k, g, i in MENU_ATTACKS]
    results.append(_run_held_menu())
    results += [_run_root_menu(n, k, g, i) for n, k, g, i in ROOT_MENU_ATTACKS]
    results.append(_run_pick_table_beyond_its_cells())
    results.append(_run_oversized_menu())
    results.append(_run_tier_host())
    results.append(_run_provisional_budget())
    results.append(_run_challenge_spend_ceiling())
    results += [_run_root_delta(n, k, g, i) for n, k, g, i in ROOT_DELTA_ATTACKS]
    results += [_run_input(n, f, g, i) for n, f, g, i in V4_INPUT_ATTACKS]
    results += [_run_input(n, f, g, i) for n, f, g, i in PICKS_ATTACKS]
    results += [_run_input(n, f, g, i) for n, f, g, i in SEAT_ATTACKS]
    logging.getLogger("fugal_subnet.memo_sync").setLevel(logging.CRITICAL)
    results += [_run_input(n, f, g, i) for n, f, g, i in PUBLISHED_ATTACKS]
    for r in results:
        if r["verdict"] != "BROKEN" and r["secs"] > budget_s:
            r.update(verdict="BROKEN", detail=f"took {r['secs']:.1f}s (I6)")
    results.append(_b64_control())
    results.append(_control())
    results.append(_v4_head_control())
    results += _v4_controls()

    print()
    print(f"{'MINER ATTACK SUITE':<46}{'INV':<5}{'VERDICT':<10}{'SECS':>6}")
    print("-" * 78)
    for r in results:
        flag = "OK" if r["verdict"] != "BROKEN" else "!!"
        print(f"{flag} {r['name'][:43]:<43}{r['invariant']:<5}"
              f"{r['verdict']:<10}{r['secs']:>6.2f}")
        if r["verdict"] == "BROKEN":
            print(f"     ↳ {r['detail']} — a miner gains: {r['gain']}")
    print("-" * 78)

    blocked = sum(1 for r in results if r["verdict"] == "BLOCKED")
    survived = sum(1 for r in results if r["verdict"] == "SURVIVED")
    broken = [r for r in results if r["verdict"] == "BROKEN"]
    print(f"{blocked} blocked, {survived} survived-safely, "
          f"{len(broken)} BROKEN")

    if broken:
        print("\nFAIL — a miner could exploit the cases marked !! above.")
        return 1
    print("PASS — no miner-controlled payload breached I2 or I6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
