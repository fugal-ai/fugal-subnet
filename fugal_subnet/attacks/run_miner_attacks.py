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
    """Serialize exactly as scripts/train_head.py:save_head does.

    The dtype matters: an object-dtype models array is rejected outright by
    allow_pickle=False, which would block every payload here for a reason that
    has nothing to do with the defense under test — a suite that passes while
    proving nothing. Only atk_object_array uses object dtype, deliberately.
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, W=W, b=b, models=np.array(models, dtype=models_dtype))
    return buf.getvalue()


def _valid_head(n_models: int = 4) -> bytes:
    rng = np.random.RandomState(0)
    W = (rng.randn(n_models, _DIM) * 0.01).astype(np.float32)
    b = np.zeros(n_models, dtype=np.float32)
    return _npz_bytes(W, b, [f"vendor/m{i}" for i in range(n_models)])


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


def _run_one(name: str, build, gain: str, inv: str, budget_s: float) -> dict:
    """Load one hostile payload, bounding both time and outcome."""
    try:
        payload = build()
    except MemoryError:
        return {"name": name, "verdict": "BLOCKED", "detail": "payload too large to build",
                "gain": gain, "invariant": inv, "secs": 0.0}

    start = time.time()
    try:
        head = load_head_from_npz(payload)
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
    results = [_run_one(n, f, g, i, budget_s) for n, f, g, i in ATTACKS]
    results.append(_b64_control())
    results.append(_control())

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
