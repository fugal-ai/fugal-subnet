"""Protocol v4 head arrays: W and b are little-endian float32 or float64.

`head_eval.load_head_v4` (the one loader the TD, the validator, the replay and
the holder-head load use) refuses every other dtype of W and b before any
arithmetic reads them. The v1-v3 loader is unchanged. No network, no spend.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from fugal_subnet import head_eval
from fugal_subnet.head_eval import load_head_from_npz, load_head_v4
from fugal_subnet.head_odds import picks_and_odds_v4
from tests.test_head_v4 import IDS, MENU, _arrays, _bytes, _hidden

REFUSED = {
    "float16": np.float16,
    "longdouble": np.longdouble,
    "big-endian float32": np.dtype(">f4"),
    "big-endian float64": np.dtype(">f8"),
    "complex64": np.complex64,
    "complex128": np.complex128,
    "int32": np.int32,
    "int64": np.int64,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def _cast(z, key, dtype):
    z = dict(z)
    z[key] = np.round(z[key]).astype(dtype) if np.dtype(dtype).kind in "iub" else \
        z[key].astype(dtype)
    return z


@pytest.mark.parametrize("key", ["W", "b"])
@pytest.mark.parametrize("name", sorted(REFUSED))
def test_every_other_dtype_of_w_and_b_is_refused_with_its_reason(key, name):
    dtype = np.dtype(REFUSED[name])
    if dtype == np.dtype(np.longdouble) and dtype.itemsize == 8:
        pytest.skip("this platform's long double is float64")
    raw = _bytes(_cast(_arrays(IDS), key, dtype))
    with pytest.raises(ValueError, match=f"{key} must be little-endian float32 or float64"):
        load_head_v4(raw)


def test_the_refusal_comes_before_any_arithmetic(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("arithmetic reached a refused dtype")

    raw = _bytes(_cast(_arrays(IDS), "W", np.float16))
    monkeypatch.setattr(head_eval.np, "isfinite", boom)
    with pytest.raises(ValueError, match="little-endian float32 or float64"):
        load_head_v4(raw)


@pytest.mark.parametrize("dtype", ["<f4", "<f8"])
def test_float32_and_float64_heads_load_unchanged(dtype):
    z = _arrays(IDS)
    z = dict(z, W=z["W"].astype(dtype), b=z["b"].astype(dtype))
    raw = _bytes(z)
    head = load_head_v4(raw)
    assert head.W.dtype == np.dtype(dtype) and head.b.dtype == np.dtype(dtype)
    # The same values in either width give the same odds and picks, byte for
    # byte (the float32 values are exact in float64).
    got = picks_and_odds_v4(head, MENU, _hidden(16))
    want = picks_and_odds_v4(load_head_v4(_bytes(_arrays(IDS))), MENU, _hidden(16))
    digest = (lambda r: hashlib.sha256(r.odds.tobytes() + repr(r.picks).encode()).hexdigest())
    assert digest(got) == digest(want)


def test_the_v3_loader_is_unchanged():
    """Protocol v1-v3 keeps what it accepted: the v4 dtype rule is not read
    there."""
    z = _cast(_arrays(["lab/x"]), "W", np.float16)
    try:
        load_head_from_npz(_bytes(z))
    except ValueError as e:
        assert "little-endian" not in str(e)


def _with_compression(raw: bytes, method: int) -> bytes:
    """`raw` (a zip) with every member's compression method set to `method`."""
    out = bytearray(raw)
    for sig, at in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
        i = out.find(sig)
        while i != -1:
            out[i + at:i + at + 2] = method.to_bytes(2, "little")
            i = out.find(sig, i + 4)
    return bytes(out)


@pytest.mark.parametrize("method", [99, 1, 97])
def test_an_archive_the_reader_cannot_unpack_is_a_refusal_not_an_error(method):
    """A head whose members name a compression method zipfile cannot unpack
    is refused with its reason (ValueError), by the v4 loader and the v4
    verdict, never another exception (verification is contracted not to
    raise on miner input)."""
    raw = _with_compression(_bytes(_arrays(IDS)), method)
    with pytest.raises(ValueError, match="not a readable"):
        load_head_v4(raw)
    why = head_eval.v4_head_verdict(raw, MENU)
    assert why is not None and why.startswith("Incompatible success head")
