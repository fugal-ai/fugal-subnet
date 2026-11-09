"""A head's model names are bounded before any name is built.

A zero-width string dtype holds any number of elements in no bytes, so a small
archive can declare millions of names; building them before checking the count
cost seconds and hundreds of MB per proof, before any hardware check. Only
success-contract heads load, and the contract bounds the count (at most 64)
before a single name is read; anything else is refused unread.
"""
import io
import time
import zipfile

import numpy as np
import pytest

from fugal_subnet import success_test_fixtures as fx
from fugal_subnet.head_eval import load_head_from_npz

HUGE = 10_000_000


def _npy(arr):
    b = io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


def _raw_npy_header(descr, shape):
    hdr = "{'descr': '%s', 'fortran_order': False, 'shape': %r, }" % (descr, shape)
    hdr += " " * ((64 - (10 + len(hdr) + 1) % 64) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + len(hdr).to_bytes(2, "little") + hdr.encode("latin1")


def _archive(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, data in members.items():
            z.writestr(f"{name}.npy", data)
    return buf.getvalue()


def _fast_refusal(data, match):
    assert len(data) < 64 * 1024
    t0 = time.monotonic()
    with pytest.raises(ValueError, match=match):
        load_head_from_npz(data)
    assert time.monotonic() - t0 < 0.5


def test_a_head_without_the_contract_is_refused_unread():
    _fast_refusal(_archive({"W": _npy(np.zeros((2, 1024), np.float32)),
                            "b": _npy(np.zeros(2, np.float32)),
                            "models": _raw_npy_header("<U0", (HUGE,))}),
                  "success-contract")


def test_a_contract_head_with_a_huge_zero_width_name_list_is_refused_before_it_is_built():
    z = fx.arrays(np.zeros((2, 1024), np.float32), np.zeros(2, np.float32), ["v/a", "v/b"])
    members = {k: _npy(v) for k, v in z.items() if k != "models"}
    members["models"] = _raw_npy_header("<U0", (HUGE,))
    _fast_refusal(_archive(members), "invalid models")


def test_a_contract_head_whose_names_do_not_match_its_rows_is_refused():
    z = fx.arrays(np.zeros((2, 1024), np.float32), np.zeros(2, np.float32), ["v/a", "v/b"])
    z["models"] = np.array(["v/a", "v/b", "v/c"])
    buf = io.BytesIO()
    np.savez(buf, **z)
    with pytest.raises(ValueError):
        load_head_from_npz(buf.getvalue())
