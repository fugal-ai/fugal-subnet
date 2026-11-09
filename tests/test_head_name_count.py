"""A head's model-name count is bounded before any name is built.

A zero-width string dtype holds any number of elements in no bytes, so a
small archive can declare millions of names; building them before checking
the count cost seconds and hundreds of MB per proof, before any hardware
check. The names array's shape must equal (rows,) first.
"""
import io
import time
import zipfile

import numpy as np
import pytest

from fugal_subnet.head_eval import load_head_from_npz


def _npy(arr):
    b = io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


def _raw_npy_header(descr, shape):
    hdr = "{'descr': '%s', 'fortran_order': False, 'shape': %r, }" % (descr, shape)
    hdr += " " * ((64 - (10 + len(hdr) + 1) % 64) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + len(hdr).to_bytes(2, "little") + hdr.encode("latin1")


def _head(names_member: bytes, rows=2):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("W.npy", _npy(np.zeros((rows, 1024), np.float32)))
        z.writestr("b.npy", _npy(np.zeros(rows, np.float32)))
        z.writestr("models.npy", names_member)
    return buf.getvalue()


def test_a_huge_zero_width_name_list_is_refused_before_it_is_built():
    data = _head(_raw_npy_header("<U0", (10_000_000,)))
    assert len(data) < 16_384
    t0 = time.monotonic()
    with pytest.raises(ValueError, match="model names have shape"):
        load_head_from_npz(data)
    assert time.monotonic() - t0 < 0.5


@pytest.mark.parametrize("shape", [(3,), (1,), (2, 1)])
def test_a_name_count_that_does_not_match_the_rows_is_refused(shape):
    names = np.array(["m/a"] * int(np.prod(shape))).reshape(shape)
    with pytest.raises(ValueError, match="model names have shape"):
        load_head_from_npz(_head(_npy(names)))


def test_too_many_rows_are_refused_before_the_names_are_read():
    with pytest.raises(ValueError, match="model rows"):
        load_head_from_npz(_head(_raw_npy_header("<U0", (10_000_000,)), rows=65))
