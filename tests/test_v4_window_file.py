"""The v4 window file (v2_epoch.save_window / load_window, layout
V4_WINDOW_FORMAT): a fixed set of concatenated arrays with an index.

A v4 window reads back to exactly the window it was written from (its
logical contents, v2_epoch.window_arrays, byte for byte); a v1-v3 window is
written and read as before; a file in the earlier v4 layout still loads; a
damaged, truncated or hostile file is refused with a ValueError before it can
allocate what it claims; and the file's member count does not grow with the
window."""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest

from fugal_subnet import v2_epoch as v2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "experiments"))


def _world_window(uids=10, epochs=8):
    import time_v4_epoch as T

    w = T.EpochWorld(T.Shape(uids=uids, epochs=epochs, dim=16, successors=2, returning=2,
                             pool=4_000, reserve=600))
    return v2.run_v2_epoch(**w.run_kwargs()).window


@pytest.fixture(scope="module")
def window():
    return _world_window()


def same_contents(a, b):
    x, y = v2.window_arrays(a), v2.window_arrays(b)
    assert set(x) == set(y)
    for k in x:
        assert x[k].dtype == y[k].dtype and x[k].shape == y[k].shape, k
        assert x[k].tobytes() == y[k].tobytes(), k


def test_a_v4_window_reads_back_to_itself(window, tmp_path):
    assert window.picks and window.slices and window.heads
    assert any(h not in window.heads for h in window.odds)        # odds of heads with no answers
    path = str(tmp_path / "w.npz")
    v2.save_window(path, window)
    assert v2._is_v4_file(path)
    back = v2.load_window(path, window.size)
    same_contents(window, back)
    assert back.picks == window.picks
    # challenge records: the holder they paired with, and items on prefixes
    assert all(r.holder is not None for r in back.picks.values())
    assert any(r.values for r in back.picks.values())
    for e, p in window.pairs.items():
        q = back.pairs[e]
        assert type(q) is type(p) and q.voters == p.voters and q.question_ids == p.question_ids
    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == sorted(f"{k}.npy" for k in v2.V4_WINDOW_ARRAYS)
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())


def test_the_member_count_does_not_grow_with_the_window(tmp_path):
    """The operation-count guard of the save and load: a window of four times
    the heads and twice the epochs is the same 24 arrays."""
    sizes = []
    for uids, epochs in ((6, 4), (24, 8)):
        w = _world_window(uids, epochs)
        path = str(tmp_path / f"w{uids}.npz")
        v2.save_window(path, w)
        with zipfile.ZipFile(path) as zf:
            sizes.append(len(zf.namelist()))
        same_contents(w, v2.load_window(path, w.size))
    assert sizes == [len(v2.V4_WINDOW_ARRAYS)] * 2


def test_a_v1_to_v3_window_is_written_and_read_as_before(tmp_path):
    from tests.test_v3_epoch import live

    results, _ = live(3)
    window = results[-1].window
    path = str(tmp_path / "w.npz")
    v2.save_window(path, window)
    assert not v2._is_v4_file(path)
    with np.load(path, allow_pickle=False) as z:
        files = list(z.files)
        stored = {k: z[k] for k in files}
    legacy = v2.window_arrays(window)
    assert files == list(legacy)
    assert all(stored[k].tobytes() == legacy[k].tobytes() for k in files)
    with zipfile.ZipFile(path) as zf:
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in zf.infolist())
    same_contents(window, v2.load_window(path, window.size))


def test_a_v4_window_in_the_earlier_layout_still_loads(window, tmp_path):
    path = str(tmp_path / "old.npz")
    with open(path, "wb") as f:
        np.savez_compressed(f, **v2.window_arrays(window))
    assert not v2._is_v4_file(path)
    same_contents(window, v2.load_window(path, window.size))


# --- damaged and hostile files ---------------------------------------------------------------


def _arrays(window):
    return dict(v2._v4_arrays(window))


def _write(path, arrays, members=None):
    """An uncompressed npz of `arrays` (np.savez's layout), or of raw
    `members` bytes when given."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for k, a in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(a), allow_pickle=False)
            zf.writestr(f"{k}.npy", buf.getvalue())
        for name, data in (members or {}).items():
            zf.writestr(name, data)


def refused(path, match=None):
    with pytest.raises(ValueError, match=match):
        v2.load_window(str(path), 48)


def test_a_truncated_file_is_refused(window, tmp_path):
    path = tmp_path / "w.npz"
    v2.save_window(str(path), window)
    data = path.read_bytes()
    for cut in (len(data) // 3, len(data) - 200, 40):
        (tmp_path / "t.npz").write_bytes(data[:cut])
        refused(tmp_path / "t.npz")
    # a member's bytes damaged in place (its CRC no longer matches)
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo("od_values.npy")
    bad = bytearray(data)
    at = info.header_offset + 30 + len(info.filename) + 200
    bad[at] ^= 0xFF
    (tmp_path / "c.npz").write_bytes(bytes(bad))
    refused(tmp_path / "c.npz")


def test_a_header_claiming_more_than_its_member_is_refused_before_reading(window, tmp_path,
                                                                           monkeypatch):
    """A member whose npy header declares 10^12 values but holds a few
    bytes: refused from the header, without allocating (np.load is never
    reached)."""
    arrays = _arrays(window)
    arrays.pop("od_values")
    head = io.BytesIO()
    np.lib.format.write_array_header_1_0(head, {"descr": "<i2", "fortran_order": False,
                                                "shape": (10**12,)})
    _write(tmp_path / "h.npz", arrays, {"od_values.npy": head.getvalue() + b"\x00" * 64})

    def no_load(*_a, **_k):
        raise AssertionError("np.load reached")
    monkeypatch.setattr(v2.np, "load", no_load)
    refused(tmp_path / "h.npz", "stored size")


def test_members_dtypes_and_declared_sizes_are_checked(window, tmp_path, monkeypatch):
    arrays = _arrays(window)
    extra = dict(arrays, junk=np.zeros(3, np.int8))
    _write(tmp_path / "x.npz", extra)
    refused(tmp_path / "x.npz", "members")
    missing = {k: v for k, v in arrays.items() if k != "pk_values"}
    _write(tmp_path / "m.npz", missing)
    refused(tmp_path / "m.npz", "members")
    wrong = dict(arrays, od_values=arrays["od_values"].astype(np.int32))
    _write(tmp_path / "d.npz", wrong)
    refused(tmp_path / "d.npz", "1-D")
    two_d = dict(arrays, pr_y=arrays["pr_y"].reshape(1, -1))
    _write(tmp_path / "2.npz", two_d)
    refused(tmp_path / "2.npz", "1-D")
    obj = dict(arrays)
    obj.pop("pr_y")
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.array([1, "a"], dtype=object), allow_pickle=True)
    _write(tmp_path / "o.npz", obj, {"pr_y.npy": buf.getvalue()})
    refused(tmp_path / "o.npz")
    _write(tmp_path / "ok.npz", arrays)
    monkeypatch.setattr(v2, "V4_WINDOW_MAX_BYTES", 1000)
    refused(tmp_path / "ok.npz", "declares more than")


def test_indexes_that_do_not_match_their_arrays_are_refused(window, tmp_path):
    base = _arrays(window)

    def variant(name, **changes):
        a = dict(base)
        a.update(changes)
        _write(tmp_path / f"{name}.npz", a)
        return tmp_path / f"{name}.npz"

    od = base["od_index"].reshape(-1, 3).copy()
    long = od.copy()
    long[-1, 2] += 5                                   # claims more odds than are stored
    refused(variant("long", od_index=long.reshape(-1)), "shorter than its index")
    short = od.copy()
    short[0, 2] -= 1                                   # leaves odds unclaimed
    refused(variant("short", od_index=short.reshape(-1)))
    head = od.copy()
    head[0, 0] = 10**6
    refused(variant("head", od_index=head.reshape(-1)), "head index")
    neg = od.copy()
    neg[0, 1] = -3
    refused(variant("neg", od_index=neg.reshape(-1)), "negative")
    refused(variant("rows", od_index=base["od_index"][:-1]), "rows of 3")
    vals = base["od_values"].copy()
    vals[0] = 10_001
    refused(variant("range", od_values=vals), "outside")
    strings = base["pr_strings"].copy()
    strings[0] = 10**7
    refused(variant("str", pr_strings=strings), "string index")
    nv = base["pr_nvoters"].copy()
    nv[0] = 0
    refused(variant("novoter", pr_nvoters=nv))
    pk = base["pk_index"].reshape(-1, 5).copy()
    pk[0, 3] = 7
    refused(variant("outage", pk_index=pk.reshape(-1)), "malformed")
    pk = base["pk_index"].reshape(-1, 5).copy()
    pk[0, 4] = -2
    refused(variant("holder", pk_index=pk.reshape(-1)), "malformed")
    lengths = base["pk_lengths"].copy()
    lengths[0] = 10**6
    refused(variant("length", pk_lengths=lengths), "past its list")
    meta = json.loads(bytes(base["meta"]).decode())
    refused(variant("fmt", meta=np.frombuffer(json.dumps(dict(meta, format="other")).encode(),
                                              np.uint8)), "not a")
    refused(variant("json", meta=np.frombuffer(b"{not json", np.uint8)), "JSON")
    dup = dict(meta, heads=meta["heads"] + meta["heads"][:1])
    refused(variant("dup", meta=np.frombuffer(json.dumps(dup).encode(), np.uint8)), "malformed")
    sl = dict(meta, slices={**meta["slices"], "12345": meta["slices"][next(iter(meta["slices"]))]})
    refused(variant("sl", meta=np.frombuffer(json.dumps(sl).encode(), np.uint8)), "slices")
