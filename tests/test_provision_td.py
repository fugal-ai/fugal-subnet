"""scripts/provision_td.py refuses, before anything is sent, a head no validator scores.

It used to check only that the .npz loads; a legacy W/b/models head loads and
is then refused by every validator, after the operator has paid for the TD.
"""
import importlib.util
import io
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


td = _load("provision_td")


def test_a_legacy_head_is_refused():
    from fugal_subnet.config import HEAD_HIDDEN_DIM

    buf = io.BytesIO()
    np.savez(buf, W=np.zeros((3, HEAD_HIDDEN_DIM), np.float32), b=np.zeros(3, np.float32),
             models=np.array(["a/x", "b/y", "c/z"]))
    assert td.head_protocols(buf.getvalue()) == []


def test_current_heads_name_their_protocol(tmp_path):
    dr = _load("dress_rehearsal")
    v1 = pathlib.Path(dr.write_head(str(tmp_path / "v1.npz"), seed=1)).read_bytes()
    v2 = pathlib.Path(dr.biased_head_v2(str(tmp_path / "v2.npz"), 1)).read_bytes()
    assert td.head_protocols(v1) == [1]
    # A v2 head has rows for entries of the pinned v4 seed menu, which is all
    # a v4 TD asks of a head at start.
    assert td.head_protocols(v2) == [2, 4]


def test_a_v4_head_can_be_pushed():
    """A head with rows keyed by the seed menu's entry ids, which no v1-v3
    validator scores, is what a v4 TD runs (head_eval.v4_head_verdict)."""
    from fugal_subnet import menu_state as ms
    from tests.test_tee_e2e import _head

    assert td.head_protocols(_head(list(ms.load_seed().ids()), seed=1)) == [4]
    assert td.head_protocols(_head(["lab-z/not-on-the-seed"], seed=1)) == []
