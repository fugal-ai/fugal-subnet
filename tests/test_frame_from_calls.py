"""The shipped bootstrap frame is provably the fold of its committed call log.

data/reference_frame.json is v1 consensus state, pinned by sha256 in
scripts/check_safety_invariants.py. The pin says the file has not changed; this
says it is what the evidence produces. The builder's default once folded the
log in 150-row decayed batches, a different frame from the pinned one, so the
documented rebuild did not reproduce the file it documents.
"""
import hashlib
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_default_recipe_reproduces_the_pinned_frame():
    builder = _load("build_frame_from_calls")
    pinned = _load("check_safety_invariants").PINNED_BOOTSTRAP_FRAME_SHA256
    frame, rows = builder.build_frame(builder.DEFAULT_LOG)    # the default batch
    payload = json.dumps(frame.to_dict(), indent=2, sort_keys=True) + "\n"
    assert hashlib.sha256(payload.encode()).hexdigest() == pinned
    assert payload == (ROOT / "data" / "reference_frame.json").read_text(encoding="utf-8")
    assert frame.epochs == 1 and rows
