"""The app half of an approved entry must be computable from the repo.

That is the whole reason the identity is a pair. The base half can only be read
off the machine that boots the image; the app half is a property of this
checkout, so it can be computed in CI and reviewed in a pull request instead of
requiring someone to hold a quote and read hex out of a register.
"""
import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location(
        "compute_app_identity", REPO / "scripts" / "compute_app_identity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_source_hash_matches_the_miner_byte_for_byte():
    """If the calculator and the miner disagree, the approved entry names an
    identity the miner cannot produce — an approved list that rejects everyone,
    for a reason nobody would look for."""
    miner_spec = importlib.util.spec_from_file_location(
        "miner_mod", REPO / "neurons" / "miner.py")
    miner = importlib.util.module_from_spec(miner_spec)
    miner_spec.loader.exec_module(miner)

    assert _script().source_hash() == miner._get_source_hash()


def test_normalisation_is_deterministic_and_key_order_independent():
    norm = _script().normalise_app_compose
    a = {"b": 1, "a": {"z": [1, 2], "y": "x"}}
    b = {"a": {"y": "x", "z": [1, 2]}, "b": 1}
    assert norm(a) == norm(b)
    assert " " not in norm(a)                       # compact separators


def test_finite_floats_are_refused_not_silently_mishashed():
    """RFC 8785 specifies ES6 number formatting, which Python does not reproduce.

    A wrong compose hash is silent and total: the approved entry names something
    dstack never extended, so every honest miner is rejected for a reason nobody
    would look for. Refusing is the correct failure.
    """
    import pytest

    norm = _script().normalise_app_compose
    with pytest.raises(ValueError, match="float"):
        norm({"scale": 1.5})
    with pytest.raises(ValueError, match=r"\$\.a\.b"):
        norm({"a": {"b": 2.0}})              # the path is named, so it is fixable


def test_non_finite_floats_become_null_not_nan():
    """json.dumps emits `NaN` and `Infinity`, which are not JSON. dstack's
    normalisation calls for null, and a hash over invalid JSON would differ
    from whatever their parser produced."""
    norm = _script().normalise_app_compose
    out = norm({"a": float("nan"), "b": float("inf"), "c": float("-inf")})
    assert "NaN" not in out and "Infinity" not in out
    assert json.loads(out) == {"a": None, "b": None, "c": None}


def test_unicode_is_emitted_directly_not_escaped():
    out = _script().normalise_app_compose({"k": "café"})
    assert "café" in out and "\\u" not in out


def test_compose_hash_changes_with_the_content(tmp_path):
    script = _script()
    p = tmp_path / "app-compose.json"
    p.write_text(json.dumps({"image": "repo@sha256:aaa"}), encoding="utf-8")
    first, _ = script.compose_hash(str(p))
    p.write_text(json.dumps({"image": "repo@sha256:bbb"}), encoding="utf-8")
    second, _ = script.compose_hash(str(p))
    assert first != second, "a different app must produce a different compose hash"
