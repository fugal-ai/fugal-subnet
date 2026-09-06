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


def test_compose_hash_is_the_raw_bytes_not_a_reserialisation():
    """dstack measures the bytes on disk, and re-serialising changes them.

    This is the bug this test exists for: hashing a normalised re-serialisation
    produced a compose hash dstack never extends, so an approved entry built
    from it would have named an identity no honest miner can ever produce —
    rejecting the entire field for a reason nobody would look for.

    dstack's own generator writes the file with indent=2 in insertion order, and
    their source says plainly: "raw byte copy - do NOT json.load/dump (would
    change hashed bytes)".
    """
    import hashlib
    import json
    import pathlib as _p
    import tempfile

    script = _script()
    obj = {"runner": "docker-compose", "docker_compose_file": "services:\n  app:\n"}
    pretty = json.dumps(obj, indent=2)                 # what dstack writes
    compact = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    assert pretty != compact, "the two forms must actually differ for this to bite"

    with tempfile.TemporaryDirectory() as td:
        f = _p.Path(td) / "app-compose.json"
        f.write_text(pretty, encoding="utf-8")
        digest, _ = script.compose_hash(str(f))

    assert digest == hashlib.sha256(pretty.encode()).hexdigest()
    assert digest != hashlib.sha256(compact.encode()).hexdigest()


def test_whitespace_alone_changes_the_compose_hash(tmp_path):
    """Byte-exactness is the property. Two files that parse to the same object
    are different apps as far as the measurement is concerned, which is why the
    file has to be shipped verbatim rather than regenerated."""
    script = _script()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text('{"x": 1}', encoding="utf-8")
    b.write_text('{"x":1}', encoding="utf-8")
    assert script.compose_hash(str(a))[0] != script.compose_hash(str(b))[0]


def test_compose_hash_changes_with_the_content(tmp_path):
    script = _script()
    p = tmp_path / "app-compose.json"
    p.write_text(json.dumps({"image": "repo@sha256:aaa"}), encoding="utf-8")
    first, _ = script.compose_hash(str(p))
    p.write_text(json.dumps({"image": "repo@sha256:bbb"}), encoding="utf-8")
    second, _ = script.compose_hash(str(p))
    assert first != second, "a different app must produce a different compose hash"
