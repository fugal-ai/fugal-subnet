"""The compose validators approve is the frozen one (review F-074).

The approved entry is sha256 of a frozen app-compose.json that embeds the
compose text. A live docker-compose edited after its freeze yields a compose
hash no entry names: the miner that deploys it is refused and nothing says
why. scripts/check_safety_invariants.py reports a live compose that no frozen
file embeds: a warning in CI (a compose moves ahead of its freeze between
releases) and a failure with --release, the launch gate.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_safety_invariants.py"


def _mod():
    spec = importlib.util.spec_from_file_location("safety_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path, live_text, frozen_text):
    d = tmp_path / "deploy" / "dstack"
    d.mkdir(parents=True)
    (d / "docker-compose.yaml").write_text(live_text, encoding="utf-8")
    (d / "app-compose.frozen-x.json").write_text(
        json.dumps({"manifest_version": 2, "docker_compose_file": frozen_text}), encoding="utf-8")
    return tmp_path


def test_a_live_compose_that_matches_a_frozen_file_passes(tmp_path):
    text = "services:\n  miner:\n    image: x@sha256:" + "a" * 64 + "\n"
    errors: list[str] = []
    _mod().check_live_composes_are_frozen(errors, root=_tree(tmp_path, text, text))
    assert errors == []


def test_a_live_compose_edited_after_its_freeze_is_refused(tmp_path):
    text = "services:\n  miner:\n    image: x@sha256:" + "a" * 64 + "\n"
    errors: list[str] = []
    _mod().check_live_composes_are_frozen(
        errors, root=_tree(tmp_path, text + "    environment:\n      - X=1\n", text))
    assert len(errors) == 1 and "no approved entry names" in errors[0]


def test_the_drift_is_a_warning_in_ci_and_a_failure_at_release(tmp_path, monkeypatch, capsys):
    """main() runs the real checks on the real checkout; only the compose gate
    is redirected here, so the test holds whatever state the checkout is in."""
    mod = _mod()
    calls = []

    def fake_gate(sink, root=None):
        calls.append(sink)
        sink.append("drifted compose")

    monkeypatch.setattr(mod, "check_live_composes_are_frozen", fake_gate)
    mod.main([])
    assert "WARNING (launch gate" in capsys.readouterr().out
    try:
        mod.main(["--release"])
    except SystemExit as e:
        assert "drifted compose" in str(e)
    else:
        raise AssertionError("--release must fail on a drifted compose")
    assert len(calls) == 2
