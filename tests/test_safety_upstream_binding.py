"""check_safety_invariants' upstream rule: the model upstream is fixed by the
measured compose and nothing else (INVARIANTS I8, diligence 2026-10 finding 1).

The rule had no test, and it matched only the list form of a compose
environment entry (`- FUGAL_OPENROUTER_BASE=...`): the mapping form
(`FUGAL_OPENROUTER_BASE: ...`) passed it (found 2026-11-08). A finney miner
would still refuse to start with the variable set, but a testnet compose only
warns, and this check is what stops such a compose being frozen at all.
"""
import importlib.util
import json
import pathlib
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("csi", REPO / "scripts" / "check_safety_invariants.py")
csi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csi)

FROZEN = sorted((REPO / "deploy" / "dstack").glob("app-compose*.json"))[0]


def _errors(tmp_path, monkeypatch, edit=lambda doc: doc, runtime_extra=""):
    (tmp_path / "deploy" / "dstack").mkdir(parents=True)
    (tmp_path / "fugal_subnet" / "tee").mkdir(parents=True)
    runtime = (REPO / "fugal_subnet" / "tee" / "runtime.py").read_text() + runtime_extra
    (tmp_path / "fugal_subnet" / "tee" / "runtime.py").write_text(runtime)
    doc = edit(json.loads(FROZEN.read_text()))
    (tmp_path / "deploy" / "dstack" / FROZEN.name).write_text(json.dumps(doc))
    monkeypatch.setattr(csi, "ROOT", tmp_path)
    errors: list[str] = []
    csi.check_upstream_is_inside_the_measurement(errors)
    return errors


def _with_env(line):
    def edit(doc):
        assert "environment:" in doc["docker_compose_file"]
        doc["docker_compose_file"] = doc["docker_compose_file"].replace(
            "environment:", "environment:\n      " + line, 1)
        return doc
    return edit


def test_the_shipped_frozen_compose_passes(tmp_path, monkeypatch):
    assert _errors(tmp_path, monkeypatch) == []


@pytest.mark.parametrize("line", [
    "- FUGAL_OPENROUTER_BASE=http://127.0.0.1:8799",
    "FUGAL_OPENROUTER_BASE: http://127.0.0.1:8799",
    'FUGAL_OPENROUTER_BASE : "http://127.0.0.1:8799"',
])
def test_a_compose_that_sets_the_upstream_is_refused(tmp_path, monkeypatch, line):
    errors = _errors(tmp_path, monkeypatch, _with_env(line))
    assert any("sets FUGAL_OPENROUTER_BASE" in e for e in errors), errors


def test_a_commented_mention_is_not_a_setting(tmp_path, monkeypatch):
    assert _errors(tmp_path, monkeypatch, _with_env("# FUGAL_OPENROUTER_BASE: never set here")) == []


def test_an_allowed_env_is_refused(tmp_path, monkeypatch):
    def edit(doc):
        doc["allowed_envs"] = ["FUGAL_OPENROUTER_BASE"]
        return doc
    assert any("allowed_envs" in e for e in _errors(tmp_path, monkeypatch, edit))


def test_an_image_by_tag_is_refused(tmp_path, monkeypatch):
    def edit(doc):
        lines = doc["docker_compose_file"].splitlines()
        doc["docker_compose_file"] = "\n".join(
            (ln.split("@sha256:")[0] if ln.strip().startswith("image:") else ln) for ln in lines)
        return doc
    assert any("by tag" in e for e in _errors(tmp_path, monkeypatch, edit))


@pytest.mark.parametrize("needle", ["ssl._create_unverified_context()", "ssl.CERT_NONE",
                                    "ctx.check_hostname = False"])
def test_a_proxy_that_stops_checking_certificates_is_refused(tmp_path, monkeypatch, needle):
    errors = _errors(tmp_path, monkeypatch, runtime_extra=f"\n# {needle}\n")
    assert any("validate the upstream certificate" in e for e in errors)


def test_no_frozen_compose_at_all_is_refused(tmp_path, monkeypatch):
    (tmp_path / "fugal_subnet" / "tee").mkdir(parents=True)
    shutil.copy(REPO / "fugal_subnet" / "tee" / "runtime.py", tmp_path / "fugal_subnet" / "tee")
    monkeypatch.setattr(csi, "ROOT", tmp_path)
    errors: list[str] = []
    csi.check_upstream_is_inside_the_measurement(errors)
    assert any("no app-compose" in e for e in errors)
