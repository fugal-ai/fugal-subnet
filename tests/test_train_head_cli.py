"""scripts/train_head.py refuses misleading input before the expensive part.

--synthetic silently ignored --manifest (the documented commands passed one),
and v2 models against the default v1 price table surfaced as a bare KeyError
only after training.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run(*args, tmp_path):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "train_head.py"), *args,
                           "--output", str(tmp_path / "h.npz")],
                          capture_output=True, text=True, cwd=ROOT, timeout=120)


def test_synthetic_refuses_a_manifest(tmp_path):
    r = _run("--synthetic", "--manifest", "data/benchmark_tokens_protocol_v2.json",
             "--models", "a", tmp_path=tmp_path)
    assert r.returncode == 2 and "test-only token manifest by design" in r.stderr


def test_unpriced_models_are_refused_before_training(tmp_path):
    r = _run("--synthetic", "--models", "anthropic/claude-haiku-5.5", tmp_path=tmp_path)
    assert r.returncode == 2 and "need --prices data/models_v2.json" in r.stderr
    assert not (tmp_path / "h.npz").exists()


def test_the_documented_synthetic_command_runs(tmp_path):
    r = _run("--synthetic", "--n-questions", "60", "--epochs", "2",
             "--prices", "data/models_v2.json",
             "--models", "anthropic/claude-haiku-5.5", "openai/gpt-6.1-terra", tmp_path=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "h.npz").exists()
