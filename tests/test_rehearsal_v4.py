"""scripts/rehearsal_v4.py runs to the end with every scenario passing, the
seat's included: no row is skipped."""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_the_in_process_v4_rehearsal_passes(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("rehearsal_v4", ROOT / "scripts" / "rehearsal_v4.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rehearsal_v4"] = mod           # its dataclasses look their module up
    spec.loader.exec_module(mod)
    assert mod.main(["--keep", str(tmp_path / "run")]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "0 failed" in out and "0 skipped" in out and "SKIP " not in out
    for name, _run, _check in mod.SEAT_HOOKS.values():
        assert name in out
