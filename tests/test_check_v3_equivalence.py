"""scripts/check_v3_equivalence.py: the comparison, and this tree's own lines.

The comparison passes two line sets only when every tag is on both sides,
every value is equal except the two differences protocol v4 intends, and
each of those has its exact shape on each side. This tree (with v4) must
produce the v4 side of both.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("check_v3_equivalence_under_test",
                                                  ROOT / "scripts" / "check_v3_equivalence.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EQ = _load()
SHA = "a" * 64
FOUR_NEW = json.dumps({"pointer": ["e00026500", SHA],
                       "commitment": ["e00026500", SHA, EQ._MENU_HASH]})
FOUR_OLD = json.dumps({"pointer": None})
NS_NEW = json.dumps({"parses": False, "error": "ValueError"})
NS_OLD = json.dumps({"parses": True, "hash": "57a1ae8286e1d24d"})


def _sides(**over_new):
    common = [("v3/all/00", "1111"), ("digest/v1_v3", "c857")]
    new = dict(common + [("commit/four_field", FOUR_NEW), ("proof/v4_namespace", NS_NEW)])
    old = dict(common + [("commit/four_field", FOUR_OLD), ("proof/v4_namespace", NS_OLD)])
    new.update(over_new)
    return list(new.items()), list(old.items())


def test_the_two_intended_differences_alone_pass():
    report, ok = EQ.compare(*_sides())
    assert ok, report
    assert sum(line.startswith("intended") for line in report) == 2


def test_any_other_difference_fails():
    report, ok = EQ.compare(*_sides(**{"v3/all/00": "2222"}))
    assert not ok and any("UNEXPECTED v3/all/00" in line for line in report)


def test_an_intended_line_with_the_wrong_shape_fails():
    # The four-field commitment read without its menu hash, or not refused
    # by the reference; the namespace proof parsing on both sides.
    wrong = json.dumps({"pointer": ["e00026500", SHA], "commitment": ["e00026500", SHA, None]})
    assert not EQ.compare(*_sides(**{"commit/four_field": wrong}))[1]
    new, old = _sides()
    old = [(t, FOUR_NEW if t == "commit/four_field" else v) for t, v in old]
    assert not EQ.compare(new, old)[1]
    assert not EQ.compare(*_sides(**{"proof/v4_namespace": NS_OLD}))[1]


def test_a_missing_line_fails():
    new, old = _sides()
    assert not EQ.compare(new[:-1], old)[1]
    assert not EQ.compare(new, old + [("menu_b/00", "x")])[1]


def _published_dirs(tmp_path):
    """Six v3 reveals, as a validator writes them (a directory per epoch) and
    as published reveals are downloaded (a flat directory, with a manifest)."""
    import shutil

    import tests.test_v3_epoch as T3
    from tests.test_v2_rebuild import write

    _results, reveals = T3.live(6)
    dirs, flat = tmp_path / "dirs", tmp_path / "flat"
    dirs.mkdir()
    flat.mkdir()
    write(dirs, json.loads(json.dumps(reveals)))
    for p in sorted(dirs.glob("*/reveal.json")):
        shutil.copy(p, flat / f"{p.parent.name}.json")
    (flat / "MANIFEST.json").write_text("{}")
    return dirs, flat


def test_published_reveals_are_read_in_either_layout(tmp_path):
    import hashlib
    import shutil

    import pytest

    dirs, flat = _published_dirs(tmp_path)
    a, b = EQ.reveal_paths(str(dirs)), EQ.reveal_paths(str(flat))
    assert [e for e, _p in a] == [e for e, _p in b] and len(a) == 6

    def blob(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

    lines = EQ._published(str(dirs), blob, None)
    assert lines == EQ._published(str(flat), blob, None)
    assert dict(lines)["reveals/count"] == "6"
    assert sum(tag.startswith("reveals/e") for tag, _v in lines) == 6
    # One epoch in both layouts in one directory is refused, by name.
    shutil.copy(flat / f"{a[0][0]}.json", dirs / f"{a[0][0]}.json")
    with pytest.raises(SystemExit, match=f"{a[0][0]} is there twice"):
        EQ.reveal_paths(str(dirs))


def test_a_reveals_directory_with_no_reveal_is_refused_before_anything_runs(
        tmp_path, capsys, monkeypatch):
    import pytest

    def ran(*_a, **_k):
        pytest.fail("a tree was run")

    monkeypatch.setattr(EQ, "run_tree", ran)
    deeper = tmp_path / "deeper" / "reveals" / "e00026466"     # a layout it does not read
    deeper.mkdir(parents=True)
    (deeper / "reveal.json").write_text("{}")
    (tmp_path / "deeper" / "notes.json").write_text("{}")
    for src in (tmp_path / "deeper", tmp_path / "missing"):
        if src.name == "missing":
            with pytest.raises(SystemExit, match="cannot list it"):
                EQ.main(["--tree", "a", "--tree", "b", "--reveals", str(src)])
            continue
        assert EQ.main(["--tree", "a", "--tree", "b", "--reveals", str(src)]) == 2
        out = capsys.readouterr().out
        assert "NO REVEALS" in out and "EQUIVALENT:" not in out
    with pytest.raises(SystemExit, match="no reveal read"):
        EQ._published(str(tmp_path / "deeper"), None, None)


def test_this_tree_produces_the_v4_side_of_both_intended_lines():
    lines = dict(EQ.run_tree(str(ROOT)))
    assert EQ._check_four_field(lines["commit/four_field"], FOUR_OLD) is None
    assert EQ._check_v4_namespace(lines["proof/v4_namespace"], NS_OLD) is None
    assert json.loads(lines["proof/v3"])["parses"] is True
    assert lines["proof/v3"] == lines["proof/v3_with_v4_keys"]
    assert len(lines) > 100
