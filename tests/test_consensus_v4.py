"""Protocol v4's part of the consensus digest (fugal_subnet/consensus_v4.py).

Every constant of a v4 module, and every constant added to a module v4 code
reads, is classified: in the v4 section of the digest, or named with the
reason it decides nothing a v4 epoch publishes, or a name the module had
before v4. A constant a later package adds fails here until it is
classified. The section exists only for a v4 epoch, so the digest of every
v1-v3 epoch is exactly what it was.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import tempfile
from pathlib import Path

import pytest

from fugal_subnet import consensus_v4 as cv4
from fugal_subnet import fingerprint

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "fugal_subnet"


def test_the_registry_is_complete():
    assert cv4.problems() == []


_TEMPLATE: dict[str, Path] = {}


def _tree(tmp_path) -> Path:
    """A copy of the packages the scan reads and the data files, to break."""
    import shutil

    if "root" not in _TEMPLATE:
        template = Path(tempfile.mkdtemp(prefix="v4-registry-"))
        for top in ("fugal_subnet", "neurons"):
            for src in (ROOT / top).rglob("*.py"):
                if "__pycache__" in src.parts:
                    continue
                dst = template / src.relative_to(ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
        for folder in ("routing_data", "signed_lists"):
            for p in (PACKAGE / folder).rglob("*"):
                if p.is_file() and p.stat().st_size < 100_000:
                    dst = template / p.relative_to(ROOT)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(p.read_bytes())
                elif p.is_file():          # a big data file: its bytes are not read here
                    dst = template / p.relative_to(ROOT)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(b"{}")
        (template / "fugal_subnet" / cv4.PRE_V4_PIN).write_bytes(
            (PACKAGE / cv4.PRE_V4_PIN).read_bytes())
        _TEMPLATE["root"] = template
    dest = Path(tmp_path)
    shutil.copytree(_TEMPLATE["root"], dest, dirs_exist_ok=True)
    return dest


def _append(root: Path, module: str, text: str) -> None:
    path = cv4._module_file(root, module)
    path.write_text(path.read_text() + "\n" + text + "\n")


def test_the_scan_reads_the_modules_neurons_run_and_the_v4_modules():
    """The set of modules read is computed from the imports, not listed."""
    scanned = cv4.scanned_modules()
    for module in ("neurons.validator", "neurons.miner", "fugal_subnet.config",
                   "fugal_subnet.benchmarks.slicer", "fugal_subnet.shared_grades",
                   "fugal_subnet.seat_v4", "fugal_subnet.tee.proof", "fugal_subnet.api",
                   "fugal_subnet.evidence", "fugal_subnet.exploration", "fugal_subnet.pricing"):
        assert module in scanned, module
    # only an entry with a stated reason takes a module out of the scan
    for module, reason in cv4.NOT_SCANNED.items():
        assert module not in scanned and reason.strip(), module
    assert "fugal_subnet.graders" in cv4.NOT_SCANNED


def test_a_new_constant_in_every_scanned_module_fails_until_classified(tmp_path):
    root = _tree(tmp_path)
    assert cv4.problems(root) == []
    scanned = sorted(cv4.scanned_modules(root))
    for module in scanned:
        _append(root, module, "NEW_UNCLASSIFIED_CONSTANT = 7")
    got = cv4.problems(root)
    for module in scanned:
        assert any(p.startswith(f"{module}:NEW_UNCLASSIFIED_CONSTANT has no protocol v4 "
                                "classification") for p in got), module


@pytest.mark.parametrize("label,text,names", [
    ("tuple assignment", "NEW_A, NEW_B = 1, 2", ("NEW_A", "NEW_B")),
    ("nested tuple and starred", "(NEW_A, [NEW_B, *NEW_C]) = (1, [2, 3])",
     ("NEW_A", "NEW_B", "NEW_C")),
    ("inside if", "if True:\n    NEW_A = 3", ("NEW_A",)),
    ("inside else", "if False:\n    pass\nelse:\n    NEW_A = 3", ("NEW_A",)),
    ("inside try", "try:\n    NEW_A = 3\nexcept Exception:\n    NEW_A = 4", ("NEW_A",)),
    ("inside an except handler", "try:\n    pass\nexcept Exception:\n    NEW_A = 4", ("NEW_A",)),
    ("inside with", "import contextlib\nwith contextlib.nullcontext():\n    NEW_A = 3",
     ("NEW_A",)),
    ("inside a loop", "while False:\n    NEW_A = 3", ("NEW_A",)),
    ("annotated with a value", "NEW_A: int = 5", ("NEW_A",)),
    ("list target", "[NEW_A] = [6]", ("NEW_A",)),
    ("class attribute", "class Rules:\n    NEW_A = 9", ("Rules.NEW_A",)),
    ("class attribute under if", "class Rules:\n    if True:\n        NEW_A: int = 9",
     ("Rules.NEW_A",)),
    ("nested class", "class Rules:\n    class Inner:\n        NEW_A = 1", ("Rules.Inner.NEW_A",)),
    ("walrus", "(NEW_A := 11)", ("NEW_A",)),
    ("walrus in a condition", "if (NEW_A := 11):\n    pass", ("NEW_A",)),
    ("for target", "for NEW_A in (1,):\n    pass", ("NEW_A",)),
    ("with target", "import contextlib\nwith contextlib.nullcontext(1) as NEW_A:\n    pass",
     ("NEW_A",)),
    ("except target", "try:\n    pass\nexcept Exception as NEW_A:\n    pass", ("NEW_A",)),
    ("match capture", "match 1:\n    case NEW_A:\n        pass", ("NEW_A",)),
    ("a def", "def NEW_A():\n    pass", ("NEW_A",)),
    ("a global declared in a function", "def f():\n    global NEW_A\n    NEW_A = 1", ("NEW_A",)),
    ("an alias of an import", "from fugal_subnet.config import EPOCH_INTERVAL as NEW_A", ()),
])
def test_every_form_of_module_level_binding_is_seen(tmp_path, label, text, names):
    root = _tree(tmp_path)
    _append(root, "fugal_subnet.seat_v4", text)
    got = cv4.problems(root)
    for name in names:
        assert any(p.startswith(f"fugal_subnet.seat_v4:{name} has no protocol v4 classification")
                   for p in got), (label, name, got[:3])
    if not names:       # an import only re-exports a value its own module classifies
        assert got == []


def test_a_write_the_scan_cannot_read_is_refused(tmp_path):
    for label, text in (("globals() write", "globals()['NEW_A'] = 1"),
                        ("globals().update", "globals().update(NEW_A=1)"),
                        ("exec", "exec('NEW_A = 1')"),
                        ("setattr on a module",
                         "import sys\nsetattr(sys.modules[__name__], 'NEW_A', 1)"),
                        ("an attribute written on a module",
                         "import sys\nsys.modules[__name__].NEW_A = 1")):
        root = _tree(tmp_path / label.replace(" ", "_").replace("(", "").replace(")", ""))
        _append(root, "fugal_subnet.seat_v4", text)
        assert any("binds names the registry scan cannot read" in p
                   for p in cv4.problems(root)), label


def test_rebinding_a_constant_by_an_augmented_assignment_is_refused(tmp_path):
    root = _tree(tmp_path)
    _append(root, "fugal_subnet.seat_v4", "STANDING_EPOCHS += 0")
    assert any("fugal_subnet.seat_v4:STANDING_EPOCHS is rebound by an augmented assignment" in p
               for p in cv4.problems(root))


def test_a_bare_annotation_binds_nothing(tmp_path):
    root = _tree(tmp_path)
    _append(root, "fugal_subnet.seat_v4", "class Fields:\n    W: int")
    assert cv4.problems(root) == []


def test_a_module_neurons_reaches_only_through_other_modules_is_scanned(tmp_path):
    """The closure is transitive: neurons -> a -> b, `from . import`, an
    import inside a function, an import under a condition, a module named by
    a string literal, `importlib.import_module` of a literal."""
    root = _tree(tmp_path)
    pkg = root / "fugal_subnet"
    (pkg / "hop_a.py").write_text("from . import hop_b\n")
    (pkg / "hop_b.py").write_text("def f():\n    import fugal_subnet.hop_c\n")
    (pkg / "hop_c.py").write_text(
        "import importlib\nM = importlib.import_module('fugal_subnet.hop_d')\n")
    (pkg / "hop_d.py").write_text("NAMES = ('fugal_subnet.hop_e',)\n")
    (pkg / "hop_e.py").write_text("if True:\n    from fugal_subnet import hop_f\n")
    (pkg / "hop_f.py").write_text("DEEP_CONSTANT = 1\n")
    assert cv4.problems(root) == []                       # nothing imports hop_a yet
    _append(root, "neurons.validator", "import fugal_subnet.hop_a")
    assert any(p.startswith("fugal_subnet.hop_f:DEEP_CONSTANT has no protocol v4 classification")
               for p in cv4.problems(root))
    assert "fugal_subnet.hop_f" in cv4.scanned_modules(root)


@pytest.mark.parametrize("importer", [
    "import importlib\nM = importlib.import_module('fugal_subnet.' + 'seat_v4')\nMARGIN = 3\n",
    "from . import seat_v4\nMARGIN = 3\n",
    "L = __import__('fugal_subnet.labels_v4')\nMARGIN = 3\n",
    "import fugal_subnet.seat_v4 as s\nMARGIN = 3\n",
])
def test_a_new_module_that_reads_v4_code_is_scanned(tmp_path, importer):
    root = _tree(tmp_path)
    (root / "fugal_subnet" / "new_reader.py").write_text(importer)
    got = cv4.problems(root)
    # a literal import puts the module in the scan, where its constant is
    # unclassified; a computed one is refused by name
    assert any(p.startswith("fugal_subnet.new_reader:MARGIN has no protocol v4 classification")
               or "fugal_subnet.new_reader imports by a name it computes" in p
               for p in got), got


def test_a_module_that_imports_by_a_computed_name_is_refused_unless_listed(tmp_path):
    root = _tree(tmp_path)
    (root / "fugal_subnet" / "dyn.py").write_text(
        "import importlib\n\n\ndef load(name):\n    return importlib.import_module(name)\n")
    _append(root, "neurons.validator", "import fugal_subnet.dyn")
    assert any("fugal_subnet.dyn imports by a name it computes" in p for p in cv4.problems(root))
    assert "fugal_subnet.benchmarks.loader" in cv4.DYNAMIC_IMPORTS


def test_a_module_not_scanned_needs_a_stated_reason(monkeypatch):
    monkeypatch.setitem(cv4.NOT_SCANNED, "fugal_subnet.graders", " ")
    assert "fugal_subnet.graders is in NOT_SCANNED without a reason" in cv4.problems()


def test_a_new_data_file_fails_until_classified(tmp_path):
    root = _tree(tmp_path)
    (root / "fugal_subnet" / "routing_data" / "published_set_v4_seed.json").write_text("{}")
    (root / "fugal_subnet" / "routing_data" / "menu_extra.json").write_text("{}")
    (root / "fugal_subnet" / "signed_lists" / "other.json").write_text("{}")
    (root / "fugal_subnet" / "signed_lists" / "sub").mkdir()
    (root / "fugal_subnet" / "signed_lists" / "sub" / "other.json").write_text("{}")
    got = "\n".join(cv4.problems(root))
    for rel in ("routing_data/published_set_v4_seed.json", "routing_data/menu_extra.json",
                "signed_lists/other.json", "signed_lists/sub/other.json"):
        assert f"fugal_subnet/{rel} is a data file in neither" in got, rel


def test_a_classified_name_that_disappears_fails(tmp_path):
    root = _tree(tmp_path)
    path = root / "fugal_subnet" / "seat_bound.py"
    path.write_text(path.read_text().replace("LN2_UP = ", "LN2_UPPER = "))
    got = cv4.problems(root)
    assert "fugal_subnet.seat_bound:LN2_UP is classified but not defined (renamed or deleted?)" in got


def _edit_registry(root: Path, old: str, new: str) -> None:
    path = root / "fugal_subnet" / "consensus_v4.py"
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new, 1))


def _problems_of_copy(root: Path) -> list[str]:
    """problems() of the copy's own consensus_v4 (its registry edits count)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cv4_copy", root / "fugal_subnet" / "consensus_v4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.problems(root)


def test_a_name_moved_from_registered_to_not_v4_fails_until_the_pin_is_updated(tmp_path):
    root = _tree(tmp_path)
    assert _problems_of_copy(root) == []
    _edit_registry(root, '"UNITS_PER_POINT", "STANDING_EPOCHS", "ODDS_MARGIN_POINTS",',
                   '"UNITS_PER_POINT", "ODDS_MARGIN_POINTS",')
    # seat_v4 already has a NOT_V4 name (RULE_C_FIRST_ROUND): its pin changes
    _edit_registry(root, '    "fugal_subnet.seat_v4": {\n',
                   '    "fugal_subnet.seat_v4": {\n        "STANDING_EPOCHS": "a reason",\n')
    got = _problems_of_copy(root)
    assert any("fugal_subnet.seat_v4: the names classified NOT_V4 or PRE_V4 changed" in p
               for p in got), got


def test_a_name_moved_within_a_pinned_module_fails_until_the_pin_is_updated(tmp_path):
    root = _tree(tmp_path)
    _edit_registry(root, '"STANDING_EPOCHS_V4", "LATE_ODDS_SLOTS_V4",', '"STANDING_EPOCHS_V4",')
    _edit_registry(root, '"fugal_subnet.config": {\n        "V4_ACTIVATION_BLOCK_ENV":',
                   '"fugal_subnet.config": {\n        "LATE_ODDS_SLOTS_V4": "a reason",\n'
                   '        "V4_ACTIVATION_BLOCK_ENV":')
    got = _problems_of_copy(root)
    assert any("fugal_subnet.config: the names classified NOT_V4 or PRE_V4 changed" in p
               for p in got), got


def test_a_name_moved_into_pre_v4_fails(tmp_path):
    root = _tree(tmp_path)
    _edit_registry(root, '"STANDING_EPOCHS_V4", "LATE_ODDS_SLOTS_V4",', '"STANDING_EPOCHS_V4",')
    _edit_registry(root, "SCORE_QUALITY_FLOOR SINGLE_MODEL_MIN_PAIRS",
                   "SCORE_QUALITY_FLOOR SINGLE_MODEL_MIN_PAIRS LATE_ODDS_SLOTS_V4")
    got = _problems_of_copy(root)
    assert any("config:LATE_ODDS_SLOTS_V4 is in PRE_V4 but not in consensus_v4_pre.json" in p
               for p in got), got
    assert any("fugal_subnet.config: the names classified NOT_V4 or PRE_V4 changed" in p
               for p in got)


def test_a_name_in_not_v4_or_pre_v4_needs_a_reason(monkeypatch):
    monkeypatch.setitem(cv4.NOT_V4["fugal_subnet.menu_publish"], "PUBLISH_BUDGET_S", "")
    monkeypatch.setitem(cv4.PRE_V4["fugal_subnet.config"], "API_TIMEOUT", " ")
    got = cv4.problems()
    assert "fugal_subnet.menu_publish:PUBLISH_BUDGET_S is in NOT_V4 without a reason" in got
    assert "fugal_subnet.config:API_TIMEOUT is in PRE_V4 without a reason" in got


def test_the_class_pins_are_what_the_tables_hash_to():
    assert cv4.CLASS_PINS == cv4.class_hashes()


def test_a_pre_v4_name_is_the_v3_releases_name_with_its_value(tmp_path):
    """Pinned in the repo (no git needed): the pin file names exactly the
    PRE_V4 names, with the v3 release's hash of the statements that bind
    each, and a name whose statements changed since fails."""
    pin = cv4.pre_v4_pin()
    assert {m: set(n) for m, n in pin.items() if n} == {
        m: set(n) for m, n in cv4.PRE_V4.items() if n}
    root = _tree(tmp_path)
    path = root / "fugal_subnet" / "config.py"
    path.write_text(path.read_text().replace("API_TIMEOUT =", "API_TIMEOUT = 1 +", 1))
    got = cv4.problems(root)
    assert any("config:API_TIMEOUT is in PRE_V4 but is not what the v3 release has" in p
               for p in got), got
    root2 = _tree(tmp_path / "second")
    (root2 / "fugal_subnet" / cv4.PRE_V4_PIN).write_text(
        json.dumps({"modules": {m: {k: v for k, v in names.items() if k != "API_TIMEOUT"}
                                for m, names in pin.items()}}))
    assert any("config:API_TIMEOUT is in PRE_V4 but not in consensus_v4_pre.json" in p
               for p in cv4.problems(root2))


def _release_file(path: str) -> bytes | None:
    import subprocess

    try:
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"origin/main:{path}"],
                               capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return shown.stdout if shown.returncode == 0 else None


def test_the_pre_v4_pin_is_what_the_v3_release_has():
    """Against the v3 release itself (origin/main), where it is available:
    every PRE_V4 name is a constant of the release's module with the same
    statements, as it is now, and the pin file records exactly that."""
    if _release_file("fugal_subnet/config.py") is None:
        pytest.skip("origin/main is not available in this checkout; the pin file "
                    "(fugal_subnet/consensus_v4_pre.json) is checked without it")
    pin = cv4.pre_v4_pin()
    for module, names in sorted(cv4.PRE_V4.items()):
        shown = _release_file(module.replace(".", "/") + ".py")
        assert shown is not None, f"{module} is not in the v3 release"
        release = cv4.binding_hashes(cv4.source_bindings(shown.decode()))
        now = cv4.pre_v4_hashes(ROOT, module)
        for name in sorted(names):
            assert name in release, f"{module}:{name} is not a constant of the v3 release"
            assert pin[module][name] == release[name], f"{module}:{name} differs from the pin"
            assert now[name] == release[name], f"{module}:{name} is not what the v3 release has"


def test_every_v4_config_name_is_classified():
    from fugal_subnet import config

    v4 = {n for n in vars(config) if cv4._CONSTANT.fullmatch(n) and "V4" in n}
    known = set(cv4.REGISTERED["fugal_subnet.config"]) | set(cv4.NOT_V4["fugal_subnet.config"])
    assert v4 <= known, sorted(v4 - known)


def test_the_pinned_files_are_hashed_from_the_package():
    hashes = cv4.file_hashes()
    assert hashes["menu_v4_seed"] == hashlib.sha256(
        (PACKAGE / "routing_data" / "menu_v4_seed.json").read_bytes()).hexdigest()
    assert set(hashes) == set(cv4.PINNED_FILES)


def test_every_registered_constant_has_an_exact_form():
    values = cv4.constants()
    count = sum(len(v) for v in cv4.REGISTERED.values())
    assert len(values) == count
    unread = {k: v for k, v in values.items() if isinstance(v, str) and v.startswith("unavailable")}
    assert not unread, unread
    json.dumps(values)
    assert values["fugal_subnet.seat_bound:A_ALPHA"] == "9211/1000"
    assert values["fugal_subnet.menu_state:MENU_LAG"] == 2
    with pytest.raises(TypeError):
        cv4.canonical(object())
    with pytest.raises(TypeError):
        cv4.canonical(len)


def _pre_v4_digest(fp: dict) -> str:
    """The digest's formula as it was before v4 was built."""
    material = {k: fp[k] for k in ("epoch", "packages", "numpy_blas", "cpu_dispatch",
                                   "grader_sha256", "routing")}
    material["scoring"] = fp.get("scoring", {})
    material["routing_protocol"] = fp.get("routing_protocol", "")
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def test_a_v1_to_v3_epoch_publishes_the_digest_it_published_before():
    plain = fingerprint.environment_fingerprint()
    for version in (None, 1, 2, 3):
        fp = fingerprint.environment_fingerprint(protocol_version=version)
        assert "v4" not in fp and set(fp) == set(plain)
        assert fingerprint.consensus_digest(protocol_version=version) == _pre_v4_digest(fp)
        assert fingerprint.consensus_digest(fp) == fingerprint.consensus_digest()
    v4 = fingerprint.environment_fingerprint(protocol_version=4)
    assert set(v4) == set(plain) | {"v4"}
    assert fingerprint.consensus_digest(v4) != fingerprint.consensus_digest(plain)
    assert {k: v for k, v in v4.items() if k not in ("v4", "code", "torch_runtime")} == {
        k: v for k, v in plain.items() if k not in ("code", "torch_runtime")}


@pytest.mark.parametrize("module,name,value", [
    ("fugal_subnet.menu_state", "MENU_LAG", 3),
    ("fugal_subnet.config", "SLICE_SIZE_V4", 199),
    ("fugal_subnet.seat_bound", "A_ALPHA", None),
    ("fugal_subnet.tee.proof", "MAX_PROVIDER_COST_NANO", 1),
    ("fugal_subnet.menu_rules", "LIST_SIGNERS_SHA256", "0" * 64),
    # the seat: a config value and one the seat module compiles
    ("fugal_subnet.config", "STANDING_EPOCHS_V4", 4),
    ("fugal_subnet.seat_v4", "MAX_PREDECESSORS", 9),
    # the memorisation check: a config value and one memo_v4 compiles
    ("fugal_subnet.config", "MEMO_MIN_LABELS_V4", 501),
    ("fugal_subnet.memo_v4", "READ_LAG", 3),
])
def test_one_differing_v4_constant_moves_the_v4_digest_and_not_the_v3_one(
        monkeypatch, module, name, value):
    v3, v4 = fingerprint.consensus_digest(), fingerprint.consensus_digest(protocol_version=4)
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, name, value)
    assert fingerprint.consensus_digest(protocol_version=4) != v4
    assert fingerprint.consensus_digest() == v3
    assert fingerprint.consensus_digest(protocol_version=3) == v3


def test_a_differing_pinned_file_or_block_moves_the_v4_digest_only(monkeypatch, tmp_path):
    v3, v4 = fingerprint.consensus_digest(), fingerprint.consensus_digest(protocol_version=4)
    copy = tmp_path / "fugal_subnet"
    for rel, _req in cv4.PINNED_FILES.values():
        src = PACKAGE / rel
        if src.exists():
            (copy / rel).parent.mkdir(parents=True, exist_ok=True)
            (copy / rel).write_bytes(src.read_bytes())
    monkeypatch.setattr(cv4, "PACKAGE", copy)
    assert fingerprint.consensus_digest(protocol_version=4) == v4
    mix = copy / "routing_data" / "pool_mix_v4.json"
    mix.write_bytes(mix.read_bytes() + b" ")
    assert fingerprint.consensus_digest(protocol_version=4) != v4
    assert fingerprint.consensus_digest() == v3
    monkeypatch.setattr(cv4, "PACKAGE", PACKAGE)
    from fugal_subnet import config

    monkeypatch.setattr(config, "FINNEY_V4_ACTIVATION_BLOCK", 9_999_360)
    assert fingerprint.consensus_digest(protocol_version=4) != v4
    assert fingerprint.consensus_digest() == v3


def test_the_reveal_of_a_v3_epoch_has_no_v4_section_and_a_v4_reveal_has_one(tmp_path,
                                                                             monkeypatch):
    from fugal_subnet import commit_reveal

    monkeypatch.setattr(commit_reveal, "EPOCH_DIR", str(tmp_path))
    questions = [{"question_id": "q1", "gold": "1"}]
    digests = {}
    for version in (3, 4):
        eid = f"e0000000{version}"
        commit_reveal.commit_epoch(eid, questions, "0xab")
        assert commit_reveal.reveal_epoch(eid, questions, [[1]], ["m/a"], {"m/a": 0.0}, {}, {},
                                          {}, {}, protocol_version=version)
        reveal = json.loads((tmp_path / eid / "reveal.json").read_text())
        assert ("v4" in reveal["environment"]) is (version == 4)
        assert reveal["consensus_digest"] == fingerprint.consensus_digest(reveal["environment"])
        digests[version] = reveal["consensus_digest"]
    assert digests[3] == fingerprint.consensus_digest()
    assert digests[4] == fingerprint.consensus_digest(protocol_version=4) != digests[3]


def test_the_seat_and_memorisation_packages_are_registered_and_the_seed_is_required():
    """The seat's and the memorisation check's constants are in the section,
    the published-set seed is a required pinned file, and the names the v4
    path stopped reading when the seat package removed their rules are
    classified as not v4, with a reason."""
    values = cv4.constants()
    for key in ("fugal_subnet.config:STANDING_EPOCHS_V4", "fugal_subnet.config:ODDS_MIN_PAIRS_V4",
                "fugal_subnet.config:ODDS_MARGIN_POINTS_V4", "fugal_subnet.config:LATE_ODDS_SLOTS_V4",
                "fugal_subnet.config:EXPLORE_PRIVATE_MIN_SAMPLES_V4",
                "fugal_subnet.config:MEMO_MARGIN_POINTS_V4", "fugal_subnet.config:MEMO_MIN_LABELS_V4",
                "fugal_subnet.config:MEMO_MIN_QUESTIONS_V4",
                "fugal_subnet.config:FRESH_RESERVE_EPOCHS_V4",
                "fugal_subnet.seat_v4:MAX_STATE_HEADS", "fugal_subnet.seat_v4:MAX_ROWS",
                "fugal_subnet.seat_v4:MAX_ENTRY_ID", "fugal_subnet.seat_v4:HISTORY_KEEP",
                "fugal_subnet.seat_v4:MAX_PREDECESSORS", "fugal_subnet.memo_v4:READ_LAG"):
        assert key in values, key
    assert cv4.PINNED_FILES["published_v4"] == ("routing_data/published_v4.json", True)
    assert cv4.file_hashes()["published_v4"] == hashlib.sha256(
        (PACKAGE / "routing_data" / "published_v4.json").read_bytes()).hexdigest()
    gone = {"CHAMPION_GHOST_OFF", "CHAMPION_GHOST_RETIRE", "CHAMPION_CONFIRM_QUESTIONS",
            "ODDS_CONFIRM_PAIRS", "CHAMPION_MIN_ACCURACY", "V2_BURN_IN_QUESTIONS",
            "EXPLORE_DISAGREE_MIN_SAMPLES", "DEDUP_SIMILARITY_THRESHOLD", "MAX_WEIGHT_DELTA"}
    assert gone <= set(cv4.NOT_V4["fugal_subnet.config"])
    assert not gone & set(cv4.PRE_V4["fugal_subnet.config"])
    for name in ("NV_UNITS_PER_POINT", "HISTORY_KEEP"):
        assert f"fugal_subnet.champion:{name}" not in values
    for module in ("fugal_subnet.attacks.seat_v4_world", "fugal_subnet.attacks.memo_seat_world"):
        assert module in cv4.NOT_SCANNED


def test_a_missing_published_set_seed_fails(tmp_path):
    root = _tree(tmp_path)
    (root / "fugal_subnet" / "routing_data" / "published_v4.json").unlink()
    assert ("fugal_subnet/routing_data/published_v4.json (published_v4) is required and missing"
            in cv4.problems(root))
