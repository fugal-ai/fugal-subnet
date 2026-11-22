"""check_safety_invariants' rules for the signed lists.

The list signers are a trust root and the shipped lists are consensus data, so
both get what the price tables and the TPM root get: a pin, and a check that
what ships is what was reviewed. A shipped list must verify under the pinned
signers; the unsigned draft is never a list. Each case builds a small tree in
a temporary directory with throwaway keys.
"""
import hashlib
import importlib.util
import pathlib

import pytest

from fugal_subnet import menu_rules as mr
from tests.signed_list_fixtures import Signer, allowlist_bytes, device_bytes

ROOT = pathlib.Path(__file__).resolve().parents[1]
NS = mr.LIST_NAMESPACE

_spec = importlib.util.spec_from_file_location(
    "csi_signed_lists", ROOT / "scripts" / "check_safety_invariants.py")
csi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csi)


class Tree:
    """A repository skeleton holding only what check_signed_lists reads."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        self.owner = Signer(root, "throwaway_release")
        (root / "deploy" / "keys").mkdir(parents=True)
        self.lists = root / "fugal_subnet" / "signed_lists"
        self.lists.mkdir(parents=True)
        (root / "data").mkdir()
        (root / "neurons").mkdir()
        self.set_signers(self.owner.signers_line(namespaces=f"git,{NS}"))
        self.pins: dict[str, str | None] = {"menu_allowlist.json": None, "device_list.json": None}

    def set_signers(self, text: str, copy: str | None = None):
        (self.root / "deploy" / "keys" / "allowed_signers").write_text(text)
        (self.lists / "allowed_signers").write_text(text if copy is None else copy)
        self.signers_pin = hashlib.sha256(text.encode()).hexdigest()

    def ship(self, name: str, payload: bytes, signature: bytes | None, pin: bool = True):
        (self.lists / name).write_bytes(payload)
        if signature is not None:
            (self.lists / f"{name}.sig").write_bytes(signature)
        self.pins[name] = hashlib.sha256(payload).hexdigest() if pin else None

    def draft(self, payload: bytes):
        (self.root / "data" / "menu_allowlist.draft.json").write_bytes(payload)

    def errors(self) -> list[str]:
        errors: list[str] = []
        csi.check_signed_lists(errors, root=self.root, signers_pin=self.signers_pin,
                               list_pins=self.pins)
        return errors


@pytest.fixture
def tree(tmp_path):
    return Tree(tmp_path)


def one_error(tree, needle):
    errors = tree.errors()
    assert len(errors) == 1 and needle in errors[0], errors


# ── the repository as it is ─────────────────────────────────────────────────

def test_the_repository_passes():
    errors: list[str] = []
    csi.check_signed_lists(errors)
    assert errors == []


def test_the_shipped_signers_are_the_release_signers_with_the_list_namespace():
    released = (ROOT / "deploy" / "keys" / "allowed_signers").read_bytes()
    assert mr.SIGNERS_PATH.read_bytes() == released
    assert mr.list_signers() == released
    assert hashlib.sha256(released).hexdigest() == mr.LIST_SIGNERS_SHA256 == \
        csi.PINNED_LIST_SIGNERS_SHA256
    permitted = mr.permitted_signers(released)
    assert sorted(permitted.values()) == ["release2@fugalai.com", "release@fugalai.com"]
    assert mr.unread_signers_lines(released) == []
    # every release key still signs git tags: the list namespace was added, not swapped in
    for line in released.decode().splitlines():
        assert f'namespaces="git,{NS}"' in line, line


def test_no_list_ships_unsigned_and_the_draft_is_not_a_list():
    """Today: a draft allowlist for the owner to review, and no signed list."""
    draft = (ROOT / "data" / "menu_allowlist.draft.json").read_bytes()
    assert mr.parse_unsigned_list(draft, mr.KIND_MENU_ALLOWLIST).serial == mr.DRAFT_SERIAL
    assert not (ROOT / "data" / "menu_allowlist.draft.json.sig").exists()
    for kind in (mr.KIND_MENU_ALLOWLIST, mr.KIND_DEVICE_LIST):
        with pytest.raises(FileNotFoundError):
            mr.shipped_list(kind)
    assert sorted(p.name for p in mr.SIGNED_LISTS_DIR.iterdir()
                  if p.name != "__pycache__") == ["allowed_signers"]


def test_the_signers_file_read_at_run_time_is_the_pinned_one(tmp_path, monkeypatch):
    swapped = tmp_path / "allowed_signers"
    swapped.write_text(Signer(tmp_path, "other").signers_line(namespaces=NS))
    monkeypatch.setattr(mr, "SIGNERS_PATH", swapped)
    with pytest.raises(RuntimeError, match="pin"):
        mr.list_signers()


# ── a signed list in a tree ─────────────────────────────────────────────────

def test_a_signed_pinned_list_passes_and_loads(tree):
    payload = allowlist_bytes(serial=1)
    tree.ship("menu_allowlist.json", payload, tree.owner.sign(payload))
    devices = device_bytes(serial=4)
    tree.ship("device_list.json", devices, tree.owner.sign(devices))
    assert tree.errors() == []
    signers = (tree.lists / "allowed_signers").read_bytes()
    got = mr.shipped_list(mr.KIND_MENU_ALLOWLIST, directory=tree.lists, signers=signers)
    assert got.serial == 1 and got.signer is not None
    assert mr.shipped_list(mr.KIND_DEVICE_LIST, directory=tree.lists, signers=signers).serial == 4


def test_a_shipped_list_without_a_signature_fails(tree):
    tree.ship("menu_allowlist.json", allowlist_bytes(), None)
    one_error(tree, "menu_allowlist.json.sig")
    with pytest.raises(FileNotFoundError):
        mr.shipped_list(mr.KIND_MENU_ALLOWLIST, directory=tree.lists,
                        signers=(tree.lists / "allowed_signers").read_bytes())


def test_a_signature_without_its_list_fails(tree):
    payload = allowlist_bytes()
    (tree.lists / "menu_allowlist.json.sig").write_bytes(tree.owner.sign(payload))
    one_error(tree, "menu_allowlist.json")


def test_a_shipped_list_edited_after_signing_fails(tree):
    payload = allowlist_bytes()
    edited = allowlist_bytes(labs=("deepseek", "openai", "added-later"))
    tree.ship("menu_allowlist.json", edited, tree.owner.sign(payload))
    one_error(tree, "bad_signature")


def test_a_shipped_list_signed_by_an_unlisted_key_fails(tree, tmp_path):
    payload = allowlist_bytes()
    tree.ship("menu_allowlist.json", payload, Signer(tmp_path, "stranger").sign(payload))
    one_error(tree, "signer_not_allowed")


def test_a_shipped_list_signed_as_a_git_object_fails(tree):
    payload = allowlist_bytes()
    tree.ship("menu_allowlist.json", payload, tree.owner.sign(payload, namespace="git"))
    one_error(tree, "wrong_namespace")


def test_a_shipped_list_must_be_pinned(tree):
    payload = allowlist_bytes()
    tree.ship("menu_allowlist.json", payload, tree.owner.sign(payload), pin=False)
    one_error(tree, "pin")
    tree.pins["menu_allowlist.json"] = "0" * 64
    one_error(tree, "pin")


def test_the_draft_shipped_as_the_list_fails_even_when_signed(tree):
    draft = allowlist_bytes(serial=mr.DRAFT_SERIAL)
    tree.ship("menu_allowlist.json", draft, tree.owner.sign(draft))
    one_error(tree, "draft")


def test_the_wrong_kind_of_list_under_a_name_fails(tree):
    devices = device_bytes()
    tree.ship("menu_allowlist.json", devices, tree.owner.sign(devices))
    one_error(tree, "wrong_kind")


def test_a_file_the_package_does_not_read_fails(tree):
    (tree.lists / "menu_allowlist.draft.json").write_bytes(allowlist_bytes(serial=0))
    one_error(tree, "menu_allowlist.draft.json")


# ── the draft ───────────────────────────────────────────────────────────────

def test_a_draft_beside_no_signed_list_passes(tree):
    tree.draft(allowlist_bytes(serial=mr.DRAFT_SERIAL))
    assert tree.errors() == []


def test_a_draft_with_a_real_serial_fails(tree):
    tree.draft(allowlist_bytes(serial=1))
    one_error(tree, "serial 0")


def test_a_draft_with_a_signature_fails(tree):
    draft = allowlist_bytes(serial=mr.DRAFT_SERIAL)
    tree.draft(draft)
    (tree.root / "data" / "menu_allowlist.draft.json.sig").write_bytes(tree.owner.sign(draft))
    one_error(tree, "draft")


def test_a_draft_left_beside_the_signed_list_fails(tree):
    payload = allowlist_bytes(serial=1)
    tree.ship("menu_allowlist.json", payload, tree.owner.sign(payload))
    tree.draft(allowlist_bytes(serial=mr.DRAFT_SERIAL))
    one_error(tree, "remove")


def test_a_draft_that_is_not_a_list_fails(tree):
    tree.draft(b'{"labs": ["openai"]}\n')
    one_error(tree, "menu_allowlist.draft.json")


# ── the signers ─────────────────────────────────────────────────────────────

def test_the_packaged_signers_must_equal_the_release_signers(tree):
    text = tree.owner.signers_line(namespaces=f"git,{NS}")
    tree.set_signers(text, copy=text + Signer(tree.root, "extra").signers_line(namespaces=NS))
    one_error(tree, "differ")


def test_the_signers_must_match_their_pin(tree):
    tree.signers_pin = "0" * 64
    one_error(tree, "pin")


def test_a_missing_signers_copy_fails(tree):
    (tree.lists / "allowed_signers").unlink()
    one_error(tree, "missing")


def test_signers_that_list_no_key_for_lists_fail(tree):
    tree.set_signers(tree.owner.signers_line(namespaces="git"))
    one_error(tree, "no key")


def test_a_signers_line_with_more_than_one_reading_fails(tree):
    """A line OpenSSH would read and this module would not (or the reverse)
    makes the trust root depend on who reads it."""
    good = tree.owner.signers_line(namespaces=f"git,{NS}")
    for line in (tree.owner.signers_line(namespaces=None),
                 tree.owner.signers_line(namespaces="*"),
                 good.replace(" ", "\t", 1)):
        tree.set_signers(good + line)
        one_error(tree, "line 2")


# ── nothing at run time reads an unsigned list ──────────────────────────────

def test_run_time_code_that_reads_an_unsigned_list_fails(tree):
    (tree.root / "neurons" / "miner.py").write_text(
        "from fugal_subnet import menu_rules\n"
        "allow = menu_rules.parse_unsigned_list(b'', 'k')\n")
    one_error(tree, "parse_unsigned_list")
    (tree.root / "neurons" / "miner.py").write_text("x = 1\n")
    (tree.root / "fugal_subnet" / "menu_rules.py").write_text(
        "def parse_unsigned_list(payload, kind):\n    return None\n")
    assert tree.errors() == []
