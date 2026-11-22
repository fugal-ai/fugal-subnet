"""Signed lists (docs/decisions/LIVE_MENU_V4.md, sections 6 and 13.4).

The lab and host allowlist, and the device list, are data signed by a release
key under a namespace of their own. Verification is offline and has one
verdict per input. Every key here is a throwaway generated for the test run,
written with its allowed-signers file to a temporary directory; no test reads
a key from the machine it runs on.

The cases marked A11 are section 14's: an old list replayed, a signature that
does not verify, and a signature the release keys did not make for lists.
"""
import base64
import hashlib
import json
import shutil
import subprocess

import pytest

from fugal_subnet import menu_rules as mr
from tests.signed_list_fixtures import (
    NS,
    Signer,
    _string,
    allowlist_bytes,
    armor,
    device_bytes,
)

SSH_KEYGEN = shutil.which("ssh-keygen")


@pytest.fixture
def owner(tmp_path):
    return Signer(tmp_path, "throwaway_release")


@pytest.fixture
def stranger(tmp_path):
    return Signer(tmp_path, "throwaway_stranger")


@pytest.fixture
def signers(owner, tmp_path):
    """A temporary allowed-signers file: the throwaway owner key, for git tags and lists."""
    path = tmp_path / "allowed_signers"
    path.write_text(owner.signers_line(namespaces=f"git,{NS}"))
    return path.read_bytes()


def verify(payload, signature, signers, kind=mr.KIND_MENU_ALLOWLIST, held_serial=0):
    return mr.verify_signed_list(payload, signature, kind=kind, held_serial=held_serial,
                                 signers=signers)


def refused(payload, signature, signers, **kw) -> str:
    with pytest.raises(mr.ListRefused) as caught:
        verify(payload, signature, signers, **kw)
    assert caught.value.refusal == mr.Refusal(caught.value.code, caught.value.detail)
    return caught.value.code


# ── a signed list is accepted, and says what it is ──────────────────────────

def test_a_list_signed_by_a_listed_key_under_the_list_namespace_is_accepted(owner, signers):
    payload = allowlist_bytes(serial=3)
    got = verify(payload, owner.sign(payload), signers, held_serial=2)
    assert isinstance(got, mr.MenuAllowlist)
    assert got.serial == 3
    assert got.labs == {"deepseek", "openai"}
    assert got.hosts == {"azure", "baidu", "openai"}
    assert got.first_party == (("openai", frozenset({"azure"})),)
    assert got.first_party_of("openai") == {"azure"} and got.first_party_of("deepseek") == set()
    assert got.sha256 == hashlib.sha256(payload).hexdigest()
    fingerprint = base64.b64encode(hashlib.sha256(
        _string(b"ssh-ed25519") + _string(owner.raw)).digest()).decode().rstrip("=")
    assert got.signer == f"SHA256:{fingerprint}"


def test_the_device_list_uses_the_same_mechanism(owner, signers):
    payload = device_bytes(serial=7)
    got = verify(payload, owner.sign(payload), signers, kind=mr.KIND_DEVICE_LIST, held_serial=6)
    assert isinstance(got, mr.DeviceList)
    assert got.serial == 7 and got.devices == {"ab" * 32, "cd" * 32}
    assert got.signer is not None


def test_the_verdict_is_the_same_every_time(owner, signers):
    payload = allowlist_bytes()
    signature = owner.sign(payload)
    assert verify(payload, signature, signers) == verify(payload, signature, signers)
    assert owner.sign(payload) == signature


def test_the_namespace_is_the_project_s_own_and_not_git_s():
    assert NS == "fugal-list@fugalai.com"
    assert NS not in ("git", "file", "email")


# ── A11: replay of an old list ──────────────────────────────────────────────

@pytest.mark.parametrize("serial,held", [(2, 2), (1, 2), (1, 10**6), (5, 5)])
def test_a11_a_list_with_a_lower_or_equal_serial_is_refused(owner, signers, serial, held):
    payload = allowlist_bytes(serial=serial)
    assert refused(payload, owner.sign(payload), signers, held_serial=held) == "rollback"


def test_a11_an_older_list_cannot_restore_a_lab_a_newer_list_removed(owner, signers):
    old = allowlist_bytes(serial=4, labs=("deepseek", "openai", "removed-lab"))
    new = allowlist_bytes(serial=5)
    held = verify(new, owner.sign(new), signers, held_serial=4)
    assert "removed-lab" not in held.labs
    assert refused(old, owner.sign(old), signers, held_serial=held.serial) == "rollback"


def test_a11_an_equal_serial_with_other_content_is_refused(owner, signers):
    other = allowlist_bytes(serial=5, labs=("deepseek", "openai", "extra"))
    assert refused(other, owner.sign(other), signers, held_serial=5) == "rollback"


def test_a11_the_serial_is_inside_the_signed_bytes(owner, signers):
    """Raising the serial of a signed list breaks its signature."""
    payload = allowlist_bytes(serial=1)
    bumped = allowlist_bytes(serial=9)
    assert refused(bumped, owner.sign(payload), signers, held_serial=5) == "bad_signature"


# ── A11: signatures that must not verify ────────────────────────────────────

def test_a11_a_changed_list_does_not_verify_under_the_old_signature(owner, signers):
    payload = allowlist_bytes()
    forged = allowlist_bytes(labs=("deepseek", "openai", "pool-fitted"))
    assert refused(forged, owner.sign(payload), signers) == "bad_signature"


def test_a11_a_flipped_signature_bit_is_refused(owner, signers):
    payload = allowlist_bytes()
    blob = bytearray(owner.blob(payload))
    blob[-1] ^= 0x01
    assert refused(payload, armor(bytes(blob)), signers) == "bad_signature"


def test_a11_a_key_that_is_not_an_allowed_signer_is_refused(stranger, signers):
    payload = allowlist_bytes()
    assert refused(payload, stranger.sign(payload), signers) == "signer_not_allowed"


def test_a11_a_signature_claiming_the_owner_s_key_but_made_by_another_is_refused(
        owner, stranger, signers):
    payload = allowlist_bytes()
    blob = stranger.blob(payload).replace(stranger.raw, owner.raw)
    assert refused(payload, armor(blob), signers) == "bad_signature"


def test_a11_a_signature_under_another_namespace_is_refused(owner, signers):
    """The same key signs git tags. A signature it made as `git` (or anything
    else) is not a signature over a list."""
    payload = allowlist_bytes()
    for namespace in ("git", "file", "fugal-list", NS + "x", ""):
        assert refused(payload, owner.sign(payload, namespace=namespace), signers) == \
            "wrong_namespace", namespace


def test_a11_a_key_listed_for_git_only_cannot_sign_a_list(owner, stranger, tmp_path):
    payload = allowlist_bytes()
    git_only = owner.signers_line(namespaces="git").encode()
    assert refused(payload, owner.sign(payload), git_only) == "signer_not_allowed"
    both = (owner.signers_line(namespaces="git") + stranger.signers_line(namespaces=NS)).encode()
    assert refused(payload, owner.sign(payload), both) == "signer_not_allowed"
    assert verify(payload, stranger.sign(payload), both).serial == 1


@pytest.mark.parametrize("line", [
    "{principal} {key}\n",                                 # no namespaces option
    '{principal} namespaces="*" {key}\n',                  # a pattern, not a name
    '{principal} namespaces="fugal-*" {key}\n',
    '{principal} namespaces="!git" {key}\n',
    '{principal} cert-authority,namespaces="{ns}" {key}\n',
    '{principal} namespaces="{ns}",valid-before="20990101" {key}\n',
    '{principal}\tnamespaces="{ns}"\t{key}\n',
    '# {principal} namespaces="{ns}" {key}\n',
    '{principal} namespaces="{ns}" {key}\r\n',
    "",
])
def test_a11_a_signers_line_that_does_not_name_the_namespace_plainly_grants_nothing(
        owner, line):
    payload = allowlist_bytes()
    text = line.format(principal="release@example.test", ns=NS, key=owner.public_line)
    assert refused(payload, owner.sign(payload), text.encode()) == "signer_not_allowed"
    assert mr.permitted_signers(text.encode()) == {}


def test_permitted_signers_reads_only_the_keys_listed_for_lists(owner, stranger):
    text = (owner.signers_line(namespaces=f"git,{NS}", principal="a@example.test")
            + "\n# a comment\n" + stranger.signers_line(namespaces="git")).encode()
    assert mr.permitted_signers(text) == {owner.raw: "a@example.test"}


def test_a11_a_device_list_cannot_be_presented_as_the_menu_allowlist(owner, signers):
    """Both lists are signed under one namespace; the kind is inside the signed bytes."""
    devices = device_bytes()
    allow = allowlist_bytes()
    assert refused(devices, owner.sign(devices), signers, kind=mr.KIND_MENU_ALLOWLIST) == \
        "wrong_kind"
    assert refused(allow, owner.sign(allow), signers, kind=mr.KIND_DEVICE_LIST) == "wrong_kind"


def test_a_draft_is_refused_even_when_signed(owner, signers):
    """Serial 0 marks a draft: no signature makes it a list."""
    draft = allowlist_bytes(serial=mr.DRAFT_SERIAL)
    assert mr.DRAFT_SERIAL == 0
    assert refused(draft, owner.sign(draft), signers) == "draft"
    assert mr.parse_unsigned_list(draft, mr.KIND_MENU_ALLOWLIST).signer is None


# ── malformed signatures ────────────────────────────────────────────────────

@pytest.mark.parametrize("signature", [
    b"", b"\n", b"not a signature", b"-----BEGIN SSH SIGNATURE-----\n",
    b"-----BEGIN SSH SIGNATURE-----\n-----END SSH SIGNATURE-----\n",
    b"-----BEGIN SSH SIGNATURE-----\n!!!!\n-----END SSH SIGNATURE-----\n",
    b"-----BEGIN SSH SIGNATURE-----\nU1NIU0lH\n-----END SSH SIGNATURE-----\n",
    b"-----BEGIN PGP SIGNATURE-----\nU1NIU0lH\n-----END PGP SIGNATURE-----\n",
    "-----BEGIN SSH SIGNATURE-----\nU1NIU0lHé\n-----END SSH SIGNATURE-----\n".encode(),
])
def test_something_that_is_not_an_ssh_signature_is_refused(signers, signature):
    assert refused(allowlist_bytes(), signature, signers) == "signature_malformed"


def test_text_around_the_signature_is_refused(owner, signers):
    payload = allowlist_bytes()
    good = owner.sign(payload)
    assert verify(payload, good, signers).serial == 1
    for wrapped in (good + b"trailing text\n", b"leading text\n" + good, good + good,
                    good + b"\n", b" " + good, good.replace(b"\n", b"\r\n")):
        assert refused(payload, wrapped, signers) == "signature_malformed", wrapped
    assert verify(payload, good.rstrip(b"\n"), signers).serial == 1


def test_a_signature_blob_with_anything_unexpected_in_it_is_refused(owner, signers):
    payload = allowlist_bytes()
    cases = {
        "trailing bytes": owner.blob(payload, tail=b"\x00"),
        "version 2": owner.blob(payload, version=2),
        "reserved field used": owner.blob(payload, reserved=b"x"),
        "another hash": owner.blob(payload, hash_name="sha256"),
        "truncated": owner.blob(payload)[:-10],
        "wrong magic": b"SSHSIX" + owner.blob(payload)[6:],
        "a length past the end": owner.blob(payload)[:10] + b"\xff\xff\xff\xff",
    }
    for name, blob in cases.items():
        assert refused(payload, armor(blob), signers) == "signature_malformed", name
    b64 = base64.b64encode(owner.blob(payload)).decode()
    one_line = f"-----BEGIN SSH SIGNATURE-----\n{b64}\n-----END SSH SIGNATURE-----\n".encode()
    assert refused(payload, one_line, signers) == "signature_malformed"  # a line over 76


def test_a_key_type_other_than_ed25519_is_refused(owner, signers):
    payload = allowlist_bytes()
    blob = owner.blob(payload).replace(b"ssh-ed25519", b"ssh-ed25518")
    assert refused(payload, armor(blob), signers) == "signature_malformed"


# ── oversized input ─────────────────────────────────────────────────────────

def test_oversized_input_is_refused_before_anything_is_parsed(owner, signers):
    payload = allowlist_bytes()
    signature = owner.sign(payload)
    assert refused(b"x" * (mr.MAX_LIST_BYTES + 1), signature, signers) == "oversized"
    assert refused(payload, b"x" * (mr.MAX_SIGNATURE_BYTES + 1), signers) == "oversized"
    assert refused(payload, signature, b"#" * (mr.MAX_SIGNERS_BYTES + 1)) == "oversized"
    big = allowlist_bytes(labs=tuple(sorted(f"lab-{i:05d}" for i in range(mr.MAX_LIST_ITEMS + 1))))
    assert len(big) <= mr.MAX_LIST_BYTES
    assert refused(big, owner.sign(big), signers) == "oversized"


def test_a_list_at_the_size_limit_is_still_read(owner, signers):
    devices = tuple(sorted(hashlib.sha256(str(i).encode()).hexdigest() for i in range(3000)))
    payload = device_bytes(devices=devices)
    assert len(payload) <= mr.MAX_LIST_BYTES
    assert len(verify(payload, owner.sign(payload), signers, kind=mr.KIND_DEVICE_LIST).devices) == 3000


# ── malformed and duplicated fields, signed correctly ───────────────────────

def canonical(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def good_object(**over):
    obj = {"format": 2, "kind": mr.KIND_MENU_ALLOWLIST, "serial": 1, "labs": ["deepseek", "openai"],
           "hosts": ["azure", "baidu", "openai"], "first_party": {"openai": ["azure"]}}
    obj.update(over)
    return {k: v for k, v in obj.items() if v is not ...}


def test_the_canonical_bytes_are_one_sorted_item_per_line():
    assert allowlist_bytes() == canonical(good_object())
    assert allowlist_bytes().endswith(b"\n}\n")


@pytest.mark.parametrize("over", [
    {"serial": -1}, {"serial": "1"}, {"serial": 1.0}, {"serial": True}, {"serial": None},
    {"serial": 2**53}, {"serial": ...},
    {"format": 1}, {"format": 3}, {"format": "2"}, {"format": ...},
    {"labs": ...}, {"hosts": ...}, {"first_party": ...},
    # list format 1's lab-agnostic array, in place of the per-lab object
    {"first_party": ..., "first_party_hosts": ["azure"]},
    {"first_party": ["azure"]}, {"first_party": {"openai": "azure"}},
    {"first_party": {"openai": []}}, {"first_party": {"OpenAI": ["azure"]}},
    {"first_party": {"openai": ["azure/fast"]}},
    {"labs": "openai"}, {"labs": {"openai": True}}, {"labs": [1, 2]}, {"labs": [["openai"]]},
    {"labs": []}, {"hosts": []},
    {"labs": ["OpenAI"]}, {"labs": ["open ai"]}, {"labs": ["openai/gpt"]}, {"labs": [""]},
    {"labs": ["x" * 65]},
    {"hosts": ["azure", "baidu", "openai/fast"]},          # a tier is never a listed host
    {"hosts": ["azure", "baidu/fp8", "openai"]},           # nor is a quantization
    {"hosts": ["Azure", "baidu", "openai"]},
    {"hosts": ["a/b/c/d", "azure", "openai"]},
    {"first_party": {"openai": ["not-a-listed-host"]}},
    {"first_party": {"not-a-listed-lab": ["azure"]}},
    {"extra": 1}, {"comment": "x"}, {"devices": []},
])
def test_a_malformed_field_is_refused(owner, signers, over):
    payload = canonical(good_object(**over))
    assert refused(payload, owner.sign(payload), signers) == "malformed"


@pytest.mark.parametrize("payload", [
    b"", b"null", b"[]", b"7", b'"list"', b"{", b"\xff\xfe", b"{}",
    '{"kind": "fugal-menu-allowlist", "labs": ["é"]}'.encode(),
    b'{"kind": "fugal-menu-allowlist", "serial": NaN}',
    b'{"kind": "fugal-menu-allowlist", "serial": 1e3}',
    b"[" * 100000,
    b'{"kind": "fugal-menu-allowlist", "serial": ' + b"9" * 5000 + b"}",
])
def test_bytes_that_are_not_a_list_are_refused(owner, signers, payload):
    assert refused(payload, owner.sign(payload), signers) in ("malformed", "wrong_kind")


def test_a_repeated_key_is_refused(owner, signers):
    """Two parsers could each keep a different one of a repeated key."""
    text = allowlist_bytes().decode().replace(
        ' "serial": 1', ' "serial": 1,\n "serial": 2')
    assert refused(text.encode(), owner.sign(text.encode()), signers) == "duplicate"
    text = allowlist_bytes().decode().replace(
        ' "labs": [', ' "labs": [\n  "evil"\n ],\n "labs": [')
    assert refused(text.encode(), owner.sign(text.encode()), signers) == "duplicate"


@pytest.mark.parametrize("field,items", [
    ("labs", ["deepseek", "openai", "openai"]),
    ("hosts", ["azure", "azure", "baidu", "openai"]),
])
def test_a_repeated_item_is_refused(owner, signers, field, items):
    payload = canonical(good_object(**{field: items}))
    assert refused(payload, owner.sign(payload), signers) == "duplicate"
    devices = canonical({"format": 1, "kind": mr.KIND_DEVICE_LIST, "serial": 1,
                         "devices": ["ab" * 32, "ab" * 32]})
    assert refused(devices, owner.sign(devices), signers, kind=mr.KIND_DEVICE_LIST) == "duplicate"


@pytest.mark.parametrize("device", ["ab" * 31, "AB" * 32, "zz" * 32, "ab" * 33, 7])
def test_a_device_id_is_64_lowercase_hex(owner, signers, device):
    payload = canonical({"format": 1, "kind": mr.KIND_DEVICE_LIST, "serial": 1,
                         "devices": [device]})
    assert refused(payload, owner.sign(payload), signers, kind=mr.KIND_DEVICE_LIST) == "malformed"


def test_bytes_other_than_the_canonical_ones_are_refused(owner, signers):
    """One content has one byte string, so its sha256 names it."""
    good = allowlist_bytes()
    obj = json.loads(good)
    variants = {
        "compact": json.dumps(obj, sort_keys=True, separators=(",", ":")).encode(),
        "no final newline": good.rstrip(b"\n"),
        "two final newlines": good + b"\n",
        "keys unsorted": (json.dumps(dict(reversed(sorted(obj.items()))), indent=1) + "\n").encode(),
        "items unsorted": canonical(good_object(labs=["openai", "deepseek"])),
        "crlf": good.replace(b"\n", b"\r\n"),
        "escaped letter": good.replace(b'"openai"', b'"\\u006fpenai"', 1),
        "a leading byte-order mark": b"\xef\xbb\xbf" + good,
    }
    for name, payload in variants.items():
        assert refused(payload, owner.sign(payload), signers) in ("not_canonical", "malformed"), name
    assert refused(variants["compact"], owner.sign(variants["compact"]), signers) == "not_canonical"
    assert refused(variants["items unsorted"], owner.sign(variants["items unsorted"]),
                   signers) == "not_canonical"


def test_the_arguments_are_checked(owner, signers):
    payload = allowlist_bytes()
    assert refused(payload.decode(), owner.sign(payload), signers) == "malformed"
    assert refused(payload, owner.sign(payload).decode(), signers) == "signature_malformed"
    with pytest.raises(ValueError):
        mr.verify_signed_list(payload, owner.sign(payload), kind="other", held_serial=0,
                              signers=signers)
    for held in (-1, None, "0", True, 1.0):
        with pytest.raises(ValueError):
            mr.verify_signed_list(payload, owner.sign(payload), kind=mr.KIND_MENU_ALLOWLIST,
                                  held_serial=held, signers=signers)


# ── agreement with OpenSSH on real files ────────────────────────────────────

needs_ssh_keygen = pytest.mark.skipif(SSH_KEYGEN is None, reason="ssh-keygen is not installed")


def _ssh_sign(key_path, namespace, file_path) -> bytes:
    out = file_path.parent / (file_path.name + ".sig")
    out.unlink(missing_ok=True)
    subprocess.run([SSH_KEYGEN, "-q", "-Y", "sign", "-f", str(key_path), "-n", namespace,
                    str(file_path)], check=True, capture_output=True, timeout=30,
                   stdin=subprocess.DEVNULL)
    return out.read_bytes()


def _ssh_verify(signers_path, namespace, payload: bytes, signature_path) -> bool:
    return subprocess.run(
        [SSH_KEYGEN, "-Y", "verify", "-f", str(signers_path), "-I", "release@example.test",
         "-n", namespace, "-s", str(signature_path)],
        input=payload, capture_output=True, timeout=30).returncode == 0


@needs_ssh_keygen
def test_a_list_signed_with_ssh_keygen_is_accepted(owner, signers, tmp_path):
    """The command the owner runs (deploy/keys/README.md) produces a signature
    this module accepts, and OpenSSH accepts the same pair."""
    path = tmp_path / "menu_allowlist.json"
    path.write_bytes(allowlist_bytes(serial=2))
    signature = _ssh_sign(owner.path, NS, path)
    assert verify(path.read_bytes(), signature, signers, held_serial=1).serial == 2
    assert signature == owner.sign(path.read_bytes())
    listed = subprocess.run([SSH_KEYGEN, "-l", "-f", f"{owner.path}.pub"], check=True,
                            capture_output=True, text=True, timeout=30).stdout.split()
    assert verify(path.read_bytes(), signature, signers).signer == listed[1]
    assert _ssh_verify(tmp_path / "allowed_signers", NS, path.read_bytes(), path.parent / (path.name + ".sig"))


@needs_ssh_keygen
def test_ssh_keygen_and_this_module_refuse_the_same_signatures(owner, stranger, signers, tmp_path):
    payload = allowlist_bytes()
    path = tmp_path / "list.json"
    path.write_bytes(payload)
    cases = {
        "signed as git": _ssh_sign(owner.path, "git", path),
        "signed by a stranger": _ssh_sign(stranger.path, NS, path),
        "signed over other bytes": owner.sign(allowlist_bytes(serial=2)),
    }
    for name, signature in cases.items():
        sig_path = tmp_path / "case.sig"
        sig_path.write_bytes(signature)
        assert not _ssh_verify(tmp_path / "allowed_signers", NS, payload, sig_path), name
        with pytest.raises(mr.ListRefused):
            verify(payload, signature, signers)


@needs_ssh_keygen
def test_a_git_only_signers_file_refuses_a_list_in_both_implementations(owner, tmp_path):
    payload = allowlist_bytes()
    path = tmp_path / "list.json"
    path.write_bytes(payload)
    signature = _ssh_sign(owner.path, NS, path)
    git_only = tmp_path / "git_only_signers"
    git_only.write_text(owner.signers_line(namespaces="git"))
    assert not _ssh_verify(git_only, NS, payload, tmp_path / "list.json.sig")
    assert refused(payload, signature, git_only.read_bytes()) == "signer_not_allowed"


# ── the owner's tool: scripts/signed_list.py ────────────────────────────────

def _tool():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "signed_list.py"
    spec = importlib.util.spec_from_file_location("signed_list_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prepare_gives_a_reviewed_draft_a_serial_and_verify_checks_the_signed_file(
        owner, signers, tmp_path, capsys):
    tool = _tool()
    draft = tmp_path / "draft.json"
    draft.write_bytes(allowlist_bytes(serial=mr.DRAFT_SERIAL))
    out = tmp_path / "menu_allowlist.json"
    assert tool.main(["prepare", "--serial", "3", "--draft", str(draft), "--out", str(out)]) == 0
    assert out.read_bytes() == allowlist_bytes(serial=3)
    assert "NOT signed" in capsys.readouterr().out
    signers_path = tmp_path / "allowed_signers"
    verify_args = ["verify", str(out), "--signers", str(signers_path)]

    (tmp_path / "menu_allowlist.json.sig").write_bytes(owner.sign(out.read_bytes()))
    assert tool.main(verify_args) == 0
    assert "serial 3" in capsys.readouterr().out
    assert tool.main(verify_args + ["--held-serial", "3"]) == 1
    assert "rollback" in capsys.readouterr().err
    (tmp_path / "menu_allowlist.json.sig").write_bytes(owner.sign(out.read_bytes(), namespace="git"))
    assert tool.main(verify_args) == 1
    assert "wrong_namespace" in capsys.readouterr().err


def test_prepare_refuses_serial_zero_and_a_draft_that_is_not_a_list(tmp_path):
    tool = _tool()
    draft = tmp_path / "draft.json"
    draft.write_bytes(allowlist_bytes(serial=mr.DRAFT_SERIAL))
    with pytest.raises(SystemExit, match="at least 1"):
        tool.main(["prepare", "--serial", "0", "--draft", str(draft), "--out", str(tmp_path / "o")])
    draft.write_bytes(b"{}\n")
    with pytest.raises(SystemExit, match="wrong_kind"):
        tool.main(["prepare", "--serial", "1", "--draft", str(draft), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()


def _derivation_files(tool, tmp_path):
    """A one-row menu file, and the plain derivation it implies."""
    menu = tmp_path / "menu.json"
    row = {"id": "openai/gpt-x@low", "model": "openai/gpt-x",
           "provider": {"order": ["openai", "azure"], "allow_fallbacks": False}}
    menu.write_text(json.dumps([row]))
    return menu, tool.draft_from_menu([row])


def test_draft_check_accepts_a_reviewed_draft_and_draft_will_not_overwrite_it(tmp_path, capsys):
    tool = _tool()
    menu, built = _derivation_files(tool, tmp_path)
    draft = tmp_path / "draft.json"
    args = ["draft", "--menu", str(menu), "--out", str(draft)]
    # no draft yet: --check fails, plain `draft` writes the derivation
    assert tool.main(args + ["--check"]) == 1
    assert tool.main(args) == 0
    assert draft.read_bytes() == built
    assert tool.main(args + ["--check"]) == 0
    assert "is the draft" in capsys.readouterr().out
    # a plain re-run over an identical draft is fine
    assert tool.main(args) == 0
    # the owner's review adds two labs, one host and one first-party mark
    base = mr.parse_unsigned_list(built, mr.KIND_MENU_ALLOWLIST)
    reviewed = mr.canonical_list_bytes(
        mr.KIND_MENU_ALLOWLIST, mr.DRAFT_SERIAL, labs=base.labs | {"a-lab", "b-lab"},
        hosts=base.hosts | {"a-host"},
        first_party={**base.first_party_dict(), "a-lab": ["a-host"]})
    draft.write_bytes(reviewed)
    capsys.readouterr()
    assert tool.main(args + ["--check"]) == 0
    out = capsys.readouterr().out
    assert "reviewed draft" in out and "contains the derivation plus 2 labs and 1 hosts" in out
    # `draft` refuses to replace it, and says why
    assert tool.main(args) == 1
    assert "Not overwritten" in capsys.readouterr().err
    assert draft.read_bytes() == reviewed
    # only the explicit flag replaces it
    assert tool.main(args + ["--overwrite"]) == 0
    assert draft.read_bytes() == built


def test_draft_check_fails_when_the_draft_dropped_something_the_derivation_has(tmp_path, capsys):
    tool = _tool()
    menu, built = _derivation_files(tool, tmp_path)
    draft = tmp_path / "draft.json"
    base = mr.parse_unsigned_list(built, mr.KIND_MENU_ALLOWLIST)
    draft.write_bytes(mr.canonical_list_bytes(
        mr.KIND_MENU_ALLOWLIST, mr.DRAFT_SERIAL, labs={"other"}, hosts=base.hosts,
        first_party={}))
    args = ["draft", "--menu", str(menu), "--out", str(draft)]
    assert tool.main(args + ["--check"]) == 1
    assert "does not contain it" in capsys.readouterr().err
    draft.write_bytes(b"not json")                      # an unreadable draft is not a superset
    assert tool.main(args + ["--check"]) == 1
    assert tool.main(args) == 1                         # and is not overwritten either
    assert draft.read_bytes() == b"not json"


def test_the_tool_never_names_a_private_key_path_of_its_own():
    """Signing is the key holder's one command; the tool reads no private key."""
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "signed_list.py").read_text()
    assert "private_bytes" not in source and "load_ssh_private_key" not in source
    assert "subprocess" not in source
