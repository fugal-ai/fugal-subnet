"""scripts/release.py: every step of a release but signing and pushing.

Real git repositories and throwaway SSH keys made here (ssh-keygen signs the
tags and the schedules, as the owner would), a fake `dstack-cloud prepare`,
and a temporary checkout for anything the tool writes. The chain head is
always passed (--head-block), so nothing here reads a chain.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("release_tool", ROOT / "scripts" / "release.py")
rt = importlib.util.module_from_spec(_spec)
sys.modules["release_tool"] = rt
_spec.loader.exec_module(rt)

from fugal_subnet import config  # noqa: E402
from fugal_subnet import protocol_schedule as ps  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None or shutil.which("git") is None,
                                reason="needs OpenSSH and git")
BPE = 360
HEAD = 9_599_760                        # the boundary of epoch 26666
DIGEST = "ghcr.io/fugal-ai/fugal-miner@sha256:" + "ab" * 32
BASE = "f4" * 32


def _git_env(key: Path | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    if key is not None:
        env.update(GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="gpg.format", GIT_CONFIG_VALUE_0="ssh",
                   GIT_CONFIG_KEY_1="user.signingkey", GIT_CONFIG_VALUE_1=f"{key}.pub")
    return env


def git(repo: Path, *args: str, key: Path | None = None, input: bytes | None = None) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, input=input,
                          env=_git_env(key), check=True).stdout.decode().strip()


def keygen(path: Path) -> Path:
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "t", "-f", str(path)],
                   check=True)
    return path


def signers_line(key: Path, namespaces="git,fugal-list@fugalai.com") -> str:
    return (f'release@example.test namespaces="{namespaces}" '
            f'{key.with_suffix(".pub").read_text().strip()}\n')


@pytest.fixture
def keys(tmp_path):
    good, other = keygen(tmp_path / "release_key"), keygen(tmp_path / "other_key")
    signers = tmp_path / "allowed_signers"
    signers.write_text(signers_line(good))
    return good, other, signers


def run(capsys, *argv) -> tuple[int, str]:
    try:
        code = rt.main(list(argv))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        print(e.code if not isinstance(e.code, int) else "")
    return code, capsys.readouterr().out


def checkout(tmp_path) -> Path:
    """A temporary checkout: the composes, two freezes and an empty schedule dir."""
    root = tmp_path / "repo"
    (root / "deploy" / "dstack").mkdir(parents=True)
    (root / "fugal_subnet" / "signed_schedule").mkdir(parents=True)
    for name in ("docker-compose.yaml", "docker-compose.test.yaml"):
        shutil.copyfile(ROOT / "deploy" / "dstack" / name, root / "deploy" / "dstack" / name)
    shutil.copyfile(ROOT / "pyproject.toml", root / "pyproject.toml")
    return root


# ── blocks and the lead ──────────────────────────────────────────────────────


def test_check_block_rules():
    p, w = rt.check_block(HEAD + 3600, head=HEAD, after={"v5": 9_585_720})
    assert p == [] and w                                    # 10 epochs: a notice warning
    p, _ = rt.check_block(HEAD + 3600 + 1, head=HEAD)
    assert any("epoch boundary" in x for x in p)
    p, _ = rt.check_block(9_585_720, head=HEAD, after={"v5": 9_585_720})
    assert any("not after" in x for x in p)
    p, _ = rt.check_block(HEAD + 2 * BPE, head=HEAD)        # first seen at +2, lead 2 more
    assert any("SCHEDULE_LEAD_EPOCHS" in x for x in p)
    assert rt.check_block(HEAD + 30 * BPE, head=HEAD) == ([], [])


def test_blocks_is_a_dry_run_and_refuses_what_a_setter_would(capsys):
    before = (ROOT / "fugal_subnet" / "config.py").read_bytes()
    code, out = run(capsys, "--head-block", str(HEAD), "blocks", f"v5={HEAD + 40 * BPE}")
    assert code == 0 and "would run" in out and "set_v5_activation_block.py" in out
    code, out = run(capsys, "--head-block", str(HEAD), "blocks", f"v5={HEAD + 40 * BPE + 7}")
    assert code == 1 and "epoch boundary" in out
    code, out = run(capsys, "--head-block", str(HEAD), "blocks", "v9=1")
    assert code != 0 and "no setter" in out
    assert (ROOT / "fugal_subnet" / "config.py").read_bytes() == before


# ── the schedule: draft, sign (owner), verify, install ───────────────────────


def _sign(key: Path, path: Path) -> None:
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", "fugal-list@fugalai.com",
                    str(path)], check=True, capture_output=True)


def _draft(capsys, root, keys, *more, head=HEAD):
    out_dir = root.parent / "out"
    return run(capsys, "--root", str(root), "--head-block", str(head), "--signers",
               str(keys[2]), "--out-dir", str(out_dir), "schedule", "draft", *more)


def test_schedule_round_trip(capsys, tmp_path, keys):
    root = checkout(tmp_path)
    block = HEAD + 40 * BPE
    code, out = _draft(capsys, root, keys, "--set", f"v5_1={block}")
    assert code == 0 and "would write" in out and "ssh-keygen -Y sign" in out
    assert not (root.parent / "out").exists()                # a dry run writes nothing
    code, out = _draft(capsys, root, keys, "--set", f"v5_1={block}", "--apply")
    path = root.parent / "out" / "schedule-1.json"
    assert code == 0 and path.is_file()
    assert ps.parse_unsigned(path.read_bytes()).as_dict() == {"v5_1": block}
    _sign(keys[0], path)
    code, out = run(capsys, "--root", str(root), "--head-block", str(HEAD), "--signers",
                    str(keys[2]), "schedule", "verify", str(path), "--install")
    assert code == 0, out
    assert "pointer sched:1:" in out and "gsutil" in out and "installed" in out
    shipped = rt.shipped_schedules(root, keys[2].read_bytes())
    assert [s.serial for s, _ in shipped] == [1]
    # The next draft starts from it, with the next serial.
    code, out = _draft(capsys, root, keys, "--set", f"v6={block + 20 * BPE}", "--apply")
    nxt = ps.parse_unsigned((root.parent / "out" / "schedule-2.json").read_bytes())
    assert code == 0 and nxt.serial == 2 and nxt.as_dict() == {"v5_1": block,
                                                               "v6": block + 20 * BPE}


def _install(capsys, root, keys, acts, serial=1):
    path = root.parent / f"s{serial}.json"
    path.write_bytes(ps.canonical_bytes(serial, config.FINNEY_GENESIS_HASH, acts))
    _sign(keys[0], path)
    code, out = run(capsys, "--root", str(root), "--offline", "--signers", str(keys[2]),
                    "schedule", "verify", str(path), "--install")
    assert code == 0, out
    return path


@pytest.mark.parametrize("case", ["older_serial", "too_soon", "backdated", "moves_passed"])
def test_schedule_draft_refusals(capsys, tmp_path, keys, case):
    root = checkout(tmp_path)
    _install(capsys, root, keys, {"v5_1": HEAD + 10 * BPE}, serial=3)
    head = HEAD + 20 * BPE if case == "moves_passed" else HEAD
    more = {"older_serial": ["--serial", "3", "--set", f"v6={HEAD + 50 * BPE}"],
            "too_soon": ["--set", f"v6={HEAD + 3 * BPE}"],
            "backdated": ["--set", f"v6={HEAD - 10 * BPE}"],
            "moves_passed": ["--set", f"v5_1={HEAD + 80 * BPE}"]}[case]
    code, out = _draft(capsys, root, keys, *more, "--apply", head=head)
    assert code != 0, out
    assert not list((root.parent / "out").glob("*.json"))


@pytest.mark.parametrize("case", ["unsigned", "wrong_signer", "other_chain", "older_serial",
                                  "too_soon"])
def test_schedule_verify_refusals(capsys, tmp_path, keys, case):
    root = checkout(tmp_path)
    if case == "older_serial":
        _install(capsys, root, keys, {"v5_1": HEAD + 40 * BPE}, serial=5)
    chain = "0x" + "cd" * 32 if case == "other_chain" else config.FINNEY_GENESIS_HASH
    block = HEAD + 3 * BPE if case == "too_soon" else HEAD + 60 * BPE
    path = tmp_path / "s.json"
    path.write_bytes(ps.canonical_bytes(4, chain, {"v6": block}))
    if case == "unsigned":
        (tmp_path / "s.json.sig").write_bytes(b"")
    else:
        _sign(keys[1] if case == "wrong_signer" else keys[0], path)
    installed = set((root / "fugal_subnet" / "signed_schedule").iterdir())
    code, out = run(capsys, "--root", str(root), "--head-block", str(HEAD), "--signers",
                    str(keys[2]), "schedule", "verify", str(path), "--install")
    assert code != 0, out
    assert set((root / "fugal_subnet" / "signed_schedule").iterdir()) == installed


# ── freeze, entries and the signed image tag ─────────────────────────────────


FAKE_PREPARE = """#!{python}
import json, os, sys
assert sys.argv[1:] == ["prepare"], sys.argv
os.makedirs("shared", exist_ok=True)
compose = open("docker-compose.yaml").read()
with open("shared/app-compose.json", "w") as f:
    json.dump({{"docker_compose_file": compose, "name": os.path.basename(os.getcwd())}}, f,
              indent=2)
"""


def _app_dirs(tmp_path) -> list[str]:
    out = []
    for net in ("finney", "test"):
        d = tmp_path / f"app-{net}"
        d.mkdir()
        (d / "app.json").write_text("{}")
        (d / "prelaunch.sh").write_text("#!/bin/sh\n")
        out += ["--app-dir", f"{net}={d}"]
    return out


def _fake_cli(tmp_path) -> Path:
    cli = tmp_path / "dstack-cloud"
    cli.write_text(FAKE_PREPARE.format(python=sys.executable))
    cli.chmod(0o755)
    return cli


def test_freeze_dry_run_then_apply(capsys, tmp_path):
    root = checkout(tmp_path)
    before = (root / "deploy" / "dstack" / "docker-compose.yaml").read_bytes()
    args = ["--root", str(root), "freeze", "--digest", DIGEST, "--tag", "2026-11-27",
            *_app_dirs(tmp_path), "--cli", str(_fake_cli(tmp_path))]
    code, out = run(capsys, *args)
    assert code == 0 and "Dry run" in out
    assert (root / "deploy" / "dstack" / "docker-compose.yaml").read_bytes() == before
    code, out = run(capsys, *args, "--apply")
    assert code == 0, out
    for net, live in (("finney", "docker-compose.yaml"), ("test", "docker-compose.test.yaml")):
        frozen = root / "deploy" / "dstack" / f"app-compose.frozen-2026-11-27-{net}.json"
        text = (root / "deploy" / "dstack" / live).read_text()
        assert f"image: {DIGEST}" in text
        assert json.loads(frozen.read_text())["docker_compose_file"] == text
    code, out = run(capsys, *args, "--apply")
    assert code != 0 and "exists" in out                     # one freeze per tag


def test_freeze_refuses_a_tag_reference(capsys, tmp_path):
    root = checkout(tmp_path)
    code, out = run(capsys, "--root", str(root), "freeze", "--digest",
                    "ghcr.io/fugal-ai/fugal-miner:latest", "--tag", "2026-11-27",
                    *_app_dirs(tmp_path))
    assert code != 0 and "sha256" in out


class Origin:
    def __init__(self, path: Path, key: Path, version: str = "0.9.0"):
        self.path, self.key = path, key
        path.mkdir()
        git(path, "init", "-q")
        (path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
        git(path, "add", "-A")
        git(path, "commit", "-q", "-m", "release")
        self.head = git(path, "rev-parse", "HEAD")

    def tag(self, name: str, message: str, key: Path | None = None) -> None:
        git(self.path, "tag", "-s", name, "-F", "-", self.head, key=key or self.key,
            input=message.encode())

    def put_tag_object(self, ref: str, raw: bytes) -> None:
        oid = git(self.path, "hash-object", "-t", "tag", "-w", "--literally", "--stdin",
                  input=raw)
        git(self.path, "update-ref", f"refs/tags/{ref}", oid)

    def raw(self, name: str) -> bytes:
        return subprocess.run(["git", "-C", str(self.path), "cat-file", "tag", name],
                              capture_output=True, check=True, env=_git_env()).stdout


def _frozen(root: Path, tag="2026-11-27"):
    """Two frozen composes naming DIGEST, as freeze writes them."""
    for net in ("finney", "test"):
        p = root / "deploy" / "dstack" / f"app-compose.frozen-{tag}-{net}.json"
        p.write_text(json.dumps({"docker_compose_file": f"image: {DIGEST}\n# {net}\n"},
                                indent=2))


def test_entries_writes_what_is_signed(capsys, tmp_path):
    root = checkout(tmp_path)
    _frozen(root)
    out_dir = tmp_path / "out"
    code, out = run(capsys, "--root", str(root), "--out-dir", str(out_dir), "entries",
                    "--digest", DIGEST, "--commit", "a" * 40, "--base", BASE, "--tag",
                    "image-v2026.11.27", "--frozen-tag", "2026-11-27", "--apply")
    assert code == 0, out
    message = (out_dir / "image-v2026.11.27.tag-message.txt").read_text()
    import hashlib
    for net in ("finney", "test"):
        h = hashlib.sha256((root / "deploy" / "dstack" /
                            f"app-compose.frozen-2026-11-27-{net}.json").read_bytes()).hexdigest()
        assert f"approved-entry {net}: {BASE}:{h}" in message
    assert "git -c gpg.format=ssh" in out and "tag -s image-v2026.11.27" in out
    code, out = run(capsys, "--root", str(root), "--out-dir", str(out_dir), "entries",
                    "--digest", DIGEST, "--commit", "a" * 40, "--base", BASE, "--tag",
                    "image-v2026.11.27", "--frozen-tag", "2026-11-99")
    assert code != 0 and "freeze first" in out


def _signed_image_tag(capsys, tmp_path, keys, *, key=None):
    root = checkout(tmp_path)
    _frozen(root)
    out_dir = tmp_path / "out"
    run(capsys, "--root", str(root), "--out-dir", str(out_dir), "entries", "--digest", DIGEST,
        "--commit", "a" * 40, "--base", BASE, "--tag", "image-v2026.11.27", "--frozen-tag",
        "2026-11-27", "--apply")
    origin = Origin(tmp_path / "origin", keys[0])
    origin.tag("image-v2026.11.27",
               (out_dir / "image-v2026.11.27.tag-message.txt").read_text(), key=key)
    return root, origin


def _verify(capsys, root, origin, keys, tag="image-v2026.11.27", *more):
    return run(capsys, "--root", str(root), "--signers", str(keys[2]), "verify-tag", tag,
               "--git-dir", str(origin.path / ".git"), *more)


def test_verify_image_tag_ok_and_matches_the_freeze(capsys, tmp_path, keys):
    root, origin = _signed_image_tag(capsys, tmp_path, keys)
    code, out = _verify(capsys, root, origin, keys, "image-v2026.11.27", "--frozen-tag",
                        "2026-11-27", "--base", BASE)
    assert code == 0 and out.strip().endswith("signed by release@example.test"), out
    (root / "deploy" / "dstack" / "app-compose.frozen-2026-11-27-test.json").write_text("{}")
    code, out = _verify(capsys, root, origin, keys, "image-v2026.11.27", "--frozen-tag",
                        "2026-11-27")
    assert code == 1 and "MISMATCH" in out


def test_verify_image_tag_wrong_signer(capsys, tmp_path, keys):
    root, origin = _signed_image_tag(capsys, tmp_path, keys, key=keys[1])
    code, out = _verify(capsys, root, origin, keys)
    assert code == 1 and "REJECTED" in out


def test_verify_image_tag_trailing_text_and_renamed(capsys, tmp_path, keys):
    root, origin = _signed_image_tag(capsys, tmp_path, keys)
    raw = origin.raw("image-v2026.11.27")
    origin.put_tag_object("image-v2026.11.28", raw)                    # renamed
    code, out = _verify(capsys, root, origin, keys, "image-v2026.11.28")
    assert code == 1 and "renamed" in out
    origin.put_tag_object("image-v2026.11.27",
                          raw + b"approved-entry finney: " + b"0" * 64 + b":" + b"1" * 64 + b"\n")
    code, out = _verify(capsys, root, origin, keys)
    assert code == 1 and "REJECTED" in out


def test_verify_unsigned_and_entryless_tags(capsys, tmp_path, keys):
    root = checkout(tmp_path)
    origin = Origin(tmp_path / "origin", keys[0])
    git(origin.path, "tag", "-a", "image-v2026.11.27", "-m", "unsigned", origin.head)
    code, out = _verify(capsys, root, origin, keys)
    assert code == 1 and "REJECTED" in out
    origin.tag("image-v2026.11.28", "Miner image image-v2026.11.28\n\nno entries\n")
    code, out = _verify(capsys, root, origin, keys, "image-v2026.11.28")
    assert code == 1 and "no approved-entry" in out


def test_validator_tag_prints_and_verify_checks_the_version(capsys, tmp_path, keys):
    origin = Origin(tmp_path / "origin", keys[0], version="0.9.0")
    code, out = run(capsys, "--root", str(origin.path), "validator-tag", "0.9.1")
    assert code != 0 and "pyproject.toml says version '0.9.0'" in out
    code, out = run(capsys, "--root", str(origin.path), "validator-tag", "0.9.0")
    assert code == 0 and "tag -s validator-v0.9.0" in out
    assert git(origin.path, "tag", "-l") == ""                          # nothing signed here
    origin.tag("validator-v0.9.0", "validator v0.9.0\n")
    code, out = _verify(capsys, origin.path, origin, keys, "validator-v0.9.0")
    assert code == 0 and "OK validator-v0.9.0" in out
    origin.tag("validator-v0.9.1", "validator v0.9.1\n")
    code, out = _verify(capsys, origin.path, origin, keys, "validator-v0.9.1")
    assert code == 1 and "version" in out


def test_build_tag_only_prints(capsys, tmp_path, keys):
    origin = Origin(tmp_path / "origin", keys[0])
    code, out = run(capsys, "--root", str(origin.path), "build-tag", "build-v2026.11.27")
    assert code == 0 and "git push origin build-v2026.11.27" in out
    assert git(origin.path, "tag", "-l") == ""
    code, out = run(capsys, "--root", str(origin.path), "build-tag", "image-v2026.11.27")
    assert code != 0


# ── notices, status, plan ────────────────────────────────────────────────────


def test_notices_fit_discord(capsys, tmp_path, keys):
    root = checkout(tmp_path)
    _install(capsys, root, keys, {"v5_1": HEAD + 40 * BPE, "v6": HEAD + 90 * BPE})
    for argv in (["notice", "schedule"],
                 ["notice", "validator", "--version", "0.9.0", "--deadline-block",
                  str(HEAD + 20 * BPE), "--summary", "the schedule fold"],
                 ["notice", "image", "--image-tag", "image-v2026.11.27", "--digest", DIGEST,
                  "--entry", f"{BASE}:{'c' * 64}", "--deadline-block", str(HEAD + 40 * BPE)]):
        code, out = run(capsys, "--root", str(root), "--head-block", str(HEAD), "--signers",
                        str(keys[2]), *argv)
        assert code == 0, out
        assert 0 < len(out) <= rt.DISCORD_LIMIT
    assert "sched:1:" in run(capsys, "--root", str(root), "--head-block", str(HEAD),
                             "--signers", str(keys[2]), "notice", "schedule")[1]


def test_status_and_plan_read_only(capsys):
    code, out = run(capsys, "--offline", "status")
    assert code == 0 and "v5_commit_fix" in out and "schedule_from" in out
    code, out = run(capsys, "plan")
    assert code == 0 and "OWNER signs" in out


def test_the_tool_never_signs_or_pushes():
    """The only processes it starts: git rev-parse, the setter scripts,
    dstack-cloud prepare, and the auto-updater's read-only git."""
    src = (ROOT / "scripts" / "release.py").read_text()
    calls = [line.strip() for line in src.splitlines() if "subprocess.run(" in line]
    assert len(calls) == 3, calls
    assert all("push" not in c and '"tag"' not in c for c in calls)
