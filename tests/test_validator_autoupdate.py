"""The opt-in validator auto-updater (deploy/autoupdate/fugal_autoupdate.py).

Real git repositories and a throwaway SSH key made here; systemd, the journal,
uv and the other user are a fake (`FakeSystem`), so nothing needs root.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fugal_autoupdate", ROOT / "deploy" / "autoupdate" / "fugal_autoupdate.py")
au = importlib.util.module_from_spec(_spec)
sys.modules["fugal_autoupdate"] = au
_spec.loader.exec_module(au)

PRINCIPAL = "release@test.example"
T0 = 1_800_000_000.0


# ------------------------------------------------------------ git fixtures


def _git_env(signing_key: Path | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    if signing_key is not None:
        env.update(GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="gpg.format", GIT_CONFIG_VALUE_0="ssh",
                   GIT_CONFIG_KEY_1="user.signingkey", GIT_CONFIG_VALUE_1=f"{signing_key}.pub")
    return env


def git(repo: Path, *args: str, key: Path | None = None, input: bytes | None = None) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, input=input,
                       env=_git_env(key), check=True)
    return r.stdout.decode().strip()


def keygen(path: Path) -> Path:
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(path)],
                   check=True)
    return path


def signers_line(key: Path, namespaces: str = "git") -> str:
    return f'{PRINCIPAL} namespaces="{namespaces}" {key.with_suffix(".pub").read_text().strip()}\n'


class Origin:
    """A stand-in for the GitHub repository."""

    def __init__(self, path: Path, key: Path):
        self.path, self.key = path, key
        path.mkdir()
        git(path, "init", "-q")
        self.commit("0.0.1")

    def commit(self, version: str) -> str:
        (self.path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "-m", f"release {version}")
        return git(self.path, "rev-parse", "HEAD")

    def tag(self, name: str, target: str = "HEAD", key: Path | None = None) -> str:
        git(self.path, "tag", "-s", name, "-m", f"{name}\n", target, key=key or self.key)
        return git(self.path, "rev-parse", f"refs/tags/{name}")

    def raw(self, name: str) -> bytes:
        return subprocess.run(["git", "-C", str(self.path), "cat-file", "tag",
                               f"refs/tags/{name}"], capture_output=True, check=True,
                              env=_git_env()).stdout

    def put_tag_object(self, ref_name: str, raw: bytes) -> None:
        oid = git(self.path, "hash-object", "-t", "tag", "-w", "--literally", "--stdin",
                  input=raw)
        git(self.path, "update-ref", f"refs/tags/{ref_name}", oid)

    @property
    def gitdir(self) -> au.Git:
        return au.Git(str(self.path / ".git"))


@pytest.fixture
def keys(tmp_path):
    good = keygen(tmp_path / "release_key")
    other = keygen(tmp_path / "other_key")
    signers = tmp_path / "allowed_signers"
    signers.write_text(signers_line(good))
    return good, other, signers


@pytest.fixture
def origin(tmp_path, keys):
    return Origin(tmp_path / "origin", keys[0])


# ------------------------------------------------------------ tag verifier


def test_good_tag_verifies(origin, keys):
    commit = origin.commit("1.0.0")
    origin.tag("validator-v1.0.0")
    tag = au.verify_release_tag(origin.gitdir, "validator-v1.0.0", str(keys[2]))
    assert (tag.version, tag.commit, tag.signer) == ((1, 0, 0), commit, PRINCIPAL)


def test_wrong_signer_is_refused(origin, keys):
    origin.commit("1.0.0")
    origin.tag("validator-v1.0.0", key=keys[1])
    with pytest.raises(au.VerifyError, match="not in the pinned allowed signers"):
        au.verify_release_tag(origin.gitdir, "validator-v1.0.0", str(keys[2]))


def test_signer_without_git_namespace_is_refused(origin, keys, tmp_path):
    origin.commit("1.0.0")
    origin.tag("validator-v1.0.0")
    list_only = tmp_path / "list_only_signers"
    list_only.write_text(signers_line(keys[0], namespaces="fugal-list@fugalai.com"))
    with pytest.raises(au.VerifyError):
        au.verify_release_tag(origin.gitdir, "validator-v1.0.0", str(list_only))


def test_trailing_text_after_signature_is_refused(origin, keys):
    origin.commit("1.0.0")
    origin.tag("validator-v1.0.0")
    raw = origin.raw("validator-v1.0.0")
    origin.put_tag_object("validator-v1.0.0", raw + b"approved: everything\n")
    with pytest.raises(au.VerifyError, match="text after the signature"):
        au.verify_release_tag(origin.gitdir, "validator-v1.0.0", str(keys[2]))


def test_second_signature_block_is_refused(origin, keys):
    origin.commit("1.0.0")
    origin.tag("validator-v1.0.0")
    raw = origin.raw("validator-v1.0.0")
    sig = raw[raw.index(au.SIG_BEGIN):]
    origin.put_tag_object("validator-v1.0.0", raw + sig)
    with pytest.raises(au.VerifyError, match="exactly one signature"):
        au.verify_release_tag(origin.gitdir, "validator-v1.0.0", str(keys[2]))


def test_edited_message_does_not_verify(origin, keys):
    origin.commit("1.0.0")
    origin.tag("validator-v1.0.0")
    raw = origin.raw("validator-v1.0.0")
    origin.put_tag_object("validator-v1.0.0", raw.replace(b"\n\nvalidator-v1.0.0\n",
                                                          b"\n\nvalidator-v1.0.0 edited\n"))
    with pytest.raises(au.VerifyError, match="does not verify"):
        au.verify_release_tag(origin.gitdir, "validator-v1.0.0", str(keys[2]))


def test_renamed_tag_is_refused(origin, keys):
    """A genuinely signed old release published under a higher version's name."""
    origin.commit("1.0.0")
    oid = origin.tag("validator-v1.0.0")
    git(origin.path, "update-ref", "refs/tags/validator-v9.0.0", oid)
    with pytest.raises(au.VerifyError, match="renamed"):
        au.verify_release_tag(origin.gitdir, "validator-v9.0.0", str(keys[2]))


def test_lightweight_and_non_commit_tags_are_refused(origin, keys):
    origin.commit("1.0.0")
    git(origin.path, "tag", "validator-v1.0.0")
    with pytest.raises(au.VerifyError, match="not an annotated tag"):
        au.verify_release_tag(origin.gitdir, "validator-v1.0.0", str(keys[2]))
    tree = git(origin.path, "rev-parse", "HEAD^{tree}")
    origin.tag("validator-v1.0.1", target=tree)
    with pytest.raises(au.VerifyError, match="not a commit"):
        au.verify_release_tag(origin.gitdir, "validator-v1.0.1", str(keys[2]))


def test_unsigned_annotated_tag_is_refused(origin, keys):
    origin.commit("1.0.0")
    git(origin.path, "tag", "-a", "validator-v1.0.0", "-m", "no signature")
    with pytest.raises(au.VerifyError):
        au.verify_release_tag(origin.gitdir, "validator-v1.0.0", str(keys[2]))


@pytest.mark.parametrize("name", ["validator-v1.0", "validator-v1.0.0-rc1", "validator-v01.0.0",
                                  "v1.0.0", "image-v2026.11.03", "validator-v1.0.0+x"])
def test_only_plain_semver_tag_names(name):
    assert au.parse_version(name) is None


def test_candidates_move_forward_only():
    names = ["validator-v0.9.0", "validator-v1.0.0", "validator-v1.0.1", "validator-v1.2.0",
             "validator-v1.10.0", "image-v2026.11.24", "validator-v2.0.0-rc1"]
    got = au.candidate_tags(names, (1, 0, 0), failed={"1.2.0"})
    assert [n for _, n in got] == ["validator-v1.10.0", "validator-v1.0.1"]
    assert au.candidate_tags(names, (1, 10, 0)) == []      # same version: never reinstalled


def test_verify_tag_cli_checks_the_package_version(origin, keys, capsys):
    origin.commit("1.0.0")
    origin.tag("validator-v1.0.0")
    origin.tag("validator-v1.0.1")       # same commit: pyproject still says 1.0.0
    args = ["--git-dir", str(origin.path / ".git"), "--signers", str(keys[2])]
    assert au.main(["verify-tag", "validator-v1.0.0", *args]) == 0
    assert au.main(["verify-tag", "validator-v1.0.1", *args]) == 1
    assert "REJECTED validator-v1.0.1" in capsys.readouterr().out


# ----------------------------------------------------------------- journal


def line(minutes_ago: float, msg: str, now: float = T0) -> tuple[float, str]:
    return now - minutes_ago * 60, f"2026-11-25 19:52:00,123 fugal.validator INFO {msg}"


HEADER = "=== Epoch e00026632 (boundary block 9587520, hash 0xabcdef0123456789, protocol v5) ==="
IDLE_KW = dict(epoch_interval=4320, block_seconds=12, min_before_collection=600)


def test_idle_after_weights_and_epoch_end():
    entries = [line(70, "Weights set successfully"), line(65, HEADER),
               line(64, "Waiting for collection block 9587700 (178 blocks, ~36 min)"),
               line(28, "Querying 40 miners"), line(10, "Weights set successfully"),
               line(10, "Weights confirmed on chain: ok"),
               line(9, "Waiting for next epoch boundary (every 360 blocks)...")]
    idle, why = au.idle_window(entries, T0, **IDLE_KW)
    assert idle and "weights set" in why


def test_not_idle_between_weights_and_epoch_end():
    """Weights set but the epoch's state not yet saved: a restart would redo the epoch."""
    entries = [line(30, HEADER), line(1, "Weights set successfully")]
    assert au.idle_window(entries, T0, **IDLE_KW)[0] is False


def test_previous_epoch_weights_do_not_count():
    entries = [line(80, "Weights set successfully"),
               line(79, "Waiting for next epoch boundary (every 360 blocks)..."),
               line(40, HEADER), line(39, "Querying 40 miners")]
    assert au.idle_window(entries, T0, **IDLE_KW)[0] is False


@pytest.mark.parametrize("blocks,minutes_ago,idle", [
    (178, 1, True),      # ~35 min left
    (100, 1, True),      # ~19 min left
    (55, 0, True),       # 11 min left
    (50, 0, False),      # exactly 10 min: not more than 10
    (40, 1, False),      # ~7 min left
    (100, 15, False),    # the line is 15 min old: ~5 min left
])
def test_collection_wait_window(blocks, minutes_ago, idle):
    entries = [line(minutes_ago + 1, HEADER),
               line(minutes_ago, f"Waiting for collection block 9587700 ({blocks} blocks, "
                                 f"~{blocks * 12 / 60:.0f} min)")]
    assert au.idle_window(entries, T0, **IDLE_KW)[0] is idle


def test_collection_wait_followed_by_other_lines_is_not_idle():
    entries = [line(5, HEADER), line(4, "Waiting for collection block 9587700 (178 blocks, ~36 min)"),
               line(3, "Could not read block height (timeout); retrying")]
    assert au.idle_window(entries, T0, **IDLE_KW)[0] is False


def test_no_header_or_stale_header_is_not_idle():
    assert au.idle_window([line(1, "Waiting for next epoch boundary")], T0, **IDLE_KW)[0] is False
    stale = [line(80, HEADER), line(70, "Weights set successfully"),
             line(69, "Waiting for next epoch boundary (every 360 blocks)...")]
    idle, why = au.idle_window(stale, T0, **IDLE_KW)
    assert idle is False and "older than one epoch" in why


def test_parse_journal_json_and_ready_status():
    text = "\n".join([
        json.dumps({"__REALTIME_TIMESTAMP": "1800000000000000", "MESSAGE": "x INFO READY: ok"}),
        json.dumps({"__REALTIME_TIMESTAMP": "1800000001500000",
                    "MESSAGE": list(b"x WARNING NOT READY: \xff")}),
        "-- No entries --",
    ])
    entries = au.parse_journal_json(text)
    assert entries[0] == (1_800_000_000.0, "x INFO READY: ok")
    assert entries[1][0] == 1_800_000_001.5 and "NOT READY" in entries[1][1]
    assert au.ready_status(entries[:1])[0] == "ready"
    assert au.ready_status(entries[1:])[0] == "not_ready"
    assert au.ready_status([(0, "fugal-validator.service: Main process exited, code=exited, "
                                "status=1/FAILURE")])[0] == "exited"
    assert au.ready_status([(0, "Setting weights")])[0] is None


# ------------------------------------------------------- environment file


def test_env_file_and_argument_expansion():
    env = au.parse_env_file('# c\n; c\nFUGAL_NETWORK=finney\nWALLET_NAME="w a"\n'
                            "HOTKEY_NAME='hk'\nFUGAL_NETUID=70\nbad line\n")
    assert env == {"FUGAL_NETWORK": "finney", "WALLET_NAME": "w a", "HOTKEY_NAME": "hk",
                   "FUGAL_NETUID": "70"}
    args = au.expand_args(au.Config().validator_args, env)
    assert args == ["--network", "finney", "--netuid", "70", "--coldkey", "w a",
                    "--hotkey", "hk", "--live"]
    with pytest.raises(au.UpdateError, match="FUGAL_NETWORK"):
        au.expand_args("--network ${FUGAL_NETWORK}", {})


def test_state_paths_must_survive_a_switch():
    good = {"FUGAL_STATE_PATH": "/var/lib/fugal/s.json", "FUGAL_EPOCH_DIR": "/var/lib/fugal/e",
            "FUGAL_EPOCH_LOG_DIR": "/var/lib/fugal"}
    assert au.state_path_problems(good, ["/opt/fugal-releases"]) == []
    assert au.state_path_problems({}, []) != []
    rel = dict(good, FUGAL_HEAD_ARCHIVE_DIR="heads")
    assert "FUGAL_HEAD_ARCHIVE_DIR" in au.state_path_problems(rel, [])[0]
    inside = dict(good, FUGAL_STATE_PATH="/opt/fugal-releases/fugal-subnet-1.0.0/s.json")
    assert au.state_path_problems(inside, ["/opt/fugal-releases"])


def test_config_rejects_unknown_keys_and_bad_paths(tmp_path):
    p = tmp_path / "c.conf"
    p.write_text("[autoupdate]\nepoch_interval = 360\nblock_seconds = 0.5\n")
    cfg = au.load_config(str(p))
    assert (cfg.epoch_interval, cfg.block_seconds) == (360, 0.5)
    p.write_text("[autoupdate]\nepoch_intervall = 360\n")
    with pytest.raises(au.UpdateError, match="unknown setting"):
        au.load_config(str(p))
    p.write_text("[autoupdate]\ninstall_root = releases\n")
    with pytest.raises(au.UpdateError, match="absolute"):
        au.load_config(str(p))


# ------------------------------------------------------------ the updater


IDLE_JOURNAL = [HEADER, "Weights set successfully",
                "Waiting for next epoch boundary (every 360 blocks)..."]
BUSY_JOURNAL = [HEADER, "Querying 40 miners"]


class FakeSystem:
    """systemd, the journal, uv and the service user, recorded."""

    def __init__(self):
        self.t = T0
        self.entries: list[tuple[float, str]] = []
        self.active = True
        self.check_result = (0, "INFO READY: approved entries 3, archive ok")
        self.on_restart = lambda n: ["INFO READY: approved entries 3, archive ok"]
        self.restarts = 0
        self.reloads = 0
        self.uv_runs = 0
        self.checks: list[list[str]] = []

    def set_journal(self, messages, minutes_ago=5):
        self.entries = [(self.t - minutes_ago * 60 + i, m) for i, m in enumerate(messages)]

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def is_active(self, service):
        return self.active

    def daemon_reload(self):
        self.reloads += 1

    def restart(self, service):
        self.restarts += 1
        self.t += 5
        self.entries += [(self.t + 20 + i, m) for i, m in enumerate(self.on_restart(self.restarts))]

    def journal(self, service, since):
        return [e for e in self.entries if e[0] >= since]

    def as_user(self, user, argv, *, cwd, timeout, log_path):
        if argv[0] in ("mkdir", "cp"):
            return subprocess.run(argv, cwd=cwd).returncode
        assert argv[1:] == ["sync", "--locked", "--extra", "tee"]
        self.uv_runs += 1
        os.makedirs(os.path.join(cwd, ".venv", "bin"), exist_ok=True)
        return 0

    def chown_tree(self, path, user):
        pass

    def run_check(self, user, workdir, env_file, argv, timeout, log_path):
        self.checks.append(argv)
        return self.check_result


ENV_TEXT = ("FUGAL_NETWORK=finney\nFUGAL_NETUID=70\nWALLET_NAME=fugal_validator\n"
            "HOTKEY_NAME=default\nFUGAL_EPOCH_INTERVAL=4320\n"
            "FUGAL_STATE_PATH=/var/lib/fugal/validator_state.json\n"
            "FUGAL_EPOCH_DIR=/var/lib/fugal/epochs\nFUGAL_EPOCH_LOG_DIR=/var/lib/fugal\n")


@pytest.fixture
def host(tmp_path, keys, origin):
    base = tmp_path / "base"
    (base / "data" / "benchmarks").mkdir(parents=True)
    (base / "data" / "embeddings").mkdir(parents=True)
    (base / "data" / "benchmarks" / "pool_v5.json.gz").write_bytes(b"pool")
    (base / "data" / "embeddings" / "pool_v5.npz").write_bytes(b"emb")
    (base / "data" / "benchmarks" / "notes.txt").write_text("not a cache")
    (base / "pyproject.toml").write_text('[project]\nversion = "1.0.0"\n')
    releases = tmp_path / "releases"
    releases.mkdir(mode=0o755)
    dropin = tmp_path / "dropin"
    dropin.mkdir(mode=0o755)
    env_file = tmp_path / "validator.env"
    env_file.write_text(ENV_TEXT)
    # The fetched code's own signers file names the attacker's key: never read.
    (origin.path / "deploy" / "keys").mkdir(parents=True)
    (origin.path / "deploy" / "keys" / "allowed_signers").write_text(signers_line(keys[1]))
    cfg = au.Config(remote=str(origin.path), base_dir=str(base), install_root=str(releases),
                    env_file=str(env_file), uv="uv", signers=str(keys[2]),
                    state_dir=str(tmp_path / "state"), dropin_dir=str(dropin), ready_timeout=60)
    fake = FakeSystem()
    up = au.Updater(cfg, fake, trusted_uid=os.getuid())
    up.init(None)                       # baseline: the base checkout's 1.0.0
    return up, fake, cfg, origin


def state(cfg):
    return json.loads(Path(cfg.state_file).read_text())


def release(origin, version, key=None):
    commit = origin.commit(version)
    origin.tag(f"validator-v{version}", key=key)
    return commit


def test_init_baseline_is_the_base_checkout_version(host):
    up, fake, cfg, origin = host
    assert state(cfg)["high_water"] == "1.0.0"


def test_up_to_date_and_downgrade_do_nothing(host):
    up, fake, cfg, origin = host
    release(origin, "0.9.0")
    release(origin, "1.0.0")
    assert up.run_once() == 0
    assert fake.uv_runs == 0 and fake.restarts == 0 and not os.listdir(cfg.install_root)


def test_success_switches_when_idle(host):
    up, fake, cfg, origin = host
    commit = release(origin, "1.1.0")
    fake.set_journal(IDLE_JOURNAL)
    assert up.run_once() == 0
    dest = Path(cfg.install_root) / "fugal-subnet-1.1.0"
    assert git(dest, "rev-parse", "HEAD") == commit
    assert (dest / "data" / "benchmarks" / "pool_v5.json.gz").read_bytes() == b"pool"
    assert (dest / "data" / "embeddings" / "pool_v5.npz").read_bytes() == b"emb"
    assert not (dest / "data" / "benchmarks" / "notes.txt").exists()
    dropin = Path(cfg.dropin_path).read_text()
    assert f"WorkingDirectory={dest}\n" in dropin
    assert "ExecStart=\n" in dropin
    assert f"ExecStart={dest}/.venv/bin/python neurons/validator.py --network ${{FUGAL_NETWORK}}" \
        in dropin
    assert fake.checks[0][:2] == [f"{dest}/.venv/bin/python", "neurons/validator.py"]
    assert fake.checks[0][-1] == "--check" and "finney" in fake.checks[0]
    s = state(cfg)
    assert s["current"] == {"version": "1.1.0", "commit": commit, "dir": str(dest)}
    assert s["high_water"] == "1.1.0" and s["staged"] is None and fake.restarts == 1
    assert up.run_once() == 0 and fake.restarts == 1          # never reinstalled


def test_check_failure_never_switches_and_is_not_retried(host, caplog):
    up, fake, cfg, origin = host
    release(origin, "1.1.0")
    fake.set_journal(IDLE_JOURNAL)
    fake.check_result = (1, "INFO loading\nWARNING NOT READY: approved entries 0 (run X)\n")
    with caplog.at_level(logging.INFO, logger="fugal-autoupdate"):
        assert up.run_once() == 1
    assert "NOT READY: approved entries 0" in caplog.text
    assert not os.path.exists(cfg.dropin_path) and fake.restarts == 0 and fake.reloads == 0
    assert not (Path(cfg.install_root) / "fugal-subnet-1.1.0").exists()
    s = state(cfg)
    assert "1.1.0" in s["failed"] and s["current"] is None and s["high_water"] == "1.0.0"
    assert up.run_once() == 0 and fake.uv_runs == 1           # not retried
    fake.check_result = (0, "READY: ok")
    release(origin, "1.1.1")                                    # a newer one is tried
    assert up.run_once() == 0
    assert state(cfg)["current"]["version"] == "1.1.1"


def test_missing_ready_rolls_back(host):
    up, fake, cfg, origin = host
    release(origin, "1.1.0")
    fake.set_journal(IDLE_JOURNAL)
    assert up.run_once() == 0                                   # 1.1.0 runs
    v110 = Path(cfg.dropin_path).read_text()
    release(origin, "1.2.0")
    fake.set_journal(IDLE_JOURNAL)
    fake.on_restart = lambda n: [] if n == 2 else ["INFO READY: back on 1.1.0"]
    assert up.run_once() == 1
    assert fake.restarts == 3                                   # switch, then rollback
    assert Path(cfg.dropin_path).read_text() == v110            # previous drop-in restored
    s = state(cfg)
    assert s["current"]["version"] == "1.1.0" and "1.2.0" in s["failed"]
    assert "no READY" in s["failed"]["1.2.0"]["reason"]
    assert not (Path(cfg.install_root) / "fugal-subnet-1.2.0").exists()
    assert (Path(cfg.install_root) / "fugal-subnet-1.1.0").exists()


def test_not_ready_after_restart_rolls_back_to_the_base_unit(host):
    up, fake, cfg, origin = host
    release(origin, "1.1.0")
    fake.set_journal(IDLE_JOURNAL)
    fake.on_restart = lambda n: (["WARNING NOT READY: v5 inputs missing"] if n == 1
                                 else ["INFO READY: base"])
    assert up.run_once() == 1
    assert not os.path.exists(cfg.dropin_path) and fake.restarts == 2
    assert state(cfg)["current"] is None and "1.1.0" in state(cfg)["failed"]


def test_waits_for_the_idle_window_then_switches_without_restaging(host):
    up, fake, cfg, origin = host
    release(origin, "1.1.0")
    fake.set_journal(BUSY_JOURNAL)
    assert up.run_once() == 0
    assert fake.restarts == 0 and state(cfg)["staged"]["version"] == "1.1.0"
    fake.active = False
    fake.set_journal(IDLE_JOURNAL)
    assert up.run_once() == 0 and fake.restarts == 0          # a stopped service stays stopped
    fake.active = True
    assert up.run_once() == 0
    assert fake.restarts == 1 and fake.uv_runs == 1 and len(fake.checks) == 1
    assert state(cfg)["current"]["version"] == "1.1.0"


def test_newer_release_supersedes_a_staged_one(host):
    up, fake, cfg, origin = host
    release(origin, "1.1.0")
    fake.set_journal(BUSY_JOURNAL)
    up.run_once()
    release(origin, "1.2.0")
    up.run_once()
    assert state(cfg)["staged"]["version"] == "1.2.0"
    assert not (Path(cfg.install_root) / "fugal-subnet-1.1.0").exists()


def test_forged_tags_are_never_staged(host, keys, caplog):
    up, fake, cfg, origin = host
    release(origin, "1.0.0")
    git(origin.path, "update-ref", "refs/tags/validator-v9.0.0",
        git(origin.path, "rev-parse", "refs/tags/validator-v1.0.0"))
    release(origin, "1.5.0", key=keys[1])     # signed by the key the fetched code trusts
    fake.set_journal(IDLE_JOURNAL)
    with caplog.at_level(logging.INFO, logger="fugal-autoupdate"):
        assert up.run_once() == 0
    assert "REJECTED tag validator-v9.0.0" in caplog.text
    assert "REJECTED tag validator-v1.5.0" in caplog.text
    assert fake.uv_runs == 0 and not os.listdir(cfg.install_root)
    assert state(cfg)["failed"] == {}          # a bad tag cannot block the real release


def test_refuses_relative_state_paths(host):
    up, fake, cfg, origin = host
    Path(cfg.env_file).write_text(ENV_TEXT.replace("FUGAL_STATE_PATH=/var/lib/fugal/",
                                                   "FUGAL_STATE_PATH="))
    release(origin, "1.1.0")
    assert up.run_once() == 1 and fake.uv_runs == 0


def test_prune_keeps_current_and_previous(host):
    up, fake, cfg, origin = host
    for v in ("1.1.0", "1.2.0", "1.3.0"):
        release(origin, v)
        fake.set_journal(IDLE_JOURNAL)
        assert up.run_once() == 0
    assert sorted(os.listdir(cfg.install_root)) == ["fugal-subnet-1.2.0", "fugal-subnet-1.3.0"]
    assert Path(cfg.base_dir).is_dir()
    assert state(cfg)["previous"]["version"] == "1.2.0"


def test_install_root_writable_by_others_is_refused(host):
    up, fake, cfg, origin = host
    os.chmod(cfg.install_root, 0o777)
    release(origin, "1.1.0")
    assert up.run_once() == 1 and fake.uv_runs == 0


def test_install_root_under_a_directory_others_can_write_is_refused(host, tmp_path):
    """The service user could rename install_root away and put a symlink there."""
    up, fake, cfg, origin = host
    release(origin, "1.1.0")
    os.chmod(tmp_path, 0o777)
    try:
        assert up.run_once() == 1 and fake.uv_runs == 0
        os.chmod(tmp_path, 0o1777)                     # sticky, like /tmp: entries are safe
        fake.set_journal(BUSY_JOURNAL)
        assert up.run_once() == 0 and fake.uv_runs == 1
    finally:
        os.chmod(tmp_path, 0o700)
