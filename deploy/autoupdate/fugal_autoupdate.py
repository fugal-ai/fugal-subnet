#!/usr/bin/env python3
"""fugal-autoupdate: opt-in automatic upgrades for a Fugal validator.

Installed once by deploy/autoupdate/install.sh and run by a systemd timer
(fugal-autoupdate.timer, every 10 minutes). An operator who does not install
it sees no change: nothing in the validator reads or imports this file.

What it trusts: only an annotated tag named `validator-v<MAJOR>.<MINOR>.<PATCH>`
whose tag object carries an SSH signature that verifies, in the `git`
namespace, against the allowed-signers file copied at install time
(/etc/fugal/autoupdate/allowed_signers). The signers file in the code it
fetches is never read. The tag object is parsed and verified here, not by
`git tag -v`: its `tag` header must equal the ref's name, its `type` must be
`commit`, there must be exactly one signature and nothing after it, and the
version must be newer than every version this host has installed (or its
install baseline). The commit is then fixed by its object id.

What one run does (`run`), each step logged on one line:

1. `git fetch` of `refs/tags/validator-v*` from the configured remote into a
   root-owned mirror (the only network call this program makes; `uv sync`,
   which it runs as the service user, downloads locked packages as a manual
   upgrade does).
2. Picks the newest verified tag above the high-water mark that has not failed
   here before.
3. Stages it side by side: a clone at that commit in
   <install_root>/fugal-subnet-<version>, owned by the service user; the data
   caches (data/benchmarks/*.json.gz, data/embeddings/*.npz) copied from the
   running directory; `uv sync --locked --extra tee`; then
   `neurons/validator.py ... --check` with the service's user, environment
   file and arguments. Only exit 0 (the READY line) goes further.
4. Waits for the idle part of an epoch, read from the validator's journal:
   after `Weights set successfully` and `Waiting for next epoch boundary` for
   the current epoch, or while it is `Waiting for collection block` with more
   than `min_minutes_before_collection` left. Otherwise it exits and the next
   timer tick looks again.
5. Switches: writes a systemd drop-in pointing WorkingDirectory and ExecStart at
   the new directory, restarts the service, and waits for the READY line in the
   journal. No READY (NOT READY, an exit, or a timeout) restores the previous
   drop-in, restarts, and records the version as failed, so it is not tried
   again until a newer one appears (`reset-failed` clears that).

Root never runs git in, or executes code from, a directory the service user
can write: the mirror and the new directory are cloned as root, then the new
directory is handed to the service user, and everything after that (cache copy,
uv, --check) runs as that user.

Standard library only, Python 3.10+, so the system python3 runs it.
"""
from __future__ import annotations

import argparse
import configparser
import dataclasses
import fcntl
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TAG_PREFIX = "validator-v"
# Strict semver core only: no pre-release, no build metadata, no leading zeros.
TAG_NAME_RE = re.compile(r"^validator-v(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})$")
MANAGED_DIR_RE = re.compile(r"^fugal-subnet-(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})$")
SIG_BEGIN = b"-----BEGIN SSH SIGNATURE-----\n"
SIG_END = b"-----END SSH SIGNATURE-----\n"
OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SIGNATURE_NAMESPACE = "git"
DROPIN_NAME = "50-fugal-autoupdate.conf"
STATE_FORMAT = 1

# Journal lines of neurons/validator.py the idle-window test reads.
EPOCH_HEADER_RE = re.compile(r"=== Epoch (\S+) \(boundary block (\d+)")
WEIGHTS_SET_RE = re.compile(r"\bWeights set successfully\b")
EPOCH_DONE_RE = re.compile(r"\bWaiting for next epoch boundary\b")
COLLECTION_WAIT_RE = re.compile(r"\bWaiting for collection block (\d+) \((-?\d+) blocks")
NOT_READY_RE = re.compile(r"\bNOT READY: ")
READY_RE = re.compile(r"(?<!NOT )\bREADY: ")
# systemd's own line when the service's process ends while starting.
PROCESS_EXIT_RE = re.compile(r"Main process exited, code=")

log = logging.getLogger("fugal-autoupdate")
Version = tuple[int, int, int]


class VerifyError(Exception):
    """A tag that must not be trusted."""


class UpdateError(Exception):
    """A step of staging or switching failed."""


# ---------------------------------------------------------------- versions


def parse_version(tag_name: str) -> Version | None:
    m = TAG_NAME_RE.match(tag_name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def parse_plain_version(text: str) -> Version:
    m = re.match(r"^(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})$", text.strip())
    if not m:
        raise ValueError(f"{text!r} is not a MAJOR.MINOR.PATCH version")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def fmt_version(v: Version) -> str:
    return "%d.%d.%d" % v


def redact_url(url: str) -> str:
    """The remote without any user:password@ part, for the log."""
    return re.sub(r"(?<=://)[^/@]*@", "", url)


# ------------------------------------------------------------ tag verifier


@dataclasses.dataclass(frozen=True)
class TagObject:
    object_id: str
    type: str
    tag: str
    tagger: str
    payload: bytes      # the signed bytes: the tag object up to the signature
    signature: bytes    # the armored SSH signature, BEGIN through END


@dataclasses.dataclass(frozen=True)
class VerifiedTag:
    name: str
    version: Version
    commit: str
    signer: str


def parse_tag_object(raw: bytes) -> TagObject:
    """Parse a raw annotated tag object strictly.

    Headers are exactly `object`, `type`, `tag`, `tagger`, in that order; the
    message is followed by exactly one SSH signature block, which ends the
    object. Anything after the END line, a second BEGIN or END armor line
    anywhere, or a PGP signature, is refused: those are the bytes `git tag -v`
    would show to a person but that the signature does not cover."""
    head, sep, body = raw.partition(b"\n\n")
    if not sep:
        raise VerifyError("tag object has no message")
    lines = head.split(b"\n")
    keys = (b"object", b"type", b"tag", b"tagger")
    if len(lines) != len(keys):
        raise VerifyError(f"tag object has {len(lines)} header lines, expected {len(keys)}")
    fields: dict[bytes, str] = {}
    for key, line in zip(keys, lines):
        k, sp, value = line.partition(b" ")
        if k != key or not sp or not value:
            raise VerifyError(f"unexpected tag header {line[:60]!r}")
        try:
            fields[key] = value.decode("ascii") if key != b"tagger" else value.decode("utf-8")
        except UnicodeDecodeError as e:
            raise VerifyError(f"tag header {key.decode()} is not text") from e
    if raw.count(b"-----BEGIN ") != 1 or raw.count(b"-----END ") != 1:
        raise VerifyError("tag must carry exactly one signature block")
    start = raw.find(SIG_BEGIN)
    if start < 0:
        raise VerifyError("tag is not SSH-signed")
    if start <= len(head) + 2 or raw[start - 1:start] != b"\n":
        raise VerifyError("signature does not start on its own line after the message")
    signature = raw[start:]
    if not signature.endswith(SIG_END):
        raise VerifyError("text after the signature block (or a truncated signature)")
    if not OID_RE.match(fields[b"object"]):
        raise VerifyError("tag object id is malformed")
    return TagObject(object_id=fields[b"object"], type=fields[b"type"], tag=fields[b"tag"],
                     tagger=fields[b"tagger"], payload=raw[:start], signature=signature)


def verify_ssh_signature(payload: bytes, signature: bytes, signers_file: str,
                         ssh_keygen: str = "ssh-keygen") -> str:
    """The principal whose key made `signature` over `payload`, in the git
    namespace, per `signers_file`; VerifyError otherwise."""
    if not os.path.isfile(signers_file):
        raise VerifyError(f"allowed signers file {signers_file} is missing")
    with tempfile.TemporaryDirectory(prefix="fugal-autoupdate-sig-") as d:
        sig_path = os.path.join(d, "tag.sig")
        with open(sig_path, "wb") as f:
            f.write(signature)
        found = subprocess.run([ssh_keygen, "-Y", "find-principals", "-f", signers_file,
                                "-s", sig_path], capture_output=True, timeout=60)
        principals = [p.strip() for p in found.stdout.decode("utf-8", "replace").splitlines()
                      if p.strip()]
        if found.returncode != 0 or not principals:
            raise VerifyError("the signing key is not in the pinned allowed signers")
        for principal in principals:
            checked = subprocess.run([ssh_keygen, "-Y", "verify", "-f", signers_file,
                                      "-I", principal, "-n", SIGNATURE_NAMESPACE, "-s", sig_path],
                                     input=payload, capture_output=True, timeout=60)
            if checked.returncode == 0:
                return principal
    raise VerifyError("the signature does not verify over the tag")


class Git:
    """git against one repository, never reading the user's or system config."""

    def __init__(self, git_dir: str, git: str = "git", timeout: int = 900):
        self.git_dir = git_dir
        self.git = git
        self.timeout = timeout

    def env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        return env

    def run(self, *args: str, check: bool = True, input: bytes | None = None) -> bytes:
        r = subprocess.run([self.git, f"--git-dir={self.git_dir}", *args], input=input,
                           capture_output=True, env=self.env(), timeout=self.timeout)
        if check and r.returncode != 0:
            raise UpdateError(f"git {args[0]} failed (exit {r.returncode}): "
                              f"{r.stderr.decode('utf-8', 'replace').strip()[-300:]}")
        return r.stdout

    def object_type(self, rev: str) -> str | None:
        r = subprocess.run([self.git, f"--git-dir={self.git_dir}", "cat-file", "-t", rev],
                           capture_output=True, env=self.env(), timeout=self.timeout)
        return r.stdout.decode().strip() if r.returncode == 0 else None

    def tag_names(self) -> list[str]:
        out = self.run("for-each-ref", "--format=%(refname)", f"refs/tags/{TAG_PREFIX}*")
        names = []
        for line in out.decode("utf-8", "replace").splitlines():
            if line.startswith("refs/tags/"):
                names.append(line[len("refs/tags/"):])
        return names


def verify_release_tag(git: Git, name: str, signers_file: str,
                       ssh_keygen: str = "ssh-keygen") -> VerifiedTag:
    """Verify refs/tags/<name> strictly (module docstring); VerifyError if not."""
    version = parse_version(name)
    if version is None:
        raise VerifyError(f"{name!r} is not a validator-vMAJOR.MINOR.PATCH tag")
    ref = f"refs/tags/{name}"
    kind = git.object_type(ref)
    if kind != "tag":
        raise VerifyError(f"{name} is not an annotated tag (it is {kind or 'missing'})")
    raw = git.run("cat-file", "tag", ref)
    obj = parse_tag_object(raw)
    if obj.tag != name:
        raise VerifyError(f"{name} holds a tag object named {obj.tag!r} (a renamed tag)")
    if obj.type != "commit":
        raise VerifyError(f"{name} points at a {obj.type}, not a commit")
    signer = verify_ssh_signature(obj.payload, obj.signature, signers_file, ssh_keygen)
    if git.object_type(obj.object_id) != "commit":
        raise VerifyError(f"{name}'s commit {obj.object_id} is not in the repository")
    return VerifiedTag(name=name, version=version, commit=obj.object_id, signer=signer)


def candidate_tags(names: list[str], high_water: Version,
                   failed: set[str] | frozenset = frozenset()) -> list[tuple[Version, str]]:
    """Tags strictly newer than `high_water` and not failed here, newest first.
    Never a downgrade, never a reinstall of the same version."""
    out = []
    for name in names:
        v = parse_version(name)
        if v is not None and v > high_water and fmt_version(v) not in failed:
            out.append((v, name))
    return sorted(out, reverse=True)


# ----------------------------------------------------------------- journal


Entry = tuple[float, str]   # (unix seconds, message)


def parse_journal_json(text: str) -> list[Entry]:
    """`journalctl -o json` lines -> [(unix seconds, message)]."""
    out: list[Entry] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
            ts = int(rec["__REALTIME_TIMESTAMP"]) / 1e6
        except (ValueError, KeyError, TypeError):
            continue
        msg = rec.get("MESSAGE")
        if isinstance(msg, list):   # journald sends non-UTF-8 messages as bytes
            msg = bytes(b & 0xFF for b in msg if isinstance(b, int)).decode("utf-8", "replace")
        if isinstance(msg, str):
            out.append((ts, msg))
    return out


def idle_window(entries: list[Entry], now: float, *, epoch_interval: float,
                block_seconds: float, min_before_collection: float) -> tuple[bool, str]:
    """Whether the validator is in the idle part of its current epoch.

    Idle means one of:
    - after the latest `=== Epoch` header (no older than one epoch), the
      journal shows `Weights set successfully` and after it `Waiting for next
      epoch boundary` (the epoch's state is saved, its tail is done); or
    - the latest line after that header is `Waiting for collection block N
      (R blocks, ...)` and R blocks at `block_seconds` each, less the time since
      the line, leave more than `min_before_collection` seconds.
    Anything else (collection, scoring, weights still pending, no header, an
    unknown line after the collection wait) is not idle."""
    header = None
    for i in range(len(entries) - 1, -1, -1):
        m = EPOCH_HEADER_RE.search(entries[i][1])
        if m:
            header = (i, m.group(1))
            break
    if header is None:
        return False, "no epoch header in the validator's recent journal"
    idx, epoch_id = header
    if now - entries[idx][0] > epoch_interval:
        return False, f"the latest epoch header ({epoch_id}) is older than one epoch"
    after = entries[idx + 1:]
    weights_at = None
    for j, (_, msg) in enumerate(after):
        if WEIGHTS_SET_RE.search(msg):
            weights_at = j
    if weights_at is not None and any(EPOCH_DONE_RE.search(m) for _, m in after[weights_at + 1:]):
        return True, f"epoch {epoch_id}: weights set and the epoch finished"
    if after:
        ts, msg = after[-1]
        m = COLLECTION_WAIT_RE.search(msg)
        if m:
            left = int(m.group(2)) * block_seconds - (now - ts)
            if left > min_before_collection:
                return True, (f"epoch {epoch_id}: waiting for collection, "
                              f"~{left / 60:.0f} min left")
            return False, (f"epoch {epoch_id}: collection in ~{max(left, 0) / 60:.0f} min "
                           f"(needs > {min_before_collection / 60:.0f})")
    return False, f"epoch {epoch_id}: collection, scoring or weights in progress"


def ready_status(entries: list[Entry]) -> tuple[str | None, str]:
    """('ready' | 'not_ready' | 'exited' | None, the line) from journal entries
    written since a restart."""
    for _, msg in entries:
        if NOT_READY_RE.search(msg):
            return "not_ready", msg
        if READY_RE.search(msg):
            return "ready", msg
        if PROCESS_EXIT_RE.search(msg):
            return "exited", msg
    return None, ""


# ------------------------------------------------------- environment file


def parse_env_file(text: str) -> dict[str, str]:
    """A systemd EnvironmentFile: KEY=VALUE lines, '#' and ';' comments,
    optional matching quotes around the value."""
    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        env[key] = value
    return env


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def expand_args(template: str, env: dict[str, str]) -> list[str]:
    """Split the ExecStart argument template and expand $VAR / ${VAR} from the
    environment file, as systemd does for the unit. An unset name is an error."""
    out = []
    for word in shlex.split(template):
        def sub(m: re.Match) -> str:
            name = m.group(1) or m.group(2)
            if name not in env:
                raise UpdateError(f"validator_args uses ${name}, which the environment file "
                                  "does not set")
            return env[name]
        out.append(_VAR_RE.sub(sub, word))
    return out


# The validator's state must live outside the code directory, or a switch
# would start the new version with none of it.
REQUIRED_ABSOLUTE = ("FUGAL_STATE_PATH", "FUGAL_EPOCH_DIR", "FUGAL_EPOCH_LOG_DIR")


def state_path_problems(env: dict[str, str], code_dirs: list[str]) -> list[str]:
    problems = []
    for key in REQUIRED_ABSOLUTE:
        if not env.get(key, "").strip():
            problems.append(f"{key} is not set (its default is inside the code directory)")
    for key, value in sorted(env.items()):
        if not re.match(r"^FUGAL_[A-Z0-9_]*_(PATH|DIR)$", key) or not value.strip():
            continue
        if not os.path.isabs(value):
            problems.append(f"{key} is a relative path (it would stay behind in the old "
                            "code directory)")
            continue
        real = os.path.normpath(value)
        for d in code_dirs:
            d = os.path.normpath(d)
            if real == d or real.startswith(d + os.sep):
                problems.append(f"{key} is inside {d}, where old versions are deleted")
    return problems


# ------------------------------------------------------------------ config


@dataclasses.dataclass
class Config:
    remote: str = "https://github.com/fugal-ai/fugal-subnet.git"
    service: str = "fugal-validator"
    service_user: str = "fugal"
    base_dir: str = "/opt/fugal/fugal-subnet"
    install_root: str = "/opt/fugal-releases"
    env_file: str = "/etc/fugal/validator.env"
    validator_args: str = ("--network ${FUGAL_NETWORK} --netuid ${FUGAL_NETUID} "
                           "--coldkey ${WALLET_NAME} --hotkey ${HOTKEY_NAME} --live")
    uv: str = "/usr/local/bin/uv"
    uv_extras: str = "tee"
    signers: str = "/etc/fugal/autoupdate/allowed_signers"
    state_dir: str = "/var/lib/fugal-autoupdate"
    dropin_dir: str = "/etc/systemd/system/fugal-validator.service.d"
    epoch_interval: int = 4320
    block_seconds: float = 12.0
    min_minutes_before_collection: float = 10.0
    ready_timeout: int = 1200
    check_timeout: int = 1800
    check_max_age: int = 7200
    sync_timeout: int = 3600
    keep_versions: int = 2

    @property
    def dropin_path(self) -> str:
        return os.path.join(self.dropin_dir, DROPIN_NAME)

    @property
    def mirror(self) -> str:
        return os.path.join(self.state_dir, "mirror.git")

    @property
    def state_file(self) -> str:
        return os.path.join(self.state_dir, "state.json")


def load_config(path: str) -> Config:
    cp = configparser.ConfigParser(interpolation=None)
    with open(path, encoding="utf-8") as f:
        cp.read_file(f)
    if not cp.has_section("autoupdate"):
        raise UpdateError(f"{path} has no [autoupdate] section")
    cfg = Config()
    known = {f.name: f for f in dataclasses.fields(Config)}
    for key, value in cp.items("autoupdate"):
        if key not in known:
            raise UpdateError(f"{path}: unknown setting {key!r}")
        default = getattr(cfg, key)
        try:
            typed = type(default)(value) if not isinstance(default, str) else value
        except ValueError as e:
            raise UpdateError(f"{path}: {key} = {value!r} is not a {type(default).__name__}") from e
        setattr(cfg, key, typed)
    for key in ("base_dir", "install_root", "state_dir", "dropin_dir", "signers", "env_file"):
        value = getattr(cfg, key)
        if not os.path.isabs(value) or any(c.isspace() for c in value) or "%" in value:
            raise UpdateError(f"{path}: {key} must be an absolute path without spaces or %")
    if cfg.keep_versions < 2:
        raise UpdateError(f"{path}: keep_versions must be at least 2 (current and previous)")
    return cfg


# ------------------------------------------------------------------ system


class SystemOps:
    """Everything that touches systemd, the journal or another user. Tests
    replace it; the real one needs root."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _systemctl(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["systemctl", *args], capture_output=True, text=True, timeout=600)

    def is_active(self, service: str) -> bool:
        return self._systemctl("is-active", "--quiet", service).returncode == 0

    def daemon_reload(self) -> None:
        r = self._systemctl("daemon-reload")
        if r.returncode != 0:
            raise UpdateError(f"systemctl daemon-reload failed: {r.stderr.strip()[-300:]}")

    def restart(self, service: str) -> None:
        r = self._systemctl("restart", service)
        if r.returncode != 0:
            raise UpdateError(f"systemctl restart {service} failed: {r.stderr.strip()[-300:]}")

    def journal(self, service: str, since: float) -> list[Entry]:
        r = subprocess.run(["journalctl", "-u", service, "-o", "json", "--no-pager",
                            "--since", f"@{int(since)}"],
                           capture_output=True, text=True, timeout=300)
        return parse_journal_json(r.stdout)

    def _user(self, user: str):
        import pwd
        pw = pwd.getpwnam(user)
        # The user's own tool directories too: uv may need cargo to build a wheel.
        path = [os.path.join(pw.pw_dir, ".local", "bin"), os.path.join(pw.pw_dir, ".cargo", "bin"),
                "/usr/local/bin", "/usr/bin", "/bin"]
        env = {"HOME": pw.pw_dir, "USER": user, "LOGNAME": user, "LANG": "C.UTF-8",
               "PATH": ":".join(path)}
        return pw, env

    def as_user(self, user: str, argv: list[str], *, cwd: str, timeout: int,
                log_path: str) -> int:
        """Run argv as `user` (its uid, gid and groups only), output to log_path."""
        pw, env = self._user(user)
        with open(log_path, "ab") as out:
            r = subprocess.run(argv, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                               user=pw.pw_uid, group=pw.pw_gid,
                               extra_groups=os.getgrouplist(user, pw.pw_gid),
                               timeout=timeout)
        return r.returncode

    def chown_tree(self, path: str, user: str) -> None:
        pw, _ = self._user(user)
        os.lchown(path, pw.pw_uid, pw.pw_gid)
        for root, dirs, files in os.walk(path):
            for name in dirs + files:
                os.lchown(os.path.join(root, name), pw.pw_uid, pw.pw_gid)

    def run_check(self, user: str, workdir: str, env_file: str, argv: list[str],
                  timeout: int, log_path: str) -> tuple[int, str]:
        """`--check` as the service would run it: its user, its environment
        file, the new directory. Returns (exit code, output)."""
        cmd = ["systemd-run", "--quiet", "--wait", "--pipe", "--collect",
               "--service-type=exec", "-p", f"User={user}",
               "-p", f"EnvironmentFile={env_file}", "-p", f"WorkingDirectory={workdir}",
               "-p", f"RuntimeMaxSec={timeout}", "--", *argv]
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                           timeout=timeout + 120)
        text = (r.stdout + r.stderr).decode("utf-8", "replace")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(text)
        return r.returncode, text


# ------------------------------------------------------------------ state


def empty_state(baseline: Version) -> dict:
    return {"format": STATE_FORMAT, "high_water": fmt_version(baseline), "current": None,
            "previous": None, "staged": None, "failed": {}, "last": None}


def load_state(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    if state.get("format") != STATE_FORMAT:
        raise UpdateError(f"{path}: unknown state format {state.get('format')!r}")
    return state


def write_atomic(path: str, data: str, mode: int = 0o644) -> None:
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_state(path: str, state: dict) -> None:
    write_atomic(path, json.dumps(state, indent=2, sort_keys=True) + "\n", mode=0o600)


# ---------------------------------------------------------------- updater


CACHE_GLOBS = (("data/benchmarks", ".json.gz"), ("data/embeddings", ".npz"))


class Updater:
    def __init__(self, cfg: Config, system: SystemOps | None = None, *,
                 trusted_uid: int = 0, ssh_keygen: str = "ssh-keygen", git: str = "git"):
        self.cfg = cfg
        self.sys = system or SystemOps(cfg)
        self.trusted_uid = trusted_uid
        self.ssh_keygen = ssh_keygen
        self.git_bin = git
        self.mirror = Git(cfg.mirror, git=git)

    # --- setup

    def init(self, baseline: Version | None) -> dict:
        os.makedirs(self.cfg.state_dir, mode=0o700, exist_ok=True)
        self._check_owned(self.cfg.state_dir)
        if not os.path.isdir(self.cfg.mirror):
            subprocess.run([self.git_bin, "init", "-q", "--bare", self.cfg.mirror],
                           check=True, capture_output=True, env=self.mirror.env(), timeout=120)
        if os.path.exists(self.cfg.state_file):
            state = load_state(self.cfg.state_file)
            log.info("already initialised: high-water v%s", state["high_water"])
            return state
        if baseline is None:
            baseline = self.base_checkout_version()
        state = empty_state(baseline)
        save_state(self.cfg.state_file, state)
        log.info("initialised: installs signed validator-v* releases newer than v%s",
                 fmt_version(baseline))
        return state

    def base_checkout_version(self) -> Version:
        """The version in the base checkout's pyproject.toml (read as a file,
        never executed), the default install baseline."""
        path = os.path.join(self.cfg.base_dir, "pyproject.toml")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, encoding="utf-8") as f:
            text = f.read(1 << 20)
        m = re.search(r'^version\s*=\s*"([0-9.]+)"', text, re.M)
        if not m:
            raise UpdateError(f"no version in {path}; pass --baseline")
        return parse_plain_version(m.group(1))

    def _check_owned(self, path: str) -> None:
        """A directory root writes into must be root's and not writable by
        anyone else, and so must every directory above it (a sticky one like
        /tmp aside), or the service user could swap what root writes into."""
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise UpdateError(f"{path} must be a real directory")
        if st.st_uid != self.trusted_uid or st.st_mode & 0o022:
            raise UpdateError(f"{path} must be owned by uid {self.trusted_uid} and not "
                              "group- or world-writable")
        parent = os.path.dirname(os.path.normpath(path))
        while True:
            pst = os.lstat(parent)
            if stat.S_ISLNK(pst.st_mode) or pst.st_uid not in (0, self.trusted_uid) or (
                    pst.st_mode & 0o022 and not pst.st_mode & stat.S_ISVTX):
                raise UpdateError(f"{parent} (above {path}) must be owned by root and not "
                                  "writable by other users")
            if parent == os.path.dirname(parent):
                return
            parent = os.path.dirname(parent)

    # --- one timer tick

    def run_once(self) -> int:
        os.makedirs(self.cfg.state_dir, mode=0o700, exist_ok=True)
        with open(os.path.join(self.cfg.state_dir, "lock"), "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log.info("another run holds the lock; skipping")
                return 0
            try:
                return self._run()
            except (UpdateError, OSError, ValueError, KeyError) as e:
                log.error("stopped: %s: %s", type(e).__name__, e)
                return 1

    def _run(self) -> int:
        if not os.path.exists(self.cfg.state_file):
            raise UpdateError("not initialised: run `fugal_autoupdate.py init` (install.sh does)")
        state = load_state(self.cfg.state_file)
        for d in (self.cfg.state_dir, self.cfg.install_root):
            self._check_owned(d)
        env = self._service_env()
        problems = state_path_problems(env, [self.cfg.install_root])
        if problems:
            raise UpdateError("refusing to manage this validator: " + "; ".join(problems)
                              + f" (fix {self.cfg.env_file})")
        expand_args(self.cfg.validator_args, env)    # an unset ${VAR} stops here, not mid-stage
        self._fetch()
        high_water = parse_plain_version(state["high_water"])
        target = self._select(high_water, set(state["failed"]))
        staged = state.get("staged")
        if staged and not os.path.isdir(os.path.join(staged["dir"], ".venv")):
            discard_why = "its directory is gone"
        elif staged and target is None:
            discard_why = "its tag no longer verifies"
        elif staged and (staged["version"] != fmt_version(target.version)
                         or staged["commit"] != target.commit):
            discard_why = "superseded"
        else:
            discard_why = None
        if staged and discard_why:
            log.info("discarding staged v%s (%s)", staged["version"], discard_why)
            self._remove_managed(staged["dir"], state)
            state["staged"] = None
            save_state(self.cfg.state_file, state)
            staged = None
        if target is None:
            current = state["current"]["version"] if state["current"] else "the base checkout"
            failed = ", ".join(f"v{v}" for v in sorted(state["failed"]))
            log.info("up to date: running %s; nothing newer than v%s%s",
                     f"v{current}" if state["current"] else current, state["high_water"],
                     f" (failed here, not retried: {failed})" if failed else "")
            return 0
        if staged is None:
            staged = self._stage(target, state, env)
            if staged is None:
                return 1
        elif self.sys.now() - staged["checked_at"] > self.cfg.check_max_age:
            log.info("re-running --check for staged v%s (older than %ds)",
                     staged["version"], self.cfg.check_max_age)
            if not self._check(target, staged["dir"], state, env):
                return 1
            staged["checked_at"] = self.sys.now()
            save_state(self.cfg.state_file, state)
        if not self.sys.is_active(self.cfg.service):
            log.info("staged v%s; %s is not running, so not switching", staged["version"],
                     self.cfg.service)
            return 0
        idle, why = self._idle(env)
        if not idle:
            log.info("staged v%s; waiting for the idle part of the epoch (%s)",
                     staged["version"], why)
            return 0
        log.info("idle window: %s", why)
        return 0 if self._switch(target, state) else 1

    def _service_env(self) -> dict[str, str]:
        with open(self.cfg.env_file, encoding="utf-8") as f:
            return parse_env_file(f.read())

    def _fetch(self) -> None:
        try:
            self.mirror.run("fetch", "--quiet", "--no-tags", "--prune", "--force",
                            self.cfg.remote,
                            f"+refs/tags/{TAG_PREFIX}*:refs/tags/{TAG_PREFIX}*")
            log.info("fetched validator-v* tags from %s", redact_url(self.cfg.remote))
        except (UpdateError, subprocess.TimeoutExpired) as e:
            log.warning("fetch failed, using the tags already fetched: %s", e)

    def _select(self, high_water: Version, failed: set[str]) -> VerifiedTag | None:
        for version, name in candidate_tags(self.mirror.tag_names(), high_water, failed):
            try:
                tag = verify_release_tag(self.mirror, name, self.cfg.signers, self.ssh_keygen)
            except VerifyError as e:
                log.warning("REJECTED tag %s: %s", name, e)
                continue
            log.info("tag %s verified: commit %s, signed by %s", name, tag.commit, tag.signer)
            return tag
        return None

    def _fail(self, state: dict, tag: VerifiedTag, reason: str) -> None:
        state["failed"][fmt_version(tag.version)] = {"commit": tag.commit, "reason": reason,
                                                     "at": int(self.sys.now())}
        state["staged"] = None
        state["last"] = {"at": int(self.sys.now()), "action": "failed",
                         "version": fmt_version(tag.version), "detail": reason}
        save_state(self.cfg.state_file, state)
        log.error("v%s FAILED: %s; it will not be retried until a newer release appears "
                  "(or `reset-failed`)", fmt_version(tag.version), reason)

    def _current_dir(self, state: dict) -> str:
        return state["current"]["dir"] if state["current"] else self.cfg.base_dir

    def _stage(self, tag: VerifiedTag, state: dict, env: dict[str, str]) -> dict | None:
        ver = fmt_version(tag.version)
        dest = os.path.join(self.cfg.install_root, f"fugal-subnet-{ver}")
        if os.path.normpath(dest) in {os.path.normpath(self._current_dir(state)),
                                      os.path.normpath(self.cfg.base_dir)}:
            raise UpdateError(f"refusing to stage over the running directory {dest}")
        log.info("staging v%s at %s", ver, dest)
        try:
            self._remove_managed(dest, state)
            os.mkdir(dest, 0o755)
            self._clone(tag, dest)
            self.sys.chown_tree(dest, self.cfg.service_user)
            self._copy_caches(self._current_dir(state), dest)
            sync_log = os.path.join(self.cfg.state_dir, f"uv-sync-{ver}.log")
            extras = [a for e in self.cfg.uv_extras.split() for a in ("--extra", e)]
            rc = self.sys.as_user(self.cfg.service_user,
                                  [self.cfg.uv, "sync", "--locked", *extras],
                                  cwd=dest, timeout=self.cfg.sync_timeout, log_path=sync_log)
            if rc != 0:
                raise UpdateError(f"uv sync exited {rc} (see {sync_log})")
            log.info("v%s: uv sync --locked done", ver)
        except (UpdateError, OSError, subprocess.SubprocessError) as e:
            self._remove_managed(dest, state)
            self._fail(state, tag, f"staging: {e}")
            return None
        if not self._check(tag, dest, state, env):
            return None
        staged = {"version": ver, "commit": tag.commit, "dir": dest,
                  "checked_at": self.sys.now()}
        state["staged"] = staged
        state["last"] = {"at": int(self.sys.now()), "action": "staged", "version": ver,
                         "detail": dest}
        save_state(self.cfg.state_file, state)
        return staged

    def _clone(self, tag: VerifiedTag, dest: str) -> None:
        """Clone the root-owned mirror at the verified commit, as root, into the
        fresh root-owned `dest`; checked, then handed to the service user."""
        env = self.mirror.env()

        def git(*args: str, cwd: str | None = None) -> str:
            r = subprocess.run([self.git_bin, *args], cwd=cwd, capture_output=True, text=True,
                               env=env, timeout=1800)
            if r.returncode != 0:
                raise UpdateError(f"git {args[0]} failed: {r.stderr.strip()[-300:]}")
            return r.stdout

        git("clone", "--quiet", "--no-checkout", "--no-hardlinks", self.cfg.mirror, dest)
        git("-C", dest, "checkout", "--quiet", "--detach", tag.commit)
        git("-C", dest, "remote", "set-url", "origin", self.cfg.remote)
        head = git("-C", dest, "rev-parse", "HEAD").strip()
        if head != tag.commit:
            raise UpdateError(f"checkout is at {head}, expected {tag.commit}")

    def _copy_caches(self, src_root: str, dest: str) -> None:
        """Copy the large data caches, as the service user (it owns the source),
        skipping anything the new checkout already has."""
        for sub, suffix in CACHE_GLOBS:
            src = os.path.join(src_root, sub)
            if not os.path.isdir(src) or os.path.islink(src):
                continue
            names = sorted(e.name for e in os.scandir(src)
                           if e.name.endswith(suffix) and e.is_file(follow_symlinks=False)
                           and not os.path.lexists(os.path.join(dest, sub, e.name)))
            if not names:
                continue
            log_path = os.path.join(self.cfg.state_dir, "copy-caches.log")
            rc = self.sys.as_user(self.cfg.service_user,
                                  ["mkdir", "-p", "--", os.path.join(dest, sub)],
                                  cwd=dest, timeout=60, log_path=log_path)
            if rc == 0:
                rc = self.sys.as_user(self.cfg.service_user,
                                      ["cp", "--no-dereference", "--preserve=timestamps",
                                       "-t", os.path.join(dest, sub), "--",
                                       *[os.path.join(src, n) for n in names]],
                                      cwd=dest, timeout=3600, log_path=log_path)
            if rc != 0:
                raise UpdateError(f"copying caches from {src} failed (see {log_path})")
            log.info("copied %d cache file(s) from %s", len(names), src)

    def _check(self, tag: VerifiedTag, dest: str, state: dict, env: dict[str, str]) -> bool:
        ver = fmt_version(tag.version)
        argv = [os.path.join(dest, ".venv", "bin", "python"), "neurons/validator.py",
                *expand_args(self.cfg.validator_args, env), "--check"]
        check_log = os.path.join(self.cfg.state_dir, f"check-{ver}.log")
        log.info("v%s: running validator --check as %s", ver, self.cfg.service_user)
        try:
            rc, text = self.sys.run_check(self.cfg.service_user, dest, self.cfg.env_file, argv,
                                          self.cfg.check_timeout, check_log)
        except (OSError, subprocess.SubprocessError) as e:
            rc, text = -1, f"{type(e).__name__}"
        line = next((ln.strip() for ln in reversed(text.splitlines())
                     if NOT_READY_RE.search(ln) or READY_RE.search(ln)), "")
        if rc == 0:
            log.info("v%s: --check READY", ver)
            return True
        self._remove_managed(dest, state)
        self._fail(state, tag, f"--check exited {rc}: {line[:400] or 'no READY line'} "
                               f"(full output in {check_log})")
        return False

    def _idle(self, env: dict[str, str]) -> tuple[bool, str]:
        try:
            interval = float(env.get("FUGAL_EPOCH_INTERVAL") or self.cfg.epoch_interval)
        except ValueError:
            interval = float(self.cfg.epoch_interval)
        now = self.sys.now()
        entries = self.sys.journal(self.cfg.service, now - interval - 600)
        return idle_window(entries, now, epoch_interval=interval,
                           block_seconds=self.cfg.block_seconds,
                           min_before_collection=self.cfg.min_minutes_before_collection * 60)

    def dropin_text(self, staged: dict) -> str:
        d = staged["dir"]
        args = self.cfg.validator_args.replace("%", "%%")
        return (f"# Written by fugal-autoupdate: validator-v{staged['version']}, "
                f"commit {staged['commit']}.\n"
                "# Delete this file, then `systemctl daemon-reload && systemctl restart "
                f"{self.cfg.service}`,\n# to run the base unit's directory again.\n"
                "[Service]\n"
                f"WorkingDirectory={d}\n"
                "ExecStart=\n"
                f"ExecStart={d}/.venv/bin/python neurons/validator.py {args}\n")

    def _read_dropin(self) -> str | None:
        try:
            with open(self.cfg.dropin_path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def _put_dropin(self, text: str | None) -> None:
        if text is None:
            try:
                os.unlink(self.cfg.dropin_path)
            except FileNotFoundError:
                pass
        else:
            self._check_owned(self.cfg.dropin_dir)
            write_atomic(self.cfg.dropin_path, text)
        self.sys.daemon_reload()

    def _restart_and_wait(self) -> tuple[str | None, str]:
        self.sys.restart(self.cfg.service)
        since = self.sys.now()
        deadline = since + self.cfg.ready_timeout
        while True:
            fresh = [e for e in self.sys.journal(self.cfg.service, since - 1) if e[0] >= since]
            status, line = ready_status(fresh)
            if status is not None:
                return status, line
            if self.sys.now() >= deadline:
                return None, f"no READY line within {self.cfg.ready_timeout}s"
            self.sys.sleep(10)

    def _switch(self, tag: VerifiedTag, state: dict) -> bool:
        staged = state["staged"]
        ver = staged["version"]
        previous = self._read_dropin()
        log.info("switching %s to v%s (%s); restarting", self.cfg.service, ver, staged["dir"])
        try:
            self._put_dropin(self.dropin_text(staged))
            status, line = self._restart_and_wait()
        except (UpdateError, OSError, subprocess.SubprocessError) as e:
            status, line = None, f"{e}"
        if status == "ready":
            log.info("v%s is running: %s", ver, line.strip()[:400])
            state["previous"] = state["current"]
            state["current"] = {"version": ver, "commit": staged["commit"], "dir": staged["dir"]}
            state["high_water"] = ver
            state["staged"] = None
            state["last"] = {"at": int(self.sys.now()), "action": "switched", "version": ver,
                             "detail": staged["dir"]}
            save_state(self.cfg.state_file, state)
            self._prune(state)
            return True
        reason = {"not_ready": "NOT READY after restart", "exited": "the process exited",
                  None: "no READY line"}[status]
        log.error("v%s did not come up (%s: %s); ROLLING BACK to %s", ver, reason,
                  line.strip()[:400], self._current_dir(state))
        try:
            self._put_dropin(previous)
            back, back_line = self._restart_and_wait()
            if back == "ready":
                log.info("rollback done: %s", back_line.strip()[:400])
            else:
                log.error("rollback restarted %s but saw no READY line (%s); check the "
                          "service by hand", self.cfg.service, back_line.strip()[:200])
        except (UpdateError, OSError, subprocess.SubprocessError) as e:
            log.error("ROLLBACK FAILED: %s; check the service by hand", e)
        self._remove_managed(staged["dir"], state)
        self._fail(state, tag, f"switch: {reason}")
        return False

    def _remove_managed(self, path: str, state: dict) -> None:
        """Delete a directory this updater created; never the base checkout, the
        running directory, the one before it, or anything outside install_root."""
        norm = os.path.normpath(path)
        keep = {os.path.normpath(self.cfg.base_dir), os.path.normpath(self._current_dir(state))}
        if state.get("previous"):
            keep.add(os.path.normpath(state["previous"]["dir"]))
        if norm in keep or not os.path.lexists(norm):
            return
        p = Path(norm)
        if (os.path.normpath(str(p.parent)) != os.path.normpath(self.cfg.install_root)
                or not MANAGED_DIR_RE.match(p.name)):
            raise UpdateError(f"refusing to remove {path}: not a directory this updater made")
        if os.path.islink(norm) or not os.path.isdir(norm):
            os.unlink(norm)
        else:
            shutil.rmtree(norm)
        log.info("removed %s", norm)

    def _prune(self, state: dict) -> None:
        """Keep the current and previous managed directories (and up to
        keep_versions in all); remove older ones."""
        dirs = []
        for e in os.scandir(self.cfg.install_root):
            m = MANAGED_DIR_RE.match(e.name)
            if m and e.is_dir(follow_symlinks=False):
                dirs.append(((int(m.group(1)), int(m.group(2)), int(m.group(3))), e.path))
        dirs.sort(reverse=True)
        for _, path in dirs[self.cfg.keep_versions:]:
            try:
                self._remove_managed(path, state)
            except UpdateError as e:
                log.warning("prune: %s", e)


# --------------------------------------------------------------------- CLI


def _status(cfg: Config) -> int:
    if not os.path.exists(cfg.state_file):
        print(f"not initialised ({cfg.state_file} missing)")
        return 1
    state = load_state(cfg.state_file)
    cur = state["current"]
    print(f"running:    {'v' + cur['version'] + ' in ' + cur['dir'] if cur else 'the base checkout ' + cfg.base_dir}")
    print(f"high-water: v{state['high_water']} (only newer releases are installed)")
    staged = state.get("staged")
    print(f"staged:     {'v' + staged['version'] + ' in ' + staged['dir'] if staged else 'none'}")
    for ver, rec in sorted(state["failed"].items()):
        print(f"failed:     v{ver} ({rec['reason']})")
    last = state.get("last")
    if last:
        print(f"last:       {last['action']} v{last['version']} at "
              f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(last['at']))}: "
              f"{last['detail']}")
    print(f"drop-in:    {cfg.dropin_path if os.path.exists(cfg.dropin_path) else 'none'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="/etc/fugal/autoupdate.conf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init", help="create the state and the mirror (install.sh runs it)")
    p_init.add_argument("--baseline", help="install only releases newer than this version "
                        "(default: the base checkout's pyproject version)")
    sub.add_parser("run", help="one pass: fetch, verify, stage, switch when idle")
    sub.add_parser("status", help="what runs, what is staged, what failed")
    p_reset = sub.add_parser("reset-failed", help="allow a failed version to be tried again")
    p_reset.add_argument("version", nargs="?", help="one version (default: all)")
    p_verify = sub.add_parser("verify-tag", help="verify one tag exactly as the updater would "
                              "(for the release signer, before pushing it)")
    p_verify.add_argument("tag")
    p_verify.add_argument("--git-dir", help="default: this checkout's repository")
    p_verify.add_argument("--signers", default="deploy/keys/allowed_signers")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)

    if args.cmd == "verify-tag":
        git_dir = args.git_dir or subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True).stdout.strip()
        git = Git(git_dir)
        try:
            tag = verify_release_tag(git, args.tag, args.signers)
        except VerifyError as e:
            print(f"REJECTED {args.tag}: {e}")
            return 1
        try:
            text = git.run("show", f"{tag.commit}:pyproject.toml").decode("utf-8", "replace")
            m = re.search(r'^version\s*=\s*"([0-9.]+)"', text, re.M)
            pkg = m.group(1) if m else None
        except UpdateError:
            pkg = None
        if pkg != fmt_version(tag.version):
            print(f"REJECTED {args.tag}: the commit's pyproject.toml says version {pkg!r}")
            return 1
        print(f"OK {args.tag}: commit {tag.commit}, signed by {tag.signer}")
        return 0

    try:
        cfg = load_config(args.config)
    except (OSError, UpdateError) as e:
        log.error("config: %s", e)
        return 1
    if args.cmd == "status":
        return _status(cfg)
    if args.cmd == "reset-failed":
        state = load_state(cfg.state_file)
        if args.version:
            state["failed"].pop(fmt_version(parse_plain_version(args.version)), None)
        else:
            state["failed"] = {}
        save_state(cfg.state_file, state)
        log.info("failed versions cleared: %s", args.version or "all")
        return 0
    updater = Updater(cfg)
    if args.cmd == "init":
        try:
            updater.init(parse_plain_version(args.baseline) if args.baseline else None)
        except (OSError, UpdateError, ValueError) as e:
            log.error("init: %s", e)
            return 1
        return 0
    return updater.run_once()


if __name__ == "__main__":
    raise SystemExit(main())
