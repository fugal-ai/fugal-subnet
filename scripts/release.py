#!/usr/bin/env python3
"""One command for a release: every step but signing and pushing.

Dry run by default: each subcommand checks what it can and prints what it
would do. `--apply` makes the local edits a step needs (files in this
checkout, `dstack-cloud prepare` in a local app directory). Nothing here
signs, pushes, uploads or publishes: where a passphrase or a public step is
needed it prints the exact command for the owner to run.

    uv run python scripts/release.py plan                   # the whole sequence, in order
    uv run python scripts/release.py status                 # blocks, schedules, composes
    uv run python scripts/release.py check-block 9600120    # boundary, order, notice
    uv run python scripts/release.py blocks v5=9600120      # a compiled block (validators + compose)
    uv run python scripts/release.py schedule draft --serial 1 --set v5_1=9600480
    uv run python scripts/release.py schedule verify release-out/schedule-1.json --install
    uv run python scripts/release.py build-tag build-v2026.11.27
    uv run python scripts/release.py freeze --digest ghcr.io/fugal-ai/fugal-miner@sha256:<hex> \\
        --tag 2026-11-27 --app-dir finney=<dir> --app-dir test=<dir>
    uv run python scripts/release.py entries --digest <ref> --commit <sha> --base <hex> \\
        --tag image-v2026.11.27 --frozen-tag 2026-11-27
    uv run python scripts/release.py verify-tag image-v2026.11.27 --frozen-tag 2026-11-27
    uv run python scripts/release.py validator-tag 0.9.0
    uv run python scripts/release.py notice schedule --serial 1

A schedulable switch's block (config.SCHEDULE_SWITCHES with None compiled)
moves by a signed schedule (docs/decisions/SIGNED_SCHEDULE.md): no image, no
compose, no approved entry. A compiled block still moves by a release.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BLOCK_SECONDS = 12
FINNEY_BLOCKS_PER_EPOCH = 360
FINNEY_NETUID = 70
STORE_BUCKET = "fugal-reveals"            # reveal_sync.FINNEY_REFERENCE_BUCKET
DISCORD_LIMIT = 2000
DEFAULT_NOTICE_EPOCHS = 20                # about a day: a warning below it, not a refusal
# The owner's key (deploy/keys/README.md). Left unquoted in printed commands
# so the shell expands $HOME.
RELEASE_KEY = "$HOME/.ssh/fugal_release"
IMAGE_RE = re.compile(r"^ghcr\.io/fugal-ai/fugal-miner@sha256:[0-9a-f]{64}$")
DATE_TAG_RE = re.compile(r"^(build|image)-v(\d{4})\.(\d{2})\.(\d{2})([a-z]?)$")
FREEZE_TAG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[a-z]?$")
SEMVER_RE = re.compile(r"^(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})$")
COMPOSE_IMAGE_RE = re.compile(r"^(    image: )ghcr\.io/fugal-ai/fugal-miner@sha256:[0-9a-f]{64}$",
                              re.M)
# The compiled blocks a release may set with this tool, and the setter that
# writes each everywhere it is written (config.py, the finney compose, docs).
SETTERS = {"v4": "set_v4_activation_block.py", "v5": "set_v5_activation_block.py"}
COMPILED = (("v2", "FINNEY_V2_ACTIVATION_BLOCK"), ("v3", "FINNEY_V3_ACTIVATION_BLOCK"),
            ("odds_seat", "FINNEY_ODDS_SEAT_BLOCK"), ("menu_b", "FINNEY_MENU_B_BLOCK"),
            ("v4", "FINNEY_V4_ACTIVATION_BLOCK"), ("v5", "FINNEY_V5_ACTIVATION_BLOCK"),
            ("v5_commit_fix", "FINNEY_V5_COMMIT_FIX_BLOCK"),
            ("schedule_from", "FINNEY_SCHEDULE_FROM_BLOCK"))


class ReleaseError(SystemExit):
    def __init__(self, message: str):
        super().__init__(f"REFUSED: {message}")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


_AUTOUPDATE = []


def autoupdate():
    """The auto-updater's strict tag verifier (parse_tag_object,
    verify_ssh_signature, verify_release_tag): one definition of "verified".
    Loaded once, so its VerifyError is one class."""
    if not _AUTOUPDATE:
        _AUTOUPDATE.append(sys.modules.get("fugal_autoupdate") or _load(
            "fugal_autoupdate", ROOT / "deploy" / "autoupdate" / "fugal_autoupdate.py"))
    return _AUTOUPDATE[0]


def say(*lines: str) -> None:
    for line in lines:
        print(line)


class Raw(str):
    """A command word printed as is (it holds $HOME for the shell)."""


def cmd(*parts) -> str:
    return " ".join(p if isinstance(p, Raw) else shlex.quote(str(p)) for p in parts)


def owner_runs(*commands: str) -> None:
    """Commands that need the release key's passphrase, or are public: printed,
    never run."""
    say("", "OWNER RUNS (this tool never signs, pushes or uploads):")
    for c in commands:
        say(f"    {c}")


# ── chain arithmetic ─────────────────────────────────────────────────────────


def epoch_of(block: int, bpe: int = FINNEY_BLOCKS_PER_EPOCH) -> int:
    return int(block) // int(bpe)


def eta(block: int, head: int | None, now: dt.datetime | None = None) -> str:
    if head is None:
        return "time unknown (pass --head-block)"
    now = now or dt.datetime.now(dt.timezone.utc)
    when = now + dt.timedelta(seconds=(int(block) - int(head)) * BLOCK_SECONDS)
    sign = "in" if block >= head else "PASSED"
    return f"~{when:%Y-%m-%d %H:%M} UTC ({sign} {abs(int(block) - int(head)) * BLOCK_SECONDS / 3600:.1f} h)"


def head_block(args) -> int | None:
    """--head-block, else finney's head read once (None when it cannot be)."""
    if getattr(args, "head_block", None) is not None:
        return int(args.head_block)
    if getattr(args, "offline", False):
        return None
    try:
        import bittensor as bt

        sub = bt.Subtensor(network="finney")
        try:
            return int(sub.get_current_block())
        finally:
            try:
                sub.close()
            except Exception:  # noqa: BLE001 - a read-only connection
                pass
    except Exception as e:  # noqa: BLE001 - reported; the checks that need it say so
        print(f"# could not read finney's head ({type(e).__name__}); pass --head-block",
              file=sys.stderr)
        return None


def check_block(block: int, *, head: int | None, bpe: int = FINNEY_BLOCKS_PER_EPOCH,
                lead_epochs: int = 2, min_notice_epochs: int = DEFAULT_NOTICE_EPOCHS,
                after: dict[str, int] | None = None) -> tuple[list[str], list[str]]:
    """(problems, warnings) for a switch block: an epoch boundary, after the
    named blocks, and at least `lead_epochs` epochs after the first boundary
    a document named now could be seen at (head's epoch + 2: the root writes
    during the next epoch). A notice below `min_notice_epochs` is a warning."""
    problems, warnings = [], []
    if block <= 0 or block % bpe:
        problems.append(f"{block} is not an epoch boundary (a multiple of {bpe})")
    for name, other in sorted((after or {}).items()):
        if other is not None and block <= other:
            problems.append(f"{block} is not after {name} ({other})")
    if head is None:
        warnings.append("the chain head is unknown: the lead and notice are not checked")
    else:
        first_seen = epoch_of(head, bpe) + 2
        floor = (first_seen + lead_epochs) * bpe
        if block < floor:
            problems.append(f"{block} is before block {floor}: a schedule first seen at epoch "
                            f"{first_seen} may set nothing before epoch {first_seen + lead_epochs}"
                            " (SCHEDULE_LEAD_EPOCHS)")
        elif epoch_of(block, bpe) - epoch_of(head, bpe) < min_notice_epochs:
            warnings.append(f"{block} is {epoch_of(block, bpe) - epoch_of(head, bpe)} epochs "
                            f"away, less than the {min_notice_epochs}-epoch notice operators "
                            "expect")
    return problems, warnings


# ── the checkout ─────────────────────────────────────────────────────────────


def compiled_blocks(root: Path) -> dict[str, int | None]:
    text = (root / "fugal_subnet" / "config.py").read_text(encoding="utf-8")
    out = {}
    for short, name in COMPILED:
        m = re.search(rf"^{name}(?:: int \| None)? = (None|\d+)$", text, re.M)
        out[short] = None if m is None or m.group(1) == "None" else int(m.group(1))
    return out


def live_image(root: Path) -> str | None:
    text = (root / "deploy" / "dstack" / "docker-compose.yaml").read_text(encoding="utf-8")
    m = re.search(r"^    image: (\S+)$", text, re.M)
    return m.group(1) if m else None


def newest_freeze(root: Path) -> str | None:
    tags = sorted({m.group(1) for p in (root / "deploy" / "dstack").glob("app-compose.frozen-*")
                   if (m := re.match(r"app-compose\.frozen-(.+)-(?:finney|test)\.json$", p.name))})
    return tags[-1] if tags else None


def shipped_schedules(root: Path, signers: bytes | None = None):
    """[(schedule, path)] the checkout ships, verified, by serial."""
    from fugal_subnet import protocol_schedule as ps

    out = []
    for digest, (payload, sig) in ps.shipped_documents(
            root / "fugal_subnet" / "signed_schedule").items():
        try:
            out.append((ps.verify(payload, sig, signers=signers),
                        root / "fugal_subnet" / "signed_schedule" / f"{digest}.json"))
        except ps.ScheduleRefused as e:
            print(f"# shipped schedule {digest[:16]}... does not verify ({e.code})",
                  file=sys.stderr)
    return sorted(out, key=lambda t: t[0].serial)


def _signers(root: Path, override: str | None) -> bytes | None:
    """The signers a schedule is checked against: the pinned copy the neurons
    read (None: the module's own, checked against its pin), or a file for tests."""
    return Path(override).read_bytes() if override else None


# ── subcommands ──────────────────────────────────────────────────────────────


def do_status(args) -> int:
    root = Path(args.root)
    head = head_block(args)
    say(f"checkout {root}", f"chain head: {head if head is not None else 'unknown'}", "",
        "compiled finney blocks:")
    for short, block in compiled_blocks(root).items():
        say(f"  {short:<15} {block if block is not None else 'None':>10}"
            + (f"  epoch e{epoch_of(block):08d}  {eta(block, head)}" if block else ""))
    from fugal_subnet import config
    say("", f"schedulable switches (config.SCHEDULE_SWITCHES): "
            f"{dict(config.SCHEDULE_SWITCHES) or 'none registered'}")
    scheds = shipped_schedules(root, _signers(root, args.signers))
    say(f"shipped signed schedules: {len(scheds)}")
    for s, path in scheds:
        say(f"  serial {s.serial}  {s.digest[:16]}...  "
            + (", ".join(f"{n}={b}" for n, b in s.activations) or "(empty)"))
    say("", f"live compose image: {live_image(root)}", f"newest freeze: {newest_freeze(root)}")
    return 0


def do_check_block(args) -> int:
    root = Path(args.root)
    head = head_block(args)
    from fugal_subnet import config
    after = {k: v for k, v in compiled_blocks(root).items()
             if k not in ("schedule_from", args.name)} if args.name else {}
    problems, warnings = check_block(args.block, head=head, lead_epochs=config.SCHEDULE_LEAD_EPOCHS,
                                     min_notice_epochs=args.min_notice_epochs, after=after)
    say(f"block {args.block:,}: epoch e{epoch_of(args.block):08d}, {eta(args.block, head)}")
    for w in warnings:
        say(f"WARNING: {w}")
    for p in problems:
        say(f"PROBLEM: {p}")
    return 1 if problems else 0


def do_blocks(args) -> int:
    """Compiled blocks: through the setter scripts, which write config.py, the
    finney compose and the docs together."""
    root = Path(args.root)
    head = head_block(args)
    from fugal_subnet import config
    compiled = compiled_blocks(root)
    failed = False
    plans = []
    for spec in args.assign:
        name, _, value = spec.partition("=")
        if name not in SETTERS:
            raise ReleaseError(f"{name!r} has no setter: compiled blocks this tool sets are "
                               f"{', '.join(sorted(SETTERS))}; a schedulable switch moves by "
                               "`schedule draft`")
        try:
            block = int(value)
        except ValueError:
            raise ReleaseError(f"{spec!r} is not NAME=BLOCK") from None
        earlier = {k: v for k, v in compiled.items()
                   if k not in ("schedule_from", "v5_commit_fix") and k != name
                   and [c[0] for c in COMPILED].index(k) < [c[0] for c in COMPILED].index(name)}
        problems, warnings = check_block(block, head=head, lead_epochs=config.SCHEDULE_LEAD_EPOCHS,
                                         min_notice_epochs=args.min_notice_epochs, after=earlier)
        say(f"{name}: {compiled.get(name)} -> {block:,} (epoch e{epoch_of(block):08d}, "
            f"{eta(block, head)})")
        for w in warnings:
            say(f"  WARNING: {w}")
        for p in problems:
            say(f"  PROBLEM: {p}")
        failed |= bool(problems)
        plans.append((name, block))
    if failed:
        return 1
    for name, block in plans:
        setter = root / "scripts" / SETTERS[name]
        if args.apply:
            subprocess.run([sys.executable, str(setter), str(block)], check=True, cwd=root)
        else:
            say(f"would run: {cmd(sys.executable, setter, block)}")
    say("", "A compiled block is in the image: after this, the image is rebuilt and frozen "
            "(build-tag, freeze, entries). Then:",
        f"    {cmd('uv', 'run', 'python', 'scripts/check_safety_invariants.py', '--release')}")
    return 0


def do_schedule_draft(args) -> int:
    from fugal_subnet import config
    from fugal_subnet import protocol_schedule as ps

    root = Path(args.root)
    head = head_block(args)
    chain = config.FINNEY_GENESIS_HASH if args.chain == "finney" else args.chain
    shipped = shipped_schedules(root, _signers(root, args.signers))
    base = shipped[-1][0] if shipped else ps.EMPTY
    acts = dict(base.activations)
    for spec in args.set or ():
        name, _, value = spec.partition("=")
        try:
            acts[name] = int(value)
        except ValueError:
            raise ReleaseError(f"--set {spec!r} is not NAME=BLOCK") from None
    for name in args.unset or ():
        acts.pop(name, None)
    serial = args.serial if args.serial is not None else base.serial + 1
    if serial <= base.serial:
        raise ReleaseError(f"serial {serial} is not above the newest shipped schedule's "
                           f"({base.serial}): nodes refuse it as a replay")
    payload = ps.canonical_bytes(serial, chain, acts)
    try:
        draft = ps.parse_unsigned(payload)
    except ps.ScheduleRefused as e:
        raise ReleaseError(f"the draft is not a schedule ({e.code}: {e.detail})") from None
    say(f"draft serial {serial} (replaces serial {base.serial}), chain {chain[:18]}...")
    unknown = sorted(set(acts) - set(config.SCHEDULE_SWITCHES))
    if unknown:
        say(f"WARNING: not a schedulable switch of this checkout: {', '.join(unknown)} "
            "(nodes without the switch's code ignore it)")
    problems = []
    for name in sorted(set(acts) | set(base.as_dict())):
        old, new = base.get(name), acts.get(name)
        if old == new:
            continue
        say(f"  {name}: {old} -> {new}" + (f" (epoch e{epoch_of(new):08d}, {eta(new, head)})"
                                            if new else ""))
        compiled = config.SCHEDULE_SWITCHES.get(name)
        if compiled is not None and new != compiled:
            problems.append(f"{name} is compiled at {compiled} by this release: the schedule "
                            "may not move it (ScheduleConflict)")
    if head is None:
        say("WARNING: the chain head is unknown: the lead and notice are not checked")
    else:
        first_seen = epoch_of(head) + 2
        try:
            ps.check_transition(base, ps.Schedule(serial=serial, chain=draft.chain,
                                                  activations=draft.activations),
                                epoch=first_seen, chain=chain,
                                blocks_per_epoch=FINNEY_BLOCKS_PER_EPOCH,
                                lead_epochs=config.SCHEDULE_LEAD_EPOCHS)
        except ps.ScheduleRefused as e:
            problems.append(f"if the root names it by epoch {first_seen}: {e.code}: {e.detail}")
        for name, block in draft.activations:
            if block != base.get(name):
                _, warn = check_block(block, head=head, lead_epochs=config.SCHEDULE_LEAD_EPOCHS,
                                      min_notice_epochs=args.min_notice_epochs)
                for w in warn:
                    say(f"WARNING: {name}: {w}")
    for p in problems:
        say(f"PROBLEM: {p}")
    if problems:
        return 1
    out = Path(args.out_dir) / f"schedule-{serial}.json"
    if args.apply:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        say(f"wrote {out}")
    else:
        say("", payload.decode("ascii").rstrip(), "", f"would write {out} (--apply)")
    owner_runs(cmd("ssh-keygen", "-Y", "sign", "-f", Raw(RELEASE_KEY), "-n",
                   "fugal-list@fugalai.com", out),
               cmd("uv", "run", "python", "scripts/release.py", "schedule", "verify", out,
                   "--install"))
    return 0


def do_schedule_verify(args) -> int:
    from fugal_subnet import config
    from fugal_subnet import protocol_schedule as ps

    root = Path(args.root)
    head = head_block(args)
    path = Path(args.file)
    sig_path = Path(args.sig) if args.sig else path.with_name(path.name + ".sig")
    payload, sig = path.read_bytes(), sig_path.read_bytes()
    try:
        doc = ps.verify(payload, sig, signers=_signers(root, args.signers))
    except ps.ScheduleRefused as e:
        raise ReleaseError(f"{path}: {e.code}: {e.detail}") from None
    chain = config.FINNEY_GENESIS_HASH if args.chain == "finney" else args.chain
    shipped = [s for s, _ in shipped_schedules(root, _signers(root, args.signers))
               if s.digest != doc.digest]
    base = shipped[-1] if shipped else ps.EMPTY
    say(f"OK: serial {doc.serial}, signed by {doc.signer}",
        f"digest {doc.digest}", f"pointer {doc.pointer.field()}")
    problems = []
    if doc.chain != str(chain).lower():
        problems.append(f"the schedule is for chain {doc.chain[:18]}..., not {args.chain}")
    if head is not None:
        try:
            ps.check_transition(base, doc, epoch=epoch_of(head) + 2, chain=doc.chain,
                                blocks_per_epoch=FINNEY_BLOCKS_PER_EPOCH,
                                lead_epochs=config.SCHEDULE_LEAD_EPOCHS)
        except ps.ScheduleRefused as e:
            problems.append(f"against serial {base.serial}, named by epoch "
                            f"{epoch_of(head) + 2}: {e.code}: {e.detail}")
    else:
        say("WARNING: the chain head is unknown: the lead is not checked")
    for p in problems:
        say(f"PROBLEM: {p}")
    if problems:
        return 1
    dest = root / "fugal_subnet" / "signed_schedule" / f"{doc.digest}.json"
    if args.install:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        shutil.copyfile(sig_path, dest.with_name(dest.name + ".sig"))
        say(f"installed {dest.relative_to(root)} and its .sig")
    else:
        say(f"would install {dest.relative_to(root)} and its .sig (--install)")
    key = ps.store_object(args.netuid, doc.digest)
    owner_runs(cmd("gsutil", "-h", "Content-Type:application/json", "cp", path,
                   f"gs://{args.bucket}/{key}"),
               cmd("gsutil", "-h", "Content-Type:text/plain", "cp", sig_path,
                   f"gs://{args.bucket}/{key}.sig"),
               "# then: commit the installed pair (a reviewed PR), or copy it into the root's "
               "<state>.schedule.d/; the root names it once the store serves it back")
    return 0


def do_build_tag(args) -> int:
    m = DATE_TAG_RE.match(args.name)
    if m is None or m.group(1) != "build":
        raise ReleaseError(f"{args.name!r} is not build-vYYYY.MM.DD[a-z]")
    root = Path(args.root)
    commit = _git(root, "rev-parse", args.commit) if _is_repo(root) else args.commit
    say(f"build tag {args.name} on {commit}: .github/workflows/image.yml builds the image and "
        "prints its digest (ghcr.io/fugal-ai/fugal-miner@sha256:...).")
    owner_runs(cmd("git", "tag", args.name, commit), cmd("git", "push", "origin", args.name))
    say("", "Then, with the digest from the workflow run:",
        f"    {cmd('uv', 'run', 'python', 'scripts/release.py', 'freeze', '--digest', '<ref>', '--tag', args.name[len('build-v'):].replace('.', '-'), '--app-dir', 'finney=<dir>', '--app-dir', 'test=<dir>')}")
    return 0


def _is_repo(root: Path) -> bool:
    return (root / ".git").exists()


def _git(root: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True,
                          check=True).stdout.strip()


def _app_dirs(specs) -> dict[str, Path]:
    out = {}
    for spec in specs or ():
        net, _, path = spec.partition("=")
        if net not in ("finney", "test") or not path:
            raise ReleaseError(f"--app-dir {spec!r} is not finney=<dir> or test=<dir>")
        out[net] = Path(path)
    return out


def do_freeze(args) -> int:
    """freeze-v080b.sh as one step: point both live composes at the digest,
    run `dstack-cloud prepare` on each network's app directory, keep the
    frozen app-compose.json, and check that it embeds the live compose."""
    root = Path(args.root)
    if not IMAGE_RE.match(args.digest):
        raise ReleaseError(f"{args.digest!r} is not ghcr.io/fugal-ai/fugal-miner@sha256:<64 hex>")
    if not FREEZE_TAG_RE.match(args.tag):
        raise ReleaseError(f"{args.tag!r} is not a freeze tag (YYYY-MM-DD[a-z])")
    dirs = _app_dirs(args.app_dir)
    copies = _app_dirs(args.copy_from)
    lives = {"finney": root / "deploy" / "dstack" / "docker-compose.yaml",
             "test": root / "deploy" / "dstack" / "docker-compose.test.yaml"}
    missing = sorted(set(lives) - set(dirs))
    if missing:
        raise ReleaseError(f"no --app-dir for {', '.join(missing)}")
    for net, live in lives.items():
        frozen = root / "deploy" / "dstack" / f"app-compose.frozen-{args.tag}-{net}.json"
        if frozen.exists() and not args.overwrite:
            raise ReleaseError(f"{frozen.relative_to(root)} exists (a freeze tag names one "
                               "freeze; pick the next letter, or --overwrite)")
    cli = args.cli or shutil.which("dstack-cloud") or "dstack-cloud"
    steps = []
    for net, live in lives.items():
        text = live.read_text(encoding="utf-8")
        if not COMPOSE_IMAGE_RE.search(text):
            raise ReleaseError(f"{live.relative_to(root)} has no `image: ...@sha256:` line")
        steps.append(("image", net, live))
    for net in lives:
        steps.append(("prepare", net, dirs[net]))
    if not args.apply:
        say(f"would point both live composes at {args.digest}")
        for net in lives:
            if net in copies:
                say(f"would copy the app settings {copies[net]} -> {dirs[net]} (without shared/)")
            say(f"would copy {lives[net].relative_to(root)} -> {dirs[net]}/docker-compose.yaml, "
                f"run `{cli} prepare` there, and keep shared/app-compose.json as "
                f"deploy/dstack/app-compose.frozen-{args.tag}-{net}.json")
        say("", "Dry run: nothing written. --apply to do it (local files only).")
        return 0
    for net, live in lives.items():
        text = live.read_text(encoding="utf-8")
        live.write_text(COMPOSE_IMAGE_RE.sub(lambda m: m.group(1) + args.digest, text),
                        encoding="utf-8")
    hashes = {}
    for net, live in lives.items():
        app = dirs[net]
        if net in copies:
            if app.exists():
                raise ReleaseError(f"{app} exists; --copy-from makes a new app directory")
            shutil.copytree(copies[net], app, symlinks=True,
                            ignore=shutil.ignore_patterns("shared"))
        if not (app / "app.json").is_file():
            raise ReleaseError(f"{app} is not a dstack-cloud app directory (no app.json)")
        shutil.rmtree(app / "shared", ignore_errors=True)
        shutil.copyfile(live, app / "docker-compose.yaml")
        subprocess.run([cli, "prepare"], cwd=app, check=True, stdout=subprocess.DEVNULL)
        frozen = root / "deploy" / "dstack" / f"app-compose.frozen-{args.tag}-{net}.json"
        shutil.copyfile(app / "shared" / "app-compose.json", frozen)
        raw = frozen.read_bytes()
        embedded = json.loads(raw.decode("utf-8")).get("docker_compose_file", "")
        if embedded != live.read_text(encoding="utf-8"):
            raise ReleaseError(f"{frozen.name} does not embed {live.name} byte for byte")
        if f"image: {args.digest}" not in embedded:
            raise ReleaseError(f"{frozen.name} does not name {args.digest}")
        hashes[net] = hashlib.sha256(raw).hexdigest()
        say(f"frozen {frozen.relative_to(root)}  sha256 {hashes[net]}")
    say("", "Next (local, no spend):",
        f"    {cmd('uv', 'run', 'python', 'scripts/check_safety_invariants.py', '--release')}",
        "    update tests that name the newest frozen finney compose (tests/test_validator_setup.py)",
        "    commit the composes and the frozen files (a reviewed PR)",
        f"    {cmd('uv', 'run', 'python', 'scripts/release.py', 'entries', '--digest', args.digest, '--commit', '<merged sha>', '--base', '<base measurement>', '--tag', 'image-v' + args.tag.replace('-', '.'), '--frozen-tag', args.tag)}")
    return 0


def do_entries(args) -> int:
    """The signed tag message and the release note (publish_approved_entry),
    and the commands to sign and publish them."""
    root = Path(args.root)
    m = DATE_TAG_RE.match(args.tag)
    if m is None or m.group(1) != "image":
        raise ReleaseError(f"{args.tag!r} is not image-vYYYY.MM.DD[a-z]")
    if not IMAGE_RE.match(args.digest):
        raise ReleaseError(f"{args.digest!r} is not a digest reference")
    if not re.fullmatch(r"[0-9a-f]{64}", args.base or ""):
        raise ReleaseError("--base is the 64-hex base measurement (provision_td.py --inspect)")
    if not re.fullmatch(r"[0-9a-f]{7,64}", args.commit):
        raise ReleaseError(f"--commit {args.commit!r} is not a commit id")
    pae = _load("publish_approved_entry", ROOT / "scripts" / "publish_approved_entry.py")
    files, entries = {}, {}
    for net in ("finney", "test"):
        p = root / "deploy" / "dstack" / f"app-compose.frozen-{args.frozen_tag}-{net}.json"
        if not p.is_file():
            raise ReleaseError(f"{p.relative_to(root)} is missing: freeze first")
        if args.digest not in json.loads(p.read_text(encoding="utf-8")).get(
                "docker_compose_file", ""):
            raise ReleaseError(f"{p.name} does not name {args.digest}")
        files[net] = str(p)
        entries[net] = f"{args.base}:{hashlib.sha256(p.read_bytes()).hexdigest()}"
    message = pae.tag_message(args.tag, entries, args.digest, args.commit)
    note = pae.release_note(args.tag, entries, files, args.base, args.digest, args.commit)
    out = Path(args.out_dir)
    msg_path, note_path = out / f"{args.tag}.tag-message.txt", out / f"{args.tag}.notes.md"
    for net in ("finney", "test"):
        say(f"approved-entry {net}: {entries[net]}")
    if args.apply:
        out.mkdir(parents=True, exist_ok=True)
        msg_path.write_text(message, encoding="utf-8")
        note_path.write_text(note, encoding="utf-8")
        say(f"wrote {msg_path} and {note_path}")
    else:
        say("", message.rstrip(), "", f"would write {msg_path} and {note_path} (--apply)")
    owner_runs(cmd("git", "-c", "gpg.format=ssh", "-c", Raw(f"user.signingkey={RELEASE_KEY}.pub"),
                   "tag", "-s", args.tag, "-F", msg_path, args.commit),
               cmd("uv", "run", "python", "scripts/release.py", "verify-tag", args.tag,
                   "--frozen-tag", args.frozen_tag, "--base", args.base),
               cmd("git", "push", "origin", args.tag),
               cmd("gh", "release", "create", args.tag, "-R", "fugal-ai/fugal-subnet",
                   "--notes-file", note_path, "--verify-tag"))
    return 0


ENTRY_LINE_RE = re.compile(r"^approved-entry (finney|test): ([0-9a-f]{64}):([0-9a-f]{64})$", re.M)


def verify_image_tag(git, name: str, signers_file: str) -> dict:
    """An image-v* tag verified as strictly as the auto-updater verifies a
    validator tag: annotated, exact headers, its own name inside, a commit,
    one SSH signature and nothing after it, by a key in the signers file
    (namespace git). Returns {commit, signer, entries {network: (base, hash)}},
    the entries read from the SIGNED message only. VerifyError otherwise."""
    au = autoupdate()
    m = DATE_TAG_RE.match(name)
    if m is None or m.group(1) != "image":
        raise au.VerifyError(f"{name!r} is not image-vYYYY.MM.DD[a-z]")
    ref = f"refs/tags/{name}"
    kind = git.object_type(ref)
    if kind != "tag":
        raise au.VerifyError(f"{name} is not an annotated tag (it is {kind or 'missing'})")
    obj = au.parse_tag_object(git.run("cat-file", "tag", ref))
    if obj.tag != name:
        raise au.VerifyError(f"{name} holds a tag object named {obj.tag!r} (a renamed tag)")
    if obj.type != "commit":
        raise au.VerifyError(f"{name} points at a {obj.type}, not a commit")
    signer = au.verify_ssh_signature(obj.payload, obj.signature, signers_file)
    message = obj.payload.partition(b"\n\n")[2].decode("utf-8", "replace")
    entries: dict[str, tuple[str, str]] = {}
    for net, base, h in ENTRY_LINE_RE.findall(message):
        if net in entries and entries[net] != (base, h):
            raise au.VerifyError(f"{name} signs two entries for {net}")
        entries[net] = (base, h)
    if not entries:
        raise au.VerifyError(f"{name}'s signed message carries no approved-entry line")
    return {"commit": obj.object_id, "signer": signer, "entries": entries}


def do_verify_tag(args) -> int:
    root = Path(args.root)
    au = autoupdate()
    git_dir = args.git_dir or _git(root, "rev-parse", "--path-format=absolute",
                                   "--git-common-dir")
    git = au.Git(git_dir)
    signers = args.signers or str(root / "deploy" / "keys" / "allowed_signers")
    try:
        if args.tag.startswith("validator-v"):
            tag = au.verify_release_tag(git, args.tag, signers)
            text = git.run("show", f"{tag.commit}:pyproject.toml").decode("utf-8", "replace")
            pm = re.search(r'^version\s*=\s*"([0-9.]+)"', text, re.M)
            if (pm.group(1) if pm else None) != au.fmt_version(tag.version):
                raise au.VerifyError(f"the commit's pyproject.toml says version "
                                     f"{pm.group(1) if pm else None!r}")
            say(f"OK {args.tag}: commit {tag.commit}, signed by {tag.signer}")
            return 0
        got = verify_image_tag(git, args.tag, signers)
    except au.VerifyError as e:
        say(f"REJECTED {args.tag}: {e}")
        return 1
    bad = False
    for net, (base, h) in sorted(got["entries"].items()):
        line = f"approved-entry {net}: {base}:{h}"
        if args.frozen_tag:
            p = root / "deploy" / "dstack" / f"app-compose.frozen-{args.frozen_tag}-{net}.json"
            want = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
            if want != h:
                line += f"  MISMATCH: {p.name} hashes to {want}"
                bad = True
        if args.base and base != args.base:
            line += f"  MISMATCH: base is not {args.base}"
            bad = True
        say(line)
    if bad:
        say(f"REJECTED {args.tag}: the signed entries are not this checkout's freeze")
        return 1
    say(f"OK {args.tag}: commit {got['commit']}, signed by {got['signer']}")
    return 0


def do_validator_tag(args) -> int:
    root = Path(args.root)
    if not SEMVER_RE.match(args.version):
        raise ReleaseError(f"{args.version!r} is not MAJOR.MINOR.PATCH")
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([0-9.]+)"', text, re.M)
    if (m.group(1) if m else None) != args.version:
        raise ReleaseError(f"pyproject.toml says version {m.group(1) if m else None!r}, not "
                           f"{args.version}: the auto-updater refuses a tag whose commit says "
                           "another version")
    tag = f"validator-v{args.version}"
    commit = _git(root, "rev-parse", args.commit) if _is_repo(root) else args.commit
    say(f"{tag} on {commit} (auto-updating validators install it in their idle window)")
    owner_runs(cmd("git", "-c", "gpg.format=ssh", "-c", Raw(f"user.signingkey={RELEASE_KEY}.pub"),
                   "tag", "-s", tag, "-m", f"validator v{args.version}", commit),
               cmd("uv", "run", "python", "scripts/release.py", "verify-tag", tag),
               cmd("git", "push", "origin", tag))
    return 0


def notice_text(kind: str, *, version: str | None = None, deadline_block: int | None = None,
                head: int | None = None, image_tag: str | None = None,
                entry: str | None = None, digest: str | None = None,
                schedule=None, summary: str | None = None) -> str:
    """A Discord-sized announcement (under 2000 characters)."""
    when = (f"block {deadline_block:,} (e{epoch_of(deadline_block):08d}, "
            f"{eta(deadline_block, head)})" if deadline_block else None)
    lines = []
    if kind == "validator":
        lines += [f"**Fugal v{version} (validators only)**" + (f": {summary}" if summary else ""),
                  "Auto-update installs it in your idle window. Otherwise: `git pull && uv sync "
                  "--locked --extra tee`, run `neurons/validator.py --check`, restart.",
                  "Miners: nothing to do."]
        if when:
            lines.append(f"Upgrade before {when}.")
    elif kind == "image":
        lines += [f"**Fugal miner image {image_tag}**" + (f": {summary}" if summary else ""),
                  f"Image: `{digest}`" if digest else "",
                  f"Approved entry (finney): `{entry}`" if entry else "",
                  f"Validators: `python scripts/fetch_approved_entries.py --tag {image_tag} "
                  "--network finney`, add it to FUGAL_TEE_MEASUREMENTS beside the current "
                  "entry, restart.",
                  "Miners: redeploy your TD from `deploy/dstack/docker-compose.yaml` and "
                  "re-provision."]
        if when:
            lines.append(f"Switch at {when}; keep the old entry until then.")
    elif kind == "schedule":
        acts = ", ".join(f"`{n}` at block {b:,} (e{epoch_of(b):08d}, {eta(b, head)})"
                         for n, b in schedule.activations) or "nothing scheduled"
        lines += [f"**Fugal protocol schedule, serial {schedule.serial}**",
                  f"Scheduled: {acts}.",
                  "No image, compose or entry changes: validators and miners' TDs read it "
                  "from the root's on-chain pointer and verify the signature themselves. "
                  "Nothing to do on a release that reads the schedule.",
                  f"Pointer: `{schedule.pointer.field()}`"]
        if summary:
            lines.insert(1, summary)
    else:
        raise ValueError(kind)
    text = "\n".join(line for line in lines if line)
    if len(text) > DISCORD_LIMIT:
        raise ReleaseError(f"the notice is {len(text)} characters, over Discord's {DISCORD_LIMIT}")
    return text


def do_notice(args) -> int:
    root = Path(args.root)
    head = head_block(args) if args.deadline_block or args.kind == "schedule" else None
    schedule = None
    if args.kind == "schedule":
        scheds = shipped_schedules(root, _signers(root, args.signers))
        pick = [s for s, _ in scheds if args.serial is None or s.serial == args.serial]
        if not pick:
            raise ReleaseError("no shipped schedule with that serial (schedule verify --install)")
        schedule = pick[-1]
    if args.kind == "validator" and not args.version:
        raise ReleaseError("--version is needed for a validator notice")
    say(notice_text(args.kind, version=args.version, deadline_block=args.deadline_block,
                    head=head, image_tag=args.image_tag, entry=args.entry, digest=args.digest,
                    schedule=schedule, summary=args.summary))
    return 0


PLAN = """\
A release, in order. Dry runs are the default; `--apply` makes local edits only.

 1. Blocks.   A schedulable switch (config.SCHEDULE_SWITCHES, compiled None): skip to 9.
              A compiled block:   release.py blocks v5=<block> --apply
              Any block:          release.py check-block <block>
 2. Checks.   uv run ruff check . && uv run pytest -q
              uv run python scripts/check_safety_invariants.py --release   (and --release-v5)
              uv run python scripts/check_determinism.py --perturb
 3. Merge the release PR (owner).
 4. Image (only when the miner image changes):
              release.py build-tag build-v<date>          -> OWNER pushes the tag; CI builds
 5.           release.py freeze --digest <ref> --tag <date> --app-dir finney=<dir> \\
                  --app-dir test=<dir> --apply            -> commit the freeze (PR, merge)
 6.           release.py entries --digest <ref> --commit <sha> --base <hex> \\
                  --tag image-v<date> --frozen-tag <date> --apply
                                                          -> OWNER signs, pushes, gh release
 7.           release.py verify-tag image-v<date> --frozen-tag <date> --base <hex>
 8. Validators: release.py validator-tag <X.Y.Z>          -> OWNER signs, verifies, pushes
 9. Schedule: release.py schedule draft --set <name>=<block> --apply
                                                          -> OWNER signs (ssh-keygen -Y sign)
              release.py schedule verify <file> --install -> OWNER uploads; PR the pair
10. Notice:   release.py notice {validator|image|schedule} ...
"""


def do_plan(args) -> int:
    say(PLAN.rstrip())
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--root", default=str(ROOT), help="the checkout (default: this one)")
    ap.add_argument("--head-block", type=int, help="finney's head block (default: read it)")
    ap.add_argument("--offline", action="store_true", help="never read the chain")
    ap.add_argument("--signers", help="an allowed-signers file to check schedules against "
                                      "(default: the pinned copy the neurons use)")
    ap.add_argument("--min-notice-epochs", type=int, default=DEFAULT_NOTICE_EPOCHS)
    ap.add_argument("--out-dir", default=str(ROOT / "release-out"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(fn=do_plan)
    sub.add_parser("status").set_defaults(fn=do_status)
    p = sub.add_parser("check-block")
    p.add_argument("block", type=int)
    p.add_argument("--name", help="the switch (checked to be after every earlier compiled block)")
    p.set_defaults(fn=do_check_block)
    p = sub.add_parser("blocks")
    p.add_argument("assign", nargs="+", metavar="NAME=BLOCK")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(fn=do_blocks)
    sched = sub.add_parser("schedule").add_subparsers(dest="schedule_cmd", required=True)
    p = sched.add_parser("draft")
    p.add_argument("--serial", type=int)
    p.add_argument("--set", action="append", metavar="NAME=BLOCK")
    p.add_argument("--unset", action="append", metavar="NAME")
    p.add_argument("--chain", default="finney", help="finney, or a genesis hash (0x...)")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(fn=do_schedule_draft)
    p = sched.add_parser("verify")
    p.add_argument("file")
    p.add_argument("--sig")
    p.add_argument("--chain", default="finney")
    p.add_argument("--install", action="store_true")
    p.add_argument("--bucket", default=STORE_BUCKET)
    p.add_argument("--netuid", type=int, default=FINNEY_NETUID)
    p.set_defaults(fn=do_schedule_verify)
    p = sub.add_parser("build-tag")
    p.add_argument("name")
    p.add_argument("--commit", default="HEAD")
    p.set_defaults(fn=do_build_tag)
    p = sub.add_parser("freeze")
    p.add_argument("--digest", required=True)
    p.add_argument("--tag", required=True, help="the freeze tag, YYYY-MM-DD[a-z]")
    p.add_argument("--app-dir", action="append", metavar="NET=DIR")
    p.add_argument("--copy-from", action="append", metavar="NET=DIR",
                   help="make --app-dir a copy of this app directory (settings only)")
    p.add_argument("--cli", help="the dstack-cloud CLI (meta-dstack-cloud/scripts/bin)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(fn=do_freeze)
    p = sub.add_parser("entries")
    p.add_argument("--digest", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--tag", required=True, help="image-vYYYY.MM.DD[a-z]")
    p.add_argument("--frozen-tag", required=True, help="the freeze tag, YYYY-MM-DD[a-z]")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(fn=do_entries)
    p = sub.add_parser("verify-tag")
    p.add_argument("tag")
    p.add_argument("--git-dir")
    p.add_argument("--frozen-tag")
    p.add_argument("--base")
    p.set_defaults(fn=do_verify_tag)
    p = sub.add_parser("validator-tag")
    p.add_argument("version")
    p.add_argument("--commit", default="HEAD")
    p.set_defaults(fn=do_validator_tag)
    p = sub.add_parser("notice")
    p.add_argument("kind", choices=("validator", "image", "schedule"))
    p.add_argument("--version")
    p.add_argument("--deadline-block", type=int)
    p.add_argument("--image-tag")
    p.add_argument("--entry")
    p.add_argument("--digest")
    p.add_argument("--serial", type=int)
    p.add_argument("--summary")
    p.set_defaults(fn=do_notice)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
