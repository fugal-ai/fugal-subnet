"""Set finney's protocol v5 activation block everywhere it is written, and
nowhere else.

    uv run python scripts/set_v5_activation_block.py 9600120     # set it
    uv run python scripts/set_v5_activation_block.py --check     # show where it is, check they agree

The block is written in exactly these places:

- fugal_subnet/config.py: `FINNEY_V5_ACTIVATION_BLOCK` (what validators read);
- deploy/dstack/docker-compose.yaml: `FUGAL_V5_ACTIVATION_BLOCK` (what the
  miner's compose carries; check_safety_invariants requires the two to be
  equal). The line is added under FUGAL_V4_ACTIVATION_BLOCK the first time;
- the release notes and guides in DOCS: before the first set they say
  `TBD-block` and `TBD-epoch` where the block and its epoch go; afterwards the
  block (9600120 and 9,600,120) and its epoch (26667 / e00026667) are moved
  as set_v4_activation_block moves v4's.

The block must be an epoch boundary at finney's 360-block tempo and after
every earlier compiled finney block (v4's included). The frozen app-compose
files are NOT written here: they come from `dstack-cloud prepare` on the live
compose after the image is built, so changing the block after the image
build means a new image and a new freeze. Run the safety check afterwards.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4_SCRIPT = Path(__file__).resolve().parent / "set_v4_activation_block.py"
CONFIG = ROOT / "fugal_subnet" / "config.py"
COMPOSE = ROOT / "deploy" / "dstack" / "docker-compose.yaml"
DOCS = (
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "MINER_GUIDE.md",
    ROOT / "docs" / "VALIDATOR_GUIDE.md",
)
FINNEY_BLOCKS_PER_EPOCH = 360
EARLIER = ("FINNEY_V2_ACTIVATION_BLOCK", "FINNEY_V3_ACTIVATION_BLOCK", "FINNEY_ODDS_SEAT_BLOCK",
           "FINNEY_MENU_B_BLOCK", "FINNEY_V4_ACTIVATION_BLOCK")
BLOCK_MARK, EPOCH_MARK = "TBD-block", "TBD-epoch"

_CONFIG_LINE = re.compile(r"^FINNEY_V5_ACTIVATION_BLOCK: int \| None = (None|\d+)$", re.M)
_COMPOSE_LINE = re.compile(r"^(\s*- FUGAL_V5_ACTIVATION_BLOCK=)(\d+)$", re.M)
_COMPOSE_V4 = re.compile(r"^(\s*)- FUGAL_V4_ACTIVATION_BLOCK=\d+$", re.M)


def _v4():
    """set_v4_activation_block's helpers (one definition of the doc forms)."""
    spec = importlib.util.spec_from_file_location(
        "set_v4_activation_block", V4_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compiled(name: str, text: str) -> int | None:
    return _v4()._compiled(name, text)


def check() -> int:
    v4 = _v4()
    config_text = CONFIG.read_text(encoding="utf-8")
    compiled = _compiled("FINNEY_V5_ACTIVATION_BLOCK", config_text)
    m = _COMPOSE_LINE.search(COMPOSE.read_text(encoding="utf-8"))
    compose = int(m.group(2)) if m else None
    print(f"config.py               FINNEY_V5_ACTIVATION_BLOCK = {compiled}")
    print(f"docker-compose.yaml     FUGAL_V5_ACTIVATION_BLOCK  = {compose}")
    bad = compiled != compose
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        marks = f"{BLOCK_MARK} x{text.count(BLOCK_MARK)}, {EPOCH_MARK} x{text.count(EPOCH_MARK)}"
        if compiled is not None:
            marks += ", " + ", ".join(f"{f} x{len(re.findall(v4._bounded(f), text))}"
                                      for f in v4.forms(compiled))
        print(f"{path.relative_to(ROOT)!s:<24}{marks}")
    if compiled is not None:
        print(f"epoch e{compiled // FINNEY_BLOCKS_PER_EPOCH:08d}; boundary: "
              f"{compiled % FINNEY_BLOCKS_PER_EPOCH == 0}")
    if bad:
        print("DISAGREE: the compiled block and the finney compose differ")
    return 1 if bad else 0


def set_block(new: int) -> None:
    v4 = _v4()
    config_text = CONFIG.read_text(encoding="utf-8")
    if new <= 0 or new % FINNEY_BLOCKS_PER_EPOCH:
        raise SystemExit(f"{new} is not an epoch boundary (a multiple of {FINNEY_BLOCKS_PER_EPOCH})")
    for name in EARLIER:
        earlier = _compiled(name, config_text)
        if earlier is not None and new <= earlier:
            raise SystemExit(f"{new} is not after {name}={earlier}")
    old = _compiled("FINNEY_V5_ACTIVATION_BLOCK", config_text)
    if not _CONFIG_LINE.search(config_text):
        raise SystemExit("config.py has no `FINNEY_V5_ACTIVATION_BLOCK: int | None = ...` line")
    config_text = _CONFIG_LINE.sub(f"FINNEY_V5_ACTIVATION_BLOCK: int | None = {new}", config_text)
    compose_text = COMPOSE.read_text(encoding="utf-8")
    if _COMPOSE_LINE.search(compose_text):
        compose_text = _COMPOSE_LINE.sub(lambda m: f"{m.group(1)}{new}", compose_text)
    else:
        m = _COMPOSE_V4.search(compose_text)
        if not m:
            raise SystemExit("deploy/dstack/docker-compose.yaml has no FUGAL_V4_ACTIVATION_BLOCK "
                             "line to put FUGAL_V5_ACTIVATION_BLOCK under")
        ind = m.group(1)
        line = (f"\n{ind}# Protocol v5 switches on at this block (an epoch boundary). The miner\n"
                f"{ind}# reads it only from here; validators have it compiled in, and\n"
                f"{ind}# check_safety_invariants requires the two to be equal.\n"
                f"{ind}- FUGAL_V5_ACTIVATION_BLOCK={new}")
        compose_text = compose_text[:m.end()] + line + compose_text[m.end():]
    CONFIG.write_text(config_text, encoding="utf-8")
    COMPOSE.write_text(compose_text, encoding="utf-8")
    print(f"config.py and docker-compose.yaml: {old} -> {new}")
    epoch = new // FINNEY_BLOCKS_PER_EPOCH
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        n = text.count(BLOCK_MARK) + text.count(EPOCH_MARK)
        text = text.replace(BLOCK_MARK, f"{new:,}").replace(EPOCH_MARK, f"e{epoch:08d}")
        if old is not None and old != new:
            text, m = v4._replace(text, old, new)
            n += m
        path.write_text(text, encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {n} replacement(s)")
    print("Next: uv run python scripts/check_safety_invariants.py --release-v5; the frozen "
          "composes come from dstack-cloud prepare after the image build.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("block", nargs="?", type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check or args.block is None:
        return check()
    set_block(args.block)
    return check()


if __name__ == "__main__":
    sys.exit(main())
