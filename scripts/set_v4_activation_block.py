"""Set finney's protocol v4 activation block everywhere it is written, and
nowhere else.

    uv run python scripts/set_v4_activation_block.py 9572760     # set it
    uv run python scripts/set_v4_activation_block.py --check     # show where it is, check they agree

The block is written in exactly these places:

- fugal_subnet/config.py: `FINNEY_V4_ACTIVATION_BLOCK` (what validators read);
- deploy/dstack/docker-compose.yaml: `FUGAL_V4_ACTIVATION_BLOCK` (what the
  miner reads; check_safety_invariants requires the two to be equal);
- the release notes and guides in DOCS, as the block (9572760 and
  9,572,760) and as its epoch (e.g. 26591 / e00026591).

The block must be an epoch boundary at finney's 360-block tempo and after
every earlier compiled finney block. The frozen app-compose files are NOT
written here: they come from `dstack-cloud prepare` on the live compose after
the image is built, so changing the block after the image build means a new
image and a new freeze. Run the safety check afterwards.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "fugal_subnet" / "config.py"
COMPOSE = ROOT / "deploy" / "dstack" / "docker-compose.yaml"
DOCS = (
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "MINER_GUIDE.md",
    ROOT / "docs" / "VALIDATOR_GUIDE.md",
)
FINNEY_BLOCKS_PER_EPOCH = 360
EARLIER = ("FINNEY_V2_ACTIVATION_BLOCK", "FINNEY_V3_ACTIVATION_BLOCK", "FINNEY_ODDS_SEAT_BLOCK",
           "FINNEY_MENU_B_BLOCK")

_CONFIG_LINE = re.compile(r"^FINNEY_V4_ACTIVATION_BLOCK: int \| None = (None|\d+)$", re.M)
_COMPOSE_LINE = re.compile(r"^(\s*- FUGAL_V4_ACTIVATION_BLOCK=)(\d+)$", re.M)


def _compiled(name: str, text: str) -> int | None:
    m = re.search(rf"^{name}: int \| None = (None|\d+)$", text, re.M)
    if not m:
        raise SystemExit(f"config.py: no `{name}: int | None = ...` line")
    return None if m.group(1) == "None" else int(m.group(1))


def forms(block: int) -> list[str]:
    """Every way the docs write a block: the number, with thousands commas,
    and its epoch (e00026591 before 26591, so the longer form goes first)."""
    epoch = block // FINNEY_BLOCKS_PER_EPOCH
    return [f"{block:,}", str(block), f"e{epoch:08d}", f"{epoch:,}", str(epoch)]


def _replace(text: str, old: int, new: int) -> tuple[str, int]:
    count = 0
    for a, b in zip(forms(old), forms(new)):
        text, n = re.subn(rf"(?<![\d,]){re.escape(a)}(?![\d,])", b, text)
        count += n
    return text, count


def check() -> int:
    config_text = CONFIG.read_text(encoding="utf-8")
    compiled = _compiled("FINNEY_V4_ACTIVATION_BLOCK", config_text)
    m = _COMPOSE_LINE.search(COMPOSE.read_text(encoding="utf-8"))
    compose = int(m.group(2)) if m else None
    print(f"config.py               FINNEY_V4_ACTIVATION_BLOCK = {compiled}")
    print(f"docker-compose.yaml     FUGAL_V4_ACTIVATION_BLOCK  = {compose}")
    bad = compiled != compose
    if compiled is not None:
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            hits = {f: len(re.findall(rf"(?<![\d,]){re.escape(f)}(?![\d,])", text))
                    for f in forms(compiled)}
            print(f"{path.relative_to(ROOT)!s:<24}" + ", ".join(f"{f} x{n}" for f, n in hits.items()))
        print(f"epoch e{compiled // FINNEY_BLOCKS_PER_EPOCH:08d}; boundary: "
              f"{compiled % FINNEY_BLOCKS_PER_EPOCH == 0}")
    if bad:
        print("DISAGREE: the compiled block and the finney compose differ")
    return 1 if bad else 0


def set_block(new: int) -> None:
    config_text = CONFIG.read_text(encoding="utf-8")
    if new <= 0 or new % FINNEY_BLOCKS_PER_EPOCH:
        raise SystemExit(f"{new} is not an epoch boundary (a multiple of {FINNEY_BLOCKS_PER_EPOCH})")
    for name in EARLIER:
        earlier = _compiled(name, config_text)
        if earlier is not None and new <= earlier:
            raise SystemExit(f"{new} is not after {name}={earlier}")
    old = _compiled("FINNEY_V4_ACTIVATION_BLOCK", config_text)
    config_text = _CONFIG_LINE.sub(f"FINNEY_V4_ACTIVATION_BLOCK: int | None = {new}", config_text)
    compose_text = COMPOSE.read_text(encoding="utf-8")
    if not _COMPOSE_LINE.search(compose_text):
        raise SystemExit("deploy/dstack/docker-compose.yaml has no `- FUGAL_V4_ACTIVATION_BLOCK=<n>` "
                         "line to set; add it under FUGAL_V3_ACTIVATION_BLOCK first")
    compose_text = _COMPOSE_LINE.sub(lambda m: f"{m.group(1)}{new}", compose_text)
    CONFIG.write_text(config_text, encoding="utf-8")
    COMPOSE.write_text(compose_text, encoding="utf-8")
    print(f"config.py and docker-compose.yaml: {old} -> {new}")
    if old is not None and old != new:
        for path in DOCS:
            text, n = _replace(path.read_text(encoding="utf-8"), old, new)
            path.write_text(text, encoding="utf-8")
            print(f"{path.relative_to(ROOT)}: {n} replacement(s)")
    print("Next: uv run python scripts/check_safety_invariants.py; the frozen composes come from "
          "dstack-cloud prepare after the image build.")


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
