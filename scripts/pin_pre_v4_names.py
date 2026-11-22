"""Writes fugal_subnet/consensus_v4_pre.json: for every name consensus_v4.PRE_V4
lists, a hash of the statements that bind it in a reference release (default
origin/main, the v3 release). A name the release does not have is left out, and
the registry check then reports it.

Run: uv run python scripts/pin_pre_v4_names.py [REF]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fugal_subnet import consensus_v4 as cv4  # noqa: E402


def release_pin(ref: str = "origin/main") -> dict[str, dict[str, str]]:
    """{module: {name: hash}} of the PRE_V4 names, read from the reference's files."""
    out: dict[str, dict[str, str]] = {}
    for module, names in sorted(cv4.PRE_V4.items()):
        rel = module.replace(".", "/") + ".py"
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:{rel}"],
                               capture_output=True, text=True)
        if shown.returncode != 0:
            out[module] = {}
            continue
        hashes = cv4.binding_hashes(cv4.source_bindings(shown.stdout))
        out[module] = {n: h for n, h in hashes.items() if n in names}
    return out


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", ref], capture_output=True,
                            text=True, check=True).stdout.strip()
    pin = {"release": commit, "modules": release_pin(ref)}
    (ROOT / "fugal_subnet" / cv4.PRE_V4_PIN).write_text(
        json.dumps(pin, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{sum(len(v) for v in pin['modules'].values())} names pinned from {ref} ({commit[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
