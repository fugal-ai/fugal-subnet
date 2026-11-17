"""Validator upgrades are "update + restart": what used to be manual per release.

Operational, not consensus: nothing here enters the consensus digest or any
score. It decides which miner images this validator accepts (the same set
every validator on the same release computes from the same checkout), where it
reads the chain, and fetches a hash-pinned file. Every release before this one
asked each operator to paste an approved entry, fetch the embeddings file and
set an archive endpoint by hand, and third-party validators got one of the
three wrong on every release (memory: validator upgrade friction, 2026-11-17).

1. Approved entries come from the code. An entry is `<base>:<compose hash>`,
   and the compose hash is sha256 of a frozen app-compose committed in this
   checkout (deploy/dstack/app-compose.frozen-<tag>-<network>.json). The frozen
   compose lands on `main` after the image is built (a compose cannot carry its
   own hash), so a validator that updates its checkout carries the entry for
   every released image. The base measurements are pinned below: they move
   only when the cloud's TD firmware does, which has happened once. A compose
   listed in SUPERSEDED is never approved from the code. FUGAL_TEE_MEASUREMENTS
   still adds entries (union); FUGAL_APPROVED_FROM_CODE=0 uses it alone.

   Trust: the same as the code. A frozen compose reaches `main` through a
   reviewed PR, like every line the validator runs; nothing is fetched.

2. The pinned embeddings file is fetched at startup when it is missing or
   wrong, and verified against the pin before it replaces anything.

3. On finney, an unset archive / fallback endpoint defaults to the public
   OpenTensor ones, so a restart mid-epoch can still read the boundary block.

4. One READY line at the end of startup names every item, or NOT READY with
   the one command that fixes each missing item.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import os
import re

logger = logging.getLogger("fugal.validator")

# The TD base measurement (MRTD + RTMR1 + RTMR2 of the dstack image miners boot).
# Measured on GCP TDX: 2026-11-03, unchanged on the v3 testnet TD 2026-11-17.
# Changes only with the cloud's TD firmware; a new one is added here in a
# release (keep the old one until no miner runs on it).
PINNED_BASE_MEASUREMENTS: dict[str, tuple[str, ...]] = {
    "finney": ("f4be9383e31c6cc87707b2b01ace9114ed958435b524425677e63d65065aa427",),
    "test": ("f4be9383e31c6cc87707b2b01ace9114ed958435b524425677e63d65065aa427",),
}

# Frozen composes committed in the repo that must never be approved from the
# code: compose hash -> why. The 2026-11-17 pair froze image fc6015e8 without
# FUGAL_V3_ACTIVATION_BLOCK, so a miner on it would have served v2 after the v3
# flip; image-v2026.11.17b (the 2026-11-17b composes) replaced them.
SUPERSEDED_COMPOSES: dict[str, str] = {
    "8c260604f11564869dd9121a189303abc12039a5ebbce98d163e26c77d0ee7ec":
        "2026-11-17 finney compose: no FUGAL_V3_ACTIVATION_BLOCK (use 2026-11-17b)",
    "95700330f06d05a30b49e4c621edacef6467d80d7f2162bda92cbdb89acb192d":
        "2026-11-17 test compose: superseded by 2026-11-17b with the finney twin",
    # Retired protocols: their images cannot produce a valid proof for the
    # protocol in force (v3 from finney block 9,527,760), so approving them adds
    # nothing; leaving them out keeps the approved set to images that can score.
    "9ab870cfe60d0c61f5764966a177d9146276979e65df0dbaa8370dcd3a75f166":
        "2026-11-03c finney compose: protocol v1 image, retired",
    "c9b5c43b57d540d0e8a321efb673b6c42d1effd011dd716934f5df2a6ff0dad8":
        "2026-11-03c test compose: protocol v1 image, retired",
    "5408259ffb5e1971f7552fdef7011415737651eb5aa737996b222dd638c7e796":
        "2026-11-10b finney compose: protocol v2 image, retired at the v3 flip",
    "ba6eb6f254dd6b4ef7f3f9860af1aeba00061b96f4a4d7260700f041eab5d458":
        "2026-11-10b test compose: protocol v2 image, retired at the v3 flip",
}

FINNEY_DEFAULT_ARCHIVE = "wss://archive.chain.opentensor.ai:443"
FINNEY_DEFAULT_FALLBACK = "wss://entrypoint-finney.opentensor.ai:443"

_FROZEN = re.compile(r"app-compose\.frozen-(?P<tag>.+)-(?P<net>finney|test)\.json$")


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def network_name(network: str | None) -> str | None:
    """'finney' or 'test' for the two public networks, else None (a local chain
    or an endpoint URL: no frozen composes apply, the env decides)."""
    n = (network or "").strip().lower()
    return n if n in ("finney", "test") else None


def frozen_compose_entries(network: str | None, root: str | None = None
                           ) -> list[tuple[str, str]]:
    """[(entry, frozen file name)] approved from the code for `network`, oldest
    tag first: every non-superseded frozen compose of that network, paired with
    every pinned base."""
    net = network_name(network)
    if net is None:
        return []
    root = root or _repo_root()
    out: list[tuple[str, str]] = []
    paths = sorted(glob.glob(os.path.join(root, "deploy", "dstack", f"app-compose.frozen-*-{net}.json")))
    for path in paths:
        name = os.path.basename(path)
        if not _FROZEN.search(name):
            continue
        with open(path, "rb") as f:
            compose_hash = hashlib.sha256(f.read()).hexdigest()
        if compose_hash in SUPERSEDED_COMPOSES:
            continue
        for base in PINNED_BASE_MEASUREMENTS.get(net, ()):
            out.append((f"{base}:{compose_hash}", name))
    return out


def from_code_enabled() -> bool:
    return os.getenv("FUGAL_APPROVED_FROM_CODE", "1").strip().lower() not in ("0", "false", "no")


def resolve_approved(env_entries: list[str], network: str | None, *, root: str | None = None,
                     from_code: bool | None = None) -> tuple[list[str], list[tuple[str, str]]]:
    """(entries, added_from_code): the env's entries, then every entry the code
    approves that the env did not already name. Order is stable (env first,
    then code by tag), so two validators on the same release log the same list."""
    if from_code is None:
        from_code = from_code_enabled()
    entries = [e for e in env_entries if e]
    if not from_code:
        return entries, []
    seen = set(entries)
    added = []
    for entry, name in frozen_compose_entries(network, root):
        if entry not in seen:
            seen.add(entry)
            entries.append(entry)
            added.append((entry, name))
    return entries, added


def default_endpoints(network: str | None, archive: list[str], fallback: list[str]
                      ) -> tuple[list[str], list[str], bool]:
    """On finney, unset archive / fallback endpoints default to the public ones."""
    if network_name(network) != "finney":
        return archive, fallback, False
    changed = False
    if not archive:
        archive, changed = [FINNEY_DEFAULT_ARCHIVE], True
    if not fallback:
        fallback, changed = [FINNEY_DEFAULT_FALLBACK], True
    return archive, fallback, changed


def prepare(network: str | None) -> dict:
    """Apply 1 and 3 to the running config (in place: the validator and the
    fingerprint read these lists) and log what changed. Returns a summary for
    the READY line."""
    from fugal_subnet import config

    entries, added = resolve_approved(list(config.TEE_APPROVED_MEASUREMENTS), network)
    config.TEE_APPROVED_MEASUREMENTS[:] = entries
    if added:
        logger.info("Approved from this release's frozen composes: %s",
                    "; ".join(f"{name} ({e.split(':', 1)[1][:12]}…)" for e, name in added))
    elif not from_code_enabled():
        logger.info("FUGAL_APPROVED_FROM_CODE=0: approved entries come from "
                    "FUGAL_TEE_MEASUREMENTS only")
    archive, fallback, changed = default_endpoints(
        network, list(config.ARCHIVE_ENDPOINTS), list(config.FALLBACK_ENDPOINTS))
    if changed:
        config.ARCHIVE_ENDPOINTS[:] = archive
        config.FALLBACK_ENDPOINTS[:] = fallback
        logger.info("Chain endpoints: archive %s, fallback %s (finney defaults; set "
                    "FUGAL_ARCHIVE_ENDPOINTS / FUGAL_FALLBACK_ENDPOINTS to override)",
                    ", ".join(archive), ", ".join(fallback))
    newest = (frozen_compose_entries(network) or [("", "")])[-1][1]
    return {"entries": len(entries), "newest": newest,
            "archive": bool(config.ARCHIVE_ENDPOINTS)}


def ensure_embeddings(path: str, pin: str | None, url: str | None, *, fetch=None) -> str:
    """Make `path` the pinned embeddings file, downloading it if missing or
    wrong. Returns a note for the log. Raises only when a needed download fails
    (the caller's existing refusal then names the problem)."""
    if not pin:
        return "embeddings: no pin applies (unpinned pool), using the file as is"
    if os.path.exists(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() == pin:
            return f"embeddings: {path} matches the pin"
    if not url:
        raise ValueError(f"{path} is missing or does not match the pin, and no EMBEDDINGS_URL "
                         "is set to fetch it from")
    from fugal_subnet.embeddings_file import fetch as _fetch

    (fetch or _fetch)(url, pin, path)
    return f"embeddings: fetched {path} from {url} (sha256 matches the pin)"


def ready_line(*, entries: int, newest: str, v3: str, archive: bool, collateral: bool | None,
               role: str) -> tuple[bool, str]:
    """(ready, line). `v3` is 'ok:<pool size>', 'not scheduled', or 'failed:<why>'."""
    missing = []
    if entries < 1:
        missing.append("no approved entries: update the checkout to the current release "
                       "(git pull), or set FUGAL_TEE_MEASUREMENTS")
    if v3.startswith("failed"):
        missing.append(f"v3 inputs {v3[7:] or 'failed'}: run `.venv/bin/python -m "
                       "fugal_subnet.embeddings_file fetch`")
    if not archive:
        missing.append(f"no archive endpoint: set FUGAL_ARCHIVE_ENDPOINTS={FINNEY_DEFAULT_ARCHIVE}")
    if collateral is False:
        missing.append("DCAP collateral endpoint did not answer: check FUGAL_PCCS_URL "
                       "(default https://pccs.phala.network)")
    if missing:
        return False, "NOT READY: " + "; ".join(missing)
    v3_part = (f"v3 inputs ok (pool {v3[3:]}, embeddings ok)" if v3.startswith("ok:")
               else f"v3 {v3}")
    newest_part = f" (newest {newest})" if newest else ""
    coll = "collateral ok" if collateral else "collateral not checked (mock)"
    return True, (f"READY: approved entries {entries}{newest_part}, {v3_part}, archive ok, "
                  f"{coll}, role {role}")
