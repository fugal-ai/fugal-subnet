#!/usr/bin/env python3
"""Fugal subnet validator — TEE proof verification and weight-setting.

Usage:
    python neurons/validator.py --netuid 1 --coldkey fugal_validator --hotkey default
    python neurons/validator.py --netuid 1 --coldkey fugal_validator --hotkey default --once

Validators never call models. Each epoch:
1. Read head commitments at the boundary; derive the nonce; select the slice
   (protocol v1 or v2 by the boundary, slicer.protocol_version_for_boundary)
2. Query miners for TEE-attested proofs (FugalProofSynapse) at the collection block
3. Verify each proof (commitment, bundle, DCAP, approved entry, nonce, slice,
   exploration, head and cost bindings)
4. Score: v1, evidence and the frontier composite; v2, shared coldkey-median
   grades, the paired comparison window, the champion seat and the 85/10/5 payout
5. Weights (set on chain, commit-reveal aware), the reveal, state saved atomically

Miners run benchmarks inside Intel TDX VMs and pay for their own inference.
The validator's only cost is bandwidth and compute for verification.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import sys
import time
from typing import NamedTuple

# isort: off
import fugal_subnet.determinism  # noqa: F401

import click  # noqa: E402
import numpy as np  # noqa: E402
# isort: on

logger = logging.getLogger("fugal.validator")

STATE_PATH = os.getenv("FUGAL_STATE_PATH", "results/success-v1/validator_state.json")


def _known_fields(cls, data: dict) -> dict:
    """Keep the keys the dataclass declares; a field added or removed in an
    upgrade must not make every existing state file unloadable."""
    names = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(set(data) - names)
    if dropped:
        logger.warning("state carries fields %s unknown to %s; ignored", dropped, cls.__name__)
    return {k: v for k, v in data.items() if k in names}


def _parse_state(raw: dict, records_cls) -> dict:
    from fugal_subnet.evidence import Evidence
    from fugal_subnet.routing_protocol import require_identity

    require_identity(raw.get("routing_protocol"))
    records = {}
    for uid, rec in raw.get("records", {}).items():
        rec = dict(rec)
        ev_data = rec.pop("evidence", None)
        record = records_cls(**_known_fields(records_cls, rec))
        if ev_data is not None:
            record.evidence = Evidence(**_known_fields(Evidence, ev_data))
        records[int(uid)] = record
    return {
        "records": records,
        "prev_uids": [int(u) for u in raw.get("prev_uids", [])],
        "prev_weights": [float(w) for w in raw.get("prev_weights", [])],
        "last_epoch_index": int(raw.get("last_epoch_index", -1)),
        # "first_commit_blocks" was persisted here until 2026-10-14. It is
        # ignored on load: seniority is now derived from chain state at
        # the boundary block (commitments.seniority_key), not remembered.
        "frame": raw.get("frame"),
        "v2": raw.get("v2"),
    }


def load_state(records_cls, path: str | None = None) -> dict:
    """Load persisted validator state across supervisor-managed restarts.

    A truncated or unparseable file (a power loss between the write and the
    rename used to be able to leave one) falls back to the previous epoch's
    copy, `<path>.bak`, with a loud warning: one epoch of stale evidence is a
    small divergence, and deleting the file was the only recovery before,
    which is a large one (I1).

    A MISSING file means a save was interrupted: save_state writes and fsyncs
    `<path>.tmp`, moves the current file to `.bak`, then renames `.tmp` in.
    Between the two renames `.tmp` already holds the epoch just applied (its
    weights submitted, its v2 window written), so it resumes from `.tmp`: the
    newest complete state, not one epoch behind. Only if `.tmp` cannot be
    loaded does it fall back to `.bak`. Starting fresh there would be the large
    divergence (empty evidence, bootstrap frame). A deliberate fresh start
    moves all three files aside; only none of them is a first start.
    """
    path = path or STATE_PATH
    bak = path + ".bak"
    try:
        with open(path) as f:
            raw = json.load(f)
        return _parse_state(raw, records_cls)
    except FileNotFoundError:
        tmp = path + ".tmp"
        if os.path.exists(tmp):
            try:
                with open(tmp) as f:
                    state = _parse_state(json.load(f), records_cls)
            except Exception as e:  # noqa: BLE001 - fall back to the previous copy below
                logger.error("Validator state %s is missing and the interrupted save %s "
                             "cannot be loaded (%s)", path, tmp, e)
            else:
                logger.warning(
                    "Validator state %s is missing: a save was interrupted between its two "
                    "renames. Resuming from the completed save %s (the latest epoch).",
                    path, tmp,
                )
                return state
        if not os.path.exists(bak):
            return _empty_state()
        problem = "missing (a save was interrupted between its two renames)"
        cause = None
    except Exception as e:
        if not os.path.exists(bak):
            raise ValueError(f"Cannot resume validator state {path}: {e}") from e
        problem, cause = f"unreadable ({e})", e
    logger.error(
        "Validator state %s is %s; resuming from the previous epoch's copy %s. This "
        "validator is one epoch behind its peers until it catches up. To start fresh "
        "on purpose, move it, its .bak and its .tmp aside.", path, problem, bak,
    )
    try:
        with open(bak) as f:
            return _parse_state(json.load(f), records_cls)
    except Exception as e:
        raise ValueError(
            f"Cannot resume validator state: {path} is {problem} and its previous copy "
            f"{bak} cannot be loaded either ({e}). To start fresh on purpose, move "
            "it, its .bak and its .tmp aside."
        ) from (cause or e)


def _empty_state() -> dict:
    return {"records": {}, "prev_uids": [], "prev_weights": [],
            "last_epoch_index": -1, "frame": None}


def save_state(records: dict, prev_uids: list[int], prev_weights: list[float],
               last_epoch_index: int, frame=None, path: str | None = None,
               v2: dict | None = None, v2_window=None, published=None):
    """Persist validator state: write, fsync, keep the previous copy, rename.

    The v2 window is written first, beside the JSON, so a crash between the
    two leaves a window one epoch ahead at worst, never a state file pointing
    at a window that does not exist. Protocol v4's published set
    (`published`, a memo_sync.ChainSet) is written the same way, before it.
    """
    path = path or STATE_PATH
    if v2_window is not None:
        from fugal_subnet.v2_epoch import save_window
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_window(path + ".v2window.npz", v2_window)
    if published is not None:
        published.save()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from fugal_subnet.routing_protocol import identity
    payload = {
        "routing_protocol": identity(),
        "records": {str(uid): dataclasses.asdict(rec) for uid, rec in records.items()},
        "prev_uids": prev_uids,
        "prev_weights": prev_weights,
        "last_epoch_index": last_epoch_index,
        "frame": frame.to_dict() if frame is not None else None,
        "v2": v2,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())      # the rename is atomic; the bytes were not
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def axon_gate_notice(stake: float, permit: bool, threshold: float) -> str | None:
    """Startup warning when this validator's own stake is below the floor live
    miners apply before answering a query (`neurons/miner.py`, `caller_allowed`).
    A refused query is indistinguishable from an absent miner on this side, so
    the one place it can be said is here, before the first epoch."""
    if stake >= threshold:
        return None
    permit_note = (" A validator permit does not substitute for it."
                   if permit else "")
    return (
        f"This hotkey's stake on the subnet is {stake:.0f} (chain total_stake: "
        f"alpha plus 0.18 x root TAO, childkey-inherited), below the "
        f"{threshold:.0f} that live miners require before answering a query "
        f"(FUGAL_MIN_VALIDATOR_STAKE in the miner image).{permit_note} Until it "
        "is staked you receive no proofs and see 'No valid proofs received' "
        "every epoch with no error on either side."
    )


def _check_live_configuration(live: bool, approved, activation_block, epoch_interval: int,
                              category_map) -> None:
    """Refuse, before any chain connection, a configuration that would make this
    validator score differently from its peers or score everyone zero.

    - --live needs FUGAL_TEE_MEASUREMENTS. With no approved entry every proof
      fails the runtime-image check and every miner is set to zero weight:
      fail-closed is right for the check, but silently deweighting the subnet
      is not how to learn the variable is unset.
    - The entries are parsed and LOGGED, so a rotation's two validators can be
      compared from their logs, and a malformed entry (a bare trailing colon
      would silently accept any application) fails here rather than at the
      first proof. A bare base is refused in --live: the application fixes the
      model upstream, so a bare base approves a TD whose upstream the miner
      chose (diligence 2026-10, finding 1).
    - The activation block must be an epoch boundary (docs/INVARIANTS.md, "I1 —
      protocol v2 activates at one named boundary"), and v2 needs its category
      map, without which no question is "hard" and the hard weight never applies.
    - --live needs a REVIEWED token manifest for every protocol it can score:
      v1's always, v2's once the activation block is set (the v2 one was not
      checked; a candidate from a re-measure would have been scored).
    """
    if live and not approved:
        raise click.ClickException(
            "--live requires FUGAL_TEE_MEASUREMENTS. With no approved "
            "measurement every proof fails the runtime-image check, so this "
            "validator would score every miner zero and set weights to match. "
            "Get the approved entry from the signed image release: "
            "`scripts/fetch_approved_entries.py --tag <image tag> --network "
            "<finney|test>` (docs/APPROVED_ENTRY_ROTATION.md)."
        )
    if approved:
        from fugal_subnet.tee.verify import parse_approved
        try:
            parsed = parse_approved(approved, require_app=live)
        except ValueError as e:
            raise click.ClickException(f"FUGAL_TEE_MEASUREMENTS: {e}") from e
        described = [
            f"{base[:12]}…:{app[:12]}…" if app else f"{base[:12]}… (bare base: any app)"
            for base, apps in sorted(parsed.items())
            for app in (sorted(apps) or [""])
        ]
        logger.info("Approved entries: %d — %s", len(described), "; ".join(described))

    from fugal_subnet.benchmarks.slicer import blocks_per_epoch, validate_activation_block
    bpe = blocks_per_epoch(epoch_interval)
    try:
        validate_activation_block(activation_block, bpe)
    except ValueError as e:
        raise click.ClickException(f"FUGAL_V2_ACTIVATION_BLOCK: {e}") from e
    if activation_block is not None and not category_map:
        msg = ("FUGAL_V2_ACTIVATION_BLOCK is set but config.BENCHMARK_CATEGORY_V2 is "
               "missing: v2 cannot tell hard questions apart")
        if live:
            raise click.ClickException(msg)
        logger.warning(msg)
    if activation_block is None:
        logger.info("Protocol v2 activation: not set; v1 rules apply to every epoch")
    else:
        logger.info("Protocol v2 activation: boundary block %d (epoch index %d); v1 rules "
                    "before it", activation_block, activation_block // bpe)

    if live:
        from fugal_subnet.routing_protocol import manifest_path
        from fugal_subnet.vendor import success_contract as contract
        for version in ((1, 2) if activation_block is not None else (1,)):
            try:
                contract.validate_manifest(json.loads(manifest_path(version).read_text()),
                                           deployable=True)
            except ValueError as e:
                raise click.ClickException(
                    f"protocol v{version} token manifest is not deployable: {e}") from e


def v2_compose_hashes(activation_block: int, root: str | None = None) -> set[str] | None:
    """Compose hashes of the frozen miner composes in this checkout that are
    built for `activation_block` (they set FUGAL_V2_ACTIVATION_BLOCK to it).

    An approved entry is `<base>:<compose hash>`, and the compose hash is
    sha256 of a frozen app-compose.json's raw bytes (deploy/dstack/). None when
    the checkout carries no frozen composes at all (an installed wheel): there
    is then nothing to compare against, and the caller says so.
    """
    import glob
    import hashlib
    import re

    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(glob.glob(os.path.join(root, "deploy", "dstack", "app-compose.frozen-*.json")))
    if not paths:
        return None
    wanted = re.compile(rf"FUGAL_V2_ACTIVATION_BLOCK=[\"']?{int(activation_block)}\b")
    out: set[str] = set()
    for path in paths:
        try:
            with open(path, "rb") as f:
                raw = f.read()
            compose = str(json.loads(raw).get("docker_compose_file", ""))
        except (OSError, ValueError, AttributeError):
            continue
        if wanted.search(compose):
            out.add(hashlib.sha256(raw).hexdigest())
    return out


def _check_collateral_endpoint(live: bool, on_finney: bool, probe=None) -> bool | None:
    """Ask the DCAP collateral endpoint once at startup, so a dead one is
    named in the log before the collection point leaves an epoch unverified.
    ADVISORY ONLY, on every chain: it never refuses to start. The first
    version refused on finney and took down the live root (v0.5.0), because
    the endpoint blocked the probe's default User-Agent while serving the
    verifier's real requests; a probe can be wrong where the verifier is not.
    Mock never verifies quotes, so it skips the probe."""
    if not live:
        return None
    from fugal_subnet import config
    from fugal_subnet.tee.attestation import probe_collateral_endpoint

    try:
        ok, why = (probe or probe_collateral_endpoint)(config.TEE_PCCS_URL)
    except Exception as e:  # noqa: BLE001 - an advisory probe never stops the validator
        ok, why = False, f"the probe itself failed ({type(e).__name__}: {e})"
    if ok:
        logger.info("DCAP collateral endpoint checked: %s", why)
        return True
    logger.warning(
        "WARNING: the DCAP collateral endpoint did not answer the startup probe: %s. "
        "If proofs go unverified at collection, check FUGAL_PCCS_URL (default "
        "https://pccs.phala.network) or the local caching PCCS it points at. "
        "Starting anyway: this probe is advisory.", why)
    return False


def _check_chain_configuration(genesis: str | None, live: bool, approved, activation_block,
                               validator_ip: str) -> int | None:
    """Refuse, once the chain is known, what would make a FINNEY validator
    score differently from its peers while looking healthy. Returns the
    fresh-data confirmation rule's activation block for this chain.

    Each of these was got wrong on mainnet by an operator following the guide,
    and each failed silently:

    - FUGAL_V2_ACTIVATION_BLOCK unset or not finney's flip block: every epoch
      is scored by the v1 rules.
    - No approved entry for a v2 miner image: every v2 proof fails the
      runtime-image check, the seat stays empty and all weight burns.
    - FUGAL_VALIDATOR_IP unset: miners that admit only validators at their
      firewall drop this validator's queries, so it verifies fewer proofs than
      its peers and its seat can differ from theirs.

    Mock, and every chain but finney, keep the warnings they had: a rehearsal
    legitimately runs without some of these.
    """
    from fugal_subnet import config
    from fugal_subnet.axon_allowlist import _routable_v4
    from fugal_subnet.tee.verify import parse_approved

    try:
        confirm_block = config.confirm_activation_block(genesis)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    if confirm_block is None:
        logger.info("Fresh-data confirmation of a seat take: not scheduled on this chain")
    else:
        logger.info("Fresh-data confirmation of a seat take: from the first epoch whose "
                    "boundary is at or after block %d", confirm_block)
    finney = genesis is not None and str(genesis).lower() == config.FINNEY_GENESIS_HASH.lower()
    if not (finney and live):
        return confirm_block

    if activation_block != config.FINNEY_V2_ACTIVATION_BLOCK:
        raise click.ClickException(
            f"FUGAL_V2_ACTIVATION_BLOCK is {activation_block if activation_block is not None else 'unset'}"
            f" on finney, where protocol v2 activated at block {config.FINNEY_V2_ACTIVATION_BLOCK}. "
            f"Set FUGAL_V2_ACTIVATION_BLOCK={config.FINNEY_V2_ACTIVATION_BLOCK}: with any other "
            "value this validator scores epochs by different rules from every peer."
        )
    v2_hashes = v2_compose_hashes(activation_block)
    if v2_hashes is None:
        logger.warning("This checkout has no deploy/dstack/app-compose.frozen-*.json, so the "
                       "approved entries could not be checked for a protocol v2 miner image.")
    else:
        apps = {app for apps in parse_approved(approved, require_app=True).values() for app in apps}
        if not apps & v2_hashes:
            raise click.ClickException(
                "FUGAL_TEE_MEASUREMENTS approves no protocol v2 miner image. None of its "
                f"{len(apps)} compose hash(es) is a frozen compose in this checkout built for "
                f"block {activation_block} ({', '.join(sorted(h[:12] + '…' for h in v2_hashes)) or 'none found'}). "
                "Every v2 proof would fail the runtime-image check, the seat would stay empty "
                "and this validator would burn all of its weight. Add the current entry: "
                "`scripts/fetch_approved_entries.py --tag <image tag> --network finney` "
                "(docs/APPROVED_ENTRY_ROTATION.md)."
            )
    if not validator_ip or not _routable_v4(validator_ip):
        raise click.ClickException(
            f"FUGAL_VALIDATOR_IP is {validator_ip or 'unset'}, not a public IPv4 address. Miners "
            "admit only the validator IPs recorded on chain at their firewall, so without it "
            "they drop this validator's queries: it verifies fewer proofs than its peers and "
            "its seat can differ from theirs. Set it to this host's public egress address "
            "(`curl -s https://ifconfig.me`)."
        )
    return confirm_block


def _sync_reference(role, netuid: int, *, since: int, budget_s: float, required: bool,
                    subtensor=None):
    """Fetch the root's reveals for a follower that syncs them itself; a no-op
    for a root or an operator-managed directory. Returns the SyncResult or None.

    With the root's hotkey known (built in on finney), every fetched reveal is
    verified against the root's on-chain reveal commitment, read here from
    `subtensor` (reveal_sync, TRUST); a reveal that differs never reaches
    disk and is logged as an ERROR.

    `required` (startup): a follower that has no reference reveals at all and
    cannot fetch any is refused, because it would start with an empty seat and
    burn its weight, the failure this exists to remove. In the epoch loop it
    is never required: a failed fetch is logged and the epoch runs on what is
    already on disk (I6).
    """
    from fugal_subnet import reveal_sync

    if role is None or role.root or not role.bucket:
        return None
    if role.root_hotkey:
        pointer = (reveal_sync.read_root_pointer(subtensor, netuid, role.root_hotkey)
                   if subtensor is not None else None)
        result = reveal_sync.sync(role.bucket, netuid, role.reference_dir, since=since,
                                  budget_s=budget_s, verify=True, pointer=pointer,
                                  ledger_path=reveal_sync.pointer_ledger_path(role.reference_dir))
    else:
        result = reveal_sync.sync(role.bucket, netuid, role.reference_dir, since=since,
                                  budget_s=budget_s)
    if result.rejected:
        logger.error(
            "Root reveals from gs://%s/%d: %d reveal(s) differ from the root's on-chain "
            "commitment and were NOT used (%s). This validator keeps the reveals it already "
            "verified; tell the subnet owner.", role.bucket, netuid, len(result.rejected),
            "; ".join(result.rejected[:3]))
    (logger.info if result.ok else logger.warning)(
        "Root reveals from gs://%s/%d: %s", role.bucket, netuid, result.summary())
    if required and result.newest is None:
        raise click.ClickException(
            f"Could not fetch the root validator's reveals from gs://{role.bucket}/{netuid} "
            f"({result.summary()}) and {role.reference_dir} holds none. A validator that is "
            "not the root starts from them; without them it would start with an empty seat "
            "and burn its weight. Check outbound HTTPS to storage.googleapis.com, or set "
            "FUGAL_ROOT=1 if this validator is the root."
        )
    return result


def _log_following(role, epoch_index: int, n_entries: int, own_seat, epoch_dir: str | None = None,
                   differ_limit: int | None = None) -> dict | None:
    """One line saying whose state this follower tracks and how far behind
    the reference is; an ERROR once its own seat has differed from the root's
    for longer than verify-and-resync takes to correct it."""
    import fugal_subnet
    from fugal_subnet import reveal_sync
    from fugal_subnet.commit_reveal import EPOCH_DIR
    from fugal_subnet.v2_sync import DRIFT_CHECKS

    if role is None or role.root:
        return None
    limit = differ_limit or DRIFT_CHECKS + 2
    st = reveal_sync.reference_status(role.reference_dir, epoch_dir or EPOCH_DIR, epoch_index,
                                      differ_limit=limit)
    if st["k"] is None:
        logger.error("Following root: NO reference reveal before e%08d in %s. This validator "
                     "cannot verify its seat against the root's.", epoch_index,
                     role.reference_dir)
        return st
    logger.info(
        "Following root: seat uid %s (head %s) at e%08d, entries %d, fugal v%s (root v%s), "
        "reference lag %d epoch(s); own seat %s",
        st["uid"] if st["uid"] is not None else "none", (st["champion"] or "none")[:12],
        st["k"], n_entries, fugal_subnet.__version__, st["version"] or "?", st["lag"],
        (getattr(own_seat, "champion", None) or "none")[:12],
    )
    if st["seat_differs"] >= limit:
        logger.error(
            "This validator's champion has differed from the root's on the newest %d reference "
            "epochs (root %s, own %s at e%08d), longer than verify-and-resync takes to correct. "
            "Check: the approved entries (FUGAL_TEE_MEASUREMENTS), FUGAL_VALIDATOR_IP on chain, "
            "and that this is the root's release (v%s here, v%s there).",
            st["seat_differs"], (st["champion"] or "none")[:12],
            (st["own_champion"] or "none")[:12], st["k"], fugal_subnet.__version__,
            st["version"] or "?",
        )
    return st


from fugal_subnet.embeddings_file import Embeddings  # noqa: E402


def _v3_explore_answers(verified_proofs: dict, coldkey_of: dict, shared: dict) -> list:
    """Every verified proof's exploration results as pool answers. An answer
    is private unless its (question, model) is in the epoch's shared set."""
    from fugal_subnet.explore_pool import ExploreAnswer

    return [
        ExploreAnswer(
            question_id=r.question_id, model=r.routed_model, coldkey=coldkey_of[uid],
            correct=bool(r.correct), failed=bool(r.failed),
            private=shared.get(r.question_id) != r.routed_model,
            prompt_tokens=int(r.prompt_tokens), completion_tokens=int(r.completion_tokens),
            latency_ms=int(getattr(r, "latency_ms", None) or 0),
        )
        for uid in sorted(verified_proofs)
        for r in verified_proofs[uid].exploration_results
    ]


def _v3_heads(verified_heads: dict) -> dict:
    """{uid: (W, b, model ids)} for odds under v3 (`_odds_heads`)."""
    return _odds_heads(verified_heads, 3)


def _load_head_for(protocol_version: int, head_bytes: bytes):
    """The head as the protocol reads it: head_eval.load_head_v4 from v4 (rows
    by entry id, token means optional and ignored), load_head_from_npz before."""
    from fugal_subnet.head_eval import load_head_from_npz, load_head_v4

    return (load_head_v4 if int(protocol_version) >= 4 else load_head_from_npz)(head_bytes)


def _odds_heads(verified_heads: dict, protocol_version: int) -> dict:
    """{uid: (W, b, model ids)} for odds. A head that cannot be loaded here was
    already refused by verify_proof; one that is missing is simply not scored
    on odds (never an epoch failure, I6)."""
    out = {}
    for uid, head_bytes in sorted(verified_heads.items()):
        try:
            head = _load_head_for(protocol_version, head_bytes)
            out[uid] = (head.W, head.b, list(head.models))
        except Exception as e:  # noqa: BLE001 - one miner, not the epoch
            logger.warning("UID %d: head not usable for odds (%s)", uid, e)
    return out


def _head_within_bounds(head_bytes: bytes, protocol_version: int = 3) -> str | None:
    """None, or why a v3 or v4 head is refused: it does not load as the
    protocol's head, or a weight or bias is outside the bound exact odds are
    computed under (head_odds.HEAD_MAX_ABS_PARAM)."""
    from fugal_subnet.head_odds import check_head_bounds

    try:
        head = _load_head_for(protocol_version, head_bytes)
        check_head_bounds(head.W, head.b)
    except Exception as e:  # noqa: BLE001 - miner input: a verdict, never a raise
        return str(e)
    return None


def _rules_tag(protocol_version: int, on_odds: bool) -> dict | None:
    """What a v2/v3 seat state was built under. Published in v3 reveals
    (`v2_state.rules`) and kept with the local state, so a state built under
    other rules is recognised and reset instead of carried over."""
    if protocol_version < 3:
        return None
    return {"protocol": int(protocol_version), "seat": "odds" if on_odds else "net_value"}


def _epoch_to_run(current_block: int, blocks_per_epoch: int, last_epoch_index: int,
                  once: bool) -> int | None:
    """The epoch to run at `current_block`, or None to wait for the next one.

    Each epoch is scored once: running one already written again would fold
    its evidence, its misses and the frame's decay in a second time. `--once`
    is the exception, an operator's deliberate re-run of the current epoch.
    """
    from fugal_subnet.benchmarks.slicer import epoch_index_for_block
    epoch_index = epoch_index_for_block(current_block, blocks_per_epoch)
    if epoch_index <= last_epoch_index and not once:
        return None
    return epoch_index


def _epoch_boundary(subtensor, epoch_index: int, blocks_per_epoch: int,
                    activation_block: int | None,
                    v3_activation_block: int | None = None, *,
                    v4_activation_block: int | None = None,
                    v5_activation_block: int | None = None) -> tuple[int, str, str, int]:
    """(boundary block, its hash, epoch id, protocol version) of an epoch.

    All four come from the boundary block, never the current one, so every
    peer and every retry of the epoch derives the same nonce, id and rules.
    The id has one source, slicer.epoch_id_for_block (I1).
    """
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block, protocol_version_for_boundary
    boundary_block = epoch_index * blocks_per_epoch
    return (boundary_block, subtensor.get_block_hash(boundary_block),
            epoch_id_for_block(epoch_index),
            protocol_version_for_boundary(boundary_block, activation_block,
                                          v3_activation_block,
                                          v4_activation_block=v4_activation_block,
                                          v5_activation_block=v5_activation_block))


class _SlicePlan(NamedTuple):
    nonce: bytes
    questions: list
    question_ids: list
    questions_hash: str
    explore_map: dict
    slice_gold: dict
    # v3: the epoch's cost-aware exploration shares (epoch_plan.explore_shares),
    # used for every miner's private plan and for the pool's pair weights.
    explore_shares: dict | None = None


def _epoch_slice(protocol_version: int, epoch_id: str, block_hash: str, pool: list,
                 gold_answers: dict, pool_models: list, slice_size: int,
                 explore_size: int) -> _SlicePlan:
    """This epoch's scored questions, exploration targets and the gold for them.

    The miner selects with the same epoch_plan functions on the same versioned
    pool and model list, or every proof fails on questions_hash. The gold is
    for the assigned questions only: handing over the whole pool would let a
    proof reference any question in it and still verify.
    """
    from fugal_subnet.benchmarks.slicer import derive_nonce
    from fugal_subnet.epoch_plan import exploration_plan, explore_shares, scored_slice
    from fugal_subnet.tee.verify import compute_questions_hash
    if int(protocol_version) >= 4:
        # v4's exploration is drawn from the epoch's menu for each hotkey, not
        # from a model list; its slice is the epoch's plan's (_epoch_slice_v4).
        raise ValueError("a protocol v4 slice is drawn by _epoch_slice_v4 on the epoch's plan")
    nonce = derive_nonce(epoch_id, block_hash)
    questions = scored_slice(protocol_version, nonce, pool, slice_size)
    question_ids = [q["question_id"] for q in questions]
    shares = explore_shares(protocol_version, pool_models)
    explore_map = exploration_plan(protocol_version, nonce, pool, set(question_ids),
                                   pool_models, explore_size, shares=shares)
    slice_gold = {qid: gold_answers[qid]
                  for qid in list(question_ids) + list(explore_map) if qid in gold_answers}
    return _SlicePlan(nonce, questions, question_ids, compute_questions_hash(question_ids),
                      explore_map, slice_gold, shares)


def _epoch_slice_v4(epoch_id: str, block_hash: str, pool: list, gold_answers: dict, plan,
                    slice_size: int) -> _SlicePlan:
    """A v4 epoch's scored questions and their gold: the epoch's plan
    (epoch_plan.plan_v4, the one the miner's TD runs) draws the slice on the
    pinned v4 mix. Exploration and measurement are each hotkey's
    (verify.expected_v4 on the epoch's menu), so the plan-wide map is empty."""
    from fugal_subnet.benchmarks.slicer import derive_nonce
    from fugal_subnet.tee.verify import compute_questions_hash
    if plan is None:
        raise RuntimeError(f"epoch {epoch_id} is protocol v4 but its plan was not built at startup")
    nonce = derive_nonce(epoch_id, block_hash)
    questions = plan.scored_slice(nonce, pool, slice_size)
    question_ids = [q["question_id"] for q in questions]
    slice_gold = {qid: gold_answers[qid] for qid in question_ids if qid in gold_answers}
    return _SlicePlan(nonce, questions, question_ids, compute_questions_hash(question_ids),
                      {}, slice_gold, None)


class _V4Tables(NamedTuple):
    """What a v4 epoch prices, indexes and categorises with: the epoch's
    MenuState (entries, prices, the pick-cost table) and the pinned v4 mix
    (categories, targets)."""
    prices: dict           # entry id -> (dollars a prompt token, dollars a completion token)
    models: list           # the menu's entry ids, its index space
    model_index: dict
    bench_category: dict   # question set -> category (epoch_plan.mix_v4)
    category_targets: dict  # category -> share of the scored slice (mix_v4, per-cent / 100)
    # entry id -> (its pick cost, {category: its pick cost there}), dollars a
    # call: what net value prices a scored call at (W13)
    pick_spend: dict = {}


def _v4_tables(v4_menu) -> _V4Tables:
    """The epoch's v4 tables. Prices are the menu's own, exact picodollars a
    token as dollars (the nearest float of the exact fraction, the same on
    every host). The pick spend is the menu's pick-cost table
    (menu_state.pick_cost) for every entry in every category of the v4 mix,
    the same nearest floats: a scored call's net value is priced at it
    (SEAT_MINIMAL_V4.md 3.4), not at its own tokens. A node without a menu
    (`v4_menu` not ok) has no prices and no entries: it checks no proof that
    epoch."""
    from fractions import Fraction

    from fugal_subnet import epoch_plan
    from fugal_subnet import menu_state as ms

    mix = epoch_plan.mix_v4()
    state = v4_menu.state if v4_menu is not None and v4_menu.ok else None
    entries = state.entries if state is not None else ()
    prices = {e.id: (float(Fraction(e.price_in, ms.PICO)), float(Fraction(e.price_out, ms.PICO)))
              for e in entries}
    models = [e.id for e in entries]
    categories = sorted(mix.targets)
    pick_spend = {e.id: (float(ms.entry_pick_cost(e)),
                         {c: float(ms.entry_pick_cost(e, c)) for c in categories})
                  for e in entries}
    return _V4Tables(prices=prices, models=models, model_index={m: i for i, m in enumerate(models)},
                     bench_category=dict(mix.bench_category),
                     category_targets={c: v / 100 for c, v in mix.targets.items()},
                     pick_spend=pick_spend)


def _v4_failure_bound(verified_proofs: dict, metagraph, state):
    """picks_v4.failure_refusals over the epoch's verified v4 proofs, each with
    its hotkey's coldkey as the pinned metagraph names it (the operator the
    outage median is taken over, as the menu step's)."""
    from fugal_subnet import menu_inputs as mi
    from fugal_subnet import picks_v4

    coldkeys = list(getattr(metagraph, "coldkeys", None) or [])
    vps = [mi.VerifiedProof(p, str(coldkeys[uid]) if uid < len(coldkeys) and coldkeys[uid]
                            else p.hotkey)
           for uid, p in sorted(verified_proofs.items())]
    return picks_v4.failure_refusals(vps, state)


def _v4_explored(expected: dict) -> dict:
    """{question id: entry} of every checked hotkey's v4 exploration, for the
    frame's mix diagnostics (each hotkey explores its own questions)."""
    out: dict = {}
    for uid in sorted(expected):
        out.update(expected[uid].exploration)
    return out


def _odds_block_for(protocol_version: int, odds_block: int | None, v4_block: int | None):
    """The block from which the seat is decided on odds in this epoch: v3's
    odds seat block, and under v4 the v4 block at the latest (v4 runs the
    v3 seat rules on odds from its first epoch, on an empty seat)."""
    if int(protocol_version) >= 4 and v4_block is not None:
        return v4_block if odds_block is None else min(int(odds_block), int(v4_block))
    return odds_block


@click.command()
@click.option("--network", default=lambda: os.getenv("FUGAL_NETWORK", "test"),
              help="Network: finney, test, local")
@click.option("--netuid", type=int,
              default=lambda: int(os.getenv("FUGAL_NETUID", "1")),
              help="Subnet netuid")
@click.option("--coldkey", default=lambda: os.getenv("WALLET_NAME", "default"),
              help="Wallet coldkey name")
@click.option("--hotkey", default=lambda: os.getenv("HOTKEY_NAME", "default"),
              help="Hotkey name")
@click.option("--wallet-path", default=lambda: os.getenv("FUGAL_WALLET_PATH") or None,
              type=click.Path(file_okay=False),
              help="Bittensor wallet root (defaults to the SDK wallet directory)")
@click.option("--once", is_flag=True, help="Run one epoch and exit")
@click.option("--log-level",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              default=lambda: os.getenv("LOG_LEVEL", "INFO"),
              help="Logging level")
@click.option(
    "--live/--mock",
    default=True,
    help="Live (default) requires real TDX attestation; --mock accepts unattested "
         "proofs and is refused on finney",
)
@click.option("--check", is_flag=True,
              help="Run every startup check, print the READY line and exit (0 when ready) "
                   "without running an epoch: what to run before a restart")
def main(network, netuid, coldkey, hotkey, wallet_path, once, log_level, live, check=False):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Upgrades are "update + restart": approved entries from this release's
    # frozen composes, finney's default endpoints (fugal_subnet/validator_setup).
    from fugal_subnet import validator_setup
    setup_summary = validator_setup.prepare(network)

    from fugal_subnet.config import (
        EPOCH_COLLECT_FRACTION,
        EPOCH_INTERVAL,
        EXPLORE_FRACTION,
        FRAME_DEFAULT_COMPLETION_TOKENS,
        REQUIRE_COMMITMENT,
        SLICE_SIZE,
        TEE_APPROVED_MEASUREMENTS,
    )
    mock = not live
    logger.info(
        "Operation mode: %s",
        "LIVE (requires real TDX attestation)" if live else "mock (accepts unattested proofs)",
    )
    from fugal_subnet import config as _cfg
    from fugal_subnet.config import V2_ACTIVATION_BLOCK
    # A `--network` URL (a validator on its own node) gets its release entries
    # once the chain's genesis is known (below), so an empty list is only
    # refused here when the network is already known.
    defer_entries = (live and not TEE_APPROVED_MEASUREMENTS
                     and validator_setup.network_name(network) is None
                     and validator_setup.from_code_enabled())
    _check_live_configuration(
        live and not defer_entries, TEE_APPROVED_MEASUREMENTS, V2_ACTIVATION_BLOCK,
        EPOCH_INTERVAL, getattr(_cfg, "BENCHMARK_CATEGORY_V2", None),
    )

    from fugal_subnet.fingerprint import assert_environment, consensus_digest
    # Consensus overrides are checked against the chain's genesis hash once the
    # chain is connected (below): refused on finney, warned about elsewhere.
    assert_environment(strict=live, consensus_overrides=False)
    logger.info("Consensus environment digest: %s", consensus_digest())

    import bittensor as bt

    from fugal_subnet.collect import CollectDendrite

    # After bittensor: importing it disables every logger that already exists,
    # so without this the neuron's own output — including every error and
    # traceback — is silently discarded. See fugal_subnet/logging_setup.
    from fugal_subnet.logging_setup import configure_logging
    configure_logging(log_level)

    from fugal_subnet import config as config_module
    from fugal_subnet.benchmarks.loader import load_all, pool_hash
    from fugal_subnet.benchmarks.slicer import (
        blocks_per_epoch as blocks_per_epoch_fn,
    )
    from fugal_subnet.benchmarks.slicer import (
        collect_block_for_epoch,
        epoch_index_for_block,
    )
    from fugal_subnet.commit_reveal import commit_epoch, reveal_epoch
    from fugal_subnet.config import (
        PAYOUT_SEAT_SHARES,
        TEE_COLLATERAL_EPOCH_BUDGET,
        V2_BURN_IN_QUESTIONS,
    )
    from fugal_subnet.dedup import find_duplicates
    from fugal_subnet.epoch_logger import (
        EpochLog,
        EpochTimer,
        detect_anomalies,
        write_epoch_log,
    )
    from fugal_subnet.frontier import build_frontier
    from fugal_subnet.head_eval import HeadScore, load_head_from_npz
    from fugal_subnet.pricing import question_input_tokens
    from fugal_subnet.protocol import FugalProofSynapse
    from fugal_subnet.reference_frame import (
        ReferenceFrame,
        accumulate_exploration,
        best_model,
        load_bootstrap,
        rebuild_from_reveals,
    )
    from fugal_subnet.scoring import MinerRecord, ScoringState, update_scores
    from fugal_subnet.tee.attestation import CollateralBudget, verification_order
    from fugal_subnet.tee.proof import BenchmarkProof
    from fugal_subnet.v2_epoch import (
        PicksInput,
        run_v2_epoch,
        seat_params_from_config,
        seat_to_dict,
    )
    from fugal_subnet.v2_epoch import (
        category_of as _v2_category_of,
    )
    # Keeps the keyword arguments of its last call: a v4 epoch's inputs.
    run_v2_epoch = _KeepsInputs(run_v2_epoch)

    wallet = bt.Wallet(name=coldkey, hotkey=hotkey, path=wallet_path)
    from fugal_subnet.config import ARCHIVE_ENDPOINTS, FALLBACK_ENDPOINTS
    subtensor = bt.Subtensor(
        network=network,
        archive_endpoints=ARCHIVE_ENDPOINTS or None,
        fallback_endpoints=FALLBACK_ENDPOINTS or None,
    )
    if live and not ARCHIVE_ENDPOINTS:
        logger.warning(
            "FUGAL_ARCHIVE_ENDPOINTS is unset. The validator pins reads to the "
            "boundary and collection blocks; a non-archive node keeps ~300 blocks "
            "and refuses older ones after a late start or a retried epoch."
        )
    from fugal_subnet.fingerprint import (
        assert_consensus_overrides_allowed,
        assert_mock_allowed,
        chain_genesis_hash,
    )
    try:
        genesis = chain_genesis_hash(subtensor)
        assert_mock_allowed(genesis, mock)
        assert_consensus_overrides_allowed(genesis)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e
    chain_setup = validator_setup.apply_for_chain(genesis, network)
    if chain_setup["newest"]:
        setup_summary["newest"] = chain_setup["newest"]
    setup_summary["entries"] = len(TEE_APPROVED_MEASUREMENTS)
    if chain_setup["endpoints_changed"]:
        subtensor = bt.Subtensor(
            network=network,
            archive_endpoints=ARCHIVE_ENDPOINTS or None,
            fallback_endpoints=FALLBACK_ENDPOINTS or None,
        )
    if chain_setup["added"] or defer_entries:
        _check_live_configuration(
            live, TEE_APPROVED_MEASUREMENTS, V2_ACTIVATION_BLOCK, EPOCH_INTERVAL,
            getattr(_cfg, "BENCHMARK_CATEGORY_V2", None),
        )
    from fugal_subnet.config import VALIDATOR_IP
    confirm_block = _check_chain_configuration(
        genesis, live, TEE_APPROVED_MEASUREMENTS, V2_ACTIVATION_BLOCK, VALIDATOR_IP)
    on_finney = genesis is not None and str(genesis).lower() == _cfg.FINNEY_GENESIS_HASH.lower()
    collateral_ok = _check_collateral_endpoint(live, on_finney)
    try:
        v3_block = _cfg.v3_activation_block(genesis)
        odds_block = _cfg.odds_seat_block(genesis)
        menu_b_blk = _cfg.menu_b_block(genesis)
        menu_grace = _cfg.menu_grace_epochs(genesis)
        # Protocol v4: no release schedules it on finney; another chain may
        # (FUGAL_V4_ACTIVATION_BLOCK, as v3's block before v3 was compiled).
        v4_block = _v4_activation_block(_cfg, genesis)
        # Protocol v5 (config.v5_activation_block): never before v4.
        v5_fn = getattr(_cfg, "v5_activation_block", None)
        v5_block = v5_fn(genesis) if callable(v5_fn) else None
        # v0.8.1: the block v5 reads raw commitment blocks from.
        fix_fn = getattr(_cfg, "v5_commit_fix_block", None)
        v5_fix_block = fix_fn(genesis) if callable(fix_fn) and v5_block is not None else None
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    _check_v3_blocks_aligned(v3_block, odds_block, EPOCH_INTERVAL, menu_b_blk, v4_block)
    if v5_block is not None:
        # An epoch boundary in BLOCKS (blocks_per_epoch), as the v2-v4 blocks are
        # checked: EPOCH_INTERVAL is seconds (finney 4320 s = 360 blocks).
        from fugal_subnet.benchmarks.slicer import blocks_per_epoch, validate_activation_block
        try:
            validate_activation_block(v5_block, blocks_per_epoch(EPOCH_INTERVAL))
            if v5_fix_block is not None:
                validate_activation_block(v5_fix_block, blocks_per_epoch(EPOCH_INTERVAL))
                if v5_fix_block < v5_block:
                    raise ValueError(f"the v5 commit-block fix block {v5_fix_block} is before "
                                     f"the v5 block {v5_block}")
        except ValueError as e:
            raise click.ClickException(f"v5 activation block: {e}") from e
    if v5_block is not None:
        logger.info("Protocol v5: from the first epoch whose boundary is at or after block %d",
                    v5_block)
        if v5_fix_block is not None:
            logger.info("Protocol v5 raw commit blocks (v0.8.1): from block %d", v5_fix_block)
    if v4_block is not None:
        logger.info("Protocol v4: from the first epoch whose boundary is at or after block %d "
                    "(not a finney release: a test or local-chain override)", v4_block)
    if menu_b_blk is not None:
        logger.info("Protocol v3 menu B: from the first epoch whose boundary is at or after "
                    "block %d; its added entries unscored for %d epochs from it",
                    menu_b_blk, menu_grace)
    if v3_block is None:
        logger.info("Protocol v3: not scheduled on this chain")
    else:
        logger.info("Protocol v3: from the first epoch whose boundary is at or after block %d; "
                    "the seat is decided on odds %s", v3_block,
                    f"from block {odds_block}" if odds_block is not None else "at no scheduled block")
    metagraph = subtensor.metagraph(netuid)
    # Not bt.Dendrite: its default pool of 100 connections lets axons that
    # never answer starve the rest of the field at collection (fugal_subnet/collect).
    dendrite = CollectDendrite(wallet=wallet)

    my_hotkey = wallet.hotkey.ss58_address
    if my_hotkey not in metagraph.hotkeys:
        raise click.ClickException(
            f"Hotkey {my_hotkey} is not registered on netuid {netuid}"
        )
    my_uid = metagraph.hotkeys.index(my_hotkey)
    logger.info("Validator UID %d on %s netuid %d", my_uid, network, netuid)
    try:
        assert_chain_weight_parameters(subtensor, netuid, live=live)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e

    # Reveal publishing: the bucket is operational configuration; the hold is
    # the chain's commit-reveal period unless overridden.
    from fugal_subnet.config import REVEAL_BUCKET, REVEAL_DELAY_EPOCHS
    from fugal_subnet.reveal_publisher import GcsUploader, delay_epochs_for
    reveal_uploader = GcsUploader(REVEAL_BUCKET) if REVEAL_BUCKET else None
    reveal_delay = REVEAL_DELAY_EPOCHS or delay_epochs_for(subtensor, netuid)
    # Root or follower in the v2 star (fugal_subnet/reveal_sync.resolve_role).
    from fugal_subnet import reveal_sync
    try:
        root_hotkey = _cfg.root_hotkey(on_finney)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    role = reveal_sync.resolve_role(
        finney=on_finney, root_flag=_cfg.ROOT_VALIDATOR, own_bucket=REVEAL_BUCKET,
        reference_bucket=_cfg.REFERENCE_BUCKET,
        explicit_dir=os.getenv("FUGAL_FRAME_BOOTSTRAP") or None,
        state_path=STATE_PATH, netuid=netuid, root_hotkey=root_hotkey,
    )
    if role is not None:
        logger.info("v2 role: %s", role.describe())
        if not role.root and role.bucket and not role.root_hotkey:
            logger.warning("The reveals this follower fetches are not verified against the "
                           "chain: set FUGAL_ROOT_HOTKEY to the root validator's hotkey.")
    # Verified heads go to the same store, by hash, with the reveal that lists
    # them (fugal_subnet/head_publisher.py). The root does by default; the
    # switch turns it on elsewhere or off. Operational: no score reads it.
    head_publisher = None
    if reveal_uploader is not None and _cfg.publish_heads_enabled(
            root=bool(role is not None and role.root), flag=_cfg.PUBLISH_HEADS):
        from fugal_subnet.commit_reveal import EPOCH_DIR as _epoch_dir
        from fugal_subnet.config import HEAD_MAX_BYTES as _head_max_bytes
        from fugal_subnet.head_publisher import HEADS_PREFIX, HeadPublisher
        head_publisher = HeadPublisher(_epoch_dir, netuid, reveal_uploader, reveal_delay,
                                       hold_epochs=_cfg.PUBLISH_HEADS_AFTER_EPOCHS,
                                       max_bytes=_head_max_bytes)
        logger.info("Verified heads publish to gs://%s/%s/%d/ by sha256: a seat holder's "
                    "with the reveal that lists it, any other head %d epoch(s) after it "
                    "first verified", REVEAL_BUCKET, HEADS_PREFIX, netuid,
                    _cfg.PUBLISH_HEADS_AFTER_EPOCHS)
    elif _cfg.PUBLISH_HEADS:
        logger.warning("FUGAL_PUBLISH_HEADS is set but FUGAL_REVEAL_BUCKET is not: heads "
                       "are not published")
    if reveal_uploader is not None:
        logger.info("Reveals publish to gs://%s after %d epoch(s)", REVEAL_BUCKET, reveal_delay)
    elif role is not None and not role.root:
        # Only the root's reveals are read by anyone (followers verify against
        # them), so a follower without a bucket is missing nothing.
        logger.info("FUGAL_REVEAL_BUCKET is unset: this validator's reveals stay on local "
                    "disk. Only the root's are read by other validators.")
    else:
        (logger.warning if live else logger.info)(
            "FUGAL_REVEAL_BUCKET is unset: epoch reveals stay on local disk and are "
            "not published. Set it to a public-read GCS bucket (docs/VALIDATOR_GUIDE.md)."
        )
    own_permit = False
    try:
        own_permit = bool(metagraph.validator_permit[my_uid])
        if not own_permit:
            logger.warning("This hotkey has NO validator permit — set_weights will "
                           "likely fail until it gains enough stake.")
    except (AttributeError, IndexError):
        pass
    try:
        from fugal_subnet.config import MIN_VALIDATOR_STAKE
        notice = axon_gate_notice(float(metagraph.S[my_uid]), own_permit, MIN_VALIDATOR_STAKE)
        if notice:
            (logger.warning if live else logger.info)(notice)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    # Publish this host's egress IP so miners can admit only validators at
    # their firewall (fugal_subnet/axon_allowlist.py). Never stops startup.
    from fugal_subnet.axon_allowlist import publish_validator_ip
    ip_ok, ip_note = publish_validator_ip(subtensor, wallet, netuid, metagraph, my_uid,
                                          VALIDATOR_IP)
    (logger.info if ip_ok or not live else logger.warning)(ip_note)

    logger.info("Loading benchmark pool...")
    benchmark_pool = load_all()
    logger.info("Benchmark pool: %d questions, pool_hash=%s",
                len(benchmark_pool), pool_hash(benchmark_pool)[:16])
    if not benchmark_pool:
        raise click.ClickException("Benchmark pool is empty — nothing to score")

    gold_answers = {q["question_id"]: q for q in benchmark_pool}
    approved_measurements = set(TEE_APPROVED_MEASUREMENTS)

    # Pinned price table — the consensus cost denominator. Loaded once so every
    # epoch prices against the same rates, and so a missing table fails at
    # startup rather than mid-epoch. Read from the SAME file the protocol
    # identity hashes (fugal_subnet/routing_data/models.json): data/models.json
    # is the reviewed copy, and the two are checked identical in CI, but a
    # checkout where they differed would have priced every proof from a table
    # the reveal's identity did not name (review 2026-11-05, F-043).
    from fugal_subnet.routing_protocol import price_table
    prices = price_table(1)
    # One global, sorted model space. Used for exploration targets and for
    # dedup indices, so every miner's routing vector is directly comparable.
    pool_models = sorted(prices)
    model_index = {m: i for i, m in enumerate(pool_models)}
    explore_size = max(1, int(round(SLICE_SIZE * EXPLORE_FRACTION)))
    logger.info("Price table: %d models; v1/v2 exploration quota %d questions/epoch "
                "(v3: per-miner exploration, logged with the v3 inputs)",
                len(prices), explore_size)

    # Every epoch input that differs by protocol version, loaded once at
    # startup and picked per epoch by the boundary-keyed version
    # (slicer.protocol_version_for_boundary): the pool and its gold, the price
    # table and the global model index space. v1 is exactly what was loaded
    # above. v2 is its own pool (loader.load_pool(2), pool_manifest_v2.json)
    # and its own price table and model list (routing_protocol, #142), so
    # from the flip every slice, exploration target, price and model index is
    # v2's, and the v2 comparison window is in the v2 model space from its
    # first epoch. Loaded only when an activation block is set, and refused at
    # startup in live if it cannot be, rather than at the flip.
    epoch_inputs = {1: (benchmark_pool, gold_answers, prices, pool_models, model_index)}
    if V2_ACTIVATION_BLOCK is not None:
        from fugal_subnet.benchmarks.loader import load_pool
        from fugal_subnet.routing_protocol import models_for_protocol, price_table
        try:
            pool_v2 = load_pool(2, strict=live)
            prices_v2 = price_table(2)
        except Exception as e:  # noqa: BLE001 - any failure here is a startup refusal
            if live:
                raise click.ClickException(f"protocol v2 inputs could not be loaded: {e}") from e
            logger.warning("protocol v2 inputs could not be loaded (mock): %s", e)
        else:
            models_v2 = models_for_protocol(2)
            epoch_inputs[2] = (
                pool_v2, {q["question_id"]: q for q in pool_v2}, prices_v2,
                models_v2, {m: i for i, m in enumerate(models_v2)},
            )
            logger.info("Protocol v2 inputs: pool %d questions (pool_hash=%s), %d models",
                        len(pool_v2), pool_hash(pool_v2)[:16], len(models_v2))
    # Protocol v3 inputs, loaded only when v3 is scheduled: its pool, its menu,
    # and the pinned embeddings file every head's odds are computed from. A
    # live validator that cannot load them refuses to start now, not at the
    # activation block.
    v3_embeddings = v3_index = epoch_inputs_v3b = None
    v3_status = "not scheduled"
    if v3_block is not None:
        from fugal_subnet import epoch_plan
        from fugal_subnet.benchmarks.loader import load_pool
        from fugal_subnet.routing_protocol import models_for_protocol, price_table
        try:
            pool_v3 = load_pool(3, strict=live)
            prices_v3 = price_table(3)
            models_v3 = models_for_protocol(3)
            if live and not getattr(_cfg, "EMBEDDINGS_SHA256", None):
                raise ValueError("config.EMBEDDINGS_SHA256 is not pinned, so the embeddings "
                                 "file could not be checked")
            # The pin, unless the pool itself is declared unpinned (never on
            # finney: FUGAL_POOL_UNPINNED is a consensus variable).
            from fugal_subnet.embeddings_file import embeddings_pin
            embeddings_path = os.getenv("FUGAL_EMBEDDINGS_FILE", "data/embeddings/pool_v3.npz")
            # Fetched here when missing or wrong (verified against the pin
            # before it replaces anything), so an upgrade needs no manual fetch.
            logger.info(validator_setup.ensure_embeddings(
                embeddings_path, embeddings_pin(), getattr(_cfg, "EMBEDDINGS_URL", None)))
            v3_embeddings = Embeddings.load(embeddings_path, embeddings_pin())
            missing = v3_embeddings.covers([q["question_id"] for q in pool_v3])
            if missing:
                raise ValueError(f"the embeddings file has no row for {len(missing)} pool "
                                 f"question(s), e.g. {missing[0]}")
            v3_index = epoch_plan.pool_index_v3(pool_v3)
        except Exception as e:  # noqa: BLE001 - any failure here is a startup refusal
            if live:
                raise click.ClickException(f"protocol v3 inputs could not be loaded: {e}") from e
            logger.warning("protocol v3 inputs could not be loaded (mock): %s", e)
            v3_status = f"failed:{e}"
        else:
            epoch_inputs[3] = (
                pool_v3, {q["question_id"]: q for q in pool_v3}, prices_v3,
                models_v3, {m: i for i, m in enumerate(models_v3)},
            )
            if menu_b_blk is not None:
                # v3 menu B: the same pool, menu A's entries plus the added
                # ones appended (routing_protocol.models_for_protocol).
                from fugal_subnet.routing_protocol import MENU_B
                models_v3b = models_for_protocol(3, MENU_B)
                epoch_inputs_v3b = (
                    pool_v3, epoch_inputs[3][1], price_table(3, MENU_B),
                    models_v3b, {m: i for i, m in enumerate(models_v3b)},
                )
            logger.info("Protocol v3 inputs: pool %d questions (pool_hash=%s), %d menu "
                        "entries, embeddings for %d questions", len(pool_v3),
                        pool_hash(pool_v3)[:16], len(models_v3), len(v3_embeddings.row))
            v3_status = f"ok:{len(pool_v3)}"
    # Protocol v4 inputs: the v4 mix is drawn on the v3 pool (pool_mix_v4.json
    # names pool_manifest_v3) and its heads read the same embeddings, so v4
    # needs v3's inputs; its prices and entries are the epoch's MenuState's,
    # read at each boundary (_v4_tables), never a table loaded here. The plan
    # is built once for the pool (epoch_plan.plan_v4, the one constructor).
    v4_plan = None
    v4_inputs_error = None
    if v4_block is not None:
        if 3 not in epoch_inputs:
            text = "protocol v4 is scheduled but the v3 pool and embeddings it runs on are not loaded"
            if live:
                raise click.ClickException(text)
            logger.warning("%s (mock)", text)
        else:
            from fugal_subnet import epoch_plan as _plan_v4
            try:
                v4_plan = _plan_v4.plan_v4(epoch_inputs[3][0])
                epoch_inputs[4] = (epoch_inputs[3][0], epoch_inputs[3][1], None, None, None)
                if v5_block is not None:
                    # Protocol v5 keeps v4's menu and its plan on the v3 pool
                    # (the epoch's nonce and committed slice); its exam,
                    # coverage and gold are the v5 pool's (v5_inputs below).
                    epoch_inputs[5] = epoch_inputs[4]
            except Exception as e:  # noqa: BLE001 - a v4 reason: refused only once v4 is due
                v4_inputs_error = f"the v4 plan cannot be built on the v3 pool ({e})"
    # Protocol v5 inputs: the v5 pool (loader.load_pool(5), verified against
    # pool_manifest_v5.json) and its routing embeddings (route_text, pinned by
    # config.EMBEDDINGS_V5_SHA256). A live validator with v5 scheduled refuses
    # to start without them; before the pin exists ("TBD") nothing can run v5.
    v5_inputs = None
    if v5_block is not None:
        try:
            v5_inputs = _v5_load_inputs(_cfg, live=live)
        except Exception as e:  # noqa: BLE001 - any failure here is a startup refusal
            if live:
                raise click.ClickException(f"protocol v5 inputs could not be loaded: {e}") from e
            logger.warning("protocol v5 inputs could not be loaded (mock): %s", e)
    role_name = ("none" if role is None else "ROOT" if role.root else
                 "follower (verified)" if role.root_hotkey else "follower")
    # Protocol v4's inputs and refusals, for the READY line and the menu
    # setup: refused at startup only from the first v4 epoch on; before it a
    # validator runs v3 and its READY line names what v4 still lacks.
    v4_status = _v4_status(
        cfg=_cfg, subtensor=subtensor, netuid=netuid, my_hotkey=my_hotkey, role=role,
        root_hotkey=root_hotkey, reveal_delay=reveal_delay, v4_block=v4_block,
        blocks_per_epoch=blocks_per_epoch_fn(EPOCH_INTERVAL), reveal_bucket=REVEAL_BUCKET,
        inputs_loaded=4 in epoch_inputs, inputs_error=v4_inputs_error,
        root_hotkeys=_cfg.root_hotkeys_v4(on_finney) if v4_block is not None else ())
    ready, ready_text = validator_setup.ready_line(
        entries=len(TEE_APPROVED_MEASUREMENTS), newest=setup_summary["newest"], v3=v3_status,
        archive=bool(ARCHIVE_ENDPOINTS), collateral=collateral_ok, role=role_name,
        v4=v4_status.found, v4_missing=v4_status.missing, v4_due=v4_status.due)
    (logger.info if ready and not v4_status.missing else logger.warning)(ready_text)


    state = load_state(MinerRecord)
    # Protocol v2 state: the seat and head owners (in the JSON state) and the
    # paired comparison window (an .npz beside it). Empty until the first v2
    # epoch; rebuilt from reveals like the frame.
    from fugal_subnet.config import CHAMPION_WINDOW_EPOCHS
    from fugal_subnet.v2_epoch import load_window, seat_from_dict
    v2_raw = state.get("v2") or {}
    v2_seat = seat_from_dict(v2_raw.get("seat"))
    # The copy rule's state (first commits, coldkey ownership, retirements),
    # published in every v2 reveal's v2_state under copy_rule.REVEAL_KEY.
    from fugal_subnet import copy_rule
    v2_copy = (copy_rule.CopyState.from_reveal(v2_raw[copy_rule.REVEAL_KEY])
               if v2_raw.get(copy_rule.REVEAL_KEY) else copy_rule.CopyState())
    v2_window = load_window(STATE_PATH + ".v2window.npz", CHAMPION_WINDOW_EPOCHS)
    # A seat with no window to back it (the .npz lost, or never written) must
    # not be judged on a partial window: hold, resume and ghosts continue, but
    # no first or take decision is made until the window is rebuilt from
    # reveals or has refilled itself (fail closed).
    v2_window_degraded = bool(v2_raw.get("window_degraded")) or bool(
        (v2_seat.champion is not None or v2_seat.ghosts) and not v2_window.slices
    )
    # The newest v2 epoch this validator did not observe and has not yet
    # re-anchored past (fugal_subnet/v2_sync.py); None when in step.
    v2_gap_until = v2_raw.get("gap_until")
    v2_gap_until = int(v2_gap_until) if v2_gap_until is not None else None
    # The epoch this validator last re-anchored at; only a reference epoch at
    # or after it can verify the re-anchor.
    v2_anchor_at = v2_raw.get("anchor_at")
    v2_anchor_at = int(v2_anchor_at) if v2_anchor_at is not None else None
    # What the seat and window were built under (None: protocol v2).
    v2_rules = v2_raw.get("rules")
    if v2_window_degraded:
        logger.error(
            "v2 seat state present but its comparison window is missing: no first or "
            "take decisions until it is rebuilt from the reference reveals or refills."
        )
    # Where state is rebuilt from when this validator has none: a follower's
    # reference reveals (fetched here, now, when it syncs them itself), or the
    # directory a root's operator named for a rebuild from its own reveals.
    blocks_per_epoch = blocks_per_epoch_fn(EPOCH_INTERVAL)
    fresh_start = not state["frame"] or (not v2_raw and not v2_window.slices)
    if role is not None and not role.root:
        now_epoch = epoch_index_for_block(subtensor.get_current_block(), blocks_per_epoch)
        # Everything on a first start (the frame is rebuilt from the whole
        # history); afterwards only what the comparison window reads.
        _sync_reference(role, netuid, required=live and fresh_start,
                        since=0 if fresh_start else now_epoch - CHAMPION_WINDOW_EPOCHS - 2,
                        budget_s=1800.0 if fresh_start else 120.0, subtensor=subtensor)
        rebuild_dir = role.reference_dir
    else:
        rebuild_dir = os.getenv("FUGAL_FRAME_BOOTSTRAP") or None
    reference_dir = role.reference_dir if role is not None and not role.root else None
    scoring_state = ScoringState(records=state["records"])
    prev_uids: list[int] = state["prev_uids"]
    prev_weights: list[float] = state["prev_weights"]
    last_epoch_index: int = state["last_epoch_index"]

    # The reference frame is subnet-level state accumulated over TIME, not over
    # the miner field — a miner's score must not move because other miners came
    # online or went dark (I4). Seeded from the shipped bootstrap prior so it is
    # well-defined at epoch 1 with zero samples.
    if state["frame"]:
        frame = ReferenceFrame.from_dict(state["frame"])
    elif rebuild_dir:
        # No local state. Rebuild from published reveals rather than starting
        # from the neutral prior, because a fresh frame is not a smaller version
        # of an established one — it is a DIFFERENT one. acc_best is the
        # denominator of every quality term, so a validator that starts cold
        # scores the whole field differently from its peers: 0.500 against 0.832
        # on measured data. Without this a lost disk, or simply a second
        # validator joining, diverges the subnet.
        frame = rebuild_from_reveals(rebuild_dir)
    else:
        # Live requires the shipped frame: every validator must value the
        # frontier from the same evidence, and "start cold" is a different
        # frontier from everyone else's (I1). Mock may start empty.
        try:
            frame = load_bootstrap(require=live)
        except FileNotFoundError as e:
            raise click.ClickException(str(e)) from e
        logger.warning(
            "Starting from the bootstrap frame with no accumulated history. If "
            "other validators have history, this one will score the field "
            "differently until it catches up. A root that lost its state sets "
            "FUGAL_FRAME_BOOTSTRAP to a directory of its published reveals to rebuild instead."
        )

    # Protocol v2 state, rebuilt the same way when this validator has none of
    # its own, or has a seat but lost its window: the seat, owners and burn-in
    # from the newest reveal's v2_state snapshot, the window replayed from the
    # last CHAMPION_WINDOW_EPOCHS reveals (fugal_subnet/v2_rebuild.py, trust
    # model in its docstring). Per-head v1 evidence is not rebuilt, as before;
    # only the E'' burn-in counter is restored, into fresh records.
    #
    # Exact only if the reveals reach the epoch just before the one this
    # validator resumes at. Published reveals lag by the commit-reveal hold,
    # so a rebuild from a peer's may be a few epochs behind: then the window
    # lacks those epochs and the seat is as of the newest reveal. That opens a
    # sync gap (below: `behind`), and verify-and-resync re-anchors on the
    # reference once it has published past it (fugal_subnet/v2_sync.py).
    # A rebuild whose newest reveal is protocol v4 waits for this node's first
    # v4 boundary: a v4 reveal is priced at the epoch's menu, which is read
    # only there (as verify-and-resync prices it).
    v4_rebuild_pending = v4_reanchor_now = False
    if (not v2_raw or v2_window_degraded) and not v2_window.slices and rebuild_dir:
        got = _rebuild_v2_state(rebuild_dir, epoch_inputs, config_module, epoch_inputs_v3b,
                                scoring_state, subtensor, blocks_per_epoch)
        if got == "v4":
            v4_rebuild_pending = True
        elif got is not None:
            v2_window, v2_seat, v2_rules, v2_copy, v2_window_degraded, gap = got
            v2_gap_until = gap if gap is not None else v2_gap_until

    _log_following(role, epoch_index_for_block(subtensor.get_current_block(), blocks_per_epoch),
                   len(approved_measurements), v2_seat)

    # Protocol v4's menu (fugal_subnet/menu_publish.py): read at every v4
    # boundary from the root's commitment, published by the root, audited by
    # the others. Only when a v4 block is scheduled, which no release does yet.
    menu_node = menu_publisher = None
    if v4_block is not None:
        menu_node, menu_publisher = _v4_menu_setup(
            cfg=_cfg, subtensor=subtensor, netuid=netuid, my_hotkey=my_hotkey, role=role,
            root_hotkey=root_hotkey, reveal_delay=reveal_delay, live=live,
            blocks_per_epoch=blocks_per_epoch, v4_block=v4_block, state_path=STATE_PATH,
            reveal_bucket=REVEAL_BUCKET, root_hotkeys=_cfg.root_hotkeys_v4(on_finney),
            status=v4_status, wallet=wallet)
    # Protocol v4's published set (fugal_subnet/memo_sync.py): held beside the
    # state, brought forward at every v4 boundary, folded from each reveal
    # this validator writes; the root publishes the snapshots its reveals name.
    pub_holder = pub_publisher = None
    if v4_block is not None and 4 in epoch_inputs:
        pub_holder, pub_publisher = _v4_published_setup(
            subtensor=subtensor, netuid=netuid, pool=epoch_inputs[4][0],
            first_epoch=v4_block // blocks_per_epoch, blocks_per_epoch=blocks_per_epoch,
            role=role, my_hotkey=my_hotkey, root_hotkey=root_hotkey,
            root_hotkeys=_cfg.root_hotkeys_v4(on_finney), reference_dir=reference_dir,
            reveal_bucket=REVEAL_BUCKET, cfg=_cfg, state_path=STATE_PATH, menu_node=menu_node,
            other_pool_ids=_v5_pool_ids(v5_inputs))
    # Protocol v5 (fugal_subnet/v5_node.py): its state beside the validator's,
    # and, on a follower, the reader of the root's published exam specs.
    v5_node = None
    if v5_block is not None:
        v5_node = _v5_setup(subtensor=subtensor, netuid=netuid, role=role, cfg=_cfg,
                            on_finney=on_finney, blocks_per_epoch=blocks_per_epoch,
                            reveal_bucket=REVEAL_BUCKET, state_path=STATE_PATH)
    if check:
        # Every startup check has run; nothing has been scored, committed or set.
        print(ready_text, flush=True)
        # The chain connection is closed here, in the main thread, not left to
        # async_substrate_interface's __del__ at interpreter shutdown, which
        # joins a thread that never finishes: on a real node `--check` printed
        # READY and then never exited (the local-chain v4 rehearsal).
        _close_chain(subtensor)
        raise SystemExit(0 if ready and not v4_status.missing else 1)

    epoch_failed = False
    while True:
        try:
            epoch_failed = False
            current_block = subtensor.get_current_block()
            epoch_index = _epoch_to_run(current_block, blocks_per_epoch, last_epoch_index, once)
            if epoch_index is None:
                time.sleep(30)
                continue
            if epoch_index <= last_epoch_index:
                logger.warning("Epoch %d already processed — running again (--once)", epoch_index)

            # Everything from the boundary block, so every peer and every retry
            # of this epoch derives the same nonce and scores it under the same
            # rules (v1 or v2).
            boundary_block, block_hash, epoch_id, protocol_version = _epoch_boundary(
                subtensor, epoch_index, blocks_per_epoch, V2_ACTIVATION_BLOCK, v3_block,
                v4_activation_block=v4_block, v5_activation_block=v5_block,
            )

            timer = EpochTimer()
            v5_commit = None
            metagraph = subtensor.metagraph(netuid)
            logger.info("=== Epoch %s (boundary block %d, hash %s, protocol v%d) ===",
                        epoch_id, boundary_block, block_hash[:16], protocol_version)

            # --- v4: THE EPOCH'S MENU, from the root's commitment at the boundary ---
            # MenuState(E) from the root's commitment for E - MENU_LAG, read
            # pinned at this boundary block (fugal_subnet/menu_publish.py). A
            # failed chain read fails the epoch, which is retried, as for head
            # commitments; a menu this validator cannot check means it can
            # verify no proof this epoch, and it says so.
            v4_menu = None
            if protocol_version >= 4:
                stop = _v4_due_refusal(menu_node, live)
                if stop is not None:
                    raise SystemExit(stop)
                if menu_node is None:
                    raise RuntimeError(f"epoch {epoch_id} is protocol v4 but the v4 menu was "
                                       "not set up at startup")
                v4_menu = menu_node.at_boundary(epoch_index)
                logger.info("Menu for %s: %s, %s", epoch_id, v4_menu.status,
                            (v4_menu.menu_hash or "none")[:16])
                if getattr(menu_node, "mirror", None) is not None and v4_menu.ok:
                    menu_node.mirror.mirror(v4_menu, menu_node.lists)
                v4_join_audit = None
                if not (role is not None and role.root):
                    v4_join_audit = _v4_join_audit(
                        subtensor, netuid, menu_node=menu_node, v4_menu=v4_menu,
                        blocks_per_epoch=blocks_per_epoch,
                        root_hotkey=(_cfg.root_hotkeys_v4(on_finney) or root_hotkey),
                        reference_dir=reference_dir,
                        bucket=(role.bucket if role is not None and role.bucket
                                else getattr(_cfg, "REFERENCE_BUCKET", "") or None))
            # The epoch's v4 prices, entries, categories and targets: the
            # menu's (prices, entries) and the pinned v4 mix's (categories,
            # targets). The exploration shares once per epoch, from the menu.
            v4_tables = _v4_tables(v4_menu) if protocol_version >= 4 else None
            v4_shares = None
            if v4_tables is not None and v4_menu.ok:
                from fugal_subnet import epoch_plan as _plan
                v4_shares = _plan.explore_shares_v4(v4_menu.state)
                if v4_rebuild_pending:
                    v4_rebuild_pending = False
                    got = _rebuild_v2_state(rebuild_dir, epoch_inputs, config_module,
                                            epoch_inputs_v3b, scoring_state, subtensor,
                                            blocks_per_epoch, v4_tables=v4_tables)
                    if got is not None and got != "v4":
                        v2_window, v2_seat, v2_rules, v2_copy, v2_window_degraded, gap = got
                        v2_gap_until = gap if gap is not None else v2_gap_until
                        if gap is not None and v2_raw:
                            # A node that had state (its window lost): the
                            # rebuild is the reference's newest reveal, behind
                            # this epoch. Its own reveals before now predate the
                            # rebuild and verify nothing (anchor_at), and the
                            # sync step below re-anchors at once on that reveal,
                            # replaying the node's own epochs since from their
                            # records, so its state is the reference's from this
                            # epoch on.
                            v2_anchor_at, v4_reanchor_now = epoch_index, True
                        save_state(scoring_state.records, prev_uids, prev_weights,
                                   last_epoch_index, frame,
                                   v2={"seat": seat_to_dict(v2_seat),
                                       copy_rule.REVEAL_KEY: v2_copy.to_reveal(),
                                       "window_degraded": v2_window_degraded,
                                       "gap_until": v2_gap_until, "anchor_at": v2_anchor_at,
                                       "rules": v2_rules},
                                   v2_window=v2_window, published=pub_holder)

            # --- v2: VERIFY AND RESYNC against a reference validator ---
            # A missed v2 epoch makes this validator's seat state differ from
            # its peers'; once a reference validator's reveal is published,
            # compare and, if they differ, re-anchor on it (v2_sync). Durable
            # at once: a re-anchor is a state correction, not an epoch, and a
            # retry recomputes the same one.
            v2_audit = None
            v4_view = None
            if protocol_version >= 2:
                # A follower that fetches the root's reveals itself brings
                # them up to date first. Never the epoch's failure (I6).
                _sync_reference(role, netuid, required=False, budget_s=120.0,
                                since=epoch_index - CHAMPION_WINDOW_EPOCHS - 2,
                                subtensor=subtensor)
                if protocol_version >= 4:
                    # --- v4: THE PUBLISHED SET, as the chain says at this boundary ---
                    # Advanced by the root's reveal commitments read pinned at
                    # every boundary through this one (fugal_subnet/memo_sync.py):
                    # the same set on every node and TD, whether or not the root
                    # is up. The epoch's plan and its memorisation verdicts both
                    # read it, and so does a re-anchor's replay of the epochs
                    # before this one (below), which is why it is read first.
                    v4_view = _v4_epoch_view(pub_holder, epoch_index)
                synced = _v2_sync_step(
                    reanchor_now=v4_reanchor_now,
                    epoch_index=epoch_index, last_epoch_index=last_epoch_index,
                    first_v2_epoch=V2_ACTIVATION_BLOCK // blocks_per_epoch,
                    reference_dir=reference_dir,
                    scoring_state=scoring_state, window=v2_window, seat=v2_seat,
                    copy_state=v2_copy, window_degraded=v2_window_degraded,
                    gap_until=v2_gap_until, anchor_at=v2_anchor_at,
                    prices=(v4_tables.prices if v4_tables is not None
                            else epoch_inputs[protocol_version][2]),
                    bench_category=(v4_tables.bench_category if v4_tables is not None else
                                    getattr(config_module,
                                            f"BENCHMARK_CATEGORY_V{protocol_version}", None)),
                    tables=_rebuild_tables(epoch_inputs, config_module, epoch_inputs_v3b,
                                           v4_tables=v4_tables),
                    # v4: a re-anchor replays this validator's own epochs
                    # since the reference epoch (v4_replay), never keeps them.
                    # Each replayed epoch j runs with the memorisation memo
                    # the epoch loop built for it: the published set as of j
                    # (read from the chain, never folded here), its plan on
                    # the epoch's pool object, and j's recorded nonce.
                    replay=(_v4_replayer(epoch_inputs[protocol_version][1], v3_embeddings,
                                         _v4_replay_span(reveal_delay),
                                         memo_for=_v4_memo_for(
                                             pub_holder, epoch_inputs[protocol_version][0]),
                                         holder_bytes=_v4_head_publication(netuid))
                            if protocol_version >= 4 else None),
                )
                v2_audit = synced.audit
                v4_reanchor_now = False
                if synced.changed:
                    (scoring_state, v2_window, v2_seat, v2_copy, v2_window_degraded,
                     v2_gap_until, v2_anchor_at) = synced.state
                    v2_rules = synced.rules if synced.reanchored else v2_rules
                    save_state(scoring_state.records, prev_uids, prev_weights,
                               last_epoch_index, frame,
                               v2={"seat": seat_to_dict(v2_seat),
                                   copy_rule.REVEAL_KEY: v2_copy.to_reveal(),
                                   "window_degraded": v2_window_degraded,
                                   "gap_until": v2_gap_until, "anchor_at": v2_anchor_at,
                               "rules": v2_rules},
                               v2_window=v2_window, published=pub_holder)
                _log_following(role, epoch_index, len(approved_measurements), v2_seat)

            # --- COMMITMENTS, pinned to the boundary block ---
            # Read NOW, while the boundary block is seconds old, not after the
            # collection wait and the 120 s query: a non-archive node serves
            # roughly the last 300 blocks, and at ~boundary+250 the pinned read
            # started failing with 50 blocks to spare. The pin makes the answer
            # identical whenever it is read, so reading early costs nothing.
            # In live a failed read is the epoch's failure (retried by the
            # outer loop), never an empty map: with REQUIRE_COMMITMENT an empty
            # map rejects every proof, and the epoch was then marked processed
            # while peers that read in time set weights — a fork with nothing
            # in the reveal to explain it (found 2026-11-02).
            timer.start_phase("commitments")
            chain_commitments = read_commitments_or_fail(
                subtensor, netuid, boundary_block,
                required=live and REQUIRE_COMMITMENT,
            )

            # --- SLICE ---
            # This epoch's inputs, by the boundary-keyed version. The miner
            # selects with the same epoch_plan functions on the same versioned
            # pool and model list, or every proof fails on questions_hash.
            timer.start_phase("slice")
            if protocol_version not in epoch_inputs:
                raise RuntimeError(
                    f"epoch {epoch_id} is protocol v{protocol_version} but its inputs "
                    "were not loaded at startup"
                )
            benchmark_pool, gold_answers, prices, pool_models, model_index = (
                epoch_inputs[protocol_version]
            )
            if v4_tables is not None:
                # v4: the menu's entries and prices (none when this validator
                # has no menu: it then checks no proof, below).
                prices, pool_models, model_index = (
                    v4_tables.prices, v4_tables.models, v4_tables.model_index)
            # v3 menu B, from its block (the miner picks it the same way).
            from fugal_subnet.routing_protocol import menu_for_epoch
            epoch_menu = (menu_for_epoch(protocol_version, boundary_block, menu_b_blk)
                          if protocol_version < 4 else None)
            if epoch_menu == "b":
                if epoch_inputs_v3b is None:
                    raise RuntimeError(f"epoch {epoch_id} is on v3 menu B but its inputs "
                                       "were not loaded at startup")
                benchmark_pool, gold_answers, prices, pool_models, model_index = epoch_inputs_v3b
            v4_epoch_plan = None
            if protocol_version >= 4:
                # v4's own compiled slice, the one its pinned mix is built for,
                # drawn by the epoch's plan: plan_v4 with the published set as
                # of this epoch, on the pool object every draw of the epoch
                # reads. Without that set (a reveal the chain names could not
                # be had) this validator checks no v4 proof (below); the slice
                # is then the plan's without it, committed only.
                from fugal_subnet import epoch_plan as _ep
                v4_epoch_plan = (_ep.plan_v4(benchmark_pool, published=v4_view, epoch=epoch_index)
                                 if v4_view is not None else None)
                plan = _epoch_slice_v4(epoch_id, block_hash, benchmark_pool, gold_answers,
                                       v4_epoch_plan or v4_plan, _cfg.SLICE_SIZE_V4)
            else:
                plan = _epoch_slice(protocol_version, epoch_id, block_hash, benchmark_pool,
                                    gold_answers, pool_models, SLICE_SIZE, explore_size)
            nonce, questions, question_ids = plan.nonce, plan.questions, plan.question_ids
            # v4: the epoch's challenge list (FAST_CROWN_V4.md 2: the slice
            # first), drawn once by the epoch's plan for every hotkey's check.
            v4_challenge_list = (
                v4_epoch_plan.challenge_list(nonce, benchmark_pool, _cfg.SLICE_SIZE_V4)
                if protocol_version == 4 and v4_epoch_plan is not None else [])
            # v5: a node with no state or a stale one (a fresh follower, a
            # restart after a gap) first rebuilds it from the root's reveals.
            if protocol_version >= 5 and v5_node is not None:
                _v5_resync(v5_node, epoch_index=epoch_index,
                           first_epoch=v5_block // blocks_per_epoch, reference_dir=reference_dir)
            # v5: the exam the TDs answer this epoch (the root's published
            # spec), its costs, and why its items do not count here (or None).
            v5_exam = (v5_node.exam_for(epoch_index) if protocol_version >= 5
                       and v5_node is not None else (None, {}, "not a v5 epoch"))
            if protocol_version >= 5:
                logger.info("Exam for %s: %s%s", epoch_id,
                            v5_exam[0].spec_hash()[:16] if v5_exam[0] is not None
                            else "none (coverage only)",
                            f"; void here: {v5_exam[2]}" if v5_exam[0] is not None
                            and v5_exam[2] else "")
            expected_questions_hash = plan.questions_hash
            expected_question_ids = set(question_ids)
            explore_map, slice_gold = plan.explore_map, plan.slice_gold
            explore_shares = plan.explore_shares
            logger.info("Slice: %d questions, hash=%s...",
                        len(questions), expected_questions_hash[:16])

            # --- COMMIT ---
            timer.start_phase("commit")
            commitment = commit_epoch(epoch_id, questions, block_hash)
            logger.info("Committed: %s", commitment.commit_hash)

            # --- WAIT FOR THE COLLECTION BLOCK ---
            # A miner cannot start benchmarking until this epoch's boundary
            # block exists, because that block's hash is what picks the
            # questions. Querying at the boundary therefore asks before any
            # proof can exist: with real models on a 300-question slice the
            # miner needs minutes, the validator would collect nothing, and —
            # worse than merely failing — it marks the epoch done and never
            # returns to it. Every epoch, forever, logged as "no valid proofs"
            # as though the miners were at fault.
            timer.start_phase("await_collect")
            collect_block = collect_block_for_epoch(
                epoch_index, blocks_per_epoch, EPOCH_COLLECT_FRACTION,
            )
            wait_for_block(subtensor, collect_block)

            # Read the metagraph AT the collection block, not at "now". Both
            # the block and the offset are derived identically by every honest
            # validator, so they query the same axons at the same chain state
            # and any difference in what they collect is a real difference
            # rather than a timing artefact.
            metagraph, metagraph_pinned = metagraph_at(subtensor, netuid, collect_block, metagraph)
            _v4_require_pinned_metagraph(protocol_version, metagraph_pinned)

            # --- QUERY MINERS FOR PROOFS ---
            timer.start_phase("query")
            nonce_hex = nonce.hex()
            synapse = FugalProofSynapse(epoch_id=epoch_id, nonce=nonce_hex)
            n_neurons = int(metagraph.n)
            logger.info("Querying %d miners for proofs at block %d (boundary %d + %d)...",
                        n_neurons, collect_block, boundary_block,
                        collect_block - boundary_block)
            responses = dendrite.query(
                metagraph.axons,
                synapse,
                timeout=120,
            )

            # --- VERIFY PROOFS ---  (commitments were read at the boundary, above)
            timer.start_phase("verify_proofs")
            verified_proofs: dict[int, BenchmarkProof] = {}
            verified_heads: dict[int, bytes] = {}
            head_hashes: dict[int, str] = {}
            commit_blocks: dict[int, float] = {}
            raw_commit_blocks: dict[int, int] = {}   # v5 (v0.8.1): the commitment's block
            n_invalid = 0
            # Kept apart from n_invalid on purpose. A proof this validator
            # could not check is not a proof it rejected, and the difference is
            # the difference between an outage and an accusation.
            n_unverifiable = 0
            outcomes: dict[int, str] = {}
            v4_expected: dict = {}         # uid -> tee/verify.V4Expected (protocol v4)

            # One ceiling for the whole phase. Every fetch is already bounded
            # individually; this bounds their SUM, which is the quantity I6
            # actually cares about -- the loop below is serial, so 256
            # individually-bounded fetches still overrun the epoch and miss the
            # weight-setting window they exist to hit.
            collateral_budget = CollateralBudget(
                TEE_COLLATERAL_EPOCH_BUDGET, expected_proofs=len(responses))

            # Verify in an order derived from the epoch nonce rather than in UID
            # order. This matters ONLY when the budget runs out, and then it
            # matters a lot: whoever is last in line goes unverified, and in UID
            # order that is the same miners every epoch -- a permanent
            # disadvantage for high UIDs that no miner caused and none could
            # escape. The nonce comes from the epoch id and a block hash, so it
            # is identical on every validator (no new I1 divergence) and no
            # miner can influence it (I4), while the disadvantage it allocates
            # rotates each epoch instead of settling on the same miners.
            order = verification_order(nonce_hex, len(responses))

            checks = _EpochChecks(
                boundary_block=boundary_block,
                require_commitment=REQUIRE_COMMITMENT,
                approved_measurements=approved_measurements,
                questions_hash=expected_questions_hash,
                nonce_hex=nonce_hex,
                slice_gold=slice_gold,
                question_ids=expected_question_ids,
                explore_map=explore_map,
                mock=mock,
                collateral_budget=collateral_budget,
                protocol_version=protocol_version,
                menu=epoch_menu,
            )
            for uid in order:
                hotkey_ss58 = metagraph.hotkeys[uid] if uid < len(metagraph.hotkeys) else ""
                if (protocol_version >= 4 and not (v4_menu is not None and v4_menu.ok
                                                   and v4_epoch_plan is not None)
                        and _offers_proof(responses[uid])):
                    # No menu, no namespace to check a v4 proof against; no
                    # published set, no plan to check it on: not a
                    # rejection, a proof this validator could not check.
                    outcomes[uid] = "unverifiable"
                    n_unverifiable += 1
                    continue
                uid_checks = checks
                if protocol_version >= 5 and hotkey_ss58 and _offers_proof(responses[uid]):
                    # v5: this hotkey's coverage cells on the epoch's menu and
                    # the exam cells the root's spec assigns it (I8).
                    try:
                        if v5_inputs is None:
                            raise ValueError("the v5 pool and embeddings are not loaded")
                        exp5 = _v5_expected(
                            v4_menu.state, epoch_index=epoch_index, nonce=nonce,
                            pool=v5_inputs.pool, hotkey=hotkey_ss58, spec=v5_exam[0],
                            plan=v4_epoch_plan)
                    except Exception as e:  # noqa: BLE001 - our expectation, not the miner's proof
                        logger.error("UID %d: no v5 expectation could be drawn (%s: %s); its "
                                     "proof is unchecked", uid, type(e).__name__, e)
                        outcomes[uid] = "unverifiable"
                        n_unverifiable += 1
                        continue
                    v4_expected[uid] = exp5.v4
                    uid_checks = _v5_uid_checks(checks, exp5, slice_gold, v5_inputs.gold)
                elif protocol_version >= 4 and hotkey_ss58 and _offers_proof(responses[uid]):
                    # v4: this hotkey's exploration and measurement on the
                    # epoch's menu, and the gold for all of them.
                    try:
                        # The epoch's plan (epoch_plan.plan_v4 with the
                        # published set as of E - 2, the one way to build
                        # it) and this epoch's menu's exploration shares
                        # (once per epoch, at the boundary) are the same for
                        # every hotkey.
                        exp = _v4_expected(
                            v4_menu.state, epoch_index=epoch_index, nonce=nonce,
                            pool=benchmark_pool, scored_ids=question_ids, hotkey=hotkey_ss58,
                            plan=v4_epoch_plan, shares=v4_shares,
                            challenge_ids=[q["question_id"] for q in v4_challenge_list])
                    except Exception as e:  # noqa: BLE001 - our expectation, not the miner's proof
                        logger.error("UID %d: no v4 expectation could be drawn (%s: %s); its "
                                     "proof is unchecked", uid, type(e).__name__, e)
                        outcomes[uid] = "unverifiable"
                        n_unverifiable += 1
                        continue
                    v4_expected[uid] = exp
                    uid_checks = checks._replace(
                        explore_map=exp.exploration, v4=exp,
                        slice_gold=_v4_gold(slice_gold, gold_answers, exp))
                elif protocol_version >= 3 and hotkey_ss58 and _offers_proof(responses[uid]):
                    # v3: this miner's own exploration set (the shared pairs and
                    # the ones drawn for its hotkey), and the gold for them.
                    from fugal_subnet import epoch_plan as _plan
                    mine = _plan.exploration_plan(3, nonce, benchmark_pool, expected_question_ids,
                                                  pool_models, 0, hotkey=hotkey_ss58,
                                                  index=v3_index, shares=explore_shares)
                    uid_checks = checks._replace(
                        explore_map=mine,
                        slice_gold={**slice_gold,
                                    **{q: gold_answers[q] for q in mine if q in gold_answers}})
                checked = _check_response(
                    uid, responses[uid], hotkey_ss58,
                    chain_commitments.get(hotkey_ss58, ("", 0)),
                    registration_block_of(metagraph, uid), uid_checks,
                )
                if (checked.status == "valid" and protocol_version >= 3
                        and (why_not := _head_within_bounds(checked.head_bytes,
                                                            protocol_version))):
                    logger.warning("UID %d proof rejected: %s", uid, why_not)
                    checked = _Checked("invalid", checked.seniority,
                                       commit_block=checked.commit_block)
                outcomes[uid] = checked.status
                if checked.seniority is not None:
                    commit_blocks[uid] = checked.seniority
                if checked.commit_block is not None:
                    raw_commit_blocks[uid] = checked.commit_block
                if checked.status == "invalid":
                    n_invalid += 1
                elif checked.status == "unverifiable":
                    n_unverifiable += 1
                elif checked.status == "valid":
                    verified_proofs[uid] = checked.proof
                    verified_heads[uid] = checked.head_bytes
                    # Keyed on the hash the attestation forced to be true, not
                    # the value the miner asserted over the axon.
                    head_hashes[uid] = checked.proof.weights_hash

            # --- v4: THE FAILURE BOUND, across the epoch's verified proofs ---
            # A proof with more than FAIL_MAX_BP_V4 of its exploration calls
            # failed is not counted, outside an outage epoch (picks_v4); the
            # v1-v3 comparison with the scored calls is not made under v4,
            # where a proof need not carry any.
            v4_failure = None
            if protocol_version >= 4 and v4_menu is not None and v4_menu.ok and verified_proofs:
                v4_failure = _v4_failure_bound(verified_proofs, metagraph, v4_menu.state)
                for uid in sorted(verified_proofs):
                    hk = verified_proofs[uid].hotkey
                    if hk in v4_failure.refused:
                        failed, calls = v4_failure.refused[hk]
                        logger.warning("UID %d proof not counted: %d of its %d exploration calls "
                                       "failed, more than the failure bound, outside an outage "
                                       "epoch", uid, failed, calls)
                        del verified_proofs[uid], verified_heads[uid], head_hashes[uid]
                        outcomes[uid] = "invalid"
                        n_invalid += 1
                if v4_failure.outage:
                    logger.warning("Epoch %s is an outage epoch (the median operator failed "
                                   "more than the outage rate of its exploration calls): the "
                                   "failure bound is waived", epoch_id)


            # Diagnostics only: keep the verified heads on private disk. Feeds
            # no score, hash, reveal or fingerprint, and can't fail the epoch.
            _archive_heads(epoch_id, verified_heads, head_hashes, metagraph)
            # Operational only: keep them for publication (with a reveal that
            # lists them, when they are due). Same guarantees as the archive.
            if head_publisher is not None:
                head_publisher.spool(verified_heads, head_hashes, epoch_index)
            # Diagnostics only: who was reached, how long it took, and what
            # became of each proof (the epoch log keeps it per UID).
            collection, collection_line = _collection_report(
                responses, list(metagraph.hotkeys), chain_commitments, outcomes)
            logger.info(collection_line)

            if not verified_proofs:
                logger.warning("No valid proofs received, skipping epoch")
                epoch_log = EpochLog(
                    epoch_id=epoch_id, block_hash=block_hash,
                    timestamp=time.time(), n_questions=len(questions),
                    n_miners_queried=n_neurons, n_heads_valid=0,
                    n_heads_invalid=n_invalid,
                    n_heads_unverifiable=n_unverifiable,
                    commit_hash=commitment.commit_hash,
                    collection=collection,
                    anomalies=(
                        ["no_valid_proofs"]
                        + ([f"menu_unavailable: {v4_menu.detail if v4_menu else 'no menu'}"]
                           if protocol_version >= 4 and not (v4_menu and v4_menu.ok) else
                           ["published_set_unavailable"]
                           if protocol_version >= 4 and v4_epoch_plan is None else
                           ["collateral_unavailable"] if n_unverifiable else [])
                        + ([] if metagraph_pinned else
                           [f"metagraph_unpinned: the read at block {collect_block} failed"])
                    ),
                    duration_s=timer.total_s,
                )
                write_epoch_log(epoch_log)
                # A v2 epoch this validator scored nothing in (a local
                # collateral outage, or miners it could not reach) may be one
                # its peers scored: a gap until it is verified again.
                burned = None
                if protocol_version >= 2:
                    v2_gap_until = epoch_index
                    # In v2 a seat pays only while its head is live, and with
                    # no proof no head is: the miner share burns. Keeping the
                    # last vector instead paid the v1 winners after the flip
                    # until the first v2 proof verified (Fable review, item 6).
                    burned = _burn_epoch_weights(
                        subtensor, wallet, netuid, my_uid, boundary_block, metagraph, live,
                        prev_uids, prev_weights, epoch_index, blocks_per_epoch,
                        protocol_version=protocol_version)
                if protocol_version >= 4:
                    _v4_publish_no_proof(
                        epoch_index=epoch_index,
                        root=role is not None and role.root, uploader=reveal_uploader,
                        reveal_delay=reveal_delay, netuid=netuid, menu=menu_publisher,
                        snapshot=pub_publisher,
                        commit=lambda data: subtensor.set_commitment(wallet, netuid, data)[0])
                last_epoch_index = epoch_index
                # The v2 seat is not advanced on an epoch with no proofs at
                # all: there is nothing to compare, and the champion's ghost
                # window is counted in epochs, so it still expires on time.
                # The burn vector is the next epoch's cap base, rebound only
                # here, at the commit point (tests/test_epoch_atomicity.py).
                prev_uids, prev_weights = burned or (prev_uids, prev_weights)
                save_state(scoring_state.records, prev_uids, prev_weights,
                           last_epoch_index, frame,
                           v2={"seat": seat_to_dict(v2_seat), copy_rule.REVEAL_KEY: v2_copy.to_reveal(),
                               "window_degraded": v2_window_degraded,
                           "gap_until": v2_gap_until, "anchor_at": v2_anchor_at,
                               "rules": v2_rules},
                           v2_window=v2_window, published=pub_holder)
                if once:
                    break
                time.sleep(60)
                continue

            logger.info("Verified %d proofs", len(verified_proofs))

            # --- REFERENCE FRAME ---
            # Exploration samples from every verified proof, pooled. The frame
            # accumulates over TIME: this epoch's samples are one decayed
            # contribution to a long-running estimate, so the reference a miner
            # is measured against does not lurch when the field size changes.
            timer.start_phase("frame")
            coldkeys = getattr(metagraph, "coldkeys", None) or []
            if protocol_version >= 4:
                # Each hotkey explores its own questions under v4: the frame's
                # mix diagnostics read all of them.
                explore_map = _v4_explored(v4_expected)
            fi = _frame_input(frame, verified_proofs, coldkeys, protocol_version, questions,
                              explore_map, gold_answers)
            observations, coldkey_of, suspicious = fi.observations, fi.coldkey_of, fi.suspicious
            samples, sample_weights, mix, bench_of = (
                fi.samples, fi.sample_weights, fi.mix, fi.bench_of)

            # EPOCH-LOCAL from here to save_state. `frame`, `scoring_state`,
            # `prev_uids` and `prev_weights` are the validator's durable state
            # and are rebound only once the epoch is written. Anything that
            # raised between scoring and save_state used to land in the outer
            # handler with the loop-level names already advanced, so the retry
            # of the SAME epoch accumulated every miner's evidence twice,
            # applied every miss twice and decayed the frame twice — and then
            # persisted it (found 2026-11-02). tests/test_epoch_atomicity.py
            # holds this shape.
            epoch_frame = accumulate_exploration(frame, samples, weights=sample_weights)
            ref_model, acc_best = best_model(epoch_frame, prices)
            logger.info(
                "Reference frame: %d samples this epoch, best model %s "
                "(acc=%.3f, %.0f trials)",
                len(samples), ref_model, acc_best, epoch_frame.trials.get(ref_model, 0.0),
            )
            # The constant-policy frontier for THIS slice: what any single model,
            # or any random mixture of models, buys at each price. Prompt tokens
            # come from the pool, so the frontier is identical for every miner
            # and on every validator. See fugal_subnet/frontier.py.
            slice_prompt_tokens = sum(
                question_input_tokens(q.get("prompt", "")) for q in questions
            )
            frontier = build_frontier(
                epoch_frame, prices, slice_prompt_tokens, len(questions),
                FRAME_DEFAULT_COMPLETION_TOKENS,
            )
            logger.info(
                "Frontier: %d hull models %s, max accuracy %.3f, confidence %.2f "
                "(least-observed hull model has %.0f trials)",
                len(frontier.models), list(frontier.models), frontier.max_accuracy,
                frontier.confidence, frontier.least_trials,
            )

            # --- SCORE FROM PROOFS ---
            timer.start_phase("score")
            epoch_scores: dict[int, HeadScore] = {}
            # Diagnostics only (epoch log + the line below): feeds no score.
            epoch_n_failed: dict[int, int] = {}
            for uid, proof in verified_proofs.items():
                score = _score_proof(proof, slice_gold, prices, epoch_frame, ref_model)
                epoch_scores[uid] = score
                epoch_n_failed[uid] = count_failed_scored(proof)
                cpq = score.total_head_cost / max(1, score.n_scored)
                logger.info(
                    "  UID %d: acc=%.3f cost=$%.4f ($%.6f/q, frontier %.3f there) "
                    "ref=$%.4f (%d/%d correct, %d failed)",
                    uid, score.accuracy, score.total_head_cost, cpq,
                    frontier.accuracy_at(cpq), score.total_oracle_cost,
                    score.n_correct, score.n_scored, epoch_n_failed[uid],
                )

            # --- DEDUP ---
            timer.start_phase("dedup_score_weight")
            routing_decisions = {
                uid: _extract_routing_decisions(
                    proof, model_index,
                    keep=expected_question_ids if protocol_version >= 4 else None)
                for uid, proof in verified_proofs.items()
            }
            # The parameter clause needs the heads themselves; they were
            # validated in verify_proof and are loaded here only for distance.
            head_params = {}
            for uid, head_bytes in verified_heads.items():
                try:
                    head_params[uid] = load_head_from_npz(head_bytes)
                except (ValueError, KeyError, TypeError):
                    continue
            # v1's copy detection; read by no v2+ decision, and not run under
            # v4, whose precedence rule refuses near-copies by measured
            # improvement (SEAT_MINIMAL_V4.md 2.1 item 21).
            dupes = (set() if protocol_version >= 4 else
                     find_duplicates(routing_decisions, commit_blocks, head_params=head_params))
            if dupes:
                logger.info("Dedup disqualified: %s", dupes)

            # --- EVIDENCE ACCUMULATION ---
            epoch_state = update_scores(
                scoring_state, epoch_scores, head_hashes,
                frontier=frontier,
                hotkeys={
                    uid: metagraph.hotkeys[uid]
                    for uid in verified_proofs
                    if uid < len(metagraph.hotkeys)
                },
                n_questions=len(questions),
                pool_size=len(benchmark_pool),
                protocol_version=protocol_version,
            )

            # --- WEIGHTS ---
            for uid in epoch_scores:
                rec = epoch_state.records.get(uid)
                if rec is not None:
                    logger.info(
                        "  UID %d scored: lcb=%.3f evidence $%.6f/q, frontier %.3f "
                        "there, headroom %.3f, composite %.5f",
                        uid, rec.wilson_lcb, rec.cost_per_question,
                        rec.reference_accuracy, rec.headroom, rec.composite_score,
                    )
            # Epsilon rule, now that this epoch's composite scores exist: a
            # head that routes like a senior one must beat it or it is a copy.
            burn_uid = resolve_burn_uid(subtensor, netuid, boundary_block, metagraph, live)
            v2_result = None
            if protocol_version >= 5:
                # --- PROTOCOL v5: the seat on real picks (fugal_subnet/v5_epoch.py) ---
                # v0.8.1: raw commitment blocks from the fix block; v0.8.0's
                # seniority keys before it (those epochs replay as published).
                v5_blocks, raw_blocks = _v5_commit_blocks(
                    boundary_block, v5_fix_block, seniority=commit_blocks, raw=raw_commit_blocks)
                v2_result, uids, weights, v5_commit = _v5_score(
                    v5_node, epoch_index=epoch_index, nonce=nonce, exam=v5_exam,
                    state=v4_menu.state, verified_proofs=verified_proofs,
                    verified_heads=verified_heads, head_hashes=head_hashes,
                    coldkey_of=coldkey_of, v5_inputs=v5_inputs,
                    commit_blocks=v5_blocks, raw_commit_blocks=raw_blocks,
                    burn_uid=burn_uid,
                    published=v4_view,
                    pooled=_v4_pool(v4_menu.state, verified_proofs=verified_proofs,
                                    expected=v4_expected, coldkey_of=coldkey_of, nonce=nonce,
                                    flagged=frozenset(), shares=v4_shares))
                d = v2_result.decision
                logger.info("v5 seat: %s, holder %s, history %s, paid %s, B* %s", d.event,
                            (d.holder or "none")[:16], [h[:12] for h in d.history],
                            sorted(h[:12] for h in d.payable), v2_result.baseline or "none")
            elif protocol_version >= 2:
                # --- PROTOCOL v2: the champion, its seats, and nothing else ---
                # Shared coldkey-median cells (grades and attested tokens),
                # the paired comparison window, the seat, and 85/10/5 across
                # the last three champions. Nothing here reads the frame, and
                # nothing burns for frame confidence: the payout is the paired
                # head-vs-head rule (docs/INVARIANTS.md, the v2 I3/I4
                # entries). The v1 composite above is still computed and
                # published as a diagnostic; it decides nothing.
                # The boundary snapshot for the copy rule: every committed
                # hotkey with its owning coldkey, read pinned at the boundary.
                snapshot = _copy_rule_snapshot(metagraph.hotkeys, coldkeys, chain_commitments)
                epoch_odds_block = _odds_block_for(protocol_version, odds_block, v4_block)
                on_odds = (protocol_version >= 3 and epoch_odds_block is not None
                           and boundary_block >= epoch_odds_block)
                epoch_rules = _rules_tag(protocol_version, on_odds)
                seniority_block = (v4_block if protocol_version >= 4 and v4_block is not None
                                   else V2_ACTIVATION_BLOCK)
                v3_odds = None
                if protocol_version >= 3:
                    from fugal_subnet import explore_weights as _xw
                    from fugal_subnet import routing_protocol as _rp
                    from fugal_subnet.v2_epoch import OddsInput
                    v3_odds = OddsInput(
                        nonce=nonce,
                        answers=_v3_explore_answers(verified_proofs, coldkey_of, explore_map),
                        benchmark_of={q: str(g.get("benchmark", ""))
                                      for q, g in gold_answers.items()},
                        # Under v4 the heads are read by the v4 loader
                        # (_v4_odds_input); without a menu no v4 proof is
                        # verified, so there are none to read here.
                        heads=(_v3_heads(verified_heads) if protocol_version < 4 else {}),
                        hidden_of=v3_embeddings.of,
                        flagged=frozenset(coldkey_of[uid] for uid in suspicious),
                        coldkey_cap=_cfg.EXPLORE_COLDKEY_CAP,
                        implausible_z=_cfg.EXPLORE_PRIVATE_Z,
                        implausible_min_samples=_cfg.EXPLORE_DISAGREE_MIN_SAMPLES,
                        implausible_min_trials=_cfg.EXPLORE_PRIVATE_MIN_TRIALS,
                        ref_prior=_cfg.ODDS_REF_PRIOR,
                        single_model_min_pairs=_cfg.SINGLE_MODEL_MIN_PAIRS,
                        category_targets=(_cfg.CATEGORY_TARGETS_V3 if v4_tables is None
                                          else v4_tables.category_targets),
                        explore_weights=(_xw.weights(explore_shares) if explore_shares else None),
                        # Menu B's grace: the added entries are drawn and
                        # pooled, but scored for no head (routing_protocol).
                        **({} if protocol_version >= 4 else dict(
                            menu=_rp.scored_menu(protocol_version, boundary_block, menu_b_blk,
                                                 blocks_per_epoch, menu_grace),
                            unscored=_rp.grace_excluded(protocol_version, boundary_block,
                                                        menu_b_blk, blocks_per_epoch,
                                                        menu_grace))),
                    )
                    if protocol_version >= 4 and v4_menu is not None and v4_menu.ok:
                        # v4: heads by the v4 loader, odds by pair_odds_v4 on
                        # the epoch's menu, pairs pooled as the menu step's.
                        v3_odds = _v4_odds_input(
                            base=v3_odds, state=v4_menu.state, shares=v4_shares,
                            expected=v4_expected, verified_proofs=verified_proofs,
                            coldkey_of=coldkey_of, verified_heads=verified_heads,
                            boundary_block=boundary_block, blocks_per_epoch=blocks_per_epoch)
                v2_result = run_v2_epoch(
                    epoch_index=epoch_index, epoch_id=epoch_id,
                    boundary_block=boundary_block, blocks_per_epoch=blocks_per_epoch,
                    questions=questions, pool_models=pool_models, prices=prices,
                    verified_proofs=verified_proofs, head_hashes=head_hashes,
                    coldkey_of=coldkey_of,
                    hotkey_of={uid: metagraph.hotkeys[uid] for uid in verified_proofs},
                    # v1's dedup clauses do not decide v2 eligibility. In v2
                    # a copy is an exact hash owned by another coldkey, or a
                    # junior that fails the margin against a senior seat
                    # holder (the copy rule, copy_rule.py).
                    records=epoch_state.records, commitments=snapshot,
                    window=v2_window, seat=v2_seat, copy_state=v2_copy, burn_uid=burn_uid,
                    params=seat_params_from_config(
                        boundary_block, confirm_block, protocol_version=protocol_version,
                        odds_seat_block=epoch_odds_block),
                    shares=PAYOUT_SEAT_SHARES,
                    burn_in_questions=V2_BURN_IN_QUESTIONS,
                    bench_category=(v4_tables.bench_category if v4_tables is not None else
                                    getattr(config_module,
                                            f"BENCHMARK_CATEGORY_V{protocol_version}", None)),
                    odds_input=v3_odds, rules=epoch_rules, state_rules=v2_rules,
                    pick_spend=(v4_tables.pick_spend if v4_tables is not None else None),
                    # v4: the real-picks record and rule B's evidence
                    # (picks_v4), priced at this epoch's own menu.
                    picks=(PicksInput(
                        state=v4_menu.state, outage=v4_failure.outage,
                        refused=dict(v4_failure.refused),
                        challenge=tuple(
                            (q["question_id"], _v2_category_of(q, v4_tables.bench_category))
                            for q in v4_challenge_list),
                        heads=_v4_holder_heads(v2_seat, epoch_index, verified_heads, netuid,
                                               responses))
                           if v4_failure is not None else None),
                    # v4: the memorisation check's input (memo_v4.EpochMemo)
                    # on the one view the epoch's plan drew on.
                    memo=(_v4_memo(v4_view, v4_epoch_plan, nonce)
                          if protocol_version >= 4 else None),
                    allow_take=not v2_window_degraded,
                    # Every head committed at or before the flip ties at the
                    # activation block, ordered by a digest salted with the
                    # activation boundary's hash, which nobody knew while the
                    # pre-flip commitments were being made. Under v4 the
                    # flip is the v4 activation block (rule C's order,
                    # SEAT_MINIMAL_V4.md 3.5).
                    seniority_floor=seniority_block,
                    seniority_salt=_activation_block_hash(subtensor, seniority_block,
                                                          boundary_block, block_hash),
                )
                if protocol_version >= 4:
                    # The epoch's inputs, kept so a re-anchor can run it
                    # again on a corrected state (v4_replay).
                    _keep_v4_record(run_v2_epoch.last, verified_heads, CHAMPION_WINDOW_EPOCHS)
                if v2_window_degraded:
                    logger.error(
                        "v2 comparison window was lost and has not been rebuilt: holding "
                        "the seat but making NO first or take decision this epoch. Restart "
                        "to rebuild it from the reference reveals (a root: set "
                        "FUGAL_FRAME_BOOTSTRAP to a directory of its own); it also refills "
                        "itself after %d epochs.",
                        CHAMPION_WINDOW_EPOCHS,
                    )
                uids, weights = v2_result.uids, v2_result.weights
                d = v2_result.decision
                if protocol_version >= 4:
                    logger.info("v4 seat: %s, holder %s, bar %s, history %s, paid %s",
                                d.event, (d.champion or "none")[:16], (d.bar or "R")[:16],
                                [h[:12] for h in d.state.history],
                                sorted(h[:12] for h in d.payable))
                else:
                    logger.info(
                        "v2 seat: %s, champion %s, history %s, ghosts %d, high-water %.4f",
                        d.event, (d.champion or "none")[:16],
                        [h[:12] for h in d.state.history], len(d.state.ghosts),
                        d.state.high_water,
                    )
                for h, why in d.detail:
                    logger.info("  v2 %s: %s", (h or "-")[:16], why)
            else:
                uids, weights, dupes = _v1_weights(
                    epoch_state.records, routing_decisions, commit_blocks, dupes,
                    frontier.confidence, burn_uid,
                )

            # The per-epoch weight cap (MAX_WEIGHT_DELTA) is not applied to a
            # v4 vector: every input to it is a published consensus decision,
            # and a validator that diverges is clipped by Yuma's stake-weighted
            # consensus (SEAT_MINIMAL_V4.md 2.2 item 28).
            uids, weights, weight_capped = ((uids, weights, False) if protocol_version >= 4
                                            else _capped(uids, weights, prev_uids, prev_weights))
            if weight_capped:
                logger.info("Weights capped vs previous epoch")

            # --- REVEAL ---
            timer.start_phase("reveal")
            epoch_score_dicts = {
                uid: {
                    "accuracy": s.accuracy,
                    "quality": epoch_state.records[uid].quality,
                    "thrift": epoch_state.records[uid].thrift,
                    "headroom": epoch_state.records[uid].headroom,
                    "reference_accuracy": epoch_state.records[uid].reference_accuracy,
                    "cost_per_question": epoch_state.records[uid].cost_per_question,
                    "score": epoch_state.records[uid].composite_score,
                }
                for uid, s in epoch_scores.items()
                if uid in epoch_state.records
            }
            epoch_weight_map = dict(zip(uids, weights))

            matrix, model_spend = _reveal_matrix(questions, pool_models, model_index,
                                                 verified_proofs)

            exploration_for_reveal = _exploration_for_reveal(observations)

            routing_for_reveal = {
                uid: decisions.tolist()
                for uid, decisions in routing_decisions.items()
            }
            # --- v4: THE MENU STEP (root) OR ITS AUDIT (every other validator) ---
            v4_fields, published_early, v4_deferred = {}, False, None
            if protocol_version >= 4:
                # The pool the seat's odds were scored on (run_v2_epoch's:
                # the detectors' and the implausibility check's exclusions),
                # so the menu's counts and the odds never pool differently.
                v4_pooled = list(v2_result.pooled)
                from fugal_subnet.v2_epoch import category_of as _category_of
                # A v5 epoch's proofs answer v5 pool questions (exam cells and
                # coverage), never the v4 slice's. The menu step sees them as
                # ONE category (_v5_menu_category_of): v5 prices a pick per
                # entry, and the pick stats allow at most PICK_MAX_CATEGORIES.
                menu_category_of = (
                    _v5_menu_category_of(v5_inputs) if protocol_version >= 5 else
                    {q: _category_of(g, v4_tables.bench_category)
                     for q, g in gold_answers.items()})
                v4_fields, published_early, v4_deferred = _v4_menu_epoch(
                    menu_node, menu_publisher, v4_menu, epoch_index=epoch_index, nonce=nonce,
                    verified_proofs=verified_proofs, coldkey_of=coldkey_of, pooled=v4_pooled,
                    category_of=menu_category_of,
                    counted_heads=(v2_result.counted_heads if protocol_version >= 5 else
                                   _v4_counted_heads(v2_result.decision.state,
                                                     v2_result.senior_bars)),
                    window_epochs=CHAMPION_WINDOW_EPOCHS,
                    root=role is not None and role.root, uploader=reveal_uploader,
                    reveal_delay=reveal_delay, netuid=netuid,
                    commit=(menu_node.root_fields.commit_reveal
                            if getattr(menu_node, "root_fields", None) is not None
                            else lambda data: subtensor.set_commitment(wallet, netuid, data)[0]),
                    root_summaries=_v4_root_summaries(
                        menu_node, root=role is not None and role.root,
                        reference_dir=reference_dir),
                    snapshot=pub_publisher,
                    served=sorted({str(getattr(p, "hotkey", "")) for p in verified_proofs.values()
                                   if getattr(p, "hotkey", "")}),
                    join_audit=v4_join_audit)

            def write_reveal(v4_fields):
                return reveal_epoch(
                epoch_id, questions,
                matrix, pool_models, model_spend,
                head_hashes, routing_for_reveal,
                epoch_score_dicts, epoch_weight_map,
                exploration=exploration_for_reveal,
                frame_excluded_coldkeys=sorted({coldkey_of[uid] for uid in suspicious}),
                protocol_version=protocol_version,
                v2_fields=(
                    {**v2_result.reveal,
                     "frame_mix_weights": dict(sorted(mix.items())),
                     "explore_benchmarks": dict(sorted(bench_of.items())),
                     # Whether this validator's state is unverified since a
                     # missed epoch: a peer re-anchoring on it skips such an
                     # epoch (fugal_subnet/v2_sync.check). Not in the
                     # consensus view.
                     # v4: `reanchored` when this epoch's state replaced its
                     # own (`_v4_reanchor`); its earlier reveals stand.
                     "v2_sync": {"unverified": v2_gap_until is not None}
                                | ({"reanchored": v2_audit} if v2_audit else {}),
                     # Protocol v4's menu fields (none before v4).
                     **v4_fields}
                    if v2_result is not None else None
                ),
                )
            # A v4 root writes its reveal after its weights, its joins for the
            # next epoch and its menu step (below); every other reveal now.
            reveal_ok = write_reveal(v4_fields) if v4_deferred is None else False

            # --- SET WEIGHTS ---
            timer.start_phase("set_weights")
            logger.info("Setting weights for %d UIDs", len(uids))
            success, msg = set_weights_with_retry(
                subtensor, wallet, netuid, my_uid, uids, weights,
                deadline_block=_weights_deadline(epoch_index, blocks_per_epoch),
            )
            weights_confirmed = False
            confirm_detail = "not attempted"
            next_prev_uids, next_prev_weights = prev_uids, prev_weights
            if success:
                logger.info("Weights set successfully")
                next_prev_uids, next_prev_weights = uids, weights
                weights_confirmed, confirm_detail = confirm_weights_on_chain(
                    subtensor, netuid, my_uid, uids, weights,
                )
                if weights_confirmed:
                    logger.info("Weights confirmed on chain: %s", confirm_detail)
                else:
                    logger.error(
                        "Weights NOT confirmed on chain: %s — the extrinsic "
                        "reported success but the chain disagrees",
                        confirm_detail,
                    )
            else:
                logger.warning("Weight-setting failed: %s", msg)

            timer.end_phase()

            # --- v4 ROOT: THE NEXT EPOCH'S JOINS, THEN THE MENU STEP AND THE REVEAL ---
            if v4_deferred is not None:
                _v4_join_step(subtensor=subtensor, netuid=netuid, dendrite=dendrite,
                              menu_node=menu_node, store=menu_node.join_store,
                              root_fields=menu_node.root_fields, root_joins=menu_node.root_joins,
                              epoch_index=epoch_index, blocks_per_epoch=blocks_per_epoch,
                              my_hotkey=my_hotkey)
                reveal_ok = write_reveal(v4_deferred())

            # --- PUBLISH REVEALS ---
            # Reveals old enough to be past the weight commit-reveal window go
            # to the public store, newest hash on chain. Operational: a failure
            # here is logged and retried, never an epoch lost. (A v4 root has
            # published them already, before its menu step.)
            if reveal_uploader is not None and not published_early:
                from fugal_subnet.commit_reveal import EPOCH_DIR
                from fugal_subnet.reveal_publisher import publish_due_reveals
                try:
                    publish_due_reveals(
                        EPOCH_DIR, epoch_index, reveal_delay, netuid, reveal_uploader,
                        commit=lambda data: subtensor.set_commitment(wallet, netuid, data)[0],
                    )
                except Exception as e:  # noqa: BLE001 - publishing never takes the epoch down
                    logger.warning("Reveal publishing failed this epoch: %s", e)

            anomalies = detect_anomalies(
                epoch_score_dicts, epoch_weight_map,
                n_neurons, len(verified_proofs),
            )
            anomalies += _epoch_anomalies(reveal_ok, metagraph_pinned, collect_block,
                                          suspicious, n_unverifiable)

            # Surfaced as an anomaly, not just a number, because this is the
            # signal that the field looks bad for reasons that are ours rather
            # than the miners'. Correlated across validators when the endpoint
            # is at fault, which is exactly what makes it diagnosable (I7).
            if collateral_budget.exhausted:
                logger.error(
                    "COLLATERAL BUDGET EXHAUSTED: this epoch spent its entire "
                    "%.0fs collateral allowance and stopped fetching. Proofs "
                    "after that point were skipped WITHOUT being checked and "
                    "are counted unverifiable, not invalid. Weights were still "
                    "set (I6) — but on a smaller verified set than the field. "
                    "Look at the PCCS before raising FUGAL_COLLATERAL_EPOCH_BUDGET.",
                    TEE_COLLATERAL_EPOCH_BUDGET,
                )

            epoch_log = EpochLog(
                epoch_id=epoch_id, block_hash=block_hash,
                timestamp=time.time(), n_questions=len(questions),
                n_miners_queried=n_neurons,
                n_heads_valid=len(verified_proofs), n_heads_invalid=n_invalid,
                n_heads_unverifiable=n_unverifiable,
                commit_hash=commitment.commit_hash,
                reveal_verified=reveal_ok,
                scores=epoch_score_dicts, n_failed=epoch_n_failed,
                weights=epoch_weight_map, collection=collection,
                dedup_disqualified=list(dupes),
                weight_capped=weight_capped,
                set_weights_success=success,
                set_weights_msg=str(msg) if msg else "",
                weights_confirmed_on_chain=weights_confirmed,
                weights_confirm_detail=confirm_detail,
                anomalies=anomalies,
                duration_s=timer.total_s,
            )
            write_epoch_log(epoch_log)

            next_v2 = _next_v2_state(v2_result if protocol_version < 5 else None,
                                     v2_window, v2_seat, v2_copy,
                                     v2_window_degraded, CHAMPION_WINDOW_EPOCHS)
            # A v5 epoch carries the v2-v4 state unchanged (it decides nothing
            # there); its own state is v5_node's, committed below.
            next_rules = (epoch_rules if v2_result is not None and protocol_version < 5
                          else v2_rules)

            # COMMIT. The epoch is complete; only now does it become the state
            # the next epoch (or a retry of this one) starts from.
            scoring_state, frame = epoch_state, epoch_frame
            prev_uids, prev_weights = next_prev_uids, next_prev_weights
            last_epoch_index = epoch_index
            v2_window, v2_seat, v2_copy, v2_window_degraded, v2_rules = (*next_v2, next_rules)
            save_state(scoring_state.records, prev_uids, prev_weights,
                       last_epoch_index, frame,
                       v2={"seat": seat_to_dict(v2_seat), copy_rule.REVEAL_KEY: v2_copy.to_reveal(),
                           "window_degraded": v2_window_degraded,
                           "gap_until": v2_gap_until, "anchor_at": v2_anchor_at,
                               "rules": v2_rules},
                       v2_window=v2_window, published=pub_holder)
            if protocol_version >= 5 and v5_commit is not None:
                v5_commit()

            # --- PUBLISH HEADS ---
            # Last, after the weights, the reveal and the saved state: the due
            # heads of every reveal published so far go to the store by hash.
            # Bounded in time and never raises; what is not sent is retried.
            if head_publisher is not None:
                head_publisher.publish(epoch_index)

        except KeyboardInterrupt:
            logger.info("Validator stopped by user")
            break
        except Exception:
            epoch_failed = True
            logger.exception("Error in epoch")

        if once:
            break

        logger.info("Waiting for next epoch boundary (every %d blocks)...", blocks_per_epoch)
        time.sleep(60)

    # Close the dendrite's HTTP session. Without this the process keeps a
    # non-daemon worker alive and `--once` never returns — an orchestrator or
    # an operator running a single epoch waits forever on a validator that has
    # already finished and said so.
    try:
        dendrite.close_session()
    except Exception as e:
        logger.debug("Dendrite session close: %s", e)

    logger.info("Validator shutdown complete")

    if once:
        # os._exit, not sys.exit. async_substrate_interface's __del__ closes its
        # websocket during interpreter finalization and joins a thread that
        # never finishes, so a normal exit hangs forever — a --once run in a
        # systemd oneshot, a cron job, or an orchestrator would appear to run
        # indefinitely despite having completed its epoch and said so.
        #
        # Safe here because everything durable is already committed: the state
        # file was written through os.replace, the epoch log was appended and
        # closed, and the weights extrinsic was confirmed on chain above. The
        # only thing skipped is teardown of connections the OS reclaims anyway.
        #
        # A failed epoch exits 1, so a oneshot or a cron job sees the failure
        # instead of a success with an error in the log.
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if epoch_failed else 0)


class _FrameInput(NamedTuple):
    observations: list         # every exploration answer, tagged with its coldkey (the reveal)
    coldkey_of: dict           # uid -> coldkey, for every verified miner
    suspicious: dict           # uid -> why its exploration stays out of the frame
    samples: list              # one (model, correct, tokens...) per (question, model)
    sample_weights: list | None
    mix: dict                  # v2: benchmark -> scored share / exploration share
    bench_of: dict             # v2: explored question -> benchmark


def _frame_input(frame, verified_proofs: dict, coldkeys, protocol_version: int, questions: list,
                 explore_map: dict, gold_answers: dict) -> _FrameInput:
    """This epoch's contribution to the reference frame, and who is kept out of it.

    One Observation per answer, voted by COLDKEY and collapsed to one sample
    per (question, model) by aggregate_observations: failed calls dropped, one
    vote per operator, median across operators. Every miner answers the same
    questions, so without this each copy was a fresh trial and a coalition
    moved the frame by its share of the field (diligence 2026-10, finding 2).

    Detection runs BEFORE this epoch's samples are folded in, so a miner is
    compared against history rather than against a frame it has just
    contributed to. Two detectors: accuracy implausible for the nonce-chosen
    models in either direction, and disagreement with the other operators on
    the same questions. The only consequence is that a flagged operator's
    samples stay OUT of the frame this epoch. Never a score: I4 says a miner's
    score cannot depend on other miners, and the frame is the one thing built
    from everyone. Keeping a flagged miner's samples out protects every other
    miner's reference rather than moving anyone's score (diligence 2026-10,
    finding 4).

    Protocol v2 weights each sample by its benchmark's scored share over its
    exploration share this epoch (slicer.mix_weights), so the frame estimates
    every model on the scored mix. The raw trial count is not weighted (it is
    what the frontier's confidence reads). The frame is informational in v2;
    this keeps it honest for the detectors and the diagnostics.
    """
    from fugal_subnet.benchmarks.slicer import mix_weights
    from fugal_subnet.reference_frame import (
        Observation,
        aggregate_observations,
        disagreeing_exploration,
        implausible_exploration,
    )

    coldkey_of = {
        uid: str(coldkeys[uid]) if uid < len(coldkeys) else f"uid-{uid}"
        for uid in verified_proofs
    }
    observations = [
        Observation(
            question_id=r.question_id, model=r.routed_model,
            correct=bool(r.correct), prompt_tokens=int(r.prompt_tokens),
            completion_tokens=int(r.completion_tokens), failed=bool(r.failed),
            coldkey=coldkey_of[uid],
        )
        for uid, proof in verified_proofs.items()
        for r in proof.exploration_results
    ]
    if protocol_version >= 4:
        # Protocol v4 runs neither the v1-frame detector nor the cross-miner
        # agreement detector (SEAT_MINIMAL_V4.md 2.3 item 32): a pooled
        # label's outcome is a coldkey median, which one operator cannot move,
        # and private answers are judged by the plausibility check against R
        # (explore_pool.private_implausible), which the odds pool applies.
        suspicious = {}
    else:
        suspicious = implausible_exploration(frame, {
            uid: [(r.routed_model, r.correct) for r in proof.exploration_results]
            for uid, proof in verified_proofs.items()
        }, protocol_version=protocol_version)
        suspicious.update(disagreeing_exploration({
            uid: [(r.question_id, r.routed_model, r.correct)
                  for r in proof.exploration_results if not r.failed]
            for uid, proof in verified_proofs.items()
        }, coldkey_of))
    for uid, why in sorted(suspicious.items()):
        logger.warning(
            "UID %d exploration excluded from the reference frame this "
            "epoch: %s. Its score is untouched (I4).", uid, why,
        )
    keyed = aggregate_observations(
        observations,
        exclude_coldkeys={coldkey_of[uid] for uid in suspicious},
        with_question=True,
    )
    samples = [row[1:] for row in keyed]
    mix: dict[str, float] = {}
    bench_of: dict[str, str] = {}
    sample_weights = None
    if protocol_version >= 2:
        explored_qs = [gold_answers[q] for q in sorted(explore_map) if q in gold_answers]
        mix = mix_weights(questions, explored_qs)
        bench_of = {q["question_id"]: str(q.get("benchmark", "")) for q in explored_qs}
        sample_weights = [mix.get(bench_of.get(row[0], ""), 1.0) for row in keyed]
    return _FrameInput(observations, coldkey_of, suspicious, samples, sample_weights,
                       mix, bench_of)


class _EpochChecks(NamedTuple):
    """What every proof this epoch is checked against.

    Every binding is passed explicitly. A check the validator does not supply
    an expectation for is a check that does not happen.
    """
    boundary_block: int
    require_commitment: bool
    approved_measurements: set
    questions_hash: str
    nonce_hex: str
    slice_gold: dict
    question_ids: set
    explore_map: dict
    mock: bool
    collateral_budget: object
    protocol_version: int
    menu: str | None = None
    # Protocol v4: this hotkey's expectation (tee/verify.V4Expected), passed
    # to verify_proof as `v4`; None before v4.
    v4: object | None = None
    # Protocol v5: this hotkey's expectation (tee/verify.V5Expected), passed
    # to verify_proof as `v5`; None before v5.
    v5: object | None = None


class _Checked(NamedTuple):
    """One miner's response, checked.

    status: "absent" (no proof offered), "invalid" (rejected), "unverifiable"
    (this validator could not check it) or "valid". `seniority` is set once the
    commitment check has passed, whatever happens after it.
    """
    status: str
    seniority: float | None = None
    proof: object = None
    head_bytes: bytes | None = None
    commit_block: int | None = None


def _check_response(uid: int, resp, hotkey: str, committed: tuple, registration_block: int,
                    checks: _EpochChecks) -> _Checked:
    """Check one miner's response: its on-chain commitment, its bundle, its proof.

    Never raises on miner input: whatever a miner sends costs that miner, never
    the epoch (I6). The one exception is ImportError, which is this validator's
    own installation and not the miner's doing.
    """
    from fugal_subnet.commitments import is_head_hash, seniority_key
    from fugal_subnet.tee.verify import verify_proof

    if resp is None or not hasattr(resp, "proof_hash") or not resp.proof_hash:
        return _Checked("absent")
    if not hotkey:
        # Responses and hotkeys come from the same metagraph, so this does not
        # happen; if it did, there is no hotkey to bind the proof to.
        logger.warning("UID %d: no hotkey in the metagraph; proof not checked", uid)
        return _Checked("invalid")
    weights_hash = getattr(resp, "weights_hash", "")
    commit_block = None

    # The on-chain commitment, read at the boundary block. Only a head hash can
    # be committed: an empty or malformed value would compare equal to an
    # equally empty weights_hash and bind the proof to no head at all.
    if (is_head_hash(weights_hash) and committed[0] == weights_hash
            and 0 < committed[1] <= checks.boundary_block):
        # Seniority from chain state at the boundary: the hotkey's
        # registration block first, its commitment block second. A retrained
        # author keeps rank over whoever copied its previous head, and every
        # validator derives the same order (commitments.seniority_key).
        seniority = seniority_key(registration_block, committed[1])
        commit_block = int(committed[1])
    elif checks.require_commitment:
        logger.warning(
            "UID %d proof rejected: weights_hash %s... not committed on-chain "
            "at or before boundary block %d",
            uid, weights_hash[:12], checks.boundary_block,
        )
        return _Checked("invalid")
    else:
        seniority = math.inf

    try:
        bundle = _get_bundle_for_uid(uid, resp)
    except Exception as e:  # noqa: BLE001 - one miner, not the epoch
        logger.warning("UID %d: failed to get bundle: %s", uid, e)
        return _Checked("invalid", seniority, commit_block=commit_block)
    if bundle is None:
        return _Checked("invalid", seniority, commit_block=commit_block)
    proof, head_bytes = bundle

    # Defence in depth. verify_proof is contracted not to raise on miner
    # input, but the cost of being wrong about that is the whole epoch, so a
    # bug there must cost one miner instead.
    try:
        result = verify_proof(
            proof,
            approved_measurements=checks.approved_measurements,
            expected_questions_hash=checks.questions_hash,
            expected_nonce=checks.nonce_hex,
            gold_answers=checks.slice_gold,
            expected_question_ids=checks.question_ids,
            expected_exploration=checks.explore_map,
            expected_weights_hash=weights_hash,
            # The hotkey of the uid we actually queried. A proof naming a
            # different one was produced for somebody else.
            expected_hotkey=hotkey,
            expected_proof_hash=getattr(resp, "proof_hash", ""),
            head_bytes=head_bytes,
            mock=checks.mock,
            collateral_budget=checks.collateral_budget,
            protocol_version=checks.protocol_version,
            menu=checks.menu,
            **({"v4": checks.v4} if checks.v4 is not None and checks.v5 is None else {}),
            **({"v5": checks.v5} if checks.v5 is not None else {}),
        )
    except ImportError:
        raise
    except Exception as e:  # noqa: BLE001 - one miner, not the epoch
        logger.warning(
            "UID %d proof verification raised %s: %s — rejecting this "
            "miner, not the epoch", uid, type(e).__name__, e,
        )
        return _Checked("invalid", seniority, commit_block=commit_block)

    if not result.valid:
        # Not scored either way, and falls through to apply_miss like any
        # absent miner. What changes is what gets RECORDED, and that is not
        # cosmetic: an operator reading "247 invalid" hunts an attack, while
        # "247 unverifiable" points at their own egress.
        #
        # Skipping is also the only I6-compatible choice. Abandoning the epoch
        # instead would let one miner with an unknown FMSPC stop every
        # validator publishing at once, because the collection point is a
        # deterministic block — a subnet kill switch for the price of one
        # registration.
        if getattr(result, "unverifiable", False):
            logger.warning(
                "UID %d proof could not be checked: %s — NOT counted "
                "as invalid; this validator never judged it",
                uid, result.reason,
            )
            return _Checked("unverifiable", seniority, commit_block=commit_block)
        logger.warning("UID %d proof failed verification: %s", uid, result.reason)
        return _Checked("invalid", seniority, commit_block=commit_block)

    for w in result.warnings or ():
        logger.info("UID %d proof warning: %s", uid, w)
    return _Checked("valid", seniority, proof, head_bytes, commit_block)


def _offers_proof(resp) -> bool:
    """Did this axon answer with a proof at all (what `_check_response` reads first)?"""
    return resp is not None and bool(getattr(resp, "proof_hash", ""))


def _query_outcome(resp) -> tuple[str, float | None]:
    """How the query to one axon ended, as the dendrite recorded it:
    (status, seconds). The status message is text the far end chose, so it is
    cut to printable ASCII and a short length before it reaches a log (I2)."""
    d = getattr(resp, "dendrite", None)
    code = getattr(d, "status_code", None)
    try:
        seconds = round(float(getattr(d, "process_time", None)), 2)
    except (TypeError, ValueError):
        seconds = None
    if code is None:
        return "no reply", seconds
    status = str(code)[:8]
    if status != "200":
        msg = str(getattr(d, "status_message", "") or "")
        msg = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in msg)[:80]
        if msg:
            status = f"{status} {msg}"
    return status, seconds


# How many UIDs the one-line collection summary names; the epoch log has all.
_COLLECTION_LINE_UIDS = 40


def _collection_report(responses, hotkeys: list, chain_commitments: dict,
                       outcomes: dict) -> tuple[dict, str]:
    """({uid: what happened} for the epoch log, the one-line summary).

    Covers every UID whose hotkey had a head committed at the boundary (a
    miner that meant to be scored) and every UID that answered anything. The
    summary names the committed miners that ended the epoch without a valid
    proof, each with how its query ended, because that is the question a
    miner asks: was I never reached, was I slow, or was my proof refused?
    Never raises: it is a log.
    """
    report: dict[str, dict] = {}
    counts = {"valid": 0, "invalid": 0, "unverifiable": 0, "absent": 0}
    missing: list[str] = []
    try:
        for uid, resp in enumerate(responses):
            result = outcomes.get(uid, "absent")
            counts[result] = counts.get(result, 0) + 1
            hotkey = hotkeys[uid] if uid < len(hotkeys) else ""
            committed = bool(hotkey) and hotkey in chain_commitments
            status, seconds = _query_outcome(resp)
            if not committed and status == "no reply" and result == "absent":
                continue
            report[str(uid)] = {"query": status, "seconds": seconds, "result": result,
                                "committed": committed}
            if committed and result != "valid":
                took = f" after {seconds:.1f}s" if seconds is not None else ""
                missing.append(f"{uid} ({result}; query {status}{took})")
    except Exception as e:  # noqa: BLE001 - diagnostics must not cost the epoch
        logger.debug("collection report failed: %s", e)
    line = (f"Collection: {len(responses)} queried; {counts['valid']} valid, "
            f"{counts['invalid']} invalid, {counts['unverifiable']} unverifiable, "
            f"{counts['absent']} absent")
    if missing:
        shown = missing[:_COLLECTION_LINE_UIDS]
        more = f", and {len(missing) - len(shown)} more" if len(missing) > len(shown) else ""
        line += (f". Committed miners without a valid proof ({len(missing)}): "
                 f"uid {', uid '.join(shown)}{more}")
    return report, line


def _get_bundle_for_uid(uid, resp):
    """Parse a miner's inline bundle, returning (proof, head_bytes) or None.

    Everything here is miner-controlled input, so it is bounded before it is
    parsed (I2). The pydantic field caps in protocol.py are the first bound;
    HEAD_MAX_BYTES is the second, applied before the head reaches a loader.
    """
    import base64

    from fugal_subnet.config import HEAD_MAX_BYTES
    from fugal_subnet.tee.proof import BenchmarkProof

    proof_json = getattr(resp, "proof_json", "") or ""
    head_b64 = getattr(resp, "head_npz_b64", "") or ""
    if not proof_json or not head_b64:
        logger.debug("UID %d: incomplete bundle in response", uid)
        return None

    try:
        proof = BenchmarkProof.from_dict(json.loads(proof_json))
    except Exception as e:
        logger.warning("UID %d: unparseable proof: %s", uid, e)
        return None

    try:
        head_bytes = base64.b64decode(head_b64, validate=True)
    except Exception as e:
        logger.warning("UID %d: undecodable head: %s", uid, e)
        return None

    if len(head_bytes) > HEAD_MAX_BYTES:
        logger.warning("UID %d: head is %d bytes (max %d)",
                       uid, len(head_bytes), HEAD_MAX_BYTES)
        return None

    return proof, head_bytes


# Imported at module scope: wait_for_block is called from the epoch loop but
# defined here, and the nominal block time is only used to render a human
# estimate — the wait itself is on block height, never on wall clock.
from fugal_subnet.benchmarks.slicer import BLOCK_TIME_S as _BLOCK_TIME_S  # noqa: E402


def assert_chain_weight_parameters(subtensor, netuid: int, *, live: bool) -> dict:
    """Refuse (live) or warn (mock) when the chain would silently drop our weights.

    Three parameters decide whether a set_weights that the SDK reports as
    submitted ever reaches Yuma, and none of them is visible in the result:
    the chain's commit-reveal VERSION must equal the one the SDK commits with
    (`IncorrectCommitRevealVersion` surfaces only at reveal), the SDK's
    `version_key` must be >= the subnet's `weights_version`, and the vector
    must satisfy `min_allowed_weights` / `max_weights_limit`. Read once at
    startup and logged; the first two are refusals in live, because a
    validator that commits every epoch into a chain that discards every reveal
    looks perfectly healthy from its own logs (validator review 2026-11-02,
    findings 3 and 4).
    """
    import inspect

    out: dict = {}
    sdk_crv = inspect.signature(subtensor.set_weights).parameters["commit_reveal_version"].default
    sdk_vk = inspect.signature(subtensor.set_weights).parameters["version_key"].default
    try:
        chain_crv = subtensor.query_subtensor("CommitRevealWeightsVersion")
        chain_crv = int(getattr(chain_crv, "value", chain_crv))
    except Exception as e:  # noqa: BLE001 - an unreadable value is reported, not guessed
        chain_crv = None
        logger.warning("Could not read CommitRevealWeightsVersion: %s", e)
    try:
        hp = subtensor.get_subnet_hyperparameters(netuid=netuid)
        out = {
            "commit_reveal_enabled": bool(getattr(hp, "commit_reveal_weights_enabled", False)),
            "commit_reveal_period": getattr(hp, "commit_reveal_period", None),
            "weights_rate_limit": getattr(hp, "weights_rate_limit", None),
            "min_allowed_weights": getattr(hp, "min_allowed_weights", None),
            "max_weights_limit": getattr(hp, "max_weights_limit", None),
            "weights_version": getattr(hp, "weights_version", None),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read subnet hyperparameters: %s", e)
    out["chain_commit_reveal_version"] = chain_crv
    out["sdk_commit_reveal_version"] = sdk_crv
    logger.info("Chain weight parameters: %s", out)

    problems = []
    if chain_crv is not None and out.get("commit_reveal_enabled") and chain_crv != sdk_crv:
        problems.append(
            f"chain CommitRevealWeightsVersion is {chain_crv} but this SDK commits "
            f"version {sdk_crv}; every reveal would be rejected"
        )
    wv = out.get("weights_version")
    if wv is not None and int(wv) > int(sdk_vk):
        problems.append(
            f"subnet weights_version {wv} exceeds the SDK version_key {sdk_vk}; "
            "set_weights would be refused"
        )
    if problems:
        msg = "; ".join(problems)
        if live:
            raise RuntimeError(msg)
        logger.warning("%s (continuing: mock)", msg)
    return out


def set_weights_with_retry(subtensor, wallet, netuid: int, my_uid: int, uids, weights,
                           *, deadline_block: int, attempts: int | None = None,
                           sleep_s: float = 12.0) -> tuple[bool, str]:
    """Submit weights, naming the reason when the SDK will not, and retry
    while the epoch has blocks left.

    The SDK's set_weights returns `(False, None)` without attempting anything
    when the chain's rate limit has not elapsed, and a transient websocket
    error becomes `(False, <exception text>)` after its own five attempts. A
    failure here used to be logged and the epoch marked done, so one blip cost
    the epoch's weights (validator review 2026-11-02, finding 3). Now the rate
    limit is read first so the log says why, and a failed submit is retried up
    to WEIGHTS_SET_RETRIES times while `deadline_block` has not passed.

    The SDK catches errors inside the extrinsic itself, but its pre-flight
    chain reads (the UID lookup, the commit-reveal check, the rate limit in its
    retry condition) are outside that, so a dropped connection there RAISES out
    of set_weights. That is one failed attempt here, retried like any other,
    not a failed epoch (which would re-query and re-score the whole epoch).
    """
    from fugal_subnet.config import WEIGHTS_SET_RETRIES

    attempts = WEIGHTS_SET_RETRIES if attempts is None else attempts
    msg = ""
    raised = False
    for attempt in range(1, max(1, attempts) + 1):
        try:
            since = int(subtensor.blocks_since_last_update(netuid, my_uid))
            limit = int(subtensor.weights_rate_limit(netuid))
        except Exception as e:  # noqa: BLE001 - diagnostic only
            since, limit = -1, -1
            logger.warning("Could not read the weights rate limit: %s", e)
        if 0 <= since <= limit:
            if raised:
                # An earlier attempt raised, and the chain now says this UID
                # updated within the rate limit: that attempt may have landed
                # before the connection dropped. Read it back rather than
                # report a failure while the weights are on chain.
                landed, detail = confirm_weights_on_chain(subtensor, netuid, my_uid, uids, weights)
                if landed:
                    return True, f"an attempt that raised had landed: {detail}"
            return False, (f"rate limit: {since} blocks since this UID's last update, "
                           f"limit {limit}; the chain would refuse the extrinsic")
        try:
            response = subtensor.set_weights(
                wallet=wallet, netuid=netuid, uids=uids, weights=weights,
                wait_for_inclusion=True, wait_for_finalization=True,
            )
        except Exception as e:  # noqa: BLE001 - a dropped connection is a failed attempt
            raised = True
            logger.warning("set_weights raised on attempt %d", attempt, exc_info=True)
            response = (False, f"set_weights raised {type(e).__name__}: {e}")
        success, msg = response
        if success:
            return True, str(msg or "")
        msg = str(msg or "submit returned no message")
        try:
            now = int(subtensor.get_current_block())
        except Exception:  # noqa: BLE001
            now = deadline_block
        if attempt >= attempts or now >= deadline_block:
            break
        logger.warning("set_weights attempt %d/%d failed (%s); retrying in %.0fs "
                       "(block %d, deadline %d)", attempt, attempts, msg, sleep_s, now, deadline_block)
        time.sleep(sleep_s)
    return False, msg


def read_commitments_or_fail(subtensor, netuid: int, block: int, *, required: bool):
    """The epoch's on-chain commitments, pinned to `block`.

    `required` is `live and REQUIRE_COMMITMENT`: then a failed read raises,
    because an empty map under REQUIRE_COMMITMENT rejects every proof and the
    epoch would be marked processed with no weights set while peers that read
    in time set theirs (I1, I6). Raising hands the epoch to the outer loop's
    retry instead. When commitments are not required (mock, local chains whose
    node has no Commitments pallet), the old behaviour stands: warn and go on.

    The commonest cause of a failure here is a non-archive endpoint asked for a
    block more than ~300 behind the head, so the message says so.
    """
    from fugal_subnet.commitments import get_commitments_with_blocks

    try:
        commitments = get_commitments_with_blocks(subtensor, netuid, block=block)
    except Exception as e:  # noqa: BLE001 - chain/SDK errors of every shape
        if required:
            raise RuntimeError(
                f"Could not read on-chain commitments at block {block}: {e}. "
                "Commitments are required, so this epoch cannot be scored "
                "without them; retrying rather than scoring every miner as "
                "uncommitted. If the block is far behind the head, the "
                "endpoint is not an archive node."
            ) from e
        logger.warning("Could not read on-chain commitments at block %d: %s", block, e)
        return {}
    logger.info("On-chain commitments at block %d: %d hotkeys", block, len(commitments))
    return commitments


def wait_for_block(subtensor, target_block: int, poll_s: int = 12) -> None:
    """Block until the chain reaches `target_block`.

    Cheap and deliberately dumb: one block-height read per poll. The wait is
    what makes the collection point deterministic, so it is not shortened on
    the grounds that a miner "looks ready" — that would reintroduce exactly the
    wall-clock dependence the fixed point exists to remove.

    How long to sleep between reads is sized from the block rate this chain is
    measured to have, never the nominal one. Sized from the nominal 12 s, an
    11-block wait on a devnet producing five blocks a second slept a minute,
    woke 240 blocks late in the *next* epoch, and asked every miner for a proof
    they had already replaced — logged as "no valid proofs", as though the
    miners were at fault.
    """
    announced = False
    last_log = 0.0
    seen: tuple[float, int] | None = None  # first (monotonic time, height) read
    s_per_block: float | None = None
    while True:
        try:
            current = subtensor.get_current_block()
        except Exception as e:  # noqa: BLE001 - a dropped socket must not end the epoch
            logger.warning("Could not read block height (%s); retrying", e)
            time.sleep(poll_s)
            continue
        if current >= target_block:
            return
        remaining = target_block - current
        if seen is None:
            seen = (time.monotonic(), current)
        elif current > seen[1]:
            s_per_block = (time.monotonic() - seen[0]) / (current - seen[1])
        # Say it once, then rarely. A line every minute for half an epoch is
        # noise an operator learns to scroll past, and the next thing they
        # scroll past is the one that mattered.
        now = time.monotonic()
        if not announced or now - last_log >= 300:
            logger.info("Waiting for collection block %d (%d blocks, ~%.0f min)",
                        target_block, remaining, remaining * _BLOCK_TIME_S / 60)
            announced = True
            last_log = now
        if s_per_block is None:
            # No rate yet: a short first sleep costs one extra read on a slow
            # chain and is the only safe guess on a fast one.
            time.sleep(1.0)
        else:
            time.sleep(min(max(s_per_block * remaining, 0.2), 60))


_ACTIVATION_HASH: dict[int, str] = {}


def _activation_block_hash(subtensor, activation_block: int, boundary_block: int,
                           boundary_hash: str) -> str:
    """The activation boundary's block hash: the v2 seniority tie-break salt.

    Chain data, the same for every validator; cached for the process. Raises
    if the chain cannot serve it, so the epoch is retried rather than ordered
    with a different salt from every peer.
    """
    a = int(activation_block)
    if a not in _ACTIVATION_HASH:
        h = boundary_hash if int(boundary_block) == a else subtensor.get_block_hash(a)
        if not h:
            raise RuntimeError(f"could not read the hash of activation block {a}")
        _ACTIVATION_HASH[a] = str(h)
    return _ACTIVATION_HASH[a]


def resolve_burn_uid(subtensor, netuid: int, block: int, metagraph, live: bool) -> int:
    """The subnet owner's UID at `block`, where unassigned weight burns.

    Read from chain each epoch (`get_subnet_owner_hotkey`, pinned to the
    boundary block so every validator resolves the same UID) and mapped
    through the pinned metagraph. It used to be a hard-coded 0 (diligence
    2026-10, finding A5). Live refuses to guess: a failed lookup raises,
    because burning to the wrong UID pays someone. Mock falls back to 0 for
    local chains whose mock subtensor has no owner query.
    """
    owner = None
    try:
        from fugal_subnet import chain_reads

        owner = chain_reads.pinned(
            subtensor, block, lambda s: s.get_subnet_owner_hotkey(netuid, block=block),
            what="Subnet owner", refused=chain_reads.is_zero_account)
    except Exception as e:  # noqa: BLE001 - reported below, never swallowed
        if live:
            raise RuntimeError(f"Could not read the subnet owner hotkey at block {block}: {e}") from e
        logger.warning("Could not read the subnet owner hotkey (%s); mock burns to UID 0", e)
    hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
    if owner and owner in hotkeys:
        return hotkeys.index(owner)
    if live:
        raise RuntimeError(
            f"Subnet owner hotkey {owner!r} is not registered on netuid {netuid} at block "
            f"{block}; refusing to burn to a guessed UID"
        )
    return 0


def registration_block_of(metagraph, uid: int) -> int:
    """The block this uid's hotkey registered at, from the pinned metagraph.

    0 when the metagraph does not carry it (a mock), which makes the
    commitment block the whole key -- the pre-2026-10 behaviour, and still
    deterministic for every validator reading the same chain.
    """
    blocks = getattr(metagraph, "block_at_registration", None)
    try:
        return int(blocks[uid]) if blocks is not None and uid < len(blocks) else 0
    except (TypeError, ValueError):
        return 0


def _close_chain(subtensor) -> None:
    """Close the chain connection now, before the process exits (`--check`).
    Never raises: a connection that will not close is no reason to fail the
    check."""
    try:
        subtensor.close()
    except Exception as e:  # noqa: BLE001 - best effort, the exit follows
        logger.debug("Closing the chain connection: %s", e)


def _v4_require_pinned_metagraph(protocol_version: int, pinned: bool) -> None:
    """Under protocol v4 the hotkey-to-coldkey map read at the collection
    block feeds the menu step (the endpoint vote, health, the coldkey bounds)
    and the audit: a consensus input. A failed pinned read then fails the
    epoch, which is retried, as a failed read of the menu commitment does;
    the current-metagraph fallback stands for v1-v3 only."""
    if int(protocol_version) >= 4 and not pinned:
        raise RuntimeError("the metagraph at the collection block could not be read; under "
                           "protocol v4 the coldkey map is a consensus input, so the epoch is "
                           "retried rather than run on the current metagraph")


def metagraph_at(subtensor, netuid: int, block: int, fallback) -> tuple:
    """(metagraph as of `block`, True), or (the live one, False) when the pinned
    read fails.

    Pinning the read is what makes two validators query the same axons. A node
    that has pruned the state for that block cannot serve it, though, and a
    validator that refuses to run because it cannot read 40 blocks of history
    is worse than one that reads the current state and says so. The False is
    what the epoch log records as `metagraph_unpinned`, so the divergence is
    attributable from the published artifacts, not only from the console.
    """
    from fugal_subnet import chain_reads

    try:
        return chain_reads.pinned(subtensor, block, lambda s: s.metagraph(netuid, block=block),
                                  what="Metagraph"), True
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not read the metagraph at block %d (%s) — using the current "
            "one. Two validators reading at different blocks can see different "
            "axon sets, so a divergence this epoch has a known cause.",
            block, e,
        )
        return fallback, False


def confirm_weights_on_chain(subtensor, netuid, my_uid, uids, weights, tol=1e-3):
    """Read the weights back off chain and check they match what we submitted.

    `set_weights` returning success means the extrinsic was included, not that
    the chain now holds what we meant. Nothing in this repo used to check the
    difference, so "weights set successfully" was an unverified claim in every
    log and every epoch artifact.

    On a commit-reveal subnet the weights are NOT readable straight away: the
    extrinsic commits an encrypted hash and the values only appear after the
    reveal period. An immediate read finds nothing, correctly — so on those
    subnets this confirms the commit was recorded (LastUpdate advanced to this
    block) rather than pretending to read values that cannot exist yet.

    Returns (confirmed, detail). Never raises — a readback failure must not
    take down an epoch that otherwise succeeded.
    """
    try:
        if subtensor.commit_reveal_enabled(netuid=netuid):
            last_update = subtensor.query_subtensor("LastUpdate", params=[netuid])
            block = subtensor.get_current_block()
            recorded = int(last_update[my_uid])
            # Allow a few blocks of slack for inclusion.
            if block - recorded > 25:
                return False, (
                    f"commit-reveal subnet, but LastUpdate for uid {my_uid} is "
                    f"block {recorded} and the chain is at {block} — the commit "
                    "was not recorded"
                )
            return True, (
                f"commit recorded at block {recorded}; values reveal after the "
                "reveal period (commit-reveal subnet)"
            )
    except Exception as e:
        logger.debug("commit-reveal probe failed, falling back to readback: %s", e)

    try:
        on_chain = subtensor.weights(netuid=netuid)
    except Exception as e:
        return False, f"readback failed: {e}"

    row = next((w for uid, w in on_chain if uid == my_uid), None)
    if row is None:
        return False, f"no weight row on chain for validator uid {my_uid}"

    # Chain stores u16-normalized weights; compare as proportions.
    got = {int(u): float(v) for u, v in row}
    total = sum(got.values()) or 1.0
    got = {u: v / total for u, v in got.items()}
    want = dict(zip(uids, weights))

    for uid in set(want) | set(got):
        if abs(want.get(uid, 0.0) - got.get(uid, 0.0)) > tol:
            return False, (
                f"uid {uid}: submitted {want.get(uid, 0.0):.4f}, "
                f"chain has {got.get(uid, 0.0):.4f}"
            )
    return True, f"{len(got)} weights match within {tol}"


def _archive_heads(epoch_id, verified_heads, head_hashes, metagraph):
    """Save this epoch's verified heads when FUGAL_HEAD_ARCHIVE_DIR is set.

    Diagnostics only: the archive feeds nothing back into scoring, and any
    error is logged and swallowed so it cannot cost the epoch.
    """
    from fugal_subnet.config import HEAD_ARCHIVE_DIR
    from fugal_subnet.head_archive import archive_verified_heads

    if not HEAD_ARCHIVE_DIR or not verified_heads:
        return
    try:
        n = len(metagraph.hotkeys)
        counts = archive_verified_heads(
            HEAD_ARCHIVE_DIR, epoch_id, verified_heads, head_hashes,
            hotkeys={u: metagraph.hotkeys[u] for u in verified_heads if u < n},
            coldkeys={u: metagraph.coldkeys[u] for u in verified_heads if u < n},
        )
        logger.info("Head archive: %d new, %d repeat, %d skipped",
                    counts["first_seen"], counts["repeat"], counts["skipped"])
    except Exception as e:  # noqa: BLE001 - diagnostics must not cost the epoch
        logger.warning("Head archive failed this epoch: %s", e)


def count_failed_scored(proof) -> int:
    """Scored calls the proof marks `failed`: upstream errors, not answers.

    Diagnostics only. A failed call is already `correct=False` in the score;
    this count is logged so a low accuracy can be read as "calls failed" or
    "answers were wrong". Exploration is excluded, as it is from the score.
    """
    return sum(1 for r in proof.scored_results if r.failed)


def _reveal_matrix(questions: list, pool_models: list, model_index: dict, verified_proofs: dict):
    """(matrix, model_spend) for the reveal.

    The matrix is in the SAME global model index space dedup uses, so a column
    means the same model in every artifact. -1 means "no miner routed this
    question to this model", which is distinct from 0 ("routed, and got it
    wrong"): as a plain zero matrix the two were indistinguishable and any
    consumer would read unexplored cells as failures.
    """
    q_idx_map = {q["question_id"]: i for i, q in enumerate(questions)}
    matrix = np.full((len(questions), max(len(pool_models), 1)), -1, dtype=np.int32)
    model_spend: dict[str, float] = {}
    for proof in verified_proofs.values():
        for r in proof.results:
            m_idx = model_index.get(r.routed_model)
            q_idx = q_idx_map.get(r.question_id)
            if m_idx is not None and q_idx is not None:
                matrix[q_idx, m_idx] = int(r.correct)
            model_spend[r.routed_model] = model_spend.get(r.routed_model, 0.0) + r.cost_usd
    return matrix, model_spend


def _exploration_for_reveal(observations: list) -> list[dict]:
    """Exploration observations for the reveal, with the voter and the failure
    flag, so rebuild_from_reveals aggregates exactly as this epoch did.
    Coldkeys are public on chain; publishing them reveals nothing new. Sorted
    deterministically in reveal_epoch, so every validator publishes the
    identical list."""
    return [
        {
            "question_id": o.question_id,
            "model": o.model,
            "correct": o.correct,
            "prompt_tokens": o.prompt_tokens,
            "completion_tokens": o.completion_tokens,
            "coldkey": o.coldkey,
            "failed": o.failed,
        }
        for o in observations
    ]


def _weights_deadline(epoch_index: int, blocks_per_epoch: int) -> int:
    """The last block this epoch's weights may be retried at: five blocks
    before the next epoch's boundary, so a retry never lands in the next
    epoch and is never confused with that epoch's weights."""
    return (epoch_index + 1) * blocks_per_epoch - 5


def _epoch_anomalies(reveal_ok: bool, metagraph_pinned: bool, collect_block: int,
                     suspicious: dict, n_unverifiable: int) -> list[str]:
    """The validator's own anomalies for the epoch log, so each is attributable
    after the fact by anyone, not only by whoever was reading the console."""
    out = []
    if not reveal_ok:
        out.append("commit_reveal_failed")
    if not metagraph_pinned:
        # Two validators that queried different axon sets this epoch have a
        # named cause in the published log (F-070).
        out.append(f"metagraph_unpinned: the read at block {collect_block} failed and the "
                   "current metagraph was used")
    for uid, why in suspicious.items():
        out.append(f"exploration_excluded_from_frame: uid {uid}: {why}")
    if n_unverifiable:
        # A signal that the field looks bad for reasons that are ours rather
        # than the miners'. Correlated across validators when the endpoint is
        # at fault, which is exactly what makes it diagnosable (I7).
        out.append(f"collateral_unavailable: {n_unverifiable} proof(s) unchecked")
    return out


def _v4_activation_block(cfg, genesis) -> int | None:
    """The protocol v4 activation block on this chain, or None
    (config.v4_activation_block). No release sets one on finney, where a set
    FUGAL_V4_ACTIVATION_BLOCK is refused (ValueError); another chain may set
    it (tests, local rehearsals). Without it no epoch is protocol v4 and
    nothing of the v4 menu runs."""
    fn = getattr(cfg, "v4_activation_block", None)
    return fn(genesis) if callable(fn) else None


class _V4Status(NamedTuple):
    """Protocol v4 on this validator, as startup finds it: `found` describes
    the inputs for the READY line (None when no v4 block applies);
    `missing` is every reason it cannot run v4 epochs, and `refusals` what
    the menu setup refuses on (the same reasons); `due` whether the current epoch is at or after
    the first v4 epoch (only then does a refusal stop the validator)."""
    found: str | None
    missing: list
    refusals: list
    due: bool
    step_params: object = None


def _v4_refusals(*, cfg, subtensor, netuid: int, my_hotkey: str, role, root_hotkey,
                 reveal_delay: int, blocks_per_epoch: int, reveal_bucket: str):
    """(why this validator cannot run the v4 menu, the menu step's parameters):
    menu_publish.startup_refusals, the root's identity and bucket, the
    proposals' category floor and the summary window."""
    from fugal_subnet import epoch_plan as _plan
    from fugal_subnet import menu_publish as mp
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block, epoch_index_for_block
    from fugal_subnet.reveal_publisher import delay_epochs_for

    root = role is not None and role.root
    root_hk = my_hotkey if root else ((role.root_hotkey if role is not None else None)
                                      or root_hotkey)
    # The production step parameters: the pinned v4 mix
    # (epoch_plan.menu_step_params_v4) and the proposals' category floor
    # (config.PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4).
    floor = getattr(cfg, "PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4", None)
    params_error = None
    try:
        step_params = _plan.menu_step_params_v4(floor) if floor is not None else None
    except (OSError, ValueError) as e:
        step_params, params_error = None, f"the v4 menu step's parameters cannot be built ({e})"
    refusals = mp.startup_refusals(
        chain_hold=delay_epochs_for(subtensor, netuid), reveal_delay=reveal_delay, root=root,
        root_hotkey=root_hk, epoch_id=epoch_id_for_block(
            epoch_index_for_block(subtensor.get_current_block(), blocks_per_epoch)))
    if root and root_hotkey and my_hotkey != root_hotkey:
        refusals.append("this validator is configured as the root, but the release names "
                        "another root hotkey")
    if root and not reveal_bucket:
        refusals.append("this root publishes no reveals (FUGAL_REVEAL_BUCKET is unset), so no "
                        "menu would ever change")
    if params_error is not None:
        refusals.append(params_error)
    elif step_params is None:
        refusals.append("the v4 proposals' category floor is not pinned "
                        "(config.PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4)")
    else:
        from fugal_subnet.config import CHAMPION_WINDOW_EPOCHS
        why = _v4_window_refusal(CHAMPION_WINDOW_EPOCHS, step_params)
        if why is not None:
            refusals.append(why)
    return refusals, step_params


def _v4_inputs(first_epoch: int, root_hotkeys) -> tuple[list, list]:
    """(what was found, what is missing) of the files and tables v4 reads:
    the pinned seed and mix, the list signers and the shipped list (the seed
    must name the one shipped), the memorisation check's published-set seed
    (the reveal it is observed through: a node brings the set forward from
    it by folding the public reveals after), the root's hotkeys from the
    first v4 epoch, and the v4 digest."""
    import hashlib

    from fugal_subnet import epoch_plan, fingerprint, memo_v4
    from fugal_subnet import menu_rules as mr
    from fugal_subnet import menu_state as ms

    found, missing = [], []
    try:
        seed = ms.load_seed()
        found.append("seed " + hashlib.sha256(ms.SEED_PATH.read_bytes()).hexdigest()[:8])
    except (OSError, ValueError) as e:
        seed = None
        missing.append(f"the pinned menu seed cannot be read ({e})")
    try:
        epoch_plan.mix_v4()
        with open(epoch_plan.MIX_V4_PATH, "rb") as f:
            found.append("mix " + hashlib.sha256(f.read()).hexdigest()[:8])
    except (OSError, ValueError) as e:
        missing.append(f"the pinned v4 mix cannot be read ({e})")
    try:
        mr.list_signers()
        try:
            labs = mr.shipped_list(mr.KIND_MENU_ALLOWLIST)
            found.append(f"list serial {labs.serial}")
        except FileNotFoundError:
            labs = None
            found.append("list none")
        if seed is not None and seed.labs.serial and (seed.labs.serial, seed.labs.sha256) != (
                (labs.serial, labs.sha256) if labs is not None else (0, "")):
            missing.append(f"the seed names lab and host list serial {seed.labs.serial}, which "
                           "this release does not ship")
    except (OSError, RuntimeError, mr.ListRefused) as e:
        missing.append(f"the signed lab and host list does not verify ({e})")
    try:
        found.append("published seed " + memo_v4.seed_label())
    except (OSError, ValueError) as e:
        missing.append(f"the pinned published-set seed cannot be read ({e})")
    starts = [int(e) for e, _hk in (root_hotkeys or ())]
    if not starts or min(starts) > int(first_epoch):
        missing.append(f"no root hotkey is known for the first v4 epoch e{int(first_epoch):08d}")
    else:
        found.append(f"root hotkeys {len(starts)}")
    found.append("v4 digest " + fingerprint.consensus_digest(protocol_version=4)[:16])
    return found, missing


def _v4_status(*, cfg, subtensor, netuid: int, my_hotkey: str, role, root_hotkey,
               reveal_delay: int, v4_block, blocks_per_epoch: int, reveal_bucket: str,
               inputs_loaded: bool, inputs_error: str | None = None,
               root_hotkeys=()) -> _V4Status:
    """Protocol v4 on this validator at startup (_V4Status). Never raises for
    a v4 reason: whether a reason stops the validator is decided by `due`."""
    from fugal_subnet.benchmarks.slicer import epoch_index_for_block

    if v4_block is None:
        return _V4Status(None, [], [], False)
    first = int(v4_block) // blocks_per_epoch
    due = epoch_index_for_block(subtensor.get_current_block(), blocks_per_epoch) >= first
    refusals, step_params = _v4_refusals(
        cfg=cfg, subtensor=subtensor, netuid=netuid, my_hotkey=my_hotkey, role=role,
        root_hotkey=root_hotkey, reveal_delay=reveal_delay, blocks_per_epoch=blocks_per_epoch,
        reveal_bucket=reveal_bucket)
    found, missing = _v4_inputs(first, root_hotkeys)
    if not inputs_loaded:
        missing.insert(0, inputs_error or "the v3 pool and embeddings v4 runs on are not loaded")
    from fugal_subnet.memo_sync import history_line
    found.append(history_line(subtensor, netuid, blocks_per_epoch))
    found = f"block {int(v4_block)}, " + ", ".join(found)
    reasons = list(refusals) + missing
    return _V4Status(found, reasons, reasons, due, step_params)


def _v4_due_refusal(menu_node, live: bool) -> str | None:
    """Why a live validator stops at its first v4 epoch, or None: what refused
    v4 at startup, before the block, and still holds (`v4_refusals`)."""
    refusals = getattr(menu_node, "v4_refusals", None)
    if live and refusals:
        return ("Protocol v4 starts at this epoch and cannot run on this validator: "
                + "; ".join(refusals) + ". Fix it and restart (run --check first).")
    return None


def _v4_menu_setup(*, cfg, subtensor, netuid: int, my_hotkey: str, role, root_hotkey,
                   reveal_delay: int, live: bool, blocks_per_epoch: int, v4_block: int,
                   state_path: str, reveal_bucket: str, root_hotkeys=None, status=None,
                   wallet=None):
    """The v4 menu node (and, on the root, its publisher), after the startup
    refusals of fugal_subnet/menu_publish.startup_refusals (_v4_refusals): a
    live validator that could not read or publish the menu in time does not
    start once v4 is due. Before the first v4 epoch the refusals are not a
    reason to stop running v3: they are logged (and on the READY line), kept
    on the node (`v4_refusals`), and stop the validator at the first v4
    boundary if they still hold.

    `root_hotkeys` is the root's hotkeys by epoch (config.root_hotkeys_v4):
    each epoch's menu commitment is read under the hotkey that was the
    root's for it, so a rebuild across a change of the root's key reaches
    the menus the live nodes ran on. Without a table, one hotkey (the one
    this node knows) for every epoch. `status` is _v4_status's, computed for
    the READY line; without it the refusals are computed here."""
    from fugal_subnet import menu_publish as mp
    from fugal_subnet import reveal_sync
    from fugal_subnet.benchmarks.slicer import epoch_index_for_block
    from fugal_subnet.menu_state import MENU_LAG as ms_lag

    root = role is not None and role.root
    root_hk = my_hotkey if root else ((role.root_hotkey if role is not None else None)
                                      or root_hotkey)
    hotkeys = tuple(root_hotkeys or ()) or (((0, root_hk),) if root_hk else ())
    if status is not None:
        refusals, step_params = list(status.refusals), status.step_params
        due = status.due
    else:
        refusals, step_params = _v4_refusals(
            cfg=cfg, subtensor=subtensor, netuid=netuid, my_hotkey=my_hotkey, role=role,
            root_hotkey=root_hotkey, reveal_delay=reveal_delay,
            blocks_per_epoch=blocks_per_epoch, reveal_bucket=reveal_bucket)
        due = epoch_index_for_block(subtensor.get_current_block(),
                                    blocks_per_epoch) >= v4_block // blocks_per_epoch
    if refusals:
        text = "Protocol v4 cannot run on this validator: " + "; ".join(refusals)
        if live and due:
            raise click.ClickException(text)
        logger.warning("%s%s", text, "" if not live else
                       f" (it runs protocol v3 until block {v4_block} and stops there unless "
                       "this is fixed and it is restarted)")
    bucket = reveal_bucket if root else ((role.bucket if role is not None else None)
                                         or getattr(cfg, "REFERENCE_BUCKET", "") or None)
    base = state_path + ".menu"
    # The root's bucket, then any mirror (FUGAL_MENU_MIRRORS); every byte is
    # checked against the hash it was asked for.
    sources = bool(bucket) or bool(getattr(cfg, "MENU_MIRRORS", ()))
    lists = mp.ListBook(
        os.path.join(base, "lists"),
        fetch=(lambda kind, sha: reveal_sync.fetch_signed_list(
            kind, sha, bucket=bucket, netuid=netuid)) if sources else None)
    book = mp.MenuBook(base)
    node = mp.MenuNode(
        first_epoch=v4_block // blocks_per_epoch, book=book, lists=lists,
        read_commitment=lambda e: mp.read_menu_commitment(
            subtensor, netuid, hotkeys, e, e * blocks_per_epoch),
        fetch_menu=(lambda h: reveal_sync.fetch_menu_bytes(h, bucket=bucket, netuid=netuid))
        if sources else None,
        step_params=step_params,
        read_join=lambda e: mp.read_join_pointer(subtensor, netuid, hotkeys, e,
                                                 e * blocks_per_epoch),
        fetch_join=(lambda h: reveal_sync.fetch_join_bytes(h, bucket=bucket, netuid=netuid))
        if sources else None)
    # Any other validator with a bucket of its own mirrors what it accepted.
    node.mirror = (mp.Mirror(netuid=netuid, store=mp.GcsStore(reveal_bucket))
                   if not root and reveal_bucket else None)
    def read_reveal_sha(k):
        # The commitment for k is read at the boundary of k + MENU_LAG; before
        # the chain reaches that block there is none. Asking anyway is worse
        # than useless: the SDK caches the node's empty answer for the block's
        # hash (async_substrate_interface, chain_getBlockHash), and every later
        # pinned read of that block on this connection fails, the epoch at
        # that boundary included, for good (the local-chain v4 rehearsal).
        block = (k + ms_lag) * blocks_per_epoch
        if block > subtensor.get_current_block():
            return None
        return mp.read_reveal_commitment(subtensor, netuid, hotkeys, k, block)

    node.read_reveal_sha = read_reveal_sha

    def fetch_reveal(epoch_id: str, sha: str):
        import hashlib

        from fugal_subnet.commit_reveal import EPOCH_DIR
        try:
            with open(os.path.join(EPOCH_DIR, epoch_id, "reveal.json"), "rb") as f:
                raw = f.read(reveal_sync.MAX_REVEAL_BYTES + 1)
            if hashlib.sha256(raw).hexdigest() == sha:
                return raw
        except OSError:
            pass
        return (reveal_sync.fetch_reveal_bytes(epoch_id, sha, bucket=bucket, netuid=netuid)
                if sources else None)

    # A walk that meets a link the endpoint no longer serves re-anchors on the
    # menu the root's newest reveal names (MenuNode._anchor).
    node.read_anchor = lambda e: mp.read_root_anchor(subtensor, netuid, hotkeys, e,
                                                     e * blocks_per_epoch,
                                                     fetch_reveal=fetch_reveal)
    publisher = (mp.MenuPublisher(netuid=netuid, store=mp.GcsStore(reveal_bucket),
                                  held=book.held, lists=lists)
                 if root and reveal_bucket else None)
    # The root's joins (fugal_subnet/join_step.py): its two commitment fields,
    # seeded from what the chain holds now, and its record of the join
    # pointers it committed.
    node.root_joins = _RootJoins(os.path.join(base, "joins.json")) if root else None
    node.root_fields = None
    node.join_store = mp.GcsStore(reveal_bucket) if root and reveal_bucket else None
    if root:
        from fugal_subnet import chain_proposals as cp
        from fugal_subnet import join_step

        try:
            have = cp.read_fields_at(subtensor, netuid, subtensor.get_current_block()).get(
                my_hotkey, ())
        except Exception:  # noqa: BLE001 - seeded at the first reveal commitment instead
            have = ()
        node.root_fields = join_step.RootFields(
            lambda fields: cp.set_commitment_fields(subtensor, wallet, netuid, fields),
            reveal=have[0] if len(have) >= 1 and have[0] else None,
            join=have[1] if len(have) >= 2 and have[1] else None)
    node.v4_refusals = list(refusals) if live else []
    logger.info("Protocol v4 menu: %s, reading the commitments of %s", "ROOT (publishes it)"
                if root else "follower (audits it)", root_hk)
    return node, publisher


def _v4_published_setup(*, subtensor, netuid: int, pool, first_epoch: int, blocks_per_epoch: int,
                        role, my_hotkey: str, root_hotkey, root_hotkeys, reference_dir,
                        reveal_bucket: str, cfg, state_path: str, menu_node=None,
                        other_pool_ids=()):
    """Protocol v4's published set on this validator (memo_sync.ChainSet,
    kept beside the state file) and, on the root, the publisher of its
    snapshots (memo_sync.SnapshotPublisher, the store the menus go to):
    (set, publisher or None). `other_pool_ids` are the v5 pool's question
    ids when v5 is scheduled (`_v5_pool_ids`): v5 reveals publish votes on
    them, so the kept set and the root's snapshots name them.

    The set advances by the root's reveal commitments read pinned at each
    boundary block, on the root as on every other validator. A reveal the
    chain names is read from this node's own epoch directory (the root), its
    reference reveals (a follower), or the root's bucket and any mirror, by
    its committed sha256; a snapshot from the bucket or a mirror, by digest."""
    import hashlib

    from fugal_subnet import commit_reveal, memo_sync, memo_v4, reveal_sync
    from fugal_subnet import menu_publish as mp
    from fugal_subnet.commitments import get_commitments_with_blocks

    root = role is not None and role.root
    root_hk = my_hotkey if root else ((role.root_hotkey if role is not None else None)
                                      or root_hotkey)
    hotkeys = tuple(root_hotkeys or ()) or (((0, root_hk),) if root_hk else ())
    bucket = reveal_bucket if root else ((role.bucket if role is not None else None)
                                         or getattr(cfg, "REFERENCE_BUCKET", "") or None)
    sources = bool(bucket) or bool(getattr(cfg, "MENU_MIRRORS", ()))

    def read_at(epoch: int):
        if not hotkeys:
            return None
        hotkey = mp.hotkey_for_epoch(hotkeys, epoch)
        data, _block = get_commitments_with_blocks(
            subtensor, netuid, block=int(epoch) * blocks_per_epoch).get(hotkey, ("", 0))
        return data or None

    def reveal_bytes(epoch_id: str, sha: str):
        for directory in (commit_reveal.EPOCH_DIR if root else reference_dir,):
            if not directory:
                continue
            try:
                with open(os.path.join(directory, epoch_id, "reveal.json"), "rb") as f:
                    raw = f.read(reveal_sync.MAX_REVEAL_BYTES + 1)
            except OSError:
                continue
            if hashlib.sha256(raw).hexdigest() == sha:
                return raw
        if not sources:
            return None
        return reveal_sync.fetch_reveal_bytes(epoch_id, sha, bucket=bucket, netuid=netuid)

    def snapshot(digest: str):
        if not sources:
            return None
        return reveal_sync.fetch_object(
            memo_sync.snapshot_object(netuid, digest), bucket=bucket,
            limit=memo_v4.MAX_SNAPSHOT_BYTES,
            check=lambda b: hashlib.sha256(b).hexdigest() == digest)

    chain_set = memo_sync.ChainSet(pool=pool, read_at=read_at, reveal_bytes=reveal_bytes,
                                   snapshot=snapshot, path=state_path + memo_sync.STATE_SUFFIX,
                                   other_pool_ids=other_pool_ids)
    publisher = (memo_sync.SnapshotPublisher(netuid=netuid, store=mp.GcsStore(reveal_bucket),
                                             held=lambda: chain_set.held)
                 if root and reveal_bucket else None)
    logger.info("Protocol v4 published set: %s", "none held yet (brought forward at the first "
                "v4 boundary)" if chain_set.at is None else
                f"read through boundary {chain_set.at}")
    return chain_set, publisher


def _v5_pool_ids(v5_inputs) -> frozenset:
    """The v5 pool's question ids (none when v5 is not scheduled or its
    inputs are not loaded): what the published set may name besides the v3
    pool's, since a v5 reveal's `votes_v5` publishes outcomes on them."""
    return frozenset(v5_inputs.gold) if v5_inputs is not None else frozenset()


def _v4_epoch_view(chain_set, epoch_index: int):
    """The published set as of epoch E (memo_sync.ChainSet.view: the root's
    commitments read pinned at every boundary through E's), or None when a
    reveal the chain names cannot be had: this validator then checks no v4
    proof this epoch, as without its menu's bytes. A failed chain read
    raises: the epoch fails and is retried, as for the menu's read."""
    from fugal_subnet import memo_sync

    if chain_set is None:
        return None
    try:
        chain_set.advance_to(epoch_index)
        return chain_set.view(epoch_index)
    except memo_sync.PublishedUnavailable as e:
        logger.error("NO PUBLISHED SET for epoch %d: %s. This validator checks no proof of the "
                     "epoch; the root's reveal comes from its bucket or a mirror.", epoch_index, e)
        return None


def _v4_memo(view, plan, nonce: bytes):
    """Epoch E's memo_v4.EpochMemo on the one view its plan drew on."""
    from fugal_subnet import epoch_plan, memo_v4

    return memo_v4.epoch_memo(view, plan.pool_index, nonce, epoch_plan.mix_v4().targets)


def _v4_memo_for(chain_set, pool):
    """What a v4 re-anchor's replay builds each replayed epoch's memo with
    (v4_replay.Replayer.memo_for): epoch j and the nonce j's record kept ->
    `_v4_memo` on the published set as of j (memo_sync.ChainSet.view, the
    chain's word at the boundaries through j's, whatever the set has read
    since) and `epoch_plan.plan_v4(pool, published=<it>, epoch=j)` on the
    epoch's pool object: the memo the epoch loop built for j. Nothing is
    folded here. A set not read through j's boundary is v4_replay.Missing:
    the epoch is not replayed without the memo it ran with. None without a
    set (no v4 schedule)."""
    if chain_set is None:
        return None
    from fugal_subnet import epoch_plan, memo_sync
    from fugal_subnet.v4_replay import Missing

    def memo_for(epoch: int, nonce: bytes):
        try:
            view = chain_set.view(int(epoch))
        except memo_sync.PublishedUnavailable as e:
            raise Missing([int(epoch)], f"no published set as of it ({e})") from None
        return _v4_memo(view, epoch_plan.plan_v4(pool, published=view, epoch=int(epoch)), nonce)

    return memo_for


def _v4_publish_no_proof(*, root: bool, uploader, epoch_index: int, reveal_delay: int,
                         netuid: int, menu, snapshot, commit) -> None:
    """The root publishes what is due in an epoch with no verified proof as
    in any other (its reveal of E - 1, the menu and the snapshot it names),
    so the reveal is committed in time. Never raises."""
    if not root or uploader is None:
        return
    from fugal_subnet.commit_reveal import EPOCH_DIR
    from fugal_subnet.reveal_publisher import publish_due_reveals
    try:
        publish_due_reveals(EPOCH_DIR, epoch_index, reveal_delay, netuid, uploader,
                            commit=commit, menu=menu, snapshot=snapshot)
    except Exception as e:  # noqa: BLE001 - publishing never takes the epoch down
        logger.warning("Reveal publishing failed this epoch: %s", e)


def _v4_root_summaries(menu_node, *, root: bool, reference_dir: str | None,
                       epoch_dir: str | None = None):
    """Where a v4 validator fills the missing epochs of its window from
    (menu_publish.root_summaries): the root reads its own reveals; another
    validator reads the root's reveals its sync brought into `reference_dir`,
    each only when its bytes hash to the root's commitment for that epoch
    (the pointer ledger sync keeps, else the commitment read at the chain
    block where it is the root's for that epoch). None without a source."""
    from fugal_subnet import menu_inputs, reveal_sync
    from fugal_subnet import menu_publish as mp
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block

    if root:
        if epoch_dir is None:
            from fugal_subnet.commit_reveal import EPOCH_DIR as epoch_dir
        return mp.root_summaries(epoch_dir, None, check=menu_inputs.summary_from_dict)
    if not reference_dir:
        return None
    ledger_path = reveal_sync.pointer_ledger_path(reference_dir)

    def sha_of(k: int) -> str | None:
        got = reveal_sync.load_pointers(ledger_path).get(epoch_id_for_block(k))
        if got is None and getattr(menu_node, "read_reveal_sha", None) is not None:
            got = menu_node.read_reveal_sha(k)
        return got

    return mp.root_summaries(reference_dir, sha_of, check=menu_inputs.summary_from_dict)


def _v4_window_refusal(window_epochs: int, step_params) -> str | None:
    """Why the v4 menu cannot run with this summary window, or None: a
    window longer than the gap between two runs of one id
    (menu_inputs.check_summary_window)."""
    from fugal_subnet import menu_inputs
    try:
        menu_inputs.check_summary_window(window_epochs, step_params)
    except ValueError as e:
        return str(e)
    return None


def _v4_expected(state, *, epoch_index: int, nonce: bytes, pool: list, scored_ids, hotkey: str,
                 plan, shares=None, challenge_ids=None):
    """What one hotkey's v4 proof must hold this epoch (tee/verify.expected_v4,
    W4a): its exploration and measurement questions, on the epoch's menu.
    `nonce` is the epoch nonce as bytes, `scored_ids` the whole slice, `index`
    `plan` is epoch_plan.plan_v4(pool) (once per pool) and `shares`
    epoch_plan.explore_shares_v4(state) (once per epoch), from the caller."""
    from fugal_subnet.tee import verify as _verify

    return _verify.expected_v4(state, epoch=epoch_index, nonce=nonce, pool=pool,
                               scored_ids=scored_ids, hotkey=hotkey, plan=plan, shares=shares,
                               challenge_ids=challenge_ids)


def _served_head_bytes(responses) -> list[bytes]:
    """The head bytes each miner served this epoch in its response, whether or
    not its proof verified (bounded by HEAD_MAX_BYTES, never parsed here):
    what `_v4_holder_heads` reads as "the miner" when the holder's own proof
    did not verify or was not counted."""
    import base64

    from fugal_subnet.config import HEAD_MAX_BYTES

    out = []
    for _uid, resp in sorted((responses or {}).items()) if isinstance(responses, dict) else \
            enumerate(responses or ()):
        text = getattr(resp, "head_npz_b64", "") or ""
        if not text or len(text) > 4 * (HEAD_MAX_BYTES // 3 + 1):
            continue
        try:
            data = base64.b64decode(text, validate=True)
        except Exception:  # noqa: BLE001 - a miner's malformed field is no head
            continue
        if len(data) <= HEAD_MAX_BYTES:
            out.append(data)
    return out


def holder_head_sources(holder: str, *, held: dict[str, bytes], replay_store, served,
                        publication) -> tuple[bytes | None, str | None]:
    """The holder's head bytes and where they came from, trying each source in
    one fixed order (FAST_CROWN_V4.md 6, 11) and keeping the first whose
    sha256 is the holder's hash: what this node holds (`held`, the heads
    verified this epoch, then `replay_store`, the heads it verified in the
    last window), then the miner (`served`, the bytes miners served this
    epoch, whether or not their proofs verified), then the root's head
    publication (`publication(holder)`, reveal_sync.fetch_head). A source
    that fails or gives other bytes is passed over. (None, None) when none
    has them: the epoch is then not scored here (v2_epoch.HolderHeadMissing),
    never scored on a guess. Any source gives the same bytes, so the order
    decides only how soon they are had."""
    import hashlib

    def ok(data) -> bool:
        return data is not None and hashlib.sha256(bytes(data)).hexdigest() == holder

    if ok(held.get(holder)):
        return bytes(held[holder]), "verified"
    sources = (("replay store", lambda: replay_store(holder)),
               ("miner", lambda: next((d for d in served() if ok(d)), None)),
               ("root publication", lambda: publication(holder)))
    for name, obtain in sources:
        try:
            data = obtain()
        except Exception as e:  # noqa: BLE001 - the next source is tried
            logger.info("The holder's head %s: %s has none (%s)", holder[:16], name,
                        type(e).__name__)
            continue
        if ok(data):
            return bytes(data), name
    return None, None


def _v4_holder_heads(seat, epoch_index: int, verified_heads: dict, netuid: int,
                     responses=None) -> dict:
    """{hash: bytes} of the holder this epoch's challenges pair with
    (seat_v4.challenge_holder, FAST_CROWN_V4.md 6) and of the holder going
    into the epoch (its picks are published), each when it is not among the
    heads verified this epoch, from `holder_head_sources` in its fixed order.
    Empty when there is no holder or it served. A holder whose bytes cannot
    be had raises in run_v2_epoch (v2_epoch.HolderHeadMissing): the epoch is
    not scored here rather than scored on a guess."""
    import hashlib

    from fugal_subnet import seat_v4

    if not isinstance(seat, seat_v4.SeatStateV4):
        return {}
    held = {hashlib.sha256(bytes(b)).hexdigest(): bytes(b) for b in verified_heads.values()}

    def replay_store(h):
        from fugal_subnet import v4_replay

        return v4_replay.Store(STATE_PATH + ".v4replay").head_bytes(h)

    publication = _v4_head_publication(netuid)

    served_cache: list = []

    def served():
        if not served_cache:
            served_cache.append(_served_head_bytes(responses))
        return served_cache[0]

    out = {}
    # The holder the challenges pair with, and the holder going into the
    # epoch (whose picks the reveal publishes if it keeps the seat).
    for holder in sorted({h for h in (seat_v4.challenge_holder(seat, epoch_index),
                                      seat.champion) if h is not None} - set(held)):
        data, source = holder_head_sources(holder, held=held, replay_store=replay_store,
                                           served=served, publication=publication)
        if data is None:
            logger.error("The holder's head %s could not be had from this node, its miner or "
                         "the root's head publication: this epoch cannot be scored here",
                         holder[:16])
            continue
        logger.info("The holder's head %s came from the %s", holder[:16], source)
        out[holder] = data
    return out


def _v4_head_publication(netuid: int):
    """hash -> the head's bytes from the root's head publication
    (reveal_sync.fetch_head on the reference bucket), checked by the caller
    against the hash: where the live epoch and a replay find a holder's head
    this node does not hold."""
    def publication(h):
        from fugal_subnet import config as _cfg
        from fugal_subnet import reveal_sync

        bucket = _cfg.REFERENCE_BUCKET or reveal_sync.FINNEY_REFERENCE_BUCKET
        return reveal_sync.fetch_head(bucket, int(netuid), h)
    return publication


def _v5_setup(*, subtensor, netuid: int, role, cfg, on_finney: bool, blocks_per_epoch: int,
              reveal_bucket: str, state_path: str):
    """This validator's v5_node.V5Node: its state beside the validator's
    (`<state>.v5`), and on a follower the reader of the root's published exam
    specs (the root's reveal of E - SPEC_LAG, as the TD reads it)."""
    from fugal_subnet import reveal_sync, v5_node

    root = role is not None and role.root
    read = None
    if not root:
        hotkeys = cfg.root_hotkeys_v4(on_finney)
        bucket = (reveal_bucket or getattr(cfg, "REFERENCE_BUCKET", "")
                  or (reveal_sync.FINNEY_REFERENCE_BUCKET if on_finney else "")) or None

        def fetch(epoch_id, sha):
            return reveal_sync.fetch_reveal_bytes(epoch_id, sha, bucket=bucket, netuid=netuid)
        read = v5_node.published_spec_reader(subtensor, netuid=netuid, root_hotkey=hotkeys,
                                             blocks_per_epoch=blocks_per_epoch, fetch_reveal=fetch)
    return v5_node.V5Node(state_path + ".v5", root=root, read_published=read)


def _v5_resync(node, *, epoch_index: int, first_epoch: int, reference_dir: str | None) -> bool:
    """Rebuild `node` (v5_node.V5Node.resync) when it is stale before epoch E,
    from the root's reveals: a follower's synced reference directory first,
    then this validator's own epoch files (the root's own). Never the epoch's
    failure: a node that cannot rebuild scores on what it holds, logged."""
    from fugal_subnet import v5_node
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block
    from fugal_subnet.commit_reveal import EPOCH_DIR

    if not node.stale(epoch_index, first_epoch):
        return False
    dirs = [d for d in (reference_dir, EPOCH_DIR) if d]

    def read(e: int):
        for d in dirs:
            try:
                with open(os.path.join(d, epoch_id_for_block(e), "reveal.json"), "rb") as f:
                    return f.read()
            except OSError:
                continue
        return None
    try:
        done = node.resync(v5_node.reference_reveals(read, epoch_index), epoch_index)
    except Exception as e:  # noqa: BLE001 - scored on what it holds; logged
        logger.error("v5 state is stale before epoch %d and could not be rebuilt from the "
                     "root's reveals (%s: %s)", epoch_index, type(e).__name__, e)
        return False
    if not done:
        logger.warning("v5 state is stale before epoch %d and the root's reveal of epoch %d is "
                       "not held: scoring on the state as it is", epoch_index, epoch_index - 1)
    return done


def _v5_expected(state, *, epoch_index: int, nonce: bytes, pool: list, hotkey: str, spec, plan):
    """What one hotkey's v5 proof must hold this epoch (tee/verify.expected_v5):
    its coverage cells and the exam cells the published spec assigns it."""
    from fugal_subnet.tee import verify as _verify

    return _verify.expected_v5(state, epoch=epoch_index, nonce=nonce, pool=pool, hotkey=hotkey,
                               spec=spec, index=getattr(plan, "pool_index", None))


def _v5_uid_checks(checks, expected, slice_gold: dict, gold_answers: dict):
    """One hotkey's checks in a v5 epoch: its V5Expected, its coverage as the
    exploration, the gold for its cells, and the questions hash and set of ITS
    exam cells (tee/verify.questions_hash_v5; [] on a coverage-only epoch).
    A v5 TD answers no v4 slice, so the epoch's slice hash and ids never apply."""
    from fugal_subnet.tee.verify import questions_hash_v5

    return checks._replace(
        explore_map=expected.v4.exploration, v5=expected,
        slice_gold=_v5_gold(slice_gold, gold_answers, expected),
        questions_hash=questions_hash_v5(expected),
        question_ids={str(q) for q, _e in expected.exam_cells})


def _v5_gold(slice_gold: dict, gold_answers: dict, expected) -> dict:
    """The gold a v5 proof is graded against: the slice's, its coverage
    questions' and its exam cells' questions'."""
    extra = set(expected.v4.exploration) | {q for q, _e in expected.exam_cells}
    return {**slice_gold, **{q: gold_answers[q] for q in sorted(extra) if q in gold_answers}}


class _V5Scored(NamedTuple):
    """A v5 epoch's result in the shape the loop's later steps read: the
    reveal fields, the pooled coverage pairs the menu step reads, the heads
    whose picks count for an entry's use (holder and history)."""
    decision: object
    reveal: dict
    pooled: tuple
    counted_heads: list
    baseline: str


class _V5Inputs(NamedTuple):
    """Protocol v5's pool inputs: the pool (verified against pool_manifest_v5),
    its gold by question id, its routing embeddings (route_text rows), the
    exam part by RouterArena dataset (arena_sets.partition) and the mix."""
    pool: list
    gold: dict
    embeddings: object
    exam_by_dataset: dict
    weights_ppm: dict


def _v5_category_of(v5_inputs: _V5Inputs | None) -> dict:
    """{question id: category} for protocol v5: every v5 pool question's
    RouterArena dataset (arena_sets.dataset_key). One source for the seat's
    strata (_v5_score) and the menu step's categories (summarize_epoch,
    step_inputs): a v5 proof's exam cells and coverage pairs are v5 pool
    questions, which the v4 gold never names. The menu step's mix stays the
    v4 one (MenuParams.mix): categories outside it carry no weight, and with
    none in it the plain mean is used (menu_state.weighted_token_means,
    menu_inputs.weighted_bounds), the same on every validator."""
    from fugal_subnet.benchmarks import arena_sets

    if v5_inputs is None:
        raise RuntimeError("protocol v5 but the v5 pool was not loaded at startup")
    return {q: arena_sets.dataset_key(g) for q, g in v5_inputs.gold.items()}


V5_MENU_CATEGORY = "arena"


def _v5_menu_category_of(v5_inputs: _V5Inputs | None) -> dict:
    """{question id: category} for the menu step in protocol v5: every v5 pool
    question in ONE category. v5 prices a pick per entry (exam_v5.exam_picks),
    so the menu's pick stats need no split, and RouterArena's ~25 datasets
    would exceed menu_state.PICK_MAX_CATEGORIES. The seat's strata stay the
    datasets (_v5_category_of)."""
    return {q: V5_MENU_CATEGORY for q in _v5_category_of(v5_inputs)}


def _v5_load_inputs(cfg, *, live: bool) -> _V5Inputs:
    """Load the v5 pool and its pinned routing embeddings (fetched from
    EMBEDDINGS_V5_URL when missing or wrong). The pin must exist: "TBD" means
    this release cannot run v5 and is refused."""
    from fugal_subnet import validator_setup
    from fugal_subnet.benchmarks import arena_sets
    from fugal_subnet.benchmarks.loader import load_pool

    pin = getattr(cfg, "EMBEDDINGS_V5_SHA256", "TBD")
    if not pin or pin == "TBD":
        raise ValueError("config.EMBEDDINGS_V5_SHA256 is not pinned (TBD): this build cannot "
                         "run protocol v5")
    pool = load_pool(5, strict=live)
    path = os.getenv("FUGAL_EMBEDDINGS_V5_FILE", "data/embeddings/pool_v5.npz")
    logger.info(validator_setup.ensure_embeddings(path, pin, getattr(cfg, "EMBEDDINGS_V5_URL", None)))
    emb = Embeddings.load(path, pin)
    missing = emb.covers([q["question_id"] for q in pool])
    if missing:
        raise ValueError(f"the v5 embeddings file has no row for {len(missing)} pool "
                         f"question(s), e.g. {missing[0]}")
    exam_part, _cover = arena_sets.partition(pool)
    logger.info("Protocol v5 inputs: pool %d questions (exam part %d), embeddings for %d",
                len(pool), sum(len(v) for v in exam_part.values()), len(emb.row))
    return _V5Inputs(pool=pool, gold={q["question_id"]: q for q in pool}, embeddings=emb,
                     exam_by_dataset=exam_part, weights_ppm=arena_sets.weights_ppm())


def _v5_commit_blocks(boundary_block: int, fix_block: int | None, *, seniority: dict,
                      raw: dict) -> tuple[dict, bool]:
    """(the commit block of each uid v5 scores the epoch at `boundary_block`
    on, whether those are raw blocks). From the v0.8.1 fix block
    (config.v5_commit_fix_block) a commitment's raw block (`raw`,
    _Checked.commit_block); before it v0.8.0's v1 seniority keys
    (`seniority`, _Checked.seniority), so those epochs replay as published."""
    raw_blocks = fix_block is not None and int(boundary_block) >= int(fix_block)
    return (raw if raw_blocks else seniority), raw_blocks


def _v5_score(node, *, epoch_index: int, nonce: bytes, exam, state, verified_proofs: dict,
              verified_heads: dict, head_hashes: dict, coldkey_of: dict, commit_blocks: dict,
              raw_commit_blocks: bool = True,
              v5_inputs: _V5Inputs | None, burn_uid: int, pooled, published):
    """Protocol v5's scoring of one epoch (v5_epoch): the votes of the
    verified proofs, the seat and weights, and the spec of epoch + SPEC_LAG.
    `published` is the epoch's published set (memo_v4.Published, the view
    its plan drew on): the memorisation rule reads it (v5_epoch).
    Returns (result, uids, weights, commit): `commit` advances `node` (called
    at the loop's commit point, after the reveal and the weights)."""
    from fugal_subnet import menu_state as ms
    from fugal_subnet import v5_epoch
    from fugal_subnet.head_eval import load_head_v4

    if v5_inputs is None:
        raise RuntimeError(f"epoch {epoch_index} is protocol v5 but the v5 pool and embeddings "
                           "were not loaded at startup")
    spec, costs, void = exam
    votes, cover = [], []
    served = []
    for uid in sorted(verified_proofs):
        proof, ck = verified_proofs[uid], coldkey_of.get(uid, "")
        ex, co = v5_epoch.votes_from_proof(epoch_index, proof, ck)
        votes += ex
        cover += co
        h = head_hashes.get(uid)
        if h:
            served.append(v5_epoch.Served(h, uid, str(proof.hotkey), ck,
                                          int(commit_blocks.get(uid, 0))))
            if uid in verified_heads:
                try:
                    node.keep_head(h, verified_heads[uid])
                except (ValueError, OSError) as e:
                    logger.warning("UID %d: its head's bytes could not be kept (%s: %s)", uid,
                                   type(e).__name__, e)
    counts = spec is not None and void is None
    record = v5_epoch.EpochRecord(
        epoch=epoch_index, spec=spec if counts else None, costs_nd=dict(costs) if counts else {},
        votes=tuple(votes), coverage=tuple(cover),
        flagged=frozenset(v5_epoch.implausible(votes)),
        void=None if counts or spec is None else void)
    # The stratum of a question is its RouterArena dataset (arena_sets.dataset_key).
    bench = _v5_category_of(v5_inputs)
    full = [e.id for e in state.entries if ms.is_full(e)]
    first_published = v5_epoch.published_reader(published)
    result = v5_epoch.score_epoch(
        epoch=epoch_index, state=node.seat, heads=node.heads, record=record,
        window=node.records, served=served, full_entries=full,
        stratum_of=lambda q: bench.get(q, ""), burn_uid=burn_uid,
        first_published=first_published, raw_commit_blocks=raw_commit_blocks)
    # The next exam, from the heads verified by now.
    loaded = {}
    for h, data in node.head_bytes.items():
        try:
            loaded[h] = load_head_v4(data)
        except ValueError:
            continue
    # The exam draws only the pool's exam part (coverage never touches it), on
    # RouterArena's dataset shares.
    try:
        next_spec, next_costs = v5_epoch.next_spec(
            epoch=epoch_index, nonce=nonce, decision=result.decision, heads=result.heads,
            served=served, loaded=loaded, state_menu=state,
            hidden_of=lambda qids: v5_inputs.embeddings.aligned(list(qids)),
            by_dataset=v5_inputs.exam_by_dataset, weights_ppm=dict(v5_inputs.weights_ppm),
            baseline=result.baseline,
            first_published=first_published,
            used=v5_epoch.spec_questions([*(s for s, _c in node.specs.values()),
                                          *(r.spec for r in node.records), spec]))
    except Exception as e:  # noqa: BLE001 - no exam then; never the epoch
        logger.error("Epoch %d: the exam spec of epoch %d could not be built (%s: %s); that "
                     "epoch is coverage only", epoch_index, epoch_index + v5_epoch.SPEC_LAG,
                     type(e).__name__, e)
        next_spec, next_costs = None, {}
    reveal = dict(result.reveal)
    if next_spec is not None:
        reveal.update(v5_epoch.spec_reveal(next_spec, next_costs))
    d = result.decision
    scored = _V5Scored(decision=d, reveal=reveal, pooled=tuple(pooled),
                       counted_heads=sorted({h for h in (d.holder, *d.history) if h}),
                       baseline=result.baseline)

    def commit():
        node.commit(epoch_index, result, record, next_spec, next_costs)
    return scored, result.uids, result.weights, commit


def _v4_gold(slice_gold: dict, gold_answers: dict, expected) -> dict:
    """The gold a v4 proof is graded against: the slice's, the hotkey's
    exploration questions' and the epoch's challenge list's."""
    extra = set(expected.exploration) | set(getattr(expected, "challenge", ()))
    return {**slice_gold, **{q: gold_answers[q] for q in sorted(extra) if q in gold_answers}}


def _v4_answers(verified_proofs: dict, expected: dict, coldkey_of: dict) -> list:
    """Every verified v4 proof's exploration results as pool answers; a pair
    is private when its question is in that hotkey's private set
    (V4Expected.private_ids)."""
    from fugal_subnet.explore_pool import ExploreAnswer

    return [
        ExploreAnswer(
            question_id=r.question_id, model=r.routed_model, coldkey=coldkey_of[uid],
            correct=bool(r.correct), failed=bool(r.failed),
            private=r.question_id in expected[uid].private_ids,
            prompt_tokens=int(r.prompt_tokens), completion_tokens=int(r.completion_tokens),
            latency_ms=int(getattr(r, "latency_ms", None) or 0))
        for uid in sorted(verified_proofs) for r in verified_proofs[uid].exploration_results]


def _v4_odds_input(*, base, state, shares, expected: dict, verified_proofs: dict,
                   coldkey_of: dict, verified_heads: dict, boundary_block: int,
                   blocks_per_epoch: int):
    """The seat's odds input under v4: `base` (the v3 OddsInput the loop
    builds) with v4's pooling (answers private by the hotkey's private set,
    both weight tables of the epoch's shares, the v4 coldkey cap), v4 heads,
    and the epoch's own menu state: every head's odds are
    head_odds.pair_odds_v4 on it, a missing row scored at what the entry's
    allowance gives (labels_v4). v4 has no grace and no full-menu rule, so
    no `menu` and no `unscored` entries; and no frame or agreement detector
    feeds the pool (`flagged` is empty: SEAT_MINIMAL_V4.md 2.3 item 32),
    whose plausibility check reads its own minimum sample
    (config.EXPLORE_PRIVATE_MIN_SAMPLES_V4)."""
    import dataclasses

    from fugal_subnet import config as _config
    from fugal_subnet import explore_pool
    from fugal_subnet import menu_state as _ms
    from fugal_subnet.benchmarks.slicer import epoch_index_for_block

    if not isinstance(state, _ms.MenuState) or (
            epoch_index_for_block(boundary_block, blocks_per_epoch) != state.epoch):
        raise ValueError("a v4 epoch's odds are computed on that epoch's own menu state")
    return dataclasses.replace(
        base, answers=_v4_answers(verified_proofs, expected, coldkey_of),
        heads=_odds_heads(verified_heads, 4), coldkey_cap=explore_pool.coldkey_cap(4),
        explore_weights=shares.shared_weights(), private_weights=shares.private_weights(),
        menu=None, unscored=frozenset(), menu_state=state, flagged=frozenset(),
        implausible_min_samples=_config.EXPLORE_PRIVATE_MIN_SAMPLES_V4)


def _v4_pool(state, *, verified_proofs: dict, expected: dict, coldkey_of: dict, nonce: bytes,
             flagged, shares=None) -> list:
    """The epoch's pooled exploration pairs under v4 (explore_pool.pool_pairs
    with both weight tables of the menu's shares and the v4 operator cap). A
    pair is private when its question is in that hotkey's private set
    (V4Expected.private_ids)."""
    from fugal_subnet import epoch_plan, explore_pool

    if shares is None:
        shares = epoch_plan.explore_shares_v4(state)
    answers = _v4_answers(verified_proofs, expected, coldkey_of)
    return explore_pool.pool_pairs(
        answers, nonce=nonce, coldkey_cap=explore_pool.coldkey_cap(4), flagged=flagged,
        weights=shares.shared_weights(), private_weights=shares.private_weights())


def _v4_counted_heads(seat, senior_bars) -> list[str]:
    """The heads whose scored picks count for an entry's use (section 9.2):
    those that held the seat (the champion and the former champions the seat
    keeps) and those that were senior bars this epoch (every head in
    `senior_bars`' values: v2_epoch.V2EpochResult.senior_bars, the bars the
    seat was decided with). A senior bar can hold no seat now: a retired
    champion still barring through its retire window."""
    bars = {h for barred_by in senior_bars.values() for h in barred_by}
    return sorted({h for h in (seat.champion, *seat.history) if h} | bars)


def _v4_menu_epoch(menu_node, menu_publisher, v4_menu, *, epoch_index: int, nonce: bytes,
                   verified_proofs: dict, coldkey_of: dict, pooled, category_of: dict,
                   counted_heads, window_epochs: int, root: bool, uploader, reveal_delay: int,
                   netuid: int, commit, root_summaries=None, snapshot=None, served=(),
                   join_audit=None) -> tuple[dict, bool, object]:
    """Protocol v4's menu work after scoring epoch k, before reveal k is
    written: (the reveal's menu fields, whether the due reveals were already
    published this epoch, and for the root a function that makes its menu
    step and returns the reveal's fields, called after its weights and its
    joins for k + 1: `_v4_join_step`; None for every other validator).

    Every v4 reveal lists the hotkeys whose proofs this validator verified
    (`served_v4`, chain_proposals.SERVED_FIELD: the root's decides who may
    propose), and another validator's carries its `join_audit` of the
    root's joins for k.

    1. Summarise epoch k (menu_inputs.summarize_epoch over its verified
       proofs, each with the coldkey the pinned metagraph gives its hotkey,
       and its pooled pairs) and keep the summary in the window beside the
       state (summary_to_dict).
    2. The root publishes the due reveals FIRST (reveal k - 1, whose
       commitment names the menu of k + 1 when its publisher confirmed the
       bytes), so it knows which menu the step for k starts from.
    3. The step's inputs are menu_inputs.step_inputs(start, MenuState(k),
       window, proofs_k, ...), a function of `start`, MenuState(k + 1): the
       root calls it now with the menu it predicts; every other validator
       keeps it and calls it at k + 2 with the menu the chain gave, for its
       audit.
    `root_summaries` (menu_publish.root_summaries, or None): where the
    window's missing epochs are filled from, the root's published summaries,
    each checked against the root's commitment; this node's own summaries are
    never replaced. The root's reveal carries epoch k's summary.
    `snapshot` (the root's memo_sync.SnapshotPublisher): the published-set
    snapshot each due reveal names goes to the store before the commitment.
    Never raises: the reveal and the weights do not wait on it."""
    from fugal_subnet import menu_publish as mp
    from fugal_subnet import menu_rules as mr
    from fugal_subnet.commit_reveal import EPOCH_DIR

    inputs, summary_doc, rebuilt = None, None, ()
    if v4_menu is not None and v4_menu.ok:
        try:
            from fugal_subnet import menu_inputs

            state_k = v4_menu.state
            proofs = [menu_inputs.VerifiedProof(verified_proofs[uid], coldkey_of[uid])
                      for uid in sorted(verified_proofs)]
            summary = menu_inputs.summarize_epoch(
                epoch_index, state_k, proofs, pooled=pooled, category_of=category_of,
                counted_heads=counted_heads)
            summary_doc = menu_inputs.summary_to_dict(summary)
            menu_node.book.keep_summary(epoch_index, summary_doc)
            if root_summaries is not None:
                filled = menu_node.book.fill_from_root(
                    range(epoch_index - window_epochs + 1, epoch_index), root_summaries)
                if filled:
                    logger.info("The v4 window took %d epoch summaries from the root's reveals",
                                len(filled))
            rebuilt = tuple(menu_node.book.from_root(through=epoch_index, size=window_epochs))
            window = tuple(menu_inputs.summary_from_dict(d) for d in menu_node.book.summaries(
                through=epoch_index, size=window_epochs))
            lists = menu_node.lists

            def inputs(start, _k=epoch_index, _state=state_k, _window=window, _proofs=tuple(proofs)):
                return menu_inputs.step_inputs(
                    start, _state, _window, _proofs, epoch=_k, nonce=nonce,
                    lab_list=lists.newest(mr.KIND_MENU_ALLOWLIST), params=menu_node.step_params,
                    category_of=category_of, device_list=lists.newest(mr.KIND_DEVICE_LIST),
                    lab_list_in_force=lists.get(mr.KIND_MENU_ALLOWLIST, start.labs))
        except Exception as e:  # noqa: BLE001 - no step inputs: no menu step this epoch
            inputs = None
            logger.error("The menu step inputs of epoch %d could not be built (%s: %s)",
                         epoch_index, type(e).__name__, e)
    published = False
    if root and uploader is not None:
        from fugal_subnet.reveal_publisher import publish_due_reveals
        try:
            publish_due_reveals(EPOCH_DIR, epoch_index, reveal_delay, netuid, uploader,
                                commit=commit, menu=menu_publisher, snapshot=snapshot)
        except Exception as e:  # noqa: BLE001 - publishing never takes the epoch down
            logger.warning("Reveal publishing failed this epoch: %s", e)
        published = True
    from fugal_subnet import chain_proposals as cp
    from fugal_subnet import join_step

    extra = {cp.SERVED_FIELD: sorted(set(served))}
    if join_audit is not None:
        extra[join_step.AUDIT_FIELD] = join_audit
    root_joins = getattr(menu_node, "root_joins", None)

    def fields_now() -> dict:
        got = mp.epoch_reveal_fields(
            menu_node, v4_menu, k=epoch_index, inputs=inputs, root=root,
            committed_menu_of=lambda j: mp.committed_menu(EPOCH_DIR, j), summary=summary_doc,
            rebuilt=rebuilt,
            committed_join_of=root_joins.pointer if root_joins is not None else None,
        ) if v4_menu else {}
        return {**got, **extra}

    if root and root_joins is not None:
        return {}, published, fields_now
    return fields_now(), published, None


# --- protocol v4: the joins (FAST_CROWN_V4.md 18; fugal_subnet/join_step.py) ------------


class _RootJoins:
    """The root's record of the join pointers it committed, by epoch
    ({epoch: delta hash or None}), in `<state>.menu/joins.json`: what its
    next step starts from (MenuNode.predict_start's `committed_join_of`)."""

    def __init__(self, path: str | None):
        self.path, self._got = path, {}
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                self._got = {int(k): (v if isinstance(v, str) else None)
                             for k, v in raw.items() if str(k).isdigit()}
            except (OSError, ValueError, AttributeError):
                self._got = {}

    def record(self, epoch: int, pointer) -> None:
        self._got[int(epoch)] = None if pointer is None else pointer.delta_hash
        for old in sorted(self._got)[:-64]:
            del self._got[old]
        if self.path:
            try:
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({str(k): v for k, v in sorted(self._got.items())}, f)
                os.replace(tmp, self.path)
            except OSError as e:
                logger.warning("Could not record the join pointer of epoch %d (%s)", epoch,
                               type(e).__name__)

    def pointer(self, epoch: int):
        from fugal_subnet import chain_proposals as cp

        if int(epoch) not in self._got:
            return None
        return cp.JoinPointer(int(epoch), self._got[int(epoch)])


def _v4_served(reveal_path: str, sha256: str | None = None) -> tuple[str, ...] | None:
    """The hotkeys a reveal file lists as verified (`served_v4`), when its
    bytes hash to `sha256` (None: the root reading its own); None when the
    file cannot be read or is not that reveal."""
    import hashlib

    from fugal_subnet import chain_proposals as cp
    from fugal_subnet.reveal_sync import MAX_REVEAL_BYTES

    try:
        with open(reveal_path, "rb") as f:
            raw = f.read(MAX_REVEAL_BYTES + 1)
        if len(raw) > MAX_REVEAL_BYTES:
            return None
        if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
            return None
        return tuple(sorted(cp.served_hotkeys(json.loads(raw))))
    except (OSError, ValueError):
        return None


def _v4_chain_read(subtensor, netuid: int, *, epoch: int, blocks_per_epoch: int,
                   root_hotkey: str, reveal_dir: str, check_sha: bool, bucket=None):
    """The chain read of epoch `epoch`'s joins pinned at its block B
    (join_step.chain_read_at), with the hotkeys the root's named reveal
    lists as verified read from `reveal_dir` (the root's own reveals, or a
    follower's synced copies of them, checked against the root's sha), else
    fetched from the root's `bucket` or a mirror by that sha."""
    from fugal_subnet import chain_proposals as cp
    from fugal_subnet import chain_reads, join_step

    def coldkeys_at(block):
        mg = chain_reads.pinned(subtensor, block, lambda s: s.metagraph(netuid, block=block),
                                what="Metagraph")
        return {str(h): str(c) for h, c in zip(mg.hotkeys, mg.coldkeys)}

    return join_step.chain_read_at(
        epoch=epoch, blocks_per_epoch=blocks_per_epoch,
        fields_at=lambda b: cp.read_fields_at(subtensor, netuid, b), coldkeys_at=coldkeys_at,
        block_hash_at=lambda b: str(subtensor.get_block_hash(b)), root_hotkey=root_hotkey,
        served_of=lambda eid, sha: _v4_served_of(reveal_dir, eid, sha, check_sha=check_sha,
                                                 netuid=netuid, bucket=bucket))


def _v4_served_of(reveal_dir, epoch_id: str, sha: str, *, check_sha: bool, netuid: int,
                  bucket=None):
    """The `served_v4` hotkeys of reveal `epoch_id`: the copy on disk when it
    is the one named, else the bytes the root's bucket or a mirror serves
    for that sha256 (reveal_sync.fetch_reveal_bytes); None when neither."""
    import json as _json

    from fugal_subnet import chain_proposals as cp
    from fugal_subnet import reveal_sync

    got = (_v4_served(os.path.join(reveal_dir, epoch_id, "reveal.json"),
                      sha if check_sha else None) if reveal_dir else None)
    if got is not None or not check_sha:
        return got
    try:
        raw = reveal_sync.fetch_reveal_bytes(epoch_id, sha, bucket=bucket, netuid=netuid)
        return None if raw is None else tuple(sorted(cp.served_hotkeys(_json.loads(raw))))
    except (ValueError, OSError):
        return None


def _v4_proposal_fetcher(dendrite, subtensor, netuid: int, block: int):
    """fetch_document(hotkey, proposal) for the join step: the document from
    the proposer's own axon (protocol.FugalProposalSynapse), as the
    metagraph at block B lists it; None when it does not answer in
    JOIN_FETCH_TIMEOUT_S. Every document of the epoch is fetched at once in
    one event loop (join_step.ProposalFetcher.fetch_many). The bytes are
    checked by hash by the admission."""
    from fugal_subnet import chain_reads, join_step

    mg = chain_reads.pinned(subtensor, block, lambda s: s.metagraph(netuid, block=block),
                            what="Metagraph")
    return join_step.ProposalFetcher(dendrite, {str(h): a for h, a in zip(mg.hotkeys, mg.axons)})


def _v4_join_step(*, subtensor, netuid: int, dendrite, menu_node, store, root_fields,
                  root_joins, epoch_index: int, blocks_per_epoch: int, my_hotkey: str,
                  fetch_listing=None) -> None:
    """The root, in epoch k: epoch k + 1's joins (join_step), after its
    weights and before its menu step for k. Waits for block B; past the
    commit deadline it does nothing (no joins, the proposals wait). Never
    raises."""
    from fugal_subnet import chain_proposals as cp
    from fugal_subnet import join_step
    from fugal_subnet import menu_publish as mp
    from fugal_subnet import menu_rules as mr
    from fugal_subnet.commit_reveal import EPOCH_DIR

    j = int(epoch_index) + 1
    block_b = cp.read_block(j, blocks_per_epoch)
    deadline = j * blocks_per_epoch - cp.commit_slack(blocks_per_epoch)
    try:
        if subtensor.get_current_block() >= deadline:
            logger.warning("Too late for epoch %d's joins; no joins, the proposals wait", j)
            root_joins.record(j, None)
            return
        wait_for_block(subtensor, block_b)
        held = menu_node.book.held(epoch_index)
        if held is None:
            logger.error("No menu held for epoch %d: no joins for %d", epoch_index, j)
            return
        base = mp.read_epoch_menu(j, held, mp.committed_menu(EPOCH_DIR, j - menu_node.lag),
                                  menu_bytes=menu_node.menu_bytes, lists=menu_node.lists,
                                  params=menu_node.reader_params)
        if base.state is None:
            logger.error("No base menu for epoch %d (%s): no joins", j, base.detail)
            return
        read = _v4_chain_read(subtensor, netuid, epoch=j, blocks_per_epoch=blocks_per_epoch,
                              root_hotkey=my_hotkey, reveal_dir=EPOCH_DIR, check_sha=False)
        docs, listings = join_step.gather(
            read, fetch_document=_v4_proposal_fetcher(dendrite, subtensor, netuid, block_b),
            fetch_listing=fetch_listing or join_step.fetch_public_listing)
        allowlist = (menu_node.lists.get(mr.KIND_MENU_ALLOWLIST, base.state.labs)
                     if base.state.labs.serial else None)

        def publish(raw: bytes, h: str) -> bool:
            from fugal_subnet import reveal_sync

            menu_node.book.keep_bytes(raw)
            if store is None:
                return False
            key = reveal_sync.join_object(netuid, h)
            store.put(key, raw)
            return store.get(key, len(raw) + 1) == raw

        out = join_step.run_join_step(
            epoch=j, base=base.state, read=read, allowlist=allowlist, documents=docs,
            listings=listings, publish=publish, commit=root_fields.commit_join,
            time_left=lambda: subtensor.get_current_block() < deadline)
        root_joins.record(j, out.pointer)
    except Exception as e:  # noqa: BLE001 - no joins this epoch, never a failed epoch
        logger.error("The join step for epoch %d failed (%s: %s); no joins", j,
                     type(e).__name__, e)


def _v4_join_audit(subtensor, netuid: int, *, menu_node, v4_menu, blocks_per_epoch: int,
                   root_hotkey, reference_dir, bucket=None) -> dict | None:
    """Another validator's `join_audit` of the root's joins for the epoch
    `v4_menu` is of (join_step.audit_join). None when it cannot be made
    (no base menu, no reveals); never raises."""
    from fugal_subnet import join_step
    from fugal_subnet import menu_publish as mp
    from fugal_subnet import menu_rules as mr

    if v4_menu is None or v4_menu.base is None:
        return None
    try:
        epoch = v4_menu.epoch
        hk = mp.hotkey_for_epoch(root_hotkey, epoch)
        read = _v4_chain_read(subtensor, netuid, epoch=epoch, blocks_per_epoch=blocks_per_epoch,
                              root_hotkey=hk, reveal_dir=reference_dir, check_sha=True,
                              bucket=bucket)
        pointer = v4_menu.join
        raw = (menu_node.join_bytes(pointer.delta_hash)
               if pointer is not None and pointer.delta_hash else None)
        base = v4_menu.base
        allowlist = (menu_node.lists.get(mr.KIND_MENU_ALLOWLIST, base.labs)
                     if base.labs.serial else None)
        return join_step.audit_join(base=base, read=read, root_delta_raw=raw,
                                    root_pointer=pointer, allowlist=allowlist)
    except Exception as e:  # noqa: BLE001 - an audit that cannot run decides nothing
        logger.warning("The join audit of epoch %d could not run (%s)", v4_menu.epoch,
                       type(e).__name__)
        return None


class _V2Sync(NamedTuple):
    changed: bool
    # (scoring_state, window, seat, copy_state, window_degraded, gap_until, anchor_at)
    state: tuple
    # After a re-anchor: what the adopted state was built under (the reference
    # reveal's `v2_state.rules`; None for a v2 state). `reanchored` says whether
    # `rules` is to be taken; otherwise the validator keeps its own.
    rules: dict | None = None
    reanchored: bool = False
    # Protocol v4: what a re-anchor replaced the state with ({"snapshot": the
    # reference epoch, "replayed": this validator's own epochs run again on
    # it}), for the reveal of the epoch it happened in; None otherwise.
    audit: dict | None = None


def _v2_sync_step(*, epoch_index: int, last_epoch_index: int, first_v2_epoch: int,
                  reference_dir: str | None, scoring_state, window, seat, copy_state,
                  window_degraded: bool, gap_until: int | None, anchor_at: int | None,
                  prices, bench_category=None, epoch_dir: str | None = None,
                  window_epochs: int | None = None, tables=None, replay=None,
                  reanchor_now: bool = False) -> _V2Sync:
    """Detect missed v2 epochs; while this validator has a gap, verify against
    the reference and re-anchor on it (fugal_subnet/v2_sync).

    - A v2 epoch this validator did not observe opens a gap (`gap_until`): the
      validator keeps deciding as usual (blocking its takes would only make it
      diverge whenever its peers take), and is re-checked against the reference.
    - Only a validator WITH a gap re-anchors. A healthy validator that
      disagrees with its reference logs an alarm and keeps its state: the
      reference may be the one that is wrong, and following it would spread
      its outage to everyone downstream.
    - A re-anchor records the epoch it happened at (`anchor_at`); only a
      reference epoch at or after it can verify it (this validator's earlier
      reveals predate the fix). The gap closes on a verified match at or after
      both the gap and the last re-anchor.
    - Protocol v4 (`replay`, a v4_replay.Replayer): a re-anchor replays this
      validator's epochs after the reference epoch instead of keeping them;
      a span it cannot replay keeps the state and moves the gap past it. The
      reference's first reveal after a gap in its own reveals (the root was
      down) is a re-anchor point at once (`_root_came_back`), and so is its
      newest reveal when this validator has just rebuilt a lost window
      (`reanchor_now`).

    Never raises (I6): any failure keeps the state and the gap.
    """
    state = (scoring_state, window, seat, copy_state, window_degraded, gap_until, anchor_at)
    changed = False
    if last_epoch_index >= 0 and epoch_index > last_epoch_index + 1 \
            and epoch_index - 1 >= first_v2_epoch:
        gap_until = max(-1 if gap_until is None else gap_until, epoch_index - 1)
        (logger.error if reference_dir else logger.warning)(
            "v2 epochs %d..%d were not run by this validator%s",
            max(last_epoch_index + 1, first_v2_epoch), epoch_index - 1,
            (": its state is unverified until it matches the reference again (decisions go "
             "on as usual; its reveals are marked v2_sync.unverified)") if reference_dir else "")
        state, changed = state[:5] + (gap_until, anchor_at), True
    try:
        return _v2_sync_inner(epoch_index=epoch_index, reference_dir=reference_dir,
                              state=state, changed=changed, prices=prices,
                              bench_category=bench_category, epoch_dir=epoch_dir,
                              window_epochs=window_epochs, tables=tables, replay=replay,
                              reanchor_now=reanchor_now)
    except Exception as e:  # noqa: BLE001 - never stops the epoch (I6); state and gap kept
        logger.error("v2 sync against the reference failed (%s); state and gap kept", e)
        return _V2Sync(changed, state)


def _check_v3_blocks_aligned(v3_block, odds_block, epoch_interval, menu_b_blk=None,
                             v4_block=None) -> None:
    """Refuse a v3 or odds-seat block that is not an epoch boundary, the way
    the v2 block is refused and the way a v3 miner refuses it
    (miner.py, validate_activation_block). Accepting it here while miners
    refuse it would leave validators scoring a flip no v3 miner can start at
    (v3 review, 2026-11-16)."""
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch, validate_activation_block

    bpe = blocks_per_epoch(epoch_interval)
    for name, block in (("v3 activation", v3_block), ("odds-seat", odds_block),
                        ("menu B", menu_b_blk), ("v4 activation", v4_block)):
        try:
            validate_activation_block(block, bpe)
        except ValueError as e:
            raise click.ClickException(f"{name} block: {e}") from e


def _rebuild_tables(epoch_inputs, config_module, inputs_v3b=None, *, v4_tables=None) -> dict:
    """{protocol: (prices, bench_category)} for every protocol whose inputs this
    validator loaded (v2 on), so a rebuild or re-anchor prices each replayed
    reveal at its own protocol's table (v3 review, finding 1).

    With v3 menu B scheduled, v3's table is menu B's: menu A's entries at
    menu A's prices plus the added ones, so a reveal from either side of the
    menu B block is priced exactly as the live epoch priced it.

    v4 has no table of its own: its prices are each epoch's menu. With
    `v4_tables` (the current epoch's _V4Tables) a v4 reveal is priced at the
    current menu's prices and pick-cost table and categorised by the v4 mix
    ((prices, bench_category, pick_spend)); without (at startup, before any
    menu is read) protocol 4 is left out."""
    out = {p: (inputs[2], getattr(config_module, f"BENCHMARK_CATEGORY_V{p}", None))
           for p, inputs in sorted(epoch_inputs.items()) if 2 <= p < 4}
    if inputs_v3b is not None and 3 in out:
        out[3] = (inputs_v3b[2], out[3][1])
    if v4_tables is not None:
        out[4] = (v4_tables.prices, v4_tables.bench_category, v4_tables.pick_spend)
    return out


def _rebuild_v2_state(rebuild_dir, epoch_inputs, config_module, inputs_v3b, scoring_state,
                      subtensor, blocks_per_epoch, *, v4_tables=None):
    """The v2 state rebuilt from `rebuild_dir`'s reveals for a validator that has
    none: (window, seat, rules, copy state, window degraded, gap until or None),
    or None when there is nothing to rebuild from. Burn-in counters are
    restored into `scoring_state` as fresh records.

    Without `v4_tables` a newest reveal of protocol v4 is not adopted
    ("v4"): its prices are its menu's, which the caller reads at its first v4
    boundary and then calls again with that epoch's tables."""
    from fugal_subnet import copy_rule
    from fugal_subnet.benchmarks.slicer import epoch_index_for_block
    from fugal_subnet.evidence import Evidence
    from fugal_subnet.scoring import MinerRecord
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals

    rebuilt = rebuild_v2_from_reveals(
        rebuild_dir, window_size=config_module.CHAMPION_WINDOW_EPOCHS,
        # Each replayed reveal is priced at its own protocol's table (a v3
        # reveal at v2's left every v3 cell unpriced; v3 review, finding 1).
        prices=epoch_inputs.get(2, epoch_inputs[1])[2],
        bench_category=getattr(config_module, "BENCHMARK_CATEGORY_V2", None),
        tables=_rebuild_tables(epoch_inputs, config_module, inputs_v3b, v4_tables=v4_tables),
    )
    if rebuilt.newest_epoch_id is None:
        return None
    rules = rebuilt.extra.get("rules")
    if v4_tables is None and int((rules or {}).get("protocol", 2)) >= 4:
        logger.info("v2 state: the newest reveal (%s) is protocol v4; rebuilding at the first "
                    "v4 boundary, with its menu", rebuilt.newest_epoch_id)
        return "v4"
    for h, uid in sorted(rebuilt.serving.items()):
        if uid in scoring_state.records or h not in rebuilt.burn_in:
            continue
        # No hotkey: `owners` is the head's FIRST committing hotkey, and
        # an owner may serve from a newer one (coldkey ownership).
        # update_scores treats "" as unknown and keeps the record; a
        # stale value made it drop the record, and the restored
        # burn-in with it, on the first verified proof (F-066).
        rec = MinerRecord(uid=uid, hotkey="", current_head_hash=h)
        rec.evidence = Evidence(weights_hash=h, burn_in_n=rebuilt.burn_in[h])
        scoring_state.records[uid] = rec
    resume_epoch = epoch_index_for_block(subtensor.get_current_block(), blocks_per_epoch)
    behind = resume_epoch - 1 - int(rebuilt.newest_epoch_id.lstrip("e"))
    gap = None
    if behind > 0:
        # The reveals lag by the hold: the epochs since are a gap the
        # verify-and-resync step closes once the reference passes them.
        gap = resume_epoch - 1
        logger.warning(
            "v2 state rebuilt from reveals that end %d epoch(s) before this "
            "validator resumes: unverified until the reference passes e%08d "
            "(verify-and-resync).", behind, gap,
        )
    else:
        logger.info("v2 state rebuilt exactly from reveals up to %s", rebuilt.newest_epoch_id)
    return (rebuilt.window, rebuilt.seat, rules,
            copy_rule.from_rebuild(rebuilt) or copy_rule.CopyState(), False, gap)


def _v2_sync_inner(*, epoch_index, reference_dir, state, changed, prices, bench_category,
                   epoch_dir, window_epochs, tables=None, replay=None,
                   reanchor_now=False) -> _V2Sync:
    import copy as _copy

    from fugal_subnet import copy_rule
    from fugal_subnet.commit_reveal import EPOCH_DIR
    from fugal_subnet.config import CHAMPION_WINDOW_EPOCHS
    from fugal_subnet.evidence import Evidence
    from fugal_subnet.scoring import MinerRecord
    from fugal_subnet.v2_sync import DRIFT_CHECKS, check, drifted, load_reveals, resync

    scoring_state, window, seat, copy_state, window_degraded, gap_until, anchor_at = state
    epoch_dir = epoch_dir or EPOCH_DIR
    window_epochs = window_epochs or CHAMPION_WINDOW_EPOCHS
    if gap_until is not None and epoch_index - gap_until > window_epochs:
        # Bounded: after a full window the gap's epochs are out of the window,
        # and a reference that never verifies (none set, or incompatible) must
        # not keep the validator re-anchoring for good.
        logger.warning("v2 gap (through e%08d) closed after a full window without a verified "
                       "match: this validator's seat state may still differ from its peers'",
                       gap_until)
        return _V2Sync(True, state[:5] + (None, None))
    if not reference_dir:
        # No reference: this validator is a ROOT (the others reference it, a
        # star). It has nothing to verify against, so it never marks itself
        # unverified, which would only stop its followers from verifying.
        if gap_until is not None:
            # Not an error for a root, and worded so it is not read as one: an
            # epoch in which no miner had a valid proof leaves the seat state
            # exactly as it was, and the root is what the others verify
            # against. It matters only if the root was DOWN while its
            # followers scored epochs.
            logger.warning("Root: no verified proof in the v2 epoch(s) through e%08d. Nothing "
                           "to do if no miner served then: the seat state is unchanged, and a "
                           "root has no reference to verify against. If this validator was "
                           "down while others scored those epochs, see docs/VALIDATOR_GUIDE.md, "
                           "\"If the root misses epochs\".", gap_until)
            return _V2Sync(True, state[:5] + (None, None))
        return _V2Sync(changed, state)
    since = epoch_index - window_epochs - 1
    ref, own = load_reveals(reference_dir, since), load_reveals(epoch_dir, since)
    chk = check(ref, own, epoch_index)
    if reanchor_now and replay is not None and chk.k is not None:
        # A lost window just rebuilt from the reference's newest reveal: bring
        # it forward over this validator's own epochs since (v4 replay).
        from fugal_subnet.v2_sync import SyncCheck

        if gap_until is None or gap_until < chk.k:
            gap_until = chk.k
        return _v4_reanchor(epoch_index=epoch_index, reference_dir=reference_dir, ref=ref,
                            own=own, chk=SyncCheck(chk.k, "differs"),
                            state=state[:5] + (gap_until, anchor_at), prices=prices,
                            bench_category=bench_category, window_epochs=window_epochs,
                            tables=tables, replay=replay)
    if (replay is not None and chk.status in ("in_sync", "differs")
            and (gap_until is None or chk.k >= gap_until)
            and _root_came_back(ref, own, chk.k, anchor_at)):
        # Protocol v4: the reference's first reveal after a gap in its own
        # reveals (the root was down while this validator ran) is a re-anchor
        # point, whether or not the views compared there differ: this
        # validator's window holds epochs the root's does not. The root is the
        # reference: this validator's epochs during the outage were its own
        # best knowledge, and from here it replays them on the root's state
        # rather than waiting for drift detection to notice a difference.
        from fugal_subnet.v2_sync import SyncCheck

        logger.warning("e%08d is the reference's first reveal after a gap in its reveals that "
                       "this validator ran through: re-anchoring on it", chk.k)
        gap_until = chk.k if gap_until is None else gap_until
        return _v4_reanchor(epoch_index=epoch_index, reference_dir=reference_dir, ref=ref,
                            own=own, chk=SyncCheck(chk.k, "differs"),
                            state=state[:5] + (gap_until, anchor_at), prices=prices,
                            bench_category=bench_category, window_epochs=window_epochs,
                            tables=tables, replay=replay)
    if gap_until is None:
        if chk.status != "differs":
            return _V2Sync(changed, state)
        if drifted(ref, own, epoch_index) is None:
            logger.error("ALARM: this validator's v2 state differs from its reference's at "
                         "e%08d with no gap to explain it; it re-anchors if that persists "
                         "for %d reference epochs. Compare the two reveals.", chk.k, DRIFT_CHECKS)
            return _V2Sync(changed, state)
        # Persistent: a proof this validator did not receive (or the
        # reference's own) left the states apart, and nothing else will bring
        # them together. In a star the reference is the arbiter, so this is a
        # gap at k: re-anchor on it below, then verify (Fable review, item 5).
        logger.error("v2 state has differed from the reference's for %d reference epochs "
                     "through e%08d: treating it as a gap and re-anchoring on the reference",
                     DRIFT_CHECKS, chk.k)
        gap_until, changed = chk.k, True
        state = state[:5] + (gap_until, anchor_at)
    if chk.status == "no_reference":
        logger.error("v2 gap open (through e%08d) but %s holds no reference reveal before "
                     "e%08d: the root's reveals have not been fetched (see the 'Root reveals' "
                     "line above, or the operator's own sync of that directory).",
                     gap_until, reference_dir, epoch_index)
        return _V2Sync(changed, state)
    if anchor_at is not None and chk.k < anchor_at:
        return _V2Sync(changed, state)              # wait for a reference epoch after the fix
    if chk.k < gap_until:
        # The reference has not yet published past the gap: a re-anchor now
        # could not cover the unobserved epochs and would only be repeated.
        return _V2Sync(changed, state)
    if chk.status == "in_sync":
        logger.info("v2 state verified against the reference at e%08d: gap closed", chk.k)
        return _V2Sync(True, state[:5] + (None, None))
    if replay is not None:
        return _v4_reanchor(epoch_index=epoch_index, reference_dir=reference_dir, ref=ref,
                            own=own, chk=chk, state=state, prices=prices,
                            bench_category=bench_category, window_epochs=window_epochs,
                            tables=tables, replay=replay)
    rs = resync(ref, own, chk.k, epoch_index, window_size=window_epochs, prices=prices,
                bench_category=bench_category, tables=tables)
    new_copy = copy_rule.from_rebuild(rs.rebuilt) or copy_rule.CopyState()
    scoring = _copy.deepcopy(scoring_state)
    for uid, (h, c) in (rs.burn_in or {}).items():
        rec = scoring.records.get(uid)
        if rec is None:
            rec = scoring.records[uid] = MinerRecord(uid=uid, hotkey="", current_head_hash=h)
        if rec.evidence is None or rec.evidence.weights_hash != h:
            rec.current_head_hash = h
            rec.evidence = Evidence(weights_hash=h)
        rec.evidence.burn_in_n = float(c)
    logger.warning("v2 state %s the reference's at e%08d: re-anchored on it; the gap stays "
                   "open until a reference epoch from e%08d on verifies it%s",
                   "differed from" if chk.status == "differs" else "was missing for", chk.k,
                   epoch_index, "" if rs.complete else
                   " (an epoch since was not observed: the next check re-anchors again)")
    return _V2Sync(True, (scoring, rs.rebuilt.window, rs.rebuilt.seat, new_copy, False,
                          gap_until, epoch_index),
                   rules=rs.rebuilt.extra.get("rules"), reanchored=True)


def _root_came_back(ref, own, k: int, anchor_at: int | None) -> bool:
    """Whether the reference's reveal of `k` is its first after a gap in its
    reveals that this validator ran through: some epoch between the
    reference's previous reveal and `k` has a reveal of this validator's own
    and none of the reference's. Not when this validator has re-anchored
    since (`anchor_at` after `k`)."""
    if anchor_at is not None and k < anchor_at:
        return False
    earlier = [j for j in ref if j < k]
    if not earlier:
        return False
    return any(j in own and j not in ref for j in range(max(earlier) + 1, k))


class _KeepsInputs:
    """`fn` (v2_epoch.run_v2_epoch), keeping the keyword arguments of its last
    call (`last`): what `_keep_v4_record` keeps of a v4 epoch."""

    def __init__(self, fn):
        self.fn, self.last = fn, None

    def __call__(self, **kwargs):
        self.last = dict(kwargs)
        return self.fn(**kwargs)


def _v4_replay_span(reveal_delay: int) -> int:
    """The most of its own epochs a v4 re-anchor replays: three, or the
    reveal hold if that is more (the steady state's span, E - 1 - hold).

    Each replayed epoch costs one whole `run_v2_epoch`, at most about 8
    minutes at 256 UIDs in the worst shapes of
    scripts/experiments/time_v4_epoch.py (2026-11-21), so three replayed
    epochs and the live epoch's own step take about 32 of the 72-minute
    epoch, under half of it. A longer span (the reference's reveals behind,
    or read late) is not replayed: v4_replay.Missing keeps the state and
    moves the gap past the span, and the validator re-anchors on a later
    reference epoch (docs/INVARIANTS.md, the v4 replay entry)."""
    return max(3, int(reveal_delay))


def _v4_replayer(gold_answers: dict, embeddings, max_span: int, memo_for=None,
                 holder_bytes=None):
    """What a v4 re-anchor replays with (v4_replay.Replayer): this
    validator's epoch records beside its state, the pinned embeddings, the
    pool's question sets, at most `max_span` epochs (`_v4_replay_span`).
    `memo_for` ((epoch, its recorded nonce) -> memo_v4.EpochMemo or None,
    `_v4_memo_for`) gives each replayed epoch's memorisation memo, as the
    epoch loop built it; None replays every epoch without one.
    `holder_bytes` (`_v4_head_publication`) gives a holder's head the
    replayed seat names and neither the record nor the head store holds."""
    from fugal_subnet.v4_replay import Replayer

    return Replayer(STATE_PATH + ".v4replay", hidden_of=embeddings.of,
                    benchmark_of={q: str(g.get("benchmark", "")) for q, g in gold_answers.items()},
                    max_span=max_span, memo_for=memo_for, holder_bytes=holder_bytes)


def _keep_v4_record(v2_kwargs: dict, verified_heads: dict, window_epochs: int) -> None:
    """Keep a v4 epoch's inputs beside the state (v4_replay.Store), and the
    records of the last `window_epochs` + 2 epochs only. Never raises (I6): a
    record not kept is an epoch a later re-anchor cannot replay, which it
    says, waiting for a reference epoch past it."""
    from fugal_subnet import v4_replay

    try:
        doc, blobs = v4_replay.record(v2_kwargs, verified_heads)
        v4_replay.Store(STATE_PATH + ".v4replay").keep(doc, blobs, keep_epochs=window_epochs + 2)
    except Exception as e:  # noqa: BLE001 - the epoch is scored and set; only its replay is lost
        logger.error("Epoch %s: its inputs could not be kept for a later re-anchor (%s: %s); "
                     "a re-anchor that needs it will wait for a reference epoch past it",
                     v2_kwargs.get("epoch_id"), type(e).__name__, e)


def _v4_reanchor(*, epoch_index, reference_dir, ref, own, chk, state, prices, bench_category,
                 window_epochs, tables, replay) -> _V2Sync:
    """Protocol v4's re-anchor: the reference's state and whole window at
    chk.k, with this validator's epochs chk.k+1 .. epoch_index-1 replayed on
    it from their records (v4_replay), not kept. A span it cannot replay (a
    record missing or unreadable, or longer than the bound) replaces
    nothing: the state is kept, and the gap moves past the epochs it could
    not replay, so the next re-anchor is on a later reference epoch."""
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block
    from fugal_subnet.v2_sync import resync
    from fugal_subnet.v4_replay import Missing

    scoring_state, _w, _s, _c, _d, gap_until, anchor_at = state
    what = "differed from" if chk.status == "differs" else "was missing for"
    # The reference's whole window at k: the reveals the check read, and the
    # older ones by path (the rebuild reads and checks each file itself).
    reference = dict(ref)
    for j in range(chk.k - window_epochs + 1, chk.k + 1):
        path = os.path.join(reference_dir, epoch_id_for_block(j), "reveal.json")
        if j not in reference and os.path.isfile(path):
            reference[j] = (path, {"epoch_id": epoch_id_for_block(j)})
    try:
        rs = resync(reference, own, chk.k, epoch_index, window_size=window_epochs,
                    prices=prices, bench_category=bench_category, tables=tables, replay=replay)
    except Missing as e:
        logger.error("v2 state %s the reference's at %s, but this validator cannot replay its "
                     "own epochs since on it (%s): its state is NOT replaced; it stays "
                     "unverified and re-anchors on a reference epoch from %s on", what,
                     epoch_id_for_block(chk.k), e, epoch_id_for_block(max(e.epochs)))
        return _V2Sync(True, state[:5] + (max(gap_until, max(e.epochs)), anchor_at))
    done = rs.replayed
    replayed = [epoch_id_for_block(j) for j in done.epochs]
    logger.warning("v2 state %s the reference's at %s: re-anchored on it and replayed this "
                   "validator's own epochs %s on it (the reveals it published for them stand as "
                   "published); the gap stays open until a reference epoch from %s on verifies it",
                   what, epoch_id_for_block(chk.k), replayed or "(none)",
                   epoch_id_for_block(epoch_index))
    return _V2Sync(True, (scoring_state, done.window, done.seat, done.copy_state, False,
                          gap_until, epoch_index),
                   rules=done.rules, reanchored=True,
                   audit={"snapshot": epoch_id_for_block(chk.k), "replayed": replayed})


def _next_v2_state(v2_result, window, seat, copy_state, degraded: bool, window_epochs: int):
    """(window, seat, copy state, degraded) carried into the next epoch.

    A v1 epoch carries the v2 state unchanged. A v2 epoch takes its result; a
    degraded window stays degraded until it has refilled to `window_epochs`.
    """
    if v2_result is None:
        return window, seat, copy_state, degraded
    return (v2_result.window, v2_result.decision.state, v2_result.copy_state,
            degraded and len(v2_result.window.slices) < window_epochs)


def _score_proof(proof, slice_gold: dict, prices: dict, epoch_frame, ref_model: str):
    """A verified proof's HeadScore, both sides of thrift from shared data.

    The denominator was already built this way — pinned rates plus the frame's
    measured verbosity — while the numerator was the miner's own account of
    what it spent, and even the denominator's prompt tokens came from the
    miner's report. One half unfakeable, the other a self-report, and a miner
    could lower its cost by shrinking numbers rather than by routing better.
    Prompt tokens come from the POOL, and the miner's cost is computed from
    which models it chose. Only scored questions count (_proof_to_head_score).
    """
    from fugal_subnet.config import FRAME_DEFAULT_COMPLETION_TOKENS
    from fugal_subnet.pricing import policy_cost_total, question_input_tokens
    from fugal_subnet.reference_frame import reference_cost

    scored = proof.scored_results
    pool_prompt_tokens = sum(
        question_input_tokens(slice_gold[r.question_id]["prompt"])
        for r in scored if r.question_id in slice_gold
    )
    ref_cost = reference_cost(
        epoch_frame, prices, ref_model,
        prompt_tokens=pool_prompt_tokens,
        n_questions=len(scored),
        default_completion_tokens=FRAME_DEFAULT_COMPLETION_TOKENS,
    )
    head_cost = policy_cost_total(
        [(r.question_id, r.routed_model) for r in scored],
        slice_gold, prices, epoch_frame, FRAME_DEFAULT_COMPLETION_TOKENS,
    )
    return _proof_to_head_score(proof, ref_cost, head_cost)


def _copy_rule_snapshot(hotkeys, coldkeys, chain_commitments: dict) -> list:
    """The boundary snapshot for the v2 copy rule: every committed hotkey with
    its owning coldkey and its commitment block, read pinned at the boundary.

    A hotkey with no coldkey in the metagraph has no owner to attribute a head
    to, so it is left out rather than given an empty one.
    """
    from fugal_subnet import copy_rule

    ck_of_hotkey = {
        hk: (str(coldkeys[u]) if u < len(coldkeys) else "")
        for u, hk in enumerate(hotkeys)
    }
    return [
        copy_rule.Commitment(hk, ck_of_hotkey[hk], c[0], int(c[1]))
        for hk, c in sorted(chain_commitments.items())
        if ck_of_hotkey.get(hk)
    ]


def _v1_weights(records: dict, routing_decisions: dict, commit_blocks: dict, dupes,
                confidence: float, burn_uid: int):
    """Protocol v1 weights: (uids, weights, every disqualified uid).

    The epsilon rule runs now that this epoch's composite scores exist: a head
    that routes like a senior one must beat it or it is a copy. The frontier's
    confidence is paid out and the rest burns. It is a factor common to every
    miner, so it is applied AFTER the scores are normalised; inside
    composite() it would cancel.
    """
    from fugal_subnet.dedup import epsilon_disqualified
    from fugal_subnet.rewards import compute_weights

    near_copies = epsilon_disqualified(
        routing_decisions, commit_blocks,
        {uid: records[uid].composite_score for uid in routing_decisions if uid in records},
    )
    if near_copies:
        logger.info("Epsilon rule disqualified: %s", sorted(near_copies))
        dupes = set(dupes) | near_copies
    uids, weights = compute_weights(
        records, dedup_disqualified=dupes, paid_fraction=confidence, burn_uid=burn_uid,
    )
    if confidence < 1.0:
        logger.info(
            "Frontier confidence %.2f: %.0f%% of emission burns to UID %d this epoch",
            confidence, 100.0 * (1.0 - confidence), burn_uid,
        )
    return uids, weights, dupes


def _burn_epoch_weights(subtensor, wallet, netuid, my_uid, boundary_block, metagraph, live,
                        prev_uids, prev_weights, epoch_index, blocks_per_epoch, *,
                        protocol_version: int = 2):
    """Set the all-burn vector for a v2 epoch with no valid proof; returns the
    (uids, weights) the next epoch's cap starts from.

    Through v3 it is capped against the previous vector like every epoch, so
    it moves at most MAX_WEIGHT_DELTA per epoch. A v4 vector is never capped
    (SEAT_MINIMAL_V4.md 2.2 #28), this one included: it is the burn uid at
    1.0 on every validator, whatever each set before. Never raises (I6): an
    unreadable owner hotkey (live refuses to guess the burn UID) or a failed
    submit logs and keeps the previous vector.
    """
    try:
        burn_uid = resolve_burn_uid(subtensor, netuid, boundary_block, metagraph, live)
        uids, weights, _ = (([burn_uid], [1.0], False) if protocol_version >= 4
                            else _capped([burn_uid], [1.0], prev_uids, prev_weights))
        ok, msg = set_weights_with_retry(
            subtensor, wallet, netuid, my_uid, uids, weights,
            deadline_block=_weights_deadline(epoch_index, blocks_per_epoch),
        )
        if ok:
            logger.info("No valid proofs: the miner share burns this epoch (UID %d)", burn_uid)
            return list(uids), list(weights)
        logger.error("No valid proofs: setting the burn vector failed: %s", msg)
    except Exception as e:  # noqa: BLE001 - an epoch with no proofs must still end cleanly
        logger.error("No valid proofs: could not set the burn vector: %s", e)
    return prev_uids, prev_weights


def _capped(uids, weights, prev_uids, prev_weights):
    """(uids, weights, capped): the change from last epoch's weights bounded,
    when there were any (rewards.cap_weight_change)."""
    from fugal_subnet.rewards import cap_weight_change

    if not prev_uids:
        return uids, weights, False
    uids, weights = cap_weight_change(uids, weights, prev_uids, prev_weights)
    return uids, weights, True


def _proof_to_head_score(proof, ref_cost, head_cost):
    """Convert a verified BenchmarkProof into a HeadScore for scoring.

    `ref_cost` is what the reference model would have cost on this same
    question set — a fact about the model pool, identical for every miner in
    the epoch and independent of how many miners are online.

    Only scored routes count. Exploration is a cost the subnet imposed, not a
    choice the miner made, so billing it against their thrift would penalise
    them for the sampling that makes everyone's scores meaningful.
    """
    from fugal_subnet.head_eval import HeadScore

    scored = proof.scored_results
    return HeadScore(
        accuracy=proof.accuracy,
        cost_efficiency=0.0,   # superseded by thrift; see scoring.composite
        kl_score=0.0,
        routing_decisions=np.array([], dtype=np.int32),
        correct_mask=np.array([r.correct for r in scored], dtype=bool),
        coverage=1.0,
        n_correct=proof.n_correct,
        n_scored=proof.n_total,
        # The POLICY's cost, computed by this validator from the pool, the
        # pinned rates and the frame. proof.scored_cost_usd is what the miner
        # says it spent; it is still checked for internal consistency because a
        # miner lying about it is worth knowing, but it decides no score.
        total_head_cost=head_cost,
        total_oracle_cost=ref_cost,
        total_kl=0.0,
    )


def _extract_routing_decisions(proof, model_index, keep=None):
    """Routing decisions as a numpy array for dedup, in a GLOBAL model space.

    `model_index` maps every model in the pinned price table to a fixed index
    shared by all miners, so index 5 means the same model in every vector.

    Previously the index space was built per-proof from that miner's own routed
    models, which broke the comparison both ways: two miners each routing 100%
    to a *different* single model both produced all-zero vectors and were
    clustered as clones, while a genuine copy that re-routed a single question
    to an alphabetically-earlier model renumbered its entire vector and evaded
    detection at 96.7% identical routing.

    Ordered by question id so two miners' vectors line up element-wise
    regardless of the order results appear in their proofs. `keep` (v4)
    limits it to the slice: a v4 proof's scored results are its challenge's
    own picks, the slice first (FAST_CROWN_V4.md 2).
    """
    scored = sorted((r for r in proof.scored_results if keep is None or r.question_id in keep),
                    key=lambda r: r.question_id)
    return np.array(
        [model_index.get(r.routed_model, -1) for r in scored],
        dtype=np.int32,
    )


if __name__ == "__main__":
    main()
