"""Verify-and-resync: a validator checks its v2 state against a reference's.

WHY. Each epoch's evidence (the miners' proofs) can only be observed live, so
a validator that missed an epoch, or saw a different set of proofs, carries a
v2 state (seat, comparison window, copy rule, burn-in) that differs from its
peers', and the seat's history keeps it different. Its peers' state is
published in their reveals, but only after the weight commit-reveal hold (a
reveal shown earlier would let a lazy validator copy the weights), so some
lag after a gap is unavoidable. This module closes the gap at that lag,
automatically:

  CHECK    once a reference validator's reveal for epoch k is available,
           compare its consensus view (the v2 state block and the weights)
           with this validator's own reveal for k;
  RESYNC   if they differ, or this validator has no reveal for k, adopt the
           reference's state AT k (v2_rebuild with snapshot_epoch_id), keep
           this validator's own observations of the epochs since (the window
           entries of its own reveals after k), and carry the burn-in counters
           forward over those epochs exactly as scoring.update_scores does
           (+ questions when a head is verified under the same hash, 0 on a
           miss). The check repeats every epoch until the views match.

What stays approximate until a later check matches: seat or copy-rule events
inside the k+1 .. now-1 span are applied at `now` instead (a take's
`champion_since`, a retirement's epoch), and the weight cap starts from this
validator's own previous weights. Both show up as a mismatch at the next
check, which re-anchors again; a span with no such event converges exactly.

PROTOCOL v4 does not keep the epochs after k: it REPLAYS them (`resync`'s
`replay`, fugal_subnet/v4_replay.py). A v4 epoch's label pool depends on the
window it was pooled on (the plausibility check's reference is fitted on the
window's earlier pairs), so an epoch kept from a window that differed
carried the difference past every re-anchor. The replay runs each epoch
k+1 .. now-1 again, in order, on the reference's state at k, from the
inputs this validator recorded when it ran it; the result is the state a
validator that ran every epoch holds, copy-rule and seat events included.

The reference is the operator's choice (a directory of a validator's
published reveals, FUGAL_FRAME_BOOTSTRAP, kept in sync with its bucket).
Nothing here touches a score rule: it only decides which state the rules run
from.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _index(epoch_id: str) -> int:
    from fugal_subnet.v2_rebuild import _epoch_index

    return _epoch_index(epoch_id)


def load_reveals(directory: str | os.PathLike, since: int = 0) -> dict[int, tuple[str, dict]]:
    """{epoch index: (path, reveal)} of every readable protocol-v2 reveal of
    an epoch at or after `since` (directories named by epoch id; older ones
    are not read)."""
    out: dict[int, tuple[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(str(directory), "*", "reveal.json"))):
        try:
            if _index(os.path.basename(os.path.dirname(path))) < since:
                continue
            with open(path, encoding="utf-8") as f:
                rv = json.load(f)
            if int(rv.get("protocol_version", 1)) < 2 or not rv.get("epoch_id"):
                continue
            out[_index(rv["epoch_id"])] = (path, rv)
        except (OSError, ValueError, TypeError, AttributeError) as e:
            logger.warning("v2 sync: skipping unreadable reveal %s: %s", path, e)
    return out


# Protocol v4's reveal fields (menu_publish.reveal_fields). `menu` names the
# menu the epoch ran on, which every node reads from the chain, so it is part
# of the view validators are compared on. The others differ by role: only the
# root publishes the menu it derived (`menu_next`, its hash, the step's events
# and the signed lists it newly names), and only the others publish an audit
# of it (`menu_audit`).
MENU_FIELD = "menu"
MENU_ROLE_FIELDS = ("menu_next", "menu_next_hash", "menu_events", "menu_lists", "menu_audit")
# The root's summary of the epoch's verified proofs (menu_inputs.summary_to_dict):
# public, so a validator that lost its window can rebuild it. Root only.
MENU_SUMMARY_FIELD = "menu_summary"


def consensus_view(reveal: Mapping) -> str:
    """The part of a reveal every honest validator must agree on, canonical.

    Zero weights are dropped. The weight cap (rewards.cap_weight_change) keeps,
    at 0.0, every uid the validator's own previous weights held, so whether a
    0.0 entry is present depends on that validator's history (a validator
    started from a rebuild has none), not on the epoch. The chain never sees
    them either: the SDK filters zero weights out before set_weights
    (convert_weights_and_uids_for_emit). Comparing them kept a rebuilt
    validator re-anchoring until its gap timed out (final review, finding 1).

    A protocol v4 reveal's `menu` (the menu the epoch ran on) joins the view;
    a reveal without one (v1-v3) gives the view it always gave."""
    view = {"v2_state": reveal.get("v2_state"),
            "weights": nonzero_weights(reveal.get("weights"))}
    if MENU_FIELD in reveal:
        view[MENU_FIELD] = reveal.get(MENU_FIELD)
    return json.dumps(view, sort_keys=True, separators=(",", ":"))


def nonzero_weights(weights):
    """`weights` without its 0.0 entries (see consensus_view); anything that
    is not a mapping is returned as is."""
    if isinstance(weights, Mapping):
        return {k: w for k, w in weights.items() if w != 0}
    return weights


# Fields of a reveal that legitimately differ between agreeing validators:
# `environment` (platform, thread counts; its score-affecting subset is
# `consensus_digest`, which is compared) and `v2_sync` (whether this
# validator's own state is verified yet), and protocol v4's role fields
# (MENU_ROLE_FIELDS: the root's derived menu, a follower's audit of it), a
# follower's audit of the root's joins and the hotkeys each validator itself
# verified (chain_proposals: only the root's list decides who may propose).
PER_VALIDATOR_FIELDS = (("environment", "v2_sync") + MENU_ROLE_FIELDS + (MENU_SUMMARY_FIELD,)
                        + ("join_audit", "served_v4"))


def comparable_reveal(reveal: Mapping) -> dict:
    """The part of a reveal two agreeing validators must publish identically:
    everything but PER_VALIDATOR_FIELDS, with zero weights dropped."""
    out = {k: v for k, v in reveal.items() if k not in PER_VALIDATOR_FIELDS}
    if "weights" in out:
        out["weights"] = nonzero_weights(out["weights"])
    return out


@dataclass(frozen=True)
class SyncCheck:
    k: int | None          # the reference epoch compared (None: nothing to compare yet)
    status: str            # "no_reference", "in_sync", "differs", "own_missing"


def check(reference: Mapping[int, tuple[str, dict]], own: Mapping[int, tuple[str, dict]],
          before: int) -> SyncCheck:
    """Compare at the newest reference epoch k < `before` that the reference
    itself had verified: a reveal published while its validator was unverified
    since a missed epoch (`v2_sync.unverified`) is skipped, so two validators
    with gaps cannot confirm each other's wrong state."""
    ks = [k for k in reference if k < before
          and not ((reference[k][1].get("v2_sync") or {}).get("unverified"))]
    if not ks:
        return SyncCheck(None, "no_reference")
    k = max(ks)
    if k not in own:
        return SyncCheck(k, "own_missing")
    same = consensus_view(reference[k][1]) == consensus_view(own[k][1])
    return SyncCheck(k, "in_sync" if same else "differs")


# A follower with no gap re-anchors on its reference once its state has
# differed on this many consecutive verified reference epochs (drifted()).
# More than one, so a difference that heals by itself (the weight cap
# converging after an outage) is not followed; few, because every epoch of a
# split costs agreement (Fable review, item 5: one lost proof split a
# validator for good).
DRIFT_CHECKS = 3


def drifted(reference: Mapping[int, tuple[str, dict]], own: Mapping[int, tuple[str, dict]],
            before: int, n: int = DRIFT_CHECKS) -> int | None:
    """The oldest of the newest `n` verified reference epochs before `before`
    if this validator has a reveal for each and every one of them differs from
    it; else None. A single lost proof (a miner's reply that reached the
    reference but not this validator) leaves no gap, because the epoch was
    observed, but leaves the state different for good: this is how it shows."""
    ks = sorted((k for k in reference if k < before
                 and not ((reference[k][1].get("v2_sync") or {}).get("unverified"))),
                reverse=True)[:n]
    if len(ks) < n or any(k not in own for k in ks):
        return None
    if all(consensus_view(reference[k][1]) != consensus_view(own[k][1]) for k in ks):
        return min(ks)
    return None


def carry_burn_in(snapshot_burn_in: Mapping[str, float], snapshot_serving: Mapping[str, int],
                  own: Mapping[int, tuple[str, dict]], k: int, before: int,
                  ) -> dict[int, tuple[str, float]] | None:
    """{uid: (head, burn-in counter)} after epoch before-1, from the snapshot
    at k and this validator's own reveals k+1 .. before-1, by the update_scores
    rule: a verified head under the same hash adds the epoch's questions, a new
    hash starts from them, a miss resets to 0. None if an own reveal in the
    span is missing (this validator did not observe that epoch)."""
    counters: dict[int, tuple[str, float]] = {
        int(uid): (h, float(snapshot_burn_in.get(h, 0.0)))
        for h, uid in snapshot_serving.items()
    }
    for j in range(k + 1, before):
        if j not in own:
            return None
        rv = own[j][1]
        verified = {int(u): str(h) for u, h in (rv.get("head_hashes") or {}).items()}
        n = float(len(rv.get("questions") or []))
        for uid in list(counters):
            if uid not in verified:
                counters[uid] = (counters[uid][0], 0.0)
        for uid, h in verified.items():
            prev = counters.get(uid)
            counters[uid] = (h, (prev[1] if prev is not None and prev[0] == h else 0.0) + n)
    return counters


@dataclass
class Resync:
    rebuilt: object                                  # v2_rebuild.V2Rebuild
    burn_in: dict[int, tuple[str, float]] | None     # None: a span epoch was not observed
    k: int
    # Protocol v4: the state after replaying k+1 .. before-1 on the
    # reference's state at k (v4_replay.Replayed); None before v4.
    replayed: object | None = None

    @property
    def complete(self) -> bool:
        return self.burn_in is not None


def resync(reference: Mapping[int, tuple[str, dict]], own: Mapping[int, tuple[str, dict]],
           k: int, before: int, *, window_size: int, prices, bench_category=None,
           tables=None, replay=None) -> Resync:
    """The reference's state at k, the window from its reveals up to k and
    this validator's own after k (both within `window_size` of before-1).

    With `replay` (protocol v4: a v4_replay.Replayer, called with the
    rebuild at k, k and `before`), the reference's window and state at k
    (its reveals k-window_size+1 .. k) are brought forward by replaying
    this validator's epochs k+1 .. before-1, not by keeping them; `own` is
    not read. v4_replay.Missing propagates: nothing is adopted."""
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals

    if replay is not None:
        paths = [reference[j][0] for j in sorted(reference) if k - int(window_size) < j <= k]
        rebuilt = rebuild_v2_from_reveals(paths, window_size=window_size, prices=prices,
                                          bench_category=bench_category,
                                          snapshot_epoch_id=reference[k][1]["epoch_id"],
                                          tables=tables)
        # No burn-in under v4: nothing to carry.
        return Resync(rebuilt, {}, k, replayed=replay(rebuilt, k, before))
    oldest = before - int(window_size)
    paths = [reference[j][0] for j in sorted(reference) if oldest <= j <= k]
    paths += [own[j][0] for j in sorted(own) if k < j < before and j >= oldest]
    rebuilt = rebuild_v2_from_reveals(paths, window_size=window_size, prices=prices,
                                      bench_category=bench_category,
                                      snapshot_epoch_id=reference[k][1]["epoch_id"],
                                      tables=tables)
    burn_in = carry_burn_in(rebuilt.burn_in, rebuilt.serving, own, k, before)
    from fugal_subnet.champion import SeatState

    if not isinstance(rebuilt.seat, SeatState):
        # Protocol v4's seat (seat_v4): whether a head stands is read from
        # the epochs it was verified in, so the heads this validator saw
        # served after k are seen then, as the live seat records them.
        from fugal_subnet import seat_v4

        rebuilt.seat = seat_v4.carry_seen(rebuilt.seat, [(j, own[j][1]) for j in sorted(own)
                                                         if k < j < before])
    return Resync(rebuilt, burn_in, k)
