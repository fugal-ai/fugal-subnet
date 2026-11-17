#!/usr/bin/env python3
"""Protocol v3 flip rehearsal, in process: v2 epochs -> v3 activation -> odds seat.

No chain, no Docker, no spend; seconds. It drives the shipped epoch code
(`fugal_subnet.v2_epoch.run_v2_epoch`, the same call neurons/validator.py makes)
through a timeline that crosses both v3 blocks, with the synthetic proofs,
heads and exploration answers of tests/test_v2_epoch.py and tests/test_v3_epoch.py:

    python scripts/rehearsal_v3.py

Timeline (chain epoch = index + 1):

    e0..e2   protocol v2: three UIDs serve heads, the seat is decided on net value.
    e3 (F)   v3 activation: rules {"protocol": 3, "seat": "net_value"}. The
             window and the seat reset (a new protocol starts from nothing).
             Exploration is pooled and every head's odds are computed and
             published in shadow.
    e8 (O)   the odds-seat block: rules {"protocol": 3, "seat": "odds"}. The seat
             resets, the window is kept, and the seat is decided on odds: the
             head that predicts (UID 6) must take it.
    e9..e12  odds epochs.

Three validators:
    A   the reference.
    B   receives every exploration answer in the opposite order, is killed after
        e5 and restarted from its persisted window (.npz) and seat (JSON).
    C   starts at e10 from nothing but A's published reveals (a rebuilt follower).
Every epoch they must publish the same reveal and set the same weights.

SHORTENED, and only here: a take needs 60 paired questions on net value and
200 pairs on odds (CHAMPION_MIN_PAIRED_QUESTIONS 1,500 and ODDS_MIN_PAIRS
1,500 in the release), the margin on odds is 0.1 points. Every other constant
is the shipped default.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from fugal_subnet.champion import OddsParams  # noqa: E402
from fugal_subnet.copy_rule import from_rebuild  # noqa: E402
from fugal_subnet.v2_epoch import (  # noqa: E402
    load_window,
    save_window,
    seat_from_dict,
    seat_params_from_config,
    seat_to_dict,
)
from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals  # noqa: E402
from fugal_subnet.v2_sync import consensus_view  # noqa: E402
from tests.test_v2_epoch import CATEGORY, PRICES, H, run  # noqa: E402
from tests.test_v2_rebuild import reveal_of, same_window  # noqa: E402
from tests.test_v3_epoch import POLICIES, UIDS, odds_input  # noqa: E402

F, O, LAST = 3, 8, 12          # v3 activation, odds-seat block, last epoch index
KILL_AFTER, C_STARTS = 5, 10


def rules_for(e):
    if e < F:
        return None
    return {"protocol": 3, "seat": "odds" if e >= O else "net_value"}


def params_for(e):
    if e < F:
        return dataclasses.replace(seat_params_from_config(protocol_version=2),
                                   min_paired_questions=60)
    p = dataclasses.replace(seat_params_from_config(protocol_version=3), min_paired_questions=60)
    if e >= O:
        p = dataclasses.replace(p, odds=OddsParams(margin_points=0.1, min_pairs=200,
                                                   nv_noninferiority=0.5, ref_prior=20,
                                                   confirm_pairs=200))
    return p


def oi_for(e, reverse=False):
    if e < F:
        return None
    oi = odds_input(e)
    return dataclasses.replace(oi, answers=list(reversed(oi.answers))) if reverse else oi


class Validator:
    def __init__(self, name, reverse=False):
        self.name, self.reverse = name, reverse
        self.window = self.seat = self.copy = None
        self.rules = None

    def step(self, e):
        pols = {u: POLICIES[u] for u in UIDS}
        r = run(e, pols, window=self.window, seat=self.seat, copy=self.copy,
                params=params_for(e), odds_input=oi_for(e, self.reverse),
                rules=rules_for(e), state_rules=self.rules)
        self.window, self.seat, self.copy = r.window, r.decision.state, r.copy_state
        self.rules = rules_for(e)
        rv = json.loads(json.dumps(reveal_of(e, r, pols)))   # as published
        rv["protocol_version"] = 3 if e >= F else 2           # as the validator stamps it
        return r, rv

    def restart(self, tmp):
        """Kill: keep only what the validator persists (window .npz, seat JSON)."""
        path = os.path.join(tmp, f"{self.name}.v2window.npz")
        save_window(path, self.window)
        seat_json = json.dumps(seat_to_dict(self.seat))
        size = self.window.size
        self.window = load_window(path, size)
        self.seat = seat_from_dict(json.loads(seat_json))


class Report:
    def __init__(self):
        self.ok, self.fail = [], []

    def check(self, cond, what):
        (self.ok if cond else self.fail).append(what)
        print(("PASS  " if cond else "FAIL  ") + what, flush=True)


def main() -> int:
    rep = Report()
    a, b = Validator("A"), Validator("B", reverse=True)
    c = None
    reveals = []
    events = {}
    with tempfile.TemporaryDirectory() as tmp:
        for e in range(LAST + 1):
            ra, rva = a.step(e)
            rb, rvb = b.step(e)
            reveals.append(rva)
            events[e] = (ra.decision.event, ra.decision.champion)
            rep.check(rva == rvb and ra.weights == rb.weights,
                      f"e{e}: B (reversed answers{', restarted' if e > KILL_AFTER else ''}) "
                      f"publishes A's reveal and weights")
            if e >= F:
                rep.check(consensus_view(rva) == consensus_view(rvb),
                          f"e{e}: A and B agree on the consensus view (odds digest)")
            if e == KILL_AFTER:
                b.restart(tmp)
            if e == C_STARTS - 1:
                d = os.path.join(tmp, "reveals")
                for rv in reveals:
                    os.makedirs(os.path.join(d, rv["epoch_id"]))
                    with open(os.path.join(d, rv["epoch_id"], "reveal.json"), "w") as f:
                        json.dump(rv, f)
                rb_ = rebuild_v2_from_reveals(d, window_size=48, prices=PRICES,
                                              bench_category=CATEGORY)
                try:
                    same_window(rb_.window, a.window)
                    same = True
                except AssertionError:
                    same = False
                rep.check(same, f"e{e}: a follower rebuilt from reveals has A's window")
                rep.check(rb_.seat == a.seat, f"e{e}: the rebuilt follower has A's seat")
                c = Validator("C")
                c.window, c.seat, c.copy, c.rules = rb_.window, rb_.seat, from_rebuild(rb_), a.rules
            if c is not None and e >= C_STARTS:
                rc, rvc = c.step(e)
                rep.check(rvc == rva and rc.weights == ra.weights,
                          f"e{e}: C (rebuilt at e{C_STARTS - 1}) publishes A's reveal and weights")

            if e == F:
                rep.check(not any(ep < F + 1 for ep in a.window.slices),
                          f"e{e}: at v3 activation the window holds no v2 epoch")
                rep.check("v2_state" in rva and rva["v2_state"].get("rules") == rules_for(e),
                          f"e{e}: the reveal publishes the rules it was built under")
                rep.check(bool(rva.get("head_odds")) and bool(rva.get("exploration_pool")),
                          f"e{e}: v3 publishes the pooled exploration and every head's odds")
            if e < F:
                rep.check(not rva.get("head_odds"), f"e{e}: v2 epochs publish no odds")
            if e == O:
                rep.check(any(ep < O + 1 for ep in a.window.pairs),
                          f"e{e}: the odds-seat block keeps the window's pooled pairs")

    first_odds_seat = [e for e in range(O, LAST + 1) if events[e][1] == H(6)]
    rep.check(bool(first_odds_seat),
              f"after the odds block the predicting head (UID 6) takes the seat "
              f"(first at e{first_odds_seat[0] if first_odds_seat else '-'})")
    rep.check(events[LAST][1] == H(6), f"e{LAST}: UID 6 still holds the seat on odds")
    print("\nseat timeline:")
    for e in range(LAST + 1):
        ev, ch = events[e]
        print(f"  e{e:<2} {('v2' if e < F else 'v3-odds' if e >= O else 'v3-nv'):8} {ev:12} "
              f"{(ch or '-')[:16]}")
    print(f"\n{len(rep.ok)} passed, {len(rep.fail)} failed")
    return 1 if rep.fail else 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
