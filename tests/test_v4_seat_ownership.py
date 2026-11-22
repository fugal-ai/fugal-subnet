"""Whose serving a v4 head's standing, pay and challenge evidence read, once
its first-commit record has lapsed (the copy rule keeps a retired hash for
STANDING_EPOCHS_V4 epochs) and the same bytes are committed again.

The world runs each epoch in the live v4 order (v2_epoch.run_v2_epoch):
copy_rule.observe with the v4 retention, copy_rule.classify on the hotkeys
that served, seat_v4.seat_serving, the challenge records from the seat's
view (v2_epoch._challenge_epoch is handed it), seat_v4.observe and
seat_v4.decide on it, late odds bounded by v2_epoch.late_epochs, and
payout.seat_weights. Three cases:

1. a holder whose own operator stops committing, the same bytes then
   committed and served by another coldkey: the holder stops standing after
   STANDING_EPOCHS_V4 epochs, and a head that beats the reference and
   challenges takes the open seat;
2. the holder's own coldkey commits the same bytes again after a lapse: it
   serves, stands and is paid again;
3. a successor whose predecessor's bytes are committed again by another
   coldkey, which challenges under that hash: those challenges are never
   read as the successor's.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from fugal_subnet import config, copy_rule, labels_v4, seat_v4
from fugal_subnet.attacks.seat_v4_world import (
    BPE,
    ENTRIES,
    SCALE,
    UNIT,
    VOTERS,
    Head,
    World,
    hash_of,
)
from fugal_subnet.champion import EpochPairs
from fugal_subnet.payout import seat_weights
from fugal_subnet.picks_v4 import PicksEpoch
from fugal_subnet.v2_epoch import late_epochs

SHARES = tuple(config.PAYOUT_SEAT_SHARES)


class OwnershipWorld(World):
    """`commit(hotkey, name, coldkey, block)` sets a hotkey's commitment (two
    hotkeys may commit the same name: the same bytes); `uncommit(hotkey)`.
    `epoch(e, served=hotkeys, answered=hotkeys, value={hotkey: points})`:
    `answered` are the hotkeys with a challenge this epoch, each item its
    value less that of the holder it pairs with."""

    def __init__(self, **kw):
        super().__init__(labels=600, questions=500, **kw)
        self.uid: dict[str, int] = {}
        self.weights: dict[int, dict[str, float]] = {}

    def head_def(self, name: str, **kw) -> Head:
        h = Head(name, f"ck-{name}", f"hk-{name}", 5, **kw)
        self.heads[name] = h
        return h

    def commit(self, hotkey: str, name: str, coldkey: str, block: int) -> None:
        self.commits[hotkey] = (name, coldkey, int(block))
        self.uid.setdefault(hotkey, 100 + len(self.uid))

    def uncommit(self, hotkey: str) -> None:
        self.commits.pop(hotkey, None)

    def epoch(self, e, *, served=(), answered=(), value=None, **_kw):
        value = value or {}
        n = self.labels
        y = (self.rng.random(n) < 0.5).astype(np.int32) * SCALE
        models = tuple(ENTRIES[i % 2] for i in range(n))
        pairs = labels_v4.label_pairs(
            EpochPairs(question_ids=tuple(f"q{e}-{i:05d}" for i in range(n)), models=models,
                       benchmarks=("b1",) * n, categories=("c1",) * n, y=y,
                       prompt_tokens=np.zeros(n, np.int32), completion_tokens=np.zeros(n, np.int32)),
            [(VOTERS[i % len(VOTERS)],) for i in range(n)], np.full(n, -1, np.int16))
        commits = [copy_rule.Commitment(hk, ck, hash_of(name), block)
                   for hk, (name, ck, block) in sorted(self.commits.items())]
        self.copy = copy_rule.observe(self.copy, e, e * BPE, commits,
                                      ghost_retire=config.STANDING_EPOCHS_V4)
        served_c = {}
        for hk in served:
            if hk in self.commits:
                name, ck, block = self.commits[hk]
                served_c[self.uid[hk]] = copy_rule.Commitment(hk, ck, hash_of(name), block)
        verdicts = copy_rule.classify(self.copy, served_c)
        serving = {served_c[u].head_hash: u for u, v in sorted(verdicts.items()) if v == "owner"}
        name_of = {hash_of(name): name for (name, _ck, _b) in self.commits.values()}
        hk_of = {u: hk for hk, u in self.uid.items()}
        seat_serving = seat_v4.seat_serving(self.state, serving, self.copy)
        rows = {h: self.heads[name_of[h]].digests() for h in seat_serving}
        observed = seat_v4.observe(self.state, epoch=e, serving=seat_serving, rows=rows,
                                   copy_state=self.copy, menu_ids=ENTRIES, blocks_per_epoch=BPE)
        pairs_w = {**self.window.pairs, e: pairs}
        odds = {hh: dict(per) for hh, per in self.window.odds.items()}
        for h in serving:
            odds.setdefault(h, {})[e] = labels_v4.effective(
                self.odds(self.heads[name_of[h]], pairs, e), pairs)
        win = dataclasses.replace(self.window, pairs=pairs_w, odds=odds)
        since = {}
        for h in serving:
            first = copy_rule.first_epoch(self.copy, h, BPE)
            since[h] = seat_v4.odds_since(observed, h, e if first is None else first)
        for h, eps in late_epochs(win, {h: None for h in serving}, since,
                                  config.LATE_ODDS_SLOTS_V4).items():
            for k in eps:
                odds.setdefault(h, {})[k] = labels_v4.effective(
                    self.odds(self.heads[name_of[h]], pairs_w[k], k), pairs_w[k])
        holder = seat_v4.challenge_holder(self.state, e)
        base = next((value.get(hk, 0) for hk, (nm, _c, _b) in self.commits.items()
                     if holder is not None and hash_of(nm) == holder), 0)
        vals = {h: (int((value.get(hk_of[u], 0) - base) * UNIT),) * self.questions
                for h, u in seat_serving.items() if hk_of[u] in answered}
        picks = {**self.window.picks,
                 e: PicksEpoch(e, ("c1",) * self.questions, vals, False, holder)}
        self.window = dataclasses.replace(self.window, pairs=pairs_w, odds=odds, picks=picks)
        d = seat_v4.decide(observed, epoch=e, window=self.window, serving=seat_serving,
                           floor_block=self.floor_block, salt=self.salt)
        uids, ws = seat_weights(d.champion, d.state.history, seat_serving, set(d.payable),
                                shares=SHARES, paid_fraction=1.0, burn_uid=0)
        self.weights[e] = {hk_of.get(u, "burn"): round(w, 6) for u, w in zip(uids, ws) if w > 1e-12}
        self.state, self.decisions[e], self.seat_serving = d.state, d, seat_serving
        return d


def _held_by_a():
    w = OwnershipWorld()
    w.head_def("A", skill=0.6)
    w.head_def("C", skill=0.4)
    w.commit("hk-A", "A", "ck-A", 5)
    w.commit("hk-C", "C", "ck-C", 6)
    for e in range(1, 11):
        w.epoch(e, served=["hk-A"], answered=["hk-A"])
    assert w.holder() == "A"
    return w


def test_another_coldkeys_commit_of_an_absent_holders_bytes_does_not_keep_it_standing():
    w = _held_by_a()
    w.uncommit("hk-A")
    for e in range(11, 40):
        if e == 16:
            w.commit("hk-T", "A", "ck-T", e * BPE + 1)
        served = (["hk-C"] if e >= 20 else []) + (["hk-T"] if e >= 16 else [])
        d = w.epoch(e, served=served, answered=served)
        if e >= 11 + config.STANDING_EPOCHS_V4:
            assert not d.evidence["holder"] or not d.evidence["holder"]["stands"] or \
                w.holder() != "A", e
    assert w.holder() == "C"
    burnt = [e for e in range(16, 40) if w.weights[e] == {"burn": 1.0}]
    assert len(burnt) <= 4, burnt


def test_the_holders_own_coldkey_committing_its_bytes_again_is_served_and_paid():
    w = _held_by_a()
    w.uncommit("hk-A")
    for e in range(11, 30):
        if e == 15:
            w.commit("hk-A", "A", "ck-A", e * BPE + 1)
        w.epoch(e, served=["hk-A"] if e >= 15 else [])
    assert w.holder() == "A"
    assert hash_of("A") in w.seat_serving
    assert all(w.weights[e].get("hk-A", 0) > 0 for e in range(16, 30))


def test_another_coldkeys_challenges_under_a_predecessors_hash_are_not_its_successors():
    w = _held_by_a()
    w.head_def("A2", skill=0.6)                 # A's rows byte for byte: a successor
    w.commit("hk-A", "A2", "ck-A", 11 * BPE + 1)
    for e in range(11, 30):
        if e == 16:
            w.commit("hk-T", "A", "ck-T", e * BPE + 1)
        served = ["hk-A"] + (["hk-T"] if e >= 16 else [])
        w.epoch(e, served=served, answered=["hk-T"] if e >= 16 else [])
    assert w.holder() == "A2"
    records = seat_v4.lineage_records(w.window.picks, w.state)
    a2 = hash_of("A2")
    assert not any(a2 in records[e].values for e in range(16, 30))
    # The successor still stands and is paid on its own exploration.
    assert all(w.weights[e].get("hk-A", 0) > 0 for e in range(12, 30))
