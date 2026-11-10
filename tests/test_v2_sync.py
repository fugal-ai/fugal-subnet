"""Verify-and-resync (fugal_subnet/v2_sync.py): a validator that missed epochs
re-anchors on a reference validator's published state and then agrees with it
exactly (I1), at the unavoidable lag of the reveal hold."""
import json
import shutil
from types import SimpleNamespace

import pytest

from fugal_subnet import copy_rule
from fugal_subnet.evidence import Evidence, accumulate_epoch_v2, apply_miss_v2
from fugal_subnet.rewards import cap_weight_change
from fugal_subnet.v2_sync import carry_burn_in, check, consensus_view, load_reveals, resync
from tests.test_v2_epoch import (
    CATEGORY,
    MODELS,
    PRICES,
    H,
    Q,
    always,
    oracle,
    proof,
    questions,
    run,
)

BURN_IN = 2 * Q                                   # two epochs, so burn-in matters here


def noisy(policy, p, seed):
    """`policy`, except a fixed share p of each epoch's questions go to m/cheap."""
    import numpy as np

    def f(d):
        flip = np.random.default_rng(int(d[0] * 1e9) + seed).random(len(d)) < p
        return ["m/cheap" if x else m for x, m in zip(flip, policy(d))]
    return f


class Sim:
    """One validator, as neurons/validator.py runs a v2 epoch: records first
    (update_scores' v2 burn-in rule), then run_v2_epoch, then the weight cap,
    then the reveal on disk. `sync_from` is the reference reveal directory."""

    def __init__(self, root, name):
        self.dir = root / name
        self.dir.mkdir()
        self.window = self.seat = self.copy = None
        self.records = {}
        self.prev = ([], [])
        self.last = -1

    def _records(self, pols, heads):
        for uid in list(self.records):
            if uid not in pols:
                self.records[uid].evidence = apply_miss_v2(self.records[uid].evidence, 200, BURN_IN)
        for uid in pols:
            rec = self.records.setdefault(uid, SimpleNamespace(evidence=None))
            rec.evidence = accumulate_epoch_v2(rec.evidence, heads[uid], n_correct=0,
                                               n_total=Q, cost=0.0, ref_cost=0.0, half_life=200)

    def sync(self, ref_dir, index):
        ref, own = load_reveals(ref_dir), load_reveals(self.dir)
        chk = check(ref, own, index)
        if chk.status not in ("differs", "own_missing"):
            return chk.status
        rs = resync(ref, own, chk.k, index, window_size=48, prices=PRICES,
                    bench_category=CATEGORY)
        self.window, self.seat = rs.rebuilt.window, rs.rebuilt.seat
        self.copy = copy_rule.from_rebuild(rs.rebuilt) or copy_rule.CopyState()
        if rs.burn_in is not None:
            for uid, (h, c) in rs.burn_in.items():
                rec = self.records.setdefault(uid, SimpleNamespace(evidence=None))
                if rec.evidence is None or rec.evidence.weights_hash != h:
                    rec.evidence = Evidence(weights_hash=h)
                rec.evidence.burn_in_n = float(c)
        return "resynced" if rs.complete else "resynced_partial"

    def epoch(self, e, pols, *, observe=True, sync_from=None):
        index = e + 1
        if not observe:
            return None
        status = self.sync(sync_from, index) if sync_from is not None else None
        heads = {u: H(u) for u in pols}
        self._records(pols, heads)
        r = run(e, pols, window=self.window, seat=self.seat, copy=self.copy,
                records=self.records, burn_in=BURN_IN)
        uids, weights = r.uids, r.weights
        if self.prev[0]:
            uids, weights = cap_weight_change(uids, weights, *self.prev)
        self.prev = (list(uids), list(weights))
        self.window, self.seat, self.copy = r.window, r.decision.state, r.copy_state
        model_index = {m: i for i, m in enumerate(MODELS)}
        decisions = {str(u): [model_index[x.routed_model] for x in
                              sorted(proof(e, pols[u]).scored_results, key=lambda x: x.question_id)]
                     for u in pols}
        reveal = {"epoch_id": f"e{index:08d}", "protocol_version": 2,
                  "questions": questions(e), "models": MODELS, "routing_decisions": decisions,
                  "head_hashes": {str(u): heads[u] for u in pols},
                  "weights": {str(u): float(x) for u, x in zip(uids, weights)}, **r.reveal,
                  "v2_sync": {"unverified": getattr(self, "gap", None) is not None}}
        d = self.dir / reveal["epoch_id"]
        d.mkdir()
        (d / "reveal.json").write_text(json.dumps(reveal))
        return status


def schedule(seed):
    """Heads that come, go, miss and get better, so the seat moves."""
    def pols(e):
        p = {5: always("m/strong"), 7: always("m/cheap")}
        if e >= 3 and (e + seed) % 9 != 0:
            p[6] = noisy(oracle, 0.4, seed)          # a good head that misses now and then
        if e >= 14 + seed % 5:
            p[8] = oracle                            # a better one arrives
        if 22 <= e < 26 + seed % 3:
            p.pop(6, None)                           # 6 away for a while (a ghost)
        return p
    return pols


def publish(src, dst, upto_index):
    """Copy the reference's reveals up to `upto_index` (the hold)."""
    for path in sorted(src.glob("*/reveal.json")):
        idx = int(path.parent.name[1:])
        target = dst / path.parent.name
        if idx <= upto_index and not target.exists():
            shutil.copytree(path.parent, target)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("hold", [1, 2])
def test_a_validator_that_missed_epochs_agrees_again_after_the_hold(tmp_path, seed, hold):
    pols = schedule(seed)
    a, b = Sim(tmp_path, "a"), Sim(tmp_path, "b")
    published = tmp_path / "published"
    published.mkdir()
    down = {10, 11, 20 + seed % 4}                   # B is down, or saw no proofs
    agree = []
    for e in range(45):
        index = e + 1
        publish(a.dir, published, index - 1 - hold)
        a.epoch(e, pols(e))
        b.epoch(e, pols(e), observe=e not in down, sync_from=published)
        rv_a = json.loads((a.dir / f"e{index:08d}" / "reveal.json").read_text())
        pb = b.dir / f"e{index:08d}" / "reveal.json"
        agree.append(pb.exists() and consensus_view(json.loads(pb.read_text())) ==
                     consensus_view(rv_a))
    last_gap = max(down)
    first_good = next(i for i in range(last_gap + 1, 45) if all(agree[i:]))
    # Exact agreement from at most a few epochs after the hold, and for good.
    assert first_good <= last_gap + hold + 4, (first_good, agree)
    # Before any gap the two never differed.
    assert all(agree[:min(down)])


def test_the_check_names_what_it_compared():
    rv = {"v2_state": {"seat": {"champion": "a"}}, "weights": {"1": 1.0}}
    other = {"v2_state": {"seat": {"champion": "b"}}, "weights": {"1": 1.0}}
    ref = {3: ("p3", rv), 5: ("p5", rv)}
    assert check({}, {}, 9).status == "no_reference"
    assert check(ref, {5: ("o5", rv)}, 9) == check(ref, {5: ("o5", dict(rv, scores={"x": 1}))}, 9)
    assert check(ref, {5: ("o5", rv)}, 9).status == "in_sync"
    assert check(ref, {5: ("o5", other)}, 9).status == "differs"
    assert check(ref, {3: ("o3", rv)}, 9) .status == "own_missing"
    assert check(ref, {}, 5).k == 3                       # only reference epochs before `before`


def test_burn_in_is_carried_by_the_update_scores_rule():
    snap = {"h5": 600.0, "h6": 60.0}
    serving = {"h5": 5, "h6": 6}

    def rv(heads, n=60):
        return ("p", {"head_hashes": {str(u): h for u, h in heads.items()},
                      "questions": [{}] * n})
    own = {11: rv({5: "h5", 6: "h6"}), 12: rv({5: "h5", 7: "h7"}), 13: rv({5: "h5x", 7: "h7"})}
    out = carry_burn_in(snap, serving, own, 10, 14)
    assert out == {5: ("h5x", 60.0), 6: ("h6", 0.0), 7: ("h7", 120.0)}
    assert carry_burn_in(snap, serving, {11: own[11]}, 10, 14) is None   # 12, 13 unobserved


# --- the validator's step (neurons/validator._v2_sync_step) ----------------------


def _step(**kw):
    from neurons.validator import _v2_sync_step

    base = dict(epoch_index=13, last_epoch_index=12, first_v2_epoch=5, reference_dir=None,
                scoring_state=SimpleNamespace(records={}), window=None, seat=None,
                copy_state=None, window_degraded=False, gap_until=None, anchor_at=None,
                prices=PRICES,
                bench_category=CATEGORY, window_epochs=48)
    base.update(kw)
    return _v2_sync_step(**base)


def test_a_missed_v2_epoch_opens_a_gap_and_a_pre_flip_one_does_not(tmp_path):
    ref = str(tmp_path)                                    # a follower: it has a reference
    out = _step(last_epoch_index=10, reference_dir=ref)    # 11 and 12 not observed
    assert out.changed and out.state[5] == 12
    assert not _step(reference_dir=ref).changed            # consecutive: nothing to do
    assert not _step(epoch_index=6, last_epoch_index=2, first_v2_epoch=6,
                     reference_dir=ref).changed            # v1 epochs


class ValidatorSim(Sim):
    """Sim, with the validator's own step doing the sync."""

    def sync(self, ref_dir, index):
        out = _step(epoch_index=index, last_epoch_index=self.last, first_v2_epoch=1,
                    reference_dir=str(ref_dir) if ref_dir else None,
                    scoring_state=SimpleNamespace(records=self.records),
                    window=self.window, seat=self.seat, copy_state=self.copy,
                    gap_until=self.gap, anchor_at=self.anchor, epoch_dir=str(self.dir))
        if out.changed:
            scoring, self.window, self.seat, self.copy, _, self.gap, self.anchor = out.state
            self.records = scoring.records
        self.anchors += out.changed and out.state[6] == index
        return out

    def epoch(self, e, pols, *, observe=True, sync_from=None):
        if not hasattr(self, "gap"):
            self.gap, self.anchor, self.anchors = None, None, 0
        if not observe:
            return None
        if sync_from is None:
            self.sync(None, e + 1)                 # the validator runs the step every epoch
        res = super().epoch(e, pols, observe=True, sync_from=sync_from)
        self.last = e + 1
        return res


@pytest.mark.parametrize("seed", range(4))
def test_the_validator_step_brings_a_validator_back_into_agreement(tmp_path, seed):
    pols = schedule(seed)
    a, b = Sim(tmp_path, "a"), ValidatorSim(tmp_path, "b")
    published = tmp_path / "published"
    published.mkdir()
    down = {10, 11, 20 + seed % 4}
    agree, gaps = [], []
    for e in range(45):
        index = e + 1
        publish(a.dir, published, index - 2)
        a.epoch(e, pols(e))
        b.epoch(e, pols(e), observe=e not in down, sync_from=published)
        gaps.append(getattr(b, "gap", None))
        rv_a = json.loads((a.dir / f"e{index:08d}" / "reveal.json").read_text())
        pb = b.dir / f"e{index:08d}" / "reveal.json"
        agree.append(pb.exists() and consensus_view(json.loads(pb.read_text())) ==
                     consensus_view(rv_a))
    last_gap = max(down)
    first_good = next(i for i in range(last_gap + 1, 45) if all(agree[i:]))
    assert first_good <= last_gap + 4, agree
    assert gaps[-1] is None                                # the gap closed
    assert any(g is not None for g in gaps)                # and it was opened


def test_a_sync_that_cannot_read_the_reference_keeps_the_state(tmp_path):
    bad = tmp_path / "ref" / "e00000012"
    bad.mkdir(parents=True)
    (bad / "reveal.json").write_text('{"protocol_version": 2, "epoch_id": "e00000012"}')
    own = tmp_path / "own"
    own.mkdir()
    out = _step(reference_dir=str(tmp_path / "ref"), epoch_dir=str(own), last_epoch_index=10)
    assert out.state[5] == 12                              # the gap stands; nothing raised


def test_the_check_sees_weights_too():
    rv = {"v2_state": {"seat": {"champion": "a"}}, "weights": {"1": 1.0}}
    capped = {"v2_state": {"seat": {"champion": "a"}}, "weights": {"1": 0.7, "0": 0.3}}
    assert check({5: ("p", rv)}, {5: ("o", capped)}, 9).status == "differs"


def test_the_resync_window_keeps_this_validators_own_epochs_after_k(tmp_path):
    pols = schedule(0)
    a, b = Sim(tmp_path, "a"), Sim(tmp_path, "b")
    for e in range(12):
        a.epoch(e, pols(e))
        b.epoch(e, pols(e))
    ref, own = load_reveals(a.dir), load_reveals(b.dir)
    rs = resync(ref, own, 8, 13, window_size=48, prices=PRICES, bench_category=CATEGORY)
    assert sorted(rs.rebuilt.window.slices) == list(range(1, 13))       # through before-1
    assert rs.rebuilt.snapshot_epoch_id == "e00000008" and rs.complete


def test_an_in_sync_check_at_or_after_the_gap_closes_it(tmp_path):
    pols = schedule(1)
    a = Sim(tmp_path, "a")
    for e in range(12):
        a.epoch(e, pols(e))
    # This validator's own reveals are the reference's: in sync at k = 12.
    closed = _step(reference_dir=str(a.dir), epoch_dir=str(a.dir), gap_until=11,
                   epoch_index=13, last_epoch_index=12)
    assert closed.changed and closed.state[5] is None
    still = _step(reference_dir=str(a.dir), epoch_dir=str(a.dir), gap_until=12,
                  epoch_index=13, last_epoch_index=12)
    assert still.state[5] is None                         # k = 12 >= 12: closed too
    early = _step(reference_dir=str(a.dir), epoch_dir=str(a.dir), gap_until=14,
                  epoch_index=13, last_epoch_index=12)
    assert early.state[5] == 14 and not early.changed     # the gap is ahead of k


def test_the_gap_is_durable_and_a_local_outage_opens_one():
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()
    assert src.count('"gap_until": v2_gap_until, "anchor_at": v2_anchor_at}') == 3
    assert 'v2_anchor_at = v2_raw.get("anchor_at")' in src
    assert 'v2_gap_until = v2_raw.get("gap_until")' in src
    skip = src[src.index('logger.warning("No valid proofs received, skipping epoch")'):]
    skip = skip[:skip.index("continue")]
    assert "if protocol_version >= 2:" in skip and "v2_gap_until = epoch_index" in skip


def _run_pair(tmp_path, seed, *, ref_down=(), b_down=(), mutual=False, epochs=45, hold=1):
    """Truth T (never misses, never syncs); A (the reference, misses ref_down);
    B (syncs from A's published reveals, misses b_down). With `mutual`, A syncs
    from B too. Returns the agreement strings of A and B against T."""
    pols = schedule(seed)
    t, a, b = Sim(tmp_path, "t"), ValidatorSim(tmp_path, "a"), ValidatorSim(tmp_path, "b")
    pub_a, pub_b = tmp_path / "pub_a", tmp_path / "pub_b"
    pub_a.mkdir()
    pub_b.mkdir()
    agree = {"a": [], "b": []}
    for e in range(epochs):
        index = e + 1
        publish(a.dir, pub_a, index - 1 - hold)
        publish(b.dir, pub_b, index - 1 - hold)
        t.epoch(e, pols(e))
        a.epoch(e, pols(e), observe=e not in ref_down, sync_from=pub_b if mutual else None)
        b.epoch(e, pols(e), observe=e not in b_down, sync_from=pub_a)
        rv_t = consensus_view(json.loads((t.dir / f"e{index:08d}" / "reveal.json").read_text()))
        for name, v in (("a", a), ("b", b)):
            p = v.dir / f"e{index:08d}" / "reveal.json"
            agree[name].append(p.exists() and consensus_view(json.loads(p.read_text())) == rv_t)
    return agree, a, b


@pytest.mark.parametrize("seed", range(3))
def test_a_follower_converges_on_a_root_that_missed_epochs(tmp_path, seed):
    """REVIEW 2026-11-09 made a follower with no gap only alarm, so it would
    never copy a root that missed epochs. FABLE REVIEW (item 5) showed the
    price: any lost proof split a validator from the root for good. In a star
    the root is the arbiter and agreement is what Yuma pays: a follower that
    differs from the root on DRIFT_CHECKS consecutive verified root epochs
    re-anchors on it. A wrong root is repaired once, by its operator, and the
    followers follow it back. Not on a single mismatch."""
    from fugal_subnet.v2_sync import DRIFT_CHECKS

    agree, a, b = _run_pair(tmp_path, seed, ref_down={10, 11}, epochs=60)
    assert b.anchors >= 1 and b.gap is None
    same = []
    for i in range(60):
        pa = a.dir / f"e{i + 1:08d}" / "reveal.json"
        pb = b.dir / f"e{i + 1:08d}" / "reveal.json"
        same.append(pa.exists() and pb.exists() and consensus_view(json.loads(pa.read_text()))
                    == consensus_view(json.loads(pb.read_text())))
    first = next(i for i in range(12, 60) if all(same[i:]))
    assert first <= 12 + DRIFT_CHECKS + 4, same


def _lost_proof_pair(tmp_path, seed, drop_epoch, drop_uid, *, epochs=90, hold=1):
    """A (root, sees everything) and B (follower) that does not receive one
    miner's proof in one epoch; the chain's commitments are the same for both,
    as they are read from chain, not from proofs."""
    import tests.test_v2_sync as me
    from tests.test_v2_epoch import C, H
    from tests.test_v2_epoch import run as _run

    full = {}

    def run_with_chain_commits(e, pols, **kw):
        kw["commits"] = [C(u, H(u), u) for u in sorted(full[e])]
        return _run(e, pols, **kw)

    pols = schedule(seed)
    a, b = Sim(tmp_path, "a"), ValidatorSim(tmp_path, "b")
    pub = tmp_path / "pub"
    pub.mkdir()
    same = []
    orig, me.run = me.run, run_with_chain_commits
    try:
        for e in range(epochs):
            index = e + 1
            publish(a.dir, pub, index - 1 - hold)
            pa = pols(e)
            full[e] = pa
            pb = {u: f for u, f in pa.items() if not (e == drop_epoch and u == drop_uid)}
            a.epoch(e, pa)
            b.epoch(e, pb, sync_from=pub)
            ra = json.loads((a.dir / f"e{index:08d}" / "reveal.json").read_text())
            rb = json.loads((b.dir / f"e{index:08d}" / "reveal.json").read_text())
            same.append(consensus_view(ra) == consensus_view(rb))
    finally:
        me.run = orig
    return same, b


@pytest.mark.parametrize("seed,drop_epoch,drop_uid",
                         [(0, 30, 5), (1, 30, 8), (2, 16, 8), (3, 40, 8), (4, 30, 5), (5, 30, 6)])
@pytest.mark.parametrize("hold", [1, 3])
def test_one_lost_proof_does_not_split_a_validator_for_good(tmp_path, seed, drop_epoch,
                                                           drop_uid, hold):
    """FABLE REVIEW, item 5 (2026-11-09): a validator that did not receive one
    miner's proof in one epoch had no gap (it observed the epoch), so it only
    alarmed; its burn-in counters, and sometimes its seat, differed from its
    peers' for good (28 of 30 cases). It now agrees again within a bounded
    number of epochs."""
    from fugal_subnet.v2_sync import DRIFT_CHECKS

    same, b = _lost_proof_pair(tmp_path, seed, drop_epoch, drop_uid, hold=hold)
    first = next((i for i in range(drop_epoch, len(same)) if all(same[i:])), None)
    assert first is not None, "".join("." if x else "X" for x in same)
    assert first <= drop_epoch + 2 * hold + DRIFT_CHECKS + 3, "".join("." if x else "X" for x in same)
    assert b.gap is None


@pytest.mark.parametrize("seed", range(3))
def test_two_validators_that_reference_each_other_both_converge(tmp_path, seed):
    agree, a, b = _run_pair(tmp_path, seed, ref_down={10, 11}, mutual=True)
    assert all(agree["b"])                                  # the healthy one is untouched
    first = next(i for i in range(12, 45) if all(agree["a"][i:]))
    assert first <= 12 + 4 and a.gap is None


@pytest.mark.parametrize("hold", [1, 2, 4, 6])
def test_a_re_anchor_is_verified_before_the_gap_closes_and_does_not_repeat(tmp_path, hold):
    """REVIEW 2026-11-09: the check compared against this validator's own
    reveals from before its re-anchor, and re-anchored again and again; each
    re-anchor moved events. Only a reference epoch at or after the re-anchor
    verifies it, and the gap closes only on a verified match."""
    agree, a, b = _run_pair(tmp_path, 3, b_down={10, 11}, hold=hold, epochs=60)
    first = next(i for i in range(12, 60) if all(agree["b"][i:]))
    # The minimum possible: exact again `hold` epochs after the gap, after ONE
    # re-anchor made once the reference has published past the gap.
    assert first == 12 + hold, agree["b"]
    assert b.gap is None and b.anchors == 1


@pytest.mark.parametrize("bad", ['"x"', '{"seat": [], "owners": [], "burn_in": []}',
                                 '{"seat": {"champion": 5}}', '[1, 2]'])
def test_a_malformed_reference_never_stops_the_epoch(tmp_path, bad):
    """REVIEW 2026-11-09 (I6): a malformed reference reveal raised out of the
    step and lost every epoch that read it."""
    ref = tmp_path / "ref" / "e00000012"
    ref.mkdir(parents=True)
    (ref / "reveal.json").write_text('{"protocol_version": 2, "epoch_id": "e00000012", '
                                     '"weights": {}, "v2_state": ' + bad + '}')
    own = tmp_path / "own"
    own.mkdir()
    out = _step(reference_dir=str(tmp_path / "ref"), epoch_dir=str(own), last_epoch_index=10)
    assert out.state[5] == 12                               # gap kept, nothing raised


def test_a_root_validator_never_marks_itself_unverified():
    """No reference: this validator is the root the others reference. It has
    nothing to verify against, and marking itself unverified would only stop
    its followers from verifying against it."""
    out = _step(epoch_index=40, last_epoch_index=30, gap_until=None)
    assert out.state[5] is None
    assert _step(epoch_index=40, last_epoch_index=39, gap_until=12).state[5] is None


def test_a_gap_is_bounded_by_the_window_even_with_a_reference(tmp_path):
    still = _step(epoch_index=40, last_epoch_index=39, gap_until=12, reference_dir=str(tmp_path))
    assert still.state[5] == 12
    closed = _step(epoch_index=61, last_epoch_index=60, gap_until=12,
                   reference_dir=str(tmp_path))
    assert closed.changed and closed.state[5] is None


# --- the reviewer's harder worlds (review 2, 2026-11-09) ---------------------------


def churn(period, seed=0, nheads=3):
    """nheads good heads; head (e // period) % nheads routes everything to
    m/cheap for `period` epochs, so the seat changes hands every few epochs."""
    def pols(e):
        bad = 5 + (e // period) % nheads
        return {5 + j: always("m/cheap") if (e >= 2 and 5 + j == bad)
                else noisy(oracle, 0.05, seed + j) for j in range(nheads)}
    return pols


class Zero(ValidatorSim):
    """ValidatorSim with the validator's zero-proof skip: the step runs, no
    reveal is written, the seat does not advance, and a gap opens."""

    def epoch(self, e, pols, *, observe=True, zero=False, sync_from=None):
        if not hasattr(self, "gap"):
            self.gap, self.anchor, self.anchors = None, None, 0
        if observe and zero:
            self.sync(sync_from, e + 1)
            self.gap, self.last = e + 1, e + 1
            # The validator sets the capped all-burn vector (_burn_epoch_weights),
            # so the next epoch's cap starts from it.
            self.prev = (cap_weight_change([0], [1.0], *self.prev) if self.prev[0]
                         else ([0], [1.0]))
            return None
        return super().epoch(e, pols, observe=observe, sync_from=sync_from)


class Truth(Sim):
    def epoch(self, e, pols, *, zero=False, **kw):
        if zero:                                   # the burn vector, as the validator sets it
            self.prev = (cap_weight_change([0], [1.0], *self.prev) if self.prev[0]
                         else ([0], [1.0]))
            return None
        return super().epoch(e, pols, **kw)


def _star(tmp_path, pols, n, *, down=None, zero=(), hold=1, mutual=False):
    """Truth T; root A (no reference, or B's with `mutual`); B and C follow A
    and B respectively. Returns per-validator agreement strings vs T."""
    down = down or {}
    t = Truth(tmp_path, "t")
    vs = {x: Zero(tmp_path, x) for x in "abc"}
    pubs = {x: tmp_path / f"pub_{x}" for x in "abc"}
    for d in pubs.values():
        d.mkdir()
    refs = {"a": pubs["b"] if mutual else None, "b": pubs["a"], "c": pubs["b"]}
    marks = {x: [] for x in "abc"}
    for e in range(n):
        index = e + 1
        for x in "abc":
            publish(vs[x].dir, pubs[x], index - 1 - hold)
        t.epoch(e, pols(e), zero=e in zero)
        for x, v in vs.items():
            v.epoch(e, pols(e), observe=e not in down.get(x, ()), zero=e in zero,
                    sync_from=refs[x])
        pt = t.dir / f"e{index:08d}" / "reveal.json"
        tv = consensus_view(json.loads(pt.read_text())) if pt.exists() else None
        for x, v in vs.items():
            p = v.dir / f"e{index:08d}" / "reveal.json"
            vv = consensus_view(json.loads(p.read_text())) if p.exists() else None
            marks[x].append(vv == tv)
    return marks, vs


@pytest.mark.parametrize("hold", [1, 2, 4])
def test_a_validator_that_missed_epochs_converges_while_the_seat_keeps_changing(tmp_path, hold):
    """REVIEW 2 (2026-11-09): with takes every few epochs, a gap that blocked
    this validator's takes guaranteed it diverged (its peers took, it did
    not) and it never agreed again. A gap no longer blocks anything."""
    pols = churn(6)
    marks, vs = _star(tmp_path, pols, 100, down={"b": {60, 61}}, hold=hold)
    assert all(marks["a"]) and all(marks["c"][:60])         # the healthy are untouched
    first = next(i for i in range(62, 100) if all(marks["b"][i:]))
    assert first <= 62 + 2 * hold + 2, marks["b"]
    assert vs["b"].gap is None and vs["b"].anchors <= 2
    assert all(marks["c"][first:])


@pytest.mark.parametrize("hold", [1, 2])
def test_an_epoch_no_validator_got_proofs_in_changes_nothing(tmp_path, hold):
    """REVIEW 2: a network-wide zero-proof epoch opened a gap on every
    validator, delayed takes everywhere, and left the root unverified for a
    window. Now nobody diverges, and every gap closes."""
    marks, vs = _star(tmp_path, churn(6), 90, zero={60}, hold=hold)
    for x in "abc":
        assert all(m for i, m in enumerate(marks[x]) if i != 60), (x, marks[x])
        assert vs[x].gap is None


def test_two_validators_with_gaps_never_confirm_each_others_wrong_state(tmp_path):
    """REVIEW 2: two validators referencing each other, both with gaps,
    verified against each other and were both wrong for good. A reveal
    published while unverified does not count; they wait (bounded by the
    window) rather than confirm each other. The supported shape is a star."""
    marks, vs = _star(tmp_path, churn(6), 90, down={"a": {58, 59}, "b": {59, 60}},
                      mutual=True)
    for x in "ab":
        assert vs[x].gap is not None or not all(marks[x][62:])   # never a false "verified"
        assert vs[x].anchors == 0


def test_a_startup_rebuild_behind_the_present_opens_a_gap():
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()
    block = src[src.index("behind = resume_epoch - 1"):]
    block = block[:block.index("else:")]
    assert "v2_gap_until = resume_epoch - 1" in block and "v2_window_degraded = False" in block
    assert "allow_take=not v2_window_degraded," in src          # a gap blocks nothing
    assert '"v2_sync": {"unverified": v2_gap_until is not None}' in src



def _startup_rebuild(v, src, resume_index):
    """What neurons/validator.py does at start with no v2 state and
    FUGAL_FRAME_BOOTSTRAP set: rebuild from `src`, fresh records carrying
    only the burn-in counters, no previous weights, and a gap if the reveals
    end before the epoch it resumes at."""
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals

    rb = rebuild_v2_from_reveals(str(src), window_size=48, prices=PRICES, bench_category=CATEGORY)
    v.window, v.seat = rb.window, rb.seat
    v.copy = copy_rule.from_rebuild(rb) or copy_rule.CopyState()
    v.records = {uid: SimpleNamespace(evidence=Evidence(weights_hash=h, burn_in_n=rb.burn_in[h]))
                 for h, uid in sorted(rb.serving.items()) if h in rb.burn_in}
    v.prev, v.last = ([], []), -1
    behind = resume_index - 1 - int(rb.newest_epoch_id[1:])
    if behind > 0:
        v.gap = resume_index - 1
    return behind


def _agrees(v, t, index):
    p, q = v.dir / f"e{index:08d}" / "reveal.json", t.dir / f"e{index:08d}" / "reveal.json"
    return p.exists() and consensus_view(json.loads(p.read_text())) == \
        consensus_view(json.loads(q.read_text()))


@pytest.mark.parametrize("pols", [schedule(1), churn(12), churn(6)],
                         ids=["schedule", "churn12", "churn6"])
@pytest.mark.parametrize("hold", [1, 2])
def test_a_validator_started_from_a_rebuild_verifies(tmp_path, pols, hold):
    """FINAL REVIEW, finding 1 (2026-11-09): the weight cap keeps every uid a
    validator's previous weights held, at 0.0; a validator started from a
    rebuild has no previous weights, so it lacked those 0.0 entries, every
    check said `differs`, and it re-anchored every few epochs (14-22 times)
    until its gap timed out. Zero weights never reach the chain and are no
    longer compared: one re-anchor, then exact agreement."""
    restart = 66
    t, a, b = Truth(tmp_path, "t"), Zero(tmp_path, "a"), Zero(tmp_path, "b")
    pa = tmp_path / "pa"
    pa.mkdir()
    marks = []
    for e in range(110):
        index = e + 1
        publish(a.dir, pa, index - 1 - hold)
        if e == restart:
            b.gap = None
            assert _startup_rebuild(b, pa, restart + 1) > 0
        t.epoch(e, pols(e))
        a.epoch(e, pols(e))
        b.epoch(e, pols(e), observe=e not in range(restart - 3, restart), sync_from=pa)
        marks.append(_agrees(b, t, index))
    assert b.anchors == 1 and b.gap is None
    assert all(marks[restart + hold + 1:]), marks[restart:]


def _view(v, index):
    p = v.dir / f"e{index:08d}" / "reveal.json"
    return consensus_view(json.loads(p.read_text())) if p.exists() else None


@pytest.mark.parametrize("pols,busy", [(schedule(1), False), (churn(12), False), (churn(6), True)],
                         ids=["schedule", "churn12", "churn6"])
@pytest.mark.parametrize("hold", [1, 2, 4])
def test_after_a_root_misses_epochs_every_validator_agrees_with_it(tmp_path, pols, busy, hold):
    """FABLE REVIEW, item 5: followers converge on the root (DRIFT_CHECKS), so
    a root that missed epochs is followed, and its missed epochs count as
    missed for the whole network, instead of splitting the root from its
    followers. (FINAL REVIEW, finding 3's recovery from a healthy follower is
    then unnecessary: no follower stays apart.)"""
    from fugal_subnet.v2_sync import DRIFT_CHECKS

    a, b, c = Zero(tmp_path, "a"), Zero(tmp_path, "b"), Zero(tmp_path, "c")
    pa = tmp_path / "pa"
    pa.mkdir()
    for e in range(110):
        publish(a.dir, pa, e - hold)
        a.epoch(e, pols(e), observe=e not in (60, 61))
        b.epoch(e, pols(e), sync_from=pa)
        c.epoch(e, pols(e), sync_from=pa)
    agree = [_view(a, i) == _view(b, i) == _view(c, i) for i in range(1, 111)]
    first = next((i for i in range(62, 110) if all(agree[i:])), None)
    row = "".join("." if x else "X" for x in agree)
    assert first is not None, row
    # A seat change inside the re-anchor span is applied late, so with the
    # seat changing every 6 epochs it takes longer (INVARIANTS: convergence
    # needs a stretch of about twice the hold with no seat change).
    bound = 8 * hold + DRIFT_CHECKS if busy else 2 * hold + DRIFT_CHECKS + 4
    assert first <= 62 + bound, row
    assert b.gap is None and c.gap is None and b.anchors >= 1


@pytest.mark.parametrize("pols", [schedule(1), churn(6)], ids=["schedule", "churn6"])
@pytest.mark.parametrize("hold", [1, 3])
def test_a_root_that_lost_its_state_rebuilds_from_a_follower(tmp_path, pols, hold):
    """VALIDATOR_GUIDE, "Recovering the root": a root whose state files are
    gone restarts with the reference set to a follower until its gap closes,
    then unsets it. It agrees with its followers again; they never move."""
    restart = 70
    a, b, c = Zero(tmp_path, "a"), Zero(tmp_path, "b"), Zero(tmp_path, "c")
    pa, pb = tmp_path / "pa", tmp_path / "pb"
    pa.mkdir()
    pb.mkdir()
    a_ref = None
    for e in range(110):
        index = e + 1
        publish(a.dir, pa, index - 1 - hold)
        publish(b.dir, pb, index - 1 - hold)
        if e == restart:
            a.gap = None
            _startup_rebuild(a, pb, restart + 1)
            a_ref = pb
        a.epoch(e, pols(e), observe=e not in range(restart - 4, restart), sync_from=a_ref)
        if a_ref is not None and e > restart and a.gap is None:
            a_ref = None
        b.epoch(e, pols(e), sync_from=pa)
        c.epoch(e, pols(e), sync_from=pa)
    assert a_ref is None and a.gap is None
    assert all(_view(a, i) == _view(b, i) == _view(c, i) for i in range(70 + 6 * hold, 111))


def test_zero_weights_are_not_compared_and_real_differences_still_are():
    from fugal_subnet.v2_sync import comparable_reveal

    base = {"v2_state": {"seat": "h1"}, "weights": {"0": 0.0, "5": 0.85, "6": 0.15}}
    assert consensus_view(base) == consensus_view({**base, "weights": {"5": 0.85, "6": 0.15}})
    for other in ({"5": 0.85, "6": 0.15, "7": 1e-12}, {"5": 0.86, "6": 0.14}, {"5": 1.0}):
        assert consensus_view(base) != consensus_view({**base, "weights": other})
    assert consensus_view(base) != consensus_view({**base, "v2_state": {"seat": "h2"}})
    assert consensus_view({"weights": None}) == consensus_view({})     # never raises
    rv = {**base, "environment": {"cpu": 8}, "v2_sync": {"unverified": True}, "scores": {"5": 1}}
    other = {**rv, "weights": {"5": 0.85, "6": 0.15}, "environment": {"cpu": 2},
             "v2_sync": {"unverified": False}}
    assert comparable_reveal(rv) == comparable_reveal(other)
    assert comparable_reveal(rv) != comparable_reveal({**other, "scores": {"5": 2}})


def test_every_agreement_check_shares_one_definition():
    """FINAL REVIEW, finding 4: watch_subnet and rehearsal_v2 dropped only
    `environment`, so a follower with an open gap showed as differing."""
    import pathlib

    from scripts.rehearsal_v2 import comparable_reveal as rehearsal_comparable

    rv = {"weights": {"0": 0.0, "5": 1.0}, "v2_sync": {"unverified": True}, "environment": {}}
    assert rehearsal_comparable(rv) == {"weights": {"5": 1.0}}
    watch = pathlib.Path(__file__).resolve().parents[1].joinpath(
        "scripts", "watch_subnet.py").read_text()
    assert "comparable_reveal(rv)" in watch and 'k != "environment"' not in watch


def test_drift_needs_every_one_of_the_newest_verified_reference_epochs_to_differ():
    from fugal_subnet.v2_sync import DRIFT_CHECKS, drifted

    assert DRIFT_CHECKS == 3

    def rv(seat, unverified=False):
        return ("p", {"v2_state": {"seat": seat}, "weights": {"5": 1.0},
                      "v2_sync": {"unverified": unverified}})

    ref = {k: rv("A") for k in range(10, 20)}
    own = {k: rv("B") for k in range(10, 20)}
    assert drifted(ref, own, 20) == 17                         # 17, 18, 19 all differ
    assert drifted(ref, own, 19) == 16
    own[18] = rv("A")                                          # one of them agrees: not drift
    assert drifted(ref, own, 20) is None
    own[18] = rv("B")
    del own[19]                                                # a reveal of ours is missing
    assert drifted(ref, own, 20) is None
    own[19] = rv("B")
    ref[19] = rv("A", unverified=True)                         # skipped, like check()
    assert drifted(ref, own, 20) == 16
    assert drifted({k: ref[k] for k in (18,)}, own, 20) is None   # too few reference epochs
    same = {k: ("p", {**ref[k][1], "weights": {"5": 1.0, "0": 0.0}}) for k in ref}
    assert drifted(ref, same, 20) is None                      # zero weights are not a difference
