#!/usr/bin/env python3
"""Fast crowning in protocol v4: the measurements and simulations behind
docs/decisions/FAST_CROWN_V4.md. Nothing here is consensus code; every
decision it simulates calls the real bound (`seat_bound.bound_from_strata`).

Two steps:

1. `measure --reveals DIR` reads published protocol v3 reveals (flat
   e*.json files, as `https://storage.googleapis.com/fugal-reveals/70/<epoch
   id>/reveal.json` serves them) and writes the measured inputs to
   `data/observations/fast_crown_v4_inputs_e00026466-e00026493.json`: how often two real heads
   pick different entries, the paired net-value items of real heads against
   the champion (priced at the v4 seed's pick-cost table, $1 a correct
   answer), the epoch-to-epoch spread of a pair's mean, the odds test's
   per-label gains, labels per epoch, and per entry its call latencies,
   failures by epoch and accuracy by category. No head, operator or answer is
   named; entries are public menu ids.

2. `a`, `b`, `c`, `e`, `time` or `all` run the simulations on those inputs
   (fixed seeds, one process, well under 1 GB):

   a  the one-epoch routing test (rule B) at 200 to 1,500 questions;
   b  repeated attempts by a challenger that is not better;
   c  the odds test's floor (rule A) with realistic label arrival;
   e  one measuring epoch for a proposed entry;
   time  how long a challenge or a measurement takes inside a TD.

Run: python scripts/experiments/experiment_fast_crown_v4.py all [--quick].
No network, no paid call.
"""
from __future__ import annotations

import argparse
import base64
import collections
import glob
import hashlib
import heapq
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fugal_subnet.seat_bound import Stratum, bound_from_strata  # noqa: E402

INPUTS = ROOT / "data" / "observations" / "fast_crown_v4_inputs_e00026466-e00026493.json"
CP = 10_000            # bound units (a millionth of a point) per centipoint
MARGIN_B = -1.5        # rule B's margin, points (SEAT_MINIMAL_V4.md 3.4)
MARGIN_A = 0.1         # rule A's margin, Brier points (3.3)
SIZES = (200, 300, 500, 750, 1000, 1500)
EDGES_B = (-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0)
EDGES_A = (0.5, 1.0, 2.0, 4.0)
CONCURRENCY = 40       # config.HARNESS_CONCURRENCY's default, what a TD runs
EXPLORE_CALLS = 112    # config.EXPLORE_PRIVATE_SIZE_V4
CALL_BUDGET_S = 30 * 60  # collection at 36 min of a 72-min epoch, ~6 min for the rest
N0_LOW = 100           # the bound tuned low (a constant), against n0 = the floor as built


# --- step 1: measure ----------------------------------------------------------------------


def _mix():
    from fugal_subnet import epoch_plan
    m = epoch_plan.mix_v4()
    return dict(m.targets), dict(m.bench_category)


def _sub(values, k, seed):
    """At most k of `values`, a fixed-seed subsample (the committed file stays small)."""
    values = list(values)
    if len(values) <= k:
        return values
    idx = np.random.default_rng(seed).choice(len(values), size=k, replace=False)
    return [values[i] for i in sorted(idx)]


def measure(reveals_dir: str) -> dict:
    from fugal_subnet import menu_state as ms

    targets, cat_of = _mix()
    seed = ms.load_seed()
    on_seed = {e.id for e in seed.entries}
    files = sorted(glob.glob(os.path.join(reveals_dir, "e*.json")))
    if not files:
        raise SystemExit(f"no e*.json reveals in {reveals_dir}")
    sources = []
    pairs = []            # one per (epoch, head) against the champion
    disagree_all = []     # every pair of distinct heads
    pick_counts = collections.Counter()
    holder_counts = collections.Counter()
    scored_fail = []      # per head-epoch share of scored calls failed
    holder_cost, challenger_cost = [], []
    lat = collections.defaultdict(list)
    fails = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    gains = collections.defaultdict(list)     # (epoch, head) -> weighted gains, Brier points
    entry_labels = collections.defaultdict(list)   # entry -> [y, weight, champion odds, median odds]
    champ_value = collections.defaultdict(lambda: [0, 0.0, 0.0])   # category -> n, sum, squares
    cell_out = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    pool_sizes, operators, hotkeys, two_op_share = [], [], [], []
    for f in files:
        raw = open(f, "rb").read()
        d = json.loads(raw)
        ep = d["epoch_id"]
        sources.append([ep, hashlib.sha256(raw).hexdigest()])
        models = d["models"]
        qids = [q["question_id"] for q in d["questions"]]
        cats = [cat_of.get(q["benchmark"], q["benchmark"]) for q in d["questions"]]
        cells = {(g["question_id"], g["model"]): float(g["grade"]) for g in d["shared_grades"]}
        cat_q = dict(zip(qids, cats))
        for g in d["shared_grades"]:
            if g["question_id"] in cat_q and g.get("completion_tokens") is not None:
                co = cell_out[g["model"]][cat_q[g["question_id"]]]
                co[0] += 1
                co[1] += float(g["completion_tokens"])
        # (an entry's accuracy is read from exploration labels only, below: the
        # graded scored cells are the questions routers chose to send it)
        champ = (d.get("champion") or {}).get("champion")
        first = {}
        for uid, h in sorted(d["head_hashes"].items(), key=lambda t: int(t[0])):
            first.setdefault(h, uid)
        vals, picks = {}, {}
        for h, uid in first.items():
            fl = set(d["failed"].get(uid, []))
            scored_fail.append(len(fl) / len(qids))
            v, p, cost = [], [], []
            for i, mi in enumerate(d["routing_decisions"][uid]):
                m = models[mi]
                p.append(m)
                if m not in on_seed:
                    v.append(np.nan)
                    continue
                t = float(ms.pick_cost(seed, m, cats[i]))
                cost.append(t)
                if qids[i] in fl:
                    v.append(0.0)
                    continue
                g = cells.get((qids[i], m))
                v.append(np.nan if g is None else 100 * g - 100 * t)
            vals[h], picks[h] = np.array(v), p
            pick_counts.update(p)
            if h == champ:
                holder_counts.update(p)
                holder_cost.append(float(np.mean(cost)))
            else:
                challenger_cost.append(float(np.mean(cost)))
        hs = sorted(vals)
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                disagree_all.append(float(np.mean([a != b for a, b in zip(picks[hs[i]], picks[hs[j]])])))
        if champ in vals:
            b = vals[champ]
            for c, x in zip(cats, b.tolist()):
                if not np.isnan(x):
                    cv = champ_value[c]
                    cv[0] += 1
                    cv[1] += x
                    cv[2] += x * x
            for h in hs:
                if h == champ:
                    continue
                a = vals[h]
                ok = ~np.isnan(a) & ~np.isnan(b)
                nv = (a - b)[ok]
                cs = np.array(cats)[ok]
                dis = np.array([x != y for x, y in zip(picks[h], picks[champ])])[ok]
                pairs.append({"epoch": ep, "head": h, "bar": champ, "nv": nv, "cats": cs, "disagree": dis,
                              "unpaired_var": float(np.var(a[ok]) + np.var(b[ok]))})
        # exploration: latencies, failures, accuracy, labels
        pool = d["exploration_pool"]
        pool_sizes.append(len(pool))
        for p_ in pool:
            if p_.get("latency_ms") is not None:
                lat[p_["model"]].append(int(p_["latency_ms"]))
            c = cat_of.get(p_["benchmark"])
            if c is not None:
                acc[p_["model"]][c][0] += 1
                acc[p_["model"]][c][1] += p_["y"] / 10_000
        voters = collections.defaultdict(set)
        for x in d["exploration"]:
            fails[x["model"]][ep][0] += 1
            fails[x["model"]][ep][1] += int(bool(x["failed"]))
            if not x["failed"]:
                voters[(x["question_id"], x["model"])].add(x["coldkey"])
        cks = sorted({x["coldkey"] for x in d["exploration"]})
        operators.append(len(cks))
        hotkeys.append(len(d["head_hashes"]))
        if len(pool) and len(cks) >= 3:
            for i in range(len(cks)):
                for j in range(i + 1, len(cks)):
                    keep = sum(1 for p_ in pool
                               if not ({cks[i], cks[j]} & voters[(p_["question_id"], p_["model"])]))
                    two_op_share.append(keep / len(pool))
        odds = {h: np.frombuffer(base64.b64decode(s), dtype="<i2").astype(np.int64)
                for h, s in (d.get("head_odds") or {}).items()}
        if champ in odds:
            y = np.array([p_["y"] for p_ in pool], dtype=np.int64)
            w = np.array([p_["weight"] for p_ in pool], dtype=np.int64)
            oc = odds[champ]
            stack = np.array([o for o in odds.values() if len(o) == len(y)])
            for i, p_ in enumerate(pool):
                col = stack[:, i]
                col = col[col >= 0]
                if oc[i] >= 0 and len(col):
                    entry_labels[p_["model"]].append(
                        [int(p_["y"]), int(p_["weight"]), int(oc[i]), int(np.median(col))])
            for h, oh in sorted(odds.items()):
                if h == champ or len(oh) != len(y):
                    continue
                ok = (oh >= 0) & (oc >= 0)
                u = (oc - y) ** 2 - (oh - y) ** 2
                uw = (2 * u * w + 1_000_000) // 2_000_000
                gains[(ep, h)].extend((uw[ok] / 1e6).tolist())

    # --- the picks summary
    disagree = np.array([p["disagree"].mean() for p in pairs])
    means = np.array([p["nv"].mean() for p in pairs])
    sds = []
    for p in pairs:
        wv = sum(((p["nv"][p["cats"] == c] - p["nv"][p["cats"] == c].mean()) ** 2).sum()
                 for c in set(p["cats"].tolist())) / len(p["nv"])
        sds.append(math.sqrt(wv))
    sds = np.array(sds)
    unpaired = np.sqrt([p["unpaired_var"] for p in pairs])
    close = np.abs(means) <= 2.0
    # the epoch-to-epoch spread of one pair's mean, beyond its sampling error
    by = collections.defaultdict(list)
    for p in pairs:
        by[(p["head"], p["bar"])].append((p["nv"].mean(), p["nv"].var(ddof=1) / len(p["nv"])))
    num = den = 0.0
    n_pairs_tau = 0
    for v in by.values():
        if len(v) < 5:
            continue
        m = np.array([x[0] for x in v])
        num += (m.var(ddof=1) - np.mean([x[1] for x in v])) * (len(v) - 1)
        den += len(v) - 1
        n_pairs_tau += 1
    tau2 = num / den if den else 0.0
    # centred items by category: nonzero items of a pair, its mean removed
    # from them, so a pair's items have mean 0 and its zeros stay zero
    items = collections.defaultdict(list)
    nz_share = []
    for p in pairs:
        nz = p["nv"] != 0
        nz_share.append(float(nz.mean()))
        if nz.sum() < 10:
            continue
        shift = p["nv"].sum() / nz.sum()
        for c, x in zip(p["cats"][nz].tolist(), (p["nv"][nz] - shift).tolist()):
            items[c].append(int(round(x * 100)))
    # --- odds gains, standardised per (epoch, head)
    unit = []
    gsd = []
    kurts = []
    for v in gains.values():
        a = np.array(v)
        if len(a) < 50 or a.std() == 0:
            continue
        gsd.append(float(a.std()))
        z = (a - a.mean()) / a.std()
        kurts.append(float((z ** 4).mean()))
        unit.extend(z.tolist())
    # --- entries
    entries = {}
    seed_cost = {e.id: (float(ms.entry_cost(e)), float(ms.entry_pick_cost(e))) for e in seed.entries}
    total_holder = sum(holder_counts.values())
    total_picks = sum(pick_counts.values())
    for m in sorted(set(lat) | set(fails) | set(acc)):
        ef = fails.get(m, {})
        entries[m] = {
            "latency_ms": _sub(sorted(lat.get(m, [])), 400, 7),
            "calls": sum(c for c, _f in ef.values()),
            "failed": sum(f_ for _c, f_ in ef.values()),
            "by_epoch": [[c, f_] for _e, (c, f_) in sorted(ef.items())],
            "accuracy": {c: [n, round(k, 3)] for c, (n, k) in sorted(acc.get(m, {}).items())},
            "call_cost_usd": seed_cost.get(m, (None, None))[0],
            "pick_cost_usd": seed_cost.get(m, (None, None))[1],
            "holder_pick_share": round(holder_counts.get(m, 0) / total_holder, 5) if total_holder else 0,
            "pick_share": round(pick_counts.get(m, 0) / total_picks, 5),
            "labels": entry_labels.get(m, []),
            "cell_out": {c: [n, round(t, 1)] for c, (n, t) in sorted(cell_out.get(m, {}).items())},
        }

    def q(a, ps):
        return [round(float(x), 4) for x in np.percentile(a, ps)]

    return {
        "format": "fast-crown-v4-inputs-1",
        "source": {"reveals": sources,
                   "url": "https://storage.googleapis.com/fugal-reveals/70/<epoch id>/reveal.json",
                   "pick_costs": "the v4 seed menu's pick-cost table (fugal_subnet/routing_data/"
                                 "menu_v4_seed.json), $1 a correct answer"},
        "picks": {
            "pairs_vs_champion": len(pairs),
            "disagree_vs_champion_q": q(disagree, [10, 25, 50, 75, 90]),
            "disagree_close_heads_median": round(float(np.median(disagree[close])), 4),
            "disagree_all_pairs_q": q(disagree_all, [10, 25, 50, 75, 90]),
            "nonzero_share_median": round(float(np.median(nz_share)), 4),
            "nv_mean_q": q(means, [10, 25, 50, 75, 90]),
            "paired_sd_q": q(sds, [10, 25, 50, 75, 90]),
            "paired_sd_close_heads_median": round(float(np.median(sds[close])), 3),
            "unpaired_sd_q": q(unpaired, [10, 25, 50, 75, 90]),
            "epoch_tau_points": round(math.sqrt(max(tau2, 0.0)), 3),
            "epoch_tau_pairs": n_pairs_tau,
            "scored_fail_share_q": q(scored_fail, [50, 90, 99]),
            "pick_cost_usd_holder_mean": round(float(np.mean(holder_cost)), 5),
            "pick_cost_usd_challenger_median": round(float(np.median(challenger_cost)), 5),
            "items_centipoints": {c: _sub(v, 4000, 11) for c, v in sorted(items.items())},
        },
        "odds": {
            "gain_sd_q": q(gsd, [10, 25, 50, 75, 90]),
            "kurtosis_median": round(float(np.median(kurts)), 2),
            "unit_gains_milli": [int(round(x * 1000)) for x in _sub(unit, 20000, 13)],
            "v3_pool_labels_per_epoch_q": q(pool_sizes, [10, 50, 90]),
            "v3_operators_per_epoch_q": q(operators, [10, 50, 90]),
            "v3_hotkeys_per_epoch_q": q(hotkeys, [10, 50, 90]),
            "v3_share_left_after_two_operators_q": q(two_op_share, [10, 50, 90]),
        },
        "entries": entries,
        "mix_targets_pct": targets,
        "champion_value_by_category": {c: [n, round(t, 3), round(q, 3)]
                                       for c, (n, t, q) in sorted(champ_value.items())},
    }


def load_inputs(path: Path = INPUTS) -> dict:
    with open(path) as f:
        return json.load(f)


# --- the bound on centipoint items --------------------------------------------------------


def bound_cp(sums: dict, margin_points: float, n_min: int, n0: int, tau0: float = 0.0):
    """seat_bound's bound on per-category sums of centipoint items:
    {category: (n, sum, sum of squares)}, all exact integers. With `tau0`
    (points) and an "_e2" entry (the sum over the record's epochs of the
    squared items an epoch), the variance gets tau0^2 x that sum: an epoch's
    evidence then counts for no more than an epoch-wide shift of SD tau0
    allows (the candidate guard of FAST_CROWN_V4.md)."""
    sums = dict(sums)
    e2 = sums.pop("_e2", (0, 0, 0))[1]
    st = {c: Stratum(int(n), int(s) * CP, int(q) * CP * CP) for c, (n, s, q) in sorted(sums.items())}
    if tau0 and st:
        c = next(iter(st))
        extra = int(e2) * int(round(tau0 * 1_000_000)) ** 2
        st[c] = Stratum(st[c].n, st[c].total, st[c].squares + extra)
    return bound_from_strata(st, margin_units=int(round(margin_points * 1_000_000)),
                             n_min=n_min, n0=n0)


def quotas(n: int, targets: dict) -> dict:
    """Largest-remainder category counts for n questions on the mix."""
    tot = sum(targets.values())
    raw = {c: n * v / tot for c, v in targets.items()}
    out = {c: int(math.floor(x)) for c, x in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - out[c], reverse=True)[: n - sum(out.values())]:
        out[c] += 1
    return out


# --- A: the one-epoch routing test --------------------------------------------------------


class PicksModel:
    """Rule B's items for a synthetic challenger: per question, with the
    measured nonzero share p a resampled centred real item plus mu / p, else
    exactly 0 (both heads routed the same and one answer served both). The
    real items are scaled so one item's SD at the v4 mix is `sd` (default:
    the median per-pair paired SD measured). An epoch shock of SD `tau`
    points is shared by every item of one epoch."""

    def __init__(self, inp: dict, sd: float | None = None):
        pk = inp["picks"]
        self.p = pk["nonzero_share_median"]
        self.targets = inp["mix_targets_pct"]
        self.items = {c: np.array(v, dtype=np.float64) for c, v in pk["items_centipoints"].items()}
        for c in self.targets:
            if c not in self.items:   # a category with no real items: pool all
                self.items[c] = np.concatenate(list(self.items.values()))
        tot = sum(self.targets.values())
        raw_var = sum(self.targets[c] / tot * self.p * (self.items[c] ** 2).mean()
                      for c in self.targets) / 1e4
        self.sd = pk["paired_sd_q"][2] if sd is None else sd
        self.scale = self.sd / math.sqrt(raw_var)

    def batch(self, rng, n: int, mu: float, runs: int, tau: float = 0.0) -> dict:
        """{category: (runs x n_c) int64 centipoint items} for `runs` epochs."""
        shock = rng.normal(0.0, tau, size=(runs, 1)) if tau > 0 else 0.0
        out = {}
        for c, k in quotas(n, self.targets).items():
            if k == 0:
                continue
            pool = self.items[c]
            x = pool[rng.integers(0, len(pool), size=(runs, k))] * self.scale
            nz = rng.random((runs, k)) < self.p
            x = np.where(nz, x + 100.0 * (mu + shock) / self.p, 0.0)
            out[c] = np.rint(x).astype(np.int64)
        return out

    @staticmethod
    def sums(batch: dict, r: int) -> dict:
        out = {c: (a.shape[1], int(a[r].sum()), int((a[r] * a[r]).sum())) for c, a in batch.items()}
        out["_e2"] = (1, sum(a.shape[1] for a in batch.values()) ** 2, 0)
        return out


def n0_for(n_min: int, tuning: str) -> int:
    return n_min if tuning == "min" else N0_LOW


TAUS = (0.0, 0.7)      # the epoch shock: none (failures fail closed), and v3's measured 0.7


def sim_a(inp: dict, runs: int = 2000, seed: int = 1, sds=(None,)) -> dict:
    """Pass rate of rule B for one attempt of n questions judged alone (n is
    the floor), for each true NV edge: the bound tuned as built (n0 = n) and
    at n0 = 100; with no epoch shock and with v3's measured one."""
    out = {}
    for sd in sds:
        model = PicksModel(inp, sd)
        for tuning in ("min", "low"):
            for tau in TAUS:
                for n in SIZES:
                    for mu in EDGES_B:
                        rng = np.random.default_rng([seed, n, int(mu * 10) + 100, int(tau * 1000),
                                                     tuning == "low", int(model.sd * 10)])
                        b = model.batch(rng, n, mu, runs, tau)
                        n0 = n0_for(n, tuning)
                        ok = sum(bound_cp(model.sums(b, r), MARGIN_B, n, n0).passes
                                 for r in range(runs))
                        out[(round(model.sd, 1), tuning, tau, n, mu)] = ok / runs
    return out


def cost_per_question(inp: dict) -> float:
    """Dollars a challenge question costs the challenger: its own pick, and
    the holder's pick where they differ (one answer serves both otherwise)."""
    pk = inp["picks"]
    return pk["pick_cost_usd_challenger_median"] + pk["disagree_vs_champion_q"][2] * \
        pk["pick_cost_usd_holder_mean"]


# --- B: repeated attempts ------------------------------------------------------------------

RULES_B = ("fresh", "run", "keep", "declared")


def sim_b(inp: dict, runs: int = 400, epochs: int = 300, seed: int = 2, floor: int = 500,
          mus=(-1.75, -2.0, -3.0), sizes=(500, 1000, 1500), taus=TAUS, tau0s=(0.0, 0.5)) -> dict:
    """A challenger that is not better (true NV `mu` below the holder) runs
    the challenge every epoch for `epochs` epochs, n0 = 100, floor 500, and
    serves a proof when it suits it. Rules:

    fresh     every attempt is judged alone;
    run       as built: the record is the head's unbroken run of answered
              epochs, and a withheld epoch restarts it; the challenger keeps
              a batch when it raises the run's bound and withholds it (so
              restarting) otherwise;
    keep      one record that never resets (48-epoch window), and the
              challenger withholds every batch that would lower its bound;
    declared  one record that never resets and cannot be withheld (a
              declared attempt that is not delivered ends the record); the
              challenger starts a new head when its bound is more than a point
              under the margin, the new head attempting 4 epochs later.
    `tau0` is the guard of `bound_cp` (0: the bound as built).
    Returns {(n, mu, tau, tau0, rule): (share with a false pass by epoch 48, by the end)}."""
    model = PicksModel(inp)
    out = {}
    n0 = n0_for(floor, "low")
    for tau in taus:
        for n in sizes:
            for mu in mus:
                rng = np.random.default_rng([seed, n, int(-mu * 100), int(tau * 100)])
                res = {(t0, k): [0, 0] for t0 in tau0s for k in RULES_B}
                for _r in range(runs):
                    b = model.batch(rng, n, mu, epochs, tau)
                    ep_sums = [model.sums(b, e) for e in range(epochs)]
                    for t0 in tau0s:
                        for rule in RULES_B:
                            _tally(res[(t0, rule)], _attacker(ep_sums, rule, floor, n0, tau0=t0), 48)
                for (t0, k), (w, a) in res.items():
                    out[(n, mu, tau, t0, k)] = (w / runs, a / runs)
    return out


VOID_RULES = ("deliver", "deliver+cap", "run+cap", "void-free", "void-once")


def sim_void(inp: dict, runs: int = 200, epochs: int = 300, seed: int = 15, floor: int = 500,
             mus=(0.0, -1.75, -2.0), sizes=(500, 1000), cap: float = 0.10,
             holder_acc: float = 0.85, window: int = 48) -> dict:
    """The holder-side failure cap (picks_v4.HOLDER_FAIL_CAP_BP): a challenger
    that is not better tries every epoch, n0 = 100, floor 500, with perfect
    knowledge of each epoch's items before it decides (the worst case).

    deliver      serves every epoch, no failure (the baseline);
    deliver+cap  serves every epoch and makes `cap` of the holder-side calls
                 fail on purpose (each scores the holder right: an item falls
                 by 100 x (1 - g_holder), g_holder ~ Bernoulli(holder_acc));
    run+cap      the same failures, and withholds an epoch that would lower
                 its run's bound (the gap restarts the run: as built);
    void-free    fails more than the cap whenever an epoch would lower its
                 bound: the challenge is void, neither counted nor breaking
                 the run, with no limit (the run is then a selection);
    void-once    the same, but a run passes through one void epoch only: a
                 second restarts it (VOID_MAX_IN_RUN_V4).
    The record is the last `window` epochs. Returns {(n, mu, rule): (share
    with a false pass by epoch 48, by the end)}."""
    model = PicksModel(inp)
    n0 = n0_for(floor, "low")
    out = {}
    for n in sizes:
        for mu in mus:
            rng = np.random.default_rng([seed, n, int(-mu * 100) + 1000])
            res = {k: [0, 0] for k in VOID_RULES}
            for _r in range(runs):
                b = model.batch(rng, n, mu, epochs)
                failed = {}
                for c, a in b.items():
                    nz = a != 0
                    hit = nz & (rng.random(a.shape) < cap)
                    wrong = rng.random(a.shape) >= holder_acc
                    failed[c] = a - np.where(hit & wrong, 100 * 100, 0)   # centipoints
                plain = [model.sums(b, e) for e in range(epochs)]
                worse = [model.sums(failed, e) for e in range(epochs)]
                for rule in VOID_RULES:
                    _tally(res[rule], _void_attacker(plain, worse, rule, floor, n0, window), 48)
            for k, (w, a) in res.items():
                out[(n, mu, k)] = (w / runs, a / runs)
    return out


def _void_attacker(plain, worse, rule, floor, n0, window):
    """The first epoch the attacker passes rule B under `rule`, or None."""
    kept: list = []          # (epoch, sums) in the run
    rec: dict = {}
    voids = 0
    for e in range(len(plain)):
        while kept and kept[0][0] <= e - window:
            rec = _add(rec, kept.pop(0)[1], -1)
        s = worse[e] if rule.endswith("+cap") else plain[e]
        trial = _add(rec, s)
        bt = bound_cp(trial, MARGIN_B, floor, n0)
        if bt.passes:
            return e
        if rule.startswith("deliver"):
            rec, kept = trial, kept + [(e, s)]
            continue
        br = bound_cp(rec, MARGIN_B, floor, n0) if rec else None
        if br is None or bt.lower > br.lower:
            rec, kept = trial, kept + [(e, s)]
        elif rule == "run+cap":
            rec, kept = {}, []
        elif rule == "void-free":
            pass
        else:
            voids += 1
            if voids > 1:
                rec, kept, voids = {}, [], 0
    return None


def _tally(acc, first, window):
    if first is not None:
        acc[1] += 1
        if first < window:
            acc[0] += 1


def _add(a: dict, b: dict, sign: int = 1) -> dict:
    out = dict(a)
    for c, (n, s, q) in b.items():
        n0, s0, q0 = out.get(c, (0, 0, 0))
        out[c] = (n0 + sign * n, s0 + sign * s, q0 + sign * q)
    return {c: v for c, v in out.items() if v[0] > 0}


def _attacker(ep_sums, rule, floor, n0, window: int = 48, gap: int = 4, tau0: float = 0.0,
              margin: float = MARGIN_B):
    """The first epoch the attacker's record passes rule B, or None."""
    rec, kept, start = {}, [], 0
    for e, s in enumerate(ep_sums):
        if e < start:
            continue
        while kept and kept[0][0] <= e - window:
            rec = _add(rec, kept.pop(0)[1], -1)
        trial = _add(rec, s)
        bt = bound_cp(trial, margin, floor, n0, tau0)
        if bt.passes:
            return e
        if rule == "fresh":
            continue
        if rule == "declared":
            rec, kept = trial, kept + [(e, s)]
            if bt.n >= floor and bt.lower < (margin - 1.0) * 1_000_000:
                rec, kept, start = {}, [], e + 1 + gap
            continue
        br = bound_cp(rec, margin, floor, n0, tau0) if rec else None
        if br is None or bt.lower > br.lower:
            rec, kept = trial, kept + [(e, s)]
        elif rule == "run":
            rec, kept = {}, []          # the withheld epoch breaks the run
    return None


def honest_b(inp: dict, runs: int = 400, seed: int = 3, sizes=(500, 1000, 1500),
             mus=(0.5, 1.0, 2.0, 4.0), floor: int = 500, limit: int = 20,
             tau0s=(0.0, 0.5), sd: float | None = None) -> dict:
    """Epochs for an honest challenger (delivers every epoch, one record) to
    pass rule B at a chosen size, the bound tuned at n0 = 100: the share
    passed after 1, 2 and 5 attempts, and the median attempts."""
    model = PicksModel(inp, sd)
    n0 = n0_for(floor, "low")
    out = {}
    for n in sizes:
        for mu in mus:
            rng = np.random.default_rng([seed, n, int(mu * 10)])
            got = {t0: [] for t0 in tau0s}
            for _r in range(runs):
                b = model.batch(rng, n, mu, limit)
                for t0 in tau0s:
                    rec = {}
                    first = None
                    for e in range(limit):
                        rec = _add(rec, model.sums(b, e))
                        if bound_cp(rec, MARGIN_B, floor, n0, t0).passes:
                            first = e + 1
                            break
                    got[t0].append(first if first is not None else 10 ** 6)
            for t0, g in got.items():
                g = np.array(g)
                out[(n, mu, t0)] = ((g <= 1).mean(), (g <= 2).mean(), (g <= 5).mean(),
                                    float(np.median(g)))
    return out


def width_factor(n: int, n0: int) -> float:
    """The bound's half-width in standard errors at n items: the formula of
    seat_bound with its pinned rationals (A_ALPHA, LN2_UP)."""
    from fugal_subnet import seat_bound
    k = (1 + -(-n // n0)).bit_length()
    return math.sqrt((1 + n0 / n) * float(seat_bound.A_ALPHA + seat_bound.LN2_UP * k))


def tickets(inp: dict, sds=(20.0, 24.0), sizes=(500, 1000, 1500), mus=(-1.6, -1.75, -2.0, -3.0),
            n0: int = N0_LOW) -> dict:
    """One attempt judged alone, normal approximation of its mean (the
    simulation finds no false pass at these sizes, so the rate is computed):
    the chance it passes rule B, and the dollars an attacker spends per
    expected false pass. Fresh attempts, extra TD runs on the same questions
    (best of N) and restarted runs are each at most one such ticket."""
    from statistics import NormalDist
    cq = cost_per_question(inp)
    out = {}
    for sd in sds:
        for n in sizes:
            se = sd / math.sqrt(n)
            for mu in mus:
                z = (MARGIN_B + width_factor(n, n0) * se - mu) / se
                p = 1 - NormalDist().cdf(z)
                out[(sd, n, mu)] = (p, n * cq / p if p > 0 else float("inf"))
    return out


# --- C: the odds test's floor ----------------------------------------------------------------


class OddsModel:
    """Rule A's per-label gains: the measured standardised gains (heavy
    tailed) times sigma, plus the edge d. A share of the labels are on fresh
    questions every other hotkey answers (clusters of size m), correlated
    within the question at rho (0.29, measured by the memorisation lane on
    the skill statistic)."""

    def __init__(self, inp: dict, sigma: float = 8.0, rho: float = 0.29):
        self.unit = np.array(inp["odds"]["unit_gains_milli"], dtype=np.float64) / 1000.0
        self.unit = (self.unit - self.unit.mean()) / self.unit.std()
        self.sigma, self.rho = sigma, rho

    def epoch_sums(self, rng, heads: int, labels: int, fresh: int, m: int, d: np.ndarray,
                   sym: bool = True):
        """(heads,) arrays of n, sum, squares in centi-Brier-points for one epoch."""
        u = self.unit
        single = labels - fresh * m
        x1 = u[rng.integers(0, len(u), size=(heads, max(single, 0)))]
        q = u[rng.integers(0, len(u), size=(heads, fresh, 1))]
        e = u[rng.integers(0, len(u), size=(heads, fresh, m))]
        x2 = (math.sqrt(self.rho) * q + math.sqrt(1 - self.rho) * e).reshape(heads, fresh * m)
        x = np.concatenate([x1, x2], axis=1)
        if sym:   # a random sign per head-epoch keeps the mean exactly 0 in distribution
            x = x * rng.choice([-1.0, 1.0], size=(heads, 1))
        x = np.rint((x * self.sigma + d[:, None]) * 100).astype(np.int64)
        return x.shape[1], x.sum(axis=1), (x * x).sum(axis=1)


FIELDS = {5: (450, 14, 4), 15: (1900, 14, 16)}   # operators: (labels a comparison, fresh, m)


def sim_c_false(inp: dict, fields: int = 20, heads: int = 30, epochs: int = 300, seed: int = 4,
                floors=(50, 100, 300, 500, 700, 1000, 1500), sigma: float = 8.0) -> dict:
    """`heads` no-edge heads over `epochs` epochs, re-tested every epoch on
    the 48-epoch window: how many of `fields` simulated fields see any head
    pass rule A, per floor (n0 = floor as built, and n0 = 100), and the
    largest bound reached."""
    model = OddsModel(inp, sigma=sigma)
    out = {}
    for ops, (labels, fresh, m) in FIELDS.items():
        rng = np.random.default_rng([seed, ops])
        hits = {(f, t): 0 for f in floors for t in ("min", "low")}
        best = {(f, t): -1e9 for f in floors for t in ("min", "low")}
        for _ in range(fields):
            hist = []
            win = np.zeros((heads, 3), dtype=object)
            win[:] = 0
            passed = {k: False for k in hits}
            for e in range(epochs):
                n, s, q = model.epoch_sums(rng, heads, labels, fresh, m, np.zeros(heads))
                hist.append((s, q))
                if len(hist) > 48:
                    so, qo = hist.pop(0)
                    win[:, 1] = win[:, 1] - so.astype(object)
                    win[:, 2] = win[:, 2] - qo.astype(object)
                    win[:, 0] = win[:, 0] - n
                win[:, 0] = win[:, 0] + n
                win[:, 1] = win[:, 1] + s.astype(object)
                win[:, 2] = win[:, 2] + q.astype(object)
                for h in range(heads):
                    sums = {"all": (int(win[h, 0]), int(win[h, 1]), int(win[h, 2]))}
                    for f in floors:
                        for t in ("min", "low"):
                            if passed[(f, t)]:
                                continue
                            b = bound_cp(sums, MARGIN_A, f, n0_for(f, t))
                            if b.lower is not None and b.n >= f:
                                best[(f, t)] = max(best[(f, t)], b.lower / 1e6)
                            if b.passes:
                                passed[(f, t)] = True
            for k, v in passed.items():
                hits[k] += int(v)
        for k in hits:
            out[(ops,) + k] = (hits[k], fields, best[k])
    return out


def sim_c_speed(inp: dict, runs: int = 300, seed: int = 5, floors=(300, 500, 700, 1000, 1500),
                sigma: float = 8.0, limit: int = 48) -> dict:
    """Epochs to pass rule A for a head d Brier points better than the bar."""
    model = OddsModel(inp, sigma=sigma)
    out = {}
    for ops, (labels, fresh, m) in FIELDS.items():
        for d in EDGES_A:
            rng = np.random.default_rng([seed, ops, int(d * 10)])
            got = {(f, t): [] for f in floors for t in ("min", "low")}
            for _ in range(runs):
                cum = [0, 0, 0]
                first = {k: None for k in got}
                for e in range(limit):
                    n, s, q = model.epoch_sums(rng, 1, labels, fresh, m, np.array([d]), sym=False)
                    cum = [cum[0] + n, cum[1] + int(s[0]), cum[2] + int(q[0])]
                    for k in got:
                        if first[k] is None and bound_cp({"all": tuple(cum)}, MARGIN_A, k[0],
                                                         n0_for(k[0], k[1])).passes:
                            first[k] = e + 1
                    if all(v is not None for v in first.values()):
                        break
                for k, v in first.items():
                    got[k].append(v if v is not None else 10 ** 6)
            for k, v in got.items():
                v = np.array(v)
                out[(ops, d) + k] = (float(np.median(v)), float((v <= 1).mean()), float((v <= 2).mean()))
    return out


# --- time: how long a TD takes ------------------------------------------------------------


def makespan(lat_s: np.ndarray, workers: int) -> float:
    """List scheduling of calls in order on `workers` slots."""
    if len(lat_s) <= workers:
        return float(lat_s.max()) if len(lat_s) else 0.0
    h = [0.0] * workers
    for x in lat_s:
        t = heapq.heappop(h)
        heapq.heappush(h, t + x)
    return max(h)


def sim_time(inp: dict, runs: int = 200, seed: int = 6, sizes=(300, 500, 1000, 1500),
             workers=(40, 80, 160)) -> dict:
    """Wall time of a TD epoch: the challenge (n questions: the challenger's
    pick and, where they differ, the holder's, drawn from the measured pick
    shares) plus 112 exploration calls (drawn from every entry), each call's
    latency resampled from that entry's measured latencies."""
    ents = inp["entries"]
    names = [m for m, e in ents.items() if e["latency_ms"]]
    lat = {m: np.array(ents[m]["latency_ms"], dtype=np.float64) / 1000 for m in names}
    share = np.array([ents[m]["pick_share"] for m in names])
    share = share / share.sum()
    hshare = np.array([ents[m]["holder_pick_share"] for m in names])
    hshare = hshare / hshare.sum() if hshare.sum() else share
    p = inp["picks"]["disagree_vs_champion_q"][2]
    rng = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        for w in workers:
            ts = []
            for _ in range(runs):
                own = rng.choice(len(names), size=n, p=share)
                hol = rng.choice(len(names), size=int(round(n * p)), p=hshare)
                exp = rng.integers(0, len(names), size=EXPLORE_CALLS)
                calls = np.concatenate([own, hol, exp])
                rng.shuffle(calls)
                lt = np.array([lat[names[i]][rng.integers(0, len(lat[names[i]]))] for i in calls])
                ts.append(makespan(lt, w))
            ts = np.array(ts)
            out[(n, w)] = (float(np.median(ts)), float(np.percentile(ts, 90)),
                           float((ts > CALL_BUDGET_S).mean()))
    return out


def sim_time_measure(inp: dict, runs: int = 200, seed: int = 7, workers: int = 40) -> dict:
    """Wall time of a proposal's measurement in one epoch (300 or 150 calls
    of the proposed entry, plus 112 exploration calls), for the measured
    entries with the slowest median latencies."""
    ents = inp["entries"]
    names = [m for m, e in ents.items() if len(e["latency_ms"]) >= 20]
    lat = {m: np.array(ents[m]["latency_ms"], dtype=np.float64) / 1000 for m in names}
    slow = sorted(names, key=lambda m: -np.percentile(lat[m], 90))[:6]
    rng = np.random.default_rng(seed)
    out = {}
    for m in slow:
        for n in (150, 300):
            ts = []
            for _ in range(runs):
                lt = np.concatenate([lat[m][rng.integers(0, len(lat[m]), size=n)],
                                     np.concatenate([lat[x][rng.integers(0, len(lat[x]), size=1)]
                                                     for x in rng.choice(names, EXPLORE_CALLS)])])
                rng.shuffle(lt)
                ts.append(makespan(lt, workers))
            ts = np.array(ts)
            out[(m, n)] = (float(np.median(ts)), float(np.percentile(ts, 90)),
                           float((ts > CALL_BUDGET_S).mean()), ents[m].get("call_cost_usd"))
    return out


# --- E: one measuring epoch ------------------------------------------------------------------


def _wilson_upper(rate: float, n_eff: float, z: float = 1.645) -> tuple[float, float]:
    if n_eff <= 0:
        return 0.0, 1.0
    z2 = z * z
    den = 1 + z2 / n_eff
    c = (rate + z2 / (2 * n_eff)) / den
    h = z / den * math.sqrt(rate * (1 - rate) / n_eff + z2 / (4 * n_eff * n_eff))
    return max(0.0, c - h), min(1.0, c + h)


def _weighted(cells: dict, targets: dict) -> tuple[float, float]:
    live = {c: v for c, v in cells.items() if v[0] > 0}
    if not live:
        return 0.0, 0.0
    tw = sum(targets.get(c, 0) for c in live)
    rate = sum(targets.get(c, 0) * k / n for c, (n, k) in live.items()) / tw
    neff = tw * tw / sum(targets.get(c, 0) ** 2 / n for c, (n, _k) in live.items() if targets.get(c))
    return rate, neff


DESIGNS_E = {"150x1": (150, 1, 150, False, 9500, False), "300x2": (150, 2, 300, False, 9500, False),
             "300x1": (300, 1, 300, False, 9500, False)}
# One epoch of 300 with the decided retry, and the candidate further changes:
# a 90% answered gate, or one re-measurement epoch when the gate fails.
DESIGNS_RETRY = {"300x1": (300, 1, 300, False, 9500, False),
                 "retry": (300, 1, 300, True, 9500, False),
                 "retry+90%": (300, 1, 300, True, 9000, False),
                 "retry+remeasure": (300, 1, 300, True, 9500, True),
                 "retry, gate 90% on first tries": (300, 1, 300, "first", 9000, False)}


# A provisional entry measured by the field (FAST_CROWN_V4.md 17): every
# hotkey's 44 coverage calls aimed at it, about 290 labels at five operators
# (7 hotkeys) and 880 at fifteen (20 hotkeys); two provisional entries share
# them; each failed call retried once.
DESIGNS_FIELD = {"5 ops, one entry (290)": (290, 1, 290, True, 9500, False),
                 "5 ops, two entries (145)": (145, 1, 145, True, 9500, False),
                 "5 ops, two epochs (290)": (145, 2, 290, True, 9500, False),
                 "15 ops (880)": (880, 1, 880, True, 9500, False)}


def sim_e(inp: dict, runs: int = 2000, seed: int = 8, designs: dict | None = None) -> dict:
    """The proposal gates as coded (menu_state step 5b: answered share 95%,
    a category with 20 graded at 50% or more, the usefulness comparison at
    one-sided 95% Wilson bounds on the mix-weighted rate) for each measured
    entry treated as a proposal: 150 questions in one epoch, 300 over two,
    300 in one. Failures are drawn epoch by epoch from the entry's measured
    failure rates (a bad hour hits the whole epoch); correctness from its
    measured accuracy by category. An existing entry is compared at 200
    pairs of its own measured accuracy."""
    targets = inp["mix_targets_pct"]
    ents = inp["entries"]

    def acc_of(m):
        a = ents[m]["accuracy"]
        tot_n = sum(v[0] for v in a.values())
        tot_k = sum(v[1] for v in a.values())
        base = (tot_k + 1) / (tot_n + 2)
        # shrink thin categories toward the entry's rate (10 pseudo-calls)
        return {c: (a.get(c, [0, 0])[1] + 10 * base) / (a.get(c, [0, 0])[0] + 10) for c in targets}

    usable = [m for m, e in ents.items()
              if e["calls"] >= 60 and sum(v[0] for v in e["accuracy"].values()) >= 40
              and e["call_cost_usd"] is not None]
    accs = {m: acc_of(m) for m in usable}
    # existing entries' lower bounds at 200 pairs of their true rate
    lowers = {}
    for m in usable:
        r = sum(targets[c] * accs[m][c] for c in targets) / sum(targets.values())
        lowers[m] = (_wilson_upper(r, 200)[0], ents[m]["call_cost_usd"], r)
    rng = np.random.default_rng(seed)
    out = {}
    designs = DESIGNS_E if designs is None else designs
    for m in usable:
        e = ents[m]
        # an epoch's failure rate: its own calls, shrunk toward the entry's
        # overall rate by 20 calls (keeps a real bad hour, damps the noise of
        # a dozen calls)
        base = e["failed"] / max(1, e["calls"])
        rates = [(f + 20 * base) / (c + 20) for c, f in e["by_epoch"] if c > 0] or [base]
        true_rate = sum(targets[c] * accs[m][c] for c in targets) / sum(targets.values())
        dominated = any(lo > true_rate + 0.03 and cost <= e["call_cost_usd"]
                        for x, (lo, cost, _r) in lowers.items() if x != m)
        good_cat = any(accs[m][c] >= 0.55 for c in targets if targets[c] >= 15)
        res = {}
        for name, (per, k_ep, total, retry, gate, remeasure) in designs.items():
            refused = collections.Counter()
            for _ in range(runs):
                for attempt in range(2 if remeasure else 1):
                    cells = collections.defaultdict(lambda: [0, 0])
                    answered = 0
                    for _e in range(k_ep):
                        f = rates[rng.integers(0, len(rates))]
                        # a retried call fails again at the same epoch's rate
                        fail = f * f if retry else f
                        for c, k in quotas(per, targets).items():
                            u = rng.random(k)
                            ok = u >= fail
                            a = int(ok.sum())
                            # retry "first": the gate reads first tries only
                            answered += int((u >= f).sum()) if retry == "first" else a
                            cells[c][0] += a
                            cells[c][1] += int((rng.random(a) < accs[m][c]).sum())
                    if answered * 10000 >= gate * total:
                        break
                if answered * 10000 < gate * total:
                    refused["answered"] += 1
                    continue
                if not any(n >= 20 and k * 10000 >= 5000 * n for n, k in cells.values()):
                    refused["category"] += 1
                    continue
                rate, neff = _weighted({c: tuple(v) for c, v in cells.items()}, targets)
                upper = _wilson_upper(rate, neff)[1]
                if any(lo > upper and cost <= e["call_cost_usd"]
                       for x, (lo, cost, _r) in lowers.items() if x != m):
                    refused["useful"] += 1
            res[name] = {k: v / runs for k, v in refused.items()}
            res[name]["any"] = sum(refused.values()) / runs
        out[m] = {"fail": e["failed"] / max(1, e["calls"]), "rate": true_rate,
                  "dominated": dominated, "good_category": good_cat, "res": res,
                  "cost": e["call_cost_usd"]}
    return out


# --- G: zero allowance ---------------------------------------------------------------------

# Labels on a new entry an epoch, pooled (coverage sends a new entry about
# 100 labels in 5 epochs at five operators, in 2 at fifteen: SEAT_MINIMAL_V4.md
# 5.6), and the share of them a comparison of two other operators' heads reads.
NEW_ENTRY_LABELS = {5: (20, 0.66), 15: (50, 0.87)}


def entry_losses(inp: dict, m: str) -> dict:
    """Per label of entry m (weighted, Brier points): what a head without the
    row loses as built (certainly wrong) and as the owner reads it (odds 0,
    never correct), and what a constant row at the entry's rate, the
    champion's fitted row and the median head's row lose."""
    lab = np.array(inp["entries"][m]["labels"], dtype=np.float64)
    y = lab[:, 0] / 10_000
    w = lab[:, 1] / lab[:, 1].mean()
    q = float((w * y).sum() / w.sum())
    oc, om = lab[:, 2] / 10_000, lab[:, 3] / 10_000
    return {"n": len(y), "rate": q, "w": w,
            "certainly_wrong": 100.0 * np.ones_like(y),
            "odds_zero": 100 * y ** 2, "constant": 100 * (q - y) ** 2,
            "fitted": 100 * (oc - y) ** 2, "median": 100 * (om - y) ** 2}


def g_entries(inp: dict) -> dict:
    """Stand-ins for a new entry from the measured entries with at least 60
    labels: strong (highest rate), mid (median), weak (lowest), and
    dominated (the dearest entry a cheaper one beats by 3 points)."""
    ents = inp["entries"]
    ok = [m for m, e in ents.items() if len(e["labels"]) >= 60 and e["call_cost_usd"]]
    rate = {m: entry_losses(inp, m)["rate"] for m in ok}
    by = sorted(ok, key=lambda m: rate[m])
    dom = [m for m in ok if any(rate[x] >= rate[m] + 0.03 and ents[x]["call_cost_usd"] < ents[m]["call_cost_usd"]
                                for x in ok)]
    dominated = max(dom, key=lambda m: ents[m]["call_cost_usd"]) if dom else by[0]
    return {"strong": by[-1], "mid": by[len(by) // 2], "weak": by[0], "dominated": dominated}


def sim_g(inp: dict, runs: int = 300, seed: int = 9, limit: int = 48) -> dict:
    """Epochs until a holder that never adds the new entry's row loses rule A
    to a challenger equal to it elsewhere that adds (a) a constant row at the
    entry's rate or (b) a fitted row (the champion's real row), at five and
    fifteen operators, under the decided constants (floor 300, n0 100), with
    the missing row scored as built (certainly wrong) and as odds 0. Also
    (c): a fitted row against a holder that added a constant row."""
    model = OddsModel(inp, sigma=8.0)
    out = {}
    for kind, m in g_entries(inp).items():
        L = entry_losses(inp, m)
        cases = {"cw/const": L["certainly_wrong"] - L["constant"],
                 "cw/fitted": L["certainly_wrong"] - L["fitted"],
                 "zero/const": L["odds_zero"] - L["constant"],
                 "zero/fitted": L["odds_zero"] - L["fitted"],
                 "const/fitted": L["constant"] - L["fitted"]}
        for ops, (pooled, share) in NEW_ENTRY_LABELS.items():
            labels, fresh, mm = FIELDS[ops]
            k = int(round(pooled * share))
            for case, g in cases.items():
                gw = g * L["w"]
                rng = np.random.default_rng([seed, ops, len(case), int(L["rate"] * 1000)])
                got = []
                for _ in range(runs):
                    cum = [0, 0, 0]
                    first = None
                    for e in range(limit):
                        n, s, q = model.epoch_sums(rng, 1, labels - k, fresh, mm, np.zeros(1))
                        x = np.rint(gw[rng.integers(0, len(gw), size=k)] * 100).astype(np.int64)
                        cum = [cum[0] + n + k, cum[1] + int(s[0]) + int(x.sum()),
                               cum[2] + int(q[0]) + int((x * x).sum())]
                        if bound_cp({"all": tuple(cum)}, MARGIN_A, 300, N0_LOW).passes:
                            first = e + 1
                            break
                    got.append(first if first is not None else 10 ** 6)
                v = np.array(got)
                out[(kind, m, ops, case)] = (float(np.median(v)), float((v <= 1).mean()),
                                             float((v <= 2).mean()), float((v <= limit).mean()),
                                             float(gw.mean()) * k / labels)
    return out


def ignore_threshold(labels: int = 450, share: float = 0.03, epochs: int = 48) -> float:
    """Under odds 0, the accuracy above which a holder ignoring an entry with
    `share` of the labels is beaten within `epochs` by an otherwise equal
    head with a constant row: share x 100 p^2 > margin + the bound's width."""
    n = labels * epochs
    width = width_factor(n, N0_LOW) * 8 / math.sqrt(n)
    return math.sqrt((MARGIN_A + width) / (100 * share))


# --- H: winning by better routing -------------------------------------------------------------


def sim_h_power(inp: dict, runs: int = 1000, seed: int = 10, margins=(0.5, 1.0, 1.5),
                sizes=(500, 1000), edges=(-1.0, 0.0, 0.5, 1.0, 2.0, 4.0), sd=None) -> dict:
    """Path H's picks test: the bound of the challenger's NV over the holder
    passes a POSITIVE margin m. Share passed in one attempt and within a run
    of two, per true edge."""
    model = PicksModel(inp, sd)
    out = {}
    for n in sizes:
        for mu in edges:
            rng = np.random.default_rng([seed, n, int(mu * 10) + 100])
            b = model.batch(rng, n, mu, 2 * runs)
            for m in margins:
                one = two = 0
                for r in range(runs):
                    s1, s2 = model.sums(b, 2 * r), model.sums(b, 2 * r + 1)
                    p1 = bound_cp(s1, m, n, N0_LOW).passes
                    one += p1
                    two += p1 or bound_cp(_add(s1, s2), m, n, N0_LOW).passes
                out[(n, mu, m)] = (one / runs, two / runs)
    return out


def sim_h_false(inp: dict, runs: int = 200, epochs: int = 300, seed: int = 11,
                margins=(0.5, 1.0), sizes=(500, 1000)) -> dict:
    """A challenger with no picks edge tries path H every epoch under the
    answered-run rule, serving what helps it: share with a false pass by
    epoch 48 / 300."""
    model = PicksModel(inp)
    out = {}
    for n in sizes:
        rng = np.random.default_rng([seed, n])
        for m in margins:
            res = [0, 0]
            for _ in range(runs):
                b = model.batch(rng, n, 0.0, epochs)
                ep_sums = [model.sums(b, e) for e in range(epochs)]
                _tally(res, _attacker(ep_sums, "run", n, N0_LOW, margin=m), 48)
            out[(n, m)] = (res[0] / runs, res[1] / runs)
    return out


def sim_h_odds(inp: dict, runs: int = 300, seed: int = 12, deltas=(0.25, 0.5, 1.0),
               edges=(0.0, -0.25, -0.5, -1.0, -2.0), limit: int = 48) -> dict:
    """Path H's odds guard: the bound of the challenger's Brier gain over the
    holder passes -delta (not worse by more than delta). Per true odds edge:
    median epochs to pass and the share passed within 1 / 48 epochs."""
    model = OddsModel(inp, sigma=8.0)
    out = {}
    for ops, (labels, fresh, mm) in FIELDS.items():
        for d in edges:
            rng = np.random.default_rng([seed, ops, int(-d * 100)])
            got = {dl: [] for dl in deltas}
            for _ in range(runs):
                cum = [0, 0, 0]
                first = {dl: None for dl in deltas}
                for e in range(limit):
                    n, s, q = model.epoch_sums(rng, 1, labels, fresh, mm, np.array([d]), sym=False)
                    cum = [cum[0] + n, cum[1] + int(s[0]), cum[2] + int(q[0])]
                    for dl in deltas:
                        if first[dl] is None and bound_cp({"all": tuple(cum)}, -dl, 300, N0_LOW).passes:
                            first[dl] = e + 1
                    if all(v is not None for v in first.values()):
                        break
                for dl, v in first.items():
                    got[dl].append(v if v is not None else 10 ** 6)
            for dl, v in got.items():
                v = np.array(v)
                out[(ops, d, dl)] = (float(np.median(v)), float((v <= 1).mean()), float((v <= limit).mean()))
    return out


def h_policies(inp: dict) -> dict:
    """Degenerate heads' NV per question at the v4 mix (points), against the
    champion's measured NV on its own picks: every constant policy (one
    entry always: its rate by category from exploration labels, its pick
    cost from the seed table) and the best switch by category (the best
    entry per category in hindsight, which favours the switch)."""
    from fugal_subnet import menu_state as ms
    seed = ms.load_seed()
    on = {e.id for e in seed.entries}
    targets = inp["mix_targets_pct"]
    tot = sum(targets.values())
    cv = inp["champion_value_by_category"]
    champ = sum(targets[c] / tot * cv[c][1] / cv[c][0] for c in targets)
    value = {}
    for m, e in inp["entries"].items():
        a = e["accuracy"]
        if m not in on or sum(v[0] for v in a.values()) < 40:
            continue
        base = (sum(v[1] for v in a.values()) + 1) / (sum(v[0] for v in a.values()) + 2)
        value[m] = {c: 100 * (a.get(c, [0, 0])[1] + 10 * base) / (a.get(c, [0, 0])[0] + 10)
                    - 100 * float(ms.pick_cost(seed, m, c)) for c in targets}
    const = {m: sum(targets[c] / tot * v[c] for c in targets) for m, v in value.items()}
    switch = sum(targets[c] / tot * max(v[c] for v in value.values()) for c in targets)
    switch_by = {c: max(value, key=lambda m: value[m][c]) for c in targets}
    # a new entry with no graded cells is priced at its call cost; how far
    # under its pick cost that is, in points, for the seed's entries
    under = sorted(100 * (float(ms.entry_pick_cost(e)) - float(ms.entry_cost(e))) for e in seed.entries)
    ratio = sorted(float(ms.entry_pick_cost(e) / ms.entry_cost(e)) for e in seed.entries
                   if ms.entry_cost(e) > 0)
    return {"champion": champ, "best_constant": max(const.items(), key=lambda kv: kv[1]),
            "constants": const, "switch": switch, "switch_by": switch_by,
            "new_entry_underpricing_points_q": [round(float(x), 2) for x in np.percentile(under, [50, 75, 90, 100])],
            "pick_over_call_q": [round(float(x), 2) for x in np.percentile(ratio, [10, 50, 90])]}


def h_nudge(inp: dict, runs: int = 2000, seed: int = 13, n: int = 1000, ks=(10, 30, 100, 300),
            margin: float = 1.0) -> dict:
    """A near-copy of the holder whose picks differ on k of n questions: its
    items are 0 elsewhere. Share passing path H at `margin` when the nudged
    items have no edge (resampled real items) and the smallest k at which a
    copy that knew every nudged outcome (+80 points an item) would pass."""
    model = PicksModel(inp)
    pool = np.concatenate([v for v in model.items.values()]) * model.scale
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        ok = 0
        for _ in range(runs):
            x = np.rint(pool[rng.integers(0, len(pool), size=k)]).astype(np.int64)
            ok += bound_cp({"all": (n, int(x.sum()), int((x * x).sum()))}, margin, 500, N0_LOW).passes
        out[k] = ok / runs
    informed = next(k for k in range(1, n) if bound_cp(
        {"all": (n, 8000 * k, 8000 * 8000 * k)}, margin, 500, N0_LOW).passes)
    return {"random": out, "informed_k": informed}


# --- K: a value floor against R's routing --------------------------------------------------


def off_policy_hits(inp: dict) -> tuple[float, float]:
    """The share of exploration labels whose entry is the holder's pick on
    that question, and R's (the category-best entry): the labels an
    off-policy estimate of each side's value can use. Label shares are v3's
    pooled shares; the holder's pick shares are the champion's."""
    ents = inp["entries"]
    lab = {m: sum(v[0] for v in e["accuracy"].values()) for m, e in ents.items()}
    tot = sum(lab.values())
    hold = sum(lab[m] / tot * ents[m]["holder_pick_share"] for m in ents)
    pol = h_policies(inp)
    t = inp["mix_targets_pct"]
    r = sum(t[c] / sum(t.values()) * lab.get(pol["switch_by"][c], 0) / tot for c in t)
    return hold, r


def sim_k(inp: dict, runs: int = 300, seed: int = 14, epochs: int = 300) -> dict:
    """Epochs until the floor unseats a holder whose routing is worse than
    R's by `gap` points (R's value minus the holder's, the bound over a
    positive margin), and the share of equal holders ever unseated in
    `epochs` epochs. Evidence:

    paid N/M  the holder's TD answers N challenge questions, its picks and
              R's where they differ, every M epochs (paired items as a
              challenge's, SD 20); a missed check counts as failed;
    off-policy  exploration labels that happen to be the holder's or R's
              pick (measured hit rates), values unpaired (SD 33 a label).
    48-epoch window, n0 100, floor 200 items."""
    model = PicksModel(inp)
    hold, r = off_policy_hits(inp)
    out = {}
    designs = {"paid 200/5": ("paid", 200, 5), "paid 500/5": ("paid", 500, 5),
               "off-policy, 5 ops": ("off", int(round(700 * hold)), int(round(700 * r))),
               "off-policy, 15 ops": ("off", int(round(2200 * hold)), int(round(2200 * r)))}
    for name, (kind, a, b) in designs.items():
        for margin in (1.0, 2.0):
            for gap in (0.0, 2.0, 3.0, 5.0):
                rng = np.random.default_rng([seed, int(gap * 10), int(margin * 10), a])
                firsts = []
                for _ in range(runs):
                    hist, win, first = [], {}, None
                    for e in range(epochs):
                        h = None
                        if kind == "paid":
                            if e % b == 0:
                                h = model.sums(model.batch(rng, a, gap, 1), 0)
                                h.pop("_e2")
                        else:
                            # the difference of two unpaired means, one item per
                            # matched label pair, its spread set by both sides' counts
                            kk = min(a, b)
                            sd = 33.0 * math.sqrt(kk * (1 / a + 1 / b))
                            x = np.rint(rng.normal(gap, sd, size=kk) * 100).astype(np.int64)
                            h = {"all": (kk, int(x.sum()), int((x * x).sum()))}
                        hist.append(h)
                        if h is not None:
                            win = _add(win, h)
                        if len(hist) > 48 and hist[-49] is not None:
                            win = _add(win, hist[-49], -1)
                        if win and bound_cp(win, margin, 200, N0_LOW).passes:
                            first = e + 1
                            break
                    firsts.append(first if first is not None else 10 ** 6)
                v = np.array(firsts)
                out[(name, margin, gap)] = (float(np.median(v)), float((v <= 48).mean()),
                                            float((v <= epochs).mean()))
    return out


# --- printing -------------------------------------------------------------------------------


def _pct(x):
    return "0" if x == 0 else ("<0.1%" if x < 0.001 else f"{100 * x:.1f}%")


def print_inputs(inp):
    pk, od = inp["picks"], inp["odds"]
    print("MEASURED INPUTS (%d published v3 epochs)" % len(inp["source"]["reveals"]))
    print("  disagreement vs the champion, q10/25/50/75/90:", pk["disagree_vs_champion_q"],
          "| heads within 2 NV points:", pk["disagree_close_heads_median"])
    print("  disagreement, every pair of heads:", pk["disagree_all_pairs_q"])
    print("  NV item SD paired (stratified):", pk["paired_sd_q"], "| close heads",
          pk["paired_sd_close_heads_median"], "| unpaired:", pk["unpaired_sd_q"])
    print("  NV mean vs champion:", pk["nv_mean_q"])
    print("  epoch spread of a pair's mean beyond sampling (tau, points): %.2f over %d pairs"
          % (pk["epoch_tau_points"], pk["epoch_tau_pairs"]))
    print("  scored calls failed per head-epoch q50/90/99:", pk["scored_fail_share_q"])
    print("  pick cost a question: holder $%.4f, challenger median $%.4f; challenge $%.4f/question"
          % (pk["pick_cost_usd_holder_mean"], pk["pick_cost_usd_challenger_median"],
             cost_per_question(inp)))
    print("  odds gain SD per label vs champion q:", od["gain_sd_q"], "kurtosis", od["kurtosis_median"])
    print("  v3 labels pooled/epoch q10/50/90:", od["v3_pool_labels_per_epoch_q"], "operators",
          od["v3_operators_per_epoch_q"], "hotkeys", od["v3_hotkeys_per_epoch_q"],
          "share left after two operators", od["v3_share_left_after_two_operators_q"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("what", choices=["measure", "inputs", "a", "b", "c", "e", "time", "retry",
                                    "g", "h", "j", "k", "v", "all"])
    p.add_argument("--reveals", default=os.path.expanduser(
        "~/fugal-deploy/router-study-2026-11-18/reveals"))
    p.add_argument("--out", default=str(INPUTS))
    p.add_argument("--quick", action="store_true", help="small runs, for a smoke check")
    a = p.parse_args()
    if a.what == "measure":
        doc = measure(a.reveals)
        Path(a.out).write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
        print(f"wrote {a.out}")
        print_inputs(doc)
        return 0
    inp = load_inputs(Path(a.out))
    q = a.quick
    t = time.time()
    if a.what in ("inputs", "all"):
        print_inputs(inp)
    if a.what in ("a", "all"):
        pk = inp["picks"]
        res = sim_a(inp, runs=200 if q else 2000, sds=(None, pk["paired_sd_q"][3]))
        cq = cost_per_question(inp)
        for key in sorted({k[:3] for k in res}):
            sd, tuning, tau = key
            if sd != round(pk["paired_sd_q"][2], 1) and (tuning, tau) != ("low", 0.0):
                continue
            print(f"\nA. rule B, one attempt judged alone: item SD {sd}, n0 = "
                  f"{'n' if tuning == 'min' else '100'}, epoch shock {tau:.1f}: pass rate by true NV")
            print("   n     $   " + "".join(f"{m:>8g}" for m in EDGES_B))
            for n in SIZES:
                print(f"{n:5d} {n * cq:6.0f}  " + "".join(
                    f"{_pct(res[(sd, tuning, tau, n, m)]):>8}" for m in EDGES_B))
        print(f"  [{time.time() - t:.0f}s]")
    if a.what in ("b", "all"):
        t = time.time()
        res = sim_b(inp, runs=40 if q else 200, epochs=100 if q else 300)
        print("\nB. a challenger not better, trying every epoch: share with a false pass by"
              " epoch 48 / by the end (floor 500, n0 100)")
        for (n, mu, tau, t0, k), (w, e) in sorted(res.items()):
            print(f"  n {n:5d}  NV {mu:+.2f}  shock {tau:.1f}  guard {t0:.1f}  {k:9s} "
                  f"{_pct(w):>7} / {_pct(e):>7}   attempt ${n * cost_per_question(inp):.0f}")
        for sd in (None, 24.0):
            hb = honest_b(inp, runs=60 if q else 400, sd=sd)
            print(f"\nB'. an honest challenger, one run (floor 500, n0 100), item SD "
                  f"{sd or inp['picks']['paired_sd_q'][2]:.0f}: passed after 1 / 2 / 5 attempts,"
                  " median attempts")
            for (n, mu, t0), (p1, p2, p5, med) in sorted(hb.items()):
                print(f"  n {n:5d}  NV {mu:+.1f}  guard {t0:.1f}  {_pct(p1):>7} {_pct(p2):>7} "
                      f"{_pct(p5):>7}  {med if med < 1e6 else '>20'}")
        print("\nB''. one ticket (a fresh attempt, a restarted run or one more TD run on the same"
              " questions): chance it passes, $ an attacker spends per expected false pass")
        for (sd, n, mu), (pp, cost) in sorted(tickets(inp).items()):
            print(f"  SD {sd:.0f}  n {n:5d}  NV {mu:+.2f}  {pp:.2e}  ${cost:,.0f}")
        print(f"  [{time.time() - t:.0f}s]")
    if a.what in ("v", "all"):
        t = time.time()
        res = sim_void(inp, runs=40 if q else 200, epochs=100 if q else 300)
        print("\nV. the holder-side failure cap (10%): a challenger not better, trying every"
              " epoch with perfect knowledge of each epoch's items: share with a false pass by"
              " epoch 48 / by the end (floor 500, n0 100)")
        for (n, mu, k), (w, e) in sorted(res.items()):
            print(f"  n {n:5d}  NV {mu:+.2f}  {k:12s} {_pct(w):>7} / {_pct(e):>7}")
        print(f"  [{time.time() - t:.0f}s]")
    if a.what in ("c", "all"):
        t = time.time()
        res = sim_c_false(inp, fields=3 if q else 20, epochs=100 if q else 300)
        print("\nC. 30 no-edge heads over 300 epochs: fields (of N) where any head passed rule A;"
              " the largest bound reached (Brier points)")
        for k, (h, n, best) in sorted(res.items()):
            print(f"  {k[0]:2d} operators  floor {k[1]:5d}  n0 {'floor' if k[2] == 'min' else '100':8s}"
                  f" {h} of {n}   best {best:+.3f}")
        sp = sim_c_speed(inp, runs=40 if q else 300)
        print("\nC'. epochs to pass rule A: median, share in 1 epoch, in 2")
        for k, (med, p1, p2) in sorted(sp.items()):
            print(f"  {k[0]:2d} operators  d {k[1]:.1f}  floor {k[2]:5d}  n0 "
                  f"{'floor' if k[3] == 'min' else '100':8s} {med:>6g} {_pct(p1):>7} {_pct(p2):>7}")
        print(f"  [{time.time() - t:.0f}s]")
    if a.what in ("time", "all"):
        t = time.time()
        res = sim_time(inp, runs=40 if q else 300)
        print(f"\nTIME. a challenge epoch (n questions, the holder's differing picks, 112 exploration):"
              f" median / p90 seconds, share past {CALL_BUDGET_S} s")
        for (n, w), (m, p9, over) in sorted(res.items()):
            print(f"  n {n:5d}  {w:3d} in flight   {m:7.0f} {p9:7.0f}  {_pct(over)}")
        res = sim_time_measure(inp, runs=40 if q else 300)
        print("\nTIME'. a measurement in one epoch at 40 in flight, slowest entries: median / p90 s,"
              " share past budget, $ for the calls")
        for (m, n), (md, p9, over, cost) in sorted(res.items()):
            print(f"  {m:42s} {n:4d}  {md:7.0f} {p9:7.0f}  {_pct(over):>6}  "
                  f"${(cost or 0) * n:.2f}")
        print(f"  [{time.time() - t:.0f}s]")
    if a.what in ("e", "all"):
        t = time.time()
        res = sim_e(inp, runs=200 if q else 2000)
        groups = {"good, not dominated": lambda r: r["good_category"] and not r["dominated"],
                  "dominated (a cheaper entry is 3+ points better)": lambda r: r["dominated"],
                  "no category at 55%": lambda r: not r["good_category"],
                  "flaky (5%+ of calls fail)": lambda r: r["fail"] >= 0.05}
        print("\nE. proposal gates: share refused, by design (150 in 1 epoch / 300 in 2 / 300 in 1)")
        for g, f in groups.items():
            ms_ = [m for m, r in res.items() if f(r)]
            if not ms_:
                continue
            row = [np.mean([res[m]["res"][d]["any"] for m in ms_]) for d in ("150x1", "300x2", "300x1")]
            print(f"  {g:50s} n={len(ms_):2d}  " + "  ".join(_pct(x) for x in row))
            for why in ("answered", "category", "useful"):
                row = [np.mean([res[m]["res"][d].get(why, 0) for m in ms_])
                       for d in ("150x1", "300x2", "300x1")]
                print(f"      of which {why:10s}{'':30s}  " + "  ".join(_pct(x) for x in row))
        print("  per entry (fail rate, mix-weighted accuracy): refused 150x1 / 300x2 / 300x1")
        for m, r in sorted(res.items(), key=lambda kv: kv[1]["rate"]):
            print(f"    {m:42s} {r['fail']:.3f} {r['rate']:.2f} {'D' if r['dominated'] else ' '}"
                  f"{'G' if r['good_category'] else ' '}  " + "  ".join(
                      _pct(r["res"][d]["any"]) for d in ("150x1", "300x2", "300x1")))
        print(f"  [{time.time() - t:.0f}s]")
    if a.what in ("retry", "all"):
        t = time.time()
        res = sim_e(inp, runs=200 if q else 2000, designs=DESIGNS_RETRY)
        print("\nE'. one measuring epoch of 300, with a retry of each failed call: share refused"
              " (of which the answered gate)")
        groups = {"good, not dominated": lambda r: r["good_category"] and not r["dominated"],
                  "dominated": lambda r: r["dominated"], "flaky (5%+ fail)": lambda r: r["fail"] >= 0.05}
        for g, f in groups.items():
            ms_ = [m for m, r in res.items() if f(r)]
            print(f"  {g:22s} n={len(ms_):2d}  " + "  ".join(
                f"{d}: {_pct(np.mean([res[m]['res'][d]['any'] for m in ms_]))}"
                f" ({_pct(np.mean([res[m]['res'][d].get('answered', 0) for m in ms_]))})"
                for d in DESIGNS_RETRY))
        print(f"  [{time.time() - t:.0f}s]")
    if a.what in ("g", "all"):
        t = time.time()
        print("\nG. zero allowance; stand-in entries:", g_entries(inp))
        res = sim_g(inp, runs=40 if q else 300)
        print("  (entry, operators, holder row / challenger row): median epochs to pass A, share in 1 /"
              " 2 / 48 epochs, mean gain a comparison label (Brier points)")
        for (kind, m, ops, case), (med, p1, p2, p48, gain) in sorted(res.items()):
            print(f"  {kind:9s} {ops:2d} ops {case:12s} {med if med < 1e6 else '>48':>5} "
                  f"{_pct(p1):>7} {_pct(p2):>7} {_pct(p48):>7}  {gain:+.3f}")
        for share in (0.01, 0.03, 0.05):
            print(f"  odds 0: an entry is beaten into the holder within 48 epochs (5 operators) above"
                  f" accuracy {ignore_threshold(450, share):.2f} at {share:.0%} of the labels")
        print(f"  [{time.time() - t:.0f}s]")
    if a.what in ("h", "all"):
        t = time.time()
        pol = h_policies(inp)
        print("\nH. degenerate policies, NV a question at the v4 mix (points): champion %.1f; best"
              " constant %s %.1f; best switch by category %.1f" % (
                  pol["champion"], pol["best_constant"][0], pol["best_constant"][1], pol["switch"]))
        print("  switch picks:", pol["switch_by"])
        print("  new entry priced at its call cost: pick - call, points, q50/75/90/max",
              pol["new_entry_underpricing_points_q"], "pick/call q10/50/90", pol["pick_over_call_q"])
        res = sim_h_power(inp, runs=100 if q else 1000)
        print("  path H picks test: passed in one attempt / within two, by true edge")
        for (n, mu, m), (p1, p2) in sorted(res.items()):
            print(f"  n {n:5d}  margin +{m:.1f}  NV {mu:+.1f}  {_pct(p1):>7} {_pct(p2):>7}")
        res = sim_h_false(inp, runs=20 if q else 200, epochs=100 if q else 300)
        print("  no picks edge, run rule, every epoch: false pass by epoch 48 / 300")
        for (n, m), (w, e) in sorted(res.items()):
            print(f"  n {n:5d}  margin +{m:.1f}  {_pct(w):>7} {_pct(e):>7}")
        res = sim_h_odds(inp, runs=40 if q else 300)
        print("  odds guard (not worse by delta): median epochs, share in 1 / 48 epochs")
        for (ops, d, dl), (med, p1, p48) in sorted(res.items()):
            print(f"  {ops:2d} ops  odds edge {d:+.2f}  delta {dl:.2f}  "
                  f"{med if med < 1e6 else '>48':>5} {_pct(p1):>7} {_pct(p48):>7}")
        nd = h_nudge(inp, runs=200 if q else 2000)
        print("  near-copy nudged on k of 1,000 questions, margin +1: passes", nd["random"],
              "; knowing every nudged outcome it needs k =", nd["informed_k"])
        print(f"  [{time.time() - t:.0f}s]")
    if a.what in ("j", "all"):
        t = time.time()
        res = sim_e(inp, runs=200 if q else 2000, designs=DESIGNS_FIELD)
        print("\nJ. a provisional entry measured by the field's coverage calls, retry, gates as"
              " coded: share refused (of which the answered gate)")
        groups = {"good, not dominated": lambda r: r["good_category"] and not r["dominated"],
                  "dominated": lambda r: r["dominated"], "flaky (5%+ fail)": lambda r: r["fail"] >= 0.05}
        for g, f in groups.items():
            ms_ = [m for m, r in res.items() if f(r)]
            print(f"  {g:22s} n={len(ms_):2d}  " + "  ".join(
                f"{d}: {_pct(np.mean([res[m]['res'][d]['any'] for m in ms_]))}"
                f" ({_pct(np.mean([res[m]['res'][d].get('answered', 0) for m in ms_]))})"
                for d in DESIGNS_FIELD))
        print(f"  [{time.time() - t:.0f}s]")
    if a.what in ("k", "all"):
        t = time.time()
        hold, r = off_policy_hits(inp)
        print(f"\nK. off-policy hits: {hold:.4f} of labels are the holder's pick, {r:.4f} R's")
        res = sim_k(inp, runs=40 if q else 200, epochs=100 if q else 150)
        print("  (evidence, margin, R minus holder): median epochs to unseat, share by 48 / by the end")
        for (name, m, gap), (med, p48, pend) in sorted(res.items()):
            print(f"  {name:20s} m {m:.0f}  gap {gap:.0f}  {med if med < 1e6 else '>end':>6}"
                  f" {_pct(p48):>7} {_pct(pend):>7}")
        print(f"  [{time.time() - t:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
