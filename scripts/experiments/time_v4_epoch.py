"""Time one protocol v4 epoch step at the subnet's full size (no network, no
spend): a worst-case-shaped world built deterministically, one whole
`v2_epoch.run_v2_epoch` with the epoch's memo, and each stage of it on its
own, with peak memory.

    uv run python scripts/experiments/time_v4_epoch.py                 # 256 UIDs, 48 epochs
    uv run python scripts/experiments/time_v4_epoch.py --uids 8 --epochs 6 --dim 16 --successors 2 --returning 2

THE WORLD (`EpochWorld`), at epoch NOW, built from keyed draws (`--seed`):

- a pool of `--pool` questions over the pinned v4 mix's sets, of which
  `--reserve` are unpublished as of the activation and are spent by a fresh
  draw of F = ceil(R / 1200) an epoch (memo_v4); the published set is folded
  through NOW - 1, every other question published long before the window
  except a tenth first published inside it;
- `--epochs` - 1 window epochs before NOW of labels (labels_v4.LabelPairs) at
  the v4 exploration sizes: each UID's hotkey (one coldkey each) calls 112
  private questions an epoch (F fresh, answered by every hotkey, the rest
  published two epochs back), pooled per (question, entry) cell with its
  voters, `--entries` menu entries with their exploration weights, the last
  four entries' labels provisional (read by no comparison, FAST_CROWN_V4.md
  18), a missing row elsewhere at odds 0;
- 3 x `--uids` heads the seat keeps (seat_v4.SeatStateV4 at NOW - 1, the
  worst-case reveal's shape): per UID its served head (real arrays of
  `--entries` rows x `--dim`), and two more heads of its operator seen in the
  two epochs before (standing, not served now). Of the served heads,
  `--successors` are first seen now and succeed a kept predecessor of the
  same hotkey (its rows byte for byte plus one off the menu), and
  `--returning` come back after twenty missed epochs (late odds); the rest
  were served through the window. The holder is the earliest served head,
  four kept heads are the history;
- the heads' odds on the window: every served head on every epoch it was
  served (`--coverage realistic`: a kept head that is not served now has
  odds only on the epoch it was verified in) or every kept head on every
  epoch (`--coverage full`: the memorisation check's worst case; heads that
  are not served share their arrays, which changes nothing they compute);
  each head is a rung of a ladder in rule C's precedence order (later heads
  0.02 Brier points better per rung), so a qualifier beats every earlier
  contender but the last few and rule C tests nearly every one of them (the
  adversarial worst case; `--no-ladder` makes every head equally good, an
  honest field, where no head beats the holder);
- challenge records for every epoch (every served head but the holder
  challenged every epoch, the kept heads in the epoch they were seen) with
  zero items, so rule B passes and nobody routes worse; this epoch every
  served head but the holder serves a challenge proof of `--challenge`
  (1,000) questions, its own picks and the holder's where they differ, which
  the validator recomputes from both heads (the holder routes question by
  question, so most picks differ).

Every input of the epoch is built directly, so the timing is of the
validator's work, not of the world's construction.

STAGES timed on their own (each on the inputs the epoch hands it):
pool and current odds (v2_epoch._pool_and_odds), picks (_picks_epoch), the
seat's observe (with seat_serving and row digests), adding the epoch with its
late odds (_add_v4_epoch), memorisation evidence (_memo_verdicts: label
eligibility, R and the split for every kept head), the seat's decide (rules
A, B and C; every comparison counted), the standing over R published for
every serving head (labels_v4.standing_for_reveal: label eligibility and
odds), building the reveal (_v4_result), serialising it (indented JSON, as
commit_reveal writes it), and saving and loading the window
(v2_epoch.save_window / load_window). Then the whole run_v2_epoch, whose
reveal must equal the one the stages built. `--td` adds what one TD spends
routing its epoch (head_odds.picks_and_odds_v4 on its scored and private
questions over the whole menu).

PEAK MEMORY is the process's resident set, sampled every 20 ms during each
stage (and the high-water mark at the end).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from fugal_subnet.attacks import memo_seat_world as msw  # noqa: E402

NOW = 1000
BPE = msw.BPE


def _block(commit_epoch: int) -> int:
    return msw._block(commit_epoch)


@dataclasses.dataclass(frozen=True)
class Shape:
    uids: int = 256
    epochs: int = 48
    entries: int = 64
    dim: int = 1024
    pool: int = 48_320
    reserve: int = 7_000
    successors: int = 8
    returning: int = 8
    coverage: str = "realistic"
    ladder: bool = True
    served_last: bool = False
    challenge: int = 1000
    seed: int = 0


def _hex(*parts) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def _entry(i: int) -> str:
    return f"lab-{i:02d}/a-model-name-{i:03d}@high"


# --- memory ---------------------------------------------------------------------------------


def rss_mib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Sampler:
    """The highest resident set seen while a stage runs (sampled every 20 ms)."""

    def __enter__(self):
        self.peak = rss_mib()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        self.t0 = time.perf_counter()
        return self

    def _run(self):
        while not self._stop.wait(0.02):
            self.peak = max(self.peak, rss_mib())

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, rss_mib())
        return False


# --- the world ------------------------------------------------------------------------------


class EpochWorld:
    """Every input of epoch NOW's run_v2_epoch (module docstring)."""

    def __init__(self, shape: Shape):
        from fugal_subnet import config, copy_rule, epoch_plan, labels_v4, memo_v4, seat_v4
        from fugal_subnet.champion import EpochPairs, Window
        from fugal_subnet.picks_v4 import PicksEpoch

        self.shape = s = shape
        self._hidden: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(s.seed)
        mix = epoch_plan.mix_v4()
        self.sets = sets = sorted(mix.bench_category)
        self.bench_category = dict(mix.bench_category)
        self.targets = dict(mix.targets)
        self.entries = entries = [_entry(i) for i in range(s.entries)]
        M, U = s.entries, s.uids
        first_window = NOW - s.epochs + 1

        # --- the pool and the published set
        P = s.pool
        q_set = np.arange(P) % len(sets)
        self.qids = qids = [f"{sets[b]}_{i:06d}" for i, b in enumerate(q_set.tolist())]
        self.q_set = q_set
        order = rng.permutation(P)
        reserve = order[:s.reserve]
        published = order[s.reserve:]
        first = np.full(P, -1, dtype=np.int64)                  # -1: unpublished
        first[published] = NOW - 300
        inside = published[: len(published) // 10]
        first[inside] = rng.integers(first_window - 10, NOW - 1, len(inside))
        f = memo_v4.fresh_count(s.reserve)
        self.fresh_per_epoch = f
        cursor = 0
        fresh_of: dict[int, np.ndarray] = {}
        for e in range(first_window, NOW):
            fresh_of[e] = reserve[cursor:cursor + f]
            first[fresh_of[e]] = e
            cursor += f
        self.unspent_reserve = reserve[cursor:]
        self.first = first
        self.published = memo_v4.Published(
            {qids[i]: int(first[i]) for i in np.flatnonzero(first >= 0).tolist()}, NOW - 1)
        self.pool = [{"question_id": qids[i], "benchmark": sets[q_set[i]], "prompt": "x"}
                     for i in range(P)]

        # --- the menu, its weights and rates
        self.menu = msw.menu(entries=tuple(entries), epoch=NOW)
        self.weights = {e: int(600_000 + (i * 7919) % 1_800_000) for i, e in enumerate(entries)}
        w_arr = np.array([self.weights[e] for e in entries], dtype=np.int64)
        rates = rng.uniform(0.2, 0.85, (M, len(sets)))
        allowance = set(range(max(0, M - 4), M))
        self.coldkeys = [f"ck-{k:03d}" for k in range(U)]
        cat_of_set = [self.bench_category[x] for x in sets]

        # --- the window's labels
        private = config.EXPLORE_PRIVATE_SIZE_V4 - f
        pairs: dict[int, object] = {}
        ys: dict[int, np.ndarray] = {}
        for e in range(first_window, NOW):
            elig = np.flatnonzero((first >= 0) & (first <= e - 2))
            pq = elig[rng.integers(0, len(elig), (U, private))].ravel()
            pe = rng.integers(0, M, U * private)
            pv = np.repeat(np.arange(U), private)
            fq = np.repeat(fresh_of[e], U)
            fe = rng.integers(0, M, len(fq))
            fv = np.tile(np.arange(U), len(fresh_of[e]))
            q = np.concatenate([pq, fq])
            ent = np.concatenate([pe, fe])
            vot = np.concatenate([pv, fv])
            key = q * M + ent
            o = np.lexsort((vot, key))
            key, vot = key[o], vot[o]
            uniq, start = np.unique(key, return_index=True)
            bounds = np.append(start, len(key)).tolist()
            vlist = vot.tolist()
            voters = [tuple(self.coldkeys[v] for v in sorted(set(vlist[a:b])))
                      for a, b in zip(bounds[:-1], bounds[1:])]
            lq, le = uniq // M, uniq % M
            n = len(uniq)
            bset = q_set[lq]
            y = (rng.random(n) < rates[le, bset]).astype(np.int32) * 10_000
            # the last four entries joined provisional: their labels score no
            # head (labels_v4.PROVISIONAL_LABEL), the rest a missing row at odds 0
            missing = np.where(np.isin(le, list(allowance)), labels_v4.PROVISIONAL_LABEL,
                               -1).astype(np.int16)
            base = EpochPairs(
                question_ids=tuple(qids[i] for i in lq.tolist()),
                models=tuple(entries[i] for i in le.tolist()),
                benchmarks=tuple(sets[b] for b in bset.tolist()),
                categories=tuple(cat_of_set[b] for b in bset.tolist()),
                y=y, prompt_tokens=np.full(n, 200, np.int32),
                completion_tokens=np.full(n, 900, np.int32), weight=w_arr[le])
            pairs[e] = labels_v4.label_pairs(base, voters, missing)
            ys[e] = y
        self.pairs = pairs

        # --- the heads
        # every commit at or before the floor ties there (salted order); with
        # `served_last` the floor is earlier and the kept heads' commits come
        # first, so every served head is a candidate tested against all of them
        floor_block = _block(NOW - 100) if s.served_last else _block(NOW - 60)
        self.floor_block, self.salt = floor_block, "0x" + "5a" * 32
        heads: list[dict] = []           # one record per head of the world
        n_succ = min(s.successors, U)
        n_back = min(s.returning, U - n_succ)
        for k in range(U):
            role = "succ" if k < n_succ else "back" if k < n_succ + n_back else "stay"
            W = (rng.standard_normal((M, s.dim)) * 1e-6).astype(np.float32)
            if role == "stay" and k == U - 1:
                # the holder (below) routes question by question, so a
                # challenger's picks differ from its picks on most questions
                # (holder-side calls; FAST_CROWN_V4.md 1 measured 87%)
                W = (rng.standard_normal((M, s.dim)) * 0.3).astype(np.float32)
            b = np.zeros(M, np.float32)
            models = list(entries)
            served = dict(name=f"served-{k}", uid=k, hotkey=f"hk-{k:03d}", owner=self.coldkeys[k],
                          role=role, W=W, b=b, models=models, block=_block(NOW - 80) + k,
                          served=True, seen=NOW - 1, kept=role == "stay")
            if role == "succ":
                pred = dict(served, name=f"pred-{k}", served=False, kept=True, role="pred")
                Ws = np.vstack([W, np.ones((1, s.dim), np.float32) * 1e-6])
                bs = np.append(b, np.float32(0)).astype(np.float32)
                served = dict(served, W=Ws, b=bs, models=models + ["lab-zz/off-menu@high"],
                              block=_block(NOW), kept=False)
                heads.append(pred)
            if role == "back":
                served = dict(served, seen=NOW - 21, kept=False)
            heads.append(served)
            for tag, seen in (("a", NOW - 1), ("b", NOW - 2)):
                heads.append(dict(name=f"kept-{k}-{tag}", uid=None, hotkey=f"hk-{k:03d}-{tag}",
                                  owner=self.coldkeys[k], role="kept", W=None, b=None,
                                  models=models, block=_block(NOW - 90 if s.served_last else NOW - 70) + k, served=False,
                                  seen=seen, kept=True))
        for h in heads:
            h["hash"] = (_hex(h["W"].tobytes(), h["b"].tobytes(), "|".join(h["models"]))
                         if h["W"] is not None else _hex("kept-head", h["name"]))
            h["since"] = -(-h["block"] // BPE)
            h["rows"] = ({e: d for e, d in seat_v4.row_digests(h["W"], h["b"], h["models"]).items()
                          if e in set(entries)} if h["W"] is not None else {})
        by_name = {h["name"]: h for h in heads}
        for k in range(n_succ):
            by_name[f"served-{k}"]["origin"] = by_name[f"pred-{k}"]["hash"]
        self.heads = heads

        def state_of(h, seen=None):
            origin = h.get("origin", h["hash"])
            ob = by_name[f"pred-{h['uid']}"]["block"] if h["role"] == "succ" else h["block"]
            return seat_v4.HeadState(
                head=h["hash"], owner=h["owner"], hotkey=h["hotkey"], block=h["block"],
                since=h["since"], seen=h["seen"] if seen is None else seen, origin=origin,
                origin_block=ob, origin_owner=h["owner"], rows=h["rows"])

        # rule C's ladder: precedence over every head the seat will keep
        ranked = sorted((h for h in heads if h["role"] != "pred"),
                        key=lambda h: seat_v4.precedence(state_of(h), floor_block=floor_block,
                                                         salt=self.salt))
        # the holder (the served head of the last UID) is the lowest rung, so
        # every other head beats it and rule C reads the ladder above it
        holder = next(h for h in ranked if h["role"] == "stay" and h["uid"] == U - 1)
        rungs = [holder] + [h for h in ranked if h is not holder]
        for p, h in enumerate(rungs):
            loss = max(1_000_000, 16_000_000 - 20_000 * p) if s.ladder else 9_000_000
            h["shift"] = 5000 - math.isqrt(loss)
        for k in range(n_succ):
            by_name[f"pred-{k}"]["shift"] = by_name[f"served-{k}"]["shift"]

        # --- the odds the window holds
        odds: dict[str, dict[int, np.ndarray]] = {}
        shared: dict[int, dict[int, np.ndarray]] = {}

        def arr(h, e):
            y = ys[e]
            return np.where(y >= 5000, 5000 + h["shift"], 5000 - h["shift"]).astype(np.int16)

        for h in heads:
            if h["role"] in ("stay", "pred"):
                epochs = range(first_window, NOW)
            elif h["role"] == "back":
                epochs = range(first_window, NOW - 20)
            elif h["role"] == "kept":
                epochs = range(first_window, NOW) if s.coverage == "full" else [h["seen"]]
            else:
                epochs = []                       # a successor: late odds only
            per = {}
            for e in epochs:
                if e < first_window:
                    continue
                if h["role"] == "kept" and s.coverage == "full" and e != h["seen"]:
                    # shared across the kept heads of one rung band: same work, less memory
                    band = h["shift"] // 50
                    per[e] = shared.setdefault(band, {}).get(e)
                    if per[e] is None:
                        per[e] = shared[band][e] = arr(h, e)
                else:
                    per[e] = arr(h, e)
            if per:
                odds[h["hash"]] = per
        # --- the challenge list: the slice, then published questions
        listed = published[:config.CHALLENGE_LIST_SIZE_V4]
        self.challenge = tuple((qids[i], self.bench_category[sets[q_set[i]]])
                               for i in listed.tolist())
        cats = tuple(c for _q, c in self.challenge)
        size = min(s.challenge, len(cats))
        zero = (0,) * size
        # every head but the holder challenged in the epochs it answered; a
        # challenge in E pairs with the holder after E - 2 (the same head)
        picks = {}
        for e in range(first_window, NOW):
            answered = {h["hash"]: zero for h in heads
                        if h is not holder and (
                            (h["role"] in ("stay", "pred"))
                            or (h["role"] == "back" and e < NOW - 20)
                            or (h["role"] == "kept" and e == h["seen"]))}
            picks[e] = PicksEpoch(e, cats, answered, False, holder["hash"])
        self.window = Window(size=s.epochs, slices={}, heads={}, pairs=pairs, odds=odds,
                             picks=picks)

        # --- the seat state at NOW - 1 and the copy rule's state
        kept_states = [state_of(h) for h in heads if h["kept"]]
        history = [h["hash"] for h in ranked if h["role"] == "kept"][:4]
        self.holder = holder["hash"]
        self.seat = seat_v4.SeatStateV4(champion=holder["hash"], champion_since=NOW - 10,
                                        history=tuple(history), heads=tuple(kept_states),
                                        epoch=NOW - 1,
                                        holders=((NOW - 1, holder["hash"]),
                                                 (NOW - 2, holder["hash"])))
        before = [copy_rule.Commitment(h["hotkey"], h["owner"], h["hash"], h["block"])
                  for h in heads if h["role"] in ("stay", "back", "pred", "kept")]
        self.copy = copy_rule.observe(copy_rule.CopyState(), NOW - 1, (NOW - 1) * BPE, before,
                                      ghost_retire=config.STANDING_EPOCHS_V4)
        self.commitments_now = [copy_rule.Commitment(h["hotkey"], h["owner"], h["hash"], h["block"])
                                for h in heads if h["role"] in ("stay", "back", "succ", "kept")]

        # --- the epoch's memo, slice, proofs and exploration answers
        index = memo_v4.reserve_index(self.published, self.pool, NOW)
        self.memo = memo_v4.epoch_memo(self.published, index,
                                       hashlib.sha256(b"time-v4-epoch").digest(), self.targets)
        self.slice_questions = [self.pool[i] for i in published[:config.SLICE_SIZE_V4].tolist()]
        served = [h for h in heads if h["served"]]
        self.served = served
        self.verified = {}
        # Every served head but the holder challenges with the first `size`
        # questions of the list: its own pick on each, and the holder's where
        # they differ (FAST_CROWN_V4.md 2); the validator recomputes both.
        from fugal_subnet.head_odds import picks_and_odds_v4

        prefix = [q for q, _c in self.challenge[:size]]
        hidden = np.stack([self.hidden_of(q) for q in prefix])
        hp = picks_and_odds_v4(SimpleNamespace(W=holder["W"], b=holder["b"],
                                               models=holder["models"]),
                               self.menu, hidden).picks
        self.differ = 0

        def call(q, m, side):
            return SimpleNamespace(question_id=q, routed_model=m,
                                   correct=bool(int(_hex("c", side, q, m)[:2], 16) % 3),
                                   failed=False, completion_tokens=100)

        for h in served:
            if h is holder:
                self.verified[h["uid"]] = SimpleNamespace(
                    scored=False, challenge=None, hotkey=h["hotkey"], scored_results=[],
                    holder_results=[])
                continue
            own = picks_and_odds_v4(SimpleNamespace(W=h["W"], b=h["b"], models=h["models"]),
                                    self.menu, hidden).picks
            held = [call(q, t, "holder") for q, m, t in zip(prefix, own, hp)
                    if t is not None and t != m]
            self.differ += len(held)
            self.verified[h["uid"]] = SimpleNamespace(
                scored=True, hotkey=h["hotkey"],
                challenge=SimpleNamespace(size=size, holder=holder["hash"]),
                scored_results=[call(q, m, "own") for q, m in zip(prefix, own)],
                holder_results=held)
        from fugal_subnet.explore_pool import ExploreAnswer
        elig = np.flatnonzero((first >= 0) & (first <= NOW - 2))
        fresh = list(self.memo.fresh)
        answers = []
        bench_of = {}
        for k in range(U):
            ck = self.coldkeys[k]
            qs = [qids[i] for i in elig[rng.integers(0, len(elig), private)].tolist()] + fresh
            for q in qs:
                m = entries[int(rng.integers(0, M))]
                answers.append(ExploreAnswer(q, m, ck, bool(rng.random() < 0.6), failed=False,
                                             private=True, prompt_tokens=200,
                                             completion_tokens=900))
        for a in answers:
            bench_of[a.question_id] = sets[q_set[int(a.question_id.rsplit("_", 1)[1])]]
        self.answers = answers
        self.bench_of = bench_of


    def hidden_of(self, qid: str) -> np.ndarray:
        v = self._hidden.get(qid)
        if v is None:
            g = np.random.default_rng(int(_hex("h", qid)[:16], 16))
            v = g.standard_normal(self.shape.dim).astype(np.float32)
            v = self._hidden[qid] = (v / np.linalg.norm(v)).astype(np.float32)
        return v

    def odds_input(self):
        from fugal_subnet import explore_pool
        from fugal_subnet.v2_epoch import OddsInput

        return OddsInput(nonce=hashlib.sha256(b"time-v4-epoch-nonce").digest(),
                         answers=self.answers, benchmark_of=self.bench_of,
                         heads={h["uid"]: (h["W"], h["b"], h["models"]) for h in self.served},
                         hidden_of=self.hidden_of, coldkey_cap=explore_pool.coldkey_cap(4),
                         category_targets=self.targets, explore_weights=self.weights,
                         private_weights=self.weights, menu_state=self.menu)

    def run_kwargs(self) -> dict:
        from fugal_subnet.v2_epoch import PicksInput, seat_params_from_config

        ids = list(self.menu.ids())
        return dict(
            epoch_index=NOW, epoch_id=f"e{NOW:08d}", boundary_block=NOW * BPE,
            blocks_per_epoch=BPE, questions=self.slice_questions, pool_models=ids,
            prices={m: (1e-6, 4e-6) for m in ids}, verified_proofs=self.verified,
            head_hashes={h["uid"]: h["hash"] for h in self.served},
            coldkey_of={h["uid"]: h["owner"] for h in self.served},
            hotkey_of={h["uid"]: h["hotkey"] for h in self.served}, records={},
            commitments=self.commitments_now, window=self.window, seat=self.seat,
            copy_state=self.copy, burn_uid=0,
            params=seat_params_from_config(1, 0, protocol_version=3),
            shares=(0.85, 0.10, 0.05), burn_in_questions=0, bench_category=self.bench_category,
            odds_input=self.odds_input(),
            picks=PicksInput(state=self.menu, challenge=self.challenge),
            seniority_floor=self.floor_block, seniority_salt=self.salt, memo=self.memo)


# --- the stages -----------------------------------------------------------------------------


class Counter:
    """Counts rule A / C comparisons (labels_v4.gain_strata calls and the
    requests of labels_v4.gain_strata_many) and the window epochs they read,
    by wrapping the module functions for one stage."""

    def __init__(self):
        self.calls = 0
        self.epochs = 0          # window epochs on which both sides have odds, over all calls
        self.batches = 0

    def _shared(self, window, x, y) -> int:
        have = [set(window.odds.get(s.head, {})) if s.head is not None else set(window.pairs)
                for s in (x, y)]
        return len(have[0] & have[1])

    def __enter__(self):
        from fugal_subnet import labels_v4

        self._orig = labels_v4.gain_strata
        self._orig_many = labels_v4.gain_strata_many
        me = self

        def counted(window, ref, x, y, **kw):
            me.calls += 1
            me.epochs += me._shared(window, x, y)
            return me._orig(window, ref, x, y, **kw)

        def counted_many(window, ref, requests, **kw):
            requests = list(requests)
            me.batches += 1
            me.calls += len(requests)
            me.epochs += sum(me._shared(window, x, y) for x, y in requests)
            return me._orig_many(window, ref, requests, **kw)

        labels_v4.gain_strata = counted
        labels_v4.gain_strata_many = counted_many
        return self

    def __exit__(self, *exc):
        from fugal_subnet import labels_v4

        labels_v4.gain_strata = self._orig
        labels_v4.gain_strata_many = self._orig_many
        return False


def run_stages(world: EpochWorld, *, whole: bool = True, td: bool = False,
               report=print) -> dict:
    """Each stage of the v4 branch of run_v2_epoch, in its order, on the
    world's inputs; then (`whole`) the whole call. Returns {stage: (seconds,
    peak MiB, note)}."""
    from fugal_subnet import config, copy_rule, labels_v4, memo_v4, seat_v4
    from fugal_subnet import v2_epoch as v2
    from fugal_subnet.shared_grades import (
        ScoredAnswer,
        shared_cell_grades,
        shared_cell_tokens,
    )

    kw = world.run_kwargs()
    out: dict = {}

    def stage(name, fn, note=lambda r: ""):
        with Sampler() as smp:
            r = fn()
        out[name] = (smp.seconds, smp.peak, note(r))
        report(f"  {name:<34} {smp.seconds:9.2f} s  {smp.peak:8.0f} MiB  {out[name][2]}")
        return r

    epoch, oi = NOW, kw["odds_input"]
    verified = kw["verified_proofs"]

    def prep():
        answers = [ScoredAnswer(question_id=r.question_id, model=r.routed_model,
                                coldkey=kw["coldkey_of"][uid], correct=bool(r.correct),
                                failed=bool(r.failed), completion_tokens=int(r.completion_tokens))
                   for uid in sorted(verified) for r in verified[uid].scored_results]
        # the shared cells are the slice's (run_v2_epoch, FAST_CROWN_V4.md 3)
        slice_ids = {q["question_id"] for q in kw["questions"]}
        answers = [a for a in answers if a.question_id in slice_ids]
        cells, tokens = shared_cell_grades(answers), shared_cell_tokens(answers)
        sl = v2.build_epoch_slice(epoch, kw["questions"], kw["pool_models"], cells, tokens,
                                  kw["prices"], kw["bench_category"])
        new_copy = copy_rule.observe(kw["copy_state"], epoch, kw["boundary_block"],
                                     kw["commitments"], ghost_retire=config.STANDING_EPOCHS_V4)
        committed = {c.hotkey: c for c in copy_rule.valid_commitments(kw["commitments"],
                                                                      kw["boundary_block"])}
        served = {uid: copy_rule.Commitment(kw["hotkey_of"][uid], kw["coldkey_of"][uid],
                                            kw["head_hashes"][uid],
                                            committed[kw["hotkey_of"][uid]].block)
                  for uid in sorted(verified)}
        verdicts = copy_rule.classify(new_copy, served)
        serving = {kw["head_hashes"][u]: u for u, v in sorted(verdicts.items()) if v == "owner"}
        model_index = {m: i for i, m in enumerate(kw["pool_models"])}
        by_head = {h: v2.head_epoch(verified[u], sl.question_ids, model_index)
                   for h, u in sorted(serving.items())}
        by_head = {h: he for h, he in by_head.items() if he is not None}
        return answers, cells, tokens, sl, new_copy, verdicts, serving, by_head

    answers, cells, tokens, sl, new_copy, verdicts, serving, by_head = stage(
        "slice, copy rule, serving", prep, lambda r: f"{len(r[6])} serving")
    pairs, odds_now, implausible, pooled = stage(
        "pool + current odds", lambda: v2._pool_and_odds(kw["window"], oi, serving, by_head,
                                                          kw["bench_category"]),
        lambda r: f"{len(r[0].question_ids)} labels, {len(r[1])} heads")
    seat_serving_now = seat_v4.seat_serving(kw["seat"], serving, new_copy)
    picks_rec, picks_block, known = stage(
        "challenges (both picks recomputed)",
        lambda: v2._challenge_epoch(kw["picks"], epoch, sl, seat_serving_now, verified,
                                    kw["seat"], kw["window"], oi, pairs,
                                    verified_by_hash=serving),
        lambda r: (f"{len(r[0].values)} counted, {len(r[1].get('refused', {}) or {})} refused "
                   f"or failed, {world.differ} holder-side calls"))

    def observe():
        ss = seat_v4.seat_serving(kw["seat"], serving, new_copy)
        rows = {h: seat_v4.row_digests(*oi.heads[u]) for h, u in sorted(ss.items())
                if u in oi.heads}
        st = seat_v4.observe(kw["seat"], epoch=epoch, serving=ss, rows=rows, copy_state=new_copy,
                             menu_ids=tuple(oi.menu_state.ids()), blocks_per_epoch=BPE)
        return ss, st

    seat_serving, observed = stage("seat observe", observe,
                                   lambda r: f"{len(r[1].heads)} heads kept")
    new_window, late = stage(
        "add epoch + late odds",
        lambda: v2._add_v4_epoch(
            kw["window"], sl, by_head, pairs, odds_now, oi, serving, new_copy, BPE, epoch,
            picks=picks_rec,
            odds_from={h: seat_v4.odds_since(observed, h, v2._since(new_copy, h, BPE, epoch))
                       for h in serving},
            late_slots=config.LATE_ODDS_SLOTS_V4,
            sides={hs.head: seat_v4.side_of(hs) for hs in observed.heads if hs.predecessors},
            seat=observed),
        lambda r: f"{sum(len(v) for v in r[1].values())} late head-epochs")
    stage("reference R (once)", lambda: labels_v4.reference(new_window, now=epoch))
    memo_ev = stage("memorisation evidence (all kept)",
                    lambda: v2._memo_verdicts(kw["memo"], new_window, observed, epoch),
                    lambda r: f"{len(r)} heads, {sum(e.refused for e in r.values())} refused")
    refused = frozenset(h for h, ev in memo_ev.items() if memo_v4.refused(ev))
    counter = Counter()

    def decide():
        with counter:
            return seat_v4.decide(observed, epoch=epoch, window=new_window, serving=seat_serving,
                                  floor_block=kw["seniority_floor"], salt=kw["seniority_salt"],
                                  memo_refused=refused, allow_take=True)

    decision = stage("seat decide (rules A, B, C)", decide,
                     lambda r: (f"{counter.calls} rule A/C comparisons over "
                                f"{counter.epochs} shared epochs, "
                                f"{len(r.evidence['order'])} contenders, "
                                f"{sum(len(d['earlier']) for d in r.evidence['heads'].values())} "
                                f"rule C rows, event {r.event}"))
    picks_block = stage(
        "holder picks on the list (published)",
        lambda: v2._publish_holder_picks(picks_block, decision.champion, known, kw["picks"],
                                         oi, serving, kw["pool_models"]),
        lambda r: f"{len(r.get('holder_picks', {}).get('picks', ''))} base64 chars")
    owners = {r.head_hash: r.owner for r in new_copy.heads}
    since = {h: v2._since(new_copy, h, BPE, epoch) for h in serving}
    stage("standing over R (serving heads)",
          lambda: labels_v4.standing_for_reveal(new_window, serving, owners, since, now=epoch),
          lambda r: f"{len(r)} heads")
    result = stage(
        "build the reveal (_v4_result)",
        lambda: v2._v4_result(
            decision=decision, window=new_window, copy_state=new_copy, serving=serving,
            seat_serving=seat_serving, shares=kw["shares"], burn_uid=0, rules=None,
            picks_block=picks_block, odds_input=oi, pairs=pairs, late=late,
            head_odds_now=odds_now, pooled=pooled, implausible=implausible,
            prices=kw["prices"], blocks_per_epoch=BPE, epoch_index=epoch, cells=cells,
            tokens=tokens, verified_proofs=verified, verdicts=verdicts,
            seniority_floor=kw["seniority_floor"], seniority_salt=kw["seniority_salt"],
            memo=kw["memo"], memo_evidence=memo_ev))
    text = stage("serialise the reveal (indented JSON)",
                 lambda: json.dumps(result.reveal, indent=2),
                 lambda r: f"{len(r) / 2**20:.1f} MiB")
    del text
    tmp = tempfile.mkdtemp(prefix="time-v4-")
    path = os.path.join(tmp, "window.npz")
    stage("save the window", lambda: v2.save_window(path, result.window),
          lambda r: f"{os.path.getsize(path) / 2**20:.1f} MiB on disk")
    stage("load the window", lambda: v2.load_window(path, result.window.size))
    os.remove(path)
    os.rmdir(tmp)
    if td:
        from fugal_subnet import head_odds

        h = world.served[-1]
        qs = [q["question_id"] for q in world.slice_questions] + \
             [a.question_id for a in world.answers if a.coldkey == h["owner"]]
        hidden = np.stack([world.hidden_of(q) for q in qs])
        art = SimpleNamespace(W=h["W"], b=h["b"], models=h["models"])
        stage("TD: route one epoch (one head)",
              lambda: head_odds.picks_and_odds_v4(art, world.menu, hidden),
              lambda r: f"{len(qs)} questions x {len(world.entries)} entries")
    if whole:
        from fugal_subnet.v2_epoch import run_v2_epoch

        r = stage("WHOLE run_v2_epoch (with memo)", lambda: run_v2_epoch(**kw))
        same = json.dumps(r.reveal, sort_keys=True) == json.dumps(result.reveal, sort_keys=True)
        report(f"  the whole epoch's reveal equals the staged one: {same}")
        out["_same_reveal"] = same
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--uids", type=int, default=256)
    p.add_argument("--epochs", type=int, default=48)
    p.add_argument("--entries", type=int, default=64)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--pool", type=int, default=48_320)
    p.add_argument("--reserve", type=int, default=7_000)
    p.add_argument("--successors", type=int, default=8)
    p.add_argument("--returning", type=int, default=8)
    p.add_argument("--coverage", choices=("realistic", "full"), default="realistic")
    p.add_argument("--served-last", action="store_true",
                   help="precedence by commit, the kept heads first: every served head is a "
                        "rule C candidate against all 512 kept heads and the served before it")
    p.add_argument("--no-ladder", action="store_true",
                   help="every head equally good (an honest field: rule C stops at once)")
    p.add_argument("--challenge", type=int, choices=(500, 1000), default=1000,
                   help="the size of every challenger's challenge")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-whole", action="store_true", help="skip the whole run_v2_epoch")
    p.add_argument("--td", action="store_true", help="also time one TD's routing")
    args = p.parse_args()
    if args.successors + args.returning >= args.uids:
        p.error("--successors + --returning must leave at least one UID served through "
                "the window (the holder is one of them)")
    shape = Shape(uids=args.uids, epochs=args.epochs, entries=args.entries, dim=args.dim,
                  pool=args.pool, reserve=args.reserve, successors=args.successors,
                  returning=args.returning, coverage=args.coverage,
                  ladder=not args.no_ladder, served_last=args.served_last,
                  challenge=args.challenge, seed=args.seed)
    print(f"shape: {shape}")
    t0 = time.perf_counter()
    with Sampler() as smp:
        world = EpochWorld(shape)
    n_labels = sum(len(p.question_ids) for p in world.window.pairs.values())
    n_odds = sum(len(v) for v in world.window.odds.values())
    odds_mib = sum(o.nbytes for v in world.window.odds.values() for o in v.values()) / 2**20
    print(f"world built in {smp.seconds:.1f} s, {smp.peak:.0f} MiB: {len(world.window.pairs)} "
          f"window epochs, {n_labels} labels, {len(world.seat.heads)} heads kept, "
          f"{n_odds} head-epochs of odds ({odds_mib:.0f} MiB as distinct arrays), "
          f"F = {world.fresh_per_epoch}, reserve {world.memo.index.reserve_size}")
    print(f"  {'stage':<34} {'seconds':>11} {'peak':>12}")
    out = run_stages(world, whole=not args.no_whole, td=args.td)
    total = sum(v[0] for k, v in out.items() if not k.startswith(("_", "WHOLE", "TD",
                                                                  "reference R",
                                                                  "standing over")))
    print(f"sum of the epoch's stages: {total:.1f} s; whole run: "
          f"{out.get('WHOLE run_v2_epoch (with memo)', (float('nan'),))[0]:.1f} s; "
          f"peak RSS {peak_mib():.0f} MiB; wall {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
