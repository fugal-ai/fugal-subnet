"""One protocol v4 epoch whose seat reads the memorisation verdict, through
the live epoch (v2_epoch.run_v2_epoch with a memo_v4.EpochMemo), on a window
built directly (tests/test_v4_seat_memo.py; the seat_v4 stage of
scripts/check_determinism.py).

The window holds WINDOW_EPOCHS epochs of pooled labels (labels_v4.LabelPairs)
before the epoch `NOW`, each on two kinds of question: OLD ones, first
published in epoch 50, long before any head's commit, and the epoch's FRESH
ones, first published by that epoch's own reveal and answered by every voter
(SEAT_MINIMAL_V4.md 6.3). Two benchmarks. Voters own no head.

A head is real arrays (so the epoch computes its odds on its own labels and
its row digests, which succession reads), and its odds on the window's
labels are set by what it knows: on a label it knows, the outcome at 9000 /
1000; otherwise its benchmark's base rate. So each verdict is set by
construction:

  mem     knows half the OLD outcomes and no FRESH one: refused;
  honest  knows a third of every outcome: not refused;
  line1   as mem, committed by the hotkey `hk-line`;
  line2   line1's rows byte for byte plus a row for an entry off the menu,
          committed by the same hotkey and coldkey in the epoch before NOW:
          line1's successor when first seen at NOW with line1 not served.
          Its odds on the window are line1's (equal rows give equal odds:
          what its late odds would be).

Each head but line2 is committed in epoch COMMIT, before the window; a head
named in `kept` stands (seen in NOW - 1, and answering the scored questions
in every window epoch) and the seat state holds it.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

NOW = 148
WINDOW_EPOCHS = 47              # NOW - 47 .. NOW - 1, all inside the 48-epoch window
COMMIT = 100
BPE = 360
DIM = 16
ENTRIES = ("lab-a/model@high", "lab-b/model@low", "lab-c/model@high")
OFF_MENU = "lab-z/model@high"
VOTERS = tuple(f"ck-voter-{i}" for i in range(8))
OLD_PER_EPOCH, FRESH_PER_EPOCH = 120, 6
SLICE_QUESTIONS, PICKS_QUESTIONS = 30, 200
BENCH_CATEGORY = {"b-easy": "knowledge", "b-hard": "hard", "mmlu": "knowledge", "math": "math",
                  "aime": "hard"}
RATE = {"b-easy": 0.8, "b-hard": 0.3}
KNOWS = {"mem": (0.5, 0.0), "honest": (0.34, 0.34), "line1": (0.5, 0.0)}   # (OLD, FRESH)
HEADS = ("honest", "line1", "line2", "mem")
OWNER = {"mem": "ck-mem", "honest": "ck-honest", "line1": "ck-line", "line2": "ck-line"}
HOTKEY = {"mem": "hk-mem", "honest": "hk-honest", "line1": "hk-line", "line2": "hk-line"}
UID = {"mem": 11, "honest": 12, "line1": 13, "line2": 13}


def _u(*parts) -> float:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8],
                          "big") / 2**64


def _block(commit_epoch: int) -> int:
    """A block whose first boundary at or after it is `commit_epoch`'s."""
    return (int(commit_epoch) - 1) * BPE + 7


def hidden_of(qid: str) -> np.ndarray:
    v = np.random.default_rng(int.from_bytes(hashlib.sha256(qid.encode()).digest()[:8], "big")
                              ).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def arrays(name: str) -> tuple:
    """(W, b, model ids) of a head; line2 is line1's rows plus one."""
    base = "line1" if name == "line2" else name
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(base.encode()).digest()[:8], "big"))
    W = (rng.standard_normal((len(ENTRIES), DIM)) * 2.0).astype(np.float32)
    b = rng.standard_normal(len(ENTRIES)).astype(np.float32)
    models = list(ENTRIES)
    if name == "line2":
        W = np.vstack([W, np.ones((1, DIM), np.float32)])
        b = np.append(b, np.float32(0.5)).astype(np.float32)
        models.append(OFF_MENU)
    return W, b, models


def hash_of(name: str) -> str:
    W, b, models = arrays(name)
    return hashlib.sha256(W.tobytes() + b.tobytes() + "|".join(models).encode()).hexdigest()


def menu(entries: tuple[str, ...] = ENTRIES, epoch: int = NOW):
    """The epoch's menu of `entries` (ids "<model>@<effort>"); the timing
    harness (scripts/experiments/time_v4_epoch.py) builds its 64 with it."""
    from fugal_subnet import menu_state as ms

    entries = [ms.Entry(id=e, rev=1, model=e.split("@")[0], reasoning=("effort", e.split("@")[1]),
                        max_tokens=8192, provider_order=("host-a/fp8",), quantization="fp8",
                        price_in=1000, price_out=4000, mean_in=500_000, mean_out=900_000,
                        token_pairs=100, joined_epoch=0, scored_from=0, measured_rate=5000,
                        proposer="") for e in entries]
    return ms.MenuState(epoch=epoch, serial=epoch, entries=tuple(entries),
                        retired=(), labs=ms.AllowlistRef(0, ""), devices=ms.AllowlistRef(0, ""))


def _odds(name: str, pairs, kinds) -> np.ndarray:
    old, fresh = KNOWS["line1" if name == "line2" else name]
    out = []
    for i, (qid, bench, y, kind) in enumerate(zip(pairs.question_ids, pairs.benchmarks,
                                                  pairs.y.tolist(), kinds)):
        knows = old if kind == "old" else fresh
        if _u("k", "line1" if name == "line2" else name, qid, i) < knows:
            out.append(9000 if y else 1000)
        else:
            out.append(int(RATE[bench] * 10_000))
    return np.array(out, np.int16)


class World:
    """The window, the published set and the heads' odds (built once)."""

    def __init__(self):
        from fugal_subnet import labels_v4
        from fugal_subnet.champion import EpochPairs
        from fugal_subnet.picks_v4 import PicksEpoch

        self.pairs, self.odds, self.picks, first = {}, {n: {} for n in HEADS}, {}, {}
        for e in range(NOW - WINDOW_EPOCHS, NOW):
            q, m, b, v, kinds = [], [], [], [], []
            for i in range(OLD_PER_EPOCH):
                qid = f"o{e}_{i}"
                first[qid] = 50
                q.append(qid)
                m.append(ENTRIES[int(_u("m", qid) * len(ENTRIES))])
                b.append("b-easy" if i % 2 else "b-hard")
                v.append((VOTERS[i % len(VOTERS)],))
                kinds.append("old")
            for j in range(FRESH_PER_EPOCH):
                qid = f"f{e}_{j}"
                first[qid] = e
                for k, ck in enumerate(VOTERS):
                    q.append(qid)
                    m.append(ENTRIES[k % len(ENTRIES)])
                    b.append("b-easy" if j % 2 else "b-hard")
                    v.append((ck,))
                    kinds.append("fresh")
            y = [10_000 if _u("y", e, qid, n) < RATE[bench] else 0
                 for n, (qid, bench) in enumerate(zip(q, b))]
            n = len(q)
            base = EpochPairs(question_ids=tuple(q), models=tuple(m), benchmarks=tuple(b),
                              categories=tuple(BENCH_CATEGORY[x] for x in b),
                              y=np.array(y, np.int32), prompt_tokens=np.zeros(n, np.int32),
                              completion_tokens=np.zeros(n, np.int32))
            pairs = labels_v4.label_pairs(base, v, np.full(n, -1, np.int16))
            self.pairs[e] = pairs
            for name in HEADS:
                self.odds[name][e] = labels_v4.effective(_odds(name, pairs, kinds), pairs)
            self.picks[e] = PicksEpoch(e, ("knowledge",) * PICKS_QUESTIONS, {
                hash_of(name): (0,) * PICKS_QUESTIONS for name in HEADS})
        self.first = first

    def published(self):
        """The published set folded through NOW - 1."""
        from fugal_subnet.memo_v4 import Published

        return Published(dict(self.first), NOW - 1)

    def memo(self):
        """NOW's EpochMemo: the reserve of a small pool of the v4 mix's
        sets, none published, and the published set above."""
        from fugal_subnet import epoch_plan, memo_v4

        sets = sorted(epoch_plan.mix_v4().bench_category)
        pool = [{"question_id": f"{s}_{i:03d}", "benchmark": s, "prompt": "x"}
                for s in sets for i in range(40)]
        pub = self.published()
        index = memo_v4.reserve_index(pub, pool, NOW)
        return memo_v4.epoch_memo(pub, index, hashlib.sha256(b"memo-seat-world").digest(),
                                  epoch_plan.mix_v4().targets)

    def window(self, kept, holder=None):
        """The window through NOW - 1: every epoch's labels, the odds of the
        heads the seat keeps (line2's when line1 is kept: equal rows) and
        their challenge records, each paired with `holder` (items 0: picks
        as good as the holder's)."""
        import dataclasses

        from fugal_subnet.champion import Window
        from fugal_subnet.picks_v4 import PicksEpoch

        names = set(kept) | ({"line2"} if "line1" in kept else set())
        hashes = {hash_of(n) for n in kept}
        return dataclasses.replace(
            Window.empty(48), pairs=dict(self.pairs),
            odds={hash_of(n): dict(self.odds[n]) for n in sorted(names)},
            picks={e: PicksEpoch(e, r.categories, {h: v for h, v in r.values.items() if h in hashes},
                                 False, None if holder is None else hash_of(holder))
                   for e, r in self.picks.items()})

    def seat(self, holder, kept, named=True):
        """A v4 seat state at NOW - 1 that keeps `kept` (each seen then),
        `holder` holding the seat since NOW - 10. NOW's challenges pair with
        it (the holder after NOW - 2) when `named`; otherwise with no holder
        (this world has no bytes of a holder that does not serve at NOW)."""
        from fugal_subnet import seat_v4

        heads = []
        for n in kept:
            W, b, models = arrays(n)
            heads.append(seat_v4.HeadState(
                head=hash_of(n), owner=OWNER[n], hotkey=HOTKEY[n], block=_block(COMMIT),
                since=COMMIT, seen=NOW - 1, origin=hash_of(n), origin_block=_block(COMMIT),
                origin_owner=OWNER[n],
                rows={e: d for e, d in seat_v4.row_digests(W, b, models).items() if e in ENTRIES}))
        return seat_v4.SeatStateV4(champion=None if holder is None else hash_of(holder),
                                   champion_since=None if holder is None else NOW - 10,
                                   heads=tuple(heads), epoch=NOW - 1,
                                   holders=((NOW - 2, None if holder is None or not named
                                             else hash_of(holder)),))

    def commitments(self, names):
        from fugal_subnet.copy_rule import Commitment

        return [Commitment(HOTKEY[n], OWNER[n], hash_of(n),
                           _block(NOW - 1 if n == "line2" else COMMIT)) for n in names]

    def run(self, *, holder, kept, served, memo, fed=list, window=None, seat=None):
        """NOW through run_v2_epoch: `kept` the heads the seat state keeps,
        `served` the heads verified at NOW (each answering the scored
        questions), committed as they are (line2 replacing line1 on its
        hotkey when served); `memo` whether the epoch's EpochMemo is passed.
        `fed` orders every input list (the determinism stage reverses them);
        `window` and `seat` replace the built ones (a node's saved copies)."""
        import dataclasses

        from fugal_subnet import copy_rule
        from fugal_subnet.champion import OddsParams
        from fugal_subnet.explore_pool import ExploreAnswer
        from fugal_subnet.v2_epoch import (
            OddsInput,
            PicksInput,
            run_v2_epoch,
            seat_params_from_config,
        )

        state = menu()
        ids = list(state.ids())
        before = [n for n in kept if n != "line2"]
        copy = copy_rule.observe(copy_rule.CopyState(), NOW - 1, (NOW - 1) * BPE,
                                 self.commitments(before), ghost_retire=3)
        now_heads = sorted({*before, *served} - ({"line1"} if "line2" in served else set()))
        answers, bench_of = [], {}
        for j in range(FRESH_PER_EPOCH):
            qid = f"x{NOW}_{j}"
            bench_of[qid] = "b-easy" if j % 2 else "b-hard"
            for k, ck in enumerate(VOTERS):
                m = ENTRIES[k % len(ENTRIES)]
                answers.append(ExploreAnswer(qid, m, ck, _u("a", qid, ck) < RATE[bench_of[qid]],
                                             failed=False, private=True, prompt_tokens=20,
                                             completion_tokens=200))
        benches = ("mmlu", "math", "aime")
        questions = [{"question_id": f"s{NOW}q{i}", "benchmark": benches[i % 3]}
                     for i in range(SLICE_QUESTIONS)]
        # Each served head answers a challenge on the slice (FAST_CROWN_V4.md
        # 2): its picks and the named holder's where they differ, as the
        # heads give them on the menu.
        from fugal_subnet.head_odds import picks_and_odds_v4
        from fugal_subnet.v2_epoch import _artifact

        named = holder is not None and holder in served
        from fugal_subnet import seat_v4 as _seat
        if seat is None:
            seat = self.seat(holder, kept, named)
        elif _seat.challenge_holder(seat, NOW) not in {hash_of(n) for n in served}:
            # A seat handed in names a holder this world has no bytes of at
            # NOW (it does not serve): NOW's challenges pair with none.
            seat = dataclasses.replace(seat, holders=((NOW - 2, None),))
        named_hash = _seat.challenge_holder(seat, NOW)
        named = named_hash is not None
        hname = next((n for n in HEADS if hash_of(n) == named_hash), None)
        challenge = tuple((q["question_id"], BENCH_CATEGORY.get(q["benchmark"], q["benchmark"]))
                          for q in questions)
        hid = np.stack([hidden_of(q["question_id"]) for q in questions])
        theirs = (picks_and_odds_v4(_artifact(arrays(hname)), state, hid).picks if named
                  else (None,) * len(questions))
        verified = {}
        for n in fed(sorted(served)):
            own = picks_and_odds_v4(_artifact(arrays(n)), state, hid).picks
            verified[UID[n]] = SimpleNamespace(
                scored=True, hotkey=HOTKEY[n],
                challenge=SimpleNamespace(size=len(questions),
                                          holder=named_hash),
                scored_results=[
                    SimpleNamespace(question_id=q["question_id"], routed_model=m,
                                    correct=_u("c", n, i) < 0.6, failed=False,
                                    completion_tokens=100)
                    for i, (q, m) in enumerate(zip(questions, own))],
                holder_results=[
                    SimpleNamespace(question_id=q["question_id"], routed_model=t,
                                    correct=_u("c", n, i) < 0.6, failed=False,
                                    completion_tokens=100)
                    for i, (q, m, t) in enumerate(zip(questions, own, theirs))
                    if t is not None and t != m])
        oi = OddsInput(nonce=hashlib.sha256(b"memo-seat-world-nonce").digest(),
                       answers=fed(answers), benchmark_of=bench_of,
                       heads={UID[n]: arrays(n) for n in fed(sorted(served))},
                       hidden_of=hidden_of, menu_state=state, single_model_min_pairs=30)
        params = dataclasses.replace(
            seat_params_from_config(1, 0, protocol_version=3), min_paired_questions=60,
            resamples=200, odds=OddsParams(margin_points=0.1, min_pairs=150, nv_noninferiority=0.5,
                                           ref_prior=20, confirm_pairs=150))
        return run_v2_epoch(
            epoch_index=NOW, epoch_id=f"e{NOW:08d}", boundary_block=NOW * BPE,
            blocks_per_epoch=BPE, questions=questions, pool_models=ids,
            prices={m: (1e-6, 4e-6) for m in ids}, verified_proofs=verified,
            head_hashes={UID[n]: hash_of(n) for n in served},
            coldkey_of={UID[n]: OWNER[n] for n in served},
            hotkey_of={UID[n]: HOTKEY[n] for n in served}, records={},
            commitments=fed(self.commitments(now_heads)),
            window=self.window(kept, holder) if window is None else window,
            seat=seat, copy_state=copy,
            burn_uid=0,
            params=params, shares=(0.85, 0.10, 0.05), burn_in_questions=0,
            bench_category=BENCH_CATEGORY, odds_input=oi,
            picks=PicksInput(state=state, challenge=challenge,
                             heads={hash_of(n): arrays(n) for n in kept}),
            seniority_floor=BPE, seniority_salt="0x" + "5a" * 32,
            memo=self.memo() if memo else None)


__all__ = ["COMMIT", "NOW", "World", "arrays", "hash_of", "hidden_of"]
