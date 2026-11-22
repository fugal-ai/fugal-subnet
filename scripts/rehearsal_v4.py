#!/usr/bin/env python3
"""Protocol v4 flip rehearsal, in process: the shipped loops from v3 across
the v4 block and through the menu's life, with fakes for the chain, the
store, the clock and the provider (tests/v4_world.py, the loop tests'
stand-ins). No Docker, no chain, no network, no paid call:

    python scripts/rehearsal_v4.py          # under two minutes; prints a table, exit 1 on a FAIL

Timeline (F the first v4 epoch; validators run `neurons/validator.main`,
miners `neurons/miner._run_epoch`, a v3 image's epochs before F and a v4
image's from F):

    F-3..F-1  protocol v3: a root, three followers, four miners.
    F         the flip: the window and the seat empty, the menu the pinned
              seed (the fixture menu here), the four-field commitment.
    F+1       miner 0 files a proposal (lab-c/model-new@high): its document
              served from its box, its hash committed beside its head. The
              root's pinned read in F+1 admits it, every node applies the
              delta at the boundary: on the menu of F+2 as provisional, every
              miner's coverage aimed at it there; the step scoring F+2 makes
              it full from F+4.
    F+1..     calls to lab-a/model-one@low fail for every operator, more
              than the retirement rate: the entry is retired.
    F+2..     the providers' listing doubles lab-b/model-two's price: it
              moves at the next refresh epoch, and only there.
    F+3       miner 1's TD restarts (a new session, nothing kept).
    F         follower C is down for the first v4 epoch; it restarts at F+1
              and agrees with the root from F+2.
    F+4       follower B stops; it restarts at F+5 from its saved state.
    F+10,F+11 the root is down; it restarts at F+12. The menus of the two
              epochs its missing commitments would have named carry, and
              the published set does not move; every TD proves and every
              follower checks every proof throughout.

SHORTENED FOR THE REHEARSAL, and only here: the menu refresh cadence is 4
epochs (MENU_REFRESH_EPOCHS is 20) on every node and TD alike, and the
compiled reader accepts the fixture's small menus, as the loop tests do.

Then the seat (SEAT_MINIMAL_V4.md 11), on a root and a follower running the
shipped loops from the v4 block (tests/v4_world.py `SeatLoop`; the follower
is handed every input in the reverse order of the root's), in three runs
whose scenarios and checks are the ones tests/test_v4_seat_loop.py runs:

    take    the seat is empty at the flip and burns until A fills it with a
            challenge; A then holds and is paid on its exploration alone; B,
            odds only, turns a 1,000-question challenge on: its first attempt
            runs out of its time budget and its second has no holder it can
            read (the root's reveal not to be had), both counting for
            nothing, and its one challenge takes the seat that epoch; a
            near-copy of B and an exact copy of A never take it.
    hold    A holds on exploration alone and adds the new entry's row by
            succession, keeping the seat; a challenger with no edge over it
            never takes it; A misses two epochs and keeps it unpaid, then
            three and the latest qualifier takes it.
    (FAST_CROWN_V4.md: the challenge, the holder on exploration.)
    memo    nothing handed to the loops: the release's published-set seed
            is the half of the pool a sharp head is confident on, every node
            and TD brings the set forward from the root's reveals, and the
            epochs' fresh draws spend the other half; A fills the seat and,
            once its unseen labels are in, is refused, so the bar is R, the
            seat open and its share burns.

Every v4 epoch of every run carries the memorisation check's fields; the
flip run's row `the published set` checks every node holds the set the
root's public reveals fold to from the seed.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.v4_world import NEW_ID, PROPOSAL  # noqa: E402 - the proposal the loop tests file

FLIP = 12
FIRST, LAST = FLIP - 3, FLIP + 15
FAILING_ID, FAILING_MODEL = "lab-a/model-one@low", "lab-a/model-one"
PRICED_ID, PRICED_MODEL = "lab-b/model-two", "lab-b/model-two"
PRICE_FROM = FLIP + 2
REFRESH = 4
PROPOSE_AT = FLIP + 1                # miner 0 files its proposal in this epoch
MINER_RESTART = FLIP + 3
FOLLOWER_STOP, FOLLOWER_BACK = FLIP + 4, FLIP + 5
ROOT_DOWN, ROOT_BACK = FLIP + 10, FLIP + 12

# The seat's rows (docs/decisions/SEAT_MINIMAL_V4.md 11), by hook: (row,
# run, check in tests/v4_world.py). Each run's loops are also checked for
# agreement and a rebuild from the root's reveals, one row per run.
SEAT_RUNS = {"take": "seat_take_scenario", "hold": "seat_hold_scenario",
             "memo": "seat_memo_scenario"}
SEAT_HOOKS = {
    "seat_fill": ("seat: the seat fills after the flip; the holder is paid on exploration "
                  "alone", "take", "check_seat_fill"),
    "seat_odds_only": ("seat: an odds-only miner takes the seat with one 1,000-question "
                       "challenge; an attempt out of time or with no readable holder counts "
                       "for nothing", "take", "check_seat_odds_only_take"),
    "seat_copies": ("seat: a near-copy and an exact copy never take it", "take",
                    "check_seat_copies"),
    "seat_succession": ("seat: a holder adds a row by succession and keeps seat and evidence",
                        "hold", "check_seat_succession"),
    "seat_holder_head": ("seat: a follower that starts without the holder's head obtains it "
                         "(its own store, the miner, the root's head publication, in that "
                         "order)", "hold", "check_seat_holder_head_from_root"),
    "seat_standing": ("seat: a challenger with no edge does not take; a holder misses two "
                      "epochs and keeps it, then three and loses it", "hold",
                      "check_seat_standing"),
    "seat_memo": ("seat: a holder that memorises is refused through the real wiring; the "
                  "seat opens, unpaid", "memo", "check_seat_memo_refusal"),
}


@dataclasses.dataclass
class Row:
    name: str
    status: str            # PASS, FAIL, SKIP
    detail: str = ""


class _Alarms:
    """Every ALARM line any node logs during the run, with the thread (the
    validator) that logged it. No scenario of this rehearsal expects one."""

    def __init__(self):
        import logging

        class Handler(logging.Handler):
            def emit(inner, record):
                try:
                    text = record.getMessage()
                except Exception:  # noqa: BLE001 - a bad record is not an alarm
                    return
                if text.startswith("ALARM"):
                    self.lines.append((record.threadName, text))

        self.lines: list[tuple[str, str]] = []
        self.handler = Handler(level=logging.ERROR)
        logging.getLogger().addHandler(self.handler)

    def close(self):
        import logging
        logging.getLogger().removeHandler(self.handler)


class _Patch:
    """monkeypatch's setattr / setenv, undone at the end."""

    def __init__(self):
        import pytest
        self.mp = pytest.MonkeyPatch()

    def setattr(self, obj, attr, value):
        self.mp.setattr(obj, attr, value)

    def setenv(self, name, value):
        self.mp.setenv(name, value)

    def undo(self):
        self.mp.undo()


def _provider_class():
    from tests.test_harness_v4 import _listing
    from tests.v4_world import Provider

    class RehearsalProvider(Provider):
        """The stub provider, plus: calls to FAILING_MODEL's low effort fail
        (three in five) from F+1, and the listing doubles PRICED_MODEL's
        price from PRICE_FROM."""

        epoch = 0

        def fetch(self, model):
            if model == PRICED_MODEL and self.epoch >= PRICE_FROM:
                return _listing(model, hosts=("host-c",),
                                pricing={"prompt": "0.000004", "completion": "0.00002"})
            return super().fetch(model)

        def call(self, proxy, model_id, question, request=None, reasoning=None, deadline_s=None):
            wire = (request or {}).get("reasoning") if isinstance(request, dict) else None
            low = (isinstance(request, dict) and request.get("model") == FAILING_MODEL
                   and isinstance(wire, dict) and wire.get("effort") == "low")
            if low and self.epoch >= FLIP + 1 and hashlib.sha256(
                    question["question_id"].encode()).digest()[0] < 154:
                return None
            return super().call(proxy, model_id, question, request, reasoning)

    return RehearsalProvider


def run(workdir: Path) -> tuple[list[Row], object]:
    from fugal_subnet import epoch_plan
    from fugal_subnet import menu_state as ms
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block
    from tests import menu_fixtures as fx
    from tests.signed_list_fixtures import Signer
    from tests.v4_world import BPE, FlipWorld, MinerSpec, ValidatorSpec

    patch = _Patch()
    rows: list[Row] = []
    alarms = _Alarms()
    try:
        from fugal_subnet import menu_rules as mr
        signer = Signer(workdir, "release")
        patch.setattr(mr, "list_signers", lambda: signer.signers_line().encode())
        fx.allow_small_menus(patch)
        # The refresh cadence, shortened on the step and on every reader alike.
        patch.setattr(ms, "READER_PARAMS", dataclasses.replace(ms.READER_PARAMS,
                                                               refresh_epochs=REFRESH))
        real_params = epoch_plan.menu_step_params_v4
        patch.setattr(epoch_plan, "menu_step_params_v4", lambda floor: dataclasses.replace(
            real_params(floor), refresh_epochs=REFRESH, min_entries=1))

        miners = [MinerSpec(f"5Fminer-{i}", f"5Fminer-cold-{i}", seed=i,
                            proposal=PROPOSAL if i == 0 else None,
                            propose_at=PROPOSE_AT if i == 0 else None)
                  for i in range(4)]
        specs = [
            ValidatorSpec("root", "root", "root", root=True, stop=ROOT_DOWN * BPE - 1),
            ValidatorSpec("root+", "root", "root", root=True, start=ROOT_BACK * BPE),
            ValidatorSpec("follower-a", "follower-a", "follower-a"),
            ValidatorSpec("follower-b", "follower-b", "follower-b",
                          stop=FOLLOWER_STOP * BPE - 1),
            ValidatorSpec("follower-b+", "follower-b", "follower-b", start=FOLLOWER_BACK * BPE),
            # Misses the first v4 epoch: stopped after the last v3 one.
            ValidatorSpec("follower-c", "follower-c", "follower-c", stop=FLIP * BPE - 1),
            ValidatorSpec("follower-c+", "follower-c", "follower-c", start=(FLIP + 1) * BPE),
        ]
        provider_cls = _provider_class()

        def on_epoch(world, epoch):
            world.provider.epoch = epoch
            if epoch == MINER_RESTART:
                spec = miners[1]
                world.miners[spec.hotkey].session = world.session(spec)

        w = FlipWorld(workdir, patch.setattr, patch.setenv, signer, flip=FLIP, first=FIRST,
                      last=LAST, validators=specs, miners=miners, on_epoch=on_epoch)
        w.provider = provider_cls(cheater=None)
        w.setup()
        t0 = time.monotonic()
        w.run()
        elapsed = time.monotonic() - t0
        rows += _checks(w, epoch_id_for_block, ms)
        # Every scenario above expects no alarm (none is a fault injected to
        # raise one): a root that misses epochs carries the menu, which is
        # the rule, and a restarted node agrees with the root again.
        rows.append(Row("no validator logged an alarm",
                        "FAIL" if alarms.lines else "PASS",
                        "; ".join(f"{who}: {text[:160]}" for who, text in alarms.lines[:3])
                        if alarms.lines else "no ALARM line in any scenario"))
        rows.append(Row("ran", "PASS", f"{LAST - FIRST + 1} epochs in {elapsed:.0f} s"))
    except Exception as e:  # noqa: BLE001 - one row, the run fails
        import traceback
        rows.append(Row("rehearsal ran to the end", "FAIL",
                        f"{type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-3]}"))
        w = None
    finally:
        patch.undo()
        alarms.close()
    rows += run_seat(workdir)
    return rows, w


def run_seat(workdir: Path) -> list[Row]:
    """The seat's runs and their rows (SEAT_RUNS, SEAT_HOOKS)."""
    from fugal_subnet import menu_rules as mr
    from tests import v4_world
    from tests.signed_list_fixtures import Signer
    from tests.test_menu_publish import LIST_BYTES

    rows: list[Row] = []
    for run_name, scenario in SEAT_RUNS.items():
        patch = _Patch()
        alarms = _Alarms()
        hooks = {h: v for h, v in SEAT_HOOKS.items() if v[1] == run_name}
        try:
            d = workdir / f"seat-{run_name}"
            d.mkdir(parents=True, exist_ok=True)
            signer = Signer(d, "release")
            patch.setattr(mr, "list_signers", lambda s=signer: s.signers_line().encode())
            t0 = time.monotonic()
            loop = getattr(v4_world, scenario)(d, patch, signer.sign(LIST_BYTES))
            detail = loop.check_agreement_and_rebuild()
            rows.append(Row(f"seat run `{run_name}`: both loops agree, a rebuild matches", "PASS",
                            f"{detail} ({time.monotonic() - t0:.0f} s)"))
            # No seat scenario injects a fault meant to raise an alarm either.
            rows.append(Row(f"seat run `{run_name}`: no validator logged an alarm",
                            "FAIL" if alarms.lines else "PASS",
                            "; ".join(f"{who}: {text[:160]}" for who, text in alarms.lines[:3])
                            if alarms.lines else "no ALARM line"))
        except Exception as e:  # noqa: BLE001 - its rows fail, the others run
            rows.append(Row(f"seat run `{run_name}`", "FAIL", f"{type(e).__name__}: {e}"[:300]))
            rows += [Row(name, "FAIL", f"run `{run_name}` failed: hook `{hook}`")
                     for hook, (name, _run, _check) in hooks.items()]
            continue
        finally:
            alarms.close()
            patch.undo()
        for hook, (name, _run, check) in hooks.items():
            try:
                rows.append(Row(name, "PASS", getattr(v4_world, check)(loop)))
            except AssertionError as e:
                rows.append(Row(name, "FAIL", f"hook `{hook}`: {e}"[:300]))
            except Exception as e:  # noqa: BLE001
                rows.append(Row(name, "FAIL", f"hook `{hook}`: {type(e).__name__}: {e}"[:300]))
    return rows


def _checks(w, epoch_id_for_block, ms) -> list[Row]:
    rows = []

    def check(name, fn):
        try:
            detail = fn()
            rows.append(Row(name, "PASS", detail or ""))
        except AssertionError as e:
            rows.append(Row(name, "FAIL", str(e)[:300]))
        except Exception as e:  # noqa: BLE001
            rows.append(Row(name, "FAIL", f"{type(e).__name__}: {e}"[:300]))

    states = sorted(w.dirs)
    ran = {s: [x["epoch_id"] for x in w.logged(s)] for s in states}
    ids = {k: epoch_id_for_block(k) for k in range(FIRST, LAST + 1)}

    def no_errors():
        assert w.clock.errors == {}, w.clock.errors
        assert w.errors == [], w.errors
        # No pinned read of a block the chain has not produced (v4_world.Chain).
        assert w.chain.future_reads == [], w.chain.future_reads[:5]
        assert ran["root"] == [ids[k] for k in ids if not ROOT_DOWN <= k < ROOT_BACK], ran["root"]
        assert ran["follower-a"] == list(ids.values())
        assert ran["follower-b"] == [ids[k] for k in ids if k != FOLLOWER_STOP], ran["follower-b"]
        bad = [(s, x["epoch_id"]) for s in states for x in w.logged(s)
               if x["n_heads_valid"] != len(w.miner_specs) or x["n_heads_invalid"]
               or x["n_heads_unverifiable"]]
        assert not bad, f"proofs not all verified: {bad}"
        assert all(m.no_proof == {} for m in w.miners.values()), {
            m.hotkey: sorted(m.no_proof) for m in w.miners.values()}
        return f"{len(states)} validators, {len(w.miner_specs)} miners, every proof verified"

    def flip():
        for k in ids:
            r = w.reveal("follower-a", k)
            assert r["protocol_version"] == (4 if k >= FLIP else 3), k
            assert r["v2_state"]["rules"]["protocol"] == (4 if k >= FLIP else 3), k
        seat = w.reveal("root", FLIP)["v2_state"]["seat"]
        assert seat["champion"] is None
        return "v3 to v4 at the block; window and seat empty at it"

    def same_menu():
        for k in range(FLIP, LAST + 1):
            on = {ms.menu_hash(w.held(s).held(k)) for s in states if w.held(s).held(k)}
            on |= {m.menus[k] for m in w.miners.values() if k in m.menus}
            assert len(on) == 1, f"{ids[k]}: {len(on)} menus"
        return "every node and TD on one menu every v4 epoch"

    def menus():
        return {k: w.held("follower-a").held(k) for k in range(FLIP, LAST + 1)}

    def proposal():
        m = menus()
        joined = [k for k, s in m.items() if NEW_ID in s.ids()]
        provisional = [k for k in joined if m[k].entry(NEW_ID).provisional]
        full = [k for k in joined if not m[k].entry(NEW_ID).provisional]
        proposer = w.miners[w.miner_specs[0].hotkey]
        assert proposer.filed.get(PROPOSE_AT), "never filed"
        assert joined and min(joined) == PROPOSE_AT + 1, f"joined at {joined}"
        assert provisional == [PROPOSE_AT + 1, PROPOSE_AT + 2], provisional
        assert full and min(full) == PROPOSE_AT + 1 + ms.MENU_LAG, full
        assert m[min(full)].entry(NEW_ID).scored_from == min(full)
        audits = [w.reveal("follower-a", k).get("join_audit") for k in range(FLIP + 2, LAST + 1)]
        assert all(a is None or a.get("differences") == [] for a in audits), audits
        return (f"filed e{PROPOSE_AT}, on the menu e{min(joined)} provisional, full from "
                f"e{min(full)}; the followers' join audits clean")

    def retired():
        m = menus()
        gone = [k for k, s in m.items() if FAILING_ID not in s.ids()]
        assert gone, f"{FAILING_ID} was never retired"
        rec = [r for r in m[gone[0]].retired if r.id == FAILING_ID]
        assert rec and rec[0].reason == ms.REASON_FAILING, rec
        return f"{FAILING_ID} retired at e{gone[0]} ({rec[0].reason})"

    def price():
        m = menus()
        prices = {k: s.entry(PRICED_ID).price_in for k, s in m.items() if PRICED_ID in s.ids()}
        moves = [k for k in sorted(prices)[1:] if prices[k] != prices[k - 1]]
        assert moves, f"{PRICED_ID}'s price never moved: {prices}"
        assert all(k % REFRESH == 0 for k in moves), f"moved outside a refresh epoch: {moves}"
        k = moves[0]
        return f"{PRICED_ID} {prices[k - 1]} -> {prices[k]} pico/token at e{k} (refresh)"

    def root_missed():
        for k in range(ROOT_DOWN, ROOT_BACK):
            assert w.reveal("root", k) is None, k
        # Proofs went on through the outage: every TD proved, and every
        # follower checked every proof, in each epoch the root was down and
        # the two after (the published set does not wait on the root).
        outage = range(ROOT_DOWN, ROOT_BACK + 2)
        for m in w.miners.values():
            assert all(k in m.menus and k not in m.no_proof for k in outage), (
                m.hotkey, sorted(m.no_proof))
        for s in states:
            if s == "root":
                continue
            logged = {x["epoch_id"]: x for x in w.logged(s)}
            for k in outage:
                x = logged[ids[k]]
                assert x["n_heads_valid"] == len(w.miner_specs) and not x["n_heads_unverifiable"], (
                    s, k)
        for k in range(ROOT_DOWN + ms.MENU_LAG, ROOT_BACK + ms.MENU_LAG):
            a, b = w.held("follower-a").held(k), w.held("follower-a").held(k - 1)
            assert a is not None and ms.menu_hash(a) == ms.menu_hash(ms.carry(b)), \
                f"e{k}: the menu did not carry"
        after = range(ROOT_BACK + 2, LAST + 1)
        for k in after:
            ra, rr = w.reveal("follower-a", k)["v2_state"], w.reveal("root", k)["v2_state"]
            skip = {"sync", "burn_in"}
            assert {x: v for x, v in ra.items() if x not in skip} == {
                x: v for x, v in rr.items() if x not in skip}, f"e{k} differs"
        return (f"down e{ROOT_DOWN}..e{ROOT_BACK - 1}; every TD proved and every follower "
                f"checked every proof through it; e{ROOT_DOWN + ms.MENU_LAG}.."
                f"e{ROOT_BACK + ms.MENU_LAG - 1} carried the menu; followers agree with the "
                f"root again from e{ROOT_BACK + 2}")

    def miner_restart():
        spec = w.miner_specs[1]
        m = w.miners[spec.hotkey]
        assert all(k in m.menus for k in range(MINER_RESTART, LAST + 1))
        return f"its TD restarted at e{MINER_RESTART} and proved every epoch after"

    def validator_restart():
        for k in range(FOLLOWER_BACK + 2, LAST + 1):
            if not ROOT_DOWN <= k < ROOT_BACK + 2:
                rb, rr = w.reveal("follower-b", k)["v2_state"], w.reveal("root", k)["v2_state"]
                skip = {"sync", "burn_in"}
                assert {x: v for x, v in rb.items() if x not in skip} == {
                    x: v for x, v in rr.items() if x not in skip}, f"e{k} differs"
        return f"stopped after e{FOLLOWER_STOP - 1}, back at e{FOLLOWER_BACK}, agrees with the root"

    def missed_first():
        # Agrees from F + 2 (the root's reveal of F is read from then; it
        # re-anchors on it and replays F + 1), outside the root's own outage.
        by, root_hk = w.weights(), w.validator_keys["root"][0]
        hk = w.validator_keys["follower-c"][0]
        assert w.reveal("follower-c", FLIP) is None
        assert w.reveal("follower-c", FLIP + 2)["v2_sync"]["reanchored"]["replayed"] == [
            ids[FLIP + 1]]
        checked = 0
        for k in range(FLIP + 2, LAST + 1):
            if ROOT_DOWN <= k < ROOT_BACK + 2:
                continue
            rc, rr = w.reveal("follower-c", k)["v2_state"], w.reveal("root", k)["v2_state"]
            assert {x: v for x, v in rc.items() if x != "sync"} == {
                x: v for x, v in rr.items() if x != "sync"}, f"e{k} differs"
            assert {u: x for u, x in by[k][hk].items() if x} == {
                u: x for u, x in by[k][root_hk].items() if x}, f"e{k}: other weights"
            checked += 1
        from tests import v4_world
        # It replays e{FLIP + 1} once, and once more the epoch after the
        # root's first epoch back: the root's first reveal after its outage
        # is a re-anchor point (read two epochs after it).
        replayed = v4_world.check_replayed_memos(w.v2_calls, {"follower-c+"},
                                                 root=("root", "root+"))
        assert replayed == {"follower-c+": [FLIP + 1, ROOT_BACK + 1]}, replayed
        return (f"missed e{FLIP}, back at e{FLIP + 1}, re-anchored and replayed e{FLIP + 1} "
                f"(with the root's memo) at e{FLIP + 2}, and e{ROOT_BACK + 1} on the root's "
                f"return; the root's state and weights in all {checked} epochs checked")

    def published():
        from fugal_subnet import memo_sync
        from tests import v4_world

        public = {k: w.reveal("root", k) for k in ids if w.reveal("root", k) is not None}
        for k in range(FLIP, LAST + 1):
            if k not in public:
                continue
            expect = v4_world.chain_view(w.chain, w.published_seed, w.pool, public, k)
            assert public[k]["memo_v4"]["published_digest"] == expect.view(k - 2).digest, k
        for s in states:
            v4_world.check_held(w.dirs[s] / ("state.json" + memo_sync.STATE_SUFFIX),
                                w.chain, w.published_seed, w.pool, public, LAST)
        # Every node that ran an epoch publishes one memo_v4 for it, the
        # epochs the root missed included (the set does not wait on it).
        for k in range(FLIP, LAST + 1):
            got = {json.dumps(w.reveal(s, k)["memo_v4"], sort_keys=True)
                   for s in states if w.reveal(s, k) is not None}
            assert len(got) == 1, f"e{k}: nodes publish different memo_v4"
        return (f"every root reveal names the set its public reveals fold to; every node holds "
                f"it at the end; one memo_v4 per epoch on every node that ran it, in all "
                f"{LAST - FLIP + 1} v4 epochs ({len(public) - (FLIP - FIRST)} with the root)")

    def weights():
        by = w.weights()
        for k in ids:
            vecs = {json.dumps(v, sort_keys=True) for v in by.get(k, {}).values()}
            assert len(vecs) <= 1, f"{ids[k]}: validators set different weights"
        return "one weight vector per epoch"

    check("every validator ran its epochs; every proof verified", no_errors)
    check("v3 -> v4 at the block", flip)
    check("one menu on every node and TD", same_menu)
    check("a proposal filed at k, provisional at k+1, full at k+3", proposal)
    check("a price move at a refresh epoch", price)
    check("a failing entry retired", retired)
    check("the root misses epochs", root_missed)
    check("a miner restart", miner_restart)
    check("a validator restart", validator_restart)
    check("a follower that misses the first v4 epoch agrees with the root again", missed_first)
    check("the published set", published)
    check("weights agree", weights)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", help="run in this directory and keep it")
    args = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.ERROR)
    os.environ.setdefault("FUGAL_ALLOW_ENV_DRIFT", "1")
    if args.keep:
        Path(args.keep).mkdir(parents=True, exist_ok=True)
        rows, _ = run(Path(args.keep))
    else:
        with tempfile.TemporaryDirectory(prefix="rehearsal-v4-") as d:
            rows, _ = run(Path(d))
    width = max(len(r.name) for r in rows)
    print(f"{'scenario'.ljust(width)}  result  detail")
    for r in rows:
        print(f"{r.name.ljust(width)}  {r.status:<6}  {r.detail}")
    failed = [r for r in rows if r.status == "FAIL"]
    print(f"\n{sum(r.status == 'PASS' for r in rows)} passed, {len(failed)} failed, "
          f"{sum(r.status == 'SKIP' for r in rows)} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
