"""Protocol v4's seat through the shipped loops (SEAT_MINIMAL_V4.md 3.3 to
3.8, 4.3, 4.4; the S4 rehearsal scenarios of section 11): `validator.main`
for a root and a follower and the miner's real `_run_epoch`, on the shared
stand-ins of tests/v4_world.py (chain, transport, store, provider), at the
design's constants (1,500 labels, 0.1 Brier points, 1,000 questions, 1.5
points, G = 3). The loops, the scenarios and the checks are in
tests/v4_world.py (`SeatLoop`, `seat_*_scenario`, `check_seat_*`), which
scripts/rehearsal_v4.py runs too.

The stub provider answers a (question, entry) right exactly when the
question's pinned embedding is positive on the entry's own coordinate, so a
head whose row for the entry weighs that coordinate predicts the outcome,
and a sharper weight predicts it better: whether a head beats R or another
head is set by its weights, not by luck. The follower is handed every
epoch's proofs, heads, answers and commitments in the reverse order of the
root's, so the two reach the same seat from different arrival orders.

1. `test_the_seat_is_filled_then_taken_by_an_odds_only_miner_and_copies_never_win`:
   an empty seat at activation is filled by the first head that beats R on
   1,500 labels and has answered 1,000 questions; an odds-only miner with
   sharper odds turns its scored questions on and takes the seat once it has
   answered 1,000 questions, including after missing an epoch (its run
   restarts); a lightly altered copy of it, committed later under another
   coldkey and answering every epoch, never passes it, including in the epoch
   it misses; an exact copy of the first holder under another coldkey is
   refused every epoch.
2. `test_a_holder_keeps_its_seat_through_succession_and_two_misses_then_loses_it`:
   a new entry joins with heads that have its row and heads that do not; the
   holder adds the row by succession and keeps the seat; it misses two epochs
   and keeps the seat, unpaid; it misses three and the seat goes at once to
   the latest qualifier.
3. `test_a_memorising_holder_is_refused_through_the_real_wiring`: nothing is
   handed to the loops. The release's published-set seed is half the pool
   (the half a sharp head is confident on); both validators and every TD
   bring the set forward from the root's reveals, the epochs' fresh draws
   spend the other half, and the holder, refused once its unseen labels are
   in, loses the seat on both nodes alike. A validator rebuilt from the
   root's reveals reaches the same set and the same evidence.

Every loop runs the memorisation check every epoch on the set the nodes
hold; in 1 and 2 the seed publishes the whole pool, so no head has an unseen
label and none is refused.
"""
from __future__ import annotations

import pytest

from tests import v4_world as w
from tests.test_menu_publish import LIST_BYTES, signer  # noqa: F401 - the fixture


@pytest.fixture
def _signature(signer):  # noqa: F811
    return signer.sign(LIST_BYTES)


def test_the_seat_is_filled_then_taken_by_an_odds_only_miner_and_copies_never_win(
        tmp_path, monkeypatch, _signature):
    loop = w.seat_take_scenario(tmp_path, monkeypatch, _signature)
    loop.check_agreement_and_rebuild()
    w.check_seat_fill(loop)
    w.check_seat_odds_only_take(loop)
    w.check_seat_copies(loop)


def test_a_holder_keeps_its_seat_through_succession_and_two_misses_then_loses_it(
        tmp_path, monkeypatch, _signature):
    loop = w.seat_hold_scenario(tmp_path, monkeypatch, _signature)
    loop.check_agreement_and_rebuild()
    w.check_seat_new_entry(loop)
    w.check_seat_succession(loop)
    w.check_seat_holder_head_from_root(loop)
    w.check_seat_standing(loop)


def test_a_memorising_holder_is_refused_through_the_real_wiring(tmp_path, monkeypatch,
                                                               _signature):
    loop = w.seat_memo_scenario(tmp_path, monkeypatch, _signature)
    loop.check_agreement_and_rebuild()
    w.check_seat_memo_refusal(loop)
    # A validator rebuilt from the root's reveals holds the window and seat
    # the verdict reads, and the published set the root's reveals fold to:
    # it reaches the same evidence.
    from fugal_subnet import epoch_plan, memo_v4
    from fugal_subnet import menu_publish as mp
    from fugal_subnet.benchmarks.slicer import derive_nonce, epoch_id_for_block
    from fugal_subnet.v2_epoch import _memo_verdicts
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals
    from neurons import validator

    last = w.MEMO_LAST
    held = mp.MenuBook(str(loop.dirs["root"] / "state.json") + ".menu").held(last)
    t = validator._v4_tables(mp.MenuRead(last, held, "carried"))
    rebuilt = rebuild_v2_from_reveals(str(loop.dirs["root"] / "epochs"), window_size=48,
                                      prices=t.prices, bench_category=t.bench_category)
    pub = w.chain_view(loop.chain, loop.seed, loop.pool, {k: loop.reveal(k) for k in loop.epochs},
                       last)
    nonce = derive_nonce(epoch_id_for_block(last), loop.chain.block_hash(last * w.BPE))
    memo = memo_v4.epoch_memo(pub, memo_v4.reserve_index(pub, loop.pool, last), nonce,
                              epoch_plan.mix_v4().targets)
    again = _memo_verdicts(memo, rebuilt.window, rebuilt.seat, last)
    assert {h: memo_v4.evidence_for_reveal(ev) for h, ev in again.items()} == (
        loop.reveal(last)["memo_v4"]["evidence"])
    assert memo_v4.memo_for_reveal(memo, again, now=last) == loop.reveal(last)["memo_v4"]


def test_a_miner_epoch_that_raises_fails_the_seat_loop_with_its_exception(
        tmp_path, monkeypatch, _signature):
    """The seat loop runs on the fail-fast clock (v4_world.Lockstep): an
    exception in the miners' epoch is the run's failure, not an error a
    validator logs while the run goes on."""
    boom = RuntimeError("the miners' epoch raised")
    spec: list = []

    def plan(e, lp):
        if e == w.SEAT_FIRST + 2:
            raise boom

    loop = w.SeatLoop(tmp_path, monkeypatch, spec, last=w.SEAT_FIRST + 6, plan=plan,
                      signature=_signature)
    spec.append((w.SEAT_A, "5Fcold-a", loop.head(loop.all_rows(w.SHARP), "a"), "on", 1))
    with pytest.raises(RuntimeError) as got:
        loop.run()
    assert got.value is boom
