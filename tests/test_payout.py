"""Protocol v2 payout: 85/10/5 across the last three champions.

docs/INVARIANTS.md, "I3 — seats, not shares".
"""
import pytest

from fugal_subnet.config import PAYOUT_SEAT_SHARES
from fugal_subnet.payout import seat_weights, validate_seat_shares

BURN = 0
SH = (0.85, 0.10, 0.05)


def w(champion, history, serving, eligible=None, paid=1.0):
    eligible = set(serving) if eligible is None else set(eligible)
    uids, ws = seat_weights(champion, history, serving, eligible,
                            shares=SH, paid_fraction=paid, burn_uid=BURN)
    assert abs(sum(ws) - 1.0) < 1e-12
    return dict(zip(uids, ws))


def test_three_live_seats_split_85_10_5():
    assert w("c", ["p1", "p2"], {"c": 5, "p1": 6, "p2": 7}) == pytest.approx(
        {5: 0.85, 6: 0.10, 7: 0.05})


def test_a_seat_whose_head_is_not_live_pays_the_champion():
    assert w("c", ["p1", "p2"], {"c": 5, "p2": 7}) == pytest.approx({5: 0.95, 7: 0.05})


def test_a_seat_that_fails_the_gates_pays_the_champion():
    got = w("c", ["p1", "p2"], {"c": 5, "p1": 6, "p2": 7}, eligible={"c", "p1"})
    assert got == pytest.approx({5: 0.90, 6: 0.10})


def test_no_champion_burns_the_champion_share_and_the_dead_seats():
    got = w(None, ["p1", "p2"], {"p1": 6})
    assert got == pytest.approx({6: 0.10, BURN: 0.90})


def test_nothing_at_all_burns_everything():
    assert w(None, [], {}) == {BURN: 1.0}


def test_one_hotkey_keeps_every_seat_it_holds():
    got = w("c", ["p1", "p2"], {"c": 5, "p1": 5, "p2": 9})
    assert got == pytest.approx({5: 0.95, 9: 0.05})


def test_extra_registrations_buy_nothing():
    # Eight UIDs serving eight heads that never won a seat: only the seats pay.
    serving = {"c": 5, **{f"farm{i}": 100 + i for i in range(8)}}
    assert w("c", [], serving) == {5: 1.0}


def test_only_three_seats_exist():
    got = w("c", ["p1", "p2", "p3", "p4"], {"c": 1, "p1": 2, "p2": 3, "p3": 4, "p4": 5})
    assert set(got) == {1, 2, 3}


def test_the_frame_confidence_share_burns_as_in_v1():
    # Seat 2 is unfilled, so its share goes to the champion; half of it all burns.
    got = w("c", ["p1"], {"c": 5, "p1": 6}, paid=0.5)
    assert got == pytest.approx({5: 0.45, 6: 0.05, BURN: 0.5})


def test_the_champion_listed_in_its_own_history_is_one_seat():
    got = w("c", ["c", "p1"], {"c": 5, "p1": 6})
    assert got == pytest.approx({5: 0.85 + 0.05, 6: 0.10})


def test_a_seat_served_by_the_burn_uid_is_not_paid_to_it_as_a_miner():
    # The burn UID is never a paid seat; that seat's share goes to the champion.
    got = w("c", ["p1"], {"c": 5, "p1": BURN})
    assert got == {5: 1.0}


@pytest.mark.parametrize("bad", [(), (0.5, 0.6), (0.85, -0.05, 0.2), (0.05, 0.10, 0.85)])
def test_bad_seat_shares_are_refused(bad):
    with pytest.raises(ValueError):
        validate_seat_shares(bad)


def test_the_shipped_shares_are_valid():
    validate_seat_shares(PAYOUT_SEAT_SHARES)
