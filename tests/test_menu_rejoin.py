"""Protocol v4: an id that retires and joins again starts at revision 1
again, so its pairs are keyed (id, rev) exactly as the first run's were.
They never meet in one window because the window the step's statistics read
is shorter than the gap between the two runs (docs/decisions/LIVE_MENU_V4.md,
4.4, "an id that comes back"; FAST_CROWN_V4.md 18).

The last pair of the first run is in the epoch before the menu without it
(R - 1, the record written at R). A join delta refuses the id while its join
epoch is before R + PROPOSAL_COOLDOWN_EPOCHS, so the second run's first label
is at R + cooldown at the earliest: two pairs of the two runs are at least
cooldown + 1 = 49 epochs apart, and the loop's window is
CHAMPION_WINDOW_EPOCHS = 48.
"""
from __future__ import annotations

import dataclasses

import pytest

from fugal_subnet import chain_proposals as cp
from fugal_subnet import config, menu_inputs
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx

ID = "lab-a/model-one@high"


def test_the_gap_is_cooldown_plus_one_and_the_window_is_inside_it():
    assert ms.rejoin_gap_epochs() == ms.PROPOSAL_COOLDOWN_EPOCHS + 1 == 49
    assert config.CHAMPION_WINDOW_EPOCHS == 48 <= ms.rejoin_gap_epochs()
    menu_inputs.check_summary_window(48)
    menu_inputs.check_summary_window(49)
    with pytest.raises(ValueError, match="49"):
        menu_inputs.check_summary_window(50)
    shorter = dataclasses.replace(fx.PARAMS, cooldown_epochs=40)
    with pytest.raises(ValueError):
        menu_inputs.check_summary_window(48, shorter)


def _summary(epoch, pairs):
    cell = menu_inputs.PairCell(ID, 1, "math", False, pairs, 0, 1000 * pairs, 9000 * pairs)
    return menu_inputs.EpochSummary(epoch, (), (), (cell,), ())


def test_the_runs_of_an_id_never_share_a_window_of_the_loops_size():
    """The tightest case: the first run's last pair at R - 1, the second
    run's join at R + cooldown, the earliest a delta admits it."""
    R = 200
    joined = R + ms.PROPOSAL_COOLDOWN_EPOCHS
    base = fx.wide_state(epoch=joined, retired=(ms.Retired(ID, R, ms.REASON_GONE, joined),))
    base = dataclasses.replace(base, entries=tuple(e for e in base.entries if e.id != ID))
    early = dataclasses.replace(base, epoch=joined - 1)
    doc, raw = fx.document("lab-a/model-one", "high", hosts=("host-a/fp8",), proposer="hk")
    for b, verdict in ((early, "refused:entry_cooldown"), (base, cp.JOINED)):
        read = cp.ChainRead(epoch=b.epoch, block=0, block_hash="ab" * 32,
                            proposals=(("hk", cp.proposal_hash(raw)),), coldkey_of={"hk": "ck"},
                            serving=frozenset({"hk"}), reveal=None)
        d = cp.admit(b, read, {cp.proposal_hash(raw): raw},
                     {cp.proposal_hash(raw): fx.listing()}, fx.ALLOW, fx.PARAMS)
        assert d.considered[0].verdict == verdict
    again = fx.provisional(base, "lab-a/model-one", "high")
    assert again.entry(ID).rev == 1 and again.entry(ID).joined_epoch == joined
    summaries = [_summary(R - 1, 7), _summary(joined, 60)]
    window = [s for s in summaries if joined - config.CHAMPION_WINDOW_EPOCHS < s.epoch <= joined]
    stats = menu_inputs.token_stats(again, window, through=joined)
    assert sum(c.pairs for s in stats for c in s.cells) == 60      # the first run's are not read


def test_a_validator_whose_window_is_longer_than_the_gap_does_not_start():
    from neurons import validator

    assert validator._v4_window_refusal(48, fx.PARAMS) is None
    assert "49" in validator._v4_window_refusal(60, fx.PARAMS)
