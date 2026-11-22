"""The claim the pick-cost table rests on, reproduced from committed data
(W13; SEAT_MINIMAL_V4.md 3.4 and section 1, the earlier seat design's 8.1).

The claim: a cost per entry measured on random question-entry pairs (the
exploration means, the call cost) under-states what an entry costs on the
questions routers actually send it about twofold, while means over graded
picked cells predict the real cost of picks within about 10%. Here every
published v3 reveal of the design's range is left out in turn: each table is
built from the others and priced on the held-out reveal's picked cells
(`scripts/build_menu_v4_seed.py accuracy`, from the committed picks file
alone; completion tokens at v3 menu B's prices, since a v3 reveal publishes
no prompt tokens for a cell). The ratio is real over predicted spend: 1.00
is exact. No network.

The design's figures (37 reveals e00026466..e00026502; champion's picks /
all heads' picks): pinned token means 2.72 / 2.70, exploration mean per
entry 2.04 / 1.91, per entry and category 1.99 / 1.80, graded cells per
entry 0.89 / 1.08, per entry and category 1.06 / 1.06. Reproduced below:
the graded-cell tables to within 0.02; the exploration and pinned tables
come out higher here (2.36 / 2.04, 2.07 / 1.38, 3.15 / 3.09), which only
strengthens the claim that exploration under-states the cost of picks.
"""
from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PICKS = ROOT / "data" / "observations" / "menu_v4_seed_picks_e00026466-e00026509.json"


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_menu_v4_seed", ROOT / "scripts" / "build_menu_v4_seed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _builder()


def _rounded(got) -> dict:
    return {who: {t: round(float(r), 3) for t, r in ratios.items()}
            for who, ratios in got["ratio"].items()}


def test_leave_one_epoch_out_on_the_designs_37_reveals():
    got = builder.accuracy(str(PICKS), first="e00026466", last="e00026502")
    assert got["epochs"] == ("e00026466", "e00026502", 37)
    assert got["picks"] == {"champion": 8_960, "heads": 149_071}
    assert _rounded(got) == {
        "champion": {"pinned": 3.152, "call_entry": 2.356, "call_category": 2.071,
                     "cells_entry": 0.876, "cells_category": 1.065,
                     "pick_entry": 0.883, "pick_category": 1.058},
        "heads": {"pinned": 3.091, "call_entry": 2.042, "call_category": 1.382,
                  "cells_entry": 1.07, "cells_category": 1.068,
                  "pick_entry": 1.079, "pick_category": 1.066},
    }
    r = got["ratio"]
    for who in ("champion", "heads"):
        # Exploration's call cost under-states picks about twofold...
        assert r[who]["call_entry"] > 2
        # ...the W13 table, per entry and category, is within 10% of them.
        assert abs(float(r[who]["pick_category"]) - 1) < 0.10
    # The design's graded-cell figures, to within 0.02.
    for who, entry, category in (("champion", 0.89, 1.06), ("heads", 1.08, 1.06)):
        assert abs(float(r[who]["cells_entry"]) - entry) <= 0.02
        assert abs(float(r[who]["cells_category"]) - category) <= 0.02


def test_leave_one_epoch_out_on_all_44_reveals_the_seed_is_built_from():
    got = builder.accuracy(str(PICKS))
    assert got["epochs"] == ("e00026466", "e00026509", 44)
    rounded = _rounded(got)
    assert rounded["champion"]["pick_category"] == 1.112
    assert rounded["heads"]["pick_category"] == 1.053
    assert rounded["champion"]["call_entry"] == 2.447
    assert rounded["heads"]["call_entry"] == 2.056


def test_the_accuracy_report_prints_from_the_committed_file(capsys):
    assert builder.main(["accuracy", "--first", "e00026466", "--last", "e00026502"]) == 0
    out = capsys.readouterr().out
    assert "37 reveals" in out and "pick_category 1.058" in out
