"""A validator rebuilt from reveals across the v3 flip holds the live state.

v3 review, 2026-11-16 (findings 1 and 2):
- the startup rebuild priced every replayed reveal at v2's table, so v3 cells
  stayed NaN and the next epoch raised "an answered question has no cell";
- the replay carried v2 epochs into a v3 window, where the live epoch had
  emptied the window at the flip (fixed in 498e7e8; tested there by
  test_v3_epoch.test_a_rebuild_across_the_v3_flip_keeps_only_the_v3_epochs,
  and here end to end with the reviewer's scenario: the next epoch's seat,
  weights and consensus view).
"""
import dataclasses
import json

import numpy as np
import pytest

from fugal_subnet.v2_epoch import seat_params_from_config
from fugal_subnet.v2_rebuild import _reveal_protocol, rebuild_v2_from_reveals
from fugal_subnet.v2_sync import consensus_view
from tests.test_v2_epoch import CATEGORY, PRICES, always, oracle, run
from tests.test_v2_rebuild import reveal_of, write

V3 = {"protocol": 3, "seat": "net_value"}


def _params(protocol):
    return dataclasses.replace(seat_params_from_config(protocol_version=protocol),
                               min_paired_questions=60)


def _live(epochs, pols, rules_of):
    window = seat = copy = None
    rules = None
    reveals = []
    for e in range(epochs):
        er = rules_of(e)
        r = run(e, pols(e), window=window, seat=seat, copy=copy,
                params=_params(3 if er else 2), blocks={5: 9, 6: 1},
                rules=er, state_rules=rules)
        window, seat, copy, rules = r.window, r.decision.state, r.copy_state, er
        reveals.append(json.loads(json.dumps(reveal_of(e, r, pols(e)))))
    return window, seat, copy, rules, reveals


def test_the_rebuild_drops_the_epochs_of_the_protocol_before_the_flip(tmp_path):
    flip = 10

    def pols(e):
        # uid 5 routes perfectly before the flip and badly after; uid 6 the reverse.
        if e < flip:
            return {5: oracle, 6: always("m/cheap")}
        return {5: always("m/mid"), 6: always("m/mid")}

    def rules_of(e):
        return V3 if e >= flip else None

    window, seat, copy, rules, reveals = _live(flip + 4, pols, rules_of)
    write(tmp_path, reveals)
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=PRICES, bench_category=CATEGORY)
    assert sorted(rb.window.slices) == sorted(window.slices) == [11, 12, 13, 14]
    assert rb.seat == seat and rb.extra.get("rules") == V3

    e = flip + 4
    a = run(e, pols(e), window=window, seat=seat, copy=copy, params=_params(3),
            blocks={5: 9, 6: 1}, rules=V3, state_rules=rules)
    b = run(e, pols(e), window=rb.window, seat=rb.seat, copy=copy, params=_params(3),
            blocks={5: 9, 6: 1}, rules=V3, state_rules=rb.extra.get("rules"))
    assert (b.decision.event, b.decision.champion) == (a.decision.event, a.decision.champion)
    assert dict(zip(b.uids, b.weights)) == dict(zip(a.uids, a.weights))
    assert consensus_view(dict(b.reveal, epoch_id="x")) == consensus_view(dict(a.reveal, epoch_id="x"))


def test_a_reanchor_on_a_reference_from_before_the_flip_holds_the_live_state(tmp_path):
    """The re-anchor path (validator._v2_sync_inner -> v2_sync.resync): the
    reference's snapshot is a v2 epoch k, and this validator's own reveals
    after k cross the flip. The live validator reset the seat at the flip and
    kept the v3 epochs; the re-anchored one must hold the same state."""
    from fugal_subnet.v2_sync import resync

    flip = 10

    def pols(e):
        if e < flip:
            return {5: oracle, 6: always("m/cheap")}
        return {5: always("m/mid"), 6: always("m/mid")}

    def rules_of(e):
        return V3 if e >= flip else None

    window, seat, copy, rules, reveals = _live(flip + 4, pols, rules_of)
    write(tmp_path, reveals)
    by = {int(rv["epoch_id"][1:]): (str(tmp_path / rv["epoch_id"] / "reveal.json"), rv)
          for rv in reveals}
    before = max(by) + 1

    # A reference snapshot from before the flip is refused: the caller keeps
    # its state and gap, and waits.
    k = flip                                       # chain epoch e00000010, the last v2 one
    with pytest.raises(ValueError, match="wait for a reference epoch after the protocol change"):
        resync({j: by[j] for j in by if j <= k}, {j: by[j] for j in by if j > k}, k, before,
               window_size=48, prices=PRICES, bench_category=CATEGORY)

    # Once the reference has published a v3 epoch, the re-anchor holds the live state.
    k = flip + 2
    rs = resync({j: by[j] for j in by if j <= k}, {j: by[j] for j in by if j > k}, k, before,
                window_size=48, prices=PRICES, bench_category=CATEGORY)
    rb = rs.rebuilt
    assert sorted(rb.window.slices) == sorted(window.slices)
    e = flip + 4
    a = run(e, pols(e), window=window, seat=seat, copy=copy, params=_params(3),
            blocks={5: 9, 6: 1}, rules=V3, state_rules=rules)
    b = run(e, pols(e), window=rb.window, seat=rb.seat, copy=copy, params=_params(3),
            blocks={5: 9, 6: 1}, rules=V3, state_rules=rb.extra.get("rules"))
    assert (b.decision.event, b.decision.champion) == (a.decision.event, a.decision.champion)
    assert consensus_view(dict(b.reveal, epoch_id="x")) == consensus_view(dict(a.reveal, epoch_id="x"))


def test_each_reveal_is_priced_at_its_own_protocols_table(tmp_path):
    def pols(e):   # champion uid 6 seated first; uid 5 becomes better from epoch 6
        return {5: always("m/mid") if e < 6 else oracle, 6: always("m/mid")}

    window, seat, copy, _, reveals = _live(8, pols, lambda e: V3)
    write(tmp_path, reveals)
    # The table the validator held for v2: it prices none of the v3 entries
    # but one (51 of 53 on the real menus).
    v2_like = {"m/cheap": PRICES["m/cheap"]}
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=v2_like,
                                 bench_category=CATEGORY,
                                 tables={2: (v2_like, CATEGORY), 3: (PRICES, CATEGORY)})
    e0 = max(window.slices)
    assert np.array_equal(rb.window.slices[e0].grades, window.slices[e0].grades, equal_nan=True)
    assert np.array_equal(rb.window.slices[e0].spends, window.slices[e0].spends, equal_nan=True)
    a = run(8, pols(8), window=window, seat=seat, copy=copy, params=_params(3),
            blocks={5: 9, 6: 1}, rules=V3, state_rules=V3)
    b = run(8, pols(8), window=rb.window, seat=rb.seat, copy=copy, params=_params(3),
            blocks={5: 9, 6: 1}, rules=V3, state_rules=V3)
    assert (b.decision.event, b.decision.champion) == (a.decision.event, a.decision.champion)


def test_a_reveals_protocol_is_its_rules_else_its_version():
    assert _reveal_protocol({"v2_state": {"rules": {"protocol": 3}}, "protocol_version": 2}) == 3
    assert _reveal_protocol({"protocol_version": 2}) == 2
    assert _reveal_protocol({}) == 2


def test_the_validator_rebuilds_and_reanchors_with_every_protocols_table():
    import pathlib
    import types

    from neurons.validator import _rebuild_tables

    src = (pathlib.Path(__file__).resolve().parents[1] / "neurons" / "validator.py").read_text()
    start = src[src.index("rebuilt = rebuild_v2_from_reveals("):]
    assert "tables=_rebuild_tables(epoch_inputs, config_module, epoch_inputs_v3b)" in start[:start.index(")\n")]
    step = src[src.index("synced = _v2_sync_step("):]
    assert "tables=_rebuild_tables(epoch_inputs, config_module, epoch_inputs_v3b)" in step[:step.index(")\n")]
    assert "bench_category=bench_category, tables=tables)" in src   # down to resync

    cfg = types.SimpleNamespace(BENCHMARK_CATEGORY_V2={"a": "x"}, BENCHMARK_CATEGORY_V3={"b": "y"})
    inputs = {1: (0, 0, {"v1": 1}), 2: (0, 0, {"v2": 2}), 3: (0, 0, {"v3": 3})}
    assert _rebuild_tables(inputs, cfg) == {2: ({"v2": 2}, {"a": "x"}),
                                           3: ({"v3": 3}, {"b": "y"})}
