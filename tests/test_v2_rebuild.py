"""Protocol v2 state rebuilt from published reveals equals the live state (I1).

docs/INVARIANTS.md, "I1 — v2 state is rebuilt from reveals: snapshot + replay".
"""
import json
import random

import numpy as np

from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals
from tests.test_v2_epoch import CATEGORY, MODELS, PRICES, always, oracle, proof, questions, run


def reveal_of(epoch, result, policies):
    """The v2 reveal the validator would publish for this epoch."""
    model_index = {m: i for i, m in enumerate(MODELS)}
    decisions = {}
    for uid, pol in policies.items():
        rs = sorted(proof(epoch, pol).scored_results, key=lambda r: r.question_id)
        decisions[str(uid)] = [model_index[r.routed_model] for r in rs]
    return {
        "epoch_id": f"e{epoch + 1:08d}", "protocol_version": 2,   # run() is chain epoch e+1
        "questions": questions(epoch), "models": MODELS,
        "routing_decisions": decisions, **result.reveal,
    }


def live(epochs, schedule):
    window = seat = copy = None
    reveals = []
    r = None
    for e in range(epochs):
        pols = schedule(e)
        r = run(e, pols, window=window, seat=seat, copy=copy)
        window, seat, copy = r.window, r.decision.state, r.copy_state
        reveals.append(json.loads(json.dumps(reveal_of(e, r, pols))))   # as published
    return r, reveals


def schedule(e):
    pols = {5: always("m/strong"), 6: oracle, 7: always("m/cheap")}
    if e % 7 == 3:
        del pols[6]                       # 6 misses now and then
    return pols


def write(tmp_path, reveals):
    for rv in reveals:
        d = tmp_path / rv["epoch_id"]
        d.mkdir()
        (d / "reveal.json").write_text(json.dumps(rv))


def same_window(a, b):
    assert sorted(a.slices) == sorted(b.slices)
    for e in a.slices:
        x, y = a.slices[e], b.slices[e]
        assert x.question_ids == y.question_ids and x.categories == y.categories
        assert x.hard == y.hard
        assert np.array_equal(x.grades, y.grades, equal_nan=True)
        assert np.array_equal(x.spends, y.spends, equal_nan=True)
    assert sorted(a.heads) == sorted(b.heads)
    for h in a.heads:
        assert sorted(a.heads[h]) == sorted(b.heads[h])
        for e in a.heads[h]:
            assert np.array_equal(a.heads[h][e].routed, b.heads[h][e].routed)
            assert np.array_equal(a.heads[h][e].failed, b.heads[h][e].failed)


def test_the_rebuild_equals_the_live_state(tmp_path):
    last, reveals = live(30, schedule)
    write(tmp_path, reveals)
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=PRICES, bench_category=CATEGORY)
    same_window(rb.window, last.window)
    assert rb.seat == last.decision.state
    assert rb.owners == last.owners
    assert rb.newest_epoch_id == "e00000030" and rb.epochs_replayed == 30
    from fugal_subnet.copy_rule import from_rebuild
    assert from_rebuild(rb) == last.copy_state


def test_the_rebuild_keeps_only_the_window(tmp_path):
    last, reveals = live(12, schedule)
    write(tmp_path, reveals)
    rb = rebuild_v2_from_reveals(tmp_path, window_size=5, prices=PRICES, bench_category=CATEGORY)
    assert sorted(rb.window.slices) == list(range(8, 13))   # chain epochs 1..12, last 5


def test_the_rebuild_does_not_depend_on_reveal_order():
    _, reveals = live(10, schedule)
    import tempfile

    paths = []
    with tempfile.TemporaryDirectory() as d:
        for rv in reveals:
            p = f"{d}/{rv['epoch_id']}.json"
            with open(p, "w") as f:
                json.dump(rv, f)
            paths.append(p)
        a = rebuild_v2_from_reveals(paths, window_size=48, prices=PRICES, bench_category=CATEGORY)
        random.Random(3).shuffle(paths)
        b = rebuild_v2_from_reveals(paths, window_size=48, prices=PRICES, bench_category=CATEGORY)
    same_window(a.window, b.window)
    assert a.seat == b.seat and a.owners == b.owners and a.burn_in == b.burn_in


def test_a_rebuilt_validator_continues_exactly_like_a_live_one(tmp_path):
    # Live for 20 epochs; then one validator continues from its own state and
    # another from a rebuild of the published reveals. Epoch 20 must agree.
    last, reveals = live(20, schedule)
    write(tmp_path, reveals)
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=PRICES, bench_category=CATEGORY)
    from fugal_subnet.copy_rule import from_rebuild
    a = run(20, schedule(20), window=last.window, seat=last.decision.state,
            copy=last.copy_state)
    b = run(20, schedule(20), window=rb.window, seat=rb.seat, copy=from_rebuild(rb))
    assert (a.uids, a.weights) == (b.uids, b.weights)
    assert a.reveal == b.reveal


def test_extra_state_is_carried_verbatim(tmp_path):
    _, reveals = live(3, schedule)
    reveals[-1]["v2_state"]["first_commit"] = {"h5": 123}
    write(tmp_path, reveals)
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=PRICES)
    assert rb.extra["first_commit"] == {"h5": 123} and "copy_rule" in rb.extra


def test_no_v2_reveals_means_empty_state(tmp_path):
    d = tmp_path / "e00000001"
    d.mkdir()
    (d / "reveal.json").write_text(json.dumps({"epoch_id": "e00000001"}))   # a v1 reveal
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=PRICES)
    assert rb.seat.champion is None and not rb.window.slices


def test_the_frame_rebuild_applies_v2_mix_weights(tmp_path, monkeypatch):
    from fugal_subnet import reference_frame
    from fugal_subnet.reference_frame import (
        Observation,
        accumulate_exploration,
        aggregate_observations,
        rebuild_from_reveals,
    )

    monkeypatch.setattr(reference_frame, "require_identity", lambda _x: None)
    exploration = [
        {"question_id": "qa", "model": "m/a", "correct": True, "prompt_tokens": 1,
         "completion_tokens": 10, "coldkey": "c1", "failed": False},
        {"question_id": "qb", "model": "m/a", "correct": False, "prompt_tokens": 1,
         "completion_tokens": 90, "coldkey": "c1", "failed": False},
    ]
    rv = {"epoch_id": "e00000001", "protocol_version": 2, "exploration": exploration,
          "frame_mix_weights": {"easy": 0.5, "hard": 3.0},
          "explore_benchmarks": {"qa": "easy", "qb": "hard"}}
    d = tmp_path / "e00000001"
    d.mkdir()
    (d / "reveal.json").write_text(json.dumps(rv))
    rebuilt = rebuild_from_reveals(tmp_path)
    obs = [Observation(e["question_id"], e["model"], e["correct"], 1, e["completion_tokens"],
                       coldkey="c1") for e in exploration]
    keyed = aggregate_observations(obs, with_question=True)
    live_frame = accumulate_exploration(None, [k[1:] for k in keyed],
                                        weights=[0.5 if k[0] == "qa" else 3.0 for k in keyed])
    assert rebuilt.to_dict() == live_frame.to_dict()
    assert rebuilt.frontier_accuracy("m/a") == 0.5 / 3.5


def test_a_malformed_reveal_is_refused_by_name_not_crashed_on(tmp_path):
    """A published reveal is someone else's file (I2). Truncated routing,
    an index outside the model space, a missing cells list or a non-list
    `models` used to surface as KeyError/ValueError tracebacks (or, for a
    string `models`, as silent acceptance) at validator startup."""
    import pytest

    _, reveals = live(3, schedule)
    cases = {
        "routing_decisions truncated": lambda rv: rv["routing_decisions"].update(
            {k: v[:5] for k, v in rv["routing_decisions"].items()}),
        "index outside the model space": lambda rv: rv["routing_decisions"].update(
            {k: [99] * len(v) for k, v in rv["routing_decisions"].items()}),
        "missing shared_grades": lambda rv: rv.pop("shared_grades"),
        "models not a list": lambda rv: rv.update(models="m/cheap"),
        "epoch_id not numeric": lambda rv: rv.update(epoch_id="exx"),
    }
    for name, mutate in cases.items():
        d = tmp_path / name.replace(" ", "_")
        d.mkdir()
        rs = json.loads(json.dumps(reveals))
        mutate(rs[1])
        write(d, rs)
        with pytest.raises(ValueError, match="reveal e0000000|epoch_id") as e:
            rebuild_v2_from_reveals(d, window_size=48, prices=PRICES, bench_category=CATEGORY)
        assert "e00000002" in str(e.value) or "exx" in str(e.value), (name, str(e.value))


def test_the_rebuild_carries_failed_calls(tmp_path):
    """Every other test's heads answer every call, so a rebuild that dropped
    the reveal's `failed` lists would pass them. Here UID 6 fails a third of
    its calls in odd epochs: a failed call grades 0 and costs 0 in the window,
    so the rebuilt window must mark exactly the same calls failed."""
    from tests.test_v2_epoch import H, Q

    pols = {5: always("m/strong"), 6: oracle, 7: always("m/cheap")}
    window = seat = copy = None
    reveals = []
    for e in range(6):
        failed = {6: range(0, Q, 3)} if e % 2 else {}
        r = run(e, pols, window=window, seat=seat, copy=copy, failed=failed)
        window, seat, copy = r.window, r.decision.state, r.copy_state
        reveals.append(json.loads(json.dumps(reveal_of(e, r, pols))))
    assert len(r.reveal["failed"]["6"]) == Q // 3                    # epoch 5 is odd
    assert sum(int(he.failed.sum()) for he in window.heads[H(6)].values()) == 3 * (Q // 3)
    write(tmp_path, reveals)
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=PRICES, bench_category=CATEGORY)
    same_window(rb.window, window)
    from fugal_subnet.champion import window_accuracy
    assert window_accuracy(rb.window, H(6)) == window_accuracy(window, H(6))
