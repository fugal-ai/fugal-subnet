"""The live menu keeps stepping in protocol v5.

A v5 epoch's verified proofs answer v5 pool questions (`arena_*` exam cells
and coverage pairs), which the v4 gold never names. The menu step's inputs
are built with the categories `neurons.validator._v5_category_of` gives (each
question's RouterArena dataset, the exam's stratum). Found by the real-chain
rehearsal: with the v4 gold's categories every v5 epoch raised
"question 'arena_…' has no category" and the live menu froze (no joins, no
price refresh, no token means, no retirement)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fugal_subnet import menu_inputs as mi
from fugal_subnet import menu_state as ms
from neurons import validator
from tests import menu_fixtures as mf
from tests.test_menu_inputs import HIGH, LOW, TWO, _pair, _r, _read, _vp

DATASETS = ("MMLUPro", "OpenTDB", "QANTA", "SuperGLUE-CausalReasoning")


def _qid(i: int) -> str:
    return f"arena_{DATASETS[i % 4]}_{i:016x}"


def _v5_inputs(n: int = 400):
    """The validator's _V5Inputs as far as the categories read it: each pool
    item's metadata names its RouterArena dataset."""
    gold = {_qid(i): {"question_id": _qid(i), "metadata": {"ra_dataset": DATASETS[i % 4]}}
            for i in range(n)}
    return SimpleNamespace(gold=gold)


def _v5_epoch_world():
    """Epoch 100's v5 proofs (exam cells and coverage pairs on arena_* questions)
    on the menu of epoch 100, an entry provisional in it, and the step from
    the menu of epoch 101."""
    epoch_state = mf.provisional(mf.state(epoch=100))
    p = epoch_state.entry("lab-c/model-new@high")
    start = ms.step(epoch_state, mf.inputs(epoch=99), mf.PARAMS)
    reads = [_read("lab-a/model-one", ("host-a/fp8", True, 120_000, 410_000, 100_000),
                   ("host-b/fp8", True, 110_000, 420_000, 131072)),
             _read("lab-b/model-two", ("host-c", True, 2_000_000, 10_000_000, 65536))]
    proofs = []
    for i in range(5):
        # coverage (exploration) calls, then exam cells (scored results)
        res = [_r(_qid(i * 10 + j), (HIGH, LOW, TWO, p.id)[j % 4], explore=True,
                  failed=(i == 4)) for j in range(12)]
        res += [_r(_qid(300 + j), (HIGH, LOW)[j % 2]) for j in range(4)]
        proofs.append(_vp(f"5Fhk{i}", f"ck{i}", res, reads=reads))
    pooled = [_pair(_qid(j), (HIGH, LOW, TWO, p.id)[j % 4]) for j in range(48)]
    return start, epoch_state, proofs, pooled


def test_the_v4_categories_name_no_v5_question():
    _start, epoch_state, proofs, pooled = _v5_epoch_world()
    v4_like = {f"q{i:03d}": "knowledge" for i in range(400)}
    with pytest.raises(ValueError, match="has no category"):
        mi.summarize_epoch(100, epoch_state, proofs, pooled=pooled, category_of=v4_like,
                           counted_heads=())


def test_v5_categories_cover_every_v5_question_and_are_its_dataset():
    cats = validator._v5_category_of(_v5_inputs())
    assert cats[_qid(1)] == "OpenTDB" and cats[_qid(3)] == "SuperGLUE-CausalReasoning"
    assert all(0 < len(c) <= ms.PICK_CATEGORY_MAX_CHARS for c in cats.values())
    with pytest.raises(RuntimeError):
        validator._v5_category_of(None)


def test_the_v5_menu_step_sees_one_category():
    # The menu's pick stats allow at most PICK_MAX_CATEGORIES categories, and
    # RouterArena has ~25 datasets: v5 prices a pick per entry, so the menu
    # step sees every v5 question in one category (the seat keeps the datasets).
    cats = validator._v5_menu_category_of(_v5_inputs())
    assert set(cats) == set(validator._v5_category_of(_v5_inputs()))
    assert set(cats.values()) == {validator.V5_MENU_CATEGORY}
    assert len(set(cats.values())) <= ms.PICK_MAX_CATEGORIES


def test_the_menu_step_runs_on_a_v5_epoch():
    start, epoch_state, proofs, pooled = _v5_epoch_world()
    cats = validator._v5_menu_category_of(_v5_inputs())
    summary = mi.summarize_epoch(100, epoch_state, proofs, pooled=pooled, category_of=cats,
                                 counted_heads={"1f" * 32})
    window = [mi.summary_from_dict(dict(mi.summary_to_dict(summary), epoch=e))
              for e in range(90, 100)] + [summary]
    inputs = mi.step_inputs(start, epoch_state, window, proofs, epoch=100, nonce=mf.NONCE,
                            lab_list=mf.ALLOW, params=mf.PARAMS, category_of=cats)
    # Every input the step reads is built: host reads, counts, tokens, gates.
    assert inputs.host_reads == window[-1].votes
    assert {c.entry_id: c.pairs for c in inputs.explore_counts}[HIGH] > 0
    assert {t.entry_id for t in inputs.token_stats} >= {HIGH, LOW}
    assert {c.category for t in inputs.token_stats for c in t.cells} <= {validator.V5_MENU_CATEGORY}
    (g,) = inputs.gates
    assert g.entry_id == "lab-c/model-new@high" and len(g.coldkeys) == 5
    out = ms.step_detail(start, inputs, mf.PARAMS)
    # The provisional entry is decided (made full, or refused at a gate): the
    # menu moves on in v5 instead of freezing.
    assert any(e.subject == "lab-c/model-new@high" and e.kind in ("full", "gate_failed")
               for e in out.events)
    # Deterministic: the same epoch gives the same step.
    again = ms.step_detail(start, mi.step_inputs(
        start, epoch_state, window, proofs, epoch=100, nonce=mf.NONCE, lab_list=mf.ALLOW,
        params=mf.PARAMS, category_of=dict(reversed(list(cats.items())))), mf.PARAMS)
    assert again == out
