"""The protocol v2 flip rehearsal (scripts/rehearsal_v2.py, dress_rehearsal v2_flip).

Two halves. The in-process harness runs here in full: whatever of v2 is on the
branch must hold across the flip; every v2 piece is on the branch, so nothing
may be skipped (a skip means a rename or a break, and fails the run).
The chain scenario needs Docker, so its CHECKS are exercised here on synthetic
epoch rows shaped like what the shipped validator publishes: a correct run
passes every row, and a run where the validators diverge is caught.
"""
from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: the module's dataclasses resolve their own
    # module by name (from __future__ import annotations).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_offline_rehearsal_has_no_failures():
    rv2 = _load("rehearsal_v2")
    report = _load("dress_rehearsal").Report()
    rv2.run_offline(report, run_v1_pipeline=False)
    assert not report.failed, report.failed
    # Every v2 piece is on this branch: a skipped row means a probe stopped
    # finding a renamed or broken function, and its checks silently stopped.
    skipped = [r for r in report.rows if r[2] is None]
    assert not skipped, skipped
    # 20 rows less the one check_determinism row run_v1_pipeline=False drops.
    assert len(report.rows) == 19, [r[1] for r in report.rows]


def test_release_v2_pieces_are_detected():
    f = _load("rehearsal_v2").probe_features()
    assert all(p.present for p in f.values()), [p.name for p in f.values() if not p.present]


def test_missing_names_every_absent_piece():
    rv2 = _load("rehearsal_v2")
    f = {n: rv2.Piece(n, n == "activation", f"src-{n}") for n, _, _ in rv2.V2_PIECES}
    assert rv2.missing(f, "activation") is None
    assert rv2.missing(f, "activation", "champion", "payout") == (
        "not on this branch: src-champion; src-payout")


# ── the chain scenario's checks, on synthetic rows ────────────────────────────

ACT = 100
M1, M2, M3 = 2, 3, 4
H = {M1: "a" * 64, M2: "b" * 64, M3: "c" * 64}


# First commits: m1 senior, then m3, then m2 (the deterministic junior).
FIRST_COMMIT = {M1: 10, M3: 20, M2: 30}


def _copy_state(idx):
    """The copy-rule block a v2 reveal publishes (copy_rule.CopyState.to_reveal)."""
    return {
        "epoch": idx,
        "first_commit": {H[u]: b for u, b in FIRST_COMMIT.items()},
        "ownership": {H[u]: f"ck{u}" for u in H},
        "first_hotkey": {H[u]: f"hk{u}" for u in H},
        "active": sorted(H.values()),
        "retired_at": {},
    }


def _row(idx, *, event=None, champion=None, weights, control=True, m2_missed=False):
    v2 = idx >= ACT
    reveal = {
        "epoch_id": f"e{idx:08d}", "consensus_digest": "same",
        "head_hashes": {str(u): h for u, h in H.items() if not (m2_missed and u == M2)},
        "scores": {"2": {"score": 0.1}}, "weights": {str(k): v for k, v in weights.items()},
        "environment": {"platform": f"host-{idx}"},
    }
    if v2:
        served = [u for u in H if not (m2_missed and u == M2)]
        reveal.update(protocol_version=2, shared_grades=[{"question_id": "q", "grade": 1.0}],
                      champion={"event": event, "champion": champion},
                      copy_verdicts={str(u): "owner" for u in served},
                      v2_state={"copy_rule": _copy_state(idx)})
    entry = {"epoch_id": f"e{idx:08d}", "scores": {"2": 0.1},
             "weights": {str(k): v for k, v in weights.items()},
             "n_heads_valid": 2 if m2_missed else 3, "n_heads_invalid": 0}
    reveals = [reveal, copy.deepcopy(reveal)]
    entries = [entry, copy.deepcopy(entry)]
    # val2 runs on a different host: environment differs, digest does not.
    reveals[1]["environment"] = {"platform": "other"}
    if control and idx <= ACT:
        ctl = copy.deepcopy(reveal)
        ctl["consensus_digest"] = "control"
        for k in ("protocol_version", "shared_grades", "champion", "copy_verdicts", "v2_state"):
            ctl.pop(k, None)
        reveals.append(ctl)
        entries.append(copy.deepcopy(entry))
    rec = {"evidence": {"burn_in_n": 0.0 if m2_missed else 80.0, "n_total": 79.7},
           "epochs_missed": 1 if m2_missed else 0}
    state = {"records": {str(M2): rec}, "v2": {"seat": champion}}
    return {"epoch_id": f"e{idx:08d}", "entries": entries, "reveals": reveals,
            "states": [state, copy.deepcopy(state)]}


def _good_run():
    rows = {i: _row(i, weights={0: 1.0}) for i in (97, 98, 99)}
    rows[100] = _row(100, event="empty", weights={0: 1.0})
    rows[101] = _row(101, event="first", champion=H[M1], weights={M1: 1.0})
    rows[102] = _row(102, event="take", champion=H[M2], weights={M2: 0.9, M1: 0.1})
    rows[103] = _row(103, event="hold", champion=H[M2], weights={M2: 0.9, M1: 0.1})
    rows[104] = _row(104, event="ghost", champion=None, weights={0: 0.95, M1: 0.05},
                     m2_missed=True)
    rows[105] = _row(105, event="hold", champion=H[M2], weights={M2: 0.9, M1: 0.1})
    miss = {"stopped": 103, "head": H[M2], "first_missed": 104, "back": 105, "resumed": 105}
    return rows, miss


def _run_checks(rows, miss, killed_at=101):
    dr, rv2 = _load("dress_rehearsal"), _load("rehearsal_v2")
    f = {n: rv2.Piece(n, True, n) for n in ("activation", "shared_grades", "champion",
                                             "payout", "miss_rule", "wiring", "miner_hold")}
    report = dr.Report()
    uid = {"dr_m1": M1, "dr_m2": M2, "dr_m3": M3}
    dr._v2_flip_checks(report, "v2_flip", f, rows, ACT, uid, killed_at, miss,
                       rv2.comparable_reveal, rv2.reveal_v2_keys)
    return report


def test_chain_checks_pass_on_a_correct_run():
    rows, miss = _good_run()
    report = _run_checks(rows, miss)
    assert not report.failed, report.failed
    assert len(report.rows) >= 15


def test_chain_checks_catch_validators_that_diverge():
    rows, miss = _good_run()
    rows[103]["entries"][1]["weights"] = {str(M2): 0.85, str(M1): 0.15}
    names = {r[1] for r in _run_checks(rows, miss).failed}
    assert "val1 and val2 agree byte for byte on every epoch across the flip" in names


def test_chain_checks_catch_a_v1_epoch_that_changed():
    rows, miss = _good_run()
    rows[98]["entries"][2]["weights"] = {"0": 0.5, str(M1): 0.5}
    names = {r[1] for r in _run_checks(rows, miss).failed}
    assert any(n.startswith("v1 epochs are unchanged") for n in names)


def test_chain_checks_catch_a_wrong_first_champion_and_split():
    rows, miss = _good_run()
    rows[101]["reveals"][0]["champion"]["champion"] = H[M3]
    rows[101]["reveals"][1]["champion"]["champion"] = H[M3]
    rows[102]["entries"][0]["weights"] = rows[102]["entries"][1]["weights"] = {
        str(M2): 0.85, str(M1): 0.10, "0": 0.05}
    names = {r[1] for r in _run_checks(rows, miss).failed}
    assert "the first seat goes to the most senior eligible head (by first commit)" in names
    assert "payout: weights sum to 1; two champions split 90/10" in names


def test_chain_checks_skip_what_the_branch_lacks():
    rows, miss = _good_run()
    dr, rv2 = _load("dress_rehearsal"), _load("rehearsal_v2")
    f = {n: rv2.Piece(n, n in ("activation", "shared_grades"), n)
         for n in ("activation", "shared_grades", "champion", "payout", "miss_rule",
                   "wiring", "miner_hold")}
    report = dr.Report()
    dr._v2_flip_checks(report, "v2_flip", f, rows, ACT,
                       {"dr_m1": M1, "dr_m2": M2, "dr_m3": M3}, 101, miss,
                       rv2.comparable_reveal, rv2.reveal_v2_keys)
    skipped = [r for r in report.rows if r[2] is None]
    assert not report.failed, report.failed
    assert len(skipped) == 7  # six seat/payout rows and the E'' evidence row
    assert report.summary() == 1  # a skip is not a pass


def test_chain_checks_fail_clearly_without_the_copy_rule_state():
    rows, miss = _good_run()
    for r in rows[101]["reveals"]:
        r.pop("v2_state", None)
    failed = {r[1]: r for r in _run_checks(rows, miss).failed}
    assert "the first seat goes to the most senior eligible head (by first commit)" in failed


def test_chain_checks_catch_a_lost_v1_epoch():
    rows, miss = _good_run()
    rows[ACT - 1]["entries"][0]["n_heads_valid"] = 2
    names = {r[1] for r in _run_checks(rows, miss).failed}
    assert any(n.startswith("no v1 epoch is lost in the handover") for n in names)


def test_rehearsal_v2_router_head_splits_a_collinear_pool(tmp_path):
    """write_head_v2 with the embeddings' mean must route across all three
    models and stay clear of CHAMPION_MIN_ACCURACY on the stub ladder.

    The backbone's pooled hidden states are nearly collinear, so the old
    0.02-scale random head was a constant router to whichever model its bias
    favoured; seed 24's was the cheapest (stub skill 0.30, exactly the floor)
    and the first champion sat on the floor, was correctly not paid on one
    take epoch, and the 90/10 payout check failed. Synthetic collinear
    hidden states reproduce the shape: a common direction plus a small
    per-question component.
    """
    import numpy as np

    from fugal_subnet.config import CHAMPION_MIN_ACCURACY, HEAD_HIDDEN_DIM
    from fugal_subnet.routing_protocol import price_table
    from fugal_subnet.vendor import success_contract as contract

    dr = _load("dress_rehearsal")
    rng = np.random.RandomState(7)
    common = rng.randn(HEAD_HIDDEN_DIM)
    hidden = common[None, :] + rng.randn(200, HEAD_HIDDEN_DIM) * 0.04
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    assert float((hidden @ hidden.T).mean()) > 0.99
    mean = hidden.mean(axis=0)
    skill = np.array([0.30, 0.60, 0.85])

    def split(path):
        z = contract.load(open(path, "rb").read())
        costs = contract.estimated_cost(z, price_table(2))
        dec = contract.rank(contract.predictions(z["W"], z["b"], hidden), costs, 1.0)[:, 0]
        return np.bincount(dec, minlength=3), float(skill[dec].mean())

    counts, acc = split(dr.write_head_v2(str(tmp_path / "router.npz"), seed=61, hidden_mean=mean))
    assert counts.min() >= 20, counts.tolist()
    assert acc >= CHAMPION_MIN_ACCURACY + 0.2, acc

    # Without the mean it is the old constant router.
    counts, _ = split(dr.write_head_v2(str(tmp_path / "constant.npz"), seed=24))
    assert counts.max() == 200, counts.tolist()


def test_rehearsal_epochs_leave_the_miner_its_late_start_margin():
    """Nightly 42428107377: at 90 s a fast devnet gave 292-block epochs, so a
    miner 30 s late had 49 blocks to collection, under its 60-block margin, and
    skipped every epoch of scenario a. The check refuses that geometry up
    front; the shipped target clears it at that block rate."""
    import pytest

    dr = _load("dress_rehearsal")
    with pytest.raises(RuntimeError, match="skip every epoch"):
        dr.check_late_start_headroom(str(292 * 12), 90)
    blocks_150 = int(150 / (90 / 292))                    # same chain speed, 150 s
    assert dr.check_late_start_headroom(str(blocks_150 * 12), dr.TARGET_EPOCH_SECONDS) >= 60
    assert dr.TARGET_EPOCH_SECONDS >= 150
