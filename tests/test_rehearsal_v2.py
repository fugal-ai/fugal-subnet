"""The protocol v2 flip rehearsal (scripts/rehearsal_v2.py, dress_rehearsal v2_flip).

Two halves. The in-process harness runs here in full: whatever of v2 is on the
branch must hold across the flip, and whatever is not is skipped, never failed.
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
    passed = [r for r in report.rows if r[2] is True]
    # Activation and shared grades are on release/v2, so the flip, digest,
    # v1-unchanged, agreement and restart rows always run.
    assert len(passed) >= 9, report.rows


def test_release_v2_pieces_are_detected():
    f = _load("rehearsal_v2").probe_features()
    assert f["activation"].present and f["shared_grades"].present


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


def _row(idx, *, event=None, champion=None, weights, control=True, m2_missed=False):
    v2 = idx >= ACT
    reveal = {
        "epoch_id": f"e{idx:08d}", "consensus_digest": "same",
        "head_hashes": {str(u): h for u, h in H.items() if not (m2_missed and u == M2)},
        "scores": {"2": {"score": 0.1}}, "weights": {str(k): v for k, v in weights.items()},
        "environment": {"platform": f"host-{idx}"},
    }
    if v2:
        reveal.update(protocol_version=2, shared_grades=[{"question_id": "q", "grade": 1.0}],
                      champion={"event": event, "champion": champion})
    entry = {"epoch_id": f"e{idx:08d}", "scores": {"2": 0.1},
             "weights": {str(k): v for k, v in weights.items()}}
    reveals = [reveal, copy.deepcopy(reveal)]
    entries = [entry, copy.deepcopy(entry)]
    # val2 runs on a different host: environment differs, digest does not.
    reveals[1]["environment"] = {"platform": "other"}
    if control and idx <= ACT:
        ctl = copy.deepcopy(reveal)
        ctl["consensus_digest"] = "control"
        for k in ("protocol_version", "shared_grades", "champion"):
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
    assert "the first seat goes to the most senior eligible head" in names
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
