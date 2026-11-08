"""The end of a validator epoch: the reveal, setting weights, publishing, the
epoch log, committing the epoch, and the exit code of a --once run.

These lived inline in neurons/validator.py's epoch loop, where planted bugs
survived the whole suite (2026-11-08), among them: unexplored cells published
as wrong answers, the reveal's exploration losing its voter or failure flag,
weights retried into the next epoch, the cap baseline moving on a failed set,
anomalies dropped from the epoch log. And `--once` exited 0 after an epoch
that raised, so a oneshot or a cron job reported success.
"""
import ast
import pathlib
from types import SimpleNamespace

import numpy as np

from neurons import validator
from neurons.validator import (
    _epoch_anomalies,
    _exploration_for_reveal,
    _next_v2_state,
    _reveal_matrix,
    _weights_deadline,
)


def _r(qid, model, correct=True, cost=0.01):
    return SimpleNamespace(question_id=qid, routed_model=model, correct=correct, cost_usd=cost)


# ------------------------------------------------------------------ the reveal --

def test_the_reveal_matrix_tells_unexplored_from_wrong():
    questions = [{"question_id": "q0"}, {"question_id": "q1"}, {"question_id": "q2"}]
    index = {"m/a": 0, "m/b": 1, "m/c": 2}
    proofs = {
        1: SimpleNamespace(results=[_r("q0", "m/c", True), _r("q1", "m/a", False)]),
        2: SimpleNamespace(results=[_r("q2", "m/b", True, cost=0.5), _r("q9", "m/a"),
                                    _r("q0", "gone/model", True)]),
    }
    matrix, spend = _reveal_matrix(questions, ["m/a", "m/b", "m/c"], index, proofs)
    assert matrix.dtype == np.int32 and matrix.shape == (3, 3)
    assert matrix.tolist() == [[-1, -1, 1],      # q0 routed to m/c, right
                               [0, -1, -1],      # q1 routed to m/a, WRONG (0, not -1)
                               [-1, 1, -1]]
    # Spend is summed across miners, including questions outside the slice.
    assert spend == {"m/c": 0.01, "m/a": 0.02, "m/b": 0.5, "gone/model": 0.01}


def test_the_reveal_matrix_has_a_column_even_with_no_models():
    matrix, spend = _reveal_matrix([{"question_id": "q0"}], [], {}, {})
    assert matrix.tolist() == [[-1]] and spend == {}


def test_the_reveal_exploration_keeps_the_voter_and_the_failure_flag():
    obs = SimpleNamespace(question_id="q", model="m", correct=False, prompt_tokens=3,
                          completion_tokens=4, coldkey="5Cold", failed=True)
    assert _exploration_for_reveal([obs]) == [{
        "question_id": "q", "model": "m", "correct": False, "prompt_tokens": 3,
        "completion_tokens": 4, "coldkey": "5Cold", "failed": True}]


# --------------------------------------------------------------- set weights --

def test_weights_are_retried_only_inside_their_own_epoch():
    assert _weights_deadline(10, 360) == 11 * 360 - 5
    assert _weights_deadline(10, 360) < 11 * 360


# ------------------------------------------------------------------ the log --

def test_a_clean_epoch_has_no_validator_anomalies():
    assert _epoch_anomalies(True, True, 730, {}, 0) == []


def test_every_validator_side_anomaly_is_logged_in_order():
    out = _epoch_anomalies(False, False, 730, {4: "too good", 9: "disagrees"}, 3)
    assert out == [
        "commit_reveal_failed",
        "metagraph_unpinned: the read at block 730 failed and the current metagraph was used",
        "exploration_excluded_from_frame: uid 4: too good",
        "exploration_excluded_from_frame: uid 9: disagrees",
        "collateral_unavailable: 3 proof(s) unchecked",
    ]


# ------------------------------------------------------------- the v2 state --

def _result(n_slices):
    return SimpleNamespace(window=SimpleNamespace(slices=dict.fromkeys(range(n_slices))),
                           decision=SimpleNamespace(state="seat-after"), copy_state="copy-after")


def test_a_v1_epoch_carries_the_v2_state_unchanged():
    assert _next_v2_state(None, "w", "s", "c", True, 48) == ("w", "s", "c", True)


def test_a_v2_epoch_takes_its_result():
    r = _result(3)
    assert _next_v2_state(r, "w", "s", "c", False, 48) == (r.window, "seat-after", "copy-after", False)


def test_a_degraded_window_recovers_only_once_it_has_refilled():
    assert _next_v2_state(_result(47), "w", "s", "c", True, 48)[3] is True
    assert _next_v2_state(_result(48), "w", "s", "c", True, 48)[3] is False


# ----------------------------------------------- inline wiring, on the syntax tree --

def _main():
    tree = ast.parse(pathlib.Path(validator.__file__).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _call(main, name):
    (call,) = [n for n in ast.walk(main) if isinstance(n, ast.Call)
               and getattr(n.func, "id", "") == name]
    return call, {k.arg: ast.unparse(k.value) for k in call.keywords}


def test_the_reveal_publishes_this_epochs_exploration_exclusions_and_version():
    main = _main()
    _, kw = _call(main, "reveal_epoch")
    assert kw["exploration"] == "exploration_for_reveal"
    assert kw["frame_excluded_coldkeys"] == "sorted({coldkey_of[uid] for uid in suspicious})"
    assert kw["protocol_version"] == "protocol_version"
    assert kw["v2_fields"].startswith("{**v2_result.reveal,")
    assert kw["v2_fields"].endswith("if v2_result is not None else None")
    assert "epoch_weight_map = dict(zip(uids, weights))" in ast.unparse(main)


def test_weights_are_set_by_the_deadline_and_the_cap_baseline_moves_only_on_success():
    main = _main()
    _, kw = _call(main, "set_weights_with_retry")
    assert kw["deadline_block"] == "_weights_deadline(epoch_index, blocks_per_epoch)"
    src = ast.unparse(main)
    assert "next_prev_uids, next_prev_weights = (prev_uids, prev_weights)" in src
    (branch,) = [n for n in ast.walk(main) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "success"]
    moved = "next_prev_uids, next_prev_weights = (uids, weights)"
    assert moved in ast.unparse(branch.body) and moved not in ast.unparse(branch.orelse)
    assert src.count(moved) == 1


def test_reveals_are_published_after_the_hold_and_a_failure_never_ends_the_epoch():
    main = _main()
    call, _ = _call(main, "publish_due_reveals")
    assert [ast.unparse(a) for a in call.args[:3]] == ["EPOCH_DIR", "epoch_index", "reveal_delay"]
    guards = [n for n in ast.walk(main) if isinstance(n, ast.Try)
              and any(call in ast.walk(p) for p in n.body)]
    guard = min(guards, key=lambda n: len(list(ast.walk(n))))     # the innermost
    assert [ast.unparse(h.type) for h in guard.handlers] == ["Exception"]


def test_the_epoch_log_records_what_happened():
    main = _main()
    logs = [n for n in ast.walk(main) if isinstance(n, ast.Call)
            and getattr(n.func, "id", "") == "EpochLog"]
    full = max(logs, key=lambda c: len(c.keywords))            # the end-of-epoch log
    kw = {k.arg: ast.unparse(k.value) for k in full.keywords}
    assert kw["n_heads_invalid"] == "n_invalid"
    assert kw["n_heads_unverifiable"] == "n_unverifiable"
    assert kw["set_weights_success"] == "success"
    assert kw["dedup_disqualified"] == "list(dupes)"
    assert kw["weights"] == "epoch_weight_map" and kw["anomalies"] == "anomalies"
    assert kw["weights_confirmed_on_chain"] == "weights_confirmed"


def test_the_epoch_commits_its_frame_its_index_and_its_v2_state():
    src = ast.unparse(_main())
    commit = src[src.index("write_epoch_log(epoch_log)\n            next_v2 ="):]
    for line in ("scoring_state, frame = (epoch_state, epoch_frame)",
                 "prev_uids, prev_weights = (next_prev_uids, next_prev_weights)",
                 "last_epoch_index = epoch_index",
                 "v2_window, v2_seat, v2_copy, v2_window_degraded = next_v2"):
        assert line in commit[:commit.index("save_state(")], line


# ----------------------------------------------------------- --once exit code --

def test_a_failed_once_run_exits_non_zero():
    """--once after an epoch that raised used to os._exit(0): a systemd oneshot
    or a cron job saw success with an error in the log."""
    main = _main()
    (exit_call,) = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == "os._exit"]
    assert ast.unparse(exit_call.args[0]) == "1 if epoch_failed else 0"
    (handler,) = [h for n in ast.walk(main) if isinstance(n, ast.Try)
                  for h in n.handlers if h.type is not None and ast.unparse(h.type) == "Exception"
                  and "Error in epoch" in ast.unparse(h)]
    assert "epoch_failed = True" in ast.unparse(handler)
    (loop,) = [n for n in ast.walk(main) if isinstance(n, ast.While)]
    first = loop.body[0]
    assert isinstance(first, ast.Try) and ast.unparse(first.body[0]) == "epoch_failed = False"
