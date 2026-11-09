"""v1 epochs compute exactly what `main` computes, through the upgrade window.

Every epoch before the activation block is protocol v1, and for the >=24 h
upgrade window validators run a MIX of main and this release. A differential
fuzz of main against this release (2026-11-08) found the v1 frame diverging in
261-278 of 300 random cases: F-061 (implausible_exploration: one trial ->
eight, posterior -> raw mean) applied to v1 epochs too, and the frame
accessors' rewrite changed the posterior's arithmetic in the last digit. The
frame pays nothing in v2, so F-061 is now v2-only, and a frame with no
weighted (v2) evidence uses main's arithmetic. After the change the same fuzz
found 0 divergences in 600 cases for the frame pipeline, accessors, frontier,
scoring and the detector.
"""
import ast
import pathlib
import random

from fugal_subnet.config import FRAME_PRIOR_ACCURACY, FRAME_PRIOR_STRENGTH
from fugal_subnet.reference_frame import ReferenceFrame, implausible_exploration


def _main_rule(frame, per_miner, min_samples=8, margin=0.35):
    """main's implausible_exploration, verbatim in logic (origin/main 678833b)."""
    flagged = {}
    for uid, results in sorted(per_miner.items()):
        by_model = {}
        for model, correct in results:
            by_model.setdefault(model, []).append(bool(correct))
        if sum(len(v) for v in by_model.values()) < min_samples:
            continue
        expected = observed = 0.0
        measured = 0
        for model, outcomes in by_model.items():
            if frame.trials.get(model, 0.0) < 1.0:
                continue
            k = float(FRAME_PRIOR_STRENGTH)
            s, n = frame.successes.get(model, 0.0), frame.trials.get(model, 0.0)
            expected += ((s + k * FRAME_PRIOR_ACCURACY) / (n + k)) * len(outcomes)
            observed += sum(outcomes)
            measured += len(outcomes)
        if measured < min_samples:
            continue
        exp_rate, obs_rate = expected / measured, observed / measured
        if abs(obs_rate - exp_rate) > margin:
            flagged[uid] = "above" if obs_rate > exp_rate else "below"
    return flagged


def _random_case(rng):
    models = [f"m/{i}" for i in range(rng.randint(3, 12))]
    frame = ReferenceFrame()
    for m in models:
        if rng.random() < 0.85:
            n = rng.choice([0.5, 1.0, 2.0, 3.0, 7.0, 8.0, 20.0, 150.0, rng.uniform(0, 300)])
            frame.trials[m] = n
            frame.successes[m] = rng.uniform(0, n)
    per_miner = {uid: [(rng.choice(models), rng.random() < rng.random()) for _ in range(rng.randint(5, 20))]
                 for uid in range(rng.randint(1, 15))}
    return frame, per_miner


def test_a_v1_epoch_flags_exactly_whom_main_flags():
    rng = random.Random(20261108)
    for _ in range(2000):
        frame, per_miner = _random_case(rng)
        got = implausible_exploration(frame, per_miner, protocol_version=1)
        assert {u: why.split()[4] for u, why in got.items()} == _main_rule(frame, per_miner)


def test_v2_keeps_f061():
    """Cold frame, one honest miner at 8/8 on a model seen 5/5: main's rule (the
    posterior pulled to 0.6) flags it; v2 (eight trials needed) does not."""
    frame = ReferenceFrame()
    frame.trials["m/a"], frame.successes["m/a"] = 5.0, 5.0
    honest = {7: [("m/a", True)] * 8}
    assert 7 in implausible_exploration(frame, honest, protocol_version=1)
    assert implausible_exploration(frame, honest, protocol_version=2) == {}
    assert implausible_exploration(frame, honest) == {}           # the default is v2


def test_a_v1_frame_uses_mains_exact_arithmetic():
    rng = random.Random(7)
    k = float(FRAME_PRIOR_STRENGTH)
    for _ in range(2000):
        frame = ReferenceFrame()
        n = rng.uniform(0.001, 500)
        s = rng.uniform(0, n)
        frame.trials["m"], frame.successes["m"] = n, s
        assert frame.accuracy("m") == (s + k * FRAME_PRIOR_ACCURACY) / (n + k)
        assert frame.frontier_accuracy("m") == min(1.0, max(0.0, s / n))
    edge = ReferenceFrame()
    edge.trials["m"], edge.successes["m"] = 1e-9, 0.7e-9
    assert edge.frontier_accuracy("m") == min(1.0, 0.7e-9 / 1e-9)     # main: measured at 1e-9


def test_the_validator_passes_the_epochs_protocol_version_to_the_detector():
    src = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_frame_input")
    (call,) = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
               and getattr(n.func, "id", "") == "implausible_exploration"]
    assert {k.arg: ast.unparse(k.value) for k in call.keywords} == {"protocol_version": "protocol_version"}
