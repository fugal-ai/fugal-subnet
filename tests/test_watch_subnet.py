"""scripts/watch_subnet.py judges an epoch when weights are set, and checks v2.

Two failures it had: it judged an epoch 30 blocks in, so validators that set
weights at +194 and +237 (finney/70, measured 2026-11-07) were reported as not
setting weights; and it could not see the flip, which MAINNET_LAUNCH asks of it
(protocol_version per reveal, identical reveals including v2's seat).
"""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("watch_subnet_under_test", ROOT / "scripts" / "watch_subnet.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)

EPOCH, BOUNDARY = "e00000010", 3600


def _write(root: pathlib.Path, reveal: dict) -> str:
    d = root / EPOCH
    d.mkdir(parents=True)
    (d / "reveal.json").write_text(json.dumps(reveal), encoding="utf-8")
    return str(root)


def _reveal(**extra):
    base = {"scores": {"5": 0.5}, "weights": {"5": 0.85, "0": 0.15}, "commit_hash": "h",
            "consensus_digest": "d", "environment": {"host": "x"}}
    return {**base, **extra}


def _run(tmp_path, reveals: dict, activation=None):
    roots = {name: _write(tmp_path / name, rv) for name, rv in reveals.items()}
    report = watch.Report()
    watch.check_reveals(report, EPOCH, BOUNDARY, {}, {}, roots, activation)
    return report


def _failed(report):
    return [line for line in report.lines if "[FAIL]" in line]


def test_an_epoch_is_judged_only_once_weights_can_be_set():
    assert not watch.ready_to_judge(9457200 + 137, 360, 340)   # the live false alarm
    assert watch.ready_to_judge(9457200 + 340, 360, 340)
    assert watch.ready_to_judge(5, 360, 0)
    assert watch.ready_to_judge(9, 10, 340)                    # clamped to the epoch


def test_v2_reveals_after_the_block_pass_and_agree(tmp_path):
    seat = {"event": "first", "champion": "ab" * 32}
    rv = _reveal(protocol_version=2, shared_grades=[[1, 0]], champion=seat)
    report = _run(tmp_path, {"val1": rv, "val2": dict(rv, environment={"host": "y"})},
                  activation=BOUNDARY)
    assert not _failed(report), report.lines


def test_a_v1_reveal_after_the_block_fails(tmp_path):
    """A validator that did not set the block keeps v1 rules after the flip."""
    report = _run(tmp_path, {"val1": _reveal(protocol_version=2), "val2": _reveal()},
                  activation=BOUNDARY)
    assert any("val2's reveal is protocol v2" in line for line in _failed(report))


def test_v1_before_the_block_is_expected(tmp_path):
    report = _run(tmp_path, {"val1": _reveal()}, activation=BOUNDARY + 360)
    assert not _failed(report), report.lines


def test_a_seat_disagreement_fails_even_when_weights_match(tmp_path):
    a = _reveal(protocol_version=2, champion={"event": "take", "champion": "aa"})
    b = _reveal(protocol_version=2, champion={"event": "hold", "champion": "bb"})
    report = _run(tmp_path, {"val1": a, "val2": b}, activation=BOUNDARY)
    assert any("reveals are identical" in line for line in _failed(report))
    assert not any("weight vectors" in line for line in _failed(report))


def _real_score(score=0.0):
    """One uid's entry as neurons/validator.py writes it to reveal.json."""
    return {"accuracy": 0.857, "quality": 0.812, "thrift": 1.95, "headroom": -0.19,
            "reference_accuracy": 1.0, "cost_per_question": 0.000141, "score": score}


def test_a_real_reveal_scores_each_uid_as_a_dict(tmp_path):
    """The first v2 epoch on testnet 552 (e00023374): two scored miners, an
    empty seat, all weight on the burn uid. The watcher died on the score
    dicts, and once past them called two identical validators outliers."""
    rv = _reveal(protocol_version=2, scores={"5": _real_score(), "6": _real_score()},
                 weights={"0": 1.0}, shared_grades=[[1, 0]],
                 champion={"event": "empty", "champion": None})
    report = _run(tmp_path, {"arm": rv, "val2": dict(rv, environment={"host": "y"})},
                  activation=BOUNDARY)
    assert not _failed(report), report.lines
    assert any("validators agree on scores and weights" in line for line in report.lines)


def test_reveal_score_reads_the_dict_and_the_bare_number():
    assert watch.reveal_score(_real_score(0.25)) == 0.25
    assert watch.reveal_score(0.5) == 0.5
    assert watch.reveal_score({"accuracy": 0.9}) == 0.0


def test_a_weight_disagreement_on_the_burn_uid_still_fails(tmp_path):
    """The other direction: weight a validator moved off the burn uid is a
    real divergence, and the burn uid being unscored must not hide it."""
    scores = {"5": _real_score(0.4)}
    a = _reveal(protocol_version=2, scores=scores, weights={"0": 1.0})
    b = _reveal(protocol_version=2, scores=scores, weights={"0": 0.5, "5": 0.5})
    report = _run(tmp_path, {"arm": a, "val2": b}, activation=BOUNDARY)
    assert any("validators agree on scores and weights" in line for line in _failed(report))
    assert any("weight vectors" in line for line in _failed(report))
