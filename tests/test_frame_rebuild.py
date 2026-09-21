"""A validator must be able to rebuild its frame from published reveals.

The frame is accumulated over time and lives in one local file. A validator that
loses it — or one that has just joined — starts from the bootstrap prior while
the field carries history, and `acc_best` is the denominator of every quality
term. Measured on real accumulation: a fresh frame gives acc_best 0.500 where an
established one gives 0.832, so the two score the whole field differently.

Without reconstruction the subnet cannot onboard a second validator or survive a
lost disk without diverging, which makes this a consensus property rather than
an operational convenience.
"""
import json

from fugal_subnet.reference_frame import (
    accumulate_exploration,
    best_model,
    rebuild_from_reveals,
)
from fugal_subnet.routing_protocol import identity


def _write_reveal(root, epoch_id, samples):
    d = root / epoch_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "reveal.json").write_text(json.dumps({
        "epoch_id": epoch_id,
        "routing_protocol": identity(),
        "exploration": [
            {"question_id": f"q{i}", "model": m, "correct": c,
             "prompt_tokens": 100, "completion_tokens": 50}
            for i, (m, c) in enumerate(samples)
        ],
    }), encoding="utf-8")


def test_rebuild_reproduces_live_accumulation(tmp_path):
    """Replaying the published reveals must land on the same frame the
    validator would have reached by processing those epochs live."""
    epochs = [
        [("a", True), ("a", False), ("b", True)],
        [("a", True), ("b", True), ("b", False)],
        [("a", False), ("b", True)],
    ]
    live = None
    for i, samples in enumerate(epochs):
        _write_reveal(tmp_path, f"e{i:08d}", samples)
        live = accumulate_exploration(live, [(m, c, 100, 50) for m, c in samples])

    rebuilt = rebuild_from_reveals(str(tmp_path))
    assert rebuilt.to_dict() == live.to_dict()


def test_replay_is_ordered_by_epoch_not_by_filesystem(tmp_path):
    """Decay is applied per epoch, so order changes the result. Directory
    listing order is not deterministic; sorting by epoch id is what makes two
    operators rebuilding from the same reveals reach the same frame."""
    _write_reveal(tmp_path, "e00000002", [("a", False)] * 8)
    _write_reveal(tmp_path, "e00000001", [("a", True)] * 8)

    first = rebuild_from_reveals(str(tmp_path)).to_dict()
    # Same data, handed over in the opposite order: must not change the answer.
    paths = sorted((tmp_path).glob("*/reveal.json"), reverse=True)
    assert rebuild_from_reveals([str(p) for p in paths]).to_dict() == first


def test_rebuild_closes_the_new_validator_divergence(tmp_path):
    """The property this exists for: a rebuilt validator agrees with an
    established one, where a cold-started validator would not."""
    from fugal_subnet.api import load_prices
    from fugal_subnet.reference_frame import load_bootstrap

    prices = load_prices()
    model = sorted(prices)[0]
    established = None
    for i in range(20):
        samples = [(model, True)] * 4
        _write_reveal(tmp_path, f"e{i:08d}", samples)
        established = accumulate_exploration(
            established, [(m, c, 100, 50) for m, c in samples])

    _, acc_established = best_model(established, prices)
    _, acc_cold = best_model(load_bootstrap(), prices)
    _, acc_rebuilt = best_model(rebuild_from_reveals(str(tmp_path)), prices)

    assert acc_cold != acc_established, "the divergence this guards against"
    assert acc_rebuilt == acc_established, "rebuild did not close it"


def test_a_corrupt_reveal_does_not_lose_the_rest(tmp_path):
    _write_reveal(tmp_path, "e00000001", [("a", True)] * 4)
    bad = tmp_path / "e00000002"
    bad.mkdir()
    (bad / "reveal.json").write_text("{not json", encoding="utf-8")
    _write_reveal(tmp_path, "e00000003", [("a", True)] * 4)

    frame = rebuild_from_reveals(str(tmp_path))
    assert frame.trials.get("a", 0) > 0


def test_no_reveals_falls_back_to_the_prior(tmp_path):
    frame = rebuild_from_reveals(str(tmp_path))
    assert frame is not None
