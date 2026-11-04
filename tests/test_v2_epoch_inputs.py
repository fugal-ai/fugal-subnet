"""Protocol v2 epoch inputs: slice, exploration, pool, prices and model index by version.

docs/INVARIANTS.md, "I1 — every epoch input is chosen by the boundary-keyed version".
"""
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()


def test_the_validator_selects_questions_only_through_epoch_plan():
    loop = SRC[SRC.index("            # --- SLICE ---"):SRC.index("            # --- COMMIT ---")]
    assert "scored_slice(protocol_version, nonce, benchmark_pool, SLICE_SIZE)" in loop
    assert "exploration_plan(\n                protocol_version, nonce, benchmark_pool" in loop
    assert "select_slice(" not in SRC and "expected_exploration_map(" not in SRC


def test_epoch_inputs_are_picked_by_the_version_before_selection():
    loop = SRC[SRC.index("            # --- SLICE ---"):SRC.index("            # --- COMMIT ---")]
    pick = loop.index("epoch_inputs[protocol_version]")
    assert pick < loop.index("scored_slice(")
    assert "benchmark_pool, gold_answers, prices, pool_models, model_index = (" in loop
    assert "if protocol_version not in epoch_inputs:" in loop


def test_v1_inputs_are_what_v1_always_loaded_and_v2_has_its_own():
    start = SRC[SRC.index("    epoch_inputs = {1:"):]
    block = start[:start.index("    state = load_state(MinerRecord)")]
    assert "epoch_inputs = {1: (benchmark_pool, gold_answers, prices, pool_models, model_index)}" in block
    assert "load_pool(2, strict=live)" in block
    assert "price_table(2)" in block and "models_for_protocol(2)" in block
    assert "if live:\n                raise click.ClickException" in block
    # v1's own loading is untouched.
    assert "benchmark_pool = load_all()" in SRC and "prices = load_prices()" in SRC


def test_v1_epoch_plan_is_byte_identical_to_the_v1_functions():
    from fugal_subnet.benchmarks.slicer import select_slice
    from fugal_subnet.epoch_plan import exploration_plan, scored_slice
    from fugal_subnet.exploration import expected_exploration

    pool = [{"question_id": f"{b}-{i}", "benchmark": b, "prompt": "p"}
            for b in ("mmlu", "math", "gsm8k") for i in range(200)]
    models = ["m/a", "m/b", "m/c"]
    for k in range(5):
        nonce = bytes([k]) * 32
        sl = scored_slice(1, nonce, pool, 60)
        assert sl == select_slice(nonce, pool, 60)
        ids = {q["question_id"] for q in sl}
        assert exploration_plan(1, nonce, pool, ids, models, 6) == expected_exploration(
            nonce, pool, ids, models, 6)


def test_v2_epoch_plan_uses_the_v2_categories():
    from fugal_subnet.config import BENCHMARK_CATEGORY_V2, CATEGORY_TARGETS_V2
    from fugal_subnet.epoch_plan import exploration_plan, scored_slice

    pool = [{"question_id": f"{b}-{i}", "benchmark": b, "prompt": "p"}
            for b in sorted(BENCHMARK_CATEGORY_V2) for i in range(400)]
    sl = scored_slice(2, b"\x07" * 32, pool, 300)
    counts = {}
    for q in sl:
        c = BENCHMARK_CATEGORY_V2[q["benchmark"]]
        counts[c] = counts.get(c, 0) + 1
    assert counts == {c: round(300 * t) for c, t in CATEGORY_TARGETS_V2.items()}
    plan = exploration_plan(2, b"\x07" * 32, pool, {q["question_id"] for q in sl},
                            ["m/a", "m/b"], 15)
    assert len(plan) == 15 and not set(plan) & {q["question_id"] for q in sl}


def test_the_startup_rebuild_prices_the_window_at_v2s_table():
    start = SRC[SRC.index("rebuilt = rebuild_v2_from_reveals("):]
    call = start[:start.index(")\n")]
    assert "prices=epoch_inputs.get(2, epoch_inputs[1])[2]" in call
    assert SRC.index("    epoch_inputs = {1:") < SRC.index("rebuilt = rebuild_v2_from_reveals(")
