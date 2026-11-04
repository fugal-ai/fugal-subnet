"""Protocol v2 image wiring: per-model reasoning on every v2 call, and the miner's hold."""
import sys

from fugal_subnet.routing_protocol import models_for_protocol, reasoning_for
from fugal_subnet.tee import harness


class _Proxy:
    records = []


def test_v2_calls_carry_each_models_pinned_reasoning_and_v1_calls_carry_none(monkeypatch):
    seen = []

    def fake_call(proxy, model_id, question, reasoning=None):
        seen.append((model_id, reasoning))
        return "ok"

    monkeypatch.setattr(harness, "_call_model", fake_call)
    m2 = models_for_protocol(2)[:3]
    list(harness._call_all(_Proxy(), [({"question_id": f"q{i}"}, m, False) for i, m in enumerate(m2)], "e1", 2))
    assert sorted(seen) == sorted((m, reasoning_for(2, m)) for m in m2)    # calls run on a thread pool
    seen.clear()
    m1 = models_for_protocol(1)[:2]
    list(harness._call_all(_Proxy(), [({"question_id": f"q{i}"}, m, False) for i, m in enumerate(m1)], "e1", 1))
    assert sorted(seen) == sorted((m, None) for m in m1)


def test_the_miner_holds_until_the_target_block(monkeypatch):
    sys.path.insert(0, "neurons")
    import miner  # noqa: E402

    blocks = iter([100, 150, 199, 200, 201])

    class _Sub:
        def get_current_block(self):
            return next(blocks)

    monkeypatch.setattr(miner.time, "sleep", lambda s: None)
    miner._wait_for_block(_Sub(), 200, "serve the axon")
    assert next(blocks) == 201          # returned exactly when the chain reached 200


def test_the_miner_never_prices_a_head_with_the_v1_only_function():
    """Structural: the miner must price/validate heads with benchmark_costs_for(head, version).
    A bare v1 benchmark_costs() call refused every genuine v2 head at startup (caught by the v2
    flip rehearsal, missed by unit tests that never start a v2 miner with a v2 head)."""
    import ast
    import pathlib
    tree = ast.parse((pathlib.Path(__file__).resolve().parent.parent / "neurons" / "miner.py").read_text())
    names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "benchmark_costs" not in names
    assert "benchmark_costs_for" in names
