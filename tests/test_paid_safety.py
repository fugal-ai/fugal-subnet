"""Regression tests for explicit live mode and atomic budget reservations."""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from fugal_subnet.api import (
    BudgetExceeded,
    LiveSpendNotEnabled,
    SpendTracker,
    build_spend_protection_prices,
    call_model,
)


def test_live_call_requires_explicit_authorization(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder-never-sent")
    tracker = SpendTracker(budget_cap_usd=1.0)

    with pytest.raises(LiveSpendNotEnabled):
        call_model(
            "provider/model",
            "hello",
            tracker=tracker,
            prices={"provider/model": (0.001, 0.001)},
        )


def test_live_call_requires_positive_budget(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder-never-sent")

    with pytest.raises(LiveSpendNotEnabled):
        call_model(
            "provider/model",
            "hello",
            tracker=SpendTracker(),
            prices={"provider/model": (0.001, 0.001)},
            live=True,
        )


def test_concurrent_reservations_cannot_overshoot():
    tracker = SpendTracker(budget_cap_usd=0.10)
    prices = {"provider/model": (0.01, 0.01)}
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def reserve():
        barrier.wait()
        try:
            tracker.reserve("provider/model", 3, 3, prices)
            outcomes.append("reserved")
        except BudgetExceeded:
            outcomes.append("rejected")

    threads = [threading.Thread(target=reserve) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["rejected", "reserved"]
    assert tracker.committed_cost_usd == pytest.approx(0.06)
    assert tracker.committed_cost_usd <= tracker.budget_cap_usd


def test_ambiguous_attempt_forfeits_reservation_before_retry(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder-never-sent")
    attempts = 0

    def fail_post(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("ambiguous timeout")

    monkeypatch.setattr(httpx, "post", fail_post)
    tracker = SpendTracker(budget_cap_usd=0.05)

    with pytest.raises(BudgetExceeded):
        call_model(
            "provider/model",
            "x",
            max_tokens=1,
            retries=2,
            tracker=tracker,
            prices={"provider/model": (0.0001, 0.0001)},
            live=True,
        )

    assert attempts == 1
    assert tracker.total_cost_usd == pytest.approx(0.0258)
    assert tracker.reserved_cost_usd == pytest.approx(0.0)
    assert tracker.total_cost_usd <= tracker.budget_cap_usd


def test_success_reconciles_worst_case_reservation(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder-never-sent")

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response())
    tracker = SpendTracker(budget_cap_usd=0.10)

    text, prompt_tokens, completion_tokens = call_model(
        "provider/model",
        "hello",
        max_tokens=10,
        retries=1,
        tracker=tracker,
        prices={"provider/model": (0.0001, 0.0002)},
        live=True,
    )

    assert (text, prompt_tokens, completion_tokens) == ("ok", 2, 1)
    assert tracker.reserved_cost_usd == pytest.approx(0.0)
    assert tracker.total_cost_usd == pytest.approx(0.0004)


def test_protection_price_uses_greater_rate_and_rejects_missing_live():
    canonical = {"provider/model": (0.001, 0.004)}
    live = {"provider/model": (0.002, 0.003)}

    assert build_spend_protection_prices(
        canonical, live, ["provider/model"],
    ) == {"provider/model": (0.002, 0.004)}

    with pytest.raises(RuntimeError, match="current live price"):
        build_spend_protection_prices(canonical, {}, ["provider/model"])


def test_validator_live_mode_requires_budget_before_chain_access(monkeypatch):
    monkeypatch.delenv("FUGAL_EPOCH_BUDGET", raising=False)
    from neurons.validator import main

    result = CliRunner().invoke(main, ["--live", "--once"])

    assert result.exit_code == 2
    assert "requires --epoch-budget" in result.output


def test_usage_overage_is_charged_not_retried(monkeypatch):
    """A provider overage must settle, never raise — raising retries a paid call.

    Regression guard: reconcile() used to raise RuntimeError when actual usage
    exceeded the reservation. call_model catches RuntimeError, forfeits, and
    retries, so a valid already-billed response was discarded and paid for
    again. Realistic for reasoning models, whose completion_tokens include
    reasoning tokens that max_tokens does not cap.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder-never-sent")
    calls = 0

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                # 5000 completion tokens against a max_tokens of 10.
                "usage": {"prompt_tokens": 2, "completion_tokens": 5000},
            }

    def counting_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr(httpx, "post", counting_post)
    tracker = SpendTracker(budget_cap_usd=10.0)

    text, _, ctok = call_model(
        "provider/model", "hello", max_tokens=10, retries=3,
        tracker=tracker, prices={"provider/model": (0.0001, 0.0002)},
        live=True,
    )

    assert (text, ctok) == ("ok", 5000)
    assert calls == 1, "an overage must not trigger a retry of a paid call"
    assert tracker.reserved_cost_usd == pytest.approx(0.0)
    # True cost is charged, not the smaller reservation.
    assert tracker.total_cost_usd == pytest.approx(2 * 0.0001 + 5000 * 0.0002)


def test_validator_embeds_on_cpu_for_consensus():
    """The backbone must never run on CUDA in the validator.

    get_backbone selects float16 on CUDA and float32 on CPU, so a GPU validator
    and a CPU validator would embed identically-worded questions differently,
    flip argmax on near-ties, and score the same head differently.
    """
    source = (Path(__file__).resolve().parents[1] / "neurons" / "validator.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "compute_hidden_states":
            device = next((kw for kw in node.keywords if kw.arg == "device"), None)
            assert device is not None, "compute_hidden_states() must pin device="
            assert isinstance(device.value, ast.Constant) and device.value.value == "cpu", (
                "validator must embed on cpu; a cuda path breaks cross-validator consensus"
            )
            return
    raise AssertionError("no compute_hidden_states() call found in neurons/validator.py")
