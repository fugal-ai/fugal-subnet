"""Regression tests for explicit live mode and atomic budget reservations."""

from __future__ import annotations

import threading

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
