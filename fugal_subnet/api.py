"""OpenRouter API client with retry/backoff and conservative spend tracking.

Adapted from fugal-core/fugal/router.py or_request. Never prints API keys.

Paid calls are fail-closed: callers must explicitly opt into live operation,
provide a positive budget, and supply both canonical and current live prices.
Each HTTP attempt atomically reserves its worst-case cost before it starts.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Mapping

import httpx

from fugal_subnet.config import API_MAX_RETRIES, API_RETRYABLE_STATUS, API_TIMEOUT

logger = logging.getLogger(__name__)

OR_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_TOKEN_OVERHEAD = 256


class BudgetExceeded(Exception):
    """A paid attempt cannot fit inside the remaining reserved budget."""


class LiveSpendNotEnabled(RuntimeError):
    """A paid code path was reached without explicit live authorization."""


class UnknownPrice(RuntimeError):
    """A model lacks canonical or current live pricing."""


@dataclass(frozen=True)
class SpendReservation:
    reservation_id: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class SpendTracker:
    """Thread-safe actual and reserved API spend for concurrent matrix work."""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    reserved_cost_usd: float = 0.0
    budget_cap_usd: float | None = None
    per_call_log: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _next_reservation_id: int = field(default=1, repr=False, compare=False)
    _reservations: dict[int, SpendReservation] = field(
        default_factory=dict, repr=False, compare=False,
    )

    def __post_init__(self):
        if self.budget_cap_usd is not None and self.budget_cap_usd <= 0:
            raise ValueError("budget_cap_usd must be positive")

    @property
    def committed_cost_usd(self) -> float:
        """Actual plus in-flight reserved spend."""
        with self._lock:
            return self.total_cost_usd + self.reserved_cost_usd

    def reserve(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        prices: Mapping[str, tuple[float, float]],
    ) -> SpendReservation:
        """Atomically reserve a worst-case paid attempt before scheduling it."""
        if self.budget_cap_usd is None:
            raise LiveSpendNotEnabled("paid calls require an explicit positive budget")
        if model not in prices:
            raise UnknownPrice(f"No spend-protection price for model {model}")
        pin, pout = prices[model]
        if pin < 0 or pout < 0:
            raise UnknownPrice(f"Invalid spend-protection price for model {model}")

        prompt_tokens = max(0, int(prompt_tokens))
        completion_tokens = max(0, int(completion_tokens))
        cost = prompt_tokens * pin + completion_tokens * pout

        with self._lock:
            committed = self.total_cost_usd + self.reserved_cost_usd
            if committed + cost > self.budget_cap_usd + 1e-12:
                raise BudgetExceeded(
                    f"Attempt for {model} needs ${cost:.6f}; budget "
                    f"${self.budget_cap_usd:.2f}, spent ${self.total_cost_usd:.6f}, "
                    f"reserved ${self.reserved_cost_usd:.6f}"
                )
            reservation = SpendReservation(
                reservation_id=self._next_reservation_id,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
            )
            self._next_reservation_id += 1
            self._reservations[reservation.reservation_id] = reservation
            self.reserved_cost_usd += cost
            return reservation

    def reconcile(
        self,
        reservation: SpendReservation,
        prompt_tokens: int,
        completion_tokens: int,
        prices: Mapping[str, tuple[float, float]],
    ) -> None:
        """Replace a reservation with provider-reported actual usage.

        An overage never raises. The provider has already billed this call, so
        rejecting it would discard a paid response and retry — paying twice.
        Instead the true cost is charged (so the overage is visible to the cap)
        and the next reserve() fails if that pushed us over budget.
        """
        pin, pout = prices[reservation.model]
        actual_cost = max(0, int(prompt_tokens)) * pin + max(0, int(completion_tokens)) * pout
        if actual_cost > reservation.cost_usd + 1e-12:
            # Realistic for reasoning models, whose completion_tokens include
            # reasoning tokens that max_tokens does not always cap.
            logger.warning(
                "Provider usage for %s cost $%.6f, above its $%.6f reservation; "
                "charging the true cost",
                reservation.model, actual_cost, reservation.cost_usd,
            )

        with self._lock:
            current = self._reservations.pop(reservation.reservation_id, None)
            if current is None:
                raise RuntimeError("Unknown or already-settled spend reservation")
            self.reserved_cost_usd -= current.cost_usd
            self.total_calls += 1
            self.total_prompt_tokens += int(prompt_tokens)
            self.total_completion_tokens += int(completion_tokens)
            self.total_cost_usd += actual_cost
            self.per_call_log.append({
                "model": reservation.model,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "cost_usd": actual_cost,
                "status": "reconciled",
            })

    def forfeit(self, reservation: SpendReservation, reason: str) -> None:
        """Charge the full reservation when provider billing is ambiguous."""
        with self._lock:
            current = self._reservations.pop(reservation.reservation_id, None)
            if current is None:
                return
            self.reserved_cost_usd -= current.cost_usd
            self.total_calls += 1
            self.total_cost_usd += current.cost_usd
            self.per_call_log.append({
                "model": current.model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": current.cost_usd,
                "status": "forfeited",
                "reason": reason,
            })

    def record(self, model: str, prompt_tokens: int, completion_tokens: int,
               cost_usd: float):
        """Record non-reserved usage (mock mode and historical compatibility)."""
        with self._lock:
            if (
                self.budget_cap_usd is not None
                and self.total_cost_usd + self.reserved_cost_usd + cost_usd
                > self.budget_cap_usd + 1e-12
            ):
                raise BudgetExceeded("Recorded usage would exceed the budget cap")
            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost_usd += cost_usd
            self.per_call_log.append({
                "model": model, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "cost_usd": cost_usd,
                "status": "recorded",
            })


def _get_api_key() -> str:
    key = os.environ.get("FUGAL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Set FUGAL_API_KEY or OPENROUTER_API_KEY env var."
        )
    return key


def call_model(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float | None = 0.0,
    timeout: int = API_TIMEOUT,
    retries: int = API_MAX_RETRIES,
    tracker: SpendTracker | None = None,
    prices: dict[str, tuple[float, float]] | None = None,
    live: bool = False,
) -> tuple[str, int, int]:
    """Call a model via OpenRouter. Returns (response_text, prompt_tokens, completion_tokens).

    Requires explicit live=True authorization, a budgeted SpendTracker, and
    spend-protection prices. Each attempt is atomically reserved before the
    HTTP call starts.
    """
    if not live:
        raise LiveSpendNotEnabled("OpenRouter calls require explicit live=True authorization")
    if tracker is None or tracker.budget_cap_usd is None:
        raise LiveSpendNotEnabled("OpenRouter calls require a positive SpendTracker budget")
    if not prices or model not in prices:
        raise UnknownPrice(f"No spend-protection price for model {model}")

    key = _get_api_key()
    messages = [{"role": "user", "content": prompt}]
    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature

    last_err = None
    for attempt in range(retries):
        reservation = tracker.reserve(
            model,
            len(prompt.encode("utf-8")) + PROMPT_TOKEN_OVERHEAD,
            max_tokens,
            prices,
        )
        try:
            # [PAID ~$0-$0.10/call]
            r = httpx.post(
                OR_URL, timeout=timeout,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
            if r.status_code in API_RETRYABLE_STATUS:
                last_err = RuntimeError(f"HTTP {r.status_code} from {model}")
                raise last_err
            r.raise_for_status()
            j = r.json()
            if "choices" not in j:
                raise ValueError(str(j.get("error", j))[:200])

            msg = j["choices"][0].get("message") or {}
            text = msg.get("content") or msg.get("reasoning") or ""
            usage = j.get("usage") or {}
            ptok = int(usage.get("prompt_tokens", 0))
            ctok = int(usage.get("completion_tokens", 0))

            tracker.reconcile(reservation, ptok, ctok, prices)

            return text, ptok, ctok

        except (httpx.TimeoutException, httpx.NetworkError) as e:
            tracker.forfeit(reservation, type(e).__name__)
            last_err = e
        except RuntimeError as e:
            tracker.forfeit(reservation, type(e).__name__)
            last_err = e
        except Exception as e:
            tracker.forfeit(reservation, type(e).__name__)
            raise

        if attempt < retries - 1:
            delay = (2 ** attempt) + random.random()
            logger.warning("Retry %d/%d for %s (%.1fs delay)", attempt + 1, retries, model, delay)
            time.sleep(delay)

    raise last_err if last_err else RuntimeError(f"call_model({model}) failed after {retries} retries")


def build_spend_protection_prices(
    canonical: Mapping[str, tuple[float, float]],
    live: Mapping[str, tuple[float, float]],
    models: list[str],
) -> dict[str, tuple[float, float]]:
    """Use the greater canonical/live token price for budget protection."""
    protected: dict[str, tuple[float, float]] = {}
    for model in models:
        if model not in canonical:
            raise UnknownPrice(f"No canonical price for model {model}")
        if model not in live:
            raise UnknownPrice(f"No current live price for model {model}")
        c_in, c_out = canonical[model]
        l_in, l_out = live[model]
        if min(c_in, c_out, l_in, l_out) < 0:
            raise UnknownPrice(f"Invalid price for model {model}")
        protected[model] = (max(c_in, l_in), max(c_out, l_out))
    return protected


def fetch_openrouter_prices(timeout: int = 30) -> dict[str, tuple[float, float]]:
    """Fetch live model prices from OpenRouter's /api/v1/models endpoint.

    Returns {model_id: (price_per_input_token, price_per_output_token)}.
    """
    r = httpx.get("https://openrouter.ai/api/v1/models", timeout=timeout)
    r.raise_for_status()
    data = r.json().get("data", [])

    prices = {}
    for model in data:
        mid = model.get("id", "")
        pricing = model.get("pricing", {})
        p_in = pricing.get("prompt")
        p_out = pricing.get("completion")
        if p_in is not None and p_out is not None:
            try:
                prices[mid] = (float(p_in), float(p_out))
            except (ValueError, TypeError):
                continue
    logger.info("OpenRouter: %d models with pricing", len(prices))
    return prices


def load_prices(path: str | None = None) -> dict[str, tuple[float, float]]:
    """Load model price sheet from local JSON fallback.

    Returns {model_id: (price_per_input_token, price_per_output_token)}.
    Prices in the JSON are per-million-token; this returns per-token.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "models.json",
        )

    with open(path, encoding="utf-8") as f:
        models = json.load(f)

    prices = {}
    for m in models:
        mid = m["id"]
        prices[mid] = (m["in"] / 1e6, m["out"] / 1e6)
    return prices
