"""OpenRouter API client with retry/backoff and spend tracking.

Adapted from fugal-core/fugal/router.py or_request. Never prints API keys.
"""
from __future__ import annotations
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field

import httpx

from fugal_subnet.config import API_TIMEOUT, API_MAX_RETRIES, API_RETRYABLE_STATUS

logger = logging.getLogger(__name__)

OR_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class SpendTracker:
    """Thread-safe running total of API spend. Safe to share across the
    concurrent workers in matrix.build_matrix."""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    budget_cap_usd: float | None = None
    per_call_log: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int,
               cost_usd: float):
        with self._lock:
            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost_usd += cost_usd
            self.per_call_log.append({
                "model": model, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "cost_usd": cost_usd,
            })

    def check_budget(self):
        with self._lock:
            if self.budget_cap_usd is not None and self.total_cost_usd >= self.budget_cap_usd:
                raise BudgetExceeded(
                    f"Budget cap ${self.budget_cap_usd:.2f} reached "
                    f"(spent ${self.total_cost_usd:.4f} across {self.total_calls} calls)"
                )


class BudgetExceeded(Exception):
    pass


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
) -> tuple[str, int, int]:
    """Call a model via OpenRouter. Returns (response_text, prompt_tokens, completion_tokens).

    Args:
        model: OpenRouter model ID (e.g. "openai/gpt-5.5").
        prompt: The question/task to send.
        max_tokens: Max response tokens.
        temperature: Sampling temperature. 0.0 for deterministic grading.
        timeout: Request timeout in seconds.
        retries: Number of retry attempts.
        tracker: Optional SpendTracker to log costs.
        prices: Optional {model_id: (price_per_input_token, price_per_output_token)}.
    """
    if tracker:
        tracker.check_budget()

    key = _get_api_key()
    messages = [{"role": "user", "content": prompt}]
    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature

    last_err = None
    for attempt in range(retries):
        try:
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

            if tracker:
                cost = 0.0
                if prices and model in prices:
                    pin, pout = prices[model]
                    cost = ptok * pin + ctok * pout
                tracker.record(model, ptok, ctok, cost)

            return text, ptok, ctok

        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_err = e
        except RuntimeError:
            pass
        except ValueError:
            raise

        if attempt < retries - 1:
            delay = (2 ** attempt) + random.random()
            logger.warning("Retry %d/%d for %s (%.1fs delay)", attempt + 1, retries, model, delay)
            time.sleep(delay)

    raise last_err if last_err else RuntimeError(f"call_model({model}) failed after {retries} retries")


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
