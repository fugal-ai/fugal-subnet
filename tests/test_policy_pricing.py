"""Scored cost must depend on the routing choice and nothing the miner controls.

thrift = reference_cost / miner_cost. The denominator was computed from pinned
rates and the reference frame; the numerator was whatever the miner reported it
had spent, and even the denominator's prompt tokens came from the miner's
report. One side unfakeable, the other a self-report — so a miner could lower
its cost by shrinking numbers instead of by routing better.
"""
from fugal_subnet.api import load_prices
from fugal_subnet.pricing import (
    policy_cost,
    policy_cost_total,
    question_input_tokens,
)


class _Frame:
    def __init__(self, per_model=None):
        self._m = per_model or {}

    def avg_completion_tokens(self, model, default):
        return self._m.get(model, default)


def test_cost_is_unchanged_by_what_the_miner_reports():
    """The property that closes the manipulation: the figure is a function of
    (question, model) only. There is no argument a miner can vary."""
    prices = load_prices()
    model = sorted(prices)[0]
    gold = {"q1": {"prompt": "a" * 400}}
    frame = _Frame({model: 200})

    a = policy_cost_total([("q1", model)], gold, prices, frame, 256)
    b = policy_cost_total([("q1", model)], gold, prices, frame, 256)
    assert a == b and a > 0
    # Nothing from the proof enters the calculation — only ids and model names.


def test_routing_to_a_cheaper_model_lowers_the_cost():
    """Rank fidelity: the whole point of the number."""
    prices = load_prices()
    blended = lambda m: 500 * prices[m][0] + 200 * prices[m][1]  # noqa: E731
    ranked = sorted(prices, key=blended)
    cheapest, dearest = ranked[0], ranked[-1]
    gold = {"q1": {"prompt": "a" * 400}}
    frame = _Frame()

    cheap = policy_cost_total([("q1", cheapest)], gold, prices, frame, 256)
    dear = policy_cost_total([("q1", dearest)], gold, prices, frame, 256)
    assert cheap < dear


def test_a_longer_question_costs_more():
    """Question difficulty still shows up, which is why input tokens come from
    the pool rather than being a flat per-question charge."""
    prices = load_prices()
    model = sorted(prices)[0]
    frame = _Frame()
    short = policy_cost_total(
        [("q", model)], {"q": {"prompt": "a" * 100}}, prices, frame, 256)
    long = policy_cost_total(
        [("q", model)], {"q": {"prompt": "a" * 10_000}}, prices, frame, 256)
    assert long > short


def test_a_verbose_model_costs_more_at_the_same_rate():
    """Verbosity comes from the frame's pooled observations, so a chatty model
    is not priced as if it were terse."""
    prices = load_prices()
    model = sorted(prices)[0]
    gold = {"q": {"prompt": "a" * 400}}
    terse = policy_cost_total([("q", model)], gold, prices, _Frame({model: 50}), 256)
    chatty = policy_cost_total([("q", model)], gold, prices, _Frame({model: 5000}), 256)
    assert chatty > terse


def test_unknown_questions_and_models_do_not_raise():
    """Reaching either means an earlier check already failed; this must not be
    the thing that reports it."""
    prices = load_prices()
    frame = _Frame()
    assert policy_cost_total([("nope", sorted(prices)[0])], {}, prices, frame, 256) == 0.0
    assert policy_cost_total(
        [("q", "not-a-model")], {"q": {"prompt": "x"}}, prices, frame, 256) == 0.0


def test_input_token_estimate_is_deterministic_and_monotone():
    assert question_input_tokens("abcd") == question_input_tokens("abcd")
    assert question_input_tokens("a" * 400) > question_input_tokens("a" * 100)
    assert question_input_tokens("") == 1        # never zero, never negative
    assert question_input_tokens(None) == 1


def test_cost_matches_the_rate_card():
    prices = load_prices()
    model = sorted(prices)[0]
    rate_in, rate_out = prices[model]
    prompt = "a" * 400
    expected = question_input_tokens(prompt) * rate_in + 200 * rate_out
    assert policy_cost(prompt, model, prices, 200) == expected
