"""Protocol v4: the TD's dollar budget for one challenge.

The operator states a ceiling (--v4-challenge-usd, FUGAL_V4_CHALLENGE_USD,
provisioning field `v4_challenge_usd`: whole dollars, 1 to
config.CHALLENGE_SPEND_MAX_USD_V4, default config.CHALLENGE_SPEND_USD_V4,
$100). It bounds REALISED spend, with an ESTIMATE pre-check:

- before the first call the TD estimates the challenge's cost from the
  epoch's pick-cost table (td_v4.challenge_estimate: each own pick and each
  holder-side pick where the two differ, at its entry's pick cost) and makes
  no challenge whose estimate exceeds the ceiling, logging both;
- while running, no challenge call starts once the challenge's realised
  spend plus the pick-cost estimate of its calls in flight reaches the
  ceiling; what is not started is handled as a time cut is.

No network, no spend.
"""
from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest

from fugal_subnet import config
from fugal_subnet import menu_state as ms
from fugal_subnet.tee import harness, provision, td_v4, v4_options
from fugal_subnet.tee.runtime import APICallRecord
from tests import menu_fixtures as mf
from tests import test_harness_v4 as th


@pytest.mark.parametrize("text,value", [("1", 1), ("5000", 5000), ("50000", 50000)])
def test_the_budget_is_whole_dollars_from_1_to_the_maximum(text, value):
    assert v4_options.parse_challenge_usd(text) == value
    assert v4_options.check_pushed({"v4_challenge_usd": text}) == {"v4_challenge_usd": value}


@pytest.mark.parametrize("text", ["0", "050", "50001", "1.5", " 5", "+5", "-5", "1e3", "",
                                  "5 000", "9999999", "x", 5000, None])
def test_anything_else_is_refused_naming_the_field(text):
    with pytest.raises(v4_options.OptionRefused) as e:
        v4_options.parse_challenge_usd(text)
    assert "FUGAL_V4_CHALLENGE_USD" in str(e.value)


def test_the_option_travels_like_the_others():
    assert v4_options.FIELDS["FUGAL_V4_CHALLENGE_USD"] == "v4_challenge_usd"
    assert "v4_challenge_usd" in provision.ALLOWED_FIELDS
    assert v4_options.pushed_options({"v4_challenge_usd": "800"}) == {
        "FUGAL_V4_CHALLENGE_USD": "800"}
    assert 1 <= config.CHALLENGE_SPEND_USD_V4 <= config.CHALLENGE_SPEND_MAX_USD_V4


def test_the_policy_carries_the_budget_and_refuses_one_out_of_range():
    pol = td_v4.ChallengePolicy(500, first_epoch=10, read_holder=lambda e: None,
                                holder_bytes=lambda h: None, spend_usd=700)
    assert pol(11) == td_v4.ChallengeSpec(500, None, None, 700)
    assert td_v4.ChallengePolicy(0, first_epoch=10).spend_usd == config.CHALLENGE_SPEND_USD_V4
    for bad in (0, config.CHALLENGE_SPEND_MAX_USD_V4 + 1, 5.0, "5"):
        with pytest.raises(ValueError, match="dollar budget"):
            td_v4.ChallengePolicy(500, first_epoch=10, read_holder=lambda e: None,
                                  holder_bytes=lambda h: None, spend_usd=bad)


def test_the_default_ceiling_is_100_dollars_and_the_maximum_unchanged():
    assert config.CHALLENGE_SPEND_USD_V4 == 100
    assert config.CHALLENGE_SPEND_MAX_USD_V4 == 50000
    assert not hasattr(td_v4, "challenge_worst_case")


def test_the_estimate_is_exact_from_the_epochs_pick_costs():
    state = ms.load_seed()
    a, b, c = state.ids()[:3]
    got = td_v4.challenge_estimate(state, [a, a, b], [b, a, None])
    cost = ms.pick_costs(state)
    # own picks always; the holder's only where it differs from the own pick
    assert got == cost[a] + cost[b] + cost[a] + cost[b]
    assert isinstance(got, Fraction)
    assert td_v4.challenge_estimate(state, [], []) == 0
    with pytest.raises(ValueError):
        td_v4.challenge_estimate(state, [a], [])


def _estimate_of(monkeypatch):
    """The estimate the harness computed in its last run."""
    seen = []
    real = td_v4.challenge_estimate

    def spy(*a, **k):
        seen.append(real(*a, **k))
        return seen[-1]
    monkeypatch.setattr(td_v4, "challenge_estimate", spy)
    return seen


def test_a_challenge_whose_estimate_exceeds_the_ceiling_is_not_made_and_is_logged(
        monkeypatch, caplog):
    mf.allow_small_menus(monkeypatch)
    monkeypatch.setattr(td_v4, "challenge_estimate", lambda *a, **k: Fraction(301, 100))
    with caplog.at_level("ERROR"):
        proof, provider, _ = th._run(monkeypatch, challenge=td_v4.ChallengeSpec(
            th.SLICE, None, None, 3))
    assert proof.challenge is None and proof.scored is False
    assert not proof.scored_results and not proof.holder_results
    text = caplog.text
    assert "$3.01" in text and "$3" in text and "FUGAL_V4_CHALLENGE_USD" in text
    # the exploration proof still goes out; only the challenge is not made
    assert proof.results


def test_a_challenge_whose_estimate_is_within_the_ceiling_runs_whatever_its_worst_case(
        monkeypatch):
    mf.allow_small_menus(monkeypatch)
    seen = _estimate_of(monkeypatch)
    proof, _p, _ = th._run(monkeypatch, challenge=td_v4.ChallengeSpec(th.SLICE, None, None, 1))
    assert seen and seen[-1] <= 1
    assert proof.challenge is not None and len(proof.scored_results) == th.SLICE


def _items(n_scored=30, n_explore=5, eid="e"):
    req = {"model": "m/x", "max_tokens": 10, "provider": {"order": ["h"], "allow_fallbacks": False}}
    return ([({"question_id": f"q{i}"}, eid, "scored", req, eid) for i in range(n_scored)]
            + [({"question_id": f"x{i}"}, eid, "explore", req, eid) for i in range(n_explore)])


def _recording_call(proxy, cost, gate=None, completion_tokens=1):
    def call(proxy_, model_id, question, request=None, reasoning=None, deadline_s=None):
        if gate is not None:
            gate.wait(5)
        proxy.records.append(APICallRecord(model_id=model_id, prompt_tokens=1,
                                           completion_tokens=completion_tokens, cost_usd=cost,
                                           timestamp=0.0, response_hash="",
                                           request_id=harness.current_request_id()))
        return "ok"
    return call


def test_no_challenge_call_starts_once_realised_spend_reaches_the_ceiling(monkeypatch):
    proxy = SimpleNamespace(records=[])
    monkeypatch.setattr(harness, "_call_model", _recording_call(proxy, 0.5))
    monkeypatch.setattr(harness, "HARNESS_CONCURRENCY", 1)
    out = harness._call_all_v4(proxy, _items(), "e", spend_cap=5,
                               call_estimate={"e": Fraction(1, 100)})
    made = [not o.not_started for o in out]
    # one at a time, $0.50 a call realised: ten challenge calls reach $5; the
    # rest are not started, and exploration is never cut
    assert made[:30] == [True] * 10 + [False] * 20
    assert all(made[30:])


def test_calls_in_flight_count_at_their_estimate_before_they_are_paid(monkeypatch):
    import threading

    proxy = SimpleNamespace(records=[])
    release = threading.Event()
    monkeypatch.setattr(harness, "_call_model", _recording_call(proxy, 0.01, gate=release))
    monkeypatch.setattr(harness, "HARNESS_CONCURRENCY", 4)
    threading.Timer(0.5, release.set).start()
    out = harness._call_all_v4(proxy, _items(n_scored=12, n_explore=3), "e", spend_cap=3,
                               call_estimate={"e": Fraction(1)})
    made = [not o.not_started for o in out]
    # $1 estimated a call and nothing paid yet: three start (0, 1 and 2 in
    # flight are under $3), the fourth would reach the ceiling, and from
    # there on none starts; exploration is never cut
    assert made[:12] == [True] * 3 + [False] * 9
    assert all(made[12:])
    # what was not started is a suffix of the call order, as a time cut's is
    assert all(o.failed for o, m in zip(out[:12], made[:12]) if not m)


def test_one_loop_applies_both_budgets_and_never_cuts_exploration(monkeypatch):
    """Challenge calls under the time budget and the dollar ceiling, a
    provisional entry's exploration under PROVISIONAL_SPEND_CAP_USD, and full
    entries' exploration under neither."""
    proxy = SimpleNamespace(records=[])
    monkeypatch.setattr(harness, "_call_model", _recording_call(proxy, 0.5,
                                                                completion_tokens=1000))
    monkeypatch.setattr(harness, "HARNESS_CONCURRENCY", 1)
    monkeypatch.setattr(td_v4, "check_wire_request", lambda r: None)
    req = {"model": "m/x"}
    items = ([({"question_id": f"q{i}"}, "full", "scored", req, "full") for i in range(20)]
             + [({"question_id": f"p{i}", "prompt": "x"}, "prov", "explore", req, "prov")
                for i in range(10)]
             + [({"question_id": f"x{i}"}, "full", "explore", req, "full") for i in range(10)])
    spend = harness._ProvisionalSpend({"prov": (10**6, 10**6, 1000)}, cap=Fraction(1, 100))
    out = harness._call_all_v4(proxy, items, "e", budget_s=1800, spend_cap=4,
                               call_estimate={"full": Fraction(1, 10)},
                               provisional=frozenset({"prov"}), spend=spend)
    made = [not o.not_started and not o.failed for o in out]
    # $0.50 realised a challenge call, one at a time: eight reach $4
    assert made[:20] == [True] * 8 + [False] * 12
    assert all(o.not_started for o in out[8:20])
    # each provisional attempt holds back 3 x (1000 x $1/M + the prompt's
    # allowance) and is settled at about $0.001: the $0.01 cap admits a few,
    # the rest are not made (failed, never not_started: the challenge's cut
    # never applies to exploration)
    prov = out[20:30]
    assert 0 < sum(not o.failed for o in prov) < 10
    assert not any(o.not_started for o in prov)
    assert all(not o.failed and not o.not_started for o in out[30:])


def test_the_miner_takes_the_budget_from_one_source():
    from neurons import miner

    assert miner._parse_v4_challenge_usd(None) is None
    assert miner._parse_v4_challenge_usd("700") == 700
    push = miner._v4_pushed_options
    assert push({"v4_challenge_usd": "800"}, 4, env=(None, "0", None), v4_from="options") == (
        None, "0", 800, "provisioning")
    assert push({}, 4, env=(None, "0", 700), v4_from="options") == (None, "0", 700, "options")
    assert push({}, 4, env=(None, "0"), v4_from="options") == (None, "0", "options")
    with pytest.raises(ValueError, match="both"):
        push({"v4_challenge_usd": "800"}, 4, env=(None, "0", 700), v4_from="options")
    ok, line = miner._miner_ready_line(4, 3600, v4_usd=800, root_hotkeys=((0, "5Froot"),))
    assert "challenge budget $800 (realised spend, estimate checked first)" in line
    assert miner._challenge_usd(None) == config.CHALLENGE_SPEND_USD_V4
