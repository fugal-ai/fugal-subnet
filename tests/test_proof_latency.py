"""Protocol v2: per-result `latency_ms` in the attested proof (for the v2 time gate).

The field follows the `failed` precedent: present only when set, so a proof without it
hashes exactly as before and a v1 miner and validator keep agreeing on every honest proof.
"""
import pytest

from fugal_subnet.tee.proof import QuestionResult


def _r(**kw):
    base = dict(question_id="q1", routed_model="m/a", correct=True, cost_usd=0.001,
                response_hash="ab" * 32, prompt_tokens=10, completion_tokens=20)
    base.update(kw)
    return QuestionResult(**base)


def test_a_result_without_latency_hashes_exactly_as_before():
    p = _r().hash_payload()
    assert "latency_ms" not in p
    assert set(p) == {"question_id", "routed_model", "correct", "cost_usd", "response_hash",
                      "prompt_tokens", "completion_tokens", "is_exploration"}


def test_latency_is_hashed_when_present_and_survives_the_json_round_trip():
    r = _r(latency_ms=41234)
    assert r.hash_payload()["latency_ms"] == 41234
    back = QuestionResult.from_dict(r.to_dict())
    assert back.latency_ms == 41234 and back.hash_payload() == r.hash_payload()
    # A v1 result round-trips without growing the field.
    assert QuestionResult.from_dict(_r().to_dict()).latency_ms is None


@pytest.mark.parametrize("bad", [True, 1.5, -1, "100", None])
def test_a_malformed_latency_is_rejected_not_reinterpreted(bad):
    d = _r(latency_ms=7).to_dict()
    d["latency_ms"] = bad
    with pytest.raises(ValueError):
        QuestionResult.from_dict(d)


def test_the_harness_measures_each_calls_wall_time(monkeypatch):
    from fugal_subnet.tee import harness

    def fake_call(proxy, model_id, question):
        return "ok"

    monkeypatch.setattr(harness, "_call_model", fake_call)

    class Proxy:
        records = []

    out = list(harness._call_all(Proxy(), [({"question_id": "a"}, "m", False)], "e1"))
    latency = out[0][-1]
    assert isinstance(latency, int) and 0 <= latency < 5000
