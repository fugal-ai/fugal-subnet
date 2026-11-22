"""Protocol v4 pooling (docs/decisions/LIVE_MENU_V4.md, section 8, "For the
validator side"): a pooled pair carries the weight of the set its question
was drawn in, the shared set's or the private set's mixture, and the v4
coldkey cap reaches the odds input. v3 pooling is unchanged byte for byte.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fugal_subnet import config, epoch_plan
from fugal_subnet import explore_pool as xp
from fugal_subnet import explore_weights as xw
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as mf

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _recorded_answers(models):
    out = []
    for i in range(400):
        h = hashlib.sha256(f"ans-{i}".encode()).digest()
        out.append(xp.ExploreAnswer(
            f"q{h[0] % 90:03d}", models[h[1] % len(models)], f"ck{h[2] % 7}", bool(h[3] & 1),
            h[4] % 13 == 0, private=bool(h[5] & 1), prompt_tokens=h[6] * 7,
            completion_tokens=h[7] * 31, latency_ms=h[8] * 3))
    return out


def test_v3_pooling_with_every_published_v3_weight_is_byte_identical():
    """Recorded before the v4 weights were added: v3's pool, with the
    weights every published v3 reveal carried, gives the same rows."""
    doc = json.loads((FIXTURES / "v3_published_explore_weights.json").read_text())
    recorded = {"menu_a": "08cb0a5fbe0598d5b7cb9997762c1f14c12dcafe6b10ebb11af8bbb53551bfb8",
                "menu_b": "262454cafc74137eea8cbc1c66f91329d5b591afcb51fb0a709674bc3a2d3f7e"}
    for key, digest in recorded.items():
        w = doc[key]["weights"]
        pairs = xp.pool_pairs(_recorded_answers(sorted(w)), nonce=b"pool-record",
                              coldkey_cap=xp.coldkey_cap(3), weights=w)
        rows = xp.pairs_for_reveal(pairs, {})
        got = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())
        assert got.hexdigest() == digest, key


def test_the_coldkey_cap_is_v3s_before_v4_and_a_private_set_and_a_half_from_v4():
    assert xp.coldkey_cap(3) == config.EXPLORE_COLDKEY_CAP
    assert xp.coldkey_cap(2) == config.EXPLORE_COLDKEY_CAP
    assert xp.coldkey_cap(4) == config.EXPLORE_COLDKEY_CAP_V4 == 168


def _shares():
    state = ms.carry(mf.state())
    counts = {"lab-a/model-one@high": 150, "lab-a/model-one@low": 10}
    return epoch_plan.explore_shares_v4(state, counts)


def _answer(qid, model, ck="ck1", private=True, correct=True):
    return xp.ExploreAnswer(qid, model, ck, correct, False, private=private,
                            prompt_tokens=100, completion_tokens=200)


def test_a_shared_pair_weighs_the_shared_sets_weight_and_a_private_pair_the_mixtures():
    shares = _shares()
    shared, private = shares.shared_weights(), shares.private_weights()
    assert shared != private
    answers = [_answer("q-shared", "lab-a/model-one@low", private=False),
               _answer("q-shared", "lab-a/model-one@low", ck="ck2", private=False),
               _answer("q-private", "lab-a/model-one@low", private=True)]
    pairs = {p.question_id: p for p in xp.pool_pairs(
        answers, nonce=b"n", coldkey_cap=xp.coldkey_cap(4), weights=shared,
        private_weights=private)}
    assert pairs["q-shared"].weight == shares.pair_weight("lab-a/model-one@low", private=False)
    assert pairs["q-private"].weight == shares.pair_weight("lab-a/model-one@low", private=True)
    assert pairs["q-private"].weight == private["lab-a/model-one@low"] != \
        shared["lab-a/model-one@low"]


def test_without_private_weights_every_pair_weighs_the_one_table_as_in_v3():
    shares = _shares()
    answers = [_answer("q1", "lab-b/model-two", private=False),
               _answer("q2", "lab-b/model-two", private=True)]
    pairs = xp.pool_pairs(answers, nonce=b"n", coldkey_cap=36, weights=shares.shared_weights())
    assert {p.weight for p in pairs} == {shares.shared_weights()["lab-b/model-two"]}


def test_a_pair_answered_in_both_sets_has_no_one_weight_and_is_refused():
    """Every node draws the shared and private sets disjoint, so a question
    is in one of them; answers that say otherwise are a caller's mistake."""
    shares = _shares()
    answers = [_answer("q1", "lab-b/model-two", private=False),
               _answer("q1", "lab-b/model-two", ck="ck2", private=True)]
    with pytest.raises(ValueError, match="shared and private"):
        xp.pool_pairs(answers, nonce=b"n", coldkey_cap=60, weights=shares.shared_weights(),
                      private_weights=shares.private_weights())


def test_private_weights_need_the_shared_table_too():
    shares = _shares()
    with pytest.raises(ValueError, match="both"):
        xp.pool_pairs([_answer("q1", "lab-b/model-two")], nonce=b"n", coldkey_cap=60,
                      private_weights=shares.private_weights())


def test_the_v4_cap_keeps_168_private_pairs_of_one_operator():
    models = list(mf.state().ids())
    answers = [_answer(f"q{i:03d}", models[i % 3]) for i in range(300)]
    w = _shares()
    pooled = xp.pool_pairs(answers, nonce=b"n", coldkey_cap=xp.coldkey_cap(4),
                           weights=w.shared_weights(), private_weights=w.private_weights())
    assert len(pooled) == 168
    assert len(xp.pool_pairs(answers, nonce=b"n", coldkey_cap=xp.coldkey_cap(3))) == \
        config.EXPLORE_COLDKEY_CAP


def test_the_weights_are_exact_inverse_probabilities_of_each_set():
    shares = _shares()
    probs = shares.part_probabilities()
    m = len(shares.models())
    for e in shares.models():
        assert shares.shared_weights()[e] == xw._round_half_up(
            xw.WEIGHT_SCALE / (m * probs["shared"][e]))
        assert shares.private_weights()[e] == xw._round_half_up(
            xw.WEIGHT_SCALE / (m * probs["private"][e]))
