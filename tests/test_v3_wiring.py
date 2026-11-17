"""Protocol v3 wiring: which epochs are v3, what a menu entry sends, the
embeddings file, and how a seat state built under other rules is reset.

docs/INVARIANTS.md, "I1 — protocol v3 activates at one named boundary".
"""
import dataclasses
import hashlib
import io
import json

import numpy as np
import pytest

from fugal_subnet import routing_protocol
from fugal_subnet.benchmarks.slicer import protocol_version_for_boundary
from fugal_subnet.head_odds import HEAD_MAX_ABS_PARAM
from fugal_subnet.tee import harness
from neurons import validator
from tests.test_v2_epoch import run
from tests.test_v3_epoch import POLICIES, UIDS, H, odds_input, params

# --- which epochs are v3 -------------------------------------------------------


def test_the_boundary_decides_the_protocol_version():
    v = protocol_version_for_boundary
    assert [v(b, 1000, 2000) for b in (999, 1000, 1999, 2000, 5000)] == [1, 2, 2, 3, 3]
    assert v(5000, 1000) == 2 and v(5000, None) == 1            # v3 not scheduled
    for bad in ((None, 2000), (3000, 2000)):
        with pytest.raises(ValueError, match="before protocol v2"):
            v(5000, *bad)


# --- a menu entry is a way of calling a model ------------------------------------

MENU = [
    {"id": "openai/gpt-6.1-sol@high", "model": "openai/gpt-6.1-sol", "in": 2.0, "out": 8.0,
     "reasoning": {"effort": "high"}, "max_tokens": 16384,
     "provider": {"order": ["openai", "azure"], "allow_fallbacks": False}},
    {"id": "openai/gpt-6.1-sol@low", "in": 2.0, "out": 8.0, "reasoning": {"effort": "low"}},
    {"id": "anthropic/claude-haiku-5.5", "in": 1.0, "out": 5.0},
]


@pytest.fixture
def menu(tmp_path, monkeypatch):
    path = tmp_path / "models_v3.json"
    path.write_text(json.dumps(MENU))
    monkeypatch.setattr(routing_protocol, "PRICES_PATH_V3", path)
    return path


def test_a_v3_entry_sends_its_pinned_request(menu):
    r = routing_protocol.request_for(3, "openai/gpt-6.1-sol@high", 16384)
    assert r == {"model": "openai/gpt-6.1-sol", "max_tokens": 16384,
                 "reasoning": {"effort": "high"},
                 "provider": {"order": ["openai", "azure"], "allow_fallbacks": False}}
    # Defaults: the model is the id before '@', the budget is the harness's.
    assert routing_protocol.request_for(3, "openai/gpt-6.1-sol@low", 9000) == {
        "model": "openai/gpt-6.1-sol", "max_tokens": 9000, "reasoning": {"effort": "low"}}
    assert routing_protocol.request_for(3, "anthropic/claude-haiku-5.5", 9000) == {
        "model": "anthropic/claude-haiku-5.5", "max_tokens": 9000}
    with pytest.raises(KeyError):
        routing_protocol.request_for(3, "openai/gpt-6.1-sol", 9000)
    # The price table and the model index space are keyed on the entry id.
    assert routing_protocol.models_for_protocol(3) == sorted(r["id"] for r in MENU)
    assert routing_protocol.price_table(3)["openai/gpt-6.1-sol@low"] == (2e-6, 8e-6)
    routing_protocol.validate_menu_v3(MENU)


def test_v2_requests_are_what_they_were():
    model = routing_protocol.models_for_protocol(2)[0]
    r = routing_protocol.request_for(2, model, 16384)
    assert r["model"] == model and r["max_tokens"] == 16384 and "provider" not in r
    assert r.get("reasoning") == routing_protocol.reasoning_for(2, model)
    assert routing_protocol.request_for(1, "x/y", 5) == {"model": "x/y", "max_tokens": 5}


@pytest.mark.parametrize("row, why", [
    ({"id": "a/b@x@y", "in": 1, "out": 1}, "at most one '@'"),
    ({"id": "a/b@high", "model": "a/c", "in": 1, "out": 1}, "not the id before"),
    ({"id": "a/b", "in": -1, "out": 1}, "non-negative price"),
    ({"id": "a/b", "in": 1, "out": 1, "reasoning": {"effort": "high", "max_tokens": 5}}, "exactly one"),
    ({"id": "a/b", "in": 1, "out": 1, "max_tokens": 0}, "positive integer"),
    ({"id": "a/b", "in": 1, "out": 1, "provider": {"order": ["p"]}}, "allow_fallbacks"),
    ({"id": "a/b", "in": 1, "out": 1, "provider": {"order": [], "allow_fallbacks": False}}, "pin an `order`"),
    ({"id": "x" * 300, "in": 1, "out": 1}, "too long"),
])
def test_a_menu_a_harness_could_not_call_is_refused(row, why):
    with pytest.raises(ValueError, match=why):
        routing_protocol.validate_menu_v3([row])
    with pytest.raises(ValueError, match="repeated"):
        routing_protocol.validate_menu_v3([MENU[2], MENU[2]])
    with pytest.raises(ValueError, match="non-empty"):
        routing_protocol.validate_menu_v3([])


def test_the_shipped_v3_menu_and_manifest_are_callable_and_agree():
    """The pinned v3 menu (check_safety_invariants pins its hash) is a menu a
    harness can call, its token manifest has one row per entry, and every
    entry sends its own answer budget and pinned hosts."""
    from fugal_subnet.vendor import success_contract as contract

    rows = json.loads(routing_protocol.PRICES_PATH_V3.read_text())
    routing_protocol.validate_menu_v3(rows)
    manifest = json.loads(routing_protocol.MANIFEST_PATH_V3.read_text())
    contract.validate_manifest(manifest)
    assert sorted(r["id"] for r in manifest["models"]) == routing_protocol.models_for_protocol(3)
    for r in rows:
        req = routing_protocol.request_for(3, r["id"], 16384)
        assert req["max_tokens"] == r["max_tokens"] and req["provider"]["allow_fallbacks"] is False


def test_the_harness_sends_a_v3_entrys_request_as_pinned(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "B"}}]}).encode())

    monkeypatch.setattr(harness, "urlopen", fake_urlopen)
    proxy = type("P", (), {"port": 1})()
    request = {"model": "openai/gpt-6.1-sol", "max_tokens": 16384, "reasoning": {"effort": "high"},
               "provider": {"order": ["openai"], "allow_fallbacks": False}}
    # Without a system prompt the request goes out as pinned, plus the user turn.
    monkeypatch.setattr(harness, "HARNESS_SYSTEM_PROMPT_V3", None)
    assert harness._call_model(proxy, "openai/gpt-6.1-sol@high", {"prompt": "2+2?"},
                               request=request) == "B"
    assert sent["body"] == {**request, "messages": [{"role": "user", "content": "2+2?"}]}
    # With a pinned system prompt it leads the messages; the entry id is never on the wire.
    monkeypatch.setattr(harness, "HARNESS_SYSTEM_PROMPT_V3", "Answer the question.")
    harness._call_model(proxy, "openai/gpt-6.1-sol@high", {"prompt": "2+2?"}, request=request)
    assert sent["body"]["messages"][0] == {"role": "system", "content": "Answer the question."}
    assert "@" not in json.dumps(sent["body"])


def test_the_v3_image_pins_a_shared_system_prompt_that_yields_to_answer_formats():
    """The shipped prompt is set (no v3 call goes out bare) and tells the model
    that a requested answer format wins: numeric_final reads the LAST number and
    letter_mcq loses a reply that opens with a letter and then explains."""
    from fugal_subnet import config as cfg

    text = cfg.HARNESS_SYSTEM_PROMPT_V3
    assert text and harness.HARNESS_SYSTEM_PROMPT_V3 == text
    assert "takes precedence" in text and "final number at the end" in text
    assert "Lead with the answer" not in text


# --- the embeddings file -------------------------------------------------------


def write_embeddings(path, ids, hidden):
    np.savez(path, question_ids=np.array(ids), H=hidden)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_embeddings_file_is_checked_against_its_pin_and_its_bounds(tmp_path):
    H_ = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
    H_ /= np.linalg.norm(H_, axis=1, keepdims=True)
    path = tmp_path / "pool_v3.npz"
    digest = write_embeddings(path, ["q1", "q2", "q3"], H_)
    emb = validator.Embeddings.load(str(path), digest)
    assert np.array_equal(emb.of("q2"), H_[1]) and emb.covers(["q1", "q9"]) == ["q9"]
    with pytest.raises(ValueError, match="the pinned one is"):
        validator.Embeddings.load(str(path), "0" * 64)
    with pytest.raises(ValueError, match="repeated"):
        validator.Embeddings(["q1", "q1"], H_[:2])
    with pytest.raises(ValueError, match="one row per question"):
        validator.Embeddings(["q1"], H_)
    with pytest.raises(ValueError, match="larger than"):
        validator.Embeddings(["q1", "q2", "q3"], H_ * 50)
    with pytest.raises(ValueError, match="not finite"):
        validator.Embeddings(["q1"], np.array([[np.nan] * 8], dtype=np.float32))


# --- what the validator turns proofs into ---------------------------------------


def test_an_answer_is_private_unless_its_pair_is_in_the_shared_set():
    from types import SimpleNamespace as NS

    def res(q, m, ok=True, failed=False):
        return NS(question_id=q, routed_model=m, correct=ok, failed=failed,
                  prompt_tokens=11, completion_tokens=22)

    proofs = {4: NS(exploration_results=[res("s1", "m/a"), res("p1", "m/b", ok=False)]),
              2: NS(exploration_results=[res("s1", "m/a", failed=True)])}
    answers = validator._v3_explore_answers(proofs, {2: "ck2", 4: "ck4"}, {"s1": "m/a"})
    assert [(a.coldkey, a.question_id, a.private, a.correct, a.failed) for a in answers] == [
        ("ck2", "s1", False, True, True), ("ck4", "s1", False, True, False),
        ("ck4", "p1", True, False, False)]
    assert answers[1].prompt_tokens == 11 and answers[1].completion_tokens == 22


def test_a_v3_head_outside_the_parameter_bound_is_refused(monkeypatch):
    from types import SimpleNamespace as NS

    ok = NS(W=np.zeros((2, 4), dtype=np.float32), b=np.zeros(2, dtype=np.float32), models=["a", "b"])
    big = NS(W=np.full((2, 4), HEAD_MAX_ABS_PARAM * 2, dtype=np.float32), b=ok.b, models=ok.models)
    from fugal_subnet import head_eval
    monkeypatch.setattr(head_eval, "load_head_from_npz", lambda data: {b"ok": ok, b"big": big}[data])
    assert validator._head_within_bounds(b"ok") is None
    assert "larger than" in validator._head_within_bounds(b"big")
    assert validator._head_within_bounds(b"missing") is not None      # a verdict, never a raise
    heads = validator._v3_heads({3: b"ok", 9: b"missing"})
    assert set(heads) == {3} and heads[3][2] == ["a", "b"]


def test_the_rules_tag():
    assert validator._rules_tag(2, False) is None
    assert validator._rules_tag(3, False) == {"protocol": 3, "seat": "net_value"}
    assert validator._rules_tag(3, True) == {"protocol": 3, "seat": "odds"}


# --- a state built under other rules is not carried over ----------------------------

NV3 = {"protocol": 3, "seat": "net_value"}
ODDS3 = {"protocol": 3, "seat": "odds"}


def _epochs(n, rules, state_rules=None, window=None, seat=None, copy=None, start=0, odds=False):
    r = None
    for e in range(start, start + n):
        r = run(e, {u: POLICIES[u] for u in UIDS}, window=window, seat=seat, copy=copy,
                params=params(odds), odds_input=odds_input(e), rules=rules,
                state_rules=state_rules)
        window, seat, copy, state_rules = r.window, r.decision.state, r.copy_state, rules
    return r


def test_the_first_v3_epoch_starts_from_an_empty_window_and_seat():
    """v3's questions and model list are different ones: a v2 window cannot be
    compared with, and a v2 champion was seated on other questions."""
    v2 = _epochs(3, rules=None)
    assert v2.decision.champion is not None and "rules" not in v2.reveal["v2_state"]
    first = run(3, {u: POLICIES[u] for u in UIDS}, window=v2.window, seat=v2.decision.state,
                copy=v2.copy_state, params=params(False), odds_input=odds_input(3),
                rules=NV3, state_rules=None)
    assert sorted(first.window.slices) == [4]                 # only this epoch
    assert first.reveal["v2_state"]["rules"] == NV3
    assert first.decision.state.history == ()                 # the v2 seat is gone
    # The copy rule's record of who committed what first is chain history: kept.
    assert first.copy_state.heads


def test_the_first_odds_epoch_keeps_the_window_and_empties_the_seat():
    nv = _epochs(4, rules=NV3, state_rules=NV3)
    champion = nv.decision.champion
    assert champion is not None
    first = run(4, {u: POLICIES[u] for u in UIDS}, window=nv.window, seat=nv.decision.state,
                copy=nv.copy_state, params=params(True), odds_input=odds_input(4),
                rules=ODDS3, state_rules=NV3)
    assert sorted(first.window.slices) == [1, 2, 3, 4, 5]      # the window is kept
    assert first.decision.event in ("first", "empty")          # decided afresh, on odds
    assert first.decision.state.history == ()
    assert first.decision.champion == H(6)                     # the head that predicts
    assert first.reveal["v2_state"]["rules"] == ODDS3


def test_the_same_rules_carry_the_state_over():
    a = _epochs(4, rules=ODDS3, state_rules=ODDS3, odds=True)
    nxt = run(4, {u: POLICIES[u] for u in UIDS}, window=a.window, seat=a.decision.state,
              copy=a.copy_state, params=params(True), odds_input=odds_input(4),
              rules=ODDS3, state_rules=ODDS3)
    assert nxt.decision.event == "hold" and sorted(nxt.window.slices) == [1, 2, 3, 4, 5]


def test_a_rebuilt_validator_reads_the_rules_its_state_was_built_under(tmp_path):
    from fugal_subnet.v2_rebuild import rebuild_v2_from_reveals
    from tests.test_v2_epoch import CATEGORY, PRICES
    from tests.test_v2_rebuild import reveal_of, write

    pols = {u: POLICIES[u] for u in UIDS}
    r = _epochs(2, rules=NV3, state_rules=NV3)
    write(tmp_path, [json.loads(json.dumps(reveal_of(1, r, pols)))])
    rb = rebuild_v2_from_reveals(tmp_path, window_size=48, prices=PRICES, bench_category=CATEGORY)
    assert rb.extra["rules"] == NV3


def test_v3_seat_params_never_read_the_anchor_model(monkeypatch):
    """From v3 the anchor model can leave the menu: V is a pinned number."""
    from fugal_subnet import v2_epoch

    def boom(model):
        raise AssertionError("the anchor model was read")

    monkeypatch.setattr(v2_epoch, "anchor_spend_per_question", boom)
    p = v2_epoch.seat_params_from_config(protocol_version=3)
    assert dataclasses.asdict(p)["anchor_spend"] == 0.49


@pytest.mark.parametrize("finney_name, env_name", [
    ("FINNEY_V3_ACTIVATION_BLOCK", "V3_ACTIVATION_BLOCK_ENV"),
    ("FINNEY_ODDS_SEAT_BLOCK", "ODDS_SEAT_BLOCK_ENV"),
    ("FINNEY_CONFIRM_ACTIVATION_BLOCK", "CONFIRM_ACTIVATION_BLOCK_ENV"),
    ("FINNEY_MENU_B_BLOCK", "MENU_B_BLOCK_ENV"),
])
def test_an_activation_variable_that_repeats_finneys_block_leaves_the_digest_alone(
        monkeypatch, finney_name, env_name):
    """Finney accepts an activation variable only when it equals the compiled
    block, so setting it changes no rule; the digest must not say otherwise
    (the Release A review fix, applied to the v3 and odds seat blocks too)."""
    from fugal_subnet import config, fingerprint

    monkeypatch.setattr(config, finney_name, 9_700_000)
    monkeypatch.setattr(config, env_name, None)
    unset = fingerprint.consensus_digest()
    monkeypatch.setattr(config, env_name, 9_700_000)
    assert fingerprint.consensus_digest() == unset
    monkeypatch.setattr(config, env_name, 1_000)   # another chain's rule still shows
    assert fingerprint.consensus_digest() != unset
