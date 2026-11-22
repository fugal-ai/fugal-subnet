"""Protocol v4 in the TD's harness (docs/decisions/LIVE_MENU_V4.md; work package W3).

The TD runs an epoch on the menu its state names, checked against the
committed hash and the compiled rules before anything is called; it reads the
endpoints listing, carries billed cost and the serving host per call, applies
the root's join delta, and runs a provisional entry's calls on their own
deadline with one retry. No network (recorded listing pages, stub model calls) and no
spend.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
from fractions import Fraction

import numpy as np
import pytest

from fugal_subnet import epoch_plan
from fugal_subnet import menu_state as ms
from fugal_subnet.benchmarks.slicer import PoolIndex, select_slice
from fugal_subnet.routing_protocol import identity_v4
from fugal_subnet.tee import harness, td_v4
from fugal_subnet.tee.proof import BenchmarkProof, Challenge
from fugal_subnet.tee.runtime import APICallRecord, MeteringProxy
from tests import menu_fixtures as mf
from tests.test_endpoint_reads import PAGES
from tests.test_tee_e2e import _head, _pool

NONCE = "ab" * 32
HOTKEY = mf.HOTKEY
SLICE = 12
EXPLORE = 4
CATEGORY = {"gsm8k": "math", "math": "math", "mmlu": "knowledge", "aime": "hard"}
TARGETS = {"hard": 0.25, "knowledge": 0.25, "math": 0.5}
FIXTURE_IDS = ["lab-a/model-one@high", "lab-a/model-one@low", "lab-b/model-two"]


@pytest.fixture(autouse=True)
def _small_menus(monkeypatch):
    mf.allow_small_menus(monkeypatch)


# --- the plan W5 and W8 will supply: a stand-in with the same interface ------------


def _exploration(nonce, pool, scored_ids, state, hotkey):
    ids = state.ids()
    rest = sorted((q["question_id"] for q in pool if q["question_id"] not in scored_ids),
                  key=lambda qid: hmac.new(nonce, f"x:{hotkey}:{qid}".encode(),
                                           hashlib.sha256).digest())
    return {qid: ids[i % len(ids)] for i, qid in enumerate(rest[:EXPLORE])}


def _plan(pool) -> td_v4.V4Plan:
    return td_v4.V4Plan(scored_slice=lambda nonce, p, size: select_slice(nonce, p, size),
                        exploration=_exploration, pool_index=PoolIndex(pool, CATEGORY),
                        targets=TARGETS)


# --- the stand-in provider -----------------------------------------------------------


class _Proxy(MeteringProxy):
    def start(self):
        pass


def _listing(model: str, hosts=("host-a/fp8", "host-b/fp8", "host-c"), **ep_over) -> bytes:
    eps = []
    for h in hosts:
        ep = {"tag": h, "pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
              "max_completion_tokens": 131072, "quantization": "fp8" if "/" in h else "unknown",
              "status": 0, "supported_parameters": ["max_tokens", "reasoning", "reasoning_effort"]}
        ep.update(ep_over)
        eps.append(ep)
    return json.dumps({"data": {"id": model, "endpoints": eps}}).encode()


class Provider:
    """Answers stub calls and listing reads; records what was asked."""

    def __init__(self, pages=None, *, fail_ids=()):
        self.calls: list[tuple[str, dict, str]] = []
        self.fetched: list[str] = []
        self.pages = pages
        self.fail_ids = set(fail_ids)

    def fetch(self, model):
        self.fetched.append(model)
        if self.pages is not None:
            return self.pages[model]
        return _listing(model)

    def call(self, proxy, model_id, question, request=None, reasoning=None, deadline_s=None):
        qid = question["question_id"]
        self.calls.append((model_id, request, qid))
        if qid in self.fail_ids:
            return None
        right = hashlib.sha256(f"{qid}|{model_id}".encode()).digest()[0] < 160
        text = question["gold"] if right else "0"
        proxy.records.append(APICallRecord(
            model_id=model_id, prompt_tokens=500, completion_tokens=300,
            cost_usd=proxy.price_call(model_id, 500, 300), timestamp=0.0,
            response_hash=hashlib.sha256(text.encode()).hexdigest(),
            request_id=harness.current_request_id(), billed_nano=1234,
            served_by="Fixture Host"))
        return text


def _inputs(pool, **over) -> td_v4.V4EpochInputs:
    # A challenge on the first SLICE questions of the list (the slice itself,
    # with no holder): the stand-in pool is far smaller than a real list.
    fields = dict(epoch=102, previous=mf.state(), committed_hash=None, published=None,
                  allowlist=mf.ALLOW, fetch_listing=None,
                  challenge=td_v4.ChallengeSpec(SLICE, None))
    fields.update(over)
    if "plan" not in fields:
        fields["plan"] = _plan(pool)
    return td_v4.V4EpochInputs(**fields)


def _run(monkeypatch, provider=None, *, head_models=FIXTURE_IDS, pool=None, head_bytes=None,
         **over):
    provider = provider or Provider()
    pool = pool or _pool()
    monkeypatch.setattr(harness, "_call_model", provider.call)
    over.setdefault("fetch_listing", provider.fetch)
    hidden = np.random.RandomState(0).randn(len(pool), 1024).astype(np.float32)
    proxy = _Proxy(port=0)
    proof = harness.run_benchmark(
        nonce=NONCE, head_bytes=head_bytes or _head(head_models), benchmark_pool=pool, proxy=proxy,
        hidden_states=hidden, slice_size=SLICE, epoch_id="e00000102", source_hash="img",
        hotkey=HOTKEY, protocol_version=4, v4=_inputs(pool, **over))
    return proof, provider, proxy


# --- the menu --------------------------------------------------------------------------


def test_a_v4_epoch_runs_on_its_menu_and_names_it(monkeypatch):
    proof, provider, _ = _run(monkeypatch)
    carried = ms.carry(mf.state())
    assert proof.menu_hash == ms.menu_hash(carried)
    assert proof.routing_protocol == identity_v4(proof.menu_hash)
    assert {r.routed_model for r in proof.results} <= set(carried.ids())
    assert len(proof.scored_results) == SLICE and len(proof.exploration_results) == EXPLORE
    # The stand-in challenge is the slice's size; a proof on the wire says 500 or 1,000.
    proof.challenge = Challenge(500, None)
    back = BenchmarkProof.from_dict(json.loads(json.dumps(proof.to_dict())))
    assert back.content_hash() == proof.content_hash()


def test_each_wire_request_is_the_entrys_named_fields_and_nothing_else(monkeypatch):
    _proof, provider, _ = _run(monkeypatch)
    state = ms.carry(mf.state())
    for entry_id, request, _qid in provider.calls:
        assert set(request) <= td_v4.WIRE_KEYS
        assert set(request["provider"]) == {"order", "allow_fallbacks"}
        assert request["provider"]["allow_fallbacks"] is False
        assert request == ms.request_for(state, entry_id)


def test_a_committed_menu_that_passes_is_the_one_run(monkeypatch):
    prev = mf.state()
    # An entry retires (a structural change), leaving its record.
    gone = prev.entries[0]
    new = dataclasses.replace(
        ms.carry(prev), serial=prev.serial + 1, entries=prev.entries[1:],
        retired=(ms.Retired(gone.id, 102, ms.REASON_GONE, 102 + ms.PROPOSAL_COOLDOWN_EPOCHS),))
    raw = ms.canonical_json(new)
    proof, _, _ = _run(monkeypatch, committed_hash=hashlib.sha256(raw).hexdigest(), published=raw)
    assert proof.menu_hash == ms.menu_hash(new) != ms.menu_hash(ms.carry(prev))


def test_a_committed_menu_the_reader_refuses_is_carried_as_on_every_node(monkeypatch):
    prev = mf.state()
    bad = dataclasses.replace(ms.carry(prev), serial=prev.serial + 5)    # out of sequence
    raw = ms.canonical_json(bad)
    proof, _, _ = _run(monkeypatch, committed_hash=hashlib.sha256(raw).hexdigest(), published=raw)
    assert proof.menu_hash == ms.menu_hash(ms.carry(prev))


def test_a_committed_price_move_outside_the_refresh_cadence_is_carried(monkeypatch):
    """Epoch 102 is no refresh epoch: a menu that moves only prices is
    refused on every node and the TD runs the carried menu."""
    prev = mf.state()
    assert not ms.refresh_due(102)
    moved = dataclasses.replace(
        ms.carry(prev), serial=prev.serial + 1,
        entries=tuple(dataclasses.replace(e, price_in=e.price_in + 1) for e in prev.entries))
    raw = ms.canonical_json(moved)
    proof, _, _ = _run(monkeypatch, committed_hash=hashlib.sha256(raw).hexdigest(), published=raw)
    assert proof.menu_hash == ms.menu_hash(ms.carry(prev))


def _no_proof(monkeypatch, exc, **over):
    provider = Provider()
    with pytest.raises(exc):
        _run(monkeypatch, provider, **over)
    assert provider.calls == [] and provider.fetched == []


def test_without_the_bytes_of_the_committed_menu_there_is_no_proof(monkeypatch):
    """The second half of section 4.3's rule: MenuUnavailable inside the TD
    means no call is made and no proof is produced for the epoch."""
    _no_proof(monkeypatch, ms.MenuUnavailable, committed_hash="ee" * 32, published=None)
    _no_proof(monkeypatch, ms.MenuUnavailable, committed_hash="ee" * 32, published=b"{}")


def test_without_the_list_the_menu_names_there_is_no_proof(monkeypatch):
    _no_proof(monkeypatch, ms.MenuUnavailable, allowlist=None)
    _no_proof(monkeypatch, ms.MenuUnavailable, allowlist=mf.allowlist(serial=7))


def test_a_menu_the_compiled_rules_refuse_inside_the_td_gives_no_proof(monkeypatch):
    """Section 14, A8: a forbidden entry is refused inside the TD. Here the
    menu the TD holds (handed in by its operator) carries an entry whose lab
    the signed list does not name."""
    forbidden = mf.state(mf.entry("lab-z/rogue", "high"), mf.entry("lab-a/model-one", "high"))
    _no_proof(monkeypatch, td_v4.MenuRejected, previous=forbidden)


def test_a_menu_naming_no_list_runs_only_if_it_is_the_pinned_seed(monkeypatch):
    unlisted = dataclasses.replace(mf.state(), labs=ms.AllowlistRef(0, ""))
    _no_proof(monkeypatch, td_v4.MenuRejected, previous=unlisted)


def test_the_first_v4_epoch_runs_on_the_pinned_seed(monkeypatch):
    seed = ms.seed_state(102)
    ids = list(seed.ids())[:3]
    proof, provider, _ = _run(monkeypatch, Provider(pages=PAGES), head_models=ids,
                              previous=None, allowlist=None)
    assert proof.menu_hash == ms.menu_hash(seed)
    assert sorted(provider.fetched) == sorted({e.model for e in seed.entries})
    assert [r.model for r in proof.endpoint_reads] == sorted({e.model for e in seed.entries})
    for entry_id, request, _ in provider.calls:
        assert request == ms.request_for(seed, entry_id)


def test_exploration_on_an_entry_off_the_menu_gives_no_proof(monkeypatch):
    pool = _pool()
    plan = dataclasses.replace(_plan(pool), exploration=lambda *a: {"q1": "lab-z/rogue@high"})
    provider = Provider()
    with pytest.raises(td_v4.MenuRejected):
        _run(monkeypatch, provider, pool=pool, plan=plan)
    assert provider.calls == []


@pytest.mark.parametrize("raw", [
    b"",
    b"\x00" * 64,
    b"[]",
    b'{"epoch": 102}',
    b"{" * 50000,
    json.dumps(ms.to_dict(ms.carry(mf.state())), indent=1).encode(),           # not canonical
    ms.canonical_json(ms.carry(mf.state())).replace(b'"serial":3', b'"serial":4.0'),
    ms.canonical_json(dataclasses.replace(ms.carry(mf.state()), serial=4, epoch=103)),
    ms.canonical_json(dataclasses.replace(ms.carry(mf.state()), serial=4,
                                          labs=ms.AllowlistRef(0, ""))),
    ms.canonical_json(dataclasses.replace(
        ms.carry(mf.state()), serial=4,
        entries=mf.state(mf.entry("lab-z/rogue", "high")).entries)),
])
def test_hostile_menu_bytes_never_reach_a_call(raw):
    """Whatever bytes carry the committed hash, the TD runs the carried menu
    (the reader refused them) and never a menu the rules forbid."""
    prev = mf.state()
    state = td_v4.epoch_menu(102, previous=prev, committed_hash=hashlib.sha256(raw).hexdigest(),
                             published=raw, allowlist=mf.ALLOW)
    assert state == ms.carry(prev)


# --- endpoint reads, billed cost, serving host -------------------------------------------


def test_the_proof_carries_a_read_of_every_menu_model_and_billed_cost_per_call(monkeypatch):
    proof, provider, _ = _run(monkeypatch, Provider(fail_ids={"q1", "q2", "q3", "q4", "q5"}))
    assert sorted(provider.fetched) == ["lab-a/model-one", "lab-b/model-two"]
    reads = {r.model: r for r in proof.endpoint_reads}
    assert [h.host for h in reads["lab-a/model-one"].hosts] == ["host-a/fp8", "host-b/fp8"]
    assert [h.host for h in reads["lab-b/model-two"].hosts] == ["host-c"]
    assert reads["lab-b/model-two"].hosts[0].price_out == 400_000
    for r in proof.results:
        if r.failed:
            assert (r.provider_cost, r.served_by) == (None, "")
        else:
            assert (r.provider_cost, r.served_by) == (1234, "Fixture Host")


def test_a_failed_read_is_no_record_and_the_epoch_goes_on(monkeypatch):
    def fetch(model):
        raise TimeoutError("listing unreachable")

    proof, provider, _ = _run(monkeypatch, fetch_listing=fetch)
    assert proof.endpoint_reads == ()
    assert len(proof.results) == SLICE + EXPLORE


def test_the_calls_are_priced_exactly_at_the_menus_prices(monkeypatch):
    proof, _, proxy = _run(monkeypatch)
    state = ms.carry(mf.state())
    for r in proof.results:
        e = state.entry(r.routed_model)
        want = Fraction(e.price_in * 500 + e.price_out * 300, ms.PICO)
        assert r.cost_usd == float(want)
    assert all(isinstance(rec.cost_usd, Fraction) for rec in proxy.records)


# --- the head's picks (section 11; W7) ------------------------------------------------------


def _head_without_cost_fields(models, seed=0) -> bytes:
    import io

    with np.load(io.BytesIO(_head(models, seed)), allow_pickle=False) as z:
        kept = {k: z[k] for k in z.files
                if k not in ("mean_in_tokens", "mean_out_tokens", "cost_profile_id")}
    buf = io.BytesIO()
    np.savez(buf, **kept)
    return buf.getvalue()


def _picks_of(proof, head_bytes, pool):
    """What `head_odds.picks_and_odds_v4` picks for the proof's scored questions."""
    from fugal_subnet.head_eval import load_head_v4
    from fugal_subnet.head_odds import picks_and_odds_v4

    hidden = np.random.RandomState(0).randn(len(pool), 1024).astype(np.float32)
    index = {q["question_id"]: i for i, q in enumerate(pool)}
    qids = [r.question_id for r in proof.scored_results]
    got = picks_and_odds_v4(load_head_v4(head_bytes), ms.carry(mf.state()),
                            hidden[[index[q] for q in qids]])
    return dict(zip(qids, got.picks))


def test_a_head_without_token_means_runs_a_v4_epoch_and_routes_by_the_shared_function(
        monkeypatch):
    raw = _head_without_cost_fields(FIXTURE_IDS, seed=2)
    proof, _, _ = _run(monkeypatch, head_bytes=raw)
    pool = _pool()
    assert len(proof.scored_results) == SLICE
    assert {r.question_id: r.routed_model for r in proof.scored_results} == \
        _picks_of(proof, raw, pool)
    assert proof.weights_hash == hashlib.sha256(raw).hexdigest()


def test_rows_off_the_menu_are_ignored_and_a_missing_row_is_never_routed_to(monkeypatch):
    raw = _head_without_cost_fields([FIXTURE_IDS[2], "lab-z/later@high", FIXTURE_IDS[0]])
    proof, _, _ = _run(monkeypatch, head_bytes=raw)
    routed = {r.routed_model for r in proof.scored_results}
    assert routed <= {FIXTURE_IDS[0], FIXTURE_IDS[2]}
    assert {r.question_id: r.routed_model for r in proof.scored_results} == \
        _picks_of(proof, raw, _pool())


def test_a_head_with_no_row_on_the_menu_makes_no_call(monkeypatch):
    provider = Provider()
    with pytest.raises(ValueError, match="no row for any entry"):
        _run(monkeypatch, provider, head_bytes=_head_without_cost_fields(["lab-z/other"]))
    assert provider.calls == [] and provider.fetched == []


# --- provisional entries (FAST_CROWN_V4.md 18) ----------------------------------------------


def _joined(prev=None, epoch=102, **listing_over):
    """A delta for `epoch` admitting lab-c/model-new@high on the carried menu,
    as (join pointer, delta bytes)."""
    from fugal_subnet import chain_proposals as cp

    base = ms.carry(prev or mf.state())
    assert base.epoch == epoch
    doc, raw = mf.document()
    read = cp.ChainRead(epoch=epoch, block=epoch * 360 - 30, block_hash="ab" * 32,
                        proposals=((mf.HOTKEY, cp.proposal_hash(raw)),),
                        coldkey_of={mf.HOTKEY: mf.COLDKEY}, serving=frozenset({mf.HOTKEY}),
                        reveal=None)
    delta = cp.admit(base, read, {cp.proposal_hash(raw): raw},
                     {cp.proposal_hash(raw): mf.listing(**listing_over)}, mf.ALLOW, ms.READER_PARAMS)
    assert delta.joined()
    data = cp.delta_bytes(delta)
    return cp.JoinPointer(epoch, hashlib.sha256(data).hexdigest()), data


def test_a_join_delta_puts_the_entry_on_the_epochs_menu_provisional(monkeypatch):
    pointer, data = _joined()
    proof, provider, _ = _run(monkeypatch, join_pointer=pointer, join_published=data)
    state = ms.carry(mf.state())
    assert proof.menu_hash != ms.menu_hash(state)
    joined = ms.apply_joins(state, __import__("fugal_subnet.chain_proposals",
                                              fromlist=["x"]).check_delta(state, data, mf.ALLOW))
    assert proof.menu_hash == ms.menu_hash(joined)
    assert "lab-c/model-new" in provider.fetched          # its hosts are read in the TD


def test_a_join_delta_without_its_bytes_gives_no_proof(monkeypatch):
    pointer, _data = _joined()
    _no_proof(monkeypatch, ms.MenuUnavailable, join_pointer=pointer, join_published=None)


def test_a_refused_delta_adds_nothing_as_on_every_node(monkeypatch):
    pointer, data = _joined()
    doc = json.loads(data)
    doc["serving"] = 0                                     # the verdicts no longer follow
    bad = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    from fugal_subnet import chain_proposals as cp

    proof, _, _ = _run(monkeypatch, join_pointer=cp.JoinPointer(102, hashlib.sha256(
        bad).hexdigest()), join_published=bad)
    assert proof.menu_hash == ms.menu_hash(ms.carry(mf.state()))


def test_a_provisional_call_is_retried_once_and_runs_on_its_own_deadline(monkeypatch):
    pointer, data = _joined()
    seen = []

    class Flaky(Provider):
        def call(self, proxy, model_id, question, request=None, reasoning=None, deadline_s=None):
            seen.append((model_id, deadline_s))
            if model_id == "lab-c/model-new@high" and len(
                    [s for s in seen if s[0] == model_id]) % 2:
                return None                                # every first attempt fails
            return super().call(proxy, model_id, question, request=request)

    proof, _, _ = _run(monkeypatch, Flaky(), join_pointer=pointer, join_published=data)
    mine = [r for r in proof.exploration_results if r.routed_model == "lab-c/model-new@high"]
    assert mine and not any(r.failed for r in mine)        # each retry answered
    prov = [d for m, d in seen if m == "lab-c/model-new@high"]
    assert len(prov) == 2 * len(mine) and set(prov) == {harness.PROVISIONAL_CALL_DEADLINE_S}
    assert {d for m, d in seen if m != "lab-c/model-new@high"} == {None}


def test_a_provisional_entry_priced_far_below_the_tds_own_read_is_not_called(monkeypatch):
    pointer, data = _joined(price_in=1, price_out=1)        # the root's read: almost free
    provider = Provider()
    proof, _, _ = _run(monkeypatch, provider, join_pointer=pointer, join_published=data)
    mine = [r for r in proof.exploration_results if r.routed_model == "lab-c/model-new@high"]
    assert mine and all(r.failed for r in mine)
    assert not any(m == "lab-c/model-new@high" for m, _r, _q in provider.calls)


def test_no_provisional_call_starts_past_the_tds_budget(monkeypatch):
    pointer, data = _joined()
    monkeypatch.setattr(harness, "PROVISIONAL_BUDGET_S", -1.0)     # already spent
    proof, provider, _ = _run(monkeypatch, join_pointer=pointer, join_published=data)
    mine = [r for r in proof.exploration_results if r.routed_model == "lab-c/model-new@high"]
    assert mine and all(r.failed for r in mine)
    assert not any(m == "lab-c/model-new@high" for m, _r, _q in provider.calls)


def test_a_v4_proof_carries_no_proposal_and_no_measurement(monkeypatch):
    proof, _, _ = _run(monkeypatch)
    doc = proof.to_dict()
    assert "proposal" not in doc and not any("measure_for" in r for r in doc["results"])
    doc["proposal"] = {"id": "x"}
    with pytest.raises(ValueError):
        BenchmarkProof.from_dict(doc)


# --- isolation -----------------------------------------------------------------------------


def test_the_v1_to_v3_path_does_not_import_the_v4_menu_modules():
    import subprocess
    import sys
    code = ("import sys; import fugal_subnet.tee.harness, fugal_subnet.tee.proof, "
            "fugal_subnet.tee.runtime, fugal_subnet.tee.verify; "
            "bad = [m for m in ('fugal_subnet.menu_state', 'fugal_subnet.menu_rules', "
            "'fugal_subnet.tee.td_v4') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr


# --- end to end through the real proxy, against a loopback upstream ----------------------


def test_through_the_real_proxy_only_the_named_fields_reach_the_provider(monkeypatch):
    """The production path: harness -> MeteringProxy -> upstream, with the
    listing read through the same base URL. The upstream is a loopback
    server; it sees exactly the entry's request plus `messages`, and its
    `usage.cost` (written as a JSON exponent) arrives exactly."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from fugal_subnet.tee import runtime as rt

    pool = _pool()
    gold = {q["question_id"]: q["gold"] for q in pool}
    posts, gets = [], []

    class H(BaseHTTPRequestHandler):
        def _send(self, body: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            gets.append((self.path, self.headers.get("Authorization")))
            model = self.path.removeprefix("/api/v1/models/").removesuffix("/endpoints")
            self._send(_listing(model))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posts.append(body)
            text = gold.get(body["messages"][-1]["content"].split()[-1].rstrip("?"), "0")
            self._send(('{"choices":[{"message":{"content":%s}}],"provider":"Loop",'
                        '"usage":{"prompt_tokens":5,"completion_tokens":2,"cost":1e-7}}'
                        % json.dumps(text)).encode())

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(rt, "_OPENROUTER_BASE", f"http://127.0.0.1:{srv.server_port}/api/v1")
    proxy = MeteringProxy(port=0, api_key="test-key")
    proxy.start()
    try:
        hidden = np.random.RandomState(0).randn(len(pool), 1024).astype(np.float32)
        proof = harness.run_benchmark(
            nonce=NONCE, head_bytes=_head(FIXTURE_IDS), benchmark_pool=pool, proxy=proxy,
            hidden_states=hidden, slice_size=SLICE, epoch_id="e00000102", source_hash="img",
            hotkey=HOTKEY, protocol_version=4,
            v4=_inputs(pool, fetch_listing=rt.fetch_endpoints_page))
    finally:
        proxy.stop()
        srv.shutdown()
    assert sorted(p for p, _auth in gets) == ["/api/v1/models/lab-a/model-one/endpoints",
                                              "/api/v1/models/lab-b/model-two/endpoints"]
    assert all(auth is None for _p, auth in gets)
    state = ms.carry(mf.state())
    by_model = {}
    for e in state.entries:
        by_model.setdefault(json.dumps(ms.request_for(state, e.id), sort_keys=True), e.id)
    assert len(posts) == SLICE + EXPLORE
    for body in posts:
        sent = {k: v for k, v in body.items() if k != "messages"}
        assert json.dumps(sent, sort_keys=True) in by_model
    assert all((r.provider_cost, r.served_by) == (100, "Loop") for r in proof.results)
    assert {r.model for r in proof.endpoint_reads} == {"lab-a/model-one", "lab-b/model-two"}


# --- W5's exploration draw, fed by the menu's own counts ------------------------------


def _v3_benchmarks(pool):
    """The fixture pool under benchmark names the v4 mix knows (the v3
    pool's sets, which v4's pool keeps): graded by grader_id as before."""
    names = {"gsm8k": "gsm8k_train", "math": "math_l35", "mmlu": "mmlu", "aime": "aime"}
    return [dict(q, benchmark=names[q["benchmark"]]) for q in pool]


def _counted(state, counts):
    return dataclasses.replace(state, entries=tuple(
        dataclasses.replace(e, explore_pairs=counts.get(e.id, 0)) for e in state.entries))


def test_the_v4_exploration_is_w5s_draw_on_the_menus_own_counts():
    from fugal_subnet import config

    pool = _v3_benchmarks(_pool(400))
    state = _counted(ms.carry(mf.state()), {"lab-a/model-one@high": 140, "lab-b/model-two": 30})
    nonce, scored = bytes.fromhex(NONCE), {q["question_id"] for q in pool[:SLICE]}
    got = td_v4.exploration_v4(nonce, pool, scored, state, HOTKEY)
    shares = epoch_plan.explore_shares_v4(state, state.explore_pairs())
    shared, cost, cover = epoch_plan.exploration_plan_v4(nonce, pool, scored, shares, HOTKEY)
    assert got == {**shared, **cost, **cover}
    assert len(got) == (config.EXPLORE_SHARED_SIZE_V4 + config.EXPLORE_PRIVATE_SIZE_V4) == 112
    assert not set(got) & scored and set(got.values()) <= set(state.ids())
    # Another published count is another coverage draw, and the same counts
    # handed in by any other route are not read.
    other = _counted(state, {"lab-a/model-one@low": 140})
    assert epoch_plan.explore_shares_v4(other) != shares
    assert td_v4.exploration_v4(nonce, pool, scored, state, HOTKEY) == got


def test_a_v4_proof_carries_w5s_exploration(monkeypatch):
    pool = _v3_benchmarks(_pool())
    from fugal_subnet import config

    plan = td_v4.V4Plan(scored_slice=lambda nonce, p, size: select_slice(nonce, p, size),
                        exploration=td_v4.exploration_v4,
                        pool_index=epoch_plan.pool_index_v4(pool),
                        targets=config.CATEGORY_TARGETS_V3)
    proof, _, _ = _run(monkeypatch, pool=pool, plan=plan)
    state = ms.carry(mf.state())
    scored = {r.question_id for r in proof.scored_results}
    want = td_v4.exploration_v4(bytes.fromhex(NONCE), pool, scored, state, HOTKEY)
    assert {r.question_id: r.routed_model for r in proof.exploration_results} == want
