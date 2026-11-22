"""Protocol v4 proof fields (docs/decisions/LIVE_MENU_V4.md, section 12).

Every v4 field is hashed only when present, the pattern `failed`,
`latency_ms` and `tpm_intermediates` set: a v1, v2 or v3 proof carries none
of them, so its JSON and its `content_hash` are the bytes they were before
the fields existed. The first test pins both, for proofs of every earlier
protocol, at the values the code produced before v4's fields were added.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from fugal_subnet import config
from fugal_subnet.protocol import PROOF_JSON_MAX_LEN
from fugal_subnet.routing_protocol import identity_for
from fugal_subnet.tee import endpoint_reads as er
from fugal_subnet.tee.proof import (
    MAX_PROVIDER_COST_NANO,
    MAX_SERVED_BY_LEN,
    BenchmarkProof,
    QuestionResult,
)

EMPTY = hashlib.sha256(b"").hexdigest()


def _legacy_proofs() -> dict[str, BenchmarkProof]:
    """Proofs as v1, v2 and v3 harnesses write them: answered, failed, with
    and without latency, an int zero cost, TPM intermediates."""
    def results(latency: bool):
        lat = {"latency_ms": 1234} if latency else {}
        return [
            QuestionResult("q-scored", "openai/gpt-6.1-sol@high", True, 0.0123, "ab" * 32,
                           812, 4096, **lat),
            QuestionResult("q-failed", "anthropic/claude-fable-5.2@max", False, 0, EMPTY, 0, 0,
                           failed=True, **({"latency_ms": 0} if latency else {})),
            QuestionResult("q-explore", "deepseek/deepseek-v4.1-flash@high", False, 0.0004,
                           "cd" * 32, 120, 950, is_exploration=True, **lat),
        ]

    def proof(version: int, *, latency: bool, tpm: list[bytes] | None = None) -> BenchmarkProof:
        return BenchmarkProof(
            hotkey="5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX",
            routing_protocol=identity_for(version),
            epoch_id="e00026500", nonce="0f" * 32, questions_hash="1e" * 32,
            weights_hash="2d" * 32, source_hash="3c" * 32, results=results(latency),
            total_cost_usd=0.0127, per_model_costs={"openai/gpt-6.1-sol@high": 0.0123,
                                                    "deepseek/deepseek-v4.1-flash@high": 0.0004},
            attestation_quote=b"\x04\x00" + b"\x00" * 30, timestamp=1795000000.25,
            tpm_intermediates=list(tpm or []))

    return {
        "v1": proof(1, latency=False),
        "v2": proof(2, latency=True),
        "v3": proof(3, latency=True),
        "v3-tpm": proof(3, latency=True, tpm=[b"\x30\x82\x01\x00" + b"\x11" * 64]),
    }


# (content_hash, sha256 of the serialised to_dict JSON) of each proof above,
# computed by the code before protocol v4's fields existed (eba95a6).
PINNED = {
    "v1": ("acfa11e576170da168e650eabcdd717c56491c2d77e7a082dab95ffa67097e65",
           "b8030a3c16bbf71e4f52e4cdadcebdb9b86c647183cabf6f8f32a7579d68430c"),
    "v2": ("f968d990fb8e42285870e424b31cf174ff44a1b9f5b4b6913354d8727c341a27",
           "fa47719d8596d6003356627d71eef07d8492bc2a4ee65be183d86ba41379348e"),
    "v3": ("e15584cbd4b8d93e6c5929751494b387002dd83d2da8d63a36829e0a8aaaed85",
           "a545d0271a1bc1b9a3b024640547d5a39122038e13c71ce58f2ccf286f8953b0"),
    "v3-tpm": ("20489244425dd2ab2b08c226a39f83d91261f1c8095e5190f5dda59617316908",
               "92627b997447dd88257dc67b7002c665072aca2174f4feccef6e3dada2746e8b"),
}


def _wire(p: BenchmarkProof) -> str:
    return json.dumps(p.to_dict(), separators=(",", ":"))


@pytest.mark.parametrize("name", sorted(PINNED))
def test_an_earlier_protocol_proof_has_the_bytes_and_hash_it_had_before_v4(name):
    p = _legacy_proofs()[name]
    assert (p.content_hash(), hashlib.sha256(_wire(p).encode()).hexdigest()) == PINNED[name]
    back = BenchmarkProof.from_dict(json.loads(_wire(p)))
    assert back.content_hash() == p.content_hash()
    assert _wire(back) == _wire(p)


@pytest.mark.parametrize("name", sorted(PINNED))
@pytest.mark.parametrize("stuffed", [
    {"menu_hash": "cd" * 32},
    {"menu_hash": "not a hash"},
    {"endpoint_reads": [{"model": "x", "hosts": "junk"}]},
    {"proposal": {"id": 1}},
    {"result": {"provider_cost": 5, "served_by": "x", "measure_for": "ab" * 32}},
    {"result": {"provider_cost": -1, "measure_for": "nope"}},
])
def test_v4_keys_on_an_earlier_protocol_proof_are_not_read(name, stuffed):
    """A proof in a v1-v3 namespace is read exactly as before v4's fields
    existed: its v4 keys are neither hashed nor refused, whatever they hold,
    so validators on either side of this change reach the same verdict."""
    p = _legacy_proofs()[name]
    d = json.loads(_wire(p))
    for k, v in stuffed.items():
        if k == "result":
            for r in d["results"]:
                r.update(v)
        else:
            d[k] = v
    back = BenchmarkProof.from_dict(d)
    assert back.content_hash() == PINNED[name][0]
    assert _wire(back) == _wire(p)
    assert back.exploration_results == p.exploration_results


# --- the v4 fields -------------------------------------------------------------

PID = "ab" * 32
MENU_HASH = "cd" * 32


def _v4_result(**over) -> QuestionResult:
    fields = dict(question_id="q1", routed_model="lab-a/model-one@high", correct=True,
                  cost_usd=0.0021, response_hash="ef" * 32, prompt_tokens=300,
                  completion_tokens=900, latency_ms=4200, provider_cost=2_104_000,
                  served_by="DeepInfra")
    fields.update(over)
    return QuestionResult(**fields)


def _reads() -> tuple[er.EndpointRead, ...]:
    return (
        er.EndpointRead("lab-a/model-one", (
            er.HostListing("host-a/fp8", True, 100_000, 400_000, 131072, "fp8", 0),
            er.HostListing("host-b/fp8", False))),
        er.EndpointRead("lab-c/model-new", (
            er.HostListing("host-a/fp8", True, 300_000, 900_000, 200_000, "fp8", -2),)),
    )


def _v4_proof(**over) -> BenchmarkProof:
    fields = dict(
        hotkey="5Fminer", routing_protocol="ff" * 32, epoch_id="e00000102", nonce="0f" * 32,
        questions_hash="1e" * 32, weights_hash="2d" * 32, source_hash="3c" * 32,
        results=[_v4_result(),
                 _v4_result(question_id="q2", is_exploration=True, provider_cost=None,
                            served_by="")],
        total_cost_usd=0.0063, per_model_costs={"lab-a/model-one@high": 0.0063},
        attestation_quote=b"", timestamp=1795000000.5, menu_hash=MENU_HASH,
        endpoint_reads=_reads())
    fields.update(over)
    return BenchmarkProof(**fields)


def test_a_v4_proof_round_trips_to_the_same_bytes_and_hash():
    p = _v4_proof()
    back = BenchmarkProof.from_dict(json.loads(_wire(p)))
    assert back == p
    assert back.content_hash() == p.content_hash()
    assert _wire(back) == _wire(p)
    assert [r.question_id for r in back.scored_results] == ["q1"]
    assert [r.question_id for r in back.exploration_results] == ["q2"]


@pytest.mark.parametrize("stuffed", [
    ("proposal", {"id": "lab-c/model-new@high"}),
    ("result", {"measure_for": PID}),
])
def test_a_v4_proof_carrying_a_proposal_or_a_measurement_is_refused(stuffed):
    """Proposals are filed on chain and measured by the field (FAST_CROWN_V4.md
    18): no v4 proof files or measures one."""
    d = json.loads(_wire(_v4_proof()))
    key, value = stuffed
    if key == "result":
        d["results"][1].update(value)
    else:
        d[key] = value
    with pytest.raises(ValueError):
        BenchmarkProof.from_dict(d)


@pytest.mark.parametrize("change", [
    {"menu_hash": "ce" * 32},
    {"endpoint_reads": _reads()[:1]},
    {"endpoint_reads": ()},
])
def test_every_v4_proof_field_is_bound_by_the_content_hash(change):
    assert _v4_proof(**change).content_hash() != _v4_proof().content_hash()


@pytest.mark.parametrize("change", [
    {"provider_cost": 2_104_001}, {"provider_cost": None}, {"served_by": "Fireworks"},
    {"served_by": None},
])
def test_every_v4_result_field_is_bound_by_the_content_hash(change):
    base = _v4_proof()
    changed = _v4_proof(results=[_v4_result(**change)] + base.results[1:])
    assert changed.content_hash() != base.content_hash()


def test_an_empty_reads_list_is_present_and_differs_from_no_field():
    assert _v4_proof(endpoint_reads=()).to_dict()["endpoint_reads"] == []
    assert "endpoint_reads" not in _v4_proof(endpoint_reads=None).to_dict()


_DELETE = object()


def _mutated(path, value):
    d = json.loads(_wire(_v4_proof()))
    target = d
    for k in path[:-1]:
        target = target[k]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return d


@pytest.mark.parametrize("path, value", [
    (("menu_hash",), "CD" * 32),
    (("menu_hash",), "cd" * 31),
    (("menu_hash",), 7),
    (("results", 0, "provider_cost"), -1),
    (("results", 0, "provider_cost"), MAX_PROVIDER_COST_NANO + 1),
    (("results", 0, "provider_cost"), 2.5),
    (("results", 0, "provider_cost"), True),
    (("results", 0, "provider_cost"), "2104000"),
    (("results", 0, "served_by"), "x" * (MAX_SERVED_BY_LEN + 1)),
    (("results", 0, "served_by"), "Deep\nInfra"),
    (("results", 0, "served_by"), 5),
    (("endpoint_reads",), {"lab-a/model-one": []}),
    (("endpoint_reads",), [{"model": "lab-a/model-one", "hosts": []}]),
    (("endpoint_reads", 0, "extra"), 1),
    (("endpoint_reads", 0, "model"), "LAB/x"),
    (("endpoint_reads", 0, "hosts", 0, "price_in"), -1),
    (("endpoint_reads", 0, "hosts", 0, "price_in"), er.MAX_PRICE_PICO + 1),
    (("endpoint_reads", 0, "hosts", 0, "price_in"), "100000"),
    (("endpoint_reads", 0, "hosts", 0, "price_in"), 1e5),
    (("endpoint_reads", 0, "hosts", 0, "limit"), -5),
    (("endpoint_reads", 0, "hosts", 0, "status"), 10**6),
    (("endpoint_reads", 0, "hosts", 0, "listed"), 1),
    (("endpoint_reads", 0, "hosts", 0, "host"), "host-a/fp8 "),
    (("endpoint_reads", 0, "hosts", 1, "price_in"), 5),
    (("endpoint_reads", 0, "hosts", 1, "host"), "host-a/fp8"),
    (("endpoint_reads", 0, "hosts", 0, "quantization"), "FP8"),
])
def test_a_malformed_v4_field_is_refused_never_reinterpreted(path, value):
    with pytest.raises((ValueError, TypeError, KeyError)):
        BenchmarkProof.from_dict(_mutated(path, value))


def test_the_reads_must_be_in_their_one_order():
    d = json.loads(_wire(_v4_proof()))
    d["endpoint_reads"].reverse()
    with pytest.raises(ValueError):
        BenchmarkProof.from_dict(d)


def test_too_many_reads_or_hosts_are_refused():
    host = {"host": "host-a/fp8", "listed": False, "price_in": 0, "price_out": 0, "limit": 0,
            "quantization": "", "status": 0}
    d = json.loads(_wire(_v4_proof()))
    d["endpoint_reads"] = [{"model": f"lab-a/m{i:03d}", "hosts": [host]}
                           for i in range(er.MAX_READ_MODELS + 1)]
    with pytest.raises(ValueError):
        BenchmarkProof.from_dict(d)
    d["endpoint_reads"] = [{"model": "lab-a/m", "hosts": [dict(host, host=f"h{i:03d}")
                                                         for i in range(er.MAX_READ_HOSTS + 1)]}]
    with pytest.raises(ValueError):
        BenchmarkProof.from_dict(d)


def test_the_largest_v4_proof_fits_the_wire_cap():
    """Section 12: PROOF_JSON_MAX_LEN against what v4 adds, at the sizes of
    SEAT_MINIMAL_V4.md section 8: 112 exploration results (68 cost-aware, 44
    coverage, no shared set), each provisional call possibly retried once
    (a retry is the same result, not a second one). Every v4 field at its
    bound: 200 scored results, the most reads a proof may carry, every
    string at its longest."""
    long_q = "q" * 120

    def result(i, **kw):
        return _v4_result(question_id=f"{long_q}{i:05d}", routed_model="l" * 200,
                          served_by="s" * MAX_SERVED_BY_LEN,
                          provider_cost=MAX_PROVIDER_COST_NANO, **kw)

    results = ([result(i) for i in range(config.SLICE_SIZE_V4)]
               + [result(1000 + i, is_exploration=True) for i in range(112)])
    assert len(results) == 200 + config.EXPLORE_SHARED_SIZE_V4 + config.EXPLORE_PRIVATE_SIZE_V4
    reads = tuple(
        er.EndpointRead(f"lab/{'m' * 190}{i:03d}", tuple(
            er.HostListing(f"{j:02d}" + "h" * (er.MAX_HOST_LEN - 2), True, er.MAX_PRICE_PICO,
                           er.MAX_PRICE_PICO, er.MAX_ANSWER_LIMIT, "q" * 16, -er.MAX_ABS_STATUS)
            for j in range(er.MAX_READ_HOSTS)))
        for i in range(er.MAX_READ_MODELS))
    p = _v4_proof(results=results, endpoint_reads=reads,
                  per_model_costs={f"{'l' * 200}{i}": 1.0 for i in range(80)})
    size = len(_wire(p))
    assert size < PROOF_JSON_MAX_LEN, size
    assert BenchmarkProof.from_dict(json.loads(_wire(p))).content_hash() == p.content_hash()
