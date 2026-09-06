"""The provisioning channel's allow-list is pinned, structurally.

`ALLOWED_FIELDS` is the security boundary between "per-miner data the enclave
may receive" and "consensus inputs a miner must not be able to choose". A field
added to it silently is the upstream-substitution exploit arriving through the
front door, so this test reads the literal out of the AST: adding one is a
failing test and a deliberate review, not a diff nobody reads.

Same treatment as tests/test_grader_policy.py gives the harness's exec policy,
and for the same reason — the value matters more than the code around it.
"""
import ast
from pathlib import Path

import pytest

from fugal_subnet.tee.provision import ALLOWED_FIELDS, ProvisionError, validate_payload

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "fugal_subnet" / "tee" / "provision.py"

# Exactly what may cross into a TD. Neither entry is an input the grader, the
# pool, the slice or the cost model reads:
#   head_b64            per-miner, changes every epoch, bound by the on-chain
#                       weights_hash committed before the nonce
#   hotkey_ss58         public, on chain; binds the proof to a miner so relay
#                       fails cryptographically rather than statistically
#   openrouter_api_key  a secret, and only the miner's own money
EXPECTED = {"head_b64", "hotkey_ss58", "openrouter_api_key"}

# Named individually so a failure says WHICH consensus input leaked in, rather
# than that a set comparison failed. Every one of these is covered by
# runtime_identity() precisely so a miner cannot choose it.
FORBIDDEN = [
    "upstream", "openrouter_base", "FUGAL_OPENROUTER_BASE",
    "pool", "pool_hash", "benchmark_pool", "FUGAL_BENCHMARK_POOL",
    "grader", "grader_hash", "source_hash",
    "slice_size", "nonce", "questions_hash",
]


def _literal_from_ast() -> set:
    """Read ALLOWED_FIELDS out of the source, not the imported module.

    Importing would report whatever the module computed; parsing reports what
    is written down. Those differ exactly when something assembles the set at
    runtime, which is the case this test exists to forbid.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", "") == "ALLOWED_FIELDS" for t in node.targets):
            continue
        call = node.value
        assert isinstance(call, ast.Call), "ALLOWED_FIELDS must be a frozenset literal"
        assert getattr(call.func, "id", "") == "frozenset", "must be frozenset(...)"
        return {ast.literal_eval(e) for e in call.args[0].elts}
    raise AssertionError("ALLOWED_FIELDS not found in provision.py")


def test_allow_list_is_a_literal_and_has_not_changed():
    assert _literal_from_ast() == EXPECTED, (
        "The provisioning allow-list changed. This is a consensus boundary: a "
        "field here can be chosen by the miner and must not be one the grader, "
        "pool, slice or cost model depends on. Review, then update EXPECTED."
    )
    assert set(ALLOWED_FIELDS) == EXPECTED, "runtime value disagrees with the source"


@pytest.mark.parametrize("field", FORBIDDEN)
def test_consensus_inputs_are_not_accepted(field):
    """A miner-chosen upstream is a total break, not a misconfiguration.

    With FUGAL_OPENROUTER_BASE under miner control the pool's gold answers are
    already inside the TD, so a miner returns them with two tokens of usage and
    collects perfect accuracy at near-zero attested cost — every hash binding,
    the DCAP signature and the approved measurement all still passing.
    """
    assert field not in ALLOWED_FIELDS
    with pytest.raises(ProvisionError, match="unknown provisioning field"):
        validate_payload({field: "x"})


def test_unknown_field_is_refused_rather_than_ignored():
    with pytest.raises(ProvisionError, match="unknown provisioning field"):
        validate_payload({"openrouter_api_key": "k", "extra": "v"})


def test_rejection_never_echoes_the_value():
    """The likeliest unknown field is a secret someone added by mistake."""
    with pytest.raises(ProvisionError) as e:
        validate_payload({"some_token": "sk-super-secret-value"})
    assert "sk-super-secret-value" not in str(e.value)


def test_allowed_payload_passes_through_unchanged():
    payload = {"head_b64": "AAAA", "hotkey_ss58": "5Fk...", "openrouter_api_key": "sk-or-v1-x"}
    assert validate_payload(payload) == payload


def test_non_string_and_non_dict_are_refused():
    with pytest.raises(ProvisionError, match="must be a string"):
        validate_payload({"openrouter_api_key": 1234})
    with pytest.raises(ProvisionError, match="must be an object"):
        validate_payload(["openrouter_api_key"])
