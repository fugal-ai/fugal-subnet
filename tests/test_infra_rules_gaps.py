"""Chain and infrastructure rules that a planted-bug review found untested.

Each test below pairs an honest control with one thing different, and each
was checked by planting the named change and watching the test fail:

- commitments: the commit block that gates seniority is read exactly, and a
  miner whose head changed re-commits (exercised through a fake subtensor, so
  the real query_map / decode path runs, not a monkeypatched reader);
- commit_reveal: a known-answer vector for the commit hash, which binds the
  routing-protocol identity;
- reveal_publisher: a commit-reveal period of 0 still holds a reveal for one
  epoch (publishing the current epoch's reveal publishes its weights early);
- fingerprint: every consensus environment variable and every digest field
  moves consensus_digest, and a package that disagrees with its pin is named;
- api.load_prices / routing_protocol.price_table: exact per-token prices
  from the pinned tables (the consensus cost denominator);
- vendor/success_contract: the 64-model bound, the stable tie-break by
  index, the decision quantum, negative token means and duplicate archive
  members;
- success_training: training and validation rows may not overlap.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import types
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fugal_subnet import commitments
from fugal_subnet import success_test_fixtures as fixtures
from fugal_subnet.vendor import success_contract as c

ROOT = Path(__file__).resolve().parents[1]


# --- commitments: exact commit block, re-commit on a new head ---------------


class _FakeSubtensor:
    """The two calls commitments.py makes, shaped like the Commitments pallet."""

    def __init__(self, entries):
        # entries: {hotkey: (head_hash, block)}
        self.entries = dict(entries)
        self.written = []

    def query_map(self, module, name, params=None, block=None):
        assert (module, name) == ("Commitments", "CommitmentOf")
        out = []
        for hotkey, (data, commit_block) in self.entries.items():
            raw = {
                "deposit": 0,
                "block": commit_block,
                "info": {"fields": [{f"Raw{len(data)}": "0x" + data.encode().hex()}]},
            }
            out.append((types.SimpleNamespace(value=hotkey), types.SimpleNamespace(value=raw)))
        return out

    def set_commitment(self, wallet, netuid, data):
        self.written.append(data)
        self.entries[wallet.hotkey.ss58_address] = (data, 10_000)
        return (True, "")


_WALLET = types.SimpleNamespace(hotkey=types.SimpleNamespace(ss58_address="5miner"))


def test_the_commit_block_is_read_exactly_as_the_chain_stores_it():
    head = "a" * 64
    sub = _FakeSubtensor({"5miner": (head, 4321), "5other": ("b" * 64, 1)})
    got = commitments.get_commitments_with_blocks(sub, netuid=7, block=5000)
    # The block gates commit-before-nonce and orders seniority: off by one is
    # a head committed AT the boundary read as committed before it.
    assert got["5miner"] == (head, 4321)
    assert got["5other"] == ("b" * 64, 1)


def test_ensure_commitment_through_the_chain_read_recommits_only_a_new_head():
    old, new = "a" * 64, "c" * 64
    # Control: the head on chain is this head, so nothing is written.
    same = _FakeSubtensor({"5miner": (old, 900)})
    assert commitments.ensure_commitment(same, _WALLET, 7, old) is True
    assert same.written == []
    # One thing different: the miner retrained, so the chain holds the OLD
    # hash and the new one must be committed.
    changed = _FakeSubtensor({"5miner": (old, 900)})
    assert commitments.ensure_commitment(changed, _WALLET, 7, new) is True
    assert changed.written == [new]


# --- commit_reveal: known-answer vector ------------------------------------


_KAT_QUESTIONS = [
    {"id": "q1", "question": "What is 2+2?", "gold": "4"},
    {"id": "q0", "question": "Capital of France?", "gold": "Paris"},
]
_KAT_GRADER = "ab" * 32


def test_commit_hash_known_answer_vector(monkeypatch):
    """sha256(canonical_json(questions) | grader_hash | routing identity).

    The identity is fixed here so the vector pins the payload FORMAT, not the
    current price table; a commit hash that dropped the identity would let a
    reveal from another protocol verify against this one's commitment.
    """
    from fugal_subnet import commit_reveal

    monkeypatch.setattr(commit_reveal, "identity", lambda: "fixed-routing-identity")
    assert commit_reveal.compute_commit_hash(_KAT_QUESTIONS, _KAT_GRADER) == (
        "11928dcd1c222e7a98f2c0c6c6df02fba4ffb82a5befe5575db9dde9737c5f02"
    )
    monkeypatch.setattr(commit_reveal, "identity", lambda: "another-routing-identity")
    assert commit_reveal.compute_commit_hash(_KAT_QUESTIONS, _KAT_GRADER) != (
        "11928dcd1c222e7a98f2c0c6c6df02fba4ffb82a5befe5575db9dde9737c5f02"
    )


# --- reveal_publisher: a period of 0 still holds one epoch -------------------


class _HP:
    def __init__(self, enabled, period):
        self.commit_reveal_weights_enabled = enabled
        self.commit_reveal_period = period


class _HPSub:
    def __init__(self, hp):
        self.hp = hp

    def get_subnet_hyperparameters(self, netuid):
        return self.hp


@pytest.mark.parametrize("period", [0, None])
def test_an_enabled_commit_reveal_period_of_zero_never_publishes_the_current_epoch(
    tmp_path, period,
):
    from fugal_subnet.reveal_publisher import delay_epochs_for, due_reveals

    for e in ("e00000004", "e00000005"):
        (tmp_path / e).mkdir()
        (tmp_path / e / "reveal.json").write_text(json.dumps({"epoch_id": e}), encoding="utf-8")
    # Control: period 2 holds two epochs.
    assert delay_epochs_for(_HPSub(_HP(True, 2)), 1) == 2
    # One thing different: the chain reports a period of 0 (or none).
    delay = delay_epochs_for(_HPSub(_HP(True, period)), 1)
    assert delay == 1
    due = [p.split(os.sep)[-2] for p in due_reveals(str(tmp_path), 5, delay)]
    assert due == ["e00000004"], "the current epoch's reveal (its weights) must be held"


# --- fingerprint: what moves the consensus digest ----------------------------


def _alternates(prices_override: str) -> dict[str, str]:
    """A valid, non-default value for every config.CONSENSUS_ENV_VARS name."""
    return {
        "FUGAL_SLICE_SIZE": "299",
        "FUGAL_SLICE_SHARE_FACTOR": "1.25",
        "FUGAL_COLLECT_FRACTION": "0.4",
        "FUGAL_EXPLORE_FRACTION": "0.5",
        "FUGAL_HARNESS_ALLOW_EXEC": "1",
        "FUGAL_WILSON_CONFIDENCE": "0.9",
        "FUGAL_SCORE_QUALITY_FLOOR": "0.75",
        "FUGAL_FRONTIER_MIN_TRIALS": "299",
        "FUGAL_BURN_IN_QUESTIONS": "2999",
        "FUGAL_V2_BURN_IN_QUESTIONS": "1499",
        "FUGAL_EXPLORE_FAIL_MARGIN": "0.3",
        "FUGAL_EXPLORE_DISAGREE_MAX": "0.35",
        "FUGAL_EXPLORE_DISAGREE_MIN_SAMPLES": "9",
        "FUGAL_EXPLORE_DISAGREE_MIN_VOTERS": "4",
        "FUGAL_DEDUP_THRESHOLD": "0.98",
        "FUGAL_DEDUP_PARAM_DISTANCE": "0.25",
        "FUGAL_DEDUP_EPSILON_AGREEMENT": "0.9",
        "FUGAL_DEDUP_EPSILON": "0.01",
        "FUGAL_EVIDENCE_HALF_LIFE": "199",
        "FUGAL_MAX_MISSED_EVIDENCE": "11",
        "FUGAL_FRAME_HALF_LIFE": "499",
        "FUGAL_FRAME_PRIOR_STRENGTH": "21",
        "FUGAL_FRAME_PRIOR_ACCURACY": "0.6",
        "FUGAL_FRAME_DEFAULT_COMPLETION_TOKENS": "257",
        "FUGAL_POOL_MIN_SIZE": "11999",
        "FUGAL_MAX_WEIGHT_DELTA": "0.25",
        "FUGAL_REQUIRE_COMMITMENT": "0",
        "FUGAL_HEAD_HIDDEN_DIM": "1023",
        "FUGAL_HEAD_MAX_MODELS": "63",
        "FUGAL_HEAD_MAX_BYTES": str(1024 * 1024 - 1),
        "FUGAL_HEAD_MAX_DECOMPRESSED": str(8 * 1024 * 1024 - 1),
        "FUGAL_HEAD_MAX_MODEL_ID_LEN": "255",
        "FUGAL_MODEL_PRICES": prices_override,
        "FUGAL_CHAMPION_ACCURACY_VALUE": "31",
        "FUGAL_CHAMPION_HARD_WEIGHT": "3",
        "FUGAL_CHAMPION_ANCHOR_MODEL": "anthropic/claude-opus-5.5",
        "FUGAL_CHAMPION_MARGIN": "0.25",
        "FUGAL_CHAMPION_CONFIDENCE": "0.95",
        "FUGAL_CHAMPION_BOOTSTRAP_RESAMPLES": "1999",
        "FUGAL_CHAMPION_HARD_GATE_MIN_QUESTIONS": "101",
        "FUGAL_CHAMPION_TIME_GATE_FACTOR": "3",
        "FUGAL_CHAMPION_GHOST_OFF": "4",
        "FUGAL_CHAMPION_GHOST_RETIRE": "25",
        "FUGAL_CHAMPION_WINDOW_EPOCHS": "47",
        "FUGAL_CHAMPION_MIN_ACCURACY": "0.35",
        "FUGAL_CHAMPION_MIN_PAIRED_QUESTIONS": "1499",
        "FUGAL_PAYOUT_SEAT_SHARES": "0.80,0.15,0.05",
        "FUGAL_SKIP_BENCHMARKS": "",
        "FUGAL_POOL_UNPINNED": "1",
        "FUGAL_OPENROUTER_BASE": "http://127.0.0.1:9/api/v1",
    }


# Runs in a child process: each variable is set, config is re-imported, the
# digest is taken, and the variable is removed again. A child keeps the
# reloads out of the test session's config module.
_DIGEST_SCRIPT = r"""
import importlib, json, os, sys
from fugal_subnet import config, fingerprint

alternates = json.loads(sys.argv[1])

def digest():
    importlib.reload(config)
    return fingerprint.consensus_digest()

out = {"__baseline__": digest()}
for name, value in alternates.items():
    os.environ[name] = value
    try:
        out[name] = digest()
    except Exception as e:  # recorded; the test reports it
        out[name] = "error: %r" % (e,)
    finally:
        del os.environ[name]
out["__baseline_again__"] = digest()
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def env_digests(tmp_path_factory):
    from fugal_subnet import config

    prices = json.loads((ROOT / "fugal_subnet" / "routing_data" / "models.json").read_text())
    prices[0] = dict(prices[0], out=prices[0]["out"] * 2)
    override = tmp_path_factory.mktemp("prices") / "models_override.json"
    override.write_text(json.dumps(prices), encoding="utf-8")
    alternates = _alternates(str(override))
    assert set(alternates) == set(config.CONSENSUS_ENV_VARS), (
        "every consensus variable needs an alternate value here, so that it "
        "is shown to move the digest"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("FUGAL_")}
    run = subprocess.run(
        [sys.executable, "-c", _DIGEST_SCRIPT, json.dumps(alternates)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, check=True,
    )
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["__baseline__"] == out["__baseline_again__"]
    return out


def _consensus_env_params():
    from fugal_subnet import config

    for name in config.CONSENSUS_ENV_VARS:
        if name == "FUGAL_MODEL_PRICES":
            yield pytest.param(name, marks=pytest.mark.xfail(
                strict=True,
                reason="FUGAL_MODEL_PRICES is a consensus override but does not move "
                       "consensus_digest: config.TEE_MODEL_PRICES_PATH is read by the "
                       "miner (neurons/miner.py) and routing_protocol.identity() hashes "
                       "the fixed routing_data/models.json, contrary to the comment in "
                       "fingerprint.consensus_digest. Reported, not fixed here.",
            ))
        else:
            yield name


@pytest.mark.parametrize("name", list(_consensus_env_params()))
def test_every_consensus_env_var_moves_the_consensus_digest(env_digests, name):
    value = env_digests[name]
    assert not value.startswith("error:"), value
    assert value != env_digests["__baseline__"], (
        f"{name} changes how a neuron scores but leaves consensus_digest unchanged"
    )


# The fields consensus_digest documents as score-affecting, and the ones it
# documents as excluded (they differ legitimately between honest validators).
_DIGESTED = ("epoch", "scoring", "routing_protocol", "packages", "numpy_blas",
             "cpu_dispatch", "grader_sha256", "routing")
_NOT_DIGESTED = ("platform", "torch_runtime", "code")


def _perturbed(fp: dict, key: str) -> dict:
    out = json.loads(json.dumps(fp))
    value = out[key]
    out[key] = {**value, "__perturbed__": 1} if isinstance(value, dict) else f"{value}-other"
    return out


@pytest.fixture(scope="module")
def fingerprint_now():
    from fugal_subnet.fingerprint import environment_fingerprint

    return environment_fingerprint()


@pytest.mark.parametrize("key", _DIGESTED)
def test_each_score_affecting_fingerprint_field_moves_the_digest(fingerprint_now, key):
    from fugal_subnet.fingerprint import consensus_digest

    base = consensus_digest(fingerprint_now)
    assert consensus_digest(_perturbed(fingerprint_now, key)) != base


@pytest.mark.parametrize("key", _NOT_DIGESTED)
def test_platform_and_code_fields_do_not_move_the_digest(fingerprint_now, key):
    from fugal_subnet.fingerprint import consensus_digest

    base = consensus_digest(fingerprint_now)
    assert consensus_digest(_perturbed(fingerprint_now, key)) == base


def test_a_different_routing_protocol_identity_moves_the_live_digest(monkeypatch):
    from fugal_subnet import routing_protocol
    from fugal_subnet.fingerprint import consensus_digest

    base = consensus_digest()
    monkeypatch.setattr(routing_protocol, "identity", lambda: "a-different-protocol")
    assert consensus_digest() != base


def test_a_package_that_disagrees_with_its_pin_is_reported(monkeypatch):
    from fugal_subnet import fingerprint

    pins = {"numpy": "2.2.6", "torch": "2.9.0"}
    monkeypatch.setattr(fingerprint, "_expected_pins", lambda: dict(pins))
    # Control: the installed versions are the pinned ones.
    monkeypatch.setattr(fingerprint, "_package_versions", lambda: dict(pins))
    assert not [p for p in fingerprint.check_environment(consensus_overrides=False)
                if "pyproject pins" in p]
    # One thing different: numpy is another build.
    monkeypatch.setattr(fingerprint, "_package_versions",
                        lambda: {**pins, "numpy": "2.3.0"})
    problems = [p for p in fingerprint.check_environment(consensus_overrides=False)
                if "pyproject pins" in p]
    assert len(problems) == 1 and problems[0].startswith("numpy 2.3.0")


# --- prices: the consensus cost denominator --------------------------------


_V1_PINS = {
    "openai/gpt-5.5": (5.0e-6, 30.0e-6),
    "google/gemini-3.1-pro-preview": (2.0e-6, 12.0e-6),
}
_V2_PINS = {
    "anthropic/claude-fable-5.1": (10.0e-6, 50.0e-6),
    "anthropic/claude-opus-5.5": (4.0e-6, 20.0e-6),
}


def test_load_prices_returns_exact_per_token_prices_for_both_tables():
    from fugal_subnet.api import load_prices
    from fugal_subnet.routing_protocol import price_table, prices_path

    v1 = load_prices()  # data/models.json
    for model, expected in _V1_PINS.items():
        assert v1[model] == pytest.approx(expected, rel=1e-12, abs=0)
    v1_shipped = load_prices(str(prices_path(1)))
    v2 = load_prices(str(prices_path(2)))
    for model, expected in _V2_PINS.items():
        assert v2[model] == pytest.approx(expected, rel=1e-12, abs=0)
    # The loader and the versioned accessor must agree exactly.
    assert v1_shipped == v1 == price_table(1)
    assert v2 == price_table(2)


# --- vendor/success_contract ------------------------------------------------


def _head(n):
    return fixtures.arrays(np.zeros((n, 1024)), np.zeros(n), [f"m{i}" for i in range(n)])


def test_a_head_may_name_64_models_and_not_65():
    assert len(_head(64)["models"]) == 64
    with pytest.raises(ValueError, match="invalid models"):
        _head(65)


def test_equal_utilities_route_to_the_lower_index_on_a_realistic_model_count():
    # 18 models (the v2 table's size) with two utility levels. A non-stable
    # sort orders ties arbitrarily above 16 elements; the route must not.
    n = 18
    p = np.where(np.arange(n) % 2 == 1, 0.8, 0.4)
    order = c.rank(p, np.zeros(n))
    expected = [i for i in range(n) if i % 2 == 1] + [i for i in range(n) if i % 2 == 0]
    assert order.tolist() == expected
    # Batched (one row per question) behaves the same.
    batch = c.rank(np.vstack([p, p[::-1]]), np.zeros(n))
    assert batch[0].tolist() == expected


def test_utilities_closer_than_one_quantum_tie_and_break_by_index():
    costs = np.array([0.1, 0.1])
    # Control: more than a quantum apart, the better one wins.
    assert c.rank(np.array([0.7, 0.7 + 3 * c.QUANTUM]), costs).tolist() == [1, 0]
    # One thing different: less than a quantum apart (float noise between
    # machines, or a sub-quantum edge) is a tie, and the lower index routes.
    assert c.rank(np.array([0.7, 0.7 + 1e-12]), costs).tolist() == [0, 1]
    assert c.rank(np.array([0.7, 0.7 + 0.3 * c.QUANTUM]), costs).tolist() == [0, 1]


@pytest.mark.parametrize("key", ["mean_in_tokens", "mean_out_tokens"])
def test_a_negative_token_mean_is_refused(key):
    z = _head(2)
    c.validate_head(z)  # control
    bad = dict(z)
    bad[key] = np.array([-1.0, 300.0])
    with pytest.raises(ValueError, match=f"negative {key}"):
        c.validate_head(bad)


def test_a_negative_lambda_is_refused():
    bad = dict(_head(2))
    bad["lam"] = np.asarray(-1.0)
    with pytest.raises(ValueError, match="negative lam"):
        c.validate_head(bad)


def test_a_manifest_with_a_negative_token_mean_is_refused():
    m = fixtures.manifest(["a", "b"])
    c.validate_manifest(m)  # control
    m["models"][1]["mean_out_tokens"] = -1.0
    with pytest.raises(ValueError, match="invalid token observations"):
        c.validate_manifest(m)


def _archive(arrays, duplicate=None):
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns on a duplicate name
        with zipfile.ZipFile(buf, "w") as zf:
            for name, value in arrays.items():
                member = io.BytesIO()
                np.save(member, value, allow_pickle=False)
                zf.writestr(f"{name}.npy", member.getvalue())
            if duplicate is not None:
                name, value = duplicate
                member = io.BytesIO()
                np.save(member, value, allow_pickle=False)
                zf.writestr(f"{name}.npy", member.getvalue())
    return buf.getvalue()


def test_an_archive_with_a_duplicate_member_name_is_refused():
    z = _head(2)
    assert set(c.load(_archive(z))) == set(z)  # control: unique names load
    # One thing different: a second b.npy. Which copy a reader sees depends
    # on the reader, so two validators could score two different heads.
    data = _archive(z, duplicate=("b", np.array([5.0, -5.0], dtype=np.float32)))
    with pytest.raises(ValueError, match="invalid archive entries"):
        c.read_archive(data)


# --- success_training -------------------------------------------------------


def test_training_and_validation_rows_may_not_overlap():
    from fugal_subnet.success_training import fit

    h = np.zeros((6, 1024), np.float32)
    y = np.ones((6, 2))
    fit(h, y, np.array([0, 1, 2]), np.array([3, 4, 5]), epochs=1)  # control
    with pytest.raises(ValueError, match="split"):
        fit(h, y, np.array([0, 1, 2, 3]), np.array([3, 4, 5]), epochs=1)
