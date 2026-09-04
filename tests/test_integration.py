"""Gate D integration test: validator ↔ miner full pipeline with mock bittensor.

Tests the complete flow:
1. Generate a synthetic head .npz
2. Miner loads and serves it
3. Validator queries miner, receives head, evaluates, scores, computes weights
"""
from __future__ import annotations

import base64
import hashlib
import os
import sys
import tempfile

import numpy as np

# Patch bittensor before any fugal imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tests.bt_mock  # noqa: E402, F401
from fugal_subnet.benchmarks.loader import load_all
from fugal_subnet.benchmarks.slicer import derive_nonce, select_slice
from fugal_subnet.config import HEAD_HIDDEN_DIM, SLICE_SIZE
from fugal_subnet.consensus import ValidatorReport, check_self_consistency, compute_consensus
from fugal_subnet.dedup import find_duplicates
from fugal_subnet.epoch_logger import (
    EpochLog,
    EpochTimer,
    detect_anomalies,
    read_epoch_logs,
    write_epoch_log,
)
from fugal_subnet.head_eval import (
    HeadArtifact,
    evaluate_head,
    load_head_from_b64,
    load_head_from_npz,
)
from fugal_subnet.matrix import build_matrix_mock
from fugal_subnet.protocol import FugalProofSynapse
from fugal_subnet.rewards import cap_weight_change, compute_weights
from fugal_subnet.scoring import ScoringState, update_scores
from fugal_subnet.soft_targets import compute_soft_targets

MODEL_POOL = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
]


def make_synthetic_head(models: list[str], hidden_dim: int = HEAD_HIDDEN_DIM,
                        seed: int = 42) -> tuple[bytes, str]:
    """Create a synthetic .npz head artifact and return (bytes, sha256_hash)."""
    rng = np.random.RandomState(seed)
    n_models = len(models)
    W = rng.randn(n_models, hidden_dim).astype(np.float32) * 0.01
    b = np.zeros(n_models, dtype=np.float32)
    models_arr = np.array(models, dtype="U100")

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f, W=W, b=b, models=models_arr)
        path = f.name

    with open(path, "rb") as f:
        data = f.read()
    os.unlink(path)

    h = hashlib.sha256(data).hexdigest()
    return data, h


def test_miner_loads_head():
    """Miner can load a valid .npz head file."""
    data, h = make_synthetic_head(MODEL_POOL)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        f.write(data)
        path = f.name

    try:
        from neurons.miner import _load_head_file
        loaded = _load_head_file(path)
        assert loaded == data
        assert len(loaded) < 1_000_000
    finally:
        os.unlink(path)

    print("  [PASS] Miner loads head")


def test_synapse_roundtrip():
    """FugalProofSynapse carries a full bundle inline, end to end.

    The bundle travels in the response rather than as a URL the validator
    fetches — see fugal_subnet/protocol.py. This asserts a realistically sized
    payload survives the synapse's own field caps.
    """
    import json

    data, h = make_synthetic_head(MODEL_POOL)
    b64 = base64.b64encode(data).decode("ascii")

    results = [{
        "question_id": f"q{i}", "routed_model": MODEL_POOL[i % len(MODEL_POOL)],
        "correct": bool(i % 3), "cost_usd": 0.001,
        "response_hash": "ab" * 32, "prompt_tokens": 500,
        "completion_tokens": 300, "is_exploration": False,
    } for i in range(SLICE_SIZE)]
    proof_json = json.dumps({
        "epoch_id": "e00000001", "nonce": "cd" * 32, "questions_hash": "ef" * 32,
        "weights_hash": h, "source_hash": "01" * 32, "results": results,
        "total_cost_usd": 0.3, "per_model_costs": {m: 0.1 for m in MODEL_POOL},
        "attestation_quote": "00" * 632, "timestamp": 0.0,
    }, separators=(",", ":"))

    synapse = FugalProofSynapse(epoch_id="e00000001", nonce="cd" * 32)
    synapse.proof_json = proof_json
    synapse.head_npz_b64 = b64
    synapse.weights_hash = h
    synapse.proof_hash = "ff" * 32

    head = load_head_from_b64(synapse.head_npz_b64)
    assert head.W.shape == (len(MODEL_POOL), HEAD_HIDDEN_DIM)
    assert head.b.shape == (len(MODEL_POOL),)
    assert list(head.models) == MODEL_POOL
    assert len(json.loads(synapse.proof_json)["results"]) == SLICE_SIZE

    kb = (len(synapse.proof_json) + len(synapse.head_npz_b64)) // 1024
    print(f"  Inline bundle: {kb} KB ({SLICE_SIZE} results + head)")
    print("  [PASS] Synapse roundtrip")


def make_synthetic_pool(n: int = 200, seed: int = 7) -> list[dict]:
    rng = np.random.RandomState(seed)
    benches = ["alpha", "beta", "gamma", "delta"]
    return [{
        "prompt": f"What is {rng.randint(1, 100)} + {rng.randint(1, 100)}?",
        "gold": str(rng.randint(2, 200)),
        "grader_id": "numeric_final",
        "benchmark": benches[i % len(benches)],
        "question_id": f"syn_{i:04d}",
        "metadata": {},
    } for i in range(n)]


def test_full_validator_pipeline():
    """Full validator epoch: slice → matrix → eval → score → weights."""
    print("\n--- Full Validator Pipeline ---")

    # 1. Load benchmarks and slice (fall back to synthetic when offline)
    try:
        pool = load_all(strict=False)
    except Exception:
        pool = []
    if not pool:
        pool = make_synthetic_pool()
    print(f"  Benchmark pool: {len(pool)} questions")

    nonce = derive_nonce("e000001_abc12345", "0xdeadbeef")
    questions = select_slice(nonce, pool, min(SLICE_SIZE, 50))
    print(f"  Slice: {questions[0]['benchmark']}... ({len(questions)} questions)")

    # 2. Build mock matrix
    def mock_fn(model, question):
        seed = hash((model, question["question_id"])) % 2**31
        rng = np.random.RandomState(seed)
        correct = int(rng.random() > 0.4)
        return ("42" if correct else "wrong"), correct

    matrix_result = build_matrix_mock(questions, MODEL_POOL, mock_fn)
    print(f"  Matrix: {matrix_result.matrix.shape}")

    # 3. Soft targets
    soft = compute_soft_targets(matrix_result.matrix)
    assert soft.shape == (len(questions), len(MODEL_POOL))
    assert np.allclose(soft.sum(axis=1), 1.0)
    print(f"  Soft targets: {soft.shape}, all rows sum to 1.0")

    # 4. Create two synthetic heads (different seeds = different quality)
    head_data_1, hash_1 = make_synthetic_head(MODEL_POOL, seed=42)
    head_data_2, hash_2 = make_synthetic_head(MODEL_POOL, seed=99)

    head1 = load_head_from_b64(base64.b64encode(head_data_1).decode())
    head2 = load_head_from_b64(base64.b64encode(head_data_2).decode())

    # 5. Mock hidden states
    np.random.seed(int.from_bytes(nonce[:4], "big"))
    hidden = np.random.randn(len(questions), HEAD_HIDDEN_DIM).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)

    model_costs = {m: 0.005 * (i + 1) for i, m in enumerate(MODEL_POOL)}

    # 6. Evaluate heads
    score1 = evaluate_head(head1, hidden, matrix_result.matrix,
                           MODEL_POOL, soft, model_costs)
    score2 = evaluate_head(head2, hidden, matrix_result.matrix,
                           MODEL_POOL, soft, model_costs)
    print(f"  Head 1: acc={score1.accuracy:.3f} cost_eff={score1.cost_efficiency:.3f} kl={score1.kl_score:.3f}")
    print(f"  Head 2: acc={score2.accuracy:.3f} cost_eff={score2.cost_efficiency:.3f} kl={score2.kl_score:.3f}")

    # 7. Scoring
    state = ScoringState()
    epoch_scores = {1: score1, 2: score2}
    head_hashes = {1: hash_1, 2: hash_2}
    state = update_scores(state, epoch_scores, head_hashes, acc_best=0.8)

    for uid, rec in state.records.items():
        print(f"  UID {uid}: composite={rec.composite_score:.4f} epochs_seen={rec.epochs_seen}")

    # 8. Dedup
    head_outputs = {1: score1.routing_decisions, 2: score2.routing_decisions}
    commit_blocks = {1: 100, 2: 200}
    dupes = find_duplicates(head_outputs, commit_blocks)
    print(f"  Dedup disqualified: {dupes}")

    # 9. Rewards
    uids, weights = compute_weights(state.records,
                                    dedup_disqualified=dupes)
    print(f"  Weights: UIDs={uids}, weights={[f'{w:.4f}' for w in weights]}")
    assert abs(sum(weights) - 1.0) < 1e-6, f"Weights don't sum to 1: {sum(weights)}"
    print(f"  Weight sum: {sum(weights):.6f} ✓")

    # 10. Mock set_weights
    import bittensor as bt
    subtensor = bt.Subtensor(network="test")
    wallet = bt.Wallet(name="test_validator")
    success, msg = subtensor.set_weights(
        wallet=wallet, netuid=1, uids=uids, weights=weights,
    )
    assert success
    print(f"  set_weights: {msg}")

    print("\n  [PASS] Full validator pipeline")


def test_dendrite_query_flow():
    """Simulate a validator querying a miner and parsing the inline bundle."""
    import json

    data, h = make_synthetic_head(MODEL_POOL)
    b64 = base64.b64encode(data).decode("ascii")
    proof_json = json.dumps({"epoch_id": "e00000001", "results": []},
                            separators=(",", ":"))

    def miner_forward(synapse: FugalProofSynapse) -> FugalProofSynapse:
        synapse.proof_json = proof_json
        synapse.head_npz_b64 = b64
        synapse.weights_hash = h
        synapse.proof_hash = "ab" * 32
        return synapse

    synapse = FugalProofSynapse(epoch_id="e00000001", nonce="cd" * 32)
    response = miner_forward(synapse)

    assert response.head_npz_b64 == b64
    assert response.proof_json == proof_json
    assert response.weights_hash == h

    # The validator recovers the head from the response and validates it.
    head = load_head_from_b64(response.head_npz_b64)
    assert head.W.shape[0] == len(MODEL_POOL)
    assert head.W.shape[1] == HEAD_HIDDEN_DIM
    assert hashlib.sha256(base64.b64decode(response.head_npz_b64)).hexdigest() == h

    print("  [PASS] Dendrite query flow")


def test_commit_reveal():
    """Commit-reveal: commit hash matches reveal hash."""
    print("\n  [TEST] Commit-reveal integrity")
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["FUGAL_EPOCH_DIR"] = tmpdir
        # Reload to pick up env var
        import fugal_subnet.commit_reveal as cr
        cr.EPOCH_DIR = tmpdir

        questions = [
            {"prompt": "What is 2+2?", "gold": "4", "question_id": "q1"},
            {"prompt": "What is 3+3?", "gold": "6", "question_id": "q2"},
        ]

        commitment = cr.commit_epoch("test_e001", questions, "block_abc")
        assert commitment.commit_hash, "Commit hash should be non-empty"
        print(f"  Committed: {commitment.commit_hash[:16]}...")

        matrix = np.array([[1, 0], [0, 1]], dtype=np.int8)
        models = ["model/a", "model/b"]
        model_costs = {"model/a": 0.001, "model/b": 0.002}
        head_hashes = {0: "head_hash_abc"}
        routing = {0: [0, 1]}
        scores = {0: {"acc": 0.9}}
        weights = {0: 1.0}
        verified = cr.reveal_epoch("test_e001", questions, matrix, models,
                                   model_costs, head_hashes, routing, scores, weights)
        assert verified, "Reveal should verify against commitment"

        assert cr.verify_epoch(os.path.join(tmpdir, "test_e001")), "Independent verify should pass"

        # The reveal must publish the data flywheel: matrix + models + costs
        import json
        with open(os.path.join(tmpdir, "test_e001", "reveal.json")) as f:
            reveal = json.load(f)
        assert reveal["matrix"] == [[1, 0], [0, 1]], "Reveal must contain the matrix"
        assert reveal["models"] == models
        assert reveal["model_costs"] == model_costs

        # Tamper with questions and verify it fails
        tampered = questions + [{"prompt": "extra", "gold": "x", "question_id": "q3"}]
        tampered_ok = cr.reveal_epoch("test_e001", tampered, matrix, models,
                                      model_costs, head_hashes, routing, scores, weights)
        assert not tampered_ok, "Tampered questions should fail verification"

        print("  [PASS] Commit-reveal integrity")


def test_weight_capping():
    """Weight capping limits per-epoch weight changes."""
    print("\n  [TEST] Weight capping")

    # Large jump from 0.1 to 0.9 should be clamped
    uids, weights = cap_weight_change(
        [1, 2], [0.9, 0.1],
        [1, 2], [0.1, 0.9],
        max_delta=0.2,
    )
    w_map = dict(zip(uids, weights))
    # 0.1 + 0.2 = 0.3, 0.9 - 0.2 = 0.7, normalized: 0.3, 0.7
    assert abs(w_map[1] - 0.3) < 0.01, f"UID 1 should be ~0.3, got {w_map[1]:.3f}"
    assert abs(w_map[2] - 0.7) < 0.01, f"UID 2 should be ~0.7, got {w_map[2]:.3f}"
    print(f"  Capped: {w_map}")

    # Small change within delta — should pass through unchanged (after normalization)
    uids2, weights2 = cap_weight_change([1, 2], [0.55, 0.45], [1, 2], [0.5, 0.5], max_delta=0.2)
    w_map2 = dict(zip(uids2, weights2))
    assert abs(w_map2[1] - 0.55) < 0.01, f"Small delta should pass through, got {w_map2[1]:.3f}"

    print("  [PASS] Weight capping")


def test_epoch_logger():
    """Structured epoch logging: write, read, anomaly detection."""
    print("\n  [TEST] Epoch logger")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        import fugal_subnet.epoch_logger as el
        el.EPOCH_LOG_DIR = tmpdir

        anomalies = detect_anomalies(
            {1: {"acc": 0.8}, 2: {"acc": 0.7}},
            {0: 0.1, 1: 0.85, 2: 0.05},
            n_miners_queried=5, n_heads_valid=2,
        )
        assert any("weight_concentration" in a for a in anomalies), f"Expected weight_concentration, got {anomalies}"

        anomalies_empty = detect_anomalies({}, {}, n_miners_queried=3, n_heads_valid=0)
        assert any("no_valid_heads" in a for a in anomalies_empty)

        log = EpochLog(
            epoch_id="test_e001", block_hash="0xabc",
            timestamp=1234567890.0, n_questions=50,
            n_miners_queried=5, n_heads_valid=2, n_heads_invalid=1,
            commit_hash="deadbeef", reveal_verified=True,
            scores={1: {"acc": 0.8}, 2: {"acc": 0.7}},
            weights={1: 0.6, 2: 0.4},
            anomalies=anomalies, duration_s=12.5,
        )
        write_epoch_log(log)

        logs = read_epoch_logs()
        assert len(logs) == 1
        assert logs[0]["epoch_id"] == "test_e001"
        assert logs[0]["n_heads_valid"] == 2

        timer = EpochTimer()
        timer.start_phase("query")
        timer.start_phase("eval")
        timer.end_phase()
        assert "query" in timer.phases
        assert "eval" in timer.phases
        assert timer.total_s >= 0

        print("  [PASS] Epoch logger")


def test_consensus():
    """Multi-validator consensus: agreement and outlier detection."""
    print("\n  [TEST] Consensus")

    r1 = ValidatorReport(
        validator_uid=0, epoch_id="e001",
        scores={1: 0.8, 2: 0.7, 3: 0.5},
        weights={1: 0.5, 2: 0.3, 3: 0.2},
        commit_hash="abc",
    )
    r2 = ValidatorReport(
        validator_uid=1, epoch_id="e001",
        scores={1: 0.82, 2: 0.68, 3: 0.52},
        weights={1: 0.48, 2: 0.32, 3: 0.2},
        commit_hash="abc",
    )
    result = compute_consensus([r1, r2])
    assert result.n_validators == 2
    assert not result.outlier_validators, f"No outliers expected, got {result.outlier_validators}"
    assert result.is_valid
    print(f"  Consensus scores: {result.consensus_scores}")

    r3 = ValidatorReport(
        validator_uid=2, epoch_id="e001",
        scores={1: 0.2, 2: 0.9, 3: 0.1},
        weights={1: 0.1, 2: 0.8, 3: 0.1},
        commit_hash="abc",
    )
    result2 = compute_consensus([r1, r2, r3])
    assert 2 in result2.outlier_validators, f"Validator 2 should be outlier, got {result2.outlier_validators}"
    print(f"  Outliers detected: {result2.outlier_validators}")

    r4 = ValidatorReport(
        validator_uid=3, epoch_id="e001",
        scores={1: 0.8}, weights={1: 1.0},
        commit_hash="different_hash",
    )
    result3 = compute_consensus([r1, r4])
    assert not result3.is_valid, "Mismatched commit hashes should invalidate"
    print(f"  Commit mismatch flags: {result3.divergence_flags}")

    warnings = check_self_consistency(
        {1: 0.8, 2: 0.7},
        {1: 0.3, 2: 0.2},
        my_uid=0,
    )
    assert len(warnings) > 0, "Large divergence should produce warnings"
    print(f"  Self-check warnings: {warnings}")

    print("  [PASS] Consensus")


def test_head_security():
    """Malicious head artifacts are rejected before any allocation."""
    print("\n  [TEST] Head security")
    import io

    # Zip bomb: tiny compressed archive, 16 MB declared decompressed size
    big = np.zeros((4000, 1024), dtype=np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, W=big, b=np.zeros(4000, dtype=np.float32),
                        models=np.array(["m"] * 4000, dtype="U10"))
    bomb = buf.getvalue()
    assert len(bomb) < 1_000_000, "test premise: bomb must fit the 1MB wire cap"
    try:
        load_head_from_npz(bomb)
        raise AssertionError("zip bomb not rejected")
    except ValueError as e:
        assert "decompresses" in str(e), f"unexpected rejection reason: {e}"
    print("  Zip bomb rejected")

    # Too many model rows
    buf = io.BytesIO()
    np.savez(buf, W=np.zeros((65, 1024), dtype=np.float32),
             b=np.zeros(65, dtype=np.float32),
             models=np.array([f"m{i}" for i in range(65)], dtype="U10"))
    try:
        load_head_from_npz(buf.getvalue())
        raise AssertionError("oversized model pool not rejected")
    except ValueError as e:
        assert "model rows" in str(e)
    print("  65-model head rejected")

    # Non-2D W
    buf = io.BytesIO()
    np.savez(buf, W=np.zeros((2, 3, 4), dtype=np.float32),
             b=np.zeros(2, dtype=np.float32),
             models=np.array(["a", "b"], dtype="U10"))
    try:
        load_head_from_npz(buf.getvalue())
        raise AssertionError("3-D W not rejected")
    except ValueError:
        pass
    print("  Non-2D W rejected")

    # Garbage bytes
    try:
        load_head_from_npz(b"not a zip at all")
        raise AssertionError("garbage not rejected")
    except ValueError:
        pass
    print("  Garbage bytes rejected")

    # NaN weights
    W = np.zeros((2, 1024), dtype=np.float32)
    W[0, 0] = np.nan
    buf = io.BytesIO()
    np.savez(buf, W=W, b=np.zeros(2, dtype=np.float32),
             models=np.array(["a", "b"], dtype="U10"))
    try:
        load_head_from_npz(buf.getvalue())
        raise AssertionError("NaN weights not rejected")
    except ValueError:
        pass
    print("  NaN weights rejected")

    print("  [PASS] Head security")


def test_cost_efficiency_cap():
    """Always-route-to-cheapest cannot earn cost_efficiency > 1 and dominate."""
    print("\n  [TEST] Cost efficiency cap")
    models = ["m/cheap", "m/mid", "m/expensive"]
    N = 50
    rng = np.random.RandomState(0)
    matrix = np.ones((N, 3), dtype=np.int8)  # every model correct everywhere
    soft = compute_soft_targets(matrix)
    hidden = rng.randn(N, HEAD_HIDDEN_DIM).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    costs = {"m/cheap": 0.001, "m/mid": 0.01, "m/expensive": 0.05}

    # Head hard-biased to the cheapest model
    W = np.zeros((3, HEAD_HIDDEN_DIM), dtype=np.float32)
    b = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    head = HeadArtifact(W=W, b=b, models=models, commit_hash="")
    score = evaluate_head(head, hidden, matrix, models, soft, costs)
    assert score.cost_efficiency <= 1.0, f"cost_efficiency uncapped: {score.cost_efficiency}"
    assert score.accuracy == 1.0
    print(f"  cost_eff={score.cost_efficiency:.3f} (capped) acc={score.accuracy:.3f}")

    # Questions no model can answer are excluded from scoring for everyone
    dead_matrix = matrix.copy()
    dead_matrix[:10, :] = 0
    soft_dead = compute_soft_targets(dead_matrix)
    score_dead = evaluate_head(head, hidden, dead_matrix, models, soft_dead, costs)
    assert score_dead.accuracy == 1.0, "dead rows must not count against accuracy"
    print("  Dead rows excluded from scoring")
    print("  [PASS] Cost efficiency cap")


def test_stratified_slicer():
    """Slices are benchmark-stratified, deterministic, and nonce-sensitive."""
    print("\n  [TEST] Stratified slicer")
    from collections import Counter
    pool = make_synthetic_pool(200)  # 4 benchmarks × 50 questions
    nonce = derive_nonce("e1", "0xabc")

    s = select_slice(nonce, pool, 40)
    counts = Counter(q["benchmark"] for q in s)
    assert all(v == 10 for v in counts.values()), f"not stratified: {counts}"

    s2 = select_slice(nonce, pool, 40)
    assert [q["question_id"] for q in s] == [q["question_id"] for q in s2], "not deterministic"

    s3 = select_slice(derive_nonce("e2", "0xabc"), pool, 40)
    assert [q["question_id"] for q in s] != [q["question_id"] for q in s3], "nonce ignored"

    # A benchmark with too few questions cedes its slots to the others
    small = [q for q in pool if q["benchmark"] == "alpha"][:2]
    rest = [q for q in pool if q["benchmark"] != "alpha"]
    s4 = select_slice(nonce, rest + small, 40)
    assert len(s4) == 40, f"slots not redistributed: {len(s4)}"
    print(f"  Stratified counts: {dict(counts)}")
    print("  [PASS] Stratified slicer")


def test_unpriced_and_expensive_models():
    """What replaced the pre-TEE model-pool policy.

    The validator used to assemble a shared, priced, cost-capped model pool
    because it paid for every call in it. Under TEE there is no shared pool and
    the miner pays, so both halves of that policy are re-expressed:

      - unpriced model  -> a hard error, because an uncosted route cannot be
                           scored (it used to be silently excluded)
      - expensive model -> allowed, and it lowers only that miner's own thrift
                           (it used to be excluded to protect a shared budget)
    """
    print("\n  [TEST] Unpriced and expensive models")
    from fugal_subnet.evidence import Evidence
    from fugal_subnet.scoring import composite
    from fugal_subnet.tee.runtime import MeteringProxy, UnpricedModel

    proxy = MeteringProxy(port=0)
    proxy.prices = {"a/cheap": (1e-7, 2e-7), "c/pricey": (5e-6, 3e-5)}

    try:
        proxy.price_call("zz/unknown", 500, 300)
        raise AssertionError("unpriced model must not be costable")
    except UnpricedModel:
        pass
    print("  Unpriced model is a hard error, never a default rate")

    cheap = proxy.price_call("a/cheap", 500, 300)
    pricey = proxy.price_call("c/pricey", 500, 300)
    assert pricey > cheap * 50, "price table must distinguish models"
    print(f"  Price table distinguishes models: ${cheap:.6f} vs ${pricey:.6f}")

    # Routing expensively is permitted and self-punishing.
    frugal = Evidence("h", n_correct=9000.0, n_total=10000.0,
                      cost_sum=1.0, ref_cost_sum=6.0, pool_size=1e9)
    lavish = Evidence("h", n_correct=9000.0, n_total=10000.0,
                      cost_sum=36.0, ref_cost_sum=6.0, pool_size=1e9)
    assert composite(frugal, 0.9) > composite(lavish, 0.9)
    print("  Expensive routing costs the miner, not the validator")
    print("  [PASS] Unpriced and expensive models")


def main():
    print("=" * 60)
    print("GATE D — Integration Test")
    print("=" * 60)

    test_miner_loads_head()
    test_synapse_roundtrip()
    test_dendrite_query_flow()
    test_full_validator_pipeline()
    test_commit_reveal()
    test_weight_capping()
    test_epoch_logger()
    test_consensus()
    test_head_security()
    test_cost_efficiency_cap()
    test_stratified_slicer()
    test_unpriced_and_expensive_models()

    print("\n" + "=" * 60)
    print("ALL TESTS PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
