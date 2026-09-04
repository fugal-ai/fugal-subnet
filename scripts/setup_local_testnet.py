#!/usr/bin/env python3
"""Set up and run a full local testnet epoch inside Docker.

This script:
1. Waits for the subtensor chain
2. Sets up Alice's pre-funded devnet wallet
3. Creates a subnet using Alice's TAO
4. Creates and funds validator/miner wallets
5. Registers both wallets on the subnet
6. Trains a synthetic head
7. Starts the miner in background
8. Runs one validator epoch
9. Verifies weights were set

Usage (inside Docker container with subtensor running):
    python scripts/setup_local_testnet.py
"""
import hashlib
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fugal.setup")

CHAIN_ENDPOINT = os.getenv("CHAIN_ENDPOINT", "ws://subtensor:9946")


def wait_for_chain(endpoint: str, timeout: int = 120):
    """Wait for the subtensor node to be ready."""
    import bittensor as bt
    logger.info("Waiting for chain at %s ...", endpoint)
    start = time.time()
    while time.time() - start < timeout:
        try:
            sub = bt.Subtensor(network=endpoint)
            block = sub.get_current_block()
            logger.info("Chain is live at block %d", block)
            return sub
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"Chain not ready after {timeout}s")


def get_alice_keypair():
    """Get Alice's well-known devnet keypair."""
    try:
        from substrateinterface import Keypair
    except ImportError:
        from bittensor_wallet import Keypair
    return Keypair.create_from_uri("//Alice")


def create_wallets():
    """Create test validator and miner wallets."""
    import bittensor as bt

    wallets = {}
    for name in ["fugal_validator", "fugal_miner"]:
        wallet = bt.Wallet(name=name, hotkey="default")
        if not wallet.coldkey_file.exists_on_device():
            logger.info("Creating wallet: %s", name)
            wallet.create_new_coldkey(use_password=False, overwrite=True)
            wallet.create_new_hotkey(overwrite=True)
        else:
            logger.info("Wallet %s already exists", name)
        wallets[name] = wallet
    return wallets


def submit_extrinsic(subtensor, call, keypair):
    """Submit a signed extrinsic using substrate interface directly."""
    extrinsic = subtensor.substrate.create_signed_extrinsic(call=call, keypair=keypair)
    receipt = subtensor.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
    if receipt.is_success:
        return True
    logger.error("  Extrinsic failed: %s", getattr(receipt, "error_message", "unknown"))
    return False


def fund_wallet(subtensor, dest_wallet, amount_tao=10000):
    """Transfer TAO from Alice to a test wallet using substrate directly."""
    alice_kp = get_alice_keypair()
    dest_addr = dest_wallet.coldkeypub.ss58_address
    logger.info("Funding %s (%s) with %d TAO...", dest_wallet.name, dest_addr, amount_tao)

    call = subtensor.substrate.compose_call(
        call_module="Balances",
        call_function="transfer_keep_alive",
        call_params={
            "dest": dest_addr,
            "value": int(amount_tao * 1e9),
        },
    )
    success = submit_extrinsic(subtensor, call, alice_kp)
    if success:
        new_balance = subtensor.get_balance(dest_addr)
        logger.info("  Funded successfully, new balance: %s", new_balance)
    else:
        logger.error("  Funding FAILED")
    return success


def create_subnet(subtensor):
    """Create a subnet using Alice's pre-funded account via substrate directly."""
    alice_kp = get_alice_keypair()
    balance = subtensor.get_balance(alice_kp.ss58_address)
    logger.info("Alice balance: %s", balance)
    logger.info("Creating subnet...")

    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="register_network",
        call_params={"hotkey": alice_kp.ss58_address},
    )
    success = submit_extrinsic(subtensor, call, alice_kp)
    if success:
        logger.info("Subnet created successfully")
    else:
        logger.warning("Subnet creation failed — may already exist")

    netuids = subtensor.get_all_subnets_netuid()
    logger.info("Active netuids: %s", netuids)
    netuid = max(netuids) if netuids else 1
    logger.info("Using netuid %d", netuid)
    return netuid


def register_wallet(subtensor, wallet, netuid):
    """Register a wallet on the subnet."""
    logger.info("Registering %s on netuid %d...", wallet.name, netuid)
    try:
        success = subtensor.register(wallet=wallet, netuid=netuid)
        if success:
            logger.info("  Registered successfully")
        else:
            logger.info("  Registration returned False — may already be registered")
    except Exception as e:
        logger.warning("  Registration: %s", e)


def train_head(output_path: str = "data/local_head.npz"):
    """Train a quick synthetic head."""
    from fugal_subnet.config import HEAD_HIDDEN_DIM
    from fugal_subnet.soft_targets import compute_soft_targets

    models = ["openai/gpt-4o-mini", "google/gemini-2.0-flash-001", "anthropic/claude-3.5-haiku"]
    n_models = len(models)
    n_questions = 100
    hidden_dim = HEAD_HIDDEN_DIM

    rng = np.random.RandomState(42)
    matrix = (rng.rand(n_questions, n_models) > 0.4).astype(np.int8)
    for i in range(n_questions):
        if matrix[i].sum() == 0:
            matrix[i, rng.randint(n_models)] = 1

    hidden = rng.randn(n_questions, hidden_dim).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)

    soft = compute_soft_targets(matrix)

    import torch
    import torch.nn.functional as F

    H = torch.tensor(hidden, dtype=torch.float32)
    T = torch.tensor(soft, dtype=torch.float32)
    W = torch.randn(n_models, hidden_dim) * 0.01
    b = torch.zeros(n_models)
    W.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.AdamW([W, b], lr=1e-3)

    for epoch in range(50):
        logits = H @ W.T + b
        loss = F.kl_div(F.log_softmax(logits, dim=1), T, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(
        output_path,
        W=W.detach().numpy().astype(np.float32),
        b=b.detach().numpy().astype(np.float32),
        models=np.array(models, dtype="U100"),
    )
    logger.info("Head trained and saved to %s (%d bytes)",
                output_path, os.path.getsize(output_path))
    return output_path


# A deterministic stand-in for a model reply. The container has no API key and
# must never spend money, but the point of this harness is to exercise the REAL
# TEE path end to end, so only the network call is replaced — the routing,
# metering, grading, attestation, verification and scoring are all genuine.
_SKILL = {}


def _stub_model_call(proxy, model_id, question):
    from fugal_subnet.tee.runtime import APICallRecord

    digest = hashlib.sha256(
        f"{question['question_id']}|{model_id}".encode()
    ).digest()
    correct = (digest[0] / 255.0) < _SKILL.get(model_id, 0.5)
    text = question["gold"] if correct else "0"
    proxy.records.append(APICallRecord(
        model_id=model_id, prompt_tokens=500, completion_tokens=300,
        cost_usd=proxy.price_call(model_id, 500, 300), timestamp=0.0,
        response_hash=hashlib.sha256(text.encode()).hexdigest(),
    ))
    return text


def _synthetic_pool(n=100):
    rng = np.random.RandomState(42)
    return [
        {
            "prompt": f"What is {rng.randint(1, 100)} + {rng.randint(1, 100)}?",
            "gold": str(rng.randint(2, 200)),
            "grader_id": "numeric_final",
            "benchmark": "synthetic",
            "question_id": f"syn_{i:04d}",
            "metadata": {},
        }
        for i in range(n)
    ]


def start_miner(wallet, subtensor_network, netuid, head_path, port=8091,
                pool=None, models=None, slice_size=50, explore_size=3):
    """Start a TEE miner in a background thread.

    Serves FugalProofSynapse — the real wire protocol — and recomputes its
    proof whenever the validator asks about an epoch it has not run yet.
    """
    import bittensor as bt

    from fugal_subnet.protocol import FugalProofSynapse
    from fugal_subnet.tee import harness as harness_mod
    from fugal_subnet.tee.runtime import MeteringProxy, TEERuntime

    with open(head_path, "rb") as f:
        head_data = f.read()

    prices = _local_prices(models)
    hidden = _local_hidden(pool, head_path)
    harness_mod._call_model = _stub_model_call

    class _Proxy(MeteringProxy):
        def start(self):
            self.prices = prices

    cache = {}

    def forward(synapse: FugalProofSynapse) -> FugalProofSynapse:
        epoch_id, nonce_hex = synapse.epoch_id, synapse.nonce
        logger.info("[Miner] Proof requested for epoch %s", epoch_id)
        if epoch_id not in cache:
            proxy = _Proxy(port=0)
            proxy.start()
            proof = harness_mod.run_benchmark(
                nonce=nonce_hex, head_bytes=head_data, benchmark_pool=pool,
                proxy=proxy, hidden_states=hidden, slice_size=slice_size,
                epoch_id=epoch_id, source_hash="local-testnet",
                explore_models=models, explore_size=explore_size,
            )
            proof.timestamp = 0.0
            proof.attestation_quote = TEERuntime(mock=True).generate_attestation(
                bytes.fromhex(proof.content_hash())
            )
            cache[epoch_id] = proof
            logger.info("[Miner] Ran benchmark: %d/%d correct, $%.6f",
                        proof.n_correct, proof.n_total, proof.total_cost_usd)
        proof = cache[epoch_id]
        synapse.proof_hash = proof.content_hash()
        synapse.weights_hash = proof.weights_hash
        # No bundle store in-container: the validator reads the proof from the
        # shared in-process cache below rather than downloading it.
        synapse.proof_bundle_url = f"memory://{epoch_id}"
        return synapse

    subtensor = bt.Subtensor(network=subtensor_network)
    axon = bt.Axon(wallet=wallet, port=port)
    axon.attach(forward_fn=forward)
    axon.serve(netuid=netuid, subtensor=subtensor)
    axon.start()
    logger.info("[Miner] TEE axon serving on port %d", port)
    return axon, cache, head_data


def _local_prices(models):
    return {m: (1e-7 * (i + 1), 2e-7 * (i + 1)) for i, m in enumerate(models)}


def _local_hidden(pool, head_path):
    """Synthetic embeddings — the backbone is too heavy for a container demo.

    Deterministic given the head, so miner and validator agree.
    """
    with np.load(head_path, allow_pickle=False) as npz:
        dim = int(npz["W"].shape[1])
    rng = np.random.RandomState(7)
    h = rng.randn(len(pool), dim).astype(np.float32)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    return h


def run_epoch(validator_wallet, subtensor, netuid, pool, models, proof_cache,
              head_data, slice_size=50, explore_size=3):
    """Run one validator epoch against the real TEE verification path."""
    import bittensor as bt

    from fugal_subnet.benchmarks.slicer import (
        derive_nonce,
        epoch_id_for_block,
        epoch_index_for_block,
        select_slice,
    )
    from fugal_subnet.commit_reveal import commit_epoch, reveal_epoch
    from fugal_subnet.dedup import find_duplicates
    from fugal_subnet.exploration import expected_exploration
    from fugal_subnet.head_eval import HeadScore
    from fugal_subnet.protocol import FugalProofSynapse
    from fugal_subnet.reference_frame import (
        ReferenceFrame,
        accumulate_exploration,
        best_model,
        reference_cost,
    )
    from fugal_subnet.rewards import compute_weights
    from fugal_subnet.scoring import ScoringState, update_scores
    from fugal_subnet.tee.proof import compute_questions_hash
    from fugal_subnet.tee.verify import verify_proof

    metagraph = subtensor.metagraph(netuid)
    dendrite = bt.Dendrite(wallet=validator_wallet)
    logger.info("[Validator] Metagraph: %d neurons", metagraph.n)

    block = subtensor.get_current_block()
    block_hash = str(block)
    # Same helper the production validator uses — see slicer.epoch_id_for_block.
    epoch_id = epoch_id_for_block(epoch_index_for_block(block, 1))
    nonce = derive_nonce(epoch_id, block_hash)
    questions = select_slice(nonce, pool, slice_size)
    question_ids = [q["question_id"] for q in questions]
    explore_map = expected_exploration(
        nonce, pool, set(question_ids), models, explore_size,
    )
    gold = {q["question_id"]: q for q in pool}

    print(f"[Validator] Epoch {epoch_id}: {len(questions)} scored + "
          f"{len(explore_map)} exploration questions", flush=True)

    commitment = commit_epoch(epoch_id, questions, block_hash)
    print(f"[Validator] Committed: {commitment.commit_hash}", flush=True)

    synapse = FugalProofSynapse(epoch_id=epoch_id, nonce=nonce.hex())
    axons = metagraph.axons
    for ax in axons:
        if ax.port and ax.port > 0:
            ax.ip = "127.0.0.1"      # miner shares this container
    responses = dendrite.query(axons, synapse, timeout=60)
    print(f"[Validator] Got {len(responses)} responses", flush=True)

    verified, head_hashes, prices = {}, {}, _local_prices(models)
    for uid, resp in enumerate(responses):
        proof_hash = getattr(resp, "proof_hash", "")
        if not proof_hash:
            continue
        proof = proof_cache.get(epoch_id)
        if proof is None:
            print(f"  UID {uid}: no proof in bundle store", flush=True)
            continue

        result = verify_proof(
            proof, approved_measurements=set(),
            expected_questions_hash=compute_questions_hash(question_ids),
            expected_nonce=nonce.hex(), gold_answers=gold,
            expected_question_ids=set(question_ids),
            expected_exploration=explore_map,
            expected_weights_hash=proof.weights_hash,
            expected_proof_hash=proof_hash,
            head_bytes=head_data, mock=True,
        )
        if not result.valid:
            print(f"  UID {uid}: proof REJECTED — {result.reason}", flush=True)
            continue
        verified[uid] = proof
        head_hashes[uid] = proof.weights_hash
        print(f"  UID {uid}: proof verified ({proof.n_correct}/{proof.n_total} "
              f"correct, ${proof.scored_cost_usd:.6f})", flush=True)

    if not verified:
        print("[Validator] No valid proofs!", flush=True)
        return False

    frame = accumulate_exploration(ReferenceFrame(), [
        (r.routed_model, r.correct, r.prompt_tokens, r.completion_tokens)
        for uid in sorted(verified)
        for r in verified[uid].exploration_results
    ])
    ref_model, acc_best = best_model(frame, prices)
    print(f"[Validator] Reference model: {ref_model} (acc={acc_best:.3f})", flush=True)

    model_index = {m: i for i, m in enumerate(sorted(prices))}
    epoch_scores, routing = {}, {}
    for uid, proof in verified.items():
        scored = proof.scored_results
        ref = reference_cost(
            frame, prices, ref_model,
            prompt_tokens=sum(r.prompt_tokens for r in scored),
            n_questions=len(scored), default_completion_tokens=300.0,
        )
        epoch_scores[uid] = HeadScore(
            accuracy=proof.accuracy, cost_efficiency=0.0, kl_score=0.0,
            routing_decisions=np.array([], dtype=np.int32),
            correct_mask=np.array([r.correct for r in scored], dtype=bool),
            n_correct=proof.n_correct, n_scored=proof.n_total,
            total_head_cost=proof.scored_cost_usd, total_oracle_cost=ref,
        )
        routing[uid] = np.array(
            [model_index.get(r.routed_model, -1)
             for r in sorted(scored, key=lambda x: x.question_id)],
            dtype=np.int32,
        )

    dupes = find_duplicates(routing, {uid: uid for uid in verified})
    print(f"[Validator] Dedup: {dupes or '{}'}", flush=True)

    state = update_scores(
        ScoringState(), epoch_scores, head_hashes, acc_best=acc_best,
        hotkeys={uid: metagraph.hotkeys[uid] for uid in verified
                 if uid < len(metagraph.hotkeys)},
        n_questions=len(questions), pool_size=len(pool),
    )
    uids, weights = compute_weights(state.records, dedup_disqualified=dupes)
    for uid, rec in state.records.items():
        print(f"  UID {uid}: quality={rec.quality:.3f} thrift={rec.thrift:.3f} "
              f"score={rec.composite_score:.4f}", flush=True)
    print(f"[Validator] Weights: UIDs={uids} "
          f"weights={[f'{w:.4f}' for w in weights]}", flush=True)

    print("[Validator] Setting weights on chain...", flush=True)
    try:
        success, msg = subtensor.set_weights(
            wallet=validator_wallet, netuid=netuid,
            uids=uids, weights=weights,
            wait_for_inclusion=True, wait_for_finalization=False,
        )
    except Exception as e:
        success, msg = False, str(e)
    if success:
        print("[Validator] Weights set successfully!", flush=True)
        # An extrinsic reporting success only means it was included, not that
        # the chain holds what we meant. Read it back.
        from neurons.validator import confirm_weights_on_chain
        my_uid = (
            metagraph.hotkeys.index(validator_wallet.hotkey.ss58_address)
            if validator_wallet.hotkey.ss58_address in metagraph.hotkeys else -1
        )
        confirmed, detail = confirm_weights_on_chain(
            subtensor, netuid, my_uid, uids, weights,
        )
        print(f"[Validator] On-chain confirmation: {confirmed} ({detail})",
              flush=True)
        success = confirmed
    else:
        # This used to force success = True with the note "chain config issue —
        # pipeline itself completed OK", which meant the demo could report a
        # green run while no weights ever landed. A failed weight-set is a
        # failed epoch.
        print(f"[Validator] Weight-setting FAILED: {msg}", flush=True)
        success = False

    all_models = sorted(prices)
    matrix = np.zeros((len(questions), len(all_models)), dtype=np.int32)
    q_index = {qid: i for i, qid in enumerate(question_ids)}
    for proof in verified.values():
        for r in proof.scored_results:
            if r.question_id in q_index and r.routed_model in model_index:
                matrix[q_index[r.question_id], model_index[r.routed_model]] = int(r.correct)

    verified_reveal = reveal_epoch(
        epoch_id, questions, matrix, all_models,
        {m: prices[m][0] + prices[m][1] for m in all_models},
        head_hashes, {uid: routing[uid].tolist() for uid in routing},
        {uid: {"quality": state.records[uid].quality,
               "thrift": state.records[uid].thrift,
               "score": state.records[uid].composite_score}
         for uid in verified},
        dict(zip(uids, weights)),
    )
    if verified_reveal:
        logger.info("[Validator] Commit-reveal verified!")
    else:
        logger.error("[Validator] Commit-reveal FAILED!")

    return success


def main():
    logger.info("=" * 60)
    logger.info("FUGAL LOCAL TESTNET SETUP")
    logger.info("=" * 60)

    subtensor = wait_for_chain(CHAIN_ENDPOINT)

    logger.info("\n--- Step 1: Create subnet (using Alice's devnet funds) ---")
    netuid = create_subnet(subtensor)

    logger.info("\n--- Step 2: Create test wallets ---")
    wallets = create_wallets()
    val_wallet = wallets["fugal_validator"]
    miner_wallet = wallets["fugal_miner"]

    logger.info("\n--- Step 3: Fund test wallets from Alice ---")
    fund_wallet(subtensor, val_wallet, amount_tao=10000)
    fund_wallet(subtensor, miner_wallet, amount_tao=10000)

    logger.info("\n--- Step 4: Register wallets ---")
    register_wallet(subtensor, val_wallet, netuid)
    register_wallet(subtensor, miner_wallet, netuid)

    time.sleep(5)

    logger.info("\n--- Step 5: Train head ---")
    head_path = train_head()

    print("--- Step 6: Start TEE miner ---", flush=True)
    pool = _synthetic_pool()
    with np.load(head_path, allow_pickle=False) as npz:
        models = [str(m) for m in npz["models"]]
    # Pricier models answer more of the synthetic pool correctly, so the
    # reference frame has a real signal to find rather than noise.
    _SKILL.update({m: 0.25 + 0.6 * i / max(1, len(models) - 1)
                   for i, m in enumerate(models)})

    axon, proof_cache, head_data = start_miner(
        miner_wallet, CHAIN_ENDPOINT, netuid, head_path,
        pool=pool, models=models,
    )
    print("Miner started, waiting 5s...", flush=True)
    time.sleep(5)

    print("--- Step 7: Run validator epoch (TEE proof verification) ---", flush=True)
    success = run_epoch(val_wallet, subtensor, netuid, pool, models,
                        proof_cache, head_data)

    logger.info("\n--- Step 8: Cleanup ---")
    axon.stop()

    if success:
        logger.info("=" * 60)
        logger.info("LOCAL TESTNET EPOCH COMPLETE — WEIGHTS SET ON CHAIN")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("LOCAL TESTNET EPOCH FAILED")
        logger.error("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print(f"\n{'='*60}", flush=True)
        print(f"FATAL: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        print(f"{'='*60}", flush=True)
        sys.exit(1)
