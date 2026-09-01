#!/usr/bin/env python3
"""Run the gated v2 protocol on a disposable real local Subtensor chain.

This acceptance harness never has a live/paid mode. It creates local-only
manifest and model-registry overrides under ``--run-root``, five validators,
two miners, the real OCI grader launcher, finalized commitments, Axon report
exchange, exact reveal verification, and finalized weights. Packaged consensus
files and public network activation remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import launch_testnet as local_chain  # noqa: E402

VALIDATOR_COUNT = 5
MINER_COUNT = 2
VALIDATOR_PORT_BASE = 8200
MINER_PORT_BASE = 8300
WORKER_IMAGE = (
    "sha256:57410e04114488e5439518597d9b51dff024a27a4f09e42d3516840e38e968d6"
)
ACTIVE_MODEL_COUNT = 2
EPOCH_BLOCKS = 1200
PRECOMMIT_OFFSET = 600
REPORT_OFFSET = 900
SLICE_SIZE = 6
ACCEPTED_EPOCH = "v2-000000000000"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(payload)
    path.chmod(0o600)


def _local_keypair(name: str, role: str):
    import bittensor as bt

    return bt.Keypair.create_from_uri(f"//FugalV2Local//{name}//{role}")


def _create_wallet(wallet_root: Path, name: str):
    import bittensor as bt

    wallet = bt.Wallet(name=name, hotkey="default", path=str(wallet_root))
    if not wallet.coldkey_file.exists_on_device():
        coldkey = _local_keypair(name, "cold")
        wallet.set_coldkey(coldkey, encrypt=False, overwrite=False)
        wallet.set_coldkeypub(coldkey, encrypt=False, overwrite=False)
    if not wallet.hotkey_file.exists_on_device():
        wallet.set_hotkey(_local_keypair(name, "hot"), overwrite=False)
    return wallet


def _balance_tao(balance) -> float:
    text = str(balance).replace("τ", "").replace("TAO", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _fund_wallets(subtensor, alice, wallets: list, amount: int = 5_000) -> None:
    import bittensor as bt

    for wallet in wallets:
        address = wallet.coldkeypub.ss58_address
        if _balance_tao(subtensor.get_balance(address)) >= amount:
            continue
        response = subtensor.transfer(
            wallet=alice,
            destination_ss58=address,
            amount=bt.Balance.from_tao(amount),
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        success, message = response
        if not success:
            raise RuntimeError(f"local funding failed for {wallet.name}: {message}")


def _create_fresh_subnet(subtensor, owner) -> int:
    before = set(subtensor.get_all_subnets_netuid())
    response = subtensor.register_subnet(
        wallet=owner,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    success, message = response
    if not success:
        raise RuntimeError(f"fresh local subnet creation failed: {message}")
    after = set(subtensor.get_all_subnets_netuid())
    created = after - before
    if len(created) != 1:
        raise RuntimeError("fresh local subnet netuid is ambiguous")
    netuid = created.pop()
    response = subtensor.start_call(
        wallet=owner,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    success, message = response
    if not success:
        raise RuntimeError(f"fresh local subnet activation failed: {message}")
    return netuid


def _stake_validators(subtensor, netuid: int, validators: list) -> None:
    import bittensor as bt

    for wallet in validators:
        response = subtensor.add_stake(
            wallet=wallet,
            netuid=netuid,
            hotkey_ss58=wallet.hotkey.ss58_address,
            amount=bt.Balance.from_tao(1_000),
            safe_staking=False,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        success, message = response
        if not success:
            raise RuntimeError(f"local validator stake failed: {message}")


def _wait_for_validator_permits(subtensor, netuid: int, validators: list) -> None:
    expected = {wallet.hotkey.ss58_address for wallet in validators}
    deadline = time.time() + 180
    while time.time() < deadline:
        metagraph = subtensor.metagraph(netuid)
        permitted = {
            str(hotkey)
            for hotkey, permit in zip(metagraph.hotkeys, metagraph.validator_permit)
            if bool(permit)
        }
        if expected <= permitted:
            return
        time.sleep(2)
    raise RuntimeError("five local validator permits were not assigned")


def _write_local_overrides(
    run_root: Path,
    *,
    activation_block: int,
    worker_image: str,
) -> tuple[Path, Path, tuple[str, ...]]:
    registry = json.loads(
        importlib.resources.files("fugal_subnet")
        .joinpath("model-registry-v2.json")
        .read_text(encoding="utf-8")
    )
    for index, model in enumerate(registry["models"]):
        enabled = index < ACTIVE_MODEL_COUNT
        model["enabled"] = enabled
        model["review_status"] = "approved" if enabled else "local_test_disabled"
    registry["registry_id"] = "fugal-models-v2-local-acceptance"
    registry["status"] = "local_mock_only"
    registry_bytes = _canonical_json(registry)
    registry_path = run_root / "consensus" / "model-registry-v2-local.json"
    _write_private(registry_path, registry_bytes)
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    active_models = tuple(
        model["id"] for model in registry["models"] if model["enabled"]
    )

    manifest = json.loads(
        importlib.resources.files("fugal_subnet")
        .joinpath("consensus-manifest.json")
        .read_text(encoding="utf-8")
    )
    v2 = next(item for item in manifest["protocols"] if item["id"] == "v2")
    v2.update({
        "status": "local_mock_acceptance",
        "enabled": True,
        "complete": True,
        "activation_blocks": {"local": activation_block, "test": None, "finney": None},
    })
    consensus = v2["consensus"]
    consensus["benchmarks"]["rollout_blockers"] = []
    consensus["model_registry"].update({
        "registry_id": registry["registry_id"],
        "canonical_sha256": registry_hash,
        "resource_sha256": registry_hash,
        "active_models": list(active_models),
        "status": "local_mock_only",
    })
    consensus["prices"]["snapshot_sha256"] = registry_hash
    consensus["worker"].update({
        "image_digest": worker_image,
        "status": "local_mock_verified",
    })
    consensus["committee"].update({
        "epoch_blocks": EPOCH_BLOCKS,
        "precommit_deadline_offset_blocks": PRECOMMIT_OFFSET,
        "report_deadline_offset_blocks": REPORT_OFFSET,
        "slice_size": SLICE_SIZE,
        "api_concurrency": 2,
    })
    manifest_path = run_root / "consensus" / "consensus-manifest-local.json"
    _write_private(manifest_path, _canonical_json(manifest))
    return manifest_path, registry_path, active_models


def _write_heads(run_root: Path, model_ids: tuple[str, ...]) -> list[Path]:
    from fugal_subnet.v2.backbone import HIDDEN_DIM

    paths = []
    for index in range(MINER_COUNT):
        generator = np.random.default_rng(10_000 + index)
        path = run_root / "heads" / f"miner-{index}.npz"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        np.savez(
            path,
            W=generator.normal(0, 0.01, (len(model_ids), HIDDEN_DIM)).astype(np.float32),
            b=generator.normal(0, 0.01, len(model_ids)).astype(np.float32),
            models=np.asarray(model_ids, dtype="U192"),
        )
        path.chmod(0o600)
        paths.append(path)
    return paths


def _advertise_axons(
    subtensor,
    netuid: int,
    wallets: list,
    external_ip: str,
    port_base: int,
) -> None:
    import bittensor as bt

    for index, wallet in enumerate(wallets):
        axon = bt.Axon(
            wallet=wallet,
            port=port_base + index,
            external_ip=external_ip,
        )
        axon.serve(netuid=netuid, subtensor=subtensor)


def _commit_miner_heads(
    subtensor,
    netuid: int,
    miners: list,
    heads: list[Path],
    *,
    activation_block: int,
) -> None:
    from fugal_subnet.v2.commitments import (
        HEAD_COMMITMENT_ID,
        finalized_block_number,
        set_commitment_finalized,
    )

    for wallet, head in zip(miners, heads):
        artifact_hash = hashlib.sha256(head.read_bytes()).hexdigest()
        set_commitment_finalized(
            subtensor,
            wallet,
            netuid=netuid,
            namespace="head",
            epoch_id=HEAD_COMMITMENT_ID,
            artifact_hash=artifact_hash,
        )
    if finalized_block_number(subtensor) >= activation_block:
        raise RuntimeError("local miner head commitments missed the activation boundary")


def _base_process_env(
    manifest_path: Path,
    registry_path: Path,
    external_ip: str,
    backbone_lock: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    env.pop("FUGAL_API_KEY", None)
    env.update({
        "FUGAL_CONSENSUS_MANIFEST": str(manifest_path),
        "FUGAL_MODEL_REGISTRY": str(registry_path),
        "FUGAL_AXON_IP": external_ip,
        "FUGAL_LOCAL_BACKBONE_LOCK": str(backbone_lock),
        "FUGAL_LOCAL_BACKBONE_CACHE": str(backbone_lock.parent / "backbone-cache"),
        # The full canonical pool and pinned backbone must already have been
        # materialized by the prerequisite checks. Avoid five redundant Hub
        # probes consuming the local epoch's precommit window.
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    return env


def _start_process(command: list[str], env: dict[str, str], log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    return process, handle


def _prewarm_local_backbone(
    run_root: Path,
    env: dict[str, str],
    *,
    boundary_hash: str,
    lock_path: Path,
) -> None:
    """Materialize local-only golden and epoch embeddings before peak load."""
    offline_keys = ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in offline_keys}
    os.environ.update({key: "1" for key in offline_keys})
    try:
        from fugal_subnet.benchmarks.slicer import derive_nonce, select_slice
        from fugal_subnet.v2.backbone import GOLDEN_PROMPTS
        from fugal_subnet.v2.benchmarks import load_pool

        pool = load_pool()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    nonce = derive_nonce(ACCEPTED_EPOCH, boundary_hash)
    questions = select_slice(nonce, pool, SLICE_SIZE)
    prompts = [question.get("prompt") for question in questions]
    if len(prompts) != SLICE_SIZE or any(not isinstance(item, str) for item in prompts):
        raise RuntimeError("local acceptance slice prompts are invalid")
    cache_root = lock_path.parent / "backbone-cache"
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    child = (
        "import json,sys; from pathlib import Path; "
        "from neurons.validator_v2 import _compute_local_serialized; "
        "prompts=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); "
        "_compute_local_serialized(prompts,Path(sys.argv[2]),Path(sys.argv[3]))"
    )
    for label, values in (("golden", list(GOLDEN_PROMPTS)), ("epoch", prompts)):
        prompt_path = run_root / "consensus" / f"prewarm-{label}.json"
        _write_private(prompt_path, _canonical_json(values))
        print(f"  Prewarming pinned CPU backbone cache ({label})...", flush=True)
        result = subprocess.run(
            [sys.executable, "-c", child, str(prompt_path), str(lock_path), str(cache_root)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"local backbone {label} prewarm failed: {result.stderr[-500:]}"
            )


def _stop_processes(processes: list[tuple[subprocess.Popen, IO[str]]]) -> None:
    for process, _handle in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.time() + 15
    for process, handle in processes:
        remaining = max(0.1, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        handle.close()


def _wait_for_socket(socket_path: Path, process: subprocess.Popen) -> None:
    deadline = time.time() + 30
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("grader launcher exited before becoming ready")
        if socket_path.exists():
            from fugal_subnet.sandbox.client import GradingClient

            if GradingClient(socket_path).health():
                return
        time.sleep(0.25)
    raise RuntimeError("grader launcher did not become ready")


def _wait_until_block(subtensor, target: int, timeout: int = 300) -> None:
    from fugal_subnet.v2.commitments import finalized_block_number

    deadline = time.time() + timeout
    while time.time() < deadline:
        if finalized_block_number(subtensor) >= target:
            return
        time.sleep(1)
    raise RuntimeError(f"local chain did not reach block {target}")


def _wait_for_acceptance(
    run_root: Path,
    processes: list[tuple[subprocess.Popen, IO[str]]],
    epoch_id: str,
    timeout: int,
) -> list[Path]:
    reveals = [
        run_root / "validators" / f"validator-{index}" / "epochs" / epoch_id / "reveal.json"
        for index in range(VALIDATOR_COUNT)
    ]
    deadline = time.time() + timeout
    next_progress = 0.0
    while time.time() < deadline:
        exited = [process.returncode for process, _ in processes if process.poll() is not None]
        if exited:
            raise RuntimeError(f"local v2 process exited early: {exited}")
        ready = sum(path.exists() for path in reveals)
        if ready == len(reveals):
            return reveals
        if time.time() >= next_progress:
            remaining = max(0, int(deadline - time.time()))
            print(
                f"  Waiting for {epoch_id}: {ready}/{len(reveals)} reveals, "
                f"{remaining}s remaining",
                flush=True,
            )
            next_progress = time.time() + 30
        time.sleep(2)
    raise RuntimeError(f"timed out waiting for {epoch_id} reveals")


def _verify_acceptance(
    subtensor,
    netuid: int,
    validators: list,
    reveals: list[Path],
    grader_socket: Path,
    env: dict[str, str],
) -> None:
    hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in reveals}
    if len(hashes) != 1:
        raise RuntimeError("five validator reveal artifacts are not byte-identical")
    representative = json.loads(reveals[0].read_text(encoding="utf-8"))
    if len(representative["committee"]) != VALIDATOR_COUNT:
        raise RuntimeError("local acceptance committee does not contain five validators")
    if len(representative["builder_reports"]) < 3:
        raise RuntimeError("local acceptance reveal lacks report quorum")
    if not representative["set_weights"] or not representative["weights"]:
        raise RuntimeError("local acceptance epoch did not derive weights")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fugal_subnet.verify_epoch",
            "--v2-reveal",
            str(reveals[0]),
            "--network",
            local_chain.LOCAL_ENDPOINT,
            "--netuid",
            str(netuid),
            "--grader-socket",
            str(grader_socket),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"offline reveal verification failed: {result.stderr[-500:]}")
    metagraph = subtensor.metagraph(netuid)
    for wallet in validators:
        uid = [str(item) for item in metagraph.hotkeys].index(wallet.hotkey.ss58_address)
        row = np.asarray(metagraph.W[uid], dtype=np.float64)
        if not np.all(np.isfinite(row)) or float(row.sum()) <= 0:
            raise RuntimeError(f"validator UID {uid} has no finalized weight row")
    for index in range(VALIDATOR_COUNT):
        journal_path = (
            reveals[index].parents[2]
            / "journals"
            / reveals[index].parent.name
            / "journal.json"
        )
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal["status"] != "complete" or journal["spend"]["actual_usd"] != "0":
            raise RuntimeError("local acceptance journal is incomplete or nonzero-spend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("results/v2-local-acceptance"),
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker-image", default=WORKER_IMAGE)
    parser.add_argument("--keep-chain", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    wallet_root = run_root / "wallets"
    processes: list[tuple[subprocess.Popen, IO[str]]] = []
    chain_created = False
    subtensor = None
    try:
        if not local_chain.check_docker():
            return 1
        ready, chain_created = local_chain.start_local_chain(archive_history=True)
        if not ready:
            return 1

        import bittensor as bt

        base_wallets = local_chain.setup_wallets(wallet_root)
        if not local_chain.fund_wallets(base_wallets):
            raise RuntimeError("base local wallets could not be funded")
        bootstrap_subtensor = bt.Subtensor(network=local_chain.LOCAL_ENDPOINT)
        netuid = _create_fresh_subnet(
            bootstrap_subtensor,
            base_wallets[local_chain.OWNER_WALLET],
        )
        bootstrap_subtensor.close()
        validators = [
            _create_wallet(wallet_root, f"fugal_v2_validator_{index}")
            for index in range(VALIDATOR_COUNT)
        ]
        miners = [
            _create_wallet(wallet_root, f"fugal_v2_miner_{index}")
            for index in range(MINER_COUNT)
        ]
        subtensor = bt.Subtensor(network=local_chain.LOCAL_ENDPOINT)
        _fund_wallets(subtensor, base_wallets["alice"], validators + miners)
        for index, wallet in enumerate(validators):
            if local_chain.register_neuron(wallet, netuid, f"v2 validator {index}") < 0:
                raise RuntimeError("local v2 validator registration failed")
        for index, wallet in enumerate(miners):
            if local_chain.register_neuron(wallet, netuid, f"v2 miner {index}") < 0:
                raise RuntimeError("local v2 miner registration failed")
        _stake_validators(subtensor, netuid, validators)
        _wait_for_validator_permits(subtensor, netuid, validators)

        external_ip = local_chain._get_local_ip()
        _advertise_axons(
            subtensor, netuid, validators, external_ip, VALIDATOR_PORT_BASE
        )
        _advertise_axons(subtensor, netuid, miners, external_ip, MINER_PORT_BASE)
        from fugal_subnet.v2.commitments import finalized_block_number

        activation_block = finalized_block_number(subtensor) + 120
        manifest_path, registry_path, active_models = _write_local_overrides(
            run_root,
            activation_block=activation_block,
            worker_image=args.worker_image,
        )
        heads = _write_heads(run_root, active_models)
        _commit_miner_heads(
            subtensor,
            netuid,
            miners,
            heads,
            activation_block=activation_block,
        )
        env = _base_process_env(
            manifest_path,
            registry_path,
            external_ip,
            run_root / "backbone.lock",
        )

        grader_socket = run_root / "grader" / "grader.sock"
        grader_process = _start_process(
            [
                sys.executable,
                "-m",
                "fugal_subnet.sandbox.launcher",
                "--socket",
                str(grader_socket),
                "--image",
                args.worker_image,
                "--allowed-uid",
                str(os.getuid()),
                "--max-concurrency",
                str(VALIDATOR_COUNT),
            ],
            env,
            run_root / "logs" / "grader.log",
        )
        processes.append(grader_process)
        _wait_for_socket(grader_socket, grader_process[0])
        _wait_until_block(subtensor, activation_block)
        boundary_hash = str(subtensor.get_block_hash(activation_block)).removeprefix("0x")
        _prewarm_local_backbone(
            run_root,
            env,
            boundary_hash=boundary_hash,
            lock_path=run_root / "backbone.lock",
        )

        for index, (wallet, head) in enumerate(zip(miners, heads)):
            process = _start_process(
                [
                    sys.executable,
                    "neurons/miner.py",
                    "--network",
                    local_chain.LOCAL_ENDPOINT,
                    "--netuid",
                    str(netuid),
                    "--coldkey",
                    wallet.name,
                    "--hotkey",
                    "default",
                    "--wallet-path",
                    str(wallet_root),
                    "--head-path",
                    str(head),
                    "--port",
                    str(MINER_PORT_BASE + index),
                ],
                env,
                run_root / "logs" / f"miner-{index}.log",
            )
            processes.append(process)
        time.sleep(8)

        for index, wallet in enumerate(validators):
            state_root = run_root / "validators" / f"validator-{index}"
            process = _start_process(
                [
                    sys.executable,
                    "neurons/validator_v2.py",
                    "--network",
                    local_chain.LOCAL_ENDPOINT,
                    "--netuid",
                    str(netuid),
                    "--coldkey",
                    wallet.name,
                    "--hotkey",
                    "default",
                    "--wallet-path",
                    str(wallet_root),
                    "--grader-socket",
                    str(grader_socket),
                    "--report-port",
                    str(VALIDATOR_PORT_BASE + index),
                    "--state-root",
                    str(state_root),
                    "--mock",
                ],
                env,
                run_root / "logs" / f"validator-{index}.log",
            )
            processes.append(process)

        reveals = _wait_for_acceptance(
            run_root,
            processes[1:],
            ACCEPTED_EPOCH,
            args.timeout,
        )
        _verify_acceptance(
            subtensor,
            netuid,
            validators,
            reveals,
            grader_socket,
            env,
        )
        print(
            "v2 local-chain acceptance passed: five validators, two miners, "
            "quorum reports, verified reveals, finalized weights, $0 API spend",
            flush=True,
        )
        return 0
    except Exception as exc:
        print(f"v2 local-chain acceptance failed: {exc}", file=sys.stderr, flush=True)
        if run_root.exists():
            print(f"diagnostic logs retained under {run_root}", file=sys.stderr, flush=True)
        return 1
    finally:
        _stop_processes(processes)
        if subtensor is not None:
            subtensor.close()
        if chain_created and not args.keep_chain:
            local_chain.stop_local_chain()


if __name__ == "__main__":
    raise SystemExit(main())
