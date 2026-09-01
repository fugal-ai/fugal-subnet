#!/usr/bin/env python3
"""Fugal testnet launch — end-to-end setup and multi-epoch test run.

Automates the full deployment against a local subtensor chain:
1. Starts a local subtensor via Docker (digest-pinned official image)
2. Creates wallets, funds them from Alice's pre-funded account
3. Creates and activates a subnet
4. Registers validator and miner neurons
5. Trains a reference head
6. Starts the miner in background
7. Runs N validator epochs (mock by default; paid only with explicit --live)
8. Reports results and costs

Usage:
    # Full run with mock API (no OpenRouter spend, no real cost):
    python scripts/launch_testnet.py --epochs 3

    # [PAID ~$30/epoch maximum] Explicit live mode and budget are required:
    OPENROUTER_API_KEY=sk-or-... python scripts/launch_testnet.py \
        --live --epoch-budget 30 --epochs 2

    # Dry run — check setup, don't create anything:
    python scripts/launch_testnet.py --dry-run

    # Skip chain/wallet setup (already running):
    python scripts/launch_testnet.py --skip-setup --netuid 2 --mock --epochs 3

Prerequisites:
    - Docker installed and running
    - OpenRouter API key (for real API mode only)
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fugal.launch")

LOCAL_ENDPOINT = "ws://127.0.0.1:9944"
DOCKER_IMAGE = (
    "ghcr.io/raofoundation/subtensor-localnet:v432"
    "@sha256:2354832a7c45ceb35703d64bd6f9477439f9a966e34a1d460965b8faf19439e2"
)
DOCKER_CONTAINER = "fugal_local_chain"
CHAIN_HISTORY_LABEL = "fugal.history"
OWNER_WALLET = "fugal_owner"
VALIDATOR_WALLET = "fugal_validator"
MINER_WALLET = "fugal_miner"
MINER_PORT = 8091

ALICE_SEED = "0xe5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a"


def _get_local_ip() -> str:
    """Get a non-loopback local IP that the chain will accept.

    The subtensor chain rejects 127.0.0.1 in serve_axon validation,
    so we need the machine's actual LAN/WSL2 IP for local testing.
    """
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "172.17.0.1"


def section(title: str):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}\n", flush=True)


def run_btcli(args: list[str], check: bool = True,
              timeout: int = 120) -> subprocess.CompletedProcess:
    cmd = ["btcli"] + args
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        logger.error("btcli failed: %s", (result.stderr or result.stdout).strip()[:300])
    return result


# ── Step 0: Docker chain ──

def check_docker():
    """Verify Docker is available."""
    if not shutil.which("docker"):
        print("ERROR: Docker not found. Install Docker first.", flush=True)
        return False
    result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    if result.returncode != 0:
        print("ERROR: Docker is not running. Start the Docker daemon.", flush=True)
        return False
    return True


def inspect_chain_container() -> tuple[bool, str, bool]:
    """Return (exists, configured image, running) for the named container."""
    result = subprocess.run(
        [
            "docker", "inspect", DOCKER_CONTAINER,
            "--format", "{{.Config.Image}}|{{.State.Running}}",
        ],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return False, "", False
    image, _, running = result.stdout.strip().partition("|")
    return True, image, running.lower() == "true"


def start_local_chain(*, archive_history: bool = False) -> tuple[bool, bool]:
    """Start the pinned local chain; return (ready, created_by_this_run)."""
    exists, configured_image, running = inspect_chain_container()
    if exists:
        if configured_image != DOCKER_IMAGE:
            print(
                f"  Refusing container '{DOCKER_CONTAINER}': image is "
                f"{configured_image!r}, expected the pinned {DOCKER_IMAGE!r}.",
                flush=True,
            )
            return False, False
        if not running:
            print(
                f"  Container '{DOCKER_CONTAINER}' exists but is stopped; "
                "start it explicitly or remove it after reviewing its state.",
                flush=True,
            )
            return False, False
        if archive_history:
            history = subprocess.run(
                [
                    "docker", "inspect", DOCKER_CONTAINER, "--format",
                    f'{{{{index .Config.Labels "{CHAIN_HISTORY_LABEL}"}}}}',
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if history.returncode != 0 or history.stdout.strip() != "archive":
                print(
                    f"  Refusing container '{DOCKER_CONTAINER}': v2 acceptance "
                    "requires archive history mode.",
                    flush=True,
                )
                return False, False
        print(f"  Reusing pinned container '{DOCKER_CONTAINER}'", flush=True)
        return wait_for_chain(), False

    print(f"  Pulling {DOCKER_IMAGE}...", flush=True)
    pull = subprocess.run(
        ["docker", "pull", DOCKER_IMAGE], capture_output=True, text=True, timeout=300,
    )
    if pull.returncode != 0:
        print(f"  Docker pull failed: {pull.stderr.strip()[:200]}", flush=True)
        return False, False

    print("  Starting local chain...", flush=True)
    history_mode = "archive" if archive_history else "default"
    command = [
        "docker", "run", "-d",
        "--name", DOCKER_CONTAINER,
        "--label", f"{CHAIN_HISTORY_LABEL}={history_mode}",
        "-p", "9944:9944", "-p", "9945:9945",
    ]
    if archive_history:
        wrapper = Path(__file__).with_name("localnet_archive_wrapper.sh").resolve()
        command.extend([
            "--volume", f"{wrapper}:/fugal/localnet_archive_wrapper.sh:ro",
            "--entrypoint", "/bin/bash",
            DOCKER_IMAGE,
            "/fugal/localnet_archive_wrapper.sh",
            "True",
        ])
    else:
        command.append(DOCKER_IMAGE)
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print(f"  Docker run failed: {result.stderr.strip()[:200]}", flush=True)
        return False, False

    print(f"  Container started: {result.stdout.strip()[:12]}", flush=True)
    ready = wait_for_chain()
    if not ready:
        stop_local_chain()
    return ready, ready


def wait_for_chain(timeout: int = 90) -> bool:
    """Wait for the local chain to be ready."""
    print(f"  Waiting for chain at {LOCAL_ENDPOINT}...", flush=True)
    import bittensor as bt
    start = time.time()
    while time.time() - start < timeout:
        try:
            sub = bt.Subtensor(network=LOCAL_ENDPOINT)
            block = sub.get_current_block()
            print(f"  Chain live at block {block}", flush=True)
            return True
        except Exception:
            time.sleep(2)
    print(f"  Chain not ready after {timeout}s", flush=True)
    return False


def stop_local_chain():
    """Stop and remove the local chain container."""
    subprocess.run(
        ["docker", "rm", "-f", DOCKER_CONTAINER],
        capture_output=True, timeout=15,
    )


# ── Step 1: Wallets ──

def setup_wallets(wallet_root: Path, dry_run: bool = False) -> dict:
    """Create deterministic disposable local wallets without printing mnemonics."""
    import bittensor as bt

    def local_keypair(name: str, role: str):
        # These public derivation URIs are intentionally local-testnet-only and
        # must never be reused on a public network.
        return bt.Keypair.create_from_uri(f"//FugalLocal//{name}//{role}")

    wallet_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    wallet_root.chmod(0o700)
    wallets = {}
    for name in [OWNER_WALLET, VALIDATOR_WALLET, MINER_WALLET]:
        wallet = bt.Wallet(name=name, hotkey="default", path=str(wallet_root))
        if wallet.coldkey_file.exists_on_device():
            print(f"  Wallet '{name}' exists: {wallet.coldkeypub.ss58_address}", flush=True)
            if not wallet.hotkey_file.exists_on_device() and not dry_run:
                wallet.set_hotkey(local_keypair(name, "hot"), overwrite=False)
        elif dry_run:
            print(f"  Wallet '{name}' does not exist (dry run — skipping)", flush=True)
            wallets[name] = None
            continue
        else:
            print(f"  Creating disposable local wallet '{name}'...", flush=True)
            coldkey = local_keypair(name, "cold")
            wallet.set_coldkey(
                coldkey, encrypt=False, overwrite=False,
            )
            wallet.set_coldkeypub(coldkey, encrypt=False, overwrite=False)
            wallet.set_hotkey(local_keypair(name, "hot"), overwrite=False)
            print(f"    Created: {wallet.coldkeypub.ss58_address}", flush=True)
        wallets[name] = wallet

    if not dry_run:
        alice = bt.Wallet(name="alice", hotkey="default", path=str(wallet_root))
        if not alice.coldkey_file.exists_on_device():
            print("  Regenerating Alice devnet wallet...", flush=True)
            alice.regenerate_coldkey(
                seed=ALICE_SEED, use_password=False, overwrite=False,
            )
        if not alice.hotkey_file.exists_on_device():
            alice.set_hotkey(local_keypair("alice", "hot"), overwrite=False)
        wallets["alice"] = alice

    return wallets


# ── Step 2: Fund wallets ──

def fund_wallets(wallets: dict):
    """Transfer TAO from Alice to owner/validator/miner wallets."""
    import bittensor as bt
    subtensor = bt.Subtensor(network=LOCAL_ENDPOINT)

    alice = wallets["alice"]
    alice_bal = subtensor.get_balance(alice.coldkeypub.ss58_address)
    print(f"  Alice balance: {alice_bal}", flush=True)

    for name, amount in [(OWNER_WALLET, 100_000), (VALIDATOR_WALLET, 10_000),
                         (MINER_WALLET, 10_000)]:
        wallet = wallets[name]
        addr = wallet.coldkeypub.ss58_address

        current = subtensor.get_balance(addr)
        bal_str = str(current).replace("τ", "").replace("TAO", "").replace(",", "").strip()
        try:
            bal_float = float(bal_str)
        except ValueError:
            bal_float = 0.0

        if bal_float >= amount:
            print(f"  {name}: already funded ({current})", flush=True)
            continue

        print(f"  Funding {name} with {amount:,} TAO...", flush=True)
        try:
            response = subtensor.transfer(
                wallet=alice,
                destination_ss58=addr,
                amount=bt.Balance.from_tao(amount),
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
            success, msg = response
            if success:
                new_bal = subtensor.get_balance(addr)
                print(f"    {name}: {new_bal}", flush=True)
            else:
                print(f"    Transfer failed: {msg}", flush=True)
                return False
        except Exception as e:
            print(f"    Transfer failed: {e}", flush=True)
            return False

    return True


# ── Step 3: Create subnet ──

def create_subnet(wallets: dict, wallet_root: Path) -> int:
    """Create a subnet and activate it. Returns the netuid."""
    import bittensor as bt
    subtensor = bt.Subtensor(network=LOCAL_ENDPOINT)
    owner = wallets[OWNER_WALLET]

    netuids_before = set(subtensor.get_all_subnets_netuid())
    print(f"  Existing netuids: {sorted(netuids_before)}", flush=True)

    if len(netuids_before) > 1:
        netuid = max(netuids_before)
        print(f"  Subnet {netuid} already exists, reusing", flush=True)
        return netuid

    print("  Creating subnet via SDK...", flush=True)
    try:
        response = subtensor.register_subnet(
            wallet=owner,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        success, msg = response
        if success:
            print("  Subnet created successfully", flush=True)
        else:
            print(f"  register_subnet returned: {msg}", flush=True)
    except Exception as e:
        print(f"  SDK register_subnet failed: {e}", flush=True)
        print("  Trying btcli...", flush=True)
        run_btcli([
            "tx", "register-subnet",
            "--wallet-name", owner.name,
            "--wallet-path", str(wallet_root),
            "--hotkey", "default",
            "--network", LOCAL_ENDPOINT,
            "-y",
        ], check=False, timeout=60)

    time.sleep(3)
    netuids_after = set(subtensor.get_all_subnets_netuid())
    new_netuids = netuids_after - netuids_before
    if new_netuids:
        netuid = max(new_netuids)
        print(f"  Subnet created: netuid {netuid}", flush=True)
    elif len(netuids_after) > 1:
        netuid = max(netuids_after)
        print(f"  Using existing netuid {netuid}", flush=True)
    else:
        print("  No subnet found after creation!", flush=True)
        return -1

    print(f"  Activating subnet {netuid}...", flush=True)
    try:
        response = subtensor.start_call(
            wallet=owner,
            netuid=netuid,
            wait_for_inclusion=True,
        )
        success, msg = response
        if success:
            print(f"  Subnet {netuid} activated", flush=True)
        else:
            print(f"  start_call: {msg} (may not be required on this chain version)", flush=True)
    except Exception as e:
        print(f"  start_call: {e} (may not be required on this chain version)", flush=True)

    return netuid


# ── Step 4: Register ──

def register_neuron(wallet, netuid: int, role: str) -> int:
    """Register a wallet on the subnet via burned registration."""
    import bittensor as bt
    subtensor = bt.Subtensor(network=LOCAL_ENDPOINT)
    metagraph = subtensor.metagraph(netuid)

    if wallet.hotkey.ss58_address in metagraph.hotkeys:
        uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        print(f"  {role} already registered as UID {uid}", flush=True)
        return uid

    print(f"  Registering {role} on netuid {netuid}...", flush=True)
    try:
        response = subtensor.burned_register(
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        success, msg = response
        if not success:
            print(f"  burned_register: {msg}", flush=True)
    except Exception as e:
        print(f"  burned_register failed: {e}, trying register...", flush=True)
        try:
            subtensor.register(wallet=wallet, netuid=netuid)
        except Exception as e2:
            print(f"  Registration failed: {e2}", flush=True)
            return -1

    time.sleep(2)
    metagraph = subtensor.metagraph(netuid)
    if wallet.hotkey.ss58_address in metagraph.hotkeys:
        uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        print(f"  {role} registered as UID {uid}", flush=True)
        return uid
    else:
        print(f"  {role} not found in metagraph after registration", flush=True)
        return -1


# ── Step 5: Train head ──

def train_head(output_path: str) -> str:
    """Train a reference head using the full training script."""
    if os.path.exists(output_path):
        print(f"  Head already exists at {output_path}, reusing", flush=True)
        return output_path

    print("  Training reference head...", flush=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    models = [
        "deepseek/deepseek-v4-flash",
        "meta-llama/llama-4-maverick",
        "openai/gpt-5.4-nano",
    ]

    result = subprocess.run([
        sys.executable, "scripts/train_head.py",
        "--synthetic", "--n-questions", "300",
        "--models", ",".join(models),
        "--output", output_path,
        "--epochs", "100",
    ], capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  Training failed: {result.stderr[-500:]}", flush=True)
        raise RuntimeError("Head training failed")

    size = os.path.getsize(output_path)
    print(f"  Head trained: {output_path} ({size} bytes)", flush=True)
    return output_path


# ── Step 6: Start miner ──

def start_miner_process(wallet_name: str, wallet_root: Path, netuid: int,
                        head_path: str) -> subprocess.Popen:
    """Start the miner as a background process."""
    cmd = [
        sys.executable, "neurons/miner.py",
        "--network", LOCAL_ENDPOINT,
        "--netuid", str(netuid),
        "--coldkey", wallet_name,
        "--hotkey", "default",
        "--wallet-path", str(wallet_root),
        "--head-path", head_path,
        "--port", str(MINER_PORT),
    ]
    env = os.environ.copy()
    local_ip = _get_local_ip()
    env["FUGAL_AXON_IP"] = local_ip

    print(f"  Starting miner on port {MINER_PORT} (axon IP={local_ip})...", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    for wait in range(6):
        time.sleep(5)
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            print(f"  Miner exited unexpectedly: {output[-500:]}", flush=True)
            raise RuntimeError("Miner failed to start")

    print(f"  Miner running (PID {proc.pid})", flush=True)

    import bittensor as bt
    sub = bt.Subtensor(network=LOCAL_ENDPOINT)
    mg = sub.metagraph(netuid)
    for uid in range(mg.n):
        ax = mg.axons[uid]
        print(f"    UID {uid}: {ax.ip}:{ax.port}", flush=True)

    return proc


def wait_for_next_epoch_boundary(timeout: int = 120, minimum_blocks: int = 0):
    """Block until the chain crosses the next epoch boundary, so the miner's
    on-chain head commitment predates the boundary the validator will use."""
    import bittensor as bt
    bpe = max(1, int(os.getenv("FUGAL_EPOCH_INTERVAL", LOCAL_EPOCH_INTERVAL)) // 12)
    sub = bt.Subtensor(network=LOCAL_ENDPOINT)
    start_block = sub.get_current_block()
    target = max(
        ((start_block // bpe) + 1) * bpe,
        start_block + minimum_blocks,
    )
    print(f"  Waiting for epoch boundary (block {target}, now {start_block})...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if sub.get_current_block() >= target:
            print(f"  Boundary reached (block {sub.get_current_block()})", flush=True)
            return
        time.sleep(2)
    print("  WARNING: epoch boundary not reached in time — continuing anyway", flush=True)


# ── Step 7: Run validator epochs ──

# Short epochs for local testing: 24s = 2 blocks per epoch at 12s block time,
# so consecutive --once runs land on distinct epoch boundaries.
LOCAL_EPOCH_INTERVAL = "24"
LOCAL_WEIGHT_RATE_LIMIT_BLOCKS = 100


def run_validator_epoch(wallet_name: str, wallet_root: Path, run_root: Path,
                        netuid: int, live: bool,
                        epoch_budget: float | None, epoch_num: int) -> dict:
    """Run a single validator epoch and return results."""
    env = os.environ.copy()
    env["FUGAL_SKIP_BENCHMARKS"] = env.get("FUGAL_SKIP_BENCHMARKS",
        "mmlu,humaneval,livecode,gpqa,ifeval,aime")
    env["FUGAL_SLICE_SIZE"] = env.get("FUGAL_SLICE_SIZE", "50")
    env["FUGAL_EPOCH_INTERVAL"] = env.get("FUGAL_EPOCH_INTERVAL", LOCAL_EPOCH_INTERVAL)
    env["FUGAL_STATE_PATH"] = str(run_root / "validator_state.json")
    env["FUGAL_RESPONSE_CACHE"] = str(run_root / "response_cache")
    env["FUGAL_EPOCH_LOG_DIR"] = str(run_root / "epoch_logs")

    cmd = [
        sys.executable, "neurons/validator.py",
        "--network", LOCAL_ENDPOINT,
        "--netuid", str(netuid),
        "--coldkey", wallet_name,
        "--hotkey", "default",
        "--wallet-path", str(wallet_root),
        "--once",
        "--log-level", "INFO",
    ]
    if live:
        cmd.extend(["--live", "--epoch-budget", str(epoch_budget)])
    else:
        cmd.append("--mock")

    mode_str = f"(LIVE — hard budget ${epoch_budget:.2f})" if live else "(mock)"
    print(f"  Running validator epoch {epoch_num} {mode_str}...", flush=True)

    start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=600, env=env,
    )
    elapsed = time.time() - start

    output = result.stdout + result.stderr
    success = result.returncode == 0

    epoch_result = {
        "epoch": epoch_num,
        "success": success,
        "elapsed_s": round(elapsed, 1),
        "mock": not live,
        "returncode": result.returncode,
    }

    for line in output.split("\n"):
        if "Weights set successfully" in line or "set_weights_success" in line:
            epoch_result["weights_set"] = True
        elif "Weight-setting failed" in line:
            epoch_result["weights_set"] = False
            epoch_result["weight_error"] = line.strip()
        elif "cost=$" in line:
            try:
                cost_str = line.split("cost=$")[1].split(",")[0].split(")")[0]
                epoch_result["api_cost"] = float(cost_str)
            except (IndexError, ValueError):
                pass

    log_path = str(run_root / "epoch_logs" / "epochs.jsonl")
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                last_line = f.readlines()[-1]
            entry = json.loads(last_line)
            epoch_result["n_heads"] = entry.get("n_heads_valid", 0)
            epoch_result["weights_set"] = entry.get("set_weights_success", False)
            if entry.get("scores"):
                epoch_result["matrix_built"] = True
        except (json.JSONDecodeError, IndexError):
            pass

    if not success:
        epoch_result["error_tail"] = output[-500:]

    return epoch_result


# ── Step 8: Report ──

def print_report(results: list[dict], total_start: float, run_root: Path):
    """Print a summary report of all epochs."""
    section("TESTNET RUN REPORT")

    total_time = time.time() - total_start
    n_success = sum(1 for r in results if r["success"])
    total_cost = sum(r.get("api_cost", 0) for r in results)

    print(f"Epochs run:     {len(results)}", flush=True)
    print(f"Successful:     {n_success}/{len(results)}", flush=True)
    print(f"Total time:     {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print(f"Total API cost: ${total_cost:.4f}", flush=True)
    print(flush=True)

    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        mode = "mock" if r.get("mock") else "real"
        cost = f"${r.get('api_cost', 0):.4f}" if not r.get("mock") else "n/a"
        heads = r.get("n_heads", "?")
        weights = "yes" if r.get("weights_set") else "no"
        print(f"  Epoch {r['epoch']}: [{status}] {mode}  "
              f"heads={heads}  weights={weights}  "
              f"cost={cost}  time={r['elapsed_s']}s", flush=True)

        if not r["success"] and "error_tail" in r:
            for line in r["error_tail"].strip().split("\n")[-5:]:
                print(f"    {line.strip()}", flush=True)

    print(flush=True)

    if n_success == len(results) and len(results) > 0:
        print("ALL LOCAL EPOCHS PASSED — this is not production/mainnet approval.", flush=True)
    elif n_success > 0:
        print("PARTIAL SUCCESS — review failures above.", flush=True)
    else:
        print("ALL EPOCHS FAILED — check setup and logs.", flush=True)

    log_dir = str(run_root / "epoch_logs")
    if os.path.isdir(log_dir):
        logs = sorted(os.listdir(log_dir))
        if logs:
            print(f"\nEpoch logs written to {log_dir}/:", flush=True)
            for f in logs[-5:]:
                print(f"  {f}", flush=True)

    print(flush=True)
    print("Future deployment gates (separate review and authorization required):", flush=True)
    print("  [ ] Purchase subnet slot: btcli tx register-subnet -w fugal_owner --network finney", flush=True)
    print("  [ ] Activate subnet:      btcli tx start-call --netuid <N> -w fugal_owner --network finney", flush=True)
    print("  [ ] Register validator:    btcli tx burned-register --netuid <N> -w fugal_validator --network finney", flush=True)
    print("  [ ] Set env: FUGAL_NETWORK=finney FUGAL_NETUID=<N>", flush=True)
    print("  [ ] Fund OpenRouter account (~$25/epoch)", flush=True)
    print("  [ ] Deploy validator on production server", flush=True)
    print("  [ ] Announce subnet — miners can find guides in docs/", flush=True)


# ── Main ──

def parse_args():
    p = argparse.ArgumentParser(
        description="Fugal testnet launch — full end-to-end deployment test on local chain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full test with mock API (no cost at all):
  python scripts/launch_testnet.py --epochs 3

  # [PAID ~$30/epoch maximum] Explicit live mode and budget required:
  OPENROUTER_API_KEY=sk-or-... python scripts/launch_testnet.py \
      --live --epoch-budget 30 --epochs 2

  # Check setup without doing anything:
  python scripts/launch_testnet.py --dry-run

  # Already running, just run more epochs:
  python scripts/launch_testnet.py --skip-setup --netuid 2 --mock --epochs 3

""",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Check setup only, don't create anything")
    p.add_argument("--skip-setup", action="store_true",
                   help="Skip chain/wallet/subnet setup (already running)")
    p.add_argument("--netuid", type=int, default=None,
                   help="Use existing netuid (required with --skip-setup)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true",
                      help="Enable paid OpenRouter calls (requires a positive budget)")
    mode.add_argument("--mock", action="store_true",
                      help="Explicitly select the default no-spend mock mode")
    p.add_argument("--epoch-budget", type=float, default=None,
                   help="Positive USD ceiling required with --live (or FUGAL_EPOCH_BUDGET)")
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of epochs to run (default: 3)")
    p.add_argument("--retrain", action="store_true",
                   help="Force retrain the head even if one exists")
    p.add_argument("--run-root", type=Path, default=Path("results/local-testnet"),
                   help="Isolated state/output directory (default: results/local-testnet)")
    p.add_argument("--wallet-root", type=Path, default=None,
                   help="Isolated wallet root (default: RUN_ROOT/wallets)")
    p.add_argument("--head-path", type=Path, default=None,
                   help="Head .npz path (default: RUN_ROOT/testnet_head.npz)")
    p.add_argument("--keep-chain", action="store_true",
                   help="Don't stop the chain container after running")
    return p.parse_args()


def main():
    args = parse_args()
    total_start = time.time()
    run_root = args.run_root.expanduser().resolve()
    wallet_root = (
        args.wallet_root.expanduser().resolve()
        if args.wallet_root is not None else run_root / "wallets"
    )
    head_path = (
        args.head_path.expanduser().resolve()
        if args.head_path is not None else run_root / "testnet_head.npz"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    epoch_budget = args.epoch_budget
    if epoch_budget is None and os.getenv("FUGAL_EPOCH_BUDGET") is not None:
        try:
            epoch_budget = float(os.environ["FUGAL_EPOCH_BUDGET"])
        except ValueError:
            print("ERROR: FUGAL_EPOCH_BUDGET must be a positive number.", flush=True)
            return 2
    if args.live and (epoch_budget is None or epoch_budget <= 0):
        print(
            "ERROR: --live requires --epoch-budget AMOUNT or a positive "
            "FUGAL_EPOCH_BUDGET.",
            flush=True,
        )
        return 2
    if not args.live and args.epoch_budget is not None:
        print("ERROR: --epoch-budget is only valid with --live.", flush=True)
        return 2

    section("FUGAL LOCAL TESTNET LAUNCH")
    print(f"Chain:    local Docker ({DOCKER_IMAGE})", flush=True)
    print(f"Endpoint: {LOCAL_ENDPOINT}", flush=True)
    mode_label = (
        f"LIVE (paid; hard budget ${epoch_budget:.2f}/epoch)"
        if args.live else "mock (default; no API cost)"
    )
    print(f"Mode:     {mode_label}", flush=True)
    print(f"Epochs:   {args.epochs}", flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Wallets:  {wallet_root}", flush=True)
    print(flush=True)

    if args.live and not (
        os.getenv("FUGAL_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    ):
        print("ERROR: OpenRouter key not set; mock mode requires no key.", flush=True)
        return 1

    # ── Dry run ──
    if args.dry_run:
        section("DRY RUN — checking setup")

        print("Docker:", flush=True)
        if check_docker():
            print("  Docker is available", flush=True)
        else:
            return 1

        exists, configured_image, running = inspect_chain_container()
        if exists:
            state = "running" if running else "stopped"
            matches = "matches pin" if configured_image == DOCKER_IMAGE else "WRONG IMAGE"
            print(f"  Chain container is {state}: {configured_image} ({matches})", flush=True)
        else:
            print("  Chain container not running (will start on launch)", flush=True)

        print("\nWallets:", flush=True)
        setup_wallets(wallet_root, dry_run=True)

        print("\nHead artifact:", flush=True)
        if head_path.exists():
            print(f"  Exists: {head_path} ({head_path.stat().st_size} bytes)", flush=True)
        else:
            print("  Not found (will train on launch)", flush=True)

        api_key = os.getenv("OPENROUTER_API_KEY")
        print("\nOpenRouter API:", flush=True)
        if api_key:
            print("  Key is set", flush=True)
        else:
            print("  Not set (use --mock for free testing)", flush=True)

        print("\nDry run complete. Run without --dry-run to proceed.", flush=True)
        return 0

    # ── Full run ──
    miner_proc = None
    chain_started = False
    netuid = args.netuid

    try:
        if not args.skip_setup:
            # Step 0: Start chain
            section("Step 0: Start Local Chain")
            if not check_docker():
                return 1
            ready, chain_started = start_local_chain()
            if not ready:
                print("Failed to start local chain", flush=True)
                return 1

            # Step 1: Wallets
            section("Step 1: Create Wallets")
            wallets = setup_wallets(wallet_root)

            # Step 2: Fund
            section("Step 2: Fund Wallets from Alice")
            if not fund_wallets(wallets):
                print("Wallet funding failed", flush=True)
                return 1

            # Step 3: Create subnet
            section("Step 3: Create & Activate Subnet")
            netuid = create_subnet(wallets, wallet_root)
            if netuid < 0:
                return 1

            # Step 4: Register
            section("Step 4: Register Neurons")
            val_uid = register_neuron(wallets[VALIDATOR_WALLET], netuid, "Validator")
            miner_uid = register_neuron(wallets[MINER_WALLET], netuid, "Miner")
            if val_uid < 0 or miner_uid < 0:
                print("Registration failed", flush=True)
                return 1

            print("\n  Metagraph:", flush=True)
            result = run_btcli([
                "query", "metagraph",
                "--netuid", str(netuid),
                "--network", LOCAL_ENDPOINT,
            ], check=False, timeout=30)
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n")[:10]:
                    print(f"    {line}", flush=True)

        else:
            if netuid is None:
                print("ERROR: --skip-setup requires --netuid", flush=True)
                return 1
            print(f"Skipping setup, using netuid {netuid}", flush=True)
            if not wait_for_chain(timeout=10):
                print("Chain not reachable. Start it first or remove --skip-setup.", flush=True)
                return 1

        # Step 5: Train head
        section("Step 5: Train Head")
        if args.retrain and head_path.exists():
            backup = head_path.with_name(f"{head_path.name}.replaced-{int(time.time())}")
            head_path.replace(backup)
            print(f"  Archived previous head: {backup}", flush=True)
        trained_head_path = train_head(str(head_path))

        # Step 6: Start miner
        section("Step 6: Start Miner")
        state_path = run_root / "validator_state.json"
        if not args.skip_setup and state_path.exists():
            backup = state_path.with_name(f"{state_path.name}.replaced-{int(time.time())}")
            state_path.replace(backup)
            print(f"  Archived stale validator state: {backup}", flush=True)
        miner_proc = start_miner_process(
            MINER_WALLET, wallet_root, netuid, trained_head_path,
        )

        # Step 7: Run epochs
        section("Step 7: Run Validator Epochs")
        results = []
        for i in range(1, args.epochs + 1):
            # Ensure the miner's head commitment predates the epoch boundary,
            # and that each run lands on a fresh boundary (fresh slice).
            wait_for_next_epoch_boundary(
                minimum_blocks=(
                    LOCAL_WEIGHT_RATE_LIMIT_BLOCKS + 1 if i > 1 else 0
                )
            )
            result = run_validator_epoch(
                VALIDATOR_WALLET, wallet_root, run_root,
                netuid, args.live, epoch_budget, i,
            )
            results.append(result)

            status = "PASS" if result["success"] else "FAIL"
            print(f"  Epoch {i}: {status} ({result['elapsed_s']}s)", flush=True)

        # Step 8: Report
        print_report(results, total_start, run_root)
        return 0 if all(r["success"] for r in results) else 1

    except KeyboardInterrupt:
        print("\nInterrupted by user", flush=True)
        return 130
    except Exception as e:
        logger.exception("Launch failed: %s", e)
        return 1
    finally:
        if miner_proc and miner_proc.poll() is None:
            print(f"Stopping miner (PID {miner_proc.pid})...", flush=True)
            miner_proc.terminate()
            try:
                miner_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                miner_proc.kill()
            print("Miner stopped", flush=True)

        if chain_started and not args.keep_chain:
            print(f"Stopping chain container '{DOCKER_CONTAINER}'...", flush=True)
            stop_local_chain()
            print("Chain stopped", flush=True)
        elif chain_started:
            print(f"Chain container '{DOCKER_CONTAINER}' left running (--keep-chain)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
