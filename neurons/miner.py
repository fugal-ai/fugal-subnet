#!/usr/bin/env python3
"""Fugal subnet TEE miner — runs benchmarks inside Intel TDX, publishes proofs.

Usage:
    python neurons/miner.py --netuid 1 --coldkey miner --hotkey default \\
        --head-path data/my_head.npz --benchmark-pool data/pool.json

Each epoch:
  1. Wait for epoch boundary (new block hash)
  2. Derive nonce from block hash
  3. Run benchmark inside TEE (route questions via head, call models, grade)
  4. Generate TDX attestation binding the proof
  5. Serve proof via axon when validator queries

The miner pays for model inference. The validator verifies the attested
proof without calling any models.
"""
import base64
import hashlib
import json
import logging
import os
import threading
import time
from typing import Tuple

import click

# isort: off
import fugal_subnet.determinism  # noqa: F401, E402 — must precede numpy
# isort: on

logger = logging.getLogger("fugal.miner")

METAGRAPH_REFRESH_S = 300


@click.command()
@click.option("--network", default=lambda: os.getenv("FUGAL_NETWORK", "test"),
              help="Network: finney, test, local")
@click.option("--netuid", type=int,
              default=lambda: int(os.getenv("FUGAL_NETUID", "1")),
              help="Subnet netuid")
@click.option("--coldkey", default=lambda: os.getenv("WALLET_NAME", "default"),
              help="Wallet coldkey name")
@click.option("--hotkey", default=lambda: os.getenv("HOTKEY_NAME", "default"),
              help="Hotkey name")
@click.option("--wallet-path", default=lambda: os.getenv("FUGAL_WALLET_PATH") or None,
              type=click.Path(file_okay=False),
              help="Bittensor wallet root")
@click.option("--port", default=lambda: int(os.getenv("FUGAL_MINER_PORT", "8091")),
              type=int, help="Axon port")
@click.option("--head-path", required=True, type=click.Path(exists=True),
              help="Path to .npz head artifact")
@click.option("--benchmark-pool", default=None, type=click.Path(exists=True),
              help="Benchmark pool JSON (default: the same load_all() the "
                   "validator uses — override only for offline or local runs)")
@click.option("--mock/--live", default=True,
              help="Mock mode (no real TDX attestation)")
@click.option("--log-level",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              default=lambda: os.getenv("LOG_LEVEL", "INFO"),
              help="Logging level")
def main(network, netuid, coldkey, hotkey, wallet_path, port, head_path,
         benchmark_pool, mock, log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import bittensor as bt

    # After bittensor: importing it disables every logger that already exists,
    # so without this the neuron's own output — including every error and
    # traceback — is silently discarded. See fugal_subnet/logging_setup.
    from fugal_subnet.logging_setup import configure_logging
    configure_logging(log_level)

    from fugal_subnet.api import load_prices
    from fugal_subnet.benchmarks.slicer import (
        blocks_per_epoch as blocks_per_epoch_fn,
    )
    from fugal_subnet.benchmarks.slicer import (
        epoch_index_for_block,
    )
    from fugal_subnet.commitments import ensure_commitment
    from fugal_subnet.config import (
        EPOCH_INTERVAL,
        EXPLORE_FRACTION,
        MIN_VALIDATOR_STAKE,
        SLICE_SIZE,
        TEE_MODEL_PRICES_PATH,
        TEE_PROXY_PORT,
    )
    from fugal_subnet.protocol import FugalProofSynapse
    from fugal_subnet.tee.runtime import TEERuntime

    head_data = _load_head_file(head_path)
    weights_hash = hashlib.sha256(head_data).hexdigest()

    # The pool is consensus state: the slice is drawn from it, so the miner and
    # the validator must derive it identically or every proof fails on
    # questions_hash for a reason that looks nothing like the cause. Default to
    # the same loader the validator calls; --benchmark-pool is an explicit
    # local override, and FUGAL_BENCHMARK_POOL overrides for both sides at once.
    from fugal_subnet.benchmarks.loader import load_all, pool_hash
    # ONE code path, always. --benchmark-pool used to json.load the file here,
    # which meant the flag bypassed everything load_all() does — and when the
    # loader learned to drop questions the harness cannot grade, the validator
    # served 125 questions while the miner served 150 and every proof failed on
    # a questions_hash that named neither side. Routing the flag through the
    # same env var the validator honours makes a second path impossible rather
    # than merely discouraged. tests/test_pool_single_source.py enforces it.
    if benchmark_pool:
        os.environ["FUGAL_BENCHMARK_POOL"] = os.path.abspath(benchmark_pool)
    pool = load_all()
    if not pool:
        raise click.ClickException("Benchmark pool is empty — nothing to benchmark")
    logger.info("Benchmark pool: %d questions, pool_hash=%s",
                len(pool), pool_hash(pool)[:16])

    # The globally agreed model space. Exploration targets are drawn from this
    # sorted list, never from the miner's own head, so the samples the reference
    # frame is built from cannot be steered.
    prices = load_prices(TEE_MODEL_PRICES_PATH or None)
    pool_models = sorted(prices)
    explore_size = max(1, int(round(SLICE_SIZE * EXPLORE_FRACTION)))
    logger.info("Price table: %d models; exploration quota %d questions/epoch",
                len(pool_models), explore_size)

    logger.info("Head loaded: %s (%d bytes)", head_path, len(head_data))
    logger.info("Weights hash: %s", weights_hash[:16])

    # Once, at startup — not per epoch. See _compute_hidden_states.
    logger.info("Computing backbone embeddings for %d questions...", len(pool))
    hidden_states = _compute_hidden_states(pool)
    logger.info("Embeddings ready: %s (backbone released)", hidden_states.shape)

    wallet = bt.Wallet(name=coldkey, hotkey=hotkey, path=wallet_path)
    subtensor = bt.Subtensor(network=network)
    metagraph = subtensor.metagraph(netuid)

    my_hotkey = wallet.hotkey.ss58_address
    if my_hotkey not in metagraph.hotkeys:
        logger.error("Hotkey %s not registered on netuid %d", my_hotkey, netuid)
        return
    my_uid = metagraph.hotkeys.index(my_hotkey)
    logger.info("Miner UID %d on %s netuid %d", my_uid, network, netuid)

    # NOTE: the on-chain commitment is deliberately NOT made here. Chains rate
    # limit transactions per hotkey, so committing immediately before
    # axon.serve() gets the serve extrinsic rejected and leaves the miner
    # unreachable at 0.0.0.0 — earning nothing, with the commitment it did
    # land being worthless because no validator can query it. Reachability
    # first; the commitment is made after the axon is serving and retried each
    # epoch until it sticks.
    committed = False

    tee_runtime = TEERuntime(mock=mock)

    # Bind what this runtime IS into RTMR3, once, before any proof exists.
    # Advisory today: on an unmeasured filesystem an attacker extends the value
    # we expect. It is written anyway so that locking the image turns this into
    # evidence rather than requiring a new mechanism. Says so out loud either
    # way — a binding nobody can see is worse than none. See INVARIANTS.md I8.
    if not mock:
        from fugal_subnet.benchmarks.loader import pool_hash as _pool_hash
        from fugal_subnet.graders import grader_hash
        from fugal_subnet.tee.attestation import extend_rtmr3, runtime_identity
        from fugal_subnet.tee.runtime import _OPENROUTER_BASE
        identity = runtime_identity(
            source_hash=_get_source_hash(),
            pool_hash=_pool_hash(pool),
            grader_hash=grader_hash(),
            upstream=_OPENROUTER_BASE,
        )
        logger.info("Model upstream: %s", _OPENROUTER_BASE)
        if extend_rtmr3(identity):
            logger.info(
                "Runtime identity %s extended into RTMR3 (ADVISORY: not "
                "enforced until the image is locked)", identity[:16],
            )
        else:
            logger.warning(
                "Runtime identity %s could NOT be extended into RTMR3. Nothing "
                "in this TD's measurement registers binds the code that will "
                "produce its proofs.", identity[:16],
            )
    current_proof = {
        "proof": None, "proof_json": "", "epoch_id": "", "lock": threading.Lock(),
    }

    mg_lock = threading.Lock()
    mg_state = {"metagraph": metagraph}

    def blacklist(synapse: FugalProofSynapse) -> Tuple[bool, str]:
        caller = getattr(synapse.dendrite, "hotkey", None)
        if not caller:
            return True, "no caller hotkey"
        with mg_lock:
            mg = mg_state["metagraph"]
        if caller not in mg.hotkeys:
            return True, f"unregistered hotkey {caller[:8]}..."
        if MIN_VALIDATOR_STAKE > 0:
            uid = mg.hotkeys.index(caller)
            permit = False
            try:
                permit = bool(mg.validator_permit[uid])
            except (AttributeError, IndexError):
                pass
            stake = 0.0
            try:
                stake = float(mg.S[uid])
            except (AttributeError, IndexError, TypeError):
                pass
            if not permit and stake < MIN_VALIDATOR_STAKE:
                return True, f"caller {caller[:8]}... lacks validator permit/stake"
        return False, "ok"

    head_b64 = base64.b64encode(head_data).decode("ascii")

    def forward(synapse: FugalProofSynapse) -> FugalProofSynapse:
        with current_proof["lock"]:
            proof = current_proof["proof"]
            proof_json = current_proof["proof_json"]
            proof_epoch = current_proof["epoch_id"]
        if proof is None:
            logger.warning("No proof available yet for epoch %s", synapse.epoch_id)
            return synapse

        # Serve only a proof for the epoch the validator actually asked about.
        # The validator rejects a stale proof anyway (its nonce won't match), so
        # returning one just burns bandwidth and buries the real reason in a
        # generic "nonce mismatch" on the validator side.
        if synapse.epoch_id and synapse.epoch_id != proof_epoch:
            logger.warning(
                "Epoch mismatch: validator asked for %s, we hold %s — not serving",
                synapse.epoch_id, proof_epoch or "<none>",
            )
            return synapse

        # The bundle rides back inline. See fugal_subnet/protocol.py for why
        # there is no external store: the validator is asking us right now, so
        # a second fetch from a third party adds a failure mode and buys nothing.
        synapse.proof_json = proof_json
        synapse.head_npz_b64 = head_b64
        synapse.proof_hash = proof.content_hash()
        synapse.weights_hash = weights_hash
        logger.info(
            "Epoch %s: serving bundle (hash=%s, %d KB proof + %d KB head)",
            synapse.epoch_id, synapse.proof_hash[:16],
            len(proof_json) // 1024, len(head_b64) // 1024,
        )
        return synapse

    external_ip = os.getenv("FUGAL_AXON_IP")
    if external_ip and _is_unroutable(external_ip):
        # The chain rejects loopback and unspecified addresses for serve_axon
        # at pool validation (Custom error 11), before the extrinsic executes.
        # bt.Axon.serve() swallows that, so the only symptom is an axon that
        # stays at 0.0.0.0 and a miner that silently earns nothing.
        logger.error(
            "FUGAL_AXON_IP=%s is not routable. The chain will reject serve_axon "
            "and no validator will ever reach this miner. Set it to an address "
            "validators can actually connect to.", external_ip,
        )
    # Fail before serving if something else already holds the port. bt.Axon
    # runs its server in a thread and a bind failure there does not surface:
    # the miner goes on to register the address on chain and log success while
    # every validator query is answered by whatever process actually owns the
    # port. Observed exactly that — a stray HTTP server returned 501 to every
    # dendrite and the miner reported itself healthy throughout.
    _assert_port_free(port)
    axon = bt.Axon(wallet=wallet, port=port,
                    external_ip=external_ip or None)
    axon.attach(forward_fn=forward, blacklist_fn=blacklist)

    for attempt in range(5):
        try:
            axon.serve(netuid=netuid, subtensor=subtensor)
        except Exception as e:
            logger.warning("axon.serve() attempt %d failed: %s", attempt + 1, e)
        metagraph = subtensor.metagraph(netuid)
        registered_ip = metagraph.axons[my_uid].ip
        if registered_ip not in ("0.0.0.0", ""):
            logger.info("Axon registered on chain: %s:%d", registered_ip, port)
            break
        if attempt < 4:
            logger.warning("Axon IP not registered yet, retrying in 15s...")
            time.sleep(15)
        else:
            logger.error(
                "axon.serve() failed after 5 attempts — the axon is still "
                "unregistered (0.0.0.0), so no validator can query this miner "
                "and it will earn nothing. Common causes: the external IP is "
                "loopback or otherwise unroutable (currently %s), the port is "
                "unreachable from outside, or the chain is rate limiting this "
                "hotkey's transactions.",
                external_ip or "auto-detected",
            )

    axon.start()
    logger.info("Miner axon serving on port %d (mock=%s)", port, mock)
    _warn_if_unreachable(metagraph.axons[my_uid].ip, port)

    # Now that the axon is serving, commit the head hash. Retried in the loop
    # below if the chain rate-limits this one.
    committed = ensure_commitment(subtensor, wallet, netuid, weights_hash)
    if not committed:
        logger.warning("Weights hash NOT committed on-chain yet — will retry.")

    blocks_per_epoch = blocks_per_epoch_fn(EPOCH_INTERVAL)

    try:
        last_refresh = time.time()
        last_epoch_index = -1
        consecutive_failures = 0
        while True:
            time.sleep(30)

            if not committed:
                committed = ensure_commitment(subtensor, wallet, netuid, weights_hash)

            if time.time() - last_refresh >= METAGRAPH_REFRESH_S:
                try:
                    fresh = subtensor.metagraph(netuid)
                    with mg_lock:
                        mg_state["metagraph"] = fresh
                except Exception as e:
                    logger.warning("Metagraph refresh failed: %s", e)
                last_refresh = time.time()

            try:
                current_block = subtensor.get_current_block()
                epoch_index = epoch_index_for_block(current_block, blocks_per_epoch)
                if epoch_index <= last_epoch_index:
                    continue

                boundary_block = epoch_index * blocks_per_epoch
                block_hash = subtensor.get_block_hash(boundary_block)
                if not block_hash:
                    continue

                last_epoch_index = epoch_index
                _run_epoch(
                    blocks_per_epoch=blocks_per_epoch,
                    head_data=head_data,
                    weights_hash=weights_hash,
                    pool=pool,
                    hidden_states=hidden_states,
                    epoch_index=epoch_index,
                    block_hash=block_hash,
                    tee_runtime=tee_runtime,
                    current_proof=current_proof,
                    slice_size=SLICE_SIZE,
                    proxy_port=TEE_PROXY_PORT,
                    explore_models=pool_models,
                    explore_size=explore_size,
                )
                consecutive_failures = 0
            except Exception:
                # A miner that silently swallows every epoch failure looks
                # alive while producing nothing. Log the traceback and escalate
                # so the operator sees it.
                consecutive_failures += 1
                logger.exception(
                    "Epoch run failed (%d consecutive)", consecutive_failures,
                )
                if consecutive_failures >= 3:
                    logger.error(
                        "%d consecutive epoch failures — this miner is earning "
                        "nothing. Fix the error above.", consecutive_failures,
                    )

    except KeyboardInterrupt:
        logger.info("Miner stopped by user")
    finally:
        axon.stop()
        tee_runtime.teardown()
        logger.info("Miner shutdown complete")


def _run_epoch(
    blocks_per_epoch,
    head_data,
    weights_hash,
    pool,
    hidden_states,
    epoch_index,
    block_hash,
    tee_runtime,
    current_proof,
    slice_size,
    proxy_port,
    explore_models,
    explore_size,
):
    """Run a single benchmark epoch."""
    from fugal_subnet.benchmarks.slicer import derive_nonce, epoch_id_for_block
    from fugal_subnet.tee.harness import run_benchmark

    # Must match the validator exactly — see slicer.epoch_id_for_block.
    epoch_id = epoch_id_for_block(epoch_index)
    nonce_bytes = derive_nonce(epoch_id, block_hash)
    nonce = nonce_bytes.hex()

    # State the deadline. Validators collect at a fixed block derived from the
    # epoch, so "my miner earns nothing" is usually "my benchmark did not
    # finish in time" — which is invisible unless the miner says what the time
    # was. A miner that consistently overruns this should cut its slice cost,
    # not wonder why its proofs are ignored.
    from fugal_subnet.benchmarks.slicer import collect_block_for_epoch
    from fugal_subnet.config import EPOCH_COLLECT_FRACTION
    collect_block = collect_block_for_epoch(
        epoch_index, blocks_per_epoch, EPOCH_COLLECT_FRACTION,
    )
    logger.info(
        "Starting epoch %s (nonce=%s...); proof must be ready by block %d",
        epoch_id, nonce[:16], collect_block,
    )

    proxy = tee_runtime.setup(proxy_port=proxy_port)
    try:
        proof = run_benchmark(
            nonce=nonce,
            head_bytes=head_data,
            benchmark_pool=pool,
            proxy=proxy,
            hidden_states=hidden_states,
            slice_size=slice_size,
            epoch_id=epoch_id,
            source_hash=_get_source_hash(),
            explore_models=explore_models,
            explore_size=explore_size,
        )

        content_hash_bytes = bytes.fromhex(proof.content_hash())
        proof.attestation_quote = tee_runtime.generate_attestation(content_hash_bytes)

        # Serialize once here, not per query: several validators may ask for
        # the same epoch and the JSON is identical for all of them.
        with current_proof["lock"]:
            current_proof["proof"] = proof
            current_proof["proof_json"] = json.dumps(
                proof.to_dict(), separators=(",", ":"),
            )
            current_proof["epoch_id"] = epoch_id

        logger.info(
            "Epoch %s complete: %d/%d correct (%.1f%%), cost=$%.4f",
            epoch_id, proof.n_correct, proof.n_total,
            proof.accuracy * 100, proof.total_cost_usd,
        )
    finally:
        proxy.stop()


def _assert_port_free(port: int) -> None:
    """Refuse to start if another process is listening on the axon port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # 0.0.0.0 is REQUIRED here and narrowing it to 127.0.0.1 would
            # silently break this check. The axon binds 0.0.0.0 — it must, or no
            # validator can reach it — so the only question worth asking is
            # whether THAT bind will succeed. A process holding a single
            # external interface (192.168.1.5:8091, say) does not conflict with
            # a loopback bind, so a loopback probe would report the port free
            # and the axon would then fail to bind the port it actually needs.
            # This socket never listens or accepts; it binds and closes, so it
            # exposes nothing. CodeQL flags the literal, not the behaviour.
            probe.bind(("0.0.0.0", port))
        except OSError as e:
            raise click.ClickException(
                f"Port {port} is already in use ({e}). Another process would "
                f"answer every validator query while this miner reported "
                f"itself healthy. Free the port, or pass a different --port."
            ) from e


def _warn_if_unreachable(published_ip: str, port: int) -> None:
    """Try to connect to the address this miner just told the chain to use.

    Serving an axon and being reachable are different things, and only the
    first one is reported anywhere. A miner on a cloud VM whose provider
    firewall does not allow the port publishes a perfectly valid address,
    logs "Axon registered on chain", and is never once contacted — it earns
    nothing for as long as it runs, with no error on either side. That is not
    hypothetical: netuid 552's miner advertised a public address for months
    behind a closed security-list rule.

    This is a warning and never fatal. Many hosts do not route their own public
    address back to themselves (no NAT hairpinning), so a failure here does not
    prove the miner is unreachable — but a success does prove it is reachable,
    and a failure is worth a line the operator can act on.
    """
    import socket
    import time as _time

    if not published_ip or published_ip in ("0.0.0.0", ""):
        return
    # axon.start() returns before its server thread is listening, so a single
    # immediate probe reports "connection refused" on a perfectly healthy
    # miner. Retry briefly: a false alarm here trains an operator to ignore
    # the one warning that means their miner earns nothing.
    deadline = _time.monotonic() + 20
    e: OSError | None = None
    while _time.monotonic() < deadline:
        try:
            with socket.create_connection((published_ip, port), timeout=5):
                # Something accepted. That is all a TCP connect can tell us —
                # it does not prove the responder is this axon, which is why
                # the port is claimed before serving rather than inferred after.
                logger.info("Axon accepts connections at %s:%d", published_ip, port)
                return
        except OSError as err:
            e = err
            _time.sleep(1)
    if e is not None:
        logger.warning(
            "Could not connect to this miner's own published address %s:%d (%s). "
            "If this host does not route its public IP back to itself this is "
            "harmless, but if the port is genuinely closed no validator will "
            "ever reach this miner and it will earn nothing while appearing "
            "healthy. Verify from another machine: nc -vz %s %d",
            published_ip, port, e, published_ip, port,
        )


def _embedding_cache_path(pool) -> str:
    """Where this pool's embeddings live on disk.

    Keyed by everything the result depends on. Two of these are consensus
    inputs and one is not: the pool and the batch size are pinned and shared,
    while the backbone build is local — and x86_64 and aarch64 genuinely
    disagree in the last bits of float32, so an architecture-blind key would
    hand a miner another machine's numbers. Keying on the machine makes the
    cache a pure memo of what this host would have computed anyway.
    """
    import platform

    from fugal_subnet.benchmarks.loader import pool_hash
    from fugal_subnet.config import BACKBONE_BATCH_SIZE, BACKBONE_MODEL

    key = hashlib.sha256("|".join([
        pool_hash(pool),
        BACKBONE_MODEL,
        str(BACKBONE_BATCH_SIZE),
        platform.machine(),
    ]).encode("utf-8")).hexdigest()[:32]
    root = os.getenv("FUGAL_EMBEDDING_CACHE", os.path.join("data", "embeddings"))
    return os.path.join(root, f"hidden-{key}.npy")


def _compute_hidden_states(pool):
    """Backbone hidden states for the benchmark pool, computed once and cached.

    Called at startup, never per epoch. Embeddings are a pure function of the
    pool, the frozen backbone and the pinned batch size — none of which change
    between epochs — so recomputing them every epoch burned minutes of CPU on a
    21K-question pool for an identical result.

    They do not change between *restarts* either, which is why the result is
    cached to disk. Measured on the real 21,717-question pool: 2h28m on an
    x86_64 laptop and 7h20m on a 4-core aarch64 VM. Paying that on every
    restart is not a startup cost, it is an availability failure — the miner is
    unreachable for hours and earns nothing, and any crash-loop is permanent.

    The cache is keyed by pool_hash so a pool change invalidates it, and it is
    loaded with allow_pickle=False so a hostile file cannot execute anything.
    A corrupt or unreadable file is recomputed.

    WHAT IT DOES NOT DEFEND AGAINST, stated plainly. These embeddings are
    computed here and passed into run_benchmark, which means they are an input
    to the attested harness rather than a product of it. Anyone who can write
    this file chooses the routing the head appears to produce — they could make
    a head look like it routes well when it does not, which matters because the
    head is the artifact other people are meant to reuse.

    That is acceptable only because of where this is supposed to run. Under
    --live the whole miner executes inside the TD, so the file lives in the
    TD's own storage and is covered by the same protection as the code the
    measurement attests. Running a --live miner with this cache on storage
    outside the TD removes a binding the attestation is assumed to provide.
    Point FUGAL_EMBEDDING_CACHE inside the enclave, or accept recomputing.

    The backbone is released afterwards: it is ~2.4GB resident and is not
    needed again once the embeddings exist.
    """
    import numpy as np

    from fugal_subnet.backbone import compute_hidden_states, release_backbone
    from fugal_subnet.config import BACKBONE_BATCH_SIZE

    cache_path = _embedding_cache_path(pool)
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if cached.shape[0] == len(pool):
                logger.info("Embeddings loaded from cache: %s", cache_path)
                return cached
            logger.warning("Cached embeddings have %d rows for a %d-question "
                           "pool — recomputing", cached.shape[0], len(pool))
        except Exception as e:  # noqa: BLE001 - a bad cache must never be fatal
            logger.warning("Could not read embedding cache %s (%s) — recomputing",
                           cache_path, e)

    questions = [q["prompt"] for q in pool]
    # batch_size is pinned in config, not left to the call site: padding is
    # batch-composition dependent, so two hosts using different batch sizes are
    # a latent cross-validator divergence.
    try:
        hidden = compute_hidden_states(questions, batch_size=BACKBONE_BATCH_SIZE)
    finally:
        release_backbone()

    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        # Write-then-rename: a miner killed mid-write must not leave a
        # truncated cache that the next start reads as complete.
        # np.save appends ".npy" unless the name already ends in it, so the
        # temp name carries the suffix or the rename below looks for a file
        # that was never written.
        tmp = f"{cache_path}.{os.getpid()}.tmp.npy"
        np.save(tmp, hidden, allow_pickle=False)
        os.replace(tmp, cache_path)
        logger.info("Embeddings cached to %s", cache_path)
    except Exception as e:  # noqa: BLE001 - caching is an optimisation
        logger.warning("Could not cache embeddings to %s: %s", cache_path, e)
    return hidden


def _get_source_hash():
    """Hash the actual source file contents for measurement verification."""
    import fugal_subnet
    src_dir = os.path.dirname(os.path.abspath(fugal_subnet.__file__))
    h = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(src_dir)):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            h.update(fpath[len(src_dir):].encode())
            h.update(open(fpath, "rb").read())
    return h.hexdigest()


def _is_unroutable(ip: str) -> bool:
    """True for addresses the chain will not accept for serve_axon."""
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_unspecified


def _load_head_file(path):
    """Load and validate the head .npz file."""
    from fugal_subnet.head_eval import load_head_from_npz
    with open(path, "rb") as f:
        data = f.read()
    try:
        load_head_from_npz(data)
    except ValueError as e:
        raise click.ClickException(f"Head failed validation: {e}")
    return data


if __name__ == "__main__":
    main()
