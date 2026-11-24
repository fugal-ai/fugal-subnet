#!/usr/bin/env python3
"""Fugal subnet TEE miner — runs benchmarks inside Intel TDX, publishes proofs.

Usage:
    python neurons/miner.py --network test --netuid <N> --coldkey miner --hotkey default \\
        --head-path data/my_head.npz              # --mock (default): no spend

    # A dstack TD (deploy/dstack/): --live --await-provisioning, the head,
    # hotkey and API key pushed over the attested channel (scripts/provision_td.py)

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
from fugal_subnet.config import MIN_VALIDATOR_STAKE  # noqa: E402
# isort: on

logger = logging.getLogger("fugal.miner")

METAGRAPH_REFRESH_S = 300



def caller_allowed(metagraph, caller, *, live: bool, min_stake: float | None = None):
    """May `caller` query this axon? Returns (allowed, reason).

    Live: a registered hotkey with stake at or above `min_stake` (default
    MIN_VALIDATOR_STAKE) may query. A validator permit on its own does not:
    subtensor grants a permit to any nonzero stake while fewer than
    `max_allowed_validators` hotkeys are staked, so on a young subnet a
    dust-staked hotkey holds one (review 2026-11-02). The axon answers with
    the full head, so anyone else is a copier with a download link (diligence
    2026-10, finding 5). The stake read is `metagraph.S`, the chain's
    `total_stake` for the UID: alpha on this subnet plus root TAO at the
    chain's root weight (0.18), after child/parent-key inheritance -- the
    same quantity the chain uses for permits and weight influence. Mock: any
    registered hotkey, for local testnets where nothing is staked. Factored
    out of the axon closure so the rule can be tested without a chain.
    """
    if not caller:
        return False, "no caller hotkey"
    hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
    if caller not in hotkeys:
        return False, f"unregistered hotkey {caller[:8]}..."
    if not live:
        return True, "ok (mock admits any registered hotkey)"
    threshold = MIN_VALIDATOR_STAKE if min_stake is None else float(min_stake)
    uid = hotkeys.index(caller)
    stake = 0.0
    try:
        stake = float(metagraph.S[uid])
    except (AttributeError, IndexError, TypeError):
        pass
    permit = False
    try:
        permit = bool(metagraph.validator_permit[uid])
    except (AttributeError, IndexError, TypeError):
        pass
    if stake >= threshold:
        return True, f"ok (stake {stake:.0f} >= {threshold:.0f}{', permit' if permit else ''})"
    if permit:
        return False, (
            f"caller {caller[:8]}... has stake {stake:.0f} below {threshold:.0f}; "
            "a validator permit alone does not admit"
        )
    return False, (
        f"caller {caller[:8]}... has stake {stake:.0f} below {threshold:.0f} "
        "and no validator permit"
    )


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
@click.option("--await-provisioning", is_flag=True, default=False,
              help="Wait for the operator to push the head and API key over an "
                   "attested channel before serving. Under dstack this is how "
                   "per-miner data arrives, because it cannot be in the measured "
                   "compose. See fugal_subnet/tee/provision.py.")
@click.option("--head-path", required=False, type=click.Path(exists=True),
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
@click.option("--v4-proposal", default=lambda: os.getenv("FUGAL_V4_PROPOSAL") or None,
              help="Protocol v4 only: a menu entry this miner proposes, as JSON "
                   '{"model": "<lab>/<name>", "reasoning": ["effort", "high"], '
                   '"provider_order": ["<host>/<quant>", ...], "quantization": "<quant>"}. '
                   "The miner serves its proposal document from its axon and commits the "
                   "document's sha256 on chain beside its head; the root admits it at the next "
                   "boundary when the rules pass it. Nothing is called or spent for it.")
@click.option("--v4-state-dir", default=lambda: os.getenv("FUGAL_V4_STATE_DIR") or None,
              help="Protocol v4 only: where the TD keeps the last menu it accepted "
                   "(menu.json). Default results/v4 under the working directory: "
                   "/app/results/v4 in the image, on the encrypted data volume. It holds "
                   "nothing secret.")
@click.option("--v4-anchor", default=lambda: os.getenv("FUGAL_V4_ANCHOR") or None,
              help="Protocol v4 only, a last resort: a menu file (its canonical bytes, as the "
                   "reveal store serves them) for the TD to start from when neither its kept "
                   "menu nor the newest root reveal can. Checked like every menu.")
@click.option("--v4-challenge", default=lambda: os.getenv("FUGAL_V4_CHALLENGE") or "0",
              type=click.Choice(["0", "500", "1000"]),
              help="Protocol v4 only: the challenge the TD answers each epoch [PAID, about $16 "
                   "for 500 and $32 for 1,000 questions]: the first 500 or 1,000 questions of "
                   "the epoch's challenge list, this head's picks and, where they differ, the "
                   "seat holder's. '0' (default): none, exploration only (an odds-only miner, "
                   "and the holder, which answers nothing). A challenger sets it for the epochs "
                   "in which it expects to beat the holder on odds.")
@click.option("--v4-challenge-usd", default=lambda: os.getenv("FUGAL_V4_CHALLENGE_USD") or None,
              help="Protocol v4 only: the TD's dollar ceiling on one challenge's realised spend, "
                   "whole dollars (default 100, at most 50000). The TD makes no challenge whose "
                   "estimate at the epoch's pick costs exceeds it, and starts no challenge call "
                   "once the challenge's realised spend plus the estimate of its calls in "
                   "flight reaches it. Calls in flight then finish and are paid.")
@click.option("--check", is_flag=True,
              help="Run the startup checks (chain, protocol, v4 options and files, pool, code "
                   "sandbox, registration), print the READY line and exit (0 when ready). "
                   "Waits for no provisioning, embeds and serves nothing.")
def main(network, netuid, coldkey, hotkey, wallet_path, port, head_path,
         await_provisioning,
         benchmark_pool, mock, log_level, check=False, v4_proposal=None,
         v4_state_dir=None, v4_anchor=None, v4_challenge="0", v4_challenge_usd=None):
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
    # Internet scanners probe any public port within seconds of it opening
    # (measured on the first live TD: 114 requests for .env, wp-config.php,
    # actuator, ... in the first minute). The SDK logs each as an ERROR-level
    # UnknownSynapseError, which buries the miner's own lines. Drop that one
    # message class; a real error from the axon still gets through.
    class _DropSynapseScans(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "UnknownSynapseError" not in record.getMessage()
    logging.getLogger("bittensor").addFilter(_DropSynapseScans())

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
        SLICE_SIZE,
        SLICE_SIZE_V4,
        TEE_MODEL_PRICES_PATH,
        TEE_PROXY_PORT,
    )
    from fugal_subnet.protocol import FugalProofSynapse
    from fugal_subnet.tee.runtime import TEERuntime

    # Refuse a forbidden configuration (mock on finney, a consensus override
    # in the environment) BEFORE asking the operator for secrets and before
    # embedding the pool (3 h 20 min on a c3-standard-8). It used to run after
    # both: a misconfigured TD received the API key and hotkey, embedded for
    # hours, and only then said no.
    genesis = _assert_chain_allows(network, mock)
    # The protocol, its flip block and the v3 menu B block this image serves:
    # finney's are compiled into the release (the variables may only repeat
    # them there); another chain reads the variables (config._chain_block).
    # Before v0.6.0 the protocol came from FUGAL_V3_ACTIVATION_BLOCK alone, so
    # a compose that left it out kept the miner on the protocol before.
    from fugal_subnet import config as _chain_cfg
    try:
        protocol = _chain_cfg.protocol_for_chain(genesis)
        flip_block = (_chain_cfg.v4_activation_block(genesis) if protocol == 4
                      else _chain_cfg.v3_activation_block(genesis) if protocol == 3
                      else _chain_cfg.V2_ACTIVATION_BLOCK)
        menu_b_block = _chain_cfg.menu_b_block(genesis) if protocol == 3 else None
        # Protocol v4 (which no finney release schedules): the operator's
        # optional proposal, checked before any secret arrives or anything is
        # embedded.
        v4_spec = _parse_v4_proposal(v4_proposal) if protocol >= 4 else None
        v4_anchor_state = _load_v4_anchor(v4_anchor) if protocol >= 4 else None
        v4_usd = _parse_v4_challenge_usd(v4_challenge_usd) if protocol >= 4 else None
        # Under --await-provisioning a TD takes them from the push (below).
        v4_from = "provisioning" if await_provisioning else "options"
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    _assert_axon_ip_routable(os.getenv("FUGAL_AXON_IP"))

    if check:
        # --check: the head is checked only when given as a file; nothing is
        # provisioned.
        head_data = _load_head_file(head_path, protocol) if head_path else None
    elif await_provisioning:
        # The wallet arrives over the same channel: the TD signs serve_axon and
        # the head commitment itself, so the keyfile has to be inside it, and
        # a pushed wallet is the only path that never touches storage the cloud
        # can read. The returned path is a tmpfs directory laid out the way
        # bt.Wallet expects, so everything below is unchanged.
        pushed: dict = {}
        head_data, wallet_path = _await_provisioning(
            coldkey=coldkey, hotkey=hotkey, wallet_path=wallet_path, v4_options=pushed,
        )
        # Protocol v4: the operator's own options arrive with the push
        # (fugal_subnet/tee/v4_options.py), checked by the receiver already.
        try:
            v4_spec, v4_challenge, v4_usd, v4_from = _v4_pushed_options(
                pushed, protocol, env=(v4_spec, v4_challenge, v4_usd), v4_from=v4_from)
        except ValueError as e:
            raise click.ClickException(str(e)) from e
    elif head_path:
        head_data = _load_head_file(head_path, protocol)
    else:
        raise click.ClickException(
            "--head-path is required unless --await-provisioning is set"
        )
    if head_data is not None:
        _startup_head_check(head_data, protocol, mock=mock, menu_b_block=menu_b_block)
    weights_hash = hashlib.sha256(head_data or b"").hexdigest()

    # The pool is consensus state: the slice is drawn from it, so the miner and
    # the validator must derive it identically or every proof fails on
    # questions_hash for a reason that looks nothing like the cause. Default to
    # the same loader the validator calls; --benchmark-pool is an explicit
    # local override, and FUGAL_BENCHMARK_POOL overrides for both sides at once.
    # ONE code path, always. --benchmark-pool used to json.load the file here,
    # which meant the flag bypassed everything load_all() does — and when the
    # loader learned to drop questions the harness cannot grade, the validator
    # served 125 questions while the miner served 150 and every proof failed on
    # a questions_hash that named neither side. Routing the flag through the
    # same env var the validator honours makes a second path impossible rather
    # than merely discouraged. tests/test_pool_single_source.py enforces it.
    # Protocol v2 images set FUGAL_V2_ACTIVATION_BLOCK in their compose: the v2
    # image serves v2 only (v1 TDs keep running the v1 image until the flip), so
    # the miner needs no dual mode, only the hold below. Unset means v1.
    from fugal_subnet.benchmarks.loader import load_pool, pool_hash

    # A protocol v3 image serves v3 only, from its block (`protocol` and
    # `flip_block` above, from the compiled finney blocks).
    # A v4 image draws on the v3 pool (the pinned v4 mix names
    # pool_manifest_v3), as every v4 validator does.
    pool_protocol = _pool_protocol(protocol)
    if benchmark_pool:
        os.environ[f"FUGAL_BENCHMARK_POOL_V{pool_protocol}" if pool_protocol >= 2
                   else "FUGAL_BENCHMARK_POOL"] = os.path.abspath(benchmark_pool)
    pool = load_pool(pool_protocol)
    if not pool:
        raise click.ClickException("Benchmark pool is empty — nothing to benchmark")
    # A pool with code questions needs a harness that grades them with the v3
    # code grader and a working sandbox, or every code answer grades 0 and is
    # still paid for (fugal_subnet/code_gate.py).
    from fugal_subnet.code_gate import release_errors
    code_errors = release_errors(pool)
    if code_errors:
        msg = "This image cannot serve the pool's code questions: " + "; ".join(code_errors)
        if not mock:
            raise click.ClickException(msg)
        logger.warning("%s (mock: continuing)", msg)
    logger.info("Benchmark pool: %d questions, pool_hash=%s",
                len(pool), pool_hash(pool)[:16])
    # v3 pools hold code questions, graded in the code sandbox. Its self-test runs
    # once at start, so a machine that cannot build it says so now, not mid-epoch.
    from fugal_subnet.benchmarks.code_sets import code_sandbox_required
    if code_sandbox_required(pool):
        from fugal_subnet.tee.harness import code_sandbox_status
        ok, why = code_sandbox_status()
        if not ok and not mock:
            raise click.ClickException(
                f"The code sandbox self-test failed ({why}). This pool has code questions, which "
                "this machine cannot grade, so it would produce no proof for any epoch. Run the "
                "miner image on x86_64 Linux with seccomp available.")

    # The globally agreed model space. Exploration targets are drawn from this
    # sorted list, never from the miner's own head, so the samples the reference
    # frame is built from cannot be steered.
    from fugal_subnet.routing_protocol import menu_for_epoch, models_for_protocol, price_table
    if protocol >= 4:
        # v4: the entries and their prices are each epoch's menu, read by the
        # v4 session; the harness prices the proxy from it (td_v4.proxy_prices).
        prices, pool_models = None, []
    elif protocol >= 2:
        prices = price_table(protocol)
        pool_models = models_for_protocol(protocol)
    else:
        # FUGAL_MODEL_PRICES is a consensus override (refused on finney); the
        # default is the copy the protocol identity hashes, as on the validator.
        prices = load_prices(TEE_MODEL_PRICES_PATH) if TEE_MODEL_PRICES_PATH else price_table(1)
        pool_models = sorted(prices)
    explore_size = max(1, int(round(SLICE_SIZE * EXPLORE_FRACTION)))
    if protocol >= 4:
        logger.info("Protocol v4: the menu is read each epoch from the root's commitment; "
                    "proposal %s; challenge %s",
                    v4_spec.id if v4_spec else "none",
                    (f"{int(v4_challenge)} questions every epoch" if int(v4_challenge)
                     else "none: exploration only"))
    elif protocol >= 3:
        from fugal_subnet.config import EXPLORE_PRIVATE_SIZE, EXPLORE_SHARED_SIZE
        logger.info("Menu: %d entries; exploration %d shared + %d private calls/epoch",
                    len(pool_models), EXPLORE_SHARED_SIZE, EXPLORE_PRIVATE_SIZE)
    else:
        logger.info("Price table: %d models; exploration quota %d questions/epoch",
                    len(pool_models), explore_size)

    logger.info("Head loaded: %s (%d bytes)", head_path or "(pushed over the attested channel)",
                len(head_data or b""))
    logger.info("Weights hash: %s", weights_hash[:16])

    # Registered? Checked before the embedding (which can take hours) as well
    # as after it, where the UID is read for serving: a registration can be
    # lost while the pool embeds.
    _assert_registered(network, netuid, coldkey, hotkey, wallet_path)

    try:
        root_hotkeys = (_chain_cfg.root_hotkeys_v4(
            str(genesis or "").lower() == _chain_cfg.FINNEY_GENESIS_HASH.lower())
            if protocol >= 4 else ())
    except ValueError:
        root_hotkeys = ()           # named by the session's own refusal below
    ready, ready_text = _miner_ready_line(
        protocol, flip_block, v4_spec=v4_spec, v4_challenge=v4_challenge, v4_usd=v4_usd,
        v4_state_dir=v4_state_dir, v4_anchor=v4_anchor_state, pool_size=len(pool),
        root_hotkeys=root_hotkeys,
        v4_from=("pushed at provisioning" if check and v4_from == "provisioning" else v4_from))
    (logger.info if ready else logger.warning)(ready_text)
    if check:
        print(ready_text)
        raise SystemExit(0 if ready else 1)

    if protocol >= 3:
        # v3: the pool's embeddings ship with the image as one pinned file, the
        # same bytes every validator computes odds from. Nothing is embedded
        # at boot, and a file that is not the pinned one is refused.
        hidden_states = _load_pinned_embeddings(pool, mock=mock)
        logger.info("Embeddings loaded from the pinned file: %s", hidden_states.shape)
    else:
        # Once, at startup — not per epoch. See _compute_hidden_states.
        logger.info("Computing backbone embeddings for %d questions...", len(pool))
        hidden_states = _compute_hidden_states(pool)
        logger.info("Embeddings ready: %s (backbone released)", hidden_states.shape)

    wallet = bt.Wallet(name=coldkey, hotkey=hotkey, path=wallet_path)
    # A fresh connection: the embedding above can take hours, longer than an
    # idle websocket is worth trusting. Protocol v4 pins reads to epoch
    # boundaries (the root's menu commitment), as the validator does, so the
    # miner takes the same archive and fallback endpoints
    # (FUGAL_ARCHIVE_ENDPOINTS / FUGAL_FALLBACK_ENDPOINTS, finney's public
    # ones by default): a node that keeps ~300 blocks serves only the newest.
    subtensor = (bt.Subtensor(network=network, **_chain_endpoints(network, genesis))
                 if protocol >= 4 else bt.Subtensor(network=network))
    metagraph = subtensor.metagraph(netuid)

    my_hotkey = wallet.hotkey.ss58_address
    my_uid = _my_uid(metagraph, my_hotkey, netuid)
    logger.info("Miner UID %d on %s netuid %d", my_uid, network, netuid)
    blocks_per_epoch = blocks_per_epoch_fn(EPOCH_INTERVAL)
    if protocol >= 4:
        from fugal_subnet.memo_sync import history_line
        logger.info("Protocol v4 %s", history_line(subtensor, netuid, blocks_per_epoch))

    # Protocol v4: the TD's menu across epochs (td_v4.V4Session), read from
    # the root's commitment at each boundary and the bytes from the root's
    # store or a mirror, each checked against the hash and the compiled rules.
    v4_session = None
    if protocol >= 4:
        v4_session = _v4_session(
            subtensor, netuid=netuid, pool=pool, first_epoch=flip_block // blocks_per_epoch,
            blocks_per_epoch=blocks_per_epoch, coldkey=_coldkey_of(metagraph, my_uid),
            genesis=genesis,
            state_dir=_v4_state_dir(v4_state_dir), anchor=v4_anchor_state,
            challenge_size=int(v4_challenge), challenge_usd=v4_usd)

    # Protocol v5 (td_v5.V5Session): from its activation the TD labels the
    # exam cells the root's spec assigns it, plus coverage; the menu is still
    # the v4 session's, and the v4 challenge answers nothing.
    v5_session = None
    v5_block = _v5_activation_block(genesis) if protocol >= 4 else None
    if v5_block is not None:
        v5_session = _v5_session(
            subtensor, netuid=netuid, v4_session=v4_session,
            first_epoch=int(v5_block) // blocks_per_epoch, blocks_per_epoch=blocks_per_epoch,
            hotkey=my_hotkey, genesis=genesis)
        logger.info("Protocol v5 from block %d (epoch %d): exam labelling plus %d coverage calls "
                    "a epoch; the --v4-challenge options are not used from then on",
                    int(v5_block), int(v5_block) // blocks_per_epoch,
                    _td_v5_coverage_calls())

    # NOTE: the on-chain commitment is deliberately NOT made here. Chains rate
    # limit transactions per hotkey, so committing immediately before
    # axon.serve() gets the serve extrinsic rejected and leaves the miner
    # unreachable at 0.0.0.0 — earning nothing, with the commitment it did
    # land being worthless because no validator can query it. Reachability
    # first; the commitment is made after the axon is serving and retried each
    # epoch until it sticks.

    tee_runtime = TEERuntime(mock=mock)
    # Google's intermediates for this TD's attestation key, carried in every
    # v2 proof so a region (or a rotated intermediate) that validators have
    # not vendored still chains to the pinned root (launch step 2).
    tpm_intermediates = [] if mock else _tpm_intermediates_at_startup(tee_runtime)

    # The miner never writes RTMR3. dstack's initrd extends it during boot and
    # the validator replays that log; an extend from here would be missing
    # from the log and every honest proof would stop replaying (I8). What
    # binds this TD's application, upstream included, is the compose hash.
    if not mock:
        from fugal_subnet.tee.runtime import _OPENROUTER_BASE
        logger.info("Model upstream: %s", _OPENROUTER_BASE)
    # A live axon serves the full head to whoever may query it, and the
    # default admits only staked validators. Zero would admit any
    # registered hotkey, which is a free download of this router (diligence
    # 2026-10, finding 5), so it is refused rather than warned about.
    if not mock and MIN_VALIDATOR_STAKE <= 0:
        raise click.ClickException(
            "--live with FUGAL_MIN_VALIDATOR_STAKE=0 would serve this miner's head "
            "to any registered hotkey. Leave it at the default or set a positive "
            "threshold; a validator permit alone does not admit a caller."
        )

    current_proof = {
        "proof": None, "proof_json": "", "epoch_id": "", "lock": threading.Lock(),
    }

    mg_lock = threading.Lock()
    mg_state = {"metagraph": metagraph}

    def blacklist(synapse: FugalProofSynapse) -> Tuple[bool, str]:
        caller = getattr(synapse.dendrite, "hotkey", None)
        with mg_lock:
            mg = mg_state["metagraph"]
        allowed, reason = caller_allowed(mg, caller, live=not mock)
        return (not allowed), reason

    head_b64 = base64.b64encode(head_data).decode("ascii")

    def forward(synapse: FugalProofSynapse) -> FugalProofSynapse:
        return _serve_proof(synapse, current_proof, head_b64, weights_hash)

    # Protocol v4: the proposal document this box serves, whose sha256 it
    # commits beside its head (chain_proposals); only the root asks for it.
    proposal_doc = _proposal_document(v4_spec, my_hotkey) if protocol >= 4 else None

    def commit_now() -> bool:
        if protocol >= 4:
            return _ensure_fields(subtensor, wallet, netuid, weights_hash, proposal_doc)
        return ensure_commitment(subtensor, wallet, netuid, weights_hash)

    committed = False
    if protocol >= 2:
        # Hold until the flip (the v2 block for a v2 image, the v3 block for a v3 one). One hotkey holds ONE on-chain commitment, and the
        # v1 TD serving this hotkey until the flip is scored against it: a v2
        # commit during the notice window would make every v1 proof from this
        # hotkey fail. So: embed (done above), commit in the epoch BEFORE the
        # activation block so the commitment is on chain at the flip boundary,
        # and register the axon only from the activation block.
        from fugal_subnet.benchmarks.slicer import validate_activation_block
        validate_activation_block(flip_block, blocks_per_epoch)
        commit_at, commit_deadline = _v2_commit_window(flip_block, blocks_per_epoch)
        _wait_for_block(subtensor, commit_at,
                        f"commit the v{protocol} head (after the last boundary before the flip)")
        committed = _commit_before_deadline(
            subtensor, wallet, netuid, weights_hash, deadline=commit_deadline, protocol=protocol,
        )
        _wait_for_block(subtensor, flip_block, f"serve the axon (the v{protocol} flip)")

    external_ip = os.getenv("FUGAL_AXON_IP")   # routable: checked at startup
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
    if proposal_doc is not None:
        from fugal_subnet.protocol import FugalProposalSynapse

        def forward_proposal(synapse: FugalProposalSynapse) -> FugalProposalSynapse:
            return _serve_proposal(synapse, proposal_doc)

        def blacklist_proposal(synapse: FugalProposalSynapse) -> Tuple[bool, str]:
            caller = getattr(synapse.dendrite, "hotkey", None)
            with mg_lock:
                mg = mg_state["metagraph"]
            allowed, reason = caller_allowed(mg, caller, live=not mock)
            return (not allowed), reason

        axon.attach(forward_fn=forward_proposal, blacklist_fn=blacklist_proposal)

    for attempt in range(5):
        metagraph = _serve_axon_once(axon, subtensor, netuid)
        if _axon_registered(metagraph, my_uid):
            logger.info("Axon registered on chain: %s:%d", metagraph.axons[my_uid].ip, port)
            break
        if attempt < 4:
            logger.warning("Axon IP not registered yet, retrying in 15s...")
            time.sleep(15)
        else:
            logger.error(
                "axon.serve() failed after 5 attempts — the axon is still "
                "unregistered (0.0.0.0), so no validator can query this miner "
                "and it earns nothing until it is. It is retried every %ds from "
                "here on. Common causes: the port is unreachable from outside, "
                "or the chain is rate limiting this hotkey's transactions.",
                METAGRAPH_REFRESH_S,
            )

    axon.start()
    logger.info("Miner axon serving on port %d (mock=%s)", port, mock)
    _warn_if_unreachable(metagraph.axons[my_uid].ip, port)

    # Now that the axon is serving, commit the head hash. Retried in the loop
    # below if the chain rate-limits this one.
    if not committed or proposal_doc is not None:
        committed = commit_now()
    if not committed:
        logger.warning("Weights hash NOT committed on-chain yet — will retry.")

    try:
        last_refresh = time.time()
        last_epoch_index = -1
        consecutive_failures = 0
        while True:
            time.sleep(30)

            if not committed:
                committed = commit_now()

            if time.time() - last_refresh >= METAGRAPH_REFRESH_S:
                try:
                    fresh = subtensor.metagraph(netuid)
                    if not _axon_registered(fresh, my_uid):
                        # Startup gave up after five tries, or the chain lost
                        # the address. Without this the miner ran at 0.0.0.0
                        # for good, alive and earning nothing.
                        logger.warning("Axon not registered on chain (0.0.0.0); serving it again")
                        fresh = _serve_axon_once(axon, subtensor, netuid)
                        if _axon_registered(fresh, my_uid):
                            logger.info("Axon registered on chain: %s:%d",
                                        fresh.axons[my_uid].ip, port)
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

                start, deadline = _should_start_epoch(current_block, epoch_index, blocks_per_epoch)
                if not start:
                    from fugal_subnet.config import LATE_START_MARGIN_BLOCKS
                    logger.info("Skipping epoch %d: %d blocks to its collection block %d, fewer than %d",
                                epoch_index, deadline - current_block, deadline, LATE_START_MARGIN_BLOCKS)
                    last_epoch_index = epoch_index
                    continue

                last_epoch_index = epoch_index
                if protocol >= 4:
                    # v4: the epoch's menu, entries and prices are the
                    # session's (V4Session.inputs_for), not a pinned table.
                    epoch_menu = None
                elif protocol >= 3:
                    # The epoch's v3 menu: B from its block (the validator
                    # picks it the same way, from the epoch's boundary).
                    epoch_menu = menu_for_epoch(protocol, boundary_block, menu_b_block)
                    pool_models = models_for_protocol(protocol, epoch_menu)
                    prices = price_table(protocol, epoch_menu)
                else:
                    epoch_menu = None
                # A miner whose startup fetch of its Google intermediates
                # failed asks again here, so one bad minute at boot does not
                # cost it every epoch until it is restarted.
                tpm_intermediates = _tpm_intermediates_for_epoch(tpm_intermediates, tee_runtime, mock)
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
                    # v4's scored slice is its own compiled size, the one
                    # its pinned mix is built for (SEAT_MINIMAL_V4.md 8).
                    slice_size=SLICE_SIZE_V4 if protocol >= 4 else SLICE_SIZE,
                    proxy_port=TEE_PROXY_PORT,
                    explore_models=pool_models,
                    explore_size=explore_size,
                    hotkey=my_hotkey,
                    protocol_version=protocol,
                    menu=epoch_menu,
                    # The table this miner routes and explores over, in both
                    # protocols. v1 passed None, so the proxy priced with its
                    # default even when FUGAL_MODEL_PRICES (testnet only) had
                    # changed the list main() uses.
                    prices=prices,
                    tpm_intermediates=tpm_intermediates,
                    block_now=subtensor.get_current_block,
                    v4_session=v4_session,
                    v5_session=v5_session,
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


def _serve_proposal(synapse, document: bytes | None):
    """The axon's answer to the root's request for this miner's proposal
    document (protocol v4, chain_proposals): the document, when the synapse
    names the sha256 of the one this box serves; else the synapse as it
    came. The root checks the bytes by hash, so nothing else is decided here."""
    import hashlib

    if document is not None and synapse.proposal == hashlib.sha256(document).hexdigest():
        synapse.document = document.decode("ascii")
    return synapse


def _proposal_document(spec, hotkey: str) -> bytes | None:
    """This miner's proposal document (chain_proposals.ProposalDoc) for the
    operator's `spec` (v4_options.ProposalSpec), under its own hotkey, or
    None without a proposal."""
    if spec is None:
        return None
    from fugal_subnet import chain_proposals as cp

    return cp.document_bytes(cp.ProposalDoc(
        proposer=hotkey, model=spec.model, reasoning=tuple(spec.reasoning),
        provider_order=tuple(spec.provider_order), quantization=spec.quantization))


def _ensure_fields(subtensor, wallet, netuid: int, weights_hash: str,
                   document: bytes | None) -> bool:
    """Protocol v4: the head pointer, and with a proposal its pointer in the
    second field, one transaction (chain_proposals.set_commitment_fields),
    unless the chain already holds exactly those fields. Never raises."""
    from fugal_subnet import chain_proposals as cp

    want = [weights_hash] + ([cp.proposal_field(cp.proposal_hash(document))] if document else [])
    try:
        have = cp.read_fields_at(subtensor, netuid, subtensor.get_current_block()).get(
            wallet.hotkey.ss58_address, ())
        if list(have) == want:
            return True
    except Exception as e:  # noqa: BLE001 - the write below says whether it holds
        logger.warning("Could not read this hotkey's commitment fields (%s)", type(e).__name__)
    ok = cp.set_commitment_fields(subtensor, wallet, netuid, want)
    if ok:
        logger.info("Committed the head%s on chain", " and the proposal" if document else "")
    return ok


def _serve_proof(synapse, current_proof: dict, head_b64: str, weights_hash: str):
    """The axon's answer: this epoch's bundle on `synapse`, or `synapse` as it
    came when there is no proof, or the proof held is another epoch's."""
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


def _pool_protocol(protocol: int) -> int:
    """The pool an image of `protocol` serves: its own up to v3; v4 draws on
    the v3 pool (routing_data/pool_mix_v4.json names pool_manifest_v3)."""
    return 3 if int(protocol) >= 4 else int(protocol)


def _miner_ready_line(protocol, flip_block, *, v4_spec=None, v4_challenge="0", v4_usd=None,
                      v4_state_dir=None, v4_anchor=None, pool_size=0, root_hotkeys=(),
                      v4_from=None):
    """(ready, line): the protocol this image serves and, under v4, its inputs
    found or missing: the pinned seed and mix, the signed lab and host list,
    the memorisation check's published-set seed, the root's hotkey for the
    first v4 epoch, and the operator's options in force, with where they came
    from (`v4_from`: "options", the command line and environment;
    "provisioning", the operator's push; "pushed at provisioning", under
    --check in a TD, where nothing is pushed yet)."""
    import hashlib as _hashlib

    parts = [f"protocol v{protocol}" + (f" from block {flip_block}" if flip_block else ""),
             f"pool {pool_size}"]
    missing = []
    if protocol >= 4:
        from fugal_subnet import epoch_plan, memo_v4
        from fugal_subnet import menu_rules as mr
        from fugal_subnet import menu_state as ms
        from fugal_subnet.benchmarks.slicer import blocks_per_epoch as _bpe
        from fugal_subnet.config import EPOCH_INTERVAL

        try:
            seed = ms.load_seed()
            parts.append("seed " + _hashlib.sha256(ms.SEED_PATH.read_bytes()).hexdigest()[:8])
        except (OSError, ValueError) as e:
            seed = None
            missing.append(f"the pinned menu seed cannot be read ({e})")
        try:
            epoch_plan.mix_v4()
            with open(epoch_plan.MIX_V4_PATH, "rb") as f:
                parts.append("mix " + _hashlib.sha256(f.read()).hexdigest()[:8])
        except (OSError, ValueError) as e:
            missing.append(f"the pinned v4 mix cannot be read ({e})")
        labs = None
        try:
            mr.list_signers()
            try:
                labs = mr.shipped_list(mr.KIND_MENU_ALLOWLIST)
                parts.append(f"list serial {labs.serial}")
            except FileNotFoundError:
                labs = None
                parts.append("list none")
            if seed is not None and seed.labs.serial and (seed.labs.serial, seed.labs.sha256) \
                    != ((labs.serial, labs.sha256) if labs is not None else (0, "")):
                missing.append(f"the seed names lab and host list serial {seed.labs.serial}, "
                               "which this image does not ship")
        except (OSError, RuntimeError, mr.ListRefused) as e:
            missing.append(f"the signed lab and host list does not verify ({e})")
        try:
            parts.append("published seed " + memo_v4.seed_label())
        except (OSError, ValueError) as e:
            missing.append(f"the pinned published-set seed cannot be read ({e})")
        first = int(flip_block) // _bpe(EPOCH_INTERVAL) if flip_block else 0
        if not root_hotkeys or min(int(e) for e, _hk in root_hotkeys) > first:
            missing.append(f"no root hotkey is known for the first v4 epoch e{first:08d}")
        if v4_from == "pushed at provisioning":
            parts.append("proposal and challenge: pushed at provisioning")
        else:
            parts.append(f"proposal {v4_spec.id if v4_spec else 'none'}")
            if v4_spec is not None and labs is not None:
                # Informational, never a refusal: the root checks the list in
                # force at admission, which a newer signed list replaces
                # without a release (tee/v4_options.listed_refusal).
                from fugal_subnet.tee import v4_options

                why = v4_options.listed_refusal(v4_spec, labs)
                if why is not None:
                    parts.append(f"proposal not allowed by the shipped list serial {labs.serial} "
                                 f"({why}): admitted only under a list in force that allows it")
            parts.append(f"challenge {int(v4_challenge)}")
            parts.append(f"challenge budget ${_challenge_usd(v4_usd)} (realised spend, "
                         "estimate checked first)")
            if v4_from:
                parts.append(f"options from {v4_from}")
        parts.append(f"state {_v4_state_dir(v4_state_dir)}")
        if v4_anchor is not None:
            parts.append("anchor given")
    if missing:
        return False, "NOT READY: " + "; ".join(missing)
    return True, "READY: " + ", ".join(parts)


def _parse_v4_proposal(text):
    """The operator's v4 proposal (FUGAL_V4_PROPOSAL / --v4-proposal, or the
    `v4_proposal` field of a TD's push) as a v4_options.ProposalSpec, or None
    when unset. Strict (tee/v4_options.parse_proposal): a JSON object with
    exactly the four keys that passes every compiled entry rule not
    depending on the signed list; anything else is a startup refusal
    (ValueError), not a proposal silently dropped. The root judges it at
    admission, against the menu, the list in force and the listing."""
    if text is None or not str(text).strip():
        return None
    from fugal_subnet.tee import v4_options

    return v4_options.parse_proposal(str(text).strip())


def _v4_pushed_options(pushed, protocol, *, env, v4_from):
    """(proposal, challenge size, where they came from) for a TD, with the
    challenge's dollar budget before the last when `env` carries one
    ((proposal, size, budget or None)): the v4 options its operator's push
    carried (`pushed`, the fields the
    receiver accepted, tee/v4_options.FIELDS), parsed by the same parsers as
    the command line's. An option set both on the command line or in the
    environment (`env`: the values parsed from them) and in the push is a
    refusal (ValueError): one operator's choice has one source. Under a
    protocol before v4 the options decide nothing and are ignored, said in
    the log."""
    from fugal_subnet.tee import v4_options

    spec, challenge, *budget = env
    usd = budget[0] if budget else None

    def out(where):
        return (spec, challenge, usd, where) if budget else (spec, challenge, where)
    given = v4_options.pushed_options(pushed)
    if not given:
        return out(v4_from)
    if protocol < 4:
        logger.warning("The push carried protocol v4 options (%s); this image serves protocol "
                       "v%d, so they decide nothing", ", ".join(sorted(given)), protocol)
        return out(v4_from)
    both = [var for var, mine in (("FUGAL_V4_PROPOSAL", spec is not None),
                                  ("FUGAL_V4_CHALLENGE", int(challenge) != 0),
                                  ("FUGAL_V4_CHALLENGE_USD", usd is not None))
            if mine and var in given]
    if both:
        raise ValueError(f"{', '.join(both)} set both in this miner's options and in the "
                         "operator's push; a TD takes them from the push only")
    if "FUGAL_V4_PROPOSAL" in given:
        spec = _parse_v4_proposal(given["FUGAL_V4_PROPOSAL"])
    if "FUGAL_V4_CHALLENGE" in given:
        challenge = v4_options.parse_challenge(given["FUGAL_V4_CHALLENGE"])
    if "FUGAL_V4_CHALLENGE_USD" in given:
        usd = v4_options.parse_challenge_usd(given["FUGAL_V4_CHALLENGE_USD"])
    logger.info("Protocol v4 options from the operator's push: %s", ", ".join(sorted(given)))
    return out("provisioning")


def _challenge_usd(value) -> int:
    """The challenge's dollar budget in force: the operator's, or the
    compiled default (config.CHALLENGE_SPEND_USD_V4)."""
    from fugal_subnet import config as _cfg

    return int(_cfg.CHALLENGE_SPEND_USD_V4) if value is None else int(value)


def _parse_v4_challenge_usd(text):
    """The operator's challenge ceiling (FUGAL_V4_CHALLENGE_USD /
    --v4-challenge-usd), whole dollars, or None when unset (the compiled
    default). Strict (tee/v4_options.parse_challenge_usd); anything else is
    a startup refusal (ValueError)."""
    if text is None or not str(text).strip():
        return None
    from fugal_subnet.tee import v4_options

    return v4_options.parse_challenge_usd(str(text).strip())



def _v4_state_dir(text):
    """The TD's v4 state directory (--v4-state-dir / FUGAL_V4_STATE_DIR):
    `results/v4` under the working directory by default, which in the image
    is /app/results/v4 on the encrypted data volume."""
    return os.path.abspath(text if text and str(text).strip() else os.path.join("results", "v4"))


def _load_v4_anchor(path):
    """The operator's anchor menu (--v4-anchor / FUGAL_V4_ANCHOR), or None
    when unset: a file holding a menu's canonical bytes. A file that is not
    one is a startup refusal (ValueError). Its rules are checked by the
    session, like every menu's."""
    if path is None or not str(path).strip():
        return None
    from fugal_subnet import menu_state as ms

    try:
        with open(path, "rb") as f:
            raw = f.read(ms.MENU_MAX_BYTES + 1)
        return ms.from_json(raw)
    except OSError as e:
        raise ValueError(f"FUGAL_V4_ANCHOR: cannot read {path} ({type(e).__name__})") from None
    except ValueError as e:
        raise ValueError(f"FUGAL_V4_ANCHOR: {path} is not a menu ({e})") from None


def _chain_endpoints(network, genesis) -> dict:
    """archive_endpoints and fallback_endpoints for bt.Subtensor, as the
    validator sets them: FUGAL_ARCHIVE_ENDPOINTS and FUGAL_FALLBACK_ENDPOINTS,
    or finney's public ones on finney when unset."""
    from fugal_subnet import config as cfg
    from fugal_subnet import validator_setup

    net = validator_setup.network_for_genesis(genesis, network)
    archive, fallback, _ = validator_setup.default_endpoints(
        net, list(cfg.ARCHIVE_ENDPOINTS), list(cfg.FALLBACK_ENDPOINTS))
    if archive and not cfg.ARCHIVE_ENDPOINTS:
        cfg.ARCHIVE_ENDPOINTS[:] = archive     # pinned reads fall back to it (chain_reads)
    return {"archive_endpoints": archive or None, "fallback_endpoints": fallback or None}


def _coldkey_of(metagraph, uid: int):
    """This hotkey's owning coldkey as the chain records it, or None."""
    try:
        return str(metagraph.coldkeys[uid])
    except (AttributeError, IndexError, TypeError):
        return None


def _v4_holder_reader(subtensor, *, netuid: int, root_hk, blocks_per_epoch: int, fetch_reveal):
    """read_holder for td_v4.ChallengePolicy: (k, the seat holder's head
    hash) from the newest root reveal, of epoch k, the reveal being the one
    the root's on-chain commitment read pinned at `epoch`'s boundary names,
    its bytes checked against the committed sha256
    (td_v4.seat_holder_from_reveal). None when there is no commitment; the
    holder is None when the reveal names none; raises when the chain or the
    reveal cannot be read."""
    from fugal_subnet import menu_publish as mp
    from fugal_subnet import reveal_sync
    from fugal_subnet.commitments import get_commitments_with_blocks
    from fugal_subnet.tee import td_v4

    def read(epoch: int):
        hotkey = mp.hotkey_for_epoch(root_hk, epoch)
        data, _b = get_commitments_with_blocks(
            subtensor, netuid, block=int(epoch) * blocks_per_epoch).get(hotkey, ("", 0))
        got = reveal_sync.parse_commitment(data)
        if got is None:
            return None
        epoch_id, sha, _menu = got
        raw = fetch_reveal(epoch_id, sha)
        if raw is None:
            raise LookupError(f"the root's reveal {epoch_id} is not to be had")
        from fugal_subnet.memo_v4 import epoch_index
        return epoch_index(epoch_id), td_v4.seat_holder_from_reveal(raw, epoch_id, sha)
    return read


def _v4_session(subtensor, *, netuid: int, pool, first_epoch: int, blocks_per_epoch: int,
                coldkey, genesis, fetch_listing=None,
                anchor=None, state_dir=None, fetch_reveal=None, challenge_size=0,
                read_holder=None, holder_bytes=None, challenge_usd=None):
    """The miner's td_v4.V4Session, from what the chain and the public store
    serve:

    - the menu hash of each epoch: the root's commitment, read pinned at the
      epoch's boundary block (menu_publish.read_menu_commitment);
    - the menu bytes and the signed lists a menu names: the root's bucket and
      then FUGAL_MENU_MIRRORS (reveal_sync.fetch_menu_bytes /
      fetch_signed_list), each checked against the hash asked for; a list
      this release ships is taken from the image (menu_publish.ListBook);
    - the plan: epoch_plan.plan_v4 on `pool` (built by the session);
    - the root's join delta of each epoch: its second commitment field, read
      pinned at the boundary (menu_publish.read_join_pointer), the delta's
      bytes from the root's bucket or a mirror (reveal_sync.fetch_join_bytes);
    - the endpoint listing: runtime.fetch_endpoints_page (keyless, free);
    - where to start a walk (td_v4.V4Session): the menu kept in `state_dir`
      (menu.json), or the
      newest root reveal verified against the root's commitment read pinned
      at the epoch's boundary (menu_publish.read_root_anchor; the reveal
      from the root's bucket or a mirror, reveal_sync.fetch_reveal_bytes),
      or `anchor`, a menu the operator hands in as a last resort;
    - each epoch's challenge: td_v4.ChallengePolicy(`challenge_size`, its
      dollar budget `challenge_usd` or the compiled default), the
      holder read from the same root reveals (`_v4_holder_reader`, or
      `read_holder`) and its head bytes from the root's head publication
      (reveal_sync.fetch_head, or `holder_bytes`), checked in the policy
      against the holder's hash.

    The root's hotkeys and bucket are the release's on finney and
    FUGAL_ROOT_HOTKEY / FUGAL_REFERENCE_BUCKET elsewhere (as a follower
    validator reads them). Without a root hotkey no menu commitment can be
    read, so the miner refuses to start. Without `state_dir` the session
    keeps nothing on disk (tests)."""
    from fugal_subnet import config as cfg
    from fugal_subnet import menu_publish as mp
    from fugal_subnet import menu_rules as mr
    from fugal_subnet import menu_state as ms
    from fugal_subnet import reveal_sync
    from fugal_subnet.tee import td_v4

    on_finney = genesis is not None and str(genesis).lower() == cfg.FINNEY_GENESIS_HASH.lower()
    try:
        # The root's hotkeys by epoch: each epoch's commitment is read under
        # the hotkey that was the root's for it (config.root_hotkeys_v4).
        root_hk = cfg.root_hotkeys_v4(on_finney)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    if not root_hk:
        raise click.ClickException(
            "protocol v4 reads each epoch's menu from the root validator's commitment: set "
            "FUGAL_ROOT_HOTKEY to the root's hotkey")
    bucket = cfg.REFERENCE_BUCKET or (reveal_sync.FINNEY_REFERENCE_BUCKET if on_finney else "")
    if not bucket and not cfg.MENU_MIRRORS:
        logger.warning("No reveal bucket (FUGAL_REFERENCE_BUCKET) and no menu mirror "
                       "(FUGAL_MENU_MIRRORS): a menu that changes cannot be read, and an epoch "
                       "on it gets no proof")
    bucket = bucket or None
    lists = mp.ListBook(None, fetch=lambda kind, sha: reveal_sync.fetch_signed_list(
        kind, sha, bucket=bucket, netuid=netuid))

    def signed(kind):
        def get(serial, sha):
            if lists.get(kind, ms.AllowlistRef(int(serial), str(sha))) is None:
                return None
            return lists.raw(kind, sha)
        return get

    if fetch_listing is None:
        from fugal_subnet.tee.runtime import fetch_endpoints_page as fetch_listing
    if fetch_reveal is None:
        def fetch_reveal(epoch_id, sha):
            return reveal_sync.fetch_reveal_bytes(epoch_id, sha, bucket=bucket, netuid=netuid)
    # The newest root reveal is read each epoch for the menu anchor, the seat
    # holder and the published set: its bytes are fetched once.
    fetch_reveal = _kept_reveals(fetch_reveal)
    if state_dir is not None:
        logger.info("Protocol v4 state (the last menu accepted): %s", state_dir)
    if read_holder is None:
        read_holder = _v4_holder_reader(subtensor, netuid=netuid, root_hk=root_hk,
                                        blocks_per_epoch=blocks_per_epoch,
                                        fetch_reveal=fetch_reveal)
    if holder_bytes is None:
        def holder_bytes(sha):
            return reveal_sync.fetch_head(bucket, netuid, sha) if bucket else None
    try:
        policy = td_v4.ChallengePolicy(int(challenge_size), first_epoch=first_epoch,
                                       read_holder=read_holder, holder_bytes=holder_bytes,
                                       spend_usd=_challenge_usd(challenge_usd))
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    published = _v4_published(subtensor, netuid=netuid, pool=pool, first_epoch=first_epoch,
                              blocks_per_epoch=blocks_per_epoch, root_hk=root_hk, bucket=bucket,
                              fetch_reveal=fetch_reveal, state_dir=state_dir)
    return td_v4.V4Session(
        first_epoch=first_epoch, pool=pool, fetch_listing=fetch_listing,
        read_commitment=lambda e: mp.read_menu_commitment(subtensor, netuid, root_hk, e,
                                                          e * blocks_per_epoch),
        read_anchor=lambda e: mp.read_root_anchor(subtensor, netuid, root_hk, e,
                                                  e * blocks_per_epoch, fetch_reveal=fetch_reveal),
        menu_bytes=lambda h: reveal_sync.fetch_menu_bytes(h, bucket=bucket, netuid=netuid),
        list_bytes=signed(mr.KIND_MENU_ALLOWLIST), device_list_bytes=signed(mr.KIND_DEVICE_LIST),
        anchor=anchor, state_dir=state_dir, challenge=policy, published=published,
        read_join=lambda e: mp.read_join_pointer(subtensor, netuid, root_hk, e,
                                                 e * blocks_per_epoch),
        join_bytes=lambda h: reveal_sync.fetch_join_bytes(h, bucket=bucket, netuid=netuid))


def _v5_activation_block(genesis):
    """The block protocol v5 applies from on this chain, or None (config's
    v5_activation_block, which lane V1 adds alongside V5_ACTIVATION_BLOCK)."""
    from fugal_subnet import config as cfg
    fn = getattr(cfg, "v5_activation_block", None)
    if callable(fn):
        return fn(genesis)
    value = getattr(cfg, "V5_ACTIVATION_BLOCK", None)
    return None if value is None else int(value)


def _td_v5_coverage_calls() -> int:
    from fugal_subnet.tee import td_v5
    return td_v5.COVERAGE_CALLS


def _v5_spec_reader(subtensor, *, netuid: int, root_hk, blocks_per_epoch: int, fetch_reveal):
    """read_spec for td_v5.ExamPolicy: (k, the exam spec) from the newest
    root reveal, of epoch k, the one the root's on-chain commitment read
    pinned at `epoch`'s boundary names, its bytes checked against the
    committed sha256 (td_v5.exam_spec_from_reveal). None when there is no
    commitment; raises when the chain or the reveal cannot be read."""
    from fugal_subnet import menu_publish as mp
    from fugal_subnet import reveal_sync
    from fugal_subnet.commitments import get_commitments_with_blocks
    from fugal_subnet.memo_v4 import epoch_index
    from fugal_subnet.tee import td_v5

    def read(epoch: int):
        hotkey = mp.hotkey_for_epoch(root_hk, epoch)
        data, _b = get_commitments_with_blocks(
            subtensor, netuid, block=int(epoch) * blocks_per_epoch).get(hotkey, ("", 0))
        got = reveal_sync.parse_commitment(data)
        if got is None:
            return None
        epoch_id, sha, _menu = got
        raw = fetch_reveal(epoch_id, sha)
        if raw is None:
            raise LookupError(f"the root's reveal {epoch_id} is not to be had")
        return epoch_index(epoch_id), td_v5.exam_spec_from_reveal(raw, epoch_id, sha)
    return read


def _v5_session(subtensor, *, netuid: int, v4_session, first_epoch: int, blocks_per_epoch: int,
                hotkey: str, genesis, fetch_reveal=None, read_spec=None):
    """The miner's td_v5.V5Session: `v4_session` for the menu (its challenge
    gated off from `first_epoch`, td_v5.gate_challenge), and an exam policy
    that reads each epoch's spec from the root's reveals as the v4 holder
    reader does (`_v5_spec_reader`, or `read_spec`)."""
    from fugal_subnet import config as cfg
    from fugal_subnet import reveal_sync
    from fugal_subnet.tee import td_v5

    if v4_session is None:
        raise click.ClickException("protocol v5 reads its menu through the v4 session: this "
                                   "image serves protocol v4 first")
    v4_session.challenge = td_v5.gate_challenge(v4_session.challenge, first_epoch)
    if read_spec is None:
        on_finney = (genesis is not None
                     and str(genesis).lower() == cfg.FINNEY_GENESIS_HASH.lower())
        root_hk = cfg.root_hotkeys_v4(on_finney)
        bucket = (cfg.REFERENCE_BUCKET
                  or (reveal_sync.FINNEY_REFERENCE_BUCKET if on_finney else "")) or None
        if fetch_reveal is None:
            def fetch_reveal(epoch_id, sha):
                return reveal_sync.fetch_reveal_bytes(epoch_id, sha, bucket=bucket,
                                                      netuid=netuid)
        read_spec = _v5_spec_reader(subtensor, netuid=netuid, root_hk=root_hk,
                                    blocks_per_epoch=blocks_per_epoch,
                                    fetch_reveal=_kept_reveals(fetch_reveal))
    policy = td_v5.ExamPolicy(first_epoch=first_epoch, hotkey=hotkey, read_spec=read_spec)
    return td_v5.V5Session(v4_session=v4_session, exam=policy, first_epoch=first_epoch)


def _kept_reveals(fetch_reveal, keep: int = 4):
    """`fetch_reveal` with the last few reveals' bytes kept by (epoch id,
    sha256): the same verified bytes serve every reader in the epoch."""
    kept: dict = {}

    def get(epoch_id, sha):
        key = (epoch_id, sha)
        if key not in kept:
            raw = fetch_reveal(epoch_id, sha)
            if raw is None:
                return None
            kept[key] = raw
            for old in list(kept)[:-keep]:
                del kept[old]
        return kept[key]
    return get


def _v4_published(subtensor, *, netuid: int, pool, first_epoch: int, blocks_per_epoch: int,
                  root_hk, bucket, fetch_reveal, state_dir=None):
    """The TD's published set (memo_sync.ChainSet), advanced as every node
    advances it: the root's reveal commitment read pinned at each boundary
    block, a reveal it newly names fetched by its committed sha256 (the
    root's bucket or a mirror), a snapshot by digest. Kept in `state_dir`
    (memo_sync.TD_FILE). A reveal whose named view contradicts the TD's own
    state stops it (no proof). Nothing an operator hands in is read."""
    import hashlib
    import os

    from fugal_subnet import memo_sync, memo_v4, reveal_sync
    from fugal_subnet import menu_publish as mp
    from fugal_subnet.commitments import get_commitments_with_blocks

    def read_at(epoch: int):
        hotkey = mp.hotkey_for_epoch(root_hk, epoch)
        data, _b = get_commitments_with_blocks(
            subtensor, netuid, block=int(epoch) * blocks_per_epoch).get(hotkey, ("", 0))
        return data or None

    def snapshot(digest: str):
        return reveal_sync.fetch_object(
            memo_sync.snapshot_object(netuid, digest), bucket=bucket,
            limit=memo_v4.MAX_SNAPSHOT_BYTES,
            check=lambda b: hashlib.sha256(b).hexdigest() == digest)

    path = None if state_dir is None else os.path.join(os.fspath(state_dir), memo_sync.TD_FILE)
    return memo_sync.ChainSet(pool=pool, read_at=read_at, reveal_bytes=fetch_reveal,
                              snapshot=snapshot, path=path, refuse=True)


def _assert_chain_allows(network: str, mock: bool):
    """Refuse what this chain forbids: --mock on finney (mock accepts
    unattested proofs), and a consensus constant overridden in the environment
    (refused on finney, warned about elsewhere, as on the validator). Keyed on
    the chain's genesis hash, so renaming the network does not get around it.
    The miner keeps --mock as its default because --live spends real money on
    inference; what it must never do is run mock on mainnet.

    Returns the chain's genesis hash (None when it could not be read), from
    which main() reads the release's compiled blocks for this chain.
    """
    import bittensor as bt

    from fugal_subnet.fingerprint import (
        assert_consensus_overrides_allowed,
        assert_mock_allowed,
        chain_genesis_hash,
    )

    subtensor = bt.Subtensor(network=network)
    try:
        genesis = chain_genesis_hash(subtensor)
        assert_mock_allowed(genesis, mock)
        assert_consensus_overrides_allowed(genesis)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e
    finally:
        try:
            subtensor.close()
        except Exception:  # noqa: BLE001 - closing a check connection is best effort
            pass
    return genesis


def _axon_registered(metagraph, uid: int) -> bool:
    """Does the chain hold a routable address for this UID's axon?"""
    try:
        ip = metagraph.axons[uid].ip
    except (AttributeError, IndexError):
        return False
    return ip not in ("0.0.0.0", "", None)


def _serve_axon_once(axon, subtensor, netuid: int):
    """One serve_axon attempt; returns the metagraph read after it. A failed
    attempt is logged, never raised: the caller decides whether to retry."""
    try:
        axon.serve(netuid=netuid, subtensor=subtensor)
    except Exception as e:  # noqa: BLE001 - reported; the registered address is the verdict
        logger.warning("axon.serve() failed: %s", e)
    return subtensor.metagraph(netuid)


def _assert_axon_ip_routable(external_ip: str | None) -> None:
    """Refuse an FUGAL_AXON_IP the chain would reject, at startup.

    The chain rejects loopback and unspecified addresses for serve_axon at pool
    validation (Custom error 11), before the extrinsic executes. bt.Axon.serve()
    swallows that, so the only symptom is an axon that stays at 0.0.0.0 and a
    miner that silently earns nothing. A setting that guarantees that is a
    startup error, and it is static, so it is checked before the operator's
    secrets arrive and before hours of embedding, not when the axon is served.
    """
    if external_ip and _is_unroutable(external_ip):
        raise click.ClickException(
            f"FUGAL_AXON_IP={external_ip} is not routable: the chain rejects "
            "serve_axon for it and no validator could ever reach this miner. Set "
            "it to an address validators can connect to."
        )


def _assert_registered(network: str, netuid: int, coldkey: str, hotkey: str,
                       wallet_path) -> None:
    """Refuse an unregistered hotkey before embedding the pool (hours), on a
    connection that is closed again rather than kept idle through it."""
    import bittensor as bt

    wallet = bt.Wallet(name=coldkey, hotkey=hotkey, path=wallet_path)
    subtensor = bt.Subtensor(network=network)
    try:
        _my_uid(subtensor.metagraph(netuid), wallet.hotkey.ss58_address, netuid)
    finally:
        close = getattr(subtensor, "close", None)
        if callable(close):
            try:
                close()
            except Exception as e:  # noqa: BLE001 - closing an idle socket
                logger.debug("closing the pre-embedding connection: %s", e)


def _my_uid(metagraph, hotkey_ss58: str, netuid: int) -> int:
    """This miner's UID, or a startup error. Not registered used to log and
    return, so the process exited 0 and a supervisor saw a clean stop."""
    if hotkey_ss58 not in metagraph.hotkeys:
        raise click.ClickException(
            f"hotkey {hotkey_ss58} is not registered on netuid {netuid}; register it "
            "(scripts/provision_subnet.py --miners ...) before starting the miner"
        )
    return metagraph.hotkeys.index(hotkey_ss58)


def _should_start_epoch(current_block: int, epoch_index: int,
                        blocks_per_epoch: int) -> tuple[bool, int]:
    """(start, collection block). Don't start an epoch that can't finish: a
    proof that misses the collection block is a miss, and its calls are paid
    for anyway (seen at a cutover: three runs started after their collection
    block). Start only with at least LATE_START_MARGIN_BLOCKS to go."""
    from fugal_subnet.benchmarks.slicer import collect_block_for_epoch
    from fugal_subnet.config import EPOCH_COLLECT_FRACTION, LATE_START_MARGIN_BLOCKS

    deadline = collect_block_for_epoch(epoch_index, blocks_per_epoch, EPOCH_COLLECT_FRACTION)
    return current_block <= deadline - LATE_START_MARGIN_BLOCKS, deadline


def _log_readiness(epoch_id: str, ready_block: int, collect_block: int) -> None:
    """Say whether this epoch's proof was ready when validators collect.

    "My miner earns nothing" is usually "my proof was late", and that is
    invisible unless the miner says so.
    """
    if ready_block <= collect_block:
        logger.info("Epoch %s: proof ready at block %d, %d blocks before collection (%d)",
                    epoch_id, ready_block, collect_block - ready_block, collect_block)
    else:
        logger.warning("Epoch %s: proof LATE by %d blocks (ready at %d, collected at %d). "
                       "Validators will not see it; cut the slice's cost or latency.",
                       epoch_id, ready_block - collect_block, ready_block, collect_block)


def _v2_commit_window(activation_block: int, blocks_per_epoch: int) -> tuple[int, int]:
    """(commit_at, deadline) for the v2 head's on-chain commitment.

    One hotkey holds ONE commitment, and validators read it at each epoch's
    BOUNDARY block. So the v2 head is committed just after the last v1
    boundary (that v1 epoch still checks its proof against the old head, and
    the flip boundary sees the new one), COMMIT_AFTER_BOUNDARY_BLOCKS clear of
    the boundary block itself, and by COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS
    before the activation block so it is on chain when the flip boundary is
    read.
    """
    from fugal_subnet.config import COMMIT_AFTER_BOUNDARY_BLOCKS, COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS

    last_v1_boundary = int(activation_block) - int(blocks_per_epoch)
    return (last_v1_boundary + COMMIT_AFTER_BOUNDARY_BLOCKS,
            int(activation_block) - COMMIT_DEADLINE_BEFORE_FLIP_BLOCKS)


def _wait_for_block(subtensor, block: int, purpose: str, poll_s: float = 30.0) -> None:
    """Block until the chain reaches `block`, logging what the miner is waiting to do.

    A failed read is logged and retried: miners start days before the flip, and
    one dropped connection must not end the process (the container would
    restart without its provisioned secrets and miss the flip)."""
    while True:
        try:
            current = subtensor.get_current_block()
        except Exception as e:  # noqa: BLE001 - a dropped socket must not end the hold
            logger.warning("Could not read the current block (%s); retrying", e)
            time.sleep(poll_s)
            continue
        if current >= block:
            return
        logger.info("Holding: %d blocks until block %d, then %s", block - current, block, purpose)
        time.sleep(poll_s)


def _commit_before_deadline(subtensor, wallet, netuid, weights_hash, deadline: int,
                            retry_blocks: int | None = None, poll_s: float = 6.0,
                            protocol: int | None = None) -> bool:
    """Commit the head, retrying every `retry_blocks` blocks until block `deadline`.

    Returns True once the chain holds the commitment. A False return means the
    deadline passed: the post-flip fallback will still commit, but validators read
    commitments at the flip boundary, so this miner's first v2 epoch(s) will be
    rejected as committed late.
    """
    from fugal_subnet.commitments import ensure_commitment
    from fugal_subnet.config import COMMIT_RETRY_BLOCKS
    retry_blocks = COMMIT_RETRY_BLOCKS if retry_blocks is None else retry_blocks
    def block_now():
        while True:
            try:
                return subtensor.get_current_block()
            except Exception as e:
                logger.warning("Could not read the current block (%s); retrying", e)
                time.sleep(poll_s)

    attempt = 0
    while True:
        current = block_now()
        if current >= deadline and protocol is not None and protocol >= 4:
            # Under v4 a head is read at each epoch's boundary: a commit after
            # the flip (a miner restarted after it) costs nothing.
            logger.info("The v%d head is committed after block %d (now %d): it counts from the "
                        "first boundary that reads it", protocol, deadline, current)
            return False
        if current >= deadline:
            logger.error(
                "v2 head NOT committed by block %d (now %d) after %d attempt(s). The commit "
                "falls back to after the flip, and validators will reject this miner's first "
                "v2 epoch(s) as committed late. Check the hotkey's chain access and balance.",
                deadline, current, attempt,
            )
            return False
        attempt += 1
        if ensure_commitment(subtensor, wallet, netuid, weights_hash):
            if attempt > 1:
                logger.info("v2 head committed on attempt %d at block %d", attempt, current)
            return True
        retry_at = current + retry_blocks
        logger.warning(
            "v2 head commit attempt %d failed at block %d; retrying at block %d "
            "(deadline %d, %d blocks left)", attempt, current, retry_at, deadline, deadline - current,
        )
        while block_now() < retry_at:
            time.sleep(poll_s)


def _tpm_intermediates_at_startup(tee_runtime) -> list:
    """The intermediates this TD's AK needs, from one startup attestation.

    Never stops the miner: a TD with no TPM half (a bare quote), no AIA, or a
    failed fetch logs why and returns [], and its proofs then verify only where
    a validator's vendored intermediates cover them, as before.
    """
    from fugal_subnet.scale import decode_dstack_attestation
    from fugal_subnet.tee.tpm import intermediates_for

    why = ("proofs will verify only where validators have vendored this TD's "
           "intermediate (us-central1 today)")
    try:
        blob = tee_runtime.generate_attestation(b"\x00" * 32)
        if int.from_bytes(blob[:2], "little") in (4, 5):
            logger.info("Bare TDX quote (no TPM half): no TPM intermediates to carry")
            return []
        ak_cert = decode_dstack_attestation(blob)["tpm"]["ak_cert"]
        chain = intermediates_for(ak_cert)
    except Exception as e:  # noqa: BLE001 - never stops the miner, whatever the TD or network returns
        logger.warning("Could not build the TPM intermediates (%s: %s); %s", type(e).__name__, e, why)
        return []
    if chain:
        logger.info("Carrying %d TPM intermediate certificate(s) in each proof", len(chain))
    else:
        logger.warning("No TPM intermediate chain could be built for this TD's key; %s", why)
    return chain


def _tpm_intermediates_for_epoch(have: list, tee_runtime, mock: bool) -> list:
    """The intermediates this epoch's proof carries: the chain already built,
    or one more attempt at building it when there is none yet.

    Outside the vendored set a proof without its chain is refused by every
    validator, and the chain used to be built once: a single failed fetch at
    boot left the miner producing refused proofs for the life of the process.
    """
    if mock or have:
        return have
    return _tpm_intermediates_at_startup(tee_runtime)


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
    hotkey,
    protocol_version=1,
    prices=None,
    block_now=None,
    menu=None,
    tpm_intermediates=(),
    v4_session=None,
    v5_session=None,
):
    """Run a single benchmark epoch; `block_now` (the chain's current block)
    lets it report whether the proof was ready by the collection block.

    Protocol v4 (`protocol_version` 4, which no release activates yet) takes
    `v4_session` (fugal_subnet.tee.td_v4.V4Session): the epoch's menu inputs
    come from it, the harness checks them before any call, and a menu the TD
    cannot read or the rules refuse ends the epoch with no proof.

    Protocol v5: with `v5_session` (td_v5.V5Session), an epoch at or after
    its first epoch runs the v5 path (the exam cells the root's spec assigns
    this hotkey, plus coverage); the menu is still the v4 session's."""
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

    v4_inputs = None
    v5_inputs = None
    if v5_session is not None and v5_session.is_v5(epoch_index):
        # Before the proxy starts, as under v4: no readable menu, no proof.
        protocol_version = max(int(protocol_version), 5)
        v5_inputs = v5_session.inputs_for(epoch_index)
    elif protocol_version >= 4:
        if v4_session is None:
            raise ValueError("a protocol v4 epoch needs the miner's v4 session")
        # Before the proxy starts: a menu this TD cannot read means no proof,
        # and nothing is called (the harness checks the menu again).
        v4_inputs = v4_session.inputs_for(epoch_index)
    proxy = tee_runtime.setup(proxy_port=proxy_port, prices=prices)
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
            # Binds the proof to THIS miner. Without it a proof is bound to an
            # epoch, a slice, a head and a runtime, but to nobody — and can be
            # read off this axon and presented by anyone.
            hotkey=hotkey,
            protocol_version=protocol_version,
            menu=menu,
            **({"v4": v4_inputs} if v4_inputs is not None else {}),
            **({"v5": v5_inputs} if v5_inputs is not None else {}),
        )
        if v4_inputs is not None:
            v4_session.ran(v4_inputs, proof)
        if v5_inputs is not None:
            v5_session.ran(v5_inputs, proof)

        # Before the content hash, which binds them (v2 proofs only: a v1
        # proof stays byte-identical for a validator not yet upgraded).
        if protocol_version >= 2 and tpm_intermediates:
            proof.tpm_intermediates = list(tpm_intermediates)
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

        if v5_inputs is not None:
            n_exam = sum(1 for r in proof.results if r.exam)
            n_cut = sum(1 for r in proof.results if r.not_started)
            logger.info("Epoch %s complete (v5): %d exam cells (spec %s), %d coverage calls, "
                        "%d failed, %d not started, cost=$%.4f", epoch_id, n_exam,
                        (proof.spec_hash or "none: coverage only")[:16],
                        len(proof.results) - n_exam, sum(1 for r in proof.results if r.failed),
                        n_cut, proof.total_cost_usd)
        elif proof.scored is False:
            # Protocol v4 with no challenge this epoch (--v4-challenge).
            logger.info("Epoch %s complete: exploration only (%d calls, no challenge), "
                        "cost=$%.4f", epoch_id, len(proof.results),
                        proof.total_cost_usd)
        else:
            logger.info(
                "Epoch %s complete: %d/%d correct (%.1f%%), cost=$%.4f",
                epoch_id, proof.n_correct, proof.n_total,
                proof.accuracy * 100, proof.total_cost_usd,
            )
        if block_now is not None:
            try:
                _log_readiness(epoch_id, block_now(), collect_block)
            except Exception as e:  # noqa: BLE001 - a report, never a reason to drop the proof
                logger.debug("Could not read the block to report readiness: %s", e)
    finally:
        proxy.stop()


def _await_provisioning(coldkey: str = "default", hotkey: str = "default",
                        wallet_path: "str | None" = None, v4_options: "dict | None" = None):
    """Block until the operator pushes per-miner data; return (head, wallet_path).

    `v4_options`, when given, receives the protocol v4 option fields the push
    carried (tee/v4_options.FIELDS; the receiver checked each), by field.

    THE AXON DOES NOT SERVE UNTIL THIS RETURNS, and that is the point. An
    unprovisioned miner is not partially ready: it would answer a validator with
    no proof, spend that validator's query budget to learn nothing, and read as
    a dead miner rather than a starting one. Blocking here fails loudly in the
    operator's own log — "waiting to be provisioned" is a sentence someone can
    act on, and silent emptiness is the failure shape this project keeps
    finding.

    The API key is put into the environment rather than passed down because
    MeteringProxy already reads it from there, and threading a secret through
    six call frames creates six places it can be logged.

    THE WALLET. If the push carries `hotkey_keyfile_b64` (and `coldkeypub_b64`,
    which the SDK's serve_axon reads for the coldkey address), both are written
    under a tmpfs wallet root laid out as bt.Wallet expects —
    `<root>/<coldkey>/hotkeys/<hotkey>` and `<root>/<coldkey>/coldkeypub.txt`,
    mode 0600 — and that root is returned as the wallet path. tmpfs, not the
    data volume: a key that outlived the attestation which justified releasing
    it would still be there for a later boot with a different measurement. The
    default root is /dev/shm; FUGAL_PROVISIONED_WALLET_DIR overrides it. When
    the push carries no keyfile the caller's own wallet_path is returned
    unchanged, so a TD that already holds a wallet by some other means keeps
    working.
    """
    import base64
    import os
    import time

    from fugal_subnet.tee.provision import PROVISION_PORT, ProvisionStore, serve

    store = ProvisionStore()
    server = serve(store, port=PROVISION_PORT)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    logger.info(
        "Waiting to be provisioned on port %d. The operator must verify this "
        "TD's attestation and push the head and API key; nothing is served "
        "until they do.", PROVISION_PORT,
    )
    waited = 0
    while not store.ready:
        time.sleep(1)
        waited += 1
        if waited % 30 == 0:
            logger.info("still waiting to be provisioned (%ds)", waited)

    head_b64 = store.get("head_b64")
    if not head_b64:
        raise click.ClickException(
            "provisioned without a head — the miner has nothing to benchmark"
        )
    key = store.get("openrouter_api_key")
    if key:
        os.environ["OPENROUTER_API_KEY"] = key
        logger.info("API key received over the attested channel")

    keyfile_b64 = store.get("hotkey_keyfile_b64")
    if keyfile_b64:
        root = os.getenv("FUGAL_PROVISIONED_WALLET_DIR", "/dev/shm/fugal-wallet")
        wallet_dir = os.path.join(root, coldkey)
        hotkeys_dir = os.path.join(wallet_dir, "hotkeys")
        os.makedirs(hotkeys_dir, mode=0o700, exist_ok=True)
        os.chmod(root, 0o700)
        os.chmod(wallet_dir, 0o700)
        _write_private(os.path.join(hotkeys_dir, hotkey), base64.b64decode(keyfile_b64))
        coldkeypub_b64 = store.get("coldkeypub_b64")
        if coldkeypub_b64:
            _write_private(os.path.join(wallet_dir, "coldkeypub.txt"),
                           base64.b64decode(coldkeypub_b64))
        else:
            logger.warning(
                "hotkey keyfile received without coldkeypub — the SDK reads the "
                "coldkey address to serve an axon, so serving will fail unless "
                "the wallet dir already holds coldkeypub.txt",
            )
        wallet_path = root
        # Names only. The keyfile is a secret; the path says where, not what.
        logger.info("Wallet received over the attested channel: %s/%s at %s",
                    coldkey, hotkey, root)
    if v4_options is not None:
        from fugal_subnet.tee.v4_options import FIELDS as _V4_FIELDS

        v4_options.update({f: store.get(f) for f in _V4_FIELDS.values() if store.get(f)})
    logger.info("Provisioned after %ds; the axon may now serve", waited)
    return base64.b64decode(head_b64), wallet_path


def _write_private(path: str, data: bytes) -> None:
    """Write a keyfile so it is never world-readable, even for an instant."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


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

    from fugal_subnet.config import BACKBONE_BATCH_SIZE, BACKBONE_MODEL
    from fugal_subnet.vendor import success_contract as contract

    # The embedded TEXT, by the contract's own cache_key (the same key load_cache
    # checks), not pool_hash: pool_hash covers question ids only, so a pool fix
    # that changed a prompt under the same id kept the path and every miner's
    # cache then failed load_cache's check.
    key = hashlib.sha256("|".join([
        contract.cache_key([q["prompt"] for q in pool]),
        BACKBONE_MODEL,
        contract.PROFILE_ID,
        str(BACKBONE_BATCH_SIZE),
        platform.machine(),
    ]).encode("utf-8")).hexdigest()[:32]
    root = os.getenv("FUGAL_EMBEDDING_CACHE", os.path.join("data", "embeddings"))
    return os.path.join(root, f"hidden-{key}.npz")


def _load_pinned_embeddings(pool, mock: bool = False):
    """The pool's embeddings from the pinned file (protocol v3), one row per
    pool question in pool order. Live refuses a file that is not pinned or
    does not match the pin; mock may load an unpinned one for a rehearsal."""
    from fugal_subnet.config import EMBEDDINGS_SHA256
    from fugal_subnet.embeddings_file import Embeddings, embeddings_pin

    if not EMBEDDINGS_SHA256 and not mock:
        raise click.ClickException(
            "config.EMBEDDINGS_SHA256 is not pinned: a v3 miner routes only from the pinned "
            "embeddings file")
    path = os.getenv("FUGAL_EMBEDDINGS_FILE", os.path.join("data", "embeddings", "pool_v3.npz"))
    try:
        # The pin, unless the pool itself is declared unpinned (never on finney).
        return Embeddings.load(path, embeddings_pin()).aligned([q["question_id"] for q in pool])
    except (OSError, ValueError, KeyError) as e:
        raise click.ClickException(f"embeddings file {path}: {e}") from e


def _compute_hidden_states(pool):
    """Backbone hidden states for the benchmark pool, computed once and cached.

    Called at startup, never per epoch. Embeddings are a pure function of the
    pool, the frozen backbone and the pinned batch size — none of which change
    between epochs — so recomputing them every epoch burned minutes of CPU on a
    21K-question pool for an identical result.

    They do not change between *restarts* either, which is why the result is
    cached to disk. Measured on mainnet TDs: 3 h 20 min for the v1 pool on a
    c3-standard-8 (docs/MAINNET_LAUNCH.md, "Miner startup cost"; the v2 pool is
    ~35% larger). Paying that on every restart is not a startup cost, it is an
    availability failure — the miner is unreachable for hours and earns nothing.

    The cache is keyed by the embedded text (see _embedding_cache_path) and
    loaded with allow_pickle=False, so a hostile file cannot execute anything.
    A cache that cannot be read, or that was built for other text, is set aside
    and recomputed. It used to raise, and in a dstack TD that is a crash loop
    the operator cannot clear: the file survives every restart and there is no
    shell to delete it.

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
    from fugal_subnet.backbone import compute_hidden_states, release_backbone
    from fugal_subnet.config import BACKBONE_BATCH_SIZE
    from fugal_subnet.vendor import success_contract as contract

    questions = [q["prompt"] for q in pool]
    cache_path = _embedding_cache_path(pool)
    if os.path.exists(cache_path):
        try:
            hidden = contract.load_cache(cache_path, questions)
        except Exception as e:  # noqa: BLE001 - any unreadable cache is recomputed, never fatal
            logger.warning("Embedding cache %s is unusable (%s); setting it aside and "
                           "recomputing", cache_path, e)
            try:
                os.replace(cache_path, cache_path + ".bad")
            except OSError:
                pass
        else:
            logger.info("Embeddings loaded from cache: %s", cache_path)
            return hidden

    # Use the configured memory bound for the 2048-token CPU profile. The shared
    # implementation excludes padding from pooling and has batch/single checks.
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
        tmp = f"{cache_path}.{os.getpid()}.tmp.npz"
        contract.save_cache(tmp, questions, hidden)
        os.replace(tmp, cache_path)
        logger.info("Embeddings cached to %s", cache_path)
    except Exception as e:  # noqa: BLE001 - caching is an optimisation
        logger.warning("Could not cache embeddings to %s: %s", cache_path, e)
    return hidden


def _get_source_hash():
    """The proof's source_hash: fugal_subnet.fingerprint.source_sha256, the one
    implementation (this used to be a second copy, kept equal only by a test)."""
    from fugal_subnet.fingerprint import source_sha256
    return source_sha256()


def _is_unroutable(ip: str) -> bool:
    """True for addresses the chain will not accept for serve_axon."""
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_unspecified


def _startup_head_check(head_data, protocol, *, mock, menu_b_block):
    """Validate the head against the protocol this image serves, before any
    epoch's inference bill: better refused here than by every validator.

    - v1-v3: the head's token means against the protocol's token manifest
      (routing_protocol.benchmark_costs_for; a v2 head against the v2 one,
      which v1's check refused at startup, found by the v2 flip rehearsal;
      once v3 menu B is scheduled a head built for either menu is accepted).
    - v4: head_eval.v4_head_verdict against the menu the TD starts from, the
      pinned seed (today's format, rows by entry id, token means optional and
      ignored; a row for some entry on the menu).
    - v3 and v4 heads are also held to the parameter bound validators refuse
      a proof for (head_odds.check_head_bounds).
    """
    from fugal_subnet.head_odds import check_head_bounds

    if protocol >= 4:
        from fugal_subnet import menu_state
        from fugal_subnet.head_eval import load_head_v4, v4_head_verdict

        why = v4_head_verdict(head_data, menu_state.load_seed())
        if why is not None:
            raise click.ClickException(f"this head would be refused by validators: {why}")
        _head = load_head_v4(head_data)
    else:
        from fugal_subnet.head_eval import load_head_from_npz
        from fugal_subnet.routing_protocol import MENU_A, MENU_B, benchmark_costs_for

        _head = load_head_from_npz(head_data)
        benchmark_costs_for(_head, protocol, require_reviewed=not mock,
                            menu=MENU_B if menu_b_block is not None else MENU_A)
    if protocol >= 3:
        try:
            check_head_bounds(_head.W, _head.b)
        except ValueError as e:
            raise click.ClickException(f"this head would be refused by validators: {e}") from e


def _load_head_file(path, protocol: int = 1):
    """Load and validate the head .npz file, in the format of the protocol
    this image serves: head_eval.load_head_v4 from v4 (token means optional),
    load_head_from_npz before it. The v1-v3 loader refused every v4 head
    without token means, so a v4 miner could not start from --head-path."""
    from fugal_subnet.head_eval import load_head_from_npz, load_head_v4
    with open(path, "rb") as f:
        data = f.read()
    try:
        (load_head_v4 if int(protocol) >= 4 else load_head_from_npz)(data)
    except ValueError as e:
        raise click.ClickException(f"Head failed validation: {e}")
    return data


if __name__ == "__main__":
    main()
