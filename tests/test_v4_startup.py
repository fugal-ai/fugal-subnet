"""A validator's startup for protocol v4: the READY line's v4 part, every v4
refusal reachable with fakes, refusals that wait for the block, and
`--check`.

Before the first v4 epoch nothing about v4 stops a validator: it runs v3, its
READY line names what v4 lacks, and it stops at the first v4 boundary if that
still holds. From the first v4 epoch a live validator refuses to start.
"""
import json
from pathlib import Path

import click
import pytest

from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from fugal_subnet import reveal_sync
from fugal_subnet import validator_setup as vs
from neurons import validator

BPE = 360
ROOT_HK = "5Froot-hotkey"
BUCKET = "fugal-test-reveals"
V4_BLOCK = 26400 * BPE


class _HP:
    def __init__(self, period):
        self.commit_reveal_weights_enabled = True
        self.commit_reveal_period = period


class _Sub:
    def __init__(self, period=1, epoch=26300):
        self.period, self.epoch = period, epoch

    def get_subnet_hyperparameters(self, netuid):
        return _HP(self.period)

    def get_current_block(self):
        return self.epoch * BPE


class _Cfg:
    PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4 = 5000
    REFERENCE_BUCKET = ""


def _role(root=False):
    return reveal_sync.Role(root, None if root else "/nonexistent", None if root else BUCKET,
                            "test", root_hotkey=None if root else ROOT_HK)


def _status(*, sub=None, cfg=_Cfg, root=False, bucket=BUCKET, my=None, hotkeys=((0, ROOT_HK),),
            loaded=True, error=None):
    return validator._v4_status(
        cfg=cfg, subtensor=sub or _Sub(), netuid=7, my_hotkey=my or (ROOT_HK if root else "5Fme"),
        role=_role(root), root_hotkey=ROOT_HK, reveal_delay=1, v4_block=V4_BLOCK,
        blocks_per_epoch=BPE, reveal_bucket=bucket, inputs_loaded=loaded, inputs_error=error,
        root_hotkeys=hotkeys)


def test_no_v4_block_no_v4_status():
    st = validator._v4_status(cfg=_Cfg, subtensor=_Sub(), netuid=7, my_hotkey="x", role=None,
                              root_hotkey=None, reveal_delay=1, v4_block=None,
                              blocks_per_epoch=BPE, reveal_bucket="", inputs_loaded=False)
    assert st.found is None and st.missing == [] and not st.due


def test_the_shipped_release_has_every_v4_input_but_its_list():
    st = _status()
    assert st.missing == [] and not st.due
    assert st.found.startswith(f"block {V4_BLOCK}, seed ") and "mix " in st.found
    assert "list none" in st.found and "root hotkeys 1" in st.found and "v4 digest " in st.found
    assert _status(sub=_Sub(epoch=26400)).due
    # The memorisation check's published-set seed, by hash and the reveal it
    # is observed through (the nodes fold the public reveals after it).
    from fugal_subnet import memo_v4
    assert f"published seed {memo_v4.seed_label()}" in st.found
    assert " through e000" in memo_v4.seed_label()


def test_an_unreadable_published_set_seed_is_a_v4_refusal_on_both_neurons(monkeypatch, tmp_path):
    from fugal_subnet import memo_v4
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch
    from fugal_subnet.config import EPOCH_INTERVAL
    from neurons import miner

    monkeypatch.setattr(memo_v4, "SEED_PATH", str(tmp_path / "absent.json"))
    st = _status()
    assert any("published-set seed cannot be read" in m for m in st.missing), st.missing
    ok, line = miner._miner_ready_line(4, 100 * blocks_per_epoch(EPOCH_INTERVAL),
                                       root_hotkeys=((0, "5Froot"),))
    assert not ok and "published-set seed cannot be read" in line


class _Unpinned(_Cfg):
    PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4 = None


@pytest.mark.parametrize("kw, why", [
    (dict(sub=_Sub(period=2)), "holds reveals for 2 epochs"),
    (dict(root=True, my="5Fother"), "the release names another root hotkey"),
    (dict(root=True, bucket=""), "publishes no reveals"),
    (dict(cfg=_Unpinned), "category floor is not pinned"),
    (dict(hotkeys=()), "no root hotkey is known for the first v4 epoch"),
    (dict(hotkeys=((26401, ROOT_HK),)), "no root hotkey is known for the first v4 epoch"),
    (dict(loaded=False), "the v3 pool and embeddings v4 runs on are not loaded"),
    (dict(loaded=False, error="the v4 plan cannot be built"), "the v4 plan cannot be built"),
])
def test_every_v4_refusal_is_reachable(kw, why):
    st = _status(**kw)
    assert any(why in m for m in st.missing), st.missing
    assert st.refusals == st.missing


def test_a_summary_window_the_menu_step_refuses_is_a_refusal(monkeypatch):
    from fugal_subnet import config
    monkeypatch.setattr(config, "CHAMPION_WINDOW_EPOCHS", 60)
    assert any("49" in m for m in _status().missing)


def test_unreadable_v4_files_are_refusals(monkeypatch):
    from fugal_subnet import epoch_plan

    def broken(*_a, **_k):
        raise ValueError("broken")

    with monkeypatch.context() as m:
        m.setattr(ms, "load_seed", broken)
        assert any("menu seed cannot be read" in x for x in _status().missing)
    with monkeypatch.context() as m:
        m.setattr(epoch_plan, "mix_v4", broken)
        assert any("v4 mix cannot be read" in x for x in _status().missing)
    with monkeypatch.context() as m:
        m.setattr(mr, "list_signers", lambda: (_ for _ in ()).throw(RuntimeError("pin")))
        assert any("does not verify" in x for x in _status().missing)
    with monkeypatch.context() as m:
        def refused(*_a, **_k):
            raise mr.ListRefused("draft", "serial 0")
        m.setattr(mr, "shipped_list", refused)
        assert any("does not verify" in x for x in _status().missing)
    with monkeypatch.context() as m:
        import dataclasses
        seed = ms.load_seed()
        m.setattr(ms, "load_seed", lambda: dataclasses.replace(
            seed, labs=ms.AllowlistRef(3, "ab" * 32)))
        assert any("names lab and host list serial 3" in x for x in _status().missing)


def test_the_ready_line_names_v4_found_or_missing():
    base = dict(entries=3, newest="app-compose.frozen-x-finney.json", v3="ok:48320",
                archive=True, collateral=True, role="follower (verified)")
    ok, line = vs.ready_line(**base)
    assert ok and line.endswith(", v4 not scheduled")
    ok, line = vs.ready_line(**base, v4="block 9504000, seed ab, mix cd")
    assert ok and line.endswith(", v4 ok (block 9504000, seed ab, mix cd)")
    # Before the block: ready for v3, and the line names what v4 lacks.
    ok, line = vs.ready_line(**base, v4="block 9504000, seed ab", v4_missing=["no list"])
    assert ok and line.startswith("READY:")
    assert "v4 MISSING from block 9504000 (runs v3 until then, stops there): no list" in line
    # From the block it is not ready.
    ok, line = vs.ready_line(**base, v4="block 9504000", v4_missing=["no list"], v4_due=True)
    assert not ok and line == "NOT READY: v4 MISSING: no list"


def _setup(tmp_path, *, sub, cfg=_Cfg, live=True):
    return validator._v4_menu_setup(
        cfg=cfg, subtensor=sub, netuid=7, my_hotkey="5Fme", role=_role(), root_hotkey=ROOT_HK,
        reveal_delay=1, live=live, blocks_per_epoch=BPE, v4_block=V4_BLOCK,
        state_path=str(tmp_path / "state.json"), reveal_bucket="")


def test_a_v4_refusal_before_the_block_runs_v3_and_stops_at_the_block(tmp_path):
    node, _ = _setup(tmp_path, sub=_Sub(period=2, epoch=26399))       # the epoch before
    assert node.v4_refusals and "holds reveals for 2 epochs" in node.v4_refusals[0]
    why = validator._v4_due_refusal(node, live=True)
    assert why.startswith("Protocol v4 starts at this epoch and cannot run on this validator")
    # From the first v4 epoch it refuses to start.
    with pytest.raises(click.ClickException, match="holds reveals for 2 epochs"):
        _setup(tmp_path, sub=_Sub(period=2, epoch=26400))
    # Mock: a warning, never a stop.
    node, _ = _setup(tmp_path, sub=_Sub(period=2, epoch=26400), live=False)
    assert node.v4_refusals == [] and validator._v4_due_refusal(node, live=False) is None
    node, _ = _setup(tmp_path, sub=_Sub(epoch=26399))
    assert node.v4_refusals == [] and validator._v4_due_refusal(node, live=True) is None


# --- --check, through the shipped validator.main on the v4 world's fakes ----------------------


def _world(tmp_path, monkeypatch, signer, **spec):
    from tests.v4_world import MINERS, FlipWorld, MinerSpec, ValidatorSpec

    w = FlipWorld(tmp_path, monkeypatch.setattr, monkeypatch.setenv, signer, flip=12, first=9,
                  last=9, validators=[ValidatorSpec("root", "root", "root", root=True, **spec)],
                  miners=[MinerSpec(hk, ck, seed=i) for i, (hk, ck) in enumerate(MINERS)])
    w.setup()
    for mod, attr, value in w.contexts()["root"]:
        monkeypatch.setattr(mod, attr, value)
    from fugal_subnet import config
    monkeypatch.setattr(config, "TEE_APPROVED_MEASUREMENTS", ["mock:entry"])
    monkeypatch.setattr(config, "ARCHIVE_ENDPOINTS", ["ws://archive.invalid"])
    return w


def _check(capsys):
    with pytest.raises(SystemExit) as e:
        validator.main.callback(network="local", netuid=7, coldkey="root", hotkey="default",
                                wallet_path=None, once=False, log_level="WARNING", live=False,
                                check=True)
    return e.value.code, capsys.readouterr().out.strip().splitlines()[-1]


def test_check_prints_the_ready_line_and_exits_without_an_epoch(tmp_path, monkeypatch, capsys):
    from tests.test_menu_publish import Signer
    from tests.v4_world import BPE as WBPE

    s = Signer(tmp_path, "release")
    monkeypatch.setattr(mr, "list_signers", lambda: s.signers_line().encode())
    w = _world(tmp_path, monkeypatch, s)
    w.chain.block = 10 * WBPE
    code, line = _check(capsys)
    assert code == 0, line
    assert line.startswith("READY:") and f"v4 ok (block {12 * WBPE}, seed " in line
    assert "list serial" in line
    assert w.chain.weights == [] and not (w.dirs["root"] / "logs").exists()
    # The same check with a v4 refusal: exit 1, the line says what is missing.
    from fugal_subnet import config
    monkeypatch.setattr(config, "PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4", None)
    code, line = _check(capsys)
    assert code == 1 and line.startswith("READY:") and "v4 MISSING from block" in line
    assert "category floor is not pinned" in line
    json.dumps(line)


# --- the miner ---------------------------------------------------------------------------------


def test_the_miners_ready_line_names_its_v4_inputs(monkeypatch):
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch
    from fugal_subnet.config import EPOCH_INTERVAL
    from neurons import miner

    ok, line = miner._miner_ready_line(3, 9527760, pool_size=48320)
    assert ok and line == "READY: protocol v3 from block 9527760, pool 48320"
    block = 100 * blocks_per_epoch(EPOCH_INTERVAL)
    spec = miner._parse_v4_proposal(json.dumps({
        "model": "openai/gpt-x", "reasoning": ["effort", "high"], "quantization": "unknown",
        "provider_order": ["openai"]}))
    ok, line = miner._miner_ready_line(4, block, v4_spec=spec, v4_challenge="1000", pool_size=10,
                                       root_hotkeys=((0, "5Froot"),))
    assert ok and line.startswith(f"READY: protocol v4 from block {block}, pool 10, seed ")
    assert f"proposal {spec.id}" in line and "ceiling" not in line
    assert "challenge 1000" in line and "list none" in line
    from fugal_subnet import memo_v4
    assert f"published seed {memo_v4.seed_label()}" in line
    ok, line = miner._miner_ready_line(4, block, root_hotkeys=())
    assert not ok and "no root hotkey is known for the first v4 epoch e00000100" in line
    monkeypatch.setattr(mr, "list_signers", lambda: (_ for _ in ()).throw(RuntimeError("pin")))
    ok, line = miner._miner_ready_line(4, block, root_hotkeys=((0, "5Froot"),))
    assert not ok and "does not verify" in line


def test_the_miners_check_exits_before_provisioning_embedding_or_serving(monkeypatch, capsys):
    from fugal_subnet import code_gate, config
    from fugal_subnet.benchmarks import code_sets, loader
    from neurons import miner
    from tests.v4_world import fixture_pool

    genesis = "0x" + "11" * 32
    monkeypatch.setattr(miner, "_assert_chain_allows", lambda network, mock: genesis)
    monkeypatch.setattr(miner, "_assert_axon_ip_routable", lambda ip: None)
    monkeypatch.setattr(miner, "_assert_registered", lambda *a, **k: None)
    monkeypatch.setattr(loader, "load_pool", lambda protocol: fixture_pool())
    monkeypatch.setattr(code_gate, "release_errors", lambda pool: [])
    monkeypatch.setattr(code_sets, "code_sandbox_required", lambda pool: False)

    def never(*_a, **_k):
        raise AssertionError("--check must not provision, embed or serve")

    monkeypatch.setattr(miner, "_await_provisioning", never)
    monkeypatch.setattr(miner, "_load_pinned_embeddings", never)
    monkeypatch.setattr(config, "V3_ACTIVATION_BLOCK_ENV", 10 * 360)
    monkeypatch.setattr(config, "V4_ACTIVATION_BLOCK_ENV", 20 * 360)
    monkeypatch.setattr(config, "EPOCH_INTERVAL", 4320)
    monkeypatch.setattr(config, "ROOT_HOTKEY_ENV", "5Froot")

    def run(**kw):
        args = dict(network="local", netuid=7, coldkey="c", hotkey="h", wallet_path=None,
                    port=8091, head_path=None, await_provisioning=True, benchmark_pool=None,
                    mock=True, log_level="WARNING", check=True)
        args.update(kw)
        with pytest.raises(SystemExit) as e:
            miner.main.callback(**args)
        return e.value.code, capsys.readouterr().out.strip().splitlines()[-1]

    code, line = run()
    assert code == 0 and line.startswith(f"READY: protocol v4 from block {20 * 360}"), line
    monkeypatch.setattr(config, "ROOT_HOTKEY_ENV", "")
    code, line = run()
    assert code == 1 and "no root hotkey is known" in line
    # A refusal of the options themselves is the same refusal as at startup.
    with pytest.raises(click.ClickException, match="FUGAL_V4_PROPOSAL"):
        run(v4_proposal="not json")


# --- FUGAL_V4_ACTIVATION_BLOCK that is not a block number -------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
_OTHER_CHAIN = "0x" + "ab" * 32

_ENV_PROBE = r'''
import json, os, sys
sys.path.insert(0, os.getcwd())
out = {}
try:
    from fugal_subnet import config, fingerprint
    finney = config.FINNEY_GENESIS_HASH

    def attempt(fn):
        try:
            return fn()
        except ValueError as e:
            return "ValueError: " + str(e)

    out["env"] = config.V4_ACTIVATION_BLOCK_ENV
    out["malformed"] = config.V4_ACTIVATION_BLOCK_MALFORMED
    out["finney_block"] = attempt(lambda: config.v4_activation_block(finney))
    out["finney_protocol"] = attempt(lambda: config.protocol_for_chain(finney))
    out["other_block"] = attempt(lambda: config.v4_activation_block(sys.argv[1]))
    out["other_protocol"] = attempt(lambda: config.protocol_for_chain(sys.argv[1]))
    out["unknown_block"] = attempt(lambda: config.v4_activation_block(None))
    out["digest_v3"] = fingerprint.consensus_digest()
    from neurons import miner, validator
    out["validator"] = attempt(lambda: validator._v4_activation_block(config, finney))
    out["neurons"] = True
except BaseException as e:
    out["error"] = type(e).__name__ + ": " + str(e)[:200]
print("PROBE " + json.dumps(out, sort_keys=True))
'''


def _probe_env(value: str | None) -> dict:
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if not k.startswith("FUGAL_")}
    if value is not None:
        env["FUGAL_V4_ACTIVATION_BLOCK"] = value
    run = subprocess.run([sys.executable, "-c", _ENV_PROBE, _OTHER_CHAIN], cwd=_ROOT, env=env,
                         capture_output=True, text=True, timeout=300)
    line = [ln for ln in run.stdout.splitlines() if ln.startswith("PROBE ")]
    assert line, run.stderr[-500:]
    return json.loads(line[0][6:])


def test_a_v4_activation_variable_that_is_not_a_block_number_is_refused_by_name_off_finney():
    clean = _probe_env(None)
    assert "error" not in clean and clean["finney_protocol"] == 3 and clean["malformed"] is None
    for bad in ("not-a-number", "12.5", "0x10", "1_000", "12abc", "٣٠٠", "--5", "12 34"):
        got = _probe_env(bad)
        # an import never fails on it, and neither neuron module
        assert "error" not in got and got["neurons"] is True, (bad, got)
        assert got["env"] is None and got["malformed"] == bad, (bad, got)
        # finney, with no v4 block compiled: the variable is ignored as the v3 release ignores it,
        # so the protocol, the v4 block, the validator's reading and the v1-v3 digest are
        # exactly what they are without the variable
        for key in ("finney_block", "finney_protocol", "validator", "digest_v3"):
            assert got[key] == clean[key], (bad, key)
        assert got["finney_block"] is None and got["finney_protocol"] == 3
        # another chain, or one whose genesis cannot be read (finney cannot be ruled out):
        # a refusal that names the variable
        for key in ("other_block", "other_protocol", "unknown_block"):
            assert (isinstance(got[key], str) and got[key].startswith("ValueError: ")
                    and "FUGAL_V4_ACTIVATION_BLOCK" in got[key] and "not a block number" in got[key]
                    ), (bad, key, got[key])


def test_a_whitespace_only_v4_activation_variable_is_unset_and_a_number_is_a_number():
    clean = _probe_env(None)
    got = _probe_env("   ")
    assert got["env"] is None and got["malformed"] is None and got == clean
    got = _probe_env(" 9600120 ")
    assert got["env"] == 9600120 and got["malformed"] is None


def test_a_malformed_v4_block_is_refused_on_finney_once_a_block_is_compiled(monkeypatch):
    from fugal_subnet import config

    monkeypatch.setattr(config, "V4_ACTIVATION_BLOCK_MALFORMED", "oops")
    assert config.v4_activation_block(config.FINNEY_GENESIS_HASH) is None
    monkeypatch.setattr(config, "FINNEY_V4_ACTIVATION_BLOCK", 9_600_120)
    with pytest.raises(ValueError, match="not a block number"):
        config.v4_activation_block(config.FINNEY_GENESIS_HASH)


def test_both_neurons_decide_their_protocol_through_the_named_refusal(monkeypatch):
    from fugal_subnet import config

    monkeypatch.setattr(config, "V4_ACTIVATION_BLOCK_MALFORMED", "oops")
    assert validator._v4_activation_block(config, config.FINNEY_GENESIS_HASH) is None
    assert config.protocol_for_chain(config.FINNEY_GENESIS_HASH) == 3
    with pytest.raises(ValueError, match="FUGAL_V4_ACTIVATION_BLOCK='oops' is not a block number"):
        validator._v4_activation_block(config, _OTHER_CHAIN)
    with pytest.raises(ValueError, match="not a block number"):
        config.protocol_for_chain(_OTHER_CHAIN)       # the miner's decision (neurons/miner.main)


def test_check_closes_the_chain_connection_before_it_exits(tmp_path, monkeypatch, capsys):
    """--check closes its chain connection itself before SystemExit. Left to
    async_substrate_interface's __del__ at interpreter shutdown, a real node's
    connection joined a thread that never finished: the check printed READY
    and never exited (the local-chain v4 rehearsal, dress_rehearsal_v4.py)."""
    from tests.test_menu_publish import Signer
    from tests.v4_world import BPE as WBPE
    from tests.v4_world import Subtensor

    s = Signer(tmp_path, "release")
    monkeypatch.setattr(mr, "list_signers", lambda: s.signers_line().encode())
    w = _world(tmp_path, monkeypatch, s)
    w.chain.block = 10 * WBPE
    closed = []
    monkeypatch.setattr(Subtensor, "close", lambda self: closed.append(True))
    code, line = _check(capsys)
    assert code == 0 and line.startswith("READY:"), line
    assert closed, "--check exited with its chain connection still open"
