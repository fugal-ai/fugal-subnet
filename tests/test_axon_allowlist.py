"""Validators publish their egress IP; miners admit only them (axon_allowlist)."""
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

from fugal_subnet import axon_allowlist as al


def _mg(rows, block=10_000):
    """rows: (hotkey, stake, ip[, permit=True[, last_update=block]])."""
    return SimpleNamespace(
        hotkeys=[r[0] for r in rows],
        S=[r[1] for r in rows],
        axons=[SimpleNamespace(ip=r[2]) for r in rows],
        validator_permit=[r[3] if len(r) > 3 else True for r in rows],
        last_update=[r[4] if len(r) > 4 else block for r in rows],
        block=block,
    )


def test_validators_are_the_active_permit_holders_with_a_public_ipv4():
    mg = _mg([("v1" * 24, 1000.0, "34.1.2.3"), ("v2" * 24, 50.0, "35.9.9.9"),
              ("m1" * 24, 10.0, "36.0.0.1"),            # under the stake rule
              ("m2" * 24, 120.0, "36.0.0.2", False),    # an earning miner: no permit
              ("ix" * 24, 9e5, "0.0.0.0", True, 1000),  # idle permit holder: never queries
              ("v3" * 24, 500.0, "0.0.0.0"),            # active, published nothing
              ("v4" * 24, 500.0, "10.0.0.5"),           # private
              ("v5" * 24, 500.0, "2600::1"),            # IPv6
              ("v7" * 24, 500.0, "224.0.0.1"),          # multicast
              ("v6" * 24, 900.0, "34.1.2.3")])          # same IP twice: one entry
    cidrs, notes = al.validator_cidrs(mg, 50.0, active_blocks=1080)
    assert cidrs == ["34.1.2.3/32", "35.9.9.9/32"]
    assert len(notes) == 4 and all("cannot be admitted" in n for n in notes)


def test_rules_that_open_the_port_broadly_to_the_instance_are_reported():
    def rule(name, src=("0.0.0.0/0",), tags=None, ports=("8091",), proto="tcp", **kw):
        r = {"name": name, "direction": "INGRESS", "sourceRanges": list(src),
             "network": "https://x/projects/p/global/networks/default",
             "allowed": [{"IPProtocol": proto, **({"ports": list(ports)} if ports else {})}]}
        if tags:
            r["targetTags"] = list(tags)
        r.update(kw)
        return r
    rules = [
        rule("fugal-axon", tags=["fugal-miner"]),
        rule("wide", ports=["8000-9000"]),
        rule("all", tags=["fugal-miner"], ports=None, proto="all"),
        rule("split", src=["0.0.0.0/1", "128.0.0.0/1"], tags=["fugal-miner"]),
        rule("v6", src=["::/0"]),
        rule("on-other-vm-tag", tags=["http-server"]),        # the VM carries it too
        rule("ssh", ports=["22"]),
        rule("other-tag", tags=["web"]),
        rule("off", disabled=True),
        rule("egress", direction="EGRESS"),
        rule("narrow", src=["34.1.2.3/32"], tags=["fugal-miner"]),
        rule("other-vpc", network="https://x/projects/p/global/networks/prod"),
        rule("sa", targetServiceAccounts=["svc@p.iam"]),
    ]
    hits, manual = al.world_open_rules(rules, 8091, {"fugal-miner", "http-server"}, "default")
    assert hits == ["all", "fugal-axon", "on-other-vm-tag", "split", "v6", "wide"]
    assert manual == ["sa"]


def test_the_managed_rule_is_current_only_when_every_field_matches():
    good = {"name": "r", "direction": "INGRESS", "sourceRanges": ["1.2.3.4/32"],
            "allowed": [{"IPProtocol": "tcp", "ports": ["8091"]}], "targetTags": ["t"],
            "network": "https://x/networks/default"}
    kw = dict(port=8091, target_tag="t", network="default", cidrs=["1.2.3.4/32"])
    assert al.rule_is_current(good, **kw)
    for change in ({"disabled": True}, {"sourceRanges": ["5.6.7.8/32"]},
                   {"allowed": [{"IPProtocol": "tcp", "ports": ["8092"]}]},
                   {"targetTags": ["other"]}, {"network": "https://x/networks/prod"}):
        assert not al.rule_is_current({**good, **change}, **kw), change
    assert not al.rule_is_current(None, **kw)


def test_the_firewall_command_creates_or_updates_and_never_empties():
    create = al.firewall_command(rule="r", project="p", port=8091, target_tag="t",
                                 cidrs=["1.2.3.4/32", "5.6.7.8/32"], exists=False)
    assert create[:4] == ["gcloud", "compute", "firewall-rules", "create"]
    assert "--rules=tcp:8091" in create and "--target-tags=t" in create
    assert "--network=default" in create
    assert "--source-ranges=1.2.3.4/32,5.6.7.8/32" in create
    update = al.firewall_command(rule="r", project="p", port=8091, target_tag="t",
                                 cidrs=["1.2.3.4/32"], exists=True)
    assert update[3] == "update" and "--source-ranges=1.2.3.4/32" in update
    assert "--no-disabled" in update and "--rules=tcp:8091" in update
    with pytest.raises(ValueError, match="empty allowlist"):
        al.firewall_command(rule="r", project="p", port=8091, target_tag="t", cidrs=[],
                            exists=True)


class _FakeAxon:
    served = []

    def __init__(self, wallet=None, port=None, external_ip=None):
        self.ip, self.port = external_ip, port

    def serve(self, netuid, subtensor):
        _FakeAxon.served.append((netuid, self.ip, self.port))
        subtensor.chain_ip = self.ip

    def start(self):  # pragma: no cover - must never be called
        raise AssertionError("a validator never starts an axon")


class _FakeSubtensor:
    def __init__(self, chain_ip="0.0.0.0", fail=False):
        self.chain_ip, self.fail = chain_ip, fail

    def metagraph(self, netuid):
        if self.fail:
            raise RuntimeError("chain down")
        return SimpleNamespace(axons=[SimpleNamespace(ip=self.chain_ip)])


@pytest.fixture
def fake_bt(monkeypatch):
    _FakeAxon.served = []
    monkeypatch.setitem(sys.modules, "bittensor", types.SimpleNamespace(Axon=_FakeAxon))
    return _FakeAxon


def test_a_validator_publishes_its_ip_once_and_never_raises(fake_bt):
    st = _FakeSubtensor()
    mg = st.metagraph(7)
    ok, msg = al.publish_validator_ip(st, None, 7, mg, 0, "34.1.2.3")
    assert ok and "published" in msg
    assert fake_bt.served == [(7, "34.1.2.3", al.VALIDATOR_ADVERTISED_PORT)]
    # Already on chain: no second extrinsic (the serving rate limit).
    ok, msg = al.publish_validator_ip(st, None, 7, st.metagraph(7), 0, "34.1.2.3")
    assert ok and "already" in msg and len(fake_bt.served) == 1
    # Unset, private, and a chain failure: reported, never raised.
    assert al.publish_validator_ip(st, None, 7, mg, 0, "")[0] is False
    assert al.publish_validator_ip(st, None, 7, mg, 0, "192.168.1.9")[0] is False
    down = _FakeSubtensor(fail=True)
    ok, msg = al.publish_validator_ip(down, None, 7, mg, 0, "34.9.9.9")
    assert not ok and "failed" in msg


def test_the_validator_publishes_at_startup_and_the_setting_is_operational():
    from fugal_subnet import config

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("neurons", "validator.py").read_text()
    assert "publish_validator_ip(subtensor, wallet, netuid, metagraph, my_uid," in src
    assert "VALIDATOR_IP" in src
    assert "FUGAL_VALIDATOR_IP" in config.OPERATIONAL_ENV_VARS
    assert "FUGAL_VALIDATOR_IP" not in config.CONSENSUS_ENV_VARS


def _script():
    import importlib.util

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "axon_firewall.py"
    spec = importlib.util.spec_from_file_location("axon_firewall", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("case", ["all_published", "one_missing", "one_missing_forced",
                                  "current", "disabled", "chain_down", "gcloud_down",
                                  "apply_fails", "other_vpc"])
def test_the_firewall_script_applies_only_a_complete_allowlist(case, monkeypatch, capsys):
    mod = _script()
    rows = [("v1" * 24, 1000.0, "34.1.2.3"), ("v2" * 24, 100.0, "35.9.9.9")]
    if case.startswith("one_missing"):
        rows.append(("v3" * 24, 500.0, "0.0.0.0"))
    mg = _mg(rows)
    fake = types.SimpleNamespace(Subtensor=lambda network: types.SimpleNamespace(
        metagraph=(lambda n: (_ for _ in ()).throw(RuntimeError("down"))) if case == "chain_down"
        else (lambda n: mg)))
    monkeypatch.setitem(sys.modules, "bittensor", fake)
    rule = {"name": "fugal-axon-validators", "direction": "INGRESS",
            "sourceRanges": ["34.1.2.3/32", "35.9.9.9/32"],
            "allowed": [{"IPProtocol": "tcp", "ports": ["8091"]}],
            "targetTags": ["fugal-miner"], "network": "https://x/networks/default"}
    existing = {"current": [rule], "disabled": [{**rule, "disabled": True}],
                "other_vpc": [{**rule, "network": "https://x/networks/prod"}]}.get(case, [])

    def gjson(cmd):
        if case == "gcloud_down":
            raise mod.subprocess.CalledProcessError(1, cmd)
        return existing
    monkeypatch.setattr(mod, "_gcloud_json", gjson)
    ran = []

    def run(cmd, **kw):
        ran.append(cmd)
        if case == "apply_fails":
            raise mod.subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(mod.subprocess, "run", run)
    args = types.SimpleNamespace(network="test", netuid=1, project="p", vpc="default",
                                 instance=None, zone="z", rule="fugal-axon-validators",
                                 port=8091, target_tag="fugal-miner", extra_cidr=None,
                                 min_stake=50.0, active_blocks=1080, apply=True,
                                 allow_partial=case == "one_missing_forced")
    rc = mod.run_once(args)
    out = capsys.readouterr().out
    if case in ("all_published", "one_missing_forced"):
        assert rc == 0 and ran and ran[0][3] == "create"
        assert "--source-ranges=34.1.2.3/32,35.9.9.9/32" in ran[0]
    elif case == "one_missing":
        assert rc == 1 and not ran and "refusing" in out
    elif case == "current":
        assert rc == 0 and not ran and "current" in out
    elif case == "disabled":
        assert rc == 0 and ran and ran[0][3] == "update" and "--no-disabled" in ran[0]
    elif case == "apply_fails":
        assert rc == 1 and "retried next run" in out
    elif case == "other_vpc":
        assert rc == 1 and not ran and "another VPC" in out
    else:
        assert rc == 1 and not ran and "left as it is" in out


def test_the_repeat_loop_survives_a_failed_run(monkeypatch, capsys):
    mod = _script()
    monkeypatch.setattr(mod, "run_once", lambda a: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(sys, "argv", ["axon_firewall.py", "--network", "test", "--netuid", "1",
                                      "--project", "p"])
    assert mod.main() == 1 and "retried next run" in capsys.readouterr().out
