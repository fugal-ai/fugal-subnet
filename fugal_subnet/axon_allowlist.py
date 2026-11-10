"""Who may reach a miner's axon: the validators, by the IP they publish.

WHY. A v2 champion is scored only while its axon answers, and a champion kept
offline past its ghost loses the seat (INVARIANTS I3, "Accepted residual: a
champion kept offline"). Only validators need to reach a miner, so a miner
that admits only them at its cloud firewall can no longer be flooded off the
network by anyone else. For that the validators' addresses must be public:
each validator records its egress IP on chain (`publish_validator_ip`, a
serve_axon with nothing listening behind it; a validator needs no inbound
traffic at all), and the miner's firewall rule is rebuilt from the metagraph
(`scripts/axon_firewall.py`).

A validator, for the allowlist, is an ACTIVE validator: a validator permit,
stake at or above MIN_VALIDATOR_STAKE (the axon's own rule, `neurons/miner.py`
`caller_allowed`), and weights set within the last `active_blocks`. Stake
alone is not enough (under dTAO an earning miner holds stake on its own
hotkey), and a permit holder that never sets weights never queries a miner.
Nothing here touches a score or a weight.
"""
from __future__ import annotations

import ipaddress
import logging

logger = logging.getLogger(__name__)

# The port a validator records beside its IP. Nothing listens on it; the
# chain only requires an address and a port to record one.
VALIDATOR_ADVERTISED_PORT = 8091


def _routable_v4(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(str(ip))
    except ValueError:
        return False
    return addr.version == 4 and addr.is_global and not addr.is_multicast


def validator_cidrs(metagraph, min_stake: float,
                    active_blocks: int = 1080) -> tuple[list[str], list[str]]:
    """(sorted /32 CIDRs to admit, notes on ACTIVE validators left out).

    Every active validator (permit, stake >= `min_stake`, weights within
    `active_blocks` of the metagraph's block) that published a global IPv4
    address. An active validator with no published address is reported, not
    guessed at: the miner cannot admit what it cannot name.
    """
    hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
    head = int(getattr(metagraph, "block", 0) or 0)
    cidrs: set[str] = set()
    notes: list[str] = []
    for uid, hk in enumerate(hotkeys):
        try:
            stake = float(metagraph.S[uid])
            permit = bool(metagraph.validator_permit[uid])
            last = int(metagraph.last_update[uid])
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        if not permit or stake < float(min_stake) or head - last > int(active_blocks):
            continue
        try:
            ip = str(metagraph.axons[uid].ip)
        except (AttributeError, IndexError):
            ip = ""
        if _routable_v4(ip):
            cidrs.add(f"{ip}/32")
        else:
            notes.append(f"uid {uid} ({hk[:8]}...) is an active validator (stake {stake:.0f}) "
                         f"with no public IPv4 on chain ({ip or 'none'}): it cannot be admitted")
    return sorted(cidrs), notes


def _broad(rng: str) -> bool:
    """A source range that admits a large part of the internet: any IPv4
    prefix of /8 or wider (0.0.0.0/0, split halves like 0.0.0.0/1) or any IPv6
    prefix of /16 or wider (::/0)."""
    try:
        net = ipaddress.ip_network(rng, strict=False)
    except ValueError:
        return False
    return net.prefixlen <= (8 if net.version == 4 else 16)


def _port_in(spec: str, port: int) -> bool:
    if "-" in str(spec):
        lo, hi = str(spec).split("-", 1)
        return int(lo) <= port <= int(hi)
    return int(spec) == port


def _network_name(rule: dict) -> str:
    return str(rule.get("network", "")).rstrip("/").rsplit("/", 1)[-1]


def world_open_rules(rules: list[dict], port: int, tags, network: str = "default"
                     ) -> tuple[list[str], list[str]]:
    """(rules that open `port` broadly to the instance, rules to check by hand).

    Enabled INGRESS allow rules on `network` whose sources include a broad
    range, applying to instances with any of `tags` (or to every instance).
    GCP allow rules add up: while one of these exists, an allowlist changes
    nothing. Rules aimed at service accounts cannot be matched to the instance
    here and are listed for a manual check.
    """
    tags = {tags} if isinstance(tags, str) else set(tags or ())
    hits, manual = [], []
    for r in rules:
        if r.get("disabled") or r.get("direction", "INGRESS") != "INGRESS":
            continue
        if not r.get("allowed") or _network_name(r) not in ("", network):
            continue
        if not any(_broad(x) for x in (r.get("sourceRanges") or [])):
            continue
        opens = False
        for a in r.get("allowed") or []:
            if a.get("IPProtocol") not in ("tcp", "all"):
                continue
            ports = a.get("ports")
            if not ports or any(_port_in(p, port) for p in ports):
                opens = True
                break
        if not opens:
            continue
        if r.get("targetServiceAccounts"):
            manual.append(r.get("name", "?"))
            continue
        rtags = r.get("targetTags") or []
        if rtags and not tags & set(rtags):
            continue
        hits.append(r.get("name", "?"))
    return sorted(hits), sorted(manual)


def rule_is_current(rule: dict | None, *, port: int, target_tag: str, network: str,
                    cidrs: list[str]) -> bool:
    """Does the managed rule already admit exactly `cidrs`, enabled, on the
    right port, tag and network?"""
    if rule is None or rule.get("disabled") or rule.get("direction", "INGRESS") != "INGRESS":
        return False
    allowed = rule.get("allowed") or []
    return (sorted(rule.get("sourceRanges") or []) == sorted(cidrs)
            and allowed == [{"IPProtocol": "tcp", "ports": [str(port)]}]
            and (rule.get("targetTags") or []) == [target_tag]
            and _network_name(rule) in ("", network))


def firewall_command(*, rule: str, project: str, port: int, target_tag: str,
                     cidrs: list[str], exists: bool, network: str = "default") -> list[str]:
    """The gcloud command that makes `rule` admit exactly `cidrs` to `port`,
    enabled, on `target_tag` (an update rewrites every field it can)."""
    if not cidrs:
        raise ValueError("refusing an empty allowlist: it would drop every validator")
    ranges = ",".join(cidrs)
    if exists:
        return ["gcloud", "compute", "firewall-rules", "update", rule,
                f"--project={project}", f"--rules=tcp:{port}", f"--target-tags={target_tag}",
                f"--source-ranges={ranges}", "--no-disabled"]
    return ["gcloud", "compute", "firewall-rules", "create", rule,
            f"--project={project}", f"--network={network}", "--direction=INGRESS",
            "--action=ALLOW", f"--rules=tcp:{port}", f"--target-tags={target_tag}",
            f"--source-ranges={ranges}"]


def publish_validator_ip(subtensor, wallet, netuid: int, metagraph, my_uid: int,
                         ip: str | None) -> tuple[bool, str]:
    """Record this validator's egress IP on chain so miners can admit it.

    Never raises (I6: nothing here may stop an epoch). Returns (ok, what
    happened), for the log. Skips the extrinsic when the chain already has
    `ip`, so a restart does not spend the serving rate limit.
    """
    if not ip:
        return False, ("FUGAL_VALIDATOR_IP is unset: miners that admit only validator IPs "
                "(scripts/axon_firewall.py) will drop this validator's queries. Set it to "
                "this host's public egress IP (docs/VALIDATOR_GUIDE.md).")
    if not _routable_v4(ip):
        return False, (f"FUGAL_VALIDATOR_IP={ip} is not a public IPv4 address: not published, "
                "and miners that admit only validator IPs cannot admit this validator.")
    try:
        current = str(metagraph.axons[my_uid].ip)
    except (AttributeError, IndexError):
        current = ""
    if current == ip:
        return True, f"validator IP {ip} already on chain"
    try:
        import bittensor as bt

        # Constructed, never started: nothing listens. serve() only records
        # the address on chain.
        axon = bt.Axon(wallet=wallet, port=VALIDATOR_ADVERTISED_PORT, external_ip=ip)
        axon.serve(netuid=netuid, subtensor=subtensor)
        after = str(subtensor.metagraph(netuid).axons[my_uid].ip)
    except Exception as e:  # noqa: BLE001 - reported; never stops the validator
        return False, f"publishing validator IP {ip} failed: {e}"
    if after == ip:
        return True, f"validator IP {ip} published on chain"
    return False, (f"validator IP {ip} not on chain after serve (chain has {after}); the serving "
            "rate limit may apply, it is retried at the next start")
