"""Network confinement for TEE benchmark execution.

Creates a Linux network namespace that restricts the benchmark process
to only communicate with the local MeteringProxy. Prevents exfiltration
and unmetered API calls.

Forked from ThirtySpokes/Chutes (MIT licensed) confinement patterns.
Requires root or CAP_NET_ADMIN.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_NAMESPACE_PREFIX = "fugal_tee_"


@dataclass
class ConfinedNamespace:
    name: str
    proxy_port: int
    created: bool = False


def create_confined_namespace(
    proxy_port: int,
    namespace_id: str = "default",
) -> ConfinedNamespace:
    """Create a network namespace that only allows traffic to localhost:proxy_port.

    The namespace gets a veth pair connecting it to the host. iptables
    rules inside the namespace restrict egress to the proxy port only.

    Args:
        proxy_port: The MeteringProxy port on the host.
        namespace_id: Unique identifier for this namespace.

    Returns:
        ConfinedNamespace with metadata about the created namespace.
    """
    ns_name = f"{_NAMESPACE_PREFIX}{namespace_id}"
    veth_host = f"veth_h_{namespace_id[:8]}"
    veth_ns = f"veth_n_{namespace_id[:8]}"
    subnet_id = _subnet_octet(namespace_id)
    host_ip = f"10.200.{subnet_id}.1"
    ns_ip = f"10.200.{subnet_id}.2"

    try:
        _run(["ip", "netns", "add", ns_name])
        _run(["ip", "link", "add", veth_host, "type", "veth", "peer", "name", veth_ns])
        _run(["ip", "link", "set", veth_ns, "netns", ns_name])
        _run(["ip", "addr", "add", f"{host_ip}/24", "dev", veth_host])
        _run(["ip", "link", "set", veth_host, "up"])

        _run_ns(ns_name, ["ip", "addr", "add", f"{ns_ip}/24", "dev", veth_ns])
        _run_ns(ns_name, ["ip", "link", "set", veth_ns, "up"])
        _run_ns(ns_name, ["ip", "link", "set", "lo", "up"])
        _run_ns(ns_name, ["ip", "route", "add", "default", "via", host_ip])

        _run_ns(ns_name, [
            "iptables", "-A", "OUTPUT",
            "-d", host_ip, "-p", "tcp", "--dport", str(proxy_port),
            "-j", "ACCEPT",
        ])
        _run_ns(ns_name, [
            "iptables", "-A", "OUTPUT",
            "-o", "lo", "-j", "ACCEPT",
        ])
        _run_ns(ns_name, [
            "iptables", "-A", "OUTPUT",
            "-j", "DROP",
        ])

        _run([
            "iptables", "-t", "nat", "-A", "PREROUTING",
            "-i", veth_host,
            "-p", "tcp", "--dport", str(proxy_port),
            "-j", "DNAT", "--to-destination", f"127.0.0.1:{proxy_port}",
        ])
        _run([
            "iptables", "-A", "FORWARD",
            "-i", veth_host, "-o", "lo",
            "-p", "tcp", "--dport", str(proxy_port),
            "-j", "ACCEPT",
        ])

        logger.info("Created confined namespace %s (proxy port %d)", ns_name, proxy_port)
        return ConfinedNamespace(name=ns_name, proxy_port=proxy_port, created=True)

    except Exception:
        logger.exception("Failed to create confined namespace %s", ns_name)
        cleanup_namespace(ns_name, proxy_port=proxy_port)
        raise


def cleanup_namespace(namespace_id_or_name: str, proxy_port: int = 0) -> None:
    """Tear down a confined namespace and its network resources."""
    ns_name = namespace_id_or_name
    if not ns_name.startswith(_NAMESPACE_PREFIX):
        ns_name = f"{_NAMESPACE_PREFIX}{namespace_id_or_name}"

    raw_id = ns_name.replace(_NAMESPACE_PREFIX, "")
    short_id = raw_id[:8]
    veth_host = f"veth_h_{short_id}"

    if proxy_port:
        _run([
            "iptables", "-t", "nat", "-D", "PREROUTING",
            "-i", veth_host,
            "-p", "tcp", "--dport", str(proxy_port),
            "-j", "DNAT", "--to-destination", f"127.0.0.1:{proxy_port}",
        ], check=False)
        _run([
            "iptables", "-D", "FORWARD",
            "-i", veth_host, "-o", "lo",
            "-p", "tcp", "--dport", str(proxy_port),
            "-j", "ACCEPT",
        ], check=False)

    try:
        _run(["ip", "netns", "delete", ns_name], check=False)
    except Exception:
        logger.debug("Namespace %s may not exist", ns_name)

    try:
        _run(["ip", "link", "delete", veth_host], check=False)
    except Exception:
        logger.debug("Veth %s may not exist", veth_host)

    logger.info("Cleaned up namespace %s", ns_name)


def _subnet_octet(namespace_id: str) -> int:
    """Derive a unique /24 subnet octet (1-254) from the namespace ID."""
    digest = int.from_bytes(namespace_id.encode()[:4], "big")
    return (digest % 254) + 1


def run_in_namespace(ns_name: str, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command inside a confined namespace."""
    return _run_ns(ns_name, cmd, **kwargs)


def _run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check, **kwargs)


def _run_ns(ns_name: str, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return _run(["ip", "netns", "exec", ns_name] + cmd, **kwargs)
