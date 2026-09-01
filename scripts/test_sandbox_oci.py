#!/usr/bin/env python3
"""Build and attack the real networkless OCI grading boundary."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fugal_subnet.sandbox.client import GradingClient
from fugal_subnet.sandbox.launcher import GradingService, OciRunner
from fugal_subnet.sandbox.protocol import SCHEMA_VERSION, canonical_json

ROOT = Path(__file__).resolve().parents[1]


class _Peer:
    def getsockopt(self, level, option, length):
        del level, option, length
        return (0).to_bytes(4, "little", signed=True) + os.getuid().to_bytes(
            4, "little", signed=True
        ) + os.getgid().to_bytes(4, "little", signed=True)


def build_image(tag: str) -> str:
    subprocess.run(
        [
            "docker", "build", "--pull=false", "--tag", tag,
            str(ROOT / "docker" / "grader-worker"),
        ],
        check=True,
    )
    return subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
        text=True,
    ).strip()


def request(code: str, inputs: list[list], expected: list, job_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "operation": "code_io",
        "code": code,
        "function": "probe",
        "inputs": inputs,
        "expected": expected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Use an existing exact sha256 image ID")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    image = args.image
    if not image:
        if args.skip_build:
            parser.error("--skip-build requires --image")
        image = build_image("fugal-grader-worker:test")
    print(f"Testing worker image {image}", flush=True)
    runner = OciRunner(engine="docker", image=image, timeout_seconds=4.0)
    service = GradingService(runner, allowed_uids={os.getuid()}, max_concurrency=1)
    peer = _Peer()

    cases = [
        (
            "honest",
            request("def probe(a, b): return a + b", [[2, 3]], [5], "honest"),
            True,
        ),
        (
            "environment",
            request(
                "import os\ndef probe(name): return os.environ.get(name)",
                [["OPENROUTER_API_KEY"]],
                [None],
                "environment",
            ),
            True,
        ),
        (
            "wallet_path",
            request(
                "import os\ndef probe(path): return os.path.exists(path)",
                [["/root/.bittensor"], ["/home/fugal-validator/.bittensor"]],
                [False, False],
                "wallet-path",
            ),
            True,
        ),
        (
            "engine_socket",
            request(
                "import os\ndef probe(path): return os.path.exists(path)",
                [["/var/run/docker.sock"], ["/run/podman/podman.sock"]],
                [False, False],
                "engine-socket",
            ),
            True,
        ),
        (
            "host_traversal",
            request(
                "import os\ndef probe(path): return os.path.exists(path)",
                [["/host"], ["/proc/1/root/home/fugal-validator"]],
                [False, False],
                "host-traversal",
            ),
            True,
        ),
        (
            "network",
            request(
                "import socket\ndef probe():\n"
                " try:\n  socket.create_connection(('1.1.1.1', 53), 0.2); return True\n"
                " except OSError:\n  return False",
                [[]],
                [False],
                "network",
            ),
            True,
        ),
        (
            "process_spawn",
            request(
                "import os\ndef probe():\n"
                " try:\n  pid=os.fork(); return pid >= 0\n"
                " except OSError:\n  return False",
                [[]],
                [False],
                "process-spawn",
            ),
            True,
        ),
        (
            "read_only_root",
            request(
                "def probe():\n"
                " try:\n  open('/owned', 'w').write('x'); return True\n"
                " except OSError:\n  return False",
                [[]],
                [False],
                "read-only-root",
            ),
            True,
        ),
    ]
    failures = []
    for label, item, wanted in cases:
        result = service.handle(peer, canonical_json(item))
        actual = result["ok"] and result["passed"]
        print(f"{label}: {'PASS' if actual == wanted else 'FAIL'}", flush=True)
        if actual != wanted:
            failures.append(label)

    timeout_payload = request("def probe():\n while True: pass", [[]], [0], "timeout")
    timeout_result = service.handle(peer, canonical_json(timeout_payload))
    timeout_ok = timeout_result["ok"] and not timeout_result["passed"]
    print(f"timeout: {'PASS' if timeout_ok else 'FAIL'}", flush=True)
    if not timeout_ok:
        failures.append("timeout")

    output_payload = request(
        "import os\ndef probe():\n os.write(1, b'x' * 2000000); return 1",
        [[]],
        [1],
        "oversized-output",
    )
    output_result = service.handle(peer, canonical_json(output_payload))
    output_ok = output_result["ok"] and not output_result["passed"]
    print(f"oversized_output: {'PASS' if output_ok else 'FAIL'}", flush=True)
    if not output_ok:
        failures.append("oversized_output")

    for label, code in (
        ("missing_function", ""),
        ("system_exit", "raise SystemExit(0)"),
        ("direct_exit", "import os\nos._exit(0)"),
    ):
        rejected = service.handle(
            peer,
            canonical_json(request(code, [[]], [0], label.replace("_", "-"))),
        )
        rejected_ok = rejected["ok"] and not rejected["passed"]
        print(f"{label}: {'PASS' if rejected_ok else 'FAIL'}", flush=True)
        if not rejected_ok:
            failures.append(label)

    math_request = {
        "schema_version": SCHEMA_VERSION,
        "job_id": "symbolic-math",
        "operation": "symbolic_math",
        "gold": "\\frac{1}{2}",
        "reply": "The answer is $0.5$.",
    }
    result = service.handle(peer, canonical_json(math_request))
    math_ok = result["ok"] and result["passed"]
    print(f"symbolic_math: {'PASS' if math_ok else 'FAIL'}", flush=True)
    if not math_ok:
        failures.append("symbolic_math")

    with tempfile.TemporaryDirectory(prefix="fugal-sandbox-") as directory:
        socket_path = Path(directory) / "grader.sock"
        launcher = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "fugal_subnet.sandbox.launcher",
                "--socket",
                str(socket_path),
                "--image",
                image,
                "--allowed-uid",
                str(os.getuid()),
                "--max-concurrency",
                "1",
                "--timeout",
                "4",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 10
            while not socket_path.exists() and launcher.poll() is None:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            client = GradingClient(socket_path, timeout_seconds=8)
            socket_ok = (
                launcher.poll() is None
                and socket_path.exists()
                and client.health()
                and client.grade_code(
                    code="def add(a, b): return a + b",
                    function="add",
                    inputs=[[10, 5]],
                    expected=[15],
                    job_id="socket-roundtrip",
                )
            )
        except Exception:
            socket_ok = False
        finally:
            launcher.terminate()
            try:
                launcher.wait(timeout=5)
            except subprocess.TimeoutExpired:
                launcher.kill()
                launcher.wait(timeout=5)
        print(f"socket_roundtrip: {'PASS' if socket_ok else 'FAIL'}", flush=True)
        if not socket_ok:
            failures.append("socket_roundtrip")

    if failures:
        print("Sandbox failures: " + ", ".join(failures), flush=True)
        return 1
    print("All real OCI sandbox checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
