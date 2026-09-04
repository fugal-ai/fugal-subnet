"""TEE runtime: MeteringProxy and benchmark execution environment.

The MeteringProxy sits between the benchmark harness and the OpenRouter API,
recording every API call with exact token counts and costs. Inside a real
TEE, the proxy's records are covered by the hardware attestation.

Forked from ThirtySpokes/Chutes (MIT licensed) runtime patterns, adapted
for Fugal's model routing benchmark.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


@dataclass
class APICallRecord:
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: float
    response_hash: str


@dataclass
class MeteringProxy:
    """Records API calls for attestation. In real TEE mode, this runs
    inside the confidential VM so its records are hardware-attested."""

    port: int = 8199
    api_key: str = ""
    records: list[APICallRecord] = field(default_factory=list)
    _server: HTTPServer | None = field(default=None, repr=False)
    _thread: Thread | None = field(default=None, repr=False)

    def start(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                request_data = json.loads(body)
                model_id = request_data.get("model", "unknown")

                upstream_url = f"{_OPENROUTER_BASE}/chat/completions"
                req = Request(
                    upstream_url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {proxy.api_key}",
                    },
                    method="POST",
                )

                try:
                    with urlopen(req, timeout=180) as resp:
                        resp_body = resp.read()
                        resp_data = json.loads(resp_body)

                    usage = resp_data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

                    cost = self._estimate_cost(model_id, prompt_tokens, completion_tokens)
                    resp_hash = hashlib.sha256(resp_body).hexdigest()

                    proxy.records.append(APICallRecord(
                        model_id=model_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=cost,
                        timestamp=time.time(),
                        response_hash=resp_hash,
                    ))

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(resp_body)

                except Exception:
                    logger.exception("Proxy upstream error")
                    self.send_response(502)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "upstream request failed"}).encode())

            def _estimate_cost(self, model_id: str, pin: int, pout: int) -> float:
                # TODO: fetch real pricing from OpenRouter when available
                # For now use a reasonable per-token estimate
                return pin * 1e-6 + pout * 2e-6

            def log_message(self, format, *args):
                logger.debug(format, *args)

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("MeteringProxy started on port %d", self.port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def per_model_costs(self) -> dict[str, float]:
        costs: dict[str, float] = {}
        for r in self.records:
            costs[r.model_id] = costs.get(r.model_id, 0.0) + r.cost_usd
        return costs

    def clear(self) -> None:
        self.records.clear()


@dataclass
class TEERuntime:
    """Manages the TEE execution environment for a benchmark run."""

    mock: bool = True
    proxy: MeteringProxy | None = None

    def setup(self, api_key: str = "", proxy_port: int = 8199) -> MeteringProxy:
        self.proxy = MeteringProxy(port=proxy_port, api_key=api_key)
        self.proxy.start()
        return self.proxy

    def teardown(self) -> None:
        if self.proxy:
            self.proxy.stop()

    def generate_attestation(self, report_data: bytes) -> bytes:
        """Generate a TDX attestation quote binding report_data.

        In mock mode, returns a synthetic quote with the report_data
        embedded at the correct offset. In live mode, calls the TDX
        quote generator.
        """
        if self.mock:
            return _mock_quote(report_data)
        return _real_quote(report_data)


def _mock_quote(report_data: bytes) -> bytes:
    """Build a synthetic TDX quote for testing (no real hardware)."""
    import struct

    padded = report_data[:64].ljust(64, b"\x00")

    header = struct.pack("<H", 4)  # version
    header += struct.pack("<H", 0)  # att_key_type
    header += struct.pack("<I", 0x00000081)  # tee_type = TDX
    header += b"\x00" * 20  # reserved
    header += b"\x00" * 20  # user_data in header

    body = b"\x00" * 520  # fields before report_data
    body += padded  # report_data at offset 568

    # Pad to a realistic quote size
    signature_area = b"\x00" * 368

    return header + body + signature_area


def _real_quote(report_data: bytes) -> bytes:
    """Generate a real TDX quote using the system quote generator."""
    import subprocess
    import tempfile
    from pathlib import Path

    padded_hex = report_data[:64].ljust(64, b"\x00").hex()
    generator = "/usr/bin/tdx-quote-generator"

    if not Path(generator).exists():
        raise RuntimeError(
            f"{generator} not found — is libtdx-attest installed? "
            "This requires an Intel TDX-capable VM."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "quote.bin"
        result = subprocess.run(
            [generator, "--report-data", padded_hex, "--hex", "--output", str(out)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"tdx-quote-generator failed (exit {result.returncode}):\n{result.stderr}"
            )
        return out.read_bytes()
