"""TEE runtime: MeteringProxy and benchmark execution environment.

The MeteringProxy sits between the benchmark harness and the OpenRouter API,
recording every API call with exact token counts and costs. Inside a real
TEE, the proxy's records are covered by the hardware attestation.

Forked from ThirtySpokes/Chutes (MIT licensed) runtime patterns, adapted
for Fugal's model routing benchmark.
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from threading import Thread
from urllib.request import Request

from fugal_subnet.config import PROXY_CALL_DEADLINE_S

logger = logging.getLogger(__name__)

# Upstream the metering proxy forwards to. Configurable because operators
# legitimately front OpenRouter with a gateway, a regional endpoint, or an
# OpenAI-compatible provider — and because it lets a local testnet point the
# REAL proxy at a local stub, so the metering, pricing and token accounting
# under test are the production ones rather than a monkeypatch.
_OPENROUTER_BASE = os.getenv(
    "FUGAL_OPENROUTER_BASE", "https://openrouter.ai/api/v1",
).rstrip("/")

# Upstream retry policy, inside the measured image so every miner has it.
# Mainnet day one (2026-11-03): ~13% of a seed's calls failed on 429 from one
# provider's shared pool and every failure was scored as a wrong answer, which
# put an otherwise frontier-level head 0.11 below the frontier. A bounded retry
# with backoff turns most of those into answers. Hard limits keep the whole
# attempt inside the harness's own 200 s wait: at most PROXY_MAX_ATTEMPTS tries,
# no new try after PROXY_RETRY_BUDGET_S, each wait capped at PROXY_MAX_WAIT_S.
# 401/402 are never retried: they mean the operator's key or balance, and
# they are logged as such so a payment problem is never read as model quality.
PROXY_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
PROXY_OPERATOR_STATUS = {
    401: "unauthorized key",
    402: "payment required: the account balance is exhausted",
    # A key with a total spend limit returns 403 once the limit is reached, on
    # every call: an operator problem exactly like 402, never transient.
    403: "forbidden: the key's spend limit is reached or the key is disabled",
}
# OpenRouter's in-flight spending budget (prepaid accounts below a balance
# threshold, and newer accounts): every running request holds its input plus
# max_tokens at the endpoint's prices, and a request that does not fit is
# refused with 402 before it reaches a provider, even with a positive balance.
# The 402 body's metadata.reason says which: "in_flight_budget_exhausted" is
# temporary (the holds of running calls release) and is retried like a 429;
# "weight_exceeds_budget" means this one request is larger than the whole budget
# and no retry can help. Any other 402 is the balance itself.
OPENROUTER_402_TRANSIENT = "in_flight_budget_exhausted"
OPENROUTER_402_TOO_LARGE = "weight_exceeds_budget"
PROXY_MAX_ATTEMPTS = 3
PROXY_RETRY_BUDGET_S = 300.0   # no new attempt after this; the call itself ends by PROXY_CALL_DEADLINE_S
PROXY_MAX_WAIT_S = 20.0


def _retry_wait(attempt: int, retry_after: str | None) -> float:
    """Backoff in seconds before attempt+1: 1, 3, 9 … with jitter, or the
    upstream's Retry-After when it is larger, capped at PROXY_MAX_WAIT_S."""
    import random
    base = min(PROXY_MAX_WAIT_S, 3.0 ** (attempt - 1)) + random.uniform(0.0, 1.0)
    if retry_after:
        try:
            base = max(base, float(retry_after))
        except ValueError:
            pass
    return min(PROXY_MAX_WAIT_S, base)


def _log_cap_overrun(model_id: str, completion_tokens) -> None:
    """Name any reply that exceeded the harness's answer budget.

    Some providers bound reasoning tokens loosely or not at all, so a reply can
    bill more than `HARNESS_MAX_TOKENS` asked for (measured 2026-11-04: one model
    returned 17,763 against 16,384). Nothing is refused, because the call
    happened and its tokens are attested and billed either way; the warning is so
    the operator sees which model, how far over and how often, from every call
    rather than from a one-off probe.
    """
    from fugal_subnet.config import HARNESS_MAX_TOKENS
    try:
        n = int(completion_tokens or 0)
    except (TypeError, ValueError):
        return
    if n > HARNESS_MAX_TOKENS:
        logger.warning("Model %s returned %d completion tokens, over the %d answer budget (+%d)",
                       model_id, n, HARNESS_MAX_TOKENS, n - HARNESS_MAX_TOKENS)


def _402_reason(err) -> str:
    """metadata.reason from an OpenRouter 402 body, or "" when there is none."""
    try:
        body = json.loads(err.read() or b"{}")
        meta = (body.get("error") or {}).get("metadata") or {}
        reason = meta.get("reason", "")
        return reason if isinstance(reason, str) else ""
    except Exception:
        return ""


class UpstreamDeadline(TimeoutError):
    """The upstream call ran past PROXY_CALL_DEADLINE_S of wall-clock time."""


def _post_with_deadline(req, deadline: float) -> bytes:
    """POST `req` and return the body, or raise by `deadline` (time.monotonic()).

    A socket timeout alone is not a deadline: it bounds each read, and OpenRouter
    keeps a non-streamed request alive while the model reasons, so a call with a
    "420 s" timeout was measured at 917 s. Here every read gets the time left as
    its timeout, and a watchdog shuts the socket down at the deadline, which also
    ends a read blocked inside a chunk. The connection is closed either way.
    A non-2xx status raises urllib's HTTPError with the body, as urlopen did.
    """
    import http.client
    import io
    import socket
    import ssl
    from urllib.error import HTTPError
    from urllib.parse import urlsplit

    def left() -> float:
        return deadline - time.monotonic()

    if left() <= 0:
        raise UpstreamDeadline("no time left before the call deadline")
    u = urlsplit(req.full_url)
    if u.scheme == "https":
        conn = http.client.HTTPSConnection(u.hostname, u.port, timeout=left(),
                                           context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=left())
    fired = threading.Event()
    sock_box: list = []

    def watchdog() -> None:
        fired.set()
        for s in sock_box:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    timer = threading.Timer(max(0.0, left()), watchdog)
    timer.daemon = True
    timer.start()
    try:
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        conn.connect()
        sock = conn.sock          # kept: http.client drops conn.sock on Connection: close
        # The watchdog shuts the connection down through a DUPLICATE descriptor.
        # When the reply closes the connection (Content-Length plus HTTP/1.0 or
        # "Connection: close"), getresponse() calls conn.close(), which marks
        # `sock` closed while the response still reads through it: any call on
        # `sock` then raises EBADF. shutdown() acts on the TCP connection, not
        # the descriptor, so the duplicate still ends a blocked read.
        ctl = sock.dup()
        sock_box.append(ctl)
        if fired.is_set():
            raise UpstreamDeadline("deadline reached while connecting")
        conn.request(req.get_method(), path, body=req.data, headers=dict(req.header_items()))
        sock.settimeout(max(0.001, left()))
        resp = conn.getresponse()
        chunks = []
        while True:
            if fired.is_set() or left() <= 0:
                raise UpstreamDeadline("deadline reached while reading the reply")
            try:
                sock.settimeout(max(0.001, left()))
            except OSError:
                # `sock` was marked closed by getresponse(); the read keeps the
                # timeout set before it, and the watchdog still enforces the deadline.
                pass
            chunk = resp.read1(65536)
            if not chunk:
                break
            chunks.append(chunk)
        if fired.is_set():
            raise UpstreamDeadline("deadline reached while reading the reply")
        body = b"".join(chunks)
        if not 200 <= resp.status < 300:
            raise HTTPError(req.full_url, resp.status, resp.reason, resp.headers, io.BytesIO(body))
        return body
    except (TimeoutError, OSError, http.client.HTTPException) as e:
        if isinstance(e, (UpstreamDeadline, HTTPError)):
            raise
        if fired.is_set() or left() <= 0:
            raise UpstreamDeadline(f"deadline reached ({type(e).__name__})") from e
        raise
    finally:
        timer.cancel()
        conn.close()
        for s in sock_box:
            try:
                s.close()
            except OSError:
                pass


def _fetch_upstream(req, model_id: str) -> bytes:
    """POST to the upstream with the retry policy above. Returns the body or raises.

    PROXY_CALL_DEADLINE_S is wall-clock time for the whole call, retries included.
    """
    from urllib.error import HTTPError
    started = time.time()
    deadline = time.monotonic() + PROXY_CALL_DEADLINE_S
    attempt = 0
    while True:
        attempt += 1
        try:
            return _post_with_deadline(req, deadline)
        except UpstreamDeadline:
            logger.warning(
                "Upstream call for %s hit the %.0f s wall-clock deadline; connection closed, "
                "recorded as failed. The provider bills a non-streamed reply in full, so the "
                "call may still be charged.", model_id, PROXY_CALL_DEADLINE_S,
            )
            raise
        except HTTPError as e:
            code = e.code
            transient = False
            if code == 402:
                reason = _402_reason(e)
                if reason == OPENROUTER_402_TRANSIENT:
                    transient = True
                elif reason == OPENROUTER_402_TOO_LARGE:
                    logger.error(
                        "Upstream 402 for %s: this one request is larger than the key's whole "
                        "OpenRouter in-flight budget (weight_exceeds_budget). OPERATOR ACTION "
                        "REQUIRED: add credits; not retried; the call is recorded as failed",
                        model_id,
                    )
                    raise
            if code in PROXY_OPERATOR_STATUS and not transient:
                logger.error(
                    "Upstream %s for %s: %s. OPERATOR ACTION REQUIRED (OpenRouter key or "
                    "balance); not retried; the call is recorded as failed",
                    code, model_id, PROXY_OPERATOR_STATUS[code],
                )
                raise
            elapsed = time.time() - started
            retryable = code in PROXY_RETRYABLE_STATUS or transient
            if retryable and attempt < PROXY_MAX_ATTEMPTS and elapsed < PROXY_RETRY_BUDGET_S:
                wait = _retry_wait(attempt, e.headers.get("Retry-After") if e.headers else None)
                logger.warning("Upstream %s%s for %s; retry %d/%d in %.1fs", code,
                               " (OpenRouter in-flight budget full)" if transient else "",
                               model_id, attempt, PROXY_MAX_ATTEMPTS - 1, wait)
                time.sleep(wait)
                continue
            if transient:
                logger.error(
                    "Upstream 402 for %s: the key's OpenRouter in-flight budget stayed full "
                    "through %d attempts; the call is recorded as failed. A low balance throttles "
                    "concurrent calls: add credits (see MINER_GUIDE, harness concurrency)",
                    model_id, attempt,
                )
            raise


class UnpricedModel(RuntimeError):
    """Raised when a routed model has no entry in the pinned price table."""


@dataclass
class APICallRecord:
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float           # priced from the pinned table — the consensus figure
    timestamp: float
    response_hash: str
    provider_cost_usd: float = 0.0  # what the provider itself reported, if any
    # Which question this call belonged to, from the X-Fugal-Request header the
    # harness sends. The harness attributes cost by this, not by list position:
    # with several calls in flight the records interleave, and "the records
    # appended since I started" would bill one question for another's tokens.
    request_id: str = ""


@dataclass
class MeteringProxy:
    """Records API calls for attestation. In real TEE mode, this runs
    inside the confidential VM so its records are hardware-attested.

    Cost is priced from the pinned table (`data/models.json`), not from
    whatever the provider happened to bill. That split is deliberate:

    - The **pinned table** is the consensus denominator. Every validator must
      reach the same cost for the same proof, and provider prices move without
      warning, so two miners benchmarking hours apart would otherwise be scored
      against different denominators.
    - The **provider's own figure** is recorded alongside it, attested, so a
      drift between the table and reality is detectable rather than silent.
    """

    port: int = 8199
    api_key: str = ""
    records: list[APICallRecord] = field(default_factory=list)
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)
    _server: HTTPServer | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _thread: Thread | None = field(default=None, repr=False)

    def price_call(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost of one call under the pinned price table.

        An unpriced model is a hard error, never a default. A silent fallback
        rate is what let every model cost the same and made routing to a cheap
        model indistinguishable from routing to a frontier one.
        """
        if model_id not in self.prices:
            raise UnpricedModel(
                f"Model {model_id!r} is not in the pinned price table "
                f"({len(self.prices)} models). Routing to an unpriced model "
                "cannot be costed, so it cannot be scored."
            )
        p_in, p_out = self.prices[model_id]
        return prompt_tokens * p_in + completion_tokens * p_out

    def start(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.prices:
            from fugal_subnet.api import load_prices
            self.prices = load_prices()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                request_data = json.loads(body)
                model_id = request_data.get("model", "unknown")
                request_id = self.headers.get("X-Fugal-Request", "")

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
                    resp_body = _fetch_upstream(req, model_id)
                    resp_data = json.loads(resp_body)

                    usage = resp_data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

                    cost = proxy.price_call(model_id, prompt_tokens, completion_tokens)
                    resp_hash = hashlib.sha256(resp_body).hexdigest()
                    _log_cap_overrun(model_id, completion_tokens)

                    provider_cost = 0.0
                    try:
                        provider_cost = float(usage.get("cost") or 0.0)
                    except (TypeError, ValueError):
                        provider_cost = 0.0

                    with proxy._lock:
                        proxy.records.append(APICallRecord(
                            model_id=model_id,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            cost_usd=cost,
                            timestamp=time.time(),
                            response_hash=resp_hash,
                            provider_cost_usd=provider_cost,
                            request_id=request_id,
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

            def log_message(self, format, *args):
                logger.debug(format, *args)

        # Fall back to an ephemeral port if the configured one is taken.
        # The port is a private detail between this proxy and the harness that
        # calls it — nothing on the wire and nothing in consensus depends on
        # its value — but it is a single fixed constant, so two miners sharing
        # a host collide on it and the second one dies with EADDRINUSE at the
        # start of every epoch, forever, benchmarking nothing. Binding 0 and
        # recording what the kernel gave us costs nothing and removes a whole
        # class of "my second miner earns zero" reports.
        # Threaded: the harness keeps several model calls in flight
        # (config.HARNESS_CONCURRENCY), and a single-threaded server would
        # serialise them again behind its accept loop.
        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            actual = self._server.server_address[1]
            logger.warning(
                "MeteringProxy port %d is in use (another miner on this host?); "
                "using ephemeral port %d instead", self.port, actual,
            )
            self.port = actual
        # port=0 asks the OS for an ephemeral port; report the one actually bound.
        self.port = self._server.server_address[1]
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

    # math.fsum, not sum: with several calls in flight the records complete in
    # a different order every run, and a left-to-right float sum rounds
    # differently with the order. fsum returns the correctly rounded total
    # whatever the order, so the attested figures do not depend on scheduling.
    @property
    def total_cost(self) -> float:
        return math.fsum(r.cost_usd for r in self.records)

    @property
    def provider_total_cost(self) -> float:
        """What the provider itself reported, where it reported anything."""
        return math.fsum(r.provider_cost_usd for r in self.records)

    @property
    def per_model_costs(self) -> dict[str, float]:
        by_model: dict[str, list[float]] = {}
        for r in self.records:
            by_model.setdefault(r.model_id, []).append(r.cost_usd)
        return {m: math.fsum(v) for m, v in sorted(by_model.items())}

    def clear(self) -> None:
        self.records.clear()


@dataclass
class TEERuntime:
    """Manages the TEE execution environment for a benchmark run."""

    mock: bool = True
    proxy: MeteringProxy | None = None

    def setup(self, api_key: str = "", proxy_port: int = 8199,
              prices: dict | None = None) -> MeteringProxy:
        self.proxy = MeteringProxy(port=proxy_port, api_key=api_key)
        if prices:
            # The epoch's protocol table (routing_protocol.price_table(version)), so a v2 call is
            # priced on v2 rates; with none, the proxy loads the v1 table as before.
            self.proxy.prices = dict(prices)
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


_CONFIGFS_TSM = "/sys/kernel/config/tsm/report"


def _quote_via_configfs(report_data: bytes) -> bytes | None:
    """Ask the kernel for a TDX quote through configfs-tsm.

    This is the standard interface on Linux 6.7+ and is what GCP c3 and Azure
    DCesv5 confidential guests actually expose. Create a report directory,
    write the 64-byte report_data to `inblob`, read the quote from `outblob`.

    Returns None if the interface is not present, so the caller can fall back.
    Requires write access to configfs — run the miner as root, or grant it.
    """
    import os
    import uuid
    from pathlib import Path

    base = Path(_CONFIGFS_TSM)
    if not base.is_dir():
        return None

    report_dir = base / f"fugal-{uuid.uuid4().hex[:12]}"
    try:
        report_dir.mkdir()
    except OSError as e:
        raise RuntimeError(
            f"configfs-tsm is present at {base} but a report could not be "
            f"created ({e}). This usually means insufficient privilege — the "
            "quote generator needs write access to configfs."
        ) from e

    try:
        (report_dir / "inblob").write_bytes(report_data)
        quote = (report_dir / "outblob").read_bytes()
    except OSError as e:
        raise RuntimeError(f"configfs-tsm quote generation failed: {e}") from e
    finally:
        try:
            os.rmdir(report_dir)
        except OSError:
            logger.warning("Could not clean up %s", report_dir)

    if not quote:
        raise RuntimeError(
            "configfs-tsm returned an empty quote — the guest reports a TSM "
            "provider but produced nothing."
        )
    return quote


def _real_quote(report_data: bytes) -> bytes:
    """Generate a real TDX quote.

    Three paths, tried in order, and the order is load-bearing:

    0. **The dstack guest agent**, when its socket is present. Under a measured
       dstack image this is the only source that yields a usable proof: it
       returns the whole envelope — quote, event log, and on GCP a TPM quote —
       and RTMR3 on a measured image is a chain including per-deploy values, so
       without the log there is nothing to replay and an approved entry naming
       a compose hash cannot be checked. Falling through to configfs here would
       produce a bare quote and a rejection that reads as "unapproved image"
       rather than "wrong API".
    1. **configfs-tsm** (`/sys/kernel/config/tsm/report`) — the kernel's own
       interface on Linux 6.7+, and what the confidential guests on GCP and
       Azure actually provide. No vendor package required.
    2. A vendor quote-generator binary, if one happens to be installed.

    NOTE: this path cannot be exercised without TDX silicon, so it is verified
    by `scripts/tdx_measurement.py --verify` on a real confidential VM rather
    than by any test in this repo. See docs/TDX_VALIDATION.md.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    padded = report_data[:64].ljust(64, b"\x00")

    from fugal_subnet.tee import dstack_client

    if dstack_client.available():
        # No try/except: under dstack this is the correct source and a failure
        # here must stop the miner, not quietly downgrade it to a proof shape
        # that every validator rejects.
        blob = dstack_client.attest(padded)
        logger.info("attestation obtained from the dstack guest agent (%d bytes)",
                    len(blob))
        return blob

    quote = _quote_via_configfs(padded)
    if quote is not None:
        logger.info("TDX quote obtained via configfs-tsm (%d bytes)", len(quote))
        return quote

    generator = "/usr/bin/tdx-quote-generator"
    if not Path(generator).exists():
        raise RuntimeError(
            f"No TDX quote source. Neither {_CONFIGFS_TSM} (the kernel's "
            f"configfs-tsm interface, Linux 6.7+) nor {generator} is present.\n"
            "This requires an Intel TDX guest — see docs/TDX_VALIDATION.md. "
            "Check with: ls /dev/tdx_guest"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "quote.bin"
        result = subprocess.run(
            [generator, "--report-data", padded.hex(), "--hex", "--output", str(out)],
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
