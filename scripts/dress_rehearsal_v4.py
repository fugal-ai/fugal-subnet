"""Scenario v4_flip of scripts/dress_rehearsal.py: protocol v4 across its
activation block, the shipped neurons against the real local chain.

    python scripts/dress_rehearsal.py --scenario v4_flip

The in-process rehearsal (scripts/rehearsal_v4.py, tests/v4_world.py) runs
the same loops on a fake chain, store and clock. This one runs them as real
OS processes against a real subtensor node: real wallets and registration,
real axon/dendrite traffic, real `set_commitment` and `set_weights`
extrinsics (through the chain's weight commit-reveal), real pinned reads
against a node that prunes state, and real wall-clock timing.

Timeline (S the epoch the scenario starts in, F = S + 3 the first v4 epoch,
epochs of --v4-epoch-seconds):

    S, S+1, S+2  protocol v3. A root (val1) and a follower (val2) run
                 continuously; one v3 miner (m1's v3 image) serves S+1 and
                 S+2, so the root publishes v3 reveals and commits them in the
                 three-field form. Four v4 miners (m1..m4, the v4 image) start
                 at S, hold their axons and commit their v4 heads after the
                 last v3 boundary.
    F            the flip: protocol 4 everywhere, the pinned seed menu, an
                 empty seat and window, the published set brought forward
                 from the seed by the root's commitments read pinned.
    F+1          m2 is restarted from its kept state.
    F+1, F+2     the root is down; miners keep proving, the follower keeps
                 verifying. It returns at F+3 and publishes its backlog.
    F+3, F+4     the follower is down; it returns at F+5, re-anchors and
                 replays, and agrees with the root from F+6.
    to F+7

Every miner explores only (--v4-challenge 0) and files nothing: the seat
and the chain proposals are scenario v4_seat_menu's
(scripts/dress_rehearsal_v4_seat.py), which reuses this module's world.

Stand-ins, the only ones (scripts/rehearsal_shim.py says why each is
needed): the public store is a local HTTP server serving a directory the way
GCS serves a public bucket, the published-set seed is a fixture seed on the
fixture pool, the signed lab and host list is the draft signed by a throwaway
key, and the listing the root reads for a proposal is the local stub's.
Model replies and listing pages come from the local stub (no key, no spend).
Everything else is the release's: the pinned menu seed, mix and constants,
the compiled v4 rules.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

S_NAME = "v4_flip"
BUCKET = "fugal-rehearsal-v4"
STORE_PORT = 8797
STUB_PORT = 8798
ARCHIVE_PORT = 9947
ARCHIVE_ENDPOINT = f"ws://127.0.0.1:{ARCHIVE_PORT}"
V3_EPOCHS_BEFORE = 3              # F = S + 3
ROOT_DOWN = (1, 2)                # the root misses F+1 and F+2
B_RESTART = 1                     # m2 restarted from its kept state at F+1
FOLLOWER_DOWN = (3, 4)            # the follower misses F+3 and F+4
END = 7                           # through F+7: the follower agrees again from F+6
MINERS = ("dr_m1", "dr_m2", "dr_m3", "dr_m4")
SHARP, SHARPER = 60.0, 240.0      # tests/v4_world's seat heads


# ── the public store ──────────────────────────────────────────────────────────

class LocalStore:
    """A directory served over HTTP the way GCS serves a public bucket:
    GET /<bucket>/<key> and the JSON API listing
    /storage/v1/b/<bucket>/o?prefix=... (name, size, md5Hash). Writes go to
    the directory itself (scripts/rehearsal_shim.py, create-only)."""

    def __init__(self, root: str, port: int = STORE_PORT):
        self.root, self.port = root, port
        self.gets: list[str] = []
        self._server = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parts = urlsplit(self.path)
                outer.gets.append(parts.path)
                if parts.path.startswith("/storage/v1/b/"):
                    bucket = parts.path.split("/")[4]
                    prefix = parse_qs(parts.query).get("prefix", [""])[0]
                    items = [{"name": k, "size": str(len(v)),
                              "md5Hash": base64.b64encode(hashlib.md5(v).digest()).decode()}
                             for k, v in sorted(outer.objects(bucket, prefix).items())]
                    return self._send(200, json.dumps({"items": items}).encode())
                bucket, _, key = parts.path.lstrip("/").partition("/")
                path = os.path.join(outer.root, bucket, key)
                if ".." in key.split("/") or not os.path.isfile(path):
                    return self._send(404, b"Not Found")
                with open(path, "rb") as f:
                    return self._send(200, f.read())

            def _send(self, code, body):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def objects(self, bucket: str = BUCKET, prefix: str = "") -> dict[str, bytes]:
        out = {}
        base = os.path.join(self.root, bucket)
        for d, _dirs, files in os.walk(base):
            for name in files:
                if name.endswith(".tmp"):
                    continue
                path = os.path.join(d, name)
                key = os.path.relpath(path, base).replace(os.sep, "/")
                if key.startswith(prefix):
                    with open(path, "rb") as f:
                        out[key] = f.read()
        return out


# ── the model provider: replies and listing pages ─────────────────────────────

SUPPORTED = ["max_tokens", "reasoning", "reasoning_effort"]


def _decimal(pico: int) -> str:
    whole, frac = divmod(int(pico), 10 ** 12)
    frac_s = str(frac).rjust(12, "0").rstrip("0")
    return f"{whole}.{frac_s}" if frac_s else f"{whole}"


class V4Stub:
    """The upstream every proxy forwards to (FUGAL_OPENROUTER_BASE), and the
    listing the v4 TDs read through the same base URL. A v4 entry's reply is
    right exactly when the question's fixture embedding is positive on the
    entry's coordinate; anything else (the v3 miner's calls) by a hash.
    Billed at the seed's listed prices; no key exists anywhere."""

    def __init__(self, pool: list, hidden, seed_doc: dict, port: int = STUB_PORT,
                 extra_models: dict | None = None):
        self.port = port
        self.by_prompt = {q["prompt"]: q for q in pool}
        self.row = {q["question_id"]: i for i, q in enumerate(pool)}
        self.hidden = hidden
        entries = seed_doc["entries"]
        self.coord = {eid: i for i, eid in enumerate(sorted(x["id"] for x in entries))}
        self.entry_of = {}
        self.listing: dict[str, list] = {}
        for e in entries:
            effort = (e.get("reasoning") or {}).get("effort")
            self.entry_of[(e["model"], effort)] = e
            hosts = self.listing.setdefault(e["model"], [])
            for tag in (e.get("provider") or {}).get("order") or []:
                if tag not in [h["tag"] for h in hosts]:
                    hosts.append({"tag": tag, "pricing": {"prompt": _decimal(e["price_in"]),
                                                          "completion": _decimal(e["price_out"])},
                                  "max_completion_tokens": int(e.get("max_tokens") or 0),
                                  "quantization": e.get("quantization") or "unknown",
                                  "status": 0, "supported_parameters": SUPPORTED})
        # Models a scenario's miners propose (chain_proposals): {model: {"hosts",
        # "quantization", "price_in", "price_out", "answers": "right" | "fail"}}.
        # "right": every reply correct; "fail": every call answered 500.
        self.extra = dict(extra_models or {})
        for model, x in self.extra.items():
            self.listing[model] = [{"tag": h, "pricing": {"prompt": _decimal(x["price_in"]),
                                                          "completion": _decimal(x["price_out"])},
                                    "max_completion_tokens": 16384,
                                    "quantization": x["quantization"], "status": 0,
                                    "supported_parameters": SUPPORTED} for h in x["hosts"]]
        self.extra_calls: dict[str, int] = {m: 0 for m in self.extra}
        self.calls = 0
        self.listing_reads = 0
        self._server = None

    def right(self, prompt: str, model: str, effort) -> bool:
        if self.extra.get(model, {}).get("answers") == "right":
            return True
        q = self.by_prompt.get(prompt)
        e = self.entry_of.get((model, effort))
        if q is not None and e is not None:
            return float(self.hidden[self.row[q["question_id"]]][self.coord[e["id"]]]) > 0
        return hashlib.sha256(f"{prompt}|{model}|{effort}".encode()).digest()[0] < 128

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = urlsplit(self.path).path
                if path.startswith("/v1/models/") and path.endswith("/endpoints"):
                    model = path[len("/v1/models/"):-len("/endpoints")]
                    outer.listing_reads += 1
                    body = json.dumps({"data": {"id": model,
                                                "endpoints": outer.listing.get(model, [])}})
                    return self._send(200, body.encode())
                return self._send(404, b"{}")

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                users = [m for m in req["messages"] if m.get("role") == "user"]
                prompt = (users or req["messages"])[-1]["content"]
                model = req.get("model", "")
                effort = (req.get("reasoning") or {}).get("effort")
                x = outer.extra.get(model)
                if x is not None:
                    outer.extra_calls[model] += 1
                    if x["answers"] == "fail":
                        return self._send(500, b'{"error": {"message": "upstream down"}}')
                q = outer.by_prompt.get(prompt)
                text = (q["gold"] if q else "0") if outer.right(prompt, model, effort) else "0"
                d = hashlib.sha256(f"{prompt}|{model}|{effort}".encode()).digest()
                pt, ct = 400 + d[1] % 200, 100 + d[2] % 100
                usage = {"prompt_tokens": pt, "completion_tokens": ct}
                e = outer.entry_of.get((model, effort)) or x
                if e is not None:
                    usage["cost"] = (pt * e["price_in"] + ct * e["price_out"]) / 1e12
                outer.calls += 1
                body = json.dumps({"choices": [{"message": {"content": text}}],
                                   "provider": "Rehearsal Host", "usage": usage}).encode()
                self._send(200, body)

            def _send(self, code, body):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 512

        self._server = Server(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


# ── fixtures ──────────────────────────────────────────────────────────────────

def fixture_pool(unpublished_per_set: int = 4) -> list[dict]:
    """Every set of the pinned v4 mix (the v3 pool's sets), numeric answers,
    each category holding comfortably more published questions than
    memo_v4.PUBLISHED_MIN_PER_CATEGORY, plus a small unpublished reserve."""
    import math

    from fugal_subnet import epoch_plan, memo_v4

    mix = epoch_plan.mix_v4()
    by_cat: dict[str, list[str]] = {}
    for b, c in sorted(mix.bench_category.items()):
        by_cat.setdefault(c, []).append(b)
    pool = []
    for _cat, sets in sorted(by_cat.items()):
        per = math.ceil((memo_v4.PUBLISHED_MIN_PER_CATEGORY + 60) / len(sets)) + unpublished_per_set
        for b in sets:
            for i in range(per):
                pool.append({"question_id": f"dr4_{b}_{i:04d}",
                             "prompt": f"[v4 {b} {i}] What is {i} + 7?",
                             "gold": str(i + 7), "grader_id": "numeric_final",
                             "benchmark": b, "metadata": {}})
    return pool


def write_fixtures(d: str, through: int) -> dict:
    """The pool, its embeddings and the published-set seed (observed through
    epoch `through`, the last questions of each set unpublished)."""
    import numpy as np

    from tests import v4_world

    pool = fixture_pool()
    pool_path = os.path.join(d, "pool_v4.json")
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(pool, f)
    rng = np.random.RandomState(4)
    H = (rng.uniform(-1, 1, size=(len(pool), 1024)) * 0.05).astype(np.float32)
    emb = os.path.join(d, "embeddings_v4.npz")
    np.savez(emb, question_ids=np.array([q["question_id"] for q in pool]), H=H)
    seed = v4_world.fixture_seed(pool, through, v4_world.fixture_unpublished(pool, per_set=4))
    seed_path = os.path.join(d, "published_v4_seed.json")
    with open(seed_path, "wb") as f:
        f.write(json.dumps(seed, sort_keys=True).encode())
    return {"pool": pool, "pool_path": pool_path, "hidden": H, "emb": emb,
            "seed": seed, "seed_path": seed_path}


def write_signed_list(d: str) -> dict:
    """The draft lab and host list at serial 1, canonical, signed by a
    throwaway key (tests/signed_list_fixtures.Signer), and that key's
    signers file: what the release will ship, for scripts/rehearsal_shim.py."""
    from pathlib import Path

    from fugal_subnet import menu_rules as mr
    from tests.signed_list_fixtures import Signer

    root = Path(os.path.dirname(__file__)).parent
    shipped = root / "fugal_subnet" / "signed_lists" / "menu_allowlist.json"
    draft = json.loads((shipped if shipped.exists()
                        else root / "data" / "menu_allowlist.draft.json").read_text())
    payload = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, 1, labs=draft["labs"],
                                      hosts=draft["hosts"],
                                      first_party=draft["first_party"])
    out = Path(d) / "signed_lists"
    out.mkdir(parents=True, exist_ok=True)
    signer = Signer(Path(d), "rehearsal_list_key")
    name = mr.SHIPPED_LISTS[mr.KIND_MENU_ALLOWLIST]
    (out / name).write_bytes(payload)
    (out / f"{name}.sig").write_bytes(signer.sign(payload))
    signers = Path(d) / "rehearsal_list_signers"
    signers.write_text(signer.signers_line())
    mr.verify_signed_list(payload, (out / f"{name}.sig").read_bytes(),
                          kind=mr.KIND_MENU_ALLOWLIST, held_serial=0,
                          signers=signers.read_bytes())
    return {"FUGAL_REHEARSAL_SIGNED_LISTS": str(out),
            "FUGAL_REHEARSAL_LIST_SIGNERS": str(signers)}


def seed_menu_doc() -> dict:
    from fugal_subnet import menu_state as ms

    return json.loads(ms.SEED_PATH.read_text(encoding="ascii"))


def v4_head(path: str, coord: dict, sharp: float | None, tag: str) -> str:
    """A v4 head over every seed entry with the given sharpness (None: one
    50% row, a labeller), tests/v4_world.seat_head_bytes."""
    from tests import v4_world

    rows = ({e: sharp for e in coord} if sharp is not None
            else {sorted(coord)[0]: 0.0})
    data = v4_world.seat_head_bytes(rows, coord, tag)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ── the chain: an archive node, and what the scenario reads itself ───────────

def start_archive_node(chain_name: str) -> bool:
    """A fourth node in the chain's container, keeping all state
    (--state-pruning archive), RPC on ARCHIVE_PORT (published by
    start_chain for this scenario). The three localnet nodes keep the
    default 256 blocks of state."""
    cmd = ("/scripts/../target/fast-runtime/release/node-subtensor --base-path /tmp/archive "
           "--chain=/scripts/specs/local.json --port 30337 "
           f"--rpc-port {ARCHIVE_PORT} --state-pruning archive --blocks-pruning archive "
           "--rpc-cors=all --allow-private-ipv4 --discover-local --unsafe-rpc-external "
           "--unsafe-force-node-key-generation > /tmp/archive.log 2>&1")
    out = subprocess.run(["docker", "exec", "-d", chain_name, "sh", "-c", cmd],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(f"    archive node did not start: {out.stderr.strip()[:200]}", flush=True)
        return False
    import bittensor as bt
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            st = bt.Subtensor(network=ARCHIVE_ENDPOINT)
            st.get_current_block()
            st.substrate.get_block_hash(1)
            return True
        except Exception:  # noqa: BLE001 - starting
            time.sleep(3)
    return False


class ChainWatch:
    """Reads, at each boundary as it passes (pinned at the boundary block,
    normally within the 256 blocks the localnet nodes keep; the archive node
    behind it otherwise), every hotkey's commitment: what the chain says,
    independently of any node. `plain` is a localnet connection with no
    archive behind it, for the pruning probe."""

    def __init__(self, subtensor, netuid: int, bpe: int, plain=None):
        self.st, self.netuid, self.bpe = subtensor, netuid, bpe
        self.plain = plain if plain is not None else subtensor
        self.at: dict[int, dict[str, tuple[str, int]]] = {}
        self.read_errors: dict[int, str] = {}
        self.last = None

    def poll(self) -> int:
        from fugal_subnet.commitments import get_commitments_with_blocks

        block = self.st.get_current_block()
        idx = block // self.bpe
        if idx != self.last and idx not in self.at:
            try:
                self.at[idx] = dict(get_commitments_with_blocks(self.st, self.netuid,
                                                                block=idx * self.bpe))
            except Exception as e:  # noqa: BLE001 - recorded, a check fails on it
                self.read_errors[idx] = f"{type(e).__name__}: {e}"[:200]
            self.last = idx
        return block

    def pruned_read(self, block: int) -> str:
        """What the localnet node (no archive) answers for a pinned read of
        `block`: 'ok' or the error's name."""
        try:
            list(self.plain.query_map("Commitments", "CommitmentOf", params=[self.netuid],
                                      block=int(block)))
            return "ok"
        except Exception as e:  # noqa: BLE001 - the answer is the point
            return type(e).__name__


# ── processes ─────────────────────────────────────────────────────────────────

class Procs:
    """The scenario's processes by name (validators and miners)."""

    def __init__(self, dr, netuid: int, base_env: dict, results: str):
        self.dr, self.netuid, self.base_env, self.results = dr, netuid, base_env, results
        self.running: dict[str, tuple[subprocess.Popen, str]] = {}
        self.all: list[tuple[subprocess.Popen, str]] = []

    def _popen(self, name: str, argv: list, env: dict) -> None:
        log = os.path.join(self.results, f"{name}.log")
        f = open(log, "a", encoding="utf-8")
        f.write(f"\n### {name} started at {time.strftime('%H:%M:%S')}\n")
        f.flush()
        p = subprocess.Popen([sys.executable, *argv], stdout=f, stderr=subprocess.STDOUT,
                             env=env, text=True, stdin=subprocess.DEVNULL)
        self.running[name] = (p, log)
        self.all.append((p, log))

    def validator(self, name: str, coldkey: str, extra: dict, check: bool = False):
        state = os.path.join(self.results, name, "state.json")
        os.makedirs(os.path.dirname(state), exist_ok=True)
        env = self.dr.validator_env(state, {**self.base_env, **extra})
        argv = ["scripts/rehearsal_shim.py", "neurons/validator.py",
                "--network", self.dr.ENDPOINT, "--netuid", str(self.netuid),
                "--coldkey", coldkey, "--hotkey", "default", "--mock"]
        if check:
            return self._check(name, argv, env)
        self._popen(name, argv, env)

    def _check(self, name: str, argv: list, env: dict, timeout: float = 300.0):
        """`--check`, its output in <name>.log as it comes (kept when it hangs):
        a result with `returncode` (None when it did not end in time),
        `stdout` (the log) and `seconds`."""
        import types

        log = os.path.join(self.results, f"{name}.log")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.Popen([sys.executable, *argv, "--check"], stdout=f,
                                 stderr=subprocess.STDOUT, env=env, text=True,
                                 stdin=subprocess.DEVNULL)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
                rc = None
        return types.SimpleNamespace(returncode=rc, stdout=log_text(self.results, name),
                                     stderr="", seconds=round(time.time() - t0, 1))

    def miner(self, name: str, coldkey: str, head: str, port: int, extra: dict,
              shim: bool = True, check: bool = False):
        env = self.dr.miner_env({"FUGAL_TEE_PROXY_PORT": str(port + 1000),
                                 **self.base_env, **extra})
        argv = (["scripts/rehearsal_shim.py"] if shim else []) + [
            "neurons/miner.py", "--network", self.dr.ENDPOINT, "--netuid", str(self.netuid),
            "--coldkey", coldkey, "--hotkey", "default", "--head-path", head,
            "--benchmark-pool", env["FUGAL_BENCHMARK_POOL_V3"], "--port", str(port), "--mock"]
        if check:
            return self._check(name, argv, env)
        self._popen(name, argv, env)

    def stop(self, name: str) -> None:
        p, _log = self.running.pop(name)
        p.send_signal(signal.SIGINT)        # an operator's Ctrl-C
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)

    def stop_all(self) -> None:
        """Every process this scenario started, whatever state it is in: an
        interrupt in the middle of a stop must not leave neurons running."""
        self.running.clear()
        for p, _log in self.all:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        deadline = time.time() + 30
        for p, _log in self.all:
            try:
                p.wait(timeout=max(1.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
            except KeyboardInterrupt:
                for q, _l in self.all:
                    if q.poll() is None:
                        q.kill()
                raise

    def exited(self) -> dict[str, int]:
        return {n: p.poll() for n, (p, _l) in self.running.items() if p.poll() is not None}


def epoch_log(results: str, name: str) -> dict[int, dict]:
    path = os.path.join(results, name, "epochs.jsonl")
    out: dict[int, dict] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    out[int(e["epoch_id"][1:])] = e
    return out


def reveal(results: str, name: str, k: int) -> dict | None:
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block

    path = os.path.join(results, name, "epochs", epoch_id_for_block(k), "reveal.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def log_text(results: str, name: str) -> str:
    try:
        with open(os.path.join(results, f"{name}.log"), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


# ── the scenario ──────────────────────────────────────────────────────────────

def scenario_v4_flip(dr, report, subtensor, netuid, wallets, procs_out, args):
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch, epoch_id_for_block

    S = S_NAME
    results = dr.RESULTS
    print("\n[Scenario v4_flip] protocol v4 across its activation block", flush=True)
    bpe = blocks_per_epoch(int(dr.EPOCH_INTERVAL_S))
    t_setup = time.time()

    archive = start_archive_node(dr.CHAIN_NAME)
    report.check(S, "an archive node joins the local chain (the localnet nodes keep 256 blocks)",
                 archive, ARCHIVE_ENDPOINT if archive else "no archive node")

    start_idx = subtensor.get_current_block() // bpe
    F = start_idx + V3_EPOCHS_BEFORE
    fx = write_fixtures(results, through=start_idx)
    seed_menu = seed_menu_doc()
    stub = V4Stub(fx["pool"], fx["hidden"], seed_menu)
    stub.start()
    store_dir = os.path.join(results, "store")
    os.makedirs(os.path.join(store_dir, BUCKET), exist_ok=True)
    store = LocalStore(store_dir)
    store.start()
    # gcloud out of reach: a regression in the shim fails, never reaches a bucket.
    no_gcloud = os.path.join(results, "no-gcloud")
    os.makedirs(no_gcloud, exist_ok=True)
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if not os.path.exists(os.path.join(p, "gcloud")))

    root_hk = wallets["dr_val1"].hotkey.ss58_address
    blocks = {"FUGAL_V2_ACTIVATION_BLOCK": str(start_idx * bpe),
              "FUGAL_V3_ACTIVATION_BLOCK": str(start_idx * bpe)}
    base = {**blocks,
            "PATH": path, "CLOUDSDK_CONFIG": no_gcloud,
            "FUGAL_OPENROUTER_BASE": f"http://127.0.0.1:{STUB_PORT}/v1",
            "FUGAL_BENCHMARK_POOL_V3": fx["pool_path"], "FUGAL_EMBEDDINGS_FILE": fx["emb"],
            "FUGAL_ROOT_HOTKEY": root_hk, "FUGAL_REFERENCE_BUCKET": BUCKET,
            "FUGAL_REHEARSAL_STORE_DIR": store_dir, "FUGAL_REHEARSAL_STORE_URL": store.url,
            "FUGAL_REHEARSAL_PUBLISHED_SEED": fx["seed_path"],
            **write_signed_list(results),
            **({"FUGAL_ARCHIVE_ENDPOINTS": ARCHIVE_ENDPOINT} if archive else {})}
    v4 = {"FUGAL_V4_ACTIVATION_BLOCK": str(F * bpe)}
    root_env = {**v4, "FUGAL_ROOT": "1", "FUGAL_REVEAL_BUCKET": BUCKET,
                "FUGAL_REFERENCE_BUCKET": ""}
    follower_env = dict(v4)
    print(f"    S = {epoch_id_for_block(start_idx)}, F = {epoch_id_for_block(F)} (block {F * bpe}, "
          f"{bpe} blocks an epoch); pool {len(fx['pool'])} questions; published seed through "
          f"{fx['seed']['through']}", flush=True)

    procs = Procs(dr, netuid, base, results)
    procs_out.extend([])   # the caller's list is not used: Procs stops its own
    coord = stub.coord
    heads = {"dr_m1": v4_head(os.path.join(results, "head_v4_a.npz"), coord, SHARP, "a"),
             "dr_m2": v4_head(os.path.join(results, "head_v4_b.npz"), coord, SHARPER, "b"),
             "dr_m3": v4_head(os.path.join(results, "head_v4_l1.npz"), coord, None, "l1"),
             "dr_m4": v4_head(os.path.join(results, "head_v4_l2.npz"), coord, None, "l2")}
    head_hash = {ck: hashlib.sha256(open(p, "rb").read()).hexdigest() for ck, p in heads.items()}
    v3_head = dr.write_head_v3(os.path.join(results, "head_v4f_v3.npz"), None, seed=41)
    ports = {"dr_m1": 8181, "dr_m2": 8182, "dr_m3": 8183, "dr_m4": 8184}
    # Exploration only on every miner: the seat is scenario v4_seat_menu's.
    scored = {ck: "0" for ck in MINERS}

    def miner_extra(ck):
        return {**v4, "FUGAL_V4_CHALLENGE": scored[ck],
                "FUGAL_V4_STATE_DIR": os.path.join(results, f"td_{ck}")}

    import bittensor as bt
    watch = ChainWatch(bt.Subtensor(network=dr.ENDPOINT, archive_endpoints=[ARCHIVE_ENDPOINT])
                       if archive else subtensor, netuid, bpe, plain=subtensor)
    timings: dict[str, float] = {}
    events: dict[str, int] = {}
    try:
        procs.validator("v4f_val1", "dr_val1", root_env)
        procs.validator("v4f_val2", "dr_val2", follower_env)
        # The v3 image of m1 (no v4 block: the release before v4).
        procs.miner("v4f_m1_v3", "dr_m1", v3_head, 8180, {}, shim=False)
        for ck in MINERS:
            procs.miner(f"v4f_{ck}", ck, heads[ck], ports[ck], miner_extra(ck))
        timings["setup_s"] = time.time() - t_setup
        _drive(dr, report, procs, watch, F, bpe, results, events, ports, heads, miner_extra,
               scored, root_env, follower_env, args)
    finally:
        procs.stop_all()
        stub.stop()
        store.stop()
        timings["total_s"] = time.time() - t_setup
    _v4_checks(report, S, results, watch, store, stub, F, bpe, fx, seed_menu, events, head_hash,
               wallets, netuid, subtensor, timings, procs, dr, base, root_env, follower_env,
               heads, ports, miner_extra, args)


def _drive(dr, report, procs, watch, F, bpe, results, events, ports, heads, miner_extra, scored,
           root_env, follower_env, args):
    """Run the schedule until the end epoch is logged by both validators."""
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block

    last_print = None
    deadline = time.time() + (END + V3_EPOCHS_BEFORE + 2) * args.v4_epoch_seconds + 600
    end = END
    while time.time() < deadline:
        block = watch.poll()
        idx = block // bpe
        phase = (block % bpe) / bpe
        root, fol = epoch_log(results, "v4f_val1"), epoch_log(results, "v4f_val2")
        if idx != last_print:
            last_print = idx
            print(f"    {epoch_id_for_block(idx)} (F{idx - F:+d}): root logged "
                  f"{sorted(k - F for k in root)[-3:]}, follower {sorted(k - F for k in fol)[-3:]}",
                  flush=True)
        dead = {n: rc for n, rc in procs.exited().items()}
        if dead:
            raise RuntimeError(f"processes exited: {dead}")

        # The v3 miner hands over: after the last v3 epoch's collection.
        if "v4f_m1_v3" in procs.running and idx == F - 1 and phase >= 0.75:
            procs.stop("v4f_m1_v3")
            events["v3_miner_stopped"] = block
        # The root: down for F+1 and F+2, back after the F+3 boundary.
        if "v4f_val1" in procs.running and F + ROOT_DOWN[0] - 1 in root and "root_down" not in events:
            procs.stop("v4f_val1")
            events["root_down"] = block
        if "root_down" in events and "root_back" not in events and idx >= F + ROOT_DOWN[1] + 1:
            procs.validator("v4f_val1", "dr_val1", root_env)
            events["root_back"] = block
        # m2 (B): restarted from its kept state.
        if ("b_restart" not in events and F + B_RESTART - 1 in root
                and F + B_RESTART - 1 in fol):
            procs.stop("v4f_dr_m2")
            procs.miner("v4f_dr_m2", "dr_m2", heads["dr_m2"], ports["dr_m2"],
                        miner_extra("dr_m2"))
            events["b_restart"] = block
        # The follower: down for F+4 and F+5, back after the F+6 boundary.
        if ("follower_down" not in events and F + FOLLOWER_DOWN[0] - 1 in fol):
            procs.stop("v4f_val2")
            events["follower_down"] = block
        if ("follower_down" in events and "follower_back" not in events
                and idx >= F + FOLLOWER_DOWN[1] + 1):
            procs.validator("v4f_val2", "dr_val2", follower_env)
            events["follower_back"] = block
        # --check while the follower is down (its wallet, a fresh state), in
        # the background so the boundary reads above go on.
        if "follower_down" in events and "checks" not in events:
            events["checks"] = block
            threading.Thread(target=lambda: events.__setitem__("check_results", _run_checks(
                procs, results, root_env, follower_env, heads, miner_extra)), daemon=True).start()
        if F + end in root and F + end in fol and ("checks" not in events
                                                   or "check_results" in events):
            break
        time.sleep(2)
    events["end"] = watch.st.get_current_block()


def _run_checks(procs, results, root_env, follower_env, heads, miner_extra) -> dict:
    """--check on a validator and a miner against the running chain: READY
    and 0; then with no root hotkey (a deliberately broken v4 input): the
    reason and 1."""
    out = {}
    tmp = os.path.join(results, "check_state")
    os.makedirs(tmp, exist_ok=True)
    # A local chain has no release entries (validator_setup resolves them for
    # finney and testnet only): its operator names them, as here. The running
    # validators do not need them in mock, and setting them there would put a
    # consensus input in their digest the miners do not have.
    entries = {"FUGAL_TEE_MEASUREMENTS": "rehearsal-local-entry"}
    ok_v = procs.validator("v4f_check_val", "dr_val2",
                           {**follower_env, **entries,
                            "FUGAL_STATE_PATH": os.path.join(tmp, "state.json")},
                           check=True)
    bad_v = procs.validator("v4f_check_val_bad", "dr_val2",
                            {**follower_env, **entries, "FUGAL_ROOT_HOTKEY": "",
                             "FUGAL_STATE_PATH": os.path.join(tmp, "bad", "state.json")},
                            check=True)
    ok_m = procs.miner("v4f_check_m", "dr_m3", heads["dr_m3"], 8190, miner_extra("dr_m3"),
                       check=True)
    bad_m = procs.miner("v4f_check_m_bad", "dr_m3", heads["dr_m3"], 8191,
                        {**miner_extra("dr_m3"), "FUGAL_ROOT_HOTKEY": ""}, check=True)
    for name, r in (("validator", ok_v), ("validator_broken", bad_v), ("miner", ok_m),
                    ("miner_broken", bad_m)):
        lines = [ln for ln in (r.stdout or "").splitlines() if "READY" in ln]
        out[name] = (r.returncode, (lines[-1] if lines else r.stdout[-300:])
                     + f" [{r.seconds}s]")
    return out


# ── the checks ────────────────────────────────────────────────────────────────

def _v4_checks(report, S, results, watch, store, stub, F, bpe, fx, seed_menu, events, head_hash,
               wallets, netuid, subtensor, timings, procs, dr, base, root_env, follower_env,
               heads, ports, miner_extra, args):
    from fugal_subnet import menu_state as ms
    from fugal_subnet import reveal_sync
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block
    from fugal_subnet.v2_sync import nonzero_weights
    from scripts.rehearsal_v2 import comparable_reveal

    names = ("v4f_val1", "v4f_val2")
    logs = {n: epoch_log(results, n) for n in names}
    ran = {n: dr._epoch_protocols(os.path.join(results, f"{n}.log")) for n in names}
    root_hk = wallets["dr_val1"].hotkey.ss58_address
    last = max([k for n in names for k in logs[n]] or [F])
    print(f"    events: { {k: v for k, v in events.items() if k != 'check_results'} }", flush=True)

    def commit_of(b):
        return (watch.at.get(b) or {}).get(root_hk, ("", 0))

    # 1. before the block: protocol v3, v3 reveals agree, the three-field form.
    before = {n: {k: v for k, v in ran[n].items() if k < F} for n in names}
    v3_rev = [k for k in range(F - V3_EPOCHS_BEFORE, F) if reveal(results, "v4f_val1", k)]
    v3_same = all(comparable_reveal(reveal(results, "v4f_val1", k))
                  == comparable_reveal(reveal(results, "v4f_val2", k) or {}) for k in v3_rev)
    report.check(S, "before the block every epoch is protocol v3 on both validators",
                 all(before[n] and set(before[n].values()) == {3} for n in names),
                 f"{ {n: sorted(before[n].items()) for n in names} }")
    report.check(S, "v3 reveals before the block: pooled exploration and odds, root = follower",
                 bool(v3_rev) and v3_same and all(
                     (reveal(results, "v4f_val1", k) or {}).get("protocol_version") == 3
                     and reveal(results, "v4f_val1", k).get("exploration_pool")
                     for k in v3_rev), f"v3 reveals at {[k - F for k in v3_rev]} (F-relative)")
    pre = {b: commit_of(b)[0] for b in sorted(watch.at) if b <= F}
    pre_parsed = {b: reveal_sync.parse_commitment(c) for b, c in pre.items() if c}
    report.check(S, "the root's commitment keeps the three-field v3 form through the block",
                 bool(pre_parsed) and all(p is not None and p[2] is None
                                          for p in pre_parsed.values()),
                 f"at boundaries {[(b - F, c) for b, c in pre.items() if c]}")

    # 2. at the block.
    rF = {n: reveal(results, n, F) for n in names}
    seed_hash = ms.menu_hash(ms.seed_state(F))
    report.check(S, "protocol 4 from the block on every validator (and every miner's proof)",
                 all(ran[n].get(F) == 4 for n in names if F in ran[n])
                 and all(v == 4 for n in names for k, v in ran[n].items() if k >= F)
                 and F in ran["v4f_val2"],
                 f"{ {n: sorted((k - F, v) for k, v in ran[n].items() if k >= F) for n in names} }")
    menuF = {n: ((rF[n] or {}).get("menu") or {}).get("hash") for n in names}
    report.check(S, "the first v4 epoch runs on the pinned seed menu on every node",
                 all(h == seed_hash for h in menuF.values() if h) and any(menuF.values()),
                 f"seed {seed_hash[:16]}; reveals {[(n, (h or '-')[:16]) for n, h in menuF.items()]}")
    seat0 = {n: ((rF[n] or {}).get("champion") or {}) for n in names}
    report.check(S, "the seat and window start empty at the block",
                 any(rF.values()) and all(s.get("champion") is None for s in seat0.values() if s)
                 and all(((r or {}).get("v2_state") or {}).get("rules", {}).get("protocol") == 4
                         for r in rF.values() if r),
                 f"{ {n: s.get('event') for n, s in seat0.items()} }")
    # The commitment the menu of epoch B is read from: the root's for B -
    # MENU_LAG, on chain at B's boundary (one that lands later, as after the
    # root's outage, names no menu for anyone, by rule).
    four = [(b, c) for b in sorted(watch.at) for c in [commit_of(b)[0]]
            if (reveal_sync.parse_commitment(c) or (0, 0, None))[2] is not None]
    timely = [(b, c) for b, c in four
              if int(reveal_sync.parse_commitment(c)[0][1:]) + ms.MENU_LAG == b]
    if timely:
        b4, c4 = timely[0]
        ep, _sha, mh = reveal_sync.parse_commitment(c4)
        read = {n: ((reveal(results, n, b4) or {}).get("menu") or {}).get("hash") for n in names}
        said = {n: f"Menu for {epoch_id_for_block(b4)}: committed, {mh[:16]}" in log_text(results, n)
                for n in names if b4 in logs[n]}
        ran_on = {n: logs[n].get(b4, {}) for n in names}
        ok = (all(h == mh for h in read.values() if h) and any(read.values())
              and all(said.values()) and bool(said)
              and all(e.get("n_heads_valid") == len(MINERS) and not e.get("n_heads_invalid")
                      for e in ran_on.values() if e))
        report.check(S, "the root's v4 commitment (four fields) is read off the chain by the "
                        "follower and the miners",
                     ok, f"four-field commitments at boundaries F+{[b - F for b, _c in four]}, "
                         f"first read for a menu at F+{b4 - F}: {c4[:40]}... ({len(c4)} bytes), "
                         f"menu {mh[:16]}; reveals there "
                         f"{ {n: (h or '-')[:16] for n, h in read.items()} }; "
                         f"'committed' in the log {said}; miners' proofs on it verified "
                         f"{ {n: e.get('n_heads_valid') for n, e in ran_on.items()} }")
    else:
        report.check(S, "the root's v4 commitment (four fields) is read off the chain by the "
                        "follower and the miners", False,
                     f"four-field commitments {[(b - F, c[:24]) for b, c in four]}, none on chain "
                     "at the boundary its menu applies from")
    pruned = watch.pruned_read(max(0, events.get("end", 0) - 2 * bpe))
    used_archive = {n: log_text(results, n).count("REHEARSAL: archive node")
                    for n in (*names, *[f"v4f_{ck}" for ck in MINERS])}
    report.check(S, "pinned reads of pruned blocks: the localnet node refuses, the archive serves",
                 pruned != "ok" and not watch.read_errors
                 and used_archive.get("v4f_val2", 0) > 0,
                 f"localnet read of a block two epochs old: {pruned}; harness reads failed at "
                 f"{sorted(watch.read_errors)}; SDK switches to the archive node by process: "
                 f"{used_archive}")

    # 3. v4 epochs.
    v4_epochs = [k for k in range(F, last + 1)]
    miners_expected = len(MINERS)
    bad = [(n, k - F, e.get("n_heads_valid"), e.get("n_heads_invalid"),
            e.get("n_heads_unverifiable")) for n in names for k, e in sorted(logs[n].items())
           if k >= F and (e.get("n_heads_valid") != miners_expected or e.get("n_heads_invalid")
                          or e.get("n_heads_unverifiable"))]
    report.check(S, "every miner's v4 proof verifies on both validators, every epoch they ran",
                 not bad and all(k in logs["v4f_val2"] for k in [F]),
                 f"short epochs (validator, F+k, valid, invalid, unverifiable): {bad[:8]}")
    excused = set()
    for k in range(F + ROOT_DOWN[0], F + ROOT_DOWN[1] + 3):
        excused.add(k)                     # the root's outage and the two epochs after it
    for k in range(F + FOLLOWER_DOWN[0], F + FOLLOWER_DOWN[1] + 3):
        excused.add(k)                     # the follower's outage, re-anchor and replay
    compared, differ = [], []
    for k in v4_epochs:
        a, b = reveal(results, "v4f_val1", k), reveal(results, "v4f_val2", k)
        if not a or not b or k in excused:
            continue
        va = {x: v for x, v in (a.get("v2_state") or {}).items() if x not in ("sync",)}
        vb = {x: v for x, v in (b.get("v2_state") or {}).items() if x not in ("sync",)}
        same = (va == vb and a.get("menu") == b.get("menu") and a.get("memo_v4") == b.get("memo_v4")
                and nonzero_weights(logs["v4f_val1"].get(k, {}).get("weights"))
                == nonzero_weights(logs["v4f_val2"].get(k, {}).get("weights")))
        (compared if same else differ).append(k - F)
    report.check(S, "root and follower publish identical v4 state and weights",
                 bool(compared) and not differ,
                 f"identical at F+{compared}; differ at F+{differ}; outage windows excused")
    cr = {n: [(k - F, bool(e.get("set_weights_success")), bool(e.get("weights_confirmed_on_chain")))
              for k, e in sorted(logs[n].items()) if k >= F] for n in names}
    landed = {n: _landed_blocks(logs[n], F, bpe) for n in names}
    report.check(S, "weights land on chain every v4 epoch (commit-reveal, confirmed off chain)",
                 all(all(s and c for _k, s, c in cr[n]) for n in names)
                 and all(ok for n in names for _k, ok in landed[n]),
                 f"(F+k, set, confirmed): {cr}; commit inside its epoch: {landed}")
    objs = store.objects(BUCKET)
    menus = {k for k in objs if "/menus/" in k}
    snaps = {k for k in objs if "/published/" in k}
    reveals = {k for k in objs if k.endswith("/reveal.json")}
    named_menus = {((reveal(results, "v4f_val1", k) or {}).get("menu") or {}).get("source", {})
                   .get("hash") for k in v4_epochs} - {None}
    missing_menus = [h[:12] for h in named_menus if reveal_sync.menu_object(netuid, h) not in objs]
    from fugal_subnet import memo_sync
    def published(k):
        return f"{netuid}/{epoch_id_for_block(k)}/reveal.json" in objs

    # A reveal, and the snapshot and heads it names, go out one epoch after
    # it (the commit-reveal hold): the newest is not due yet.
    named_snaps = set()
    for k in v4_epochs:
        r = reveal(results, "v4f_val1", k) if published(k) else None
        nv = memo_sync.named_view(r) if r else None
        if nv is not None:
            named_snaps.add(nv[1])
    missing_snaps = [d[:12] for d in named_snaps if memo_sync.snapshot_object(netuid, d) not in objs]
    audits = [(k - F, (reveal(results, "v4f_val2", k) or {}).get("menu_audit"))
              for k in v4_epochs if (reveal(results, "v4f_val2", k) or {}).get("menu_audit")]
    bad_audits = [(k, a) for k, a in audits if a.get("differences")]
    report.check(S, "menus, published-set snapshots and reveals are in the store by hash; the "
                    "follower's audits are empty",
                 bool(menus) and not missing_menus and not missing_snaps and bool(reveals)
                 and not bad_audits,
                 f"{len(menus)} menus, {len(snaps)} snapshots ({len(named_snaps)} named, missing "
                 f"{missing_snaps}), {len(reveals)} reveals; menus missing {missing_menus}; "
                 f"{len(audits)} audits, {len(bad_audits)} with differences "
                 f"{[(k, a.get('why')) for k, a in bad_audits][:2]}")
    fold_rows = _fold_check(results, watch, store, fx, F, last, bpe, netuid, root_hk, names)
    report.check(S, "every node's published-set digest equals an independent fold of the "
                    "root's committed reveals", fold_rows[0], fold_rows[1])


    # 5. the follower's outage.
    back = F + FOLLOWER_DOWN[1] + 1
    resync = [(k - F, (reveal(results, "v4f_val2", k) or {}).get("v2_sync", {}).get("reanchored"))
              for k in range(back, last + 1)]
    agree_from = next((k for k in range(back, last + 1)
                       if all(_same(results, logs, j) for j in range(k, last + 1))), None)
    report.check(S, "a follower stopped two epochs re-anchors, replays and agrees with the root",
                 any(r for _k, r in resync) and agree_from is not None and agree_from <= back + 2,
                 f"back at F+{back - F}; re-anchors {[(k, r) for k, r in resync if r]}; agrees "
                 f"from F+{agree_from - F if agree_from else None} (in process: two epochs after "
                 "the miss)")

    # 6. the root's outage.
    alarms = {n: [ln for ln in log_text(results, n).splitlines() if "ALARM" in ln]
              for n in (*names, *[f"v4f_{ck}" for ck in MINERS])}
    down = range(F + ROOT_DOWN[0], F + ROOT_DOWN[1] + 1)
    fol_ok = all(logs["v4f_val2"].get(k, {}).get("n_heads_valid") == miners_expected for k in down)
    root_back = F + ROOT_DOWN[1] + 1
    agree_after = next((k for k in range(root_back, last + 1) if _same(results, logs, k)), None)
    report.check(S, "the root down two epochs: miners prove, the follower verifies, no ALARM",
                 fol_ok and not any(alarms.values()) and agree_after is not None,
                 f"follower valid in F+{ROOT_DOWN}: "
                 f"{[logs['v4f_val2'].get(k, {}).get('n_heads_valid') for k in down]}; alarms "
                 f"{ {n: a[:1] for n, a in alarms.items() if a} }; agree again from "
                 f"F+{agree_after - F if agree_after else None}")
    backlog = _backlog_fold(results, watch, F, root_hk, names)
    report.check(S, "the root's backlog is folded on every node at the same boundary",
                 backlog[0], backlog[1])

    # 7. a miner restart.
    b_after = [(k - F, logs["v4f_val1"].get(k, {}).get("n_heads_valid"))
               for k in range(F + B_RESTART, last + 1)]
    kept = os.path.exists(os.path.join(results, "td_dr_m2", "menu.json"))
    report.check(S, "a restarted miner proves again from its kept state",
                 "b_restart" in events and kept
                 and all(v == miners_expected for _k, v in b_after if v is not None),
                 f"restarted at block {events.get('b_restart')}; kept menu {kept}; root valid "
                 f"after: {b_after}")

    # 8. --check.
    cres = events.get("check_results") or {}
    ok = (cres.get("validator", (1,))[0] == 0 and "READY" in cres.get("validator", (1, ""))[1]
          and cres.get("miner", (1,))[0] == 0 and "READY" in cres.get("miner", (1, ""))[1]
          and cres.get("validator_broken", (0,))[0] == 1
          and cres.get("miner_broken", (0,))[0] == 1
          and "v4 MISSING" in cres.get("validator_broken", (0, ""))[1]
          and "root hotkey" in cres.get("validator_broken", (0, ""))[1]
          and "root" in cres.get("miner_broken", (0, ""))[1])
    report.check(S, "--check prints READY and exits 0; a broken v4 input prints why and exits 1",
                 ok, "; ".join(f"{k}: rc={v[0]} {v[1][:150]}" for k, v in cres.items())
                 or "never run")

    # 9. timing and the commitments pallet.
    durs = {n: [round(e.get("duration_s", 0), 1) for k, e in sorted(logs[n].items()) if k >= F]
            for n in names}
    epoch_s = float(args.v4_epoch_seconds)
    after = {n: _after_collection(log_text(results, n), F) for n in names}
    report.check(S, "each validator's v4 epoch step fits in the epoch",
                 all(durs[n] and max(durs[n]) < epoch_s for n in names),
                 f"seconds per v4 epoch, boundary to logged {durs}; of it after the collection "
                 f"block (query, verify, score, publish, weights) {after}; epoch {epoch_s:.0f}s; "
                 f"setup "
                 f"{timings.get('setup_s', 0):.0f}s, scenario {timings.get('total_s', 0):.0f}s")
    rows = commitment_rows(watch, F, bpe, root_hk, wallets, subtensor, netuid, results)
    report.check(S, "every commitment lands inside its epoch, none refused by the pallet",
                 rows[0], rows[1])
    logs_row(report, S, results)
    print(f"    stub: {stub.calls} model calls, {stub.listing_reads} listing reads; store: "
          f"{len(store.gets)} GETs", flush=True)


def _after_collection(text: str, F: int) -> list:
    """Seconds from 'Querying ... miners' to the weights confirmed on chain
    (else 'Epoch eN logged') per v4 epoch: the step's own work, without the
    root's wait for its join read before it writes its reveal."""
    import datetime
    import re

    out, start, stop = [], None, None
    for ln in text.splitlines():
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)", ln)
        if not m:
            continue
        t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
        if "Querying" in ln and "miners for proofs" in ln:
            start = t
        if start is not None and "Weights confirmed on chain" in ln:
            stop = t
        g = re.search(r"Epoch e(\d+) logged", ln)
        if g and start is not None:
            if int(g.group(1)) >= F:
                out.append(round(((stop or t) - start).total_seconds(), 1))
            start = stop = None
    return out


def _tracebacks(text: str) -> list[str]:
    """Each traceback in a log with the line before it (the message of the
    logger.exception that handled it, when it was logged as handled)."""
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        if ln.startswith("Traceback"):
            j = i + 1
            while j < len(lines) and (lines[j].startswith(" ") or not lines[j][:4].isdigit()):
                j += 1
            out.append("\n".join(lines[max(0, i - 1):j]))
    return out


def _first_traceback(text: str) -> str:
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "Traceback" in ln:
            tail = [x for x in lines[i:i + 60] if x and not x.startswith(" ")]
            return " | ".join(tail[:3])[:300]
    return ""


def _same(results, logs, k) -> bool:
    from fugal_subnet.v2_sync import nonzero_weights

    a, b = reveal(results, "v4f_val1", k), reveal(results, "v4f_val2", k)
    if not a or not b:
        return False
    va = {x: v for x, v in (a.get("v2_state") or {}).items() if x != "sync"}
    vb = {x: v for x, v in (b.get("v2_state") or {}).items() if x != "sync"}
    return (va == vb and a.get("menu") == b.get("menu")
            and nonzero_weights(logs["v4f_val1"].get(k, {}).get("weights"))
            == nonzero_weights(logs["v4f_val2"].get(k, {}).get("weights")))


def _landed_blocks(log: dict, F: int, bpe: int) -> list:
    """(F+k, whether the weights commit was recorded inside epoch k), from
    the validator's own read-back ('commit recorded at block N')."""
    import re

    out = []
    for k, e in sorted(log.items()):
        if k < F:
            continue
        m = re.search(r"commit recorded at block (\d+)", str(e.get("weights_confirm_detail", "")))
        out.append((k - F, bool(m) and k * bpe <= int(m.group(1)) < (k + 1) * bpe))
    return out


def _fold_check(results, watch, store, fx, F, last, bpe, netuid, root_hk, names):
    """memo_v4's fold from the fixture seed through the root's commitments as
    this harness read them pinned at each boundary, and the reveal bytes the
    store serves for each (checked against the committed sha256): every v4
    reveal's published digest, and every node's held set, must be it."""
    from fugal_subnet import memo_sync, memo_v4, reveal_sync

    objs = store.objects(BUCKET)
    pub = memo_v4.seed_published(fx["pool"], fx["seed"])
    named = pub.through
    states = {}
    problems = []
    for b in range(pub.through + memo_v4.READ_LAG + 1, last + 1):
        if b not in watch.at:
            problems.append(f"no harness read at F{b - F:+d}")
            break
        got = reveal_sync.parse_commitment((watch.at[b].get(root_hk) or ("", 0))[0])
        ids: set = set()
        if got is not None and memo_v4.epoch_index(got[0]) > named:
            raw = objs.get(f"{netuid}/{got[0]}/reveal.json")
            if raw is None or hashlib.sha256(raw).hexdigest() != got[1]:
                problems.append(f"reveal {got[0]} named at F{b - F:+d} is not in the store")
                break
            named = memo_v4.epoch_index(got[0])
            ids = memo_v4.published_ids(json.loads(raw))
        pub = memo_v4.fold(pub, b - memo_v4.READ_LAG, ids)
        states[b] = pub
    checked, wrong = 0, []
    for k in range(F, last + 1):
        if k not in states:
            continue
        want = memo_v4.fold(states[k].view(k - memo_v4.READ_LAG), k - 1, ()).view(
            k - memo_v4.READ_LAG).digest
        for n in names:
            r = reveal(results, n, k)
            if r is None:
                continue
            checked += 1
            if (r.get("memo_v4") or {}).get("published_digest") != want:
                wrong.append((n, k - F))
    held = {}
    for n in names:
        path = os.path.join(results, n, "state.json" + memo_sync.STATE_SUFFIX)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            at = doc["at"]
            h = memo_v4.published_from_doc(doc["set"])
            held[n] = (at - F, at in states
                       and h.digest == states[at].view(at - memo_v4.READ_LAG).digest)
        except (OSError, ValueError, KeyError) as e:
            held[n] = (None, f"{type(e).__name__}")
    ok = not problems and checked > 0 and not wrong and all(v[1] is True for v in held.values())
    return ok, (f"{checked} reveal digests checked, wrong {wrong[:6]}; held sets (at F+k, equal): "
                f"{held}; {problems[:3]}")


def _backlog_fold(results, watch, F, root_hk, names):
    """The root's reveals of the epochs before its outage, published when it
    came back: the boundary whose commitment first names each, and that
    every node folded it there (its held set or reveals name the same
    published digest from that boundary; checked by _fold_check)."""
    from fugal_subnet import reveal_sync

    back_block = None
    seen = []
    for b in sorted(watch.at):
        got = reveal_sync.parse_commitment((watch.at[b].get(root_hk) or ("", 0))[0])
        if got is not None and (not seen or seen[-1][1] != got[0]):
            seen.append((b - F, got[0]))
    down_first = F + ROOT_DOWN[0]
    after = [(b, e) for b, e in seen if b >= ROOT_DOWN[1] + 1]
    if after:
        back_block = after[0]
    named_down = [e for _b, e in seen if int(e[1:]) == down_first - 1]
    ok = back_block is not None and bool(named_down)
    return ok, (f"root commitments by boundary (F+k, reveal named): {seen}; the reveal of "
                f"F+{ROOT_DOWN[0] - 1} (published by the root on its return) first named at "
                f"F+{back_block[0] if back_block else None}; every node's fold through it is "
                "checked by the published-set row")


PALLET_ERRORS = ("SpaceLimitExceeded", "TooManyFieldsInCommitmentInfo", "AccountNotAllowedCommit",
                 "CommitmentSetRateLimitExceeded", "TooManyUnrevealedCommits", "Custom error",
                 "RateLimit")


def commitment_rows(watch, F, bpe, root_hk, wallets, subtensor, netuid, results):
    """Each account's commitments as the harness saw them at the boundaries:
    the block each landed at (inside the epoch it belongs to), its size,
    and the bytes per epoch; what the pallet holds (MaxSpace, UsedSpaceOf)."""
    from fugal_subnet import reveal_sync

    per: dict[str, list] = {}
    for b in sorted(watch.at):
        for hk, (data, block) in (watch.at[b] or {}).items():
            rows = per.setdefault(hk, [])
            if not rows or rows[-1][1] != block:
                rows.append((data, block))
    names = {w.hotkey.ss58_address: n for n, w in wallets.items()}
    late = []
    for data, block in per.get(root_hk, []):
        got = reveal_sync.parse_commitment(data)
        if got is None:
            continue
        e = int(got[0][1:])
        # The reveal of E is published and committed in E + 1 (one epoch's
        # hold), or later when the root was down; never before E + 1.
        if block < (e + 1) * bpe:
            late.append((got[0], block))
    sizes = {names.get(hk, hk[:8]): sorted({len(d.encode()) for d, _b in rows})
             for hk, rows in per.items()}
    counts = {names.get(hk, hk[:8]): len(rows) for hk, rows in per.items()}
    per_epoch: dict[str, dict[int, int]] = {}
    for hk, rows in per.items():
        for d, block in rows:
            if block >= F * bpe:
                e = per_epoch.setdefault(names.get(hk, hk[:8]), {})
                e[block // bpe - F] = e.get(block // bpe - F, 0) + len(d.encode())
    most = {n: max(v.values()) for n, v in per_epoch.items() if v}
    try:
        space = subtensor.substrate.query("Commitments", "MaxSpace").value
    except Exception as e:  # noqa: BLE001
        space = f"? ({type(e).__name__})"
    try:
        hp = subtensor.get_subnet_hyperparameters(netuid)
        tempo = getattr(hp, "tempo", "?")
    except Exception:  # noqa: BLE001
        tempo = "?"
    refused = []
    for name in sorted(os.listdir(results)):
        if name.endswith(".log"):
            for ln in log_text(results, name[:-4]).splitlines():
                if any(x in ln for x in PALLET_ERRORS):
                    refused.append(f"{name[:-4]}: {ln.strip()[:160]}")
    ok = bool(per.get(root_hk)) and not late and not refused
    return ok, (f"pallet or rate-limit errors in the logs: {refused[:3]}; "
                f"commitments seen (account: count, sizes in bytes): "
                f"{ {n: (counts[n], sizes[n]) for n in counts} }; most bytes committed in one v4 "
                f"epoch by account {most}; root commitments before their "
                f"epoch's end: {late}; pallet MaxSpace {space} bytes per account per tempo "
                f"(tempo {tempo} blocks; the scenario's epoch is {bpe})")


def logs_row(report, S, results, expected=None) -> None:
    """The in-process recorder's rule on the real chain (tests/v4_world.Chain
    future_reads): no neuron read a block the chain had not reached (the
    shim notes every block-hash read that came back empty for a block above
    the chain's head), and no ALARM line or unhandled exception in any
    neuron's log."""
    names = sorted(n[:-4] for n in os.listdir(results)
                   if n.endswith(".log") and n.startswith(("v4f_", "vs_")))
    future = {n: [ln for ln in log_text(results, n).splitlines() if "REHEARSAL: FUTURE READ" in ln]
              for n in names}
    alarms = {n: [ln for ln in log_text(results, n).splitlines() if "ALARM" in ln] for n in names}
    tracebacks = {}
    for n in names:
        got = _tracebacks(log_text(results, n))
        bad = [b for b in got if not (expected and expected(b))]
        if bad:
            tracebacks[n] = f"{len(bad)}: " + " | ".join(x for x in bad[0].splitlines()
                                                         if x and not x.startswith(" "))[:300]
    report.check(S, "no read of a block the chain had not reached, no ALARM, no unhandled "
                    "exception, in any neuron's log",
                 not any(future.values()) and not any(alarms.values()) and not tracebacks,
                 f"future reads { {n: f[:1] for n, f in future.items() if f} }; alarms "
                 f"{ {n: a[:1] for n, a in alarms.items() if a} }; exceptions {tracebacks}"[:700]
                 + f" ({len(names)} logs read)")
