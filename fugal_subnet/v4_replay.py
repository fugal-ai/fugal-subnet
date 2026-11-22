"""Protocol v4: a re-anchored validator replays its own epochs since the anchor.

WHY. Verify-and-resync (fugal_subnet/v2_sync.py) adopts the reference's
state at an epoch k. The reference's reveals lag the present by the reveal
hold, so the epochs k+1 .. now-1 are ones this validator ran on its own
state. Under v4 a kept epoch is not a fixed fact about that epoch: its label
pool depends on the window it was pooled on (the plausibility check's
reference R is fitted on the window's earlier pairs, `odds_score.
expected_odds`), and so do its late odds and its seat step. An epoch this
validator ran on a window that differed from the reference's is therefore
different from the reference's epoch, and keeping it carries the difference
past every re-anchor.

So under v4 a re-anchor does not keep those epochs: it REPLAYS them, in
epoch order, through `v2_epoch.run_v2_epoch` itself, on the corrected state,
from the inputs this validator recorded when it ran each one. The state that
results is the function of (the reference's state at k, the inputs of epochs
k+1 .. now-1) that a validator which ran every epoch holds.

WHAT IS RECORDED (`record`), and why the reveal is not enough. Each v4
epoch's inputs to `run_v2_epoch`, once the epoch is committed
(`Store.keep`), beside the state:

- the slice (question id and question set; the v4 slice is priced at the
  menu's pick-cost table, so the prompt is not read), the menu's entries,
  prices, pick-cost table and categories;
- every verified proof's scored results (question, routed entry, correct,
  failed, completion tokens) and whether it answered the scored questions;
  the uid's head hash, coldkey and hotkey;
- the boundary snapshot of commitments (the copy rule's input);
- every verified proof's exploration answers, each with its coldkey and
  whether it was private (re-pooling needs every answer, including those of
  operators the original pool left out; the reveal publishes only the pool);
- the bytes of every verified head (by sha256, once however many epochs it
  served): odds on a re-pooled epoch, and late odds on the reference's
  earlier pairs, are computed from them;
- the epoch nonce, the exploration weight tables, the epoch's menu state,
  the pool's constants, the real-picks input (outage flag, refused hotkeys),
  the seniority floor and salt, the burn uid and shares, and the rules tag.

What is not recorded, because it is the same for every epoch of a release:
the pinned embeddings (`hidden_of`) and the pool's question sets
(`benchmark_of`), which the caller passes. The memorisation check's memo
for an epoch is the caller's too (`memo_for(epoch, nonce)`, handed the
epoch's recorded nonce): it is built from the published set as of that
epoch, which the chain decides (fugal_subnet/memo_sync.py), never from this
validator's own reveals. What is not used: v1-v3's evidence records
and seat parameters (a v4 epoch reads neither).

REPLAYED, NOT REPUBLISHED. A replay decides only this validator's state. The
replayed epochs' weights and reveals are dropped: those epochs are past, and
the reveals this validator published for them stand as published (its
commitments to them are on chain). The reveal of the epoch at which the state
was replaced says so (`v2_sync.reanchored`, written by the validator).

BOUNDED. A re-anchor replays at most `max_span` epochs (the validator
passes the reveal hold, neurons/validator._v4_replay_span: the steady
state's span; records are kept for the comparison window plus two). It
costs one `run_v2_epoch` per replayed epoch and holds one epoch's inputs at
a time, each head loaded once. A span with an epoch whose record or heads
are missing or unreadable, or a longer span, is not replayed: `Missing`
names the epochs, and the caller keeps its state and waits for a reference
epoch past them (never a silent keep of an epoch it cannot replay).

Deterministic: records are JSON (floats in their shortest round-trip form),
written with every map and list in sorted order; epochs are replayed in
index order; and `run_v2_epoch` is order-independent in its inputs (a record
rewritten in another order replays to the same bytes,
tests/test_v4_resync_replay.py).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FORMAT = "fugal-v4-replay/2"
_SUFFIX = ".json.gz"
_SHA = re.compile(r"[0-9a-f]{64}")


class Missing(ValueError):
    """A replay that cannot run: the epochs whose inputs are missing or
    unreadable, or a span longer than the bound."""

    def __init__(self, epochs: Iterable[int], why: str):
        self.epochs = tuple(sorted(int(e) for e in epochs))
        super().__init__(why)


# --- the record ---------------------------------------------------------------------------


def _uid_map(d: Mapping, value=lambda v: v) -> dict:
    return {str(int(uid)): value(d[uid]) for uid in sorted(d, key=int)}


def _results(proof, results=None) -> list:
    rows = proof.scored_results if results is None else results
    # A holder-side call not started carries a sixth field, true.
    return [[str(r.question_id), str(r.routed_model), bool(r.correct), bool(r.failed),
             int(r.completion_tokens)] + ([True] if getattr(r, "not_started", None) else [])
            for r in rows]


def _challenge(proof):
    ch = getattr(proof, "challenge", None)
    return None if ch is None else [int(ch.size), ch.holder]


def record(kwargs: Mapping, head_bytes: Mapping[int, bytes]) -> tuple[dict, dict[str, bytes]]:
    """(the record of one v4 epoch, {sha256: head bytes} it names) from the
    keyword arguments `run_v2_epoch` was called with and the verified heads'
    bytes. Raises ValueError for an epoch that is not a v4 one."""
    from fugal_subnet import menu_state as ms

    oi, pi = kwargs.get("odds_input"), kwargs.get("picks")
    if oi is None or oi.menu_state is None:
        raise ValueError("only a protocol v4 epoch (an odds input with a menu state) is recorded")
    if kwargs.get("pick_spend") is None:
        raise ValueError("a v4 epoch's slice is priced at its pick-cost table")
    menu = ms.canonical_json(oi.menu_state).decode("ascii")
    if pi is not None and ms.menu_hash(pi.state) != ms.menu_hash(oi.menu_state):
        raise ValueError("the real-picks input and the odds input name different menus")
    heads, blobs = {}, {}
    for uid in sorted(oi.heads):
        data = bytes(head_bytes[uid])
        sha = hashlib.sha256(data).hexdigest()
        heads[str(int(uid))] = sha
        blobs[sha] = data
    proofs = kwargs["verified_proofs"]
    return {
        "format": FORMAT,
        "epoch_index": int(kwargs["epoch_index"]),
        "epoch_id": str(kwargs["epoch_id"]),
        "boundary_block": int(kwargs["boundary_block"]),
        "blocks_per_epoch": int(kwargs["blocks_per_epoch"]),
        "questions": [[str(q["question_id"]), str(q.get("benchmark", ""))]
                      for q in kwargs["questions"]],
        "pool_models": [str(m) for m in kwargs["pool_models"]],
        "prices": {str(m): [float(a), float(b)] for m, (a, b) in sorted(kwargs["prices"].items())},
        "bench_category": (None if kwargs.get("bench_category") is None else
                           {str(k): str(v) for k, v in sorted(kwargs["bench_category"].items())}),
        "pick_spend": {str(e): [float(v), {str(c): float(x) for c, x in sorted(by.items())}]
                       for e, (v, by) in sorted(kwargs["pick_spend"].items())},
        "proofs": _uid_map(proofs, lambda p: {
            "scored": getattr(p, "scored", None), "results": _results(p),
            "challenge": _challenge(p),
            "held": _results(p, list(getattr(p, "holder_results", ())))}),
        "head_hashes": _uid_map(kwargs["head_hashes"], str),
        "coldkey_of": _uid_map(kwargs["coldkey_of"], str),
        "hotkey_of": _uid_map(kwargs["hotkey_of"], str),
        "commitments": sorted([str(c.hotkey), str(c.coldkey), str(c.head_hash), int(c.block)]
                              for c in kwargs["commitments"]),
        "burn_uid": int(kwargs["burn_uid"]),
        "shares": [float(s) for s in kwargs["shares"]],
        "seniority_floor": (None if kwargs.get("seniority_floor") is None
                            else int(kwargs["seniority_floor"])),
        "seniority_salt": str(kwargs.get("seniority_salt", "")),
        "rules": None if kwargs.get("rules") is None else dict(kwargs["rules"]),
        "menu_state": menu,
        "odds": {
            "nonce": bytes(oi.nonce).hex(),
            "answers": sorted([str(a.question_id), str(a.model), str(a.coldkey), bool(a.correct),
                               bool(a.failed), bool(a.private), int(a.prompt_tokens),
                               int(a.completion_tokens), int(a.latency_ms)] for a in oi.answers),
            "heads": heads,
            "flagged": sorted(str(c) for c in oi.flagged),
            "coldkey_cap": int(oi.coldkey_cap),
            "implausible_z": int(oi.implausible_z),
            "implausible_min_samples": int(oi.implausible_min_samples),
            "implausible_min_trials": int(oi.implausible_min_trials),
            "ref_prior": int(oi.ref_prior),
            "single_model_min_pairs": int(oi.single_model_min_pairs),
            "category_targets": (None if oi.category_targets is None else
                                 {str(k): float(v) for k, v in sorted(oi.category_targets.items())}),
            "explore_weights": (None if oi.explore_weights is None else
                                {str(k): int(v) for k, v in sorted(oi.explore_weights.items())}),
            "private_weights": (None if oi.private_weights is None else
                                {str(k): int(v) for k, v in sorted(oi.private_weights.items())}),
        },
        "picks": (None if pi is None else
                  {"outage": bool(pi.outage),
                   "refused": {str(hk): [int(f), int(n)] for hk, (f, n) in sorted(pi.refused.items())},
                   "challenge": [[str(q), str(c)] for q, c in pi.challenge],
                   "heads": sorted(_held(pi, blobs))}),
    }, blobs


def _held(pi, blobs: dict) -> list[str]:
    """The heads a challenge check read beyond the verified ones (the
    holder's), kept with the record's other heads."""
    out = []
    for sha, data in sorted(pi.heads.items()):
        data = bytes(data)
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("a challenge head's bytes are not its hash")
        blobs[sha] = data
        out.append(sha)
    return out


@dataclass(frozen=True)
class _Result:
    """The fields of a proof's scored result that a v4 epoch reads."""
    question_id: str
    routed_model: str
    correct: bool
    failed: bool
    completion_tokens: int
    not_started: bool | None = None


@dataclass(frozen=True)
class _Challenge:
    size: int
    holder: str | None


@dataclass(frozen=True)
class _Proof:
    """The fields of a verified proof that a v4 epoch reads."""
    scored: bool | None
    scored_results: tuple
    challenge: _Challenge | None = None
    holder_results: tuple = ()


def inputs(doc: Mapping, *, hidden_of, benchmark_of: Mapping[str, str],
           read_head: Callable[[str], bytes], loaded: dict | None = None) -> dict:
    """The keyword arguments of `run_v2_epoch` for a recorded epoch, but its
    state (`window`, `seat`, `copy_state`, `state_rules`) and its `memo`.
    `read_head(sha256)` returns a recorded head's bytes; `loaded` ({sha256: (W,
    b, model ids)}, filled here) keeps heads loaded across a replay's
    epochs. Raises ValueError for a record that does not read back."""
    from fugal_subnet import copy_rule
    from fugal_subnet import menu_state as ms
    from fugal_subnet.explore_pool import ExploreAnswer
    from fugal_subnet.head_eval import load_head_v4
    from fugal_subnet.v2_epoch import OddsInput, PicksInput

    loaded = {} if loaded is None else loaded
    try:
        if doc.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} record")
        state = ms.from_json(str(doc["menu_state"]).encode("ascii"))
        o = doc["odds"]
        heads = {}
        for uid, sha in sorted(o["heads"].items(), key=lambda kv: int(kv[0])):
            if sha not in loaded:
                h = load_head_v4(read_head(str(sha)))
                loaded[sha] = (h.W, h.b, list(h.models))
            heads[int(uid)] = loaded[sha]
        odds_input = OddsInput(
            nonce=bytes.fromhex(o["nonce"]),
            answers=[ExploreAnswer(question_id=a[0], model=a[1], coldkey=a[2], correct=bool(a[3]),
                                   failed=bool(a[4]), private=bool(a[5]), prompt_tokens=int(a[6]),
                                   completion_tokens=int(a[7]), latency_ms=int(a[8]))
                     for a in o["answers"]],
            benchmark_of=benchmark_of, heads=heads, hidden_of=hidden_of,
            flagged=frozenset(o["flagged"]), coldkey_cap=int(o["coldkey_cap"]),
            implausible_z=int(o["implausible_z"]),
            implausible_min_samples=int(o["implausible_min_samples"]),
            implausible_min_trials=int(o["implausible_min_trials"]), ref_prior=int(o["ref_prior"]),
            single_model_min_pairs=int(o["single_model_min_pairs"]),
            category_targets=o["category_targets"], explore_weights=o["explore_weights"],
            private_weights=o["private_weights"], menu_state=state)
        picks = doc["picks"]
        return {
            "epoch_index": int(doc["epoch_index"]), "epoch_id": str(doc["epoch_id"]),
            "boundary_block": int(doc["boundary_block"]),
            "blocks_per_epoch": int(doc["blocks_per_epoch"]),
            "questions": [{"question_id": q, "benchmark": b} for q, b in doc["questions"]],
            "pool_models": list(doc["pool_models"]),
            "prices": {m: (float(a), float(b)) for m, (a, b) in doc["prices"].items()},
            "verified_proofs": {
                int(uid): _Proof(p["scored"], tuple(_Result(*r) for r in p["results"]),
                                 None if p["challenge"] is None else _Challenge(*p["challenge"]),
                                 tuple(_Result(*r) for r in p["held"]))
                for uid, p in doc["proofs"].items()},
            "head_hashes": {int(u): str(h) for u, h in doc["head_hashes"].items()},
            "coldkey_of": {int(u): str(c) for u, c in doc["coldkey_of"].items()},
            "hotkey_of": {int(u): str(h) for u, h in doc["hotkey_of"].items()},
            # v1-v3's evidence records and seat parameters: a v4 epoch reads neither.
            "records": {}, "params": None, "burn_in_questions": 0,
            "commitments": [copy_rule.Commitment(str(a), str(b), str(c), int(d))
                            for a, b, c, d in doc["commitments"]],
            "burn_uid": int(doc["burn_uid"]), "shares": tuple(float(s) for s in doc["shares"]),
            "bench_category": doc["bench_category"],
            "seniority_floor": doc["seniority_floor"], "seniority_salt": str(doc["seniority_salt"]),
            "odds_input": odds_input, "rules": doc["rules"],
            "pick_spend": {e: (float(v), {c: float(x) for c, x in by.items()})
                           for e, (v, by) in doc["pick_spend"].items()},
            "picks": (None if picks is None else
                      PicksInput(state=state, outage=bool(picks["outage"]),
                                 refused={hk: (int(f), int(n))
                                          for hk, (f, n) in picks["refused"].items()},
                                 challenge=tuple((str(q), str(c)) for q, c in picks["challenge"]),
                                 heads={str(sha): read_head(str(sha))
                                        for sha in picks["heads"]})),
        }
    except (KeyError, TypeError, IndexError, AttributeError) as e:
        raise ValueError(f"replay record is malformed ({type(e).__name__}: {e})") from None


# --- the store beside the state -----------------------------------------------------------


class Store:
    """The records of this validator's recent v4 epochs, in `directory`
    (beside its state): `eNNNNNNNN.json.gz` per epoch, a `.heads` list beside
    each, and every head's bytes once under `heads/<sha256>`."""

    def __init__(self, directory: str | os.PathLike):
        self.directory = str(directory)

    def _path(self, epoch: int) -> str:
        from fugal_subnet.benchmarks.slicer import epoch_id_for_block

        # The epoch id's one formatter (slicer.epoch_id_for_block).
        return os.path.join(self.directory, epoch_id_for_block(int(epoch)) + _SUFFIX)

    def _write(self, path: str, data: bytes) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def keep(self, doc: Mapping, blobs: Mapping[str, bytes], *, keep_epochs: int) -> None:
        """Write one epoch's record (and any head not held yet), then drop
        the records older than `keep_epochs` epochs and the heads no kept
        record names."""
        os.makedirs(os.path.join(self.directory, "heads"), exist_ok=True)
        for sha, data in sorted(blobs.items()):
            path = os.path.join(self.directory, "heads", sha)
            if not os.path.exists(path):
                self._write(path, bytes(data))
        epoch = int(doc["epoch_index"])
        raw = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self._write(self._path(epoch) + ".heads", "\n".join(sorted(blobs)).encode())
        self._write(self._path(epoch), gzip.compress(raw, mtime=0))
        self.prune(oldest=epoch - int(keep_epochs) + 1)

    def epochs(self) -> list[int]:
        from fugal_subnet.v2_rebuild import _epoch_index

        try:
            names = os.listdir(self.directory)
        except OSError:
            return []
        out = []
        for name in names:
            if name.endswith(_SUFFIX):
                try:
                    out.append(_epoch_index(name[:-len(_SUFFIX)]))
                except ValueError:
                    continue
        return sorted(out)

    def prune(self, *, oldest: int) -> None:
        for epoch in self.epochs():
            if epoch < oldest:
                for path in (self._path(epoch), self._path(epoch) + ".heads"):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        named: set[str] = set()
        for epoch in self.epochs():
            try:
                with open(self._path(epoch) + ".heads", encoding="ascii") as f:
                    named |= {line for line in f.read().split() if line}
            except OSError:
                continue
        try:
            held = os.listdir(os.path.join(self.directory, "heads"))
        except OSError:
            return
        for sha in held:
            if _SHA.fullmatch(sha) and sha not in named:
                try:
                    os.remove(os.path.join(self.directory, "heads", sha))
                except OSError:
                    pass

    def has(self, epoch: int) -> bool:
        return os.path.exists(self._path(epoch))

    def load(self, epoch: int) -> dict | None:
        """The record of `epoch`, or None when there is none. Raises
        ValueError for one that does not read back as that epoch's."""
        path = self._path(epoch)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                doc = json.loads(gzip.decompress(f.read()))
        except (OSError, ValueError, EOFError) as e:
            raise ValueError(f"{path}: unreadable ({e})") from None
        if not isinstance(doc, dict) or doc.get("epoch_index") != int(epoch):
            raise ValueError(f"{path}: not the record of epoch {epoch}")
        return doc

    def head_bytes(self, sha: str) -> bytes:
        if not _SHA.fullmatch(str(sha)):
            raise ValueError("a head is named by its sha256")
        path = os.path.join(self.directory, "heads", sha)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            raise ValueError(f"head {sha[:12]} is not held ({e})") from None
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"head {sha[:12]}: its bytes do not match its hash")
        return data


# --- the replay ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Replayed:
    """The state after replaying `epochs` on the reference's state at an
    anchor: what the next epoch starts from."""
    window: object
    seat: object
    copy_state: object
    rules: dict | None
    epochs: tuple[int, ...]


def _holder_heads(kw: dict, doc: Mapping, seat, epoch: int, store: Store,
                  holder_bytes: Callable[[str], bytes | None] | None) -> dict:
    """`kw` (an epoch's run_v2_epoch arguments) with the bytes of the holder
    its challenges pair with and of the holder going into it, when the
    replayed seat names one that is not a verified head of the epoch and not
    in the record: from this node's head store, then `holder_bytes` (the
    root's head publication), each checked by its hash, as the live epoch
    seeks them (neurons/validator._v4_holder_heads). The seat replayed is
    the reference's, so its holder may be one this node's record of the
    epoch never read."""
    import dataclasses

    from fugal_subnet import seat_v4

    pi = kw.get("picks")
    if pi is None or not isinstance(seat, seat_v4.SeatStateV4):
        return kw
    have = set(str(s) for s in doc["odds"]["heads"].values()) | set(pi.heads)
    want = sorted({h for h in (seat_v4.challenge_holder(seat, epoch), seat.champion)
                   if h is not None} - have)
    extra = {}
    for h in want:
        try:
            data = store.head_bytes(h)
        except ValueError:
            data = None
        if data is None and holder_bytes is not None:
            try:
                got = holder_bytes(h)
            except Exception:  # noqa: BLE001 - not to be had: the epoch says so below
                got = None
            if got is not None and hashlib.sha256(bytes(got)).hexdigest() == h:
                data = bytes(got)
        if data is not None:
            extra[h] = data
    if not extra:
        return kw
    return dict(kw, picks=dataclasses.replace(pi, heads={**dict(pi.heads), **extra}))


def replay(store: Store, *, window, seat, copy_state, rules, epochs: Iterable[int],
           hidden_of, benchmark_of: Mapping[str, str], memo_for=None,
           max_span: int, holder_bytes: Callable[[str], bytes | None] | None = None) -> Replayed:
    """Run each of `epochs`, in order, through `v2_epoch.run_v2_epoch` from
    its record, starting from (`window`, `seat`, `copy_state`, `rules`): the
    reference's state at the anchor. More than `max_span` epochs, an epoch
    with no record, or one whose record or heads do not read back, raises
    Missing naming the epochs, and nothing is replayed: the state is only
    returned once every epoch has run (one epoch's inputs are held at a
    time, and each head is loaded once). `memo_for(epoch, nonce)` gives an
    epoch's memorisation memo from its recorded nonce (None: none); it may
    raise Missing for an epoch it cannot give the memo of. The holder's
    head, when the replayed seat names one the record does not hold, comes
    from this node's head store, then `holder_bytes` (the root's head
    publication), checked by hash (`_holder_heads`); without it the epoch
    raises Missing. The epochs' weights and reveals are not kept.

    The take decisions run as on a validator with a whole window
    (`allow_take`): the state replayed from is the reference's, rebuilt."""
    from fugal_subnet import v2_epoch
    from fugal_subnet.benchmarks.slicer import epoch_id_for_block

    epochs = sorted(int(e) for e in epochs)
    if len(epochs) > int(max_span):
        raise Missing(epochs, f"{len(epochs)} epochs to replay, more than {max_span}")
    absent = [j for j in epochs if not store.has(j)]
    if absent:
        raise Missing(absent, "no record for " + ", ".join(epoch_id_for_block(j) for j in absent))
    loaded: dict = {}
    for j in epochs:
        try:
            doc = store.load(j)
            if doc is None:
                raise ValueError("no record")
            kw = inputs(doc, hidden_of=hidden_of, benchmark_of=benchmark_of, read_head=store.head_bytes,
                        loaded=loaded)
        except ValueError as e:
            raise Missing([j], f"{epoch_id_for_block(j)}: {e}") from None
        kw = _holder_heads(kw, doc, seat, j, store, holder_bytes)
        try:
            result = v2_epoch.run_v2_epoch(
                **kw, window=window, seat=seat, copy_state=copy_state, state_rules=rules,
                allow_take=True,
                memo=None if memo_for is None else memo_for(j, kw["odds_input"].nonce))
        except v2_epoch.HolderHeadMissing as e:
            raise Missing([j], f"{epoch_id_for_block(j)}: {e}") from None
        window, seat, copy_state = result.window, result.decision.state, result.copy_state
        rules = kw["rules"]
    return Replayed(window, seat, copy_state, rules, tuple(epochs))


@dataclass(frozen=True)
class Replayer:
    """What `v2_sync.resync` calls under v4 to bring the reference's state at
    k forward over this validator's epochs k+1 .. before-1. `memo_for`
    ((epoch, its recorded nonce) -> memo_v4.EpochMemo or None) gives each
    replayed epoch's memo, as the live loop built it; None replays every
    epoch without one."""
    directory: str
    hidden_of: object
    benchmark_of: Mapping[str, str]
    max_span: int
    memo_for: Callable | None = None
    # The root's head publication (hash -> bytes or None), for a holder the
    # replayed seat names that neither the record nor the head store holds.
    holder_bytes: Callable | None = None

    def __call__(self, rebuilt, k: int, before: int) -> Replayed:
        from fugal_subnet import copy_rule

        return replay(Store(self.directory), window=rebuilt.window, seat=rebuilt.seat,
                      copy_state=copy_rule.from_rebuild(rebuilt) or copy_rule.CopyState(),
                      rules=rebuilt.extra.get("rules"), epochs=range(int(k) + 1, int(before)),
                      hidden_of=self.hidden_of, benchmark_of=self.benchmark_of,
                      memo_for=self.memo_for, max_span=self.max_span,
                      holder_bytes=self.holder_bytes)


__all__ = ["FORMAT", "Missing", "Replayed", "Replayer", "Store", "inputs", "record", "replay"]
