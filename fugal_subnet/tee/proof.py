"""Proof data models and shared helpers for TEE-attested benchmarks.

BenchmarkProof is what a miner produces inside the TEE after running
the routing benchmark. QuestionResult holds per-question routing
decisions and grades.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from fugal_subnet.routing_protocol import identity
from fugal_subnet.tee.endpoint_reads import (
    EndpointRead,
    reads_from_json,
    reads_to_json,
)


def compute_questions_hash(question_ids: list[str]) -> str:
    """Compute the canonical hash for a set of question IDs."""
    canonical = json.dumps(sorted(question_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class QuestionResult:
    question_id: str
    routed_model: str
    correct: bool
    cost_usd: float
    response_hash: str
    # Attested token counts. Carrying them makes every model's counterfactual
    # cost on this question exactly computable from the pinned price table
    # (cost_m = p_in_m * prompt_tokens + p_out_m * completion_tokens), so cost
    # comparisons need no assumed "typical question" constant anywhere.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # True for nonce-forced exploration routes. These never count toward the
    # miner's own score — being made to route randomly must not be a penalty —
    # they exist only to sample the model pool for the reference frame.
    is_exploration: bool = False
    # True when the model call did not complete (timeout, connection drop,
    # non-2xx from the proxy). Recorded as its own fact, in the attested proof,
    # because "graded wrong" and "never answered" mean different things to the
    # reference frame: a wrong answer measures the model, a failed call
    # measures the network, and a miner controls its own network. Before this
    # field existed a coalition could lower every model's measured accuracy by
    # dropping connections during exploration only (diligence 2026-10, finding
    # 2). A failed result is always `correct=False` and `verify_proof` holds
    # it to that.
    failed: bool = False
    # Wall time the harness spent on this call, retries included, in whole
    # milliseconds (protocol v2). Attested like everything else in the proof,
    # because the v2 time gate ("per-call time within 2x the anchor's") needs a
    # number the miner cannot choose. None on v1 proofs, and then absent from
    # the hashed payload, so a v1 proof hashes exactly as it did before.
    latency_ms: int | None = None
    # Protocol v4 (docs/decisions/LIVE_MENU_V4.md, section 12). Each is None on
    # an earlier protocol's result and then absent from the hashed payload and
    # from the JSON, so such a result hashes exactly as before.
    # `provider_cost`: what the provider billed for this call (`usage.cost` of
    # its replies), in integer nanodollars; None when no reply reported one.
    provider_cost: int | None = None
    # `served_by`: the provider that answered, as its reply names it ("" when
    # it names none).
    served_by: str | None = None
    # `holder_pick`: True on a challenge's holder-side call (FAST_CROWN_V4.md
    # 2): the seat holder's pick on a challenge question where it differs from
    # the challenger's own. Such a result is not the challenger's answer
    # (`BenchmarkProof.scored_results` leaves it out; `holder_results` lists
    # it). Absent (None) on every other result.
    holder_pick: bool | None = None
    # `not_started`: True on a holder-side call the challenger's TD did not
    # start (its time or dollar budget ran out first, in the order
    # picks_v4.call_order_key gives). Such a result is failed, wrong, with no
    # tokens and no cost; it is no item of the challenge
    # (picks_v4.challenge_items). Absent (None) on every other result.
    not_started: bool | None = None

    def hash_payload(self) -> dict:
        """The fields `content_hash` covers for this result.

        `failed` is included ONLY when True. A proof with no failed calls then
        hashes exactly as it did before the field existed, so a miner and a
        validator on either side of this change agree on every honest proof;
        only a proof that actually records a failure needs both sides to know
        the field. Deterministic either way: the same object always yields the
        same payload.
        """
        payload = {
            "question_id": self.question_id,
            "routed_model": self.routed_model,
            "correct": self.correct,
            "cost_usd": self.cost_usd,
            "response_hash": self.response_hash,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "is_exploration": self.is_exploration,
        }
        if self.failed:
            payload["failed"] = True
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.provider_cost is not None:
            payload["provider_cost"] = self.provider_cost
        if self.served_by is not None:
            payload["served_by"] = self.served_by
        if self.holder_pick is not None:
            payload["holder_pick"] = self.holder_pick
        if self.not_started is not None:
            payload["not_started"] = self.not_started
        return payload

    def to_dict(self) -> dict:
        return {**self.hash_payload(), "failed": self.failed}

    @classmethod
    def from_dict(cls, d: dict, *, v4_fields: bool = True) -> QuestionResult:
        # `v4_fields` False reads a result of a v1-v3 proof: protocol v4's
        # keys are not read at all, exactly as before they existed.
        if not v4_fields:
            d = {k: v for k, v in d.items() if k not in V4_RESULT_KEYS}
        elif "measure_for" in d:
            # Proposals are filed on chain and measured by the field's
            # exploration (FAST_CROWN_V4.md 18): no v4 result measures one.
            raise ValueError("a v4 result carries no measure_for: no proof measures a proposal")
        # Numbers are kept EXACTLY as the JSON carried them. `content_hash`
        # canonicalises `hash_payload()`, and JSON distinguishes `0` from
        # `0.0`: the miner hashes the in-memory object it serialised, so a
        # validator that coerced `cost_usd` to float re-hashed a *different*
        # payload whenever a result arrived with an integer cost. That is what
        # every failed call carries (`cost_usd: 0`), and on mainnet day one
        # (2026-11-03, epoch e00026188) it made both honest seeds' proofs fail
        # `report_data == content_hash` while the quotes were genuine. Types
        # are still checked: a bool (a JSON true/false, which Python would
        # happily treat as 1/0) or a string is rejected, never reinterpreted.
        r = cls(
            question_id=d["question_id"],
            routed_model=d["routed_model"],
            correct=bool(d["correct"]),
            cost_usd=_json_number(d["cost_usd"], "cost_usd"),
            response_hash=d["response_hash"],
            prompt_tokens=_json_number(d.get("prompt_tokens", 0), "prompt_tokens"),
            completion_tokens=_json_number(d.get("completion_tokens", 0), "completion_tokens"),
            is_exploration=bool(d.get("is_exploration", False)),
            failed=bool(d.get("failed", False)),
            latency_ms=_latency_ms(d["latency_ms"]) if "latency_ms" in d else None,
            provider_cost=(_provider_cost(d["provider_cost"]) if "provider_cost" in d else None),
            served_by=_served_by(d["served_by"]) if "served_by" in d else None,
            holder_pick=_holder_pick(d["holder_pick"]) if "holder_pick" in d else None,
            not_started=_not_started(d["not_started"]) if "not_started" in d else None,
        )
        if r.holder_pick and r.is_exploration:
            raise ValueError("a holder-side challenge call is not exploration")
        if r.not_started and not (r.holder_pick and r.failed and not r.correct
                                  and r.prompt_tokens == 0 and r.completion_tokens == 0
                                  and r.cost_usd == 0 and r.provider_cost is None):
            raise ValueError("only a holder-side call is reported not started, and then failed, "
                             "with no tokens and no cost")
        return r


def _holder_pick(value):
    """Present only as JSON true: a holder-side challenge call."""
    if value is not True:
        raise ValueError("holder_pick is present only as true")
    return value


def _not_started(value):
    """Present only as JSON true: a holder-side call not started."""
    if value is not True:
        raise ValueError("not_started is present only as true")
    return value


def _latency_ms(value):
    """A latency is a non-negative JSON integer. Kept exactly as carried (the
    #120 lesson: re-typing a number changes the hash); a bool, float, string
    or negative value is rejected, never reinterpreted."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"latency_ms must be a non-negative integer, got {value!r}")
    return value


# Protocol v4's keys. A proof in a v1-v3 namespace is read as it was before
# they existed: its v4 keys are not read, so they are neither hashed nor
# refused, on every validator alike (`BenchmarkProof.from_dict`). `proposal`
# and `measure_for` are kept in these sets for that reason only: a v4 proof
# that carries either is refused (proposals are filed on chain).
V4_RESULT_KEYS = frozenset({"provider_cost", "served_by", "measure_for", "holder_pick",
                            "not_started"})
V4_PROOF_KEYS = frozenset({"menu_hash", "endpoint_reads", "proposal", "scored", "challenge"})

# Bounds on protocol v4's per-result fields (I2).
MAX_PROVIDER_COST_NANO = 10**12      # $1,000 for one call: far above any real one
MAX_SERVED_BY_LEN = 64


def _provider_cost(value):
    """Billed nanodollars: a non-negative JSON integer, bounded, kept as carried."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_PROVIDER_COST_NANO:
        raise ValueError(f"provider_cost must be an integer in 0..{MAX_PROVIDER_COST_NANO}")
    return value


def served_by_ok(value) -> bool:
    """A provider name a proof may carry: printable ASCII, at most MAX_SERVED_BY_LEN."""
    return (isinstance(value, str) and len(value) <= MAX_SERVED_BY_LEN
            and all(0x20 <= ord(c) <= 0x7E for c in value))


def _served_by(value):
    if not served_by_ok(value):
        raise ValueError(f"served_by must be printable ASCII of at most {MAX_SERVED_BY_LEN} "
                         "characters")
    return value


def _json_number(value, name: str):
    """A JSON number as it arrived: int stays int, float stays float.

    Rejects bool (a subclass of int in Python, but `true` is not a cost) and
    anything that is not a finite number. No coercion in either direction: the
    value goes back into `hash_payload()` byte-identical to what the miner
    hashed.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number, got {type(value).__name__}")
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise ValueError(f"{name} must be finite")
    return value


# Bounds on proof-carried TPM intermediates (I2: every miner-controlled input
# is bounded before it is parsed). A real chain needs one or two; a Google
# intermediate is ~1.5 KB DER.
TPM_INTERMEDIATES_MAX = 4
TPM_INTERMEDIATE_MAX_BYTES = 8192


def _parse_tpm_intermediates(raw) -> list[bytes]:
    """Hex strings -> DER bytes, bounded. Anything else is a malformed proof."""
    if not isinstance(raw, list):
        raise ValueError("tpm_intermediates must be a list")
    if len(raw) > TPM_INTERMEDIATES_MAX:
        raise ValueError(f"tpm_intermediates carries {len(raw)} certificates "
                         f"(max {TPM_INTERMEDIATES_MAX})")
    out = []
    for c in raw:
        if not isinstance(c, str) or len(c) > 2 * TPM_INTERMEDIATE_MAX_BYTES:
            raise ValueError("tpm_intermediates entry is not a bounded hex string")
        out.append(bytes.fromhex(c))
    return out


# A challenge's sizes as a proof may carry them (config.CHALLENGE_SIZES_V4).
CHALLENGE_SIZES = (500, 1000)
_CHALLENGE_KEYS = frozenset({"size", "holder"})


@dataclass(frozen=True)
class Challenge:
    """What a v4 proof's challenge is (FAST_CROWN_V4.md 2): the first `size`
    questions of the epoch's challenge list, paired with the seat holder whose
    head hash is `holder` (None: no holder was named, the open form, and no
    holder-side call is made)."""

    size: int
    holder: str | None

    def to_json(self) -> dict:
        return {"size": self.size, "holder": self.holder}

    @classmethod
    def from_json(cls, d) -> Challenge:
        """One encoding, strict: exactly `size` (500 or 1000, a JSON integer)
        and `holder` (64 lowercase hex characters, or null)."""
        if not isinstance(d, dict) or set(d) != _CHALLENGE_KEYS:
            raise ValueError(f"a challenge must be an object with exactly {sorted(_CHALLENGE_KEYS)}")
        size, holder = d["size"], d["holder"]
        if type(size) is not int or size not in CHALLENGE_SIZES:
            raise ValueError(f"a challenge's size is one of {CHALLENGE_SIZES}")
        if holder is not None:
            _menu_hash(holder)
        return cls(size=size, holder=holder)


def _earlier_namespace(routing_protocol) -> bool:
    """True when a proof names the v1, v2 or v3 evidence namespace."""
    from fugal_subnet.routing_protocol import identity_for
    return isinstance(routing_protocol, str) and routing_protocol in {
        identity_for(v) for v in (1, 2, 3)}


def _menu_hash(value) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("menu_hash must be 64 lowercase hex characters")
    return value


@dataclass
class BenchmarkProof:
    epoch_id: str
    nonce: str
    questions_hash: str
    weights_hash: str
    source_hash: str
    results: list[QuestionResult]
    total_cost_usd: float
    per_model_costs: dict[str, float]
    attestation_quote: bytes
    timestamp: float
    # WHOSE proof this is. Defaults to "" so that a proof carrying no hotkey is
    # representable — it has to be, or the validator could not reject one — but
    # an empty value is never valid under verification. See verify_proof.
    #
    # Before this existed, a proof was bound to an epoch, a slice, a head and a
    # runtime, and to NO MINER. Any party could take an honest miner's proof off
    # its public axon response, commit the same weights_hash, and present it as
    # their own; verification passed, and what stopped it was dedup plus
    # commit-block seniority — statistical, not cryptographic.
    hotkey: str = ""
    routing_protocol: str = field(default_factory=identity)
    # DER intermediates for the TPM attestation key's certificate chain, carried
    # so a miner whose region (or whose rotated intermediate) is not vendored
    # still verifies. Untrusted on purpose: a supplied certificate is worthless
    # unless it chains to the pinned Google root (tpm.verify_ak_chain), exactly
    # as in TLS. In content_hash ONLY WHEN NON-EMPTY (2026-11-08), so it is bound
    # through report_data (one proof_hash, one set of intermediates for every
    # validator: I1) while a proof without it keeps its hash. The miner attaches
    # them to v2 proofs only (tpm.intermediates_for, at startup), so a v1 proof
    # from the v2 image is byte-identical for a validator not yet upgraded.
    # Omitted from to_dict when empty. Bounded in from_dict (I2).
    tpm_intermediates: list[bytes] = field(default_factory=list)
    # Protocol v4 (docs/decisions/LIVE_MENU_V4.md, section 12). None on every
    # earlier proof, and then absent from content_hash and from to_dict, so
    # such a proof keeps its bytes and its hash. A v4 proof always carries
    # `menu_hash` (the menu its run used, whose identity_v4 is its
    # `routing_protocol`) and `endpoint_reads` (possibly empty: a failed read
    # leaves no record).
    menu_hash: str | None = None
    endpoint_reads: tuple[EndpointRead, ...] | None = None
    # `scored`: whether the TD answered the epoch's scored questions
    # (docs/decisions/SEAT_MINIMAL_V4.md 3.4: they are optional under v4).
    # True: the results hold the whole scored slice; False: none of it, and
    # the proof is exploration only. A v4 proof always
    # says which (verify_proof refuses one that does not, and one whose
    # results disagree with what it says: all of the slice or none).
    scored: bool | None = None
    # `challenge` (FAST_CROWN_V4.md 2): present exactly when `scored` is true.
    # The proof's scored results are then its own picks on the first
    # `challenge.size` questions of the epoch's challenge list (the scored
    # slice first), and its `holder_results` the holder's picks where they
    # differ from its own.
    challenge: Challenge | None = None

    @property
    def scored_results(self) -> list[QuestionResult]:
        """Results the miner is judged on — exploration is excluded, and so
        is a v4 challenge's holder-side call (it is not the miner's answer)."""
        return [r for r in self.results if not r.is_exploration and not r.holder_pick]

    @property
    def holder_results(self) -> list[QuestionResult]:
        """A v4 challenge's holder-side calls (`holder_pick` set)."""
        return [r for r in self.results if r.holder_pick]

    @property
    def exploration_results(self) -> list[QuestionResult]:
        """Exploration answers: the nonce assigned the entry."""
        return [r for r in self.results if r.is_exploration]

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.scored_results if r.correct)

    @property
    def n_total(self) -> int:
        return len(self.scored_results)

    @property
    def accuracy(self) -> float:
        scored = self.scored_results
        if not scored:
            return 0.0
        return self.n_correct / len(scored)

    @property
    def scored_cost_usd(self) -> float:
        """Cost of the scored routes only.

        Exploration is a forced cost the miner did not choose, so billing it
        against their thrift would punish them for the subnet's own sampling.
        """
        return sum(r.cost_usd for r in self.scored_results)

    def content_hash(self) -> str:
        """SHA256 of the proof content (excluding attestation).

        This is the value bound into the TDX report_data field so the
        hardware attestation covers the proof's content.
        """
        payload = {
            # First, because it is the thing that was missing. content_hash is
            # bound into the TDX report_data, so including the hotkey here makes
            # the HARDWARE attest whose proof this is. Relay then fails on a
            # signature rather than on a similarity threshold.
            "hotkey": self.hotkey,
            "routing_protocol": self.routing_protocol,
            "epoch_id": self.epoch_id,
            "nonce": self.nonce,
            "questions_hash": self.questions_hash,
            "weights_hash": self.weights_hash,
            "source_hash": self.source_hash,
            "results": [r.hash_payload() for r in self.results],
            "total_cost_usd": self.total_cost_usd,
            "per_model_costs": self.per_model_costs,
            "timestamp": self.timestamp,
        }
        if self.tpm_intermediates:
            # Bound when present (see the field). Absent otherwise, so every
            # proof without intermediates hashes exactly as before.
            payload["tpm_intermediates"] = [c.hex() for c in self.tpm_intermediates]
        payload.update(self._v4_fields())
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "hotkey": self.hotkey,
            "routing_protocol": self.routing_protocol,
            "epoch_id": self.epoch_id,
            "nonce": self.nonce,
            "questions_hash": self.questions_hash,
            "weights_hash": self.weights_hash,
            "source_hash": self.source_hash,
            "results": [r.to_dict() for r in self.results],
            "total_cost_usd": self.total_cost_usd,
            "per_model_costs": self.per_model_costs,
            "attestation_quote": self.attestation_quote.hex(),
            "timestamp": self.timestamp,
            **({"tpm_intermediates": [c.hex() for c in self.tpm_intermediates]}
               if self.tpm_intermediates else {}),
            **self._v4_fields(),
        }

    def _v4_fields(self) -> dict:
        """The v4 fields that are present, in their JSON form: the same in
        content_hash and in to_dict."""
        out: dict = {}
        if self.menu_hash is not None:
            out["menu_hash"] = self.menu_hash
        if self.endpoint_reads is not None:
            out["endpoint_reads"] = reads_to_json(self.endpoint_reads)
        if self.scored is not None:
            out["scored"] = self.scored
        if self.challenge is not None:
            out["challenge"] = self.challenge.to_json()
        return out

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkProof:
        # A proof in a v1, v2 or v3 namespace is read exactly as before
        # protocol v4's fields existed: its v4 keys are not read, so they are
        # neither hashed nor refused, and every validator, whichever release
        # it runs, reaches the same verdict on it. Any other proof has its v4
        # fields read, strictly.
        if _earlier_namespace(d.get("routing_protocol", "")):
            d = {k: v for k, v in d.items() if k not in V4_PROOF_KEYS}
            results = [QuestionResult.from_dict(r, v4_fields=False) for r in d["results"]]
        elif "proposal" in d:
            raise ValueError("a v4 proof carries no proposal: proposals are filed on chain")
        else:
            results = [QuestionResult.from_dict(r) for r in d["results"]]
        return cls(
            # Absent rather than empty for a proof written before this field
            # existed. Either way it fails verification; the distinction only
            # matters for reading old artifacts without an exception.
            hotkey=d.get("hotkey", ""),
            routing_protocol=d.get("routing_protocol", ""),
            epoch_id=d["epoch_id"],
            nonce=d["nonce"],
            questions_hash=d["questions_hash"],
            weights_hash=d["weights_hash"],
            source_hash=d["source_hash"],
            results=results,
            total_cost_usd=d["total_cost_usd"],
            per_model_costs=d["per_model_costs"],
            attestation_quote=bytes.fromhex(d["attestation_quote"]),
            timestamp=d["timestamp"],
            tpm_intermediates=_parse_tpm_intermediates(d.get("tpm_intermediates", [])),
            menu_hash=_menu_hash(d["menu_hash"]) if "menu_hash" in d else None,
            endpoint_reads=(reads_from_json(d["endpoint_reads"]) if "endpoint_reads" in d
                            else None),
            scored=_scored_flag(d["scored"]) if "scored" in d else None,
            challenge=Challenge.from_json(d["challenge"]) if "challenge" in d else None,
        )


def _scored_flag(value) -> bool:
    """A JSON true or false, kept as carried; anything else is refused."""
    if not isinstance(value, bool):
        raise ValueError("scored must be true or false")
    return value
