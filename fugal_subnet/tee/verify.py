"""Proof verification for TEE-attested benchmarks.

The validator calls verify_proof() for each miner's submission. It checks
attestation, measurements, question consistency, and cost consistency —
without ever calling a model.

The security model in one line: *nothing a miner asserts about itself is
trusted; only what the hardware measured, or what a hash chain forces to be
true, is trusted.*

That distinction is the reason this module rejects `proof.source_hash` as an
image identity. A workload running inside a genuine TDX VM produces a genuine
Intel-signed quote, so DCAP verification passes for an attacker who simply
runs *modified* code on real hardware. What separates honest from modified is
the measurement register the CPU filled in, not a string the workload wrote
about itself.

The chain that makes a proof trustworthy (docs/INVARIANTS.md I8, "what
attested actually means", is the full statement):

    Google TPM quote, signed by a Google-chained AK, over sha256(tdx_quote)
      -> the TDX quote came from a Google Confidential VM (unwrap_attestation)
    Intel DCAP signature
      -> quote is genuine, from real TDX hardware
    TDATTRIBUTES.DEBUG clear
      -> the host could not read or change the TD
    measurement_id(quote) in approved_measurements
      -> the image that booted is the published one
    the RTMR3 event log replays to the quote, and names an approved compose hash
      -> the application that ran is the published one
    report_data == proof.content_hash()
      -> the proof body is exactly what that image produced
    proof.hotkey == this uid's hotkey, nonce == this epoch's
      -> the proof is this miner's, for this epoch
    proof.weights_hash == on-chain commitment, == sha256(head bytes)
      -> the head that ran is the head that was committed, before the nonce
    result question ids == the assigned slice, exploration == the nonce's targets
      -> those answers are to the questions we asked
    per-question costs reconcile with the totals
      -> the cost figures are internally consistent

Protocol v4 (docs/decisions/LIVE_MENU_V4.md, sections 4.1, 5, 8 and 12) adds,
against the epoch's MenuState (`V4Expected`, built by `expected_v4`):

    proof.menu_hash == menu_hash(MenuState(E)), namespace == identity_v4(it)
      -> the run used the epoch's menu, not a stale or another one
    every scored route is an entry of that menu
      -> the head picked among the entries the epoch offers
    exploration == the shared set, the cost-aware and the coverage parts
      -> the nonce, the hotkey and the menu's shares chose every pair
    proof.challenge: a prefix of the epoch's challenge list of 500 or 1000
      -> a challenge is optional (FAST_CROWN_V4.md 2), never a chosen subset

and under v4 the exploration failure bound is absolute and taken across the
epoch's proofs (picks_v4.failure_refusals: more than FAIL_MAX_BP_V4 of a
proof's exploration calls failed, outside an outage epoch), not the v1-v3
comparison with the scored calls a v4 proof need not carry.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from fugal_subnet.config import EXPLORE_FAIL_MARGIN
from fugal_subnet.tee.attestation import (
    CollateralUnavailable,
    extract_report_data,
    measurement_id,
    parse_quote,
    replay_event_log,
    td_is_debug,
    verify_dcap,
)
from fugal_subnet.tee.proof import (
    BenchmarkProof,
)
from fugal_subnet.tee.proof import (
    compute_questions_hash as compute_questions_hash,  # noqa: F401 — re-export
)

logger = logging.getLogger(__name__)

# Both halves of a live entry: measurement_id() is a sha256, and the app half is
# the sha256 of the frozen app-compose.json, replayed out of the RTMR3 log.
_HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass
class VerifyResult:
    valid: bool
    reason: str = ""
    warnings: list[str] | None = None
    # "This validator could not check the proof", as distinct from "the proof
    # is bad". Defaults False so every existing construction keeps the meaning
    # it has today and only a deliberate one opts in — a third truthy state
    # someone can forget to branch on would be worse than no distinction.
    #
    # NEVER set this for anything a miner's bytes could have caused. A miner who
    # can reach this branch becomes UNSCOREABLE AT WILL: neither rewarded nor
    # punished, absent from the invalid count, and invisible to the attack
    # suite. That is an I4 problem wearing an I6 costume, and it is why the only
    # things that set it are the operator's own infrastructure.
    unverifiable: bool = False


def parse_approved(entries, require_app: bool = False) -> dict[str, set[str]]:
    """Turn the approved list into {base_measurement: {app_identity, ...}}.

    An entry is either

        <base>                 the image is approved, nothing is required of RTMR3
        <base>:<compose_hash>  and the compose hash in the RTMR3 log must match

    Two halves rather than one hash, because they have different lifecycles and
    different approvers. The base rotates when the image or kernel changes,
    which is the cloud provider's schedule; the app identity rotates when our
    code, pool or grader changes, which is ours. Hashing them together would
    force a single rotation for either event and make "what code is approved"
    unreadable. Kept apart, the app identity is computable off-hardware from the
    repo, so it is reviewable in a pull request instead of requiring someone to
    hold a quote.

    A bare base is the pre-existing behaviour and stays PARSEABLE, for mock
    runs and local testnets where no application is being bound. It is never a
    live configuration: the model upstream is fixed only by the compose that
    the app identity measures, so a bare base approves a TD whose upstream the
    miner chose -- the exploit `scripts/stub_upstream.py` demonstrates
    (diligence 2026-10, finding 1). `require_app=True` makes a bare base a
    loud ValueError; the live validator passes it at startup, and
    `verify_proof` refuses a bare-base match outside mock regardless.

    `require_app=True` also holds both halves to the one live shape, 64 hex
    each (case-insensitive, stored lower-case). Anything else is a typo that
    would match no proof, or an identity the live verifier cannot check, and
    either way the operator would see the whole field refused with no hint.

    THE APP IDENTITY IS NEVER AN RTMR3 VALUE, and storing one here would be a
    subtle, expensive mistake. A measured image extends RTMR3 several times
    during boot and one of those events is `instance-id`, which changes on every
    deploy — so two honest miners running the identical approved application
    produce different RTMR3 values, as does the same miner after a redeploy.
    Measured on two real dstack deploys of the same compose.

    What is approved is the compose hash carried in an event PAYLOAD. The replay
    authenticates the log against the register the CPU signed; the payload is the
    thing compared. An approved list holding RTMR3 values would pass its first
    test and reject every honest miner from their second deploy onward, and the
    symptom would look like an attack rather than a design error.
    """
    out: dict[str, set[str]] = {}
    for raw in entries:
        entry = str(raw).strip()
        if not entry:
            continue
        base, sep, app = entry.partition(":")
        base, app = base.strip().lower(), app.strip().lower()
        # A colon with nothing after it is a typo, not an intention, and
        # interpreting it as the weaker form silently disables the app binding
        # while the operator believes they configured one. That is the exact
        # failure this codebase keeps finding: a security check that reports
        # success and does nothing. An unexpanded shell variable or a trailing
        # copy-paste is all it takes, so it fails loudly instead.
        if sep and not app:
            raise ValueError(
                f"Approved measurement {entry!r} ends in a colon with no runtime "
                "identity. Write '<base>' for an image-only entry, or "
                "'<base>:<app_identity>' to require one — never a bare colon, "
                "which would silently accept any runtime."
            )
        if not base:
            raise ValueError(f"Approved measurement {entry!r} has no base measurement")
        if require_app and not app:
            raise ValueError(
                f"Approved measurement {entry!r} is a bare base with no app "
                "identity. A bare base approves ANY application on that image, "
                "including one whose model upstream the miner chose. Live entries "
                "must be '<base>:<compose_hash>' (docs/APPROVED_ENTRY_ROTATION.md)."
            )
        if require_app and not (_HEX64.fullmatch(base) and _HEX64.fullmatch(app)):
            raise ValueError(
                f"Approved measurement {entry!r} is not '<base>:<compose_hash>' "
                "with 64 hex characters in each half (docs/APPROVED_ENTRY_ROTATION.md)."
            )
        out.setdefault(base, set())
        if app:
            out[base].add(app)
    return out


# A bare TDX quote's first field is a u16 little-endian version. Anything else
# at offset 0 is an envelope, not a quote.
_TDX_QUOTE_VERSIONS = (4, 5)


def unwrap_attestation(blob: bytes, extra_certs: "list[bytes] | None" = None
                       ) -> tuple[bytes, list | None]:
    """Return (tdx_quote, event_log) for either a bare quote or a dstack blob.

    A miner running under dstack sends the whole attestation — TDX quote, event
    log, and on GCP a TPM quote — not a bare quote. Both shapes have to verify,
    and which one arrived must be decided from the bytes rather than from
    configuration, because two validators configured differently would disagree
    about the same proof.

    **The TPM half is verified here, not optionally later.** If the envelope
    carries one and it does not verify, the proof is rejected: a validator that
    checked only the TDX half would accept proofs another validator rejects,
    which is a fork with no bug behind it. That is also why an absent
    `cryptography` raises rather than returning invalid — a missing library is
    the operator's misconfiguration, and downgrading it to "this proof is bad"
    would let a --live validator reject the whole field while looking healthy.
    Identical reasoning to the ImportError handling around `verify_dcap`.
    """
    if len(blob) >= 2 and int.from_bytes(blob[:2], "little") in _TDX_QUOTE_VERSIONS:
        return blob, None

    from fugal_subnet.scale import ScaleError, decode_dstack_attestation

    try:
        att = decode_dstack_attestation(blob)
    except ScaleError as e:
        raise ValueError(f"attestation is neither a TDX quote nor a dstack blob: {e}") from e

    from fugal_subnet.tee.tpm import TpmError, parse_attest, verify_tpm_quote

    try:
        # extra_certs: intermediates the proof carries (BenchmarkProof.
        # tpm_intermediates). Safe to accept from the miner: they only help if
        # they chain to the pinned root by signature. Since 2026-11-08 they are
        # inside content_hash when present, so they are bound to proof_hash and
        # report_data and every validator sees the same set (I1). The miner
        # attaches them to v2 proofs (tpm.intermediates_for).
        ok = verify_tpm_quote(att["tpm"], extra_certs=extra_certs)
    except TpmError as e:
        raise ValueError(f"TPM quote rejected: {e}") from e
    if not ok:
        raise ValueError(
            "TPM quote verification failed — the attestation key certificate "
            "does not chain to Google's pinned EK/AK root, or the quote was not "
            "signed by it"
        )

    # The TPM half must be THIS TDX quote's TPM half. dstack's GCP agent puts
    # sha256(tdx_quote) in TPMS_ATTEST.extra_data (measured on both real
    # fixtures), which is the only thing tying the two signatures together.
    # Without this check a genuine Google-signed TPM quote from any Confidential
    # VM could be stapled onto any TDX quote, and the "second trust root" proved
    # only that some GCP vTPM exists somewhere (found 2026-11-02; I8).
    try:
        extra = parse_attest(att["tpm"]["message"])["extra_data"]
    except (TpmError, KeyError, ValueError) as e:
        raise ValueError(f"TPM quote rejected: unreadable TPMS_ATTEST: {e}") from e
    expected = hashlib.sha256(att["quote"]).hexdigest()
    if extra != expected:
        raise ValueError(
            "TPM quote is not bound to this TDX quote: TPMS_ATTEST.extra_data is "
            f"{extra[:16]}..., sha256(tdx_quote) is {expected[:16]}... — the two "
            "halves of the attestation come from different quotes"
        )

    return att["quote"], att["events"]


@dataclass(frozen=True)
class V4Expected:
    """What one hotkey's protocol v4 proof must hold for epoch E, from
    published inputs only (`expected_v4`): the epoch's menu and its hash, the
    exploration the nonce, the hotkey and the menu's shares assign (the
    shared set and the miner's two private parts)."""

    state: object                                  # menu_state.MenuState of epoch E
    menu_hash: str
    hotkey: str
    shared: dict[str, str]
    cost_aware: dict[str, str]
    coverage: dict[str, str]
    # The epoch's challenge list, in order (epoch_plan.challenge_list_v4: the
    # scored slice first). A challenge of n answers its first n.
    challenge: tuple[str, ...] = ()

    @property
    def exploration(self) -> dict[str, str]:
        """{question id: entry id}: the shared set and both private parts."""
        return {**self.shared, **self.cost_aware, **self.coverage}

    @property
    def private_ids(self) -> frozenset[str]:
        """The questions of the miner's private set: a pair on one of them is
        weighed by the private set's weight when pooled (explore_pool)."""
        return frozenset(self.cost_aware) | frozenset(self.coverage)


def expected_v4(state, *, epoch: int, nonce: bytes, pool: list[dict], scored_ids,
                hotkey: str, plan, shares=None, challenge_ids=None) -> V4Expected:
    """`V4Expected` for `hotkey` in v4 epoch `epoch`, whose menu is `state`
    (menu_state.menu_for_epoch's result for the epoch, carried or committed).

    - exploration: `epoch_plan.exploration_plan_v4` on
      `epoch_plan.explore_shares_v4(state)`, section 8's draw, which the TD
      runs as `td_v4.exploration_v4`.

    `nonce` is the epoch nonce as bytes (the harness's: bytes.fromhex of the
    64-hex nonce), `scored_ids` the epoch's whole scored slice, and `plan` the
    epoch's `td_v4.V4Plan`: `epoch_plan.plan_v4(pool)`, the one way to build
    it, which the miner's TD runs too (its pool index). `shares` is the epoch's `epoch_plan.explore_shares_v4(state)`:
    a function of the menu alone, so a validator computes it once per epoch
    and hands it to every hotkey's expectation (None computes it here).
    Shares over another set of entries are refused. Pure: every validator
    derives the same expectation."""
    from fugal_subnet import epoch_plan
    from fugal_subnet import menu_state as ms

    if not isinstance(state, ms.MenuState):
        raise ValueError("a v4 expectation is read from the epoch's menu (a menu_state.MenuState)")
    ms.validate(state)
    if state.epoch != int(epoch):
        raise ValueError(f"the menu given is the menu of epoch {state.epoch}, not of {epoch}")
    if not hotkey:
        raise ValueError("a v4 expectation is drawn for a hotkey")
    scored = set(scored_ids)
    if shares is None:
        shares = epoch_plan.explore_shares_v4(state)
    elif sorted(getattr(shares, "cost", ())) != list(state.ids()):
        raise ValueError("the exploration shares handed in are not over this epoch's menu")
    shared, cost_aware, coverage = epoch_plan.exploration_plan_v4(
        nonce, pool, scored, shares, hotkey, plan.pool_index)
    if challenge_ids is None:
        # A plan that cannot draw the list (a slice of another size than its
        # own) gives none: every challenge on it is then refused, never
        # checked against a guess.
        try:
            challenge_ids = [q["question_id"]
                             for q in plan.challenge_list(nonce, pool, len(scored))]
        except (ValueError, AttributeError):
            challenge_ids = []
    return V4Expected(state=state, menu_hash=ms.menu_hash(state), hotkey=hotkey,
                      shared=dict(shared), cost_aware=dict(cost_aware), coverage=dict(coverage),
                      challenge=tuple(challenge_ids))


@dataclass(frozen=True)
class V5Expected:
    """What one hotkey's protocol v5 proof must hold for epoch E: the menu
    and coverage cells as a v4 expectation with no shared, cost-aware or
    challenge part (`v4`), the root's exam spec hash it answers (None on a
    coverage-only epoch) and the exam cells it must call: the spec's cells
    assigned to it whose question is in the pool and whose entry is on the
    epoch's menu (the TD calls no other: tee/harness, td_v5)."""

    v4: V4Expected
    spec_hash: str | None
    exam_cells: frozenset


def expected_v5(state, *, epoch: int, nonce: bytes, pool: list[dict], hotkey: str,
                spec=None, index=None) -> V5Expected:
    """`V5Expected` for `hotkey` in v5 epoch `epoch` on menu `state`, with
    `spec` the exam spec the root published for the epoch (None: coverage
    only). The coverage cells are td_v5.coverage_v5's, drawn outside the
    spec's questions, exactly as the TD draws them. Pure."""
    from fugal_subnet import menu_state as ms
    from fugal_subnet.exam_v5 import assigned_cells
    from fugal_subnet.tee import td_v5

    if not isinstance(state, ms.MenuState):
        raise ValueError("a v5 expectation is read from the epoch's menu (a menu_state.MenuState)")
    ms.validate(state)
    if state.epoch != int(epoch):
        raise ValueError(f"the menu given is the menu of epoch {state.epoch}, not of {epoch}")
    if not hotkey:
        raise ValueError("a v5 expectation is drawn for a hotkey")
    if spec is not None and int(spec.epoch) != int(epoch):
        raise ValueError(f"the exam spec given is for epoch {spec.epoch}, not {epoch}")
    spec_q = set(spec.questions) if spec is not None else set()
    cells: frozenset = frozenset()
    if spec is not None:
        in_pool = {str(q["question_id"]) for q in pool}
        on_menu = set(state.ids())
        cells = frozenset((q, e) for q, e in assigned_cells(spec, hotkey)
                          if q in in_pool and e in on_menu)
    coverage = td_v5.coverage_v5(nonce, pool, spec_q, state, hotkey, index=index)
    v4 = V4Expected(state=state, menu_hash=ms.menu_hash(state), hotkey=hotkey, shared={},
                    cost_aware={}, coverage=dict(coverage), challenge=())
    return V5Expected(v4=v4, spec_hash=(spec.spec_hash() if spec is not None else None),
                      exam_cells=cells)


def questions_hash_v5(v5: V5Expected) -> str:
    """The questions hash a v5 proof carries: the sorted, distinct question ids
    of its exam cells (`compute_questions_hash`), exactly as the TD computes it
    (tee/harness, v5 path: `compute_questions_hash(sorted(set(exam_qids)))` over
    the assigned cells it calls). A coverage-only epoch's is the hash of []. It
    is never the v4 scored slice's hash: a v5 TD answers no slice."""
    return compute_questions_hash(sorted({str(q) for q, _e in v5.exam_cells}))


def _v5_exam_check(proof: BenchmarkProof, v5: V5Expected) -> str | None:
    """A v5 proof answers no challenge and its exam calls are exactly its
    assigned cells (I8): each once, each to the cell's entry, the spec's hash
    bound into the proof; on a coverage-only epoch, no exam call at all."""
    if proof.scored is not False or proof.challenge is not None:
        return "A v5 proof answers no challenge (`scored` false, no `challenge`)"
    if proof.holder_results:
        return "A v5 proof makes no holder-side call"
    exam = [r for r in proof.results if r.exam]
    if v5.spec_hash is None:
        if exam or getattr(proof, "spec_hash", None) is not None:
            return "This epoch is coverage only: the proof carries exam calls or a spec hash"
        return None
    if getattr(proof, "spec_hash", None) != v5.spec_hash:
        return (f"The proof answers exam spec {str(getattr(proof, 'spec_hash', None))[:16]}, "
                f"not this epoch's {v5.spec_hash[:16]}")
    got = [(r.question_id, r.routed_model) for r in exam]
    if len(got) != len(set(got)):
        return f"{len(got) - len(set(got))} exam cells answered twice"
    if set(got) != set(v5.exam_cells):
        missing = len(set(v5.exam_cells) - set(got))
        extra = len(set(got) - set(v5.exam_cells))
        return (f"The exam calls are not this hotkey's assigned cells ({missing} missing, "
                f"{extra} not assigned)")
    return None


def _v4_menu_check(proof: BenchmarkProof, v4: V4Expected) -> str | None:
    """The proof ran on the epoch's menu: it names its hash, and its
    namespace is identity_v4 of that hash (section 4.1; A7). A proof made on
    the previous epoch's menu, or on any other, is refused here."""
    from fugal_subnet.routing_protocol import identity_v4

    if proof.menu_hash != v4.menu_hash:
        return (f"The proof ran on menu {str(proof.menu_hash)[:16]}..., not on this epoch's menu "
                f"{v4.menu_hash[:16]}...")
    if proof.routing_protocol != identity_v4(v4.menu_hash):
        return "The proof's evidence namespace is not identity_v4 of this epoch's menu"
    if proof.endpoint_reads is None:
        return "A v4 proof carries its endpoint reads (possibly none)"
    return None


def _v4_scored_check(proof: BenchmarkProof, v4: V4Expected,
                     slice_ids: set[str] | None) -> str | None:
    """The proof says whether it answered a challenge (its `scored` field),
    and its results are what it says (FAST_CROWN_V4.md 2, package F4):

    - `scored` false: no challenge, no scored result and no holder-side call:
      the proof is exploration (and any measurement) only;
    - `scored` true: a `challenge` of 500 or 1,000 questions; its scored
      results are exactly the first `size` questions of the epoch's challenge
      list (which begins with the whole slice), each once; with no holder
      named, no holder-side call; with one, holder-side calls only on those
      questions, each at most once and each to an entry other than the
      proof's own pick on it;
    - a result marked `not_started` is a holder-side call, and the calls so
      marked are the last of the holder-side calls in the order the TD
      starts them (picks_v4.call_order_key of the proof's nonce): a TD stops
      starting calls when a budget runs out, never picks which to skip.

    Anything else is refused whole: no proof keeps the questions its head
    did well on, carries answers from anywhere else, or pairs a call the
    challenge does not make. Whether each pick is the one the heads give,
    and whether the holder is the one the epoch names, is the epoch's check
    (v2_epoch._challenge_epoch): a challenge that fails it is not counted."""
    got = [r.question_id for r in proof.scored_results]
    held = proof.holder_results
    if any(r.not_started and not r.holder_pick for r in proof.results):
        return "Only a challenge's holder-side call may be reported not started"
    if proof.scored is None:
        return "A v4 proof says whether it answered a challenge (its `scored` field)"
    if not proof.scored:
        if got or held or proof.challenge is not None:
            return (f"The proof says it answered no challenge but carries {len(got)} scored "
                    f"results, {len(held)} holder-side calls"
                    + (" and a challenge" if proof.challenge is not None else ""))
        return None
    ch = proof.challenge
    if ch is None:
        return "A v4 proof that answers a challenge says which (its `challenge` field)"
    from fugal_subnet.tee.proof import CHALLENGE_SIZES

    if ch.size not in CHALLENGE_SIZES:
        return f"A challenge answers {' or '.join(map(str, CHALLENGE_SIZES))} questions"
    if not v4.challenge:
        return "This epoch's challenge list could not be drawn: the challenge cannot be checked"
    if ch.size > len(v4.challenge):
        return (f"The challenge answers {ch.size} questions; this epoch's challenge list holds "
                f"{len(v4.challenge)}")
    prefix = v4.challenge[:ch.size]
    if slice_ids is not None and not set(slice_ids) <= set(prefix):
        return "The challenge list does not begin with the epoch's slice"
    if len(got) != len(set(got)) or set(got) != set(prefix):
        missing = len(set(prefix) - set(got))
        extra = len(set(got) - set(prefix))
        return (f"The proof answers {len(got)} scored results against the first {ch.size} "
                f"questions of the challenge list ({missing} missing, {extra} not in it, "
                f"{len(got) - len(set(got))} repeated): a challenge is a prefix of the list")
    if ch.holder is None:
        if held:
            return f"The challenge names no holder but carries {len(held)} holder-side calls"
        return None
    own = {r.question_id: r.routed_model for r in proof.scored_results}
    hq = [r.question_id for r in held]
    if len(hq) != len(set(hq)) or not set(hq) <= set(prefix):
        return (f"The challenge's {len(hq)} holder-side calls are not distinct questions of its "
                "prefix")
    same = sorted(r.question_id for r in held if r.routed_model == own[r.question_id])
    if same:
        return (f"{len(same)} holder-side calls go to the proof's own pick (e.g. {same[0]}): the "
                "holder is called only where the two picks differ")
    from fugal_subnet.picks_v4 import call_order_key

    nonce = proof.nonce
    nonce_bytes = bytes.fromhex(nonce) if len(nonce) == 64 else nonce.encode()
    in_order = sorted(held, key=lambda r: call_order_key(nonce_bytes, r.question_id))
    flags = [bool(r.not_started) for r in in_order]
    if any(flags) and flags != sorted(flags):
        return ("The holder-side calls marked not started are not the last ones in the order "
                "the TD starts them")
    return None


def _v4_route_check(proof: BenchmarkProof, v4: V4Expected) -> str | None:
    """Every scored result is routed to an entry of the epoch's menu."""
    ids = set(v4.state.ids())
    off = sorted({r.routed_model for r in proof.scored_results + proof.holder_results
                  if r.routed_model not in ids})
    if off:
        return f"{len(off)} scored routes name an entry not on this epoch's menu (e.g. {off[0]!r})"
    return None


def _v4_exploration_check(proof: BenchmarkProof, v4: V4Expected) -> str | None:
    """The exploration results are exactly the assigned set: the shared set
    and both private parts, each question once, each to its drawn entry
    (section 8)."""
    expected = v4.exploration
    explored = {r.question_id: r.routed_model for r in proof.exploration_results}
    if len(proof.exploration_results) != len(explored):
        return (f"Duplicate exploration results: {len(proof.exploration_results)} entries for "
                f"{len(explored)} questions")
    if set(explored) != set(expected):
        return (f"Exploration set mismatch: {len(expected)} questions assigned, "
                f"{len(explored)} answered")
    wrong = sorted(q for q, m in expected.items() if explored[q] != m)
    if wrong:
        return (f"{len(wrong)} exploration questions routed to an entry other than the one drawn "
                f"(e.g. {wrong[0]})")
    return None


def _v4_head_check(head_bytes: bytes | None, v4: V4Expected) -> str | None:
    """The head is a v4 head (head_eval.load_head_v4: today's format, rows
    keyed by entry id, token means optional and ignored) with a row for an
    entry of the epoch's menu; rows for other ids are never read (section
    11.1). head_eval.v4_head_verdict, the one rule the TD and the miner use."""
    from fugal_subnet.head_eval import v4_head_verdict
    try:
        return v4_head_verdict(head_bytes, v4.state)
    except (KeyError, TypeError) as e:
        return f"Incompatible success head: {e}"


def verify_proof(
    proof: BenchmarkProof,
    approved_measurements: set[str],
    expected_questions_hash: str,
    expected_nonce: str,
    gold_answers: dict[str, dict],
    *,
    expected_question_ids: set[str] | None = None,
    expected_exploration: dict[str, str] | None = None,
    expected_weights_hash: str = "",
    expected_hotkey: str = "",
    expected_proof_hash: str = "",
    head_bytes: bytes | None = None,
    event_log: list | None = None,
    mock: bool = False,
    collateral_budget=None,
    protocol_version: int = 1,
    menu: str | None = None,
    v4: V4Expected | None = None,
    v5: V5Expected | None = None,
) -> VerifyResult:
    """Verify a miner's TEE-attested benchmark proof.

    Structural and hash-chain checks run in every mode — they cost nothing and
    they are what makes a local testnet meaningful. Only the checks that need
    real hardware (the DCAP signature chain and approved-image matching) are
    skipped under `mock`.

    Args:
        proof: The miner's benchmark proof.
        approved_measurements: Approved `measurement_id()` values.
        expected_questions_hash: SHA256 of the expected question IDs.
        expected_nonce: The expected epoch nonce.
        gold_answers: {question_id: task_dict} for the assigned slice.
        expected_question_ids: The exact question set the miner was assigned.
        expected_exploration: {question_id: required_model} for this epoch.
        expected_weights_hash: Head hash committed on-chain before the boundary.
        expected_proof_hash: content_hash the miner advertised over the axon.
        head_bytes: The head artifact shipped in the bundle.
        mock: If True, skip hardware-trust checks (DCAP, measurement).
        protocol_version: the epoch's version (slicer.protocol_version_for_boundary).
            The proof's routing evidence namespace and the head's cost profile
            are checked against that version's (routing_protocol.identity_for,
            benchmark_costs_for(head, version)); the harness stamps the same.
        menu: the epoch's v3 menu (routing_protocol.menu_for_epoch); on menu B
            a head built for menu A or menu B is accepted.
        v4: protocol v4 only, and then required: this hotkey's `V4Expected`
            (`expected_v4`). The namespace, the routes, the exploration and
            the measurement are checked against it; `expected_exploration`,
            if also given, must be its exploration. `gold_answers` must cover
            the exploration and measurement questions too.

    Returns:
        VerifyResult with pass/fail and reason.
    """
    warnings: list[str] = []
    if int(protocol_version) >= 5:
        # Protocol v5: the menu, routes, coverage and head are v4's checks on
        # the coverage-only V4Expected inside `v5`; the exam replaces the
        # challenge (_v5_exam_check).
        if not isinstance(v5, V5Expected) or v4 is not None:
            raise ValueError("verify_proof under protocol v5 needs this hotkey's V5Expected "
                             "(expected_v5) and no V4Expected")
        v4 = v5.v4
        # A v5 proof's questions hash is its own exam cells' questions, not the
        # epoch's slice: it is read from `v5` and from nothing else, so a caller
        # cannot hold every v5 proof to a v4 slice hash (questions_hash_v5).
        expected_questions_hash = questions_hash_v5(v5)
    elif v5 is not None:
        raise ValueError("a V5Expected is for a protocol v5 epoch")
    if int(protocol_version) >= 4:
        # Protocol v4: the namespace is identity_v4 of the epoch's menu hash,
        # so it is read from the epoch's menu and from nothing else.
        if not isinstance(v4, V4Expected):
            raise ValueError("verify_proof under protocol v4 needs this hotkey's V4Expected "
                             "(expected_v4): the epoch's menu decides the namespace")
        if v4.hotkey != (expected_hotkey or proof.hotkey):
            raise ValueError("the V4Expected given was drawn for another hotkey")
        if expected_exploration is not None and dict(expected_exploration) != v4.exploration:
            raise ValueError("expected_exploration differs from the V4Expected's exploration")
        expected_exploration = None          # checked against v4 below, once
        why = _v4_menu_check(proof, v4)
        if why:
            return VerifyResult(False, why)
    else:
        # The namespace the harness stamps (harness: identity_for(version)); a
        # second formatting of it here once refused every v3 proof (2026-11-16).
        from fugal_subnet.routing_protocol import identity_for
        expected_namespace = identity_for(int(protocol_version))
        if proof.routing_protocol != expected_namespace:
            return VerifyResult(
                False,
                f"Incompatible routing protocol evidence namespace for protocol v{int(protocol_version)}",
            )

    # 1. DCAP attestation — proves the quote is genuine Intel-signed hardware.
    #    It does NOT prove the code was unmodified; check 3 does that.
    #
    #    VERIFICATION OF UNTRUSTED INPUT RETURNS A VERDICT, IT DOES NOT THROW.
    #    verify_dcap raises on a quote it cannot parse or whose collateral it
    #    cannot fetch, and those bytes come from a miner. Unguarded, the
    #    exception left verify_proof, left the validator's per-miner loop, and
    #    was caught by the epoch loop's catch-all — abandoning the WHOLE epoch.
    #    Any registered hotkey could therefore halt every validator on the
    #    subnet, every epoch, for the price of one registration, by returning
    #    forty bytes of garbage. That is I6, broken by miner input, and it only
    #    appears under --live, which is why it was never seen.
    #
    #    ImportError is deliberately NOT caught: a missing dcap-qvl is the
    #    operator's misconfiguration, not a miner's doing, and silently
    #    downgrading it to "this proof is invalid" would let a --live validator
    #    reject the entire field while looking like it was working.
    # 0a. WHOSE proof this is. First, because it is the cheapest check that can
    #     reject, it needs no hardware, and a relayed proof should not cost a
    #     DCAP verification to refuse.
    #
    #     Enforced in EVERY mode, like the report_data binding below and for the
    #     same reason: this is a property of the proof's content, not of the
    #     silicon, so a local testnet must catch relay too. Mock mode is the
    #     absence of hardware checks, not a weaker set of rules.
    #
    #     A missing expected_hotkey is a caller error rather than a licence to
    #     skip: verifying without it silently restores the property this exists
    #     to remove. Outside mock that is fatal. Inside mock — used by attack
    #     fixtures and determinism runs that have no metagraph — the check is
    #     simply not requested.
    if not expected_hotkey and not mock:
        raise ValueError(
            "verify_proof called without expected_hotkey outside mock mode. A "
            "proof that is not bound to a miner can be relayed by anyone who "
            "can read it off that miner's axon."
        )
    if expected_hotkey and proof.hotkey != expected_hotkey:
        return VerifyResult(
            False,
            f"Proof is bound to hotkey {proof.hotkey or '(none)'}, but this uid's "
            f"hotkey is {expected_hotkey}. A proof names the miner it was "
            "produced for; this one names somebody else.",
        )

    # 0. Unwrap. Under dstack the miner sends an envelope, not a bare quote, and
    #    the TPM half inside it is verified as part of unwrapping — see
    #    unwrap_attestation for why that is not optional.
    try:
        attestation_quote, embedded_events = unwrap_attestation(
            proof.attestation_quote, extra_certs=proof.tpm_intermediates)
    except ValueError as e:
        return VerifyResult(False, str(e))
    if event_log is None:
        event_log = embedded_events

    # 2. Quote parses, and report_data binds the proof body to the hardware.
    #    Enforced in every mode: the mock quote generator embeds report_data
    #    correctly, so tamper detection works on a local testnet too.
    try:
        quote = parse_quote(attestation_quote)
        report_data = extract_report_data(attestation_quote)
    except ValueError as e:
        return VerifyResult(False, f"Invalid attestation quote: {e}")

    # 2a. A debug TD's host can read and change its memory, so its quote proves
    #     nothing about what ran. Free, so it runs in every mode, before any
    #     network fetch.
    if td_is_debug(quote):
        return VerifyResult(
            False,
            "The quote comes from a DEBUG TD: its host can read and change the "
            "TD's memory, so nothing it attests is evidence",
        )

    content_hash = proof.content_hash()
    expected_report_data = bytes.fromhex(content_hash).ljust(64, b"\x00")[:64]
    if report_data != expected_report_data:
        return VerifyResult(
            False,
            f"report_data mismatch: quote has {report_data.hex()[:32]}..., "
            f"expected {expected_report_data.hex()[:32]}... — the proof body "
            "was altered after it was attested",
        )

    # 4. Nonce — ties the proof to this epoch's unpredictable block hash.
    if proof.nonce != expected_nonce:
        return VerifyResult(
            False,
            f"Nonce mismatch: proof has {proof.nonce[:16]}..., "
            f"expected {expected_nonce[:16]}...",
        )

    # 5. Questions hash.
    if proof.questions_hash != expected_questions_hash:
        return VerifyResult(
            False,
            f"Questions hash mismatch: proof has {proof.questions_hash[:16]}..., "
            f"expected {expected_questions_hash[:16]}...",
        )

    # 6. The answers must be to the questions we actually assigned.
    #    questions_hash alone proves nothing here: it is a public value, so a
    #    miner can copy it while grading an easier set entirely of its own
    #    choosing. Only comparing the result ids against the slice closes that.
    if v5 is not None:
        why = _v5_exam_check(proof, v5)
        if why:
            return VerifyResult(False, why)
    elif v4 is not None:
        # Protocol v4: the scored questions are optional, all or none, and
        # the proof says which (_v4_scored_check).
        why = _v4_scored_check(proof, v4, expected_question_ids)
        if why:
            return VerifyResult(False, why)
    elif expected_question_ids is not None:
        actual_ids = {r.question_id for r in proof.scored_results}
        if actual_ids != expected_question_ids:
            missing = expected_question_ids - actual_ids
            extra = actual_ids - expected_question_ids
            return VerifyResult(
                False,
                f"Result set does not match the assigned slice: "
                f"{len(missing)} missing, {len(extra)} unassigned "
                f"(e.g. {sorted(extra)[:3] or sorted(missing)[:3]})",
            )
        if len(proof.scored_results) != len(expected_question_ids):
            return VerifyResult(
                False,
                f"Duplicate results: {len(proof.scored_results)} entries for "
                f"{len(expected_question_ids)} questions",
            )

    # 6b. The exploration set must be exactly what the nonce assigned.
    #     Exploration costs the miner money and earns it nothing directly, so
    #     the only thing making it happen is that an incomplete or re-targeted
    #     set is a rejected proof. Both sides derive the assignment from public
    #     inputs, so this never relies on the miner's account of it.
    if expected_exploration is not None:
        explored = {r.question_id: r.routed_model for r in proof.exploration_results}
        if len(proof.exploration_results) != len(explored):
            return VerifyResult(
                False,
                f"Duplicate exploration results: {len(proof.exploration_results)} "
                f"entries for {len(explored)} questions",
            )
        if set(explored) != set(expected_exploration):
            return VerifyResult(
                False,
                f"Exploration set mismatch: {len(expected_exploration)} questions "
                f"assigned, {len(explored)} answered",
            )
        wrong = [
            qid for qid, model in expected_exploration.items()
            if explored.get(qid) != model
        ]
        if wrong:
            return VerifyResult(
                False,
                f"{len(wrong)} exploration questions routed to a model other than "
                f"the one the nonce assigned (e.g. {wrong[0]}: got "
                f"{explored.get(wrong[0])!r}, required "
                f"{expected_exploration[wrong[0]]!r})",
            )

    # 6c. Protocol v4: the routes, the exploration and the measurement are
    #     what the epoch's menu and the nonce assign (V4Expected).
    if v4 is not None:
        for check in (_v4_route_check, _v4_exploration_check):
            why = check(proof, v4)
            if why:
                return VerifyResult(False, why)

    # 7. The head that ran must be the head committed on-chain before the nonce
    #    was knowable — otherwise a miner commits one head and runs another,
    #    and the evidence accumulator is keyed on a value it controls freely.
    if expected_weights_hash and proof.weights_hash != expected_weights_hash:
        return VerifyResult(
            False,
            f"Head mismatch: proof ran weights {proof.weights_hash[:16]}..., "
            f"on-chain commitment is {expected_weights_hash[:16]}...",
        )
    if head_bytes is not None:
        actual = hashlib.sha256(head_bytes).hexdigest()
        if actual != proof.weights_hash:
            return VerifyResult(
                False,
                f"Bundled head does not match the attested weights_hash: "
                f"head is {actual[:16]}..., proof claims {proof.weights_hash[:16]}...",
            )

    if not mock and v4 is not None:
        why = _v4_head_check(head_bytes, v4)
        if why:
            return VerifyResult(False, why)
    elif not mock:
        from fugal_subnet.head_eval import load_head_from_npz
        from fugal_subnet.routing_protocol import benchmark_costs_for
        try:
            if head_bytes is None:
                raise ValueError("missing success head bytes")
            benchmark_costs_for(load_head_from_npz(head_bytes), int(protocol_version),
                                menu=menu)
        except (ValueError, KeyError, TypeError) as e:
            return VerifyResult(False, f"Incompatible success head: {e}")

    # 8. The bundle we downloaded must be the one the miner advertised.
    if expected_proof_hash and content_hash != expected_proof_hash:
        return VerifyResult(
            False,
            f"Bundle mismatch: downloaded proof hashes to {content_hash[:16]}..., "
            f"axon advertised {expected_proof_hash[:16]}...",
        )

    # 9. Every graded question must be one we have gold for, or it cannot have
    #    been graded against anything.
    unknown = [r.question_id for r in proof.results if r.question_id not in gold_answers]
    if unknown:
        return VerifyResult(
            False,
            f"{len(unknown)} results reference questions with no gold answer "
            f"(e.g. {unknown[:3]})",
        )

    # 9a. Token counts are bounded (I2). They are attested, so an honest proof
    #     never comes near the bound; a JSON integer is unbounded, and every
    #     value used downstream (the frame, the v2 spend) must be bounded.
    from fugal_subnet.config import PROOF_MAX_TOKENS
    bad_tokens = [
        r.question_id for r in proof.results
        if not (0 <= r.prompt_tokens <= PROOF_MAX_TOKENS
                and 0 <= r.completion_tokens <= PROOF_MAX_TOKENS)
    ]
    if bad_tokens:
        return VerifyResult(
            False,
            f"{len(bad_tokens)} results carry a token count outside 0..{PROOF_MAX_TOKENS} "
            f"(e.g. {bad_tokens[:3]})",
        )

    # 9b. A failed call is a wrong answer and nothing more. The harness
    #     records `failed` for a call that did not complete, and the frame
    #     drops such results rather than reading them as the model being
    #     wrong (diligence 2026-10, finding 2). That makes the flag worth
    #     lying about in both directions, so it is held to its meaning: a
    #     failed result cannot be correct and cannot carry a reply.
    empty_reply = hashlib.sha256(b"").hexdigest()
    bad_failed = [
        r.question_id for r in proof.results
        if r.failed and (r.correct or r.response_hash != empty_reply)
    ]
    if bad_failed:
        return VerifyResult(
            False,
            f"{len(bad_failed)} results are marked failed but carry a grade or "
            f"a reply (e.g. {bad_failed[:3]})",
        )

    # 9c. Failed calls are free of consequence for the frame, so failing
    #     exploration must not be free for the miner. A miner controls its
    #     own network and exploration is 5% of its bill; a proof whose
    #     exploration failed far more often than its scored pass is the
    #     signature of blocking that phase, and it is rejected whole. An
    #     outage that takes both down equally passes this -- it is the
    #     DIFFERENCE that is judged -- and is then merely a bad epoch.
    #     Under v4 the bound is absolute and taken across the epoch's proofs
    #     (picks_v4.failure_refusals), because a v4 proof need not carry a
    #     scored pass to compare with.
    n_explore = len(proof.exploration_results)
    n_scored = len(proof.scored_results)
    if n_explore and n_scored and v4 is None:
        explore_fail = sum(1 for r in proof.exploration_results if r.failed) / n_explore
        scored_fail = sum(1 for r in proof.scored_results if r.failed) / n_scored
        if explore_fail - scored_fail > EXPLORE_FAIL_MARGIN:
            return VerifyResult(
                False,
                f"Exploration failure rate {explore_fail:.0%} exceeds the scored "
                f"failure rate {scored_fail:.0%} by more than "
                f"{EXPLORE_FAIL_MARGIN:.0%}: exploration was blocked, not run",
            )

    # 10. Costs must be internally consistent. Under real attestation this can
    #     never legitimately fail — the metering proxy produced every figure
    #     inside the same measured image — so an inconsistency means the proof
    #     is not what it claims to be, not that a number drifted.
    per_q_sum = sum(r.cost_usd for r in proof.results)
    if not _costs_consistent(per_q_sum, proof.total_cost_usd):
        return VerifyResult(
            False,
            f"Cost inconsistency: per-question sum ${per_q_sum:.4f} vs "
            f"attested total ${proof.total_cost_usd:.4f}",
        )

    per_model_sum = sum(proof.per_model_costs.values())
    if not _costs_consistent(per_model_sum, proof.total_cost_usd):
        return VerifyResult(
            False,
            f"Cost inconsistency: per-model sum ${per_model_sum:.4f} vs "
            f"attested total ${proof.total_cost_usd:.4f}",
        )

    # 11. ONLY NOW the hardware-trust checks, because everything above was free
    #     and local, and these are not: DCAP fetches collateral over the network
    #     against a per-epoch budget shared by every miner (I4, I6), and the
    #     measurement match follows it. Running them first meant a replayed or
    #     stale proof — wrong nonce, wrong report_data, wrong advertised hash —
    #     cost every validator a fetch whose FMSPC target was still unsigned at
    #     parse time, for a proof that ten lines of comparison would have
    #     refused (security read 2026-11-02, finding 5). A fetch now only ever
    #     follows a fresh quote over a fresh content hash that names this
    #     miner, this epoch and this slice. tests/test_verify_order.py counts
    #     the fetches.
    if not mock:
        try:
            dcap_ok = verify_dcap(
                attestation_quote, budget=collateral_budget)
        except ImportError:
            raise
        except CollateralUnavailable as e:
            # The operator's infrastructure, not the miner's proof. Sits here
            # beside the ImportError re-raise because it is the same category:
            # in both, nothing was learned about this miner, and reporting
            # "invalid" would be an accusation the evidence does not support.
            #
            # It does NOT return valid=True. Nothing was verified, so nothing is
            # accepted — this only changes what the failure is called, and what
            # the caller is permitted to conclude from it.
            return VerifyResult(
                False,
                f"DCAP collateral unavailable, so this proof was never judged: {e}",
                unverifiable=True,
            )
        except Exception as e:  # noqa: BLE001 - miner-controlled bytes
            return VerifyResult(
                False, f"DCAP attestation verification failed: {type(e).__name__}: {e}",
            )
        if not dcap_ok:
            return VerifyResult(False, "DCAP attestation verification failed")

    # 3. Approved runtime image, from the hardware's own measurement registers.
    if not mock:
        approved = parse_approved(approved_measurements)
        measured = measurement_id(quote)
        if measured not in approved:
            return VerifyResult(
                False,
                f"Unapproved runtime image: measurement {measured[:16]}... "
                f"not among {len(approved)} approved measurements",
            )

        # 3b. The application, from the compose hash in the RTMR3 event log.
        #     RTMR3 is an EXTEND, not a set, and a dstack TD extends it nine
        #     times during boot, including `instance-id`, which changes on
        #     every deploy. So the register is never compared to a fixed value:
        #     the log is replayed from zero, the result must equal what the CPU
        #     signed, and only then is the compose-hash payload read. A log that
        #     does not reproduce the register is discarded whole.
        if not approved[measured]:
            # A bare base says nothing about the application, and the
            # application is what fixes the model upstream. Outside mock that
            # is the upstream-substitution exploit with every signature intact
            # (diligence 2026-10, finding 1), so it is a rejection and not a
            # warning. The live validator already refuses to START with such an
            # entry; this is the same rule at the point of use, for scripts
            # that call verify_proof directly.
            return VerifyResult(
                False,
                f"Approved entry for image {measured[:16]}... is a bare base: no "
                "app identity, so the application (and its model upstream) is "
                "unchecked. Live verification requires <base>:<compose_hash>.",
            )
        # Only a compose hash is checkable: it is the one app identity the
        # replayed log carries. Any other app half approves nothing.
        required_apps = {a for a in approved[measured] if _HEX64.fullmatch(a)}
        if not required_apps:
            return VerifyResult(
                False,
                f"Approved entry for image {measured[:16]}... names no compose "
                "hash. Live verification requires <base>:<compose_hash>.",
            )
        if not event_log:
            return VerifyResult(
                False,
                "The proof carries no TDX event log. RTMR3 on a measured image is "
                "a chain that includes per-deploy values, so the application can "
                "only be checked by replaying the log, never by comparison.",
            )
        try:
            replayed, events = replay_event_log(event_log, imr=3)
        except ValueError as e:
            return VerifyResult(False, f"Malformed TDX event log: {e}")
        if replayed != quote.rtmr3:
            return VerifyResult(
                False,
                f"Event log does not reproduce RTMR3: replayed "
                f"{replayed[:16]}... but the quote says "
                f"{quote.rtmr3[:16]}...",
            )
        app_seen = events.get("compose-hash", b"").hex()
        if app_seen not in required_apps:
            return VerifyResult(
                False,
                f"Unapproved application: compose-hash {app_seen[:16]}... "
                f"not among {len(required_apps)} approved for this image",
            )

    return VerifyResult(
        valid=True,
        reason="All checks passed",
        warnings=warnings if warnings else None,
    )


def _costs_consistent(a: float, b: float) -> bool:
    """Check cost consistency using both relative (5%) and absolute ($0.001) tolerance."""
    diff = abs(a - b)
    return diff <= max(0.05 * max(abs(a), abs(b)), 0.001)
