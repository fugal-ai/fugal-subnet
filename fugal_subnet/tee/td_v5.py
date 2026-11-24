"""Protocol v5, the TD side (~/projects/fugal-v2-handoff/v5/DESIGN.md sections 5 and 7).

What a miner's TD does in a v5 epoch. v5 keeps v4's menu machinery (the
live menu, read and checked by `td_v4.V4Session` exactly as under v4) and
replaces its challenge and its random exploration with two kinds of calls:

1. **Exam cells.** The root publishes epoch E's exam spec (`exam_v5.ExamSpec`)
   in its reveal of E - SPEC_LAG. At E's boundary the TD reads the newest
   root reveal named by the root's on-chain commitment, checks its bytes
   against the committed sha256 and its epoch, parses the spec, checks the
   spec's own hash, and calls exactly the (question, entry) cells the spec
   assigns to this hotkey (`exam_v5.assigned_cells`). The TD makes no picks
   and needs no head but its own (which the proof binds, as before): the
   cells are the union of every contender's differing picks, which every
   validator recomputes from the same published inputs.
2. **Coverage cells.** COVERAGE_CALLS cells drawn for this hotkey as v4
   draws its coverage part (`coverage_v5`): the training labels that let a
   new or unpicked entry be learned at all.

The first SPEC_LAG epochs from the activation are coverage only: no reveal
before them can carry a spec. An epoch whose spec cannot be read or checked
is coverage only too (logged), never an exam of cells the TD could not
verify.

Timing: exam and coverage calls are started in that order and none is
started more than EXAM_BUDGET_S after the first call (`not_started`), and
each call is held to at most EXAM_CALL_DEADLINE_S, so the proof is ready
before the validators' collection at boundary + EPOCH_COLLECT_FRACTION of
the epoch whatever an entry's host does.

v4's challenge is not part of v5: `gate_challenge` makes a v4 challenge
policy (the miner's --v4-challenge) answer nothing from the v5 activation,
with a log line.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

from fugal_subnet import config
from fugal_subnet.reveal_sync import MAX_REVEAL_BYTES

logger = logging.getLogger(__name__)

# The v5 constants this module reads (fugal_subnet/config.py, the v5 block).
SPEC_LAG = int(config.SPEC_LAG_V5)
COVERAGE_CALLS = int(config.COVERAGE_CALLS_V5)
Y_SCALE = int(config.Y_SCALE_V5)
# No exam or coverage call is started later than this after the epoch's first
# call, and none waits longer than EXAM_CALL_DEADLINE_S: the last reply is in
# by about 30 minutes after the first call, inside the 36-minute window to the
# validators' collection (boundary + 0.5 of a 360-block epoch).
EXAM_BUDGET_S = float(config.EXAM_BUDGET_S_V5)
EXAM_CALL_DEADLINE_S = int(config.EXAM_CALL_DEADLINE_S_V5)

# Where the root's reveal of E - SPEC_LAG carries epoch E's spec:
# {"epoch": E, "spec": <ExamSpec.to_json()>, "spec_hash": <ExamSpec.spec_hash()>}.
REVEAL_SPEC_KEY = "exam_spec_v5"
MAX_SPEC_REVEAL_BYTES = MAX_REVEAL_BYTES

# The call kinds of a v5 epoch, in the order they are started.
EXAM_KINDS = ("exam", "coverage")


def _exam_v5():
    """The exam module (fugal_subnet/exam_v5.py: ExamSpec, assigned_cells)."""
    from fugal_subnet import exam_v5
    return exam_v5


def _hex64(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def exam_spec_from_reveal(raw: bytes, epoch_id: str, sha256: str):
    """The exam spec a root reveal carries (`REVEAL_SPEC_KEY`), or None when
    it carries none. The bytes must hash to the `sha256` the root committed
    and be the reveal of `epoch_id`, and the spec's recomputed hash must be
    the one the reveal names; otherwise ValueError."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_SPEC_REVEAL_BYTES:
        raise ValueError("not a reveal's bytes")
    if hashlib.sha256(bytes(raw)).hexdigest() != sha256:
        raise ValueError("the bytes are not the reveal the root committed")
    doc = json.loads(bytes(raw))
    if not isinstance(doc, dict) or doc.get("epoch_id") != epoch_id:
        raise ValueError(f"the bytes are not the reveal of {epoch_id}")
    field = doc.get(REVEAL_SPEC_KEY)
    if field is None:
        return None
    if (not isinstance(field, dict) or set(field) != {"epoch", "spec", "spec_hash"}
            or type(field["epoch"]) is not int or not isinstance(field["spec"], str)
            or not _hex64(field["spec_hash"])):
        raise ValueError(f"the reveal's {REVEAL_SPEC_KEY} is not an exam spec record")
    spec = _exam_v5().ExamSpec.from_json(field["spec"])
    if spec.spec_hash() != field["spec_hash"]:
        raise ValueError("the reveal's exam spec does not hash to the hash it names")
    if int(spec.epoch) != field["epoch"]:
        raise ValueError("the reveal's exam spec is not for the epoch it names")
    return spec


@dataclass(frozen=True)
class ExamTask:
    """Epoch `epoch`'s exam as this TD answers it: the root's spec, its hash
    (bound into the proof) and this hotkey's assigned cells, (question id,
    entry id), sorted."""

    epoch: int
    spec: object
    spec_hash: str
    cells: tuple[tuple[str, str], ...]


class ExamPolicy:
    """This TD's exam in epoch E, or None for a coverage-only epoch.

    `read_spec(E)` returns (k, spec or None) from the newest root reveal
    verified at E's boundary (`exam_spec_from_reveal`), k its epoch; None
    when the root has no commitment; it raises when the chain or the reveal
    cannot be read. The epoch's spec is the one in the reveal of
    E - SPEC_LAG: any other newest reveal means no exam this epoch."""

    def __init__(self, *, first_epoch: int, hotkey: str,
                 read_spec: Callable[[int], tuple[int, object | None] | None]):
        if not hotkey:
            raise ValueError("an exam is answered for a hotkey")
        self.first_epoch = int(first_epoch)
        self.hotkey = hotkey
        self.read_spec = read_spec

    def __call__(self, epoch: int) -> ExamTask | None:
        epoch = int(epoch)
        if epoch < self.first_epoch:
            raise ValueError(f"epoch {epoch} is before the first v5 epoch {self.first_epoch}")
        if epoch < self.first_epoch + SPEC_LAG:
            logger.info("Epoch %d: the first %d v5 epochs are coverage only (no reveal before "
                        "them carries an exam spec)", epoch, SPEC_LAG)
            return None
        want = epoch - SPEC_LAG
        try:
            got = self.read_spec(epoch)
        except Exception as e:  # noqa: BLE001 - coverage only this epoch; logged
            logger.warning("Epoch %d: the exam spec could not be read (%s): coverage only this "
                           "epoch", epoch, type(e).__name__)
            return None
        if got is None or int(got[0]) != want:
            logger.warning("Epoch %d: the root's newest reveal at the boundary is not of epoch "
                           "%d: no exam spec, coverage only this epoch", epoch, want)
            return None
        spec = got[1]
        if spec is None:
            logger.warning("Epoch %d: the root's reveal of epoch %d carries no exam spec: "
                           "coverage only this epoch", epoch, want)
            return None
        if int(spec.epoch) != epoch:
            logger.warning("Epoch %d: the exam spec in the reveal of epoch %d is for epoch %d: "
                           "coverage only this epoch", epoch, want, int(spec.epoch))
            return None
        cells = tuple(sorted((str(q), str(e))
                             for q, e in _exam_v5().assigned_cells(spec, self.hotkey)))
        return ExamTask(epoch=epoch, spec=spec, spec_hash=spec.spec_hash(), cells=cells)


def gate_challenge(policy, first_v5_epoch: int):
    """A v4 challenge policy (td_v4.ChallengePolicy, or None) that answers
    nothing from the v5 activation on: v5 has no challenge. The operator's
    --v4-challenge options become no-ops there, said once in the log."""
    first = int(first_v5_epoch)
    said = [False]

    def gated(epoch: int):
        if int(epoch) >= first:
            if policy is not None and getattr(policy, "size", 0) and not said[0]:
                said[0] = True
                logger.info("Protocol v5 from epoch %d: there is no challenge; the --v4-challenge "
                            "options are not used", first)
            return None
        return None if policy is None else policy(epoch)
    return gated


def coverage_v5(nonce: bytes, pool: list[dict], excluded_ids: set[str], state, hotkey: str,
                index=None) -> dict[str, str]:
    """This hotkey's coverage cells, {question id: entry id}: COVERAGE_CALLS
    questions drawn as v4 draws a miner's private set (the epoch's fresh
    questions first when `index` is the epoch's memo_v4.ReserveIndex, then
    slicer.select_private_explore), disjoint from `excluded_ids` (the exam's
    questions), every one sent to an entry by the menu's coverage shares
    (epoch_plan.explore_shares_v4(state).coverage, explore_weights.
    assign_by_share under exploration.coverage_nonce). Every validator draws
    the same.

    TODO(coordinator): the category targets are v4's pinned mix until lane P's
    RouterArena mix replaces them."""
    from fugal_subnet import epoch_plan, memo_v4
    from fugal_subnet.benchmarks.slicer import select_private_explore
    from fugal_subnet.exploration import coverage_nonce, private_nonce
    from fugal_subnet.explore_weights import assign_by_share

    n = COVERAGE_CALLS
    if not hotkey or n <= 0:
        return {}
    shares = epoch_plan.explore_shares_v4(state)
    targets, _bench_category, _bench_weight = epoch_plan._mix_v4()
    mine = private_nonce(nonce, hotkey)
    index = index or epoch_plan.pool_index_v4(pool)
    excluded = set(excluded_ids)
    fresh: list[dict] = []
    if isinstance(index, memo_v4.ReserveIndex):
        fresh = [q for q in memo_v4.fresh_questions(nonce, index, targets)
                 if q["question_id"] not in excluded][:n]
        excluded |= {q["question_id"] for q in fresh}
    chosen = fresh + select_private_explore(mine, index, excluded, n - len(fresh), dict(targets))
    if not chosen:
        return {}
    return assign_by_share(coverage_nonce(mine), chosen, shares.coverage)


def grade_y(question: dict, text: str) -> int:
    """One reply's label, an integer 0..Y_SCALE (RouterArena's metrics may be
    fractional). Through lane P's arena grader (fugal_subnet/graders_arena.py,
    `grade(question, text)`) when the image ships it; a bool counts as 0 or
    Y_SCALE, a Fraction in [0, 1] is floored to the scale, an int is taken
    as already on the scale. Floats are refused (labels are consensus data).

    TODO(lane P / coordinator): drop the fallback to the v4 grader once
    graders_arena.py is integrated."""
    grade = None
    try:
        from fugal_subnet import graders_arena  # type: ignore[attr-defined]
        grade = getattr(graders_arena, "grade", None)
    except ImportError:
        grade = None
    if grade is None:
        from fugal_subnet.tee.harness import grade_question
        return Y_SCALE if grade_question(question, text) else 0
    v = grade(question, text)
    if isinstance(v, bool):
        return Y_SCALE if v else 0
    if isinstance(v, int):
        y = v
    elif isinstance(v, Fraction):
        if not 0 <= v <= 1:
            raise ValueError("an arena grade is in [0, 1]")
        y = (v.numerator * Y_SCALE) // v.denominator
    else:
        raise ValueError(f"an arena grade is a bool, an int on the scale or a Fraction, not "
                         f"{type(v).__name__}")
    if not 0 <= y <= Y_SCALE:
        raise ValueError(f"an arena grade is on 0..{Y_SCALE}")
    return y


# --- the epoch's inputs, as the harness takes them ------------------------------------------


@dataclass(frozen=True)
class V5EpochInputs:
    """Everything a v5 epoch needs: the v4 inputs that give its menu (never
    with a challenge) and its exam (None: coverage only)."""

    epoch: int
    v4: object                       # td_v4.V4EpochInputs, challenge None
    exam: ExamTask | None
    budget_s: float = EXAM_BUDGET_S
    call_deadline_s: int = EXAM_CALL_DEADLINE_S

    def menu(self):
        return self.v4.menu()


class V5Session:
    """A miner's v5 state across epochs: its v4 session (the menu it last
    accepted, read and checked as under v4) and its exam policy."""

    def __init__(self, *, v4_session, exam: ExamPolicy, first_epoch: int,
                 protocol_of=None, blocks_per_epoch: int | None = None):
        self.v4 = v4_session
        self.exam = exam
        self.first_epoch = int(first_epoch)
        # `protocol_of(boundary_block) -> int` is the one source of an epoch's
        # protocol, slicer.protocol_version_for_boundary (the miner binds it
        # to the chain's activation blocks); without it, the first v5 epoch.
        self.protocol_of = protocol_of
        self.blocks_per_epoch = None if blocks_per_epoch is None else int(blocks_per_epoch)

    def is_v5(self, epoch: int) -> bool:
        if self.protocol_of is not None and self.blocks_per_epoch:
            return int(self.protocol_of(int(epoch) * self.blocks_per_epoch)) >= 5
        return int(epoch) >= self.first_epoch

    def inputs_for(self, epoch: int) -> V5EpochInputs:
        """Epoch E's inputs: the v4 session's (MenuUnavailable or
        MenuRejected propagate: no proof), with no challenge, and the exam
        policy's task. Raises before the first v5 epoch."""
        epoch = int(epoch)
        if not self.is_v5(epoch):
            raise ValueError(f"epoch {epoch} is before the first v5 epoch {self.first_epoch}")
        v4_inputs = self.v4.inputs_for(epoch)
        if getattr(v4_inputs, "challenge", None) is not None:
            v4_inputs = dataclasses.replace(v4_inputs, challenge=None)
        return V5EpochInputs(epoch=epoch, v4=v4_inputs, exam=self.exam(epoch))

    def ran(self, inputs: V5EpochInputs, proof) -> object:
        """Record the menu `proof` ran on (td_v4.V4Session.ran)."""
        return self.v4.ran(inputs.v4, proof)
