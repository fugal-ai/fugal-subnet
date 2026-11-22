"""The menu as published state (protocol v4; docs/decisions/LIVE_MENU_V4.md, section 4).

WHAT. From protocol v4 the menu is not a file in the image. It is a value,
`MenuState`, that changes by one pure step an epoch and is named by its hash
in every proof's namespace (`routing_protocol.identity_v4`). The image carries
the rules and a pinned seed (`routing_data/menu_v4_seed.json`), the state of
the first v4 epoch.

This module is the data model, its one encoding, and the three functions that
decide a menu:

    step(MenuState(k + lag - 1), inputs(k))  ->  MenuState(k + lag)
    menu_for_epoch(MenuState(E - 1), the hash committed for E)  ->  base(E)
    apply_joins(base(E), the entries a join delta admits for E)  ->  MenuState(E)

All three are pure: no clock, no network, no chain, no file. What produces
the inputs (the TD's endpoint reads, the validator's votes over verified
proofs, the chain read of the commitments, the root's join delta checked in
`chain_proposals`) lives elsewhere and hands its results in as the plain
typed values defined below.

A NEW ENTRY'S LIFE (FAST_CROWN_V4.md section 18). A miner files a proposal
by an on-chain commitment; the root admits it into a join delta read at a
block just before a boundary, and at that boundary it joins the menu as a
PROVISIONAL entry (`Entry.scored_from == PROVISIONAL`, a `Provisional`
record beside it). In its join epoch every box's coverage exploration is
aimed at it, and its labels score for no head. The ordinary step that scores
its join epoch applies the gates to the field's labels: it becomes a full
entry (`scored_from` the epoch the step's menu applies to; a head with no row
for it is then scored as odds 0 on its labels, and the displacement shield
of ENTRY_SHIELD_EPOCHS starts), or it retires with the gate's reason, the
cooldown on its id and on its proposer's coldkey.

THE RULES ARE APPLIED HERE, NOT HANDED IN. Which entry or proposal the
compiled rules refuse (`menu_rules`) is never an input: the step is given the
signed list a node holds and what proofs carried, and calls the rules itself.

EXACT. Everything in a state is an integer or a string (two fixed booleans
aside: `allow_fallbacks`, always false, and a `reasoning` of `enabled`).
Prices are integer picodollars per token, token means integer milli-tokens,
rates integer parts per million or pairs of integers. Arithmetic is on
integers and Fractions and every rounding is half up, so a state and its hash
are the same bytes on any host (I1).

ONE ENCODING. `canonical_json` is sorted keys, no whitespace, ASCII;
`menu_hash` is its sha256. `from_json` accepts only those bytes, so one hash
names one state.

Nothing here is reachable from protocol v1-v3.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from fugal_subnet import menu_rules

PICO = 10**12                # picodollars in a dollar: a price is picodollars per token
MILLI = 1000                  # milli-tokens in a token: a token mean is milli-tokens
PPM = 1_000_000               # a rate of 1.0 in parts per million
BP = 10_000                   # a rate of 1.0 in basis points
MAX_INT = 2**53 - 1           # every integer in a state: exact in any JSON reader
MAX_STRING = 256
SEED_PATH = Path(__file__).resolve().parent / "routing_data" / "menu_v4_seed.json"

# The design's constants (docs/decisions/LIVE_MENU_V4.md, sections 3-9), each
# defined here once. They are compiled, change by release only, and are the
# defaults of `MenuParams`; none is read from the environment. The constants
# of the entry rules are menu_rules' and are not restated here: the answer cap
# (menu_rules.MAX_ANSWER_TOKENS) reaches the step through
# menu_rules.answer_limit, the one place an answer is limited.
MENU_LAG = 2                       # a menu derived while scoring epoch k applies from k + 2
MENU_CAP = 64                      # entries on the menu, at most
# Entries on the menu, at least. The epoch's exploration draw and every head's
# picks need a menu that is not empty, and a menu this small is already a
# subnet that needs a release, not a step: the seed has 59. Eight is the
# pipeline's own width (MENU_MAX_PROVISIONAL), so the proposals in flight can
# restock a menu at the floor, and with health retirements at most
# max(2, a tenth of the menu) a step (`health_retirement_bound`) the menu
# reaches it only over many steps. The minimum counts FULL entries: the gates
# may retire every provisional entry in one step, whatever else happens, so a
# provisional entry holds no place under the floor. The reader refuses a menu
# with fewer full entries; the step never retires a full entry below it (for
# health, the price ceiling or displacement: the rest wait), and keeps the
# list in force rather than adopt a newer one that would leave fewer.
MENU_MIN_ENTRIES = 8
MENU_MAX_PROVISIONAL = 8           # provisional entries on one menu, at most
# Provisional entries that join at one boundary, at most: MENU_JOINS_SMALL_FIELD
# while fewer than MENU_JOINS_FIELD_HOTKEYS hotkeys served the epoch the root's
# reveal names, else MENU_MAX_JOINS_PER_EPOCH (FAST_CROWN_V4.md 18(c): the
# field's coverage measures one new entry an epoch at five operators, about 290
# labels, and two at fifteen).
MENU_MAX_JOINS_PER_EPOCH = 2
MENU_JOINS_SMALL_FIELD = 1
MENU_JOINS_FIELD_HOTKEYS = 10
# `scored_from` of a provisional entry: no label of it is scored for any head
# until the gates make it full and the step sets `scored_from` to the epoch
# its menu applies to. The largest integer a state holds, so every label's
# epoch is before it.
PROVISIONAL = 2**53 - 1
# A full entry is not displaced for this many epochs from `scored_from`
# (FAST_CROWN_V4.md section 0: a separate 20-epoch shield), and a provisional
# entry is never displaced.
ENTRY_SHIELD_EPOCHS = 20
# An entry's measured rate is in odds units: head_odds.ODDS_SCALE (a test
# holds the two equal; menu_state does not import the head code).
RATE_SCALE = 10_000
ENDPOINT_MIN_VOTERS = 3            # coldkeys a voted endpoint read needs to move anything
TOKEN_MIN_PAIRS = 30               # pairs an entry's token means need to be replaced
# Prices, answer limits, token means and exploration-pair counts move in the
# published state only in a step whose new epoch is a multiple of this (about
# a day), except on the entry that joins or is updated in that very step. A
# structural change elsewhere does not license a refresh. The menu hash then
# changes rarely, so a miner's box needs new menu bytes rarely and no epoch
# depends on fetching them just to keep running.
MENU_REFRESH_EPOCHS = 20
# A refreshed price moves by at most this factor, up or down, in one refresh
# (an entry's, a proposal's): what one published state can do to a price is
# bounded, and a real larger move arrives over consecutive refreshes (x16
# in two, x64 in three, about a day apart).
MENU_PRICE_STEP = 4
# The gates a provisional entry meets on the field's labels (FAST_CROWN_V4.md
# 18(c)-(d)), decided by the step that scores its join epoch:
# - answered: judged per coldkey, on the median coldkey's share of its
#   exploration calls to the entry that were answered (each failed call is
#   retried once inside the TD), so one operator failing its own calls moves
#   nothing;
# - category: some category with at least PROPOSAL_MIN_CATEGORY_GRADED labels
#   at the accuracy floor (MenuParams.min_category_accuracy_bp);
# - useful: no entry with at least PROPOSAL_USEFUL_MIN_PAIRS labels that costs
#   no more a call is clearly better (its lower bound above the entry's upper);
# - cost: its measured call costs something at its price.
PROPOSAL_MIN_ANSWERED_BP = 9500    # the median coldkey's answered share, at least
PROPOSAL_MIN_CATEGORY_GRADED = 20  # labels the category gate needs
PROPOSAL_USEFUL_MIN_PAIRS = 100    # labels an entry needs before a new entry is compared with it
# - sibling: one model takes one place per setting only when the settings
#   differ. A provisional entry fails when a full entry of the same model
#   (another reasoning setting) with PROPOSAL_USEFUL_MIN_PAIRS labels has an
#   accuracy inside the new entry's 95% interval widened by
#   SIBLING_MARGIN_PPM on each side, and the new entry is not cheaper a call
#   by more than SIBLING_CHEAPER_BP. Two settings that differ by more than
#   that in accuracy, or in cost, both stay.
SIBLING_MARGIN_PPM = 20_000        # two points of accuracy
SIBLING_CHEAPER_BP = 2000          # a call 20% cheaper is a different offer
PROPOSAL_COOLDOWN_EPOCHS = 48      # epochs before a retired or refused id, or the coldkey
#                                    whose proposal failed a gate, may join again
# Health retirement (section 9.2, the rule W0's replay fixed). The aggregation
# over verified proofs (menu_inputs.health_stats) reads the first five; the
# step reads RETIRE_FAIL_MIN_COLDKEYS (MenuParams.retire_fail_min_coldkeys).
RETIRE_WINDOW_EPOCHS = 12          # counted epochs (outages left out) a coldkey's calls are summed over
RETIRE_MIN_CALLS = 5               # exploration calls to the entry that make a coldkey eligible on it
RETIRE_FAIL_RATE_BP = 3000         # an eligible coldkey fails on the entry above this failure rate...
RETIRE_FAIL_MIN_EPOCHS = 3         # ...with its failed calls in at least this many distinct epochs
RETIRE_FAIL_MIN_COLDKEYS = 4       # eligible coldkeys an entry needs before it can retire as failing;
#                                    it retires when more than half of them are failing on it
RETIRE_OUTAGE_RATE_BP = 3000       # an epoch whose median coldkey failed above this share of all its
#                                    exploration calls is an outage and is not counted
RETIRE_GONE_EPOCHS = 6             # consecutive epochs with no pinned host listed
# In a small field the coldkey minimums above (RETIRE_FAIL_MIN_COLDKEYS for
# failing, ENDPOINT_MIN_VOTERS for each unlisted vote of gone) could never be
# met: an entry nobody can call would stay. They follow the serving field
# instead (`retire_minimum`): a majority of the coldkeys serving, never more
# than the compiled minimum and never fewer than RETIRE_MIN_FLOOR, so one
# operator, however many hotkeys it runs, never retires an entry alone.
RETIRE_MIN_FLOOR = 2
RETIRE_USE_FLOOR_BP = 100          # an entry used more than this share is never displaced
RETIRED_KEEP = 256                 # `retired` holds this many records: every cooldown still
#                                    running, then its newest records of cooldowns that ran out
# The most proposers' coldkeys in cooldown a menu holds (`coldkey_cooldown_cap`):
# a coldkey's cooldown is written by a failed gate, at most MENU_MAX_PROVISIONAL
# a step (every gate is of a provisional entry), and it stays while it can
# still refuse a join, MENU_LAG + PROPOSAL_COOLDOWN_EPOCHS - 1 steps. So
# MENU_MAX_PROVISIONAL x (PROPOSAL_COOLDOWN_EPOCHS + MENU_LAG) = 400 is never
# reached by an honest step; the reader refuses a menu above it.
# The largest committed menu document a node reads, in bytes; larger bytes are
# refused before they are parsed. The largest document the caps allow, with
# every string at MAX_STRING and every integer at 2^53 - 1 (64 entries each
# with a pick-cost table of PICK_MAX_CATEGORIES categories, eight provisional
# records, RETIRED_KEEP retired records and the coldkey
# cap's 400 coldkeys), is about 0.43 MB, and the seed about 52 kB
# (tests/test_menu_reader_checks.py holds both under fractions of this limit),
# so an honest root never comes near it, while a TD parses at most this much
# from the root. Every list in a menu is bounded by a compiled cap the reader
# checks (`check_caps`), so no committed menu can grow the next honest step's
# document past this limit.
MENU_MAX_BYTES = 1_048_576
# The pick-cost table (docs/decisions/SEAT_MINIMAL_V4.md 3.4 and section 8,
# PICK_COST_PRIOR_CELLS; the earlier seat design's 8.2). An entry's pick cost
# is the token means of its graded scored-question cells shrunk toward its
# call cost as if this many more cells had cost exactly the call cost, and its
# pick cost in one category is that category's cells shrunk toward the
# entry's pick cost by the second. An entry with no graded cell is priced at
# its call cost, a thin category near its entry's pick cost, and many cells
# at their own mean.
PICK_COST_PRIOR_CELLS_ENTRY = 30
PICK_COST_PRIOR_CELLS_CATEGORY = 20
# A pick-cost table lists at most this many categories, each named in at
# most this many characters: the v4 mix's seven (epoch_plan.mix_v4, the
# categories a scored question has) and room for one more. Like every list
# in a menu it is capped, so the largest document stays under MENU_MAX_BYTES.
PICK_MAX_CATEGORIES = 8
PICK_CATEGORY_MAX_CHARS = 32

# Why an entry or a proposal left (the `reason` of a `retired` record).
REASON_FAILING = "failing"                # health: too many of its calls fail
REASON_GONE = "gone"                      # health: none of its pinned hosts is listed
REASON_LAB_REMOVED = "lab_removed"        # its lab left the signed list
REASON_HOST_REMOVED = "host_removed"      # one of its pinned hosts left the signed list
REASON_DISPLACED = "displaced"            # the least-used entry of a full menu made room
REASON_GATE_ANSWERED = "gate_answered"    # the median coldkey answered too few of its calls
REASON_GATE_CATEGORY = "gate_category"    # no category at the accuracy floor
REASON_GATE_USEFUL = "gate_useful"        # dearer than, and clearly worse than, an entry
REASON_GATE_COST = "gate_cost"            # its measured call costs nothing at its price
REASON_GATE_SIBLING = "gate_sibling"      # another setting of its model already on the menu
#                                           does as well at no greater cost
REASON_PRICE_CEILING = "price_ceiling"    # its refreshed price would be above the ceiling
#                                           (menu_rules.check_price)
REASONS = frozenset({
    REASON_FAILING, REASON_GONE, REASON_LAB_REMOVED, REASON_HOST_REMOVED, REASON_DISPLACED,
    REASON_GATE_ANSWERED, REASON_GATE_CATEGORY, REASON_GATE_USEFUL, REASON_GATE_COST,
    REASON_GATE_SIBLING, REASON_PRICE_CEILING})
# The reasons that are health retirements (section 9.2), bounded per step.
HEALTH_REASONS = frozenset({REASON_FAILING, REASON_GONE})
# The reasons an entry on the menu leaves for, and those a proposal fails a gate for.
_ENTRY_REASONS = HEALTH_REASONS | {REASON_LAB_REMOVED, REASON_HOST_REMOVED, REASON_DISPLACED,
                                   REASON_PRICE_CEILING}
_GATE_REASONS = frozenset({REASON_GATE_ANSWERED, REASON_GATE_CATEGORY, REASON_GATE_USEFUL,
                           REASON_GATE_COST, REASON_GATE_SIBLING})
# The rule an entry fails under a newer signed list, and the reason it then
# retires for: its lab left the list, or one of its pinned hosts did (a host
# that lost its first-party mark no longer covers an entry that pins no
# quantization on it). The keys are menu_rules.ALLOWLIST_REFUSALS.
LIST_REFUSAL_REASONS = {
    "lab_not_allowed": REASON_LAB_REMOVED,
    "host_not_allowed": REASON_HOST_REMOVED,
    "unknown_quantization": REASON_HOST_REMOVED,
}


class MenuUnavailable(ValueError):
    """The chain names a menu hash and this node does not hold what it takes
    to check it: bytes with that hash, or the signed list that menu names.
    Nothing is carried and nothing is guessed, because other nodes may hold
    the bytes and accept the menu. A miner's TD in this state produces no
    valid proof for the epoch; the bytes and the list come from the public
    reveal store or any mirror of it, and fetching them is its operator's job.
    """


class MenuRefused(ValueError):
    """Bytes with the committed hash are not a menu that can follow the
    previous one: malformed, out of sequence, or refused by the compiled
    rules. The verdict depends on the bytes, the previous menu and the signed
    list alone, so every node that holds them reaches it, and every one of
    them runs the epoch on `carried`: the previous menu, one epoch on. The
    caller logs the refusal as an alarm and uses `carried`."""

    def __init__(self, message: str, carried: MenuState):
        super().__init__(message)
        self.carried = carried


# --- parameters ----------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MenuParams:
    """The constants of the derivation step. The defaults are the design's;
    the two without one are not fixed by it and must be passed."""

    # The scored category mix, (category, integer weight): what token means
    # are weighted to (section 7.2). Pinned with the v4 pool.
    mix: tuple[tuple[str, int], ...]
    # PROPOSAL_MIN_CATEGORY_ACCURACY in basis points (section 5).
    min_category_accuracy_bp: int
    menu_lag: int = MENU_LAG
    shield_epochs: int = ENTRY_SHIELD_EPOCHS
    menu_cap: int = MENU_CAP
    min_entries: int = MENU_MIN_ENTRIES
    max_joins_per_epoch: int = MENU_MAX_JOINS_PER_EPOCH
    joins_small_field: int = MENU_JOINS_SMALL_FIELD
    joins_field_hotkeys: int = MENU_JOINS_FIELD_HOTKEYS
    max_provisional: int = MENU_MAX_PROVISIONAL
    endpoint_min_voters: int = ENDPOINT_MIN_VOTERS
    token_min_pairs: int = TOKEN_MIN_PAIRS
    refresh_epochs: int = MENU_REFRESH_EPOCHS
    min_answered_bp: int = PROPOSAL_MIN_ANSWERED_BP
    min_category_graded: int = PROPOSAL_MIN_CATEGORY_GRADED
    useful_min_pairs: int = PROPOSAL_USEFUL_MIN_PAIRS
    cooldown_epochs: int = PROPOSAL_COOLDOWN_EPOCHS
    retire_fail_min_coldkeys: int = RETIRE_FAIL_MIN_COLDKEYS
    retire_gone_epochs: int = RETIRE_GONE_EPOCHS
    retire_use_floor_bp: int = RETIRE_USE_FLOOR_BP
    retired_keep: int = RETIRED_KEEP
    pick_prior_cells_entry: int = PICK_COST_PRIOR_CELLS_ENTRY
    pick_prior_cells_category: int = PICK_COST_PRIOR_CELLS_CATEGORY

    def __post_init__(self):
        cats = [c for c, _ in self.mix]
        if (not self.mix or cats != sorted(set(cats))
                or any(not _is_int(w) or w < 0 for _, w in self.mix)
                or not any(w for _, w in self.mix)):
            raise ValueError("mix must be sorted (category, non-negative integer weight) pairs")
        for f in dataclasses.fields(self):
            if f.name != "mix" and (not _is_int(getattr(self, f.name)) or getattr(self, f.name) < 0):
                raise ValueError(f"{f.name} must be a non-negative integer")
        if (self.menu_lag < 1 or self.menu_cap < 1
                or self.refresh_epochs < 1 or self.min_entries < 1):
            raise ValueError("menu_lag, menu_cap, min_entries and refresh_epochs "
                             "must be positive")
        if self.min_entries > self.menu_cap:
            raise ValueError("min_entries must not be above menu_cap")


# --- the state -----------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """What is sent for an entry: the fields `request_for` reads (section 4.2).
    `reasoning` is the one (key, value) of OpenRouter's unified parameter, or None."""

    id: str
    rev: int
    model: str
    reasoning: tuple[str, str | int | bool] | None
    max_tokens: int
    provider_order: tuple[str, ...]
    quantization: str


@dataclass(frozen=True)
class PickCategory:
    """An entry's pick cost in one category: its graded cells there and their
    token means, shrunk toward the entry's pick means."""

    category: str
    cells: int               # graded scored-question cells of the category, at least 1
    mean_in: int             # milli-tokens
    mean_out: int


@dataclass(frozen=True)
class PickTable:
    """An entry's pick cost (W13): the token means of its graded
    scored-question cells of the current revision in the window, shrunk
    toward its call means (`pick_table`), and the same per category. A
    category not listed is priced at the entry's values."""

    cells: int               # all its graded cells, at least 1: the categories' sum
    mean_in: int             # milli-tokens
    mean_out: int
    categories: tuple[PickCategory, ...] = ()    # sorted by category, each with a cell


@dataclass(frozen=True)
class Entry(Request):
    """An entry on the menu."""

    price_in: int            # picodollars per prompt token
    price_out: int           # picodollars per completion token
    mean_in: int             # milli-tokens: the measured mean prompt tokens of a call
    mean_out: int            # milli-tokens: the measured mean completion tokens
    token_pairs: int         # the pairs the means rest on
    joined_epoch: int        # the first epoch on the menu
    # The first epoch whose labels of it are scored: PROVISIONAL while it is
    # provisional (no label of it scores for any head), then the epoch the
    # menu that made it full applies to. From then a head with no row for it
    # is scored as odds 0 on its labels. A seed entry has scored_from =
    # joined_epoch.
    scored_from: int
    # q_e: (k + 1) / (n + 2) over the field's answered labels of its join
    # epoch, in RATE_SCALE units rounded half up (`measured_rate`), set when
    # the gates make it full; (0 + 1) / (0 + 2) while provisional and for a
    # seed entry. Published; it scores nothing.
    measured_rate: int
    proposer: str            # the proposing hotkey ("" for a seed entry); decides nothing
    # The entry's pooled EXPLORATION pairs in the window, for its current
    # `rev` (section 8's n_e): what exploration's coverage part reads.
    # 0 when it joins. Authored by whoever publishes the state.
    explore_pairs: int = 0
    # The pick-cost table (W13): what a call of this entry costs on the
    # questions routers send it, from its graded scored-question cells.
    # None when it has none (it joined, its request was updated, or no head
    # routed to it in the window): a pick is then priced at the call cost.
    # Refreshed with the token means, authored by whoever publishes the state.
    pick: PickTable | None = None

    @property
    def provisional(self) -> bool:
        """Whether the gates have yet to decide it (`scored_from == PROVISIONAL`)."""
        return self.scored_from == PROVISIONAL

    def in_allowance(self, epoch: int) -> bool:
        """Whether its labels of `epoch` score for no head: it is provisional,
        or `epoch` is before the epoch it became full."""
        return int(epoch) < self.scored_from

    def shielded(self, epoch: int, shield_epochs: int = ENTRY_SHIELD_EPOCHS) -> bool:
        """Whether it may not be displaced in `epoch`: it is provisional, or
        fewer than `shield_epochs` epochs have passed since it became full."""
        return self.provisional or int(epoch) < self.scored_from + int(shield_epochs)

    def allowance_rate(self, epoch: int) -> int | None:
        """What a head with no row for this entry is scored as on a pair of
        `epoch`, as a rate: None, always. A label of a provisional entry is
        scored for no head (labels_v4.missing_odds marks it), and after it a
        missing row is odds 0."""
        return None


@dataclass(frozen=True)
class MeasureCell:
    """An entry's labels in one category over the epochs a gate reads."""

    category: str
    calls: int               # labels
    answered: int            # of them, graded (every label is)
    correct: int             # correct ones, in whole labels (the outcome sum rounded down)
    tokens_in: int           # attested prompt tokens over the labels
    tokens_out: int


@dataclass(frozen=True)
class Provisional:
    """A provisional entry's proposal: what joined it, and whose it is. Its
    entry is on the menu with `scored_from == PROVISIONAL`."""

    id: str
    proposal: str            # sha256 of the proposal document (chain_proposals)
    proposer: str            # hotkey
    coldkey: str             # the proposer's coldkey: it shares the cooldown of a failed gate


@dataclass(frozen=True)
class Retired:
    """Why an id left, or why a proposal for it ended, and the first epoch it
    may be filed again."""

    id: str
    epoch: int
    reason: str
    cooldown_until: int


@dataclass(frozen=True)
class ColdkeyCooldown:
    """A proposer's coldkey whose proposals may not join before `until`: its
    provisional entry failed a gate. Kept apart from `retired`, which is keyed by entry id
    and would lose a coldkey when the id's record is replaced."""

    coldkey: str
    until: int


@dataclass(frozen=True)
class AllowlistRef:
    """The signed list in force: its serial and the sha256 of its bytes.
    Serial 0 with an empty hash is "none adopted yet"."""

    serial: int
    sha256: str


@dataclass(frozen=True)
class MenuState:
    epoch: int                       # the epoch this state applies to
    serial: int                      # +1 whenever anything but `epoch` changes
    entries: tuple[Entry, ...]       # sorted by id
    retired: tuple[Retired, ...]     # sorted by id
    labs: AllowlistRef               # the lab and host list (section 6)
    devices: AllowlistRef            # the device list (section 13.4)
    # Proposers' coldkeys in cooldown, sorted by coldkey; a record leaves when
    # it can no longer refuse a join.
    coldkey_cooldowns: tuple[ColdkeyCooldown, ...] = ()
    # The provisional entries' proposals, sorted by id: exactly one per entry
    # with scored_from == PROVISIONAL.
    provisional: tuple[Provisional, ...] = ()

    def provisional_record(self, entry_id: str) -> Provisional | None:
        for p in self.provisional:
            if p.id == entry_id:
                return p
        return None

    def cooldowns(self) -> dict[str, int]:
        """{coldkey: the first epoch it may file again}."""
        return {c.coldkey: c.until for c in self.coldkey_cooldowns}

    def entry(self, entry_id: str) -> Entry:
        for e in self.entries:
            if e.id == entry_id:
                return e
        raise KeyError(f"{entry_id!r} is not on the menu of epoch {self.epoch}")

    def ids(self) -> tuple[str, ...]:
        """The epoch's entry ids, sorted: its index space."""
        return tuple(e.id for e in self.entries)

    def explore_pairs(self) -> dict[str, int]:
        """{entry id: its pooled exploration pairs in the window, for its
        current revision}: section 8's n_e, as this state publishes it."""
        return {e.id: e.explore_pairs for e in self.entries}


# --- the one encoding ----------------------------------------------------------


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _int(v, what: str, lo: int = 0) -> int:
    if not _is_int(v) or not lo <= v <= MAX_INT:
        raise ValueError(f"{what} must be an integer in {lo}..2^53-1, got {v!r}")
    return v


def _rate(v, what: str) -> int:
    _int(v, what)
    if v > RATE_SCALE:
        raise ValueError(f"{what} must be at most {RATE_SCALE}, got {v!r}")
    return v


def measured_rate(correct: int, answered: int) -> int:
    """q_e: (correct + 1) / (answered + 2) in RATE_SCALE units, rounded half
    up. Exact. A measurement with no answered call gives one half."""
    _int(correct, "correct")
    _int(answered, "answered")
    if correct > answered:
        raise ValueError("more correct answers than answered calls")
    return _round_half_up(Fraction(correct + 1, answered + 2) * RATE_SCALE)


def cells_rate(cells: Iterable[MeasureCell]) -> int:
    """`measured_rate` over a measurement's cells: every category's answered
    calls and correct answers summed."""
    cells = tuple(cells)
    return measured_rate(sum(c.correct for c in cells), sum(c.answered for c in cells))


def _str(v, what: str, *, empty: bool = False) -> str:
    if (not isinstance(v, str) or len(v) > MAX_STRING or (not v and not empty)
            or any(not 0x21 <= ord(c) <= 0x7E for c in v)):
        raise ValueError(f"{what} must be 1..{MAX_STRING} printable ASCII characters with no "
                         f"spaces, got {v!r}")
    return v


def _hex64(v, what: str) -> str:
    if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        raise ValueError(f"{what} must be 64 lowercase hex characters, got {v!r}")
    return v


def _keys(d, required: Iterable[str], optional: Iterable[str], what: str) -> None:
    if not isinstance(d, Mapping):
        raise ValueError(f"{what} must be an object")
    required, optional = set(required), set(optional)
    got = set(d)
    if not all(isinstance(k, str) for k in got) or got - required - optional or required - got:
        raise ValueError(f"{what}: fields {sorted(map(str, got))} are not the expected ones")


def _reasoning_dict(reasoning) -> dict | None:
    return None if reasoning is None else {reasoning[0]: reasoning[1]}


def _request_dict(r: Request) -> dict:
    out = {"id": r.id, "rev": r.rev, "model": r.model, "max_tokens": r.max_tokens,
           "provider": {"order": list(r.provider_order), "allow_fallbacks": False},
           "quantization": r.quantization}
    if r.reasoning is not None:
        out["reasoning"] = _reasoning_dict(r.reasoning)
    return out


_REQUEST_KEYS = ("id", "rev", "model", "max_tokens", "provider", "quantization")


def _request_fields(d: Mapping, what: str) -> dict:
    provider = d["provider"]
    _keys(provider, ("order", "allow_fallbacks"), (), f"{what}.provider")
    if provider["allow_fallbacks"] is not False or not isinstance(provider["order"], list):
        raise ValueError(f"{what}: `provider` must pin an `order` and forbid fallbacks")
    reasoning = None
    if "reasoning" in d:
        r = d["reasoning"]
        if not isinstance(r, Mapping) or len(r) != 1:
            raise ValueError(f"{what}: `reasoning` must set exactly one field")
        reasoning = next(iter(r.items()))
    return {"id": d["id"], "rev": d["rev"], "model": d["model"], "reasoning": reasoning,
            "max_tokens": d["max_tokens"], "provider_order": tuple(provider["order"]),
            "quantization": d["quantization"]}


def _entry_dict(e: Entry) -> dict:
    """An entry as it stands in the published document, which is also the
    form the compiled rules read (`menu_rules.check_entry`)."""
    out = dict(_request_dict(e), price_in=e.price_in, price_out=e.price_out,
               mean_in=e.mean_in, mean_out=e.mean_out, token_pairs=e.token_pairs,
               joined_epoch=e.joined_epoch, scored_from=e.scored_from,
               measured_rate=e.measured_rate, proposer=e.proposer,
               explore_pairs=e.explore_pairs)
    if e.pick is not None:
        # Present only for an entry with graded cells, so a state has one
        # document and an entry without them reads exactly as before.
        out["pick"] = {"cells": e.pick.cells, "mean_in": e.pick.mean_in,
                       "mean_out": e.pick.mean_out,
                       "categories": [{"category": c.category, "cells": c.cells,
                                       "mean_in": c.mean_in, "mean_out": c.mean_out}
                                      for c in e.pick.categories]}
    return out


def to_dict(state: MenuState) -> dict:
    """The state as the JSON document that is hashed and published."""
    return {
        "coldkey_cooldowns": {c.coldkey: c.until for c in state.coldkey_cooldowns},
        "epoch": state.epoch,
        "serial": state.serial,
        "entries": [_entry_dict(e) for e in state.entries],
        "provisional": [{"id": p.id, "proposal": p.proposal, "proposer": p.proposer,
                         "coldkey": p.coldkey} for p in state.provisional],
        "retired": {r.id: {"epoch": r.epoch, "reason": r.reason,
                           "cooldown_until": r.cooldown_until} for r in state.retired},
        "allowlists": {"labs": {"serial": state.labs.serial, "sha256": state.labs.sha256},
                       "devices": {"serial": state.devices.serial, "sha256": state.devices.sha256}},
    }


_ENTRY_KEYS = _REQUEST_KEYS + ("price_in", "price_out", "mean_in", "mean_out", "token_pairs",
                               "joined_epoch", "scored_from", "measured_rate", "proposer",
                               "explore_pairs")
_PROVISIONAL_KEYS = ("id", "proposal", "proposer", "coldkey")
_PICK_KEYS = ("cells", "mean_in", "mean_out", "categories")
_PICK_CATEGORY_KEYS = ("category", "cells", "mean_in", "mean_out")


def _pick_from_dict(d) -> PickTable:
    _keys(d, _PICK_KEYS, (), "pick table")
    if not isinstance(d["categories"], list):
        raise ValueError("a pick table's `categories` must be a list")
    cats = []
    for c in d["categories"]:
        _keys(c, _PICK_CATEGORY_KEYS, (), "pick table category")
        cats.append(PickCategory(**{k: c[k] for k in _PICK_CATEGORY_KEYS}))
    return PickTable(cells=d["cells"], mean_in=d["mean_in"], mean_out=d["mean_out"],
                     categories=tuple(cats))


def from_dict(doc) -> MenuState:
    """The state a document describes. Any field missing, unknown, of another
    type or out of range is a ValueError: a document is the state or nothing."""
    try:
        _keys(doc, ("epoch", "serial", "entries", "provisional", "retired", "allowlists",
                    "coldkey_cooldowns"), (), "menu state")
        for name in ("entries", "provisional"):
            if not isinstance(doc[name], list):
                raise ValueError(f"`{name}` must be a list")
        entries = []
        for d in doc["entries"]:
            _keys(d, _ENTRY_KEYS, ("reasoning", "pick"), "entry")
            entries.append(Entry(**_request_fields(d, "entry"),
                                 **{k: d[k] for k in _ENTRY_KEYS[len(_REQUEST_KEYS):]},
                                 pick=_pick_from_dict(d["pick"]) if "pick" in d else None))
        provisional = []
        for d in doc["provisional"]:
            _keys(d, _PROVISIONAL_KEYS, (), "provisional record")
            provisional.append(Provisional(**{k: d[k] for k in _PROVISIONAL_KEYS}))
        if not isinstance(doc["retired"], Mapping):
            raise ValueError("`retired` must be an object keyed by id")
        retired = []
        for rid in sorted(doc["retired"], key=str):
            r = doc["retired"][rid]
            _keys(r, ("epoch", "reason", "cooldown_until"), (), "retired record")
            retired.append(Retired(id=rid, epoch=r["epoch"], reason=r["reason"],
                                   cooldown_until=r["cooldown_until"]))
        if not isinstance(doc["coldkey_cooldowns"], Mapping):
            raise ValueError("`coldkey_cooldowns` must be an object keyed by coldkey")
        cooldowns = tuple(ColdkeyCooldown(ck, doc["coldkey_cooldowns"][ck])
                          for ck in sorted(doc["coldkey_cooldowns"], key=str))
        lists = doc["allowlists"]
        _keys(lists, ("labs", "devices"), (), "allowlists")
        refs = {}
        for name in ("labs", "devices"):
            _keys(lists[name], ("serial", "sha256"), (), f"allowlists.{name}")
            refs[name] = AllowlistRef(lists[name]["serial"], lists[name]["sha256"])
        state = MenuState(epoch=doc["epoch"], serial=doc["serial"], entries=tuple(entries),
                          retired=tuple(retired), labs=refs["labs"], devices=refs["devices"],
                          coldkey_cooldowns=cooldowns, provisional=tuple(provisional))
        validate(state)
    except (TypeError, KeyError, AttributeError, IndexError) as e:
        # validate checks every type before it uses a value; this names any
        # document it missed a ValueError too, never another exception.
        raise ValueError(f"not a menu state document: {type(e).__name__}: {e}") from None
    return state


def _check_request(r: Request, what: str) -> None:
    from fugal_subnet.routing_protocol import validate_menu_v3

    _str(r.id, f"{what} id")
    _int(r.rev, f"{r.id}: rev", 1)
    _str(r.model, f"{r.id}: model")
    _int(r.max_tokens, f"{r.id}: max_tokens", 1)
    _str(r.quantization, f"{r.id}: quantization")
    order = r.provider_order
    if not isinstance(order, tuple) or not order:
        raise ValueError(f"{r.id}: the provider order must name each pinned host once")
    for h in order:
        _str(h, f"{r.id}: host")
    if len(set(order)) != len(order):
        raise ValueError(f"{r.id}: the provider order must name each pinned host once")
    if r.reasoning is not None:
        if not isinstance(r.reasoning, tuple) or len(r.reasoning) != 2:
            raise ValueError(f"{r.id}: `reasoning` must be one (key, value)")
        key, value = r.reasoning
        ok = ((key == "effort" and isinstance(value, str) and _str(value, f"{r.id}: effort"))
              or (key == "max_tokens" and _is_int(value) and 1 <= value <= MAX_INT)
              or (key == "enabled" and value is True))
        if not ok:
            raise ValueError(f"{r.id}: `reasoning` must be an effort, a token budget or enabled")
    # The shape rules of a v3 menu row, unchanged (section 6): the id, the
    # model before `@`, one reasoning field, a pinned order with no fallbacks.
    validate_menu_v3([dict(_request_dict(r), **{"in": 0, "out": 0})])


def validate(state: MenuState) -> None:
    """Refuse a state no honest step produces: wrong types, values out of
    range, lists out of order, an id held twice. Raises ValueError."""
    if not isinstance(state, MenuState):
        raise ValueError("not a MenuState")
    _int(state.epoch, "epoch")
    _int(state.serial, "serial")
    for name in ("entries", "provisional", "retired"):
        if not isinstance(getattr(state, name), tuple):
            raise ValueError(f"`{name}` must be a tuple")

    for e in state.entries:
        if not isinstance(e, Entry):
            raise ValueError("`entries` must hold Entry values")
        _check_request(e, "entry")
        for f in ("price_in", "price_out", "mean_in", "mean_out", "token_pairs", "joined_epoch",
                  "explore_pairs"):
            _int(getattr(e, f), f"{e.id}: {f}")
        _int(e.scored_from, f"{e.id}: scored_from", e.joined_epoch)
        _rate(e.measured_rate, f"{e.id}: measured_rate")
        _str(e.proposer, f"{e.id}: proposer", empty=True)
        if e.pick is not None:
            _check_pick(e.pick, e.id)
    ids = [e.id for e in state.entries]
    if ids != sorted(set(ids)):
        raise ValueError("entries must be sorted by id with no id repeated")

    for p in state.provisional:
        if not isinstance(p, Provisional):
            raise ValueError("`provisional` must hold Provisional values")
        _str(p.id, "provisional id")
        _hex64(p.proposal, f"{p.id}: proposal")
        _str(p.proposer, f"{p.id}: proposer")
        _str(p.coldkey, f"{p.id}: coldkey")
    pids = [p.id for p in state.provisional]
    if pids != sorted(set(pids)):
        raise ValueError("provisional must be sorted by id with no id repeated")
    if set(pids) != {e.id for e in state.entries if e.scored_from == PROVISIONAL}:
        raise ValueError("every provisional entry, and only those, has a provisional record")
    for e in state.entries:
        if e.scored_from == PROVISIONAL and (
                e.rev != 1 or e.proposer != state.provisional_record(e.id).proposer):
            raise ValueError(f"{e.id}: a provisional entry is revision 1 of its record's proposer")

    for r in state.retired:
        if not isinstance(r, Retired):
            raise ValueError("`retired` must hold Retired values")
        _str(r.id, "retired id")
        _int(r.epoch, f"{r.id}: retired epoch")
        _int(r.cooldown_until, f"{r.id}: cooldown_until", r.epoch)
        if not isinstance(r.reason, str) or r.reason not in REASONS:
            raise ValueError(f"{r.id}: unknown retirement reason {r.reason!r}")
    rids = [r.id for r in state.retired]
    if rids != sorted(set(rids)):
        raise ValueError("retired must be sorted by id with no id repeated")

    if not isinstance(state.coldkey_cooldowns, tuple):
        raise ValueError("`coldkey_cooldowns` must be a tuple")
    for c in state.coldkey_cooldowns:
        if not isinstance(c, ColdkeyCooldown):
            raise ValueError("`coldkey_cooldowns` must hold ColdkeyCooldown values")
        _str(c.coldkey, "cooldown coldkey")
        _int(c.until, f"{c.coldkey}: cooldown until")
    cks = [c.coldkey for c in state.coldkey_cooldowns]
    if cks != sorted(set(cks)):
        raise ValueError("coldkey_cooldowns must be sorted by coldkey with none repeated")

    for name, ref in (("labs", state.labs), ("devices", state.devices)):
        if not isinstance(ref, AllowlistRef):
            raise ValueError(f"allowlists.{name} must be an AllowlistRef")
        _int(ref.serial, f"allowlists.{name}.serial")
        if ref.serial == 0 and ref.sha256 == "":
            continue
        _hex64(ref.sha256, f"allowlists.{name}.sha256")


def _pick_category(v, what: str) -> str:
    _str(v, what)
    if len(v) > PICK_CATEGORY_MAX_CHARS:
        raise ValueError(f"{what} must be at most {PICK_CATEGORY_MAX_CHARS} characters")
    return v


def _check_pick(pick, what: str) -> None:
    """A pick table's shape: integers in range, at least one cell, categories
    sorted with none repeated, each with a cell, adding up to the table's."""
    if not isinstance(pick, PickTable):
        raise ValueError(f"{what}: `pick` must be a PickTable")
    _int(pick.cells, f"{what}: pick cells", 1)
    _int(pick.mean_in, f"{what}: pick mean_in")
    _int(pick.mean_out, f"{what}: pick mean_out")
    if not isinstance(pick.categories, tuple) or len(pick.categories) > PICK_MAX_CATEGORIES:
        raise ValueError(f"{what}: pick categories must be a tuple of at most "
                         f"{PICK_MAX_CATEGORIES}")
    for c in pick.categories:
        if not isinstance(c, PickCategory):
            raise ValueError(f"{what}: pick categories must hold PickCategory values")
        _pick_category(c.category, f"{what}: pick category")
        _int(c.cells, f"{what}: pick cells of {c.category}", 1)
        _int(c.mean_in, f"{what}: pick mean_in of {c.category}")
        _int(c.mean_out, f"{what}: pick mean_out of {c.category}")
    cats = [c.category for c in pick.categories]
    if cats != sorted(set(cats)):
        raise ValueError(f"{what}: pick categories must be sorted with none repeated")
    if sum(c.cells for c in pick.categories) != pick.cells:
        raise ValueError(f"{what}: the pick categories' cells do not add up to the table's")


def _check_cells(cells, what: str) -> None:
    if not isinstance(cells, tuple):
        raise ValueError(f"{what}: cells must be a tuple")
    for c in cells:
        if not isinstance(c, MeasureCell):
            raise ValueError(f"{what}: cells must hold MeasureCell values")
        _str(c.category, f"{what}: category")
        _int(c.calls, f"{what}: calls")
        _int(c.tokens_in, f"{what}: tokens_in")
        _int(c.tokens_out, f"{what}: tokens_out")
        if not (_is_int(c.answered) and _is_int(c.correct) and 0 <= c.correct <= c.answered <= c.calls):
            raise ValueError(f"{what}: a cell needs 0 <= correct <= answered <= calls")
    cats = [c.category for c in cells]
    if cats != sorted(set(cats)):
        raise ValueError(f"{what}: cells must be sorted by category with none repeated")


def held_list(allowlist) -> menu_rules.MenuAllowlist:
    """The signed lab and host list a node hands in, checked as far as a pure
    function can: it is a list `menu_rules.verify_signed_list` returned
    (`signer` is set, which nothing else does; a draft or a parsed file has
    none), with a signed list's serial, and its `sha256` is the hash of the
    canonical bytes of its content. So the hash a state names is the list the
    rules were applied under. ValueError otherwise."""
    if not isinstance(allowlist, menu_rules.MenuAllowlist):
        raise ValueError("the menu needs the signed lab and host list itself "
                         "(menu_rules.MenuAllowlist, from verify_signed_list)")
    _int(allowlist.serial, "lab list serial", 1)
    _hex64(allowlist.sha256, "lab list sha256")
    if not isinstance(allowlist.signer, str) or not allowlist.signer:
        raise ValueError("the lab and host list was not verified: it names no signer")
    try:
        raw = menu_rules.canonical_list_bytes(
            menu_rules.KIND_MENU_ALLOWLIST, allowlist.serial, labs=allowlist.labs,
            hosts=allowlist.hosts, first_party=allowlist.first_party)
    except (TypeError, ValueError):
        raw = b""
    if hashlib.sha256(raw).hexdigest() != allowlist.sha256:
        raise ValueError("the lab and host list does not hash to the sha256 it carries")
    return allowlist


def held_device_list(device_list) -> menu_rules.DeviceList:
    """The signed device list (section 13.4) a node hands in, checked as
    `held_list` checks the lab and host list: the list itself as
    `menu_rules.verify_signed_list` returned it (a bare `AllowlistRef` is
    not a list), with a signer, a signed list's serial, and a `sha256` that is
    the hash of the canonical bytes of its content, so a reference is adopted
    only together with the list it names. ValueError otherwise."""
    if not isinstance(device_list, menu_rules.DeviceList):
        raise ValueError("a device list is adopted only as the signed device list itself "
                         "(menu_rules.DeviceList, from verify_signed_list), not a reference")
    _int(device_list.serial, "device list serial", 1)
    _hex64(device_list.sha256, "device list sha256")
    if not isinstance(device_list.signer, str) or not device_list.signer:
        raise ValueError("the device list was not verified: it names no signer")
    try:
        raw = menu_rules.canonical_list_bytes(menu_rules.KIND_DEVICE_LIST, device_list.serial,
                                              devices=device_list.devices)
    except (TypeError, ValueError):
        raw = b""
    if hashlib.sha256(raw).hexdigest() != device_list.sha256:
        raise ValueError("the device list does not hash to the sha256 it carries")
    return device_list


def check_rules(state: MenuState, allowlist: menu_rules.MenuAllowlist) -> menu_rules.Refusal | None:
    """The compiled rules on a whole state, under the list it names: every
    entry, provisional ones included, passes `menu_rules.check_entry`. None,
    or the first refusal in the state's own order. `allowlist` must be the
    list the state names (ValueError otherwise): a verdict under another list
    says nothing."""
    validate(state)
    held_list(allowlist)
    if (allowlist.serial, allowlist.sha256) != (state.labs.serial, state.labs.sha256):
        raise ValueError(f"the menu of epoch {state.epoch} names list serial {state.labs.serial}, "
                         f"not the list offered (serial {allowlist.serial})")
    for e in state.entries:
        refusal = menu_rules.check_entry(_entry_dict(e), allowlist)
        if refusal is not None:
            return menu_rules.Refusal(refusal.code, f"entry {e.id}: {refusal.detail}")
    return None


def canonical_json(state: MenuState) -> bytes:
    """The bytes that are hashed: sorted keys, no whitespace, ASCII."""
    validate(state)
    return json.dumps(to_dict(state), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def menu_hash(state: MenuState) -> str:
    """sha256 of the canonical JSON: the name of a menu in a proof's namespace
    and in the root's commitment."""
    return hashlib.sha256(canonical_json(state)).hexdigest()


def _no_float(text: str):
    raise ValueError(f"a menu state holds no fractional number ({text})")


def _no_repeat(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"field {k!r} appears twice")
        out[k] = v
    return out


def from_json(raw: bytes) -> MenuState:
    """The state of canonical bytes. Bytes in any other layout, with a
    fractional number or with a repeated field, are a ValueError: a hash is
    of bytes, so a state has exactly one document."""
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("a menu document is bytes")
    try:
        doc = json.loads(bytes(raw).decode("ascii"), parse_float=_no_float,
                         parse_constant=_no_float, object_pairs_hook=_no_repeat)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as e:
        raise ValueError(f"not a menu state document: {e}") from None
    state = from_dict(doc)
    try:
        canonical = canonical_json(state)
    except (TypeError, OverflowError) as e:
        raise ValueError(f"not a menu state document: {e}") from None
    if canonical != bytes(raw):
        raise ValueError("the bytes are not the canonical document of the state they describe")
    return state


# --- which state an epoch runs on ------------------------------------------------


def load_seed() -> MenuState:
    """The pinned seed (`routing_data/menu_v4_seed.json`, built by
    scripts/build_menu_v4_seed.py), as written: epoch 0."""
    try:
        doc = json.loads(SEED_PATH.read_text(encoding="ascii"), parse_float=_no_float,
                         parse_constant=_no_float, object_pairs_hook=_no_repeat)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{SEED_PATH.name} is not a menu state document: {e}") from None
    return from_dict(doc)


def seed_state(first_epoch: int) -> MenuState:
    """The menu of the first v4 epoch: the seed, at that epoch."""
    return dataclasses.replace(load_seed(), epoch=_int(first_epoch, "first v4 epoch"))


def carry(state: MenuState) -> MenuState:
    """The same menu one epoch later: nothing changes but `epoch`."""
    return dataclasses.replace(state, epoch=state.epoch + 1)


# The compiled constants `menu_for_epoch` checks a committed menu against when
# it is given no parameters. The reader reads menu_lag, menu_cap,
# max_provisional, shield_epochs, cooldown_epochs and refresh_epochs; the mix
# and the category floor are the step's and are not read here.
READER_PARAMS = MenuParams(mix=(("any", 1),), min_category_accuracy_bp=0)


def rejoin_gap_epochs(params: MenuParams | None = None) -> int:
    """The fewest epochs between a pair of one run of an id and a pair of its
    next run (4.4, "an id that comes back"). The last pair of a run is in the
    epoch before the first menu without it (R - 1, the record written at R);
    the id may join again at R + cooldown_epochs, and its labels count from
    there. A window of the step's statistics no longer than this never holds
    pairs of two runs, which share (id, rev): a new run starts at revision 1
    again."""
    params = READER_PARAMS if params is None else params
    return params.cooldown_epochs + 1


def menu_for_epoch(previous: MenuState, committed_hash: str | None,
                   published: bytes | None = None, *,
                   allowlist: menu_rules.MenuAllowlist | None = None,
                   device_list: menu_rules.DeviceList | None = None,
                   params: MenuParams | None = None) -> MenuState:
    """MenuState(E) from MenuState(E - 1): the one rule for which menu an
    epoch runs on (section 4.3).

    `committed_hash` is the menu hash in the root's commitment for epoch
    E - MENU_LAG, read at E's boundary block, or None when there is none.
    `published` is bytes from any source. `allowlist` is the signed lab and
    host list this node holds for the menu those bytes describe, and
    `device_list` the signed device list it holds, each as
    `menu_rules.verify_signed_list` returned it. `params` are the step's
    constants (default: the compiled ones, `READER_PARAMS`).

    - No commitment: the menu is carried, and the changes wait.
    - A commitment to the carried menu: the carried menu. It needs no bytes
      and no list, since nothing in it is new.
    - A commitment to another menu: the state with that hash, if it can
      follow the previous one. In order, every check but the last deciding
      on the bytes and the previous menu alone: at most `MENU_MAX_BYTES`;
      well-formed; for this epoch with the next serial; naming the signed
      lists in force or newer ones (never none, never an older one); within
      the compiled caps (`check_caps`); one step on from the previous menu
      (`check_transition`); and, last, passing the compiled rules under the
      lab and host list it names (`check_rules`). If it cannot follow,
      `MenuRefused`: the epoch runs on `MenuRefused.carried` on every node,
      and the caller logs an alarm.
    - A commitment this node cannot check, for want of bytes with that hash,
      of the lab and host list the menu names, or of a newer device list it
      names: `MenuUnavailable`. No menu is returned. A list is adopted only
      together with the list itself: a reference to a list no node holds
      gives every node `MenuUnavailable`, never the reference.

    A list that was not verified, or does not hash to what it carries, is a
    ValueError: the caller's mistake, never a verdict on the menu.
    """
    params = READER_PARAMS if params is None else params
    carried = carry(previous)
    if committed_hash is None:
        return carried
    _hex64(committed_hash, "the committed menu hash")
    if committed_hash == menu_hash(carried):
        return carried
    if published is None or hashlib.sha256(bytes(published)).hexdigest() != committed_hash:
        raise MenuUnavailable(
            f"epoch {carried.epoch}: the chain names menu {committed_hash[:16]} and no bytes "
            "with that hash were offered")
    if allowlist is not None:
        held_list(allowlist)
    if device_list is not None:
        held_device_list(device_list)

    def refused(why: str) -> MenuRefused:
        return MenuRefused(f"epoch {carried.epoch}: {why}", carried)

    try:
        return _check_committed(previous, carried, bytes(published), allowlist, device_list,
                                params, refused)
    except (MenuRefused, MenuUnavailable, MemoryError):
        raise
    except Exception as e:  # noqa: BLE001 - bytes that break a check are not a menu
        # Every check decides on the bytes, the previous menu and the list
        # alone, so whatever they make a check raise, every node that holds
        # them meets it too: such bytes are refused like any malformed menu,
        # never "unavailable".
        raise refused(f"the committed menu could not be checked ({type(e).__name__}: {e})") from None


def _check_committed(previous: MenuState, carried: MenuState, published: bytes,
                     allowlist, device_list, params: MenuParams, refused) -> MenuState:
    """`menu_for_epoch`'s checks of committed bytes whose hash is the
    committed one, in the order its docstring gives."""
    if len(published) > MENU_MAX_BYTES:
        raise refused(f"the committed menu is {len(published)} bytes, more than the "
                      f"{MENU_MAX_BYTES} bytes a menu may hold")
    try:
        state = from_json(published)
    except ValueError as e:
        raise refused(f"the committed menu is malformed: {e}") from None
    if state.epoch != carried.epoch or state.serial != previous.serial + 1:
        raise refused(
            f"the committed menu is for epoch {state.epoch} with serial {state.serial}; a "
            f"changed menu is for this epoch with serial {previous.serial + 1}")
    for name, new, old in (("lab and host", state.labs, previous.labs),
                           ("device", state.devices, previous.devices)):
        if new.serial < old.serial or (new.serial == old.serial and new != old):
            raise refused(
                f"the committed menu names {name} list serial {new.serial} "
                f"({new.sha256[:16]}); the list in force is serial {old.serial} "
                f"({old.sha256[:16]}), and only a newer one replaces it")
    if state.labs.serial == 0:
        raise refused("the committed menu names no signed list, and a menu changes only "
                      "under one")
    for why in (check_caps(state, params), check_transition(previous, state, params)):
        if why is not None:
            raise refused(why)
    if allowlist is None or (allowlist.serial, allowlist.sha256) != (
            state.labs.serial, state.labs.sha256):
        raise MenuUnavailable(
            f"epoch {carried.epoch}: the committed menu names lab and host list serial "
            f"{state.labs.serial} ({state.labs.sha256[:16]}), and that list was not offered")
    if state.devices != previous.devices and (device_list is None or (
            device_list.serial, device_list.sha256) != (state.devices.serial, state.devices.sha256)):
        raise MenuUnavailable(
            f"epoch {carried.epoch}: the committed menu adopts device list serial "
            f"{state.devices.serial} ({state.devices.sha256[:16]}), and that list was not offered")
    why = _list_verdicts(previous, state, allowlist, params)
    if why is not None:
        raise refused(why)
    refusal = check_rules(state, allowlist)
    if refusal is not None:
        raise refused(str(refusal))
    return state


def _list_verdicts(previous: MenuState, state: MenuState,
                   allowlist: menu_rules.MenuAllowlist, params: MenuParams) -> str | None:
    """What leaves for a list reason in a menu that adopts a newer lab and
    host list is what that list refuses, with the reason it refuses it for,
    or why not. The reader holds the list the menu names, so it recomputes
    the step's verdicts (4.4, step 1) rather than take them from the menu:
    an entry (a provisional one included) that leaves with a `lab_removed`
    or `host_removed` record is one `menu_rules.check_entry` refuses under
    the list, for that reason. Without a newer list no list reason is
    accepted at all (`check_transition`)."""
    if state.labs == previous.labs:
        return None
    before = {r.id: r for r in previous.retired}
    after = {r.id: r for r in state.retired}
    entries_now = {e.id for e in state.entries}
    list_reasons = frozenset(LIST_REFUSAL_REASONS.values())
    for e in previous.entries:
        r = after.get(e.id)
        if e.id in entries_now or r is None or before.get(e.id) == r or r.reason not in list_reasons:
            continue
        refusal = menu_rules.check_entry(_entry_dict(e), allowlist)
        got = None if refusal is None else LIST_REFUSAL_REASONS.get(refusal.code)
        if got != r.reason:
            return (f"entry {e.id} leaves for {r.reason}, and the list it adopts "
                    + (f"refuses it for {refusal.code}" if refusal is not None
                       else "refuses it for nothing"))
    return None


def check_caps(state: MenuState, params: MenuParams = READER_PARAMS) -> str | None:
    """The compiled caps on one menu, or None: at least `min_entries` full
    entries (provisional ones are not counted: the next step's gates may
    retire them all) and at most `menu_cap` entries, every entry with an
    expected cost a call above zero, at most `max_provisional` provisional
    entries, at most
    `retired_keep` records in `retired` and at most `coldkey_cooldown_cap`
    coldkeys in cooldown. The first are what the epoch's exploration draw and
    every head's picks need of a menu (`epoch_plan.explore_shares_v4`,
    `head_odds.picks_and_odds_v4`); the last two bound the document, so that
    no menu can bring the next one past `MENU_MAX_BYTES`. Every menu the step
    derives from a menu within them is within them, and so is every menu a
    join delta the reader accepts (`chain_proposals.check_delta`) leaves."""
    if len(state.entries) > params.menu_cap:
        return f"the menu holds {len(state.entries)} entries, more than the cap of {params.menu_cap}"
    if full_count(state.entries) < params.min_entries:
        return (f"the menu holds {full_count(state.entries)} full entries (of "
                f"{len(state.entries)}), fewer than the minimum of {params.min_entries}")
    for e in state.entries:
        if entry_cost(e) <= 0:
            return f"entry {e.id}: its expected cost a call is not above zero"
        why = _pick_outside_range(e, params)
        if why is not None:
            return why
    if len(state.provisional) > params.max_provisional:
        return (f"{len(state.provisional)} provisional entries, more than the "
                f"{params.max_provisional} allowed")
    if len(state.retired) > params.retired_keep:
        return (f"the retired record holds {len(state.retired)} ids, more than the "
                f"{params.retired_keep} it keeps")
    if len(state.coldkey_cooldowns) > coldkey_cooldown_cap(params):
        return (f"{len(state.coldkey_cooldowns)} coldkeys in cooldown, more than the "
                f"{coldkey_cooldown_cap(params)} the steps can leave")
    return None


def _shrink(sum_milli: int, cells: int, prior: int, prior_cells: int) -> int:
    """(sum + prior_cells x prior) / (cells + prior_cells), rounded half up,
    capped at MAX_INT: the shrinkage of `pick_table`, in milli-tokens."""
    return min(MAX_INT, _round_half_up(Fraction(sum_milli + prior_cells * prior,
                                                cells + prior_cells)))


def _pick_outside_range(e: Entry, params: MenuParams) -> str | None:
    """Why `e`'s pick table is not one its graded cells could give, or None.

    Every cell's tokens are at most the answer cap (`menu_rules.
    MAX_ANSWER_TOKENS`: `menu_inputs` bounds each cell by it), so a table of
    n cells lies between its prior's share of what it shrinks toward (every
    cell free) and that share plus n cells at the cap; a category likewise
    toward the entry's pick means. And a pick never costs nothing. The cells
    themselves are only in the publisher's proofs, so this is what a node
    that holds the menu alone can check of a table."""
    p = e.pick
    if p is None:
        return None
    cap = MILLI * menu_rules.MAX_ANSWER_TOKENS

    def within(value: int, cells: int, prior: int, k: int) -> bool:
        return _shrink(0, cells, prior, k) <= value <= _shrink(cells * cap, cells, prior, k)

    ke, kc = params.pick_prior_cells_entry, params.pick_prior_cells_category
    if not (within(p.mean_in, p.cells, e.mean_in, ke) and within(p.mean_out, p.cells, e.mean_out, ke)):
        return (f"entry {e.id}: its pick means {p.mean_in}/{p.mean_out} are not what "
                f"{p.cells} graded cells shrunk toward its call means can give")
    if _cost(e.price_in, e.price_out, p.mean_in, p.mean_out) <= 0:
        return f"entry {e.id}: its pick cost is not above zero"
    for c in p.categories:
        if not (within(c.mean_in, c.cells, p.mean_in, kc)
                and within(c.mean_out, c.cells, p.mean_out, kc)):
            return (f"entry {e.id}: its pick means in {c.category} are not what {c.cells} "
                    "graded cells shrunk toward its pick means can give")
        if _cost(e.price_in, e.price_out, c.mean_in, c.mean_out) <= 0:
            return f"entry {e.id}: its pick cost in {c.category} is not above zero"
    return None


def coldkey_cooldown_cap(params: MenuParams = READER_PARAMS) -> int:
    """The most proposers' coldkeys in cooldown one menu holds: at most
    `max_provisional` failed gates a step, each kept for fewer than
    `cooldown_epochs + menu_lag` steps."""
    return params.max_provisional * (params.cooldown_epochs + params.menu_lag)


# What a refresh moves (MENU_REFRESH_EPOCHS) on an entry that keeps its
# revision and is not decided in the step.
_REFRESHED = ("price_in", "price_out", "max_tokens", "mean_in", "mean_out", "token_pairs",
              "explore_pairs", "pick")


def refresh_due(epoch: int, params: MenuParams = READER_PARAMS) -> bool:
    """Whether the step that derives the menu of `epoch` refreshes prices,
    answer limits, token means, exploration-pair counts and pick-cost tables
    whatever else it does: `epoch` is a multiple of `refresh_epochs`."""
    return int(epoch) % params.refresh_epochs == 0


def structural_change(previous: MenuState, state: MenuState) -> bool:
    """Whether `state` differs from `previous` in its structure, read from
    the two menus alone: an entry joined, retired, was decided (left its
    provisional state) or had its request updated (a new revision), or a
    newer signed list was adopted (lab and host, or device). A record pruned
    from `retired` and a cooldown running out are not."""
    return (previous.ids() != state.ids()
            or any(a.rev != b.rev or a.scored_from != b.scored_from
                   for a, b in zip(previous.entries, state.entries))
            or (previous.labs, previous.devices) != (state.labs, state.devices)
            or previous.provisional != state.provisional)


def _decided(old: Entry, new: Entry) -> bool:
    """Whether a step made provisional `old` full as `new`."""
    return old.provisional and not new.provisional


def _refreshed_pairs(previous: MenuState, state: MenuState):
    """(what, old, new, fields) for every entry a refresh may move: one that
    keeps its revision and is not decided in this step."""
    before = {e.id: e for e in previous.entries}
    for e in state.entries:
        old = before.get(e.id)
        if old is not None and old.rev == e.rev and not _decided(old, e):
            yield f"entry {e.id}", old, e, _REFRESHED


def _refresh_outside_cadence(previous: MenuState, state: MenuState) -> str | None:
    """What `state` refreshes of `previous`, or None (for a step that may not
    refresh)."""
    for what, old, new, fields in _refreshed_pairs(previous, state):
        if not _same(new, old, fields):
            return what
    return None


def _price_beyond_step(previous: MenuState, state: MenuState) -> str | None:
    """A price that moved more than MENU_PRICE_STEP, on an entry refreshed or
    decided in this step, or None."""
    before = {e.id: e for e in previous.entries}
    for e in state.entries:
        old = before.get(e.id)
        if old is None or old.rev != e.rev:
            continue
        for f in ("price_in", "price_out"):
            if not _price_within_step(getattr(e, f), getattr(old, f)):
                return (f"entry {e.id}: {f} moved from {getattr(old, f)} to {getattr(e, f)}, more "
                        f"than a factor of {MENU_PRICE_STEP} in one step")
    return None


_IDENTITY = ("id", "rev", "model", "reasoning", "provider_order", "quantization", "proposer")


def _records_follow(previous: MenuState, state: MenuState, *, k: int, new_epoch: int,
                    params: MenuParams) -> str | None:
    """`retired` and `coldkey_cooldowns` are what the step leaves of
    `previous`'s, or why not. A cooldown decides who may join, so a record
    the step did not write would refuse joins for as long as it says:

    - a record that is new, or differs from `previous`'s for its id, is
      written in this step for an entry that left the menu here, with this
      epoch, a reason the step gives for that (a gate reason only for a
      provisional entry the step decides), and exactly the compiled
      cooldown. So nothing is invented or lengthened;
    - a record leaves only by the step's pruning (over `retired_keep`,
      cooldowns that can no longer refuse a join, the oldest first): the
      reader recomputes which;
    - a coldkey's cooldown is exactly what the step leaves: those still
      running, and the coldkey of each provisional entry that failed a gate
      here, at exactly the compiled length."""
    full = new_epoch + params.cooldown_epochs
    before = {r.id: r for r in previous.retired}
    after = {r.id: r for r in state.retired}
    entries_now = {e.id for e in state.entries}
    allowed: dict[str, frozenset] = {}
    decided_here = set()
    for e in previous.entries:
        if e.id in entries_now:
            continue
        gate = e.provisional and e.joined_epoch <= k
        allowed[e.id] = _ENTRY_REASONS | (_GATE_REASONS if gate else frozenset())
        if gate:
            decided_here.add(e.id)
    for r in state.retired:
        if before.get(r.id) == r:
            continue
        if (r.epoch, r.cooldown_until) != (new_epoch, full) or r.reason not in allowed.get(r.id, ()):
            return (f"the retired record of {r.id} ({r.reason}, epoch {r.epoch}, cooldown until "
                    f"{r.cooldown_until}) is not one the step writes: a record is written only "
                    f"for an id that leaves the menu in this step, at epoch {new_epoch} with the "
                    f"cooldown until {full}")
    kept = dict(before)
    kept.update({rid: r for rid, r in after.items() if before.get(rid) != r})
    excess = len(kept) - params.retired_keep
    for r in sorted(kept.values(), key=lambda r: (r.epoch, r.id)):
        if excess <= 0:
            break
        if not _cooldown_runs(r.cooldown_until, k):
            del kept[r.id]
            excess -= 1
    if set(kept) != set(after):
        extra, missing = sorted(set(after) - set(kept)), sorted(set(kept) - set(after))
        return (f"the retired record keeps {extra[:3]} and drops {missing[:3]}, not what the "
                "step keeps (a record leaves only for room, the oldest cooldown that ran out "
                "first)")
    expected = {ck: until for ck, until in previous.cooldowns().items() if _cooldown_runs(until, k)}
    for eid in sorted(decided_here):
        r = after.get(eid)
        rec = previous.provisional_record(eid)
        if r is not None and before.get(eid) != r and r.reason in _GATE_REASONS and rec is not None:
            expected[rec.coldkey] = max(full, expected.get(rec.coldkey, 0))
    if state.cooldowns() != expected:
        diff = sorted(set(state.cooldowns().items()) ^ set(expected.items()))
        return (f"the proposers' cooldowns are not the ones the step leaves ({len(diff)} differ: "
                "a coldkey's cooldown is written only by a failed gate in this step, at the "
                "compiled length, and leaves when it can no longer refuse a join)")
    return None


def _same(a, b, fields) -> bool:
    return all(getattr(a, f) == getattr(b, f) for f in fields)


def check_transition(previous: MenuState, state: MenuState,
                     params: MenuParams = READER_PARAMS) -> str | None:
    """Whether `state` can be one step on from `previous` (section 4.3), on
    what every node holds: the two menus and the compiled constants. None, or
    why not. The step's inputs (votes, labels, use) are not needed and not
    read, so the checks are those the step's output meets whatever its
    inputs:

    - no entry joins in a step (an entry joins only at a boundary, by the
      root's join delta: `apply_joins`), and no entry's request, revision,
      proposer or `joined_epoch` changes;
    - a provisional entry stays provisional, unchanged but for a refresh,
      while it joined after the scored epoch; one that joined by the scored
      epoch is decided in this step: it becomes full with `scored_from` this
      epoch, or it leaves with a gate's record (or for a list, health or
      displacement reason), and its provisional record leaves with it. No
      other entry's `scored_from` or `measured_rate` moves;
    - an entry leaves only with the `retired` record a retirement writes
      (its reason, this epoch, the full cooldown), a list reason only with
      a newer list (and, in `menu_for_epoch`, which holds the list, only for
      what that list refuses: `_list_verdicts`), a displacement only of an
      entry that is not shielded (`Entry.shielded`), at most one a step,
      and only to keep one place free on a menu that would otherwise be at
      its cap, and at most `health_retirement_bound` health retirements;
    - no cooldown that can still refuse a join (an id's in `retired`, a
      coldkey's) is dropped or shortened, and none is invented or
      lengthened: `retired` and the coldkeys' cooldowns are exactly what the
      step leaves (`_records_follow`);
    - prices, answer limits, token means, exploration-pair counts and
      pick-cost tables move only at the refresh cadence (`refresh_due`),
      except on an entry decided in this step, and a price by at most a
      factor of `MENU_PRICE_STEP`.

    The values of prices, answer limits, token means, exploration-pair
    counts and pick-cost tables (when they may move; `check_caps` holds a
    table to what its cells could give), retirements and which provisional
    entries pass their gates are decided by inputs only the root's proofs
    carry (another node holds only its own proof set and cannot recompute
    them exactly), and are not checked here (section 4.3 says what that
    leaves)."""
    k = previous.epoch - params.menu_lag + 1           # the scored epoch of the step
    new_epoch = previous.epoch + 1
    before = {e.id: e for e in previous.entries}
    for e in state.entries:
        old = before.get(e.id)
        if old is None:
            return (f"entry {e.id} joins in a step: an entry joins only at a boundary, by the "
                    "root's join delta")
        if not _same(e, old, _IDENTITY) or e.joined_epoch != old.joined_epoch:
            return (f"entry {e.id}: its request, revision, proposer or joined_epoch changed in "
                    "a step")
        if old.provisional:
            if e.provisional and old.joined_epoch <= k:
                return (f"entry {e.id} joined at {old.joined_epoch}, by the scored epoch {k}: the "
                        "step decides it (full, or retired)")
            if not e.provisional:
                if old.joined_epoch > k:
                    return (f"entry {e.id} joined at {old.joined_epoch}, after the scored epoch "
                            f"{k}: its labels are not in this step, so it stays provisional")
                if e.scored_from != new_epoch:
                    return (f"entry {e.id} becomes full scored from {e.scored_from}; the step "
                            f"that decides it scores it from {new_epoch}")
            elif e.measured_rate != old.measured_rate:
                return f"entry {e.id}: a provisional entry's measured_rate moved"
        elif (e.scored_from, e.measured_rate) != (old.scored_from, old.measured_rate):
            return (f"entry {e.id}: its scored_from and measured_rate moved from "
                    f"{old.scored_from}/{old.measured_rate} to {e.scored_from}/{e.measured_rate}")
    was = {p.id: p for p in previous.provisional}
    for p in state.provisional:
        if was.get(p.id) != p:
            return f"provisional record of {p.id}: a step writes none and changes none"

    records = {r.id: r for r in state.retired}
    new_ids = {e.id for e in state.entries}
    health = displaced = 0
    for e in previous.entries:
        if e.id in new_ids:
            continue
        r = records.get(e.id)
        reasons = _ENTRY_REASONS | (_GATE_REASONS if e.provisional and e.joined_epoch <= k
                                    else frozenset())
        if (r is None or r.epoch != new_epoch or r.cooldown_until != new_epoch + params.cooldown_epochs
                or r.reason not in reasons):
            return (f"entry {e.id} leaves the menu without the record a retirement writes "
                    f"(why, epoch {new_epoch}, cooldown until "
                    f"{new_epoch + params.cooldown_epochs})")
        if r.reason in LIST_REFUSAL_REASONS.values() and state.labs == previous.labs:
            return f"entry {e.id} leaves for {r.reason} with no newer list adopted"
        if r.reason == REASON_PRICE_CEILING and not (
                refresh_due(new_epoch, params) and menu_rules.check_price(
                    price_step(e.price_in * MENU_PRICE_STEP, e.price_in),
                    price_step(e.price_out * MENU_PRICE_STEP, e.price_out)) is not None):
            return (f"entry {e.id} leaves for {r.reason}, which only a refresh whose price move "
                    "could pass the ceiling gives")
        health += r.reason in HEALTH_REASONS
        if r.reason == REASON_DISPLACED:
            displaced += 1
            if e.shielded(new_epoch, params.shield_epochs):
                return f"entry {e.id} is displaced while it is shielded"
    if health > health_retirement_bound(len(previous.entries)):
        return (f"{health} health retirements in one step, at most "
                f"{health_retirement_bound(len(previous.entries))} on a menu of "
                f"{len(previous.entries)}")
    if displaced > 1 or (displaced and len(state.entries) != params.menu_cap - 1):
        return (f"{displaced} entries displaced, leaving {len(state.entries)} of the menu's "
                f"{params.menu_cap} places: the step displaces one entry only to keep one place "
                "free on a menu that would otherwise be at its cap")
    why = _records_follow(previous, state, k=k, new_epoch=new_epoch, params=params)
    if why is not None:
        return why
    kept = {r.id: r.cooldown_until for r in state.retired}
    for r in previous.retired:
        if _cooldown_runs(r.cooldown_until, k) and kept.get(r.id, -1) < r.cooldown_until:
            return (f"the cooldown of {r.id} runs until {r.cooldown_until}, and the menu drops "
                    "or shortens it")
    held = state.cooldowns()
    for ck, until in previous.cooldowns().items():
        if _cooldown_runs(until, k) and held.get(ck, -1) < until:
            return f"a proposer's cooldown runs until {until}, and the menu drops or shortens it"
    was = {e.id: e for e in previous.entries}
    for e in state.entries:
        if e.pick is not None and e.id in was and was[e.id].provisional and not e.provisional:
            return (f"{e.id} is made full with a pick table: an entry made full starts at its "
                    "call cost")
    if not refresh_due(new_epoch, params):
        what = _refresh_outside_cadence(previous, state)
        if what is not None:
            return (f"{what}: a price, answer limit, token mean, exploration-pair count or "
                    f"pick cost moved in epoch {new_epoch}, which is no refresh epoch (every "
                    f"{params.refresh_epochs})")
    return _price_beyond_step(previous, state)


# --- the boundary's joins ----------------------------------------------------------


@dataclass(frozen=True)
class NewEntry:
    """One entry a join delta admits (chain_proposals.check_delta): its
    request as the proposal document gives it, the listed price of its
    dearer pinned host, and whose proposal it is."""

    request: Request
    price_in: int            # picodollars per prompt token
    price_out: int
    proposal: str            # sha256 of the proposal document
    proposer: str            # hotkey
    coldkey: str


def join_means(base: MenuState) -> tuple[int, int]:
    """(mean_in, mean_out) a joining entry is priced at until its own labels
    give its token means: the lower median, separately, of the full entries'
    means (of every entry's when none is full). Integers, exact."""
    pool = [e for e in base.entries if not e.provisional] or list(base.entries)
    ins = sorted(e.mean_in for e in pool)
    outs = sorted(e.mean_out for e in pool)
    return ins[(len(ins) - 1) // 2], outs[(len(outs) - 1) // 2]


def apply_joins(base: MenuState, joins: Iterable[NewEntry],
                params: MenuParams | None = None) -> MenuState:
    """MenuState(E) from base(E) (what `menu_for_epoch` gives) and the
    entries the epoch's join delta admits, in the delta's order: each joins
    as a provisional entry at revision 1, joined_epoch E, scored_from
    PROVISIONAL, priced at its listed price and `join_means`, with no
    exploration pairs and no pick-cost table, and with its provisional
    record; a record of its id in `retired` (whose cooldown has run out)
    leaves. `serial` + 1 when anything joins. Pure; ValueError for a join
    the menu cannot take (an id already on it, a request that is not one,
    caps exceeded): `chain_proposals.check_delta` refuses those first."""
    joins = tuple(joins)
    if not joins:
        return base
    mean_in, mean_out = join_means(base)
    entries = {e.id: e for e in base.entries}
    retired = {r.id: r for r in base.retired}
    records = {p.id: p for p in base.provisional}
    for j in joins:
        if not isinstance(j, NewEntry) or not isinstance(j.request, Request):
            raise ValueError("a join is a NewEntry")
        r = j.request
        if r.id in entries:
            raise ValueError(f"{r.id} is on the menu already")
        if r.rev != 1:
            raise ValueError(f"{r.id}: an entry joins at revision 1")
        entries[r.id] = Entry(
            id=r.id, rev=1, model=r.model, reasoning=r.reasoning, max_tokens=r.max_tokens,
            provider_order=r.provider_order, quantization=r.quantization,
            price_in=j.price_in, price_out=j.price_out, mean_in=mean_in, mean_out=mean_out,
            token_pairs=0, joined_epoch=base.epoch, scored_from=PROVISIONAL,
            measured_rate=measured_rate(0, 0), proposer=j.proposer, explore_pairs=0, pick=None)
        records[r.id] = Provisional(id=r.id, proposal=_hex64(j.proposal, "proposal"),
                                    proposer=j.proposer, coldkey=j.coldkey)
        retired.pop(r.id, None)
    state = dataclasses.replace(
        base, serial=base.serial + 1, entries=tuple(entries[i] for i in sorted(entries)),
        retired=tuple(retired[i] for i in sorted(retired)),
        provisional=tuple(records[i] for i in sorted(records)))
    validate(state)
    why = check_caps(state, READER_PARAMS if params is None else params)
    if why is not None:
        raise ValueError(why)
    return state


# --- reading a state -------------------------------------------------------------


def request_for(state: MenuState, entry_id: str) -> dict:
    """The request fields the harness sends for this entry: what
    `routing_protocol.request_for` reads from a v3 table row, read from the
    entry."""
    e = state.entry(entry_id)
    out = {"model": e.model, "max_tokens": e.max_tokens}
    if e.reasoning is not None:
        out["reasoning"] = _reasoning_dict(e.reasoning)
    out["provider"] = {"order": list(e.provider_order), "allow_fallbacks": False}
    return out


def _cost(price_in: int, price_out: int, mean_in: int, mean_out: int) -> Fraction:
    return Fraction(price_in * mean_in + price_out * mean_out, PICO * MILLI)


def entry_cost(entry: Entry) -> Fraction:
    """Expected dollars of one call: prices times token means, exactly."""
    return _cost(entry.price_in, entry.price_out, entry.mean_in, entry.mean_out)


def costs(state: MenuState) -> dict[str, Fraction]:
    """{entry id: expected dollars a call} for the epoch's menu: the CALL
    cost, what a call costs on a question drawn at random. Exploration's
    shares, weights and spend caps read it."""
    return {e.id: entry_cost(e) for e in state.entries}


def pick_means(entry: Entry, category: str | None = None) -> tuple[int, int]:
    """(mean_in, mean_out) in milli-tokens a pick of `entry` is priced at:
    its pick-cost table's means in `category` when the table lists it, else
    the table's own, else (no graded cells) its call means."""
    p = entry.pick
    if p is None:
        return entry.mean_in, entry.mean_out
    for c in p.categories:
        if c.category == category:
            return c.mean_in, c.mean_out
    return p.mean_in, p.mean_out


def entry_pick_cost(entry: Entry, category: str | None = None) -> Fraction:
    """Expected dollars of one call of `entry` on a question routers send it
    (in `category`, when given): its prices times its pick means, exactly."""
    return _cost(entry.price_in, entry.price_out, *pick_means(entry, category))


def pick_costs(state: MenuState) -> dict[str, Fraction]:
    """{entry id: expected dollars a pick} for the epoch's menu: the PICK
    cost, one number per entry, because a pick is made without knowing a
    question's category and the entry's cells already carry the categories
    it is picked on. What a head's pick (`head_odds.picks_and_odds_v4`, the
    TD and the validator) and the product read. An entry with no graded
    cells is priced at its call cost."""
    return {e.id: entry_pick_cost(e) for e in state.entries}


def pick_cost(state: MenuState, entry_id: str, category: str | None = None) -> Fraction:
    """Expected dollars of a call of `entry_id` on a question of `category`,
    for the real-picks check (SEAT_MINIMAL_V4.md 3.4): the table's value in
    that category, or the entry's pick cost when the table does not list it
    (or `category` is None). KeyError for an id not on the menu."""
    return entry_pick_cost(state.entry(entry_id), category)


def pick_table(cells: Iterable[PickCell], entry: Entry, params: MenuParams) -> PickTable | None:
    """The pick-cost table of `entry` from its graded scored-question cells,
    `cells` per category (token sums in tokens), shrunk toward the entry's
    call means (W13; the earlier seat design's 8.2):

        pick    = (1000 x T + K_E x call means) / (n + K_E)     K_E = pick_prior_cells_entry
        pick_c  = (1000 x T_c + K_C x pick) / (n_c + K_C)       K_C = pick_prior_cells_category

    per token kind, in milli-tokens, each rounded half up; a category shrinks
    toward the entry's rounded (published) value. None with no cells, or
    when the means would make a pick cost nothing (the call cost stands); a
    category whose means would make a call cost nothing is priced at the
    entry's. Exact, and independent of the order of `cells`."""
    sums = {}
    for c in cells:
        _check_pick_cell(c, entry.id)
        if c.category in sums:
            raise ValueError(f"pick cells of {entry.id}: category {c.category!r} appears twice")
        if c.cells:
            sums[c.category] = (c.cells, MILLI * c.tokens_in, MILLI * c.tokens_out)
    return pick_table_from_sums(sums, entry, params)


def pick_table_from_sums(sums: Mapping[str, tuple[int, int, int]], entry: Entry,
                         params: MenuParams) -> PickTable | None:
    """`pick_table` on {category: (cells, prompt milli-tokens, completion
    milli-tokens)}, each with at least one cell: sums in milli-tokens, so a
    builder with means rather than token sums (the seed's prompt side) loses
    nothing to rounding."""
    n = sum(v[0] for v in sums.values())
    if not n:
        return None
    ke, kc = params.pick_prior_cells_entry, params.pick_prior_cells_category
    p_in = _shrink(sum(v[1] for v in sums.values()), n, entry.mean_in, ke)
    p_out = _shrink(sum(v[2] for v in sums.values()), n, entry.mean_out, ke)
    if _cost(entry.price_in, entry.price_out, p_in, p_out) <= 0:
        return None
    cats = []
    for cat in sorted(sums):
        cells, t_in, t_out = sums[cat]
        c_in, c_out = _shrink(t_in, cells, p_in, kc), _shrink(t_out, cells, p_out, kc)
        if _cost(entry.price_in, entry.price_out, c_in, c_out) <= 0:
            c_in, c_out = p_in, p_out
        cats.append(PickCategory(cat, cells, c_in, c_out))
    return PickTable(cells=n, mean_in=p_in, mean_out=p_out, categories=tuple(cats))


def _round_half_up(x: Fraction) -> int:
    return int((2 * x + 1) // 2)


@dataclass(frozen=True)
class TokenCell:
    """An entry's attested tokens in one category: sums over `pairs` pairs."""

    category: str
    pairs: int
    tokens_in: int
    tokens_out: int


def weighted_token_means(cells: Iterable[TokenCell],
                         mix: Iterable[tuple[str, int]]) -> tuple[int, int, int]:
    """(mean_in, mean_out, pairs): the mean tokens of a call in milli-tokens,
    each category's mean weighted by its share of the scored mix (section
    7.2), renormalised over the categories measured, as the single-model
    table weights accuracy. Categories outside the mix carry no weight; if no
    measured category is in the mix, the plain mean over all pairs. Exact,
    rounded half up; (0, 0, 0) with no pairs."""
    weight = dict(mix)
    by_cat: dict[str, TokenCell] = {}
    for c in cells:
        for v in (c.pairs, c.tokens_in, c.tokens_out):
            _int(v, f"token cell {c.category}")
        if c.category in by_cat:
            raise ValueError(f"category {c.category!r} appears twice")
        if c.pairs:
            by_cat[c.category] = c
    pairs = sum(c.pairs for c in by_cat.values())
    if not pairs:
        return 0, 0, 0
    total_w = sum(weight.get(cat, 0) for cat in by_cat)
    means = []
    for field in ("tokens_in", "tokens_out"):
        if total_w:
            mean = sum(weight.get(cat, 0) * Fraction(getattr(c, field), c.pairs)
                       for cat, c in by_cat.items()) / total_w
        else:
            mean = Fraction(sum(getattr(c, field) for c in by_cat.values()), pairs)
        means.append(min(MAX_INT, _round_half_up(mean * MILLI)))
    return means[0], means[1], pairs


# --- the inputs of a step --------------------------------------------------------
#
# Each is what one aggregation over an epoch's verified proofs, or over the
# window, hands to the step. The step reads them by key, so the order they
# arrive in never matters; a key given twice is a ValueError.


@dataclass(frozen=True)
class HostRead:
    """The voted read of one host of one model for the epoch (section 7.1):
    the median across coldkeys of what their TDs read from the listing.
    `listed` false means the read found no such host. `limit` is the host's
    answer limit in tokens, 0 when it states none."""

    model: str
    host: str
    voters: int
    listed: bool
    price_in: int = 0            # picodollars per token
    price_out: int = 0
    limit: int = 0


@dataclass(frozen=True)
class TokenStat:
    """An entry's attested tokens over its window pairs of revision `rev`,
    by category."""

    entry_id: str
    rev: int
    cells: tuple[TokenCell, ...]


@dataclass(frozen=True)
class PickCell:
    """An entry's graded scored-question cells in one category: how many,
    and the sums of their consensus tokens (each cell's coldkey median,
    whole tokens, at most the answer cap: menu_inputs.graded_cells)."""

    category: str
    cells: int
    tokens_in: int
    tokens_out: int


@dataclass(frozen=True)
class PickStat:
    """An entry's graded scored-question cells over the window, for
    revision `rev`, by category: what the pick-cost table is computed from
    (`pick_table`)."""

    entry_id: str
    rev: int
    cells: tuple[PickCell, ...]


def _check_pick_cell(c, what: str) -> None:
    """A graded-cell sum the step can read: a category, integers in range,
    and no more tokens than its cells could hold at the answer cap (a cell
    is bounded by it, so sums above it contradict themselves)."""
    if not isinstance(c, PickCell):
        raise ValueError(f"pick cells of {what} must be PickCell values")
    _pick_category(c.category, f"pick cells of {what}: category")
    for f in ("cells", "tokens_in", "tokens_out"):
        _int(getattr(c, f), f"pick cells of {what}: {f}")
    bound = c.cells * menu_rules.MAX_ANSWER_TOKENS
    if c.tokens_in > bound or c.tokens_out > bound:
        raise ValueError(f"pick cells of {what} in {c.category}: more tokens than {c.cells} "
                         f"cells of at most {menu_rules.MAX_ANSWER_TOKENS} hold")


@dataclass(frozen=True)
class ExploreCount:
    """An entry's pooled exploration pairs in the window through the scored
    epoch, for revision `rev` (section 8's n_e). Pairs of a proposal's
    measurement are never in it."""

    entry_id: str
    rev: int
    pairs: int


@dataclass(frozen=True)
class HealthStat:
    """An entry's health for its revision `rev` (section 9.2), as the
    aggregation over verified proofs counts it (menu_inputs.health_stats).

    Over the last RETIRE_WINDOW_EPOCHS counted epochs (an outage epoch is not
    counted), `eligible` is how many coldkeys made at least RETIRE_MIN_CALLS
    exploration calls to the entry, and `failing` how many of those had more
    than RETIRE_FAIL_RATE_BP of them fail, with failed calls in at least
    RETIRE_FAIL_MIN_EPOCHS distinct epochs. `unlisted_epochs` is how many
    consecutive epochs, ending with the scored one, the voted reads listed
    none of its pinned hosts."""

    entry_id: str
    rev: int
    eligible: int = 0
    failing: int = 0
    unlisted_epochs: int = 0
    # The coldkeys serving over those epochs (`retire_minimum`); None: the
    # compiled minimum applies as it stands.
    field: int | None = None


def is_full(entry: Entry) -> bool:
    """Whether `entry` is FULL (not provisional): the one definition the
    menu minimum (`full_count`, `MENU_MIN_ENTRIES`) and a challenge's picks
    (picks_v4.challenge_picks) both read."""
    return not entry.provisional


def full_count(entries: Iterable[Entry]) -> int:
    """How many of `entries` are full: what `MENU_MIN_ENTRIES` counts."""
    return sum(1 for e in entries if is_full(e))


def health_retirement_bound(entries: int) -> int:
    """The most health retirements ("failing" and "gone") one step makes on
    a menu of `entries` entries: max(2, a tenth of the menu). One bad epoch,
    one listing change or one coordinated report then costs the menu at most
    this many entries a step; an entry truly broken still leaves within a
    few steps."""
    return max(2, int(entries) // 10)


def retire_minimum(field: int, compiled: int) -> int:
    """The coldkeys a health rule needs in a field of `field` serving
    coldkeys: a majority of them, at most `compiled` and at least
    RETIRE_MIN_FLOOR."""
    return max(RETIRE_MIN_FLOOR, min(int(compiled), int(field) // 2 + 1))


def health_retires(h: HealthStat, params: MenuParams) -> str | None:
    """The health reason `h` retires its entry for, or None (section 9.2):
    failing when at least `retire_fail_min_coldkeys` coldkeys are eligible
    (in a smaller field, `retire_minimum` of it) and more than half of them
    are failing on it; gone when none of its pinned hosts has been listed for
    `retire_gone_epochs` consecutive epochs. A majority of coldkeys, not of
    calls or hotkeys, is what one operator cannot supply."""
    need = (params.retire_fail_min_coldkeys if h.field is None
            else retire_minimum(h.field, params.retire_fail_min_coldkeys))
    if h.eligible >= need and 2 * h.failing > h.eligible:
        return REASON_FAILING
    if h.unlisted_epochs >= params.retire_gone_epochs:
        return REASON_GONE
    return None


@dataclass(frozen=True)
class AccuracyStat:
    """An entry's measured accuracy over its window pairs of revision `rev`,
    weighted to the scored mix: the rate and its one-sided 95% lower bound,
    in parts per million."""

    entry_id: str
    rev: int
    pairs: int
    rate_ppm: int
    lower_ppm: int


@dataclass(frozen=True)
class EntryUse:
    """How many scored-slice picks went to an entry over the window, by heads
    that held the seat or were senior bars (section 9.2)."""

    entry_id: str
    picks: int


@dataclass(frozen=True)
class CallTally:
    """One coldkey's exploration calls to an entry over the epochs a gate
    reads, and how many of them failed (each failed call was retried once
    inside the TD)."""

    coldkey: str
    calls: int
    failed: int


@dataclass(frozen=True)
class GateStat:
    """What the gates read of one provisional entry's revision `rev`, over
    the window epochs from its join through the scored epoch
    (menu_inputs.gate_stats): each coldkey's calls and failures, its labels
    by category (`MeasureCell`), and the one-sided 95% upper bound, in parts
    per million rounded up, on its accuracy weighted to the scored mix."""

    entry_id: str
    rev: int
    coldkeys: tuple[CallTally, ...]
    cells: tuple[MeasureCell, ...]
    upper_ppm: int
    lower_ppm: int = 0               # the one-sided 95% lower bound, rounded down


@dataclass(frozen=True)
class StepInputs:
    """Everything `step` reads besides the state: the published inputs of
    scored epoch `epoch` (section 4.4)."""

    epoch: int                                   # k: the epoch being scored
    nonce: bytes                                 # epoch k's nonce
    # The signed lab and host list the node holds, as verify_signed_list
    # returned it: the one the state names, or a newer one, which the step
    # adopts. Always given. What the list covers is never an input: the step
    # applies the compiled rules under it.
    lab_list: menu_rules.MenuAllowlist | None = None
    # The signed lab and host list the state names (the list in force), as
    # verify_signed_list returned it. Read only when `lab_list` is newer and
    # is held back (adopting it would leave fewer than `min_entries`); it
    # must then be the list the state names.
    lab_list_in_force: menu_rules.MenuAllowlist | None = None
    # The signed device list the node holds, as verify_signed_list returned
    # it (the one in force or a newer one), or None when it holds none.
    device_list: menu_rules.DeviceList | None = None
    host_reads: tuple[HostRead, ...] = ()
    token_stats: tuple[TokenStat, ...] = ()
    explore_counts: tuple[ExploreCount, ...] = ()
    # Each entry's graded scored-question cells over the window, for the
    # pick-cost table (W13). An entry with none, or with cells only of
    # another revision, is priced at its call cost after a refresh.
    pick_stats: tuple[PickStat, ...] = ()
    health: tuple[HealthStat, ...] = ()
    accuracy: tuple[AccuracyStat, ...] = ()
    use: tuple[EntryUse, ...] = ()
    use_total: int = 0                           # all counted picks: the denominator of use
    # What the gates read of each provisional entry the step decides.
    gates: tuple[GateStat, ...] = ()


@dataclass(frozen=True)
class Event:
    """One thing a step did, for the reveal: what, to which id, and why."""

    kind: str
    subject: str
    detail: str = ""


@dataclass(frozen=True)
class StepResult:
    state: MenuState
    events: tuple[Event, ...]


# --- the step --------------------------------------------------------------------


def _keyed(items, key, what: str) -> dict:
    out = {}
    for x in items:
        k = key(x)
        if k in out:
            raise ValueError(f"step inputs: {what} {k!r} is given twice")
        out[k] = x
    return out


def _price_and_limit(req: Entry, reads: Mapping[tuple[str, str], HostRead], *,
                     min_voters: int) -> tuple[int, int, int] | None:
    """(price_in, price_out, max_tokens) of a request already in the state
    from its hosts' voted reads, or None for "no change": the price of the
    dearer pinned host (the higher input price and the higher output price
    among the listed ones; a voted 0 keeps the price it has) and the rules'
    answer limit over the listed ones (`menu_rules.answer_limit(on_menu=True)`). Every pinned host needs a read
    with enough voters. A host the reads show as no longer listed is skipped,
    provided another is listed; with none listed nothing changes, and the
    health rule retires the entry if that lasts."""
    got = []
    for host in req.provider_order:
        r = reads.get((req.model, host))
        if r is None or r.voters < min_voters:
            return None
        got.append(r)
    listed = [r for r in got if r.listed]
    if not listed:
        return None
    # A voted price of 0 is no change: a request in the state is never priced
    # at 0 (an entry with no expected cost is a menu no node can draw on).
    price_in = price_step(max(r.price_in for r in listed) or req.price_in, req.price_in)
    price_out = price_step(max(r.price_out for r in listed) or req.price_out, req.price_out)
    return (price_in, price_out,
            menu_rules.answer_limit(req.provider_order, {r.host: r.limit for r in listed},
                                    on_menu=True))


def price_step(new: int, old: int) -> int:
    """`new`, moved no further than a factor of MENU_PRICE_STEP from `old`:
    one refresh's move of a price."""
    return min(max(int(new), -(-int(old) // MENU_PRICE_STEP)), int(old) * MENU_PRICE_STEP)


def _price_within_step(new: int, old: int) -> bool:
    return price_step(new, old) == new


def _check_read(r: HostRead) -> None:
    _str(r.model, "host read model")
    _str(r.host, "host read host")
    for f in ("voters", "price_in", "price_out", "limit"):
        _int(getattr(r, f), f"host read {r.model} {r.host}: {f}")
    if not isinstance(r.listed, bool):
        raise ValueError(f"host read {r.model} {r.host}: `listed` must be a boolean")


def _token_cells(cells: Iterable[MeasureCell]) -> tuple[TokenCell, ...]:
    return tuple(TokenCell(c.category, c.answered, c.tokens_in, c.tokens_out) for c in cells)


def _merge_cells(have: tuple[MeasureCell, ...], add: tuple[MeasureCell, ...]) -> tuple[MeasureCell, ...]:
    out = {c.category: c for c in have}
    for c in add:
        old = out.get(c.category)
        out[c.category] = c if old is None else MeasureCell(
            c.category, old.calls + c.calls, old.answered + c.answered, old.correct + c.correct,
            old.tokens_in + c.tokens_in, old.tokens_out + c.tokens_out)
    return tuple(out[c] for c in sorted(out))


def step(state: MenuState, inputs: StepInputs, params: MenuParams) -> MenuState:
    """MenuState(k + lag) from MenuState(k + lag - 1) and the inputs of scored
    epoch k. Pure and exact; see `step_detail`."""
    return step_detail(state, inputs, params).state


def step_detail(state: MenuState, inputs: StepInputs, params: MenuParams) -> StepResult:
    """The derivation step of section 4.4, with what it did.

    In order: (1) adopt a newer signed list and retire what the compiled
    rules refuse under it, or hold it back when it would leave fewer than
    `min_entries`; (2) prices and answer limits from the epoch's voted
    endpoint reads; (3) token means, exploration-pair counts and the
    pick-cost table from the window; (4) health retirements (never of a
    provisional entry, which the gates decide); (5) the gates, on every
    provisional entry that joined by the scored epoch: full from the new
    epoch, or retired with the gate's reason, the cooldown on its id and on
    its proposer's coldkey; (6) when the menu is then at its cap, the
    least-used entry that is not shielded is displaced, so that one place
    stays free for a join; (7) `serial` +1 if anything changed. No entry
    joins in a step: entries join at a boundary (`apply_joins`).

    Steps 2 and 3 are the refresh. They apply only when the new epoch is a
    refresh epoch (`refresh_due`); otherwise prices, answer limits, token
    means, counts and pick-cost tables stay as published, whatever else the
    step does (an entry the gates make full takes the values its labels and
    the epoch's votes give). A refreshed
    price moves at most a factor of `MENU_PRICE_STEP` up or down. The
    cadence is read from the state's own epoch and a compiled constant, so
    the step stays a pure function.

    The result depends on the state, the inputs and the parameters only, and
    not on the order any input collection arrives in. Inputs that contradict
    themselves (a key twice, a value out of range, the wrong epoch, a list
    that is neither the one in force nor a newer one) are a ValueError;
    inputs about ids the state no longer holds are ignored.
    """
    validate(state)
    return _step(state, inputs, params, refresh=refresh_due(state.epoch + 1, params))


def _step(state: MenuState, inputs: StepInputs, params: MenuParams, *, refresh: bool) -> StepResult:
    """`step_detail`, with steps 2 and 3 applied only when `refresh`. Their
    inputs are checked either way."""
    k = _int(inputs.epoch, "the scored epoch")
    if state.epoch != k + params.menu_lag - 1:
        raise ValueError(f"the step for scored epoch {k} starts from the menu of epoch "
                         f"{k + params.menu_lag - 1}, not {state.epoch}")
    if not isinstance(inputs.nonce, (bytes, bytearray)) or not inputs.nonce:
        raise ValueError("the step needs the scored epoch's nonce")
    new_epoch = state.epoch + 1
    cooldown_until = new_epoch + params.cooldown_epochs
    events: list[Event] = []

    entries: dict[str, Entry] = {e.id: e for e in state.entries}
    records: dict[str, Provisional] = {p.id: p for p in state.provisional}
    retired: dict[str, Retired] = {r.id: r for r in state.retired}
    labs, devices = state.labs, state.devices
    coldkeys: dict[str, int] = state.cooldowns()

    def retire(entry_id: str, reason: str) -> None:
        """Record why an id left, and its cooldown. A record is keyed by id:
        the newest one replaces the last."""
        retired[entry_id] = Retired(entry_id, new_epoch, reason, cooldown_until)

    def retire_entry(entry_id: str, reason: str) -> None:
        del entries[entry_id]
        retire(entry_id, reason)
        records.pop(entry_id, None)
        events.append(Event("retired", entry_id, reason))

    # 1. Signed lists. The node hands in the lab and host list it holds. A
    # newer one is adopted, and whatever the compiled rules refuse under it
    # leaves, provisional entries included. Nothing here is told what a list
    # refuses.
    if inputs.lab_list is None or isinstance(inputs.lab_list, AllowlistRef):
        raise ValueError("the step needs the signed lab and host list: the one in force or a "
                         "newer one")
    allow = held_list(inputs.lab_list)

    def list_reason(refusal: menu_rules.Refusal, what: str) -> str:
        if refusal.code not in LIST_REFUSAL_REASONS:
            raise ValueError(f"the menu holds {what}, which the compiled rules refuse whatever "
                             f"the list ({refusal})")
        return LIST_REFUSAL_REASONS[refusal.code]

    refused_entries = {}
    if allow.serial > labs.serial:
        for eid in sorted(entries):
            refusal = menu_rules.check_entry(_entry_dict(entries[eid]), allow)
            if refusal is not None:
                refused_entries[eid] = list_reason(refusal, f"entry {eid}")
    full_left = full_count(e for eid, e in entries.items() if eid not in refused_entries)
    if allow.serial > labs.serial and full_left < params.min_entries:
        # A list that would leave fewer than the minimum of full entries is
        # not adopted (a provisional entry it leaves is no place under the
        # floor: the gates below may retire it in this very step): the list
        # in force stays, so the menu can grow by joins until the newer list
        # leaves enough. It is adopted in the first step where it does; until
        # then the entries it would refuse stay on the menu.
        events.append(Event("lab_list_held", str(allow.serial),
                            f"it would leave {full_left} full entries, "
                            f"fewer than {params.min_entries}"))
        allow = None
        if inputs.lab_list_in_force is not None:
            in_force = held_list(inputs.lab_list_in_force)
            if (in_force.serial, in_force.sha256) != (labs.serial, labs.sha256):
                raise ValueError(f"the list in force handed to the step is serial "
                                 f"{in_force.serial}, not the one the menu names "
                                 f"(serial {labs.serial})")
            allow = in_force
    elif allow.serial > labs.serial:
        labs = AllowlistRef(allow.serial, allow.sha256)
        events.append(Event("lab_list", str(allow.serial), allow.sha256))
        for eid, reason in sorted(refused_entries.items()):
            retire_entry(eid, reason)
    elif (allow.serial, allow.sha256) != (labs.serial, labs.sha256):
        raise ValueError(f"the step needs the list in force (serial {labs.serial}, "
                         f"{labs.sha256[:16]}) or a newer one, not serial {allow.serial}")

    if inputs.device_list is not None:
        # The device list the node holds: the one in force, or a newer one,
        # which is adopted. Never a bare reference (held_device_list).
        new = held_device_list(inputs.device_list)
        if new.serial > devices.serial:
            devices = AllowlistRef(new.serial, new.sha256)
            events.append(Event("device_list", str(new.serial), new.sha256))
        elif (new.serial, new.sha256) != (devices.serial, devices.sha256):
            raise ValueError(f"the step needs the device list in force (serial {devices.serial}, "
                             f"{devices.sha256[:16]}) or a newer one, not serial {new.serial}")

    # 2. Prices and answer limits, for every entry.
    for r in inputs.host_reads:
        _check_read(r)
    reads = _keyed(inputs.host_reads, lambda r: (r.model, r.host), "host read")
    for key in sorted(entries) if refresh else ():
        x = entries[key]
        got = _price_and_limit(x, reads, min_voters=params.endpoint_min_voters)
        if got is None or got == (x.price_in, x.price_out, x.max_tokens):
            continue
        if menu_rules.check_price(got[0], got[1]) is not None:
            # Above the ceiling: never explored at that price. The entry
            # leaves (its id's cooldown runs), unless that would leave fewer
            # than `min_entries` entries, or, for a full entry, fewer than
            # `min_entries` full ones (as health retirements count: the gates
            # may yet retire provisional ones): then nothing about it moves
            # this step.
            if len(entries) <= params.min_entries or (not x.provisional and sum(
                    not y.provisional for y in entries.values()) <= params.min_entries):
                events.append(Event("retire_waits", x.id, REASON_PRICE_CEILING))
                continue
            retire_entry(key, REASON_PRICE_CEILING)
            continue
        entries[key] = dataclasses.replace(x, price_in=got[0], price_out=got[1], max_tokens=got[2])
        events.append(Event("price", x.id, f"{got[0]}/{got[1]} limit {got[2]}"))

    # 3. Token means, from the pairs of each entry's current revision.
    for eid, stat in sorted(_keyed(inputs.token_stats, lambda s: s.entry_id, "token stat").items()):
        e = entries.get(eid)
        if not refresh or e is None or stat.rev != e.rev:
            continue
        mean_in, mean_out, pairs = weighted_token_means(stat.cells, params.mix)
        if pairs < params.token_min_pairs or (mean_in, mean_out, pairs) == (
                e.mean_in, e.mean_out, e.token_pairs):
            continue
        if _cost(e.price_in, e.price_out, mean_in, mean_out) <= 0:
            continue                      # means that make a call cost nothing are no change
        entries[eid] = dataclasses.replace(e, mean_in=mean_in, mean_out=mean_out,
                                           token_pairs=_int(pairs, f"{eid}: pairs"))
        events.append(Event("token_means", eid, f"{mean_in}/{mean_out} on {pairs}"))

    # 3b. Each entry's exploration pairs in the window, for its current
    # revision: the count is published as handed in (section 8). An entry
    # with no count keeps the one it has.
    counts = _keyed(inputs.explore_counts, lambda c: c.entry_id, "exploration count")
    for eid, c in sorted(counts.items()):
        _int(c.rev, f"exploration count {eid}: rev", 1)
        _int(c.pairs, f"exploration count {eid}: pairs")
        e = entries.get(eid)
        if not refresh or e is None or c.rev != e.rev or c.pairs == e.explore_pairs:
            continue
        entries[eid] = dataclasses.replace(e, explore_pairs=c.pairs)

    # 3c. The pick-cost table, from each entry's graded scored-question cells
    # of its current revision, shrunk toward the call means just refreshed
    # (W13). Recomputed whole at a refresh: an entry with no cells of its
    # revision (none handed in, or only another revision's) is priced at its
    # call cost.
    picks = _keyed(inputs.pick_stats, lambda s: s.entry_id, "pick stat")
    for eid, s in sorted(picks.items()):
        _int(s.rev, f"pick stat {eid}: rev", 1)
        if not isinstance(s.cells, tuple):
            raise ValueError(f"pick stat {eid}: cells must be a tuple")
        for c in s.cells:
            _check_pick_cell(c, eid)
        cats = [c.category for c in s.cells]
        if len(set(cats)) != len(cats) or len(cats) > PICK_MAX_CATEGORIES:
            raise ValueError(f"pick stat {eid}: a category appears twice, or more than "
                             f"{PICK_MAX_CATEGORIES} categories")
    for eid in sorted(entries) if refresh else ():
        e, s = entries[eid], picks.get(eid)
        table = pick_table(s.cells if s is not None and s.rev == e.rev else (), e, params)
        if table != e.pick:
            entries[eid] = dataclasses.replace(e, pick=table)
            events.append(Event("pick_cost", eid, "call cost" if table is None else
                                f"{table.mean_in}/{table.mean_out} on {table.cells} cells"))

    # 4. Health. Use plays no part: an entry that fails or is gone leaves; at
    # most `health_retirement_bound` of them a step, in id order, and never
    # below `min_entries`. The rest wait: the next step that sees them still
    # failing or gone retires them. A provisional entry is the gates'.
    allowed = health_retirement_bound(len(entries))
    for eid, h in sorted(_keyed(inputs.health, lambda s: s.entry_id, "health stat").items()):
        for f in ("eligible", "failing", "unlisted_epochs"):
            _int(getattr(h, f), f"health {eid}: {f}")
        if h.field is not None:
            _int(h.field, f"health {eid}: field")
            if h.eligible > h.field:
                raise ValueError(f"health {eid}: more coldkeys eligible than serving")
        if h.failing > h.eligible:
            raise ValueError(f"health {eid}: more coldkeys failing than eligible")
        e = entries.get(eid)
        if e is None or h.rev != e.rev or e.provisional:
            continue
        reason = health_retires(h, params)
        if reason is None:
            continue
        if allowed <= 0 or sum(not x.provisional for x in entries.values()) <= params.min_entries:
            events.append(Event("retire_waits", eid, reason))
            continue
        retire_entry(eid, reason)
        allowed -= 1

    # 5. The gates, on every provisional entry that joined by the scored
    # epoch (FAST_CROWN_V4.md 18): on the field's labels of its revision from
    # its join through k. It becomes full from the new epoch, or it leaves
    # with the gate's record, and its proposer's coldkey waits the cooldown.
    # An entry that joined after k stays provisional: its labels are a later
    # step's.
    accuracy = _keyed(inputs.accuracy, lambda s: s.entry_id, "accuracy stat")
    for s in accuracy.values():
        for f in ("pairs", "rate_ppm", "lower_ppm"):
            _int(getattr(s, f), f"accuracy {s.entry_id}: {f}")
    gates = _keyed(inputs.gates, lambda g: g.entry_id, "gate stat")
    for g in gates.values():
        _check_gate(g)
    for eid in sorted(records):
        rec, e = records[eid], entries.get(eid)
        if e is None or e.joined_epoch > k:
            continue
        g = gates.get(eid)
        if g is not None and g.rev != e.rev:
            g = None
        reason = _gate_verdict(e, g, entries=entries, accuracy=accuracy, params=params)
        if reason is not None:
            retire_entry(eid, reason)
            coldkeys[rec.coldkey] = max(cooldown_until, coldkeys.get(rec.coldkey, 0))
            events.append(Event("gate_failed", eid, reason))
            continue
        del records[eid]
        entries[eid] = _made_full(e, g, reads, new_epoch=new_epoch, params=params,
                                  priced=refresh)
        events.append(Event("full", eid, f"scored from {new_epoch}"))

    # 6. A place kept free for a join: when the menu is at its cap, the
    # least-used entry is displaced (lowest use, then lowest measured
    # accuracy, an unmeasured entry first, then id). Never a shielded entry
    # (provisional, or full for fewer than `shield_epochs` epochs), never one
    # above the use floor, never without any counted pick to judge use by, and
    # never when it would leave fewer than `min_entries` full entries (only a
    # full entry is displaced; the provisional ones hold no place under the
    # floor): then it waits, and the menu reports itself full.
    use = _keyed(inputs.use, lambda u: u.entry_id, "entry use")
    for u in use.values():
        _int(u.picks, f"use {u.entry_id}: picks")
    use_total = _int(inputs.use_total, "use_total")
    if any(u.picks > use_total for u in use.values()):
        raise ValueError("step inputs: an entry has more picks than there were")
    if len(entries) >= params.menu_cap and full_count(entries.values()) <= params.min_entries:
        events.append(Event("menu_full", str(len(entries)),
                            f"displacing would leave fewer than {params.min_entries} full "
                            "entries; joins wait for a place"))
    elif len(entries) >= params.menu_cap:
        def displaceable(e: Entry) -> bool:
            picks = use[e.id].picks if e.id in use else 0
            return (use_total > 0 and not e.shielded(new_epoch, params.shield_epochs)
                    and picks * BP <= params.retire_use_floor_bp * use_total)

        def rank(e: Entry):
            s = accuracy.get(e.id)
            rate = s.rate_ppm if s is not None and s.rev == e.rev else -1
            return (use[e.id].picks if e.id in use else 0, rate, e.id)

        candidates = [e for e in entries.values() if displaceable(e)]
        if candidates and len(entries) == params.menu_cap:
            retire_entry(min(candidates, key=rank).id, REASON_DISPLACED)
        elif not candidates:
            events.append(Event("menu_full", str(len(entries)),
                                "no entry can be displaced; joins wait for a place"))

    # 7. The new state. `retired` holds `retired_keep` records. A cooldown that
    # is still running is never dropped for room: only records whose cooldown
    # can refuse no further join go, the oldest first.
    excess = len(retired) - params.retired_keep
    for r in sorted(retired.values(), key=lambda r: (r.epoch, r.id)):
        if excess <= 0:
            break
        if not _cooldown_runs(r.cooldown_until, k):
            del retired[r.id]
            excess -= 1
    kept = retired.values()
    out = MenuState(
        epoch=state.epoch, serial=state.serial,
        entries=tuple(entries[i] for i in sorted(entries)),
        retired=tuple(sorted(kept, key=lambda r: r.id)), labs=labs, devices=devices,
        # A coldkey's cooldown leaves the state once it has run out.
        coldkey_cooldowns=tuple(ColdkeyCooldown(ck, coldkeys[ck]) for ck in sorted(coldkeys)
                                if _cooldown_runs(coldkeys[ck], k)),
        provisional=tuple(records[i] for i in sorted(records)))
    out = dataclasses.replace(out, epoch=new_epoch,
                              serial=state.serial + (0 if out == state else 1))
    validate(out)
    return StepResult(out, tuple(events))


def _check_gate(g: GateStat) -> None:
    """A gate statistic the step can read: integers in range, coldkeys and
    categories each once and sorted, failures within calls."""
    if not isinstance(g, GateStat):
        raise ValueError("gate stats must be GateStat values")
    _str(g.entry_id, "gate stat entry")
    _int(g.rev, f"gate stat {g.entry_id}: rev", 1)
    _int(g.upper_ppm, f"gate stat {g.entry_id}: upper_ppm")
    _int(g.lower_ppm, f"gate stat {g.entry_id}: lower_ppm")
    if not g.lower_ppm <= g.upper_ppm <= PPM:
        raise ValueError(f"gate stat {g.entry_id}: bounds out of order or above one")
    if not isinstance(g.coldkeys, tuple) or not isinstance(g.cells, tuple):
        raise ValueError(f"gate stat {g.entry_id}: coldkeys and cells must be tuples")
    for c in g.coldkeys:
        if not isinstance(c, CallTally):
            raise ValueError(f"gate stat {g.entry_id}: coldkeys must hold CallTally values")
        _str(c.coldkey, f"gate stat {g.entry_id}: coldkey")
        _int(c.calls, f"gate stat {g.entry_id}: calls")
        _int(c.failed, f"gate stat {g.entry_id}: failed")
        if c.failed > c.calls:
            raise ValueError(f"gate stat {g.entry_id}: more failed calls than calls")
    cks = [c.coldkey for c in g.coldkeys]
    if cks != sorted(set(cks)):
        raise ValueError(f"gate stat {g.entry_id}: coldkeys must be sorted with none repeated")
    _check_cells(g.cells, g.entry_id)


def median_answered(g: GateStat | None) -> Fraction:
    """The median, over the coldkeys that made at least one call to the
    entry, of each coldkey's answered share of its calls (the mean of the
    two middle ones for an even count). 0 with no call at all."""
    shares = sorted(Fraction(c.calls - c.failed, c.calls) for c in (g.coldkeys if g else ())
                    if c.calls > 0)
    if not shares:
        return Fraction(0)
    mid = len(shares) // 2
    return shares[mid] if len(shares) % 2 else (shares[mid - 1] + shares[mid]) / 2


def _gate_verdict(e: Entry, g: GateStat | None, *, entries: Mapping[str, Entry],
                  accuracy: Mapping[str, AccuracyStat], params: MenuParams) -> str | None:
    """The gate a provisional entry fails on its labels, or None (it becomes
    full). In order: answered (the median coldkey's answered share, at least
    `min_answered_bp`); cost (its measured call costs something at its
    price); category (some category with `min_category_graded` labels at the
    accuracy floor); useful (no full entry with `useful_min_pairs` labels,
    costing no more a call, whose lower bound is above this entry's upper);
    sibling (no full entry of the same model with `useful_min_pairs` labels
    whose accuracy is inside this entry's interval widened by
    SIBLING_MARGIN_PPM, unless this entry is cheaper a call by more than
    SIBLING_CHEAPER_BP)."""
    if median_answered(g) * BP < params.min_answered_bp:
        return REASON_GATE_ANSWERED
    mean_in, mean_out, pairs = weighted_token_means(_token_cells(g.cells), params.mix)
    if not pairs:
        return REASON_GATE_CATEGORY
    cost = _cost(e.price_in, e.price_out, mean_in, mean_out)
    if cost <= 0:
        return REASON_GATE_COST
    if not any(c.answered >= params.min_category_graded
               and c.correct * BP >= params.min_category_accuracy_bp * c.answered
               for c in g.cells):
        return REASON_GATE_CATEGORY
    for oid in sorted(entries):
        o, s = entries[oid], accuracy.get(oid)
        if (oid != e.id and not o.provisional and s is not None and s.rev == o.rev
                and s.pairs >= params.useful_min_pairs and entry_cost(o) <= cost
                and s.lower_ppm > g.upper_ppm):
            return REASON_GATE_USEFUL
    for oid in sorted(entries):
        o, s = entries[oid], accuracy.get(oid)
        if (oid != e.id and o.model == e.model and not o.provisional and s is not None
                and s.rev == o.rev and s.pairs >= params.useful_min_pairs
                and g.lower_ppm - SIBLING_MARGIN_PPM <= s.rate_ppm <= g.upper_ppm + SIBLING_MARGIN_PPM
                and cost * BP > entry_cost(o) * (BP - SIBLING_CHEAPER_BP)):
            return REASON_GATE_SIBLING
    return None


def _made_full(e: Entry, g: GateStat, reads: Mapping[tuple[str, str], HostRead], *,
               new_epoch: int, params: MenuParams, priced: bool = False) -> Entry:
    """A provisional entry the gates passed, as a full entry: scored from
    `new_epoch`; its measured rate (k + 1) / (n + 2) over its labels; its
    token means from its labels when they rest on `token_min_pairs` (else
    the join's); its price and answer limit from the epoch's voted reads when
    every pinned host was voted on (each price within the price step), unless
    the step's refresh has just priced it (`priced`). Its picks are priced at
    its call cost (no pick table) until the next refresh epoch."""
    rate = measured_rate(sum(c.correct for c in g.cells), sum(c.answered for c in g.cells))
    out = dataclasses.replace(e, scored_from=new_epoch, measured_rate=rate, pick=None)
    mean_in, mean_out, pairs = weighted_token_means(_token_cells(g.cells), params.mix)
    if pairs >= params.token_min_pairs and _cost(e.price_in, e.price_out, mean_in, mean_out) > 0:
        out = dataclasses.replace(out, mean_in=mean_in, mean_out=mean_out, token_pairs=pairs)
    got = None if priced else _price_and_limit(out, reads, min_voters=params.endpoint_min_voters)
    if got is not None and menu_rules.check_price(got[0], got[1]) is None:
        # (A voted price above the ceiling is not taken here; the next
        # refresh retires the entry for it, `step_detail` step 2.)
        out = dataclasses.replace(out, price_in=got[0], price_out=got[1], max_tokens=got[2])
    return out


def _cooldown_runs(until: int, k: int) -> bool:
    """Whether a cooldown can still refuse a join after the step for scored
    epoch k. A join at epoch J is refused while J is before `until`, and the
    menus this step's leads to apply from k + 2."""
    return until > k + 1
