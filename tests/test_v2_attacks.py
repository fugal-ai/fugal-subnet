"""Adversarial tests for protocol v2 consensus: what a miner or a coalition would try.

Each test implements one attack against the shipped v2 code (shared grades,
the champion seat, the copy rule, the seat payout, the cross-flip state) and
passes only when the attack is REFUSED for the reason the docstring names.
docs/INVARIANTS.md sections are cited per test.

Attacks are modelled at their upper bound where the real lever is weaker: a
coalition voter's grade and completion tokens are set by the test, although on
the subnet they are attested model output it can only bias (by provider choice
on its own OpenRouter account), not choose.
"""
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fugal_subnet.champion import SeatState, Window, pair_units
from fugal_subnet.copy_rule import Commitment, CopyState, v2_seniority
from fugal_subnet.evidence import Evidence, burn_in_complete
from fugal_subnet.shared_grades import ScoredAnswer, shared_cell_grades, shared_cell_tokens
from fugal_subnet.tee.proof import QuestionResult
from fugal_subnet.v2_epoch import run_v2_epoch, seat_params_from_config
from tests.test_v2_epoch import ACC, BPE, CATEGORY, MODELS, PRICES, H, burned, oracle

PARAMS = seat_params_from_config()
SHARES = (0.85, 0.10, 0.05)
# The harness answer budget: no attested completion can be longer.
MAX_COMPLETION = 16384
BENCHES = ("mmlu", "math", "aime")          # knowledge, math, hard (tests/test_v2_epoch.CATEGORY)


# --- a v2 epoch with any question count, any routing, and adversarial voters --------


def _questions(epoch, q):
    return [{"question_id": f"e{epoch}q{i}", "prompt": "p" * (10 + i % 50),
             "benchmark": BENCHES[(3 * i) // q]} for i in range(q)]


def _difficulty(epoch, q):
    return np.random.default_rng(10_000 + epoch).random(q)


def _always(model):
    return lambda d: [model] * len(d)


def _noisy_oracle(p):
    """The oracle, except a fixed per-epoch fraction p of questions go to m/cheap."""
    def policy(d):
        flip = np.random.default_rng(int(d[0] * 1e9) + 7).random(len(d)) < p
        return ["m/cheap" if f else m for f, m in zip(flip, oracle(d))]
    return policy


def _tokens(model):
    return 100 + 10 * MODELS.index(model)


@dataclass
class Srv:
    """One UID serving one head this epoch."""

    uid: int
    head: str
    coldkey: str
    block: int
    routed: list
    wrong: tuple = ()                 # questions it answers wrong whatever the truth
    tokens: dict | None = None        # {question index: completion tokens}
    record: object = None
    hotkey: str | None = None

    @property
    def hk(self):
        return self.hotkey or f"hk{self.uid}"


def _proof(epoch, q, s: Srv):
    d = _difficulty(epoch, q)
    toks = s.tokens or {}
    return SimpleNamespace(scored_results=[
        QuestionResult(
            question_id=f"e{epoch}q{i}", routed_model=m,
            correct=i not in s.wrong and bool(d[i] < ACC[m]), cost_usd=0.0,
            response_hash="x", completion_tokens=int(toks.get(i, _tokens(m))), failed=False,
        )
        for i, m in enumerate(s.routed)
    ])


def _unburned(h):
    """A record whose E'' burn-in counter restarted (a miss, or a new head)."""
    return SimpleNamespace(evidence=Evidence(weights_hash=h, n_total=10_000.0,
                                             n_correct=8000.0, burn_in_n=0.0))


def _epoch(epoch, q, servers, *, prev=None, seat=None, commits=None, burn_in=0,
           floor=None, salt=""):
    """One v2 epoch (chain epoch `epoch` + 1), threading state from `prev`."""
    e = epoch + 1
    if commits is None:
        commits = [Commitment(s.hk, s.coldkey, s.head, s.block) for s in servers]
    state = (dict(window=prev.window, seat=prev.decision.state, copy_state=prev.copy_state)
             if prev is not None else
             dict(window=Window.empty(48), seat=seat or SeatState(), copy_state=CopyState()))
    return run_v2_epoch(
        epoch_index=e, epoch_id=f"e{e:08d}", boundary_block=e * BPE, blocks_per_epoch=BPE,
        questions=_questions(epoch, q), pool_models=MODELS, prices=PRICES,
        verified_proofs={s.uid: _proof(epoch, q, s) for s in servers},
        head_hashes={s.uid: s.head for s in servers},
        coldkey_of={s.uid: s.coldkey for s in servers},
        hotkey_of={s.uid: s.hk for s in servers},
        records={s.uid: s.record or burned(s.head) for s in servers},
        commitments=commits, burn_uid=0, params=PARAMS, shares=SHARES,
        burn_in_questions=burn_in, bench_category=CATEGORY,
        seniority_floor=floor, seniority_salt=salt, **state,
    )


def _w(r):
    return dict(zip(r.uids, r.weights))


# --- 1. low-voter cell capture --------------------------------------------------------
#
# INVARIANTS, "I4 — shared grades: a cell is a vote of operators" and "I3/I4 — v2
# pays from the paired rule alone" (rule 3, spend is measured): one vote per
# coldkey, the median across coldkeys, for grades and completion tokens alike.


def _cell(n_honest, coalition_hotkeys_per_coldkey):
    """One (question, model) cell: n_honest coldkeys answer correctly at 100
    tokens; each coalition coldkey answers wrong at the harness's maximum."""
    honest = [ScoredAnswer("q", "m", f"honest{i}", True, False, 100) for i in range(n_honest)]
    coalition = [ScoredAnswer("q", "m", f"coal{j}", False, False, MAX_COMPLETION)
                 for j, n in enumerate(coalition_hotkeys_per_coldkey) for _ in range(n)]
    answers = honest + coalition
    return shared_cell_grades(answers)[("q", "m")], shared_cell_tokens(answers)[("q", "m")]


@pytest.mark.parametrize("n_honest,k", [(n, k) for n in (1, 2, 3, 5) for k in range(n)])
def test_a1_a_coalition_smaller_than_a_cells_honest_voters_cannot_move_it(n_honest, k):
    """ATTACK: k coldkeys route the champion's question to the champion's model
    and answer it wrong at the longest completion the harness allows, to lower
    its grade and raise its spend.
    DEFENCE: the cell is the median over coldkeys, so k below the cell's honest
    voters moves neither its grade nor its tokens (INVARIANTS: "A cell moves only
    when at least half of its voting coldkeys answer differently")."""
    grade, tokens = _cell(n_honest, [1] * k)
    assert grade == 1.0 and tokens == 100.0


@pytest.mark.parametrize("hotkeys", [1, 5, 50])
@pytest.mark.parametrize("n_honest", [1, 2, 3])
def test_a1_sybil_hotkeys_under_one_coldkey_are_one_vote(n_honest, hotkeys):
    """ATTACK (sybil variant): one operator registers many hotkeys and has every
    one of them vote the champion's cell down.
    DEFENCE: one vote per coldkey (the mean over its hotkeys): fifty hotkeys move
    the cell exactly as far as one does, and against two or more honest
    coldkeys not at all (INVARIANTS I4, "Hotkeys are not a lever")."""
    assert _cell(n_honest, [hotkeys]) == _cell(n_honest, [1])
    if n_honest >= 2:
        assert _cell(n_honest, [hotkeys]) == (1.0, 100.0)


@pytest.mark.parametrize("coalition", [[1], [1, 1], [10]], ids=["k1", "k2", "sybil-1x10"])
def test_a1_a_minority_coalition_on_the_champions_cells_does_not_move_the_seat(coalition):
    """ATTACK, end to end: champion C (an oracle router) is seated; challenger X
    routes everything to m/mid and is clearly worse. Two honest operators route
    like C, so C's cells where it differs from X have three honest coldkeys. A
    coalition routes like C and, on exactly the questions where C and X differ,
    answers wrong at the maximum completion, for five epochs (X's burn-in and
    paired floor).
    DEFENCE: below the honest voters the cells are untouched, so C's window
    grades and spends are byte-identical to an epoch with no coalition, X's NV
    is what it would honestly be, and C keeps the seat and the pay (INVARIANTS
    I4 shared grades; I3 "one champion")."""
    q, epochs = 300, 6

    def play(with_coalition):
        r = None
        for e in range(epochs):
            d = _difficulty(e, q)
            rc, rx = oracle(d), _always("m/mid")(d)
            differ = tuple(i for i in range(q) if rc[i] != rx[i])
            servers = [Srv(5, H("C"), "ckC", 5, rc), Srv(6, H("X"), "ckX", 6, rx),
                       Srv(7, H("f1"), "ckF1", 7, rc), Srv(8, H("f2"), "ckF2", 8, rc)]
            if with_coalition:
                uid = 20
                for j, hotkeys in enumerate(coalition):
                    for _ in range(hotkeys):
                        servers.append(Srv(uid, H(f"coal{uid}"), f"ck-coal{j}", 9 + uid, rc,
                                           wrong=differ,
                                           tokens={i: MAX_COMPLETION for i in differ}))
                        uid += 1
            r = _epoch(e, q, servers, prev=r,
                       seat=SeatState(champion=H("C"), champion_since=0))
        return r

    clean, attacked = play(False), play(True)
    assert attacked.decision.event == "hold" and attacked.decision.champion == H("C")
    assert _w(attacked) == pytest.approx({5: 1.0})
    for e in clean.window.slices:
        for head in (H("C"), H("X")):
            g0, s0 = clean.window.head_vectors(head, e)
            g1, s1 = attacked.window.head_vectors(head, e)
            assert np.array_equal(g0, g1) and np.array_equal(s0, s1)
    kw = dict(accuracy_value=PARAMS.accuracy_value, hard_weight=PARAMS.hard_weight,
              anchor_spend=PARAMS.anchor_spend)
    nv = pair_units(attacked.window, H("X"), H("C"), **kw)
    assert sum(int(x.sum()) for x in nv.by_category.values()) < 0       # X is worse, still


# --- 2. pre-flip seniority lottery ----------------------------------------------------
#
# INVARIANTS, "I3 — one champion ..." (First: "Pre-flip commits tie") and "I3/I4 —
# the v2 copy rule" (seniority is the first commit; a junior must beat every
# senior seat holder by the margin).

FLIP = BPE        # the activation block: the first v2 boundary (chain epoch 1)


def _lottery_servers(e, q, n_copies, copy_block):
    d = _difficulty(e, q)
    out = [Srv(5, H("orig"), "ck-orig", 100, oracle(d))]
    out += [Srv(10 + j, H(f"copy{j}"), f"ck-copy{j}", copy_block + j, oracle(d))
            for j in range(n_copies)]
    return out


@pytest.mark.parametrize("n_copies", [1, 4, 8])
def test_a2_copies_committed_after_the_flip_never_win_the_first_seat(n_copies):
    """ATTACK: N coldkeys each commit a byte-perturbed copy of a strong head
    (a distinct hash, identical routing) to win the first seat by tie-break.
    Here the copies land after the activation block.
    DEFENCE: commits after the flip keep their real block, so every pre-flip
    head is senior to them whatever the salt; the FIRST rule seats the most
    senior eligible head, and the copies are never paid (INVARIANTS I3, "Commits
    after the flip keep their real block")."""
    for s in range(24):
        # The flip at block 120: the original (block 100) is pre-flip, the
        # copies (150 + j) are not.
        r = _epoch(0, 60, _lottery_servers(0, 60, n_copies, 150), floor=120,
                   salt=f"0x{s:04x}")
        assert r.decision.event == "first" and r.decision.champion == H("orig")
        assert _w(r) == pytest.approx({5: 1.0})


@pytest.mark.parametrize("n_copies", [1, 4, 8])
def test_a2_copies_committed_before_the_flip_never_win_the_first_seat(n_copies):
    """ATTACK: N coldkeys each commit a byte-perturbed copy of a strong head
    BEFORE the activation block, where every commit ties at the seniority
    floor and the salted draw decided the first seat: N copies, N tickets.
    DEFENCE (2026-11-09): the empty seat goes to the best measured head
    (champion.solo_value), and an exact tie, which copies always are, is broken
    by the real first-commit block on chain before the salted draw, so the
    original, committed first, wins for every salt."""
    for s in range(24):
        r = _epoch(0, 60, _lottery_servers(0, 60, n_copies, 101), floor=120,
                   salt=f"0x{s:04x}")
        assert r.decision.event == "first" and r.decision.champion == H("orig")
        assert _w(r) == pytest.approx({5: 1.0})


@pytest.mark.parametrize("n_junk", [1, 3, 8])
def test_a2_junk_heads_buy_no_tickets_for_the_first_seat(n_junk):
    """ATTACK: N coldkeys commit cheap junk heads (always the cheap model,
    accuracy 0.55, above the 0.30 floor) before the flip, as lottery tickets
    for the first seat (they won 48-98 of 200 draws under the old rule).
    DEFENCE: the empty seat goes to the best measured head; junk never is."""
    for s in range(24):
        def servers(e, q):
            d = _difficulty(e, q)
            out = [Srv(5, H("orig"), "ck-orig", 100, oracle(d))]
            out += [Srv(20 + j, H(f"junk{j}"), f"ck-junk{j}", 101 + j, ["m/cheap"] * q)
                    for j in range(n_junk)]
            return out
        r = _epoch(0, 60, servers(0, 60), floor=120, salt=f"0x{s:04x}")
        assert r.decision.event == "first" and r.decision.champion == H("orig")


def test_a2_a_copy_never_takes_the_seat_from_a_seated_original():
    """ATTACK: four pre-flip copies of the original (distinct hashes, identical
    routing) serve beside it for 30 epochs, hoping noise or a re-ordering hands
    one of them the seat after the first decision went to the original.
    DEFENCE: taking needs the margin on the paired test; an identical router's
    NV is exactly 0, so the original holds and the copies earn nothing
    (INVARIANTS I3, "Hold": "copies of the champion never take the seat")."""
    salt = next(f"0x{s:04x}" for s in range(64)
                if _epoch(0, 60, _lottery_servers(0, 60, 4, 150), floor=FLIP,
                          salt=f"0x{s:04x}").decision.champion == H("orig"))
    r = None
    for e in range(30):
        r = _epoch(e, 60, _lottery_servers(e, 60, 4, 150), prev=r, floor=FLIP, salt=salt)
        assert r.decision.champion == H("orig")
    assert r.decision.event == "hold" and _w(r) == pytest.approx({5: 1.0})
    assert any("NV=+0.000" in m for h, m in r.decision.detail if h == H("copy0"))


# --- 3. re-serialised copy ------------------------------------------------------------
#
# INVARIANTS, "I3/I4 — the v2 copy rule" (a hash belongs to the coldkey that first
# committed it; a junior must beat every senior seat holder; ghost incumbents).

FIXTURE_HEAD = "tests/fixtures/success_bundle/head.npz"


def _members(data):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return [(i, zf.read(i.filename)) for i in zf.infolist()]


def _rezip(members, compression):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as zf:
        for info, payload in members:
            zf.writestr(info.filename, payload)
    return buf.getvalue()


def _with_extra_array(data):
    members = _members(data)
    extra = io.BytesIO()
    np.lib.format.write_array(extra, np.zeros(3, dtype=np.float32))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for info, payload in members:
            zf.writestr(info.filename, payload)
        zf.writestr("unused.npy", extra.getvalue())
    return buf.getvalue()


def test_a3_a_reserialised_head_is_a_different_hash_with_the_same_routing():
    """ATTACK: re-zip a seat holder's .npz (another compression, members in
    another order) so the hash differs while the routing is identical, or add
    an unused array.
    DEFENCE: the hash differs, so it is a different head, junior to the
    original, with no claim on its seat or age (the copy rule); an extra array
    is not a head at all (validate_head takes exactly the contract's fields)."""
    from fugal_subnet.head_eval import load_head_from_npz

    original = (Path(__file__).resolve().parents[1] / FIXTURE_HEAD).read_bytes()
    base = load_head_from_npz(original)
    members = _members(original)
    variants = {
        "deflated": _rezip(members, zipfile.ZIP_DEFLATED),
        "reordered": _rezip(list(reversed(members)), zipfile.ZIP_STORED),
    }
    for name, data in variants.items():
        assert hashlib.sha256(data).hexdigest() != hashlib.sha256(original).hexdigest(), name
        head = load_head_from_npz(data)
        assert np.array_equal(head.W, base.W) and np.array_equal(head.b, base.b), name
        assert head.models == base.models, name
    with pytest.raises(ValueError, match="unsupported head fields"):
        load_head_from_npz(_with_extra_array(original))


def _copy_world(e, *, orig_serves=True, copier_head="rezip", orig_record=None, q=60):
    d = _difficulty(e, q)
    servers = []
    if orig_serves:
        servers.append(Srv(5, H("orig"), "ck-orig", 100, oracle(d), record=orig_record))
    if copier_head is not None:
        servers.append(Srv(9, H(copier_head), "ck-copier", 150, oracle(d)))
    commits = [Commitment("hk5", "ck-orig", H("orig"), 100)]
    if copier_head is not None:
        commits.append(Commitment("hk9", "ck-copier", H(copier_head), 150))
    return servers, commits


def test_a3_a_reserialised_copy_is_not_paid_does_not_take_and_cannot_retire_the_original():
    """ATTACK: a new coldkey commits a re-zipped copy of the champion (same
    routing, new hash) and serves it for 30 epochs; then stops committing it;
    then commits the champion's EXACT bytes.
    DEFENCE: the re-zip is a junior head with NV 0 against the champion, so it
    never takes and is never paid; only the owner's commitments retire a head,
    so the copier leaving retires nothing; the exact bytes under another
    coldkey are an exact copy ("copy" verdict) and are not paid as the original
    (INVARIANTS, the copy rule: "A hash belongs to the coldkey that first
    committed it")."""
    r = None
    for e in range(30):
        servers, commits = _copy_world(e)
        r = _epoch(e, 60, servers, prev=r, commits=commits)
        assert r.decision.champion == H("orig") and 9 not in r.uids
    # The copier stops committing: the original is untouched.
    servers, commits = _copy_world(30, copier_head=None)
    r = _epoch(30, 60, servers, prev=r, commits=commits)
    assert r.decision.event == "hold" and not r.decision.state.ghosts
    assert r.copy_state.get(H("orig")).active
    # The copier serves the original's exact bytes under its own coldkey.
    d = _difficulty(31, 60)
    servers = [Srv(5, H("orig"), "ck-orig", 100, oracle(d)),
               Srv(9, H("orig"), "ck-copier", 3000, oracle(d))]
    commits = [Commitment("hk5", "ck-orig", H("orig"), 100),
               Commitment("hk9", "ck-copier", H("orig"), 3000)]
    r = _epoch(31, 60, servers, prev=r, commits=commits)
    assert r.reveal["copy_verdicts"] == {"5": "owner", "9": "copy"}
    assert _w(r) == pytest.approx({5: 1.0}) and r.owners[H("orig")] == "hk5"


@pytest.mark.parametrize("outage", [1, 2, 3])
def test_a3_inside_its_ghost_the_champion_cannot_lose_the_seat_to_a_copy(outage):
    """ATTACK: the copier knocks the champion offline so that its re-zipped
    copy, burned in beside it, takes the emptied seat.
    DEFENCE: a champion that missed leaves a ghost bar for CHAMPION_GHOST_OFF
    (3) epochs; the copy must beat the ghost by the margin (NV 0: never), and
    the champion resumes when it is back (INVARIANTS I3, "Ghost")."""
    last = 29
    r = None
    for e in range(last + outage + 8):
        out = last < e <= last + outage
        servers, commits = _copy_world(e, orig_serves=not out)
        r = _epoch(e, 60, servers, prev=r, commits=commits, burn_in=1500)
        assert r.decision.champion != H("rezip")
        assert 9 not in r.uids
    assert r.decision.champion == H("orig")


@pytest.mark.parametrize("outage", [4, 12])
def test_a3_past_its_ghost_the_copy_keeps_the_seat_the_documented_residual(outage):
    """Accepted residual (INVARIANTS I3, "Accepted residual: a champion kept
    offline"): a champion kept offline for longer than its ghost loses the
    seat to the best other head, and a re-serialised copy that ties it keeps
    it, because the original can never beat a tie by the margin. The original
    is paid seat 2. The defence is operational: a miner's axon accepts only
    validator IPs (scripts/axon_firewall.py, MINER_GUIDE). Pinned so that a
    change to it is deliberate."""
    last = 29
    r = None
    for e in range(last + outage + 12):
        out = last < e <= last + outage
        back = last + outage < e <= last + outage + 5     # E'' restarted its burn-in
        servers, commits = _copy_world(e, orig_serves=not out,
                                       orig_record=_unburned(H("orig")) if back else None)
        r = _epoch(e, 60, servers, prev=r, commits=commits, burn_in=1500)
    assert r.decision.champion == H("rezip")
    assert r.decision.state.history[:1] == (H("orig"),)
    assert _w(r) == pytest.approx({5: 0.1, 9: 0.9})


def test_a3_a_former_champion_serving_rarely_does_not_entrench_the_seat():
    """REGRESSION (review of the first A3 fix, 2026-11-09): extending the
    comparison set to every former seat holder let a seat-2 holder that serves
    one epoch in twenty bar every junior head for good. A clearly better junior
    Y must take the seat from champion A once it is burned in."""
    q, a_pol, b_pol = 300, _noisy_oracle(0.25), _noisy_oracle(0.30)
    y_block = 2 * BPE + 5
    r, took = None, None
    for e in range(40):
        d = _difficulty(e, q)
        commits = [Commitment("hkA", "ckA", H("A"), 3), Commitment("hkB", "ckB", H("B"), 1)]
        servers = [Srv(1, H("A"), "ckA", 3, a_pol(d), hotkey="hkA")]
        if e % 20 == 0:
            servers.append(Srv(2, H("B"), "ckB", 1, b_pol(d), hotkey="hkB"))
        if e >= 2:
            commits.append(Commitment("hkY", "ckY", H("Y"), y_block))
            servers.append(Srv(3, H("Y"), "ckY", y_block, oracle(d), hotkey="hkY"))
        r = _epoch(e, q, servers, prev=r, commits=commits,
                   seat=SeatState(champion=H("A"), champion_since=0, history=(H("B"),)))
        if took is None and r.decision.champion == H("Y"):
            took = e
    assert took is not None and took <= 2 + 5 + 1        # burn-in, then the take


def test_a3_an_honest_outage_burns_only_the_ghost():
    """REGRESSION (same review): an honest champion that goes offline for good
    leaves a 3-epoch ghost; then the best junior refills the seat. The first A3
    fix made it 24 epochs of burn."""
    q, x_pol, off = 300, _noisy_oracle(0.10), 10
    commits = [Commitment("hk5", "ckC", H("C"), 1), Commitment("hk6", "ckX", H("X"), 2)]
    r, burned_epochs = None, 0
    for e in range(off + 30):
        d = _difficulty(e, q)
        servers = [Srv(6, H("X"), "ckX", 2, x_pol(d))]
        if e < off:
            servers.insert(0, Srv(5, H("C"), "ckC", 1, oracle(d)))
        r = _epoch(e, q, servers, prev=r, commits=commits,
                   seat=SeatState(champion=H("C"), champion_since=0))
        if e >= off and _w(r).get(0, 0.0) > 0.5:
            burned_epochs += 1
    assert r.decision.champion == H("X")
    assert burned_epochs == PARAMS.ghost_off


# --- 4. strategic withholding ---------------------------------------------------------
#
# INVARIANTS, "I3 — one champion ..." ("Accepted residual: withholding"; "Ghost")
# and "I3/I4 — the v2 copy rule" (retirement is counted from the first one).


def _epoch_nv(e, q, rc, rx):
    """X's NV over C on epoch e alone: what an informed champion would compute
    before deciding to serve (both heads' routes are knowable to a staked
    operator, and the benchmarks' gold is public)."""
    r = _epoch(e, q, [Srv(5, H("C"), "ckC", 1, rc), Srv(6, H("X"), "ckX", 2, rx)])
    pu = pair_units(r.window, H("X"), H("C"), accuracy_value=PARAMS.accuracy_value,
                    hard_weight=PARAMS.hard_weight, anchor_spend=PARAMS.anchor_spend)
    return float(np.concatenate(list(pu.by_category.values())).mean()) / 1e6


def _withholding(p, withhold, epochs, q=300):
    """Champion C (the oracle with a fraction p of questions sent to m/cheap)
    against challenger X (the oracle). Returns (C's summed weight, the epoch X
    took the seat or None)."""
    cpol = _noisy_oracle(p)
    commits = [Commitment("hk5", "ckC", H("C"), 1), Commitment("hk6", "ckX", H("X"), 2)]
    r, earned, took, misses = None, 0.0, None, 0
    for e in range(epochs):
        d = _difficulty(e, q)
        rc, rx = cpol(d), oracle(d)
        serve = True
        if withhold and r is not None and r.decision.state.champion == H("C"):
            # Withhold an epoch X wins, as long as the missed ghost still stands.
            serve = not (_epoch_nv(e, q, rc, rx) > 0 and misses < PARAMS.ghost_off - 1)
        misses = 0 if serve else misses + 1
        servers = ([Srv(5, H("C"), "ckC", 1, rc)] if serve else []) + [Srv(6, H("X"), "ckX", 2, rx)]
        r = _epoch(e, q, servers, prev=r, commits=commits,
                   seat=SeatState(champion=H("C"), champion_since=0))
        earned += _w(r).get(5, 0.0)
        if took is None and r.decision.state.champion == H("X"):
            took = e
    return earned, took


def test_a4_withholding_does_not_block_a_clearly_better_challenger_nor_pay():
    """ATTACK: the champion serves only epochs in which its paired NV against the
    challenger is favourable and rides the 3-epoch missed ghost through the
    rest (resuming at once, since holding needs no burn-in), so the epochs it
    loses never enter the comparison.
    DEFENCE: the ghost lasts CHAMPION_GHOST_OFF epochs, so the champion must
    serve at least one epoch in three, and every withheld epoch burns its whole
    share. Against a challenger clearly better than the margin (here about 2 NV
    points, ten times it) the epochs it is forced to serve still carry the
    challenger past the margin: the take is delayed, not blocked, and the
    withholding champion earns less than an honest one at every horizon
    (INVARIANTS I3, "Accepted residual: withholding": "it pays only while a
    challenger sits within a hair of the margin")."""
    for epochs in (6, 10, 14):
        honest, took_h = _withholding(0.05, False, epochs)
        withheld, took_w = _withholding(0.05, True, epochs)
        assert withheld < honest, (epochs, withheld, honest)
    assert took_h is not None and took_w is not None and took_w <= 10


def test_a4_retiring_and_recommitting_neither_resumes_the_seat_nor_restarts_the_ghost():
    """ATTACK: the champion's owner commits a different head for one boundary
    (retiring it: a 24-epoch ghost bar), then recommits the champion's hash and
    serves it again, hoping to resume the seat at once behind the longer bar, or
    to restart the retire window each time it recommits.
    DEFENCE: only a "missed" ghost resumes; a retired one does not, and a
    ghost's own head is not a challenger to it, so the seat stays empty for the
    whole retire window, counted from the first retirement; the head returns
    only through the FIRST rule after the bar expires (INVARIANTS I3, "Ghost";
    the copy rule, "a retired head bars for at most CHAMPION_GHOST_RETIRE
    epochs over its lifetime")."""
    q, retire = 60, 2
    first_index = retire + 1                      # chain epoch of the retirement
    last_bar = first_index + PARAMS.ghost_retire - 1
    r = None
    for e in range(last_bar + 2):
        d = _difficulty(e, q)
        c_head = H("C-next") if e == retire else H("C")
        commits = [Commitment("hk5", "ckC", c_head, 1 if c_head == H("C") else e * BPE + 10),
                   Commitment("hk6", "ckX", H("X"), 2)]
        servers = [Srv(6, H("X"), "ckX", 2, _always("m/mid")(d))]
        if c_head == H("C"):
            servers.insert(0, Srv(5, H("C"), "ckC", 1, _always("m/mid")(d)))
        r = _epoch(e, q, servers, prev=r, commits=commits,
                   seat=SeatState(champion=H("C"), champion_since=0))
        index = e + 1
        if first_index <= index <= last_bar:
            (g,) = r.decision.state.ghosts
            assert (g.head_hash, g.reason, g.last_epoch) == (H("C"), "retired", last_bar)
            assert r.decision.champion is None and r.decision.event == "ghost"
            assert not any("resumed" in m for _, m in r.decision.detail)
        if index == last_bar + 1:
            assert r.decision.event == "first" and r.decision.champion == H("C")
    assert r.copy_state.get(H("C")).retired_at == first_index


# --- 5. cross-flip carry-over ---------------------------------------------------------
#
# INVARIANTS, "I1 — every epoch input is chosen by the boundary-keyed version"
# (proof namespace and head cost profile), "I3 — the v2 miss rule (E'')", and the
# pre-flip tie ("I3 — one champion", First).


def test_a5_a_v1_proof_in_the_first_v2_epoch_is_refused_for_its_namespace():
    """ATTACK: a miner serves its v1-namespace proof in the first epoch whose
    boundary is the activation block, to be scored under v2 without a v2 head.
    DEFENCE: the version is the boundary's, so the activation boundary itself
    is v2, and verify_proof refuses a proof whose routing evidence namespace is
    not the version's before any other check (INVARIANTS I1, "Proof namespace
    and head cost profile")."""
    from fugal_subnet.benchmarks.slicer import protocol_version_for_boundary
    from fugal_subnet.routing_protocol import identity
    from tests.test_v2_epoch_inputs import _proof_with_namespace, _verify

    act = 10 * BPE
    assert protocol_version_for_boundary(act - BPE, act) == 1
    assert protocol_version_for_boundary(act, act) == 2
    result = _verify(_proof_with_namespace(identity()), protocol_version_for_boundary(act, act))
    assert not result.valid and "namespace" in result.reason


def test_a5_no_head_is_valid_in_both_namespaces():
    """ATTACK (the premise of carrying evidence across the flip): serve ONE head
    hash in v1 and in v2, so its v1 record (and its burn-in) is the v2 head's.
    DEFENCE: a head binds its cost profile to one token manifest
    (cost_profile_id), so a v1 head is refused by v2's check and a v2 head by
    v1's; the same bytes can never be served on both sides of the flip."""
    import json

    from fugal_subnet.head_eval import load_head_from_npz
    from fugal_subnet.routing_protocol import (
        benchmark_costs_for,
        manifest_path,
        models_for_protocol,
    )
    from fugal_subnet.vendor import success_contract as sc

    v1 = load_head_from_npz((Path(__file__).resolve().parents[1] / FIXTURE_HEAD).read_bytes())
    benchmark_costs_for(v1, 1)
    with pytest.raises(ValueError, match="identity mismatch"):
        benchmark_costs_for(v1, 2)
    models = models_for_protocol(2)[:3]
    z = sc.make_head(np.zeros((3, 1024)), np.zeros(3), models,
                     json.loads(manifest_path(2).read_text()), "attack-test")
    buf = io.BytesIO()
    np.savez(buf, **z)
    v2 = load_head_from_npz(buf.getvalue())
    benchmark_costs_for(v2, 2)
    with pytest.raises(ValueError, match="identity mismatch"):
        benchmark_costs_for(v2, 1)


def test_a5_phantom_v1_misses_do_not_buy_v2_burn_in():
    """ATTACK: before the flip, serve one v1 epoch and then miss for weeks: v1's
    miss rule adds phantom zeros to n_total, and a record written before the
    burn-in counter existed reads its burn-in as n_total. Carry that into v2 to
    be burned in at the first v2 epoch and take the empty first seat days
    before anyone who served honestly.
    DEFENCE: the record IS inflated (asserted, so this test fails if the premise
    stops holding), but it belongs to the v1 hash; the v2 head is another hash
    (previous test), and the first v2 accumulation of a new hash restarts the
    counter at that epoch's questions. In v2 a miss never adds to burn-in.
    Through the wiring, the head is not burned in and the seat stays empty
    (INVARIANTS I3, "the v2 miss rule (E'')")."""
    from fugal_subnet.config import V2_BURN_IN_QUESTIONS
    from fugal_subnet.scoring import ScoringState, update_scores
    from tests.frame_fixtures import frame_from_truth, frontier_of
    from tests.test_miss_rule_v2 import _score

    frontier = frontier_of(frame_from_truth(400))
    s = update_scores(ScoringState(), {1: _score()}, {1: H("v1-head")}, frontier=frontier,
                      n_questions=300, protocol_version=1)
    for _ in range(20):                                  # v1 misses: phantom zeros
        s = update_scores(s, {}, {}, frontier=frontier, n_questions=300, protocol_version=1)
    inflated = s.records[1].evidence
    assert inflated.burn_in_n < 0 and inflated.burn_in_progress >= V2_BURN_IN_QUESTIONS
    # The flip: the UID serves its v2 head.
    s = update_scores(s, {1: _score()}, {1: H("v2-head")}, frontier=frontier,
                      n_questions=300, protocol_version=2)
    ev = s.records[1].evidence
    assert ev.weights_hash == H("v2-head") and ev.burn_in_progress == 300.0
    assert not burn_in_complete(ev, V2_BURN_IN_QUESTIONS)
    # v2 misses never add to burn-in, before or after it completes.
    for _ in range(3):
        before = s.records[1].evidence.burn_in_progress
        s = update_scores(s, {}, {}, frontier=frontier, n_questions=300, protocol_version=2)
        assert s.records[1].evidence.burn_in_progress <= before
    # Through the wiring: the first v2 epoch has no burned-in head, so no champion.
    d = _difficulty(0, 300)
    rec = SimpleNamespace(evidence=ev)
    r = _epoch(0, 300, [Srv(1, H("v2-head"), "ck1", 100, oracle(d), record=rec)],
               burn_in=V2_BURN_IN_QUESTIONS, floor=FLIP, salt="0xabc")
    assert r.decision.event == "empty" and _w(r) == {0: 1.0}


@pytest.mark.parametrize("salt", [f"0x{s:02x}" for s in range(16)])
def test_a5_a_commit_once_the_salt_is_known_is_junior_to_every_pre_flip_head(salt):
    """ATTACK: wait for the activation block's hash (the pre-flip tie-break
    salt), then commit, choosing the head's bytes so its salted digest sorts
    first; or commit in the activation block itself.
    DEFENCE: the salt is the activation block's own hash, so a commit in that
    block was made before it existed (it ties, ordered by a digest it could not
    grind), and any commit after it keeps its real block and is junior to every
    pre-flip head whatever its digest; the FIRST seat goes to a pre-flip head
    (INVARIANTS I3, "Pre-flip commits tie"). These three route identically, so
    the empty seat is an exact value tie, broken by the real first-commit block
    (since 2026-11-09): the earliest commit wins whatever the salt."""
    d = _difficulty(0, 60)
    servers = [Srv(5, H("early"), "ck-early", 10, oracle(d)),
               Srv(6, H("at-flip"), "ck-at", FLIP, oracle(d)),
               Srv(7, H("after"), "ck-after", FLIP + 1, oracle(d))]
    r = _epoch(1, 60, servers, floor=FLIP, salt=salt)
    st = r.copy_state
    key = {h: v2_seniority(st, h, floor_block=FLIP, salt=salt)
           for h in (H("early"), H("at-flip"), H("after"))}
    assert key[H("at-flip")][0] == key[H("early")][0] == FLIP
    assert key[H("after")] > max(key[H("early")], key[H("at-flip")])
    assert r.decision.event == "first" and r.decision.champion != H("after")
    assert r.decision.champion == H("early")
