"""A stalled PCCS must not be able to stop a validator setting weights.

This is I6, and the failure it guards is not hypothetical: `verify_dcap`
fetches DCAP collateral over the network once per proof, inside the epoch loop,
from one host, and every validator does it at the same moment because the
collection point is a deterministic block. An upstream that accepts the
connection and then never answers used to block the epoch outright -- no
weights, and no log line either, because the code that records the failure sits
downstream of the block and never runs.

The bound has to be asserted in WALL-CLOCK TIME, not by checking that some
timeout argument is present. The previous code had `timeout=30` on the
threadpool branch and it did nothing: `with ThreadPoolExecutor()` calls
`shutdown(wait=True)` on exit, so the TimeoutError could not propagate until
the hung call finished anyway. Reading the code found a timeout and stopped
there; only running it showed the timeout was decorative. Hence the clocks
below.
"""
import asyncio
import pathlib
import subprocess
import sys
import time

import pytest

from fugal_subnet.tee import verify as verify_mod
from fugal_subnet.tee.attestation import (
    CollateralBudget,
    CollateralUnavailable,
    _run_coro,
    verification_order,
    verify_dcap,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# `dcap_qvl` is an OPTIONAL extra, and CI runs pytest twice on purpose: the
# `test` job installs `--extra dev` only, the `tee-verifier` job adds
# `--extra tee` and asserts the module is importable before running. The two
# environments differ deliberately, because `verify_dcap` takes a different
# path when the verifier is absent.
#
# So the cases below that drive the REAL binding skip where it is missing and
# run where it is present. The ones that do not need it -- the `_run_coro`
# bound, the budget arithmetic, the verification order -- must keep running in
# BOTH jobs, which is why this is per-test rather than a module-level skip.
def _needs_dcap():
    pytest.importorskip(
        "dcap_qvl",
        reason="optional [tee] extra; these cases run in the tee-verifier job",
    )


def _REAL_QUOTE():
    """The live-hardware quote, unwrapped. Needed because verify_dcap
    parses before it fetches, so garbage never reaches the code here."""
    _needs_dcap()
    blob = (FIXTURES / "attestation_A.bin").read_bytes()
    quote, _events = verify_mod.unwrap_attestation(blob)
    return quote

# Comfortably above the bound under test, far below the stall being simulated.
# Any assertion using it fails loudly if the bound stops working at all, and
# does not flake on a loaded CI box.
SLACK = 8.0
STALL = 60.0


def _stalled(_url, _quote):
    """Stand-in for a PCCS that accepts the connection and never answers."""
    async def _never():
        await asyncio.sleep(STALL)
    return _never()


@pytest.fixture()
def real_quote():
    """A quote captured from a live dstack CVM, unwrapped from its envelope.

    It has to be a real one. `verify_dcap` parses BEFORE it fetches -- a quote
    that will not parse is the miner's problem and must never cost a round trip
    or be reported as an outage -- so garbage bytes never reach the code under
    test here and the timeout would go unexercised while the test still passed.
    """
    _needs_dcap()
    blob = (FIXTURES / "attestation_A.bin").read_bytes()
    quote, _events = verify_mod.unwrap_attestation(blob)
    return quote


@pytest.fixture()
def stalled_pccs(monkeypatch):
    _needs_dcap()
    import dcap_qvl
    monkeypatch.setattr(dcap_qvl, "get_collateral", _stalled, raising=False)
    monkeypatch.setattr(
        "fugal_subnet.config.TEE_COLLATERAL_TIMEOUT", 1.0, raising=False)


# --- the unit that was broken -------------------------------------------


def test_the_bound_holds_on_the_branch_a_synchronous_validator_takes():
    """`run_until_complete` had no bound at all, and it is the branch the real
    validator takes. A stub sleeping 60 s blocked for the full 60 s."""
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        _run_coro(lambda: asyncio.sleep(STALL), timeout=1.0)
    assert time.monotonic() - t0 < SLACK


def test_the_bound_holds_on_the_branch_that_only_looked_bounded():
    """THE REGRESSION THIS FILE EXISTS FOR. The threadpool branch carried a
    `timeout=30` that never took effect, because pool shutdown waited for the
    hung call before the TimeoutError could escape. The clock is the only thing
    that can tell the two apart -- an assertion that a timeout was *passed*
    passes against the broken code."""
    async def driver():
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            _run_coro(lambda: asyncio.sleep(STALL), timeout=1.0)
        return time.monotonic() - t0

    t0 = time.monotonic()
    inner = asyncio.run(driver())
    outer = time.monotonic() - t0
    assert inner < SLACK
    # Measured from OUTSIDE the loop as well: the old shape returned from
    # `result()` on time and then blocked here, in `shutdown(wait=True)`.
    assert outer < SLACK


def test_no_timeout_means_no_bound_and_that_is_still_available():
    """The bound is opt-in per call site, so nothing else that runs a coroutine
    through this helper silently acquires a deadline it was not written for."""
    assert _run_coro(lambda: asyncio.sleep(0, result=7)) == 7


# --- what the validator actually observes -------------------------------


def test_a_stalled_pccs_raises_rather_than_hanging(stalled_pccs, real_quote):
    """Both halves matter, and the second half was learned by falsifying this
    test rather than by writing it carefully.

    Against the pre-fix code this passed for the wrong reason: the old branch
    called `asyncio.get_event_loop()`, which raises RuntimeError once an
    earlier test has closed the loop, and that RuntimeError was caught by the
    generic handler and re-raised as CollateralUnavailable instantly. The right
    exception, in the right time budget, from entirely the wrong cause. So the
    assertion has to name WHY the fetch was abandoned, not merely that it was.
    """
    t0 = time.monotonic()
    with pytest.raises(CollateralUnavailable) as e:
        verify_dcap(real_quote, pccs_url="https://pccs.invalid")
    assert time.monotonic() - t0 < SLACK
    assert "exceeded" in str(e.value), (
        "raised in time, but not because the bound fired -- see the docstring")


def test_a_timeout_says_it_was_a_timeout(stalled_pccs, real_quote):
    """An operator reading the epoch log has to be able to tell a stalled
    upstream from a refused connection, and to see the budget they would raise.
    `str(TimeoutError())` is empty, so this needs its own message."""
    with pytest.raises(CollateralUnavailable) as e:
        verify_dcap(real_quote, pccs_url="https://pccs.invalid")
    assert "exceeded" in str(e.value)
    assert "never judged" in str(e.value)


def test_a_timeout_is_an_outage_and_never_an_accusation(stalled_pccs, monkeypatch):  # noqa: E501
    """The prerequisite that made this timeout safe to add at all. Before the
    unverifiable path existed, a TimeoutError fell into `except Exception:
    return False` and the miner was scored as a cheat -- so a slow PCCS became
    a simultaneous false accusation against every honest miner in the field.

    Asserted end to end through verify_proof, because that is where the
    accusation would actually have been recorded."""
    from tests.test_unverifiable import _proof

    p = _proof()
    # The real envelope, so the parse-first gate is passed rather than stubbed.
    p.attestation_quote = (FIXTURES / "attestation_A.bin").read_bytes()
    monkeypatch.setattr(
        verify_mod, "verify_dcap",
        lambda q, **kw: verify_dcap(q, pccs_url="https://pccs.invalid", **kw))
    r = verify_mod.verify_proof(
        p, approved_measurements=set(), expected_questions_hash=p.questions_hash,
        expected_nonce=p.nonce, gold_answers={},
        expected_hotkey=p.hotkey, mock=False,
    )
    assert r.valid is False          # nothing was checked, so nothing is accepted
    assert r.unverifiable is True    # but it is an outage, not fraud
    # And for the right reason -- see the sibling test's docstring on why an
    # unverifiable verdict alone does not prove the bound is what produced it.
    assert "exceeded" in r.reason


# --- the aggregate bound ------------------------------------------------
#
# The per-proof bound above stops one fetch hanging. It does not stop the EPOCH
# hanging: the verify loop is serial over every UID, so 256 individually-bounded
# fetches still cost 256 x the bound -- 2560 s at the default, against an 1800 s
# post-collection window. The epoch returns and has already missed the window it
# exists to hit, which is an I6 failure that looks like a successful run.


def test_the_allowance_is_whichever_runs_out_first():
    """Handing the last fetch a full per-call budget would let it overshoot the
    epoch ceiling by that much, which is the whole quantity being bounded."""
    b = CollateralBudget(2.5)
    assert b.allowance(10.0) == 2.5     # ceiling binds
    b.spend(2.0)
    assert b.allowance(10.0) == 0.5     # ceiling still binds, and has shrunk
    b2 = CollateralBudget(100.0)
    assert b2.allowance(10.0) == 10.0   # per-call binds


def test_a_spent_budget_is_exhausted_and_never_reports_negative_time():
    b = CollateralBudget(1.0)
    assert not b.exhausted
    b.spend(5.0)
    assert b.exhausted
    assert b.remaining == 0.0
    assert b.allowance(10.0) == 0.0


def test_an_exhausted_budget_refuses_without_touching_the_network(real_quote, monkeypatch):
    """Checked BEFORE dialling, not after. A call that will be abandoned anyway
    must never leave the process, or the ceiling bounds nothing."""
    _needs_dcap()
    called = []

    def _record(url, quote):
        called.append(url)
        async def _c():
            return None
        return _c()

    import dcap_qvl
    monkeypatch.setattr(dcap_qvl, "get_collateral", _record, raising=False)

    spent = CollateralBudget(0.0)
    with pytest.raises(CollateralUnavailable) as e:
        verify_dcap(real_quote, pccs_url="https://pccs.invalid", budget=spent)
    assert "budget is exhausted" in str(e.value)
    assert called == [], "an exhausted budget still made a network call"


def test_many_stalled_proofs_cost_the_budget_not_the_sum_of_their_timeouts(
    stalled_pccs, real_quote,
):
    """THE POINT OF THE WHOLE MECHANISM, asserted on the clock.

    Twenty proofs at a 1 s per-call bound would be 20 s unbounded-in-aggregate.
    Against a 3 s epoch ceiling it must cost ~3 s, because once the ceiling is
    spent the remaining proofs are refused without dialling.
    """
    budget = CollateralBudget(3.0)
    t0 = time.monotonic()
    outcomes = []
    for _ in range(20):
        try:
            verify_dcap(real_quote, pccs_url="https://pccs.invalid", budget=budget)
        except CollateralUnavailable as e:
            outcomes.append(str(e))
    elapsed = time.monotonic() - t0

    assert len(outcomes) == 20, "every proof must resolve, none may be accepted"
    assert elapsed < 3.0 + SLACK, (
        f"20 stalled proofs took {elapsed:.1f}s against a 3s ceiling — the "
        "aggregate bound is not holding")
    # Something other than the per-call bound stopped the later ones: either
    # the ceiling ran out or the breaker opened. Which one wins depends on the
    # numbers, and both are correct -- what must not happen is 20 dials.
    assert any("was not attempted" in o for o in outcomes)


def test_the_breaker_stops_dialling_an_endpoint_that_has_stopped_answering(
    stalled_pccs, real_quote, monkeypatch,
):
    """The fd bound, not a time bound.

    An abandoned fetch keeps its socket -- measured at 1.02 leaked descriptors
    each -- because cancelling it would unwind a Rust future inside pyo3 and
    crash the interpreter. A container on the common 1024 default would run out
    in a few epochs against a stalled PCCS and lose its axon and subtensor
    sockets with it. The only way to leak fewer is to dial fewer times, so the
    third consecutive timeout is treated as evidence about the ENDPOINT.
    """
    dials = []
    import dcap_qvl
    real = dcap_qvl.get_collateral

    def _counting(url, quote):
        dials.append(url)
        return real(url, quote)

    monkeypatch.setattr(dcap_qvl, "get_collateral", _counting, raising=False)

    # A budget far too large to be what stops us, so only the breaker can.
    b = CollateralBudget(10_000.0, max_timeouts=3)
    for _ in range(12):
        with pytest.raises(CollateralUnavailable):
            verify_dcap(real_quote, pccs_url="https://pccs.invalid", budget=b)

    assert len(dials) == 3, (
        f"dialled {len(dials)} times after 3 consecutive timeouts — every dial "
        "past the third leaks a descriptor to learn something already known")
    assert b.circuit_open
    assert b.exhausted


def test_an_endpoint_that_errors_between_stalls_is_flaky_not_down(monkeypatch):
    """Consecutive, and asserted THROUGH THE ERROR PATH rather than by poking
    the counter.

    The earlier version of this test called the reset method directly, so it
    could not see that the reset fired only on SUCCESS: a 5xx, a DNS failure or
    a reset left the counter standing, and three timeouts separated by two
    instant errors opened a breaker that three comments called "consecutive".
    A test that drives the object instead of the code path agrees with whatever
    the object does.

    An endpoint that returns an error ANSWERED. No descriptor leaked, and it
    proved it is reachable — the opposite of what this counter accumulates.
    """
    _needs_dcap()
    import dcap_qvl

    calls = {"n": 0}

    def _alternating(_url, _quote):
        calls["n"] += 1
        stall = calls["n"] % 2 == 1

        async def _c():
            if stall:
                await asyncio.sleep(STALL)
            raise RuntimeError("HTTP 500 from upstream")
        return _c()

    monkeypatch.setattr(dcap_qvl, "get_collateral", _alternating, raising=False)
    monkeypatch.setattr(
        "fugal_subnet.config.TEE_COLLATERAL_TIMEOUT", 0.3, raising=False)

    b = CollateralBudget(10_000.0, max_timeouts=3)
    for _ in range(10):
        with pytest.raises(CollateralUnavailable):
            verify_dcap(
                _REAL_QUOTE(), pccs_url="https://pccs.invalid", budget=b)

    assert not b.circuit_open, (
        "the breaker opened on an endpoint that answered every other dial in "
        "~0ms — that is flaky, not down, and the time budget is what handles it")


def test_a_nearly_spent_budget_does_not_blame_a_healthy_endpoint(monkeypatch):
    """The clock running out is not evidence against the endpoint.

    `allowance` is min(per-call, remaining), so at the tail of a spent epoch a
    perfectly healthy endpoint answering in a flat 100 ms is abandoned at
    0.047 s and would take a strike it did not earn. Same outage-versus-
    accusation distinction the unverifiable path was built for, one level down.
    """
    _needs_dcap()
    import dcap_qvl

    def _prompt(_url, _quote):
        async def _c():
            await asyncio.sleep(0.1)
            return None
        return _c()

    monkeypatch.setattr(dcap_qvl, "get_collateral", _prompt, raising=False)
    monkeypatch.setattr(
        "fugal_subnet.config.TEE_COLLATERAL_TIMEOUT", 10.0, raising=False)

    b = CollateralBudget(0.02, max_timeouts=3)   # almost nothing left
    with pytest.raises(CollateralUnavailable) as e:
        verify_dcap(_REAL_QUOTE(), pccs_url="https://pccs.invalid", budget=b)

    assert b._timeouts == 0, (
        "a healthy endpoint was charged a timeout strike because the EPOCH "
        "budget ran out, not because the endpoint failed")
    assert "not evidence against the endpoint" in str(e.value)


def test_a_miner_cannot_drain_the_budget_with_a_quote_that_does_not_parse():
    """THE I4 CASE FOR THE BUDGET, and the reason parse-before-fetch matters
    more now than it did before the budget existed.

    A shared epoch ceiling is a shared resource, so the question "what can one
    miner make everyone else pay?" has to be asked of it. If garbage bytes
    reached the fetch, a miner could burn the whole allowance and starve the
    honest miners behind it in the queue -- one miner lowering another's score,
    which is exactly what I4 forbids.

    They cannot: verify_dcap parses first, and an unparseable quote is judged
    locally and rejected without a round trip. The budget is untouched, and the
    verdict is `False` -- the miner's own fault -- not `unverifiable`.
    """
    _needs_dcap()
    b = CollateralBudget(10.0)
    before = b.remaining
    assert verify_dcap(b"\x00" * 8, pccs_url="https://pccs.invalid", budget=b) is False
    assert b.remaining == before, "a malformed quote consumed shared budget"


def test_a_failed_fetch_still_costs_the_budget(stalled_pccs, real_quote):
    """The other direction. A budget that only charged for SUCCESS would drain
    slowest exactly when the upstream is at its worst, which is the case it
    exists for -- so time is charged on every exit, not just the happy one."""
    b = CollateralBudget(10.0)
    with pytest.raises(CollateralUnavailable):
        verify_dcap(real_quote, pccs_url="https://pccs.invalid", budget=b)
    assert b.remaining < 10.0


# --- the real binding ---------------------------------------------------


def test_abandoning_a_real_fetch_does_not_crash_the_interpreter(tmp_path):
    """THE TEST THAT WAS MISSING, and its absence is the more useful lesson.

    Every other case in this file stubs `get_collateral` with an
    `asyncio.sleep`. A Python coroutine cancels cleanly. The Rust future behind
    the real binding does not: cancelling it unwinds inside pyo3, and its tokio
    worker then touches the interpreter during finalization --

        assertion failed: The Python interpreter is not initialized

    -- so the process exits 134 or 139 *after* the bound has already fired
    correctly. Measured at 4 runs in 10. The whole suite passed throughout,
    because the stub's semantics differ from the real thing at exactly the
    point under test.

    That is the same error as the decorative `timeout=30` this file was written
    to catch: verified against a stand-in that could not exhibit the failure.
    An assertion that the bound FIRED is not an assertion that the process
    SURVIVED it, so this asserts the exit code, in a subprocess, against a
    socket that accepts and never answers. No network, no spend.
    """
    _needs_dcap()
    script = tmp_path / "abandon.py"
    script.write_text(
        "import socket, threading, pathlib\n"
        "srv = socket.socket()\n"
        "srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "srv.bind(('127.0.0.1', 0)); srv.listen(64)\n"
        "port = srv.getsockname()[1]\n"
        # Accept and then say nothing at all -- the stall this guards against.
        "threading.Thread(target=lambda: [srv.accept() for _ in iter(int, 1)],\n"
        "                 daemon=True).start()\n"
        "from fugal_subnet.tee import verify as v\n"
        "from fugal_subnet.tee.attestation import _run_coro\n"
        "import dcap_qvl\n"
        f"blob = pathlib.Path({str(FIXTURES / 'attestation_A.bin')!r}).read_bytes()\n"
        "q, _ = v.unwrap_attestation(blob)\n"
        "try:\n"
        "    _run_coro(lambda: dcap_qvl.get_collateral(\n"
        "        f'http://127.0.0.1:{port}', q), timeout=1.0)\n"
        "except TimeoutError:\n"
        "    print('bound fired')\n"
        # Prove the interpreter is still usable, then exit -- exit is where it
        # used to die, so the test is worthless without actually returning.
        "dcap_qvl.parse_quote(q)\n"
        "print('still alive')\n"
    )

    # Repeated because the crash is a race and a single clean run proves
    # nothing: the pre-fix implementation passed 6 runs in 10.
    for attempt in range(6):
        r = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=120,
            cwd=str(pathlib.Path(__file__).resolve().parent.parent),
        )
        assert r.returncode == 0, (
            f"attempt {attempt}: exited {r.returncode} "
            f"(134=SIGABRT, 139=SIGSEGV) — abandoning a real collateral fetch "
            f"must not take the interpreter down.\nstderr: {r.stderr[-600:]}")
        assert "bound fired" in r.stdout
        assert "still alive" in r.stdout


def test_one_stalled_miner_cannot_eat_the_share_of_the_miners_behind_it():
    """I4 applied to the shared ceiling.

    Without a per-proof share, a single stalling fetch consumes time that
    belonged to everyone behind it in the queue. Whether a miner can actually
    cause that turns on how fast a PCCS refuses an FMSPC it does not know --
    which nobody has measured. Enforcing the share means the answer stops
    deciding whether the design is safe.
    """
    b = CollateralBudget(600.0, expected_proofs=256)
    # No single proof may take more than its slice, even though the per-call
    # hang guard is far larger and there is plenty left in the ceiling.
    assert b.allowance(10.0) == pytest.approx(600.0 / 256, rel=1e-6)

    # Unused time flows forward: after 100 fast proofs the survivors are
    # entitled to more, so a healthy epoch is never squeezed by this.
    for _ in range(100):
        b.spend(0.05)
    assert b.allowance(10.0) > 600.0 / 256

    # And the per-call hang guard still caps it once the share exceeds it.
    generous = CollateralBudget(10_000.0, expected_proofs=2)
    assert generous.allowance(10.0) == 10.0


def test_an_unknown_field_size_falls_back_to_the_plain_ceiling():
    """`expected_proofs=None` means "unknown", and an unknown denominator must
    not silently become 1 — that would hand the first proof everything."""
    b = CollateralBudget(600.0)
    assert b.allowance(10.0) == 10.0


# --- verification order -------------------------------------------------


def test_two_validators_derive_the_same_verification_order():
    """I1. The order decides who goes unverified once the ceiling binds, so if
    two validators disagreed about it they would drop different miners from the
    same evidence — a fork manufactured by the fix for a fork."""
    a = verification_order("ab" * 32, 256)
    b = verification_order("ab" * 32, 256)
    assert a == b


def test_every_miner_is_in_the_queue_exactly_once():
    """A 'permutation' that quietly dropped or duplicated a uid would silently
    unverify miners no matter how much budget was left."""
    order = verification_order("cd" * 32, 256)
    assert sorted(order) == list(range(256))


def test_the_disadvantage_rotates_between_epochs():
    """The whole reason not to use UID order. If the queue were stable, the
    same high-UID miners would go unverified every epoch — a standing penalty
    no miner caused and none could escape."""
    first = verification_order("ab" * 32, 256)
    second = verification_order("ef" * 32, 256)
    assert first != second
    # And the tail — the miners actually at risk — genuinely turns over.
    assert set(first[-32:]) != set(second[-32:])


def test_the_order_is_not_uid_order():
    """Guards the degenerate implementation that passes the two tests above."""
    assert verification_order("ab" * 32, 256) != list(range(256))


def test_an_empty_field_is_not_a_special_case():
    assert verification_order("ab" * 32, 0) == []
