"""The DCAP verdict itself, against real collateral, with no network.

Until 2026-11-08 no test saw `verify_dcap` say "no": every attack-suite case
stubs it, and the only non-mock test fails on a synthetic quote before the
signature is checked. A `verify_dcap` that returned True on a failed signature
passed the whole suite. These cases run the real library on the real fixtures
against collateral recorded once from the configured PCCS (2026-11-09, FMSPC
00806F050000), at a pinned time, so they never touch the network or expire.

They also hold the per-epoch collateral cache: one fetch per platform, and
each quote judged by its OWN PCK chain (the fetched collateral carries the
first quote's chain, and `verify` prefers it when present).
"""
import json
import pathlib

import pytest

from fugal_subnet.tee.attestation import (
    CollateralBudget,
    CollateralUnavailable,
    collateral_window,
    verify_dcap,
)

dcap_qvl = pytest.importorskip(
    "dcap_qvl", reason="optional [tee] extra; these cases run in the tee-verifier job")

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
RECORDED = json.loads((FIXTURES / "collateral_00806F050000.json").read_text())
FETCHED_AT = 1794187155              # 2026-11-09T00:39:15Z
PCK_CRL_EXPIRES = 1796773717         # 2026-12-08T23:48:37Z, the earliest expiry


def _collateral(**override):
    return dcap_qvl.QuoteCollateralV3.from_json(json.dumps({**RECORDED, **override}))


def _quote(name="A"):
    from fugal_subnet.tee.verify import unwrap_attestation
    return unwrap_attestation((FIXTURES / f"attestation_{name}.bin").read_bytes())[0]


@pytest.fixture()
def pccs(monkeypatch):
    """A PCCS that answers from the recording and counts what it is asked."""
    calls = []
    served = {"collateral": _collateral()}

    async def get_collateral(url, quote):
        calls.append(url)
        if isinstance(served["collateral"], Exception):
            raise served["collateral"]
        return served["collateral"]

    monkeypatch.setattr(dcap_qvl, "get_collateral", get_collateral)
    return type("Pccs", (), {"calls": calls, "served": served})


# --- the verdict --------------------------------------------------------------

@pytest.mark.parametrize("name", ["A", "B"])
def test_a_genuine_quote_verifies(pccs, name):
    assert verify_dcap(_quote(name), now=FETCHED_AT) is True


@pytest.mark.parametrize("offset, field", [(600, "report_data"), (200, "mrtd"),
                                           (530, "rtmr3"), (20, "header")])
def test_a_quote_with_one_byte_changed_is_refused(pccs, offset, field):
    quote = bytearray(_quote())
    quote[offset] ^= 0x01
    assert verify_dcap(bytes(quote), now=FETCHED_AT) is False


def test_a_signature_failure_is_a_refusal_not_a_pass(pccs, monkeypatch):
    """The branch every stub skips: the library says no, inside the window."""
    def refuses(*_a):
        raise ValueError("Verification failed: ISV enclave report signature is invalid")
    monkeypatch.setattr(dcap_qvl, "verify", refuses)
    assert verify_dcap(_quote(), now=FETCHED_AT) is False


# --- stale collateral is the operator's, not the miner's ----------------------

def test_the_window_ends_at_the_earliest_expiry_which_is_a_crl():
    start, end = collateral_window(_collateral())
    assert start <= FETCHED_AT < end == PCK_CRL_EXPIRES
    # And it is the library's own boundary, to the second.
    dcap_qvl.verify(_quote(), _collateral(), end - 1)
    with pytest.raises(ValueError):
        dcap_qvl.verify(_quote(), _collateral(), end)


@pytest.mark.parametrize("now", [PCK_CRL_EXPIRES, PCK_CRL_EXPIRES + 86400 * 30,
                                 FETCHED_AT - 86400 * 2])
def test_collateral_outside_its_window_is_an_outage_not_an_accusation(pccs, now):
    """Expired collateral fails EVERY genuine quote ("TCBInfo expired"). That
    is a stale PCCS or a wrong clock, so nothing was learned about a miner."""
    with pytest.raises(CollateralUnavailable, match="stale|clock"):
        verify_dcap(_quote(), now=now)


def test_a_tampered_quote_inside_the_window_is_still_the_miners(pccs):
    quote = bytearray(_quote())
    quote[600] ^= 0x01
    assert verify_dcap(bytes(quote), now=PCK_CRL_EXPIRES - 1) is False


# --- one fetch per platform per epoch ------------------------------------------

def test_one_fetch_serves_every_quote_on_a_platform(pccs):
    budget = CollateralBudget(600.0, expected_proofs=3)
    for name in ("A", "B", "A"):
        assert verify_dcap(_quote(name), budget=budget, now=FETCHED_AT) is True
    assert len(pccs.calls) == 1


def test_without_a_budget_every_call_fetches(pccs):
    verify_dcap(_quote(), now=FETCHED_AT)
    verify_dcap(_quote(), now=FETCHED_AT)
    assert len(pccs.calls) == 2


def test_cached_collateral_still_serves_once_the_budget_is_spent(pccs):
    budget = CollateralBudget(600.0)
    assert verify_dcap(_quote(), budget=budget, now=FETCHED_AT)
    budget.spend(10_000)
    assert budget.exhausted
    assert verify_dcap(_quote("B"), budget=budget, now=FETCHED_AT)
    assert len(pccs.calls) == 1


def test_a_failed_fetch_is_not_cached(pccs):
    budget = CollateralBudget(600.0)
    pccs.served["collateral"] = OSError("connection reset")
    with pytest.raises(CollateralUnavailable):
        verify_dcap(_quote(), budget=budget, now=FETCHED_AT)
    pccs.served["collateral"] = _collateral()
    assert verify_dcap(_quote(), budget=budget, now=FETCHED_AT)
    assert len(pccs.calls) == 2


def test_each_quote_is_judged_by_its_own_pck_chain(pccs):
    """The fetched collateral carries the asking quote's PCK chain and `verify`
    prefers it. Shared across proofs, a miner on another machine would be
    checked against the first miner's certificate. A collateral carrying a
    chain that matches nothing must still verify a genuine quote, because the
    chain is dropped before use."""
    pccs.served["collateral"] = _collateral(
        pck_certificate_chain="-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n")
    budget = CollateralBudget(600.0)
    assert verify_dcap(_quote(), budget=budget, now=FETCHED_AT) is True
    (kept,) = budget._collateral.values()
    assert json.loads(kept.to_json()).get("pck_certificate_chain") is None


# --- configuration and resource bounds ----------------------------------------

def test_an_empty_pccs_url_is_refused_not_defaulted(pccs, monkeypatch):
    monkeypatch.setattr("fugal_subnet.config.TEE_PCCS_URL", "")
    with pytest.raises(ValueError, match="No PCCS URL"):
        verify_dcap(_quote(), now=FETCHED_AT)
    assert pccs.calls == []


def test_unreadable_collateral_is_an_outage_not_an_accusation(pccs):
    """Collateral comes from the PCCS, not the miner. If it is damaged, the
    library refuses every quote and no window can be read: an outage."""
    pccs.served["collateral"] = _collateral(tcb_info="not json")
    with pytest.raises(CollateralUnavailable, match="unreadable"):
        verify_dcap(_quote(), now=FETCHED_AT)
