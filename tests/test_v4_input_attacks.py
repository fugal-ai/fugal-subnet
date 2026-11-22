"""The protocol v4 step-input cases of the miner attack suite
(fugal_subnet/attacks/run_miner_attacks.py, V4_INPUT_ATTACKS), each held,
and each shown to break once the rule that holds it is taken out: the rule,
not the shape of the case, is what holds."""
from __future__ import annotations

import pytest

from fugal_subnet import chain_proposals as cp
from fugal_subnet import menu_inputs as mi
from fugal_subnet import menu_state as ms
from fugal_subnet.attacks import run_miner_attacks as suite


def _per_hotkey(monkeypatch):
    monkeypatch.setattr(mi, "_voter", lambda vp: vp.hotkey)


def _any_read(monkeypatch):
    monkeypatch.setattr(mi, "_read_ok", lambda read: True)


def _no_majority(monkeypatch):
    real = ms.health_retires

    def any_failing(h, params):
        if h.failing:
            return ms.REASON_FAILING
        return real(h, params)
    monkeypatch.setattr(ms, "health_retires", any_failing)


def _no_outage_guard(monkeypatch):
    monkeypatch.setattr(mi, "outage", lambda summary: False)


def _no_cell_bound(monkeypatch):
    monkeypatch.setattr(mi, "cell_token_bounds", lambda entry: (10**12, 10**12))


def _no_occupied_rule(monkeypatch):
    monkeypatch.setattr(cp, "occupied", lambda *a, **k: set())


def _no_coldkey_once(monkeypatch):
    monkeypatch.setattr(cp, "coldkey_admitted", lambda *a, **k: False)


def _every_hotkey_served(monkeypatch):
    monkeypatch.setattr(cp, "is_served", lambda *a, **k: True)


def _lenient_documents(monkeypatch):
    """Junk bytes read as some proposal, of whatever hotkey sent them."""
    real = cp.parse_document

    def parse(raw):
        try:
            doc = real(raw)
        except ValueError:
            doc = real(cp.document_bytes(cp.ProposalDoc(
                proposer="5Fhk-a", model="lab-x/junk", reasoning=None,
                provider_order=("host-x/fp8",), quantization="fp8")))
        return doc

    monkeypatch.setattr(cp, "parse_document", parse)


RULES = {
    "one operator's 20 hotkeys vote a price": _per_hotkey,
    "one operator's 20 hotkeys unlist a host": _per_hotkey,
    "a forged read reaches the vote": _any_read,
    "one coldkey fails every call to an entry": _no_majority,
    "every call fails for three epochs": _no_outage_guard,
    "propose an entry already on the menu": _no_occupied_rule,
    "two miners propose one entry": _no_occupied_rule,
    "one coldkey's 6 hotkeys propose": _no_coldkey_once,
    "a junk proposal document": _lenient_documents,
    "a proposal from a hotkey that served nothing": _every_hotkey_served,
    "one coldkey's 20 hotkeys inflate cells": _per_hotkey,
    "absurd tokens in one graded cell": _no_cell_bound,
}


def test_every_input_case_names_its_rule():
    assert {n for n, *_ in suite.V4_INPUT_ATTACKS} == set(RULES)


@pytest.mark.parametrize("case", suite.V4_INPUT_ATTACKS, ids=[c[0] for c in suite.V4_INPUT_ATTACKS])
def test_each_input_case_is_held_by_its_rule(case, monkeypatch):
    name, fn, gain, inv = case
    assert suite._run_input(name, fn, gain, inv)["verdict"] == "BLOCKED"
    RULES[name](monkeypatch)
    assert suite._run_input(name, fn, gain, inv)["verdict"] == "BROKEN"


@pytest.mark.parametrize("name", ["a read price as a float", "a read price as decimal text",
                                  "a host listed twice in one read"])
def test_a_forged_read_in_a_proof_is_refused_when_the_proof_is_read(name):
    (case,) = [c for c in suite.PROOF_FIELD_ATTACKS if c[0] == name]
    assert suite._run_proof_field(*case)["verdict"] == "BLOCKED"
