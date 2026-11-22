"""Chain proposals through the shipped loops (neurons/validator.main and
neurons/miner._run_epoch on tests/v4_world.py's fake chain, store and
provider), with the cases a live field brings at once (FAST_CROWN_V4.md 18):

    F+1   four miners commit proposals before the root's pinned read:
          two of them the same entry (lab-c/model-new@high), one another
          entry (lab-c/model-other@high), and one the hash of bytes that are
          not a proposal document (junk). Four serving hotkeys: one join an
          epoch, so the valid proposals compete for one place.
    F+2   one of the two entries is on the menu, provisional; the junk is
          refused as malformed; the other valid entry waits (no place).
    F+2   the root is late for F+3's joins: it commits no join pointer.
    F+3   every node and TD applies the empty delta: the base menu runs.
    F+3   the root admits the waiting entry: on the menu of F+4.

Every node and TD runs one menu every epoch, the follower's join audits are
clean, and no validator logs an alarm."""
from __future__ import annotations

import dataclasses

import pytest

FLIP = 12
FIRST, LAST = FLIP - 1, FLIP + 6
FILE_AT, LATE = FLIP + 1, FLIP + 2
OTHER = {"model": "lab-c/model-other", "reasoning": ["effort", "high"],
         "provider_order": ["host-a/fp8"], "quantization": "fp8"}
OTHER_ID = "lab-c/model-other@high"
JUNK = b'{"this": "is not a proposal document"}'


@pytest.fixture
def world(tmp_path, monkeypatch, caplog):
    import logging

    from fugal_subnet import epoch_plan
    from fugal_subnet import menu_rules as mr
    from fugal_subnet import menu_state as ms
    from neurons import validator
    from tests import menu_fixtures as fx
    from tests.signed_list_fixtures import Signer
    from tests.v4_world import PROPOSAL, FlipWorld, MinerSpec, Provider, ValidatorSpec

    signer = Signer(tmp_path, "release")
    monkeypatch.setattr(mr, "list_signers", lambda: signer.signers_line().encode())
    fx.allow_small_menus(monkeypatch)
    real_params = epoch_plan.menu_step_params_v4
    monkeypatch.setattr(epoch_plan, "menu_step_params_v4", lambda floor: dataclasses.replace(
        real_params(floor), min_entries=1))
    miners = [MinerSpec("5Fminer-0", "5Fminer-cold-0", seed=0, proposal=PROPOSAL,
                        propose_at=FILE_AT),
              MinerSpec("5Fminer-1", "5Fminer-cold-1", seed=1, proposal=PROPOSAL,
                        propose_at=FILE_AT),
              MinerSpec("5Fminer-2", "5Fminer-cold-2", seed=2, proposal=OTHER,
                        propose_at=FILE_AT),
              MinerSpec("5Fminer-3", "5Fminer-cold-3", seed=3)]
    specs = [ValidatorSpec("root", "root", "root", root=True),
             ValidatorSpec("follower", "follower", "follower")]

    def on_epoch(w, epoch):
        if epoch == FILE_AT:
            # Miner 3 serves junk and commits its hash beside its head.
            from fugal_subnet import chain_proposals as cp

            m = w.miners["5Fminer-3"]
            m.proposal_doc = JUNK
            w.chain.commits[m.hotkey].append(
                (epoch * 10 + 1, (m.weights_hash, cp.proposal_field(cp.proposal_hash(JUNK)))))

    real_join = validator._v4_join_step

    def late_root(**kw):
        if kw["epoch_index"] == LATE:
            kw["root_joins"].record(LATE + 1, None)      # past the commit deadline: nothing
            return None
        return real_join(**kw)
    monkeypatch.setattr(validator, "_v4_join_step", late_root)

    w = FlipWorld(tmp_path, monkeypatch.setattr, monkeypatch.setenv, signer, flip=FLIP,
                  first=FIRST, last=LAST, validators=specs, miners=miners, on_epoch=on_epoch)
    w.provider = Provider(cheater=None)
    w.setup()
    with caplog.at_level(logging.ERROR):
        w.run()
    w.alarms = [r.getMessage() for r in caplog.records if "ALARM" in r.getMessage()]
    w.ms = ms
    # The root's pinned read before the boundary, every node's read of its join
    # pointer, the published-set reads and the TDs' reads of the root's
    # reveals: none asks for a block the chain has not produced.
    assert w.chain.future_reads == [], w.chain.future_reads[:5]
    return w


def test_proposals_through_the_loops_with_a_junk_document_two_of_one_entry_and_a_late_root(world):
    from fugal_subnet import chain_proposals as cp
    from tests.v4_world import NETUID, NEW_ID

    w, ms = world, world.ms
    menus = {k: w.held("follower").held(k) for k in range(FLIP, LAST + 1)}
    root_menus = {k: w.held("root").held(k) for k in range(FLIP, LAST + 1)}
    assert {k: ms.menu_hash(m) for k, m in menus.items()} == \
        {k: ms.menu_hash(m) for k, m in root_menus.items()}
    for k in range(FLIP, LAST + 1):
        on = {ms.menu_hash(menus[k])} | {m.menus[k] for m in w.miners.values() if k in m.menus}
        assert len(on) == 1, k

    def ids(k):
        return set(menus[k].ids())

    # F+2: exactly one of the two entries joined, provisional; F+3 (the late
    # root): nothing new; F+4: the other one.
    first = (ids(FILE_AT + 1) - ids(FILE_AT)) & {NEW_ID, OTHER_ID}
    assert len(first) == 1
    (first_id,) = first
    assert menus[FILE_AT + 1].entry(first_id).provisional
    assert ids(LATE + 1) - ids(LATE) == set()
    second = {NEW_ID, OTHER_ID} - first
    assert ids(LATE + 2) - ids(LATE + 1) == second
    # The follower audited every join step it could: no difference.
    audits = {k: (w.reveal("follower", k) or {}).get("join_audit") for k in range(FLIP, LAST + 1)}
    assert all(a is None or a.get("differences") == [] for a in audits.values()), audits
    assert any(a is not None for a in audits.values())
    # The deltas the root published, by their hash, as every node fetched them.
    raw = [v for key, v in w.store.objects.items() if f"{NETUID}/joins/" in key]
    parsed = [cp.parse_delta(b) for b in raw]
    by_epoch = {d.epoch: d for d in parsed}
    assert LATE + 1 not in by_epoch                          # the late root published nothing
    v = {c.hotkey: c.verdict for c in by_epoch[FILE_AT + 1].considered}
    # All four documents filed in the one epoch came back over the root's
    # fetch (real loopback HTTP, every request at once): none is unfetched.
    assert len(v) == 4 and cp.UNFETCHED not in v.values(), v
    assert v["5Fminer-3"] == cp.MALFORMED
    assert sorted(v[h] for h in ("5Fminer-0", "5Fminer-1", "5Fminer-2")).count(cp.JOINED) == 1
    assert by_epoch[FILE_AT + 1].serving == 4
    later = {c.hotkey: c.verdict for c in by_epoch[LATE + 2].considered}
    assert later["5Fminer-3"] == cp.MALFORMED
    assert list(later.values()).count(cp.JOINED) == 1
    assert w.alarms == []
