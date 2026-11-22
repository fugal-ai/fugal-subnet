"""Protocol v4: proposals filed on chain, the root's join delta, provisional
entries and their gates (docs/decisions/FAST_CROWN_V4.md, section 18).

No network, no chain: documents, listings and chain reads are written out.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from fugal_subnet import chain_proposals as cp
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx

BLOCK_HASH = "ab" * 32
REVEAL = ("e00000099", "cd" * 32)


def _read(base, proposals, *, serving=None, coldkeys=None, block_hash=BLOCK_HASH):
    """A chain read of `base`'s epoch: `proposals` is [(hotkey, raw document)]."""
    hk_p = tuple(sorted((hk, cp.proposal_hash(raw)) for hk, raw in proposals))
    serving = {hk for hk, _ in proposals} if serving is None else set(serving)
    coldkeys = coldkeys or {hk: f"ck-{hk}" for hk, _ in proposals}
    return cp.ChainRead(epoch=base.epoch, block=base.epoch * 360 - cp.JOIN_READ_MARGIN_BLOCKS,
                        block_hash=block_hash, proposals=hk_p, coldkey_of=coldkeys,
                        serving=frozenset(serving), reveal=REVEAL)


def _docs(proposals):
    return {cp.proposal_hash(raw): raw for _hk, raw in proposals}


def _listings(proposals, **over):
    out = {}
    for _hk, raw in proposals:
        doc = cp.parse_document(raw)
        out[cp.proposal_hash(raw)] = fx.listing(doc.provider_order, **over)
    return out


def _proposal(hotkey, model="lab-c/model-new", effort="high", **kw):
    return hotkey, fx.document(model, effort, proposer=hotkey, **kw)[1]


def _admit(base, proposals, *, listings=None, params=fx.PARAMS, **read_kw):
    return cp.admit(base, _read(base, proposals, **read_kw), _docs(proposals),
                    _listings(proposals) if listings is None else listings, fx.ALLOW, params)


# --- the document and the commitment fields --------------------------------------------


def test_a_document_has_one_canonical_form_and_a_bounded_size():
    doc, raw = fx.document()
    assert cp.parse_document(raw) == doc
    for bad in (json.dumps(cp.document_dict(doc), indent=1).encode(),        # another layout
                cp.canonical(dict(cp.document_dict(doc), x=1)),               # an unknown key
                raw[:-1],                                                     # not JSON
                b"{" + b'"a":"' + b"x" * cp.PROPOSAL_MAX_BYTES + b'"}',       # too large
                raw.replace(b'"domain":"fugal-chain-proposal-v1"', b'"domain":"other"')):
        with pytest.raises(ValueError):
            cp.parse_document(bad)


def test_a_document_pins_at_most_two_hosts_and_forbids_fallbacks():
    doc, raw = fx.document(hosts=("host-a/fp8", "host-b/fp8"))
    assert cp.parse_document(raw).provider_order == ("host-a/fp8", "host-b/fp8")
    d = cp.document_dict(doc)
    d["provider"]["order"] = ["host-a/fp8", "host-b/fp8", "host-c/fp8"]
    with pytest.raises(ValueError):
        cp.parse_document(cp.canonical(d))
    d = cp.document_dict(doc)
    d["provider"]["allow_fallbacks"] = True
    with pytest.raises(ValueError):
        cp.parse_document(cp.canonical(d))


def test_commitment_fields_have_one_spelling_each():
    p = "a" * 64
    assert cp.parse_proposal_field(cp.proposal_field(p)) == p
    for bad in ("proposal:" + "A" * 64, "proposal:" + "a" * 63, " proposal:" + p, p, None, 5):
        assert cp.parse_proposal_field(bad) is None
    assert cp.parse_join_field(cp.join_field(105, None), 105) == cp.JoinPointer(105, None)
    assert cp.parse_join_field(cp.join_field(105, p), 105) == cp.JoinPointer(105, p)
    assert cp.parse_join_field(cp.join_field(104, p), 105) is None          # another epoch
    assert cp.parse_join_field("join:e105:" + p, 105) is None               # not the epoch id
    assert len(cp.proposal_field(p)) <= 128 and len(cp.join_field(10**8, p)) <= 128


def test_a_reader_of_the_first_field_sees_the_head_whatever_the_second_holds():
    """The SDK's decode_metadata reads field 0 only: what every pre-v4
    reader of commitments goes through (commitments.get_commitments_with_blocks)."""
    from bittensor.core.chain_data.utils import decode_metadata

    head = "f" * 64
    value = {"block": 7, "info": {"fields": [
        {"Raw64": "0x" + head.encode().hex()},
        {"Raw73": "0x" + cp.proposal_field("a" * 64).encode().hex()}]}}
    assert decode_metadata(value) == head
    assert cp.commitment_fields(value) == (head, cp.proposal_field("a" * 64))
    one = {"block": 7, "info": {"fields": [{"Raw64": "0x" + head.encode().hex()}]}}
    assert decode_metadata(one) == head and cp.commitment_fields(one) == (head,)


def test_the_two_field_write_is_the_sdks_own_shape():
    info = cp.commitment_info(["f" * 64, cp.proposal_field("a" * 64)])
    assert info == {"fields": [[{"Raw64": b"f" * 64},
                                {"Raw73": cp.proposal_field("a" * 64).encode()}]]}
    for bad in ([], ["x"] * 4, ["x" * 129]):
        with pytest.raises(ValueError):
            cp.commitment_info(bad)


# --- admission -----------------------------------------------------------------------------


def test_a_valid_proposal_joins_as_provisional_at_the_boundary():
    base = fx.wide_state(epoch=101)
    p = _proposal("hk-1")
    delta = _admit(base, [p])
    assert [c.verdict for c in delta.considered] == [cp.JOINED]
    raw = cp.delta_bytes(delta)
    entries = cp.check_delta(base, raw, fx.ALLOW, params=fx.PARAMS)
    after = ms.apply_joins(base, entries)
    e = after.entry("lab-c/model-new@high")
    assert e.provisional and e.joined_epoch == 101 and e.rev == 1
    assert (e.price_in, e.price_out) == (300_000, 900_000)
    assert after.provisional_record(e.id).coldkey == "ck-hk-1"
    assert after.serial == base.serial + 1


@pytest.mark.parametrize("change, verdict", [
    (dict(serving=()), cp.NOT_SERVED),
    (dict(model="lab-z/unknown"), "refused:lab_not_allowed"),
    (dict(hosts=("host-z/fp8",)), "refused:host_not_allowed"),
    (dict(model="lab-a/model-one", effort="high"), "refused:already_on_menu"),
])
def test_refusals_name_the_rule(change, verdict):
    base = fx.wide_state(epoch=101)
    serving = change.pop("serving", None)
    p = _proposal("hk-1", **change)
    delta = _admit(base, [p], serving=serving)
    assert delta.considered[0].verdict == verdict
    assert cp.check_delta(base, cp.delta_bytes(delta), fx.ALLOW, params=fx.PARAMS) == ()


def test_a_document_that_does_not_match_its_hash_or_its_hotkey_is_not_admitted():
    base = fx.wide_state(epoch=101)
    hk, raw = _proposal("hk-1")
    read = _read(base, [(hk, raw)])
    other = fx.document("lab-c/model-two", "high", proposer=hk)[1]
    delta = cp.admit(base, read, {cp.proposal_hash(raw): other}, {}, fx.ALLOW, fx.PARAMS)
    assert delta.considered[0].verdict == cp.UNFETCHED
    stolen = _proposal("hk-2")[1]                       # hk-2's document committed by hk-1
    read = _read(base, [(hk, stolen)])
    delta = cp.admit(base, read, _docs([(hk, stolen)]), _listings([(hk, stolen)]), fx.ALLOW,
                     fx.PARAMS)
    assert delta.considered[0].verdict == cp.MALFORMED


def test_an_effort_is_admitted_only_when_every_pinned_host_takes_one():
    base = fx.wide_state(epoch=101)
    p = _proposal("hk-1")
    delta = _admit(base, [p], listings=_listings([p], efforts=False))
    assert delta.considered[0].verdict == "refused:effort_unsupported"
    doc = cp.ProposalDoc(proposer="hk-1", model="lab-c/model-new", reasoning=("enabled", True),
                         provider_order=("host-a/fp8",), quantization="fp8")
    plain = ("hk-1", cp.document_bytes(doc))            # thinking on, no effort named: allowed
    assert _admit(base, [plain], listings=_listings([plain], efforts=False)
                  ).considered[0].verdict == cp.JOINED


def test_a_host_the_listing_does_not_show_refuses_the_proposal():
    base = fx.wide_state(epoch=101)
    hk, raw = _proposal("hk-1")
    listing = {cp.proposal_hash(raw): (cp.HostListing("host-a/fp8", False),)}
    delta = _admit(base, [(hk, raw)], listings=listing)
    assert delta.considered[0].verdict == "refused:not_listed"


def test_one_join_an_epoch_below_ten_serving_hotkeys_and_two_above():
    base = fx.wide_state(epoch=101)
    props = [_proposal(f"hk-{i}", model=f"lab-c/model-{i}") for i in range(4)]
    small = _admit(base, props)
    assert sum(c.verdict == cp.JOINED for c in small.considered) == 1
    assert sum(c.verdict == cp.NO_PLACE for c in small.considered) == 3
    serving = {hk for hk, _ in props} | {f"other-{i}" for i in range(8)}
    large = _admit(base, props, serving=serving)
    assert sum(c.verdict == cp.JOINED for c in large.considered) == 2


def test_the_admission_order_is_the_hmac_of_block_b_and_cannot_be_chosen():
    base = fx.wide_state(epoch=101)
    props = [_proposal(f"hk-{i}", model=f"lab-c/model-{i}") for i in range(6)]
    winners = set()
    for b in ("11" * 32, "22" * 32, "33" * 32, "44" * 32):
        d = _admit(base, props, block_hash=b)
        order = [c.proposal for c in d.considered]
        assert order == sorted(order, key=lambda p: cp.place(b, p))
        winners.add(d.joined()[0].hotkey)
    assert len(winners) > 1                         # another block, another winner


def test_one_proposal_per_coldkey_an_epoch():
    base = fx.wide_state(epoch=101)
    props = [_proposal("hk-1", model="lab-c/model-1"), _proposal("hk-2", model="lab-c/model-2")]
    serving = {hk for hk, _ in props} | {f"other-{i}" for i in range(10)}
    d = _admit(base, props, serving=serving, coldkeys={"hk-1": "ck", "hk-2": "ck"})
    assert sorted(c.verdict for c in d.considered) == [cp.COLDKEY_ONCE, cp.JOINED]


def test_two_coldkeys_filing_the_same_entry_admit_the_first_in_order():
    base = fx.wide_state(epoch=101)
    props = [_proposal("hk-1"), _proposal("hk-2")]       # the same model and effort
    serving = {hk for hk, _ in props} | {f"other-{i}" for i in range(10)}
    d = _admit(base, props, serving=serving)
    assert [c.verdict for c in d.considered] == [cp.JOINED, "refused:already_on_menu"]


def test_a_proposal_in_cooldown_waits_and_after_it_is_a_fresh_filing():
    base = fx.wide_state(epoch=101, retired=(ms.Retired("lab-c/model-new@high", 90,
                                                        ms.REASON_GATE_CATEGORY, 138),))
    p = _proposal("hk-1")
    assert _admit(base, [p]).considered[0].verdict == "refused:entry_cooldown"
    later = dataclasses.replace(base, epoch=138)
    assert _admit(later, [p]).considered[0].verdict == cp.JOINED
    ck = dataclasses.replace(base, retired=(), coldkey_cooldowns=(ms.ColdkeyCooldown("ck-hk-1",
                                                                                       140),))
    assert _admit(ck, [p]).considered[0].verdict == "refused:proposer_cooldown"


def test_a_full_menu_or_a_full_provisional_count_admits_nothing():
    base = fx.wide_state(epoch=101, fillers=61)          # 64 entries: the cap
    assert len(base.entries) == ms.MENU_CAP
    p = _proposal("hk-1")
    assert _admit(base, [p], params=ms.READER_PARAMS).considered[0].verdict == cp.NO_PLACE


# --- what every node checks of a delta ---------------------------------------------------


def _delta_doc(base, proposals):
    return json.loads(cp.delta_bytes(_admit(base, proposals)))


def _bytes(doc):
    return cp.canonical(doc)


def test_a_delta_stating_a_verdict_the_rules_do_not_give_is_refused():
    base = fx.wide_state(epoch=101)
    junk = _proposal("hk-1", model="lab-z/unknown")
    doc = _delta_doc(base, [junk])
    doc["considered"][0]["verdict"] = cp.JOINED
    with pytest.raises(cp.DeltaRefused):
        cp.check_delta(base, _bytes(doc), fx.ALLOW, params=fx.PARAMS)


def test_a_delta_that_leaves_out_a_valid_proposal_as_unfetched_passes_every_node_but_the_audit():
    """A TD cannot tell an omission from a box that did not answer; a
    follower, reading the chain itself, sees the proposal and the root's
    word "unfetched" while the document is inside the delta: it does not
    match, the follower alarms."""
    base = fx.wide_state(epoch=101)
    p = _proposal("hk-1")
    honest = _admit(base, [p])
    doc = json.loads(cp.delta_bytes(honest))
    doc["considered"][0]["verdict"] = cp.UNFETCHED
    del doc["considered"][0]["document"], doc["considered"][0]["listing"]
    omitted = cp.parse_delta(_bytes(doc))
    assert cp.check_delta(base, _bytes(doc), fx.ALLOW, params=fx.PARAMS) == ()
    # A follower re-runs on what the root's delta carries: without the
    # document it reaches "unfetched" too (the root's word, stated plainly).
    follower = cp.admit(base, _read(base, [p]), *cp.contents_of(omitted), fx.ALLOW, fx.PARAMS)
    assert cp.audit(follower, omitted) == []
    # Leaving the proposal out of the delta altogether is seen.
    stated = cp.audit(follower, dataclasses.replace(omitted, considered=()))
    assert stated and "leaves out" in stated[0]


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(epoch=d["epoch"] + 1),
    lambda d: d.update(serving=-1),
    lambda d: d["considered"][0].update(document="{}"),
    lambda d: d["considered"][0].update(verdict="maybe"),
    lambda d: d.update(considered=d["considered"] * 2),
    lambda d: d.update(extra=1),
])
def test_malformed_deltas_are_refused(mutate):
    base = fx.wide_state(epoch=101)
    doc = _delta_doc(base, [_proposal("hk-1")])
    mutate(doc)
    with pytest.raises(cp.DeltaRefused):
        cp.check_delta(base, _bytes(doc), fx.ALLOW, params=fx.PARAMS)


def test_an_oversized_delta_is_refused_unparsed():
    base = fx.wide_state(epoch=101)
    with pytest.raises(cp.DeltaRefused):
        cp.check_delta(base, b" " * (cp.DELTA_MAX_BYTES + 1), fx.ALLOW, params=fx.PARAMS)


def test_a_delta_computed_on_another_base_is_stale_and_adds_nothing():
    base = fx.wide_state(epoch=101)
    raw = cp.delta_bytes(_admit(base, [_proposal("hk-1")]))
    other = dataclasses.replace(base, serial=base.serial + 1)
    with pytest.raises(cp.DeltaStale):
        cp.check_delta(other, raw, fx.ALLOW, params=fx.PARAMS)


def test_the_follower_audit_finds_a_join_the_chain_does_not_support():
    base = fx.wide_state(epoch=101)
    p = _proposal("hk-1")
    root = _admit(base, [p])
    mine = cp.admit(base, _read(base, [p], serving=()), _docs([p]), _listings([p]), fx.ALLOW,
                    fx.PARAMS)
    diff = cp.audit(mine, root)
    assert diff and "joined (root) vs not_served" in diff[-1]


# --- the gates ------------------------------------------------------------------------------


def _decide(base, gate, *, accuracy=(), extra=None):
    """The step scoring `base`'s epoch, from base carried one epoch (the
    entry joined at base.epoch is decided)."""
    start = ms.carry(base)
    inputs = fx.inputs(epoch=base.epoch, gates=(gate,) if gate else (), accuracy=tuple(accuracy),
                       **(extra or {}))
    return ms.step(start, inputs, fx.PARAMS)


def test_a_provisional_entry_that_passes_its_gates_is_full_from_the_steps_epoch():
    base = fx.provisional(fx.wide_state(epoch=100))
    after = _decide(base, fx.gate("lab-c/model-new@high"))
    e = after.entry("lab-c/model-new@high")
    assert not e.provisional and e.scored_from == 102
    assert e.measured_rate == ms.measured_rate(240, 240) and after.provisional == ()
    assert ms.check_transition(ms.carry(base), after, fx.PARAMS) is None


def test_the_answered_gate_is_judged_on_the_median_coldkey():
    base = fx.provisional(fx.wide_state(epoch=100))
    eid = "lab-c/model-new@high"
    one_bad = fx.gate(eid, coldkeys=(("ck-1", 60, 60), ("ck-2", 60, 0), ("ck-3", 60, 1)))
    assert not _decide(base, one_bad).entry(eid).provisional          # one operator moves nothing
    most_bad = fx.gate(eid, coldkeys=(("ck-1", 60, 60), ("ck-2", 60, 10), ("ck-3", 60, 1)))
    after = _decide(base, most_bad)
    assert eid not in after.ids()
    rec = {r.id: r for r in after.retired}[eid]
    assert rec.reason == ms.REASON_GATE_ANSWERED and rec.cooldown_until == 102 + 48
    assert after.cooldowns()[fx.COLDKEY] == 102 + 48
    assert ms.check_transition(ms.carry(base), after, fx.PARAMS) is None


def test_no_labels_at_all_fail_the_answered_gate():
    base = fx.provisional(fx.wide_state(epoch=100))
    after = _decide(base, None)
    assert {r.id: r.reason for r in after.retired}["lab-c/model-new@high"] == \
        ms.REASON_GATE_ANSWERED


def test_the_category_floor_and_the_usefulness_comparison():
    base = fx.provisional(fx.wide_state(epoch=100))
    eid = "lab-c/model-new@high"
    weak = fx.gate(eid, correct=60)
    assert {r.id: r.reason for r in _decide(base, weak).retired}[eid] == ms.REASON_GATE_CATEGORY
    cheap = base.entry("lab-a/model-one@low")
    acc = [ms.AccuracyStat(cheap.id, cheap.rev, 500, 990_000, 980_000)]
    after = _decide(base, fx.gate(eid, upper_ppm=950_000), accuracy=acc)
    assert {r.id: r.reason for r in after.retired}[eid] == ms.REASON_GATE_USEFUL


def test_a_sibling_setting_that_does_as_well_at_no_greater_cost_is_refused():
    """lab-a/model-one is on the menu at high; the same model at xhigh,
    priced the same with the same outcomes, takes no second place; at a
    much lower cost, or clearly better, it stays."""
    wide = fx.wide_state(epoch=100)
    on = wide.entry("lab-a/model-one@high")
    base = fx.provisional(wide, "lab-a/model-one", "xhigh", hosts=on.provider_order,
                          price_in=on.price_in, price_out=on.price_out)
    eid = "lab-a/model-one@xhigh"
    acc = [ms.AccuracyStat(on.id, on.rev, 500, 800_000, 770_000)]
    same = fx.gate(eid, correct=192, tokens_in=250, tokens_out=900, lower_ppm=750_000,
                   upper_ppm=850_000)
    assert {r.id: r.reason for r in _decide(base, same, accuracy=acc).retired}[eid] == \
        ms.REASON_GATE_SIBLING
    better = fx.gate(eid, correct=230, tokens_in=250, tokens_out=900, lower_ppm=900_000,
                     upper_ppm=980_000)
    assert not _decide(base, better, accuracy=acc).entry(eid).provisional
    cheaper = fx.gate(eid, correct=192, tokens_in=250, tokens_out=300, lower_ppm=750_000,
                      upper_ppm=850_000)
    assert not _decide(base, cheaper, accuracy=acc).entry(eid).provisional


def test_an_entry_that_joined_after_the_scored_epoch_stays_provisional():
    base = fx.provisional(fx.wide_state(epoch=101))       # joined at 101; the step scores 100
    start = base
    after = ms.step(start, fx.inputs(epoch=100, gates=(fx.gate("lab-c/model-new@high"),)),
                    fx.PARAMS)
    assert after.entry("lab-c/model-new@high").provisional


def test_the_reader_refuses_a_step_that_joins_an_entry_or_skips_a_gate():
    base = fx.provisional(fx.wide_state(epoch=100))
    start = ms.carry(base)
    honest = _decide(base, fx.gate("lab-c/model-new@high"))
    joined = fx.provisional(dataclasses.replace(honest, epoch=101), "lab-c/model-x")
    joined = dataclasses.replace(joined, epoch=102)
    assert "joins in a step" in ms.check_transition(start, joined, fx.PARAMS)
    kept = ms.step(start, fx.inputs(epoch=100, gates=(fx.gate("lab-c/model-new@high"),)),
                   fx.PARAMS)
    still = dataclasses.replace(kept, entries=start.entries, provisional=start.provisional)
    assert "decides it" in ms.check_transition(start, still, fx.PARAMS)
    early = dataclasses.replace(honest, entries=tuple(
        dataclasses.replace(e, scored_from=101) if e.id == "lab-c/model-new@high" else e
        for e in honest.entries))
    assert "scores it from 102" in ms.check_transition(start, early, fx.PARAMS)


def test_a_provisional_entry_and_a_shielded_one_are_never_displaced():
    e = fx.entry("lab-c/model-z", None, provider_order=("host-c",), scored_from=95,
                 joined_epoch=90)
    assert e.shielded(100) and e.shielded(114) and not e.shielded(115)
    base = fx.provisional(fx.wide_state(epoch=100))
    assert base.entry("lab-c/model-new@high").shielded(10**6)


def test_coverage_is_aimed_first_at_the_entry_in_its_join_epoch():
    from fugal_subnet import epoch_plan

    base = fx.provisional(fx.wide_state(epoch=100))
    assert epoch_plan.provisional_first(base) == ("lab-c/model-new@high",)
    shares = epoch_plan.explore_shares_v4(base)
    probs = shares.part_probabilities()["coverage"]
    assert probs["lab-c/model-new@high"] == 1
    assert epoch_plan.provisional_first(ms.carry(base)) == ()          # its second epoch: not first


def test_labels_drawn_while_provisional_score_for_no_head_and_a_missing_row_is_odds_zero():
    import numpy as np

    from fugal_subnet import labels_v4

    base = fx.provisional(fx.wide_state(epoch=100))
    eid = "lab-c/model-new@high"
    assert labels_v4.missing_odds(base, [eid, "lab-a/model-one@high"]).tolist() == [
        labels_v4.PROVISIONAL_LABEL, -1]
    full = dataclasses.replace(base, epoch=102, entries=tuple(
        dataclasses.replace(e, scored_from=102) if e.id == eid else e for e in base.entries),
        provisional=())
    assert labels_v4.missing_odds(full, [eid]).tolist() == [-1]
    from fugal_subnet.champion import EpochPairs, Window

    z = np.zeros(2, dtype=np.int32)
    pairs = EpochPairs(question_ids=("q1", "q2"), models=(eid, eid), benchmarks=("b", "b"),
                       categories=("math", "math"), y=np.array([10_000, 0], dtype=np.int32),
                       prompt_tokens=z, completion_tokens=z)
    lp = labels_v4.label_pairs(pairs, [("ck-a",), ("ck-a",)],
                               [labels_v4.PROVISIONAL_LABEL] * 2)
    x = labels_v4.Side("h" * 64, "owner", 0)
    assert not labels_v4.eligible(lp, 100, x, labels_v4.REFERENCE, now=100).any()
    window = Window(size=48, slices={}, heads={}, pairs={100: labels_v4.label_pairs(pairs, [("ck-a",), ("ck-a",)], [-1, -1])},
                    odds={"h" * 64: {100: np.array([-1, -1], dtype=np.int16)}})
    assert labels_v4._scored(window, None, x, 100).tolist() == [0, 0]


def test_the_cut_to_the_most_considered_comes_after_registration_and_serving():
    """Commitments of hotkeys that are not registered at B or did not serve
    the epoch never take a place among the MAX_CONSIDERED proposals."""
    base = fx.wide_state(epoch=101)
    honest, raw = _proposal("5Z" + "h" * 46)
    fields = {honest: ("ab" * 32, cp.proposal_field(cp.proposal_hash(raw)))}
    for i in range(cp.MAX_CONSIDERED):
        p = cp.proposal_field(f"{i:064x}")
        fields[f"5A{i:046d}"] = ("cd" * 32, p)            # sorts first, never served
        fields[f"5B{i:046d}"] = ("cd" * 32, p)            # served, not registered at B
    fields["bad hotkey"] = ("cd" * 32, cp.proposal_field("ee" * 32))
    serving = {honest, "bad hotkey"} | {f"5B{i:046d}" for i in range(cp.MAX_CONSIDERED)}
    read = cp.candidates(fields, {honest: "ck-honest", "bad hotkey": "ck"}, epoch=101,
                         block=101 * 360 - 30, block_hash=BLOCK_HASH, serving=serving,
                         reveal=REVEAL)
    assert read.proposals == ((honest, cp.proposal_hash(raw)),)
    delta = cp.admit(base, read, _docs([(honest, raw)]), _listings([(honest, raw)]), fx.ALLOW,
                     fx.PARAMS)
    assert [c.hotkey for c in delta.joined()] == [honest]
    # Still at most MAX_CONSIDERED, the lowest eligible hotkeys.
    many = {f"5C{i:046d}": ("cd" * 32, cp.proposal_field(f"{i:064x}"))
            for i in range(cp.MAX_CONSIDERED + 5)}
    read = cp.candidates(many, dict.fromkeys(many, "ck"), epoch=101, block=101 * 360 - 30,
                         block_hash=BLOCK_HASH, serving=set(many), reveal=REVEAL)
    assert [hk for hk, _ in read.proposals] == sorted(many)[:cp.MAX_CONSIDERED]


def test_a_delta_stating_more_serving_hotkeys_than_any_subnet_holds_is_refused_unchecked():
    """The check every node and TD makes stands in for each serving hotkey a
    delta states: a delta above MAX_SERVING is refused before it."""
    base = fx.wide_state(epoch=101)
    delta = _admit(base, [_proposal("hk-1")])
    doc = json.loads(cp.delta_bytes(delta))
    assert cp.parse_delta(cp.canonical(dict(doc, serving=cp.MAX_SERVING))).serving == \
        cp.MAX_SERVING
    for serving in (cp.MAX_SERVING + 1, 2**62):
        with pytest.raises(cp.DeltaRefused, match="serving"):
            cp.check_delta(base, cp.canonical(dict(doc, serving=serving)), fx.ALLOW,
                           params=fx.PARAMS)
