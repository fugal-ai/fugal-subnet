"""Hand-made protocol v4 menu states and step inputs for the tests.

Nothing here reads the network, the chain or the pinned seed: every value is
written out, so a test states exactly the menu and the inputs it is about.
"""
from __future__ import annotations

import hashlib

from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms

# The scored category mix the tests weight token means to (integer weights).
MIX = (("code", 15), ("hard", 20), ("knowledge", 20), ("math", 45))
# Refreshed in every step, so that each test states only the rule it is
# about; tests/test_menu_refresh.py holds the compiled cadence. The menus here
# are three entries, below the compiled minimum (menu_state.MENU_MIN_ENTRIES);
# tests/test_menu_liveness.py holds the minimum on `wide_state()`.
PARAMS = ms.MenuParams(mix=MIX, min_category_accuracy_bp=5000, refresh_epochs=1, min_entries=1)
NONCE = bytes(range(32))
HOTKEY, COLDKEY = "5Fproposer-hotkey", "5Fproposer-coldkey"

LABS = ("lab-a", "lab-b", "lab-c", "lab-d", "lab-q")
HOSTS = ("host-a", "host-b", "host-c", "host-d", "host-e")
# The hosts an entry may pin without a quantization (a closed-weights lab's),
# per lab (list format 2): here the same three for every lab of the fixtures.
FIRST_PARTY = {lab: ("host-c", "host-d", "host-e") for lab in LABS}


def allowlist(serial: int = 2, *, labs=LABS, hosts=HOSTS, first_party=FIRST_PARTY,
              signer: str | None = "SHA256:menu-fixture-key") -> mr.MenuAllowlist:
    """A lab and host list as `verify_signed_list` returns one: its sha256 is
    the hash of its canonical bytes, and `signer` says a signature was
    verified (no key is involved here; pass None for a list nobody verified).
    `first_party` is {lab: hosts}; labs not in `labs` are left out."""
    fp = {lab: tuple(hs) for lab, hs in dict(first_party).items() if lab in labs and hs}
    raw = mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, serial, labs=labs, hosts=hosts,
                                  first_party=fp)
    return mr.MenuAllowlist(serial=serial, labs=frozenset(labs), hosts=frozenset(hosts),
                            first_party=mr.first_party_map(fp),
                            sha256=hashlib.sha256(raw).hexdigest(), signer=signer)


# The list in force in `state()`.
ALLOW = allowlist()


def device_list(serial: int = 1, devices=("ab" * 32, "cd" * 32), *,
                signer: str | None = "SHA256:menu-fixture-key") -> mr.DeviceList:
    """A device list as `verify_signed_list` returns one (see `allowlist`)."""
    raw = mr.canonical_list_bytes(mr.KIND_DEVICE_LIST, serial, devices=devices)
    return mr.DeviceList(serial=serial, devices=frozenset(devices),
                         sha256=hashlib.sha256(raw).hexdigest(), signer=signer)


def entry(model: str = "lab-a/model-one", effort: str | None = "high", **over) -> ms.Entry:
    fields = dict(
        id=f"{model}@{effort}" if effort else model, rev=1, model=model,
        reasoning=("effort", effort) if effort else ("enabled", True), max_tokens=131072,
        provider_order=("host-a/fp8", "host-b/fp8"),
        price_in=100_000, price_out=400_000, mean_in=250_000, mean_out=900_000,
        token_pairs=120, joined_epoch=0, scored_from=0, measured_rate=5000, proposer="")
    fields.update(over)
    # The one quantization its hosts carry, unless the test says otherwise.
    fields.setdefault("quantization", mr.pinned_quantization(fields["provider_order"]))
    return ms.Entry(**fields)


def state(*entries: ms.Entry, epoch: int = 101, **over) -> ms.MenuState:
    """A menu for `epoch` holding `entries` (three by default)."""
    if not entries:
        entries = (entry("lab-a/model-one", "high"),
                   entry("lab-a/model-one", "low", price_in=100_000, price_out=400_000,
                         mean_out=300_000),
                   entry("lab-b/model-two", None, provider_order=("host-c",),
                         price_in=2_000_000, price_out=10_000_000))
    fields = dict(epoch=epoch, serial=3, entries=tuple(sorted(entries, key=lambda e: e.id)),
                  retired=(),
                  labs=ms.AllowlistRef(ALLOW.serial, ALLOW.sha256),
                  devices=ms.AllowlistRef(0, ""))
    fields.update(over)
    out = ms.MenuState(**fields)
    ms.validate(out)
    return out


def wide_state(*extra: ms.Entry, epoch: int = 101, fillers: int = 6, **over) -> ms.MenuState:
    """`state()`'s three entries, `fillers` more on a first-party host of
    lab-d (`lab-d/filler-<i>`), and `extra`: a menu above the compiled
    minimum (menu_state.MENU_MIN_ENTRIES), for the tests of rules that read
    the menu's size."""
    base = state(epoch=epoch).entries
    filler = tuple(entry(f"lab-d/filler-{i}", None, provider_order=("host-d",))
                   for i in range(fillers))
    return state(*(base + filler + extra), epoch=epoch, **over)


def inputs(epoch: int = 100, **over) -> ms.StepInputs:
    """Inputs of scored epoch `epoch` with nothing in them but the list in
    force; `state()` is the menu of epoch 101, so these derive the menu of
    epoch 102."""
    over.setdefault("lab_list", ALLOW)
    over.setdefault("nonce", NONCE)
    return ms.StepInputs(epoch=epoch, **over)


def read(model: str, host: str, *, price_in: int = 100_000, price_out: int = 400_000,
         limit: int = 131072, listed: bool = True, voters: int = 5) -> ms.HostRead:
    return ms.HostRead(model=model, host=host, voters=voters, listed=listed,
                       price_in=price_in, price_out=price_out, limit=limit)


def document(model: str = "lab-c/model-new", effort: str | None = "high", *,
             hosts=("host-a/fp8",), proposer: str = HOTKEY, quantization: str = "fp8"):
    """A chain proposal document (chain_proposals.ProposalDoc) and its bytes."""
    from fugal_subnet import chain_proposals as cp

    doc = cp.ProposalDoc(proposer=proposer, model=model,
                         reasoning=("effort", effort) if effort else None,
                         provider_order=tuple(hosts), quantization=quantization)
    return doc, cp.document_bytes(doc)


def listing(hosts=("host-a/fp8",), *, price_in: int = 300_000, price_out: int = 900_000,
            limit: int = 200_000, efforts: bool = True):
    """The root's read of a proposal's hosts' listing: every host listed."""
    from fugal_subnet import chain_proposals as cp

    return tuple(cp.HostListing(h, True, price_in, price_out, limit, efforts)
                 for h in sorted(hosts))


def gate(entry_id: str, *, rev: int = 1, coldkeys=(("ck-1", 60, 0), ("ck-2", 60, 0),
                                                   ("ck-3", 60, 0)),
         labels: int = 240, correct: int | None = None, tokens_in: int = 200,
         tokens_out: int = 1000, upper_ppm: int = 900_000, lower_ppm: int = 0,
         categories=("code", "hard", "knowledge", "math")) -> ms.GateStat:
    """What the gates read of a provisional entry: each coldkey's (calls,
    failed), and `labels` labels split as evenly as they go over `categories`
    (a remainder to the first ones), all correct unless `correct` says."""
    correct = labels if correct is None else correct
    n = len(categories)

    def share(total: int, i: int) -> int:
        return total // n + (1 if i < total % n else 0)

    cells = tuple(
        ms.MeasureCell(category=c, calls=share(labels, i), answered=share(labels, i),
                       correct=share(correct, i), tokens_in=tokens_in * share(labels, i),
                       tokens_out=tokens_out * share(labels, i))
        for i, c in enumerate(sorted(categories)))
    return ms.GateStat(entry_id=entry_id, rev=rev,
                       coldkeys=tuple(ms.CallTally(*c) for c in sorted(coldkeys)),
                       cells=cells, upper_ppm=upper_ppm, lower_ppm=lower_ppm)


def provisional(base: ms.MenuState, model: str = "lab-c/model-new", effort: str | None = "high",
                *, hosts=("host-a/fp8",), proposer: str = HOTKEY, coldkey: str = COLDKEY,
                **listing_over) -> ms.MenuState:
    """`base` with one entry joined as provisional at base's epoch, as a join
    delta admitting `document(model, effort, hosts=hosts)` leaves it."""
    import hashlib

    from fugal_subnet import chain_proposals as cp

    doc, raw = document(model, effort, hosts=hosts, proposer=proposer)
    price_in, price_out, max_tokens = cp.priced(doc, listing(hosts, **listing_over))
    return ms.apply_joins(base, [ms.NewEntry(
        request=doc.request(max_tokens), price_in=price_in, price_out=price_out,
        proposal=hashlib.sha256(raw).hexdigest(), proposer=proposer, coldkey=coldkey)], PARAMS)


def allow_small_menus(monkeypatch) -> None:
    """Let the compiled reader (menu_state.READER_PARAMS, which every node's
    reader uses when it is handed no parameters) accept this module's
    three-entry menus: tests of the publishing, session and harness plumbing
    run on `state()`, below menu_state.MENU_MIN_ENTRIES. The minimum itself
    is held in tests/test_menu_liveness.py, on `wide_state()`."""
    import dataclasses

    monkeypatch.setattr(ms, "READER_PARAMS", dataclasses.replace(ms.READER_PARAMS, min_entries=1))
