#!/usr/bin/env python3
"""Build the protocol v4 menu seed (data/menu_v4_seed.json).

The seed is the menu of the first v4 epoch (docs/decisions/LIVE_MENU_V4.md,
section 4.5): v3 menu B's entries with their requests, prices and answer
limits read from a saved OpenRouter listing, and token means measured on the
pairs the v3 field published. An entry none of whose pinned hosts is listed,
or that fails the design's health rule on those pairs, is left out. Every
entry kept must pass the compiled rules (fugal_subnet/menu_rules.py) as an
entry on the menu, against that listing and the lab and host list.

Nothing here touches the network. Every input is a file, and the same files
give the same seed byte for byte:

    python scripts/build_menu_v4_seed.py build \\
        --listing data/observations/openrouter_endpoints_2026-11-18.json \\
        --pairs data/observations/menu_v4_seed_pairs_e00026466-e00026484.json \\
        --picks data/observations/menu_v4_seed_picks_e00026466-e00026509.json

writes the reviewed copy, data/menu_v4_seed.json, and the copy the package
ships, fugal_subnet/routing_data/menu_v4_seed.json. Those three inputs are the
defaults. `build --check` rebuilds and compares with both copies instead of
writing them.

THE THREE INPUT FILES are themselves built by this script from public pages
saved with a generic client and no key, so anyone can repeat them:

    # the listing: one page a model, as served
    for m in <each `model` of data/models_v3b.json>; do
      curl -A listing-snapshot/1.0 -o "pages/${m/\\//__}.json" \\
        "https://openrouter.ai/api/v1/models/$m/endpoints"; done
    python scripts/build_menu_v4_seed.py snapshot pages --fetched-utc 2026-11-18T17:29:56Z \\
        --out data/observations/openrouter_endpoints_2026-11-18.json

    # the pairs: the published v3 reveals of the named epochs
    for e in e00026466 ... e00026484; do
      curl -o "reveals/reveal_$e.json" \\
        "https://storage.googleapis.com/fugal-reveals/70/$e/reveal.json"; done
    python scripts/build_menu_v4_seed.py pairs reveals --first e00026466 --last e00026484 \\
        --out data/observations/menu_v4_seed_pairs_e00026466-e00026484.json

    # the graded cells and picks: the same reveals, through e00026509
    python scripts/build_menu_v4_seed.py picks reveals --first e00026466 --last e00026509 \\
        --out data/observations/menu_v4_seed_picks_e00026466-e00026509.json

The listing file keeps each page's bytes and their sha256. The pairs file
keeps each reveal's sha256 and, per entry, integer sums only: pairs and
attested tokens by benchmark over all the named epochs, and calls and failed
calls by operator over the last HEALTH_EPOCHS of them (operators are not
named: the health rule reads the counts alone). The picks file keeps each
reveal's sha256 and, per reveal, entry and benchmark, integer sums only: its
graded scored-question cells (the published shared cells) and their
completion tokens, the cells the champion and all served heads picked and
those cells' completion tokens, and its pooled exploration pairs and their
completion tokens. No head, operator or answer is named.

`accuracy` prints how well each cost table predicts what the picks really
cost, leaving one epoch out (the claim of SEAT_MINIMAL_V4.md 3.4 and the
earlier seat design's 8.1), from the picks file alone.

RULES (each is the design's rule for a live menu, applied once at build time):

- Request: the v3 menu B row's model, reasoning, provider order and no
  fallbacks, unchanged. `rev` 1.
- Answer limit (`max_tokens`): menu_rules.answer_limit over the pinned hosts
  the listing shows, the rule the step applies to an entry every epoch:
  min(131072, the smallest answer limit among them). A pinned host the
  listing does not show is skipped.
- Price: the dearer listed pinned host's, in integer picodollars per token:
  the higher input price and the higher output price among the pinned hosts
  the listing shows. The listing gives decimal strings, exact in picodollars.
- Quantization: the listed pinned hosts', which must agree.
- Token means: the mean attested tokens of the entry's pooled pairs, by
  category, weighted to the pinned v4 mix (menu_state.weighted_token_means;
  epoch_plan.mix_v4), in
  milli-tokens. With fewer than TOKEN_MIN_PAIRS pairs the previous value
  stands: the row of v3 menu B's pinned token manifest.
- Exploration pairs (`explore_pairs`, section 8's n_e): the entry's pooled
  pairs in the named reveals, all of them exploration pairs (v3 pools
  nothing else), whether or not the token means come from them.
- Pick-cost table (`pick`, W13; menu_state.pick_table_from_sums): the
  entry's graded cells in the picks file's reveals, by v4 category
  (epoch_plan.mix_v4's), shrunk toward the token means above as the step
  shrinks them (PICK_COST_PRIOR_CELLS_ENTRY and _CATEGORY). A v3 reveal
  publishes a cell's coldkey-median completion tokens and not its prompt
  tokens, so the prompt side of every cell is taken at the entry's call
  mean (the table's prompt means equal its call means); a cell's
  completion tokens are rounded half up and held to the answer limit the
  call was sent with (v3 menu B's `max_tokens`), as the live rule holds
  them to the epoch's menu (menu_inputs.graded_cells). An entry with no
  graded cell has no table: it is priced at its call cost.
- Left out: an entry with no pinned host in the listing; an entry whose
  median failure rate across operators (those with at least
  HEALTH_MIN_CALLS calls, at least HEALTH_MIN_OPERATORS of them) is above
  HEALTH_FAIL_RATE over the last HEALTH_EPOCHS named epochs.
- Seed entries are on the menu from the start and have no allowance: a
  head without a row for one is certainly wrong on it from the first v4
  epoch. `joined_epoch` and `scored_from` are 0, `proposer` is empty, and
  `measured_rate` is what no measurement gives, (0 + 1) / (0 + 2), never
  read (SEAT_MINIMAL_V4.md 4.1).
- The rules: each entry passes menu_rules.check_entry under the lab and host
  list (the signed list the release ships, or the draft under review until
  the first one is signed) with the listing's answer limits. The build stops
  on a refusal.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fugal_subnet import menu_state  # noqa: E402  (the repo root is on the path from here)

LISTING_FORMAT = "openrouter-endpoints-snapshot-v1"
LISTING_SOURCE = "https://openrouter.ai/api/v1/models/{model}/endpoints"
PAIRS_FORMAT = "menu-seed-pairs-v1"
REVEAL_SOURCE = "https://storage.googleapis.com/fugal-reveals/70/{epoch_id}/reveal.json"
MENU_PATH = os.path.join(ROOT, "data", "models_v3b.json")
MANIFEST_PATH = os.path.join(ROOT, "data", "benchmark_tokens_protocol_v3b.json")
SEED_PATH = os.path.join(ROOT, "data", "menu_v4_seed.json")
PACKAGE_SEED_PATH = os.path.join(ROOT, "fugal_subnet", "routing_data", "menu_v4_seed.json")
LISTING_PATH = os.path.join(ROOT, "data", "observations", "openrouter_endpoints_2026-11-18.json")
PAIRS_PATH = os.path.join(ROOT, "data", "observations",
                          "menu_v4_seed_pairs_e00026466-e00026484.json")
PICKS_FORMAT = "menu-seed-picks-v1"
PICKS_PATH = os.path.join(ROOT, "data", "observations",
                          "menu_v4_seed_picks_e00026466-e00026509.json")
# The columns of a picks row after (reveal, entry, benchmark): counts and
# completion-token sums of the graded cells, of the cells the champion
# picked, of the cells every served head picked (each head once), and of the
# pooled exploration pairs.
PICKS_COLUMNS = ("graded", "graded_out", "champion", "champion_out", "heads", "heads_out",
                 "pairs", "pairs_out")
DRAFT_LIST_PATH = os.path.join(ROOT, "data", "menu_allowlist.draft.json")

# The step's constant, read from menu_state and not restated here.
TOKEN_MIN_PAIRS = menu_state.TOKEN_MIN_PAIRS
# The seed's own health filter, applied once to the named reveals: the median
# failure rate across operators with enough calls. It is not the live rule of
# section 9.2 (menu_inputs.health_stats: a majority of eligible coldkeys,
# failed calls in distinct epochs, the outage guard), which needs per-epoch
# counts the pairs file does not keep. The seed it built is pinned by hash.
HEALTH_MIN_OPERATORS = 3           # operators with enough calls the median needs
HEALTH_FAIL_RATE = Fraction(1, 5)  # the median failure rate above which an entry is left out
HEALTH_EPOCHS = 12                 # "over the last 12 epochs"
HEALTH_MIN_CALLS = 10              # "coldkeys with at least 10 calls"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _epoch_number(epoch_id: str) -> int:
    """The number in a published reveal's epoch id, to order and range-check
    the input files. Ids are read, never formatted, here."""
    if len(epoch_id) < 2 or epoch_id[0] != "e" or not epoch_id[1:].isdigit():
        raise SystemExit(f"{epoch_id!r} is not an epoch id")
    return int(epoch_id[1:])


def _lines(head: dict, name: str, rows: list[str]) -> str:
    """A JSON object with `rows` (already encoded) one a line under `name`."""
    out = ["{"]
    for k in sorted(head):
        out.append(f"{json.dumps(k)}:{json.dumps(head[k], sort_keys=True, separators=(',', ':'))},")
    out.append(f"{json.dumps(name)}:[")
    out.append(",\n".join(rows))
    out.append("]")
    out.append("}")
    return "\n".join(out) + "\n"


# --- snapshot: saved listing pages -> one file -----------------------------------


def snapshot(pages_dir: str, fetched_utc: str) -> str:
    """The listing file's text: every saved page, as served, with its sha256."""
    models = sorted({r["model"] for r in json.load(open(MENU_PATH, encoding="utf-8"))})
    rows = []
    for model in models:
        path = os.path.join(pages_dir, model.replace("/", "__") + ".json")
        raw = open(path, "rb").read()
        text = raw.decode("utf-8")
        if json.loads(text)["data"]["id"] != model:
            raise SystemExit(f"{path} is not the page of {model}")
        rows.append(json.dumps({"model": model, "sha256": _sha256(raw), "raw": text},
                               sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    head = {"format": LISTING_FORMAT, "source": LISTING_SOURCE, "fetched_utc": fetched_utc}
    return _lines(head, "pages", rows)


def read_listing(path: str) -> dict[str, dict[str, dict]]:
    """{model: {host tag: endpoint}} of a listing file, each page checked
    against its recorded sha256."""
    doc = json.load(open(path, encoding="utf-8"))
    if doc.get("format") != LISTING_FORMAT:
        raise SystemExit(f"{path} is not a {LISTING_FORMAT} file")
    out: dict[str, dict[str, dict]] = {}
    for page in doc["pages"]:
        if _sha256(page["raw"].encode("utf-8")) != page["sha256"]:
            raise SystemExit(f"{path}: the page of {page['model']} does not match its sha256")
        data = json.loads(page["raw"])["data"]
        if data["id"] != page["model"] or page["model"] in out:
            raise SystemExit(f"{path}: page {page['model']} is mislabelled or repeated")
        hosts: dict[str, dict] = {}
        for endpoint in data["endpoints"]:
            if endpoint["tag"] in hosts:
                raise SystemExit(f"{path}: {page['model']} lists host {endpoint['tag']} twice")
            hosts[endpoint["tag"]] = endpoint
        out[page["model"]] = hosts
    return out


# --- pairs: published reveals -> integer sums --------------------------------------


def pairs(reveals_dir: str, first: str, last: str) -> str:
    """The pairs file's text, from `reveal_<epoch id>.json` files of every
    epoch from `first` to `last`."""
    lo, hi = _epoch_number(first), _epoch_number(last)
    files = {}
    for name in sorted(os.listdir(reveals_dir)):
        if name.startswith("reveal_") and name.endswith(".json"):
            files[_epoch_number(name[len("reveal_"):-len(".json")])] = os.path.join(reveals_dir, name)
    missing = [n for n in range(lo, hi + 1) if n not in files]
    if missing or hi < lo:
        raise SystemExit(f"no reveal file for epoch number(s) {missing} in {reveals_dir}")
    health_from = max(lo, hi - HEALTH_EPOCHS + 1)
    reveals, cells, calls = [], {}, {}
    for n in range(lo, hi + 1):
        raw = open(files[n], "rb").read()
        reveal = json.loads(raw)
        epoch_id = reveal["epoch_id"]
        if _epoch_number(epoch_id) != n or reveal.get("protocol_version") != 3:
            raise SystemExit(f"{files[n]} is not the protocol v3 reveal of epoch number {n}")
        reveals.append({"epoch_id": epoch_id, "sha256": _sha256(raw), "bytes": len(raw)})
        seen = set()
        for p in reveal["exploration_pool"]:
            key = (p["question_id"], p["model"])
            if key in seen:
                raise SystemExit(f"{epoch_id}: pair {key} is pooled twice")
            seen.add(key)
            c = cells.setdefault(p["model"], {}).setdefault(p["benchmark"], [0, 0, 0])
            for i, v in enumerate((1, p["prompt_tokens"], p["completion_tokens"])):
                if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                    raise SystemExit(f"{epoch_id}: pair {key} holds a token count that is not a count")
                c[i] += v
        if n >= health_from:
            for x in reveal["exploration"]:
                c = calls.setdefault(x["model"], {}).setdefault(x["coldkey"], [0, 0])
                c[0] += 1
                c[1] += 1 if x["failed"] else 0
    entries = {}
    for entry_id in sorted(set(cells) | set(calls)):
        entries[entry_id] = {
            "cells": {b: cells[entry_id][b] for b in sorted(cells.get(entry_id, {}))},
            # One [calls, failed] an operator, sorted: the health rule needs no names.
            "calls": sorted(calls.get(entry_id, {}).values()),
        }
    head = {"format": PAIRS_FORMAT, "source": REVEAL_SOURCE, "reveals": reveals,
            "health_epochs": [reveals[health_from - lo]["epoch_id"], reveals[-1]["epoch_id"]]}
    rows = [json.dumps({"id": k, **v}, sort_keys=True, separators=(",", ":"))
            for k, v in entries.items()]
    return _lines(head, "entries", rows)


def read_pairs(path: str) -> dict[str, dict]:
    doc = json.load(open(path, encoding="utf-8"))
    if doc.get("format") != PAIRS_FORMAT:
        raise SystemExit(f"{path} is not a {PAIRS_FORMAT} file")
    out = {}
    for row in doc["entries"]:
        if row["id"] in out:
            raise SystemExit(f"{path}: entry {row['id']} appears twice")
        out[row["id"]] = row
    return out


# --- picks: published reveals -> graded cells and picks -------------------------------


def _round_half_up(x: Fraction) -> int:
    return int((2 * x + 1) // 2)


def picks(reveals_dir: str, first: str, last: str) -> str:
    """The picks file's text, from `reveal_<epoch id>.json` files of every
    epoch from `first` to `last` (see the module docstring)."""
    lo, hi = _epoch_number(first), _epoch_number(last)
    files = {}
    for name in sorted(os.listdir(reveals_dir)):
        if name.startswith("reveal_") and name.endswith(".json"):
            files[_epoch_number(name[len("reveal_"):-len(".json")])] = os.path.join(reveals_dir, name)
    missing = [n for n in range(lo, hi + 1) if n not in files]
    if missing or hi < lo:
        raise SystemExit(f"no reveal file for epoch number(s) {missing} in {reveals_dir}")
    limit = {r["id"]: r["max_tokens"] for r in json.load(open(MENU_PATH, encoding="utf-8"))}
    reveals, acc, clamped = [], {}, 0

    def add(key, column, tokens):
        row = acc.setdefault(key, [0] * len(PICKS_COLUMNS))
        row[2 * column] += 1
        row[2 * column + 1] += tokens

    for n in range(lo, hi + 1):
        raw = open(files[n], "rb").read()
        reveal = json.loads(raw)
        epoch_id = reveal["epoch_id"]
        if _epoch_number(epoch_id) != n or reveal.get("protocol_version") != 3:
            raise SystemExit(f"{files[n]} is not the protocol v3 reveal of epoch number {n}")
        r = len(reveals)
        reveals.append({"epoch_id": epoch_id, "sha256": _sha256(raw), "bytes": len(raw)})
        bench = {q["question_id"]: q["benchmark"] for q in reveal["questions"]}
        cell = {}
        for g in reveal["shared_grades"]:
            key = (g["question_id"], g["model"])
            if key in cell:
                raise SystemExit(f"{epoch_id}: cell {key} is published twice")
            tokens = _round_half_up(Fraction(repr(g["completion_tokens"])))
            if tokens < 0:
                raise SystemExit(f"{epoch_id}: cell {key} holds a negative token count")
            if tokens > limit[g["model"]]:
                clamped += 1
                tokens = limit[g["model"]]
            cell[key] = tokens
            add((r, g["model"], bench[g["question_id"]]), 0, tokens)
        # routing_decisions are per uid, one model index per question in
        # question-id order; each served head counts once, on its uid.
        by_id = sorted(bench)
        champion = (reveal.get("champion") or {}).get("champion")
        serving = (reveal.get("v2_state") or {}).get("serving") or {}
        for head in sorted(serving):
            uid = str(serving[head])
            row = reveal["routing_decisions"].get(uid)
            if row is None:
                continue
            if len(row) != len(by_id):
                raise SystemExit(f"{epoch_id}: uid {uid} routes {len(row)} of {len(by_id)} questions")
            failed = set(reveal["failed"].get(uid) or [])
            for qid, j in zip(by_id, row):
                model = reveal["models"][j]
                if qid in failed or (qid, model) not in cell:
                    continue
                key = (r, model, bench[qid])
                add(key, 2, cell[(qid, model)])
                if head == champion:
                    add(key, 1, cell[(qid, model)])
        for p in reveal["exploration_pool"]:
            if not isinstance(p["completion_tokens"], int) or p["completion_tokens"] < 0:
                raise SystemExit(f"{epoch_id}: a pooled pair holds a token count that is not a count")
            add((r, p["model"], p["benchmark"]), 3, p["completion_tokens"])
    entries = sorted({k[1] for k in acc})
    benchmarks = sorted({k[2] for k in acc})
    e_ix = {e: i for i, e in enumerate(entries)}
    b_ix = {b: i for i, b in enumerate(benchmarks)}
    rows = [json.dumps([r, e_ix[e], b_ix[b], *v], separators=(",", ":"))
            for (r, e, b), v in sorted(acc.items(), key=lambda kv: (kv[0][0], e_ix[kv[0][1]],
                                                                     b_ix[kv[0][2]]))]
    head = {"format": PICKS_FORMAT, "source": REVEAL_SOURCE, "reveals": reveals,
            "entries": entries, "benchmarks": benchmarks, "columns": list(PICKS_COLUMNS),
            "clamped_cells": clamped}
    return _lines(head, "rows", rows)


def read_picks(path: str) -> tuple[list[str], list[tuple[int, str, str, tuple[int, ...]]]]:
    """(the reveals' epoch ids in order, [(reveal index, entry, benchmark,
    the PICKS_COLUMNS values)]) of a picks file."""
    doc = json.load(open(path, encoding="utf-8"))
    if doc.get("format") != PICKS_FORMAT or doc.get("columns") != list(PICKS_COLUMNS):
        raise SystemExit(f"{path} is not a {PICKS_FORMAT} file")
    out, seen = [], set()
    for row in doc["rows"]:
        if (len(row) != 3 + len(PICKS_COLUMNS)
                or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in row)):
            raise SystemExit(f"{path}: a row is not {3 + len(PICKS_COLUMNS)} counts")
        key = (row[0], doc["entries"][row[1]], doc["benchmarks"][row[2]])
        if key in seen or row[0] >= len(doc["reveals"]):
            raise SystemExit(f"{path}: row {key} is repeated or names no reveal")
        seen.add(key)
        out.append((*key, tuple(row[3:])))
    return [r["epoch_id"] for r in doc["reveals"]], out


def graded_sums(path: str) -> dict[str, dict[str, tuple[int, int]]]:
    """{entry: {v4 category: (graded cells, their completion tokens)}} over
    every reveal of a picks file: the cells the seed's pick-cost tables are
    computed from."""
    from fugal_subnet import epoch_plan

    category = dict(epoch_plan.mix_v4().bench_category)
    _epochs, rows = read_picks(path)
    out: dict[str, dict[str, list[int]]] = {}
    for _r, entry, benchmark, v in rows:
        if v[0] == 0:
            continue
        if benchmark not in category:
            raise SystemExit(f"{path}: benchmark {benchmark} has no v4 category")
        a = out.setdefault(entry, {}).setdefault(category[benchmark], [0, 0])
        a[0] += v[0]
        a[1] += v[1]
    return {e: {c: (a[0], a[1]) for c, a in sorted(by.items())} for e, by in sorted(out.items())}


def accuracy(path: str, *, first: str | None = None, last: str | None = None) -> dict:
    """How well each cost table predicts what heads' picks really cost, one
    epoch left out at a time: for every reveal in [first, last] of the picks
    file, each table is built from the other reveals in that range and
    priced on the held-out reveal's picked cells. The result is real over
    predicted spend, summed over the held-out reveals, for the champion's
    picks and for every served head's: 1.00 is exact, 2.00 a table that
    under-states the picks' cost twice over.

    Completion tokens at v3 menu B's completion price (a v3 reveal publishes
    no prompt tokens for a cell). The tables: `pinned` (v3 menu B's token
    manifest), `call_entry` and `call_category` (the pooled exploration
    pairs' mean per entry, and per entry and category: the call cost),
    `cells_entry` and `cells_category` (the graded cells' plain means), and
    `pick_entry` and `pick_category` (the W13 table: the cells shrunk toward
    the call mean per entry by PICK_COST_PRIOR_CELLS_ENTRY, and per category
    toward that by PICK_COST_PRIOR_CELLS_CATEGORY). A table without data for
    an entry falls back to the call mean, and that to the pinned mean.
    Exact Fractions; the ratios are returned as Fractions."""
    from fugal_subnet import epoch_plan

    category = dict(epoch_plan.mix_v4().bench_category)
    price = {r["id"]: Fraction(repr(r["out"])) for r in json.load(open(MENU_PATH, encoding="utf-8"))}
    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))["models"]
    pinned = {r["id"]: Fraction(repr(r["mean_out_tokens"])) for r in manifest}
    epochs, rows = read_picks(path)
    lo = 0 if first is None else epochs.index(first)
    hi = len(epochs) - 1 if last is None else epochs.index(last)
    kept = range(lo, hi + 1)
    data: dict[tuple[int, str, str], list[int]] = {}
    for r, entry, benchmark, v in rows:
        if r in kept:
            a = data.setdefault((r, entry, category[benchmark]), [0] * len(PICKS_COLUMNS))
            for i, x in enumerate(v):
                a[i] += x
    ke, kc = menu_state.PICK_COST_PRIOR_CELLS_ENTRY, menu_state.PICK_COST_PRIOR_CELLS_CATEGORY
    tables = ("pinned", "call_entry", "call_category", "cells_entry", "cells_category",
              "pick_entry", "pick_category")
    sums = {who: {t: [Fraction(0), Fraction(0)] for t in tables} for who in ("champion", "heads")}
    for held in kept:
        by_entry: dict[tuple[str, int], list[int]] = {}
        by_cat: dict[tuple[str, str, int], list[int]] = {}
        for (r, entry, cat), v in data.items():
            if r == held:
                continue
            for col in (0, 3):                        # graded cells, pooled pairs
                a = by_entry.setdefault((entry, col), [0, 0])
                b = by_cat.setdefault((entry, cat, col), [0, 0])
                for x in (a, b):
                    x[0] += v[2 * col]
                    x[1] += v[2 * col + 1]

        def mean(acc, fallback):
            return Fraction(acc[1], acc[0]) if acc and acc[0] else fallback

        for (r, entry, cat), v in data.items():
            if r != held:
                continue
            p0 = pinned[entry]
            call_e = mean(by_entry.get((entry, 3)), p0)
            call_c = mean(by_cat.get((entry, cat, 3)), call_e)
            cells_e = mean(by_entry.get((entry, 0)), call_e)
            cells_c = mean(by_cat.get((entry, cat, 0)), cells_e)
            g = by_entry.get((entry, 0), [0, 0])
            pick_e = (g[1] + ke * call_e) / (g[0] + ke)
            c = by_cat.get((entry, cat, 0), [0, 0])
            pick_c = (c[1] + kc * pick_e) / (c[0] + kc)
            predicted = dict(zip(tables, (p0, call_e, call_c, cells_e, cells_c, pick_e, pick_c)))
            for who, col in (("champion", 1), ("heads", 2)):
                n, real = v[2 * col], v[2 * col + 1]
                for t in tables:
                    sums[who][t][0] += real * price[entry]
                    sums[who][t][1] += n * predicted[t] * price[entry]
    picked = {who: sum(v[2 * col] for v in data.values())
              for who, col in (("champion", 1), ("heads", 2))}
    return {"epochs": (epochs[lo], epochs[hi], len(kept)), "picks": picked,
            "ratio": {who: {t: a / b for t, (a, b) in s.items()} for who, s in sums.items()}}


# --- build -------------------------------------------------------------------------


def _picodollars(price: str, what: str) -> int:
    """A listed price per token (a decimal string) in picodollars, exactly."""
    from fugal_subnet.menu_state import PICO

    value = Fraction(price) * PICO
    if value.denominator != 1 or value < 0:
        raise SystemExit(f"{what}: listed price {price!r} is not a whole number of picodollars")
    return int(value)


def _milli(value, what: str) -> int:
    """A token mean of the pinned manifest (a JSON number with decimals) in
    milli-tokens, exactly, from its shortest decimal form."""
    from fugal_subnet.menu_state import MILLI

    out = Fraction(repr(value)) * MILLI
    if out.denominator != 1 or out < 0:
        raise SystemExit(f"{what}: manifest mean {value!r} is not a whole number of milli-tokens")
    return int(out)


def scored_mix() -> tuple[tuple[str, int], ...]:
    """The scored category mix the token means are weighted to: the pinned v4
    mix (epoch_plan.mix_v4().menu_mix(), integer per-cent weights), the one
    the derivation step's production parameters weight to
    (epoch_plan.menu_step_params_v4)."""
    from fugal_subnet import epoch_plan

    return epoch_plan.mix_v4().menu_mix()


def lab_list():
    """The lab and host list the seed's entries are checked under: the signed
    list the release ships, or, until the owner has signed the first one,
    the draft under review (data/menu_allowlist.draft.json)."""
    from fugal_subnet import menu_rules

    try:
        return menu_rules.shipped_list(menu_rules.KIND_MENU_ALLOWLIST)
    except FileNotFoundError:
        with open(DRAFT_LIST_PATH, "rb") as f:
            return menu_rules.parse_unsigned_list(f.read(), menu_rules.KIND_MENU_ALLOWLIST)


def host_limits(hosts: dict[str, dict], what: str) -> dict[str, int]:
    """{host tag: the answer limit the listing states, 0 when it states
    none} for the hosts the listing shows for one model."""
    out = {}
    for tag, endpoint in hosts.items():
        limit = endpoint.get("max_completion_tokens")
        if limit is None:
            limit = 0
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise SystemExit(f"{what}: host {tag} lists an answer limit that is not a count")
        out[tag] = limit
    return out


def failing(calls: list[list[int]]) -> bool:
    """The seed's health filter on [calls, failed] an operator."""
    rates = sorted(Fraction(failed, n) for n, failed in calls if n >= HEALTH_MIN_CALLS)
    if len(rates) < HEALTH_MIN_OPERATORS:
        return False
    mid = len(rates) // 2
    median = rates[mid] if len(rates) % 2 else (rates[mid - 1] + rates[mid]) / 2
    return median > HEALTH_FAIL_RATE


def build(listing_path: str, pairs_path: str, allowlist=None,
          picks_path: str = PICKS_PATH) -> tuple[str, list[str]]:
    """(the seed file's text, one line of notes an entry). `allowlist` is the
    lab and host list the entries must pass under (default: `lab_list()`)."""
    from fugal_subnet import config, menu_rules, menu_state

    allowlist = lab_list() if allowlist is None else allowlist
    listing = read_listing(listing_path)
    measured = read_pairs(pairs_path)
    graded = graded_sums(picks_path)
    # The step's constants, the pinned v4 mix: the seed's tables are what a
    # refresh would give on these cells.
    params = menu_state.MenuParams(mix=scored_mix(), min_category_accuracy_bp=0)
    menu = json.load(open(MENU_PATH, encoding="utf-8"))
    manifest = {r["id"]: r for r in json.load(open(MANIFEST_PATH, encoding="utf-8"))["models"]}
    mix = scored_mix()
    entries, notes = [], []
    for row in menu:
        eid, order = row["id"], row["provider"]["order"]
        hosts = [listing[row["model"]][h] for h in order if h in listing.get(row["model"], {})]
        got = measured.get(eid, {"cells": {}, "calls": []})
        if not hosts:
            notes.append(f"LEFT OUT {eid}: none of its pinned hosts {order} is listed")
            continue
        if failing(got["calls"]):
            notes.append(f"LEFT OUT {eid}: failing on the named epochs' calls")
            continue
        quantizations = sorted({str(h["quantization"]) for h in hosts})
        if len(quantizations) != 1:
            raise SystemExit(f"{eid}: its listed pinned hosts differ in quantization {quantizations}")
        price_in = max(_picodollars(h["pricing"]["prompt"], eid) for h in hosts)
        price_out = max(_picodollars(h["pricing"]["completion"], eid) for h in hosts)
        by_category: dict[str, list[int]] = {}
        for benchmark, (n, t_in, t_out) in got["cells"].items():
            c = by_category.setdefault(config.BENCHMARK_CATEGORY_V3[benchmark], [0, 0, 0])
            c[0] += n
            c[1] += t_in
            c[2] += t_out
        mean_in, mean_out, n_pairs = menu_state.weighted_token_means(
            (menu_state.TokenCell(cat, *by_category[cat]) for cat in sorted(by_category)), mix)
        # Its pooled exploration pairs in the named reveals (v3 pools nothing
        # else), whatever the token means are taken from.
        explore_pairs = sum(n for n, _, _ in got["cells"].values())
        basis = f"{n_pairs} pooled pairs"
        if n_pairs < TOKEN_MIN_PAIRS:
            man = manifest[eid]
            mean_in, mean_out = _milli(man["mean_in_tokens"], eid), _milli(man["mean_out_tokens"], eid)
            basis = f"the v3 manifest row ({man['samples']} calls; {n_pairs} pooled pairs)"
            n_pairs = int(man["samples"])
        reasoning = row.get("reasoning")
        # The answer limit by the rule the step applies to an entry on the
        # menu: over the pinned hosts the listing shows.
        max_tokens = menu_rules.answer_limit(
            order, host_limits(listing[row["model"]], eid), on_menu=True)
        entry = menu_state.Entry(
            id=eid, rev=1, model=row["model"],
            reasoning=None if reasoning is None else next(iter(reasoning.items())),
            max_tokens=max_tokens, provider_order=tuple(order),
            quantization=quantizations[0], price_in=price_in, price_out=price_out,
            mean_in=mean_in, mean_out=mean_out, token_pairs=n_pairs, joined_epoch=0,
            scored_from=0, measured_rate=menu_state.measured_rate(0, 0), proposer="",
            explore_pairs=explore_pairs)
        # Its graded cells: the prompt side at its call mean, exactly (a v3
        # reveal publishes no prompt tokens for a cell).
        cells = {cat: (n, n * mean_in, menu_state.MILLI * out)
                 for cat, (n, out) in graded.get(eid, {}).items()}
        entry = dataclasses.replace(
            entry, pick=menu_state.pick_table_from_sums(cells, entry, params))
        entries.append(entry)
        unlisted = [h for h in order if h not in listing[row["model"]]]
        # For the reviewer: dollars per million tokens, beside v3's pinned price.
        now = (Fraction(price_in, 10**6), Fraction(price_out, 10**6))
        was = (Fraction(repr(row["in"])), Fraction(repr(row["out"])))
        notes.append(
            f"{eid}: ${float(now[0]):.6g}/${float(now[1]):.6g} per M"
            + (f" (v3 ${float(was[0]):.6g}/${float(was[1]):.6g})" if was != now else "")
            + f"; means {mean_in}/{mean_out} milli-tokens from {basis}"
            + (f"; pick {entry.pick.mean_in}/{entry.pick.mean_out} on {entry.pick.cells} graded "
               "cells" if entry.pick is not None else "; no graded cell: picked at its call cost")
            + (f"; answer limit {max_tokens} (v3 {row['max_tokens']})"
               if max_tokens != row["max_tokens"] else "")
            + (f"; pinned host(s) {unlisted} not listed" if unlisted else ""))
    state = menu_state.MenuState(
        epoch=0, serial=0, entries=tuple(sorted(entries, key=lambda e: e.id)),
        retired=(), labs=menu_state.AllowlistRef(0, ""),
        devices=menu_state.AllowlistRef(0, ""))
    menu_state.validate(state)
    # The compiled caps on every entry (a positive call and pick cost, each
    # pick-cost table within what its cells could give); the menu's size is
    # the safety check's (check_menu_v4_seed_pinned).
    why = menu_state.check_caps(state, dataclasses.replace(menu_state.READER_PARAMS,
                                                           min_entries=1))
    if why is not None:
        raise SystemExit(f"the seed is not within the compiled caps: {why}")
    doc = menu_state.to_dict(state)
    for entry in doc["entries"]:
        refusal = menu_rules.check_entry(
            entry, allowlist, host_limits=host_limits(listing[entry["model"]], entry["id"]))
        if refusal is not None:
            raise SystemExit(f"{entry['id']} does not pass the compiled rules: {refusal}")
    rows = [json.dumps(e, sort_keys=True, separators=(",", ":")) for e in doc.pop("entries")]
    text = _lines(doc, "entries", rows)
    if menu_state.from_dict(json.loads(text)) != state:
        raise SystemExit("the seed file does not read back as the state it was written from")
    return text, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("snapshot", help="saved listing pages -> the listing file")
    p.add_argument("pages_dir")
    p.add_argument("--fetched-utc", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("pairs", help="published v3 reveals -> the pairs file")
    p.add_argument("reveals_dir")
    p.add_argument("--first", required=True)
    p.add_argument("--last", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("picks", help="published v3 reveals -> the picks file")
    p.add_argument("reveals_dir")
    p.add_argument("--first", required=True)
    p.add_argument("--last", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("accuracy", help="the picks file -> each cost table's accuracy on picks")
    p.add_argument("--picks", default=PICKS_PATH)
    p.add_argument("--first", default=None)
    p.add_argument("--last", default=None)
    p = sub.add_parser("build", help="the listing, pairs and picks files -> the seed")
    p.add_argument("--listing", default=LISTING_PATH)
    p.add_argument("--pairs", default=PAIRS_PATH)
    p.add_argument("--picks", default=PICKS_PATH)
    p.add_argument("--out", default=None,
                   help="write (or check) this one file instead of the data/ copy and the "
                        "package copy")
    p.add_argument("--check", action="store_true",
                   help="compare with the file(s) instead of writing")
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        text = snapshot(args.pages_dir, args.fetched_utc)
    elif args.command == "pairs":
        text = pairs(args.reveals_dir, args.first, args.last)
    elif args.command == "picks":
        text = picks(args.reveals_dir, args.first, args.last)
    elif args.command == "accuracy":
        got = accuracy(args.picks, first=args.first, last=args.last)
        first, last, n = got["epochs"]
        print(f"real over predicted spend of the picked cells, one epoch left out, "
              f"{first}..{last} ({n} reveals; completion tokens at v3 menu B's prices)")
        for who, ratios in got["ratio"].items():
            print(f"  {who} ({got['picks'][who]} picks): " + ", ".join(
                f"{t} {float(r):.3f}" for t, r in ratios.items()))
        return 0
    else:
        text, notes = build(args.listing, args.pairs, picks_path=args.picks)
        for line in notes:
            print(line)
        for name, path in (("listing", args.listing), ("pairs", args.pairs),
                           ("picks", args.picks)):
            print(f"{name} sha256 {_sha256(open(path, 'rb').read())}  {os.path.relpath(path, ROOT)}")
    data = text.encode("utf-8")
    if args.command == "build" and args.out is None:
        outs = [SEED_PATH, PACKAGE_SEED_PATH]
    else:
        outs = [args.out]
    for out in outs:
        shown = os.path.relpath(out, ROOT) if out.startswith(ROOT) else out
        if getattr(args, "check", False):
            if not os.path.exists(out) or open(out, "rb").read() != data:
                raise SystemExit(f"{shown} is not what these inputs build")
            print(f"{shown} rebuilds byte for byte (sha256 {_sha256(data)})")
            continue
        with open(out, "wb") as f:
            f.write(data)
        print(f"wrote {shown} ({len(data)} bytes, sha256 {_sha256(data)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
