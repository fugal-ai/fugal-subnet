"""Protocol v2, one epoch: from verified proofs to the seat and the weight vector.

The pure modules (shared_grades, champion, payout, evidence's E'' rule) know
nothing about proofs, metagraphs or files. This module is the seam: it turns
what the validator has at the end of verification into their inputs, and
turns their outputs into a weight vector, reveal fields and persisted state.
Everything here is deterministic given the epoch's inputs (I1) and every
input is in the published reveal, so the window and the seat are rebuilt from
reveals exactly.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from fugal_subnet.champion import (
    Candidate,
    EpochPairs,
    EpochSlice,
    Ghost,
    HeadEpoch,
    OddsParams,
    Pending,
    SeatDecision,
    SeatParams,
    SeatState,
    Window,
    seat_step,
)
from fugal_subnet.pricing import policy_cost

HARD_CATEGORY = "hard"


def category_of(question: Mapping, bench_category: Mapping[str, str] | None) -> str:
    """A question's category: the benchmark's category when one is configured,
    else the benchmark itself (so the bootstrap still stratifies by benchmark)."""
    bench = str(question.get("benchmark", ""))
    if bench_category:
        return str(bench_category.get(bench, bench))
    return bench


def build_epoch_slice(
    epoch: int,
    questions: list[Mapping],
    pool_models: list[str],
    cells: Mapping[tuple[str, str], float],
    cell_tokens: Mapping[tuple[str, str], float],
    prices: Mapping[str, tuple[float, float]],
    bench_category: Mapping[str, str] | None = None,
    pick_spend: Mapping[str, tuple[float, Mapping[str, float]]] | None = None,
) -> EpochSlice:
    """The epoch's slice in the global model index space.

    A cell's spend is the pool's prompt tokens for the question and the cell's
    coldkey-median attested completion tokens, at the pinned prices: measured,
    shared, and never read from the frame. Cells nobody answered are NaN in
    both arrays; a head's own non-failed answer always has its cell.

    `pick_spend` (protocol v4: {entry: (its pick cost, {category: its pick
    cost there})}, dollars a call, from the epoch's menu) prices a cell
    instead at its entry's pick cost in the question's category, or the
    entry's own when the category is not listed (SEAT_MINIMAL_V4.md 3.4):
    the table every head is priced at, whatever one call happened to cost.
    """
    q = len(questions)
    m = len(pool_models)
    grades = np.full((q, m), np.nan, dtype=np.float64)
    spends = np.full((q, m), np.nan, dtype=np.float64)
    cats = tuple(category_of(x, bench_category) for x in questions)
    for i, question in enumerate(questions):
        qid = question["question_id"]
        prompt = question.get("prompt", "")
        for j, model in enumerate(pool_models):
            g = cells.get((qid, model))
            if g is None or model not in prices:
                continue
            grades[i, j] = float(g)
            if pick_spend is not None:
                entry_value, by_category = pick_spend[model]
                spends[i, j] = float(by_category.get(cats[i], entry_value))
                continue
            spends[i, j] = policy_cost(prompt, model, prices, float(cell_tokens.get((qid, model), 0.0)))
    return EpochSlice(
        epoch=int(epoch),
        question_ids=tuple(x["question_id"] for x in questions),
        categories=cats,
        hard=tuple(c == HARD_CATEGORY for c in cats),
        grades=grades,
        spends=spends,
    )


def head_epoch(proof, question_ids: tuple[str, ...], model_index: Mapping[str, int]) -> HeadEpoch | None:
    """One verified proof's scored answers aligned to the slice, or None if they do not cover it.

    A verified proof answers exactly the assigned slice (verify_proof checks
    the question set), so a gap here is a bug upstream; it is refused rather
    than guessed at.
    """
    by_q = {r.question_id: r for r in proof.scored_results}
    routed = np.empty(len(question_ids), dtype=np.int64)
    failed = np.zeros(len(question_ids), dtype=bool)
    for i, qid in enumerate(question_ids):
        r = by_q.get(qid)
        if r is None or r.routed_model not in model_index:
            return None
        routed[i] = model_index[r.routed_model]
        failed[i] = bool(r.failed)
    return HeadEpoch(routed=routed, failed=failed)


# --- persistence -------------------------------------------------------------
#
# The window is arrays, so it is kept beside the JSON state file as an .npz
# (loaded with allow_pickle=False: the file is local, but the rule has no
# exceptions). The seat is small and goes into the JSON state.


def _labels_v4():
    """Protocol v4's labels module, imported only where a v4 window is."""
    from fugal_subnet import labels_v4

    return labels_v4


def save_window(path: str, window: Window) -> None:
    """Write the window beside the state file: a protocol v4 window (one
    with v4 labels or real-picks records) in the v4 layout (`_v4_arrays`), any
    other exactly as before (`window_arrays`)."""
    v4 = bool(window.picks) or any(isinstance(p, _labels_v4().LabelPairs)
                                   for p in window.pairs.values())
    arrays = _v4_arrays(window) if v4 else window_arrays(window)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        (np.savez if v4 else np.savez_compressed)(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def window_arrays(window: Window) -> dict[str, np.ndarray]:
    """The window as the v1-v3 file holds it: one array per slice, head,
    epoch and field, and a JSON `meta`. Also the logical form of any window
    (check_determinism digests it, whatever layout the file has)."""
    arrays: dict[str, np.ndarray] = {}
    meta = {"size": window.size, "slices": {}, "heads": {}}
    for e, sl in sorted(window.slices.items()):
        meta["slices"][str(e)] = {
            "question_ids": list(sl.question_ids),
            "categories": list(sl.categories),
            "hard": [bool(x) for x in sl.hard],
        }
        arrays[f"g_{e}"] = sl.grades
        arrays[f"s_{e}"] = sl.spends
    for i, (h, per) in enumerate(sorted(window.heads.items())):
        meta["heads"][h] = {"key": i, "epochs": sorted(per)}
        for e, he in sorted(per.items()):
            arrays[f"r_{i}_{e}"] = he.routed
            arrays[f"f_{i}_{e}"] = he.failed
        # v3: the head's odds on that epoch's pooled pairs.
        odds_epochs = sorted(window.odds.get(h, {}))
        if odds_epochs:
            meta["heads"][h]["odds_epochs"] = odds_epochs
            for e in odds_epochs:
                arrays[f"p_{i}_{e}"] = window.odds[h][e]
    if window.pairs:
        meta["pairs"] = {}
        for e, ep in sorted(window.pairs.items()):
            meta["pairs"][str(e)] = {
                "question_ids": list(ep.question_ids), "models": list(ep.models),
                "benchmarks": list(ep.benchmarks), "categories": list(ep.categories),
            }
            arrays[f"x_y_{e}"] = ep.y
            arrays[f"x_p_{e}"] = ep.prompt_tokens
            arrays[f"x_c_{e}"] = ep.completion_tokens
            if ep.weight is not None:
                arrays[f"x_w_{e}"] = ep.weight
            if ep.scored is not None:
                arrays[f"x_s_{e}"] = ep.scored
    if any(isinstance(p, _labels_v4().LabelPairs) for p in window.pairs.values()) or any(
            h not in window.heads for h in window.odds):
        # Protocol v4's labels and the odds of heads with no answers in the
        # window (labels_v4). A v1-v3 window has neither: its file is as before.
        v4_meta, v4_arrays = _labels_v4().window_to_arrays(window)
        meta.update(v4_meta)
        arrays.update(v4_arrays)
    if window.picks:
        # Protocol v4's real-picks records, only when there are any, so a
        # v1-v3 window's file is what it was.
        meta["picks"] = {}
        for e, rec in sorted(window.picks.items()):
            heads_e = sorted(rec.values)
            meta["picks"][str(e)] = {"categories": list(rec.categories), "outage": rec.outage,
                                     "heads": heads_e, "holder": rec.holder,
                                     "lengths": [len(rec.values[h]) for h in heads_e]}
            arrays[f"k_{e}"] = np.array([x for h in heads_e for x in rec.values[h]],
                                        dtype=np.int64)
    arrays["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    return arrays


def load_window(path: str, size: int) -> Window:
    if not os.path.exists(path):
        return Window.empty(size)
    if _is_v4_file(path):
        return _load_v4(path, size)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        slices = {
            int(e): EpochSlice(
                epoch=int(e),
                question_ids=tuple(v["question_ids"]),
                categories=tuple(v["categories"]),
                hard=tuple(bool(x) for x in v["hard"]),
                grades=np.array(z[f"g_{e}"], dtype=np.float64),
                spends=np.array(z[f"s_{e}"], dtype=np.float64),
            )
            for e, v in meta["slices"].items()
        }
        heads = {
            h: {
                int(e): HeadEpoch(
                    routed=np.array(z[f"r_{v['key']}_{e}"], dtype=np.int64),
                    failed=np.array(z[f"f_{v['key']}_{e}"], dtype=bool),
                )
                for e in v["epochs"]
            }
            for h, v in meta["heads"].items()
        }
        pairs = {
            int(e): EpochPairs(
                question_ids=tuple(v["question_ids"]), models=tuple(v["models"]),
                benchmarks=tuple(v["benchmarks"]), categories=tuple(v["categories"]),
                y=np.array(z[f"x_y_{e}"], dtype=np.int32),
                prompt_tokens=np.array(z[f"x_p_{e}"], dtype=np.int32),
                completion_tokens=np.array(z[f"x_c_{e}"], dtype=np.int32),
                weight=(np.array(z[f"x_w_{e}"], dtype=np.int64) if f"x_w_{e}" in z.files else None),
                scored=(np.array(z[f"x_s_{e}"], dtype=bool) if f"x_s_{e}" in z.files else None),
            )
            for e, v in meta.get("pairs", {}).items()
        }
        odds = {
            h: {int(e): np.array(z[f"p_{v['key']}_{e}"], dtype=np.int16)
                for e in v["odds_epochs"]}
            for h, v in meta["heads"].items() if v.get("odds_epochs")
        }
        picks = {}
        if meta.get("picks"):
            from fugal_subnet.picks_v4 import PicksEpoch
            for e, v in meta["picks"].items():
                flat = np.array(z[f"k_{e}"], dtype=np.int64).tolist()
                ends = np.cumsum([0] + [int(n) for n in v["lengths"]]).tolist()
                if ends[-1] != len(flat):
                    raise ValueError(f"the window's challenge record of epoch {e} is malformed")
                picks[int(e)] = PicksEpoch(
                    int(e), tuple(v["categories"]),
                    {h: tuple(flat[ends[i]:ends[i + 1]]) for i, h in enumerate(v["heads"])},
                    bool(v["outage"]), v["holder"])
        window = Window(size=int(size), slices=slices, heads=heads, pairs=pairs, odds=odds,
                        picks=picks)
        if "labels_v4" in meta or "odds_only" in meta:
            window = _labels_v4().window_from_arrays(window, meta, z)
    # A changed window size is a changed rule; shrink (or keep) to the
    # configured size from the newest epoch, never silently extend.
    return window


# --- the protocol v4 window file ---------------------------------------------------------
#
# A v4 window holds about 13,000 head-epochs of odds and 1.3 million labels;
# one array per head and epoch made a 77 MB file of ~25,000 members that took
# over a minute to write. The v4 layout is a fixed set of concatenated arrays
# (V4_WINDOW_ARRAYS), each with an index, every string once in a table, and
# a JSON `meta` naming the format (V4_WINDOW_FORMAT). It is written
# uncompressed (np.savez) and read with allow_pickle=False. Before any array
# is read, the loader checks the archive's members, their dtypes and that
# each array's declared size is what its member holds, within
# V4_WINDOW_MAX_BYTES in all, so a hostile or truncated file is refused with
# a ValueError before it can allocate; then every index, offset and string
# reference is checked against the arrays it addresses.

V4_WINDOW_FORMAT = "fugal-window-v4-2"
# No valid window comes near it (48 epochs x 288 head-arrays x 30,000 labels
# x 2 bytes of odds is 0.8 GiB); it bounds what a damaged file can make the
# loader allocate.
V4_WINDOW_MAX_BYTES = 4 << 30
# name -> dtype, every array one-dimensional
V4_WINDOW_ARRAYS = {
    "meta": "u1", "v4_format": "u1",
    "sl_index": "<i8", "sl_grades": "<f8", "sl_spends": "<f8",
    "an_index": "<i8", "an_routed": "<i8", "an_failed": "|b1",
    "pr_index": "<i8", "pr_strings": "<i4", "pr_y": "<i4", "pr_prompt": "<i4",
    "pr_completion": "<i4", "pr_weight": "<i8", "pr_scored": "|b1", "pr_missing": "<i2",
    "pr_nvoters": "<i4", "pr_voters": "<i4",
    "od_index": "<i8", "od_values": "<i2",
    "pk_index": "<i8", "pk_heads": "<i4", "pk_categories": "<i4", "pk_lengths": "<i8",
    "pk_values": "<i8",
}


def _is_v4_file(path: str) -> bool:
    """Whether the window file is in the v4 layout (it holds `v4_format`):
    read from the archive's directory only."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            return "v4_format.npy" in zf.namelist()
    except (zipfile.BadZipFile, OSError) as e:
        raise ValueError(f"window file {path} is not a readable archive: {e}") from e


def _v4_arrays(window: Window) -> dict[str, np.ndarray]:
    """The window in the v4 layout (module comment above). Values are cast to
    the dtypes load_window reads them as."""
    labels = _labels_v4()
    strings: dict[str, int] = {}

    def sid(s) -> int:
        return strings.setdefault(str(s), len(strings))

    def sids(seq) -> list[int]:
        return [strings.setdefault(x, len(strings)) for x in map(str, seq)]

    cat = {k: [] for k in V4_WINDOW_ARRAYS if k not in ("meta", "v4_format")}
    slices = {}
    for e, sl in sorted(window.slices.items()):
        slices[str(e)] = {"question_ids": list(sl.question_ids), "categories": list(sl.categories),
                          "hard": [bool(x) for x in sl.hard]}
        g = np.asarray(sl.grades, dtype=np.float64)
        cat["sl_index"].append([int(e), g.shape[0], g.shape[1]])
        cat["sl_grades"].append(g.ravel())
        cat["sl_spends"].append(np.asarray(sl.spends, dtype=np.float64).reshape(g.shape).ravel())
    heads = sorted(set(window.heads) | set(window.odds))
    head_id = {h: i for i, h in enumerate(heads)}
    for h in sorted(window.heads):
        for e, he in sorted(window.heads[h].items()):
            routed = np.asarray(he.routed, dtype=np.int64)
            cat["an_index"].append([head_id[h], int(e), len(routed)])
            cat["an_routed"].append(routed)
            cat["an_failed"].append(np.asarray(he.failed, dtype=bool).reshape(len(routed)))
    for e, p in sorted(window.pairs.items()):
        n = len(p.question_ids)
        v4 = isinstance(p, labels.LabelPairs)
        flags = (p.weight is not None) | (p.scored is not None) << 1 | v4 << 2
        cat["pr_index"].append([int(e), n, flags])
        cat["pr_strings"].append(np.column_stack([
            np.array(sids(f), dtype=np.int32).reshape(n)
            for f in (p.question_ids, p.models, p.benchmarks, p.categories)]).reshape(-1))
        cat["pr_y"].append(np.asarray(p.y).astype(np.int32))
        cat["pr_prompt"].append(np.asarray(p.prompt_tokens).astype(np.int32))
        cat["pr_completion"].append(np.asarray(p.completion_tokens).astype(np.int32))
        if p.weight is not None:
            cat["pr_weight"].append(np.asarray(p.weight).astype(np.int64))
        if p.scored is not None:
            cat["pr_scored"].append(np.asarray(p.scored, dtype=bool))
        if v4:
            cat["pr_missing"].append(np.asarray(p.missing).astype(np.int16))
            cat["pr_nvoters"].append(np.array([len(v) for v in p.voters], dtype=np.int32))
            cat["pr_voters"].append(np.array(sids(c for v in p.voters for c in v), dtype=np.int32))
    for h in heads:
        for e, o in sorted(window.odds.get(h, {}).items()):
            o = np.asarray(o, dtype=np.int16)
            cat["od_index"].append([head_id[h], int(e), len(o)])
            cat["od_values"].append(o)
    for e, rec in sorted(window.picks.items()):
        # a challenge record: each head's items on its own prefix of the
        # list (pk_lengths), and the holder the challenges paired with
        hs = sorted(rec.values)
        q = len(rec.categories)
        cat["pk_index"].append([int(e), len(hs), q, int(bool(rec.outage)),
                                -1 if rec.holder is None else sid(rec.holder)])
        cat["pk_heads"].append(np.array([sid(h) for h in hs], dtype=np.int32))
        cat["pk_categories"].append(np.array([sid(c) for c in rec.categories], dtype=np.int32))
        cat["pk_lengths"].append(np.array([len(rec.values[h]) for h in hs], dtype=np.int64))
        cat["pk_values"].append(np.array([x for h in hs for x in rec.values[h]], dtype=np.int64))
    meta = {"format": V4_WINDOW_FORMAT, "size": int(window.size), "heads": heads,
            "strings": list(strings), "slices": slices}
    out = {"meta": np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8),
           "v4_format": np.frombuffer(V4_WINDOW_FORMAT.encode("ascii"), dtype=np.uint8)}
    for k, parts in cat.items():
        dt = np.dtype(V4_WINDOW_ARRAYS[k])
        if k.endswith("_index"):
            width = {"sl_index": 3, "an_index": 3, "pr_index": 3, "od_index": 3, "pk_index": 5}[k]
            out[k] = np.array(parts, dtype=dt).reshape(-1) if parts else np.zeros(0, dt)
            assert out[k].size == width * len(parts)
        else:
            out[k] = (np.concatenate([np.asarray(a, dtype=dt).reshape(-1) for a in parts])
                      if parts else np.zeros(0, dt))
    return out


def _v4_checked_arrays(path: str) -> dict[str, np.ndarray]:
    """The v4 window file's arrays, after checking, from the archive alone,
    that it holds exactly V4_WINDOW_ARRAYS, each a stored one-dimensional
    array of its dtype whose declared size is what its member holds, within
    V4_WINDOW_MAX_BYTES in all. ValueError otherwise, before anything large
    is read."""
    import zipfile

    want = {f"{k}.npy": k for k in V4_WINDOW_ARRAYS}
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            names = [i.filename for i in infos]
            if sorted(names) != sorted(want):
                raise ValueError("its members are not the v4 window's arrays")
            if sum(i.file_size for i in infos) > V4_WINDOW_MAX_BYTES:
                raise ValueError(f"it declares more than {V4_WINDOW_MAX_BYTES} bytes")
            for info in infos:
                with zf.open(info) as f:
                    version = np.lib.format.read_magic(f)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                    elif version in ((2, 0), (3, 0)):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                    else:
                        raise ValueError(f"member {info.filename} has npy version {version}")
                    header = f.tell() if hasattr(f, "tell") else None
                key = want[info.filename]
                if (dtype != np.dtype(V4_WINDOW_ARRAYS[key]) or len(shape) != 1
                        or header is None or header + shape[0] * dtype.itemsize != info.file_size):
                    raise ValueError(f"member {info.filename} is not a 1-D "
                                     f"{V4_WINDOW_ARRAYS[key]} array of its stored size")
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in V4_WINDOW_ARRAYS}
    except ValueError as e:
        raise ValueError(f"window file {path}: {e}") from e
    except (zipfile.BadZipFile, OSError, EOFError, KeyError) as e:
        raise ValueError(f"window file {path} is damaged: {e}") from e


def _load_v4(path: str, size: int) -> Window:
    """A v4-layout window file (`_v4_arrays`), checked (`_v4_checked_arrays`,
    then every index against what it addresses) and rebuilt through the
    window's own types, which check the rest."""
    from fugal_subnet.head_odds import ODDS_SCALE
    from fugal_subnet.picks_v4 import PicksEpoch

    labels = _labels_v4()
    a = _v4_checked_arrays(path)

    def bad(why):
        return ValueError(f"window file {path}: {why}")

    try:
        meta = json.loads(bytes(a["meta"]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise bad("its meta is not JSON") from e
    if (not isinstance(meta, dict) or set(meta) != {"format", "size", "heads", "strings", "slices"}
            or meta["format"] != V4_WINDOW_FORMAT or bytes(a["v4_format"]) != V4_WINDOW_FORMAT.encode()
            or type(meta["size"]) is not int):
        raise bad(f"it is not a {V4_WINDOW_FORMAT} window")
    strings, heads, slices_meta = meta["strings"], meta["heads"], meta["slices"]
    if (not isinstance(strings, list) or not all(isinstance(s, str) for s in strings)
            or len(set(strings)) != len(strings)
            or not isinstance(heads, list) or not all(isinstance(h, str) for h in heads)
            or heads != sorted(set(heads)) or not isinstance(slices_meta, dict)):
        raise bad("its string table, heads or slices are malformed")

    def table(key, width):
        t = a[key]
        if t.size % width:
            raise bad(f"{key} is not rows of {width}")
        return t.reshape(-1, width)

    class Cursor:
        """Consecutive segments of one concatenated array, each checked to
        lie inside it; `done` checks that every value was claimed."""

        def __init__(self, key):
            self.key, self.at = key, 0

        def take(self, n):
            if n < 0 or self.at + n > len(a[self.key]):
                raise bad(f"{self.key} is shorter than its index says")
            out = a[self.key][self.at:self.at + n]
            self.at += n
            return out

        def done(self):
            if self.at != len(a[self.key]):
                raise bad(f"{self.key} holds values its index does not name")

    def epoch_of(v):
        e = int(v)
        if e < 0:
            raise bad("an epoch index is negative")
        return e

    def strs(ix):
        if len(ix) and (int(ix.min()) < 0 or int(ix.max()) >= len(strings)):
            raise bad("a string index is out of range")
        return [strings[i] for i in ix.tolist()]

    try:
        slices = {}
        grades, spends = Cursor("sl_grades"), Cursor("sl_spends")
        for e, q, m in table("sl_index", 3).tolist():
            e = epoch_of(e)
            v = slices_meta.get(str(e))
            if e in slices or not isinstance(v, dict) or set(v) != {"question_ids", "categories",
                                                                      "hard"} or q < 0 or m < 0:
                raise bad(f"slice {e} is malformed")
            slices[e] = EpochSlice(epoch=e, question_ids=tuple(str(x) for x in v["question_ids"]),
                                   categories=tuple(str(x) for x in v["categories"]),
                                   hard=tuple(bool(x) for x in v["hard"]),
                                   grades=grades.take(q * m).reshape(q, m).copy(),
                                   spends=spends.take(q * m).reshape(q, m).copy())
        grades.done()
        spends.done()
        if set(slices_meta) != {str(e) for e in slices}:
            raise bad("the slices' metadata and arrays name different epochs")

        def head_named(i):
            if not 0 <= i < len(heads):
                raise bad("a head index is out of range")
            return heads[i]

        answers: dict[str, dict] = {}
        routed, failed = Cursor("an_routed"), Cursor("an_failed")
        for i, e, n in table("an_index", 3).tolist():
            h, e = head_named(i), epoch_of(e)
            if e in answers.get(h, {}):
                raise bad("a head's answers appear twice for one epoch")
            answers.setdefault(h, {})[e] = HeadEpoch(routed=routed.take(n).copy(),
                                                     failed=failed.take(n).copy())
        routed.done()
        failed.done()

        pairs: dict[int, EpochPairs] = {}
        cur = {k: Cursor(k) for k in ("pr_strings", "pr_y", "pr_prompt", "pr_completion",
                                      "pr_weight", "pr_scored", "pr_missing", "pr_nvoters",
                                      "pr_voters")}
        for e, n, flags in table("pr_index", 3).tolist():
            e = epoch_of(e)
            if e in pairs or n < 0 or not 0 <= flags < 8:
                raise bad(f"pairs of epoch {e} are malformed")
            s = strs(cur["pr_strings"].take(4 * n)) if n else []
            base = EpochPairs(
                question_ids=tuple(s[0::4]), models=tuple(s[1::4]), benchmarks=tuple(s[2::4]),
                categories=tuple(s[3::4]), y=cur["pr_y"].take(n).copy(),
                prompt_tokens=cur["pr_prompt"].take(n).copy(),
                completion_tokens=cur["pr_completion"].take(n).copy(),
                weight=cur["pr_weight"].take(n).copy() if flags & 1 else None,
                scored=cur["pr_scored"].take(n).copy() if flags & 2 else None)
            if flags & 4:
                counts = cur["pr_nvoters"].take(n)
                if n and int(counts.min()) < 1:
                    raise bad(f"a label of epoch {e} has no voter")
                flat = strs(cur["pr_voters"].take(int(counts.sum(dtype=np.int64))))
                bounds = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).tolist()
                voters = [tuple(flat[bounds[k]:bounds[k + 1]]) for k in range(n)]
                base = labels.label_pairs(base, voters, cur["pr_missing"].take(n).copy())
            pairs[e] = base
        for c in cur.values():
            c.done()

        odds: dict[str, dict] = {}
        values = Cursor("od_values")
        for i, e, n in table("od_index", 3).tolist():
            h, e = head_named(i), epoch_of(e)
            o = values.take(n).copy()
            p = pairs.get(e)
            if p is None or n != len(p.question_ids) or e in odds.get(h, {}):
                raise bad(f"odds of {h[:12]} for epoch {e} do not align with its pairs")
            if n and (int(o.min()) < -1 or int(o.max()) > ODDS_SCALE):
                raise bad("odds outside -1..ODDS_SCALE")
            odds.setdefault(h, {})[e] = o
        values.done()

        picks = {}
        ph, pc, pn, pv = (Cursor("pk_heads"), Cursor("pk_categories"), Cursor("pk_lengths"),
                          Cursor("pk_values"))
        for e, nh, q, outage, holder in table("pk_index", 5).tolist():
            e = epoch_of(e)
            if e in picks or nh < 0 or q < 0 or outage not in (0, 1) or holder < -1:
                raise bad(f"the picks record of epoch {e} is malformed")
            hs = strs(ph.take(nh))
            if hs != sorted(set(hs)):
                raise bad(f"the picks record of epoch {e} names a head twice or out of order")
            cats = strs(pc.take(q))
            lengths = pn.take(nh).tolist()
            if any(not 0 < n <= q for n in lengths):
                raise bad(f"the picks record of epoch {e} has items past its list")
            rows = [pv.take(n).tolist() for n in lengths]
            picks[e] = PicksEpoch(e, tuple(cats), {h: tuple(r) for h, r in zip(hs, rows)},
                                  bool(outage), None if holder < 0 else strs(np.array([holder]))[0])
        for c in (ph, pc, pn, pv):
            c.done()
    except (TypeError, IndexError) as e:
        raise bad(f"it is malformed: {e}") from e
    return Window(size=int(size), slices=slices, heads=answers, pairs=pairs, odds=odds,
                  picks=picks)


def seat_to_dict(state: SeatState) -> dict:
    if not isinstance(state, SeatState):
        # Protocol v4's seat (seat_v4.SeatStateV4): its own published shape,
        # marked "protocol": 4; v1-v3 seats are written as before.
        from fugal_subnet import seat_v4

        return seat_v4.to_dict(state)
    out = {
        "champion": state.champion,
        "champion_since": state.champion_since,
        "history": list(state.history),
        "ghosts": [{"head_hash": g.head_hash, "last_epoch": g.last_epoch, "reason": g.reason}
                   for g in state.ghosts],
        "high_water": state.high_water,
        "high_water_hard": state.high_water_hard,
    }
    # The confirmation rule's state, only when there is any: before its
    # activation block there is none, and the published seat (which is in the
    # consensus view, v2_sync) stays byte-identical to the one before the rule.
    if state.pending:
        out["pending"] = [{"head_hash": p.head_hash, "since_epoch": p.since_epoch, "bar": p.bar}
                          for p in state.pending]
    if state.confirm_from is not None:
        out["confirm_from"] = state.confirm_from
    return out


# A published seat is someone else's file (I2). At most one pending record per
# registered hotkey can exist; this only bounds what a hand-made reveal can load.
MAX_PENDING = 4096


def seat_from_dict(raw: Mapping | None) -> SeatState:
    if not raw:
        return SeatState()
    if isinstance(raw, Mapping) and "protocol" in raw:
        # Protocol v4's seat (seat_v4.to_dict), validated there.
        from fugal_subnet import seat_v4

        return seat_v4.from_dict(raw)
    pending = raw.get("pending") or ()
    if not isinstance(pending, (list, tuple)) or len(pending) > MAX_PENDING:
        raise ValueError("seat state: `pending` must be a list of at most "
                         f"{MAX_PENDING} records")
    confirm_from = raw.get("confirm_from")
    if confirm_from is not None and (isinstance(confirm_from, bool)
                                     or not isinstance(confirm_from, int)):
        raise ValueError("seat state: `confirm_from` must be an epoch index")
    return SeatState(
        champion=raw.get("champion"),
        champion_since=raw.get("champion_since"),
        history=tuple(raw.get("history", ())),
        ghosts=tuple(Ghost(str(g["head_hash"]), int(g["last_epoch"]), str(g["reason"]))
                     for g in raw.get("ghosts", ())),
        high_water=float(raw.get("high_water", 0.0)),
        high_water_hard=float(raw.get("high_water_hard", 0.0)),
        pending=tuple(Pending(str(p["head_hash"]), int(p["since_epoch"]), str(p["bar"]))
                      for p in pending),
        confirm_from=confirm_from,
    )


def anchor_spend_per_question(model: str) -> float:
    """The anchor's expected spend per question: the reviewed v2 token
    manifest's mean input and output tokens for `model`, at the pinned v2
    prices. Pure function of pinned files, so every validator agrees."""
    import math

    from fugal_subnet.routing_protocol import manifest_path, price_table

    manifest = json.loads(manifest_path(2).read_text())
    rows = {r["id"]: r for r in manifest.get("models", [])}
    prices = price_table(2)
    if model not in rows or model not in prices:
        raise KeyError(f"anchor model {model!r} is not in the v2 token manifest and price table")
    r = rows[model]
    rate_in, rate_out = prices[model]
    spend = math.fsum([float(r["mean_in_tokens"]) * rate_in,
                       float(r["mean_out_tokens"]) * rate_out])
    if not spend > 0:
        raise ValueError(f"anchor model {model!r} has no positive expected spend")
    return spend


def seat_params_from_config(boundary_block: int | None = None,
                            confirm_block: int | None = None, *,
                            protocol_version: int = 2,
                            odds_seat_block: int | None = None) -> SeatParams:
    """The seat rules for the epoch whose boundary is `boundary_block`.

    The fresh-data confirmation rule and the shorter retired-ghost window are
    in force from the first epoch whose boundary is at or after
    `confirm_block` (config.confirm_activation_block). With either argument
    None they are not, and the result is the rule set from before them.

    Protocol v3 prices one extra correct answer at a pinned number of dollars
    (CHAMPION_VALUE_PER_CORRECT_USD) instead of a multiple of the anchor
    model's spend, and from `odds_seat_block` decides the seat on odds
    (OddsParams), with net value kept as a guard.
    """
    from fugal_subnet import config

    confirm = (confirm_block is not None and boundary_block is not None
               and int(boundary_block) >= int(confirm_block))
    v3 = int(protocol_version) >= 3
    on_odds = (v3 and odds_seat_block is not None and boundary_block is not None
               and int(boundary_block) >= int(odds_seat_block))
    if v3:
        # V = accuracy_value x anchor_spend (champion.pair_units); pinning V in
        # dollars is accuracy_value 1 at an "anchor spend" of V itself.
        accuracy_value, anchor = 1.0, float(config.CHAMPION_VALUE_PER_CORRECT_USD)
        if not anchor > 0:
            raise ValueError("CHAMPION_VALUE_PER_CORRECT_USD must be positive")
    else:
        accuracy_value = config.CHAMPION_ACCURACY_VALUE
        anchor = anchor_spend_per_question(config.CHAMPION_ANCHOR_MODEL)
    odds = OddsParams(
        margin_points=config.BRIER_MARGIN, min_pairs=config.ODDS_MIN_PAIRS,
        nv_noninferiority=config.NV_NONINFERIORITY, ref_prior=config.ODDS_REF_PRIOR,
        confirm_pairs=config.ODDS_CONFIRM_PAIRS,
    ) if on_odds else None
    return SeatParams(
        margin_points=config.CHAMPION_MARGIN,
        confidence=config.CHAMPION_CONFIDENCE,
        resamples=config.CHAMPION_BOOTSTRAP_RESAMPLES,
        accuracy_value=accuracy_value,
        hard_weight=config.CHAMPION_HARD_WEIGHT,
        hard_gate_min_questions=config.CHAMPION_HARD_GATE_MIN_QUESTIONS,
        ghost_off=config.CHAMPION_GHOST_OFF,
        ghost_retire=(config.CHAMPION_GHOST_RETIRE_CONFIRM if confirm
                      else config.CHAMPION_GHOST_RETIRE),
        wilson_confidence=config.WILSON_CONFIDENCE,
        min_accuracy=config.CHAMPION_MIN_ACCURACY,
        min_paired_questions=config.CHAMPION_MIN_PAIRED_QUESTIONS,
        anchor_spend=anchor,
        confirm_questions=config.CHAMPION_CONFIRM_QUESTIONS if confirm else 0,
        confirm_open_epochs=config.CHAMPION_CONFIRM_OPEN_EPOCHS if confirm else 0,
        odds=odds,
    )


def v2_state_for_reveal(
    *,
    seat: SeatState,
    owners: Mapping[str, str],
    serving: Mapping[str, int],
    burn_in: Mapping[str, float] | None,
    extra: Mapping[str, object] | None = None,
) -> dict:
    """The post-epoch snapshot of everything the next v2 epoch reads.

    Published in every v2 reveal as `v2_state`, so a validator that lost its
    state, or has just joined, can rebuild it from the newest reveal, with the
    window replayed from the last CHAMPION_WINDOW_EPOCHS reveals (the rebuild
    from reveals); until it does, a validator whose window was lost fails
    closed (no first or take decisions). `extra` is where later v2 state (the copy rule's first-commit
    seniority, ownership and ghosts) is published: one block, one channel, so
    the rebuild reads one place. Keys are sorted for byte-identical reveals.
    `burn_in` None (protocol v4, which has no burn-in) leaves the field out.
    """
    block = {
        "seat": seat_to_dict(seat),
        "owners": {h: owners[h] for h in sorted(owners)},
        "serving": {h: int(serving[h]) for h in sorted(serving)},
    }
    if burn_in is not None:
        block["burn_in"] = {h: float(burn_in[h]) for h in sorted(burn_in)}
    for key in sorted(extra or {}):
        if key in block:
            raise ValueError(f"v2_state key {key!r} is reserved")
        block[key] = extra[key]
    return block


def decision_for_reveal(decision: SeatDecision) -> dict:
    return {
        "event": decision.event,
        "champion": decision.champion,
        "bar": decision.bar,
        "seat": seat_to_dict(decision.state),
        "detail": [[h, why] for h, why in decision.detail],
    }


@dataclass(frozen=True)
class OddsInput:
    """What a protocol v3 epoch needs beyond a v2 one: the exploration answers
    of every verified proof, and what turns a head into odds on them."""

    nonce: bytes
    # Every verified proof's exploration results (explore_pool.ExploreAnswer).
    answers: list
    # {question_id: benchmark} for every explored question.
    benchmark_of: Mapping[str, str]
    # {uid: (W, b, model ids)} of each verified head, as loaded and bounded.
    heads: Mapping[int, tuple]
    # question_id -> its embedding row from the pinned embeddings file.
    hidden_of: object
    # Coldkeys the validator's exploration detectors flagged this epoch.
    flagged: frozenset[str] = frozenset()
    coldkey_cap: int = 36
    implausible_z: int = 4
    implausible_min_samples: int = 8
    implausible_min_trials: int = 50
    ref_prior: int = 20
    single_model_min_pairs: int = 300
    category_targets: Mapping[str, float] | None = None
    # {entry: inverse-probability weight} of the epoch's cost-aware exploration
    # (explore_weights.weights of epoch_plan.explore_shares); None weighs one.
    explore_weights: Mapping[str, int] | None = None
    # The protocol's menu (v3 only). A head whose rows do not cover every
    # entry makes no prediction at all (every pair -1), so it is not eligible
    # for the odds seat (champion._no_full_menu). None checks only the pairs
    # drawn. It is the epoch's menu without menu B's added entries during
    # their grace (routing_protocol.scored_menu). A head's rows are matched
    # by entry id, so a row for an entry not on the menu is never read.
    menu: tuple[str, ...] | None = None
    # Entries whose pairs are pooled but not scored this epoch, for every head
    # alike (v3 only; routing_protocol.grace_excluded): menu B's added entries
    # during its grace period. Empty when none is.
    unscored: frozenset[str] = frozenset()
    # Protocol v4 only (None before it). The private set's weight table
    # (SharesV4.private_weights; `explore_weights` is then the shared set's),
    # and the epoch's menu_state.MenuState: every verified head's odds are
    # head_odds.pair_odds_v4 on it, with a missing row scored at what the
    # entry's allowance gives (labels_v4). v4 has no grace and no full-menu
    # rule, so `menu` and `unscored` are not given with it.
    private_weights: Mapping[str, int] | None = None
    menu_state: object | None = None


@dataclass(frozen=True)
class PicksInput:
    """What a protocol v4 epoch needs for the real-picks check (picks_v4;
    FAST_CROWN_V4.md 2): the epoch's MenuState, whose pick-cost table prices
    each challenge call and on which every pick is recomputed, the epoch's
    outage flag and the hotkeys the failure bound did not count (published
    with the rest), the epoch's challenge list as (question id, category) in
    list order (epoch_plan.challenge_list_v4), and heads by hash that the
    epoch's checks read beyond the verified heads, the holders' when they did
    not serve: bytes (checked by their hash; what a validator passes and a
    replay record keeps) or, from a node that holds it loaded, the head as
    (W, b, entry ids)."""

    state: object
    outage: bool = False
    refused: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    challenge: tuple[tuple[str, str], ...] = ()
    heads: Mapping[str, object] = field(default_factory=dict)


class HolderHeadMissing(ValueError):
    """The head of the holder an epoch's challenges pair with is not held:
    its picks cannot be recomputed, so the epoch cannot be scored here."""


def _artifact(head: tuple):
    """(W, b, entry ids) as head_odds.picks_and_odds_v4 reads a head."""
    from types import SimpleNamespace

    W, b, models = head
    return SimpleNamespace(W=np.asarray(W), b=np.asarray(b), models=list(models))


def _challenge_epoch(pi: PicksInput, epoch: int, sl: EpochSlice, serving, verified_proofs, seat,
                     window: Window, odds_input, pairs=None, verified_by_hash=None):
    """The epoch's challenge record (picks_v4.PicksEpoch) and every serving
    head's rule-B evidence against the holder going into the epoch (none: the
    open-seat count), as the reveal publishes them (FAST_CROWN_V4.md 2, 6).

    A proof's challenge counts when it pairs with the holder the epoch names
    (seat_v4.challenge_holder) and every pick it states is the one the heads
    give: the challenger's own on each question of its prefix, recomputed
    from its verified head, and the holder's wherever the two differ,
    recomputed from the holder's bytes (head_odds.picks_and_odds_v4 on this
    epoch's menu over its full entries: picks_v4.challenge_picks). Anything
    else is refused and not counted, so the run restarts. Its items are
    picks_v4.challenge_items (a failed call fails closed, a holder-side call
    not started is no item); a challenge with more than the cap of its
    holder-side calls failed or not started is void (picks_v4.
    challenge_verdict): published with its reason, its items all NO_ITEM,
    and its calls in no mean and no cell. Raises HolderHeadMissing when a
    holder is named and its head is not held: no node scores the epoch
    without it."""
    from fugal_subnet import picks_v4, seat_v4

    state = pi.state
    qids = [q for q, _c in pi.challenge]
    cats = [c for _q, c in pi.challenge]
    holder = seat_v4.challenge_holder(seat, epoch) if isinstance(seat, seat_v4.SeatStateV4) else None
    heads = {h: odds_input.heads[uid] for h, uid in sorted(serving.items())
             if uid in odds_input.heads}
    # The holder's bytes may come from any verified proof that carries them
    # (bytes are checked by their hash); whose challenges count is `serving`.
    any_head = {h: odds_input.heads[uid] for h, uid in sorted((verified_by_hash or {}).items())
                if uid in odds_input.heads}
    hidden_rows: dict[str, np.ndarray] = {}
    stacked: dict[tuple, np.ndarray] = {}

    def hidden(ids):
        # every challenger of one size reads the same prefix: stacked once
        key = tuple(ids)
        if key not in stacked:
            for q in ids:
                if q not in hidden_rows:
                    hidden_rows[q] = np.asarray(odds_input.hidden_of(q))
            stacked[key] = (np.stack([hidden_rows[q] for q in ids]) if ids
                            else np.zeros((0, 1)))
        return stacked[key]

    holder_art = None
    if holder is not None:
        held = heads.get(holder, any_head.get(holder))
        raw = pi.heads.get(holder)
        if held is None and isinstance(raw, tuple):
            held = raw
        elif held is None and raw is not None:
            import hashlib

            from fugal_subnet.head_eval import load_head_v4
            if hashlib.sha256(bytes(raw)).hexdigest() == holder:
                h = load_head_v4(bytes(raw))
                held = (h.W, h.b, list(h.models))
        if held is None:
            raise HolderHeadMissing(f"epoch {epoch}: the head of the holder {holder[:16]} the "
                                    "challenges pair with is not held")
        holder_art = _artifact(held)
    holder_picks: list = []

    def holder_picks_through(n: int) -> list:
        if holder_art is None:
            return [None] * n
        if len(holder_picks) < n:
            got = picks_v4.challenge_picks(holder_art, state, hidden(qids[len(holder_picks):n]))
            holder_picks.extend(got)
        return holder_picks[:n]

    values: dict[str, tuple[int, ...]] = {}
    refused: dict[str, str] = {}
    void: dict[str, str] = {}
    means: dict[str, list[int]] = {}
    cells: dict[str, list[int]] = {}
    in_slice = set(sl.question_ids)
    for h, uid in sorted(serving.items()):
        proof = verified_proofs[uid]
        if getattr(proof, "scored", None) is not True or getattr(proof, "challenge", None) is None:
            continue
        ch = proof.challenge
        if ch.holder != holder:
            refused[h] = "pairs with a holder other than the one this epoch names"
            continue
        if ch.size > len(qids) or h not in heads:
            refused[h] = "its challenge cannot be checked against this epoch's list and heads"
            continue
        prefix = qids[:ch.size]
        own_picks = picks_v4.challenge_picks(_artifact(heads[h]), state, hidden(prefix))
        hps = holder_picks_through(ch.size)
        own = {r.question_id: r for r in proof.scored_results}
        held_calls = {r.question_id: r for r in proof.holder_results}
        if set(own) != set(prefix) or len(own) != len(proof.scored_results):
            refused[h] = "its answers are not its challenge's prefix"
            continue
        want_held = {q for q, mine, hp in zip(prefix, own_picks, hps)
                     if hp is not None and hp != mine}
        if any(own[q].routed_model != mine for q, mine in zip(prefix, own_picks)):
            refused[h] = "states a pick other than its head's"
            continue
        if (set(held_calls) != want_held
                or any(held_calls[q].routed_model != hp for q, hp in zip(prefix, hps)
                       if q in want_held)):
            refused[h] = "states a holder pick other than the holder's"
            continue
        answers = [picks_v4.Answer(own[q].routed_model, bool(own[q].correct),
                                   bool(own[q].failed)) for q in prefix]
        held_by_pos = {i: picks_v4.Answer(held_calls[q].routed_model,
                                          bool(held_calls[q].correct),
                                          bool(held_calls[q].failed),
                                          not getattr(held_calls[q], "not_started", None))
                       for i, q in enumerate(prefix) if q in held_calls}
        why = picks_v4.challenge_verdict(held_by_pos)
        if why is not None:
            # void: counted for and against nobody; the run passes through it
            values[h] = picks_v4.void_items(ch.size)
            void[h] = why
            continue
        items = picks_v4.challenge_items(state, cats[:ch.size], answers, hps, held_by_pos)
        values[h] = items
        for e, (n, t) in picks_v4.item_means(items, [a.entry for a in answers], hps).items():
            acc = means.setdefault(e, [0, 0])
            acc[0] += n
            acc[1] += t
        # The calls the slice's shared cells do not hold: the challenger's own
        # past the slice, and every holder-side call.
        for i, q in enumerate(prefix):
            calls = ([own[q]] if q not in in_slice else []) + (
                [held_calls[q]] if q in held_calls
                and not getattr(held_calls[q], "not_started", None) else [])
            for r in calls:
                acc = cells.setdefault(f"{r.routed_model} {cats[i]}", [0, 0, 0])
                acc[0] += 1
                acc[1] += int(bool(r.correct) and not r.failed)
                acc[2] += int(bool(r.failed))
    rec = picks_v4.PicksEpoch(int(epoch), tuple(cats), values, bool(pi.outage), holder)
    records = {**{e: r for e, r in window.picks.items()
                  if e >= epoch - window.size + 1 and e != epoch}, epoch: rec}
    bar = seat.champion
    evidence = {h: picks_v4.evidence_to_reveal(
                    picks_v4.picks_check(records, h, bar, through=epoch),
                    answered_now=h in rec.values)
                for h in sorted(serving)}
    versus = {}
    if holder_art is not None and pairs is not None:
        versus = picks_v4.routing_vs_r(state, pairs, window, holder_art, odds_input.hidden_of,
                                       now=int(epoch))
    block = picks_v4.reveal_block(
        rec, evidence, pi.refused, questions=qids, means={e: tuple(v) for e, v in means.items()},
        cells=cells, challenges_refused=refused,
        routing_vs_r=versus, void=void)
    # The holder's picks on the whole list, kept for publication
    # (_publish_holder_picks), computed once.
    known = {holder: holder_picks_through(len(qids))} if holder_art is not None else {}
    return rec, block, known


def _publish_holder_picks(block: dict, champion: str | None, known: Mapping, pi: PicksInput,
                          odds_input, serving, pool_models) -> dict:
    """The `picks_v4` block with `holder_picks`: the picks of the seat
    holder this epoch's decision names (champion.champion) on every question
    of the challenge list (picks_v4.holder_picks_doc), present whenever there
    is a holder, whether or not anyone challenged; empty with none. Its head
    is read from any verified proof carrying its bytes, else from PicksInput
    `heads` (bytes checked by hash); without either, HolderHeadMissing."""
    import hashlib

    from fugal_subnet import picks_v4

    if champion is None or not pi.challenge:
        return dict(block, holder_picks={})
    qids = [q for q, _c in pi.challenge]
    picks = known.get(champion)
    if picks is None:
        head = None
        uid = serving.get(champion)
        if uid is not None and uid in odds_input.heads:
            head = odds_input.heads[uid]
        elif isinstance(pi.heads.get(champion), tuple):
            head = pi.heads[champion]
        elif champion in pi.heads and hashlib.sha256(bytes(pi.heads[champion])).hexdigest() \
                == champion:
            from fugal_subnet.head_eval import load_head_v4
            h = load_head_v4(bytes(pi.heads[champion]))
            head = (h.W, h.b, list(h.models))
        if head is None:
            raise HolderHeadMissing(f"the seat holder {champion[:16]}'s head is not held: its "
                                    "picks cannot be published")
        hidden = np.stack([np.asarray(odds_input.hidden_of(q)) for q in qids])
        picks = list(picks_v4.challenge_picks(_artifact(head), pi.state, hidden))
    return dict(block, holder_picks=picks_v4.holder_picks_doc(champion, picks, pool_models))


def odds_b64(odds: np.ndarray) -> str:
    """A head's odds as published: little-endian int16, base64."""
    import base64

    return base64.b64encode(np.asarray(odds, dtype="<i2").tobytes()).decode("ascii")


def odds_from_b64(text: str, n: int) -> np.ndarray:
    """Inverse of `odds_b64`, validated: exactly `n` values in -1..ODDS_SCALE."""
    import base64

    from fugal_subnet.head_odds import ODDS_SCALE

    raw = base64.b64decode(text, validate=True)
    if len(raw) != 2 * n:
        raise ValueError(f"head odds hold {len(raw) // 2} values for {n} pairs")
    out = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    if n and (int(out.min()) < -1 or int(out.max()) > ODDS_SCALE):
        raise ValueError("head odds outside -1..ODDS_SCALE")
    return out


def odds_digest(pool_rows: list, odds_by_head: Mapping[str, str]) -> str:
    """One hash over an epoch's pooled pairs and every head's odds on them.

    It goes into `v2_state` (as `odds_digest`), which is the consensus view two
    validators are compared on, so validators that pooled different pairs or
    computed different odds differ visibly in the epoch it happened."""
    import hashlib

    canonical = json.dumps({"pool": pool_rows, "odds": {h: odds_by_head[h] for h in sorted(odds_by_head)}},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pool_and_odds(window: Window, oi: OddsInput, serving: Mapping[str, int],
                   answered: Mapping[str, HeadEpoch], bench_category):
    """(EpochPairs, {head: odds}, flagged coldkeys with reasons, pooled pairs)
    for a v3 epoch.

    Under v4 (`oi.menu_state`) the pairs are labels_v4.LabelPairs (each
    pair's voting coldkeys and what a missing row is scored at in this epoch)
    and every verified head served by its owner has odds, whether or not it
    answered the scored questions: the odds it is scored on, its row's or
    its entry's allowance value (labels_v4.effective)."""
    from fugal_subnet import explore_pool, head_odds, odds_score

    v4 = oi.menu_state is not None
    if v4 and (oi.menu is not None or oi.unscored):
        raise ValueError("a v4 epoch has no grace and no full-menu rule: its odds input names "
                         "no `menu` and no `unscored` entries")

    cells = {(a.question_id, a.model): str(oi.benchmark_of.get(a.question_id, ""))
             for a in oi.answers}
    expected = odds_score.expected_odds(window.pairs, oi.ref_prior, cells,
                                        min_model_pairs=oi.implausible_min_trials)
    implausible = explore_pool.private_implausible(
        oi.answers, expected, z=oi.implausible_z, min_samples=oi.implausible_min_samples)
    pooled = explore_pool.pool_pairs(
        oi.answers, nonce=oi.nonce, coldkey_cap=oi.coldkey_cap,
        flagged=set(oi.flagged) | set(implausible), weights=oi.explore_weights,
        private_weights=oi.private_weights)
    benches = tuple(str(oi.benchmark_of.get(p.question_id, "")) for p in pooled)
    pairs = EpochPairs(
        question_ids=tuple(p.question_id for p in pooled),
        models=tuple(p.model for p in pooled),
        benchmarks=benches,
        categories=tuple(category_of({"benchmark": b}, bench_category) for b in benches),
        y=np.array([p.y for p in pooled], dtype=np.int32),
        prompt_tokens=np.array([p.prompt_tokens for p in pooled], dtype=np.int32),
        completion_tokens=np.array([p.completion_tokens for p in pooled], dtype=np.int32),
        weight=(np.array([p.weight for p in pooled], dtype=np.int64)
                if oi.explore_weights is not None else None),
        scored=(np.array([p.model not in oi.unscored for p in pooled], dtype=bool)
                if oi.unscored else None),
    )
    if v4:
        labels = _labels_v4()
        voters = explore_pool.pair_voters(
            oi.answers, nonce=oi.nonce, coldkey_cap=oi.coldkey_cap,
            flagged=set(oi.flagged) | set(implausible))
        pairs = labels.label_pairs(
            pairs, [voters[(p.question_id, p.model)] for p in pooled],
            labels.missing_odds(oi.menu_state, pairs.models))
    hidden = [oi.hidden_of(q) for q in pairs.question_ids]
    odds: dict[str, np.ndarray] = {}
    if v4:
        # Every verified owner head at once (head_odds.pair_odds_v4_many:
        # the integers pair_odds_v4 gives, the epoch's embeddings read once).
        from types import SimpleNamespace
        mine = [(h, oi.heads[uid]) for h, uid in sorted(serving.items()) if uid in oi.heads]
        computed = head_odds.pair_odds_v4_many(
            [SimpleNamespace(W=W, b=b, models=list(models)) for _h, (W, b, models) in mine],
            oi.menu_state, hidden, list(pairs.models))
        return (pairs, {h: _labels_v4().effective(o, pairs) for (h, _hd), o in zip(mine, computed)},
                implausible, pooled)
    for h, uid in sorted(serving.items()):
        if (h not in answered and not v4) or uid not in oi.heads:
            continue
        W, b, models = oi.heads[uid]
        if oi.menu is not None and not set(oi.menu) <= set(models):
            # From v3 a head gives odds for every menu entry. One that leaves
            # an entry out is published with no odds anywhere (-1), which the
            # reveal shows and a rebuild reads the same way.
            odds[h] = np.full(len(pairs.models), -1, dtype=np.int16)
            continue
        odds[h] = head_odds.head_odds(W, b, list(models), hidden, list(pairs.models))
    return pairs, odds, implausible, pooled


def _since(copy_state, h: str, blocks_per_epoch: int, epoch_index: int) -> int:
    """The first epoch whose boundary is at or after `h`'s first commit
    (copy_rule.first_epoch); this epoch for a head with no record."""
    from fugal_subnet import copy_rule

    since = copy_rule.first_epoch(copy_state, h, blocks_per_epoch)
    return int(epoch_index if since is None else since)


def late_epochs(window: Window, heads, since: Mapping[str, int], slots: int | None,
                sides: Mapping | None = None, first: Iterable[str] = ()) -> dict[str, list[int]]:
    """{head: the window epochs whose late odds are computed now}: for each
    head, the v4 epochs from `since[head]` on for which the window holds no
    odds of it, newest first; at most `slots` head-epochs in all (None: no
    bound), taken a round at a time, each head its newest missing epoch per
    round, the heads in an order keyed by the window's newest epoch
    (sha256(epoch | head)), so no head is always last. What is left is
    computed in the epochs after, when the head is verified again. A function
    of the window, so every node computes the same and a rebuild reads
    exactly what was published. An epoch a successor's predecessors' stored
    odds stand for (`sides`, seat_v4.side_of: labels_v4.predecessor_odds) is
    not computed: it is read from them. The heads in `first` (the seat's:
    seat_v4.late_odds_first) take their rounds before any other head, in
    the same order among themselves; the others take theirs with the slots
    left. With no such head, or slots for every head-epoch, the choice is
    what it is without them."""
    import hashlib

    labels = _labels_v4()
    newest = max(window.pairs, default=0)
    wanted = {}
    for h in sorted(heads, key=lambda h: hashlib.sha256(f"{newest}|{h}".encode()).hexdigest()):
        have = window.odds.get(h, {})
        wanted[h] = [e for e in sorted(window.pairs, reverse=True)
                     if isinstance(window.pairs[e], labels.LabelPairs)
                     and e >= int(since[h]) and e not in have
                     and (h not in (sides or {})
                          or labels.predecessor_odds(window, sides[h], e) is None)]
    first = frozenset(first)
    chosen: dict[str, list[int]] = {}
    used = 0
    for group in ([h for h in wanted if h in first], [h for h in wanted if h not in first]):
        depth = 0
        while any(depth < len(wanted[h]) for h in group):
            for h in group:
                todo = wanted[h]
                if depth < len(todo) and (slots is None or used < slots):
                    chosen.setdefault(h, []).append(todo[depth])
                    used += 1
            depth += 1
            if slots is not None and used >= slots:
                return chosen
    return chosen


def _add_v4_epoch(window: Window, sl, answers_by_head, pairs, odds_now, oi: OddsInput,
                  serving: Mapping[str, int], copy_state, blocks_per_epoch: int,
                  epoch_index: int, picks=None, odds_from: Mapping[str, int] | None = None,
                  late_slots: int | None = None, sides: Mapping | None = None, seat=None):
    """(the window with v4 epoch `sl` added, the late odds computed now).

    Every verified owner head has odds this epoch, answered or not; and for
    each window epoch from its commit on for which the window holds no odds
    of it (an epoch it was not verified in), they are computed now from its
    bytes and the stored pairs (labels_v4.late_odds). `picks` is the epoch's
    real-picks record (picks_v4.PicksEpoch, rule B), added with the epoch.

    `odds_from` (the v4 seat: seat_v4.odds_since) moves a successor's first
    epoch back to its oldest inherited row's; `late_slots` bounds the late
    odds computed (and published) this epoch (`late_epochs`), which reads a
    successor's predecessors' odds through its seat view (`sides`) and
    serves first the heads the seat state `seat` reads (seat_v4.late_odds_first,
    on the window holding this epoch)."""
    labels = _labels_v4()
    answered = {h: o for h, o in odds_now.items() if h in answers_by_head}
    out = window.add_epoch(sl, answers_by_head, pairs, answered, picks=picks)
    out = labels.with_odds(out, {h: {sl.epoch: o} for h, o in odds_now.items()
                                 if h not in answers_by_head})
    heads = {h: oi.heads[uid] for h, uid in sorted(serving.items()) if uid in oi.heads}
    since = {h: (odds_from or {}).get(h, _since(copy_state, h, blocks_per_epoch, epoch_index))
             for h in heads}
    first = ()
    if seat is not None:
        from fugal_subnet import seat_v4

        first = seat_v4.late_odds_first(seat, out, epoch=epoch_index)
    chosen = late_epochs(out, heads, since, late_slots, sides, first)
    by_epoch: dict[int, list[str]] = {}
    for h, epochs in chosen.items():
        for e in epochs:
            by_epoch.setdefault(e, []).append(h)
    got = {}
    for e, hs in sorted(by_epoch.items()):       # each epoch's embeddings read once
        for h, o in zip(hs, labels.odds_on_many([heads[h] for h in hs], out.pairs[e],
                                                oi.hidden_of)):
            got[(h, e)] = o
    late = {h: {e: got[(h, e)] for e in sorted(epochs)} for h, epochs in chosen.items()}
    return labels.with_odds(out, late), late


def _v4_label_fields(window: Window, pairs, late, serving: Mapping[str, int], copy_state,
                     blocks_per_epoch: int, epoch_index: int) -> dict:
    """The reveal fields of a v4 epoch's labels (labels_v4): each pair's
    voters and missing-row odds, the late odds computed this epoch, and each
    serving head's labels and gain over R."""
    labels = _labels_v4()
    owners = {r.head_hash: r.owner for r in copy_state.heads}
    since = {h: _since(copy_state, h, blocks_per_epoch, epoch_index) for h in serving}
    return {
        "labels_v4": labels.labels_for_reveal(pairs),
        "late_odds": labels.late_for_reveal(late),
        "odds_v4": labels.standing_for_reveal(window, serving, owners, since, now=epoch_index),
    }


def _memo_verdicts(memo, window: Window, state, epoch_index: int) -> dict:
    """{head: memo_v4.MemoEvidence} for every head the v4 seat keeps after
    `seat_v4.observe` (`state`): the holder, the history and every standing
    head, serving this epoch or not, each read through the seat's own view
    of it (seat_v4.side_of: a successor's inherited rows count from its
    predecessor's commit), on `window` holding this epoch. The seat reads
    the refused ones and the reveal publishes all of them."""
    from fugal_subnet import memo_v4, seat_v4

    return memo_v4.verdicts(memo, window, {hs.head: seat_v4.side_of(hs) for hs in state.heads},
                            now=epoch_index)


def _digest_of(obj) -> str:
    import hashlib

    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _odds_shadow(window: Window, ref_prior: int, serving: Mapping[str, int],
                 champion: str | None) -> dict:
    """Where every scored head's odds stand, published each v3 epoch whether or
    not odds decide the seat yet: its mean skill over the reference on its
    current run, and its mean Brier gain over the champion."""
    from fractions import Fraction

    from fugal_subnet import odds_score

    ref = odds_score.reference(window.pairs, ref_prior)
    out = {}
    for h in sorted(serving):
        if h not in window.odds:
            continue
        skill, n = odds_score.solo_skill(window, ref, h)
        row = {"uid": int(serving[h]), "pairs": n, "skill_points": odds_score.points(skill)}
        if champion is not None and champion != h and champion in window.odds:
            bu = odds_score.brier_units(window, ref, h, champion)
            if bu.n_pairs:
                row["vs_champion"] = {
                    "pairs": bu.n_pairs,
                    "brier_gain_points": odds_score.points(Fraction(bu.total, bu.n_pairs)),
                }
        out[h] = row
    return out


@dataclass(frozen=True)
class V2EpochResult:
    window: Window
    decision: SeatDecision
    copy_state: object          # copy_rule.CopyState after this epoch
    uids: list[int]
    weights: list[float]
    reveal: dict
    # {head: the senior seat holders that barred it} (copy_rule.senior_bars),
    # as handed to seat_step. Not in the reveal; protocol v4's entry use
    # counts the picks of the heads in it (LIVE_MENU_V4.md 9.2).
    senior_bars: dict = field(default_factory=dict)
    # The epoch's pooled exploration pairs (explore_pool.pool_pairs), exactly
    # the pool the odds were scored on: the validator's detectors' and the
    # implausibility check's exclusions applied. Not in the reveal; protocol
    # v4's menu summary reads this pool, so the menu step and the seat's odds
    # never pool with different exclusions. Empty without an odds input.
    pooled: tuple = ()

    @property
    def owners(self) -> dict[str, str]:
        """{head hash: the hotkey that first committed it}, for logs and the reveal."""
        return {r.head_hash: r.first_hotkey for r in self.copy_state.heads}


def _v4_result(*, decision, window: Window, copy_state, serving, seat_serving, shares, burn_uid,
               rules, picks_block, odds_input: OddsInput, pairs, late, head_odds_now, pooled,
               implausible, prices, blocks_per_epoch: int, epoch_index: int, cells, tokens,
               verified_proofs, verdicts, seniority_floor, seniority_salt, memo=None,
               memo_evidence=None) -> V2EpochResult:
    """A protocol v4 epoch's weights and reveal from its seat decision
    (seat_v4.decide): 85 / 10 / 5 over the holder and the two before it,
    each seat paid while its head is payable (decision.payable), an unpaid
    seat's share to the holder and an unpaid holder's share burnt
    (payout.seat_weights). No weight cap is applied to a v4 vector (the
    validator's loop). The reveal publishes the seat state (`v2_state.seat`),
    its evidence (`seat_v4`, its digest `v2_state.seat_digest`) and the
    labels, odds and real-picks fields of the earlier packages; it carries no
    burn-in and no v1-v3 seat shadow (`odds_shadow`)."""
    from fugal_subnet import copy_rule, explore_pool, seat_v4
    from fugal_subnet.payout import seat_weights
    from fugal_subnet.shared_grades import cells_for_reveal
    from fugal_subnet.single_model import single_model_table

    uids, weights = seat_weights(decision.champion, decision.state.history, seat_serving,
                                 set(decision.payable), shares=shares, paid_fraction=1.0,
                                 burn_uid=burn_uid)
    owners = {r.head_hash: r.first_hotkey for r in copy_state.heads}
    extra = {copy_rule.REVEAL_KEY: copy_state.to_reveal()}
    if rules is not None:
        extra["rules"] = dict(rules)
    if picks_block is not None:
        from fugal_subnet import picks_v4
        extra[picks_v4.DIGEST_KEY] = picks_v4.digest(picks_block)
    pool_rows = explore_pool.pairs_for_reveal(pooled, odds_input.benchmark_of)
    odds_rows = {h: odds_b64(o) for h, o in sorted(head_odds_now.items())}
    extra["odds_digest"] = odds_digest(pool_rows, odds_rows)
    label_fields = _v4_label_fields(window, pairs, late, serving, copy_state, blocks_per_epoch,
                                    epoch_index)
    extra["labels_digest"] = _digest_of({k: label_fields[k] for k in ("labels_v4", "late_odds")})
    if memo is not None:
        from fugal_subnet import memo_v4

        label_fields["memo_v4"] = memo_v4.memo_for_reveal(memo, memo_evidence, now=epoch_index)
        extra["memo_digest"] = _digest_of(label_fields["memo_v4"])
    extra[seat_v4.DIGEST_KEY] = seat_v4.digest(decision.evidence)
    reveal = {
        # No per-UID `failed` lists in a v4 reveal (FAST_CROWN_V4.md 3): a
        # challenge's failures are inside its items, and the reveal's size
        # budget needs the room.
        "shared_grades": cells_for_reveal(cells, tokens),
        "copy_verdicts": {str(uid): v for uid, v in sorted(verdicts.items())},
        # What orders commits at or before the v4 activation block (rule C):
        # the block they tie at and its hash, which breaks the tie.
        "v2_seniority": {"floor_block": seniority_floor, "salt": seniority_salt},
        "champion": seat_v4.decision_for_reveal(decision),
        "v2_state": v2_state_for_reveal(seat=decision.state, owners=owners, serving=serving,
                                        burn_in=None, extra=extra),
        **label_fields,
        "exploration_pool": pool_rows,
        "head_odds": odds_rows,
        "exploration_excluded_coldkeys": {ck: why for ck, why in sorted(implausible.items())},
        "single_model_table": single_model_table(
            window.pairs, prices, odds_input.category_targets or {},
            min_pairs=odds_input.single_model_min_pairs),
        **({"picks_v4": picks_block} if picks_block is not None else {}),
        seat_v4.REVEAL_KEY: decision.evidence,
    }
    return V2EpochResult(window, decision, copy_state, uids, weights, reveal,
                         dict(decision.tested), tuple(pooled or ()))


def run_v2_epoch(
    *,
    epoch_index: int,
    epoch_id: str,
    boundary_block: int,
    blocks_per_epoch: int,
    questions: list[Mapping],
    pool_models: list[str],
    prices: Mapping[str, tuple[float, float]],
    verified_proofs: Mapping[int, object],
    head_hashes: Mapping[int, str],
    coldkey_of: Mapping[int, str],
    hotkey_of: Mapping[int, str],
    records: Mapping[int, object],
    commitments: list,
    window: Window,
    seat: SeatState,
    copy_state,
    burn_uid: int,
    params: SeatParams,
    shares: tuple[float, ...],
    burn_in_questions: int,
    bench_category: Mapping[str, str] | None = None,
    allow_take: bool = True,
    seniority_floor: int | None = None,
    seniority_salt: str = "",
    odds_input: OddsInput | None = None,
    rules: Mapping | None = None,
    state_rules: Mapping | None = None,
    pick_spend: Mapping[str, tuple[float, Mapping[str, float]]] | None = None,
    picks: PicksInput | None = None,
    memo=None,
) -> V2EpochResult:
    """One v2 epoch, from verified proofs to the weight vector. Pure.

    Ownership, seniority, retirement and the senior bars are the copy rule's
    (fugal_subnet/copy_rule.py). `commitments` is the boundary snapshot,
    one `copy_rule.Commitment` per registered hotkey (with its owning coldkey),
    read pinned at `boundary_block`; `copy_state` is the copy rule's state
    after the previous epoch. A head hash belongs to the coldkey that first
    committed it, only its canonical owner UID serves it (`classify`), its
    seniority is its first commit, a champion is retired when its owner stops
    committing it, and a head takes a seat only by also beating every senior
    seat holder by the margin (`senior_bars`). Copies, duplicates and unknown
    hashes are not candidates at all: they are neither paid, nor measured,
    nor able to retire anything. Nothing here reads the frame, and
    `paid_fraction` is 1.

    `odds_input` (protocol v3) adds the exploration pool and every scored
    head's odds on it to the window and the reveal. Whether they decide the
    seat is `params.odds`; they are computed and published either way.

    `rules` names what this epoch is scored under (protocol, and whether the
    seat is on odds); `state_rules` what the incoming window and seat were
    built under. Where they differ the state is not carried over: a new
    protocol starts from an empty window and an empty seat (its questions and
    its model list are different ones), and a new seat objective starts from
    an empty seat on the window it has. `rules` is published in `v2_state`, so
    a rebuilt validator knows what the state it loaded was built under.

    `pick_spend` (protocol v4) prices the slice's cells at the epoch's
    pick-cost table (`build_epoch_slice`); without it, at their attested
    tokens and `prices`, as in v2 and v3.

    `picks` (protocol v4) adds the epoch's real-picks record to the window
    and publishes rule B's evidence for every serving head (`picks_v4`: the
    reveal's `picks_v4` field and `v2_state`'s `picks_digest`). It decides
    nothing here: the seat package reads it.

    `memo` (protocol v4, memo_v4.EpochMemo) is the memorisation check's
    input: every head the seat keeps (serving or not) is judged on the
    window holding this epoch (`_memo_verdicts`), the refused ones are the
    seat's `memo_refused`, and the reveal's `memo_v4` publishes the same
    evidence with the published set's digest as the epoch's draws read it
    and its fresh questions, their digest in `v2_state` (`memo_digest`).
    Without it no head is refused and no memorisation field is written.

    PROTOCOL v4 (an `odds_input` with a `menu_state`) decides the seat by
    seat_v4 (SEAT_MINIMAL_V4.md 3.3 to 3.8): `seat` is a seat_v4.SeatStateV4
    (an empty v1-v3 seat is read as an empty v4 one), and `seniority_floor`
    and `seniority_salt` are the v4 activation block and its hash. None of
    v1-v3's seat rules run:
    burn-in, the senior bars, ghosts and the accuracy floor are not read, and
    the copy rule keeps a retired head's record for STANDING_EPOCHS_V4
    epochs. The reveal carries the seat's evidence (`seat_v4`, its digest in
    `v2_state`) and no burn-in.
    """
    from fugal_subnet import copy_rule
    from fugal_subnet.champion import window_accuracy
    from fugal_subnet.evidence import burn_in_complete
    from fugal_subnet.payout import seat_weights
    from fugal_subnet.shared_grades import (
        ScoredAnswer,
        cells_for_reveal,
        shared_cell_grades,
        shared_cell_tokens,
    )

    v4 = odds_input is not None and odds_input.menu_state is not None
    if rules is not None and dict(rules) != dict(state_rules or {}):
        if (state_rules or {}).get("protocol") != rules.get("protocol"):
            window = Window.empty(window.size)
        seat = SeatState()
    if v4:
        from fugal_subnet import config, seat_v4

        if isinstance(seat, SeatState):
            if seat != SeatState():
                raise ValueError("a v1-v3 seat cannot be carried into a v4 epoch: a protocol "
                                 "change starts from an empty seat")
            seat = seat_v4.SeatStateV4()

    answers = [
        ScoredAnswer(
            question_id=r.question_id, model=r.routed_model, coldkey=coldkey_of[uid],
            correct=bool(r.correct), failed=bool(r.failed),
            completion_tokens=int(r.completion_tokens),
        )
        for uid in sorted(verified_proofs)
        for r in verified_proofs[uid].scored_results
    ]
    if v4:
        # A v4 proof's scored results are its challenge's own picks, the
        # slice first: the shared cells are the slice's (FAST_CROWN_V4.md 3);
        # the challenge's other calls are published as sums (_challenge_epoch).
        slice_ids = {q["question_id"] for q in questions}
        answers = [a for a in answers if a.question_id in slice_ids]
    cells = shared_cell_grades(answers)
    tokens = shared_cell_tokens(answers)
    sl = build_epoch_slice(epoch_index, questions, pool_models, cells, tokens, prices,
                           bench_category, pick_spend=pick_spend)
    model_index = {m: i for i, m in enumerate(pool_models)}

    # The copy rule's fold of this boundary's commitments. Idempotent across a
    # retry of the same epoch: a state already folded through this epoch (a
    # --once re-run after the state was written) is used as it is.
    if copy_state.epoch is not None and int(epoch_index) <= copy_state.epoch:
        new_copy = copy_state
    else:
        new_copy = copy_rule.observe(copy_state, epoch_index, boundary_block, commitments,
                                     ghost_retire=(config.STANDING_EPOCHS_V4 if v4
                                                   else params.ghost_retire))

    # Who serves each head: only its canonical owner UID.
    committed_by = {c.hotkey: c for c in copy_rule.valid_commitments(commitments, boundary_block)}
    served = {}
    for uid in sorted(verified_proofs):
        c = committed_by.get(hotkey_of[uid])
        if c is not None and c.head_hash == head_hashes[uid]:
            served[uid] = copy_rule.Commitment(hotkey_of[uid], coldkey_of[uid],
                                               head_hashes[uid], c.block)
    verdicts = copy_rule.classify(new_copy, served)
    serving: dict[str, int] = {
        head_hashes[uid]: uid for uid, v in sorted(verdicts.items()) if v == "owner"
    }

    answers_by_head: dict[str, HeadEpoch] = {}
    for h, uid in sorted(serving.items()):
        he = head_epoch(verified_proofs[uid], sl.question_ids, model_index)
        if he is not None:
            answers_by_head[h] = he
    pairs = head_odds_now = pooled = None
    implausible: dict[str, str] = {}
    late: dict = {}
    if odds_input is not None:
        pairs, head_odds_now, implausible, pooled = _pool_and_odds(
            window, odds_input, serving, answers_by_head, bench_category)
    picks_rec = picks_block = None
    if picks is not None and not v4:
        raise ValueError("the challenge records are a v4 epoch's: they need its odds input")
    if v4:
        # The seat's view of the epoch's heads first (succession), so a
        # successor's odds are computed from its oldest inherited row.
        seat_serving = seat_v4.seat_serving(seat, serving, new_copy)
        if picks is not None:
            # A challenge counts for a head only when the seat's view says its
            # own operator served it (seat_serving), never for the same bytes
            # committed by another coldkey after the first commit lapsed.
            picks_rec, picks_block, known = _challenge_epoch(picks, epoch_index, sl, seat_serving,
                                                      verified_proofs, seat, window, odds_input,
                                                      pairs, verified_by_hash=serving)
        rows = {h: seat_v4.row_digests(*odds_input.heads[uid])
                for h, uid in sorted(seat_serving.items()) if uid in odds_input.heads}
        menu_ids = tuple(odds_input.menu_state.ids())
        observed = seat_v4.observe(seat, epoch=epoch_index, serving=seat_serving, rows=rows,
                                   copy_state=new_copy, menu_ids=menu_ids,
                                   blocks_per_epoch=blocks_per_epoch)
        new_window, late = _add_v4_epoch(
            window, sl, answers_by_head, pairs, head_odds_now, odds_input, serving, new_copy,
            blocks_per_epoch, epoch_index, picks=picks_rec,
            odds_from={h: seat_v4.odds_since(observed, h, _since(new_copy, h, blocks_per_epoch,
                                                                 epoch_index))
                       for h in serving},
            late_slots=config.LATE_ODDS_SLOTS_V4,
            sides={hs.head: seat_v4.side_of(hs) for hs in observed.heads if hs.predecessors},
            seat=observed)
        # The memorisation verdict on the window as it now stands, for every
        # head the seat keeps (serving or not), before the seat reads it.
        memo_evidence = refused = None
        if memo is not None:
            from fugal_subnet import memo_v4

            memo_evidence = _memo_verdicts(memo, new_window, observed, epoch_index)
            refused = frozenset(h for h, ev in memo_evidence.items() if memo_v4.refused(ev))
        decision = seat_v4.decide(observed, epoch=epoch_index, window=new_window,
                                  serving=seat_serving, floor_block=seniority_floor,
                                  salt=seniority_salt, memo_refused=refused or frozenset(),
                                  allow_take=allow_take)
        if picks_block is not None:
            picks_block = _publish_holder_picks(picks_block, decision.champion, known, picks,
                                                odds_input, serving, pool_models)
        return _v4_result(
            decision=decision, window=new_window, copy_state=new_copy, serving=serving,
            seat_serving=seat_serving, shares=shares, burn_uid=burn_uid, rules=rules,
            picks_block=picks_block, odds_input=odds_input, pairs=pairs, late=late,
            head_odds_now=head_odds_now, pooled=pooled, implausible=implausible, prices=prices,
            blocks_per_epoch=blocks_per_epoch, epoch_index=epoch_index, cells=cells,
            tokens=tokens, verified_proofs=verified_proofs, verdicts=verdicts,
            seniority_floor=seniority_floor, seniority_salt=seniority_salt, memo=memo,
            memo_evidence=memo_evidence)
    new_window = window.add_epoch(sl, answers_by_head, pairs, head_odds_now, picks=picks_rec)

    candidates: dict[str, Candidate] = {}
    for h, uid in sorted(serving.items()):
        if h not in answers_by_head:
            continue
        rec = records.get(uid)
        since = copy_rule.first_epoch(new_copy, h, blocks_per_epoch)
        candidates[h] = Candidate(
            head_hash=h,
            seniority=copy_rule.v2_seniority(new_copy, h, floor_block=seniority_floor,
                                             salt=seniority_salt),
            eligible=True,
            since_epoch=int(since if since is not None else epoch_index),
            burned_in=burn_in_complete(getattr(rec, "evidence", None), burn_in_questions),
            first_block=(new_copy.get(h).first_block if new_copy.get(h) is not None else None),
        )

    # Retired: the champion or a standing ghost whose OWNER no longer commits
    # it. A missed ghost that becomes retired is upgraded by seat_step.
    watched = {g.head_hash for g in seat.ghosts} | ({seat.champion} if seat.champion else set())
    # A watched head with no copy record at all has been retired too: the copy
    # rule drops a record once its FIRST retirement is older than ghost_retire,
    # so a champion whose head was retired once, re-committed and then dropped
    # again has no record the epoch its owner moves on, and was read as merely
    # missed (a 3-epoch ghost instead of the 24-epoch retired one; theme-2
    # review, 2026-11-08).
    retired = (set(copy_rule.retired_hashes(new_copy))
               | {h for h in watched if h not in new_copy}) & watched

    bars = copy_rule.senior_bars(
        new_copy, epoch_index, new_window, candidates,
        holders=copy_rule.seat_holders(seat, len(shares)),
        ghost_off=params.ghost_off, ghost_retire=params.ghost_retire,
        floor_block=seniority_floor, salt=seniority_salt,
    )
    decision = seat_step(seat, epoch_index, epoch_id, new_window, candidates,
                         retired=retired, params=params, allow_take=allow_take,
                         senior_bars=bars)

    # Payout eligibility is the HOLD rule: live, canonical owner, window
    # accuracy at or above the floor. Burn-in is not required, so a champion
    # that resumes after a miss (its E'' counter restarted) is paid at once.
    payable = set()
    for h in candidates:
        acc, n = window_accuracy(new_window, h)
        if n > 0 and acc >= params.min_accuracy:
            payable.add(h)
    # Seat 1 is the champion's, and a champion that is gone but still barring
    # (a ghost) keeps it: its share burns, and seats 2 and 3 keep paying the
    # two champions before it.
    ghost_hashes = {g.head_hash for g in decision.state.ghosts}
    seat_one = decision.champion
    history = tuple(decision.state.history)
    if seat_one is None and history and history[0] in ghost_hashes:
        seat_one, history = history[0], history[1:]
        # The ghost holds the seat; it is not paid for it, even when its head
        # is live again (a retired champion that recommitted its old hash
        # serves as its own ghost and was paid seat 1 every epoch it served;
        # v2 attack review, 2026-11-09).
        payable = payable - {seat_one}
    uids, weights = seat_weights(
        seat_one, history, serving, payable,
        shares=shares, paid_fraction=1.0, burn_uid=burn_uid,
    )
    owners = {r.head_hash: r.first_hotkey for r in new_copy.heads}
    extra = {copy_rule.REVEAL_KEY: new_copy.to_reveal()}
    if rules is not None:
        extra["rules"] = dict(rules)
    if picks_block is not None:
        from fugal_subnet import picks_v4
        extra[picks_v4.DIGEST_KEY] = picks_v4.digest(picks_block)
    v3_fields: dict = {}
    if odds_input is not None:
        from fugal_subnet import explore_pool
        from fugal_subnet.single_model import single_model_table

        pool_rows = explore_pool.pairs_for_reveal(pooled, odds_input.benchmark_of)
        odds_rows = {h: odds_b64(o) for h, o in sorted(head_odds_now.items())}
        extra["odds_digest"] = odds_digest(pool_rows, odds_rows)
        v3_fields = {
            **v3_fields,
            "exploration_pool": pool_rows,
            **({"odds_unscored_entries": sorted(odds_input.unscored)}
               if odds_input.unscored else {}),
            "head_odds": odds_rows,
            "exploration_excluded_coldkeys": {
                ck: why for ck, why in sorted(implausible.items())},
            "odds_shadow": _odds_shadow(new_window, odds_input.ref_prior, serving,
                                        decision.champion),
            "single_model_table": single_model_table(
                new_window.pairs, prices, odds_input.category_targets or {},
                min_pairs=odds_input.single_model_min_pairs),
        }
    if memo is not None:
        raise ValueError("the memorisation fields are a v4 epoch's: they need its odds input")
    reveal = {
        "shared_grades": cells_for_reveal(cells, tokens),
        "failed": {
            str(uid): sorted(r.question_id for r in verified_proofs[uid].scored_results if r.failed)
            for uid in sorted(verified_proofs)
        },
        "copy_verdicts": {str(uid): v for uid, v in sorted(verdicts.items())},
        # What ordered pre-flip commits: the activation block they tie at and
        # the boundary hash that breaks the tie (anyone can check it on chain).
        "v2_seniority": {"floor_block": seniority_floor, "salt": seniority_salt},
        "champion": decision_for_reveal(decision),
        "v2_state": v2_state_for_reveal(
            seat=decision.state, owners=owners, serving=serving,
            burn_in={
                h: float(getattr(getattr(records.get(uid), "evidence", None),
                                 "burn_in_progress", 0.0) or 0.0)
                for h, uid in serving.items()
            },
            extra=extra,
        ),
        **v3_fields,
        **({"picks_v4": picks_block} if picks_block is not None else {}),
    }
    return V2EpochResult(new_window, decision, new_copy, uids, weights, reveal, bars,
                         tuple(pooled or ()))


__all__ = [
    "Candidate", "OddsInput", "PicksInput", "V2EpochResult", "build_epoch_slice", "odds_b64", "odds_digest",
    "odds_from_b64", "run_v2_epoch", "category_of", "decision_for_reveal", "head_epoch",
    "load_window", "save_window", "seat_from_dict", "seat_params_from_config", "seat_to_dict",
    "v2_state_for_reveal",
]
