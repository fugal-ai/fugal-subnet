#!/usr/bin/env python3
"""Build v3 menu B (models_v3b.json) and its token manifest (benchmark_tokens_protocol_v3b.json).

Menu B is menu A (data/models_v3.json, unchanged and still pinned) with the
entries below APPENDED, and its manifest is menu A's reviewed manifest with
one measured row per added entry appended. Nothing in menu A changes, so a head
built for menu A keeps its cost profile and stays valid after the menu B block
(docs/decisions/V3_MENU_B.md).

    python scripts/build_menu_v3b.py data/observations/tokens_v3b_new_entries_2026-11-17.jsonl
    cp data/models_v3b.json data/benchmark_tokens_protocol_v3b.json fugal_subnet/routing_data/

The measured rows file holds one JSON object per paid call made through the v3
request path (pinned hosts, entry budget, system prompt): id, question_id, ok,
prompt_tokens, completion_tokens. Only ok rows count; an added entry needs at
least MIN_SAMPLES of them.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_SAMPLES = 10

_PROVIDER = {"order": ["openai", "azure"], "allow_fallbacks": False}
# The added entries (approved 2026-11-17): OpenAI's GPT-6.2 Astra and Sol, at
# the effort levels the menu offers OpenAI's top models.
ADDED = (
    [{"id": f"openai/gpt-6.2-astra@{e}", "model": "openai/gpt-6.2-astra", "in": 8.0, "out": 40.0,
      "reasoning": {"effort": e}, "max_tokens": 128000, "provider": _PROVIDER}
     for e in ("low", "high", "max")]
    + [{"id": f"openai/gpt-6.2-sol@{e}", "model": "openai/gpt-6.2-sol", "in": 2.0, "out": 10.0,
        "reasoning": {"effort": e}, "max_tokens": 128000, "provider": _PROVIDER}
       for e in ("low", "medium", "high", "max")]
)


def _menu_text(rows: list[dict]) -> str:
    """The menu file's layout: one entry per line, as data/models_v3.json."""
    return "[\n" + ",\n".join("  " + json.dumps(r) for r in rows) + "\n]\n"


def main(obs_path: str) -> int:
    from fugal_subnet.routing_protocol import validate_menu_v3

    menu_a = json.load(open(os.path.join(ROOT, "data", "models_v3.json")))
    man_a = json.load(open(os.path.join(ROOT, "data", "benchmark_tokens_protocol_v3.json")))
    ids_a = {r["id"] for r in menu_a}
    added = sorted(ADDED, key=lambda r: r["id"])
    if ids_a & {r["id"] for r in added}:
        raise SystemExit("an added entry is already in menu A")
    menu_b = list(menu_a) + added
    validate_menu_v3(menu_b)

    rows: dict[str, list[dict]] = {}
    with open(obs_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("ok"):
                    rows.setdefault(r["id"], []).append(r)
    man_rows = list(man_a["models"])
    for entry in added:
        got = rows.get(entry["id"], [])
        if len(got) < MIN_SAMPLES:
            raise SystemExit(f"{entry['id']}: {len(got)} measured calls, {MIN_SAMPLES} needed")
        n = len(got)
        man_rows.append({
            "id": entry["id"], "samples": n,
            "mean_in_tokens": round(sum(r["prompt_tokens"] for r in got) / n, 1),
            "mean_out_tokens": round(sum(r["completion_tokens"] for r in got) / n, 1),
            "basis": f"measured: v3 2026-11-17, {n} calls on the non-code v3 pool through the "
                     "v3 request path (pinned hosts, entry budget, system prompt)",
        })
    obs_sha = hashlib.sha256(open(obs_path, "rb").read()).hexdigest()
    man_b = dict(man_a)
    man_b["sources"] = dict(man_a["sources"], **{os.path.basename(obs_path): obs_sha})
    man_b["models"] = man_rows
    man_b["status"] = "reviewed"
    man_b["review"] = dict(man_a.get("review") or {}, menu_b={
        "date": "2026-11-17",
        "status_note": "reviewed: menu A's reviewed rows unchanged, plus one row per added entry "
                       "measured on the v3 path (10 calls each, all served by OpenAI); approved "
                       "by the owner on 2026-11-17 for v0.6.0",
        "added": [r["id"] for r in added],
    })
    with open(os.path.join(ROOT, "data", "models_v3b.json"), "w", encoding="utf-8") as f:
        f.write(_menu_text(menu_b))
    with open(os.path.join(ROOT, "data", "benchmark_tokens_protocol_v3b.json"), "w",
              encoding="utf-8") as f:
        f.write(json.dumps(man_b, indent=1) + "\n")
    for r in man_rows[len(man_a["models"]):]:
        print(f"{r['id']:32s} n={r['samples']} in={r['mean_in_tokens']} out={r['mean_out_tokens']}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    raise SystemExit(main(sys.argv[1]))
