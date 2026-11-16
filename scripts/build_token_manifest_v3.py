"""Build the PROVISIONAL v3 token manifest (benchmark_tokens_protocol_v3.json).

    python scripts/build_token_manifest_v3.py \\
      data/benchmark_tokens_protocol_v2.json data/models_v3.json data/benchmark_tokens_protocol_v3.json \\
      [v3-measured.jsonl]
    cp data/benchmark_tokens_protocol_v3.json fugal_subnet/routing_data/

The optional 4th argument is a JSONL of v3 observations written by
`scripts/record_token_observations.py --protocol 3` (entry id in `model`).
Rule 0 comes first when it is given:
  0. measured v3: >= MIN_V3_SAMPLES completed calls for the entry -> their
     means. A rule-1 row (v2, same model and setting) is kept unless the v3
     means differ from it by more than 2x either way (then v3 wins, noted).
No paid calls here. Rule, per v3 menu entry (model m, reasoning r):
  1. measured: v2 measured m with exactly r -> v2's means, as they are.
  2. scaled:   v2 measured m (or its named predecessor/sibling) with another
     setting -> v2's prompt mean as is; completion mean x EFFORT[r] / EFFORT[v2's],
     capped at the entry's max_tokens.
  3. default:  no v2 model to lean on -> the median v2 prompt and completion
     means, completion x EFFORT[r].
Every row except rule 1 is marked provisional. The monthly menu update replaces
provisional rows with the free per-entry statistics that protocol v3
exploration publishes (single_model.single_model_table, attested tokens of
pooled pairs); a new entry gets a small paid sample (about 10 calls) appended,
never a re-measure of rows already measured.
"""
import hashlib
import json
import statistics
import sys

V2, MENU, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
V3_OBS = sys.argv[4] if len(sys.argv) > 4 else ""
MIN_V3_SAMPLES = 10
v2 = json.load(open(V2))
menu = json.load(open(MENU))
v2_rows = {r["id"]: r for r in v2["models"]}
v2_table = {r["id"]: r for r in json.load(open(MENU.replace("models_v3.json", "models_v2.json")))}

# Completion-length factor by effort, relative to a medium-equivalent setting.
# v2 pinned `effort: medium` (OpenAI), `max_tokens: 8192` (Anthropic, Google,
# Qwen) or `enabled: true` (open-weight models); all three are read as medium.
EFFORT = {"minimal": 0.25, "low": 0.5, "medium": 1.0, "high": 2.0, "xhigh": 3.0, "max": 4.0}
PREDECESSOR = {  # v3 model -> the v2 model its lengths are scaled from
    "deepseek/deepseek-v4-pro": "deepseek/deepseek-v4-pro-0813",
    "deepseek/deepseek-v4-flash": "deepseek/deepseek-v4.1-flash",
    "deepseek/deepseek-v4.1-flash-1017": "deepseek/deepseek-v4.1-flash",
    "qwen/qwen4-max": "qwen/qwen3.8-2.4t-a95b",
    "xiaomi/mimo-v2.5-pro": "xiaomi/mimo-v2.6-pro",
    "google/gemini-4-carbon": "google/gemini-4-argon",
}


def effort_of(reasoning):
    if reasoning and "effort" in reasoning:
        return reasoning["effort"]
    return "medium"


med_in = statistics.median(r["mean_in_tokens"] for r in v2["models"])
med_out = statistics.median(r["mean_out_tokens"] for r in v2["models"])
v3_obs: dict[str, list[dict]] = {}
if V3_OBS:
    for line in open(V3_OBS):
        if line.strip():
            o = json.loads(line)
            if o.get("protocol") == 3:
                v3_obs.setdefault(o["model"], []).append(o)


def v3_row(entry_id):
    obs = v3_obs.get(entry_id, [])
    if len(obs) < MIN_V3_SAMPLES:
        return None
    n = len(obs)
    return {"id": entry_id, "samples": n,
            "mean_in_tokens": round(sum(o["prompt_tokens"] for o in obs) / n, 3),
            "mean_out_tokens": round(sum(o["completion_tokens"] for o in obs) / n, 3),
            "basis": (f"measured: v3 2026-11-16, {n} calls on the non-code v3 pool through the v3 "
                      f"harness path (system prompt, entry budget, pinned hosts); "
                      f"{sum(1 for o in obs if o.get('hit_budget'))} used the whole budget")}


def far(a, b):
    return a > 2 * b or b > 2 * a


rows, counts = [], {"measured_v3": 0, "measured": 0, "scaled": 0, "default": 0}
kept_v2_over_v3, v3_over_v2 = [], []
for e in menu:
    m, r, cap = e["model"], e.get("reasoning"), int(e["max_tokens"])
    src_id = m if m in v2_rows else PREDECESSOR.get(m)
    if src_id is not None and m in v2_rows and v2_table[m].get("reasoning") == r:
        src = v2_rows[m]
        row = {"id": e["id"], "samples": src["samples"], "mean_in_tokens": src["mean_in_tokens"],
               "mean_out_tokens": src["mean_out_tokens"], "basis": "measured: v2 2026-11-09, same model and reasoning setting"}
        counts["measured"] += 1
    elif src_id is not None:
        src = v2_rows[src_id]
        f = EFFORT[effort_of(r)] / EFFORT[effort_of(v2_table[src_id].get("reasoning"))]
        out = min(float(src["mean_out_tokens"]) * f, cap)
        row = {"id": e["id"], "samples": src["samples"], "mean_in_tokens": src["mean_in_tokens"],
               "mean_out_tokens": round(out, 3), "provisional": True,
               "basis": f"scaled: v2 {src_id} (completion x {f:g} for effort {effort_of(r)})"}
        counts["scaled"] += 1
    else:
        f = EFFORT[effort_of(r)]
        row = {"id": e["id"], "samples": 1, "mean_in_tokens": round(med_in, 3),
               "mean_out_tokens": round(min(med_out * f, cap), 3), "provisional": True,
               "basis": f"default: median of the 19 v2 means (completion x {f:g} for effort {effort_of(r)})"}
        counts["default"] += 1
    v3r = v3_row(e["id"])
    if v3r is not None:
        if not row.get("provisional"):
            if far(v3r["mean_out_tokens"], row["mean_out_tokens"]) or far(v3r["mean_in_tokens"], row["mean_in_tokens"]):
                v3_over_v2.append(e["id"])
            else:
                kept_v2_over_v3.append(e["id"])
                rows.append(row)
                continue
            counts["measured"] -= 1
        else:
            counts["scaled" if row["basis"].startswith("scaled") else "default"] -= 1
        v3r["mean_out_tokens"] = min(v3r["mean_out_tokens"], float(cap))
        row = v3r
        counts["measured_v3"] += 1
    rows.append(row)

manifest = {
    "version": "fugal-token-statistics-v1",
    "profile_id": v2["profile_id"],
    "status": "candidate",
    "sources": {"benchmark_tokens_protocol_v2.json": hashlib.sha256(open(V2, "rb").read()).hexdigest(),
                **({"tokens-2026-11-16.jsonl": hashlib.sha256(open(V3_OBS, "rb").read()).hexdigest()}
                   if V3_OBS else {})},
    "worker_profile": (
        "Protocol v3 PROVISIONAL token statistics, built 2026-11-16 with no paid calls "
        "(scripts/build_token_manifest_v3.py rule): rows measured in v2 at the same model and "
        "reasoning setting are reused as they are; every other row is an estimate from v2's "
        "measured means, marked provisional. Replaced row by row at the monthly menu update by "
        "the attested per-entry token medians protocol v3 exploration publishes "
        "(single_model.single_model_table); new entries get a small paid sample appended."),
    "omitted_missing_or_zero_input": 0,
    "models": rows,
    "review": {
        "date": "2026-11-16",
        "status_note": "candidate: the v3 release review promotes it to reviewed",
        "counts": counts,
        "v2_rows_kept_over_v3": kept_v2_over_v3,
        "v2_rows_replaced_by_v3_over_2x": v3_over_v2,
        "effort_factors": EFFORT,
        "predecessors": PREDECESSOR,
        "limitations": [
            "v2 means were measured on the v2 pool with a 16,384-token answer budget; v3 adds code "
            "questions and raises budgets to each entry's max_tokens, so provisional completion "
            "means are likely low for long reasoners.",
            "Effort factors are a stated assumption, not a measurement."],
    },
}
text = json.dumps(manifest, indent=1) + "\n"
open(OUT, "w").write(text)
print(counts, "kept v2:", kept_v2_over_v3, "v3 over v2 (>2x):", v3_over_v2)
for r in rows:
    print(f"{r['id']:42s} in={r['mean_in_tokens']:>8} out={r['mean_out_tokens']:>9} {r['basis']}")
