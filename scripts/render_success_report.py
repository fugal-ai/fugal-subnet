#!/usr/bin/env python3
"""Render both completed offline mechanism runs into one reviewable report."""
import argparse
import json
from pathlib import Path


def ci(value, scale=1):
    a, b = value["ci95"]
    return f"{value['mean']*scale:.4f} [{a*scale:.4f}, {b*scale:.4f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    src = Path(args.results)
    reports = {n: json.loads((src / f"report-{n}.json").read_text()) for n in (512, 2048)}
    a, b = reports[512], reports[2048]
    assert a["sources"] == b["sources"] and a["selected_indices"] == b["selected_indices"]
    assert a["splits"] == b["splits"]
    lines = ["# Offline success-contract mechanism report", "",
             "This is a SPROUT mechanism test on historical models and recorded outcomes. It does not establish live-subnet calibration or nominate a release head. The production contract remains 2048 tokens regardless of this comparison.", "",
             "## Data and method", "",
             f"The existing selected sample contains {len(b['selected_indices']):,} questions and {len(b['per_model'])} model columns. Exact duplicate prompts stay together in deterministic 60/20/20 groups (seed 42), yielding " + ", ".join(f"{name}: {len(ids):,}" for name, ids in b["splits"].items()) + ". The 512- and 2048-token heads are trained separately, with the same sample and splits. Each checkpoint is selected by validation BCE; the test split never selects a checkpoint.", "",
             b["label_conversion"] + ". Observed all-failure questions are retained; missing labels are not failures. Training uses independent masked BCE and no CMA refinement.", "",
             "Sampling inherits the earlier experiment's source-stratified selection and character-length filter. It is not an unbiased draw from all SPROUT prompts or the live subnet pool. The recorded sample indices and exact splits are included in both JSON reports.", "",
             b["token_limitations"], "", b["worker_profile"] + ".", "",
             "Costs below use recorded per-question input/output tokens multiplied by the historical experiment's **assumed** price snapshot. They are not provider-reported actual dollar charges. The complete evaluated snapshot is bundled. Core can explicitly use a different snapshot, which may change selections.", "",
             "| Source extract | SHA256 |", "| --- | --- |"]
    for name, value in b["sources"].items():
        lines.append(f"| {name} | `{value}` |")
    lines += ["", "## Embedding and calibration comparison", "",
              "Full embedding passes run on CPU float32 with the pinned model/tokenizer and exact core prompt. Timing is wall time on this WSL host during development; concurrent checks and memory pressure can affect it. Peak RSS is per-process high-water memory, not GPU memory. These timings are mechanism measurements, not a controlled hardware performance claim.", "",
              "See [execution notes](RUN_NOTES.md) for the discarded memory-limited attempt and the checkpointed rerun. The comparison below uses the successful passes only.", "",
              "| Input limit | Truncated | Total seconds | Seconds/question | Peak RSS MiB | Selected epoch | Pooled test Brier |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, r in reports.items():
        e = r["embedding"]
        lines.append(f"| {n} | {e['truncated']}/{e['n']} ({100*e['truncated']/e['n']:.2f}%) | {e['seconds']:.1f} | {e['seconds_per_question']:.4f} | {e['peak_rss_kib']/1024:.1f} | {r['selected_epoch']} | {r['pooled']['brier']:.6f} |")
    delta = b["pooled"]["brier"] - a["pooled"]["brier"]
    lines += ["", f"The 2048-token variant changes pooled test Brier by {delta:+.6f} relative to 512 (lower is better). This result does not change the 2048-token contract default.", "",
              "## Held-out routing", "",
              "Each cell is a mean with a paired 95% bootstrap interval (2,000 question resamples, seed 7). Bootstrap draws are shared across policies. Intervals condition on this selected sample and the fitted head; they do not include retraining or dataset-selection uncertainty.", "",
              "| Input limit | λ | Accuracy % [95% CI] | Recorded-token cost $/1,000 queries [95% CI] |",
              "| --- | --- | --- | --- |"]
    for n, r in reports.items():
        for lam, route in r["routes"].items():
            lines.append(f"| {n} | {lam} | {ci(route['accuracy'],100)} | {ci(route['recorded_cost'],1000)} |")
    lines += ["", "## Fixed-model and cheapest baselines", "",
              f"Cheapest by train-split estimated query cost: `{b['cheapest_model']}`. Fixed-model baselines use the same held-out outcomes and actual recorded tokens as routing.", "",
              "| Model | Accuracy % [95% CI] | Cost $/1,000 queries [95% CI] | 2048 λ=1 accuracy minus fixed, percentage points [95% CI] |",
              "| --- | --- | --- | --- |"]
    for m, v in b["fixed_models"].items():
        lines.append(f"| {m} | {ci(v['accuracy'],100)} | {ci(v['recorded_cost'],1000)} | {ci(b['routes']['1']['paired_accuracy_minus_fixed'][m],100)} |")
    lines += ["", "## Fixed-mixture frontier", "",
              "Frontier endpoints are chosen from validation-set fixed-model cost/accuracy, then evaluated on test. Mixtures are expected outcomes of a question-independent randomized policy; the probability of the right endpoint is shown. These are offline baselines, not changes to live frontier rewards.", "",
              "| Left model | Right model | Right probability | Accuracy % [95% CI] | Cost $/1,000 queries [95% CI] |",
              "| --- | --- | --- | --- | --- |"]
    for v in b["fixed_mixture_frontier"]:
        lines.append(f"| {v['left']} | {v['right']} | {v['right_weight']:.2f} | {ci(v['accuracy'],100)} | {ci(v['recorded_cost'],1000)} |")
    lines += ["", "## Per-model discrimination and calibration", "",
              "AUC is undefined when a held-out model column has only one label class. Brier and calibration bins use observed labels only.", "",
              "| Model | Observed test labels | 512 AUC | 2048 AUC | 512 Brier | 2048 Brier |", "| --- | --- | --- | --- | --- | --- |"]
    def number(v):
        return "undefined" if v is None else f"{v:.6f}"
    for m, v in b["per_model"].items():
        old = a["per_model"][m]
        lines.append(f"| {m} | {v['n']} | {number(old['auc'])} | {number(v['auc'])} | {old['brier']:.6f} | {v['brier']:.6f} |")
    for n, r in reports.items():
        lines += ["", f"### {n}-token pooled calibration", "", "| Probability bin | Observations | Mean prediction | Observed success |", "| --- | --- | --- | --- |"]
        for v in r["pooled"]["calibration"]:
            lines.append(f"| {v['lo']:.1f}–{v['lo']+.1:.1f} | {v['n']} | {v['mean_prediction']:.6f} | {v['success_rate']:.6f} |")
    lines += ["", "Per-model calibration tables, every λ-specific paired fixed-model contrast, model selection counts, and training histories are retained in the adjacent JSON reports.", "",
              "## Reproduction", "", "Use the exact three source extracts whose hashes appear above, and the pinned local Qwen snapshot. Reuse the existing selected indices; do not resample.", "", "```bash",
              "python scripts/evaluate_success_contract.py --data /path/to/experiments \\",
              "  --backbone /path/to/Qwen3-0.6B --output /path/to/results --max-length 2048",
              "python scripts/evaluate_success_contract.py --data /path/to/experiments \\",
              "  --backbone /path/to/Qwen3-0.6B --output /path/to/results --max-length 512",
              "python scripts/render_success_report.py --results /path/to/results \\",
              "  --output /path/to/results/REPORT.md", "```", "",
              "The experimental 512 artifact uses a distinct, unsupported production contract marker and cannot be loaded as a 2048 success head. The 2048 artifact can be exported offline and loaded in core through FUGAL_HEAD with its evaluated price snapshot. SPROUT token provenance is incomplete, so this bundle is not deployable.", "",
              "No model calls were purchased, no winner was selected, no shipped head was replaced, and no testnet reset or release publication was performed.", ""]
    Path(args.output).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
