#!/usr/bin/env python3
"""Render completed evaluation results and an explicitly deferred comparison."""
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
    p.add_argument("--partial-512", help="Explicitly report an interrupted comparison using this summary JSON")
    args = p.parse_args()
    src = Path(args.results)
    limits = (2048,) if args.partial_512 else (512, 2048)
    reports = {n: json.loads((src / f"report-{n}.json").read_text()) for n in limits}
    a, b = reports.get(512), reports[2048]
    if a:
        assert a["sources"] == b["sources"] and a["selected_indices"] == b["selected_indices"]
        assert a["splits"] == b["splits"] and a["training"] == b["training"]
    partial = json.loads(Path(args.partial_512).read_text()) if args.partial_512 else None
    if partial:
        assert partial["sources"] == b["sources"]
    training = b["training"]
    lines = ["# Offline success-contract mechanism report", "",
             "This is a SPROUT mechanism test on historical models and recorded outcomes. It does not establish live-subnet calibration or nominate a release head. The production contract remains 2048 tokens regardless of this comparison.", "",
             "## Data and method", "",
             f"The existing selected sample contains {len(b['selected_indices']):,} questions and {len(b['per_model'])} model columns. Exact duplicate prompts stay together in deterministic 60/20/20 groups (seed 42), yielding " + ", ".join(f"{name}: {len(ids):,}" for name, ids in b["splits"].items()) + ". Completed heads use these same sample indices and splits. Each checkpoint is selected by validation BCE; the test split never selects a checkpoint.", "",
             b["label_conversion"] + f". There are {b['observed_label_cells']:,} observed label cells, including {b['all_observed_failed_rows']:,} questions where every observed model failed. Those questions are retained; missing labels are not failures. Training uses independent masked BCE and no CMA refinement.", "",
             f"Training uses AdamW for {training['epochs']} epochs with learning rate {training['learning_rate']}, weight decay 0.0001, seed {training['seed']}, and zero-initialized weights and biases. There is no hyperparameter search; validation BCE selects the saved checkpoint.", "",
             "Sampling inherits the earlier experiment's source-stratified selection and character-length filter. It is not an unbiased draw from all SPROUT prompts or the live subnet pool. The recorded sample indices and exact splits are included in the JSON evaluation report.", "",
             b["token_limitations"], "", b["worker_profile"] + ".", "",
             "Costs below use recorded per-question input/output tokens multiplied by the historical experiment's **assumed** price snapshot. They are not provider-reported actual dollar charges. The complete evaluated snapshot is bundled. Core can explicitly use a different snapshot, which may change selections.", "",
             "| Source extract | SHA256 |", "| --- | --- |"]
    if partial:
        lines[4:4] = [f"**Conclusion:** 512 tokens would truncate {100*partial['would_truncate_at_512']/partial['total_questions']:.1f}% of the selected prompts; 2048 truncates none. The 2048 mechanism evaluation is complete. The 512 head comparison is deferred, so relative routing quality remains unknown.", ""]
    for name, value in b["sources"].items():
        lines.append(f"| {name} | `{value}` |")
    lines += ["", "## Embedding and calibration comparison", "",
              "Full embedding passes run on CPU float32 with the pinned model/tokenizer and exact core prompt. Timing is wall time on this WSL host during development; concurrent checks and memory pressure can affect it. Peak RSS is per-process high-water memory, not GPU memory. These timings are mechanism measurements, not a controlled hardware performance claim.", "",
              "See [execution notes](RUN_NOTES.md) for the discarded memory-limited attempt and the checkpointed rerun. The table below includes completed passes only.", "",
              "| Input limit | Truncated | Total seconds | Seconds/question | Peak RSS MiB | Selected epoch | Pooled test Brier |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, r in reports.items():
        e = r["embedding"]
        lines.append(f"| {n} | {e['truncated']}/{e['n']} ({100*e['truncated']/e['n']:.2f}%) | {e['seconds']:.1f} | {e['seconds_per_question']:.4f} | {e['peak_rss_kib']/1024:.1f} | {r['selected_epoch']} | {r['pooled']['brier']:.6f} |")
    if a:
        delta = b["pooled"]["brier"] - a["pooled"]["brier"]
        lines += ["", f"The 2048-token variant changes pooled test Brier by {delta:+.6f} relative to 512 (lower is better). This result does not change the 2048-token contract default."]
    else:
        lines += ["", "### Limited 512 versus 2048 conclusion", "",
                  f"The user requested wrapping up with available evidence. The 512 pass was stopped after {partial['completed_questions']:,}/{partial['total_questions']:,} questions, processed in ascending token length. No 512 head was trained or evaluated. This is not a representative partial quality comparison.", "",
                  f"Across the full selected sample, {partial['would_truncate_at_512']:,} questions would exceed 512 tokens and {partial['would_truncate_at_2048']:,} would exceed 2048. Among the completed 512 inputs, {partial['completed_inputs_exceeding_512']:,} exceeded 512 tokens. For {partial['untruncated_embedding_comparisons']:,} completed inputs that fit both limits, embeddings agree within 1e-5 (maximum absolute difference {partial['untruncated_max_abs_difference']}).", "",
                  "**The evidence establishes compatible 2048 routing; it does not establish that 2048 outperforms 512.** Full 512 accuracy, calibration, cost, latency and peak-memory comparison remains deferred. The 2048 default is retained as the specified contract, not as an experimentally selected winner. Partial timings are recorded in `partial-512.json` and are not compared with a full pass."]
    lines += ["",
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
        old = a["per_model"][m] if a else None
        lines.append(f"| {m} | {v['n']} | {number(old['auc']) if old else 'not evaluated'} | {number(v['auc'])} | {number(old['brier']) if old else 'not evaluated'} | {v['brier']:.6f} |")
    for n, r in reports.items():
        lines += ["", f"### {n}-token pooled calibration", "", "| Probability bin | Observations | Mean prediction | Observed success |", "| --- | --- | --- | --- |"]
        for v in r["pooled"]["calibration"]:
            lines.append(f"| {v['lo']:.1f}–{v['lo']+.1:.1f} | {v['n']} | {v['mean_prediction']:.6f} | {v['success_rate']:.6f} |")
    lines += ["", "Per-model calibration tables, every λ-specific paired fixed-model contrast, model selection counts, and training histories are retained in the adjacent JSON reports.", "",
              "## Reproduction", "", "Use the exact three source extracts whose hashes appear above, and the pinned local Qwen snapshot. Reuse the existing selected indices; do not resample. The commands below include the optional, deferred full 512 comparison; it is not required to reproduce the limited report from its checked-in JSON results.", "", "```bash",
              "python scripts/evaluate_success_contract.py --data /path/to/experiments \\",
              "  --backbone /path/to/Qwen3-0.6B --output /path/to/results --max-length 2048",
              "python scripts/evaluate_success_contract.py --data /path/to/experiments \\",
              "  --backbone /path/to/Qwen3-0.6B --output /path/to/results --max-length 512",
              "python scripts/render_success_report.py --results /path/to/results \\",
              "  --output /path/to/results/REPORT.md", "```", "",
              "If completed later, experimental 512 artifacts use a distinct, unsupported production contract marker and cannot be loaded as 2048 success heads. The 2048 artifact can be exported offline and loaded in core through FUGAL_HEAD with its evaluated price snapshot. SPROUT token provenance is incomplete, so this bundle is not deployable. The model identifiers come from the historical extract; the bundle does not configure corresponding worker endpoints or establish their availability.", "",
              "No model calls were purchased, no winner was selected, no shipped head was replaced, and no testnet reset or release publication was performed.", ""]
    if partial:
        lines += ["To render this explicitly limited report without resuming the deferred run:", "", "```bash",
                  "python scripts/render_success_report.py --results docs/evaluation/success-contract \\",
                  "  --partial-512 docs/evaluation/success-contract/partial-512.json \\",
                  "  --output /tmp/success-report.md", "```", ""]
    Path(args.output).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
