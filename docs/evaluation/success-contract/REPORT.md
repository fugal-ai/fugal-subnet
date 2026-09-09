# Offline success-contract mechanism report

This is a SPROUT mechanism test on historical models and recorded outcomes. It does not establish live-subnet calibration or nominate a release head. The production contract remains 2048 tokens regardless of this comparison.

**Conclusion:** 512 tokens would truncate 14.1% of the selected prompts; 2048 truncates none. The 2048 mechanism evaluation is complete. The 512 head comparison is deferred, so relative routing quality remains unknown.

## Data and method

The existing selected sample contains 10,000 questions and 13 model columns. Exact duplicate prompts stay together in deterministic 60/20/20 groups (seed 42), yielding train: 5,995, validation: 2,002, test: 2,003. Completed heads use these same sample indices and splits. Each checkpoint is selected by validation BCE; the test split never selects a checkpoint.

finite score >= 1 => 1; lower => 0; NaN remains missing. There are 130,000 observed label cells, including 615 questions where every observed model failed. Those questions are retained; missing labels are not failures. Training uses independent masked BCE and no CMA refinement.

Training uses AdamW for 200 epochs with learning rate 0.01, weight decay 0.0001, seed 42, and zero-initialized weights and biases. There is no hyperparameter search; validation BCE selects the saved checkpoint.

Sampling inherits the earlier experiment's source-stratified selection and character-length filter. It is not an unbiased draw from all SPROUT prompts or the live subnet pool. The recorded sample indices and exact splits are included in the JSON evaluation report.

Original importer replaced missing token counts with zero. Positive input only used for manifest. Output zero may be missing. No deployable export.

SPROUT historical provider requests; exact generation settings unavailable in local extract.

Costs below use recorded per-question input/output tokens multiplied by the historical experiment's **assumed** price snapshot. They are not provider-reported actual dollar charges. The complete evaluated snapshot is bundled. Core can explicitly use a different snapshot, which may change selections.

| Source extract | SHA256 |
| --- | --- |
| sprout_matrix.npz | `85f5e1dd864de2b4af01801bbee54379cd10464f9ca383fbe615c62c82af20cf` |
| sprout_prompts.json | `f2e54bf8fca58610026f4aa47f908b0ae5cf15902220f80584876f0e347b0ab9` |
| sprout_embed_index.json | `06716cc881a3dc209281c6eb9c0ab32354a95e3a90f926630326187d615eee19` |

## Embedding and calibration comparison

Full embedding passes run on CPU float32 with the pinned model/tokenizer and exact core prompt. Timing is wall time on this WSL host during development; concurrent checks and memory pressure can affect it. Peak RSS is per-process high-water memory, not GPU memory. These timings are mechanism measurements, not a controlled hardware performance claim.

See [execution notes](RUN_NOTES.md) for the discarded memory-limited attempt and the checkpointed rerun. The table below includes completed passes only.

| Input limit | Truncated | Total seconds | Seconds/question | Peak RSS MiB | Selected epoch | Pooled test Brier |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 0/10000 (0.00%) | 8466.7 | 0.8467 | 4661.1 | 200 | 0.197961 |

### Limited 512 versus 2048 conclusion

The user requested wrapping up with available evidence. The 512 pass was stopped after 6,400/10,000 questions, processed in ascending token length. No 512 head was trained or evaluated. This is not a representative partial quality comparison.

Across the full selected sample, 1,410 questions would exceed 512 tokens and 0 would exceed 2048. Among the completed 512 inputs, 0 exceeded 512 tokens. For 6,400 completed inputs that fit both limits, embeddings agree within 1e-5 (maximum absolute difference 0.0).

**The evidence establishes compatible 2048 routing; it does not establish that 2048 outperforms 512.** Full 512 accuracy, calibration, cost, latency and peak-memory comparison remains deferred. The 2048 default is retained as the specified contract, not as an experimentally selected winner. Partial timings are recorded in `partial-512.json` and are not compared with a full pass.

## Held-out routing

Each cell is a mean with a paired 95% bootstrap interval (2,000 question resamples, seed 7). Bootstrap draws are shared across policies. Intervals condition on this selected sample and the fitted head; they do not include retraining or dataset-selection uncertainty.

| Input limit | λ | Accuracy % [95% CI] | Recorded-token cost $/1,000 queries [95% CI] |
| --- | --- | --- | --- |
| 2048 | 0 | 75.6865 [73.6395, 77.5349] | 3.2026 [3.0218, 3.4122] |
| 2048 | 0.5 | 75.7863 [73.7881, 77.6835] | 3.0943 [2.9169, 3.3090] |
| 2048 | 1 | 75.6365 [73.5896, 77.5337] | 2.9567 [2.7766, 3.1656] |
| 2048 | 2 | 75.4368 [73.4398, 77.3340] | 2.6776 [2.4983, 2.8856] |
| 2048 | 5 | 74.6380 [72.6897, 76.5364] | 2.0517 [1.8773, 2.2439] |

## Fixed-model and cheapest baselines

Cheapest by train-split estimated query cost: `wxai-llama-3-2-1b-instruct`. Fixed-model baselines use the same held-out outcomes and actual recorded tokens as routing.

| Model | Accuracy % [95% CI] | Cost $/1,000 queries [95% CI] | 2048 λ=1 accuracy minus fixed, percentage points [95% CI] |
| --- | --- | --- | --- |
| aws-claude-3-5-sonnet-v1 | 70.8437 [68.8467, 72.7908] | 7.0104 [6.8568, 7.1678] | 4.7928 [2.8457, 6.4903] |
| aws-titan-text-premier-v1 | 35.5467 [33.4498, 37.6935] | 0.5228 [0.5051, 0.5408] | 40.0899 [37.5936, 42.5362] |
| openai-gpt-4o | 72.5911 [70.6940, 74.5394] | 4.6071 [4.3592, 4.8853] | 3.0454 [1.5477, 4.4433] |
| openai-gpt-4o-mini | 69.5956 [67.5986, 71.5427] | 0.3232 [0.3105, 0.3379] | 6.0409 [4.2436, 7.8382] |
| wxai-granite-3-2b-instruct-8k-max-tokens | 40.9386 [38.6920, 42.9868] | 0.0827 [0.0788, 0.0872] | 34.6980 [32.3015, 37.1942] |
| wxai-granite-3-8b-instruct-8k-max-tokens | 48.5771 [46.3804, 50.6740] | 0.1513 [0.1453, 0.1582] | 27.0594 [24.7616, 29.4558] |
| wxai-llama-3-1-70b-instruct | 63.8542 [61.7074, 65.9523] | 0.4960 [0.4589, 0.5355] | 11.7823 [9.6855, 13.7793] |
| wxai-llama-3-1-8b-instruct | 49.4758 [47.3789, 51.6226] | 0.0683 [0.0632, 0.0739] | 26.1608 [23.8143, 28.3575] |
| wxai-llama-3-2-1b-instruct | 27.5087 [25.5105, 29.5557] | 0.0211 [0.0196, 0.0228] | 48.1278 [45.7314, 50.4743] |
| wxai-llama-3-2-3b-instruct | 42.8357 [40.7376, 45.0824] | 0.0323 [0.0299, 0.0350] | 32.8008 [30.3045, 35.3470] |
| wxai-llama-3-3-70b-instruct | 57.9131 [55.7664, 60.0599] | 0.3716 [0.3596, 0.3845] | 17.7234 [15.6752, 19.9201] |
| wxai-llama-3-405b-instruct | 55.5667 [53.4199, 57.7634] | 2.0172 [1.9258, 2.1210] | 20.0699 [17.8732, 22.2167] |
| wxai-mixtral-8x7b-instruct-v01 | 42.0369 [39.8402, 44.1837] | 0.1793 [0.1742, 0.1848] | 33.5996 [31.2032, 36.0459] |

## Fixed-mixture frontier

Frontier endpoints are chosen from validation-set fixed-model cost/accuracy, then evaluated on test. Mixtures are expected outcomes of a question-independent randomized policy; the probability of the right endpoint is shown. These are offline baselines, not changes to live frontier rewards.

| Left model | Right model | Right probability | Accuracy % [95% CI] | Cost $/1,000 queries [95% CI] |
| --- | --- | --- | --- | --- |
| wxai-llama-3-2-1b-instruct | wxai-llama-3-2-3b-instruct | 0.00 | 27.5087 [25.5105, 29.5557] | 0.0211 [0.0196, 0.0228] |
| wxai-llama-3-2-1b-instruct | wxai-llama-3-2-3b-instruct | 0.25 | 31.3405 [29.5057, 33.1631] | 0.0239 [0.0225, 0.0256] |
| wxai-llama-3-2-1b-instruct | wxai-llama-3-2-3b-instruct | 0.50 | 35.1722 [33.3749, 37.0444] | 0.0267 [0.0251, 0.0285] |
| wxai-llama-3-2-1b-instruct | wxai-llama-3-2-3b-instruct | 0.75 | 39.0040 [37.0816, 41.0135] | 0.0295 [0.0275, 0.0317] |
| wxai-llama-3-2-1b-instruct | wxai-llama-3-2-3b-instruct | 1.00 | 42.8357 [40.7376, 45.0824] | 0.0323 [0.0299, 0.0350] |
| wxai-llama-3-2-3b-instruct | wxai-llama-3-1-8b-instruct | 0.00 | 42.8357 [40.7376, 45.0824] | 0.0323 [0.0299, 0.0350] |
| wxai-llama-3-2-3b-instruct | wxai-llama-3-1-8b-instruct | 0.25 | 44.4958 [42.4984, 46.5926] | 0.0413 [0.0387, 0.0441] |
| wxai-llama-3-2-3b-instruct | wxai-llama-3-1-8b-instruct | 0.50 | 46.1558 [44.2336, 48.0785] | 0.0503 [0.0471, 0.0539] |
| wxai-llama-3-2-3b-instruct | wxai-llama-3-1-8b-instruct | 0.75 | 47.8158 [45.8309, 49.7507] | 0.0593 [0.0552, 0.0638] |
| wxai-llama-3-2-3b-instruct | wxai-llama-3-1-8b-instruct | 1.00 | 49.4758 [47.3789, 51.6226] | 0.0683 [0.0632, 0.0739] |
| wxai-llama-3-1-8b-instruct | openai-gpt-4o-mini | 0.00 | 49.4758 [47.3789, 51.6226] | 0.0683 [0.0632, 0.0739] |
| wxai-llama-3-1-8b-instruct | openai-gpt-4o-mini | 0.25 | 54.5057 [52.6457, 56.3034] | 0.1320 [0.1264, 0.1380] |
| wxai-llama-3-1-8b-instruct | openai-gpt-4o-mini | 0.50 | 59.5357 [57.7883, 61.2581] | 0.1957 [0.1885, 0.2041] |
| wxai-llama-3-1-8b-instruct | openai-gpt-4o-mini | 0.75 | 64.5657 [62.7309, 66.3633] | 0.2595 [0.2496, 0.2708] |
| wxai-llama-3-1-8b-instruct | openai-gpt-4o-mini | 1.00 | 69.5956 [67.5986, 71.5427] | 0.3232 [0.3105, 0.3379] |
| openai-gpt-4o-mini | openai-gpt-4o | 0.00 | 69.5956 [67.5986, 71.5427] | 0.3232 [0.3105, 0.3379] |
| openai-gpt-4o-mini | openai-gpt-4o | 0.25 | 70.3445 [68.4598, 72.1923] | 1.3942 [1.3278, 1.4677] |
| openai-gpt-4o-mini | openai-gpt-4o | 0.50 | 71.0934 [69.3460, 72.9156] | 2.4651 [2.3388, 2.6065] |
| openai-gpt-4o-mini | openai-gpt-4o | 0.75 | 71.8422 [70.0696, 73.6274] | 3.5361 [3.3495, 3.7466] |
| openai-gpt-4o-mini | openai-gpt-4o | 1.00 | 72.5911 [70.6940, 74.5394] | 4.6071 [4.3592, 4.8853] |

## Per-model discrimination and calibration

AUC is undefined when a held-out model column has only one label class. Brier and calibration bins use observed labels only.

| Model | Observed test labels | 512 AUC | 2048 AUC | 512 Brier | 2048 Brier |
| --- | --- | --- | --- | --- | --- |
| aws-claude-3-5-sonnet-v1 | 2003 | not evaluated | 0.704822 | not evaluated | 0.187079 |
| aws-titan-text-premier-v1 | 2003 | not evaluated | 0.713188 | not evaluated | 0.204390 |
| openai-gpt-4o | 2003 | not evaluated | 0.686629 | not evaluated | 0.183658 |
| openai-gpt-4o-mini | 2003 | not evaluated | 0.694371 | not evaluated | 0.192755 |
| wxai-granite-3-2b-instruct-8k-max-tokens | 2003 | not evaluated | 0.715989 | not evaluated | 0.210664 |
| wxai-granite-3-8b-instruct-8k-max-tokens | 2003 | not evaluated | 0.722693 | not evaluated | 0.214121 |
| wxai-llama-3-1-70b-instruct | 2003 | not evaluated | 0.711888 | not evaluated | 0.203210 |
| wxai-llama-3-1-8b-instruct | 2003 | not evaluated | 0.711646 | not evaluated | 0.216552 |
| wxai-llama-3-2-1b-instruct | 2003 | not evaluated | 0.729654 | not evaluated | 0.174909 |
| wxai-llama-3-2-3b-instruct | 2003 | not evaluated | 0.742685 | not evaluated | 0.203108 |
| wxai-llama-3-3-70b-instruct | 2003 | not evaluated | 0.784966 | not evaluated | 0.182323 |
| wxai-llama-3-405b-instruct | 2003 | not evaluated | 0.765988 | not evaluated | 0.188028 |
| wxai-mixtral-8x7b-instruct-v01 | 2003 | not evaluated | 0.713348 | not evaluated | 0.212701 |

### 2048-token pooled calibration

| Probability bin | Observations | Mean prediction | Observed success |
| --- | --- | --- | --- |
| 0.0–0.1 | 196 | 0.079619 | 0.066327 |
| 0.1–0.2 | 1558 | 0.160347 | 0.121309 |
| 0.2–0.3 | 3038 | 0.252819 | 0.219223 |
| 0.3–0.4 | 3471 | 0.351070 | 0.321809 |
| 0.4–0.5 | 3804 | 0.449331 | 0.459253 |
| 0.5–0.6 | 3506 | 0.550795 | 0.566743 |
| 0.6–0.7 | 3814 | 0.651592 | 0.662821 |
| 0.7–0.8 | 4663 | 0.751073 | 0.769033 |
| 0.8–0.9 | 1976 | 0.830891 | 0.870951 |
| 0.9–1.0 | 13 | 0.918817 | 0.923077 |

Per-model calibration tables, every λ-specific paired fixed-model contrast, model selection counts, and training histories are retained in the adjacent JSON reports.

## Reproduction

Use the exact three source extracts whose hashes appear above, and the pinned local Qwen snapshot. Reuse the existing selected indices; do not resample. The commands below include the optional, deferred full 512 comparison; it is not required to reproduce the limited report from its checked-in JSON results.

```bash
python scripts/evaluate_success_contract.py --data /path/to/experiments \
  --backbone /path/to/Qwen3-0.6B --output /path/to/results --max-length 2048
python scripts/evaluate_success_contract.py --data /path/to/experiments \
  --backbone /path/to/Qwen3-0.6B --output /path/to/results --max-length 512
python scripts/render_success_report.py --results /path/to/results \
  --output /path/to/results/REPORT.md
```

If completed later, experimental 512 artifacts use a distinct, unsupported production contract marker and cannot be loaded as 2048 success heads. The 2048 artifact can be exported offline and loaded in core through FUGAL_HEAD with its evaluated price snapshot. SPROUT token provenance is incomplete, so this bundle is not deployable. The model identifiers come from the historical extract; the bundle does not configure corresponding worker endpoints or establish their availability.

No model calls were purchased, no winner was selected, no shipped head was replaced, and no testnet reset or release publication was performed.

To render this explicitly limited report without resuming the deferred run:

```bash
python scripts/render_success_report.py --results docs/evaluation/success-contract \
  --partial-512 docs/evaluation/success-contract/partial-512.json \
  --output /tmp/success-report.md
```
