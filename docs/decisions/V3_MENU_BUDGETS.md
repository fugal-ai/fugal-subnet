# v3 menu: answer budgets, call deadlines, token statistics, exploration cost

2026-11-16. What protocol v3 pins about each menu entry beyond its price, and
why. The menu itself (23 models, 53 `model@effort` entries) was approved by
the owner the same day; its pin comment is in `scripts/check_safety_invariants.py`.

## First principle

The router's odds must predict what a user of the routed product would get.
Nobody cuts a model off at 16,384 tokens in production, so a harness cutoff
teaches heads that slow, careful entries fail, on exactly the hard questions
where they are worth paying for. Time is the user's trade-off, not ours: it is
measured (and published) rather than capped, except where the protocol's own
clock forces a cap. What actually limits the measurement is its cost.

## Answer budgets (`max_tokens` per entry, `data/models_v3.json`)

`min(131072, the smallest answer limit of the entry's pinned hosts)`, from
OpenRouter's `/endpoints` of 2026-11-16:

| Budget | Entries |
|---|---|
| 128,000 | every OpenAI and Anthropic entry (their hosts' limit) |
| 131,072 | Gemini 4 Argon, Muse Ember/Spark/Wonder, DeepSeek, GLM-5.3 Flash, Kimi K3.1 Pro, Qwen4 Max, MiMo, Gemma 4 |
| 117,964 | Granite 4.2 8B |
| 65,536 | Gemini 3.8 Flash; Gemini 4 Carbon (pinned Vertex global + us: Google AI Studio capped it at 32,768) |

**Provisional.** The 128k measurement of 2026-11-16 (each model's highest
effort on one hard question) decides the final budgets; the hash is re-pinned
then (`TODO(v3 release)` in the pin comment).

## Call deadlines (`config.call_deadline_s`)

| Effort | Deadline (proxy, wall clock) |
|---|---|
| low, minimal | 600 s |
| medium, none (thinking on, or no reasoning) | 900 s |
| high, xhigh, max | 1,800 s |

The harness waits `HARNESS_WAIT_MARGIN_S` (30 s) longer. v1 and v2 calls keep
`PROXY_CALL_DEADLINE_S` (420 s) byte for byte. The live check measured
100-300 s for high/max entries at a 16k budget and one low-effort cheap
reasoner at 1,117 s.

**The ceiling is the collection point.** Validators collect at
`EPOCH_COLLECT_FRACTION` (0.5) of a 72-minute epoch: 2,160 s after the
boundary, and the proof must be built and attested before then. 1,800 s
leaves ~6 min.

**Throughput.** A miner makes 300 scored calls (the router's choices) and 32
exploration calls (drawn), `HARNESS_CONCURRENCY` (24) at a time. Its epoch
takes about max(longest call, total call seconds / concurrency):

| Case | Total call seconds | At 24 in flight | Fits 2,160 s? |
|---|---|---|---|
| Expected: scored mean 60 s; exploration 8 calls at 1,800 s, 24 at 120 s | 18,000 + 14,400 + 2,880 = 35,280 | 1,470 s | yes |
| Worst exploration: scored mean 60 s; all 32 exploration calls at 1,800 s | 18,000 + 57,600 = 75,600 | 3,150 s | **no** |

The worst case needs 75,600 / 2,000 ≈ **38 in flight** (2,000 s leaves time to
build the proof). Scored calls are the router's own choice: a router that
picks slow entries for many questions risks its proof, which is a real cost
it learns to weigh. Exploration calls are not chosen, so their worst case is
the miner's exposure; cost-aware sampling (below) makes the slow, expensive
entries rarer, but cheap reasoners can be slow too.

## Token statistics (`benchmark_tokens_protocol_v3.json`)

The cost side of every routing decision. **No paid calls** built it
(`scripts/build_token_manifest_v3.py`):

1. **measured** (4 rows): v2 measured the same model at the same reasoning
   setting; its means are reused.
2. **scaled** (36 rows): v2 measured the model, or a named predecessor or
   sibling, at another setting; prompt mean as is, completion mean × an effort
   factor (low 0.5, medium 1, high 2, xhigh 3, max 4; v2's `max_tokens: 8192`
   and `enabled: true` read as medium), capped at the entry's budget.
3. **default** (13 rows): no v2 model to lean on; the median v2 means, ×
   the effort factor.

Every row but rule 1 is marked `provisional`. Status `candidate`; the release
review promotes it.

**Reuse policy (owner's rule, 2026-11-16): never pay twice for the same
measurement.** v3 exploration publishes, per pooled pair, the attested
prompt and completion tokens of real calls on the real pool at the real
budgets (`single_model.single_model_table` aggregates them per entry). At
each monthly menu update, provisional rows are replaced by those free
medians; a NEW entry gets a small paid sample (about 10 calls) appended;
rows already measured are never re-measured. Paid results are saved under
`~/projects/fugal-v2-handoff/v3-prep/measurements/`.

## Cost-aware exploration (`fugal_subnet/explore_weights.py`)

Design and the unbiasedness argument: INVARIANTS, "I4 — every miner measures
the models (v3)". The numbers at the provisional token means (which are LOW
for long reasoners: they were measured at a 16k budget):

| | Uniform | Cost-aware (cap $0.20, floor 1/4 → exponent 1/2) |
|---|---|---|
| Expected spend per miner per epoch | $0.316 | $0.163 |
| p99 / max over 20,000 simulated epochs | $0.431 / $0.464 | $0.288 / $0.358 |
| Per miner per day (20 epochs) | $6.32 | $3.25 |
| Design effect of the weights | 1.00 | 1.32 |
| Weight range | 1 | 0.33 to 2.68 |
| Chance per call: cheapest / dearest entry | 1.9% / 1.9% | 5.7% / 0.70% |

As measured token means replace the provisional ones, the dearest entries'
costs rise and the cap selects a steeper exponent automatically (up to 2,
design effect ≤ ~3 at floor 1/4). The floor bounds that: the dearest entry
keeps ≥ 0.47% of calls, so if Fable 5.2 @max really costs ~$1 a call, the
floor alone costs a miner ~$0.15 an epoch, and the cap would then bind at
the steepest exponent. Revisit the cap and floor with the measured means.

## Open questions

- The worst-case exploration epoch needs ~38 calls in flight; the default is
  24 (raised from 8 for v2, bounded by OpenRouter's in-flight budget). Either
  raise `HARNESS_CONCURRENCY` for v3 miners, shorten the high-effort deadline,
  or let the harness ship the proof at a fixed cutoff with unfinished calls
  marked failed (bounded by `EXPLORE_FAIL_MARGIN` today).
- A later collection point (`EPOCH_COLLECT_FRACTION` 0.75 = 54 min) would
  allow 45-minute deadlines, at the cost of the validators' post-collection
  budget (collateral fetches, scoring, weights, reveal).
- The validator's live startup checks require a `reviewed` token manifest
  for v1 and v2 only; v3's is `candidate` until the release review.
