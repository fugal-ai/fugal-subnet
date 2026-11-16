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
exploration calls (drawn), `HARNESS_CONCURRENCY` (40 from v3; 24 in v2) at a time. Its epoch
takes about max(longest call, total call seconds / concurrency):

| Case | Total call seconds | At 24 in flight | At 40 (v3) | Fits 2,160 s at 40? |
|---|---|---|---|---|
| Expected: scored mean 60 s; exploration 8 calls at 1,800 s, 24 at 120 s | 18,000 + 14,400 + 2,880 = 35,280 | 1,470 s | 882 s | yes |
| Worst exploration: scored mean 60 s; all 32 exploration calls at 1,800 s | 18,000 + 57,600 = 75,600 | 3,150 s | 1,890 s | yes |

The worst case needs 75,600 / 2,000 ≈ **38 in flight** (2,000 s leaves time to
build the proof). Scored calls are the router's own choice: a router that
picks slow entries for many questions risks its proof, which is a real cost
it learns to weigh. Exploration calls are not chosen, so their worst case is
the miner's exposure; cost-aware sampling (below) makes the slow, expensive
entries rarer, but cheap reasoners can be slow too.

## Token statistics (`benchmark_tokens_protocol_v3.json`)

The cost side of every routing decision (`scripts/build_token_manifest_v3.py`,
rules in order):

0. **measured in v3** (49 rows): 10 calls per entry on the same 10 non-code
   v3 questions (category mix of the v3 targets without code), through the
   v3 harness path (shared system prompt, the entry's own budget, its pinned
   hosts), 2026-11-16, $3.86 in all
   (`scripts/record_token_observations.py --protocol 3`; rows in
   `~/projects/fugal-v2-handoff/v3-prep/measurements/tokens-2026-11-16.jsonl`).
   No call used its whole budget; median latency 2-29 s per entry, the
   slowest single call 705 s (gpt-6.1-terra@max).
1. **measured in v2** (3 rows): terra@medium, sol@medium, mimo-v2.6-flash,
   kept because the v3 sample agrees within 2x. mimo-v2.6-pro's v2 mean was
   5.8x the v3 sample's, so v3 replaced it.
2. **scaled** (0 rows) and 3. **default** (1 row: gemma-4-31b, 9 of its 10
   calls landed; one question's upstream kept returning 502): the v2-based
   estimates, marked `provisional`.

Effort does not always do what its name says: glm-5.3-flash@low answered
longer than @high on this sample, and qwen4-max@low longer than @high. Ten
questions is a small sample and long answers are heavy-tailed (one hard
question took 54k tokens at a 128k budget on 2026-11-16), so these means are
a floor for the hard categories; code questions are not in the sample.
Status `candidate`; the release review promotes it.

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
the models (v3)". At the measured token means of 2026-11-16:

| | Uniform | Cost-aware (cap $0.20, floor 1/4 → exponent 1/2) |
|---|---|---|
| Expected spend per miner per epoch | $0.233 | $0.117 |
| Per miner per day (20 epochs) | $4.66 | $2.34 |
| Weight range | 1 | 0.36 to 2.87 |
| Chance per call: cheapest / dearest entry | 1.9% / 1.9% | 5.3% / 0.66% |

The dearest entry per call is gemini-4-carbon@high ($0.075 a question). The
cap selects the same exponent as at the provisional means, so the release
constants stand. As exploration's attested means replace these (code
questions included), the cap selects a steeper exponent automatically if
the costs rise; the floor keeps every entry measured.

## Open questions

- Settled 2026-11-16: `HARNESS_CONCURRENCY` is 40 from v3, which fits the
  worst-case exploration epoch (1,890 s < 2,160 s). Shortening deadlines or
  shipping at a cutoff was rejected: both would drop exactly the slow, long
  answers the router must learn the true odds of.
- A later collection point (`EPOCH_COLLECT_FRACTION` 0.75 = 54 min) would
  allow 45-minute deadlines, at the cost of the validators' post-collection
  budget (collateral fetches, scoring, weights, reveal).
- The validator's live startup checks require a `reviewed` token manifest
  for v1 and v2 only; v3's is `candidate` until the release review.
