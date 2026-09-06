# Does the subnet measure anything? Two experiments

The subnet's premise is that a learned routing head sends a question to a
better model than chance does, and that a miner cannot satisfy the scoring
function without actually doing it. Neither had been tested. This is the test.

**Every number below is labelled by how it was obtained** — *measured*, *read*,
or *assumed*. This project has been burned repeatedly by prose that blurs the
three, so the labels are not decoration.

Reproduce everything:

```bash
python scripts/experiment_routing_dataset.py fetch
python scripts/experiment_routing_dataset.py embed --limit 10000 --device cuda --batch-size 8
python scripts/experiment_head_efficacy.py --json results/head_efficacy.json
python scripts/experiment_pool_memorization.py embed-pool --device cuda --batch-size 8 --limit 12000
python scripts/experiment_pool_memorization.py capacity --json results/pool_capacity.json
python scripts/experiment_pool_memorization.py gap --json results/pool_gap.json
```

No paid API call happens in any of it.

---

## The two answers, first

**1. A trained head does beat chance — and loses to a head that ignores the
question.** *(measured)* The shipped trainer, over the shipped routing rule, on
5,000 held-out questions: score **1.057** against a random head's 0.868 and
chance's 0.815. It beats the best single model (0.985). It also beats a
per-benchmark lookup table (1.036), so it is genuinely routing rather than
recognising domains. But **"always route to `openai-gpt-4o-mini`" scores
1.183** — 12% better than the best head I could train, from a head with `W = 0`
and a one-hot bias. Adding the shipped sep-CMA-ES stage moves the trained head
to 1.060. It does not close the gap.

**2. A miner cannot memorise the pool — because the pool is big enough, and
nothing says so.** *(measured)* The head is a linear map over 1,024 dimensions.
On the most favourable geometry that exists it fits arbitrary labels for ~2,000
questions perfectly and only 16.9% of them at 21,717. So the pool being public
and finite is survivable at its current size. At a tenth of that size it would
not be, and **no check anywhere ties the pool's size to that fact.**

The second finding is reassuring. The first is not, and it is the one that
should change what happens next.

---

## Why this could not be answered from this repository

*(read)* A TEE proof records the one model a miner routed to and what happened
to it. It does not record what the other twelve models would have done, because
the miner never called them — that missing counterfactual is the entire reason
`exploration.py` exists. So there is no question-by-model correctness matrix in
`results/`, and ordinary operation will never produce one.

Building one in-house means calling every model on every question. Priced from
the pinned table at `FRAME_DEFAULT_COMPLETION_TOKENS = 256` over the real
21,717-question pool, that is **$688 for all 17 models**, $40 per model
*(computed, via `pricing.question_input_tokens`)*. Not run, not proposed.

### The substitute, and its limits

`CARROT-LLM-Routing/SPROUT` on HuggingFace — free, parquet, no API key. 13
hosted LLMs scored on 44,241 real prompts, with per-model correctness *and*
per-model input/output token counts. Fetched with parquet column pruning, so
only the columns needed crossed the wire rather than the full 580 MB.

What it is **not**, stated plainly because the temptation to over-read it is
strong:

- **Not this subnet's pool, models, or graders.** SPROUT's labels come from an
  LLM judge scoring 0–1; `graders.py` is mechanical and binary. Binarised here
  at 1.0, the strictest threshold, because a mechanical verdict is binary by
  construction.
- **Not a prediction of miner scores.** These results are evidence about the
  *mechanism* — can a linear head over a frozen 0.6B embedding learn to route —
  not about what a Fugal miner will earn.
- **`acc_best` here is a raw test-split rate, not the frame's posterior.** The
  live `reference_frame` shrinks toward `FRAME_PRIOR_ACCURACY`, so real miner
  scores will run slightly *higher* than these for identical routing.

Prompts over 5,000 characters were dropped before sampling *(the real pool's
longest is 4,891; its mean is 396)*. SPROUT carries RAG contexts up to 87,000
characters, the backbone truncates at 512 tokens, and keeping them would have
measured routing on a question population this subnet does not have.

---

## Experiment 1 — does a trained head beat a random one?

10,000 questions sampled stratified by source, split 5,000 train / 5,000 test.
Every strategy is fit on train and reported on test. Routing decisions come
from `head_eval.evaluate_head`, the same rule `tee/harness._route_question`
applies — softmax, quantise, argmax.

Reference model: `openai-gpt-4o`, test accuracy 0.734 *(measured)*.

| strategy | accuracy | $/question | thrift | **score** | models | top share | in-source share | bits |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.937 | 0.000874 | 5.41 | **1.464** | 13 | 0.24 | 0.31 | 3.09 |
| **best_constant** | 0.698 | 0.000322 | 10.00 | **1.183** | 1 | 1.00 | 1.00 | 0.00 |
| trained_sft_cma | 0.737 | 0.002332 | 2.03 | **1.060** | 12 | 0.41 | 0.52 | 2.07 |
| trained_sft | 0.755 | 0.002994 | 1.58 | **1.057** | 13 | 0.30 | 0.45 | 2.39 |
| domain_oracle | 0.758 | 0.003821 | 1.24 | **1.036** | 4 | 0.54 | 1.00 | 1.66 |
| best_single | 0.734 | 0.004731 | 1.00 | **0.985** | 1 | 1.00 | 1.00 | 0.00 |
| random_head | 0.552 | 0.001639 | 6.00 | **0.868** | 8.6 | 0.63 | — | 1.47 |
| trained_ce | 0.475 | 0.000220 | 10.00 | **0.828** | 13 | 0.29 | 0.40 | 2.82 |
| random_routing | 0.518 | 0.001242 | 3.82 | **0.815** | 13 | 0.08 | — | 3.70 |
| cheapest | 0.263 | 0.000022 | 10.00 | **0.479** | 1 | 1.00 | 1.00 | 0.00 |

All *measured*. `random_head` and `random_routing` are means over 20 seeds
(accuracy sd 0.099 and 0.005).

### The premise holds

The trained head beats a random head by **+0.189** and chance by **+0.242**,
both far outside the seed spread. That is the founding claim, and it survives.

It is also *really routing*, which is a separate claim and needed its own
check. It uses all 13 models, sends at most 30% to any one of them, and — the
diagnostic that matters — its **within-source** top-model share is 0.45. A head
that had only learned to recognise the benchmark would score 1.00 there, the
way `domain_oracle` does by construction. The head varies its choice *inside*
each domain, and it beats `domain_oracle` on the composite.

The gains concentrate exactly where they should *(measured, per source)*:

| source | n | best_single | trained_sft | domain_oracle | oracle |
|---|---|---|---|---|---|
| `lighteval/MATH/all` | 790 | 0.775 | **0.895** | 0.908 | 0.984 |
| `ragbench` | 642 | 0.783 | **0.826** | 0.840 | 0.980 |
| `openhermes/teknium` | 1619 | 0.704 | **0.723** | 0.704 | 0.887 |
| `TIGER-Lab/MMLU-Pro` | 1084 | 0.780 | 0.759 | 0.780 | 0.952 |
| `Idavidrein/gpqa` | 289 | 0.516 | 0.526 | 0.495 | 0.952 |

On MATH the head reaches 0.895 against the best single model's 0.775, and it
gets there by routing to `wxai-llama-3-3-70b-instruct`, which is **12× cheaper
than `gpt-4o`** and better at maths. That is precisely the product the subnet
exists to discover, and the head found it without being told.

### And the scoring function does not reward it

`scoring.py`'s docstring is explicit about why the score is a geometric mean:

> each degenerate strategy scores well on one axis — route everything to the
> cheapest model, or everything to the best — and collects that term's weight
> regardless of the other. Under a product neither axis can rescue the other:
> both degenerate strategies score badly.

**Measured, that argument is half right.** Both degenerates it *names* do score
badly: `cheapest` 0.479, `best_single` 0.985. But the argument only covers the
two *ends* of the price range, and the winning degenerate is in the middle.
`openai-gpt-4o-mini` gets 0.698 against the reference's 0.734 — 95% of the
quality — at **1/15th the price**:

```
quality = wilson_lcb(0.698) / 0.734 = 0.933      0.933^0.9 = 0.939
thrift  = 0.004731 / 0.000322 = 14.7, capped     10.00^0.1 = 1.259
score   = 0.939 * 1.259 = 1.183
```

The thrift cap is already binding and still not enough. `SCORE_QUALITY_EXPONENT
= 0.9` is *derived* in `config.py` from the requirement that "a router that
gives up 40% of quality must not outscore a quality match". This router gives
up **5%** and wins. The derivation constrains the wrong corner of the space.

**This is an I3 problem.** "A miner cannot raise its score except by routing
better or more cheaply" holds only in the letter: the highest-scoring strategy
available today is a head with `W = 0` that never reads the question, and every
honest router is 12% behind it.

### It is not an artefact of SPROUT's model list

The break-even is a property of the scoring constants, so it can be computed on
`data/models.json` directly with **no accuracy data at all** *(computed)*.
Ignoring the Wilson penalty, a constant policy on model *k* beats the reference
when

```
acc_k / acc_best  >  thrift ^ (-(1-w)/w)
```

At `w = 0.9` that threshold is astonishingly low:

| the model is… | …and it beats the reference at only |
|---|---|
| 2× cheaper | 92.6% of the reference's accuracy |
| 4× cheaper | 85.7% |
| 6× cheaper | 81.9% |
| 10× cheaper (the thrift cap) | **77.4%** |

To beat the best trained head measured above (score 1.057), a model 10× cheaper
needs **82.3%** of the reference's accuracy.

On the pinned table this is not hypothetical. Costed at
`FRAME_DEFAULT_COMPLETION_TOKENS` over the pool's measured 99.1 input tokens
per question, the 17 models span **168×**, from `deepseek/deepseek-v4-flash` at
$0.000049 to `openai/gpt-5.5` at $0.008176 — and **8 of the 17 sit at or beyond
the 10× thrift cap** relative to `gpt-5.5`. If the reference is
`gpt-5.5` or `claude-opus-4.8`, eight different constant policies are each one
`W = 0` head away from the maximum thrift term.

The single remaining unknown is whether any of those eight reaches 82.3% of the
frontier model's accuracy on this pool. That is exactly the matrix nobody has
built *(the $688 above)*. For calibration, the equivalent ratio measured on
SPROUT was **95%** — `gpt-4o-mini` against `gpt-4o` — which is 13 points of
headroom above what the exploit needs.

Three things worth separating before anyone reaches for a fix:

- It is **tuning, not architecture.** The exponent, the thrift cap and the
  reference choice are all constants. Nothing about the TEE design or the head
  format causes this.
- Any fix is a **consensus change** and needs an INVARIANTS entry plus a check.
  The obvious candidates — raising `w`, lowering `SCORE_THRIFT_CAP`, or making
  the reference the best *scoring* constant rather than the most *accurate*
  model — all re-score every miner.
- The check that would have caught it is cheap and does not exist: **evaluate
  every constant policy and assert that a trained head beats all of them.**
  That is 13 evaluations. A test that asserts `max(score(constant_k)) <
  score(trained_head)` on a fixture would have failed the day the exponent
  moved to 0.9.

### One methodological trap, recorded because I fell into it

`trained_ce` scores 0.828 — worse than a random head. It is the same
architecture trained *harder*, by cross-entropy against the cheapest-correct
model. That label is dominated by cheap models (`llama-3-2-1b` is the
cheapest-correct answer for 27% of questions), so the objective taught the head
to route cheap and be wrong. **The label choice mattered more than the
optimiser.** A miner tuning against the wrong target will underperform a random
head while believing it is training properly, and `train_head.py`'s soft-target
default is doing real work that is nowhere documented as load-bearing.

---

## Experiment 2 — can a miner memorise the pool?

### The structural fact first

*(read)* Every question a miner is ever scored on comes from the pinned public
pool. `slicer.select_slice` samples it; `exploration.select_explore_set` samples
it; `benchmarks/pool_manifest.json` pins the dataset revisions by commit SHA and
hashes the content. **There is no out-of-pool evaluation anywhere in the
subnet.** So a head that had learned the pool's routing labels by rote, with no
generalisable structure at all, would be scored exactly as highly as one that
learned to route.

That means the defence is *not* slicing and *not* nonce-derived exploration.
Both hide **which** questions will be asked; neither hides the pool. A miner
that learns all 21,717 is prepared for every slice that can ever be drawn. The
only thing standing between the subnet and a lookup table is **how much the head
can represent.**

### Capacity: how much can a linear head memorise?

Fitting **uniformly random labels** — no structure to generalise, so anything
above chance is memorisation and nothing else. On isotropic Gaussian unit
vectors, the easiest geometry that exists and therefore a strict upper bound on
what any embedding permits *(measured)*:

| questions | train accuracy on random labels | chance | memorised |
|---|---|---|---|
| 1,000 | 1.000 | 0.125 | 100% |
| 2,000 | 1.000 | 0.125 | 100% |
| 4,000 | 0.812 | 0.125 | 79% |
| 8,000 | 0.427 | 0.125 | 35% |
| 16,000 | 0.307 | 0.125 | 21% |
| **21,717** | **0.269** | 0.125 | **17%** |

**The ceiling is architectural, not under-training.** Four optimiser settings —
1,500 / 4,000 / 4,000 / 10,000 epochs at lr 0.05 / 0.05 / 0.20 / 0.10 — return
train accuracy **0.269 to three decimals in every case** *(measured)*. Full-batch
AdamW with no weight decay and no early stop, i.e. harder-working than
`train_head.train_sft`.

#### Confirmed on the real pool, not just argued

The Gaussian table above is an upper bound by an argument — clustered points
span fewer effective dimensions, so they should be *harder* to separate
arbitrarily. This project's own method note says arguments about an adjacent
layer are where things go wrong, so it was run rather than trusted. 6,000 real
pool questions embedded with the actual backbone *(measured)*:

| questions | real pool embeddings | Gaussian upper bound |
|---|---|---|
| 500 | 1.000 | 1.000 |
| 1,000 | 1.000 | 1.000 |
| 2,000 | **0.958** | 1.000 |
| 4,000 | **0.625** | 0.812 |
| 6,000 | **0.486** | ~0.55 (interpolated) |

The real pool is consistently **harder** to memorise than the bound predicts,
which is the direction the argument required. The conclusion is now measured at
both ends rather than measured at one and reasoned at the other.

So at the pool's current size a head cannot hold a lookup table. **This is a
statement about 21,717, not about the design.** At 2,000 questions memorisation
is complete and the subnet would measure nothing at all — and the only thing
keeping the pool large is that MMLU happens to have 14,042 items.

> **The pool size is a security parameter.** Nothing in `INVARIANTS.md`,
> `config.py` or `check_safety_invariants.py` says so or checks it. The code
> benchmark exclusion already removed 164 questions; a future exclusion that
> removed MMLU would take the pool to 7,675 and roughly double what a head can
> memorise, and no test would notice.

### The head is wider than anyone intended

*(measured)* `load_head_from_npz` caps rows at `HEAD_MAX_MODELS = 64` but
**never checks that the model names are distinct** — verified: a head with 64
rows all naming `openai/gpt-4o` loads without complaint. Since routing is an
argmax over rows, R rows over L models partitions the space into R regions
rather than L, which is strictly more capacity for the same models:

| head rows (over 8 models) | random-label fit at N=21,717 |
|---|---|
| 8 | 0.269 |
| 16 | 0.318 |
| 32 | **0.441** |
| 64 | 0.320 |

The 64-row number is **a lower bound, not a ceiling**: capacity is monotone in
rows by construction (extra rows can be zeroed), so a drop is the optimiser
giving up at a fixed budget, not the architecture. Read the row as "at least
0.441 is reachable".

**But that is a Gaussian result, and the real pool does not reproduce it.**
Re-run on the actual embeddings, 32 rows buys nothing at all *(measured)*:

| questions | 8 rows | 32 rows | difference |
|---|---|---|---|
| 500 | 1.000 | 1.000 | 0.000 |
| 1,000 | 1.000 | 1.000 | 0.000 |
| 2,000 | 0.958 | 0.941 | −0.017 |
| 4,000 | 0.625 | 0.621 | −0.004 |
| 6,000 | 0.486 | 0.522 | +0.036 |

Three of the five differences are zero or negative and the largest is +0.036,
against a Gaussian gain of +0.172 at 21,717. The extra rows helped on isotropic
vectors and do not materially help on clustered ones — the binding constraint on
real embeddings is how few directions the questions actually span, and more
partitions of a low-dimensional cloud do not separate more points.

So the honest version of this finding is narrower than it first looked:
`HEAD_MAX_MODELS` **does** permit duplicate model names, which is a real gap
between what the constant is named and what it bounds, and it is worth closing
on those grounds. It is **not** a measurable memorisation advantage on this
pool. Recorded rather than deleted because the Gaussian number is what a
reviewer would find if they checked the synthetic case alone.

### The generalisation gap, on real routing labels

Train on K questions, evaluate on 2,818 held out *(measured)*. Reported two
ways because they answer different questions: **label match** asks whether the
head named the same model the oracle would have, and **routed correct** asks
whether the model it chose answered the question — which is what
`proof.accuracy` counts.

| K | label match in / out | routed correct in / out | routed gap |
|---|---|---|---|
| 93 | 1.000 / 0.251 | 1.000 / 0.427 | +0.573 |
| 234 | 0.996 / 0.228 | 0.996 / 0.488 | +0.507 |
| 943 | 0.994 / 0.249 | 0.996 / 0.484 | +0.512 |
| 1,881 | 0.976 / 0.243 | 0.985 / 0.486 | +0.499 |
| 3,759 | 0.798 / 0.250 | 0.866 / 0.511 | +0.356 |
| 6,567 | 0.613 / 0.272 | 0.724 / 0.513 | +0.210 |

Majority-class label baseline 0.276; uniform chance 0.077.

Two things to read off this, and the second is the one that matters:

- **In-pool accuracy is memorisation and it is total below ~1,000 questions.**
  The head reproduces its training set essentially perfectly up to K≈1,881 and
  then degrades — the same capacity ceiling, now on real labels.
- **The gap closes because in-pool falls, not because out-of-pool rises.**
  Out-of-pool routed-correct sits at 0.43–0.51 across a 70× range of K. Nothing
  the head learns from more pool questions transfers. On label match it never
  beats the 0.276 majority class at all.

*Caveat, stated because it changes the reading:* this experiment trains on the
hard cheapest-correct label, the same objective that made `trained_ce` fail in
experiment 1. Its out-of-pool routed-correct of ~0.5 is therefore well below
what the soft-target trainer achieves (0.755). The **gap** is still measured
like-for-like — same objective, in versus out — but the absolute out-of-pool
number is a property of the objective, not a ceiling on routing.

### What actually defends the subnet, ranked

1. **Head capacity.** The only real defence, and it is a consequence of the
   linear architecture and the pool's size rather than a decision anyone made.
2. **Pool size.** A parameter with a security role that is not written down.
3. **Slicing and nonce-derived exploration.** *(read)* Real, but they defend a
   different property — that a miner cannot pre-compute *answers* to a known
   slice. Against learning the pool they do nothing, because they draw from it.
4. **Behavioural dedup.** *(read)* Would catch many miners shipping the *same*
   memorised head, not one miner shipping its own.

And a matter of proportion: `INVARIANTS.md` already records an open,
exploitable-today hole where a miner points `FUGAL_OPENROUTER_BASE` at a server
that returns the pool's own `gold` for every question. **That attack is strictly
stronger than memorisation and strictly cheaper.** Pool memorisation is worth
understanding because it survives fixing the upstream hole; it should not be
prioritised above it.

---

## Two method notes, both earned the hard way in this session

**A falsification that passes is a contradiction in terms.** A regex plant that
silently matched nothing reported "3 passed" and looked like a verified claim.
The only cheap check is to confirm the plant actually planted.

**A number that cannot exist should be checked before it is quoted.** This
document originally reported the reference model scoring **1.065 against
itself**, which is impossible: `thrift = 1` by definition for the reference, so
the score is `quality^0.9`, and `quality > 1` requires `wilson_lcb(acc) >
acc_best`. The cause was a population mismatch — the numerator computed over
answerable questions only, the denominator over all of them. Corrected, the
reference scores 0.994 against itself, the shortfall being the Wilson penalty.

Chasing it turned up something worth keeping:

> **There are two definitions of miner accuracy in this repository, differing
> by 7 points of apparent score.** `proof.accuracy` divides by every scored
> question — a question no model could answer counts as wrong, and the
> validator could not do otherwise, since it holds one proof and does not know
> what the other models would have said. `head_eval.evaluate_head` excludes
> those questions, with a comment saying why. `_proof_to_head_score` builds the
> HeadScore straight from the proof, so **nothing on the consensus path calls
> `evaluate_head` at all.** Its exclusion rule is dead code with respect to
> consensus — but it lives in the function anyone would reach for when
> reasoning about scores, and it reads 7% high with no warning.

---

## What this does not establish

- **Nothing here is a Fugal miner's score.** Different pool, different models,
  LLM-judge labels rather than mechanical graders, and a raw `acc_best` rather
  than the frame's shrunk posterior.
- **Whether a pinned model actually clears the 82.3% bar is unmeasured.** The
  break-even and the 168x price spread are computed from `data/models.json`;
  the accuracies of those 17 models on this pool are not known to anyone, and
  finding out costs $688. The SPROUT ratio of 95% is calibration, not proof.
- **The prices are assumed.** SPROUT ships no prices; the table in
  `experiment_routing_dataset.py` is public list prices as of 2026-09. Accuracy
  results do not depend on them at all; thrift and the composite do.
- **Only one train/test split was run.** The margins between the trained head,
  `domain_oracle` and `best_single` are 2–7 points on 5,000 questions and were
  not repeated across splits.

## What to do next

1. **Treat the degenerate constant as live on the pinned table until shown
   otherwise.** The break-even is 82.3% of the reference's accuracy at the
   thrift cap, eight pinned models sit at that cap, and the analogous ratio
   measured on SPROUT is 95%. The burden of proof is now on the claim that no
   pinned model clears it.
2. **Add the missing check either way:** a trained head must beat every
   constant policy. Thirteen evaluations, and it would have caught this on the
   commit that moved the exponent to 0.9.
3. **Write down that the pool size is a security parameter**, with the measured
   capacity curve as the justification and a floor enforced in
   `check_safety_invariants.py`.
4. **Make `HEAD_MAX_MODELS` mean what it says** — either require distinct model
   names or document that 64 rows over few models is permitted and why. On the
   naming gap, not on memorisation: the real pool shows no capacity gain.
5. **Decide the two-accuracy-definitions question.** Not a live bug; a trap.
