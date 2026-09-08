# Miner Economics — is a miner net-positive?

**Yes, with room, though less than the first version of this document
claimed.** At the median alpha price of a live Bittensor subnet, a Fugal miner
in a 32-miner field nets **+$54/day** against **$18.76/day** of cost. The
subnet stays profitable for the **marginal** entrant up to a field of **~124
earning miners** — still more than any subnet on the network currently carries
(1–63 observed), but the margin is ~3.9×, not the 5.8× first reported.

> **Revised 2026-09-06 after cross-review.** The first version quoted `pot /
> cost` = 186 as the equilibrium. That is where the *average* miner breaks
> even, and nobody decides to enter on the average: weights are
> score-proportional, so the entrant whose decision sets the equilibrium is the
> lowest-scoring one. The same bug made `--incentive-share` self-contradicting
> — it could report a loss-making miner beside an equilibrium field size larger
> than the field it was losing money in. Fixed, with the marginal and average
> figures now reported separately. The **§ Does the scoring function keep
> miners solvent?** section was also re-derived and its conclusion reversed.

Reproduce every number here:

```bash
python scripts/model_miner_economics.py --sweep         # base case + sweeps
python scripts/model_miner_economics.py --live-chain    # re-read the chain
python scripts/model_miner_economics.py --recompute-slice-tokens
```

No paid API calls. The script cannot make one — it never imports
`fugal_subnet.api` and never reads `OPENROUTER_API_KEY`. `--live-chain` is a
read-only subtensor query and is free.

---

## Every number, labelled by how it was obtained

This project has been burned repeatedly by prose that blurs measured and
reasoned, so nothing below is unlabelled.

| Input | Value | How obtained |
|---|---|---|
| Alpha emitted to participants, per subnet | **1.000 α/block** | **read from chain** — `DynamicInfo.alpha_out_emission`, *identical* for all 124 active subnets at block 9,009,022. Not proportional to subnet rank |
| Miner share of that | **0.41** | **read from chain** — confirmed against chain; see derivation below |
| Alpha price, new subnet | **0.00491 τ/α** | **read from chain** — median of 128 non-root subnets |
| TAO price | **$240.91** | **market** — CoinGecko 2026-09-06; Kraken TAOUSD $241.44 the same minute |
| Block time | **12 s** | **read from chain** — `bittensor.core.settings.BLOCKTIME`, SDK 10.5.0 |
| Registration burn | **0.0005 τ ≈ $0.12** | **read from chain** — `Subtensor.recycle()` on netuids 64/90/99 |
| Input tokens per 300-question slice | **22,095** (sd 1,010) | **computed** — 50 nonce-seeded `slicer.select_slice` draws over the real 21,553-question pool |
| Completion tokens per answer | **256** | **assumed** — `config.FRAME_DEFAULT_COMPLETION_TOKENS`. Never measured against a real provider |
| Confidential VM | **$150/month** | **measured** — GCP `c3-standard-4`, the shape the rehearsal ran on |
| Score spread (best:worst) | **2.0×** | **assumed** — from a 3-miner rehearsal with hand-made heads; sets how far below `1/N` the marginal miner earns |
| Head-training compute | **$0/month** | **assumed** — not measured, defaulted to zero so the omission is visible; see § What this model does not establish |
| Earning miners | **32** | **assumed** — no Fugal field exists; calibrated against live subnets below |

Four inputs are assumed. The sweep below shows the conclusion survives a 4×
error in completion length, a 5× score spread, and $300/month of unmodelled
GPU — but not all three at once, which is stated plainly rather than buried.

### The 41% is confirmed against chain, not taken on trust

`Metagraph.emission` summed over all UIDs comes to exactly `tempo × 0.82` on
three subnets with two different tempos:

| netuid | tempo | Σ emission | tempo × 0.82 |
|---|---|---|---|
| 1 (Apex) | 99 | 81.1805 | 81.18 |
| 90 (KubeTEE) | 360 | 295.2017 | 295.2 |
| 99 (Thirty Spokes) | 360 | 295.2017 | 295.2 |

and on netuid 99 the UIDs carrying incentive hold 147.6008 of that 295.2017 —
exactly half. So the split is **18% owner / 41% validators / 41% miners**, and
`alpha_out_emission` is uniform. This is the documented split, and the point
is that it was *confirmed* rather than assumed — across two different tempos,
which is what makes the `tempo × 0.82` identity evidence rather than a
coincidence.

### Alpha price does not discount for being new

The obvious worry is that a fresh subnet trades far below the network median.
It does not. Grouping all 128 non-root subnets by age (**read from chain**):

| Subnet age | n | p10 | median | p90 |
|---|---|---|---|---|
| < 3 months | 11 | 0.00317 | **0.00528** | 0.01254 |
| 3–12 months | 30 | 0.00284 | **0.00507** | 0.02292 |
| > 1 year | 87 | 0.00275 | **0.00463** | 0.02829 |

The median is flat across cohorts — if anything slightly *higher* for young
subnets. So the population median is the right default for Fugal, not a
discount to it.

**This table is drawn from survivors, and that limits what it can settle.** All
128 rows are subnets that still exist at block 9,009,022. A subnet whose alpha
price collapsed and whose miners left is not in the cross-section, so
"age-independent" means "among subnets that made it, age does not predict
price" — not "a new subnet's price will not collapse". It rules out a
*systematic* new-subnet discount, which is the specific worry it was built to
answer. It does not establish that Fugal will be a survivor; that is the
product question flagged below, and no arithmetic reaches it.

---

## The base case

32 earning miners, mid-range routing, median alpha price, and — the change
from the first version — the **marginal** miner rather than the average one. At
a 2× score spread the lowest-scoring miner earns 0.67 of an equal `1/N` share:

```
  subnet-wide miner pot           3491.83   (fixed by chain; does not grow with the field)
  revenue                           72.75   (0.67 x an equal 1/32 share)
  - API, routed slice              -13.17
  - API, forced exploration         -0.66
  - confidential VM                 -4.93
                               ----------
  NET                              +53.99
```

At 24 epochs/day (`EPOCH_INTERVAL=3600`, 300 blocks each).

**The structural fact that drives everything:** subnet-wide miner revenue is
fixed at 0.41 α/block = 2,952 α/day regardless of how many miners there are.
Miners divide a constant pot. So "is a miner net-positive?" is really "is the
field smaller than the equilibrium size N\*?", and N\* is the number worth
quoting.

**Which N\*, though, is the correction this document needed.** Two exist:

| | Value | What it means |
|---|---|---|
| Average-miner break-even (`pot / cost`) | 186 | Where the field as a whole stops covering its costs |
| **Marginal-miner break-even** (`pot × m / cost`) | **124** | Where the *next* entrant stops covering its own |

The second is the equilibrium, because entry and exit are decided one miner at
a time and the one deciding is the lowest-scoring. `m = 2/(1+spread)` is the
lowest miner's share of an equal split under scores spread uniformly over a
best:worst ratio; at the rehearsal's 2× that is 0.67. The first version of this
document quoted 186 and called it the marginal figure, which overstated the
headline by 50%.

---

## How far the answer is from flipping

Each solved holding everything else at base case:

| Axis | Now | Break-even | Margin |
|---|---|---|---|
| Earning miners | 32 | **124** | 3.9× |
| Alpha price | 0.00491 τ/α | **0.001266 τ/α** | 3.9× |
| TAO price | $240.91 | **$62.12** | 3.9× |

These are one margin stated three ways, not three independent confirmations —
revenue is linear in each, so the same 3.9× falls out whichever variable is
solved for. It is quoted three ways because each way is separately checkable
against something real, and the alpha-price row is checkable against the whole
network.

The alpha-price row is still the striking one, and it survives the correction
with less room than before. **0.001266 τ/α remains below the lowest price on
the entire network** — the cheapest of 128 subnets at block 9,009,022 is netuid
16 at 0.001638 (**read from chain**). A 32-miner Fugal field would stay
profitable even if Fugal became the least valuable subnet on Bittensor, but the
gap is now 1.3×, not the 1.9× first reported. That is a real conclusion holding
on a smaller margin, not a comfortable one.

### The corner cases that matter

All for the marginal miner at a 2× spread:

| Scenario | Net USD/day |
|---|---|
| 32 miners, median price, mid routing (base) | +$53.99 |
| 32 miners, completion tokens 4× the assumption (1024) | +$14.76 |
| 32 miners, 5× score spread instead of 2× | +$17.61 |
| 32 miners, $300/mo of head-training GPU | +$44.12 |
| 256 miners, median price, cheap routing | +$3.17 |
| 128 miners, median price, mid routing | **−$0.57** |
| 256 miners, p10 alpha price (0.00275), cheap routing | **−$0.83** |
| 256 miners, median price, mid routing | **−$9.67** |
| 64 miners, everything routed to `openai/gpt-5.5` | **−$27.16** |

**Two of these flipped sign in the correction and the change matters.** A full
256-UID field at the p10 alpha price on the cheapest possible routing was
+$1.71 and is now −$0.83; 128 miners on mid routing was +$8.52 and is now
−$0.57. So the marginal miner sits almost exactly at break-even at a
128-miner field, and the honest statement is that the subnet supports a field
in the low hundreds, not that every corner is comfortably positive.

Every losing row still requires either a field far larger than any subnet on
the network sustains, or the expensive end of the routing range — and the next
section is about why that second one is no longer safe to dismiss.

---

## The cost side, computed rather than quoted

`pricing.policy_cost` is linear in tokens, so an epoch's whole bill collapses
to `Σ(input tokens) × rate_in + n × completion × rate_out`. That means one pool
statistic (22,095 input tokens per slice) prices every routing policy exactly.

Per-epoch API bill at completion = 256, priced from the hash-pinned
`data/models.json` (**computed**):

| Model routed to | $/epoch |
|---|---|
| `deepseek/deepseek-v4-flash` (cheapest) | 0.0140 |
| `minimax/minimax-m3` | 0.0988 |
| `mistralai/mistral-large-2512` | 0.1262 |
| `openai/gpt-5.4-mini` (median) | 0.3622 |
| uniform mix over all 17 | 0.5487 |
| `google/gemini-3.5-flash` | 0.7243 |
| `openai/gpt-5.5` (dearest) | 2.4145 |

**This independently reproduces the $0.13–0.73/epoch figure carried into this
task.** That range is exactly the span from `mistral-large-2512` ($0.1262) to
`gemini-3.5-flash` ($0.7243) — the middle 8 of the 17 pinned models. Two
derivations, arrived at separately, agree. The full envelope is wider in both
directions ($0.014 to $2.41) because it includes the extremes a head could
actually pick.

Two cost floors no routing strategy removes:

- **Forced exploration**, $0.0274/epoch = $0.66/day. `EXPLORE_FRACTION=0.05` of
  the slice goes to a nonce-chosen model the miner does not pick, priced here
  at the uniform mix because that is what "nonce-chosen" means.
- **The CVM**, $4.93/day. This dominates for a cheap-routing miner: at the
  cheapest policy the API bill is $0.34/day against $4.93 of hardware. The
  floor for any miner is about **$5.60/day**.

---

## Sensitivity

All figures are marginal-miner N\*.

By alpha price × routing policy:

| alpha price | cheapest | median | mix | dearest |
|---|---|---|---|---|
| 0.00275 (p10) | 220.0 | 91.3 | 69.5 | 20.5 |
| 0.00491 (median) | 392.8 | 163.0 | **124.1** | 36.6 |
| 0.00949 (p75) | 759.2 | 315.0 | 239.9 | 70.8 |
| 0.02775 (p90) | 2220.0 | 921.2 | 701.4 | 207.1 |

By completion tokens — the largest unmeasured *cost* input:

| completion tokens | cheapest | median | mix | dearest |
|---|---|---|---|---|
| 64 | 444.4 | 306.5 | 260.0 | 107.8 |
| **256 (assumed)** | 392.8 | 163.0 | **124.1** | 36.6 |
| 512 | 340.2 | 100.3 | 73.1 | 19.5 |
| 1024 | 268.3 | 56.7 | 40.1 | 10.1 |

By score spread — the largest unmeasured *revenue* input, and the one this
document previously buried in a caveat:

| spread | multiplier | cheapest | median | mix | dearest |
|---|---|---|---|---|---|
| 1.0× (equal) | 1.00 | 589.2 | 244.5 | 186.1 | 55.0 |
| **2.0× (rehearsal)** | 0.67 | 392.8 | 163.0 | **124.1** | 36.6 |
| 3.0× | 0.50 | 294.6 | 122.2 | 93.1 | 27.5 |
| 5.0× | 0.33 | 196.4 | 81.5 | 62.0 | 18.3 |
| 10.0× | 0.18 | 107.1 | 44.5 | 33.8 | 10.0 |

The 2× comes from **three** miners with hand-made heads in the rehearsal. A
real field has dedup, burn-in and evidence accumulation acting on it and should
spread wider, and N\* scales linearly in `2/(1+spread)`. At a 5× spread the
marginal miner breaks even at 62 on mixed routing — inside the 1–63 range of
earning miners actually observed on live subnets. **This is now the input most
likely to move the answer, ahead of completion length.** It is cheap to measure
the moment a real field exists: it is the ratio of the highest to lowest
non-zero incentive in the metagraph.

By head-training compute, which the first version omitted entirely:

| GPU $/month | cheapest | median | mix | dearest |
|---|---|---|---|---|
| **0 (assumed)** | 392.8 | 163.0 | **124.1** | 36.6 |
| 100 | 252.6 | 132.5 | 105.6 | 34.8 |
| 300 | 147.4 | 96.4 | 81.3 | 31.7 |
| 900 | 65.5 | 53.1 | 48.1 | 25.0 |

**The conclusion survives any single assumption being wrong by 4–5×.** It does
not survive all of them being wrong at once: a 5× spread with 1024-token
completions and $300/month of GPU puts mixed-routing N\* in the twenties. No
row here is evidence that will happen — but "wide margin" was the wrong summary
and "holds unless several assumptions fail together" is the right one.

---

## Does the scoring function keep miners solvent? At the shipped exponent, no

**This section previously claimed the opposite, and the claim was wrong.** It
was derived against `quality^0.8 * thrift^0.2`, which is what five documents in
this repo said the scoring function was. The shipped value is
`config.SCORE_QUALITY_EXPONENT = 0.9`. Re-derived at 0.9, the conclusion
reverses, so it is re-derived here rather than repaired.

N\* still swings enormously on routing choice — 36.6 for everything to
`gpt-5.5` against 392.8 for everything to `deepseek-v4-flash`, a **10.7×
range**. The question is which end the score points at.

Score is `quality^w · thrift^(1-w)` with `thrift = ref_cost / miner_cost`.
Setting `d(ln score) = 0` gives the rate at which a miner will trade cost for
quality:

```
d(ln cost) / d(ln quality) = w / (1 - w)
```

| w | Cost rise accepted per 1% quality gain | Quality ratio needed to justify the dearest model over the cheapest |
|---|---|---|
| 0.8 (what this section assumed) | 4% | **3.62×** |
| **0.9 (shipped)** | 9% | **1.77×** |

The dearest model costs 172× the cheapest per epoch ($2.4145 vs $0.0140).
`docs/design-decisions.md` records that quality spans **~3×** while thrift
spans `SCORE_THRIFT_CAP²` = 100×. So:

- At **w=0.8**, justifying the dearest model needed a 3.62× quality advantage —
  *outside* the ~3× that quality can actually span. The expensive corner was
  unreachable, cost pressure won, and the old claim held.
- At **w=0.9**, it needs only 1.77× — comfortably *inside* the ~3× span. The
  expensive corner is score-justified, and N\* there is **36.6**.

That last number is not academic: 36.6 sits inside the 1–63 earning miners
observed on live subnets. So at the shipped exponent the score does **not**
steer miners toward the solvent end of the routing range. What keeps miners
solvent is exit — the field shrinks until it fits N\* — not the incentive.

**This is an unremarked consequence of a correction made for other reasons.**
The move from 0.8 to 0.9 was forced by a scoring-integrity failure: at 0.8 a
63% router beat a 93% one at 13× cheaper. The fix was right and is well
derived. Its economic side effect is that cost pressure fell from
`100^0.2 = 2.51` of effective range to `100^0.1 = 1.58`, against quality's
`3^0.9 = 2.69` — quality now dominates ~1.7:1 where the two were previously
balanced. The subnet buys better routing and pays for it in a smaller
sustainable field.

Nothing here argues for changing `w` back: 0.8 is refuted on its own terms and
`design-decisions.md` carries the proof. It argues that **`SCORE_QUALITY_EXPONENT`
is an economic parameter as well as a scoring one**, and that the same note in
`design-decisions.md` requiring `SCORE_THRIFT_CAP` changes to re-derive the
exponent should also require re-running this model.

---

## What this model does *not* establish

- **Alpha price is exogenous here, and it isn't in reality.** The model takes
  0.00491 τ/α as given. What alpha actually trades at depends on whether anyone
  buys it, which depends on whether the subnet produces something useful. That
  is a product question, and no amount of arithmetic answers it. What the chain
  data *does* establish is that the cross-sectional distribution is
  age-independent and tightly clustered — p10 to p75 spans only 3.5× — and that
  the answer stays positive across all of it.
- **Sell pressure is not modelled.** Miners and validators together realise
  5,904 α/day. The AMM haircut on that is small (**computed**: 1.3% against
  netuid 99's pool at block 9,009,022, the thinnest young pool sampled), but a
  sustained sell into a pool nobody is buying moves the price itself, which is
  the exogeneity caveat above wearing different clothes.
- **It assumes the miner produces a valid proof every epoch.** A miner that
  misses epochs falls through `apply_miss` and earns less than `1/N`.
- **Cost and revenue share are modelled as independent, and the scoring
  function couples them.** Every N\* table varies routing policy while holding
  the miner's incentive share fixed, but routing policy is an input to
  `thrift`, which is an input to the score, which sets the share. The
  `cheapest` column is the clearest casualty: routing everything to
  `deepseek-v4-flash` maximises `thrift` and collapses `quality`, so a miner
  doing it would not hold a `2/(1+spread)` share of a normal field — N\* = 392.8
  is an upper bound nobody can occupy. The `dearest` column is distorted the
  same way in the other direction. Treat the columns as bracketing the cost
  axis, not as four reachable strategies. Closing this properly needs per-model
  accuracy data the reference frame does not have yet (`FRAME_PRIOR_ACCURACY`
  is deliberately neutral at 0.5 until testnet measures it).
- **Head-training compute is assumed to be zero.** The subnet's premise is
  continuous head optimisation, which implies ongoing training spend that this
  model does not know. Zero is the floor — a miner that trains once and then
  only serves. `--gpu-usd-month` sweeps it, and $300/month takes mixed-routing
  N\* from 124 to 81.
- **The score spread is from three miners with hand-made heads.** It is the
  revenue-side equivalent of the completion-token assumption and now has its
  own sweep row rather than a caveat sentence. Use `--spread`, or
  `--incentive-share` for a specific miner.
- **Registered UIDs are not the denominator; earning miners are.** At block
  9,009,022 (**read from chain**), of 256 UIDs each: netuid 64 had 17 earning,
  netuid 51 had 63, netuid 4 had 6, netuid 120 had 5, netuid 44 had 1. An idle
  registered UID costs its holder nothing ongoing and takes nothing from the
  pot. N\* = 186 should be compared against 1–63, not against 256.

---

## What is worth measuring next

One input carries all the residual uncertainty, and measuring it is already on
the roadmap for another reason.

**Completion length against a real provider.** `docs/LIVE_API_VALIDATION.md`
already asks for a single live epoch to check the pinned price table against
real OpenRouter billing. That same run yields real `completion_tokens`, which
is the assumed input here. The ~112–131 tokens visible in
`results/rehearsal/` came from `scripts/stub_upstream.py` and are synthetic —
they are not evidence about real models. [PAID — the doc's own estimate is
"well under $1" at `FUGAL_SLICE_SIZE=40`.] Not run; flagged for approval rather
than executed.

Everything else in this model is already read from the chain or computed from
repo code, and `--live-chain` / `--recompute-slice-tokens` re-derive those in
place. Both flag a drift from the pinned values (>10% and >5% respectively) so
a stale pin announces itself instead of quietly re-scoring the conclusion.

---

## Implication for the rest of the project

The premise behind the TEE work holds. Miners are net-positive with a 3.9×
margin to break-even at a plausible field size, so the attestation effort is
protecting a subnet that can attract participants rather than measuring the
wrong thing. The margin is smaller than first reported and the answer is
"yes, in a field of low hundreds" rather than "yes, comfortably".

The economics also corroborate **I5** from a direction it had not been checked
from. I5 says validators bear no inference cost, and they collect the same
0.41 α/block pot that miners do — so validator economics are positive by a much
larger margin than miner economics, and the asymmetry the TEE architecture
created is in the direction the subnet needs.

The binding constraint on field size is the **CVM, not the API bill**. At
cheap routing, hardware is 83% of a miner's cost, and hardware plus forced
exploration — the two things no routing strategy can reduce — are 94%. That
is the number to attack if the field ever needs to grow past N\*:
`docs/INVARIANTS.md` records AMD SEV-SNP bare metal at ~$207/mo observed and
hourly TDX as non-existent, so
there is no cheaper confidential option today. `--cvm-usd-month 0` puts the
ceiling at N\* = 168 — so even free hardware only moves the equilibrium 36%.
The pot is the constraint, and the pot is fixed by the chain.

One further lever nobody had costed as a lever: **`SCORE_QUALITY_EXPONENT` sets
the sustainable field size**, because it decides where in the 10.7× routing
range miners settle. That is an argument for measuring per-model accuracy
early — not for changing the exponent, which is derived and correct.

## Measured on the first live epochs (2026-09-07/08)

Two heads routing only among the three cheapest pinned models, 300-question
slices plus the 15-question exploration quota, real OpenRouter calls inside
dstack TDs:

- Pinned cost per epoch per miner: **$0.16–$0.22** (proof totals), against
  the $0.014 + $0.027 this document's table implied for that routing. Two
  reasons: completions averaged **534 tokens** (max 2,049) where 256 was
  assumed, and the exploration quota — nonce-assigned across all 16 priced
  models including the dearest — was **~60% of the epoch's cost** for a head
  that otherwise never calls them.
- OpenRouter's account meter tracked the pinned totals within ~5% across four
  miner-epochs (account-level; the proof does not yet carry provider cost).
- Startup: 7 h 39 m of `c3-standard-4` time (~$1.60) to embed the pool once.

The exploration share is a product question: the quota is what makes the
reference frame unbiased, and it is charged to miners in proportion to how
cheap their own routing is.
