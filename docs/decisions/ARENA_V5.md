# Protocol v5: routers are paid for RouterArena's score

Status: **ACCEPTED, 2026-11-24**, for v0.8.0 (activation block in
`config.FINNEY_V5_ACTIVATION_BLOCK`). Invariants: `docs/INVARIANTS.md`, the
v5 entries. Code: `arena_score.py`, `exam_v5.py`, `seat_v5.py`,
`v5_epoch.py`, `tee/td_v5.py`.

## 1. The goal, and why v4's rule did not reach it

The subnet's product is a router that gives the best answers for the lowest
price, as RouterArena measures it. RouterArena scores a router by one
formula over its accuracy a and its cost c per 1,000 queries:

    Arena(a, c) = 100(1 + β) a C / (β a + C),   β = 0.1,
    C = (log2 200 − log2 c) / (log2 200 − log2 0.0044),  c clamped to [0.0044, 200]

v4 seated a head on prediction odds (Brier gain) and valued real picks, only
as a guard, at $1 per correct answer. At the operating points routers reach
(80–90% accuracy at $0.4–1.6 per 1K), RouterArena values a correct answer
at roughly $0.03, so v4 rewarded a far more accuracy-heavy trade-off, and
odds measure prediction on every entry, including entries a router never
picks. v5 pays for the quantity RouterArena measures, on the router's own
picks.

## 2. The score

Locally, Arena is a monotone function of

    S = a − ln(c̄) / k,     k = C(c)² · ln 2 · log2(200/0.0044) / (β · a_ref²)

with a_ref = 0.82 (`ARENA_A_REF_V5`). Two routers X and Y are compared at k
evaluated at the logarithmic mean of their mean pick costs (label-free, so
the bound stays valid). Per exam question q:

    x_q = 100 · [ (y_X − y_Y) − (t_X − t_Y) / V_XY ],   V_XY = k_XY · LM(ĉ_X, ĉ_Y)

y is the cell's pooled label in [0, 1] (RouterArena's metrics may be
fractional), t its median billed cost, ĉ each router's mean pick cost over
the compared questions, LM the logarithmic mean (integer arithmetic, no
logarithm). Equal picks give 0 and still count. Offline, on 44 graded
RouterArena runs, ranking by S agrees with ranking by Arena (Spearman 0.98;
92% of pairs within 1 Arena point ordered the same), and predicted Arena
gaps between our heads matched measured ones to about 0.2 points.

A head picks argmax over entries of (predicted success − lam × pick cost)
with its own `lam` in (0, 10000]: choosing the trade-off is the router's
job.

## 3. The exam

- For epoch E the root publishes, in its reveal of E − 2, a spec: questions
  drawn by the epoch nonce from the RouterArena-like pool, each contender's
  picks, B*'s entry on the first 100 questions, and the cells to label: the
  union of picks that differ (a question where everyone agrees needs no
  call). Every validator recomputes the spec; a mismatch voids that exam
  locally.
- Each cell is labelled by 3 miners of distinct coldkeys, assigned by a
  keyed walk of the verified (hotkey, coldkey) ring; up to 320 calls per
  hotkey an epoch, plus 48 coverage calls for training labels and new menu
  entries.
- Labels are pooled by median; a coldkey whose votes stray from the others'
  is flagged for the epoch. **Neither compared operator's votes judge their
  comparison:** an operator cannot help its own router by re-running calls,
  and cannot hurt a rival's.
- A cell whose usable votes all failed (two coldkeys or more) counts as a
  wrong answer at zero cost, as a product that could not answer would.

## 4. The seat

One rule, walked over the contenders (the newest standing head per
coldkey) in first-commit order:

- a contender replaces the incumbent when its lower bound over the
  incumbent passes **+1/4** S-point and its bound over **B*** (the best
  single full menu entry on the window's coverage labels) passes 0;
- with no incumbent the seat fills with the first contender whose bound
  over B* passes: on merit, never by a draw;
- the holder's own successor (same coldkey) replaces it unless the holder's
  bound over it passes +1/4 (it is shown worse), and must itself beat B*;
- a seat that does not beat B* is not paid: its share burns;
- exact copies are refused by ownership; a near-copy of an earlier head
  would need +1/4 over it.

The bound is `seat_bound.lower_bound` (valid at every look, α = 1%),
stratified by dataset, at least 2,000 paired questions, window 48 epochs.
Measured spread: picks differ on 60–98% of questions with a per-question
SD of about 28 points, so an edge of 2 S-points is seated after about 4,000
paired questions and an edge of 1 after about 18,000. Pay stays 85/10/5.

## 5. What was removed, and why

The odds test, the reference R and precedence chains (the score is real
picks, not odds); the challenge (every contender is examined every epoch,
and an operator no longer answers its own comparison); the activation tie
draw (the empty seat fills on merit); the cost-aware random exploration
share (labels are bought where routers disagree). The live menu, measured
costs, attestation, commit-reveal, the memorisation machinery and the
payout are unchanged.

## 6. Accepted residuals

- The score is RouterArena's, so it inherits RouterArena's trade-off; k is
  re-derived each release.
- The exam pool is public and questions are reused under the memorisation
  check; a private held-out set was ruled out so that every input stays
  auditable.
- Collusion needs two of a cell's three votes; the votes are published.
- The menu bounds the ceiling: RouterArena's leaders use cheap instruct
  models, which miners can propose through the live menu.
