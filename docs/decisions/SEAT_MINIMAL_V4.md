# Protocol v4: the seat reduced to its core (design)

Status: **DESIGN, 2026-11-19.** It describes rules for a future protocol v4
and applies only from a v4 activation block that no release has set.
Protocol v3 (`docs/INVARIANTS.md`) is what runs today. Nothing here is built.
It supersedes the rule set of the earlier seat design on branch
`v4/seat-design` (`SEAT_AND_ECONOMICS_V4.md`), whose measurements it reuses,
and it settles the seat items `LIVE_MENU_V4.md` leaves open (its sections 8,
9.1 and 17). Words as in `LIVE_MENU_V4.md`: an **entry** is one way of calling
a model, a **pair** (or label) is one pooled exploration answer nobody chose,
**R** is the reference odds, the **scored questions** are an epoch's slice.

Every number is either **from published epochs** (the 44 protocol v3 reveals
`e00026466`..`e00026509`; for when a question was first published, also the
117 protocol v2 reveals `e00026349`..`e00026465`), or a **model** (a stated
simulation), and says which. No paid call was made.

## 0. The decision on one page

A router is a predictor: for each question and each entry, the chance the
entry answers correctly, and what it costs; a pick maximises odds minus
lambda x cost. The subnet has two jobs: **buy labels** (question, entry,
right or wrong, cost) cheaply and honestly, and **pay whoever turns labels
into the best predictions**. Every other rule exists to stop a named cheat.

The v3 seat, its pay and its evidence rest on 35 rules, gates, thresholds
and windows (section 2). This
record keeps what stops a real cheat and folds the rest into three rules
built on one statistical bound:

| | Rule | What it decides |
|---|---|---|
| **A** | **The odds test.** X beats Y when, on at least 1,500 labels drawn after both were committed and supplied by neither one's operator, the lower bound of X's Brier gain over Y is above 0.1 Brier points. The bound is valid at every look, so it is re-run every epoch for every head with no confirmation phase | who is better at predicting |
| **B** | **The real-picks check.** A head about to take the seat answers the 200 scored questions; its net value over the seat holder, at $1 per correct answer and each call priced at the measured pick cost of its entry and category, must have a lower bound above -1.5 points on at least 1,000 questions. The seat holder answers them every epoch. The answers also build the pick-cost table | that better odds do not route worse |
| **C** | **Precedence.** A head may take the seat only if it beats (A) every earlier head still in the race. R is the earliest head of all | that a later near-copy never passes an earlier head, and that one missed epoch costs nobody its place |

Four additions survive because none of the three can do their job:
ownership of a hash (an exact copy is the same head, so no comparison can
refuse it), a standing window of three epochs (liveness: how long an absent
head keeps its place and the holder its seat), the label-pool rules (an
operator cap and the plausibility check: they protect labels, not the seat),
and one memorisation check (section 6). Pay stays 85 / 10 / 5.

What goes: burn-in, the confirmation phase and its open window, pending
records, ghosts and their three lengths, the high-water marks, the accuracy
floor, the hard-question weight, the bootstrap, the senior-bar copy rule, the
first-fill ranking, v1 dedup, the cross-miner agreement detector, the shared
exploration set and the weight ramp (section 12).

New entries are usable at once: from the epoch an entry joins, a head with a
row is scored on it and may route to it; a head without one is scored, for
20 epochs, as if its row said the entry's measured success rate, a row any
head could write in one number, so omitting a row gains nothing and lacking
one costs only what a real row earns; after 20 epochs a missing row is
certainly wrong (section 4).

The recommended label budget is 112 exploration calls per hotkey-epoch,
about $20 a day with the confidential VM: at ten hotkeys a head 0.5 Brier
points better is seated in about 8 epochs, every entry has 100 labels in
about 7 (section 5).

## 1. What the subnet buys, and what follows

Two things are bought and one thing is paid for:

- **Labels**, each an attested model call graded by a pinned grader inside a
  TD. They must be cheap (every hotkey buys them), honest (no operator chooses
  a label's outcome or which labels judge its own head), and plentiful (they
  are the evidence every decision reads).
- **Real picks**, the scored questions: what a router actually costs and
  earns, which is also the only place the cost of a pick can be measured
  (exploration under-states it about twofold, `LIVE_MENU_V4.md` 7, the earlier
  seat design's section 8).
- **Predictions**, paid through the seat. A head is only bytes on a locked
  public encoder; validators compute its odds on every label themselves.

From this:

1. **Evidence must be labels the head could not have seen.** A head is fixed
   by its commitment; a label drawn after that cannot have been fitted.
   Whether its operator served that epoch changes nothing about what the head
   predicts, so it must change nothing about how the head is scored (3.2).
2. **No head is judged on labels its own operator supplied.** An operator
   decides which of its runs to serve; if its own calls could judge its own
   head, that choice would be a lever (3.2).
3. **One bound, valid under repeated looks.** Every decision is a paired mean
   compared with a margin, re-run every epoch. A bound that is valid at every
   look makes a second "confirmation" sample unnecessary (3.1).
4. **Speed comes from labels, never from a lower bar.** The margin, the
   confidence and the minimum evidence are fixed; the label budget is what
   moves (section 5).
5. **Precedence is "not materially better than an earlier head", not a
   resemblance measure**, because the margin is the one test a copy cannot
   pass by being changed (the earlier seat design, 6.1, measured why).

## 2. Inventory: every rule that decides the seat, the pay or eligibility today

Read from the code at `4d2dc3a` (`champion.py`, `odds_score.py`,
`copy_rule.py`, `dedup.py`, `v2_epoch.py`, `routing_protocol.py`,
`explore_pool.py`, `payout.py`, `rewards.py`, `evidence.py`, `tee/verify.py`,
the validator's v2/v3 wiring, `config.py`). "Cost" is what it costs honest
miners, from published epochs where they show it. The verdict says what
stops the cheat afterwards when the rule goes.

Costs seen in the 44 published v3 epochs, used below: the net-value guard
refused the eventual champion and two other heads for 10 epochs
(`e00026485`..`e00026494`) while their odds bound over the champion was
+0.16 to +0.29 Brier points, above the 0.1 margin; when the champion lost
its seat for lacking new rows at the end of a menu grace, six heads beat the
bar's odds by +4 to +19 points for three epochs (`e00026495`..`e00026497`)
and were refused by the guard while the seat burned; the high-water marks
refused 14 heads 68 times (one at an accuracy equal to the mark to four
places); "too few pairs or questions" refused 23 heads 75 times; the
confirmation phase recorded one pending head and confirmed none.

### 2.1 Who may take, hold or fill the seat

| # | Rule (code) | Cheat or failure it stops (test) | Cost to honest miners | Verdict |
|---|---|---|---|---|
| 1 | **Paired odds test**: Brier gain over the bar, LCB above `BRIER_MARGIN` 0.1 at `CHAMPION_CONFIDENCE` 99%, `ODDS_MIN_PAIRS` 1,500 (`_challenge_odds`, `paired_test`) | a head no better than the holder trading the seat on noise (`test_better_odds_take_the_seat_and_equal_or_lookup_odds_do_not`) | the margin's price: heads within twice the margin wait long | **KEEP simplified** as rule A: the same margin and minimum, the bootstrap replaced by a closed-form bound valid at every look |
| 2 | Bootstrap: 2,000 PCG64 resamples per test, seeds from epoch and hashes (`paired_test`, `pair_seed`) | (determinism of 1) | validator time; a numpy-version dependency in consensus | **REMOVE**: the bound of 3.1 is exact integer arithmetic with no random stream |
| 3 | **Fresh-pair confirmation**: a first pass records a `Pending`; take only after passing again on `ODDS_CONFIRM_PAIRS` 1,500 pairs drawn later; the record lapses when the pass leaves the window or the run breaks (`_live_pending`) | many heads with no edge passing by luck, k heads being k chances (`test_seat_confirmation.py`; `experiment_seat_confirmation.py`) | roughly doubles the time to seat (INVARIANTS, confirmation); a miss drops the record | **REMOVE**: the margin already makes a zero-edge pass vanish and the anytime bound makes repeated looks honest (3.9: 0 zero-edge takes in every simulated setting, at the margin at most 1% over 300 epochs, against 0.7-1.2% with confirmation) |
| 4 | `CHAMPION_CONFIRM_OPEN_EPOCHS` 5: a seat filled from empty is open to the plain test (`confirm_from`) | a lucky first fill being protected by confirmation | none | **REMOVE** with 3 |
| 5 | **Net-value guard**: NV over the bar on the scored questions, LCB above -`NV_NONINFERIORITY` 0.5, realized cost, V = `CHAMPION_VALUE_PER_CORRECT_USD` $0.49, at least `CHAMPION_MIN_PAIRED_QUESTIONS` 1,500 (`pair_units`) | sharper odds that route worse taking the seat (`test_sharper_odds_that_route_worse_do_not_take_the_seat`) | the binding constraint today: per-question SD 47 points, 42 of them cost; it held the eventual champion 10 epochs and blocked every head for the three burn epochs above | **KEEP simplified** as rule B: $1 per correct, priced at the measured pick-cost table, hard weight 1, 200 questions an epoch, only for a head about to take and the holder |
| 6 | `CHAMPION_HARD_WEIGHT` 2 on the hard category | routers that give up the hard questions | doubles the guard's variance at the v4 mix (SD 32.5 against 21.4, earlier seat design 3.2) | **REMOVE**: the v4 mix already gives hard questions 30%, derived from where outcomes vary |
| 7 | **Burn-in** E'': 1,500 scored questions since the last reset or miss before a take; a miss before it resets the evidence, after it decays and restarts it (`evidence.burn_in_complete`, `apply_miss_v2`) | a young head shaping its record by withholding bad epochs (`test_miss_rule_v2.py`, `test_a_withheld_epoch_never_enters_the_pairing_and_a_miss_restarts_it`) | about five epochs after every commit and every miss | **REMOVE**: odds are scored on every label after commit whether or not the head served (3.2), so withholding selects nothing; the minimum evidence of A and B remains |
| 8 | **Current run**: a challenger is measured only on its unbroken run of served epochs (`Window.current_run`) | the same withholding (`test_a_head_is_scored_only_on_pairs_from_its_own_run`) | a miss forfeits all evidence | **REMOVE for odds** (3.2 makes it unnecessary); **KEEP for picks** (B reads the head's unbroken run of answered epochs: picks cannot be recomputed for an epoch nobody answered) |
| 9 | **High-water marks**: overall and hard accuracy at or above the seated head's Wilson lower bound, never falling while a champion or ghost stands; `WILSON_CONFIDENCE` 95%, `CHAMPION_HARD_GATE_MIN_QUESTIONS` 100 (`_passes_gates`, `_seated`) | a cheaper but less accurate router taking the seat (`test_the_high_water_mark_gates_and_only_rises`) | 68 refusals of 14 heads; absolute, so it does not follow a harder mix or a provider change (INVARIANTS residual) | **REMOVE**: B prices accuracy at $1 a correct answer, the product's own exchange rate; a router that loses a point of accuracy must save a cent a question to pass, which is the trade the product makes |
| 10 | **Accuracy floor** `CHAMPION_MIN_ACCURACY` 0.30 to take, hold and be paid | a broken head (every call failing) seated and paid (`test_a_broken_head_cannot_take_an_empty_seat`) | none seen | **REMOVE**, replaced by the failure bound of the liveness rule (3.7): an epoch with more than 34% of scored calls failed is a missed epoch. A head's picks are a function of its odds (`picks_and_odds_v4`), so a head with good odds cannot pick at random |
| 11 | **Skill over R**: a challenger must beat R (no margin); a fill needs significant skill; a holder loses the seat when R beats it (`_challenge_odds`, `can_hold`, the first-fill branch) | a table of rates, or a head with no skill, filling or holding the seat (`test_reference_skill_gate`, `test_beating_a_bad_champion_is_not_enough_without_beating_the_reference`, `test_a_champion_whose_odds_are_worse_than_the_reference_does_not_hold`) | none: live heads are +0.8 to +1.9 points over R | **MERGE** into A and C: R is the earliest head, so every taker must beat it by the same test; a holder that R beats is no longer the bar |
| 12 | **Full-menu rule**: a head missing any scored entry's row is not eligible to take, confirm, fill or hold (`_no_full_menu`) | a copy of the champion with its weakest rows removed, the gaps filled by R (`test_a_copy_of_the_champion_with_a_row_removed_does_not_take_the_seat`) | a champion lost its seat for this at the end of a menu grace, followed by three burn epochs | **MERGE** into the per-pair odds: a missing row is the entry's measured rate during the allowance and certainly wrong after (section 4); no separate eligibility test |
| 13 | **Per-entry grace**: an entry's pairs are scored for no head until `scored_from` (`grace_excluded`, `EpochPairs.scored`), `ENTRY_GRACE_EPOCHS` 24 | heads being penalised for an entry nobody has labels for (`test_a_head_without_the_row_is_not_penalised_during_grace_and_is_ineligible_after`) | the first good row cannot win for 24 epochs | **REPLACE** by the allowance of section 4: scored from the join epoch, missing rows at the measured rate for 20 epochs |
| 14 | **First fill** ranks eligible heads by measured skill over R (v3) or solo net value (v2); exact ties by real first-commit block, then seniority (`solo_skill`, `solo_value`, `Candidate.first_block`) | a salted draw among copies at the flip (`test_a2_copies_committed_after_the_flip_never_win_the_first_seat`) | the best of several noisy measurements wins, not the earliest equal | **REPLACE** by C: the latest qualifier that beats every earlier one; among heads not measurably different the earliest wins |
| 15 | **Ghosts**: a champion that misses leaves a bar for `CHAMPION_GHOST_OFF` 3 epochs; one whose owner retires it for `CHAMPION_GHOST_RETIRE` 24 (8 from the confirmation block, `_CONFIRM`); a missed ghost is upgraded to retired; while a ghost stands the seat burns and the ghost keeps seat 1 | copies taking an absent or retired champion's seat (`test_a3_inside_its_ghost_the_champion_cannot_lose_the_seat_to_a_copy`) | burns the miner share for the whole window when nobody beats the ghost | **MERGE** into one standing window G = 3 for every head (3.7): an absent head keeps its place and the holder its seat for G epochs; after that the seat goes to the latest qualifier at once, with no burn if one exists |
| 16 | **Resume**: a missed champion takes its seat back inside its ghost window | an outage handing the seat to whoever is next | none | **MERGE** into standing: the holder never leaves while it stands |
| 17 | **Senior bars** (copy rule): a head must beat every senior seat holder (champion, seats 2-3, ghosts) by the margin (`copy_rule.senior_bars`, `seat_holders`, `comparison_set`) | perturbed copies of a seat holder taking a seat (`test_a_near_copy_of_a_former_champion_does_not_take_the_seat`) | none measured; protects seat holders only, so a near-copy of an unseated head can pass it | **MERGE** into C, which protects every earlier head in the race, not only seat holders |
| 18 | **Seniority** = first commit of the hash; pre-flip commits tie at the activation block, broken by a salted hash (`copy_rule.seniority`, `floor_block`, `salt`) | committing early to buy the first seat; grinding the order (`test_the_pre_flip_order_depends_on_the_activation_hash_so_it_cannot_be_ground`) | none | **KEEP** as C's order; the salted tie only at the v4 activation boundary |
| 19 | **Ownership**: a hash belongs to the coldkey that first committed it; the same hash under another coldkey is a copy; the owner's second hotkey on the same hash is a duplicate; an unknown hash is ineligible (`classify`) | serving another operator's exact bytes (`test_the_same_hash_under_another_coldkey_is_a_copy`) | none | **KEEP** (3.7): an exact copy is the same head, so no comparison can refuse it |
| 20 | Retirement fold: `retired_at`, records kept while committed or a ghost (`observe`) | (state for 15 and 17) | none | **KEEP simplified**: a record is kept while committed or standing |
| 21 | v1 **dedup**: routing cosine 0.99, parameter distance 0.2, epsilon rule 0.95 / 0.005 (`dedup.py`, `head_distance.py`) | copy-and-tweak under v1 (`test_finding5_near_copy_dedup.py`) | computed every epoch, read by no v2 or v3 decision | **REMOVE** from the v4 path: C does the job by measured improvement |
| 22 | `CHAMPION_TIME_GATE_FACTOR` 2 (per-call time gate) | slow routers | never enforced (no hash-bound latency yet) | **REMOVE** (dead constant) |
| 23 | `allow_take=False` when a validator's window is not trusted | judging the seat on a partial window | none | **KEEP** (operational fail-closed, not a seat rule) |
| 24 | A protocol change starts an empty window and seat (`rules` tag) | carrying state across incompatible pools | the activation burn (about five epochs here, 3.6) | **KEEP** |

### 2.2 Who is paid

| # | Rule | Cheat it stops (test) | Cost | Verdict |
|---|---|---|---|---|
| 25 | **85 / 10 / 5** across the holder and the two before it; an unpaid seat's share to the holder; with no holder it burns (`payout.seat_weights`) | registering more hotkeys to buy shares (`test_extra_registrations_buy_nothing`) | none | **KEEP** (owner) |
| 26 | A seat pays only while live and passing the hold gates (accuracy floor) | paying a broken or absent former champion (`test_a_seat_that_fails_the_gates_pays_the_champion`) | none | **KEEP simplified**: live and not beaten by R and not refused by the memorisation check |
| 27 | A ghost keeps seat 1 unpaid; seats 2-3 do not shift | an absent champion's share flowing to former champions | burn while a ghost stands | **REMOVE** with ghosts |
| 28 | **`MAX_WEIGHT_DELTA`** 0.3: each UID's weight moves at most 0.3 an epoch (`rewards.cap_weight_change`) | v1 score manipulation in one epoch | a new champion reaches its share over three epochs while the replaced one keeps being paid | **REMOVE**: every input to the weight vector is a published consensus decision; a validator that diverges is clipped by Yuma's stake-weighted consensus, which is where one validator's error belongs |

### 2.3 What enters the evidence

| # | Rule | Cheat it stops (test) | Cost | Verdict |
|---|---|---|---|---|
| 29 | One vote per coldkey per pair, outcome the coldkey median (`pool_pairs`) | sybil hotkeys outvoting (`test_one_operator_is_one_vote_however_many_hotkeys_it_runs`) | none | **KEEP** |
| 30 | **Operator cap** on private pairs, `EXPLORE_COLDKEY_CAP_V4` 60 = 1.5x the private set, lowest keyed digest kept (`capped`) | one operator supplying most of the pool (`test_an_operators_private_pairs_are_capped_and_it_cannot_choose_which`) | an operator's third hotkey adds little | **KEEP**, scaled to 1.5x the new private set |
| 31 | **Plausibility check**: an operator whose private answers are more than 4 SD from R (after 50 pairs of the model) is left out of the pool (`private_implausible`) | a tampered upstream or harness supplying impossible outcomes (`test_implausible_private_answers_are_flagged_in_either_direction`) | it excluded nobody in the 44 published epochs | **KEEP** |
| 32 | v1-frame implausibility and **cross-miner disagreement** detectors feeding `flagged` (`reference_frame`, `EXPLORE_DISAGREE_*`) | deviant answers on the shared set (`test_finding4_two_sided_detector.py`) | they left one operator out of the pool for 8 of the 44 epochs | **REMOVE**: a shared pair's outcome is a median, which one operator cannot move, and private answers are judged by 31 |
| 33 | The **shared exploration set** (8 calls every miner answers) | the agreement check's input | 8 calls a hotkey-epoch | **REMOVE**: its only reader is 32; and under 3.2 a pair every operator voted on judges no head |
| 34 | **Exploration failure bound**: a proof whose exploration fails more often than its scored calls by `EXPLORE_FAIL_MARGIN` 34% is refused (`verify_proof`) | skipping exploration by blocking one's own network (`test_a_proof_that_fails_exploration_but_not_scoring_is_rejected`) | none | **KEEP simplified**: absolute, 34% of exploration calls, waived in an outage epoch (the median coldkey above 30% failing, `LIVE_MENU_V4.md` 9.2), since the scored questions are now optional |
| 35 | Odds scored only on pairs of the head's run, after its first commit | fitting to published outcomes | see 8 | **KEEP simplified** (3.2): after commit, every epoch, not only served ones |

Out of scope and unchanged: the proof checks (attestation, nonce, the head
against its commitment, head bounds), the menu step's entry rules and
retirement, and the head publication hold (operational).

## 3. The minimal rule set

### 3.1 One bound

Every decision below compares a paired mean with a margin, and is re-run
every epoch. They all use one function, exact on any CPU:

```
items       x_i  integers (one-millionth of a point), each with a category c_i
n           the number of items;  S = sum x_i;  D = S - m x 10^6 x n   (m the margin, points)
V           sum over categories c of  [ sum_{i in c} x_i^2 - (sum_{i in c} x_i)^2 / n_c ]
            (exact Fraction: the within-category spread, so the bound is stratified)
rho         V x n0 / n                                   (n0: the rule's minimum evidence)
passes      n >= n_min  and  D > 0  and
            D^2 > (V + rho) x ( A_ALPHA + LN2_UP x bit_length(ceil(1 + n / n0)) )
A_ALPHA     9211/1000   (>= 2 ln(1/alpha), alpha = 1%)
LN2_UP      6932/10000  (>= ln 2)
```

This is the one-sided normal-mixture confidence sequence (Robbins; Howard,
Ramdas, McAuliffe and Sekhon, 2021) with the variance estimated from the
items and the logarithm replaced by an integer upper bound, which only
widens it. Its property is the one the seat needs: for a head whose true gain
is exactly the margin, the chance that it **ever** passes, however many
epochs it is re-tested, is at most about alpha. That is what the confirmation
phase was built to approximate with a second sample. Unlike today's bootstrap
it needs no random stream, no resample count and no numpy version in the
consensus digest, and it costs one pass over the items.

### 3.2 The labels a head is scored on

For two heads X and Y (Y may be R), the **labels** are the pooled
exploration pairs that

1. were drawn in the last `SEAT_WINDOW_EPOCHS` (48) epochs;
2. were drawn after both rows they read were committed: a head's row for an
   entry counts from the first epoch whose boundary is at or after the commit
   of the first head (itself, or a predecessor by succession, 4.4) that
   carried that row byte for byte;
3. were supplied by neither X's nor Y's operator (no voting coldkey is either
   head's owner);
4. are not proposal measurement pairs (unscored for every head, as built).

A head's odds on a pair are computed by validators from its committed bytes
and the pinned embeddings, **for every epoch in the window, whether or not
the head served it**: a validator computes them when it verifies the head,
and for the epochs the head missed it computes them, from the stored pairs,
in the epoch the head is next verified, and publishes them in that reveal.
A head's row missing for a pair's entry gives the entry's measured rate
during the entry's allowance and is certainly wrong after it (section 4).
Pairs carry their exploration weights, as today.

Why each condition:

- (2) and "every epoch" together make evidence a function of the head alone:
  no label can have been fitted, and serving or not serving an epoch selects
  nothing. Burn-in and the current-run rule existed only to price
  withholding; with nothing to select there is nothing to price.
- (3) makes evidence independent of the head's own operator: the only labels
  an operator can influence are its own calls (which run it serves, which
  calls it lets fail), and those never judge its own head. It costs a
  comparison about two operators' share of the pool (2/H of the labels with H
  equal operators), which the label budget of section 5 counts.
- (1) bounds the state: per head and epoch, one int16 per pair.

### 3.3 Rule A: the odds test

**X beats Y** when the bound of 3.1 passes on X's Brier gain over Y on their
labels (x_i = the pair's weighted loss of Y minus that of X), with margin
`ODDS_MARGIN` 0.1 Brier points, `ODDS_MIN_PAIRS` 1,500 labels and n0 = 1,500.

The margin and the minimum are v3's. The per-pair SD of the weighted Brier
difference against the champion is 3.8 to 8.3 points on published epochs;
the tables below use 8.

### 3.4 Rule B: the real-picks check

A head H **passes the picks check** against the holder B when the bound of
3.1 passes with margin -`PICKS_MARGIN` (-1.5 points), `PICKS_MIN_QUESTIONS`
1,000 and n0 = 1,000, on

```
NV_i = 100 x [ (g_H,i - g_B,i) - (t_H,i - t_B,i) / V ]          (points)
g     the shared cell grade (I4); 0 for a failed call
t     0 for a failed call, else the epoch's pick cost of the routed entry in
      question i's category (MenuState: token means at the epoch's price)
V     1 / BENCHMARK_LAMBDA = $1.00, so a cent of cost is one point
```

over the questions of H's **answered run** (its consecutive epochs with the
200 scored questions answered, at most 34% failed) on which B also answered.
A miss restarts the run: grades cannot be recomputed for an epoch nobody
answered, and a run with gaps would let a head keep only its good epochs.
When the seat is open (3.6) and H has fewer than 1,000 questions paired with
the last holder, the check asks only that H's answered run holds 1,000
questions.

- **Who answers.** The holder, every epoch: its answers are the bar, the
  routing the product checks parity against, and most of the cost table; an
  epoch without them is a missed epoch. Any other head when it chooses. Its
  standing on A (its gain over the bar and the bound) is published every
  epoch, so a head can start answering about five epochs before it expects
  to pass.
- **It produces the cost table.** Graded scored cells are where routers
  spend, so the pick cost per (entry, category) is their token means, shrunk
  toward the entry's mean and then toward its exploration call cost (the
  earlier seat design's 8.2, unchanged: `PICK_COST_PRIOR_CELLS` 30 and 20).
  The product's pick uses the per-entry pick cost.
- **Why these choices.** At the v4 mix, pricing at the table and weighting
  hard questions 1 gives a per-question SD of 21.4 points (the earlier seat
  design, 33 pairings on published epochs) against 51 for v3's estimator. No
  separate realized-cost check is added: picks are a function of odds and the
  per-entry table cost (`picks_and_odds_v4`), so a head cannot choose
  long-answer questions apart from its odds; the table is fed by the same
  picks; and its measured bias against realized cost is +0.03 points.

What it lets through (model: SD 21.4, 200 questions an epoch; share of
challengers that have passed by epoch 5 / 10 / 20 / 40 of answering):

| True NV over the holder | the bound, margin 1.5 (recommended) | the bound, margin 1.0 | a fixed 99% bound, margin 1.0 (v3's kind of test) |
|---|---|---|---|
| +1.0 | 22% / 91% / 100% / 100% | 7% / 62% / 99% / 100% | 74% / 98% / 100% / 100% |
| +0.5 | 7% / 62% / 99% / 100% | 1% / 23% / 80% / 100% | 45% / 84% / 99% / 100% |
| 0 | 1% / 23% / 81% / 100% | 0 / 4% / 28% / 78% | 20% / 50% / 81% / 98% |
| -1.0 | 0 / 0 / 2% / 10% | 0 / 0 / 0 / 0 | 1% / 3% / 4% / 6% |
| -2.0 | 0 | 0 | 0 |

The check is a guard, not a race: the head has already beaten the holder on
odds. Margin 1.5 under the bound lets a challenger one point better through
in about 10 epochs and one a point worse through about 2% of the time in 20
epochs; margin 1.0 holds an equal router for about 30 epochs.

### 3.5 Rule C: precedence

Heads are ordered by their first commit (of themselves or, for a successor,
of their earliest predecessor, 4.4); commits at or before the v4 activation
block tie at it, broken by the salted digest of `copy_rule.seniority`. **R is
the earliest head of all.**

A head S is a **contender** when it stands (3.7), answered the scored
questions in one of the last G epochs, beats the bar (A), and is not shown to
route worse than the holder (the picks check run the other way: the bound of
the holder's NV over S, margin 1.5, does not pass). R is always a contender.

**A head may take the seat only if it beats (A) every contender earlier than
it.**

- A copy, perturbed or retrained a little, of an earlier contender is not
  better than it by the margin, so it cannot pass it, whatever it looks like,
  and whether or not the earlier head served last epoch.
- An earlier head keeps its place through a missed epoch: its odds are
  computed for the missed epoch anyway (3.2) and it stands for G epochs.
- An earlier head bars only while it pays for the scored questions and is
  not shown to route worse, so no head can hold a place it never tries to
  use.
- Two independent heads that are not measurably different: the earlier goes
  first. The later loses only a seat it could not have taken from an equal
  holder either.

### 3.6 The seat, each epoch

1. **The bar** is the holder if it stands, R does not beat it, and the
   memorisation check (section 6) does not refuse it. Otherwise the bar is R
   and the seat is **open**.
2. **Qualifiers** are heads verified this epoch with their scored answers,
   not exact copies, not refused by the memorisation check, that beat the bar
   (A), pass the picks check (B), and beat every earlier contender (C).
3. If there are qualifiers, **the one committed last takes the seat.** It is
   unique: every other qualifier is a contender earlier than it, so it beat
   them all. The holder it replaces goes to the history.
4. Otherwise the holder keeps the seat, open or not: a seat is never emptied
   without a replacement.

The decision reads no arrival order and no ranking of margins. At the v4
activation the seat starts empty (as at every protocol change); the first
qualifier needs 1,500 labels over R and 1,000 answered questions, so the
miner share burns for about five epochs.

### 3.7 What else is needed, and why it cannot be folded in

- **Ownership.** A hash belongs to the coldkey that first committed it; the
  same hash under another coldkey is not a head (not scored, seated or
  paid), and the owner's second hotkey on it earns nothing. An exact copy is
  the same head, with the same commit and the same odds, so no comparison
  can tell the two apart (`copy_rule.classify`, unchanged).
- **Standing**, `STANDING_EPOCHS` G = 3. A head stands while it was verified
  in this epoch or one of the two before it; the holder stands while it
  answered the scored questions (at most 34% failed) in one of them. One or
  two missed epochs cost no head its place and no holder its seat; after
  three the holder's seat opens and goes at once to the latest qualifier,
  with no burn if one exists. A, B and C compare heads; none of them says how
  long an absent head counts, so this one constant does what the ghost
  lengths, the resume rule and the pending record's miss rule did. G is the
  copy-after-outage trade (section 13).
- **The label pool rules**: one vote per coldkey, the operator cap (1.5x a
  hotkey's private calls), the plausibility check against R, and the
  exploration failure bound (a proof with more than 34% of its exploration
  calls failed is refused, except in an outage epoch). A trusts labels; these
  rules make them trustworthy, and they decide nothing about who wins.
- **The memorisation check** (section 6). A measures prediction on the
  pool, and a head that has memorised the public pool predicts it well for
  the wrong reason. Nothing in A, B or C can see that.

### 3.8 Pay

The miner share goes 85 / 10 / 5 to the holder and the two holders before it
(`PAYOUT_SEAT_SHARES`). A seat pays in an epoch when its head is live
(verified; the holder with its scored answers), R does not beat it, and the
memorisation check does not refuse it. An unpaid seat's share goes to the
holder; an unpaid holder's share burns. Seats are paid to the UID serving
the seat's head or its successor. No weight ramp.

(As built, S4: `fugal_subnet/seat_v4.py`; the details chosen where this
record is silent are listed in `docs/INVARIANTS.md`, "I3 — v4: the seat in
three rules". One differs from 3.2's wording: a reveal publishes at most
`LATE_ODDS_SLOTS_V4` (32) head-epochs of late odds, newest first and the
heads in an order that turns with the epoch, so a reveal at 256 UIDs stays
inside the bound every node reads (`reveal_sync.MAX_REVEAL_BYTES`, 64 MiB when
written, 96 MiB since); the rest are computed and published in
the epochs after, and a comparison reads an epoch once both heads' odds for
it are held.)

### 3.9 What the single bound buys, and what it costs

Model: each epoch a challenger adds n labels whose gains over the bar are
normal with mean d and SD sigma; 2,000 to 20,000 runs per cell. "Pass and
confirm" is v3's odds rule (99% on the run within 48 epochs, then 99% again
on 1,500 fresh labels, the record lapsing at 48 epochs); "single bound" is
rule A on its sliding 48-epoch window.

**False takes** (share of heads that ever take the seat):

| True gain d | sigma | labels an epoch | pass and confirm (150 epochs) | single bound (300 epochs) |
|---|---|---|---|---|
| 0 (a copy, or no better) | 4, 8, 12 | 160 to 2,000 | 0 to 0.07% | **0 in every setting** |
| 0.08 (just under the margin) | 8 | 370 / 1,000 / 2,000 | 0.3% / 0.3% / 0.1% | 0.2% / 0.1% / 0.1% |
| 0.10 (at the margin) | 8 | 370 / 1,000 / 2,000 | 0.7% / 0.7% / 1.2% | 1.0% / 0.7% / 0.5% |

With the margin inside the test, a head that is no better than the bar never
passes in any simulated setting, with or without confirmation: the margin
is 0.5 to 1.8 standard errors at the minimum evidence and grows with the
square root of the evidence. Repeated testing matters only at the margin,
and there the bound keeps a head's lifetime chance near alpha even over 300
epochs. Many heads are many chances only for heads that really are within a
hair of 0.1 points better, and seating one of those is no loss.

**Speed** (median epochs from the first label to taking the seat, sigma 8):

| d | 370 labels an epoch: pass and confirm / single | 1,000 an epoch: pass and confirm / single |
|---|---|---|
| 0.3 | 42 / 51 | 17 / 19 |
| 0.5 | 14 / 15 | 6 / 6 |
| 1.0 | 10 / 5 | 4 / 2 |

One rule: as fast for a head half a point better, twice as fast for one a
point better, 10-20% slower for one 0.3 better, with no phase, record or
lapse to track. (Simulated with the bound exactly as specified in 3.1,
integer logarithm included.)

**Residuals, accepted:**

- **Serving the best of several runs of the scored questions.** An operator
  can run the scored questions in several TDs and serve the best, which
  helps it through the picks check. As in v3 (INVARIANTS, best-of-N), it
  costs N times the calls, and the head must already have beaten the holder
  on odds, which no run can move (3.2). Closing it needs one run per epoch
  per TD identity, a later image change.
- **The holder withholding scored epochs** to slow a challenger's picks
  check. It is unpaid for each one, as in v3.
- **A copy after a long outage.** A head absent for more than G epochs no
  longer bars its copies. The defence is operational, as today (validators'
  published egress addresses and the axon firewall).

## 4. New entries are usable at once

### 4.1 The rule

An entry joins at epoch J (the first epoch whose menu lists it).

- **From J its labels are scored for every head**, and a head with a row for
  it may route scored questions and users to it (`picks_and_odds_v4` already
  prices every entry with a row).
- **A head without the row**, in epochs J to J + `ENTRY_ALLOWANCE_EPOCHS` - 1
  (20 epochs, a day), is scored on those labels as if its row said **q_e**:
  the entry's success rate on its proposal measurement, (k + 1) / (n + 2)
  over the answered calls, an integer in odds units written into the
  `MenuState` that adds the entry and fixed for the allowance.
- **From J + 20 a missing row is certainly wrong** (v3's rule).
- Seed entries have no allowance (they are on the menu from the first v4
  epoch); a request update to an entry starts none.

### 4.2 Why the measured rate, and the numbers

**Omitting a row gains nothing.** q_e is a single published number fixed
before any head is scored on the entry, and a row with zero weights and the
matching bias gives exactly it (the pinned threshold table reaches every
integer). So a head without the row is scored precisely as a head that
committed that one-number row. Deleting a real row can only beat keeping it
if the row is worse than a constant, and then the head could write the
constant instead: omission is never better than a row the head could have.

**Lacking a row costs only what a real row earns.** The head without the row
is scored as the least informed row consistent with the published
measurement: not certainly wrong, not 0.5, not ineligible.

The alternatives fail one side or the other:

| Substitute for a missing row | Omitting a weak row | Lacking a row |
|---|---|---|
| R (per entry and benchmark, fitted on the window's outcomes) | gains whenever the row is worse than R: 3 of the 25 heads below | costs little |
| Drop the entry's labels from that comparison | gains whenever the row is worse than the other head's | costs nothing, so the first good row cannot win |
| 0.5 | never gains | costs about 12 points a label more than a constant at the entry's rate |
| Certainly wrong | never gains | disqualifying: no allowance at all |
| **The measured rate q_e** | **never gains over a one-number row** | **costs what a real row earns over the base rate** |

From published epochs: menu B's seven entries joined at `e00026475` (each
then reached 38-117 labels over 35 epochs); 25 heads gave odds on at least
40 of their labels. Mean Brier loss per label, in points (lower is better):

| Epochs since join | Heads' rows | Entry rate, labels to date | Entry x benchmark rate, to date | R (fitted on the window) | 0.5 |
|---|---|---|---|---|---|
| 0-4 | 9.8 | 19.3 | 19.5 | 12.7 | 25.0 |
| 5-9 | 11.4 | 18.0 | 17.7 | 13.5 | 25.0 |
| 10-19 | 9.1 | 11.3 | 11.2 | 10.7 | 25.0 |
| 20-34 | 11.2 | 13.0 | 12.2 | 11.9 | 25.0 |

Every one of the 25 heads' rows beat even the best constant there could be,
the entry's rate over all its labels in hindsight, by 2.0 points a label on
average and 0.16 at the least. A 300-call measured rate is a constant no
better than that one. On the published field, then, deleting a row would
have cost every head; and the first heads with rows were ahead by 9.5 points
a label in the entry's first five epochs.

(As built, S2: those figures are reproduced from 35 named reveals in
`tests/test_allowance_rows.py`, per head and per label, unweighted. In the
score's own units, each label weighted by its exploration weight, 21 of the
25 heads beat the hindsight constant and 4 would have gained 0.42 to 0.86
points a label of those entries by deleting their rows; per row (one head,
one entry) 56 of 175 rows are worse than it. What holds for every row is
the identity: deleting a row is scored exactly as the one-number row at
q_e.)

**What that means for the seat.** Under v4's coverage part a new entry gets
160-260 labels in its first 4-8 epochs (about a tenth of the labels then,
`LIVE_MENU_V4.md` 8). A measured rate from 300 calls is close to the
hindsight constant above, so a good row is worth 2-3 points a label over it,
and a head with one gains about 0.2-0.3 Brier points of average gain over an
equal head without one in those epochs: more than the margin. The first miner with a good row can win at once, as
decided, and a head without the row loses exactly that much and no more.

### 4.3 A holder that lacks the row when the allowance ends

From J + 20 its labels of the entry are certainly wrong: 100 points a label
against R's 12 or so. At a sixtieth of the labels that takes about 1.5 points
off its skill, so within a few epochs R beats it and any challenger with the
row beats it on A. Then, by 3.6: the latest qualifier takes the seat that
epoch; if there is none the holder keeps the seat, unpaid while R beats it.
No ghost, no empty seat, no burn while a qualifier exists. A holder that adds
the row by succession (4.4) keeps everything.

### 4.4 Succession: adding rows keeps a head's place

A head H' is the **successor** of H when the same hotkey committed H before
it, both belong to the same coldkey, every row H has for an entry still on
the menu is in H' byte for byte (the row's weights and bias), and H' only
adds rows (or drops rows of entries that left the menu). It is checked with
per-row digests, `sha256(entry id || row bytes)`, published in the state for
every standing head, so a rebuild reproduces it.

A successor **is** its predecessor for the rules: its seat or history seat,
its precedence, its standing and its evidence. Its odds on H's rows are H's;
its new rows count from its own commit (3.2, condition 2). It applies to
every head, not only the holder: a contender that adds a row for a new entry
must not become junior to near-copies committed in between.

**Residual, accepted.** An earlier contender can fit a later head's new row
from its published odds and add it by succession, keeping its precedence;
the later head then passes it only by being better by the margin on
everything else. The row's author still wins if it beats the holder before
the copy lands, and the product loses nothing either way.

### 4.5 What feeds heads and the reference

- **The proposer's measurement** (300 labels, public at join): unscored for
  every head, as built; it seeds q_e, R (which reads every pooled and
  measurement label of the window, leave one epoch out), the entry's call
  cost and its pick cost's prior. Every head may train on it.
- **The entry's first exploration labels**: the coverage part's deficits send
  them (160-260 in 4-8 epochs); they are scored for every head from J, and
  they are the labels a row is judged on.

### 4.6 The join delay

Today an entry filed in epoch k is listed pending in `MenuState(k + 2)`,
measured in k + 2 .. k + 6 (60 questions an epoch), decided while scoring
k + 6, and joins at k + 8: about 9.6 hours.

The proposer cannot choose its measurement questions however many epochs
the measurement takes: they are drawn by `HMAC(epoch nonce, proposal id)`
from nonces that do not exist when it files. What sets the number of epochs
is time inside the TD: 300 extra calls in one epoch, on top of 112
exploration calls and possibly 200 scored ones, at 24 in flight, does not
reliably fit an epoch for a dear reasoning entry at minutes a call. Two
epochs of 150 do.

**Recommendation:** `PROPOSAL_MEASURE_EPOCHS` 2 (was 5). An entry then joins
at k + 5, three epochs (3.6 hours) sooner. The two epochs of `MENU_LAG` stay:
a menu is committed before the epoch it applies to. The make-up bound is
`PROPOSAL_MAKEUP_EPOCHS` 3, `MENU_LAG` + 1 (decided with S2; S6 had built 2):
any one step the root skips is made up, whichever measurement epoch it hits.
A step skipped for the first measurement epoch is made up in k + 5, one
skipped for the last in k + 6, the first menu the next step writes and the
bound; two skipped steps are not, and the proposal leaves with no record or
cooldown, to be filed again. One way a proposer
could touch its own measurement is to let calls
fail after seeing their timing: the 95% answered gate bounds that to 15
calls, at most about 0.05 on q_e, which is 0.25 points a label of the entry,
for 20 epochs, on heads that have no row.

## 5. The label budget

### 5.1 What a call costs, and what a hotkey pays

At measured costs a hotkey-epoch of exploration under setting B of
`LIVE_MENU_V4.md` (48 calls) costs $0.32 (0.28-0.37), $0.0067 a call; the
confidential VM is $4.93 a day, $0.25 an epoch; a day is 20 epochs. The
scored questions are optional, so a hotkey's bill is exploration plus the VM
until it is close to the seat, and about $4.65 an epoch more (200 questions
at the v4 mix) while it answers them.

| Daily cost per hotkey | Calls a hotkey-epoch |
|---|---|
| about $11 | 48 |
| about $20 | 112 |
| about $40 | 262 |

### 5.2 What labels buy

Model: H hotkeys, each its own operator; 3% of calls fail; a comparison
leaves out both heads' operators' labels (3.2); design effect 1.11; per-label
SD 8 points; rule A as specified (1,500, 0.1, the bound). The labels a test
needs at the median: 5,480 for a head 0.5 points better, 20,430 for 0.3,
86,480 for 0.2 (which a 48-epoch window holds only at the large budgets).
"Every entry at 100" is W0's calibration (about 7,500 pooled labels from an
empty window, 60 entries); "a new entry at 100" includes the two-epoch lag.

| $ a day | H | Labels an epoch (pool / one test) | SE of a test an epoch (pts) | Epochs to seat a head 0.2 / 0.3 / 0.5 better | Epochs until every entry has 100 | A new entry to 100 | Field $ a day |
|---|---|---|---|---|---|---|---|
| 11 | 5 | 202 / 116 | 0.78 | 825 / 195 / 53 | 37 | 7 | 56 |
| 11 | 10 | 396 / 310 | 0.48 | 310 / 74 / 20 | 19 | 5 | 113 |
| 11 | 15 | 590 / 504 | 0.38 | 191 / 45 / 13 | 13 | 4 | 170 |
| 11 | 20 | 784 / 698 | 0.32 | 138 / 33 / 9 | 10 | 3 | 226 |
| 11 | 30 | 1,172 / 1,086 | 0.26 | 89 / 21 / 6 | 6 | 3 | 339 |
| **20** | **5** | **517 / 306** | **0.48** | **315 / 75 / 20** | **15** | **4** | **100** |
| **20** | **10** | **1,026 / 815** | **0.30** | **118 / 28 / 8** | **7** | **3** | **200** |
| **20** | **15** | **1,536 / 1,324** | **0.23** | **73 / 18 / 5** | **5** | **3** | **300** |
| **20** | **20** | **2,045 / 1,833** | **0.20** | **53 / 13 / 4** | **4** | **3** | **400** |
| **20** | **30** | **3,064 / 2,852** | **0.16** | **34 / 8 / 3** | **2** | **2** | **600** |
| 40 | 5 | 1,245 / 742 | 0.31 | 130 / 31 / 9 | 6 | 3 | 200 |
| 40 | 10 | 2,482 / 1,979 | 0.19 | 49 / 12 / 4 | 3 | 2 | 400 |
| 40 | 15 | 3,718 / 3,216 | 0.15 | 30 / 8 / 2 | 2 | 2 | 600 |
| 40 | 20 | 4,955 / 4,452 | 0.13 | 22 / 6 / 2 | 2 | 2 | 800 |
| 40 | 30 | 7,428 / 6,926 | 0.10 | 14 / 4 / 1 | 1 | 2 | 1,200 |

The published field today is 5 operators and 8 hotkeys, and 160 pooled labels
an epoch.

### 5.3 Recommendation: 112 calls a hotkey-epoch, about $20 a day

- **It is where labels stop being the constraint.** At ten hotkeys a head
  half a point better is seated in about 8 epochs (10 hours) and every entry
  is measured to 100 labels in about 7; at $11 those are 20 and 19. Past
  $20 the other minimums bind: the picks check's 1,000 questions are five
  epochs whatever the label budget, and a new entry's two-epoch lag is fixed.
- **It is affordable.** A challenger's day goes from about $119 in v3 to $20,
  a sixth; the confidential VM, a quarter of it, is a fixed cost that makes
  the first labels the dearest ($11 buys 48 calls, $20 buys 112). Ten
  hotkeys spend $200 a day, about 6% of the miner pot at the prices of
  `MINER_ECONOMICS.md`; thirty spend $600 (17%). At $40, thirty hotkeys would
  spend a third of the pot for a test error 1.6 times smaller.
- **The cost of winning does not rise with it.** At ten hotkeys a head 0.3
  points better spends about $41 on labels and the VM to be seated at $11 a
  day (3.7 days), $28 at $20 (1.4 days), $24 at $40 (0.6 days), plus $25-45
  of scored questions in each case: faster seats cost the winner about the
  same. The budget's real price is what the field pays while nobody wins.

**As constants** (scaling `LIVE_MENU_V4.md` setting B by 112 / 40, the shared
set removed, 2.3 item 33): a private set of `EXPLORE_PRIVATE_SIZE_V4` 112 =
68 cost-aware + 44 coverage calls; cost-aware expected-spend cap $0.55 and
coverage cap $1.10 (the same headroom over the measured $0.35 and $0.43 as
v3's caps have); `EXPLORE_COLDKEY_CAP_V4` 168 (1.5x); coverage target 100;
damping 1,400 labels (setting B's 500 scaled by the coverage calls). These
scale a replayed setting; W0's model is re-run on them before they are
pinned.

### 5.4 What stops a hotkey that pays for labels and supplies junk

- **It cannot supply an answer it did not get.** Every label is a model call
  made inside its TD through the metered proxy and graded by the pinned
  grader; the proof binds them (I8). It chooses neither the question nor the
  entry (both from the epoch nonce and its hotkey).
- **Failures say nothing and are bounded.** A failed call is dropped; a proof
  with more than 34% of its exploration failed is refused outside an outage
  epoch, and a refused proof is a missed epoch.
- **Impossible answers are left out.** An operator whose answers are more
  than four SD from what R expects of those entries on those kinds of
  question has its labels left out of the epoch's pool.
- **Its own labels never judge its own head** (3.2), so the one choice it has,
  which run to serve, moves nothing it is paid for.

### 5.5 What stops one operator's many hotkeys dominating the pool

One vote per coldkey per label, and the operator cap: one coldkey's labels
are cut to 1.5 hotkeys' worth, keeping the lowest keyed digest, so it cannot
choose which. An operator that registers more coldkeys pays a full hotkey
($20 a day) for each, and what it adds is still attested calls it did not
choose; volume buys it no say over any outcome, and its labels never judge
its own heads.

### 5.6 W0's model re-run on these sizes (package S6, 2026-11-19)

The constants of 5.3 are built (`config.py`; section 8). W0's model
(`a_sizes.simulate` of the 2026-11-18 replay, its 24 published v3 epochs'
measured costs, failure rates and hotkeys per operator; 200 seeded runs, 144
epochs) was re-run on them: no shared set, 68 cost-aware calls under $0.55,
44 coverage calls, damping 1,400, coverage cap $1.10, operator cap 168.
**Model** figures; the replay's inputs are unchanged.

Spend, one hotkey-epoch, at measured costs: the cost-aware part $0.346
(exponent 1/2, as at 32 calls), uniform coverage $0.433, steady state
**$0.78 (0.67-0.91)**, so $1.03 an epoch, $20.6 a day, with the VM, as 5.3
says. The first epochs from an empty window peak at $1.04. **The coverage cap
no longer binds on a single dear join**: the dearest entry ($0.129 a call)
joining a full window draws $0.81 of damped coverage, under $1.10, so the
worst epoch is $1.15 (it was $0.56 at 40 calls, with the cap binding). At the
seed menu's costs the cap binds only when a dozen or more of the dearest
entries are short at once. Design effect 1.11, largest weight 4.8, as at 40
calls.

Against 5.2's $20 rows (the model with one hotkey per operator, as 5.2
assumes; ranges 5-95%):

| H | Pooled labels an epoch: 5.2 / model | Every entry at 100: 5.2 / model | A new entry to 100: 5.2 / model |
|---|---|---|---|
| 5 | 517 / 555 | 15 / 15 (15-16) | **4 / 7 (6-9)** |
| 10 | 1,026 / 1,110 | **7 / 8** (8-8) | 3 / 3 (2-4) |
| 15 | 1,536 / 1,665 | **5 / 6** (5-6) | **3 / 2** (2-2) |
| 20 | 2,045 / 2,221 | 4 / 4 (4-5) | **3 / 2** (2-2) |
| 30 | 3,064 / 3,331 | **2 / 3** (3-4) | **2 / 1** (1-1) |

Bold where the figures differ. The model pools 7-9% more labels than 5.2
(it fails calls at each entry's measured rate, not a flat 3%), fills an
empty window an epoch slower at 10, 15 and 30 hotkeys, and brings a new entry
to 100 labels three epochs later at five hotkeys and an epoch sooner at 15 or
more (its count starts at the join and reads the window as published two
epochs back, against 5.2's two-epoch lag added to a fill rate). With the
measured hotkeys per operator (2.9 on average, cut to 168 labels per coldkey)
and 5 / 7 / 10 operators: 726 / 1,015 / 1,452 labels an epoch, every entry at
100 in 12 / 9 / 6 epochs, a new entry in 5 / 3 / 2. The seat-speed columns
of 5.2 were not re-modelled.

## 6. The memorisation check

### 6.1 How often questions repeat today

From published epochs (a question counts as published once any reveal
lists it: a scored question, a graded cell, an exploration answer or a
pooled label; v1 reveals were not read, so these are lower bounds):

- **68% of the v3 pool has been published**: 33,007 of 48,320 questions,
  after the 117 v2 epochs and the first 44 v3 epochs.
- **71% of pooled labels in the last ten epochs were on a question already
  published**; 29%, about 48 an epoch at 164 labels, were first appearances.
  The same (question, entry) cell came back in 18 of 8,235 labels.
- **Relative to each head's commit**, 54% of the labels heads were scored on
  were on questions not yet published when the head was committed (its
  first epoch, for heads committed before v3).
- **No head shows a memorisation gap.** A head's skill over R on questions
  published before its commit against its skill on the others, compared
  within each benchmark: over 44 heads the gap's z-score has mean -0.50, SD
  0.82, maximum +1.43. Compared without stratifying, low-skill heads show
  "gains" on repeated questions of 2 to 17 points; those come entirely from
  which benchmarks repeat (small sets repeat sooner), so the check must
  compare within benchmark.

A linear head on the locked encoder has too little capacity to memorise the
pool, which is why the encoder is locked, and the data agrees. The check is a
tripwire: it costs honest heads nothing, it refuses gross memorisation, and
it matters most once the locked encoder is swapped for a bigger one.

### 6.2 The check

For a head H, split its labels against R (3.2: drawn after the row's commit,
not its own operator's) into **unseen** (the question had not been published
when the row was committed) and **seen**. The **gap** is, within each
benchmark, H's mean skill over R on seen labels minus that on unseen ones,
averaged over benchmarks with weights `1 / (1/n_seen + 1/n_unseen)`
(benchmarks with fewer than 5 of either are left out).

**H is refused** when the bound of 3.1 passes on the gap with margin
`MEMO_MARGIN` 1.0 Brier point, with at least `MEMO_MIN_LABELS` labels in
each part (500 as built, was 1,000: 6.3 says why; n0 = the same). The items
are the **questions**, not the labels: every hotkey answers every fresh
question (6.3), so one question's labels move together, and each question's
labels enter as one item, their summed weighted skill scaled by the label's
share of the gap, in the stratum of its benchmark and part. The variance is
then the within-stratum spread of the questions, and a benchmark is left out
with fewer than 5 questions in either part. (As built, S5: the scaling is a
common integer multiple of every coefficient, so the verdict is the bound's
inequality on the gap itself, exactly; `memo_v4.evidence_from_parts`.)

A refused head cannot take the seat, is not a contender, is not paid, and a
refused holder opens its seat (3.6). The refusal lifts when the bound no
longer passes as the window moves.

- **False refusals.** On published epochs: none of 44 heads (the largest gap
  is +1.86 points with an SE of 3.1, on a head with negative skill; the next
  is +1.29 with an SE of 0.91). In the model, an honest head (true gap 0)
  needs a measured gap of about 2.6 to 5 points to be refused at the sample
  sizes seen (SE 0.4 to 0.9), four or more standard errors: far below one in
  a thousand.
- **Power.** Half the time it refuses a head whose skill on seen questions
  exceeds its skill on unseen ones by about 2.1 points with 3,000 labels in
  each part, 3.3 points with 1,000 (per-label SD of skill about 11). That is gross memorisation: a head that knows
  outcomes, not one that has learned a little about a question's difficulty.

(As built, S5, both re-measured; 6.3 has the power under the rationed
reserve. Published epochs, from `data/observations/
memo_v4_labels_e00026466-e00026509.json` with first publication read from
every named reveal from `e00026189`, v1 included, and pinned in
`tests/test_memo_v4.py`: 53 heads have a comparison and none is refused;
the gap's z-score has mean -0.28, SD 0.85, maximum +1.93; the largest gap
is +4.04 points (SE 2.09, on 74 seen and 38 unseen labels) and no head's
lower bound comes within 2.6 points of the margin. Pooled across
benchmarks, two heads would be refused, at +19.2 and +3.7 points: the
within-benchmark comparison is what spares them. The per-label SD of skill
over R is about 20 points (20.5 unweighted, 24 weighted, every head of the
published epochs), not 11, and a head's labels on one question correlate
(intra-question correlation 0.29 over 2,649 questions with two or more of
a head's labels), so the power figures above are optimistic by about a
factor of two.)

### 6.3 What keeps it meaningful: a rationed reserve of unpublished questions

The check needs unseen labels, and drawing questions publishes them. With
draws uniform over the pool, the unpublished share decays with time constant
(pool size) / (questions drawn an epoch): at the recommended budget and ten
hotkeys, about 48,320 / 1,300 = 37 epochs. So v4 does not draw uniformly:
the unpublished questions are a **reserve**, spent only by one rationed
draw, and **no question generator is built** (the owner's decision,
2026-11-19; it stays a later option, below). This replaces the design's
`FRESH_PER_HOTKEY` 10, which spent the reserve in about a week at ten
hotkeys.

**The reserve, measured** (`data/observations/
published_v4_e00026189-e00026509.json`). A question is published when a
reveal publishes an outcome on it: a scored question, an exploration answer
(failed or not, pooled or not) or a pooled label. Reading every reveal the
store holds for netuid 70, e00026189..e00026509 (v1 160, v2 117, v3 44: v1's
pool shares AIME, IFEval, MMLU and MATH levels 3-5 with v4's, and its
reveals published 6,737 of v4's questions first, 958 of them never
published again), 32,106 of the 48,320 are published and **16,214 are
not**: easy maths 8,020, knowledge 2,511, code 1,931, instruction 1,590,
hard 1,183, everyday 880, maths 99. (6.1's 33,007 could not be reproduced:
the v2 and v3 reveals publish 31,148 of the pool, and no pool question id
appears in any reveal field but the three read here.) **v3 keeps spending it until v4 activates**: about
140 questions an epoch now (118-194 over the last ten), falling as the
reserve shrinks; on the v3 mix the model leaves about 7,000 after one more
week of v3 and 4,200 after two, three quarters of it easy maths (GSM8K
train, MATH levels 1-2), the maths, everyday, instruction and code reserves
nearly gone. Each further week of v3 costs the check about half its reserve.

**The rule.** For epoch E, read the published set as of E - 2 (every node
holds that reveal at E's boundary, as it holds the menu committed in it);
the reserve is the pool's questions not published through E - 2, R of them.

1. **The fresh draw.** The epoch spends F = ceil(R / `FRESH_RESERVE_EPOCHS`),
   1,200, questions of the reserve, drawn by the epoch nonce on the scored
   mix over the categories that still have a reserve
   (`memo_v4.fresh_questions`: the private draw keyed by the epoch nonce
   alone). **Every hotkey's private set holds all F of them**, each sent to
   an entry drawn by the hotkey's own part and shares, as any private
   question is. The number of hotkeys changes how many labels the fresh
   questions give, never how many questions are spent: the reserve shrinks
   by a 1/1200 share an epoch, rounded up, whatever the field.
2. **Everything else reads published questions.** The scored slice, the rest
   of each private set (112 - F questions) and proposal measurement draw
   only questions published through E - 2. A category with fewer than 463
   published questions (the slice, a private set and a measurement: what one
   draw can take or exclude) is left out of those draws, its share going to
   the other categories as the slicer does for an absent category, so no draw
   comes back short; it is reached by the fresh draw until enough of it is
   published. A category whose reserve is empty gives its fresh slots to the
   others.
3. **Weights stay exact.** The private set keeps its 112 questions and its
   68 / 44 split by the miner's keyed rank, so a fresh question is in each
   part with the same chance as any other, and its entry is drawn by the
   part's shares: every pair weighs 1 / (M pi_private) exactly, as before.
   No miner chooses which of its questions are fresh: the set is the epoch's,
   and a proof that answers another set is refused by the exploration check.
4. **The published set is state**: per pool question the first epoch a reveal
   published it, seeded at activation from a pinned file built from named
   reveals (`data/published_v4.json`, its unpublished ids by set; every
   other question of a named set is published as of the seed's last reveal),
   folded from each reveal by the live validator and by a rebuild alike, its
   digest in each v4 reveal and in `v2_state` (`memo_digest`). The seed must
   be re-observed through the reveal of the activation epoch minus two: the
   v3 epochs in between publish questions too.

**What it gives** (model: per-label SD of skill 20 points, intra-question
correlation 0.29, the bound of 3.1 with questions as items, 48-epoch window,
300 to 2,000 seeded runs of 300 epochs; m is the number of other operators'
hotkeys answering, each giving one label per fresh question).

| R at activation | F at first / on day 60 | Left on day 60 | Reserve empty after |
|---|---|---|---|
| 16,214 (activation now) | 14 / 5 | 5,602 (35%) | 193 days |
| 7,000 (one more week of v3) | 6 / 2 | 2,214 (32%) | 145 days |
| 4,200 (two more weeks) | 4 / 1 | 1,151 (27%) | 118 days |

Any reserve of 1,200 or more lasts at least 60 days with no new questions,
and one of 2,400 or more keeps at least a quarter of itself to then.

At R = 7,000 a head gets about F x m unseen labels an epoch: 54 at ten
operators (m = 9) and 30 at six (m = 5), so 500 in 10 and 17 epochs; on day
60 (F = 2), 18 and 10 an epoch, 500 in 28 epochs at ten operators, while six
operators' 48-epoch window holds 480 and the check waits for the field to
grow. A question answered by m operators is worth m / (1 + (m - 1) 0.29)
independent labels, 2.7 at m = 9: sharing the fresh questions buys 2.7
times the evidence of giving each to one hotkey, for the same reserve.

| F, m | Gap (seen minus unseen, points) | Refused within 300 epochs | Median epochs to the refusal |
|---|---|---|---|
| 6, 9 | 0 | 0 of 2,000 | |
| 6, 9 | 1 (the margin) | 0.6% | |
| 6, 9 | 2 / 3 / 4 / 6 | 26% / 94% / 100% / 100% | 144 / 79 / 34 / 13 |
| 3, 5 | 0 | 0 of 2,000 | |
| 3, 5 | 1 (the margin) | 0.9% | |
| 3, 5 | 4 / 6 / 10 | 97% / 100% / 100% | 75 / 34 / 34 |
| 10, 9 | 4 / 6 | 100% / 100% | 20 / 12 |

So the false-refusal rate is the design's: no honest head refused in any
setting, a head exactly at the margin under 1% over 300 epochs. A head with
4 points of skill from published outcomes is refused in under two days at
ten operators, 6 points in under a day.

**The minimum is 500, not 1,000.** At 1,000 the check is out of reach of the
48-epoch window once F is 2 or 3 and the field is a few operators (the window
holds 480 to 720 unseen labels), and in the model 1,000 bought nothing in
false refusals, which the margin and the bound set. What 500 gives up is
only a refusal at the very minimum: there the smallest gap the bound can pass
is about 6.9 points against 5.2 at 1,000 (m = 9); past it the evidence and
the bound are the same.

**When a release adds a standard benchmark set**, its questions enter the
pool unpublished, so R rises by the set's size at the release and F with it
(5,000 questions add about 4 an epoch), with no constant to change: the seed
names the sets it was observed on and a set it does not name is unpublished
whole, and after activation a new question is simply not in the folded
state. Until some of a new set is published it is reached only by the fresh
draw; a new set needs a weight in the pinned mix in any case (a rebuilt mix,
as any change of sets does). **A generator** of fresh maths and code
questions (problems from the epoch nonce, answers computed or tests generated
alongside, graded by pinned checkers, unpublished by construction) remains
the way to a reserve that never runs down: not built, a later option, most
useful before the encoder grows (7). Without growth the reserve runs down in
four to six months and the locked encoder's capacity is again the only
defence, as it is today.

**Residual.** The check sees outcomes the subnet published. A miner can call
models on reserve questions itself, outside the subnet, and memorise those
outcomes: at about $0.0067 a call (5.1), labelling a 7,000-question reserve
on the dozen entries a router mostly uses costs about $560. The fresh draw's
order is unpredictable (the nonce), so that buys knowledge of questions
drawn later, not of a chosen few.

## 7. What the product needs, and the encoder

The product needs three things from this, and gets them:

1. **The holder's head**, published at once by hash with each reveal
   (`LIVE_MENU_V4.md` 11.3); parity is checked against the reveal's routing
   of the scored questions, which the holder answers every epoch.
2. **The cost table at $1 per correct answer**: the per-entry pick cost in
   each epoch's `MenuState`, which is what the subnet's picks use at
   `BENCHMARK_LAMBDA` 1 and what rule B is priced at. fugal-core's selector
   must read it (`LIVE_MENU_V4.md` 11.2, W10).
3. **New entries routable at once.** A pick may use any entry with a row
   from the epoch it joins; when the holder has a row for a new entry,
   users reach it. q_e and the allowance are scoring only; the product never
   reads them.

**Nothing here depends on the encoder's size.** Rules A and C, the
allowance and the memorisation check read integer odds per label; rule B
reads grades and costs; succession compares row bytes whatever their shape;
the window stores one int16 per label per head. Swapping the locked encoder
is a protocol boundary like the v4 activation: a new pinned embeddings file,
a new head format, every head new, the salted tie for precedence, an empty
seat and the minimums. A bigger encoder raises what a head can memorise, so
the memorisation check and pool growth (6.3: the rationed reserve runs down
in months, and a generator is the later option) should be in place before it.

## 8. Constants

All compiled (not read from the environment), classified in
`check_safety_invariants.py` and in the consensus digest.

| Constant | v3 | v4 | Where |
|---|---|---|---|
| `SEAT_WINDOW_EPOCHS` (was `CHAMPION_WINDOW_EPOCHS`) | 48 | 48 | 3.2 |
| `ODDS_MARGIN` (was `BRIER_MARGIN`) | 0.1 | 0.1 | A |
| `ODDS_MIN_PAIRS` | 1,500 | 1,500 | A |
| The bound: alpha, `A_ALPHA`, `LN2_UP` | 99% bootstrap, 2,000 resamples | 1%: 9211/1000, 6932/10000 | 3.1 |
| `ODDS_CONFIRM_PAIRS`, `CHAMPION_CONFIRM_QUESTIONS`, `CHAMPION_CONFIRM_OPEN_EPOCHS` | 1,500, 1,500, 5 | none | 2.1 #3-4 |
| `SLICE_SIZE_V4` | 300 | 200 (the mix rebuilt for it) | B |
| `PICKS_MIN_QUESTIONS` (was `CHAMPION_MIN_PAIRED_QUESTIONS`) | 1,500 | 1,000 | B |
| `PICKS_MARGIN` (was `NV_NONINFERIORITY`) | 0.5 (fixed 99%) | 1.5 (the bound) | B |
| V, the value of a correct answer | $0.49 | 1 / `BENCHMARK_LAMBDA` = $1.00 | B |
| `CHAMPION_HARD_WEIGHT` | 2 | none (1) | B |
| `PICK_COST_PRIOR_CELLS` | none | 30 (entry), 20 (entry x category) | B |
| `FAIL_MAX` (was `EXPLORE_FAIL_MARGIN`, relative) | 0.34 relative | 0.34 absolute, exploration and scored calls, waived in an outage epoch | 3.7 |
| `STANDING_EPOCHS` | ghosts 3 / 24 / 8 | 3 | 3.7 |
| `V2_BURN_IN_QUESTIONS`, `CHAMPION_MIN_ACCURACY`, the high-water marks' Wilson 95% and 100 hard questions | 1,500, 0.30, 0.95, 100 | none | 2.1 |
| `ODDS_REF_PRIOR` | 20 | 20 | R |
| `ENTRY_ALLOWANCE_EPOCHS` (was `ENTRY_GRACE_EPOCHS`) | 24 | 20 | 4 |
| `PROPOSAL_MEASURE_EPOCHS` | 5 | 2 | 4.6 |
| `MEMO_MARGIN`, `MEMO_MIN_LABELS` (n0 the same) | none | 1.0, 500 a part (S5; was 1,000) | 6 |
| `MEMO_MIN_QUESTIONS` (a benchmark's least questions in each part) | none | 5 | 6.2 |
| `FRESH_RESERVE_EPOCHS` (was `FRESH_PER_HOTKEY` 10) | none | 1,200: ceil(R / 1,200) fresh questions an epoch for the field | 6.3 |
| `EXPLORE_SHARED_SIZE` (v4) | 8 | 0 | 2.3 #33 |
| `EXPLORE_PRIVATE_SIZE_V4` = cost-aware + coverage | 40 = 24 + 16 | 112 = 68 + 44 | 5 |
| `EXPLORE_SPEND_CAP_USD_V4`, coverage cap | $0.25, $0.40 | $0.55, $1.10 | 5 |
| `EXPLORE_COLDKEY_CAP_V4` | 60 | 168 | 5 |
| `EXPLORE_COVERAGE_DAMP_PAIRS_V4` | 500 | 1,400 | 5 |
| `EXPLORE_PRIVATE_Z`, `EXPLORE_PRIVATE_MIN_TRIALS` | 4, 50 | unchanged | 3.7 |
| `PAYOUT_SEAT_SHARES` | 0.85 / 0.10 / 0.05 | unchanged | 3.8 |
| `MAX_WEIGHT_DELTA` | 0.3 | not applied to a v4 vector | 2.2 #28 |
| `HISTORY_KEEP` | 4 | 4 | |

## 9. Invariants: add, change, delete

Written into `docs/INVARIANTS.md` with the build, one entry per package; v2
and v3 entries stay while a v3 epoch can still be rebuilt, then go.

**Add**

- *I1/I3 — v4: one bound.* Every v4 seat decision is the bound of
  `champion.lower_bound` on exact integer items: a margin, a minimum, the
  stratified variance, and the pinned rational constants `A_ALPHA` and
  `LN2_UP`; no random stream. For a true mean at the margin the chance of
  ever passing is at most about alpha over any number of epochs.
- *I3/I4 — v4: the labels a head is scored on.* A comparison reads only
  pooled labels of the last `SEAT_WINDOW_EPOCHS` epochs drawn after both rows
  it reads were committed and supplied by neither head's operator. A head's
  odds are computed for every epoch in the window whether or not it served;
  missed epochs are computed when it is next verified and published then.
  Nothing an operator does with its own calls or proofs moves its own head's
  evidence.
- *I3 — v4: the seat in three rules.* The bar is the standing holder that R
  does not beat and the memorisation check does not refuse, else R. A
  qualifier beats the bar on odds (A), passes the picks check against the
  holder (B), and beats every earlier contender (C); the qualifier committed
  last takes the seat; otherwise the holder keeps it. A seat is never emptied
  without a replacement. Exact copies are not heads.
- *I3 — v4: new entries are usable at once.* From its join epoch an entry's
  labels are scored for every head and any head with a row may route to it;
  a head without the row is scored at the entry's measured rate (fixed in
  `MenuState`) for `ENTRY_ALLOWANCE_EPOCHS`, then certainly wrong. Omitting a
  row is scored exactly as the one-number row at that rate. A successor (same
  hotkey and coldkey, every existing row byte-identical) is its predecessor.
- *I3 — v4: the memorisation check.* A head whose skill on questions
  published before its commit exceeds its skill on the others, within
  benchmark, by more than `MEMO_MARGIN` under the bound is refused. An
  epoch spends ceil(R / `FRESH_RESERVE_EPOCHS`) unpublished questions, drawn
  by its nonce and held by every hotkey's private set; every other draw
  reads published questions only.

**Change**

- *I3 — v4 grace is per entry*: replaced by "new entries are usable at once".
- *I4/I5 — v4 exploration measures every entry*: the shared set is removed,
  the private set is 112 calls with the caps of section 8, the coldkey cap
  168, the fresh reserve added; an odds-only miner's spend is its
  exploration, bounded by the caps.
- *I3 — seats, not shares*: for v4 epochs a seat pays while live, not beaten
  by R and not refused; no ghost keeps seat 1; no weight ramp.
- *I1 — every scoring constant is in the digest*: the constants of section 8.

**Delete for v4 epochs**: the confirmation entry; the E'' miss rule; ghosts
and resume; the high-water marks and the accuracy floor; the copy rule's
senior bars (ownership and the salted tie stay); the first-fill ranking; the
net-value guard's v3 description; v1 dedup's mention for v2+.

## 10. Tests, attack cases, determinism

| Rule | Tests | Attack cases (miner suite or `tests/test_v4_seat_attacks.py`) |
|---|---|---|
| The bound | integer-exact and order-independent; equals the float formula within one unit; a mean at the margin never passes; fewer than the minimum never passes; the integer log bound is never below ln; seeded simulation at the margin passes at most 1.5% over 300 epochs | 30 heads with no edge over 300 simulated epochs: none seated |
| Labels | a head that missed an epoch gets the same odds as if it had served; a label voted by a head's owner is not read for it; a label drawn before a row's commit is not read; measurement labels never scored; late odds rebuild identically | an operator re-running its exploration, or failing its own calls, moves its own head's evidence by exactly 0; withholding any pattern of epochs changes no head's evidence |
| A | a copy never beats its original; a head twice the margin better passes; a successor's odds equal its predecessor's | perturbed, retrained and odds-distilled copies of an earlier head (the heads-publication scenarios, re-run): the original always keeps precedence |
| B | NV units are exact; identical heads have NV 0; a failed call is 0 grade and 0 cost; V is read from `BENCHMARK_LAMBDA` only; a missed answered epoch restarts the run; the open-seat path asks only for 1,000 answered questions | a head with better odds and picks two points worse never takes |
| C | equal independent heads: the earlier takes; a contender that stops answering stops barring after G epochs; one shown to route worse stops barring; the latest qualifier is unique and order-independent | an earlier head that never answers the scored questions bars nobody |
| Seat and pay | a holder missing one or two epochs keeps the seat; after three the latest qualifier takes it that epoch; with none the holder keeps it; R beating the holder opens the seat; 85 / 10 / 5 with unpaid shares to the holder; no ramp | a copy of the holder after a G-epoch outage (the documented residual) |
| New entries | an omitted row scores exactly as the bias-only row at q_e; certainly wrong from J + 20; a holder that adds a row by succession keeps seat, precedence and evidence; a byte change makes a new head; a successor by another coldkey inherits nothing; picks may use a new entry from J | deleting a weak row during the allowance gains nothing; a proposer failing measurement calls moves q_e at most as bounded |
| Memorisation | the stratified gap is exact; a synthetic honest head is not refused; a synthetic memoriser (outcomes known on seen questions) is refused; the fresh reserve draws only unpublished questions and is a function of published state | a memoriser cannot take or hold the seat |

Determinism (`scripts/check_determinism.py`, every input reversed under
`--perturb`): `bound_v4` (the bound on adversarial integer items);
`seat_v4` (twenty epochs with misses and late odds, a new entry joining with
heads with and without rows, a succession, an open seat filled, a holder
lapsing); `memo_v4` (the gap and the published-set fold); `pick_cost` (the
earlier design's W13).

## 11. Build packages

These replace the earlier seat design's W12 (guard), W14 (optional slice),
W15 (seniority among challengers), W16 (succession, handover, routing grace)
and W17 (payout split, dropped: no change). W13 (the pick-cost table) is
kept as designed there. Every package adds its invariant text and
determinism stage in the same change.

| | Package | Depends on | Main files | Checks |
|---|---|---|---|---|
| S1 | **One bound** | none | `champion.py` (`lower_bound`), `config.py`, `fingerprint.py` | `bound_v4` |
| S2 | **Labels and odds**: label eligibility (after both rows, other operators' labels), odds for missed epochs, the allowance and q_e, certainly wrong after; v4 grace masking removed | S1; W1 (`MenuState` q_e field), W4, W7 | `odds_score.py`, `v2_epoch._pool_and_odds`, `menu_state.py`, `routing_protocol.py`, `v2_rebuild.py`, reveal fields | part of `seat_v4` |
| S3 | **Picks**: rule B at the table and $1, hard weight 1, the answered run, the optional scored questions in the proof, the absolute failure bound | S1; W13; W3 (proof schema) | `champion.py`, `tee/proof.py`, `tee/verify.py`, `tee/harness.py`, `neurons/miner.py`, `v2_epoch.py` | picks tests; miner attack suite |
| S4 | **Seat**: the bar, qualifiers, contenders, standing, succession by row digests, ownership kept, pay eligibility, the published state | S1, S2, S3 | `champion.seat_step` (v4), `copy_rule.py`, `payout.py`, `v2_epoch.py`, `v2_rebuild.py`, `v2_sync.py` | `seat_v4` |
| S5 | **Memorisation**: the published-set state and its pinned seed, the gap, the refusal, the fresh reserve | S1, S2; W5 | `odds_score.py`, `epoch_plan.py`, `explore_weights.py`, a seed script (as built: `memo_v4.py`, `epoch_plan.py`, `v2_epoch.py`, `scripts/build_published_v4.py`) | `memo_v4` |
| S6 | **Budget and join**: the sizes and caps of section 8, no shared set, the mix rebuilt for 200 questions, two measurement epochs; W0's model re-run on them | W1, W5 | `config.py`, `explore_weights.py`, `epoch_plan.py`, `scripts/build_mix_v4.py`, `menu_state.py` | `explore_weights`, `token_means` stages |

Order: S1 first; S2, S6 and W13 in parallel; S3 after W13; S4 after S2 and
S3; S5 after S2. The activation package (W11) gains rehearsal scenarios: an
odds-only miner starting the scored questions and taking the seat; a holder
missing two epochs and keeping it, then three and losing it to the latest
qualifier; a new entry joining with heads with and without rows; a holder
adding a row by succession; a fill at activation.

## 12. What is deleted

From the v4 path (the v2 and v3 code goes too once no rebuild can reach a v3
epoch, 48 epochs after the activation):

- `champion.py`: the bootstrap and its seeds; `Pending`, `_live_pending`,
  `confirm_from` and the open period; `Ghost`, ghost upgrade and resume; the
  high-water marks, `_passes_gates`, the accuracy floor; `solo_value` and
  `solo_skill` as first-fill rankings; `_no_full_menu`; the anchor-spend
  and `V = accuracy_value x anchor` paths.
- `evidence.py`: the E'' burn-in counter and miss rule for v4.
- `copy_rule.py`: `comparison_set`, `seat_holders`, `senior_bars` and the
  retire windows (ownership, `classify`, seniority and the fold stay).
- `dedup.py`, `head_distance.py`: off the v4 path (v1 only).
- The v1-frame and agreement detectors feeding `flagged` for v4; the shared
  exploration set in v4.
- `routing_protocol.grace_excluded` / `scored_menu` for v4 (the allowance
  replaces them).
- `rewards.cap_weight_change` for v4 vectors.
- Constants: every `CHAMPION_*` seat constant except the window, `V2_BURN_IN_QUESTIONS`,
  `ODDS_CONFIRM_PAIRS`, `NV_NONINFERIORITY`, `CHAMPION_VALUE_PER_CORRECT_USD`,
  `EXPLORE_DISAGREE_*` (the plausibility check's minimum sample gets its own
  name), `ENTRY_GRACE_EPOCHS`.
- Reveal and state fields for v4: `pending`, `confirm_from`, `ghosts`,
  `high_water`, `high_water_hard`, `burn_in`, `frame_excluded_coldkeys`.

## 13. What the owner decides

Only the real decisions. Already decided and applied here: the scored
questions are optional until close to winning, 200 of them, $1 a correct
answer, 85 / 10 / 5, the precedence principle, new entries usable at once
with an allowance of about a day (20 epochs), all verified heads published.

1. **The label budget.** Recommended: 112 exploration calls a hotkey-epoch,
   about $20 a day with the VM (section 5.3). $11 (48 calls) is cheaper and
   about 2.5 times slower at every field size; $40 (262) is about twice as
   fast and costs the field twice as much.
2. **The picks check's margin.** Recommended: 1.5 net-value points under the
   anytime bound: a challenger a point better passes in about 10 epochs, an
   equal one in about 15, one a point worse about 2% of the time within 20
   epochs. With 1.0, an equal router waits about 30 epochs.
3. **The standing window G.** Recommended: 3 epochs. Longer bars copies of
   an absent head for longer, and leaves an absent holder's seat burning for
   longer; shorter does the reverse.
4. **Pool growth.** Decided 2026-11-19: no generator; the unpublished
   reserve is rationed (6.3), so the check keeps working for four to six
   months with no new questions and each release that adds a standard set
   extends it. A generator stays a later option, before any encoder swap.

## 14. What this record could not measure

- **v4's spreads.** The per-label SD of rule A (8 points) and of rule B
  (21.4) are today's, at today's picks and mix; heads will re-route under the
  v4 mix and price.
- **The bound's coverage with real labels.** Its guarantee is for the plug-in
  variance; the simulations used normal labels. The `bound_v4` stage's
  seeded simulation and a replay on published labels are the checks.
- **Few operators.** The budget table counts one operator per hotkey. With
  five operators behind eight hotkeys, as now, leaving out both heads'
  operators removes about two fifths of a comparison's labels.
- **A memoriser.** None is in the data, so the check's power is a model.
- **A dear entry's measurement time** inside a TD, which sets whether one
  measurement epoch would do (4.6).
