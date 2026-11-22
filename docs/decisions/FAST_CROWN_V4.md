# Protocol v4: a better router seated in one epoch (design)

Status: **DESIGN, 2026-11-21; the decisions of section 0 are taken.** It
changes the unreleased protocol v4 of `SEAT_MINIMAL_V4.md` (as built on
`v4/wip` at `08c6698`) and applies only from a v4 activation block no release
has set. Packages F1-F7 (the fast-crown core) are built (`docs/INVARIANTS.md`,
"v4: the fast crown", lists the details chosen where this record is silent);
F8, F9 and sections 18-19's proposals are another package's. Words as there: an **entry** is one way of
calling a model, a **label** one pooled exploration answer, **R** the
reference, **NV** the net value of a pick at $1 a correct answer (a point is a
cent), J the epoch an entry joins.

Every number is **measured** on the 28 published protocol v3 reveals
`e00026466`..`e00026493`
(`data/observations/fast_crown_v4_inputs_e00026466-e00026493.json`) or comes
from a stated **model**. Every pass or fail is decided by the real bound
(`seat_bound.bound_from_strata`). The simulations are
`scripts/experiments/experiment_fast_crown_v4.py` (`all` re-runs them, one
process, fixed seeds). No paid call was made.

## 0. Decided

| | As built (`SEAT_MINIMAL_V4.md`) | Decided |
|---|---|---|
| The routing test (rule B) | the epoch's 200 scored questions, 1,000 paired: five epochs at least | the challenger answers the first **500 or 1,000** questions of the epoch's **challenge list** in one epoch; the TD keeps 40 calls in flight |
| Who answers | the holder every epoch, the challenger its own picks | the challenger only: its own pick, and the holder's where they differ |
| A failed call in rule B | 0 grade, 0 cost on either side | **fails closed**: the challenger's own call counts as wrong at its table cost, the holder's as right at its table cost |
| The holder | stands and is paid while it answers the scored questions | stands and is paid on its **exploration proof** alone |
| The bound's tuning n0 | the rule's minimum | `BOUND_N0_V4` = **100** for rules A and B |
| Floors | rule A 1,500 labels, rule B 1,000 questions | `ODDS_MIN_PAIRS_V4` **300**, `PICKS_MIN_QUESTIONS_V4` **500** |
| Repeated attempts | the answered run: a gap restarts it | **unchanged** |
| Measuring a proposal | 2 epochs of 150 | **1 epoch of 300**, each failed call retried once |
| An epoch-wide shift | (none) | no guard variance; each epoch's mean challenge item **per entry is published** so a shift can be seen |
| A new entry's allowance | 20 epochs at the measured rate | **1 epoch** (`ENTRY_ALLOWANCE_EPOCHS`): the join epoch only, so a holder one epoch late is not beaten by a copy of itself (section 8); succession stays |
| A missing row after the allowance | certainly wrong (100 Brier points a label) | **odds 0**: "never answers correctly", 100 y² a label |
| Displacement of a new entry | shielded by its allowance | a separate **20-epoch shield** (`ENTRY_SHIELD_EPOCHS`) |
| A seat path on better picks | (none) | **not adopted** (section 9) |
| A finder's fee for a proposer | (none) | none for now |

The owner's goal: a new model on a live, updated router in as few epochs as
possible, ideally the epoch after a miner acts. Section 18 checks a design
for that (chain proposals that join first and are measured by the field);
section 19 a value floor against R.

Not adopted (section 15): challenges of 1,500, the guard variance, a seat
path on better picks.

## 1. What the data say

| Input | Measured | Used for |
|---|---|---|
| Heads' picks that differ from the champion's on a question | median **87%** (IQR 68-93%); every pair of heads 94% | the challenger's extra calls: 0.87 a question |
| Per-question SD of the paired NV item | median **20.0** (IQR 15.4-24.1); unpaired 41.3 | rule B's spread |
| One call's grade, call to call | SD 16 points | single-call items are noisier than v3's median cells: tables also at SD 24 |
| Spread of a pair's epoch mean beyond sampling | 0.70 over 22 pairs; **0** without head-epochs of more than 3 failed calls (95% upper 1.0) | the one-epoch exposure |
| Pick cost a question (seed table, $1 a correct answer) | holder $0.0191, challenger $0.0153; a challenge question **$0.032** | dollars |
| Odds gain a label against the champion | SD 7.9 (IQR 5.1-11.9), kurtosis 31 | rule A, resampled with its tails |
| Labels a v3 epoch pooled / left after two operators | 199 / 66% | v4: about 700 pooled at five operators, **about 450 a comparison** |
| Entries' accuracy on random questions (exploration labels) | 68% to 95% | the cost of ignoring an entry |
| Call latency per entry | medians 1.4-40 s, 90th percentiles to 434 s, slowest 1,252 s | time |

## 2. The one-epoch routing test (rule B)

Each epoch has a **challenge list** of 1,000 published questions on the scored
mix, drawn by the epoch nonce, the scored slice first. A challenger's TD
answers the first 500 or 1,000. For each question it computes the holder's
pick from the holder's published bytes (`picks_and_odds_v4` on the epoch's
menu, as a validator does), calls its own pick and, where they differ, the
holder's. The item is

```
same entry:      0                                   (one answer serves both)
different entry: v_challenger - v_holder,  v = 100 x (grade - table cost)
                 failed challenger call: grade 0 at its table cost
                 failed holder call:     grade 1 at its table cost
                 no holder pick (no row for any entry): v_holder = 0
```

Rule B is otherwise as built (margin -1.5, the answered run).

One attempt judged alone, items resampled from the measured items and shifted
to the true NV (points), n0 = 100, 2,000 attempts (model):

| n | $ | -3 | -2 | -1 | 0 | +0.5 | +1 | +2 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 16 | 0 | 0 | <0.1% | 2.3% | 7.2% | 18% | 57% | 98% |
| 750 | 24 | 0 | 0 | 0 | 4.0% | 14% | 35% | 83% | 100% |
| **1,000** | **32** | 0 | 0 | 0.1% | 7.8% | 28% | **56%** | **95%** | 100% |

Over a run of attempts at 1,000 (SD 20, SD 24 in brackets), passed after one /
two: +0.5 27% / 73%; **+1 55% / 97%** (27% / 78%); +2 97% (77%) in one. Tuned
as built (n0 = n) the +1 column at 1,000 is 21%: retuning n0 is worth half
again the questions. Equal or slightly worse routers are inside the -1.5
margin by design (the odds test decides who is better); routers 2 and 3
points worse never passed.

**One bad hour.** The only epoch-level effect in the data is failed calls.
As built, a holder-side failure scores 0, so an hour in which the holder's
picked entries fail moves every item toward the challenger; failing closed
removes that. An epoch-wide shift in accuracy without failures is not seen
but not ruled out above about 1 point; at 0.7 points one attempt of 1,000
passes a router 2 points worse 0.4% of the time.

## 3. Time and reveal size at 1,000

A challenge epoch is n own picks, 0.87 n holder picks and 112 exploration
calls (latencies resampled per entry, 40 in flight, 30 minutes of calls):
500 takes 1,050 s median / 1,292 s 90th percentile, 1,000 takes 1,424 / 1,664
s, 2% of epochs past 30 minutes (calls cut off count as failed). The floor is
the slowest single call, about 900 s.

**Reveal size.** The worst-case reveal (the real writer at 256 UIDs,
`tests/test_v4_reveal_budget.py`) is 60.78 MiB of 64; its test asks for 3 MiB
spare. Measured field by field: per-head picks values (200 int64) 0.54 MiB,
the per-UID `failed` lists 1.71 MiB, `routing_decisions` 0.49 MiB, the
slice's `shared_grades` 2.82 MiB at 205 bytes a cell. At 256 challengers of
1,000:

| Change | MiB |
|---|---|
| items as int64, every challenger | +2.62 (replacing 0.54) |
| items as **int32** | **+1.32** (replacing 0.54) |
| the list as 1,000 ids | +0.03 |
| challenge cells published per cell (up to 1,000 x 72) | up to +14 |
| challenge cells as per-(entry, category) sums | +0.07 |

With int32 items and summed cells the reveal is 61.6 MiB: under 64 but with
2.4 MiB spare, short of the test's 3. **Still needed at 1,000:** int32 items
for challengers only, challenge cells as sums (the slice's 200 keep their
cells), and one more saving: drop the per-UID `failed` lists from v4 reveals
(failures are inside the items), which gives 59.9 MiB. The prompts of the 800
questions past the slice are not published (ids only).

## 4. Repeated attempts

A challenger that is not better may try every epoch, under several hotkeys or
in several TDs, and serve a proof only when it suits it. Share of 200 runs of
300 epochs with a false pass by epoch 48 / 300 (floor 500, n0 100):

| Rule | NV -1.75, n 500 | NV -2, n 500 | NV -1.75 / -2, n 1,000 | with a 0.7 shift, n 1,000: -1.75 / -2 |
|---|---|---|---|---|
| each attempt alone | 0 / 0.5% | 0 / 0 | 0 / 0 | 12% & 57% / 4.5% & 27% |
| **the answered run** (decided) | **0 / 0** | **0 / 0** | **0 / 0** | 3% & 24% / 2% & 11% |
| never resets, bad epochs withheld | 1% / **73%** | 0 / **11%** | 20% / 1% (by 300) | 27% & 100% / 6% & 99% |
| never resets, attempts declared | 0 / 0.5% | 0 / 0 | 0 / 0 | 2.5% & 15% / 1% & 7% |

One ticket (a fresh attempt, a restarted run, or one more TD run on the same
questions) at NV -2 passes with probability 1e-5 at 500 and 5e-6 at 1,000:
**$1.4M and $6.5M of challenges per expected false pass.** A record that never
resets is worse, not better: it turns withholding into a selection of good
epochs.

## 5. The odds test's floor

Per-label gains resampled from the measured gains, SD 8, fresh-reserve
questions answered by every other hotkey correlated at 0.29; about 450 labels
a comparison an epoch at five operators, 1,900 at fifteen. **30 no-edge heads
over 300 epochs, 20 fields per setting: none passed at any floor from 50 to
1,500**; the highest bound reached was +0.054 against the 0.1 margin. Median
epochs to pass (share in one epoch), floor 300, n0 100: an edge of 0.5 takes
11 epochs at five operators and 3 at fifteen; 1 point 3 (11%) and 1 (89%); 2
points 1 (88%) and 1 (100%). The bound decides; the floor only sets whether a
clearly better head passes on its first epoch, and 300 is below one epoch of
a small field.

## 6. The holder without daily scored questions

A head, the holder included, stands while it (or a predecessor it succeeded)
was verified in this epoch or one of the two before. A seat is paid when its
head is verified that epoch, R does not beat it and the memorisation check
does not refuse it: the holder's cost falls to about $20 a day. R still
beats a stale holder (rule A), and any challenger's test measures the
holder's current routing, recomputed on the epoch's menu and answered that
epoch.

The holder is the champion named by the seat state at the epoch's start (the
root's newest reveal); its bytes are published with the reveal. The TD and
every validator compute its picks; a proof naming another holder or stating a
pick other than the computed one has its challenge refused (the run
restarts). An entry without a row is never the holder's pick; a question with
no entry it has a row for has no holder pick (its side is 0). The pick-cost
table is fed by challengers' answers only; the reveal publishes the holder's
computed picks on the slice for the product's parity check. No attack the
scored-every-epoch rule stopped comes back: each was about the holder's own
answers. New residual: a holder whose favourite entry starts failing for
everyone is shielded from rule B (its failed side scores right) until the
health rule retires the entry, about 3-5 epochs.

## 7. One measuring epoch, with a retry

The gates as coded, each measured entry as a proposal, failures drawn epoch by
epoch from its measured rates, 2,000 runs (model). Share refused (of which
the 95% answered gate):

| Entries | 300 over two (as built) | 300 in one | **300 in one, one retry** | retry, 90% gate on first tries |
|---|---|---|---|---|
| Good, not dominated (36) | 22% (15%) | 24% (16%) | **9.0% (<0.1%)** | 10.8% (2.1%) |
| Dominated (18) | 90% | 90% | 90% | 90% |
| Flaky, 5% or more failing (5) | 61% (60%) | 57% (55%) | **7.6% (0.1%)** | 15% (10%) |

The retry removes the failure-driven refusals entirely. The 9% that remains is
the usefulness comparison refusing entries a cheaper entry measurably beats
(within 3 points of my "dominated" cut): the gate doing its job, so **no
further change is proposed for false refusals.** What the retry costs is the
gate's filter on flaky entries: an entry failing 5-10% of calls now joins,
and only the health rule (30%) removes it. If the lead wants that filter
back, judge the answered share on first tries at 90% (last column): good
entries 10.8% refused, flaky 15%. Time: 300 calls fit 30 minutes at 40 in
flight for every measured entry but granite-4.2-8b@high (7% of epochs past);
the dearest costs $6.82.

## 8. Zero allowance

**What a missing row costs.** As built (`labels_v4._scored`), a head with no
row for an entry is **certainly wrong**: the opposite of every outcome, 100
Brier points a label, whatever the entry's accuracy. The owner's reading,
"predicts it never answers correctly", is odds 0: 100 y² a label, cheap for an
entry that is rarely right. They differ for weak entries only, and **no
measured entry is weak**: on random questions every entry is right 68-95% of
the time. A head's real row costs about 2-15 points a label.

**A lazy holder** (never adds the row) against a challenger equal to it
elsewhere that adds a constant row at the entry's rate or a fitted row (the
champion's real row), stand-ins from real entries: strong
`gpt-6.1-sol@high`, mid `claude-sonnet-5.5@low`, weak `granite-4.2-8b@low`,
dominated `gpt-6.1-sol@medium`. The new entry gets about 20 labels an epoch at
five operators and 50 at fifteen (coverage). Median epochs to pass rule A
(share in one epoch / two):

| Entry | as built, 5 ops | odds 0, 5 ops | as built, 15 ops | odds 0, 15 ops | gain a label, as built / odds 0 (5 ops) |
|---|---|---|---|---|---|
| strong | 2 (11% / 96%) | 2 (8% / 86%) | 1 (100%) | 1 (99%) | +2.8 / +2.7 |
| mid | 2 (7% / 90%) | 3 (4% / 49%) | 1 (99%) | 1 (82%) | +2.6 / +2.2 |
| weak | 2 (8% / 86%) | 4 (0 / 10%) | 1 (98%) | 2 (41% / 91%) | +2.3 / +1.6 |
| dominated | 2 (10% / 89%) | 2 (3% / 52%) | 1 (97%) | 1 (85%) | +2.6 / +2.3 |

(Constant and fitted challenger rows give the same speeds within a few
points.) **A weak entry is not cheap to ignore** under either reading at the
accuracies the pool has; a dominated entry costs as much as a useful one,
because rule A scores predictions, not usefulness. Under odds 0 an entry is
cheap to ignore only below about 0.56 accuracy at 1% of the labels, 0.32 at
3% and 0.25 at 5% (above those, an otherwise equal head with a row passes
within 48 epochs at five operators); as built, never.

**Fitted against constant.** A challenger whose only edge is a fitted row,
against a holder that added a constant row by succession, gains +0.02 to
+0.08 points a label (-0.05 on the strong entry): it **never passed rule A
within 48 epochs.** Rule A rewards having a row, not a good one.

**One epoch late, and copies.** A holder that adds the row by succession is
forgiven its late epochs (a new row counts from the successor's commit, so
earlier labels of the entry are not read for it). But in the epoch it lacked
the row it is beaten by any head equal to it elsewhere with a row: 7-12% of
the time at five operators, **97-100% at fifteen**. That head can be a copy of
the holder's own published bytes plus the row, committed by the proposer, who
knew the entry first. It then holds the seat, and the original cannot take
it back: by succession it is earlier, but it is not better than the copy by
the margin. With an allowance of even one epoch (the missing row at the
measured rate in the join epoch only) the copy gains only fitted-against-
constant and never passes.

**Griefing.** The rules allow at most 2 joins an epoch, 8 proposals in the
pipeline, 4 per coldkey; an id is a model and reasoning setting (host order
and quantization are request updates); a refused or retired id waits 48
epochs. One coldkey can force about one join an epoch while distinct ids
last, for its $20.6 a day plus $0.1-7 a measurement: **about $22-30 a day for
up to 20 joins.** An honest miner answers each join with a successor carrying
a constant row at the published rate: no calls, one commitment and one push
to its TD. Its risk is only being late: as above. Zero allowance also makes a
new entry displaceable at once (`in_allowance` shielded it; it has no picks
yet), so at the menu cap a junk join can displace a real entry that joined
the epoch before.

**What zero allowance touches in the code:** `menu_state`:
`ENTRY_ALLOWANCE_EPOCHS`, `MenuParams.allowance_epochs`, `Entry.allowance_rate`
(always None: dead), `Entry.in_allowance` (always false: the
`MENU_MAX_PENDING` count and the displacement shield change meaning),
`Entry.scored_from` (= `joined_epoch`) and the reader's join and carry checks,
`Entry.measured_rate` (no longer scored; still published); `labels_v4`:
`missing_odds` and `LabelPairs.missing` (all -1: removable from window and
reveal), `_scored` if odds 0 replaces certainly wrong; `consensus_v4` (the
constant is registered); `attacks/run_miner_attacks.py`; the tests pinning
the 20 (`test_menu_state`, `test_menu_step`, `test_menu_measured_rate`,
`test_menu_reader_checks`, `test_menu_pick_cost`, `test_menu_rejoin`,
`test_v4_labels`, `test_v4_seat_odds`, `test_v4_loop`, `test_v4_end_to_end`,
`test_allowance_rows`, `tests/v4_world.py`, `tests/menu_fixtures.py`);
`SEAT_MINIMAL_V4.md` 4.1-4.3 is superseded.

## 9. Winning by better routing: not adopted

The proposal: a challenger also takes the seat when its paired challenge shows
its picks **better** than the holder's by a margin m while its odds are not
worse by more than delta, rule C applying alike.

**Power** (one attempt / within two, model on measured items, SD 20):

| True NV | n 500, m 0.5 | n 1,000, m 0.5 | n 1,000, m 1.0 |
|---|---|---|---|
| -1, 0, +0.5 | 0 | 0 | 0 |
| +1 | 0.2% / 0.3% | 0.1% / 0.2% | 0 |
| +2 | 2.4% / 8.9% | 8.0% / 32% | 1.0% / 6.6% |
| +4 | 55% / 96% | 95% / 100% | 84% / 100% |

No-edge heads trying every epoch under the answered run: no false pass in 300
epochs at m 0.5 or 1. The paired test does not see a routing gain "at once":
with SD 20 a question, 1,000 questions see only gains of about 3 points or
more. **The odds guard** (not worse by delta): a head with equal odds shows it
in 8 epochs at delta 0.5 (five operators), 3 at delta 1.0; a head 0.5 worse
passes delta 1.0 in 7.

**The gaming cases** (the record: `INVARIANTS.md` I3, `I3_DECISION.md`,
`HEAD_EFFICACY.md`; the v2 champion that was a question-type switch):

| Head | Passes path H? | Why |
|---|---|---|
| Constant policy (always `gemini-4-argon@low`) | **yes**, almost surely at 1,000 | its NV at v4 prices is 92.3 points a question against the v3 champion's 87.2 (+5.1); a head can copy the holder's odds and raise one entry's by 0.1 for 0.03 Brier points: the odds guard cannot see it |
| Switch by category (the best entry per category, in hindsight) | **yes** | 95.1 (+7.9): the v2 failure, back |
| Tuned to the pick-cost table's quirks | partly | a new entry is priced at its call cost until it has graded cells; pick cost runs 0.36-1.76x call cost (10th-90th pct), up to 13.7 points a pick under-priced: a head sending 40% of questions to such an entry gains up to about 0.4 points (90th pct) of value that is not real |
| Odds miscalibrated to move picks inside delta | **yes** | picks follow odds minus cost; biases of a few points move picks wholesale at a cost of 100 eps² a label of one entry |
| Near-copy of the holder with picks nudged | no | random nudges on 10-300 of 1,000 questions never passed; it would need to know the outcome of 34 of them, which a head on the locked encoder cannot store |

The v3 champion is beaten by constant policies because it was tuned to v3's
price of accuracy ($0.49); a v4-tuned holder would route at least as well as
the constant. But path H rewards whichever router the *current field* routes
worse than, by measure of the exam, and the exam's mix and table are not user
traffic: that is exactly why scoring moved to odds. **Rule C** with two paths
still gives a unique winner (the latest-committed qualifier beats every
earlier contender by construction), but blocking copies under path H needs a
paired picks test against every earlier contender, so a challenge's calls
grow with the number of contenders.

**Recommendation: do not adopt.** At 500-1,000 questions it sees only gains
of about 3 points or more, and because picks are a small re-biasing of odds,
its odds guard cannot stop a constant or category switch that beats the field
on the exam's prices. What does make a new entry used quickly: zero
allowance makes every head add a row (section 8), and the holder that adds a
good row by succession uses the model from J; a challenger wins when its
predictions are better (rule A). A routing gain of 1-2 points from a new
entry is too small for any test at these sizes to see in an epoch or two
(it needs 4,000-8,000 paired questions).

## 10. Propose and measure in the same epoch

The proposer's TD files in epoch k's proof **and** measures 300 calls in the
same proof; the root's step scoring k decides the gates; the entry joins at
k + 2 (`MENU_LAG`), two epochs sooner than k + 4.

**(1) Fixing the questions.** Draw them from k's nonce and the **entry id**
alone (model and reasoning setting), not the proposal id (which hashes the
proposer, host order and quantization): then grinding proposal bytes or
hotkeys gives no other draw, and one entry id is measured once an epoch. What
remains: k's nonce is known when k starts, so a proposer can look at each
epoch's draw, call the model on it outside the TD (300 calls, $0.1-7) and file
in a favourable epoch: best of E draws. It buys a borderline entry past the
gates, and token means a few points short until 30 exploration pairs replace
them; the measured rate no longer scores anything at zero allowance. Closing
it needs the proposal id on chain before k's boundary (the miner's
commitment), which is the announce step moved to the chain.

**(2) What the announce step did, and how it holds.** The window is k alone,
decided by the step for k; an epoch with no step expires the proposal,
writing nothing (it is filed again). A TD measures only the proposal in its
own proof (`MeasureLedger` unchanged, its ceiling unchanged). The pending,
per-coldkey and room limits are checked by the step after measuring: the TD
checks them against `MenuState(k)` before spending, but proposals filed in
the same epoch can take the room, and the later ones in the step's order
(HMAC of the nonce and proposal id) are refused with their 300 calls spent.
Two miners filing one entry: the first in that order is the proposer; the
second is a duplicate, refused with no cooldown and its measurement not
pooled; nobody is refunded (nobody is today). Cooldowns, the queue and its
24-epoch expiry, and two joins an epoch (first passed, first joined) are
unchanged.

**(3) Notice.** The others see the entry in the root's reveal of k, which
appears during k + 1 (held one epoch for weight commit-reveal); it joins at
the boundary of k + 2. An automated holder has from that reveal to the
boundary, under 72 minutes and possibly minutes, to commit a successor and
push it to its TD. Missing it leaves it lazy for epoch k + 2: section 8's
"one epoch late" (lost 7-12% of the time at five operators, 97-100% at fifteen,
possibly to a copy of itself prepared by the proposer). Under k + 4 the
pending entry is public for about two epochs before it joins. Proposal spam
gets worse in the same way: each join must be answered within an epoch.

**(4) Data at join.** The same 300 calls as today, in one epoch: the token
means and call cost come from them; the pick-cost table has no graded cells
either way. Unchanged.

**(5) Join at k + 1** would need the menu for k + 1 public before its
boundary, while the menu travels in reveal k, which is held for one epoch
(that is why `MENU_LAG` is 2): a second, unheld menu object and commitment,
a measurement proof collected mid-epoch with its own deadline, followers
auditing it before the boundary, and the root's commitment naming the menu
inside the epoch. **That is a redesign of menu publication, not a contained
change,** and it puts a new deadline on the path every node needs to start an
epoch.

**Recommendation: k + 2 is safe with the decided one-epoch allowance** (the
missing row at the measured rate in the join epoch): it removes the copy
theft and the spam pressure and lets a holder that adds the row use the model
from k + 2. Packages: F8 grows (section 13). Section 18 goes further.

## 11. From a filed model to the seat

Minutes from the moment a miner acts, at the start of epoch k (72-minute
epochs), for each way an entry can get on the menu. "Products following
reveals" see the holder's successor when the root publishes the reveal of
the epoch it was first verified in, which appears during the next epoch.

| Milestone | As built (2 measuring epochs, allowance 20) | Decided k + 4 | k + 2 (10) | Chain proposals (18) |
|---|---|---|---|---|
| (i) on the menu | 360 (k + 5) | 288 (k + 4) | 144 (k + 2) | **72 (k + 1)**, if committed before block B |
| (ii) a head with a real row routable | 360 | 288 | 144 | 72 |
| (iii) the field's first labels public | about 432 | about 360 | about 216 | about 144-216 |
| (iv) gates decided (in force) | 360 | 288 | 144 | 216 (k + 3) |
| (v) a lazy holder penalised / its seat lost (5 ops) | 1,800 / 1,900 | 360 / 432-504 | 216 / 288-360 | 216 / 288-360 |
| a product following reveals uses the holder's successor | about 432 | about 360 | about 216 | about 144 |

Per case, decided constants:

| Case | Decided (k + 4) | k + 2 | Chain proposals |
|---|---|---|---|
| The holder adds the row itself | uses it at k + 4 | k + 2 | k + 1 |
| A challenger with a clear overall edge (2 Brier points) adds the row | k + 4, a challenge of 1,000 at J (95%) | k + 2 | k + 1 to k + 2 |
| A challenger whose only edge is a well-fitted row | never on rule A (+0.02-0.08 a label) | never | never |
| A lazy holder | loses its seat J + 2 to J + 3 at five operators, J + 1 to J + 2 at fifteen | the same from J | from J + 2 (provisional labels score nobody) |

**The earliest a product can use the new head.** Following reveals, the
epoch after the successor is first verified. Following the chain directly,
from the join epoch: the product reads the holder's hotkey commitment, fetches
the successor's bytes from the holder's miner (or a URL it publishes), checks
the sha256 against the commitment and, against the published predecessor's
row digests, that every existing row is byte for byte the same and rows were
only added; the new entry's request fields come from the menu naming it (in
section 18, the join delta, published before the boundary). The subnet
should publish a holder's successor at its first verification, through the
same unheld object as the join delta, not at its commitment: before
verification it is only a miner's claim, and the publication hold exists for
heads that are not seated.

Dollars a day (exploration and VM $20.6 a hotkey-day; measured pick costs):

| Party | As built | Decided |
|---|---|---|
| Holder | about $96 (200 scored questions an epoch) | **$20.6**, plus a successor commit for each join |
| Challenger | $20.6, plus about $3 an epoch while answering | $20.6, plus **$16 / $32 an attempt** (500 / 1,000) |
| Other miners | $20.6 | $20.6, plus a successor commit for each join |
| Proposer | 300 calls ($0.1-7) over two epochs | the same in one, plus retries |
| A griefer | (an allowance absorbs joins) | about $22-30 a day for up to 20 joins |

## 12. The rules as they would be written into INVARIANTS

- *I3 — v4: the challenge.* The epoch's challenge list is 1,000 published
  questions on the scored mix, drawn by the epoch nonce, the slice first. A
  challenger answers its first 500 or 1,000 (`CHALLENGE_SIZES_V4`): its own
  pick, and the pick of the holder named at the epoch's start where they
  differ, both recomputed by every validator; anything else refuses the
  challenge.
- *I3 — v4: rule B's items.* 0 where the picks are the same entry; else the
  challenger's value minus the holder's, 100 x (grade - table cost), a failed
  challenger call wrong and a failed holder call right, each at its table
  cost; no holder pick 0. A failed call never raises a challenger's items.
  Each epoch's mean item by entry is published.
- *I3 — v4: evidence over attempts.* Rule B reads the answered run of
  consecutive challenge epochs; a gap restarts it; at least 500 items.
- *I1/I3 — v4: one bound.* Rules A and B use n0 = `BOUND_N0_V4` (100); rule
  A needs 300 labels.
- *I3 — v4: standing and pay.* Every head, the holder included, stands while
  verified in one of the last `STANDING_EPOCHS_V4`; a seat is paid when its
  head is verified that epoch, R does not beat it and it is not refused. A
  contender bars later heads while it has a counted challenge in the last
  `STANDING_EPOCHS_V4` epochs.
- *I3 — v4: new entries.* In the epoch an entry joins, a head without its
  row is scored at the entry's measured rate; after it, at odds 0. A new
  entry is not displaced for `ENTRY_SHIELD_EPOCHS` (20). Succession keeps a
  head's place, and its new row counts from its own commit.
- *I3 — v4: measuring a proposal.* 300 questions in one epoch, each failed
  call retried once inside the TD.

## 13. Build packages

| | Package | Files |
|---|---|---|
| F1 | Constants and the bound's tuning: `BOUND_N0_V4`, the floors, `CHALLENGE_SIZES_V4`, `CHALLENGE_LIST_SIZE_V4` | `config.py`, `seat_v4.py`, `picks_v4.py`, `consensus_v4.py`, `scripts/check_safety_invariants.py` |
| F2 | The challenge list in the plan: nested, published questions, the slice first | `epoch_plan.py` (`plan_v4`), `memo_v4.py`, `tee/td_v4.py` (`V4Plan`) |
| F3 | Proof and TD: a `challenge` field (n, the holder's hash, both sides' results), the holder's picks from its published bytes, the size option replacing `v4_scored` | `tee/proof.py`, `tee/harness.py`, `tee/td_v4.py` (`ScoredPolicy`), `tee/v4_options.py`, `tee/provision.py`, `neurons/miner.py` |
| F4 | Verification: the prefix, the named holder, every pick recomputed | `tee/verify.py` (`_v4_scored_check`), `menu_inputs.py` (`graded_cells`) |
| F5 | Picks: the items, failing closed, int32, the run over challenge epochs, per-entry means published | `picks_v4.py`, `v2_epoch.py` (`_picks_epoch`) |
| F6 | Seat: standing and pay by verification, contenders by challenge | `seat_v4.py`, `v2_rebuild.py`, `v2_sync.py` |
| F7 | Reveal: int32 items for challengers, summed challenge cells, ids for the list, no per-UID `failed` lists in v4, the holder's computed slice picks; the budget test at 256 x 1,000 | `commit_reveal.py`, `v2_epoch.py`, `tests/test_v4_reveal_budget.py` |
| F8 | Measurement: one epoch of 300, one retry; with k + 2, filing and measuring in one proof, the draw keyed by entry id, the step deciding a proposal filed in the epoch it scores | `menu_state.py` (5a, 5b, 5d, `measure_window`), `menu_inputs.py` (`filings`, `proposal_bounds`), `exploration.measure_nonce`, `tee/td_v4.py`, `tee/harness.py`, `tee/verify.py` (`_v4_measurement_check`) |
| F9 | Allowance: one epoch, a missing row at odds 0 after it (`labels_v4._scored`), the 20-epoch displacement shield apart from the allowance | `menu_state.py`, `labels_v4.py`, the tests of section 8 |

Every package updates `docs/INVARIANTS.md`, `docs/MINER_GUIDE.md` where a
miner's choice changes, and the `picks_v4` / `seat_v4` determinism stages.

## 14. Attack cases the build must add

1. A challenger's TD fails any subset of either side's calls: its items never
   rise.
2. A challenger serves only the epochs that help: the run restarts at each
   gap; a router 2 points worse is not seated over 300 simulated epochs.
3. A challenger reports 500 of a 1,000 run because the prefix looks better:
   allowed (prefixes are looks at one sequence); a non-prefix set is refused.
4. A proof pairing against another head than the holder, or misstating a
   holder pick: challenge refused.
5. A holder serving exploration only keeps the seat and is paid; one that
   stops serving loses it after three epochs.
6. A copy of the holder plus a new entry's row, against a holder one epoch
   late: the copy gains only a fitted row over the measured rate in the join
   epoch and does not take the seat.
7. Proposals filed to force rows: at most two joins an epoch, a new entry
   not displaced before its shield ends; with k + 2, varied proposal bytes
   get the same questions and a second filing of an entry is a duplicate.

## 15. Considered and not adopted

- **Challenges of 1,500**: past 30 minutes in 45% of epochs at 40 in flight
  (80 needed) and 2 MiB of reveal; two attempts at 1,000 cover a router 1
  point better (97%).
- **A guard variance of (0.5 points)² an epoch**: it held a 0.7 shift to 0-1%
  but cut a 2-point router's one-attempt pass at 1,000 from 97% to 75%.
- **Evidence that never resets** (withholding becomes selection), **a
  fixed-sample bound per attempt** (every attempt a fresh ticket), **failed
  calls as missing** (the network's controller would choose the items), **a
  label floor of 700** (two epochs for a clearly better head, no safety
  added), **a cooldown or attempt cap** (tickets already cost millions).
- **A seat path on better picks** (section 9).
- **Join at k + 1** (section 10): a redesign of menu publication.

## 16. Open decisions for the lead

Settled 2026-11-21: the missing-row reading (odds 0), the one-epoch
allowance, the 20-epoch shield, path H, no finder's fee.

1. **How a new entry gets on the menu.** k + 4 (proposer measures), k + 2
   (section 10), or chain proposals measured by the field (section 18).
   Recommended: **section 18 with its changes**; if its publication path is
   judged too risky for v4, k + 2.
2. **The flaky-entry filter.** With the retry, entries failing 5-10% of calls
   join. Recommended: accept, the health rule retires real failures; the
   alternative is a 90% gate on first tries.
3. **The value floor** (section 19). Recommended: publish the comparison
   only.

## 17. What this record could not measure

Single-call items in v4 (SD 20 to about 26); an epoch-wide accuracy shift
without failures (not seen, not ruled out above about 1 point); provider rate
limits at 2,000 calls an epoch (paid); labels a comparison and a new entry's
share in a real v4 epoch; v4-tuned heads (the degenerate-policy figures
compare with a champion tuned to v3's prices).

## 18. Chain proposals that join first and are tested live

The design checked: a miner files a proposal by an on-chain commitment; at
each boundary valid proposals join as **provisional** entries; the field
measures them in their first epoch through exploration it already makes; the
gates then make them full entries or retire them. Read against `v4/wip` at
`d87deda` (which adds the published-set wiring and the v4 replay; nothing
here changes either, but both read the menu each epoch and must read the
provisional state below).

**(a) The chain.** A hotkey holds one `Commitments` field, at most 128 bytes
(`reveal_publisher.COMMITMENT_MAX_BYTES`, the SDK's `Raw128`); the pallet keeps
only the latest. A miner's head pointer is the bare 64-hex sha256 of its head
(`commitments.is_head_hash`; `copy_rule` ignores anything else), and a
snapshot is capped at `MAX_COMMITMENTS` (4,096). The chain rate-limits
transactions per hotkey (`neurons/miner.py` notes it); the code records no
number, so the pallet's limit must be read before building. A proposal
(model id, reasoning setting, host order, quantization; `max_tokens` is
derived from the hosts' read limits, not chosen) is 150-300 bytes in
canonical form: it does not fit beside the pointer. It fits as a hash:
`<head 64 hex>:p<sha256 of the canonical proposal, 43 base64url>` is 109
bytes. The bytes live with the proposer (served by its axon, or carried in
its previous epoch's proof); the root fetches them at its pinned read and
copies them into the menu delta it publishes, so every node verifies them
by hash from the root's bucket and no miner writes there. A proposal the root
cannot fetch is skipped that epoch and stays on chain. Head and proposal
change together in one transaction, which is atomic; the head's first-commit
record is unchanged by re-committing the same hash with a suffix, so the
copy rule's readers must parse the suffix (they would ignore the field
today).

**(b) Who computes the menu.** Every node deriving it from a pinned chain read
would put, inside every TD, a chain read the operator feeds and fetches from
other miners' boxes: rejected. **Recommended: the root** reads the
commitments at block B = boundary - 30 blocks (6 minutes: fetches at 30 s
each, `reveal_sync.REQUEST_TIMEOUT_S`, an upload read back, a commitment
included in a block or two), publishes a **join delta** by hash and commits
the hash before the boundary. Its own commitment field is full (125 of 128
bytes: reveal id, digest, menu hash), so the delta's hash needs a second root
hotkey in the compiled table. A root that misses the deadline commits
nothing; the delta is empty, and the proposals wait an epoch on chain.
Followers audit it exactly: chain state at B, the documents inside the delta,
the rules. This removes `MENU_LAG` **for joins only.** Prices, retirements,
token means, pick tables and the gates read epoch summaries from verified
proofs, published in the reveal held one epoch for weight commit-reveal, so
they stay two epochs behind.

**(c) Measurement by the field.** As built a new entry gets about 20 labels an
epoch (coverage damped by its deficit over 1,400). Coverage aimed first at
provisional entries, undamped, gives up to 44 calls a hotkey within the
$1.10 coverage cap (4 of 59 seed entries are dear enough that the cap cuts
below 44, the dearest to 16): **about 290 labels at five operators (7
hotkeys), 880 at fifteen (20)**, against the 300 the gates were tuned on. The
gates (with the retry, model on measured rates; share refused):

| Labels | Good entries | Dominated | Flaky |
|---|---|---|---|
| 290 (5 ops, one entry) | 9.2% | 89% | 7.2% |
| 145 (5 ops, two entries) | 6.3% | **75%** | 5.6% |
| 2 epochs of 145 | 9.3% | 89% | 7.5% |
| 880 (15 ops) | 16.5% | 99% | 13% |

One entry an epoch fits at five operators; two do not (a quarter of
dominated entries would join): admit **one provisional entry an epoch below
ten hotkeys**, two above, or keep a second one provisional for two epochs.
The rest of the menu loses its coverage for that epoch (44 of 112 calls),
which only matters to entries still under 100 labels.

**(d) Safety that rested on the proposer paying.**

- *Spam.* A commitment costs a fee and a registered hotkey. Rules: the
  proposer's hotkey had a verified proof in the epoch before (so about $20.6
  a day per proposing hotkey); one proposal per coldkey per epoch; a coldkey
  whose proposal fails a gate waits 48 epochs (as today); the admission order
  when more arrive than may join is HMAC(hash of block B, proposal hash):
  unknown when proposals are committed, and miners do not author blocks, so
  grinding needs more hotkeys and front-running gains nothing (no finder's
  fee; a copied proposal of the same entry is a duplicate). Filling every
  slot for a day with junk needs a fresh coldkey per failed proposal per 48
  epochs: about 20-48 serving coldkeys, **$400-1,000 a day**. Each junk entry
  costs the field one epoch of coverage (at most $1.10 a hotkey) and no
  successor commits: provisional labels score nobody, and junk is retired at
  J + 2.
- *A broken or slow host.* Provisional calls must be left out of the
  exploration failure bound (34%), the outage rule and the health counts, run
  with a short deadline (300 s), and be cut at the TD's time budget; a cut or
  failed call counts only against the entry's answered share, judged per
  coldkey (the median coldkey), so one operator failing its own calls moves
  nothing.
- *Distortion.* While provisional, the entry's labels are read by no rule A
  comparison (this replaces the one-epoch allowance, since there is no
  measured rate to score a missing row at); heads with a row may route users
  and challenge questions to it; a challenge pick of a provisional entry is
  priced at its attested realized cost (it has no table); a head that routes
  there risks only that the entry is retired at J + 2.
- *Cap and shield.* Provisional entries count toward the cap of 64 and need a
  displaceable entry to join; the 20-epoch shield starts when one becomes
  full.
- *Values at join.* No measured rate and no token means: exploration prices
  it at its listed prices times the menu's median token means until its first
  epoch's labels (30 pairs is `TOKEN_MIN_PAIRS`) give its own.

**(e) What goes, what stays.** Goes: the proposer's measurement
(`td_v4.measurement_due`, `measurement_to_run`, `MeasureLedger`,
`MEASURE_CALLS_PER_EPOCH`; the harness's measurement calls; the proof's
`proposal` field and measurement results; `verify._v4_measurement_check`;
`exploration.measure_nonce`; `epoch_plan.measurement_questions`;
`menu_inputs.filings` and `proposal_bounds`); in `menu_state`, `Pending`,
`MeasureReport`, `measure_window`, `measure_deadline`, `_measure_expires`,
`PROPOSAL_MEASURE_EPOCHS`, `PROPOSAL_MEASURE_PAIRS`, `PROPOSAL_MAKEUP_EPOCHS`,
the queue and `QUEUE_WAIT_EPOCHS`, steps 5a-5d rewritten; the TD option
`FUGAL_V4_MEASURE_CAP_USD`, and `FUGAL_V4_PROPOSAL` moves to the miner's
commitment tool; their attack cases and tests (`test_menu_step`'s proposal
sections, `test_td_v4_session`'s measurement, `test_menu_measured_rate`,
`test_menu_rejoin`). Stays: the entry rules and signed lists
(`menu_rules`), duplicates and cooldowns, two joins an epoch, the gates, health,
displacement. New: the chain proposal reader, document fetch and check, the
root's join delta and second commitment, base menu plus delta in TD, miner
and validator, coverage priority and the failure-bound exclusions, the
provisional state in `MenuState` and the reveal, the replay reading it.

What the earlier reviews' fixes guarded, and where it goes: the proposer
choosing its questions (no proposer measurement now; each hotkey's coverage
draw is its own nonce's); the proposer failing calls after seeing timing
(becomes any operator failing its own provisional calls: the median-coldkey
answered share); a TD measuring proposals it did not file, and the spend
ceiling (become the field's coverage, capped at $1.10 a hotkey-epoch); the
measurement window and make-up when the root skips (a skipped root commits
no delta; proposals wait on chain); pending limits and the queue (one or two
provisional joins an epoch; the rest wait on chain); duplicates in one epoch
(the admission order decides).

Size: packages J1 chain proposals and documents; J2 the root's join delta and
second commitment; J3 base-plus-delta menus in TD, miner, validator, replay;
J4 provisional state, coverage priority, failure-bound exclusions; J5 gates
on field labels and the provisional-to-full step; J6 removal of the proposer
pipeline and its tests. About the size of the fast-crown core (F1-F7), and it
touches the path every node needs to start an epoch.

**(f) Timeline.** Section 11.

**(g) Judgement: adopt with changes**: the proposal as a 43-character hash
suffix on the head commitment with the document served by the proposer; the
root's pinned read 30 blocks before the boundary, an unheld join delta under
a second root hotkey, carried when late; provisional labels scored by
nobody; one provisional join an epoch below ten hotkeys; provisional calls
outside every failure and outage count; realized cost for challenge picks of
a provisional entry; proposers must have served the epoch before. Consensus
risk is higher than k + 2: a new publication path with a deadline before the
boundary, a second root key and miner-served documents. Its failure mode is
safe (the delta is empty and proposals wait). What it still cannot deliver:
the next epoch only for a commitment made before block B; the field's
verdict two epochs later (the held reveal); a lazy holder penalised only from
J + 2; and a product sees the new head without trusting a miner only once
the root publishes it.

## 19. A value floor against R

A guard, not a path to win: a holder may not hold the seat while its routing
value (correct minus cost at the table, the challenge's metric) is shown worse
than R's routing (per category, the entry R's rates and the table would
pick) by a margin, and the comparison is published every epoch.

**Where the holder's side comes from.** The holder no longer answers.
*Off-policy:* exploration labels whose entry happens to be the holder's pick
on that question, or R's: measured, **1.5% and 1.8% of labels**, about 11 and
13 an epoch at five operators (700 labels), 34 and 40 at fifteen; the two
sides are different questions, so items are unpaired (SD about 33 points).
*Holder-paid:* every M epochs the holder's TD answers N challenge questions,
its picks and R's where they differ (paired, SD 20); a missed check counts as
failed. Epochs until the floor unseats a holder worse than R by the gap
(median; share by 48 epochs), 200 runs, n0 100, 48-epoch window:

| Evidence, margin | gap 0 (150 epochs) | gap 2 | gap 3 | gap 5 | holder's cost |
|---|---|---|---|---|---|
| off-policy, 5 ops, m 1 | never | 0.5% | 0.5% | 8% | none |
| off-policy, 15 ops, m 1 | never | 1% | 5.5% | 45 (57%) | none |
| paid 200 every 5, m 1 | never | 12% | 36 (70%) | 6 (100%) | about $26 a day |
| paid 500 every 5, m 1 | never | 61 (33%) | 16 (100%) | 1 (100%) | about $64 a day |
| paid 500 every 5, m 2 | never | never | 69 (37%) | 6 (100%) | about $64 a day |

A holder equal to R was never unseated. The degenerate heads of section 9
gain nothing: the floor only removes a holder; taking the seat still needs
rule A. When it unseats a holder, R is the bar and the seat is open (3.6):
any head that beats R on odds and passes rule B against the last holder
takes it; with none, the holder keeps the seat unpaid.

**Recommendation: publish the comparison, do not enforce it.** The free
off-policy estimate sees only gaps of 5 points or more, and only at fifteen
operators; the paid check that sees 3 points costs the holder $26-64 a day,
which brings back what the decided design removed. Published each epoch with
its bound, it tells the product and the field when the holder routes worse
than R; if it ever shows a real gap, a paid check of 200 every 5 epochs at
margin 1 is the version worth building.
