# Protocol v4: the live menu, measured costs, a harder exam (design)

Status: **DESIGN, 2026-11-18.** It describes a future protocol v4. Protocol v3
(`docs/INVARIANTS.md`) is what runs today, and every rule below applies only
from a v4 activation block that no release has set. Code references were
checked against `main` at `066c601` (v0.6.0).

Five packages of section 18 are built and inert (nothing on the v1-v3 path
imports them, and nothing reaches them before a v4 block): W1,
`fugal_subnet/menu_state.py`; W2, `fugal_subnet/menu_rules.py`; W3, the TD
side (`fugal_subnet/tee/td_v4.py`, `fugal_subnet/tee/endpoint_reads.py`, the
v4 path of `tee/harness.py` and the v4 proof fields of `tee/proof.py`); W5,
section 8's exploration (`explore_weights.plan_shares_v4`,
`epoch_plan.exploration_plan_v4`); and W6, per-entry grace
(`routing_protocol.grace_excluded` and `scored_menu` under protocol version
4). W4 is built the same way in two halves: verification and aggregation
(W4a: `verify_proof` under protocol 4, `tee/verify.py`'s `expected_v4` and
`V4Expected`, the step's inputs from verified proofs in
`fugal_subnet/menu_inputs.py`, and v4 pooling in `explore_pool`), and
publication (W4b: `fugal_subnet/menu_publish.py`, the v4 branch of the
validator's epoch loop, the commitment's menu field and the read helpers in
`reveal_sync`). Sections 4, 5, 6, 8, 9, 12 and 15 say what that code does
where it exists. Nothing else here is built.

W0, the offline replay that fixes the sizes and thresholds of sections 8,
9.2 and 10, is done: 24 published v3 reveals, `e00026466`..`e00026489`
(28.8 hours), replayed with no paid call (a decision record of 2026-11-18,
kept outside the repo and reproducible from those reveals, which are named
with their sha256). Sections 2, 8, 9.2, 10 and 17 carry its numbers, and
the code follows them (W4a brought the W1 step's health rule to 9.2's).

Words used throughout: an **entry** is one way of calling a model
(`openai/gpt-6.2-sol@high`); the **menu** is the set of entries a head gives
odds for; a **pair** is one exploration answer, a (question, entry) nobody
chose; **R** is the reference odds a head is compared with; the **root** is the
validator whose hotkey the release names (`config.root_hotkey`).

## 1. What v4 is for

The subnet exists to produce a router that is measurably better for real users
than the best single model. Three things have to hold for that, and v4 is
organised around them:

1. **The menu is the models users can actually call today, at today's
   prices.** In v3 every change to it costs a release, a new image, a new
   approved entry and a validator upgrade.
2. **What a head is told a call costs is what it costs.** A router that prices
   an entry at a third of its real spend routes to it too often.
3. **The questions separate routers.** Odds can only be shown to be good where
   models differ; on a question every entry answers, every head is right.

A fourth point is supply rather than scoring: miners need more than one place
to run a sealed machine (section 13, a separable work package).

## 2. What the live v3 data says

Measured on the published reveals `e00026466`..`e00026483` (the first 18 v3
epochs, 2026-11-17/18) and on OpenRouter's public listing on 2026-11-18. The
numbers are small-sample; where that matters it is said.

| Finding | Number | Consequence |
|---|---|---|
| Routers are close to each other and to single entries | 23 heads at 86-93% accuracy on the scored slice (all but one at 89-93%); several single cheap entries at 92-95% (+-3 points) on exploration pairs | The exam does not yet separate a router from a constant choice |
| Hindsight headroom exists | Best graded entry per question: 95.9% at about a fifth of the champion's spend | The menu can support a better router; the measurement has to reward finding it |
| Pinned token means under-state spend | Attested tokens x pinned prices vs the pinned mean: `gpt-6.1-sol@medium` $0.0080 vs $0.0023 a call, `kimi-k3.1-pro@max` $0.0276 vs $0.0103, `deepseek-v4.1-flash-1017@max` $0.0022 vs $0.0004. Over all entries (W0): 20 cost more than 1.5x their pinned figure, 6 less than 0.67x, median ratio 1.34; the dearest, `claude-opus-5.5@max`, $0.129 a call against $0.023 pinned | Heads and exploration shares price many entries too low; the gap varies by entry |
| Dear entries are barely measured | 27 of 60 entries (every Fable 5.2 and GPT-6.2 entry) have fewer than 40 pooled pairs after 18 epochs; after 24, 11 are under 40 and 43 under 100; `single_model_table` is still empty (its most-drawn entry has 252 pairs) | R, token means and any retirement rule have no data exactly where calls cost most |
| Two benchmarks are saturated | On exploration pairs (an entry nobody chose, inverse-probability weighted; W0, 24 epochs): ARC 99.4% (99.1-99.7), GSM8K 97.2% (95.3-98.9). MMLU is 98% on the entries the leading heads pick but 91.1% (87.8-94.2) on exploration pairs, and is not saturated. `supergpqa_hard` 67.2% (64.5-69.5) | ARC alone is 15% of v3's scored mix and cannot show skill; the seat is decided on exploration pairs, so those are the ones to read |
| Cost counts, and is noisy | The champion spends $0.0193 a question (0.0146-0.0241; W0, from the shared grades and routing decisions). At `CHAMPION_VALUE_PER_CORRECT_USD` = $0.49 that is 3.9 net-value points (3.0-4.9). One long answer from a dear entry is worth 100-200 points on one question: of the 47-point per-question SD of a challenger's paired net value, the cost term alone is 42 | The net-value guard is held back by the noise in realized cost more than by accuracy (section 8, the slice) |
| Entries fail | `xiaomi/mimo-v2.5-pro` 35% of exploration calls failed, `google/gemma-4-31b-it` 10% | Nothing removes an entry that stopped working |
| Pinned rows drift from the listing | Of six models checked, four differ: one entry's only pinned host is no longer listed (the entry failing 35% of calls), one model's second pinned host is gone, one pinned host now lists $0.15/$0.60 against the pinned $0.098/$0.294, one model lists 5% below its pinned price | Prices and hosts move on a timescale of days; a pin per release cannot follow |

Why dear entries are under-measured follows from v3's own constants: 32
exploration calls a miner, an expected-spend cap of $0.20, and a floor of a
quarter of a uniform share. A dear entry is drawn with probability
(1/4)/60 = 0.42%, the pool holds about 220 pairs an epoch, so it collects
about one pair an epoch. That is the design working as written, with a budget
too small for a 60-entry menu.

## 3. Scope and decisions already taken

Owner decisions (2026-11-17 and 2026-11-18), not reopened here:

- v4 is **one reset release** carrying: (a) the live menu; (b) costs measured
  from exploration; (c) a new head format if the offline router study supports
  one (it did not: 11.2); (d) a harder question mix and automatic retirement of failing entries.
- Adding a model needs no release, no owner action and no validator or miner
  upgrade. **The proposing miner pays** for the measurement. **No extra
  proposer reward.**
- A **lab allowlist** stays as the one human control.
- Defaults: grace 24 epochs, menu cap 64 entries, at most 2 joins an epoch,
  a 300-question measurement.
- Verified heads are **published each epoch**, so the product can follow the
  champion without a manual copy.
- A **second attestation path** (Phala Cloud) is a work package of v4 that may
  slip to a later release without blocking the rest.

Section 16 lists where this design departs from the owner's plan notes of
2026-11-17 and why. Section 17 lists what is still open.

## 4. The menu becomes published state

### 4.1 What changes

In v3 the menu is two files inside the measured image
(`routing_data/models_v3b.json` and its token manifest), hashed into the
proof's namespace (`routing_protocol.identity_v3`). In v4 the image carries
**rules** (section 6) and a **seed** (4.5). The menu itself is state:

```
MenuState
  epoch            the epoch this state applies to
  serial           +1 whenever anything below changes
  entries[]        sorted by id (section 4.2), each with its costs, its grace
                   epochs and its exploration-pair count
  pending[]        proposals being measured (section 5)
  queue[]          proposals that passed and wait for a slot
  retired{}        id -> {epoch, reason, cooldown_until}, at most 256 (4.4)
  coldkey_cooldowns{}  coldkey -> the first epoch it may file again (section 5)
  allowlists       {labs: {serial, sha256}, devices: {serial, sha256}}
```

Who may file is consensus, so both cooldowns are in the hashed state: an
entry id's in `retired`, a proposer's coldkey's in `coldkey_cooldowns`. The
coldkey has its own record because `retired` is keyed by id, and a later
record for the same id would replace it. A coldkey's record leaves the state
when its cooldown has run out.

`menu_hash(E)` is the sha256 of the canonical JSON of `MenuState(E)`
(`sort_keys`, no whitespace, integers and strings only: section 4.2 has no
floats). It replaces the two file hashes in the proof namespace:

```
identity_v4(E) = digest({protocol: "fugal-success-v4", contract, profile,
                         lambda, menu: menu_hash(E)})
```

so a proof produced on any other menu fails `verify_proof` exactly where a v3
proof from another table fails today (`tee/verify.py`, the namespace check).

### 4.2 An entry

```
Entry
  id, rev                  rev +1 when the request changes (4.4)
  model, reasoning, max_tokens, provider{order, allow_fallbacks:false}
  quantization             as pinned at proposal
  price_in, price_out      integer picodollars per token (1e-12 USD)
  mean_in, mean_out        integer milli-tokens: the measured means (section 7)
  token_pairs              how many pairs the means rest on
  explore_pairs            its pooled EXPLORATION pairs in the window, for the
                           current rev (section 8's n_e); 0 when it joins and
                           when a request update bumps rev
  joined_epoch             first epoch on the menu
  scored_from              joined_epoch + MENU_GRACE_EPOCHS (section 9); a seed
                           entry's is its joined_epoch (no grace, 4.5)
  proposer                 the proposing hotkey (audit only; decides nothing)
```

`explore_pairs` is not `token_pairs`: the token means of a new entry rest on
its proposer's measurement, and stop moving below `TOKEN_MIN_PAIRS`, while
`explore_pairs` counts only pairs the field explored, so a new or re-pinned
entry shows its whole deficit to exploration (section 8). `MenuState` hands
the counts to exploration as `{entry id: explore_pairs}`
(`MenuState.explore_pairs()`).

Integers because the state is hashed and compared across hosts: OpenRouter
reports prices as decimal strings per token (`"0.00000004368"`), which are
exact in picodollars.

`request_for` reads the request fields from the entry, as v3 reads them from
the table row. Heads key rows by `id`. Each epoch's slice is indexed by that
epoch's own sorted id list, as it already is across the menu B block; the
build must check that every window consumer aligns by id, since entries now
leave as well as join.

### 4.3 Who decides the state, and when it applies

All TDs must run an epoch on one menu, so the state has one author per epoch
and everyone else audits it:

- When the root scores epoch `k` it computes `MenuState(k + MENU_LAG)` from
  `MenuState(k + MENU_LAG - 1)` and epoch `k`'s inputs (4.4), and publishes it
  in reveal `k` as `menu_next`. Its hash goes in the root's on-chain reveal
  commitment (`reveal:<epoch>:<digest>` gains `:<menu_hash>`). As built,
  the menu hash is written as its 32 bytes in unpadded base64url (43
  characters, one spelling per hash): `set_commitment` writes one `Raw`
  field of at most 128 bytes, the hex form would be 146, and this form is
  125 (`reveal_publisher.COMMITMENT_MAX_BYTES`, checked at startup by
  `menu_publish.startup_refusals`). The whole commitment is
  `reveal:<epoch_id>:<reveal sha256, 64 hex>:<menu hash, 43 base64url>`:
  7 + 9 + 1 + 64 + 1 + 43 = 125 bytes against the SDK's `Raw128`.
  **Followers on code older than this cannot read the four-field form**
  (their parser takes `reveal:<epoch_id>:<sha256>` only, so their reveal
  sync stops at the first such commitment): the activation package must
  have every validator upgraded before the root writes one.
- **Publish before commit.** The root's publisher puts a menu hash in its
  on-chain commitment only after the menu bytes, and any newer signed list
  they name, are confirmed readable from the public store (read back and
  hashed). A failed or unconfirmed upload means no menu hash in that epoch's
  commitment, so the previous menu carries and the changes wait. The
  validator-side package (W4) implements this in the publisher and tests
  it: a commitment never names bytes the store did not serve back.
- `MENU_LAG` = 2 on finney. A reveal is held for the weight commit-reveal
  period (`reveal_publisher.delay_epochs_for`, at least one epoch), so reveal
  `k` appears during epoch `k + 1` and is public before the boundary of
  `k + 2`. Finney's commit-reveal period for this netuid was read as 1 on
  2026-11-19, so the hold is one epoch and `MENU_LAG` = 2 holds there. The lag is compiled and in the consensus digest; a validator whose
  chain reports a longer hold than `MENU_LAG - 1` refuses to start.
- **The rule every node applies at the boundary of epoch E:** read the root's
  commitment for epoch `E - MENU_LAG` at the boundary block (pinned-block
  read, like head commitments). If it exists and names a menu hash, that is
  `menu_hash(E)`. If it does not exist, `MenuState(E)` is `MenuState(E - 1)`
  with `epoch` advanced, and the changes that would have applied wait. Chain
  state at a named block is the same for everyone, so a late or absent root
  delays menu changes and never splits the subnet.
- The miner's TD needs the bytes only when the hash changed. It accepts them
  from any source (the public reveal store, a mirror, the operator) because
  it checks them against the on-chain hash, then applies the compiled rules
  of section 6 to every entry before it calls anything. When the changed
  menu names a newer signed list it needs that list and its signature too,
  from the same sources.
- **A committed menu that fails the check is refused by every node, and the
  previous menu is carried.** A menu that differs from the carried one must
  be at most `MENU_MAX_BYTES` (1 MiB, checked before it is parsed),
  well-formed, for this epoch with the next serial, name the signed lists
  in force or newer ones (never none, never an older one), stay within the
  compiled caps (at least `MENU_MIN_ENTRIES` (8) full entries, provisional
  ones not counted, and at most `MENU_CAP`
  entries, every entry with an expected cost a call above zero; at most
  `MENU_MAX_PENDING` proposals in measurement or waiting and entries in
  grace together; at most `RETIRED_KEEP` (256) records in `retired` and
  `coldkey_cooldown_cap` (`MENU_MAX_PENDING` x (`PROPOSAL_COOLDOWN_EPOCHS`
  + `MENU_LAG`) = 400) coldkeys in cooldown, so that every list in a menu
  is capped and the largest document the caps allow, about 0.43 MB with
  every string and integer at its longest and a pick-cost table of
  `PICK_MAX_CATEGORIES` (8) categories on every entry, stays under
  `MENU_MAX_BYTES`),
  be one step on from the previous menu (the transition checks below), and
  pass the compiled rules under the list it names: every entry, and every
  proposal in measurement or waiting. Bytes of any
  other kind are refused too, whatever they make a check raise: a value of
  another type anywhere is a refusal, never "unavailable". The verdict
  depends on the bytes, the previous menu and the list alone, so every node
  that holds them reaches it; everything but the rules is decided on the
  bytes and the previous menu, before any list is asked for. Each node runs
  the epoch on `MenuState(E - 1)` with `epoch` advanced, exactly as if
  there had been no commitment, and logs an alarm. A miner's TD checks that
  carried menu under the list the menu it holds names, not under any list
  the refused bytes name, so a refused menu costs every node, TDs
  included, one epoch of menu changes and nothing else. The root's own node
  does the same, so its next step starts from the carried menu like
  everyone's.
- **The transition checks** (`menu_state.check_transition`) are what the
  step's output meets whatever its inputs, so they need only the two menus
  and the compiled constants:
  - an entry that was not on the menu joins only from a proposal in the
    queue, or in measurement with its last epoch recorded by this step, as
    that proposal's request, revision and proposer, in this epoch, with
    `scored_from = joined_epoch + MENU_GRACE_EPOCHS`; an entry's request
    changes only by a request update that is ready in the same way, one
    revision up; no entry's `joined_epoch` or `scored_from` moves; at most
    `MENU_MAX_JOINS_PER_EPOCH` entries join or are updated;
  - an entry that joins, and a request update, carries its proposal's
    values (4.4): its price and answer limit are the proposal's, moved only
    as a refresh in that very step would move the proposal's own (within
    the price step, in a refresh epoch), and from the queue its token means
    and pair count are the measured ones exactly. A proposal decided and
    joined in one step joins on means its last measurement report gives,
    which only the root's proofs carry, so there the reader holds the pair
    count to no fewer than the answered calls the held menu records
    (`_carries_proposal`);
  - an entry leaves the menu only with the `retired` record a retirement
    writes: its reason, this epoch, and the full `PROPOSAL_COOLDOWN_EPOCHS`
    cooldown; a list reason only with a newer list, and only for an entry
    that list refuses for that reason (the next item); a displacement only
    for an entry out of its grace that made room for one that joined, on a
    menu the step leaves at its cap (the step displaces only when a join
    needs the place); and at most max(2, a tenth of the menu) health
    retirements (failing, gone) in one step;
  - with the newer list it adopts in hand, what leaves for a list reason
    (`lab_removed`, `host_removed`) is what that list refuses for that
    reason: `menu_rules.check_entry` on the entry, `check_request` on a
    proposal, as the step's first stage computes them; a request update
    that leaves with nothing written is one the list refuses
    (`_list_verdicts`, in `menu_for_epoch`, which holds the list);
  - a proposal new in measurement starts unmeasured from this epoch for its
    full run, filed by the scored epoch, under the id of what it proposes;
    it meets the step's proposal check on what both menus show (no
    duplicate of what is being decided or of the entry on the menu, no
    running cooldown on the id or on the proposer's coldkey, those the step
    writes before it files included), and leaves its proposer's coldkey
    with at most `MENU_MAX_PROPOSALS_PER_COLDKEY` (4) proposals in
    measurement or waiting;
  - one already in measurement keeps what it proposes, records one epoch
    exactly when the scored epoch is one of its measurement epochs, keeps
    the window the step gives it (`measure_window`: it moves only to make
    up an epoch no step ran for, section 5), its results do not shrink, and
    it is decided in the step its measurement ends; a proposal enters the
    queue only then, and a waiting one keeps its measured costs;
  - a proposal leaves measurement or the queue only as the step lets it,
    with what the step writes for it: the queue, the menu, a gate record
    (when its measurement ends), a lapse record and its coldkey's cooldown
    (when it was due), or with its entry or under a newer list; or, writing
    nothing, a queued proposal that has waited `QUEUE_WAIT_EPOCHS` and a
    proposal whose make-up would run past `measure_deadline` (section 5),
    and none is kept past either;
  - `retired` and `coldkey_cooldowns` are exactly what the step leaves
    (`_records_follow`): a record that is new or differs from the held
    menu's is for an id that left the menu or whose proposal ended in this
    step, at this epoch, for a reason the step gives for that, with exactly
    the compiled cooldown; a coldkey's cooldown is new or changed only for
    the proposer of a proposal that lapsed in this step, at exactly the
    compiled length (a failed gate gives none); a record leaves only when
    its id joins the menu, or by the step's own pruning (over
    `RETIRED_KEEP`, records that can refuse no filing, the oldest first),
    and a coldkey's when it can refuse no filing, which the reader
    recomputes. So no cooldown is dropped, shortened, invented or
    lengthened;
  - prices, answer limits, token means and exploration-pair counts move
    only in a refresh epoch, except on the entry that joins or is updated
    in that step (4.4, the refresh cadence), and a refreshed price moves by
    at most a factor of `MENU_PRICE_STEP` (4) up or down.
- **A newer signed list is adopted only with the list itself.** A committed
  menu that names a newer lab and host list or a newer device list is
  checked only by a node that holds that list (as `verify_signed_list`
  returned it: its content hashes to the reference); the step is likewise
  handed the lists, never a reference. A reference to a list no node holds
  is adopted by none.
- **A node that cannot check a changed menu has no menu for the epoch.** With
  no bytes for the committed hash, or without a list the menu newly names, a
  node cannot tell an acceptable menu from a refused one, and it does not
  guess. A miner's TD in that state produces no valid proof for the epoch.
  That is its operator's problem to solve: the reveal store and its mirrors
  are where the bytes and the lists come from.
- Every other validator recomputes `menu_next` from its own verified proofs.
  A difference is published (`menu_audit` in its reveal) and logged as an
  alarm. It then follows the root's state, as it already re-anchors its
  window and seat on the root (`v2_sync`). Two validators that verified
  different proof sets can differ here for the same honest reason they can
  differ on a shared grade.

**What the root can and cannot do** (stated plainly, because this makes the
root's reveal an input to every proof).

It cannot, because every TD and every validator checks it on what it holds
and carries the previous menu otherwise (a refused menu costs the subnet one
epoch of menu changes and nothing else, in a TD as on a validator):

- add an entry the compiled rules or the signed lists forbid;
- put an entry on the menu that was not a proposal in measurement or
  waiting in the menu before, or more than `MENU_MAX_JOINS_PER_EPOCH` in
  one epoch, or exceed the caps;
- take the menu below `MENU_MIN_ENTRIES`, publish an entry that costs
  nothing, or retire more than max(2, a tenth of the menu) entries for
  health in one step, so no menu it publishes leaves an epoch nobody can
  draw;
- remove an entry without the `retired` record and full cooldown a
  retirement writes, or end a proposal without what the step writes for it
  (a lapse gives the coldkey its cooldown too);
- remove a healthy entry, or drop a proposal, in the step that adopts a
  newer list unless that list refuses it, for the reason it refuses it;
- displace an entry from a menu that is not full after the step, or one in
  its grace;
- put an entry on the menu, or update one, on values other than its
  proposal's (price, answer limit, measured token means, request);
- write, re-date or lengthen a cooldown record, an id's or a coldkey's,
  that no step would write, or keep or drop one other than as the step
  does: so one menu cannot ban an id or a coldkey for good, or fill the
  document with records until no later menu fits `MENU_MAX_BYTES`;
- keep a passed proposal waiting past `QUEUE_WAIT_EPOCHS`, or a proposal in
  measurement past its make-up bound;
- skip a new entry's grace, or move any entry's `joined_epoch` or
  `scored_from`, so it cannot make an entry count at once and disqualify
  every head that has not trained a row for it;
- change an entry's request (hosts, quantization) except by a request
  update that was in the pipeline;
- file a proposal for an id or from a coldkey whose cooldown is still
  running, a duplicate, or a fifth for one coldkey;
- stretch a measurement: a proposal's window is the step's, it is decided
  in the step its measurement ends, and no window runs past
  `measure_deadline` however many epochs the root skips;
- shorten or drop a cooldown that is still running;
- move prices, answer limits, token means or exploration-pair counts
  outside a refresh epoch (beyond the entry that joins or is updated in that
  step), or move a price by more than a factor of four in one refresh;
- adopt a signed list nobody holds, or go back to an older one;
- hand a TD more than `MENU_MAX_BYTES` to parse, or bytes that stop it:
  anything that is not a menu is refused and carried.

It still can, because these follow from inputs only the root's verified
proofs carry, which a TD does not hold and another validator holds only as
its own, possibly different, proof set:

- **delay or omit**: publish no commitment, withhold a step, or leave out a
  valid filing, a measurement report, a price vote or a retirement;
- **stop the epochs**: commit a hash whose bytes it never publishes, or a
  menu that names a newer list it never publishes. Every node then has no
  menu (`MenuUnavailable`) and no TD produces a valid proof, for as long as
  it withholds them, because each later menu follows from that one. It
  stops validators as well as miners: a validator with no menu can check
  no v4 proof, so it scores the epoch under the no-proof rule and sets burn
  weights. **This is a new dependence**: in v3 a miner needs nothing from
  the root to produce a valid proof, and in v4 every proof waits on the
  root's availability whenever the menu changes. Publish-before-commit (above)
  stops an honest root's failed upload from doing this; it does nothing
  against a root that withholds on purpose. The candidate mitigation is
  mirrors: any node that holds the bytes can serve them, because every
  reader checks them against the on-chain hash, so one honest copy anywhere
  ends the stop. **Built:** any validator other than the root with a
  bucket of its own re-publishes each menu it accepted at a boundary, and
  each signed list it holds for it, under the root's content-addressed
  names (`menu_publish.Mirror`, create-only, each object once); every
  reader (the validators' node, a miner's TD through
  `reveal_sync.fetch_menu_bytes` / `fetch_signed_list`) also asks the https
  locations in `FUGAL_MENU_MIRRORS` (comma-separated, an operational
  variable), after the root's bucket, and takes only bytes that hash to
  what it asked for. Nothing depends on a mirror existing; who runs mirrors
  and where they are listed is the activation package's. A chain-anchored fallback (carry the previous menu when the
  bytes are "unavailable" for long enough) was considered and not designed:
  unavailable is not something every node observes identically, so such a
  rule would split the subnet on the nodes that did and did not fetch the
  bytes. Open item for the activation package (section 17);
- **author what the step decides from proofs**: prices, answer limits,
  token means and the exploration-pair counts, which proposals pass their
  gates and the measurement results they pass on (the token means of a
  proposal decided and joined in the same step included, bounded below by
  the pairs the held menu records), which entries retire (failing, gone,
  displaced) and which proposals lapse, and a new filing's coldkey and
  first price, which come from a proof only the root's proof set holds;
- **fabricate a proposal** the rules allow, with a filing no proof carried
  and measurement results of its choosing: the transition checks make it
  show in every published menu, in measurement for exactly its
  `PROPOSAL_MEASURE_EPOCHS` epochs (plus any make-up) and then in grace for
  `MENU_GRACE_EPOCHS`, before it is scored. It costs no miner anything: a
  TD measures only a proposal its own session filed, so one written under
  a miner's hotkey is not measured by that miner's box, and a proposer's
  own measurement is held to its ceiling at the larger of the menu's price
  and the billed cost (section 5).

The refresh cadence (4.4) bounds when the root may move prices, answer
limits, token means and counts: only in a refresh epoch, which every node
checks, and a price by at most a factor of four each time.

What it authors this way shows as a difference in every other validator's
`menu_audit`, which is published, but no node refuses a menu for it: two
honest validators can differ here for the same reason they can differ on a
shared grade. The lab and host allowlist bounds what a fabricated proposal
can name.

**Accepted limits, stated plainly:**

- **Selective serving.** A root that serves a menu's bytes, or a newer
  signed list, to some nodes and then withholds them leaves the nodes that
  fetched early on that menu and the rest with `MenuUnavailable` (no menu,
  no valid proof, burn weights on a validator) until a mirror serves the
  bytes. Every reader checks bytes against the on-chain hash, so one honest
  copy anywhere ends it; nothing in the protocol makes the root serve them.
- **An anchored TD takes the root's word for one menu.** A miner's TD that
  starts or has fallen behind anchors on the menu the root's newest reveal
  says its epoch ran on (5, "a TD that starts or falls behind"); it checks
  those bytes against the hash, the caps and the rules, but not the chain of
  steps before them. A root that names another menu there leaves restarted
  miners on a menu the validators do not hold: their proofs are refused on
  the namespace, which costs them, never the validators' consensus.
- **The root's hotkey is a compiled table.** Each epoch's commitment is
  read under the root hotkey in force for it (`config.FINNEY_ROOT_HOTKEYS_V4`,
  (first epoch, hotkey) rows; `menu_publish.hotkey_for_epoch`), so a node
  that rebuilds menus across a change of the root's key reads every epoch
  under the right one. A key rotation is a release that appends a row and
  keeps the old ones; a rebuild that meets an epoch no row covers, or a
  link it cannot verify, stops there and says so, and never goes on from an
  unverified menu.

### 4.4 The derivation step

`MenuState(k+2) = step(MenuState(k+1), inputs(k))`, a pure function. In order:

1. **Allowlists.** The step is handed the signed list the node holds: the one
   the state names, or a newer one (higher serial, signature verified). A
   newer one is adopted, and the step applies the compiled rules of section 6
   under it to every entry and to every proposal in measurement or waiting.
   An entry whose lab left the list retires (`lab_removed`). An entry with a
   pinned host that left the list, or that pins no quantization on a host
   that lost its first-party mark, retires (`host_removed`). A proposal the
   new list refuses is dropped the same way, with the cooldown; a refused
   request update leaves its entry as it was. **What a list refuses is
   computed by the step and is never an input to it**, and every node that
   holds the list recomputes it from the two menus. A newer list that would
   leave fewer than `MENU_MIN_ENTRIES` full entries is held back (a
   provisional entry does not count: the gates may retire it in the same
   step): the list in
   force stays, the epoch's filings are judged and admitted under it (the
   step is handed the list in force too, `StepInputs.lab_list_in_force`),
   and the newer list is adopted in the first step where it leaves at least
   the minimum. Until then the entries of a lab or host it removes stay on
   the menu. The seed names no list, so the first step adopts the one the
   release ships, and no step runs without a list.
2. **Prices and answer limits** from epoch `k`'s endpoint reads (section
   7.1): the price of the dearer listed pinned host, and the answer limit of
   section 6 over the listed ones. A pinned host the reads show as no longer
   listed is skipped while another is listed.
3. **Token means** from the window's pairs through epoch `k` (section 7.2),
   and each entry's **exploration-pair count**: the pooled exploration pairs
   of its current `rev` in the window through epoch `k`, handed in per
   (entry id, rev) by the aggregation over verified proofs and published as
   handed in. Measurement pairs never count. A count for another `rev` is
   ignored, and an entry with no count keeps its own. These counts, like the
   prices and the token means, are authored by whoever publishes the state:
   another validator's count comes from its own proof set and can differ for
   an honest reason, so no node refuses a menu for one; every node checks
   only what both know, that an entry joins and a request update starts at 0.
4. **Health retirements** (section 9.2).
5. **Proposals**: record epoch `k`'s measurement results (a proposal whose
   make-up would run past `measure_deadline` leaves, writing nothing:
   section 5); decide the ones that finished; admit up to
   `MENU_MAX_JOINS_PER_EPOCH` (2) from the queue, and let a queued proposal
   that has waited `QUEUE_WAIT_EPOCHS` (24) and is not admitted go, writing
   nothing (no record, no coldkey cooldown: its proposer may file again at
   once); file new ones up to `MENU_MAX_PENDING` (section 5). The step
   applies section 6's rules to each filing itself, under the list in force.

   **The queue's wait.** At the cap with nothing displaceable (no counted
   pick yet, as at activation, when no head holds the seat) passed proposals
   would wait for good and, counting towards `MENU_MAX_PENDING`, refuse every
   filing for room. `QUEUE_WAIT_EPOCHS` is the grace length, 24 epochs (a
   day and a bit): time for two joins an epoch and a seat that forms to make
   room, and it bounds a proposal's time in the pipeline by its
   measurement, a wait and a grace, the bound a joined entry already has.
   The proposer's measurement is spent either way; the published pairs
   remain.
6. `serial += 1` if anything changed.

**The refresh cadence.** Steps 2 and 3 are the refresh: prices, answer
limits, token means and exploration-pair counts (and the prices and limits
of proposals in measurement or waiting). They apply only in a step whose new
epoch is a multiple of `MENU_REFRESH_EPOCHS` = 20 (about a day). Any other
step leaves the refreshed values as published, whatever else it does: an
entry that joins or is updated takes its proposal's values, and a
structural change (an entry retires, a newer list is adopted, a proposal is
filed, decided or lapses) moves nothing else. One structural change is not
a licence to reprice the whole menu.

**A refreshed price moves at most a factor of `MENU_PRICE_STEP` = 4** up or
down (`menu_state.price_step`): one published state can do that much to a
price, and a real larger move arrives over consecutive refreshes (x16 in
two, x64 in three). A voted price of 0 is no change (an entry is never
priced at 0), and token means that would make a call cost nothing are not
taken.

The reason is the menu hash. A step that changes nothing gives the carried
menu, whose hash every node computes without new bytes; one that moves a
price changes the hash, and every miner's box then needs the new bytes
before its next proof. With the cadence the hash changes about once a day
plus on structural changes, so no epoch depends on fetching bytes just to
keep running. The coverage damping (section 8) already tolerates day-old
counts, and day-old prices are what v3's pinned table is at its best.

The step stays a pure function: the cadence is read from the state's own
epoch and a compiled constant. Every node checks the cadence and the price
step from the two menus alone (`check_transition`): a committed menu that
moves a refreshed value outside a refresh epoch, on any entry but one that
joins or is updated in that step, or a price beyond the step, is refused.
`menu_state.MENU_REFRESH_EPOCHS`, `MENU_PRICE_STEP`, `refresh_due` and
`price_step`; tests in `tests/test_menu_refresh.py` and
`tests/test_menu_root_limits.py`. A refresh epoch for which the root
published nothing is not made up: its refresh waits for the next refresh
epoch.

**`rev` counts changes of what is called, and nothing else.** A **request
update** is a filing for an id on the menu with a different provider order
or quantization. It goes through the proposal path of section 5 and bumps
`rev` when it joins. A price move or an answer-limit move (step 2) does not
change `rev`: the same hosts serve the same model, and the pairs measured
before the move still describe the entry. Pairs measured under an older
`rev` leave R and the token means for that entry, because a different host
can be different hardware and a different quantization. Heads keep their
row, and the entry keeps its `joined_epoch` and `scored_from`.

**An id that comes back starts at revision 1 again**, so its pairs carry the
same (id, rev) as its previous run's, and the statistics the step reads
from the window of epoch summaries (token means, exploration-pair counts,
accuracy, health) would mix the two runs if one window held both. It
cannot: the last pair of a run is in the epoch before the first menu
without it (R - 1); the id may be filed again in scored epoch R +
`PROPOSAL_COOLDOWN_EPOCHS` and is measured from R + cooldown +
`MENU_LAG`; measurement pairs count towards an entry's token means, so the
second run's first pair is that one, not the first scored one. Two pairs
of the two runs are therefore at least cooldown + lag + 1 = 48 + 2 + 1 =
**51 epochs apart**, and the window is `CHAMPION_WINDOW_EPOCHS` = **48 <=
51**. A proposal that fails its gates or lapses also puts its id in
cooldown, and its gap is larger (cooldown + 2 x lag). Measurement epochs
and grace do not enter: the bound has to hold from the second run's
measurement, which comes before both. `menu_state.rejoin_gap_epochs`;
`menu_inputs.check_summary_window` refuses a longer window at import and
the validator's v4 setup refuses to start with one
(`tests/test_menu_rejoin.py`). Revisions are kept as they are; a window or
cooldown changed later must keep the inequality.

A filing identical to the entry on the menu (same order, same quantization)
is refused, and so is a second filing for an id that already has a proposal
in measurement or waiting, a request update included.

**The `retired` record never drops a cooldown that is still running.** It
holds at most 256 records. When it is over that size, records whose cooldown
has run out make room, the oldest first. So that the size and the running
cooldowns cannot collide, a filing is accepted only while its id has a place
in the record: every id on the menu, in measurement or waiting may come to
need one, and each running cooldown keeps its own. A new id beyond that is
refused with the reason `no room`. A request update needs no new place.

### 4.5 The seed

`routing_data/menu_v4_seed.json`, built by a script from named, published v3
reveals and pinned by hash: v3's entries with their model, reasoning setting
and host order, prices and answer limits read from the listing at build time
and reviewed, and token means and exploration-pair counts from v3's pooled
pairs (the same committed pairs file). So v4 starts with
measured costs, and with entries that are failing at build time left out.
`MenuState` for the first v4 epoch is the seed.

**Seed entries have no grace.** They are on the menu and scored from the
first v4 epoch: the seed is written at epoch 0 with `joined_epoch` and
`scored_from` both 0, and `seed_state` moves only the state's `epoch` to the
first v4 epoch, so no seed entry is in grace in any v4 epoch.

Every seed entry passes the compiled rules of section 6 as an entry on the
menu, against the listing snapshot the seed was built from; the builder
stops otherwise. Of the 59, six pin a second host that snapshot does not
show (four `claude-fable-5.2` entries, two `gemini-4-carbon` entries) and
are priced and limited by the host that is listed. The seed names no signed
list: the first step adopts the one the release ships (4.4).

## 5. Proposals: commit first, then measure

A miner proposes one entry inside its epoch proof (no new commitment
channel). The order is the point of the design: the proposal is binding
**before** anyone, the proposer included, knows which questions will measure
it or how it will do.

| Phase | Epoch | What happens |
|---|---|---|
| **File** | `k` | The proposer's TD puts a `proposal` in its proof: model id, reasoning setting, provider order, quantization, and its own attested read of the model's endpoints (7.1). The TD checks the compiled rules first and refuses to file a proposal that fails them |
| **Accept for measurement** | root scoring `k` | The step applies the static checks (section 6), then: no duplicate (4.4), no cooldown on the id or the proposer's coldkey, at most `MENU_MAX_PROPOSALS_PER_COLDKEY` (4) of the proposer's coldkey's proposals in measurement and waiting together (a fifth is refused, `coldkey_limit`), room under `MENU_MAX_PENDING` (8), a place in the `retired` record (4.4). More valid proposals than room: order by `HMAC(nonce_k, proposal_id)`. Listed in `pending` of `MenuState(k+2)` with its measurement epochs |
| **Measure** | `k+2` .. `k+2+S-1`, later if the root publishes no step for one of them (below) | In each of `S` = `PROPOSAL_MEASURE_EPOCHS` (5) epochs the proposer's TD answers `PROPOSAL_MEASURE_PAIRS / S` (60) questions on the candidate, drawn by `HMAC(epoch nonce, proposal_id)` on the scored category mix, disjoint from its slice and exploration. Validators check the set exactly, as they check exploration. Calls run after the scored and exploration calls; one that misses its deadline is a failed call |
| **Decide** | root scoring the last measurement epoch | The gates below, on all 300 results |
| **Join** | two epochs later | The entry is on the menu, explored by every miner, scored for nobody until `scored_from` (section 9.1) |

A model is therefore explored by the whole field about 9 epochs (11 hours)
after it is proposed and scored about 29 hours after that.

**Gates** (all on the attested measurement):

- at least 95% of calls answered;
- at least one category at or above `PROPOSAL_MIN_CATEGORY_ACCURACY` on at
  least 20 graded questions, so a broken or junk model never joins. As
  built: 50% (`config.PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4` = 5000), the
  lead's choice pending the seat-rule package (the value was not examined:
  section 17);
- **plausibly useful**: a candidate is refused when some entry that costs no
  more, with at least 100 pairs, has an accuracy lower bound above the
  candidate's upper bound (one-sided 95% bounds on the mix-weighted rate).
  One such entry is enough; an entry that costs more, or has fewer pairs, is
  not compared with. Every entry on the menu dilutes exploration for all the
  others, so an entry that is both dearer and clearly worse than what is
  there has a cost and no use.

**A proposer that stops delivering** (no report for a measurement epoch that
a step ran for) loses the proposal; the entry and the proposer's coldkey get
a cooldown (`PROPOSAL_COOLDOWN_EPOCHS`, 48). A failed gate gives the entry
the same cooldown. All measured pairs are published either way.

**An epoch the root did not publish for is not the proposer's miss.** The
step for scored epoch `k` reads epoch `k`'s measurement report and no other.
When the root publishes no step for an epoch (no commitment, or a menu every
node refused, so the menu was carried: 4.3), nothing records what the
proposer measured in it. That epoch is neither counted nor missed:

- the proposal does not lapse and the proposer's coldkey gets no cooldown;
- the measurement is not decided on fewer than `S` epochs. It counts the
  epochs a step ran for, and the first step that runs afterwards adds the
  epochs the proposal is short of;
- the added epochs start where the proposer can see them. Its TD decides
  whether to measure in epoch `E` from `MenuState(E)`, and when a step runs
  the menus up to the one it starts from already exist. So a step changes
  the schedule only from the first menu it writes: a window still open
  there is lengthened, and one that has already closed gets a new run
  starting there;
- a report for an epoch no step ran for cannot be handed in later. Which
  epochs count is decided by the epochs, not by whoever assembles a step's
  inputs.

A proposer whose TD measures in exactly the epochs its menu names is
therefore never lapsed by the root's absence, and a make-up epoch is a
measurement epoch like any other: a step that runs for it and finds no
report lapses the proposal.

**Measurement pairs are never scored for any head.** They enter the window as
unscored pairs (`EpochPairs.scored` false, as menu B's grace pairs are), where
they seed R and the token means. That also settles the question of the
proposer scoring on pairs only it could prepare for.

**Make-up is bounded.** A root that kept skipping a proposal's epochs
would keep its proposer's box measuring, at the proposer's cost, for as long
as it liked. A proposal's window ends by `measure_deadline` = filed epoch +
`MENU_LAG` + `PROPOSAL_MEASURE_EPOCHS` - 1 + `PROPOSAL_MAKEUP_EPOCHS` (3,
`MENU_LAG` + 1: any single skipped step is made up; `SEAT_MINIMAL_V4.md`
4.6): its proposer measures in at most the planned epochs plus three. A step that would
take it past that lets the proposal go, writing nothing: no record and no
coldkey cooldown, because the epochs it could not finish are the root's,
and the proposer may file again at once (a new measurement, at its cost). A
filing too old for its first window to fit the bound is refused. Every node
holds a committed menu to the same rule, and the TD never measures past the
bound.

**The proposer's spend is bounded by its own setting.** It states one
ceiling in dollars for the box (`--v4-measure-cap-usd`): the total its TD
spends on measurement across all its proposals. The TD stops measuring at
it and the proposal lapses. Each call counts at the larger of its price on
the menu and what the provider billed; a call the provider reports no
billed cost for counts at the larger of its menu price and its cost at the
TD's own listing read of the proposal's hosts that epoch (the dearest
listed pinned host: the provider does not reliably name the host that
served it). The ceiling is checked before each batch of calls (so a
measurement stops at most one batch past it), and the TD measures only a
proposal its own session filed: its session records the id of each
proposal it files and ignores a pending proposal that merely names its
hotkey. The ledger of filed ids, each proposal's spend and the running
total is kept in the TD's state directory
(`<state dir>/measure_ledger.json`, on its encrypted data volume), written
after every change, so a restart does not reset the ceiling; it holds
proposal ids and dollar amounts, nothing secret. A measurement window cannot
be stretched by the menu: it is the step's (`measure_window`), within the
make-up bound, and the proposal is decided in the step its measurement
ends. A proposal whose measured call costs nothing at its price fails its
gates (`gate_cost`).

**A TD that starts or falls behind does not replay history.** A full menu
state is self-contained. For epoch E the TD (`td_v4.V4Session`) starts from
the first of these that reaches MenuState(E - 1) through every reader
check: the last menu it accepted, kept in memory and in its state directory
(`<state dir>/menu.json`, the canonical bytes, nothing secret), when at most
`RESUME_MAX_EPOCHS` (8) old; the newest root reveal published and verified
against the root's on-chain commitment read pinned at E's boundary block,
whose `menu` field (`{epoch, hash}`, already in the consensus view) names
the menu its epoch k ran on, those bytes fetched from any source and checked
against the hash, the caps and the compiled rules; the operator's anchor
file (`--v4-anchor`), a last resort; the kept menu however old; the seed.
From there it walks forward with the full reader checks: from a reveal, two
links (reveal k is public before the boundary of k + 2). The root puts the
bytes of the menu each reveal's epoch ran on in its store with the reveal,
so a carried menu's bytes are there too. Each pinned read is kept, and the
epoch's own menu is kept when it is read, before any call, so a node that
serves only recent blocks (about 300) costs at most the epoch whose older
boundary it cannot read, and the next epoch anchors on a newer reveal. A
read that fails costs that epoch's proof only, never the session, and the
session never goes on from a menu it could not verify. The miner takes the
validator's archive and fallback endpoints for its pinned reads
(`FUGAL_ARCHIVE_ENDPOINTS`, `FUGAL_FALLBACK_ENDPOINTS`; finney's public ones
by default).

**What being first buys.** The proposer's head has a row trained on its own
data before anyone else has seen the entry. After grace every head needs a
row to hold or take the seat (v3's full-menu rule), so a proposal sets a
deadline for the whole field that the proposer has already met. That is the
whole reward, as decided.

## 6. Rules compiled into the image

These change by release only (`fugal_subnet/menu_rules.py`). The TD applies
them to every entry of a menu it is given and to every proposal it files;
validators apply them to every proposal and to `menu_next`. Two functions
every node runs apply them without being asked: the derivation step, to
every filing and, when a newer list is adopted, to everything in the state
(4.4); and `menu_for_epoch`, to every committed menu that differs from the
carried one (4.3). Each check is a pure function that returns the rule it
refused under.

- **Lab allowlist.** The model's author (the id before `/`) is on the signed
  list. This is the defence against a model fitted to the public pool being
  listed and proposed (section 14, A1).
- **Host allowlist.** Every pinned host is on the signed list. A service tier
  (`/flex`, `/fast`, `/priority`, `/spot`) is a different product at a
  different price and is never a pinned host or a listed one. A regional or
  other variant of a host (`google-vertex/global`) is allowed only when the
  signed list names that exact host string: listing the plain host does not
  cover its variants, and listing one variant does not cover another.
- **Pinned order, no fallbacks**, at most `MENU_MAX_HOSTS` (2) hosts, one
  quantization across them. `unknown` is accepted only for the model author's
  own host and the hosts the list marks first-party: the list carries that
  mark per host (`first_party_hosts`), and a host that loses it no longer
  covers an entry that pins no quantization on it.
- **Reasoning** is one of a fixed set; `max_tokens` = min(131072, the smallest
  answer limit among the pinned hosts), as in v3.
- **Pinned hosts and the listing.** For a new proposal every pinned host must
  be in the listing (the proposer's attested read, 7.1) and state an answer
  limit. For an entry already on the menu, a pinned host the listing no
  longer shows is skipped when the answer limit and the price are computed,
  provided at least one pinned host is listed; a listed host that states no
  limit sets none. Such an entry stays on the menu. It retires only under
  the health rule, when none of its pinned hosts has been listed for 6
  consecutive epochs (9.2). The six seed entries of 4.5 are the case today.
- `validate_menu_v3`'s shape rules, unchanged.
- **One id per model and reasoning setting** (`model@effort`, or the model id
  for a setting without an effort), so the same call cannot sit on the menu
  under two names. A filing for an id on the menu with a different host
  order or quantization is a request update (4.4), unless that id already
  has a proposal in measurement or waiting, which refuses it. A filing
  identical to the entry on the menu is refused.

**The allowlists are signed data, not code**, so the owner adds a lab or a host
without a release. One file, `{format, kind, serial, labs[], hosts[],
first_party_hosts[]}` in one canonical byte layout, signed with a release key
under its own signature namespace, `fugal-list@fugalai.com` (the keys in
`deploy/keys/allowed_signers` are listed for `git` and for that namespace; a
signature made for one is not a signature for the other). A newer list
reaches the subnet through the root's reveal and applies at the menu lag, so
every node switches at the same epoch. A list with a lower serial or a bad
signature is ignored; serial 0 marks a draft and is refused whatever signs
it. The image and the validator ship the list current at release, and the
state names the list in force by serial and hash (4.1). No list is signed
yet: an unsigned draft derived from the v3 menu is in `data/` for the owner
to review.

## 7. Prices and token means are measured

### 7.1 Prices: every TD reads the listing, every epoch

OpenRouter's `GET /api/v1/models/{id}/endpoints` is public (checked
2026-11-18 without a key: HTTP 200). Per host it returns `tag`
(`deepinfra/fp8`), `quantization`, `max_completion_tokens`, `pricing.prompt`
and `pricing.completion` as decimal strings per token, `status` and uptime.
It is under the one base URL the TD may call.

Each epoch the TD reads the endpoints of every model on the menu and in
`pending` (about 30 GETs, no spend) and puts a compact record in its proof.
Validators take, per (model, host, field), the **median across coldkeys**,
with at least `ENDPOINT_MIN_VOTERS` (3) coldkeys or no change. An entry's
price is its dearer pinned host's (v3's rule), its answer limit the smallest,
both among the pinned hosts that are listed (section 6).
A read that fails is no vote. Nothing waits on it.

So a price change needs no proposal and no person, and one operator cannot
move a price. The trust is what it already is: the TD's upstream is inside
the measurement (I8), and OpenRouter is believed about OpenRouter.

**Billed cost is attested too.** `MeteringProxy` already reads `usage.cost`
(`tee/runtime.py`, `provider_cost_usd`) and the proof does not carry it. v4
adds it per result. Validators publish, per entry, billed spend against
tokens x listed price. It decides no score. A ratio outside a band over 50
pairs is published as drift, which is how a host that bills differently from
its listing becomes visible. The band is kept per entry and attributed to a
host from the endpoint reads: a reply names its provider by display name
(`served_by`), not by the pinned host tag (section 12). (Checked on already-paid answers from
2026-11-17: `usage.cost` came back on every successful call for 19 of the 25
menu models; the other six were not called.)

### 7.2 Token means: from the pool, weighted to the scored mix

An entry's `mean_in` and `mean_out` are the means of attested tokens over its
window pairs of the current `rev`, per category, weighted to the scored
category mix (the weighting `single_model_table` already uses for accuracy),
in exact rational arithmetic. Means, not medians: what a user pays in
expectation is the mean, and a few very long answers are part of it. With
fewer than `TOKEN_MIN_PAIRS` (30) pairs the previous value stands. A new
entry starts from its proposal measurement.

This replaces the pinned token manifest from the v4 block. v3's manifests
stay pinned for v3 epochs.

### 7.3 Where the measured cost is used

| Use | v3 | v4 |
|---|---|---|
| A head's pick (odds minus lambda x cost) | `rank`, with token means baked into the head, checked against the manifest | `head_odds.picks_and_odds_v4`: exact odds and the epoch's `MenuState` pick costs (`menu_state.pick_costs`, W13); token means in the head are not read (section 11) |
| Exploration shares and weights | pinned prices x pinned means | the epoch's `MenuState` call costs (`menu_state.costs`) |
| Net value on the scored slice | attested tokens of the cell x pinned price | the epoch's pick cost of the entry in the question's category (`menu_state.pick_cost`, W13) |
| The product | vendored tables | `MenuState` from the root's reveal: the per-entry pick cost |

As built (W13; `SEAT_MINIMAL_V4.md` 3.4 and the earlier seat design's 8.2):
an entry's call cost is its token means over exploration pairs; its pick
cost is the token means of its graded scored-question cells, shrunk toward
the call means by 30 cells and per category toward its pick means by 20, an
optional `pick` object on the entry, refreshed at the cadence. The two rows
for a pick and for net value above read it; exploration reads the call
cost. `docs/INVARIANTS.md` (the W13 entry) says what holds.

A head's picks therefore depend on its bytes and on published state, both
fixed before the epoch's nonce exists.

## 8. Exploration that measures every entry

A 300-pair proposal measurement from one operator would outweigh what the
whole field knows about a dear entry for days (section 2). The fix is not a
smaller measurement. It is exploration that keeps measuring every entry.

**Mechanism.** A miner draws the shared set as in v3 (8 calls) and a
private set of `EXPLORE_PRIVATE_SIZE` = 40 calls in two parts:

- the **cost-aware part**, 24 calls, drawn as v3 draws its private set, with
  its expected-spend cap `EXPLORE_SPEND_CAP_USD` = $0.25 a hotkey-epoch. As
  in v3 the cap is on every call the cost-aware shares draw: the 8 shared
  calls and these 24, 32 in all;
- the **coverage part**, `EXPLORE_COVERAGE_SIZE` = 16 calls, which keeps
  every entry measured. Its shares are fixed in three steps, each a function
  of published state:
  1. Each entry's deficit is `max(0, EXPLORE_COVERAGE_TARGET - n_e)`, with
     `EXPLORE_COVERAGE_TARGET` = 100 and `n_e` the entry's **exploration**
     pairs in the window as published `MENU_LAG` epochs back: its
     `explore_pairs` in the epoch's `MenuState` (4.2; the step writes the
     count through epoch `k` into the menu of `k + MENU_LAG`), so every node
     reads the same number from chain-anchored state
     (`epoch_plan.explore_shares_v4(state)` reads `state.explore_pairs()`
     and nothing else). Under the refresh cadence (4.4) the count is at
     most about a day old, which the damping of step 2 tolerates. It counts the current
     `rev` only, so a re-pinned entry shows a deficit again. A proposer's
     measurement pairs are not counted: if they were, a new entry would
     arrive with 300 pairs from one operator and the field would never
     cover it, which is the problem this section opens with. The count is
     the publisher's (4.4), so the root can steer the coverage part's shares
     by misstating it, within the spend cap of step 3 and visibly in
     `menu_audit`.
  2. Only the fraction `f = min(1, sum of the deficits / 500 pairs)` of the
     coverage part follows the deficits (shares proportional to them); the
     rest is uniform over the menu. Without the damping a single entry with
     a deficit (one that has just joined, been re-pinned or aged out) takes
     every coverage call of every miner for three epochs, until its pairs
     show in the published window: 320-570 pairs where 100 were wanted, and
     $2.22 an epoch if it is priced like the dearest entry today. With the
     damping and step 3's cap the same join costs $0.56 at its worst epoch and lands at
     160-260 pairs in 4-8 epochs.
  3. The coverage part is mixed toward uniform until its expected spend is
     within $0.40 a hotkey-epoch: shares `t x q + (1 - t) x u`, `q` from step
     2 and `u` uniform, with `t` the largest value in [0, 1] whose expected
     spend is within the cap. Spend is linear in `t`, so `t` is an exact
     fraction. Uniform coverage costs $0.155 (0.131-0.189) at measured costs,
     so the cap binds only on a spike.

`EXPLORE_COLDKEY_CAP` = 60, 1.5x the private size as in v3: the top operator
stays at 16-18% of the pool.

Every draw probability is known exactly to every node, so the pair weight
`1 / (M x pi_e)` stays an exact inverse probability and the odds score stays
unbiased (`explore_weights`). The shared set and the private set are now
drawn with different probabilities, so each pair carries the weight of the
set it was drawn in: **a shared pair the shared set's (the cost-aware
shares), a private pair the mixture of the private set's two parts**, never
the weight of the one part that happened to draw it. As built (W5): the 40 private questions are
drawn as v3 draws its private set and split by a keyed rank of the question
id, the first 24 to the cost-aware part, so a private question is in each
part with probability 24/40 or 16/40 whatever its benchmark and the mixture
is its entry's exact probability. A weight from the coverage part alone
would not do: that part can give an entry probability zero. The cost-aware
cap is on the 32 calls its shares draw (the shared set and the cost-aware
part), as v3's cap is.

**Neither cap is an absolute bound** when the whole menu is dear. The
cost-aware part takes the steepest cost tilt the floor allows
(`EXPLORE_COST_MAX_HALVES`) when no tilt meets $0.25, which happens when
even that tilt's mean call costs more than $0.25 / 32, about $0.0078: a menu
whose cheap entries all cost that much. The coverage part, when uniform
coverage alone costs more than $0.40 (a menu whose mean call costs more than
$0.025), takes the cheaper of uniform and the damped deficit draw, and its
expected spend is above the cap. Today's measured menu is far from either
(the cost-aware part $0.163, uniform coverage $0.155). Both are still caps on
expected spend, and the bill is still bounded by the menu's dearest entry
times the calls.

**For the validator side (W4):** pool the pairs with both weight tables,
`SharesV4.shared_weights()` for shared pairs and `private_weights()` for
private ones, passed to `explore_pool.pool_pairs`; and apply the v4 coldkey
cap (`EXPLORE_COLDKEY_CAP_V4` = 60) in the odds input, not v3's. The
activation package adds the v4 constants to the consensus fingerprint. Lifting the dear entries' probability also
brings their weights toward one, which lowers the variance of every seat
test.

**What it buys and costs** (W0's model, at measured costs, a 59-entry menu,
200 runs; ranges 5-95%):

| | v3 today | This setting |
|---|---|---|
| Calls a hotkey-epoch | 32 | 48 |
| Epochs from an empty window until every entry has 100 pairs, at 5 / 7 / 10 operators | never / never / never | 29 (28-30) / 21 (20-22) / 15 (14-16) |
| Exploration spend a hotkey-epoch: steady / worst epoch | $0.16 / $0.16 | $0.32 (0.28-0.37) / $0.49 |
| Design effect of the weights | 1.31 | 1.11 |
| Seat-test variance an epoch, against v3 | 1.00 | 0.52 |

A head 0.25 points ahead clears the 0.1 margin in about 8 epochs instead of
16. A joined entry reaches 100 field pairs in 4-8 epochs, inside its 24-epoch
grace. The largest possible weight rises from 2.9 to 4.8 (an entry with no
deficit while others have one). From where the subnet is now (the real
24-epoch window) every entry reaches 100 pairs in 12 epochs (9-13); with a
64-entry menu, 30 / 22 / 16 epochs and $0.34. With three operators nothing
reaches 100 pairs reliably; the damping keeps them from oscillating. Every
size here assumes at least five operators.

**The scored slice stays at 300 questions** unless the seat-rule package
changes the net-value guard so that it needs fewer questions; it goes to 200
only together with that change. This is an open design item of the seat
rules (section 17), and the evidence is W0's:

- The slice costs a hotkey $5.53 an epoch (4.84-6.29), 36 times the
  exploration bill. 100 fewer questions save $1.84 (1.61-2.10) at today's
  mix and $0.90 (0.79-1.02) at the mix of section 10, against $0.16
  (0.13-0.19) for the 16 coverage calls.
- But the slice is what the net-value guard reads, and the guard is what
  holds the seat today: two heads have passed the odds test and the skill
  test since `e00026485` and are refused each epoch on the guard's lower
  bound while their measured net value is positive. At 200 questions they
  would wait 1.5 times longer.
- The guard's per-question SD is 47 points, 42 of them from realized cost.
  Pricing the guard at the routed entry's mean cost (the number the head's
  own pick uses) instead of each cell's realized tokens brings it to 26: a
  router one net-value point better then passes after 10 epochs with
  probability 60% at 200 questions, against 27% today at 300. About 5,400
  scored questions give such a router an even chance today, about 1,600
  with mean costs (2,200 at section 10's mix). The other way, with no
  change in kind, is a wider margin: an equal router passes on a full
  window with probability 58% at -1.0 and 93% at -1.5, at 300.
- If the slice goes to 200, the three 1,500-question counts (burn-in,
  minimum paired, confirmation) go to 1,000, so they stay five epochs. The
  accuracy gates are not the constraint at either size.

Pricing the guard at mean cost changes section 7.3's row for net value and
needs the owner's OK.

The spend caps stay caps on expected spend, for the reason v3 gives: a rule
that stopped at a realised total would break the weights.

## 9. Grace, retirement and the cap

### 9.1 Grace is per entry

**Replaced for v4** by `SEAT_MINIMAL_V4.md` section 4 (built as package S2,
`fugal_subnet/labels_v4.py`): an entry's labels are scored for every head
from its join, and a head without its row is scored at the entry's measured
rate for a 20-epoch allowance, certainly wrong after. `scored_from` is now
the end of that allowance. The text below is the rule it replaced.

v3's menu B rule, made general. From `joined_epoch` an entry is explored by
everyone. Until `scored_from` its pairs are unscored for every head alike and
the full-menu rule does not ask for its row. From `scored_from` a head
without a row for it cannot take or hold the odds seat.
`routing_protocol.grace_excluded` and `scored_menu` become functions of the
entry's own epochs instead of one block: under protocol version 4 they take
the epoch's own `MenuState` (`menu_state=`) and refuse to run without it or
with another epoch's. Scoring a missing row at R's odds during grace stays
rejected, for the reason recorded in `V3_MENU_B.md`. Rows are matched by
entry id, so a row for an entry not on the epoch's menu is never read, and
an entry retired during its grace never asks for a row. Seats, burn-in and
the window carry across menu changes.

Seed entries have no grace: their `scored_from` is the first v4 epoch (4.5).
A request update does not start a new grace period either (4.4).

At most `MENU_MAX_PENDING` (8) entries are in measurement or grace at once, so
the field is never asked to add more than a few rows a day.

**Routing to an entry in grace is not prevented.** A head may carry a row
for an entry still in grace, and its picks may then route scored questions
to it (`head_odds.picks_and_odds_v4` prices every entry with a row), though
no head's odds are scored on that entry until its `scored_from`. The scored
slice's grades of those picks count as any other pick's. Whether routing
should also wait out the grace (picks restricted to scored entries) is an
open item for the seat-rule package (section 17).

### 9.2 Retirement

**Health** (automatic, any time):

- *Failing.* Over the last `RETIRE_WINDOW_EPOCHS` (12) counted epochs, a
  coldkey with at least `RETIRE_MIN_CALLS` (5) exploration calls to the
  entry is eligible. An eligible coldkey is failing on the entry when more
  than `RETIRE_FAIL_RATE` (30%) of those calls failed and its failed calls
  fall in at least 3 distinct epochs. The entry retires when at least 4
  coldkeys are eligible and more than half of them are failing. **The
  outage guard:** an epoch in which the median coldkey failed more than 30%
  of all its exploration calls, to every entry, is an outage and is not
  counted. The majority of coldkeys is what stops one operator retiring an
  entry by dropping its own connections; the distinct epochs and the guard
  are what stop one bad epoch retiring healthy entries.
- *Gone.* The endpoint reads (7.1) show none of its pinned hosts listed for 6
  consecutive epochs. While at least one is listed the entry stays, priced
  and limited by the hosts that are (section 6). **The same kind of outage
  guard as failing:** an epoch whose decided reads show more than 30% of
  all hosts unlisted is a listing outage (a format change, an outage that
  still answers) and is not counted (`menu_inputs.listing_outage`).
- *At most a few a step, and never below the minimum.* A step retires at
  most max(2, a tenth of the menu) entries for health, in id order, and
  never takes the menu below `MENU_MIN_ENTRIES` (8) full entries (the
  minimum counts full entries only, because the gates may retire every
  provisional entry in one step; displacement waits at the minimum too);
  the rest wait for the
  next step that still sees them failing or gone. A newer signed list that
  would leave fewer than the minimum is not adopted: the list in force
  stays, the step's filings are judged under it, and the newer list is
  adopted in the first step where it leaves at least the minimum (the
  entries of a lab or host it removes stay on the menu until then, 4.4).
  So no menu the step derives leaves an epoch that cannot be drawn.
- *Lab or host removed* from the signed list: the step applies the compiled
  rules under the newer list and retires what they refuse (4.4).

The thresholds are W0's. On the replay (24 epochs, 59 healthy entries and
one real break: `xiaomi/mimo-v2.5-pro` failed every call of every operator
from `e00026478`) this rule retires the broken entry four epochs after it
broke, with no false retirement, none under one hostile operator (even with
the broken host the data shows), and none under one to three epochs in which
every call fails. The rule it replaces (10 calls, 20%, at least 3 coldkeys,
no distinct-epoch count, no guard) retires up to 8 healthy entries after one
such epoch, because a coldkey with 10 calls can have 3 of them in it; lets
one hostile operator plus one broken host make a majority of three; and
catches a broken dear entry 15% of the time within 16 epochs, because a
coldkey makes a median of 4 calls to a dear entry in 12 epochs at v3's draw
(under this rule, 40% in 8 epochs, and 92% in 6 for entries of middle
cost, which is about how often section 8 calls a dear one). What it cannot
stop: two colluding large operators can still retire up to 7 entries (a
coldkey majority is as strong as the number of operators, 5-7 today), and a
whole lab failing for three epochs retires at most 2 of its entries.

As built (W4a): the aggregation over verified proofs
(`menu_inputs.health_stats`) counts, per entry and current revision, the
eligible and the failing coldkeys over the last 12 counted epochs, and the
step (`menu_state.health_retires`) retires on at least
`RETIRE_FAIL_MIN_COLDKEYS` (4) eligible with more than half failing. The
constants are `menu_state.RETIRE_WINDOW_EPOCHS`, `RETIRE_MIN_CALLS`,
`RETIRE_FAIL_RATE_BP`, `RETIRE_FAIL_MIN_EPOCHS`, `RETIRE_FAIL_MIN_COLDKEYS`
and `RETIRE_OUTAGE_RATE_BP`. Coldkeys are the chain's owners of the proofs'
hotkeys, so one operator is one coldkey however many hotkeys it runs.
"Gone" counts consecutive epochs, ending with the scored one, in which the
endpoint vote shows every pinned host unlisted by at least
`ENDPOINT_MIN_VOTERS` coldkeys; a missing epoch or too few voters ends the
run. Exploration calls before an entry's `joined_epoch` are an earlier run
of the id and are not counted. The seed's build-time filter
(`scripts/build_menu_v4_seed.py`) keeps the earlier median rule as its own
constants: its pairs file holds per-operator totals, not per-epoch counts,
and the seed it built is pinned.

Two limits of the built rule, accepted (W4a):

- **A coldkey majority that fails every call makes every epoch an
  outage.** The outage guard leaves out an epoch in which the median
  coldkey failed more than 30% of all its exploration calls; operators who
  are most of the field and fail everything (an outage of their own, or on
  purpose) make every epoch such an epoch, so nothing retires as failing
  while they do. It costs those operators their exploration and their
  scores, and an entry that is truly broken is still caught once they stop.
- **The vote trusts the chain's hotkey-to-coldkey map.** Coldkeys are read
  from the metagraph at the collection block. v1-v3 fall back to the
  current metagraph when that pinned read fails, and say so in the epoch
  log. That is not acceptable for a consensus input: under v4 the map feeds
  the root's step (the endpoint vote, health, the coldkey bounds) and every
  audit, so a v4 epoch whose pinned read fails is retried instead
  (`validator._v4_require_pinned_metagraph`), as a failed read of the menu
  commitment is. The pinned read is at most one epoch old, within what a
  node serves, so the retry costs liveness only when the node cannot read
  the chain at all.

A retired entry leaves exploration and the menu at the lag. Heads' rows for
it are ignored. It can be proposed again after its cooldown, typically with
other hosts. Its record in `retired` is kept for as long as that cooldown
runs, whatever else is retired meanwhile (4.4).

**The cap** (`MENU_CAP`, 64). Nothing is retired for being unpopular while
there is room. When a proposal has passed and the menu is full, it takes the
slot of the entry with the lowest **use**: the share of scored-slice picks
that went to it, over the window, by heads that held the seat or were senior
bars. As built: the champion and the former champions the seat keeps, and
every head among the senior bars the epoch's seat was decided with
(`v2_epoch.V2EpochResult.senior_bars`, which `copy_rule.senior_bars` gave
`seat_step`; a senior bar can hold no seat now, as a retired champion still
barring through its retire window). A head is counted once, by weights
hash. Ties go to the lower measured accuracy, then the id. An entry in grace,
or with use above `RETIRE_USE_FLOOR` (1%), cannot be displaced. If nothing can
be, the proposal waits in `queue`.

Use, not a frontier: a model that is not on the accuracy-against-cost frontier
overall can still be the right call for one kind of question, and whether the
best routers actually call it is the direct evidence. A miner can keep an
entry alive only by routing real scored questions to it, which costs it.

## 10. The question mix

Odds can be shown to be good only where outcomes vary. On a benchmark where
every entry is right 99% of the time, saying 0.99 for everything is correct
and no head can beat it, so those pairs add noise to every seat test and
nothing else.

For v4:

- **Targets are derived, not hand-set: the p(1-p) rule.** A category's
  share of the scored slice and of exploration is proportional to the
  outcome variance of a pair, `p(1-p)`, with `p` the accuracy of an entry
  drawn uniformly (inverse-probability weighted) on the category's
  exploration pairs; floor 5%; rounded to 5% by largest remainder. That
  variance is the most a head's odds can explain on the category, and it
  is the quantity the published data pins down: its intervals are narrow
  enough to order the categories (hard 0.191-0.211, the next 0.074-0.114).
  Inside a category, sets split by their own `p(1-p)` x pool size, a set
  with under 100 exploration pairs taking its category's value, and no set
  gets more than pool size / (slice x 96 epochs), so none is cycled faster
  than two windows. Computed once by a script from named reveals, pinned in
  the v4 pool manifest, in the digest. Not adaptive at run time: a mix that
  moved with the window would be a feedback loop inside a consensus input.
  As built (W8): `scripts/build_mix_v4.py` from the observations of reveals
  `e00026466`..`e00026489` (the replay's epochs; it reproduces the table
  below), pinned as its own file, `routing_data/pool_mix_v4.json` (sha256 in
  `check_safety_invariants.py`; v4's pool is v3's, so there is no v4 pool
  manifest yet), and read through `epoch_plan.plan_v4`, the one constructor
  of a v4 epoch's plan. The p of a category pools the cells of the seed
  menu's entries. **What W8 measured:** at a 300-question slice the cycle
  cap binds for every hard set except `aime`, and for `code_contests` and
  `code_mbppplus`; hard can hold at most 30.288% of the slice, so its 30%
  fits with 0.29 points to spare, and about 8.6 questions an epoch spill
  onto `aime` (96.7% accuracy on exploration pairs, close to saturated). At
  a 200-question slice nothing binds. **The slice size is the open item it
  already is** (section 8, 17: the seat-rule package decides 200 together
  with the guard change); the mix file is built for one size
  (`config.SLICE_SIZE_V4` = 300, which `epoch_plan.plan_v4` checks), so
  changing it means rebuilding and re-pinning the mix with
  `scripts/build_mix_v4.py build --slice 200`. Every number weighted to the
  scored mix uses this one under v4: the step's token means and proposal
  gates (`epoch_plan.menu_step_params_v4`) and the seed's token means.

  From W0's 9,970 exploration pairs:

  | Category | v3 | v4 | Exact (95% interval) | Uniform-entry accuracy |
  |---|---|---|---|---|
  | hard | 20 | **30** | 30.9 (28.8-33.1) | 71.9% |
  | code | 15 | **15** | 13.3 (11.4-15.1) | 90.4% |
  | math | 10 | **15** | 14.5 (11.8-17.1) | 89.4% |
  | knowledge | 20 | **15** | 14.3 (12.6-16.0) | 89.6% |
  | instruction | 15 | **15** | 14.0 (12.2-15.9) | 89.8% |
  | easy_math | 5 | **5** | 8.0 (5.0-11.3) | 94.4% |
  | everyday | 15 | **5** | 5.0 (the floor) | 99.4% |

  The rule this replaces, a share proportional to the gap between the best
  entry's and the cheapest tier's accuracy, rests on a number the data
  separates from zero for one category only (code): it would give code 35%
  and could not place a tenth of the slice under the cycle cap. Every rule
  compared (the gap, the between-entry SD, scored disagreement, `p(1-p)`)
  agrees on the two large moves, everyday (ARC) to the floor and knowledge
  down, and they differ on code (15-35), math (10-25) and instruction
  (10-25), which is the honest size of the uncertainty. At a 200-question
  slice this mix draws about 53 `supergpqa_hard` questions an epoch (the
  whole set about every 130 epochs) and no cycle cap binds. A question
  costs 26% more under it ($0.0224 against $0.0179 at the heads' current
  picks).
- **No question leaves the pool for being answered by every graded entry.**
  The earlier draft removed questions every graded entry answered, on at
  least four entries. It is not built: 81% of the scored questions seen
  (5,695 of 6,991) meet it, 58% even of `supergpqa_hard`, and the graded
  entries are the ones heads picked, so it says nothing about the cheap
  entries a router would send the question to instead. A saturated set is
  handled by the mix (ARC at the floor), not by removing questions.
- **More hard material** from sets already loaded
  (`supergpqa_hard`, `omnimath_hard`, `beyondaime`, the contest code sets)
  and any new set the owner approves (licence, RouterArena overlap and
  contamination checked as for v3). New questions need embeddings, which the
  release recomputes anyway if the encoder changes (section 11).

The longer-term answer, already in the roadmap, is to derive the mix from the
kinds of question the product is asked. v4 does not depend on it.

**The price at which the exam is taken** is a related owner decision
(section 17). Picks are made at `BENCHMARK_LAMBDA` = 1, which values a correct
answer at a dollar; the guard uses $0.49. At either, cost is small next to
accuracy (3.9 of the champion's net-value points at the guard's price: W0,
section 2), which is why the scored slice goes to dear entries and costs
miners what it does. Taking the exam at the product's default price setting would
make the slice measure the router users get and cut what the slice costs.

## 11. Heads: format, cost, publication

### 11.1 What v4 fixes whatever the study finds

- **Rows are keyed by entry id.** A head may carry up to `HEAD_MAX_ROWS_V4`
  (72) rows: the menu cap (64) plus `MENU_MAX_PENDING` (8), so a head can
  hold rows for entries that have not joined yet (v3's cap is 64).
  Rows for ids not on the epoch's menu are ignored, so a proposer can commit
  a row before its entry joins and a head survives a retirement.
- **A head need carry no token means and no cost profile.** `mean_in_tokens`,
  `mean_out_tokens` and `cost_profile_id` are optional, checked as today when
  present, and never read; cost comes from the epoch's `MenuState` (7.3).
  The vendored contract (`vendor/success_contract.py`) requires them and is
  byte-compared with fugal-core's in CI, so the v4 acceptance is written
  beside it (`head_eval.validate_head_v4`) and the vendored file is
  unchanged; what fugal-core's contract must add is listed in 11.2.
- **Odds stay exact integers.** Whatever a head computes, the validator must
  compute it to the same integer on every host (`head_odds.py`: exact
  products, `math.fsum`, a pinned threshold table, no exponential).

### 11.2 The head format: unchanged (decided 2026-11-19)

**Decision.** v4 keeps today's head: a linear layer, `W` (rows x 1024) and a
bias `b`, on the frozen Qwen3-0.6B question embedding already pinned, under
today's contract, profile and backbone. No encoder change, no hidden layer,
no per-question cost head, no extra features. What changes is 11.1: rows
keyed by entry id and aligned to the epoch's menu, and cost taken from the
menu.

**Why.** An offline study on published epochs (20 held-out v3 epochs fitted
time-forward; 95% bootstrap intervals over questions; one point is 0.01
Brier, positive favours the alternative; the first interval is on the seat's
pooled pairs, the second on raw exploration cells) found the purpose-built
Qwen3-Embedding-0.6B worse (-0.34 [-0.67, -0.02]; -0.34 [-0.52, -0.15]) and
a 64-wide MLP no better (-0.02 [-0.18, +0.13]; -0.06 [-0.17, +0.04]) than
today's linear head. A per-question cost head added nothing beyond a
per-entry cost table measured where routers spend (net value x100 at lambda
1: +1.23 for the head, +1.38 [+0.95, +1.87] for the table, both against
pinned means), which is what v4's measured costs are (7.2), and prompt-shape
features gave +0.06 [+0.03, +0.08] on one set only, too little for a new
consensus surface. The one clear gain was in training, a shared direction
plus per-entry rows over independent rows (+0.40 [+0.12, +0.68]; +0.68
[+0.52, +0.84]), and today's format already holds such a head
(`W_m = w0 + V_m`), so it is miner guidance, not a protocol change.

**A v4 head, as built** (`head_eval.load_head_v4`, `INVARIANTS.md`):

- *Required:* `W` (rows x 1024) and `b` (rows), numeric and finite, every
  value within `HEAD_MAX_ABS_PARAM` (10^6); `models`, 1 to 72 unique,
  non-empty entry ids of at most 256 characters; `contract`
  (`fugal-success-v1`), `profile_id` and `backbone_revision` equal to the
  pinned values; a non-empty `provenance`; a finite non-negative `lam`,
  which no pick reads (as in v3).
- *Optional, never read:* `mean_in_tokens`, `mean_out_tokens` (rows, finite,
  non-negative) and `cost_profile_id` (64 lowercase hex), checked as today
  when present. Any other field is refused. Every head valid today is a
  valid v4 head; its rows must name v4 entry ids to be read.
- *Bytes:* as today: 1 MB, 8 MB expanded, at most 32 `.npy` members, no
  object dtype, no pickle.
- *Picks and odds:* `head_odds.picks_and_odds_v4(head, state, hidden,
  lam=None) -> MenuOdds(entries, odds, picks)`, the one function the TD
  routes with and a validator recomputes with. Odds are v3's exact integers
  for the state's entries in the state's order (-1 where the head has no
  row). A pick maximises odds / 10,000 - lambda x `menu_state.costs(state)`
  in Fractions, lambda the benchmark setting (section 17, item 1), ties to
  the first entry in the state's order; a question with no row on the menu
  has no pick. `pair_odds_v4` gives the same cells for pooled pairs. A
  head with no row on the epoch's menu cannot run the epoch
  (`head_eval.v4_head_verdict`).

**What fugal-core's contract must add** before the product can serve a v4
head (W10); the subnet then re-vendors it and `validate_head_v4` becomes a
call into it:

1. Acceptance of a head without `mean_in_tokens`, `mean_out_tokens` and
   `cost_profile_id` (each checked as today when present), with up to 72
   rows, under the same `contract` string, or under a new one if fugal-core
   prefers a version (the subnet's acceptance then follows it).
2. The v4 pick from costs it is handed: exact odds (exact products,
   `math.fsum`, the pinned threshold table of `head_odds.py`), Fraction
   costs, ties to the first entry in the menu's order. `rank`'s float
   utility on a 1e-4 grid is a different rule, and the product's routes
   must equal the reveals' (11.3).
3. No call of `estimated_cost` or `check_manifest` for a v4 head.

**Still open: whether a hidden layer pays at miners' data scale.** The
study's heads were fitted on about 60,000 public rows and reach about +0.3
skill; live heads reach +1.2 to +1.9, so miners hold far more labels, and a
hidden layer that gives nothing at public scale could still help at ten
times more. The cheap experiment (offline, free, about 30 minutes): a
learning curve on 25, 50 and 100% of the public rows, the 64-wide MLP
against the shared-direction linear head; if the gap does not grow with
data, the MLP is dropped. The decisive one: an optional MLP head scored in
shadow, with no weight, on the same pairs for 10 testnet epochs. A later
format must meet the constraints below.

Constraints any later format must meet:

1. **Exact evaluation.** A hidden layer is admissible if each layer is exact
   products, a correctly rounded sum, a rounding to float32 and a ReLU, so
   the output integer is the same on any CPU. Anything needing a transcendental
   function inside the network is not.
2. **Validator cost.** Odds for every head on every pooled pair, each epoch,
   on a 2-vCPU validator inside the epoch: this bounds hidden width and row
   count.
3. **One encoder, pinned.** A new encoder means a new pinned embeddings file
   for the whole pool (the owner computes it once) and the product embedding
   every user prompt with the same model and profile.
4. **Size and parse bounds** as today (`HEAD_MAX_BYTES`, no pickle, bounded
   arrays, `HEAD_MAX_ABS_PARAM`).
5. A predicted cost, if adopted, enters the pick only. It is not scored by
   itself: a head that mis-states cost picks worse and pays for it in net
   value.

### 11.3 Verified heads are published

With each reveal, the root uploads the bytes of every head it verified that
epoch to the public store as `heads/<netuid>/<sha256>.npz`
(`head_publisher`; `reveal_sync.fetch_head` reads one back and checks the
hash), on the reveal's own delay for a seat holder's head and after
`FUGAL_PUBLISH_HEADS_AFTER_EPOCHS` for any other. The name is the sha256 of
the bytes, so anyone checks them against the reveal's `head_hashes` and the
miner's on-chain commitment. This is operational: no score reads it, and
other validators may do the same.

The product then follows the champion by itself: read the newest root
reveal from the public store, take `champion`, fetch the head by hash, prove
parity against the reveal's routing decisions, serve. Parity binds the head
to the reveal. As built, the product does not check the reveal against the
root's on-chain reveal commitment (it has no chain client): it relies on the
bucket for which reveal is the root's. That check is listed in
`docs/INVARIANTS.md`, "Known gaps".

Published bytes let a head be continued from another's. v3 already accepts
that for published odds: a faithful copy has no edge and cannot take the seat,
ownership is by exact hash, and a head that changes rows gains only where its
odds are really better.

**This part has no consensus effect and can ship before v4** in a validator
release, with a notice to miners.

## 12. Proof and reveal changes

**Proof** (all inside `content_hash`, so bound through `report_data`; each
new field hashed only when present, the pattern `failed` and `latency_ms`
set):

| Field | Where | Purpose |
|---|---|---|
| `menu_hash` | proof | names the menu the run used; must equal the epoch's |
| `routing_protocol` | proof | `identity_v4(E)` |
| `provider_cost` | each result | billed cost, integer nanodollars |
| `served_by` | each result | the provider that answered, as the reply names it: its display name, not the host tag. Bounded string |
| `endpoint_reads` | proof | per model, per pinned host: prices, limit, quantization, status. Bounded |
| `proposal` | proof, optional, at most one | section 5 |
| `measure_for` | each measurement result | the proposal id; such a result is exploration and is never scored |

`PROOF_JSON_MAX_LEN` is checked against what v4 adds: the 16 calls of the
coverage part (section 8; 48 exploration results in all) and the
measurement results, 60 an epoch for each of a hotkey's proposals in
measurement, of which there can be as many as the menu holds
(`MENU_MAX_PENDING`, 8, though each coldkey has at most
`MENU_MAX_PROPOSALS_PER_COLDKEY`, 4, and a hotkey has one coldkey), so 240
at most; the size check keeps the bound of 480 as a margin.

As built (W3, `tee/proof.py` and `tee/endpoint_reads.py`), each field absent
on an earlier protocol's proof and bounded on a v4 one:

- `menu_hash`: 64 lowercase hex.
- `endpoint_reads`: a list sorted by `model`, at most 80, each
  `{model, hosts}` with 1 to 32 `hosts` sorted by `host`, each
  `{host, listed, price_in, price_out, limit, quantization, status}`: prices
  integer picodollars per token up to 10^9, `limit` an integer up to 10^9
  (0 when the listing states none), `status` an integer within 1,000 of
  zero, `quantization` as listed. A host the listing does not show is
  `listed: false` with every other value zero or empty. A model whose read
  failed has no record.
- `proposal`: `{id, model, reasoning, provider: {order, allow_fallbacks:
  false}, quantization}`, at most 4 hosts; the proposer is the proof's
  hotkey, the filing epoch the proof's, and its read is its own
  `endpoint_reads`.
- per result: `provider_cost`, integer nanodollars up to 10^12, absent when
  the provider reported none; `served_by`, printable ASCII up to 64
  characters ("" when the reply names no provider); `measure_for`, the
  proposal id, only on a result with `is_exploration` true.

The largest proof these bounds allow (300 scored, 48 exploration and 480
measurement results, 80 reads of 32 hosts, every string at its longest) is
about 1.24 MB against the 4 MB cap (0.92 MB with one proposal in
measurement; `tests/test_proof_v4_fields.py`).

Notes from the build (W3):

- **`served_by` is the provider's display name from the reply**, not the
  host tag the request pinned: the reply does not carry the tag. Host-level
  drift (a pinned host that bills or answers differently from its listing)
  is therefore attributed from the endpoint reads, which are per host tag,
  and not from replies. `served_by` shows only which provider answered.
- **The measurement runs as its own phase**, after the scored and
  exploration calls of the epoch: a proposal's calls never delay or crowd
  out the calls a miner is scored on, and one that misses its deadline is a
  failed call (section 5).
- The exploration a TD runs is `td_v4.exploration_v4`: section 8's draw,
  on the shares of the epoch's menu alone (its costs and the counts it
  publishes). A TD reads a menu that adopts a newer device list only with
  that list, which the session takes from any source and verifies
  (`V4Session(device_list_bytes=)`), as it does the lab and host list.

Verification and the step's inputs, as built (W4a, `tee/verify.py` and
`fugal_subnet/menu_inputs.py`):

- `verify.expected_v4(MenuState(E), epoch=, nonce=, pool=, scored_ids=,
  hotkey=, index=, targets=)` is one hotkey's `V4Expected`: the menu hash,
  the exploration by part (shared, cost-aware, coverage: the TD's
  `td_v4.exploration_v4`) and, per proposal of that hotkey the menu
  schedules in E, its 60 drawn questions (drawn after the whole slice and
  the exploration, as the harness draws them). `index` and `targets` must be
  the ones the miners' `td_v4.V4Plan` carries (`epoch_plan.pool_index_v4`
  and the scored mix). `verify_proof(protocol_version=4, v4=...)` requires
  it and refuses: another `menu_hash`, a namespace not `identity_v4` of it,
  no `endpoint_reads`, a scored route off the menu, exploration other than
  the assigned set, measurement for a proposal not scheduled for this
  miner in E, measurement that is not exactly the draw (none at all is a
  valid proof, and the step lapses the proposal), and a head with no row on
  the menu. `gold_answers` must cover the measurement questions.
- The step's inputs are taken per coldkey, the chain's owner of each
  proof's hotkey (`menu_inputs.VerifiedProof`). The endpoint vote: a
  coldkey's hotkeys are one vote (listed when at least half read the host
  listed, then their median); a host is listed when at least half the
  coldkeys vote it listed, and its `voters` are those coldkeys, its values
  their median rounded half up. A read not in the field's one shape is no
  vote.
- Each epoch leaves an `EpochSummary` (the vote, each coldkey's exploration
  calls and failures per entry revision, pooled and measurement pairs by
  revision and category, the counted heads' picks), with one JSON form.
  From the window of summaries: token sums over exploration and
  measurement pairs of the current revision; exploration-pair counts over
  exploration pairs only; accuracy and the proposals' upper bounds by
  Wilson's interval at z = 1.645 on the mix-weighted rate with the
  stratified effective size (exact, an integer square root; lower bounds
  rounded down, upper up, in ppm); use summed over the window; health as
  9.2 says. Epoch k's reports and filings come from its proofs, the
  filings through `td_v4.filing_from_proof` with the chain's coldkey.
- Pooling under v4 weighs a pair on a private question (`V4Expected.
  private_ids`) by the private mixture and any other by the shared set
  (`explore_pool.pool_pairs(private_weights=)`), with
  `explore_pool.coldkey_cap(4)` = 60.

**The epoch summary is public.** The root's reveal of epoch k carries
`menu_summary`: epoch k's summary of its verified proofs
(`menu_inputs.summary_to_dict`: the endpoint vote, each coldkey's
exploration calls and failures per entry revision, the pooled and
measurement pairs per entry revision and category, the counted heads'
picks). It is a root-only field, outside the view validators are compared
on (`v2_sync.PER_VALIDATOR_FIELDS`), and no v1-v3 reveal field changes. A
validator whose window is missing epochs (a restart with an empty state, a
new validator) fills them from the root's reveals, each read only when its
bytes hash to the root's reveal commitment for that epoch (the pointer
ledger follower sync keeps, else the commitment read at the boundary block
of k + `MENU_LAG`, where it is the root's for k), and never in place of a
summary of its own (`MenuBook.fill_from_root`, `menu_publish.root_summaries`).
Its `menu_audit` then names those epochs (`window_from_root`): on them it
checks the root against the root's own summary, which is weaker than an
audit from its own proofs, and says so. Size: on the seed menu (59
entries) with 11 coldkeys and 15 hotkeys, a summary is 50-94 kB as compact
JSON (mean 86 kB; 497 call records, 291 pair cells, 47 votes in the
largest), against 1.39 MB for a published v3 reveal (mean of the 24 W0
replayed, largest 1.65 MB): about 6% more. It grows with coldkeys x
entries called and entries x categories, not with hotkeys.

**Reveal**: `menu` (hash and state used), `menu_next`, `proposals` (every
decision with its reason), `endpoint_votes`, `billed_cost` per entry,
`menu_audit` (non-root validators), the signed allowlists when they changed.
`menu_hash` joins `v2_state`, the view validators are compared on. A rebuild
reads the menu from reveals, as it reads odds.

As built (W4b, `menu_publish`), on a v4 reveal and on no other:

- `menu`: `{epoch, hash, source: {epoch, hash}}`: the menu the epoch ran
  on, and `source`, the last published menu it was carried from
  (`menu_publish.menu_field`, `MenuBook.source_of`): the first epoch of the
  run of epochs whose menus have the same content but for `epoch` (a
  changed menu takes the next serial), with its hash. A menu committed at
  its own epoch is its own source. The source is either a committed menu,
  whose bytes the root put in its store and read back before committing
  them, or the pinned seed at the first v4 epoch, whose bytes the root puts
  with the reveal that names it (and which any reader also builds from the
  release, `menu_state.seed_state`). So a reader with no history fetches
  `<netuid>/menus/<source hash>.json`, checks its hash, checks it is the
  menu of the source epoch, sets `epoch` to the reveal's and checks the
  rebuilt canonical bytes against `hash` (`menu_publish.menu_from_field`).
  Every validator computes the field from the menus it held, so the root's
  and a follower's agree byte for byte; it joins the compared view
  (`v2_sync.consensus_view`) beside `v2_state`, where a root naming another
  source differs from every follower, and a TD anchors only on a reveal
  whose field has this shape (`menu_publish.anchor_from_reveal`), rebuilding
  from the source when it has no bytes for the hash.
- the root only: `menu_next`, MenuState(k + 2) as its canonical document
  (an ASCII string, exactly the bytes hashed and committed);
  `menu_next_hash`; `menu_events`, the step's events in its order as
  `[kind, subject, detail]` (the proposals' decisions with their reasons
  are among them); `menu_lists`, every signed list the new menu names that
  the menu it was stepped from did not, with its bytes and signature.
- every other validator: `menu_audit`, `{scored_epoch, menu_epoch, root,
  own, status, differences}` with `differences` the paths at which its own
  step differs from the root's committed menu ([] when they agree, null with
  a `why` when it could not run the step). For an epoch with no commitment
  (the root published no step) the menu carries by rule: `differences` are
  then those of the menu the epoch runs on from the carry of the menu
  before ([] for a correct carry), with a `why` saying the root committed
  no menu, and its own step is not compared.

`endpoint_votes` and `billed_cost` are the aggregation's (W4a). The rebuild
of the menu walks the chain rather than the reveals (4.3, and
`docs/INVARIANTS.md`, the publication entry): a reveal says which menu the
root derived, and only the chain at each boundary says which one ran.

## 13. A second attestation path: Phala Cloud (separable work package)

**Why.** Every miner today runs on one provider (GCP TDX, through dstack).
That is a quota wall and a single supplier. Phala Cloud runs dstack on Intel
TDX and gives a tenant a raw TDX quote and the RTMR3 event log, with the
compose hash recorded the same way. It does not have GCP's vTPM. Targon is
**not** buildable today (its CPU tiers are AMD, a tenant cannot boot its own
image and no tenant-facing quote is documented); it is noted only as a later
SEV-SNP port.

**This package touches the verifier, the approved entries and the miner's
compose, and nothing in sections 4-12.** It can slip to a later release
without blocking them. Every validator must upgrade before a miner uses it,
which is why it rides a release that already requires that.

### 13.1 The trust statement

A proof makes six claims (`tee/verify.py`). Five do not depend on the
provider: real TDX hardware (Intel DCAP), the TD is not debuggable, the
published OS booted, the published application ran, and the proof body is
what that TD produced for this miner and epoch. The sixth, **whose machine it
ran on**, is what GCP's TPM half answers: Intel attests the TD, Google
attests that the machine underneath is one of its Confidential VMs.

For a Phala base the sixth claim is answered differently:

> Intel's signature says a genuine TDX platform produced the quote and names
> that platform (the PPID in the quote's PCK certificate). The platform's
> device id, sha256 of the PPID, is on a list of the provider's machines that
> is signed by the release key and pinned.

So the roots are Intel, and the release key vouching for a published list of
the provider's machines, in place of Intel and Google's CA. What is trusted
that was not before: the provider's operators and the physical custody of
the listed machines, as the GCP path trusts Google's. There is **no "any
hardware" path**: a TD on a machine nobody vouches for is refused, because
the subnet is as strong as the weakest path it accepts.

One difference is stated because the build must design for it: on this path
the persistent volume is not sealed against the operator. Everything a TD
reads back from disk is untrusted input and must be checked against a pinned
hash before use, on both paths.

### 13.2 The rule follows from the approved base

**The rule a proof must meet is decided by its approved base, never by what
the miner sends.** Otherwise the miner picks the weaker rule.

- An approved entry gains a platform tag: `<platform>/<base>:<compose_hash>`,
  platform one of `gcp`, `phala`. An entry with no tag reads as `gcp`, so
  today's entries are unchanged. A base may appear under one platform only;
  a validator refuses to start otherwise.
- `verify_proof` decodes the attestation envelope by its variant tag
  (`scale.py` gains dstack's TDX-only variant beside the GCP one), computes
  the base measurement, and looks up its platform.
  - `gcp`: the TPM half must be present and valid, as today.
  - `phala`: the envelope must be the TDX-only variant, and the quote's
    device id must be on the device list.
  - An envelope whose variant does not match its base's platform is refused,
    in both directions.

### 13.3 Bases, and how they are computed

`base = sha256(MRTD || RTMR1 || RTMR2)` as today. On Phala the tenant chooses
among the provider's published OS images, built from source, whose
measurements are reproducible with dstack's own tool (`dstack-mr`). One base
is pinned per production OS image version, per host firmware behaviour that
moves MRTD, and per memory class where the image's RTMR1 depends on guest
memory. Each is computed offline from the published image and confirmed on
one real CVM before it is approved. **Development OS images are never
approved**: they allow a shell into the TD.

The application half is a separate frozen compose (same image digest, same
docker compose; the key provider setting differs from GCP's, so its bytes and
its hash differ).

### 13.4 The device list, updated without a release

`{serial, devices[]}`: device ids of the provider's machines, taken from the
provider's public node list, signed by the release key under the allowlist
namespace of section 6. The validator ships the list current at release,
pinned by hash. A newer signed list travels in the root's reveal and applies
at the menu lag, so every validator accepts or refuses a given machine from
the same epoch. **It is never fetched per proof**: two validators fetching at
different moments would disagree, which is the reason the Google chain is
pinned too.

A machine's PPID changes with some BIOS and microcode updates, and a miner on
it fails verification until a new list is signed. That is the operational
cost of this path, and who keeps the list current is an open owner question.

### 13.5 Attack cases the suite needs

`run_tee_attacks.py`, each refused:

1. a GCP base presented with no TPM half (TDX-only envelope);
2. a Phala base with a valid quote from a device not on the list;
3. a Phala base with a TPM half attached, and a GCP base in a TDX-only
   envelope whose quote is otherwise valid (variant and platform must agree);
4. a development OS image's measurement;
5. an event log from the other platform, or one that does not replay to the
   quote;
6. a quote whose PCK certificate has no PPID extension, or a malformed one;
7. a device list with a bad signature, or a lower serial than the one held;
8. a listed device with the debug attribute set;
9. the same base listed under two platforms (startup refusal).

### 13.6 The invariant change

`docs/INVARIANTS.md` I8 gains an entry: *a proof's attestation rule is chosen
by its approved base; each accepted platform names its second trust root; no
platform is accepted without one.* The existing statement that a validator
must check both halves becomes "must check every half its base's platform
requires". `check_safety_invariants` pins the device list's hash and the
allowlist signing key, and checks that no development image base is approved.
The determinism check is unchanged.

### 13.7 Open measurements (before any of it is approved)

| Question | How it is answered |
|---|---|
| Is the provider's final app-compose byte-identical for every tenant who submits the same file, so all miners share one compose hash? | The provider's provision call, which returns the final compose and hash without creating a machine |
| Can validators reach the axon at a public host and port, and is the port stable across restarts? | One CVM with a port mapping |
| Does the miner fit the smaller instance (2 vCPU / 4 GB)? If it needs 4 vCPU / 8 GB the price is level with GCP | Memory high-water on one epoch |
| Is MRTD the same across the provider's nodes and regions? | Compare computed bases with quotes from two nodes |
| Which key-provider settings does the platform accept for a tenant? | The same provision call |

The first two cost nothing or under a dollar and need the owner's OK and an
account.

## 14. Attacks on the live menu, and the test each needs

| | Attack | Defence | Test |
|---|---|---|---|
| A1 | A model fitted to the public pool is listed and proposed. It looks perfect and cheap and is useless to users. Scores cannot detect it | The lab allowlist, the host allowlist. Later: accuracy on questions newer than the model against the old pool | Rule tests for both lists. **Residual, documented in Known gaps** |
| A2 | Spam proposals to churn the menu and force retraining, or fill the pipeline so no one else can propose | Proposer pays; `MENU_MAX_PENDING`; at most 4 proposals in flight per coldkey; 2 joins an epoch; cooldowns on entry and coldkey; grace; every node holds a committed menu's new filings to the duplicate, cooldown and coldkey rules | Derivation tests at each limit; the transition check refuses a fifth proposal of one coldkey (`tests/test_menu_coldkey_cap.py`) and a filing in cooldown or duplicated (`tests/test_menu_root_limits.py`); a full queue frees room within `QUEUE_WAIT_EPOCHS` (`tests/test_menu_step.py`); `run_miner_attacks`: "root files a fifth proposal of one coldkey", "root drops a running cooldown" |
| A3 | Choosing the measurement questions, or filing only a lucky sample | Commit first; questions from later nonces; a proposal that stops is dropped and every measured pair is published; a report counts only in the step for its own epoch | A measurement on another seed is refused; an epoch the proposer skipped lapses the proposal, and an epoch the root published no step for does not |
| A4 | Moving a price | Prices from TD reads of the listing, coldkey median, minimum voters; billed-cost drift published | One operator's reads cannot move the median; fewer than the minimum changes nothing |
| A5 | Retiring a rival's favourite entry by failing calls to it, or the whole menu through a listing change | A majority of at least 4 eligible coldkeys, failed calls in 3 distinct epochs, the outage guard; the listing-outage guard on "gone"; at most max(2, a tenth of the menu) health retirements a step, never below `MENU_MIN_ENTRIES` (9.2) | One operator failing every call retires nothing, with or without a broken host beside it; one to three epochs in which every call fails retire nothing; a listing that shows every host unlisted retires nothing (`tests/test_menu_liveness.py`) |
| A6 | Keeping a dead entry alive, or pushing one out, through use | Use counts only seat holders' and senior bars' scored picks; health retirement ignores use | Displacement tests |
| A7 | A stale or different menu in a TD | Namespace carries `menu_hash`; the epoch's hash comes from the chain | Proof on another menu refused |
| A8 | The root publishes a menu the rules forbid, or withholds one | TDs and validators apply the compiled rules and the transition checks to every committed menu and refuse one that fails, carrying the previous one (a TD under the list the menu it holds names); any bytes that are not a menu are refused, never "unavailable"; the carry rule on an absent commitment; `menu_audit` | A forbidden entry is refused inside the TD; a forbidden menu and an absent commitment each carry the previous menu on every node; a refused menu naming another list is carried by the TD in that epoch and every later one (`tests/test_td_v4_session.py`); a one-value mutation fuzz of committed bytes gives the menu or a refusal and nothing else; invented, re-dated or lengthened cooldowns, records over the caps, a list reason the list does not give, a join on other values, a displacement from a menu not full or in grace, a queue or a measurement kept past its bound, each refused (`tests/test_menu_root_limits.py`); `run_miner_attacks`: the root's menu cases (grace moved, repriced outside a refresh epoch, bytes over the limit unread) |
| A9 | Scoring on proposal pairs the proposer prepared for | Measurement pairs are never scored; grace | Unit test on `EpochPairs.scored` |
| A10 | OpenRouter outage or a model removed mid-epoch | Failed calls as today; a failed endpoint read is no vote; retirement follows | Liveness test: no call on the epoch path is unbounded |
| A11 | Allowlist key compromise, or replay of an old list | Same key and custody as image tags; monotonic serial; rotation written down | Lower serial and bad signature refused |
| A12 | A request update that swaps in different hardware under a trusted row | `rev` bump; pairs of the old `rev` leave R and the token means; full measurement again | Derivation test |
| A13 | The root publishes a menu every node accepts and no node can draw an epoch on (empty, an entry priced at 0, everything retired at once) | The reader refuses fewer than `MENU_MIN_ENTRIES` entries and an entry that costs nothing; the step never prices at 0, never retires below the minimum, at most max(2, a tenth) for health a step | `tests/test_menu_liveness.py`, including the property that every menu the reader's per-menu checks pass gives an exploration draw and picks |
| A14 | The root keeps a proposer's box measuring, or writes a proposal under a miner's hotkey for its box to pay for | The reader holds a proposal's window to the step's, within its make-up bound, and requires it decided when its measurement ends; a TD measures only what its own session filed; one ceiling for the box across its proposals, counting the larger of the menu's price and the billed cost (or, unbilled, the TD's own listing read), before each batch, kept across a restart | `tests/test_menu_reader_checks.py`, `tests/test_menu_root_limits.py`, `tests/test_harness_v4.py`; `run_miner_attacks`: "measurement past the spend ceiling", "measurement priced at nothing, billed" |
| A15 | A miner restarted long after activation, or on a node that keeps only recent blocks, can never prove again; a node that rebuilds across a change of the root's key ends on another menu | A TD anchors on the root's newest reveal or its kept menu, keeps its pinned reads and its menus, and a failed read costs one epoch; the root's hotkeys are a compiled table read per epoch | `tests/test_td_v4_session.py` (2,000 epochs in; a 300-block node after a skipped epoch), `tests/test_menu_publish.py` (a rebuild across a key change) |

## 15. Invariants, checks, activation

**`docs/INVARIANTS.md`** (written with the build, not before):

- New, I1: *the menu of an epoch is a pure function of published inputs and
  is read through one function.* One reader, `menu_state.menu_for_epoch`; a
  second way to obtain the menu is a consensus break of the same kind as a
  second epoch-id format. The reader has three outcomes and no fourth: the
  menu (carried, or the committed one after it passed the checks of 4.3);
  `MenuRefused`, which names the carried menu as the one every node runs the
  epoch on and is logged as an alarm; and `MenuUnavailable`, with no menu,
  for a node that lacks the bytes of a changed hash or the list it names. A
  miner's TD with no menu produces no valid proof for that epoch.
- I1: the compiled rules are applied by the step and by the reader, never
  handed to them as verdicts; every menu the step derives passes the reader.
- I1 (determinism): `MenuState` is integers and strings; the step is exact.
- I1: prices, answer limits, token means and exploration-pair counts move
  only in a refresh epoch (`MENU_REFRESH_EPOCHS`), except on the entry that
  joins or is updated in that step, a price by at most a factor of
  `MENU_PRICE_STEP` (4); every node checks that from the two menus (4.4).
- I1: every menu the reader accepts can run an epoch: at least
  `MENU_MIN_ENTRIES` full entries, each with a positive expected cost; the step
  keeps both. Any committed bytes that are not such a menu are
  `MenuRefused`, never another outcome.
- I3: per-entry grace; the full-menu rule over entries past `scored_from`.
- I4: prices, failure retirement and token means are votes of operators; a
  proposer cannot choose its measurement, and is not lapsed by an epoch the
  root published no step for. A cooldown still running is never dropped
  from the state.
- I5: a miner's spend is bounded by the exploration caps (the cost-aware
  part's and the coverage part's, both on expected spend) and by its own
  proposal ceiling, one total for the box, charged at the larger of the
  menu's price and the billed cost (or, unbilled, its own listing read),
  kept across restarts, spent only on proposals its own session filed, and
  never past a proposal's make-up bound.
- I6: no epoch waits on the reveal store, the listing or the root, **with
  one exception**: an epoch whose committed menu changed needs that menu's
  bytes (and any newer list it names) before any proof is valid. The root's
  publisher commits a menu hash only after the bytes are confirmed readable
  from the public store (publish before commit, 4.3), so an honest root's
  failed upload carries the previous menu instead of stopping proofs; the
  refresh cadence keeps such epochs to about one a day plus structural
  changes. W4 implements the publisher rule and its test. A TD that
  starts or falls behind anchors on the root's newest reveal or its kept
  menu and walks a few links; a pinned read that fails costs that epoch
  only.
- I8: the image carries rules and the pinned seed, and no other menu; a v4
  entry passes the compiled rules inside the TD; the signed lists. Section
  13's entry if that package ships.
- I9: R and the token means over a changing menu and `rev`.
- Known gaps: A1's residual; the root as the menu's author (4.3); **the
  root's availability**: a root that commits a hash and withholds the bytes
  stops every proof until the bytes appear, a dependence v3 miners do not
  have. Mirrors are the candidate mitigation; a chain-anchored fallback was
  considered and not designed, because "unavailable" is not observed
  identically by every node (4.3).

**`scripts/check_determinism.py`** stages, identical under `--perturb`:
`menu_derivation`, `endpoint_vote`, `token_means`, `proposal_measurement`,
`explore_shares_v4`, `seat_v4` (was `grace_scoring`), `retirement`, `head_odds_v4`.

**`scripts/check_safety_invariants.py`**: the seed is pinned and every entry
in it passes the compiled rules; no menu other than the pinned seed ships in
the image; the allowlist key and the shipped lists are pinned and
signature-checked; `MENU_LAG` and every new constant are classified and in
the digest; v3's pins stay; the frozen compose carries what the miner reads.

**Activation.** `FINNEY_V4_ACTIVATION_BLOCK`, compiled, epoch-aligned;
`slicer.protocol_version_for_boundary` returns 4 from it. As built ahead of
activation, `FINNEY_V4_ACTIVATION_BLOCK` is None and another chain (tests, a
local rehearsal) reaches v4 through `FUGAL_V4_ACTIVATION_BLOCK`, read as
v3's variable was before v3's block was compiled and refused on finney; the
loops run v4 end to end in `tests/test_v4_loop.py` (`docs/INVARIANTS.md`). The first v4 epoch
starts from an empty window and an empty seat, as v3's did, with the seed as
its menu. **Before the block every output is byte-identical to v0.6.0**:
shown by the 40-epoch equivalence script used for v0.6.0 (kept outside the
repo so far; W11 brings it in under `scripts/`), extended with a v3 scenario
and run on the old and new trees. Miners need the v4 image and its approved
entry before the block; validators the release, **every one of them before
the root writes its first four-field commitment** (4.3: older followers
cannot parse it). The frozen compose is checked
to set every variable the miner reads (the 2026-11-17 lesson).

## 16. Where this departs from the plan, and why

| | The plan | This design | Why |
|---|---|---|---|
| 1 | The TD measures inside the proposal | File first; measurement assigned over the next epochs | A proposer who measures first can file only the samples it likes, and the seed of R for that entry would carry the bias for a whole window. It also does not fit one epoch's call budget for slow entries |
| 2 | Every validator computes the menu | The root's committed state is the menu; the others audit it; an absent commitment carries the previous menu | All TDs need one menu before the epoch starts. Chain state at a block is the only input every node shares in time |
| 3 | Price changes through update proposals | Every TD reads the listing every epoch; coldkey median | Four of six models checked had drifted from their pin within days. A proposal per price change is the per-model work the owner wants gone |
| 4 | Token means: medians from exploration at n >= 100 | Mix-weighted means, n >= 30, and no token means inside the head | Expected spend is a mean. A head that bakes in means cannot follow a measured table |
| 5 | Exploration unchanged | A coverage part of 16 calls, damped and spend-capped; the scored slice shrinks only with a guard that needs fewer questions | Section 2: at one pair an epoch a dear entry is never measured, and a 300-pair proposal would be most of what is known about it |
| 6 | At the cap, displace the lowest-value entry; retire entries off the frontier | Displace the least-used entry, only when a slot is needed; otherwise wait | A frontier over noisy single-model estimates retires on noise and ignores entries that are best for one kind of question |
| 7 | Gate: answered and above a floor | Adds "plausibly useful" | Each entry thins exploration for all the others |
| 8 | Proposal pairs must be kept from the proposer's score | Never scored for anyone | Simpler, and consistent with grace |
| 9 | Allowlist as signed data | Signed data that travels in the root's reveal and applies at the lag | A list each node fetched by itself would be read at different times |
| 10 | (not in the plan) | `MENU_MAX_PENDING`; `rev` and request updates; billed cost and serving host in the proof | Bounds on retraining load; hosts change; drift must be visible |
| 11 | Hand-set category targets | Targets derived from the measured outcome variance `p(1-p)`, pinned | So the mix follows evidence and can be re-derived at the next release; the best-to-cheapest gap is too noisy to derive it from (section 10) |
| 12 | (not in the plan) | Head publication, shippable before v4 | Already decided; the product needs it |
| 13 | (not in the plan) | Section 13 | Owner decision 2026-11-18 |

## 17. Open items

**For the owner:**

1. **The price setting of the exam** (section 10): keep picks at lambda 1 and
   the guard at $0.49, or take the exam at the product's default price
   setting. Recommended: the product's default.
2. **The scored slice and the net-value guard** (section 8), an open
   design item of the seat-rule package: price the guard at the routed
   entry's mean cost and take the slice to 200 questions (with the
   1,500-question counts at 1,000), or keep the guard and the slice at 300.
   The coverage part is adopted either way; at 300 it adds about $0.16 to a
   hotkey's $5.7 an epoch. Recommended (W0): the mean-cost guard at 200.
3. **Head publication** (11.3): all verified heads at the reveal's delay, in a
   validator release before v4. Recommended: yes.
4. **The first lab and host lists**, and who keeps the Phala device list
   current (13.4). Until the first list is signed the seed names none; the
   seed is rebuilt and re-pinned to be checked under the signed list before
   an activation block is set.
5. **New question sets** for the harder mix, each needing the approval v3's
   sets had.
6. Phala: an account and under a dollar for the measurements of 13.7.

**For the seat-rule package:** whether a head's picks may route to an
entry in grace (9.1: today they may, and no head is scored on its odds for
it yet); the scored slice's size with the net-value guard (item 2).

**For the activation package:**

7. **The root's availability** (4.3): a root that commits a menu hash and
   withholds the bytes stops every proof. Candidate mitigation: mirrors,
   since any node holding the bytes can serve them and every reader checks
   them against the on-chain hash (where they are listed, who runs them, how
   a TD's operator finds one). A chain-anchored fallback was considered and
   not designed: "unavailable" is not something every node observes
   identically.
8. The v4 constants (`config.py`'s `_V4` names, `menu_state`'s constants
   including `MENU_REFRESH_EPOCHS`, `QUEUE_WAIT_EPOCHS`,
   `PROPOSAL_MAKEUP_EPOCHS`, and `config.FINNEY_ROOT_HOTKEYS_V4`) go into the
   consensus fingerprint.
9. **Merge `v4/wip` with `main` before release.** The branch was cut before
   two changes merged to `main`: the TPM attestation key accepted across
   regions (PR 202) and head publication (11.3). A release built from the
   branch as it stands would drop both; the merge keeps them.

**What the published data cannot tell** (W0 replayed every published v3
reveal it could; these questions need data the reveals do not hold):

- **Whether a question separates routers.** Only picked cells are graded, so
  the scored cells depend on the heads' choices; exploration gives one entry
  per question, so no question is seen on two entries nobody chose. The
  scored disagreement is a lower bound among sensible picks and cannot be
  split into routable difference and chance. (The one direct handle: the
  same shared pair answered by three or more operators was not unanimous in
  18% of 183 pairs, 13-24%.)
- **The gap between the best and the cheap entries, by set.** The ten best
  entries have 38-208 pairs a category; resolving a 3-point difference needs
  about 1,000 a category, roughly two windows of section 8's setting.
- **Question sets not in the pool**, and how heads re-route, spend and
  disagree under another mix or another exam price. The guard figures at
  section 10's mix use today's picks.
- **The "gone" rule** (a reveal carries no endpoint reads), **the
  billed-cost drift band** (`usage.cost` is not published), and **scored
  calls in the retirement rule** (a reveal does not say which coldkey a uid
  belongs to; there are 128,700 scored calls against 13,728 exploration
  calls). v4's reveal publishes per-coldkey failure counts so that these
  can be measured.
- **The retirement rule's false-retirement rate** below a few percent per
  entry per day (no false retirement among 59 entries over 29 hours cannot
  exclude about 5% per entry per day), its behaviour in a real outage (the
  data has none), and its detection time on more than one real break.
- **How many operators v4 will have.** Every size in section 8 is given at
  3, 5, 7 and 10; the setting assumes at least 5.
- `PROPOSAL_MIN_CATEGORY_ACCURACY` and the plausibly-useful gate (section 5)
  were not examined.

**Suggested by W0 and not adopted here:** `TOKEN_MIN_PAIRS` 100 instead of 30
(the cost of a call has a coefficient of variation of 1.9 for the median
entry, so a 30-pair mean is off by 35% at one SE, and even at 100 a third of
the table is not within 25%; publish the interval with the mean), and
`SINGLE_MODEL_MIN_PAIRS` 100 instead of 300 (the table has been empty for 24
epochs). The seed builder and `menu_state.TOKEN_MIN_PAIRS` use 30.

**Not verified** (no paid call was made): `usage.cost` on the six menu models
absent from the 2026-11-17 answers; whether every window consumer aligns by
id (whether the Commitments field has room for a second hash was answered
by W4b against the SDK: not in hex, 146 bytes against 128, so the menu hash
is written in 43 characters of base64url, 125 bytes in all, 4.3; not yet
tried on a chain); whether the
answer limit the 2026-11-18 listing states for `google/gemini-4-carbon` on
`google-vertex/global` (943,718) is one the host honours. Section 6's rule
takes the listing at its word, which moves that entry's answer limit from
v3's 65,536 to the 131,072 cap in the seed.

## 18. Build breakdown

Lanes can run in parallel where the table shows no dependency. Every package
that touches consensus code adds its invariant text and its determinism stage
in the same change. "Checks" is in addition to `ruff`, the safety-invariant
check and the package's own tests.

| | Package | Depends on | Main files | Checks it must pass |
|---|---|---|---|---|
| W0 | **Replay and parameters** (offline, no repo change): replay published reveals to fix section 8's sizes, 9.2's thresholds and section 10's targets. **Done 2026-11-18**; its numbers are in sections 2, 8, 9.2, 10 and 17 | none | a script outside the repo; a short decision record | reproducible from named reveals |
| W1 | **Menu state**: dataclasses, canonical JSON, `menu_hash`, the step, the seed builder and its pin, `identity_v4` | none | new `menu_state.py`; `routing_protocol.py`; `routing_data/menu_v4_seed.json`; `check_safety_invariants.py` | `menu_derivation` stage; seed rebuilds byte for byte |
| W2 | **Rules and signed lists**: compiled rules, list format, signature verification, signing namespace | none | new `menu_rules.py`; `deploy/keys/`; `validate_menu_v3` reused | rule tests; A1, A11 |
| W3 | **TD side**: endpoint reads, billed cost and host capture, proof fields, menu from state, measurement calls, proposal filing | W1, W2 | `tee/runtime.py`, `tee/proof.py`, `tee/harness.py`, `neurons/miner.py` | proof hash unchanged for v3 proofs; miner attack suite; no paid call in tests |
| W4 | **Validator side**: namespace and menu checks, proposal and measurement verification, endpoint vote, token means, health retirement (9.2's rule, with the step's health input and constants brought to it), the cap, reveal fields, rebuild and sync | W1, W2; W3's schema | `tee/verify.py`, `v2_epoch.py`, `explore_pool.py`, `single_model.py`, `v2_rebuild.py`, `v2_sync.py`, `commit_reveal.py`, `neurons/validator.py` | `endpoint_vote`, `token_means`, `proposal_measurement`, `retirement` stages; A2-A10, A12; TEE attack suite |
| W5 | **Exploration v4**: shares from state costs, the coverage part with its damping and spend cap, each pair weighted by its own part | W1; W0's numbers | `explore_weights.py`, `epoch_plan.py`, `exploration.py`, `config.py` | `explore_shares_v4` stage; weights sum checks; spend-cap tests |
| W6 | **Per-entry grace** | W1 | `routing_protocol.py`, `odds_score.py`, `champion.py`, `v2_epoch.py` | `grace_scoring` stage; menu B's tests still pass on v3 epochs |
| W7 | **Head format v4** (the study reported 2026-11-19: format unchanged, 11.2): optional cost fields, id-keyed rows up to 72, picks on the menu's costs in exact arithmetic, one function for the TD and the validator. **Built** (branch `v4/w7-head-format`) | the study; W1 for costs | `head_eval.py`, `head_odds.py`, `tee/harness.py`; fugal-core's contract later (11.2), then re-vendored | `head_odds_v4` stage; vendored copy unchanged and byte-equal; head property tests |
| W8 | **Pool v4 and the mix** | W0 (the encoder is unchanged, 11.2, so not W7) | `benchmarks/`, `exclusions`, pool manifest, `config.py` | pool tests; manifest pin |
| W9 | **Head publication** (operational; can ship early) | none | `head_archive.py`, `reveal_publisher.py`, `neurons/validator.py`, guides | upload failure never fails an epoch |
| W10 | **Product** (fugal-core): menu and costs from the root's reveal, head by hash, parity | W1's format; W9 | fugal-core | parity against reveals |
| W11 | **Activation and proof of equivalence**: v4 block, protocol switch, state reset, `v2_equiv` v3 scenario, `dress_rehearsal --scenario v4_menu` (a proposal joins, grace, a retirement, a price move), one real TD check, INVARIANTS, guides, notice | W1-W8 | `config.py`, `slicer.py`, `fingerprint.py`, `scripts/`, `docs/` | equivalence hash equal on both trees; rehearsal green; full suite; release gate |
| WP | **Phala path** (section 13; separable, may slip) | none of the above | `scale.py`, `tee/attestation.py`, `tee/verify.py`, `tee/roots/`, approved-entry table, `deploy/dstack/`, `run_tee_attacks.py` | 13.5's cases; fixtures from dstack's repo; one real CVM; independent review |

Suggested first wave, all independent: W0 (done), W1, W2, W9, WP's offline half.
Second: W3, W5, W6. Third: W4. W7 and W8 start when the study reports. W10
and W11 close. An independent review by execution, not by diff, comes before
the release, as for v3.
