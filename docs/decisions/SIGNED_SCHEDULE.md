# A signed protocol schedule

**Status:** built, switched off. `config.FINNEY_SCHEDULE_FROM_BLOCK` is `None`
in this release, so no node reads the schedule on finney and the root writes
nothing new. The rollout (section 8) switches it on in a later release.

**Code:** `fugal_subnet/protocol_schedule.py` (the document, the fold, the
tracker), `fugal_subnet/join_step.RootFields` (the root's third commitment
field), `scripts/release.py` (the owner's tooling). Invariant: INVARIANTS.md,
"I1 — the protocol schedule is a fold over the root's pointer".

## 1. The problem

A protocol switch block (`FINNEY_V5_ACTIVATION_BLOCK`, v0.8.1's
`FINNEY_V5_COMMIT_FIX_BLOCK`) is compiled into `config.py`, and the finney
compose repeats it (`FUGAL_V5_ACTIVATION_BLOCK`). The compose repetition adds
no coupling of its own: on finney the miner refuses a compose value that
differs from the compiled one, and an unset variable means the compiled one.
The coupling is the compilation. The miner's image carries the block, so
moving a switch date meant a new image, a new freeze (`dstack-cloud prepare`),
new approved entries, a re-signed `image-v*` tag (about three hours), and a
TD redeploy by every miner.

The goal: a switch whose code is already in the image can have its block set
or moved by a signed statement, with no new image or compose, and every
validator and every miner's TD must derive the same blocks for every epoch.

## 2. Requirements, as given

1. Trust is the release key pinned in the image or checkout, never fetched.
2. The schedule is versioned by a serial and accepted only forward.
3. A block may be set only at least N epochs after the schedule is first
   seen, and an activation that has passed can never be moved or removed.
4. Validators and TDs derive the identical schedule for every epoch.
5. Compiled blocks stay valid and take precedence; v0.8.1 nodes keep working
   until every node runs the new code.
6. The TD verifies it offline with what is inside the measured image.

## 3. Options considered

**A. Compiled blocks, schedule only for the TD.** Validators keep compiled
blocks (moved by a validator release, which the auto-updater now makes cheap)
and the TD reads a signed file the operator pushes. Simple, but nothing binds
the two: a TD on an older signed file runs another protocol than the
validators, and "first seen" is a per-node clock, so requirement 3 has no
common meaning. Rejected.

**B. A signed file in the checkout and the image, newest serial wins.** Every
node holds "the newest file it has". Nodes that have not fetched the newest
file disagree with those that have, and nothing on chain says which is in
force. Rejected for the same reason: requirement 4 needs one source every
node reads at the same block.

**C. The signed document names its own announce epoch.** The document says
"announced at epoch A" and nodes check that the root named it at A. That stops
backdating, but not replay: the root's hotkey (a hot key, on a server) can name
an older document later, and every check on the old document still passes.
Refusing it needs to know what was in force before, which is history.

**D. (chosen) The root names the document on chain; every node folds the
root's pointer from a fixed epoch.** The pointer's first appearance is a chain
fact, so "first seen" needs no signed claim; replay is refused because the
serial must exceed the one in force; every rule is a function of chain reads
pinned at boundary blocks and of the bytes, which every node checks the same
way. The cost is that a node must read the pointer at every boundary since
the mechanism started (one single-key read an epoch, kept on disk), and that
it needs the bytes of every document the root ever named.

**E. A dedicated hotkey whose commitment block is the "first seen".** Cleaner
in one respect (the commitment's own block is the time, so no walk is
needed), but every spare hotkey on netuid 70 can be pruned (only the owner
hotkey is immune, and it is the root, whose commitment is rewritten every
epoch). Rejected for that operational reason.

## 4. The document

One canonical JSON object (sorted keys, one space of indent, ASCII, a final
newline: the signed lists' layout), at most 16 KiB:

```json
{
 "activations": {
  "v5_1": 9600120
 },
 "chain": "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03",
 "format": 1,
 "kind": "fugal-protocol-schedule",
 "serial": 1
}
```

- `activations`: switch name (`[a-z][a-z0-9_]{0,39}`) to block, at most 64.
  The whole schedule every time, not a diff. A name absent means "not
  scheduled".
- `chain`: the genesis hash of the chain it is for, so a test-network
  document is never valid on finney.
- `serial`: 1 or more (0 marks a draft and is refused whatever signs it).

It is signed with a release key from `deploy/keys/allowed_signers`, under the
namespace the signed lists use, `fugal-list@fugalai.com`:

    ssh-keygen -Y sign -f ~/.ssh/<release key> -n fugal-list@fugalai.com schedule.json

Reusing that namespace keeps the trust root and its pins
(`menu_rules.LIST_SIGNERS_SHA256`, in every live TD image) unchanged. A list
and a schedule are told apart by `kind`, which is inside the signed bytes, the
way a device list and a menu allowlist already are; each parser refuses any
other kind. The neurons verify the signature in process
(`menu_rules.verify_list_signature`, the same verifier as the lists) against
`fugal_subnet/signed_lists/allowed_signers`, the pinned copy that is inside
the measured image. Nothing is fetched to decide trust (requirements 1, 6).

## 5. The pointer

The root's commitment gains a field: `sched:<serial>:<digest>`, where the
digest is sha256 of the document's bytes followed by its armored signature
(at most 87 bytes). The digest covers the signature on purpose: with a hash
of the document alone, two sources could serve one document with two
signatures, one valid and one not, and two nodes would reach different
verdicts on the same pointer. With the digest, the bytes a pointer names are
fixed, so the verdict is too. The root writes `[reveal pointer, join
pointer, sched pointer]` (or `[reveal, sched]` with no join); the pallet takes
up to three fields. Field 0 is still the reveal pointer and field 1 the join
pointer, so v0.8.1 code (`decode_metadata` reads field 0; the join reader reads
field 1 and treats anything that is not a join as no join) reads the
commitment exactly as before (requirement 5).

`P(e)` is the pointer in the root's commitment read pinned at epoch e's
boundary block, under the root hotkey of epoch e (`hotkey_for_epoch`). A field
that is not exactly a pointer, or no such field, is `None`. No commitment at
all is not `None` but an unreadable block: the root commits every epoch, and
a pruned endpoint can answer the storage default for a block it no longer
keeps (the read then falls back to the archive endpoint, as other pinned
reads do).

The bytes are found by the digest: in the release
(`fugal_subnet/signed_schedule/<digest>.json` and `.sig`, in the checkout and
the image), in the node's own cache, or in the public store
(`<netuid>/lists/fugal-protocol-schedule/<digest>.json` and `.sig`, beside
the signed lists). Every source is checked against the digest, and bytes that
do not match are treated as not served, never as a refusal, so a source
decides availability, never the verdict. The root names a document only
after reading it back from the public store.

## 6. The rule: a fold over the pointer

From `E0`, the first epoch at or after `SCHEDULE_FROM_BLOCK` (compiled for
finney; `None` = the mechanism is off and the schedule is empty), every node
computes, for each epoch e in order:

```
in_force = EMPTY (serial 0, no activations)
for e = E0, E0 + 1, ...:
    p = P(e)
    if p is None or p == in_force.pointer: continue          # absent: carry
    bytes = the document named by p                          # none anywhere: UNRESOLVED (stop)
    d = verify(bytes, signature)                             # refused: carry
    refuse (carry) unless:
        d.kind, d.format are this format's, d.chain is this chain, d.serial == p.serial
        d.serial > in_force.serial                           # forward only, no replay
        for every name whose block differs between in_force and d:
            the old block (if any) and the new block (if any) are both
            at or after the boundary of e + LEAD             # lead; history is frozen
        every block is an epoch boundary
    in_force = d (adopted at e)
```

The result for epoch E is `in_force` after step E. Why each requirement holds:

- **Determinism (4).** Each step reads chain state at a named block and bytes
  checked against a hash and a signature, with compiled constants (`LEAD`,
  `SCHEDULE_FROM_BLOCK`, the signers). Two nodes, whatever they hold or when
  they start, compute the same sequence or one of them stops (UNRESOLVED);
  none computes a different one. A TD and a validator run the same function.
- **First seen is a chain fact (3).** The epoch a document is adopted is the
  first epoch whose boundary shows the pointer, after the in-force serial. The
  owner cannot backdate it and the root cannot either.
- **Forward only (2).** A pointer to an older document (a replay by whoever
  holds the root's hotkey) fails `serial > in_force.serial` and is carried
  over, at every epoch it stays named.
- **History is frozen (3).** A change at e touches only blocks at or after
  `(e + LEAD) x blocks_per_epoch`, on both sides, so whether a switch is
  active at any epoch before e + LEAD is the same under every later schedule.
  In particular a node may use today's schedule to replay any past epoch.
- **Absence carries.** A root that writes its commitment without the field
  (an old version, a bug) changes nothing: the schedule in force stays.
- **Refusals are final.** Every refusal condition only gets stricter as e
  grows (the lead bound rises, the in-force serial only rises), so a refused
  pointer that stays named stays refused.

`LEAD` is `SCHEDULE_LEAD_EPOCHS = 2` (compiled, consensus). Two epochs is the
least that works with the root's timing (it writes its commitment during
epoch e - 1; nodes adopt at e's boundary); any human notice beyond that is
an announcement matter, and `scripts/release.py` warns below one day
(`--min-notice-epochs`, default 20). LEAD is also the staleness bound: a node
whose fold reached epoch S knows the activation status of every epoch before
S + LEAD, so a node that cannot read the chain or fetch bytes for a while keeps
scoring for up to LEAD epochs and stops after (UNRESOLVED is reported with
the last certain epoch, `certain_through`).

## 7. Compiled blocks take precedence (5)

A switch's block is `config.switch_block(name, genesis, schedule)`: the
compiled finney value when there is one, else the schedule's. When both exist
and differ, the node raises `ScheduleConflict` rather than choose, because a
release and a schedule that disagree would split nodes on two versions; the
release check (`check_safety_invariants --release`) refuses a release whose
compiled blocks contradict the newest shipped schedule. The blocks compiled
today (v2 to v5, the v5 commit fix) stay compiled; a schedule may repeat them
and may not move them. The fold itself never reads compiled blocks, so its
result is the same on every version that has the mechanism.

A switch becomes schedulable when a release ships its code with a compiled
block of `None` and registers its name in `config.SCHEDULE_SWITCHES`. Its
consumer reads `switch_block` (or `ScheduleView.active(name, epoch)`) at each
boundary, never once at startup, because a block may move while the node runs
(by at least LEAD epochs of notice). A node that cannot read the schedule for
an epoch past `certain_through` must not score or prove that epoch.

## 8. Rollout

1. **This release (off).** The code, the tooling and the checks ship with
   `FINNEY_SCHEDULE_FROM_BLOCK = None`. Nothing reads or writes the pointer on
   finney; a v0.8.1 node and a node on this release behave identically.
   Off finney, `FUGAL_SCHEDULE_FROM_BLOCK` turns it on for rehearsals.
2. **The switch-on release (the next one that needs a miner image anyway).**
   It compiles `FINNEY_SCHEDULE_FROM_BLOCK` to an epoch boundary after the
   upgrade deadline every validator and miner gets, as for any protocol
   block, and registers its own new switch with a compiled block of `None`.
   This is the last time a date needs an image.
3. **From then on.** To set or move a block: `scripts/release.py schedule
   draft`, sign (owner), `schedule verify`, upload to the store, put it in the
   root's schedule directory; the root names it at its next commitment and
   every node adopts it at the next boundary. No image, no compose, no
   approved entry, no TD redeploy.
4. **Freezing.** A release after a switch has passed may compile its block
   (it must equal the schedule's). It changes nothing; it shortens the fold a
   new node needs only if the release also moves `SCHEDULE_FROM_BLOCK`
   forward, which is a consensus change and is not planned.

## 9. What still needs a new image

- Any switch whose code is not in the image (a new protocol's rules).
- The schedule's own constants: `SCHEDULE_LEAD_EPOCHS`,
  `FINNEY_SCHEDULE_FROM_BLOCK`, the document format.
- The signers (`signed_lists/allowed_signers`, pinned in the image), the root
  hotkey table, any other compiled consensus constant.

## 10. Risks

- **Liveness on the root and the store.** A pointer whose bytes no node can
  fetch stops every node's fold at that epoch (UNRESOLVED); they keep going
  for LEAD epochs on what they know, then stop scoring until the bytes are
  served. The root confirms the bytes in the store before naming them, and
  the bytes also ship in the repository, so this needs the root's hotkey or a
  root bug, not an outage. The fix is to serve the bytes; nothing is lost.
- **The fold grows with time.** One single-key pinned read an epoch since
  `SCHEDULE_FROM_BLOCK`, kept on disk (`ScheduleTracker` state). A fresh TD
  after a month reads about 600 boundaries once, from the archive endpoint.
- **A wrong compiled block in a release** makes `ScheduleConflict` stop that
  release's nodes (the release check exists to catch it first).
- **The signer set must be the same on every node.** A key added to the
  signers reaches nodes with their next release; sign schedules only with a
  key every deployed release lists (the lists' rule, deploy/keys/README.md).
- **The root's hotkey** can withhold (delay) a schedule and can name any
  document the owner signed; it cannot backdate, replay an older serial or
  move a passed block.
