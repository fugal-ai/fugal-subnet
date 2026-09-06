# Open Work — pick up here

Last updated **2026-09-06**, after PR #9 merged. Written as a handoff, so it
says what is *decided* as well as what is *left* — the decisions cost several
sessions to reach and re-deriving them is the expensive part.

For the invariants themselves, read [INVARIANTS.md](INVARIANTS.md). For the
mainnet sequence, [MAINNET_LAUNCH.md](MAINNET_LAUNCH.md). This file is the
narrower question: **what is unfinished in the PCCS/collateral work, and what
should happen next.**

---

## State as of this writing

| | |
|---|---|
| `main` | `e4adf7e`, all of PRs #4–#9 merged |
| Remote branches | `origin/main` only — everything else merged and deleted |
| Full gate on `main` | green: ruff, safety invariants, axon smoke, all three attack suites, determinism both modes, integration, 282 passed / 1 skipped |

One local worktree still sits on a deleted branch: `/home/jtdoherty/fugal-td`
on `td-provisioning`. Merged in PR #7, so it is safe to remove
(`git worktree remove`) once nobody is working in it.

---

## The one-sentence summary

The validator fetches DCAP collateral **once per proof, over the network,
inside the epoch loop**, and that single fact causes the hang, the fork, the
privacy leak and 99.6% of the cost of verifying a proof. Everything below is
either a consequence of it or a step toward removing it.

---

## 1. The timeout — unblocked, small, highest value per risk

**Do this first.** It is the most dangerous open item and it became cheap to
fix only after PR #9.

The problem: `verify_dcap` has no timeout on the validator's real code path.
The `timeout=30` guards the `ThreadPoolExecutor` branch, which only runs when
an event loop is already running; a synchronous validator takes
`run_until_complete`, which has no bound of its own. **Measured**: a stub
sleeping 60 s blocked the call for the full 60 s.

So a stalled PCCS blocks the epoch. No weights, no log line — the code that
records the failure is downstream of the block and never runs. Every validator
hits it at the same moment, because the collection point is a deterministic
block.

**Why it was not fixed earlier, and why that no longer applies.** A bare
timeout used to convert a hang into a *false accusation*: `TimeoutError` fell
into `except Exception: return False`, the proof was recorded invalid, and the
miner was scored as a cheat. PR #9 built the `unverifiable` path, so a timeout
can now raise `CollateralUnavailable` and be reported as "not checked in time"
instead.

**What to do:** bound the fetch on *both* branches of `_run_coro`, raising
`CollateralUnavailable` rather than returning `False`. Pick the budget from the
measured round trip (690 ms warm, 788 ms cold, GCP us-central1 → Phala) with
generous headroom — this is a hang guard, not a latency target.

**Also worth doing with it:** widen I6. It currently reads "no *miner*
behaviour can stop a validator completing an epoch", and the hang arrives
through a third-party dependency instead. The invariant is what was wrong; the
timeout is only the implementation. Widening it to "no external party" plus a
check is what stops the next network call added inside the epoch loop from
recreating this — and there will be one.

---

## 2. Item 3 — remove the per-proof fetch

This is the real fix. **Two viable routes**, and they are not equivalent.

### Route A — local caching PCCS (operations, no consensus change)

Point `FUGAL_PCCS_URL` at a PCCS running alongside each validator. It re-serves
the same Intel-signed bytes, so it adds no trust root, and it collapses 256
fetches into roughly one per distinct platform (FMSPC).

- No code change beyond configuration — `config.TEE_PCCS_URL` already exists.
- Fixes the hang's blast radius, the cost, and the privacy leak.
- Reduces the fork's *rate*; does not remove its cause.
- Leaves the crafted-FMSPC hole open.
- Cost: every validator operator must run and maintain a daemon.

### Route B — collateral-in-proof (the clean end state, consensus change)

The miner fetches its own collateral once and ships it inside the proof bundle.
The validator verifies with **no network call at all**. Safe in principle
because collateral is Intel-signed and checked against a root CA compiled into
`dcap-qvl` — a miner cannot forge it.

Its best argument is structural, not performance: it converts a **globally
correlated** validator-side failure into an **uncorrelated per-miner** one,
which is the same principle as I5 (miners bear their own costs).

**Three parts, and they must land together.** Any two without the third leaves
a hole:

1. **Collateral travels in the bundle.** Miner side — harness, `protocol.py`,
   MINER_GUIDE. Note collateral is *not* covered by `content_hash`/`report_data`,
   so it is unattested miner data riding inside an attested bundle. It is new
   ingestion surface: I2 applies, so size caps before parse plus
   `run_miner_attacks` cases.

2. **A freshness bound.** Collateral contains `root_ca_crl` and `pck_crl` — so
   shipping it with the proof means **the miner supplies the revocation list
   that would revoke them**. A miner with a revoked PCK certificate ships it
   alongside a pre-revocation CRL and every signature validates.

   > Under this design the freshness bound is **the entire security of
   > revocation, single-layered** — not a staleness nuisance with defence in
   > depth behind it. Stated the weaker way, someone later relaxes it for miner
   > convenience and does not know what they removed.

   It must be a **pinned shared constant**, not a per-operator TTL, or it
   reintroduces the divergence it exists to remove. `tcb_info` is exposed as a
   JSON string, so `issueDate`/`nextUpdate` parse in pure Python —
   **verified**, the bound is implementable.

3. **`now_secs` from the epoch block, not `time.time()`.** The freshness check
   is `next_update > now`; with local clocks, collateral near expiry is valid
   for one validator and expired for another, and the fork simply moves.

**The sanitiser this needs, already worked out and verified:**

Never hand miner JSON to `from_json`. The Rust struct has **ten** fields while
the Python wrapper exposes nine — the hidden one is `pck_certificate_chain`,
and it survives a `from_json` → `to_json` round trip, so miner JSON can inject
a certificate chain. Parse the JSON, take the nine known fields, and rebuild
through the constructor, which produces collateral with no chain attached.

- Four fields are `bytes` and encode as **lowercase hex** (`root_ca_crl`,
  `pck_crl`, `tcb_info_signature`, `qe_identity_signature`) — use
  `bytes.fromhex`. The other five are `str`. Passing hex strings straight
  through raises `TypeError: Can't extract 'str' to 'Vec'`.
- This cannot break an honest miner: a real dstack quote carries its own chain,
  and it is a **complete** one. Measured on `attestation_A.bin` — 3677 bytes,
  three certificates, `Intel SGX PCK Certificate` -> `Intel SGX PCK Platform
  CA` -> `Intel SGX Root CA`. The quote alone therefore chains to Intel's root
  with nothing borrowed from collateral, which is *why* discarding the supplied
  chain is safe rather than merely observed to be.
- The sanitiser *is* the security boundary, so it needs a `run_miner_attacks`
  case that feeds hostile collateral through it and asserts the chain is gone —
  not a unit test that only checks the honest round trip.

### Which route

If validators are few and technical, Route A buys most of the benefit for none
of the consensus risk. At scale, Route B is better because it dissolves the
ambiguity rather than managing it. **What cannot continue indefinitely is the
per-proof fetch itself.**

---

## 3. Still open, and only real hardware settles them

Two questions survived several sessions of reading and will not yield to more
of it. Both need a live run.

- **Does `verify()` raise on `Revoked`, or return it as a status?** The binary
  carries the literal `TCB status is invalid: Revoked`, but the `.status`
  docstring lists `REVOKED` as an observable return value and
  `QuotePolicy.strict` would be pointless if plain `verify` rejected everything
  but UpToDate. The rehearsal saw only `UpToDate`, so the question did not
  arise — **an absent observation is not a null result.**
- **Does `verify()` prefer the collateral's certificate chain over the
  quote's?** Decides whether the sanitiser is essential or merely correct.
  Cheap experiment once a real quote and its collateral exist: replace the
  collateral's `pck_certificate_chain` with garbage and see whether
  verification still succeeds. If it does, the quote's chain is being used.

`verify_dcap` now logs `.status` and `.advisory_ids`, so the first question
answers itself the next time a live box runs — no extra work needed.

---

## 4. TCB status policy — deliberately not decided

`verify_dcap` accepts every status. A passing DCAP check proves the quote is
genuine, **not that the hardware is patched**, and an `UpToDate` box and a
`REVOKED` box are currently indistinguishable to the validator.

Not enforced on purpose. On GCP and Azure the firmware belongs to the cloud
provider, so rejecting `OUT_OF_DATE` would drop honest miners for their host's
maintenance schedule — and drop every miner on a platform generation at once
when Intel publishes. Enforcement is also a consensus change: two validators
with different thresholds disagree about identical bytes, so it needs a pinned
grace period and an INVARIANTS entry.

**The blocker is data, not design.** Decide it once the logs show the real
distribution of statuses across a live field.

---

## Decisions already made — do not re-litigate

- **Per-epoch abandon is not an option.** It was once recorded as the
  consensus-safe choice. It is the most dangerous of the three: I6 says no
  miner behaviour may stop a validator setting weights, and with the
  crafted-FMSPC path open, per-epoch abandon hands **every miner a subnet kill
  switch** for the price of one registration, fired simultaneously because the
  collection point is a deterministic block. A threshold hybrid inherits it at
  N+1 registrations. This is encoded as
  `test_an_unverifiable_proof_never_stops_the_epoch`.
- **Skip-the-miner is the only I6-compatible answer**, and it is what the
  validator already did. Skipped miners fall through to `apply_miss` like any
  absent miner — correct because weights are *relative*: the failure it
  mishandles is correlated (everyone docked, ranking preserved), the one it
  prevents is uncorrelated (one miner escaping a miss gains on everyone).
- **Skip-the-miner does not make weights agree.** Least bad local option. The
  divergence is in the input; only removing the fetch closes it.
- **Phala is not a trust root.** Collateral is Intel-signed and checked against
  a compiled-in root CA, so a mirror can withhold or serve stale, never forge.
  The endpoint is an availability, privacy and latency choice.
- **The crafted-FMSPC path is still miner-reachable.** `get_collateral` derives
  its request from the FMSPC in the quote, and `dcap-qvl` surfaces every failure
  as `ValueError`, so miner-caused and network-caused are separable only by
  message prose. Parse-first narrowed it; only removing the fetch closes it.

---

## Numbers worth not re-measuring

| Quantity | Value | How obtained |
|---|---|---|
| Local verify cost per proof | 2.55–2.68 ms | measured, real proofs, 300-question slice |
| Same, × 256 miners | 0.67 s — 0.04% of budget | measured |
| Collateral round trip | 788 ms cold, ~690 ms warm | measured, GCP us-central1 → Phala |
| Round trip as share of proof cost | ~99.6% | measured |
| 256 miners × measured RTT | 178 s of the 1800 s budget (~10%) | modelled from the above |
| Post-collection budget | 1800 s, not 3600 | `EPOCH_COLLECT_FRACTION=0.5` |

The budget is half the epoch because collection starts halfway in, and verify,
score, weight-setting and the reveal all share what remains.

`scripts/bench_verify_throughput.py` reproduces all of it, including the
abstention model. Its RTT sweep marks the one measured row and labels the rest
as swept.

---

## Suggested order

1. **The timeout**, plus widening I6 and its check. Small, unblocked, and the
   most dangerous thing still open.
2. **Run the rehearsal against merged `main`.** The accounting from PR #9 makes
   an outage legible — `n_heads_unverifiable` in the epoch log is the real
   failure rate, and it is the missing input to how urgently item 3 is needed.
   It also answers the `Revoked` question for free.
3. **Choose Route A or Route B** with that number in hand.
4. **TCB status policy**, once the logs show the real distribution.

---

## A method note, earned the hard way

Everything that settled a question in this work came from **running** something
— a JSON round trip, a 60-second sleep, a closed port, a real quote, a
falsified test. Everything that went wrong came from careful reading of an
adjacent layer: a docstring, a symbol table, a Python wrapper hiding a Rust
field, a branch reasoned about instead of traced. Care did not prevent any of
them; execution did.

Two consequences worth keeping:

- **Label claims by how they were obtained.** "Measured" and "read" have very
  different track records here and prose does not distinguish them.
- **Prefer a guard to a conclusion.** The kill-switch argument took several
  rounds and two wrong turns; it is now a test that fails CI, so nobody has to
  reconstruct the reasoning. Most of what a session produces is conclusions —
  the ones worth keeping become checks.
