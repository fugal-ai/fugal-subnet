# Open Work — pick up here

Written 2026-09-06 as a handoff after PR #9; trimmed 2026-11-06 to what is
still open. The decisions recorded here cost several sessions to reach and
re-deriving them is the expensive part, so the "do not re-litigate" list and
the numbers stay. Everything that has since been done lives where it landed:
the timeout and the epoch budget in `INVARIANTS.md` (I6), the live subnet and
its entries in `MAINNET_LAUNCH.md`, protocol v2 in `INVARIANTS.md` and
`design-decisions.md` Step 8.

## Status as of 2026-11-06

- finney netuid 70 is live under protocol v1 (4 validators, 194 routable
  miners, 144 earning at block 9449357); epochs have run end to end on real
  hardware since 2026-11-04, so the September note "no epoch has been observed
  running end to end" no longer applies.
- The per-proof collateral timeout and the per-epoch budget are built
  (`INVARIANTS.md`, I6 entries; `tests/test_collateral_timeout.py`).
- Miner economics, head efficacy and pool memorisation are answered in their
  own documents (`MINER_ECONOMICS.md`, `HEAD_EFFICACY.md`); the v2 economics
  re-derivation is open (`design-decisions.md`, "Open questions", v2).
- Still open, below: removing the per-proof collateral fetch (§2), the two
  questions only real hardware settles (§3), and the TCB status policy (§4).

---

## 2. Item 3 — remove the per-proof fetch

This is the real fix. **Two viable routes**, and they are not equivalent.

*Update 2026-11-08:* a third, smaller step is done: the validator keeps one
collateral per platform per epoch (`CollateralBudget.collateral_for`;
INVARIANTS "one collateral fetch per platform per epoch"). It gives most of
Route A's benefit with no daemon. Unchanged: the crafted-FMSPC path (an
unknown platform is still a fetch, bounded by the fair share), and Route B
remains the end state that removes the fetch entirely.

*Residual noted 2026-11-09 (ledger review):* the per-epoch collateral is not
re-fetched within the epoch. If its `nextUpdate` passes mid-epoch, this
validator's later proofs that epoch become unverifiable (an outage, not a
refusal) while a peer that fetched later could verify them. Low risk: PCCS
serves refreshed collateral well before `nextUpdate`, and validators fetch at
about the same time. Route B removes it.

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
- This cannot break an honest miner: a real dstack quote carries its own chain
  (**measured** on `attestation_A.bin` — 3677 bytes, 3 certificates).
- The sanitiser *is* the security boundary, so it needs a `run_miner_attacks`
  case that feeds hostile collateral through it and asserts the chain is gone —
  not a unit test that only checks the honest round trip.

### Which route

If validators are few and technical, Route A buys most of the benefit for none
of the consensus risk. At scale, Route B is better because it dissolves the
ambiguity rather than managing it. **What cannot continue indefinitely is the
per-proof fetch itself.**

---

## 2b. One cheap measurement worth taking during the rehearsal

**How long does the configured PCCS take to refuse an FMSPC it does not know?**

It decides how many hostile registrations it would take to drain a validator's
collateral ceiling: a fast 404 (~0.7 s) needs >850 UIDs and is impossible at a
field of 256; a stall to the full per-call bound needs ~60. The per-proof fair
share means safety no longer depends on the answer, but tuning does — it is the
difference between a 2.34 s floor being generous and being tight.

Recipe, and it costs nothing beyond one request: take a real quote, rewrite the
FMSPC bytes, call `get_collateral` against the configured PCCS, time it. Worth
folding into the rehearsal rather than running as an unsolicited request to a
third party's endpoint.

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
- **`measurement_id` does not include RTMR0**, so the machine shape does not
  affect it and miners may choose any instance size — measured byte-identical
  across `c3-standard-4` and `c3-standard-8`. MAINNET_LAUNCH.md carried the old
  four-register formula and concluded the subnet would fork by instance size;
  corrected here, and worth not reintroducing, because that document is read
  immediately before spending real TAO.
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

`scripts/experiments/bench_verify_throughput.py` reproduces all of it, including the
abstention model. Its RTT sweep marks the one measured row and labels the rest
as swept.

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
