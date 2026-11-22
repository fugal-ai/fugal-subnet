# Decision records and measurements

Dated documents that explain *why* the subnet is the way it is. Nobody runs the
subnet from these; they exist so a decision is not re-made from scratch. Each
says at its top which protocol it describes and whether it is superseded. The
current rules, with what enforces each, are in `../INVARIANTS.md`; the
operator guides are beside it in `docs/`.

| file | what |
|---|---|
| `design-decisions.md` | the design, step by step; Step 8 is protocol v2 |
| `I3_DECISION.md` | the 2026-09-08 scoring decision (v1; superseded for v2) |
| `CODE_BENCHMARK_PLAN.md` | why the code benchmarks are out, and the phases to return them |
| `HEAD_EFFICACY.md` | the experiments: trained vs random heads, pool memorisation, copies |
| `MINER_ECONOMICS.md` | the v1 miner economics model; the v2 seat arithmetic in its banner |
| `REHEARSAL_2026-09-06.md` | the first live-hardware rehearsal, every line measured or read |
| `LIVE_MENU_V4.md` | DESIGN (not built): protocol v4, the menu as published state, measured prices and token means, per-entry grace, retirement, a second attestation path |
| `SEAT_MINIMAL_V4.md` | DESIGN (not built): the v4 seat reduced to three rules on one bound (odds test, real-picks check, precedence), new entries usable at once, the label budget, a memorisation check; an inventory of every v3 seat rule with its verdict |
| `FAST_CROWN_V4.md` | DESIGN (decided, not built): a better router seated in one epoch: one-epoch challenges the challenger pays for on both sides, the holder on exploration only, the bound retuned, lower floors, one measuring epoch, a one-epoch entry allowance and a 20-epoch displacement shield; a seat path on better picks and a join at k + 1 considered and not adopted; a value floor against R proposed |
| `OPEN_WORK.md` | the September handoff, trimmed to what is still open |
| `evaluation/` | the success-contract evaluation outputs the miner guide links |
