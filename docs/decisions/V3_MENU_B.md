# v3 menu B: entries added at a block, with a grace period (v0.6.0)

Decided 2026-11-17, owner-approved for v0.6.0.

## Upgrade timing (required)

Miners must re-provision on the v0.6.0 image (its frozen compose comes in the
follow-up composes PR) **before** block 9,531,000: from that block validators
plan exploration over menu B's 60 entries, and a v0.5.0 image, which plans over
menu A's 53, fails verification. Validators must update to `main` at or after
the composes PR merge and restart before the block, because approved entries
are read at startup; a v0.5.0 validator left running past the block keeps
menu A and diverges from the upgraded ones. The grace length (20 epochs) is
compiled for finney like the block (`config.menu_grace_epochs`).

## What changes

Seven entries join the v3 menu: `openai/gpt-6.2-astra@{low,high,max}` ($8 / $40 per
million tokens) and `openai/gpt-6.2-sol@{low,medium,high,max}` ($2 / $10), each with
`max_tokens` 128000, `reasoning.effort` set to the level in its id, and pinned to
OpenAI then Azure with no fallbacks. No existing entry changes or leaves the menu.

- **Menu A** (`data/models_v3.json`, 53 entries, unchanged, same pins) applies
  before the menu B block.
- **Menu B** (`data/models_v3b.json`, 60 entries) applies from the first v3 epoch
  whose boundary is at or after `config.menu_b_block`: finney's block is compiled
  in (`FINNEY_MENU_B_BLOCK` = 9,531,000, epoch 26475, ~2026-11-18 04:22 UTC, one
  epoch after the odds seat block). On another chain the block comes from
  `FUGAL_MENU_B_BLOCK`, and on finney that variable is refused unless it repeats
  the compiled value.
- Menu B's token manifest is menu A's reviewed manifest with one measured row
  per added entry: 10 calls each on the v3 request path, all served by OpenAI,
  $0.72 in total (`data/observations/tokens_v3b_new_entries_2026-11-17.jsonl`).
  `scripts/build_menu_v3b.py` rebuilds both files byte for byte.

## Why nothing resets

Menu B's rows for menu A's entries are menu A's rows, unchanged. That gives
three properties:

1. **Index space.** Menu B's model list is menu A's sorted list followed by the
   added entries, sorted (`routing_protocol.models_for_protocol(3, "b")`). Every
   menu A entry keeps its index, so a window spanning the block stays aligned.
   Each epoch's slice is indexed by its own list anyway: a slice before the block
   has 53 columns and one after it has 60.
2. **Heads.** A head built for menu A has menu A's manifest digest as its cost
   profile. On menu B both manifests are accepted (`benchmark_costs_for(...,
   menu="b")`), so an A head's proofs keep verifying after the block, priced at
   menu B's table (identical prices for A's entries). A head built for menu B is
   refused on epochs before the block. Miners should commit one only after it.
3. **State.** Seat, burn-in, copy-rule seniority and the window carry across the
   block untouched. This is not a protocol change: the protocol is still v3 and
   nothing that resets at a protocol change (the window, the seat) runs.

## The grace period

From the block, the added entries are part of exploration at once: their
cost-aware shares are drawn from menu B's prices and token means, so every miner
measures them from the first menu B epoch. That data feeds the reference R, the
implausibility detector and the single-model table.

For `MENU_GRACE_EPOCHS` (20, ~24 h) epochs from the block, the added entries'
pooled pairs are scored for **no head**. They are kept in the window with
`EpochPairs.scored` false, published in the reveal with the list
`odds_unscored_entries`, and never scored later. The odds seat's menu
(`OddsInput.menu`, the full-menu rule) is menu A for those epochs.

The rejected alternative was to score the added pairs during grace at R's odds
for any head that lacks a row. That is gameable: R is fitted on the window's
outcomes, and a head can choose to omit exactly the rows where R beats it (the
v3 review of 2026-11-16 found this for menu rows in general). Leaving the pairs
unscored for every head treats all heads equally, so leaving out the new rows
during grace gains a head nothing (`tests/test_v3_menu_b.py`).

After grace the full-menu rule applies to menu B. A head without a row for an
added entry publishes no odds (-1 on every pair) and is not eligible to take
or hold the odds seat. Miners have the grace period to retrain and commit a
60-row head.

## How readers pick the menu

Every per-epoch reader picks the menu from the epoch's boundary through
`routing_protocol.menu_for_epoch`:

- the validator's slice, exploration plan, pricing and proof checks;
- the miner, per epoch, inside the TD;
- the odds input, through `scored_menu` and `grace_excluded`;
- `verify_live_miner.py`.

A rebuild or re-anchor prices every v3 reveal at menu B's table. That table is
a superset holding menu A's prices for A's entries, so it prices reveals from
before the block exactly as they were priced live. It reads the grace mask from
the reveal's `odds_unscored_entries`. The block, the grace length and both
menu B files are in the consensus digest.

## The miner reads compiled blocks

The miner used to pick its protocol from `FUGAL_V3_ACTIVATION_BLOCK` in its
compose alone. It now reads the release's compiled finney blocks for the
protocol, the flip block and the menu B block (`config.protocol_for_chain`,
`v3_activation_block`, `menu_b_block`), keyed on the chain's genesis hash. The
compose variables may only repeat them on finney. The compose check in
`check_safety_invariants.py` is kept.
