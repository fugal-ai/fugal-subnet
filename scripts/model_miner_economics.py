#!/usr/bin/env python3
"""Is a Fugal miner net-positive? A runnable model, not a prose estimate.

WHY THIS EXISTS. The whole TEE/attestation effort assumes miners show up. If a
miner spends more than it earns, nobody runs one, and every consensus property
this repo defends is a property of an empty subnet. Nobody had computed the
number, so this script computes it.

WHAT IT DOES NOT DO. It makes no paid API calls, and it cannot: it never
imports `fugal_subnet.api` and never reads OPENROUTER_API_KEY. The only network
call it can make is `--live-chain`, a read-only subtensor query, which is free.
Everything else runs from pinned constants recorded below.

HOW TO READ THE NUMBERS. Every input carries a provenance tag, because this
project has been burned by prose that blurs the three:

    read-from-chain   queried from finney, reproducible with --live-chain
    computed          derived by running repo code over repo data
    market            a public price quote, with source and date
    assumed           a choice nobody has measured; the sensitivity sweep is
                      how you find out whether it matters

THE SHAPE OF THE ANSWER. Miner revenue for the whole subnet is fixed by the
chain (0.41 alpha/block, uniform across every active subnet) and does not grow
with the number of miners. Miner cost is per-miner and roughly constant. So
there is an equilibrium field size N* where the marginal miner breaks even, and
"is a miner net-positive?" is really "is N below N*?". N* is the headline
output.

Usage:
    python scripts/model_miner_economics.py                  # base case
    python scripts/model_miner_economics.py --live-chain     # re-read the chain
    python scripts/model_miner_economics.py --sweep          # sensitivity grid
    python scripts/model_miner_economics.py --recompute-slice-tokens
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Pinned inputs. Each is (value, provenance, source).
#
# Re-derive the chain rows with --live-chain and the cost rows with
# --recompute-slice-tokens. If a re-derivation disagrees with a pin, the pin is
# stale: update it here and say so in docs/MINER_ECONOMICS.md.
# --------------------------------------------------------------------------

CHAIN_SNAPSHOT_BLOCK = 9_009_022
CHAIN_SNAPSHOT_DATE = "2026-09-06"


@dataclass(frozen=True)
class Input:
    value: float
    unit: str
    provenance: str
    source: str


PINNED = {
    # --- emission side -----------------------------------------------------
    "alpha_out_per_block": Input(
        1.0, "alpha/block", "read-from-chain",
        f"DynamicInfo.alpha_out_emission, identical (1.0) for all 124 active "
        f"subnets at block {CHAIN_SNAPSHOT_BLOCK}; the 5 exceptions are netuid 0 "
        f"(root) and 4 subnets registered within the last ~10 days",
    ),
    "miner_emission_share": Input(
        0.41, "fraction", "read-from-chain",
        "Metagraph.emission summed over UIDs equals tempo * 0.82 exactly on "
        "netuids 1 (tempo 99 -> 81.180), 90 and 99 (tempo 360 -> 295.2016). "
        "Half of that 0.82 carries incentive (netuid 99: 147.6008 of 295.2017), "
        "so the split is 18% owner / 41% validators / 41% miners. Not assumed "
        "from documentation -- checked against three subnets with two tempos",
    ),
    "alpha_price_tao": Input(
        0.00491, "tao/alpha", "read-from-chain",
        f"median DynamicInfo.price across 128 non-root subnets at block "
        f"{CHAIN_SNAPSHOT_BLOCK}. Age-independent: median is 0.00528 for subnets "
        f"under 3 months old, 0.00507 at 3-12 months, 0.00463 past a year, so a "
        f"new subnet's expected price is the population median, not a discount "
        f"to it. p10 0.00275, p90 0.02775",
    ),
    "block_seconds": Input(
        12.0, "s/block", "read-from-chain",
        "bittensor.core.settings.BLOCKTIME (SDK 10.5.0)",
    ),
    "tao_usd": Input(
        240.91, "USD/TAO", "market",
        f"CoinGecko simple/price bittensor, {CHAIN_SNAPSHOT_DATE}. Kraken "
        f"TAOUSD last trade the same minute: 241.44. Two independent venues "
        f"within 0.2%",
    ),
    "registration_burn_tao": Input(
        0.0005, "TAO", "read-from-chain",
        f"Subtensor.recycle() on netuids 64, 90, 99 at block "
        f"{CHAIN_SNAPSHOT_BLOCK}; the floor for a young subnet. Established "
        f"subnets are higher (netuid 51: 0.5 TAO). One-off, and at the floor it "
        f"is ~$0.12 -- included for completeness, not because it matters",
    ),

    # --- cost side ---------------------------------------------------------
    "slice_input_tokens": Input(
        22094.6, "tokens/epoch", "computed",
        "mean over 50 nonce-seeded slices of pricing.question_input_tokens "
        "across slicer.select_slice(size=300) on the real pool "
        "(loader.load_all -> 21,553 questions after the humaneval exclusion). "
        "sd 1010 (4.6% CV), min 20498, max 24903 -- the slice is stratified, so "
        "epoch-to-epoch cost variance is small",
    ),
    "completion_tokens": Input(
        256.0, "tokens/answer", "assumed",
        "config.FRAME_DEFAULT_COMPLETION_TOKENS. THIS IS THE WEAKEST INPUT IN "
        "THE MODEL. No completion length has ever been measured against a real "
        "provider -- docs/LIVE_API_VALIDATION.md says so outright, and the "
        "~112-131 tokens visible in results/rehearsal/ came from "
        "scripts/stub_upstream.py, a synthetic stub. Output rates are 3-6x "
        "input rates across the pinned table, so this term dominates the bill. "
        "Swept in --sweep",
    ),
    "cvm_usd_month": Input(
        150.0, "USD/month", "measured",
        "GCP c3-standard-4 confidential VM, the shape the rehearsal actually "
        "ran on. Corroborated by docs/TDX_VALIDATION.md's $0.20/hour = "
        "$146/month. Excludes egress and the boot disk",
    ),

    "score_spread": Input(
        2.0, "best:worst", "assumed",
        "The rehearsal produced weights 0.298 / 0.240 / 0.463 -- a 1.93x spread "
        "-- across THREE miners with hand-made heads, which is the weakest "
        "input in this model after completion_tokens. A real field has dedup, "
        "burn-in and evidence accumulation working on it and should spread "
        "wider. N* scales linearly in the multiplier 2/(1+spread), so this is "
        "swept rather than trusted",
    ),
    "gpu_usd_month": Input(
        0.0, "USD/month", "assumed",
        "NOT MEASURED, and defaulted to zero deliberately so the omission is "
        "visible rather than silent. The subnet's premise is continuous head "
        "optimisation, which implies ongoing training compute; a miner that "
        "trains on rented GPU pays for it and this model does not know what. "
        "Zero is the floor case -- a miner training once and then only serving. "
        "Swept in --sweep",
    ),

    # --- shape of the field ------------------------------------------------
    "n_miners": Input(
        32.0, "miners", "assumed",
        "No Fugal field exists. For calibration, live subnets carry far fewer "
        "EARNING miners than registered UIDs: at block "
        f"{CHAIN_SNAPSHOT_BLOCK}, of 256 UIDs each, netuid 64 had 17 earning, "
        "51 had 63, 4 had 6, 120 had 5, 44 had 1. Registered-but-idle UIDs cost "
        "their holder nothing ongoing and earn nothing, so they are not the "
        "denominator -- earning miners are",
    ),
}

# Repo config values, read from the repo rather than retyped.
sys.path.insert(0, str(REPO))
from fugal_subnet import config  # noqa: E402
from fugal_subnet.benchmarks import slicer  # noqa: E402

SECONDS_PER_DAY = 86_400.0
DAYS_PER_MONTH = 365.0 / 12.0


def load_prices() -> dict[str, tuple[float, float]]:
    """Per-token USD rates from the hash-pinned consensus table. Read only."""
    models = json.loads((REPO / "data" / "models.json").read_text())
    return {m["id"]: (m["in"] / 1e6, m["out"] / 1e6) for m in models}


# --------------------------------------------------------------------------
# Cost side
# --------------------------------------------------------------------------

def slice_cost_usd(
    prices: dict[str, tuple[float, float]],
    rate_in: float,
    rate_out: float,
    slice_input_tokens: float,
    slice_size: int,
    completion_tokens: float,
) -> float:
    """Cost of answering one epoch's slice at one (rate_in, rate_out) pair.

    Exact, not sampled: policy_cost is linear in tokens, so the whole slice
    collapses to (sum of input tokens) * rate_in + n * completion * rate_out.
    That is why only one pool statistic needs pinning.
    """
    return slice_input_tokens * rate_in + slice_size * completion_tokens * rate_out


def policy_rates(
    prices: dict[str, tuple[float, float]], policy: str
) -> tuple[float, float, str]:
    """(rate_in, rate_out, label) for a named routing policy.

    A head is a routing policy over the pool, so its per-epoch bill lies
    between routing everything to the cheapest model and routing everything to
    the dearest. Those bounds are the honest way to state miner spend; a single
    figure hides which policy it assumes.
    """
    ranked = sorted(prices.items(), key=lambda kv: kv[1][0] + kv[1][1])
    if policy == "cheapest":
        mid, (ri, ro) = ranked[0]
        return ri, ro, f"all -> {mid}"
    if policy == "dearest":
        mid, (ri, ro) = ranked[-1]
        return ri, ro, f"all -> {mid}"
    if policy == "median":
        mid, (ri, ro) = ranked[len(ranked) // 2]
        return ri, ro, f"all -> {mid}"
    if policy == "mix":
        n = len(ranked)
        ri = sum(v[0] for _, v in ranked) / n
        ro = sum(v[1] for _, v in ranked) / n
        return ri, ro, f"uniform over {n} models"
    raise ValueError(f"unknown policy {policy!r}")


def epoch_api_cost_usd(
    prices: dict[str, tuple[float, float]],
    policy: str,
    slice_input_tokens: float,
    slice_size: int,
    completion_tokens: float,
    explore_fraction: float,
) -> tuple[float, float, str]:
    """(routed cost, exploration cost, label) for one epoch, in USD.

    Exploration is priced separately and at the UNIFORM mix, not at the miner's
    policy: config.EXPLORE_FRACTION of the slice is answered with a
    nonce-chosen model the miner does not pick. It is a cost floor no routing
    strategy can optimise away, which is exactly what it is there to be.
    """
    ri, ro, label = policy_rates(prices, policy)
    routed = slice_cost_usd(
        prices, ri, ro, slice_input_tokens, slice_size, completion_tokens
    )
    mri, mro, _ = policy_rates(prices, "mix")
    n_explore = explore_fraction * slice_size
    per_q_input = slice_input_tokens / slice_size
    explore = n_explore * (per_q_input * mri + completion_tokens * mro)
    return routed, explore, label


# --------------------------------------------------------------------------
# Revenue side
# --------------------------------------------------------------------------

def subnet_miner_revenue_usd_day(
    alpha_out_per_block: float,
    miner_emission_share: float,
    block_seconds: float,
    alpha_price_tao: float,
    tao_usd: float,
) -> float:
    """USD/day flowing to ALL miners on one subnet.

    Fixed by the chain and independent of how many miners there are, which is
    the single most important fact in this model: miners divide a constant pot.
    """
    blocks_per_day = SECONDS_PER_DAY / block_seconds
    alpha_per_day = alpha_out_per_block * miner_emission_share * blocks_per_day
    return alpha_per_day * alpha_price_tao * tao_usd


def realisable_fraction(
    alpha_per_day: float, alpha_in: float, tao_in: float, alpha_price_tao: float
) -> float:
    """Share of face value a miner keeps after AMM slippage on selling.

    Miners are paid in alpha and quoted in USD only after selling into the
    subnet's constant-product pool. Measured at block 9,009,022 this is a ~1%
    haircut on the whole subnet's daily miner emission (netuid 99: 2952 alpha
    into alpha_in 219,960 / tao_in 697 gives 9.23 TAO against a face value of
    9.36), so it is reported and then not carried through the headline. It
    would matter for a pool an order of magnitude thinner.
    """
    if alpha_in <= 0 or tao_in <= 0:
        return 1.0
    tao_out = tao_in * alpha_per_day / (alpha_in + alpha_per_day)
    face = alpha_per_day * alpha_price_tao
    return tao_out / face if face > 0 else 1.0


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

@dataclass
class Result:
    revenue_usd_day: float
    api_usd_day: float
    explore_usd_day: float
    cvm_usd_day: float
    gpu_usd_day: float
    cost_usd_day: float
    net_usd_day: float
    breakeven_miners: float
    breakeven_miners_average: float
    share_multiplier: float
    breakeven_alpha_price: float
    breakeven_tao_usd: float
    epochs_per_day: float
    policy_label: str
    subnet_pot_usd_day: float
    incentive_share: float


def model(args, prices) -> Result:
    routed, explore, label = epoch_api_cost_usd(
        prices, args.policy, args.slice_input_tokens, args.slice_size,
        args.completion_tokens, args.explore_fraction,
    )
    epochs_per_day = SECONDS_PER_DAY / args.epoch_seconds
    api = routed * epochs_per_day
    exp = explore * epochs_per_day
    cvm = args.cvm_usd_month / DAYS_PER_MONTH
    gpu = args.gpu_usd_month / DAYS_PER_MONTH
    cost = api + exp + cvm + gpu

    pot = subnet_miner_revenue_usd_day(
        args.alpha_out_per_block, args.miner_emission_share,
        args.block_seconds, args.alpha_price_tao, args.tao_usd,
    )
    # Weights are score-proportional, so the miner whose entry decision sets
    # the equilibrium is the LOWEST-scoring one, not the average one. Under
    # scores spread uniformly over a best:worst ratio R, the lowest miner's
    # share is 2/(1+R) of an equal split -- 0.67 at the 2x spread the rehearsal
    # produced (0.298/0.240/0.463). An explicit --incentive-share overrides the
    # derivation.
    multiplier = 2.0 / (1.0 + args.spread)
    if args.incentive_share:
        share = args.incentive_share
        multiplier = share * args.n_miners
    else:
        share = multiplier / args.n_miners
    revenue = pot * share

    # Break-evens: each solves cost == revenue for one variable, holding the
    # rest at their current values.
    #
    # `pot / cost` is where the AVERAGE miner breaks even, and quoting it as the
    # equilibrium was wrong: it ignored the share multiplier entirely, so
    # --incentive-share could report a loss-making miner alongside an
    # equilibrium field size larger than the field it was losing money in. The
    # equilibrium is set by the marginal entrant, hence the multiplier.
    be_miners = pot * multiplier / cost if cost > 0 else float("inf")
    be_miners_avg = pot / cost if cost > 0 else float("inf")
    be_price = (
        args.alpha_price_tao * cost / revenue if revenue > 0 else float("inf")
    )
    be_tao = args.tao_usd * cost / revenue if revenue > 0 else float("inf")

    return Result(
        revenue_usd_day=revenue, api_usd_day=api, explore_usd_day=exp,
        cvm_usd_day=cvm, gpu_usd_day=gpu, cost_usd_day=cost,
        net_usd_day=revenue - cost,
        breakeven_miners=be_miners, breakeven_miners_average=be_miners_avg,
        share_multiplier=multiplier, breakeven_alpha_price=be_price,
        breakeven_tao_usd=be_tao, epochs_per_day=epochs_per_day,
        policy_label=label, subnet_pot_usd_day=pot, incentive_share=share,
    )


# --------------------------------------------------------------------------
# Re-derivation (the flags that make this a model rather than a table)
# --------------------------------------------------------------------------

def refresh_from_chain(args) -> list[str]:
    """Re-read the emission inputs from finney. Read-only, no spend."""
    import bittensor as bt

    notes = []
    st = bt.Subtensor(network=args.network)
    block = st.get_current_block()
    subs = [s for s in st.all_subnets() if s.netuid != 0]
    prices = sorted(float(s.price) for s in subs)
    med = prices[len(prices) // 2]
    aoe = [float(s.alpha_out_emission) for s in subs]
    active = [a for a in aoe if a > 0]
    notes.append(f"block {block}, {len(subs)} non-root subnets")
    notes.append(
        f"alpha_out_emission: {len(active)} active, all "
        f"{'equal to ' + str(active[0]) if len(set(active)) == 1 else 'DIFFERING ' + str(sorted(set(active)))}"
    )
    notes.append(
        f"alpha price p10/median/p90 = {prices[len(prices)//10]:.5f} / "
        f"{med:.5f} / {prices[9*len(prices)//10]:.5f}"
    )
    if active:
        args.alpha_out_per_block = active[0]
        args._rederived.add("alpha_out_per_block")
    args.alpha_price_tao = med
    args._rederived.add("alpha_price_tao")
    for name, pinned in (
        ("alpha_price_tao", PINNED["alpha_price_tao"].value),
        ("alpha_out_per_block", PINNED["alpha_out_per_block"].value),
    ):
        live = getattr(args, name)
        if pinned and abs(live - pinned) / pinned > 0.10:
            notes.append(
                f"!! {name} drifted >10% from the pin ({pinned} -> {live:.6g}); "
                f"update PINNED and docs/MINER_ECONOMICS.md"
            )
    return notes


def refresh_slice_tokens(args) -> list[str]:
    """Re-derive the pool statistic by running the real slicer. No spend."""
    import secrets
    import statistics

    from fugal_subnet.benchmarks import loader
    from fugal_subnet.pricing import question_input_tokens

    pool = loader.load_all(strict=False)
    totals = []
    for _ in range(args.slice_samples):
        sl = slicer.select_slice(secrets.token_bytes(32), pool, args.slice_size)
        totals.append(sum(question_input_tokens(q.get("prompt", "")) for q in sl))
    mean = statistics.mean(totals)
    notes = [
        f"pool {len(pool)} questions, {args.slice_samples} slices of "
        f"{args.slice_size}: mean {mean:.1f} input tokens "
        f"(sd {statistics.stdev(totals):.1f}, min {min(totals)}, max {max(totals)})"
    ]
    pin = PINNED["slice_input_tokens"].value
    if abs(mean - pin) / pin > 0.05:
        notes.append(f"!! drifted >5% from the pin ({pin} -> {mean:.1f})")
    args.slice_input_tokens = mean
    args._rederived.add("slice_input_tokens")
    return notes


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_inputs(args) -> None:
    print("INPUTS")
    print(f"  {'input':24s} {'value':>12s}  {'provenance':16s} unit")
    print("  (* = re-derived live this run, same provenance, fresher reading)")
    rows = [
        ("alpha_out_per_block", args.alpha_out_per_block),
        ("miner_emission_share", args.miner_emission_share),
        ("alpha_price_tao", args.alpha_price_tao),
        ("tao_usd", args.tao_usd),
        ("block_seconds", args.block_seconds),
        ("slice_input_tokens", args.slice_input_tokens),
        ("completion_tokens", args.completion_tokens),
        ("cvm_usd_month", args.cvm_usd_month),
        ("gpu_usd_month", args.gpu_usd_month),
        ("score_spread", args.spread),
        ("n_miners", float(args.n_miners)),
    ]
    rederived = getattr(args, "_rederived", set())
    for name, value in rows:
        p = PINNED[name]
        tag = p.provenance
        if name in rederived:
            # Same provenance, fresher reading -- not a hand override.
            tag = f"{p.provenance} *"
        elif abs(value - p.value) > 1e-12:
            tag = "overridden"
        print(f"  {name:24s} {value:12.6g}  {tag:16s} {p.unit}")
    print(f"  {'epoch_seconds':24s} {args.epoch_seconds:12.6g}  "
          f"{'repo config':16s} s (config.EPOCH_INTERVAL)")
    print(f"  {'slice_size':24s} {args.slice_size:12.6g}  "
          f"{'repo config':16s} questions (config.SLICE_SIZE)")
    print(f"  {'explore_fraction':24s} {args.explore_fraction:12.6g}  "
          f"{'repo config':16s} fraction (config.EXPLORE_FRACTION)")
    print()


def print_result(r: Result, args) -> None:
    print("PER-MINER ECONOMICS, USD/day")
    print(f"  routing policy               {r.policy_label}")
    print(f"  epochs/day                   {r.epochs_per_day:.1f}")
    print(f"  incentive share              {r.incentive_share:.4f} "
          f"({'explicit' if args.incentive_share else 'marginal miner: %.2f of an equal 1/%d split, from a %.1fx score spread' % (r.share_multiplier, args.n_miners, args.spread)})")
    print()
    print(f"  subnet-wide miner pot        {r.subnet_pot_usd_day:10.2f}   "
          f"(fixed by chain, does not grow with the field)")
    print(f"  revenue                      {r.revenue_usd_day:10.2f}")
    print(f"  - API, routed slice          {-r.api_usd_day:10.2f}")
    print(f"  - API, forced exploration    {-r.explore_usd_day:10.2f}")
    print(f"  - confidential VM            {-r.cvm_usd_day:10.2f}")
    if r.gpu_usd_day:
        print(f"  - head-training compute      {-r.gpu_usd_day:10.2f}")
    print(f"  {'':28s} {'-' * 10}")
    print(f"  NET                          {r.net_usd_day:10.2f}   "
          f"({'PROFITABLE' if r.net_usd_day > 0 else 'LOSS-MAKING'})")
    print()
    print("BREAK-EVEN FRONTIER (each solved holding the others fixed)")
    print(f"  equilibrium field size       {r.breakeven_miners:10.1f} miners  "
          f"-- the MARGINAL (lowest-scoring) entrant; above this it exits")
    print(f"  ... for the AVERAGE miner    {r.breakeven_miners_average:10.1f} miners  "
          f"-- not the equilibrium: nobody decides to enter on the average")
    print(f"  break-even alpha price       {r.breakeven_alpha_price:10.6f} tao/alpha "
          f"(now {args.alpha_price_tao:.6f})")
    print(f"  break-even TAO price         {r.breakeven_tao_usd:10.2f} USD "
          f"(now {args.tao_usd:.2f})")
    print()


def print_sweep(args, prices) -> None:
    import copy

    print("SENSITIVITY SWEEP")
    print()

    print("  net USD/day by field size x alpha price (policy=%s, completion=%d)"
          % (args.policy, args.completion_tokens))
    price_grid = [0.00275, 0.00491, 0.00949, 0.02775]
    n_grid = [4, 8, 16, 32, 64, 128, 256]
    print(f"    {'N miners':>9s} " + "".join(f"{p:>12.5f}" for p in price_grid))
    print(f"    {'':>9s} " + "".join(f"{lbl:>12s}" for lbl in
                                     ("p10", "median", "p75", "p90")))
    for n in n_grid:
        cells = []
        for p in price_grid:
            a = copy.copy(args)
            a.n_miners, a.alpha_price_tao = n, p
            cells.append(model(a, prices).net_usd_day)
        print(f"    {n:>9d} " + "".join(f"{c:>12.2f}" for c in cells))
    print()

    print("  equilibrium field size N* by alpha price x routing policy")
    policies = ["cheapest", "median", "mix", "dearest"]
    print(f"    {'alpha price':>12s} " + "".join(f"{p:>12s}" for p in policies))
    for p in price_grid:
        cells = []
        for pol in policies:
            a = copy.copy(args)
            a.alpha_price_tao, a.policy = p, pol
            cells.append(model(a, prices).breakeven_miners)
        print(f"    {p:>12.5f} " + "".join(f"{c:>12.1f}" for c in cells))
    print()

    print("  equilibrium field size N* by completion tokens x routing policy")
    print("  (completion length is the only unmeasured cost input)")
    comp_grid = [64, 128, 256, 512, 1024]
    print(f"    {'completion':>12s} " + "".join(f"{p:>12s}" for p in policies))
    for c in comp_grid:
        cells = []
        for pol in policies:
            a = copy.copy(args)
            a.completion_tokens, a.policy = float(c), pol
            cells.append(model(a, prices).breakeven_miners)
        print(f"    {c:>12d} " + "".join(f"{x:>12.1f}" for x in cells))
    print()

    print("  per-epoch API bill by routing policy (USD, routed slice only)")
    for pol in policies:
        routed, explore, label = epoch_api_cost_usd(
            prices, pol, args.slice_input_tokens, args.slice_size,
            args.completion_tokens, args.explore_fraction,
        )
        print(f"    {pol:>12s}  {routed:8.4f}  (+{explore:.4f} exploration)  {label}")
    print()

    print("  equilibrium N* by score spread -- how much less the marginal miner earns")
    print("  (the rehearsal's 1.93x came from 3 miners with hand-made heads;")
    print("   a real field with dedup, burn-in and evidence accumulation spreads wider)")
    print(f"    {'spread':>12s} {'multiplier':>11s} " + "".join(f"{p:>12s}" for p in policies))
    for sp in (1.0, 2.0, 3.0, 5.0, 10.0):
        cells = []
        for pol in policies:
            a = copy.copy(args)
            a.spread, a.policy, a.incentive_share = sp, pol, None
            cells.append(model(a, prices).breakeven_miners)
        print(f"    {sp:>11.1f}x {2.0/(1.0+sp):>11.2f} " + "".join(f"{c:>12.1f}" for c in cells))
    print()

    print("  equilibrium N* by head-training compute (unmeasured; 0 = train once, then serve)")
    print(f"    {'GPU $/mo':>12s} " + "".join(f"{p:>12s}" for p in policies))
    for g in (0.0, 100.0, 300.0, 900.0):
        cells = []
        for pol in policies:
            a = copy.copy(args)
            a.gpu_usd_month, a.policy = g, pol
            cells.append(model(a, prices).breakeven_miners)
        print(f"    {g:>12.0f} " + "".join(f"{c:>12.1f}" for c in cells))
    print()

    print("  cost floor sensitivity: N* if the CVM were cheaper or free")
    for cvm in (0.0, 50.0, 150.0, 300.0):
        a = copy.copy(args)
        a.cvm_usd_month = cvm
        print(f"    ${cvm:6.0f}/mo -> N* = {model(a, prices).breakeven_miners:8.1f}")
    print()


def print_slippage(args) -> None:
    """Report the AMM haircut using the thinnest young pool measured."""
    blocks_per_day = SECONDS_PER_DAY / args.block_seconds
    alpha_day = args.alpha_out_per_block * args.miner_emission_share * blocks_per_day
    # netuid 99 (Thirty Spokes), block 9,009,022 -- the thinnest pool among the
    # young subnets sampled, so this is a conservative haircut.
    frac = realisable_fraction(alpha_day, 219_959.80, 697.267, 0.003169973)
    print("REALISATION HAIRCUT (read-from-chain pool depth, netuid 99)")
    print(f"  whole-subnet miner emission  {alpha_day:.0f} alpha/day")
    print(f"  kept after AMM slippage      {frac * 100:.2f}% of face value")
    print("  -> ~1%, so the headline is quoted at face value and this is a note,")
    print("     not a correction. It would matter for a pool 10x thinner.")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--alpha-out-per-block", type=float,
                   default=PINNED["alpha_out_per_block"].value)
    p.add_argument("--miner-emission-share", type=float,
                   default=PINNED["miner_emission_share"].value)
    p.add_argument("--alpha-price-tao", type=float,
                   default=PINNED["alpha_price_tao"].value)
    p.add_argument("--tao-usd", type=float, default=PINNED["tao_usd"].value)
    p.add_argument("--block-seconds", type=float,
                   default=PINNED["block_seconds"].value)
    p.add_argument("--slice-input-tokens", type=float,
                   default=PINNED["slice_input_tokens"].value)
    p.add_argument("--completion-tokens", type=float,
                   default=PINNED["completion_tokens"].value)
    p.add_argument("--cvm-usd-month", type=float,
                   default=PINNED["cvm_usd_month"].value)
    p.add_argument("--n-miners", type=int, default=int(PINNED["n_miners"].value))
    p.add_argument("--incentive-share", type=float, default=None,
                   help="this miner's share of subnet incentive; overrides --spread")
    p.add_argument("--spread", type=float,
                   default=PINNED["score_spread"].value,
                   help="best:worst score ratio across the field; sets how much "
                        "less than 1/N the marginal miner earns")
    p.add_argument("--gpu-usd-month", type=float,
                   default=PINNED["gpu_usd_month"].value,
                   help="head-training compute, if the miner rents it")
    p.add_argument("--epoch-seconds", type=float, default=float(config.EPOCH_INTERVAL))
    p.add_argument("--slice-size", type=int, default=config.SLICE_SIZE)
    p.add_argument("--explore-fraction", type=float, default=config.EXPLORE_FRACTION)
    p.add_argument("--policy", default="mix",
                   choices=["cheapest", "median", "mix", "dearest"])
    p.add_argument("--live-chain", action="store_true",
                   help="re-read emission inputs from subtensor (read-only, free)")
    p.add_argument("--network", default="finney")
    p.add_argument("--recompute-slice-tokens", action="store_true",
                   help="re-derive the pool statistic by running the real slicer")
    p.add_argument("--slice-samples", type=int, default=50)
    p.add_argument("--sweep", action="store_true", help="print the sensitivity grids")
    p.add_argument("--json", action="store_true", help="emit the base case as JSON")
    args = p.parse_args(argv)

    if args.n_miners < 1:
        p.error("--n-miners must be >= 1")

    args._rederived: set[str] = set()
    notes: list[str] = []
    if args.recompute_slice_tokens:
        notes += refresh_slice_tokens(args)
    if args.live_chain:
        notes += refresh_from_chain(args)

    prices = load_prices()
    r = model(args, prices)

    if args.json:
        print(json.dumps(r.__dict__, indent=2))
        return 0

    print()
    print("Fugal miner economics -- no paid API calls, no spend")
    print(f"pinned chain snapshot: block {CHAIN_SNAPSHOT_BLOCK}, {CHAIN_SNAPSHOT_DATE}")
    print()
    if notes:
        print("RE-DERIVED THIS RUN")
        for n in notes:
            print(f"  {n}")
        print()
    print_inputs(args)
    print_result(r, args)
    print_slippage(args)
    if args.sweep:
        print_sweep(args, prices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
