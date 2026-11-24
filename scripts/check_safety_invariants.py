"""Static checks for repository safety invariants that must never regress."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AXON_PROTOCOL_FILES = (
    ROOT / "neurons" / "miner.py",
    ROOT / "fugal_subnet" / "protocol.py",
)
_GUARDED = {"fugal_subnet", "neurons"}
# The grading contract. Any byte of graders.py is a consensus change: the hash
# is in every validator's epoch commitment and in the consensus digest, and the
# miner image grades with the file it was built from. History:
#   895809de...  v1, and the v2 image build-v2026.11.10 (never activated on finney)
#   319340e8...  2026-11-10, before the v2 flip (v0.3.1): letter_mcq reads an explicit
#                answer marker (a capital standing alone), then a line that is
#                only the letter, then a marked-up letter opening the reply,
#                before the 40-character fallback. Measured on 1,703 real
#                replies from the 19 v2 models (INVARIANTS, I4).
GRADER_SHA256 = (
    "319340e8aeaac5fedf2b488c94ad11c1e79e0606fcfc5e002552a60a4108a9d9"
)
# The v3 code grading contract (exec_stdio, exec_func), pinned for the same
# reason as graders.py: a byte change changes which code answers pass, so it is
# a consensus change and a new grader version. graders.py itself is untouched;
# protocol v3 adds this file beside it. History:
#   92bceddc...  2026-11-16, first version (docs/decisions/CODE_GRADER_V3.md):
#                seccomp-filtered child, CPU-time limits, token compare.
#   8ed4730b...  2026-11-16, v3 review finding 5: setpriority/ioprio_set/
#                sched_* and fchmod/fchown denied for any target; new threads
#                get 8 MiB stacks, two malloc arenas and a pinned CPU count
#                (64+ threads, same on every TD; the main thread's stack is
#                unchanged); wall backstop max(3x CPU + 2 s, 20 s) per run and
#                60 s per question. CPU limits and the comparison are unchanged.
CODE_GRADER_SHA256 = (
    "8ed4730b532b026ea4ae41a2777a634e6b907f04d143bf9286740406cb6f9265"
)
# The v3 shared system prompt (config.HARNESS_SYSTEM_PROMPT_V3): every v3 model
# call carries it, so a byte change moves every v3 grade and the token means.
# Pinned like the graders; changing it is a deliberate act — update the hash in
# the same commit, say why, and change fugal-core's text with it. History:
#   7b300e49...  2026-11-16, first version: fugal-core's house prompt with
#                "a requested answer format wins" (DRAFT, review before the v3
#                image is frozen).
SYSTEM_PROMPT_V3_SHA256 = (
    "7b300e491b7a5d8cb9629b8799115c52983ddfb93f4981b145f85ce3f31591d1"
)
# The v3 pool's two hand-built data files. Each decides which questions are in
# the pool (the manifest's hashes cover the result, and this pin makes an edit a
# reviewed act). Changing either: rebuild pool_manifest_v3.json, update the pin
# in the same commit, and say why.
#   exclusions_v3.json  RouterArena overlap, its LeetCode slugs, the all-fail rule
#   code_keep_v3.json   the calibrated code problems and tests
#                       (scripts/build_code_keep_v3.py)
POOL_V3_DATA_SHA256 = {
    "exclusions_v3.json": "ee346f7f486372351336a6cfb23143e927a29222d47c1d41562b9de5dc1cf26e",
    "code_keep_v3.json": "a02c71940b7765b45fcb315153f0686aeaa78d56e6180afa76aafa604a20f4bc",
}
# Protocol v5 (RouterArena) grading contract: fugal_subnet/graders_arena.py, RouterArena's
# metrics ported (parity: scripts/parity_graders_arena.py). Pinned like graders_code.py.
#   (first version) 2026-11-24, lane P.
ARENA_GRADER_SHA256 = "PENDING"
# The v5 pool's pinned data (scripts/build_pool_v5.py writes the manifest and the exclusions;
# arena_prompts.json is RouterArena's prompt templates, verbatim). Rebuild and re-pin together.
POOL_V5_DATA_SHA256 = {
    "arena_prompts.json": "PENDING",
    "exclusions_routerarena.json": "PENDING",
    "pool_manifest_v5.json": "PENDING",
}
# The price table is the consensus cost denominator: every validator prices
# every proof against it, so an unreviewed edit silently re-scores the whole
# subnet. Pinned for the same reason graders.py is. Changing prices is a
# deliberate act — update this hash in the same commit and say why.
PINNED_PRICE_TABLE_SHA256 = (
    "7bc8e332a43833bb4eaccef21e784f57ffde9e49e343275f822f35d9a6d19b27"
)
# Protocol v2's price table (data/models_v2.json; identical copy in
# fugal_subnet/routing_data/). 18 models (x-ai/grok-4.7 dropped 2026-11-05:
# it exceeded the 16,384 answer budget and its provider accepts effort only,
# so the cap could not be enforced; the other 18 rows are unchanged), prices from the public OpenRouter
# catalogue of 2026-11-04, one pinned `reasoning` object each. Five rates were
# then re-pinned from MEASURED effective prices (the v2 token measurement of
# 2026-11-04, 757 metered calls): OpenRouter routed those models to endpoints
# priced differently from the catalogue default, billed/table = glm-5.3-flash
# 2.8594, glm-5.3 0.4858, deepseek-v4-pro-0813 0.7979, minimax-m3.1-flash
# 1.1522, qwen3.8-27b 1.0857; input and output scaled by that ratio. Consensus
# from V2_ACTIVATION_BLOCK; a change re-scores every v2 miner.
# 2026-11-09: anthropic/claude-fable-5.2 added (19 models) at the public
# OpenRouter catalogue price ($8 / $40 per million tokens), with fable-5.1's
# reasoning setting; fable-5.1 stays the anchor (CHAMPION_ANCHOR_MODEL). The
# other 18 rows are unchanged.
PINNED_PRICE_TABLE_V2_SHA256 = (
    "9b18e3fbc786c3aea90966b69984717ead66c27f4bf4a98a388a9674f1be2aaa"
)
# Protocol v3's menu (data/models_v3.json; identical copy in
# fugal_subnet/routing_data/). 2026-11-16, user-approved: 23 models, 53
# `model@effort` entries = the union of the arena.ai Text, Agent and WebDev
# Pareto frontiers available on OpenRouter (Granite 4.2 8B for 4.1) plus Qwen4
# Max, Muse Spark 1.3 and Muse Wonder; each model at its frontier effort(s)
# plus low and high. Every entry pins its OpenRouter hosts (same quantization,
# no flex tiers, no fallbacks) and is priced at the dearer pinned host from
# OpenRouter's /endpoints of 2026-11-16. Gemini 4 Carbon pins Vertex global +
# us (Google AI Studio capped answers at 32k), so $2.2 / $11. Live-checked
# 2026-11-16 (every entry answered; ~/projects/fugal-v2-handoff/v3-prep/
# measurements/livecheck-2026-11-16.jsonl).
# TODO(v3 release): `max_tokens` per entry is PROVISIONAL = min(131072, the
# smallest answer limit of its pinned hosts). The final budgets are set from
# the 128k measurement of 2026-11-16 (results-128k.jsonl) before the release;
# re-pin this hash then.
PINNED_PRICE_TABLE_V3_SHA256 = (
    "1494ad3f597f77f20a4bafb0c7401a23c357a4afb140fcab895d49bb6b6881f2"
)
# Protocol v3's token manifest (the cost denominator), status `reviewed`
# (approved by the owner 2026-11-17; live miners require a reviewed manifest).
# 2026-11-16: 49 rows MEASURED in v3 (10 calls each, the same 10 non-code v3
# questions for every entry, through the v3 harness path: system prompt,
# entry budget, pinned hosts; $3.86, rows in ~/projects/fugal-v2-handoff/
# v3-prep/measurements/tokens-2026-11-16.jsonl, none used its whole budget);
# 3 rows kept from v2 (same model and setting, within 2x of the v3 sample);
# mimo-v2.6-pro replaced by its v3 sample (v2 was 5.8x longer); gemma-4-31b
# estimated (9 of 10 calls; one question's upstream kept failing). Rule:
# scripts/build_token_manifest_v3.py. Code questions are not in the sample;
# v3 exploration's attested token medians replace rows at the monthly menu
# update. Pinned like the menu: a change re-prices every v3 miner's routing.
PINNED_TOKEN_MANIFEST_V3_SHA256 = (
    "17d362de9f5675bf3a677c5dae5fa7dc1a4363ae966e22bc73b8a91691e2526a"
)
# Protocol v3 menu B (data/models_v3b.json and its manifest; identical copies
# in fugal_subnet/routing_data/), from config.menu_b_block. 2026-11-17,
# owner-approved for v0.6.0: menu A byte for byte (the pins above unchanged),
# plus 7 entries appended: openai/gpt-6.2-astra@{low,high,max} at $8/$40 and
# openai/gpt-6.2-sol@{low,medium,high,max} at $2/$10 per M tokens, max_tokens
# 128000, pinned to openai then azure with no fallbacks. The manifest is menu
# A's reviewed rows plus one MEASURED row per added entry (10 calls each on the
# v3 request path, all served by OpenAI, $0.72; data/observations/
# tokens_v3b_new_entries_2026-11-17.jsonl). Built by scripts/build_menu_v3b.py,
# which reproduces both files byte for byte. Each check below also holds menu
# B's first 53 rows equal to menu A's, which is what keeps A heads valid.
PINNED_PRICE_TABLE_V3B_SHA256 = (
    "28903ad0054bbc33ddc8d4c50e9713010c370e7e4551bb1075e93dbf884891ff"
)
PINNED_TOKEN_MANIFEST_V3B_SHA256 = (
    "57e23781338ab466a3fbdf556c03946f86f90e89e8a1cdea3c51ea26dc75040f"
)
# Protocol v4's menu seed (data/menu_v4_seed.json; identical copy in
# fugal_subnet/routing_data/): the menu of the first v4 epoch
# (docs/decisions/LIVE_MENU_V4.md, section 4.5). No release schedules v4 yet,
# and nothing before v4 reads this file. Pinned 2026-11-18, the first build:
# 59 of v3 menu B's 60 entries with their model, reasoning and host order
# unchanged (xiaomi/mimo-v2.5-pro is left out: its one pinned host is not in
# the listing); prices in picodollars per token from OpenRouter's public
# /endpoints pages saved 2026-11-18T17:29Z (data/observations/
# openrouter_endpoints_2026-11-18.json, sha256 de00e0e9...aedd0522), the
# dearer listed pinned host; token means in milli-tokens from the pooled pairs
# of the published v3 reveals e00026466..e00026484 (data/observations/
# menu_v4_seed_pairs_e00026466-e00026484.json, sha256 d9dd92c8...99eb16af),
# weighted to v3's scored mix, for the 48 entries with at least 30 pairs, and
# the v3 menu B manifest row for the other 11. Built by
# scripts/build_menu_v4_seed.py, which reproduces the file byte for byte from
# those two inputs (tests/test_menu_v4_seed.py). PROVISIONAL until the v4
# release: the prices are to be reviewed by the owner, and the seed is rebuilt
# and re-pinned when the first signed lab list and the v4 scored mix exist. A
# change after v4 activates moves the first epoch's menu hash, and with it
# every proof's namespace.
# Re-pinned 2026-11-18 (was 6d056b45...61b6c347, never released): the state
# document gained the `coldkey_cooldowns` map, empty in the seed; no entry,
# price or token mean changed (the file differs by that one line).
# Re-pinned 2026-11-18 (was 936123f7...2b62d31f, never released): the builder
# now takes each entry's answer limit from the listing by the rule the step
# applies to an entry on the menu (menu_rules.answer_limit over the pinned
# hosts the listing shows), so that every seed entry passes the compiled
# rules against the listing it was built from. Two entries moved,
# google/gemini-4-carbon@high and @low, from v3's 65,536 to 131,072: their one
# listed pinned host (google-vertex/global) states 943,718 in the snapshot
# and the other (google-vertex/us) is not listed. The other 57 entries'
# limits already equalled the rule's. No price or token mean changed (the
# file differs by those two lines).
# Re-pinned 2026-11-19 (was d4732e2c...498bd4f7, never released): an entry
# gained `explore_pairs`, its pooled exploration pairs in the window (section
# 8's n_e, which exploration's coverage part reads). Each seed entry's is the
# sum of its pooled pairs in the same committed pairs file the token means
# come from. No price, limit or token mean changed (each of the 59 entry lines
# differs by that one field).
# Re-pinned 2026-11-19 (was 1a39bba3...89a5351b, never released): the token
# means are weighted to the pinned v4 question mix (epoch_plan.mix_v4, W8:
# hard 30, code 15, math 15, knowledge 15, instruction 15, easy_math 5,
# everyday 5) instead of v3's, as the derivation step's production
# parameters weight them (epoch_plan.menu_step_params_v4). Only mean_in and
# mean_out moved; ids, prices, limits, token_pairs and explore_pairs are
# unchanged.
# Re-pinned 2026-11-19 (was 849ff539...1c362076, never released): an entry
# gained its pick-cost table (`pick`, W13: SEAT_MINIMAL_V4.md 3.4, the
# earlier seat design's 8.2), what a call costs on the questions routers
# send it, which v4 picks, the product and the real-picks check price at.
# Each seed entry's is computed by the builder from the graded scored cells
# of the 44 published v3 reveals e00026466..e00026509 (data/observations/
# menu_v4_seed_picks_e00026466-e00026509.json, sha256 0071ddfd...6658a3f4):
# per entry and v4 category, the cells' coldkey-median completion tokens
# (rounded half up, held to the answer limit the call was sent with),
# shrunk toward the entry's call means with PICK_COST_PRIOR_CELLS 30 and 20;
# the prompt side is the call mean (a v3 reveal publishes no prompt tokens
# for a cell). The entries with no graded cell carry no table. No price,
# limit, token mean, token_pairs or explore_pairs changed.
# Re-pinned 2026-11-19 (was 86d0bf45...4bbabf28, never released): an entry
# gained `measured_rate`, q_e of SEAT_MINIMAL_V4.md 4.1, the rate a head with
# no row for a new entry is scored at during the entry's allowance. It is
# fixed when the entry joins, from its proposal's measurement. A seed entry
# had no proposal measurement and no allowance (`scored_from` = `joined_epoch`
# = 0, unchanged), so the value is never read; each is (0 + 1) / (0 + 2) =
# 5000. No other field changed (each of the 59 entry lines differs by that one
# field).
# Re-pinned 2026-11-21 (was 724c018b...5cad3363, never released): the menu
# document drops the proposer-measurement pipeline's two lists, `pending` and
# `queue` (both empty in the seed), for `provisional`, the records of entries
# that joined by a chain proposal and await their gates (FAST_CROWN_V4.md
# section 18), empty in the seed. No entry, price, limit or token mean changed.
PINNED_MENU_V4_SEED_SHA256 = (
    "a9fc62753beced7d8771ccb4cc5b63960ea240be9919166eb7fa660b31045fdb"
)
# Protocol v4's question mix (data/pool_mix_v4.json; identical copy in
# fugal_subnet/routing_data/): the share of the scored slice and of
# exploration each category and each question set gets in a v4 epoch
# (docs/decisions/LIVE_MENU_V4.md, section 10; epoch_plan.mix_v4). No release
# schedules v4 yet, and nothing before v4 reads this file. Re-pinned
# 2026-11-19 for a 200-question slice (docs/decisions/SEAT_MINIMAL_V4.md,
# section 8: the scored slice goes from 300 to 200 with the real-picks check;
# was 5ccc39a8...0b64d9988, the first build at 300). The targets are those of
# the first build, which do not depend on the slice: hard 30, code 15, math 15, knowledge 15,
# instruction 15, easy_math 5, everyday 5 per cent, each proportional to the
# outcome variance p(1 - p) of a uniformly drawn seed-menu entry on the
# category's exploration pairs (floor 5%, rounded to 5% by largest
# remainder), from the 9,729 placed exploration cells of the published v3
# reveals e00026466..e00026489 (data/observations/
# mix_v4_cells_e00026466-e00026489.json, sha256 4659bd52...aa4cc59f), the
# epochs and the targets of the W0 replay of 2026-11-18. Inside a category
# sets split by their own p(1 - p) x pool size (the category's value for a
# set with under 100 cells), none above pool size / (200 x 96) of the slice
# (at 200 no set reaches its cap; at 300 every hard set but aime and two
# code sets did), in millionths of a 200-question slice; pool sizes from the
# v3 pool manifest, v4's pool. Built by scripts/build_mix_v4.py, which
# reproduces the file byte for byte from those inputs
# (tests/test_pool_mix_v4.py). A change after v4 activates changes every v4
# epoch's questions; another slice size needs a rebuild with --slice and a
# new pin.
PINNED_POOL_MIX_V4_SHA256 = (
    "e21461379d2eb72e97e7cca81a6797ad154a34e5693cb662600a0aa63d2a3bfb"
)
# Protocol v4's published-set seed (data/published_v4.json; identical copy in
# fugal_subnet/routing_data/): per question set of the v3 pool (v4's pool),
# the questions no named reveal published an outcome on, which the
# memorisation check reads as unseen and the fresh draw spends
# (docs/decisions/SEAT_MINIMAL_V4.md, section 6; memo_v4.seed_published).
# Pinned 2026-11-19 (package S5): from the 321 public reveals e00026189..e00026509
# (v1 160, v2 117, v3 44: every reveal the store holds for netuid 70 up to
# the S2-S6 measurements), 32,106 of the pool's 48,320 questions were
# published and 16,214 were not (data/observations/
# published_v4_e00026189-e00026509.json, sha256 12f0af44...cd6058, each reveal
# with its sha256). Built by scripts/build_published_v4.py, which reproduces
# the file byte for byte from that file (tests/test_memo_v4.py). Nothing
# before v4 reads this file.
#
# The seed is a starting point, not the set at the activation: validators and
# TDs bring the published set forward from it by folding the root's reveals
# the chain names at each boundary after its `through` (memo_sync.ChainSet),
# as they do after the activation, so the seed need not be observed through
# the reveal just before the block, which would not exist when the image is
# built. The release gate (`--release-v4`) demands it ships, matches this pin,
# and is observed no more than PUBLISHED_V4_SEED_MAX_AGE_EPOCHS epochs before
# the activation epoch (and through at most that epoch minus memo_v4.READ_LAG),
# which bounds how many reveals a node folds to catch up: at the block every
# TD and every validator without state replays one pinned chain read and
# reads one root reveal (about 1.44 MB of a v3 reveal) per epoch since the
# seed, so 150 epochs is about 216 MB and 150 archive reads.
PINNED_PUBLISHED_V4_SHA256 = (
    "2aa4da74fc51859d7a50e727a61226f46e5f282719cf44f8af1ee53be75851eb"
)
PUBLISHED_V4_SEED_MAX_AGE_EPOCHS = 150

# Google's EK/AK CA root, vendored at fugal_subnet/tee/roots/. Verifying a TPM
# quote means trusting this CA in addition to Intel's — see docs/INVARIANTS.md.
# data/reference_frame.json is the bootstrap reference frame: decayed
# exploration counts every live validator starts from, so the frontier every
# miner is scored against is valued from it. Consensus state like models.json.
# None means the file must not exist yet; a builder (build_frame_from_calls.py,
# or build_bootstrap_frame.py from reveals) prints the digest to pin here in
# the same commit, saying why.
# Pinned 2026-11-03: built by scripts/build_frame_from_calls.py from
# data/observations/bootstrap_frame_calls_v1.jsonl (17 models x up to 300 direct,
# logged, graded calls; mistral stopped at 80 after upstream 429s and is not on
# the hull). `build_frame_from_calls.py --build-only` reproduces this digest
# (one undecayed batch, the default; tests/test_frame_from_calls.py).
PINNED_BOOTSTRAP_FRAME_SHA256: str | None = (
    "c31417505fe7a80e253a11261665fa5f7e63343d28f963f3ed96da2dd521a1c0"
)
PINNED_TPM_ROOT_SHA256 = (
    "594759594b9b61524f6c8ef668f7177f7b9066bc0674f2f49fe9e052be78beb2"
)
# Protocol v4's signed lists (docs/decisions/LIVE_MENU_V4.md, sections 6 and
# 13.4; fugal_subnet/menu_rules.py). Nothing before a v4 activation block
# reads them.
#
# The list signers: fugal_subnet/signed_lists/allowed_signers, the keys whose
# signature makes a lab and host allowlist (or a device list) trusted. A trust
# root like the TPM root above, and byte-identical to
# deploy/keys/allowed_signers, so the release signers and the list signers
# cannot drift apart. Changing a signer means updating this pin and
# menu_rules.LIST_SIGNERS_SHA256 in the same commit, and saying why. History:
#   5a6bc2ef...  2026-11-18, first version: the two release keys
#                (release@fugalai.com, release2@fugalai.com), each listed for
#                the namespaces `git` and `fugal-list@fugalai.com`.
PINNED_LIST_SIGNERS_SHA256 = (
    "5a6bc2ef0b3fd93d193f4162791b62364f10f2c48025a3b32c05903efde41ead"
)
# The lists a release ships under fugal_subnet/signed_lists/, by file name.
# None means the file must not exist yet. A shipped list is consensus data:
# it decides which entries every node accepts. It must verify under the
# signers above and match its pin; `scripts/signed_list.py verify` prints the
# digest to pin here in the commit that adds the signed file.
# History:
#   menu_allowlist.json e3f95a7b...  2026-11-22, v0.7.0: the first lab and host
#       list, serial 1, signed by release@fugalai.com: 19 labs, 26 hosts, 8
#       first-party marks (list format 2), the reviewed draft's content
#       unchanged (`signed_list.py prepare --serial 1`). Every entry of the
#       pinned v4 seed passes it, so it is adopted at the first v4 step.
PINNED_SIGNED_LISTS_SHA256: dict[str, str | None] = {
    "menu_allowlist.json": "e3f95a7b8daac512bda812146b05eb3f38e9b19759088d8b0eaec7a135166a39",
    "device_list.json": None,
}


def python_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.py")
        if not any(part in {".git", ".venv", "venv"} for part in path.parts)
    ]


def check_np_load_calls(errors: list[str]) -> None:
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_np_load = (
                isinstance(func, ast.Attribute)
                and func.attr == "load"
                and isinstance(func.value, ast.Name)
                and func.value.id == "np"
            )
            if not is_np_load:
                continue
            allow_pickle = next(
                (kw.value for kw in node.keywords if kw.arg == "allow_pickle"),
                None,
            )
            if not (
                isinstance(allow_pickle, ast.Constant)
                and allow_pickle.value is False
            ):
                relative = path.relative_to(ROOT)
                errors.append(
                    f"{relative}:{node.lineno}: np.load() must use allow_pickle=False"
                )


def check_runtime_annotations(errors: list[str]) -> None:
    for path in AXON_PROTOCOL_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
                and any(alias.name == "annotations" for alias in node.names)
            ):
                relative = path.relative_to(ROOT)
                errors.append(
                    f"{relative}:{node.lineno}: deferred annotations break bt.Axon.attach()"
                )


def check_deserialize_contract(errors: list[str]) -> None:
    """Every Axon-attached Synapse must define deserialize() returning self.

    Checked against whatever Synapse subclasses protocol.py actually declares,
    rather than a hardcoded list — a hardcoded list silently stops covering a
    synapse that gets renamed, and silently fails on one that gets deleted.
    """
    path = ROOT / "fugal_subnet" / "protocol.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    synapse_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(b, ast.Attribute) and b.attr == "Synapse")
            or (isinstance(b, ast.Name) and b.id == "Synapse")
            for b in node.bases
        )
    ]
    if not synapse_names:
        errors.append("fugal_subnet/protocol.py: no bt.Synapse subclass found")
        return

    for synapse_name in synapse_names:
        found = False
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == synapse_name:
                found = True
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "deserialize":
                        returns = [child for child in ast.walk(item) if isinstance(child, ast.Return)]
                        if not (len(returns) == 1 and isinstance(returns[0].value, ast.Name) and returns[0].value.id == "self"):
                            errors.append(
                                f"fugal_subnet/protocol.py: {synapse_name}.deserialize() must return only self"
                            )
                        break
                else:
                    errors.append(
                        f"fugal_subnet/protocol.py: {synapse_name}.deserialize() was not found"
                    )
        if not found:
            errors.append(f"fugal_subnet/protocol.py: {synapse_name} class was not found")


def check_pinned_grader(errors: list[str]) -> None:
    path = ROOT / "fugal_subnet" / "graders.py"
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != GRADER_SHA256:
        errors.append(
            f"fugal_subnet/graders.py: grader hash {actual[:16]}... does not match "
            f"the pin {GRADER_SHA256[:16]}... — "
            "a grader change is a consensus break"
        )


def check_pinned_code_grader(errors: list[str]) -> None:
    path = ROOT / "fugal_subnet" / "graders_code.py"
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != CODE_GRADER_SHA256:
        errors.append(
            f"fugal_subnet/graders_code.py: code grader hash {actual[:16]}... does not "
            f"match the pin {CODE_GRADER_SHA256[:16]}... — a code grader change is a "
            "consensus break"
        )


def check_code_questions_are_graded(errors: list[str], root: Path | None = None) -> None:
    """A v3 pool manifest that holds code questions (exec_stdio / exec_func)
    ships only with a harness that grades them with graders_code and a working
    sandbox (fugal_subnet/code_gate.py; v3 review, 2026-11-16, finding 7).
    Otherwise every code answer grades 0 and is still paid for."""
    import sys

    root = root or ROOT
    manifest = root / "fugal_subnet" / "benchmarks" / "pool_manifest_v3.json"
    if not manifest.is_file():
        return
    text = manifest.read_text(encoding="utf-8")
    if '"exec_stdio"' not in text and '"exec_func"' not in text:
        return
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fugal_subnet import code_gate, graders_code

    ok, why = graders_code.sandbox_selftest()
    if not ok:
        errors.append(f"{manifest.relative_to(root)}: code questions, but the code sandbox "
                      f"self-test fails here: {why}")
    ok, why = code_gate.harness_grades_code()
    if not ok:
        errors.append(f"{manifest.relative_to(root)}: code questions, but {why}")


def check_pinned_system_prompt_v3(errors: list[str]) -> None:
    from fugal_subnet import config

    text = config.HARNESS_SYSTEM_PROMPT_V3
    if not text:
        errors.append("fugal_subnet/config.py: HARNESS_SYSTEM_PROMPT_V3 is empty; the v3 "
                      "image must pin the shared system prompt")
        return
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual != SYSTEM_PROMPT_V3_SHA256:
        errors.append(
            f"fugal_subnet/config.py: HARNESS_SYSTEM_PROMPT_V3 hash {actual[:16]}... does "
            f"not match the pin {SYSTEM_PROMPT_V3_SHA256[:16]}... — the shared system "
            "prompt is part of every v3 grade; change the pin deliberately")


def check_v3_release_pins(errors: list[str]) -> None:
    """A release that schedules protocol v3 on finney must pin everything a v3
    epoch reads that is not in the repository: the embeddings file (its hash, and
    where the image build and validators fetch it from) and the shared system
    prompt. Without them a v3 miner cannot route and a validator cannot compute
    odds, so the flip would stall every node at the activation block."""
    from fugal_subnet import config

    if config.FINNEY_V3_ACTIVATION_BLOCK is None:
        return
    missing = [name for name in ("EMBEDDINGS_SHA256", "EMBEDDINGS_URL",
                                 "HARNESS_SYSTEM_PROMPT_V3")
               if not getattr(config, name, None)]
    if missing:
        errors.append("fugal_subnet/config.py: FINNEY_V3_ACTIVATION_BLOCK is set but "
                      + ", ".join(missing) + " is not pinned; the v3 release must pin it")


def check_pool_v3_data_pinned(errors: list[str]) -> None:
    for name, pin in POOL_V3_DATA_SHA256.items():
        path = ROOT / "fugal_subnet" / "benchmarks" / name
        if not path.exists():
            errors.append(f"fugal_subnet/benchmarks/{name} is missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != pin:
            errors.append(f"fugal_subnet/benchmarks/{name}: hash {actual[:16]}... does not match "
                          f"the pin {pin[:16]}... (it decides which questions are in the v3 pool)")


def check_pinned_arena_grader(errors: list[str]) -> None:
    path = ROOT / "fugal_subnet" / "graders_arena.py"
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != ARENA_GRADER_SHA256:
        errors.append(
            f"fugal_subnet/graders_arena.py: v5 grader hash {actual[:16]}... does not match "
            f"the pin {ARENA_GRADER_SHA256[:16]}... — a v5 grader change is a consensus break")


def check_pool_v5_data_pinned(errors: list[str]) -> None:
    for name, pin in POOL_V5_DATA_SHA256.items():
        path = ROOT / "fugal_subnet" / "benchmarks" / name
        if not path.exists():
            errors.append(f"fugal_subnet/benchmarks/{name} is missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != pin:
            errors.append(f"fugal_subnet/benchmarks/{name}: hash {actual[:16]}... does not match "
                          f"the pin {pin[:16]}... (it decides the v5 exam pool or its prompts)")


def check_epoch_id_single_source(errors: list[str]) -> None:
    """Both neurons must derive epoch identity from the same helper.

    The nonce is sha256(f"{epoch_id}:{block_hash}"), so if the miner and the
    validator format the identifier differently they derive different nonces,
    select different question slices, and every proof fails the nonce check.
    That shipped: the miner used f"e_{block_hash[:16]}" and the validator used
    f"e{epoch_index:08d}", their slices overlapped 45/300, and the subnet could
    not have set weights.

    A behavioural test cannot catch this — a test naturally derives the id once
    and passes it to both sides, which is exactly the divergence it needs to
    detect. So the invariant is structural: neither neuron may build an
    epoch_id itself.
    """
    for name in ("miner.py", "validator.py"):
        path = ROOT / "neurons" / name
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        if "epoch_id_for_block" not in source:
            errors.append(
                f"neurons/{name}: must derive epoch identity from "
                "slicer.epoch_id_for_block, the single source of truth"
            )
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "epoch_id" not in targets:
                continue
            # A locally constructed string is the bug; a call is the fix.
            if isinstance(node.value, (ast.JoinedStr, ast.Constant, ast.BinOp)):
                errors.append(
                    f"neurons/{name}:{node.lineno}: epoch_id is built locally. "
                    "Call slicer.epoch_id_for_block instead — a second "
                    "formatting of the epoch id is a consensus break"
                )


def check_tpm_dependency_pin(errors: list[str]) -> None:
    """`cryptography` is pinned in two extras and both copies must agree.

    It is in `dev` as well as `tee` because the TPM tests need it, but writing
    `fugal-subnet[tee]` would drag dcap-qvl into the dev environment and make
    the `test` and `tee-verifier` CI jobs identical — they differ on purpose.
    The cost of that choice is a duplicated version string, and this is what
    keeps it from drifting.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pins = set(re.findall(r'"cryptography==([0-9][^"]*)"', text))
    if not pins:
        errors.append(
            "pyproject.toml no longer pins cryptography — the TPM verifier is a "
            "consensus rule for --live validators on GCP and its library is "
            "pinned exactly, like dcap-qvl"
        )
    elif len(pins) > 1:
        errors.append(
            f"pyproject.toml pins cryptography at {sorted(pins)} in different "
            "places — validators would disagree on TPM verification depending on "
            "which extra they installed"
        )


def check_tpm_trust_anchor(errors: list[str]) -> None:
    """The pinned Google EK/AK root is a trust anchor, so it gets a hash gate.

    A swapped anchor does not fail loudly on its own: every forged TPM quote
    signed under the substituted CA would simply verify. That is the same
    failure shape as an edited grader, so it gets the same treatment.
    """
    path = ROOT / "fugal_subnet" / "tee" / "roots" / "google_ek_ak_root.der"
    if not path.exists():
        errors.append(
            "fugal_subnet/tee/roots/google_ek_ak_root.der is missing — TPM quotes "
            "on GCP cannot be chained to any trust anchor"
        )
        return

    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != PINNED_TPM_ROOT_SHA256:
        errors.append(
            f"google_ek_ak_root.der hash {actual[:16]}... does not match pin "
            f"{PINNED_TPM_ROOT_SHA256[:16]}... — this is the trust anchor for "
            "every TPM quote the subnet accepts; changing it accepts a different "
            "CA's attestations"
        )

    # The module's own constant must agree, or the runtime check is checking
    # something the CI gate never saw.
    tpm_src = (ROOT / "fugal_subnet" / "tee" / "tpm.py").read_text(encoding="utf-8")
    if f'_ROOT_SHA256 = "{PINNED_TPM_ROOT_SHA256}"' not in tpm_src:
        errors.append(
            "fugal_subnet/tee/tpm.py:_ROOT_SHA256 disagrees with the pin in this "
            "script — two pins that can drift are worse than one"
        )


def check_signed_lists(errors: list[str], root: Path | None = None,
                       signers_pin: str | None = None,
                       list_pins: dict[str, str | None] | None = None) -> None:
    """Protocol v4's signed lists: pinned signers, signed and pinned lists,
    and a draft that is never a list.

    - The list signers shipped in the package are byte-identical to the
      release signers in deploy/keys/, match their pin (here and in
      menu_rules), read one way only, and list at least one key for lists.
    - A list under fugal_subnet/signed_lists/ ships with its signature,
      verifies under those signers with menu_rules.verify_signed_list (the
      list namespace, a listed key, a serial of at least 1), and matches its
      pin. Nothing else may sit in that directory.
    - The draft in data/ carries serial 0, has no signature beside it, and is
      removed once the signed list ships.
    - No code under fugal_subnet/ or neurons/ reads a list without checking
      its signature (menu_rules.parse_unsigned_list is for tools).

    `root`, `signers_pin` and `list_pins` exist for the tests, which build a
    tree with throwaway keys.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from fugal_subnet import menu_rules as mr

    real = root is None
    root = ROOT if root is None else Path(root)
    signers_pin = PINNED_LIST_SIGNERS_SHA256 if signers_pin is None else signers_pin
    list_pins = PINNED_SIGNED_LISTS_SHA256 if list_pins is None else list_pins
    lists_dir = root / "fugal_subnet" / "signed_lists"
    rules = root / "fugal_subnet" / "menu_rules.py"

    for base in (root / "fugal_subnet", root / "neurons"):
        for path in sorted(base.rglob("*.py")) if base.is_dir() else ():
            if path != rules and "parse_unsigned_list" in path.read_text(encoding="utf-8"):
                errors.append(
                    f"{path.relative_to(root)}: uses parse_unsigned_list, which checks no "
                    "signature; a neuron reads a list through verify_signed_list or shipped_list"
                )

    released, shipped = root / "deploy" / "keys" / "allowed_signers", lists_dir / "allowed_signers"
    missing = [str(p.relative_to(root)) for p in (released, shipped) if not p.exists()]
    if missing:
        errors.append(f"{', '.join(missing)} missing: the list signers are a trust root and "
                      "ship in the package as a copy of the release signers")
        return
    signers = shipped.read_bytes()
    if signers != released.read_bytes():
        errors.append("deploy/keys/allowed_signers and fugal_subnet/signed_lists/allowed_signers "
                      "differ: the release signers and the list signers are one file")
        return
    actual = hashlib.sha256(signers).hexdigest()
    if actual != signers_pin:
        errors.append(f"fugal_subnet/signed_lists/allowed_signers: hash {actual[:16]}... does "
                      f"not match pin {signers_pin[:16]}... (a changed signer changes whose "
                      "lists every node accepts)")
        return
    if real and f'LIST_SIGNERS_SHA256 = "{signers_pin}"' not in rules.read_text(encoding="utf-8"):
        errors.append("fugal_subnet/menu_rules.py:LIST_SIGNERS_SHA256 disagrees with the pin "
                      "in this script")
    for number in mr.unread_signers_lines(signers):
        errors.append(
            f"deploy/keys/allowed_signers line {number}: not of the form `<principal> "
            'namespaces="<name>,..." ssh-ed25519 <key>`, the one form menu_rules reads; a line '
            "OpenSSH and menu_rules read differently makes the trust root depend on the reader"
        )
    if not mr.permitted_signers(signers):
        errors.append(f"deploy/keys/allowed_signers lists no key for {mr.LIST_NAMESPACE}: "
                      "no list could ever be accepted")
        return

    known = {"allowed_signers", "__pycache__"}
    for kind, name in mr.SHIPPED_LISTS.items():
        known |= {name, f"{name}.sig"}
        payload_path, sig_path = lists_dir / name, lists_dir / f"{name}.sig"
        where = f"fugal_subnet/signed_lists/{name}"
        if not payload_path.exists() and not sig_path.exists():
            if list_pins.get(name) is not None:
                errors.append(f"{where} is pinned but missing")
            continue
        if not sig_path.exists():
            errors.append(f"{where} ships without {name}.sig: an unsigned list is not a list")
            continue
        if not payload_path.exists():
            errors.append(f"{where}.sig is present without {name}")
            continue
        payload = payload_path.read_bytes()
        try:
            mr.verify_signed_list(payload, sig_path.read_bytes(), kind=kind, held_serial=0,
                                  signers=signers)
        except mr.ListRefused as e:
            errors.append(f"{where} does not verify as a signed list ({e})")
            continue
        digest, pin = hashlib.sha256(payload).hexdigest(), list_pins.get(name)
        if pin is None:
            errors.append(f"{where} ships but its pin is None; set the pin to {digest} in the "
                          "commit that adds the file, and say why")
        elif digest != pin:
            errors.append(f"{where}: hash {digest[:16]}... does not match pin {pin[:16]}... "
                          "(a shipped list decides which entries every node accepts)")
    for path in sorted(lists_dir.iterdir()):
        if path.name not in known:
            errors.append(f"fugal_subnet/signed_lists/{path.name}: not a file the package "
                          "reads; only the signers and the signed lists ship there")

    draft = root / "data" / "menu_allowlist.draft.json"
    if draft.with_name(draft.name + ".sig").exists():
        errors.append("data/menu_allowlist.draft.json.sig: a draft is never signed; "
                      "`scripts/signed_list.py prepare` writes the bytes to sign")
    if draft.exists():
        try:
            parsed = mr.parse_unsigned_list(draft.read_bytes(), mr.KIND_MENU_ALLOWLIST)
        except mr.ListRefused as e:
            errors.append(f"data/menu_allowlist.draft.json: {e}")
            return
        if parsed.serial != mr.DRAFT_SERIAL:
            errors.append(f"data/menu_allowlist.draft.json: a draft must carry serial "
                          f"{mr.DRAFT_SERIAL}, which no signature makes a list; it carries "
                          f"{parsed.serial}")
        elif (lists_dir / mr.SHIPPED_LISTS[mr.KIND_MENU_ALLOWLIST]).exists():
            errors.append("data/menu_allowlist.draft.json: remove the draft in the commit that "
                          "ships the signed list")


def check_collateral_endpoint_explicit(errors: list[str]) -> None:
    """Every DCAP collateral fetch must name its endpoint.

    `get_collateral_and_verify(quote)` with no url resolves, inside dcap-qvl,
    to Phala's PCCS. That is how every --live validator acquired a third-party
    availability and privacy dependency nobody chose, and how INVARIANTS came
    to record the collateral as coming from Intel: nothing in this repo named
    the endpoint, so nothing contradicted the assumption.

    Restoring the one-argument call would restore that silence, and it would
    look like a simplification in review. So it is checked rather than trusted:
    the endpoint is this project's decision and must appear in this project.
    """
    src = (ROOT / "fugal_subnet" / "tee" / "attestation.py").read_text(encoding="utf-8")

    # Both entry points, because the code moved from one to the other once
    # already. A check naming only the function that happens to be called today
    # stops protecting anything the moment someone switches back, and it does so
    # silently -- it keeps passing, having found nothing to inspect.
    seen = 0
    for fn, min_args in (("get_collateral_and_verify", 2), ("get_collateral", 2)):
        for call in re.findall(rf"(?<![\w.]){fn}\(([^)]*)\)", src):
            seen += 1
            args = [a.strip() for a in call.split(",") if a.strip()]
            if len(args) < min_args:
                errors.append(
                    f"fugal_subnet/tee/attestation.py calls {fn} with "
                    f"{len(args)} argument(s) ({call.strip()!r}) — the PCCS url "
                    "must be passed explicitly, or dcap-qvl silently substitutes "
                    "its own default and the collateral source becomes invisible"
                )
    if seen == 0:
        errors.append(
            "fugal_subnet/tee/attestation.py makes no recognised collateral call — "
            "this check inspects call sites by name, so a rename turns it into a "
            "no-op that still reports success; teach it the new name"
        )

    cfg = (ROOT / "fugal_subnet" / "config.py").read_text(encoding="utf-8")
    if "TEE_PCCS_URL" not in cfg:
        errors.append(
            "config.TEE_PCCS_URL is missing — the collateral endpoint is a "
            "consensus-path dependency and must be pinned by this project "
            "rather than inherited from dcap-qvl's default"
        )


def check_external_calls_bounded(errors: list[str]) -> None:
    """I6. No external party may stop a validator completing an epoch.

    I6 used to say "no MINER behaviour", and that wording is what let the hang
    through: the collateral fetch is a call to a THIRD PARTY, so no reading of
    the old invariant was violated while a stalled PCCS blocked every validator
    in the field simultaneously -- no weights, and no log line either, because
    the code that records the failure sits downstream of the block.

    The lesson is that the invariant was wrong, not merely unenforced. A
    timeout on the one call that happened to be unbounded fixes today; this
    check is what stops the NEXT network call added inside the epoch loop from
    recreating it, and there will be one.

    Two things are enforced:

    1. Every `_run_coro` call passes a `timeout`. The helper runs a coroutine
       from sync code and defaults to unbounded, which is correct for a helper
       and fatal for a call on the epoch path.
    2. Every blocking network primitive passes a `timeout`. Each of these
       defaults to "wait forever" if the argument is omitted, and the omission
       is invisible at the call site -- it looks like every other call.

    Deliberately a syntactic check on argument presence, not a claim about the
    VALUE. A budget can be wrong; an absent budget is unbounded, and only the
    second kind produces a subnet that stops setting weights.
    """
    # Split by whether the NAME alone is evidence. `urlopen` and
    # `create_connection` mean one thing in Python; `get` and `post` are also
    # dict and mapping methods, so those need a receiver that looks like an
    # HTTP client before they count.
    #
    # They were one set for a commit, gated as a whole on the receiver. That
    # silently disabled the two unambiguous names in their normal
    # module-qualified form -- `urllib.request.urlopen(url)` has receiver
    # `request`, `socket.create_connection(addr)` has receiver `socket`, and
    # neither is in the client list. Recall measured at 2 of 6 planted calls.
    # The fix for a false positive had quietly created false negatives, which
    # is worse: the check kept reporting success while inspecting less.
    unambiguous_apis = {"urlopen", "create_connection"}
    ambiguous_apis = {
        "get", "post", "put", "head", "delete", "patch", "request",
    }
    unbounded_apis = unambiguous_apis | ambiguous_apis
    # Bare `get`/`post` are far too common as method names, so an attribute
    # call only counts when its receiver looks like an HTTP client.
    #
    # `s` was in this set for one commit, for `s = requests.Session()`. It
    # matched `s.get("accuracy", ...)` on a plain dict in epoch_logger instead.
    # Dropped deliberately: the cost of a false positive here is not noise, it
    # is that somebody eventually silences the whole check, and then the real
    # unbounded call lands unremarked. Precision over recall.
    #
    # KNOWN GAP, stated rather than papered over, and MEASURED rather than
    # guessed at. Against six deliberately planted unbounded calls this catches
    # four: `requests.get(url)`, `urlopen(url)`, `urllib.request.urlopen(url)`
    # and `socket.create_connection(addr)`. It misses the two that bind a
    # client to a name first -- `sess = requests.Session(); sess.get(url)` and
    # `httpx.Client().get(url)` -- because that needs assignment tracking.
    #
    # Four of six is worth having and is not worth mistaking for six of six.
    # I6 claims every call leaving the process on the epoch path is bounded and
    # names this check as the enforcement, so the distance between the claim
    # and the check belongs in writing, next to the check.
    http_receivers = {"requests", "httpx", "session", "client", "http"}

    for path in python_files():
        rel = path.relative_to(ROOT)
        if rel.parts[0] not in {"fugal_subnet", "neurons"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            has_timeout = any(k.arg == "timeout" for k in node.keywords)
            if has_timeout:
                continue

            if isinstance(node.func, ast.Name):
                name, receiver = node.func.id, None
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                receiver = (
                    node.func.value.id
                    if isinstance(node.func.value, ast.Name) else None
                )
            else:
                continue

            if name == "_run_coro":
                errors.append(
                    f"{rel}:{node.lineno} calls _run_coro without a timeout — "
                    "the helper is unbounded by default, so this is a call that "
                    "can block a validator's epoch forever (I6)"
                )
                continue
            if name not in unbounded_apis:
                continue
            if name in ambiguous_apis and isinstance(node.func, ast.Attribute) and (
                receiver is None or receiver.lower() not in http_receivers
            ):
                continue
            errors.append(
                f"{rel}:{node.lineno} calls {name}() with no timeout — a "
                "blocking network call on the validator's path waits forever by "
                "default, and one stalled upstream then halts every validator "
                "at once, because the collection point is a deterministic block "
                "(I6)"
            )

    cfg = (ROOT / "fugal_subnet" / "config.py").read_text(encoding="utf-8")
    if "TEE_COLLATERAL_TIMEOUT" not in cfg:
        errors.append(
            "config.TEE_COLLATERAL_TIMEOUT is missing — the collateral fetch is "
            "the one external call inside the epoch loop, and its bound is what "
            "keeps a stalled PCCS from halting the subnet (I6)"
        )


def check_measured_registers_documented(errors: list[str]) -> None:
    src = (ROOT / "fugal_subnet" / "tee" / "attestation.py").read_text(encoding="utf-8")
    body = re.search(
        r"def measurement_id\(.*?\n(.*?)(?=\n(?:def |class |@))", src, re.S)
    if body is None:
        errors.append("could not locate measurement_id() to derive its register set (I8)")
        return
    # The registers the CODE actually hashes.
    actual = {m.lower() for m in re.findall(r"quote\.(mrtd|rtmr\d)", body.group(1))}
    if not actual:
        errors.append("measurement_id() names no registers — cannot check docs against it (I8)")
        return

    # LIMITATION, found by this check firing on INVARIANTS.md prose that quoted
    # a wrong register set as an example of what not to write: it cannot tell an
    # assertion from an illustration. Documentation discussing a wrong set
    # should describe it rather than format it as a claim. Loosening the pattern
    # to fix that would cost more than it buys.
    #
    # Only claims of MEMBERSHIP are checked. Docs discuss RTMR0 and RTMR3 at
    # length to explain why they are EXCLUDED, and that prose must not trip.
    # Two assertion shapes, both anchored on MRTD appearing with the register
    # list, which is what a membership claim looks like:
    #   "sha256(MRTD || RTMR1 || RTMR2)"   explicit formula
    #   "(MRTD, RTMR0-2)"                  range form
    claim_re = re.compile(
        r"\(\s*MRTD\s*(?:[,‖|]|and)\s*RTMR\s*(\d)\s*(?:[-–]\s*(\d))?"
        r"(?:\s*(?:[,‖|]|and)\s*RTMR\s*(\d))?[^)]*\)", re.I)

    for rel in ("docs/MINER_GUIDE.md", "docs/INVARIANTS.md", "docs/VALIDATOR_GUIDE.md",
                "docs/decisions/design-decisions.md", "AGENTS.md", "README.md"):
        path = ROOT / rel
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in claim_re.finditer(line):
                lo, hi, extra = m.group(1), m.group(2), m.group(3)
                claimed = {"mrtd"}
                if hi:                      # a range: RTMR0-2
                    claimed |= {f"rtmr{i}" for i in range(int(lo), int(hi) + 1)}
                else:
                    claimed.add(f"rtmr{lo}")
                    if extra:
                        claimed.add(f"rtmr{extra}")
                if claimed != actual:
                    why = (
                        " RTMR0 is host-chosen, so a reader who believes it is "
                        "measured expects the approved list to fork by instance "
                        "size, and it does not."
                        if "rtmr0" in claimed - actual else ""
                    )
                    errors.append(
                        f"{rel}:{lineno} says the base measurement is "
                        f"{sorted(claimed)} but measurement_id() hashes "
                        f"{sorted(actual)} — a miner reading this predicts the "
                        f"wrong rejection.{why} (I8)"
                    )


def check_one_accuracy_definition_on_the_consensus_path(errors: list[str]) -> None:
    """I1. Every validator must divide by the same denominator.

    Two accuracy definitions exist in this repo and they differ by ~7 points:

      - `proof.accuracy` = `n_correct / len(scored_results)` — every scored
        question is in the denominator. THIS is what scoring reads, via
        `_proof_to_head_score`.
      - `head_eval.evaluate_head` drops questions no model in the pool answered
        correctly ("carry no routing signal", head_eval.py:203) before dividing.

    Both are defensible. Only one is consensus, and `evaluate_head` is not on
    that path at all — verified: it has zero callers under `fugal_subnet/` or
    `neurons/`, and every reference lives in tests, scripts or comments.

    So this guards a hazard rather than a bug. The trap is the NAME: it is the
    function you would reach for to reason about head scoring, and it is not
    the scoring function. Someone did reach for it and got a number 7% high
    with no warning — a reference model scoring 1.065 against itself, which the
    shipped definitions make impossible. A docstring would not have helped;
    docstrings are read by people already looking at the function, and the
    failure mode is someone grepping for a scoring function and finding one
    with the right name. Wiring it in would move every miner's apparent
    accuracy by ~7 points while reading as a refactor.

    NOTE FOR ANY RENAME. This check hardcodes the string "evaluate_head". If
    the function is renamed — `evaluate_head_offline` and
    `training_head_score` have both been suggested, and the name IS the trap —
    the string must move in the same commit, or this guard silently stops
    guarding. Deferred for now only because the rename would break a teammate's
    in-flight experiment, not because it is wrong.
    """
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.parts[0] not in _GUARDED or rel.parts[-1] == "head_eval.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            hit = (
                isinstance(node, ast.ImportFrom)
                and any(a.name == "evaluate_head" for a in node.names)
            ) or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "evaluate_head"
            ) or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "evaluate_head"
            )
            if hit:
                errors.append(
                    f"{rel}:{node.lineno} puts head_eval.evaluate_head on the "
                    f"consensus path. It uses a DIFFERENT accuracy denominator "
                    f"from proof.accuracy -- it drops questions no model "
                    f"answered, which inflated a measured score from 0.994 to "
                    f"1.065. Scoring must read accuracy from the proof "
                    f"(_proof_to_head_score), so that every validator divides "
                    f"by the same thing (I1)"
                )


ROUTING_DATA_REQUIRED = (
    "models.json", "models_v2.json", "models_v3.json",
    "benchmark_tokens_v1.json", "benchmark_tokens_protocol_v2.json",
    "benchmark_tokens_protocol_v3.json",
    "models_v3b.json", "benchmark_tokens_protocol_v3b.json",
    "menu_v4_seed.json", "pool_mix_v4.json", "published_v4.json",
)


def check_routing_data_mirrors_data(errors: list[str]) -> None:
    """Every file under fugal_subnet/routing_data/ is byte-identical to its copy
    in data/ (CLAUDE.md rule 7).

    routing_data/ is what the neurons read and what the protocol identity
    hashes; data/ is the reviewed copy the pins are checked against. A
    difference means the pins check one file and consensus uses another.
    """
    mirror = ROOT / "fugal_subnet" / "routing_data"
    names = set(ROUTING_DATA_REQUIRED)
    if mirror.is_dir():
        names |= {p.name for p in mirror.iterdir() if p.is_file()}
    for name in sorted(names):
        copy, reviewed = mirror / name, ROOT / "data" / name
        missing = [str(p.relative_to(ROOT)) for p in (copy, reviewed) if not p.exists()]
        if missing:
            errors.append(f"{', '.join(missing)} missing (routing_data/ and data/ must both hold {name})")
        elif copy.read_bytes() != reviewed.read_bytes():
            errors.append(f"data/{name} and fugal_subnet/routing_data/{name} differ")


def check_price_table_v2_pinned(errors: list[str]) -> None:
    """The v2 table matches its pin, ids are unique, and every entry pins
    exactly one valid reasoning setting. (Its routing_data/ copy is
    checked by check_routing_data_mirrors_data.)"""
    a = ROOT / "data" / "models_v2.json"
    if not a.exists():
        errors.append("data/models_v2.json is missing")
        return
    actual = hashlib.sha256(a.read_bytes()).hexdigest()
    if actual != PINNED_PRICE_TABLE_V2_SHA256:
        errors.append(f"data/models_v2.json: hash {actual[:16]}... does not match pin "
                      f"{PINNED_PRICE_TABLE_V2_SHA256[:16]}... (a v2 price change re-scores every v2 miner)")
    rows = json.loads(a.read_text())
    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("data/models_v2.json: duplicate model id")
    for r in rows:
        rs = r.get("reasoning")
        if not isinstance(rs, dict) or len(rs) != 1 or not (
            rs.get("effort") in ("minimal", "low", "medium", "high")
            or (isinstance(rs.get("max_tokens"), int) and 0 < rs["max_tokens"] < 16384)
            or rs.get("enabled") is True
        ):
            errors.append(f"data/models_v2.json: {r.get('id')!r} must pin exactly one valid reasoning setting")


def check_menu_v3b_pinned(errors: list[str]) -> None:
    """Menu B and its manifest match their pins, pass the same checks as menu
    A, and extend menu A: every menu A row (entry and token row) is in menu B
    unchanged, in order, ahead of the added ones."""
    from fugal_subnet.routing_protocol import validate_menu_v3
    from fugal_subnet.vendor import success_contract

    files = (("models_v3b.json", PINNED_PRICE_TABLE_V3B_SHA256, "v3 menu B"),
             ("benchmark_tokens_protocol_v3b.json", PINNED_TOKEN_MANIFEST_V3B_SHA256,
              "v3 menu B token manifest"))
    for name, pin, what in files:
        path = ROOT / "data" / name
        if not path.exists():
            errors.append(f"data/{name} is missing")
            return
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != pin:
            errors.append(f"data/{name}: hash {actual[:16]}... does not match pin "
                          f"{pin[:16]}... (a {what} change re-scores every v3 miner)")
    menu_a = json.loads((ROOT / "data" / "models_v3.json").read_text())
    menu_b = json.loads((ROOT / "data" / "models_v3b.json").read_text())
    man_a = json.loads((ROOT / "data" / "benchmark_tokens_protocol_v3.json").read_text())
    man_b = json.loads((ROOT / "data" / "benchmark_tokens_protocol_v3b.json").read_text())
    try:
        validate_menu_v3(menu_b)
        success_contract.validate_manifest(man_b, deployable=True)
    except (ValueError, KeyError) as e:
        errors.append(f"data/models_v3b.json or its manifest: {e}")
        return
    if menu_b[:len(menu_a)] != menu_a or len(menu_b) <= len(menu_a):
        errors.append("data/models_v3b.json must be menu A's rows unchanged, then the added entries")
    if man_b["models"][:len(man_a["models"])] != man_a["models"]:
        errors.append("data/benchmark_tokens_protocol_v3b.json must keep menu A's rows unchanged")
    budget = {r["id"]: int(r.get("max_tokens") or 0) for r in menu_b}
    if sorted(r["id"] for r in man_b["models"]) != sorted(budget):
        errors.append("data/benchmark_tokens_protocol_v3b.json: rows do not match menu B one to one")


def check_menu_v4_seed_pinned(errors: list[str]) -> None:
    """The v4 menu seed matches its pin and is a state the first v4 epoch can
    start from: a valid MenuState at epoch 0 and serial 0 with nothing in
    measurement, waiting or retired, whose every entry is a v3 menu B entry
    with that entry's model, reasoning and host order unchanged, on the menu
    and scored from the start, and passes the compiled rules
    (menu_rules.check_entry) under the lab and host list: the signed list the
    release ships, or the draft under review until the first one is signed.
    It is within the compiled caps (menu_state.check_caps, which holds each
    pick-cost table to what its cells could give).
    (Its routing_data/ copy is checked by check_routing_data_mirrors_data;
    its answer limits against the listing it was built from by
    tests/test_menu_v4_seed.py.)"""
    from fugal_subnet import menu_rules, menu_state

    path = ROOT / "data" / "menu_v4_seed.json"
    if not path.exists():
        errors.append("data/menu_v4_seed.json is missing")
        return
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != PINNED_MENU_V4_SEED_SHA256:
        errors.append(f"data/menu_v4_seed.json: hash {actual[:16]}... does not match pin "
                      f"{PINNED_MENU_V4_SEED_SHA256[:16]}... (the seed is the first v4 epoch's "
                      "menu: a change moves every v4 proof's namespace)")
        return
    try:
        seed = menu_state.from_dict(json.loads(data))
    except ValueError as e:
        errors.append(f"data/menu_v4_seed.json: {e}")
        return
    if ((seed.epoch, seed.serial) != (0, 0) or seed.provisional or seed.retired
            or seed.coldkey_cooldowns or any(e.provisional for e in seed.entries)):
        errors.append("data/menu_v4_seed.json must be at epoch 0 and serial 0 with nothing "
                      "provisional, retired or in cooldown")
    if not seed.entries:
        errors.append("data/menu_v4_seed.json holds no entry")
    try:
        allowlist = menu_rules.shipped_list(menu_rules.KIND_MENU_ALLOWLIST)
    except FileNotFoundError:
        draft = ROOT / "data" / "menu_allowlist.draft.json"
        allowlist = (menu_rules.parse_unsigned_list(draft.read_bytes(),
                                                    menu_rules.KIND_MENU_ALLOWLIST)
                     if draft.exists() else None)
    except menu_rules.ListRefused:
        allowlist = None          # check_signed_lists reports a shipped list that does not verify
    if allowlist is None:
        errors.append("data/menu_v4_seed.json: no lab and host list to check its entries under "
                      "(neither a signed list nor the draft)")
    v3 = {r["id"]: r for r in json.loads((ROOT / "data" / "models_v3b.json").read_text())}
    documents = {d["id"]: d for d in menu_state.to_dict(seed)["entries"]}
    for e in seed.entries:
        row = v3.get(e.id)
        request = menu_state.request_for(seed, e.id)
        request.pop("max_tokens")
        if row is None or request != {k: row[k] for k in ("model", "reasoning", "provider")
                                      if k in row}:
            errors.append(f"data/menu_v4_seed.json: {e.id} is not a v3 menu B entry with its "
                          "model, reasoning and host order unchanged")
        refusal = (menu_rules.check_entry(documents[e.id], allowlist)
                   if allowlist is not None else None)
        if refusal is not None:
            errors.append(f"data/menu_v4_seed.json: {e.id} does not pass the compiled rules "
                          f"({refusal})")
        if (e.rev, e.joined_epoch, e.scored_from, e.proposer) != (1, 0, 0, ""):
            errors.append(f"data/menu_v4_seed.json: {e.id} must be at rev 1, on the menu and "
                          "scored from the start, with no proposer")
        if min(e.price_in, e.price_out, e.mean_in, e.mean_out) <= 0:
            errors.append(f"data/menu_v4_seed.json: {e.id} needs positive prices and token means")
        ceiling = menu_rules.check_price(e.price_in, e.price_out)
        if ceiling is not None:
            errors.append(f"data/menu_v4_seed.json: {e.id} is above the price ceiling ({ceiling})")
    # Within the compiled caps the reader holds every menu to, a pick-cost
    # table within what its cells could give among them.
    why = menu_state.check_caps(seed)
    if why is not None:
        errors.append(f"data/menu_v4_seed.json: {why}")


def check_pool_mix_v4_pinned(errors: list[str]) -> None:
    """The v4 question mix matches its pin and reads as a mix: integer
    per-cent targets summing to 100, every set of the v3 pool manifest (v4's
    pool) at its size with a positive share, no set above its cycle cap
    (epoch_plan.mix_v4 checks the rest when it reads the package copy, which
    check_routing_data_mirrors_data holds byte-identical). Its rebuild from
    the committed observations is tests/test_pool_mix_v4.py."""
    path = ROOT / "data" / "pool_mix_v4.json"
    if not path.exists():
        errors.append("data/pool_mix_v4.json is missing")
        return
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != PINNED_POOL_MIX_V4_SHA256:
        errors.append(f"data/pool_mix_v4.json: hash {actual[:16]}... does not match pin "
                      f"{PINNED_POOL_MIX_V4_SHA256[:16]}... (the mix decides every v4 epoch's "
                      "questions)")
        return
    doc = json.loads(data)
    manifest = json.loads((ROOT / "fugal_subnet" / "benchmarks" / "pool_manifest_v3.json")
                          .read_text(encoding="utf-8"))["per_benchmark"]
    if {b: (r["category"], r["pool"]) for b, r in doc["benchmarks"].items()} != {
            b: (r["category"], r["n"]) for b, r in manifest.items()}:
        errors.append("data/pool_mix_v4.json: its sets, categories and sizes are not the v3 "
                      "pool manifest's")
    if any(r["weight_ppm"] <= 0 for r in doc["benchmarks"].values()):
        errors.append("data/pool_mix_v4.json: a set of the pool has no share")
    try:
        from fugal_subnet import epoch_plan
        epoch_plan.mix_v4.cache_clear()
        epoch_plan.mix_v4()
    except (ValueError, KeyError, OSError) as e:
        errors.append(f"fugal_subnet/routing_data/pool_mix_v4.json does not read as a v4 mix: {e}")


def check_published_v4_pinned(errors: list[str]) -> None:
    """The v4 published-set seed matches its pin and reads as a seed of the
    v3 pool manifest: its pool hash, its sets at their sizes, and no set
    listing more unpublished questions than it holds (memo_v4.check_seed
    checks the rest when it reads the package copy, which
    check_routing_data_mirrors_data holds byte-identical). Its rebuild from
    the committed observations is tests/test_memo_v4.py."""
    path = ROOT / "data" / "published_v4.json"
    if not path.exists():
        errors.append("data/published_v4.json is missing")
        return
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != PINNED_PUBLISHED_V4_SHA256:
        errors.append(f"data/published_v4.json: hash {actual[:16]}... does not match pin "
                      f"{PINNED_PUBLISHED_V4_SHA256[:16]}... (the seed decides which questions "
                      "v4's memorisation check counts unseen and its fresh draw spends)")
        return
    doc = json.loads(data)
    manifest = json.loads((ROOT / "fugal_subnet" / "benchmarks" / "pool_manifest_v3.json")
                          .read_text(encoding="utf-8"))
    if (doc["pool"]["pool_hash"] != manifest["pool_hash"]
            or doc["pool"]["per_set"] != {b: r["n"] for b, r in manifest["per_benchmark"].items()}):
        errors.append("data/published_v4.json: it was not observed on the v3 pool manifest's pool")
    try:
        from fugal_subnet import memo_v4
        memo_v4.seed_doc.cache_clear()
        memo_v4.seed_doc()
    except (ValueError, KeyError, OSError) as e:
        errors.append(f"fugal_subnet/routing_data/published_v4.json does not read as a seed: {e}")


def check_no_other_v4_menu_ships(errors: list[str], root: Path | None = None) -> None:
    """No menu other than the pinned seed ships in the package (and so in the
    image). From v4 the menu is published state: an epoch's menu is the seed
    or what the step derived from it, named by the root's commitment. A
    second v4 menu document in the package would be a menu no hash pins and
    no step derived, one import away from being read. A file counts as one
    when it is a JSON object with a menu state's fields. `root` is for the
    tests."""
    root = ROOT if root is None else Path(root)
    seed = root / "fugal_subnet" / "routing_data" / "menu_v4_seed.json"
    for path in sorted((root / "fugal_subnet").rglob("*.json")):
        if path == seed:
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(doc, dict) and {"entries", "provisional", "retired",
                                      "allowlists"} <= set(doc):
            errors.append(f"{path.relative_to(root)}: a protocol v4 menu document other than "
                          "the pinned seed; the package carries the rules and the seed only")


def check_menu_v3_pinned(errors: list[str]) -> None:
    """The v3 menu and token manifest match their pins; the menu passes
    routing_protocol.validate_menu_v3; the manifest has exactly one row per
    menu entry, and no row's completion mean exceeds its entry's budget."""
    from fugal_subnet.routing_protocol import validate_menu_v3
    from fugal_subnet.vendor import success_contract

    menu_path = ROOT / "data" / "models_v3.json"
    man_path = ROOT / "data" / "benchmark_tokens_protocol_v3.json"
    for path, pin, what in ((menu_path, PINNED_PRICE_TABLE_V3_SHA256, "v3 menu"),
                            (man_path, PINNED_TOKEN_MANIFEST_V3_SHA256, "v3 token manifest")):
        if not path.exists():
            errors.append(f"data/{path.name} is missing")
            return
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != pin:
            errors.append(f"data/{path.name}: hash {actual[:16]}... does not match pin "
                          f"{pin[:16]}... (a {what} change re-scores every v3 miner)")
    rows = json.loads(menu_path.read_text())
    try:
        validate_menu_v3(rows)
    except ValueError as e:
        errors.append(f"data/models_v3.json: {e}")
        return
    manifest = json.loads(man_path.read_text())
    try:
        success_contract.validate_manifest(manifest)
    except (ValueError, KeyError) as e:
        errors.append(f"data/benchmark_tokens_protocol_v3.json: {e}")
        return
    budget = {r["id"]: int(r.get("max_tokens") or 0) for r in rows}
    ids = [r["id"] for r in manifest["models"]]
    if sorted(ids) != sorted(budget):
        errors.append("data/benchmark_tokens_protocol_v3.json: rows do not match the v3 menu "
                      "entries one to one")
    for r in manifest["models"]:
        if budget.get(r["id"]) and r["mean_out_tokens"] > budget[r["id"]]:
            errors.append(f"data/benchmark_tokens_protocol_v3.json: {r['id']} completion mean "
                          "exceeds the entry's max_tokens")


def check_price_table_pinned(errors: list[str]) -> None:
    """The consensus price table must match its pin, and be well-formed.
    (Its routing_data/ copy is checked by check_routing_data_mirrors_data.)
    """
    path = ROOT / "data" / "models.json"
    if not path.exists():
        errors.append("data/models.json is missing — validators cannot price proofs")
        return

    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != PINNED_PRICE_TABLE_SHA256:
        errors.append(
            f"data/models.json: price table hash {actual[:16]}... does not match "
            f"pin {PINNED_PRICE_TABLE_SHA256[:16]}... — a price change re-scores "
            "every miner, so update the pin deliberately"
        )

    try:
        models = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"data/models.json is not valid JSON: {e}")
        return

    seen = set()
    for entry in models:
        mid = entry.get("id")
        if not mid:
            errors.append("data/models.json: an entry has no 'id'")
            continue
        if mid in seen:
            errors.append(f"data/models.json: duplicate model id {mid!r}")
        seen.add(mid)
        for key in ("in", "out"):
            price = entry.get(key)
            if not isinstance(price, (int, float)) or price < 0:
                errors.append(
                    f"data/models.json: {mid!r} has non-numeric or negative {key!r} price"
                )


def check_paid_call_guards(errors: list[str]) -> None:
    """Ensure every call_model() invocation has an explicit live= keyword."""
    for path in python_files():
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_call_model = (
                isinstance(func, ast.Name) and func.id == "call_model"
            )
            if is_call_model and not any(kw.arg == "live" for kw in node.keywords):
                relative = path.relative_to(ROOT)
                errors.append(
                    f"{relative}:{node.lineno}: call_model() requires explicit live="
                )


def _declared_flags(path: Path) -> set[str]:
    """Every long flag an entry point defines, read statically from its AST.

    Parsed rather than executed: importing these modules pulls in torch and
    bittensor, and a lint check should not cost thirty seconds or have import
    side effects. Covers both click (@click.option) and argparse
    (add_argument on a parser or any group).
    """
    flags: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in ("option", "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                for piece in arg.value.split("/"):        # click's --live/--mock
                    if piece.startswith("--"):
                        flags.add(piece)
    return flags


def check_documented_flags(errors: list[str]) -> None:
    """Every flag the docs show must exist on the command they show it for.

    This class of bug recurred repeatedly: docs advertising --live and
    --epoch-budget on a script that had neither, a --models example using
    comma separation against a nargs="+" parser, and a competitive-training
    command missing --use-backbone, which silently trained on random
    embeddings. Prose drifts from argparse and nothing catches it.
    """
    doc_paths = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").glob("*.md"))
    command = re.compile(
        r"python\s+((?:scripts|neurons)/[\w_]+\.py)((?:\s+\\\s*\n|[^\n`])*)"
    )
    cache: dict[str, set[str]] = {}

    for doc in doc_paths:
        for match in command.finditer(doc.read_text(encoding="utf-8")):
            target, rest = match.group(1), match.group(2)
            script = ROOT / target
            if not script.exists():
                errors.append(f"{doc.name}: documents {target}, which does not exist")
                continue
            if target not in cache:
                cache[target] = _declared_flags(script)
            declared = cache[target]
            for flag in sorted(set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", rest))):
                if flag not in declared:
                    errors.append(
                        f"{doc.relative_to(ROOT)}: documents `{flag}` for {target}, "
                        "which does not define it"
                    )


def check_tee_safety(errors: list[str]) -> None:
    """TEE-specific safety invariants."""
    # Miner must not use deferred annotations
    miner_path = ROOT / "neurons" / "miner.py"
    tree = ast.parse(miner_path.read_text(encoding="utf-8"), filename=str(miner_path))
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
        ):
            errors.append(
                "neurons/miner.py: deferred annotations break bt.Axon.attach()"
            )

    # TEE harness must use the same graders.py (import check)
    harness_path = ROOT / "fugal_subnet" / "tee" / "harness.py"
    if harness_path.exists():
        source = harness_path.read_text(encoding="utf-8")
        if "fugal_subnet.graders" not in source:
            errors.append(
                "fugal_subnet/tee/harness.py: must import from fugal_subnet.graders "
                "(hash-pinned grader ensures TEE grades match validator)"
            )

    # Verify module must not import or call models
    verify_path = ROOT / "fugal_subnet" / "tee" / "verify.py"
    if verify_path.exists():
        vtree = ast.parse(verify_path.read_text(encoding="utf-8"), filename=str(verify_path))
        forbidden = {"build_matrix", "call_model", "compute_hidden_states"}
        for node in ast.walk(vtree):
            if isinstance(node, ast.Name) and node.id in forbidden:
                errors.append(
                    f"fugal_subnet/tee/verify.py: references {node.id} — "
                    "verification must never call models"
                )


def check_upstream_is_inside_the_measurement(errors: list[str]) -> None:
    """The model upstream must be fixed by what the TD measures, and nothing else.

    Diligence 2026-10, finding 1. The metering proxy reads its upstream from
    FUGAL_OPENROUTER_BASE inside the miner's own TD, and the TD holds every
    gold answer. The only thing that binds the upstream is the dstack compose
    hash: the compose fixes the container environment, the hash is replayed
    out of the RTMR3 event log, and the approved entry names it. That chain
    has three links an edit can silently break, and each is checked here:

    1. every frozen app-compose must have `allowed_envs: []` -- a listed env
       is delivered through dstack's encrypted channel AFTER attestation and is
       unmeasured by design, so a compose that allowed FUGAL_OPENROUTER_BASE
       would hand the miner the upstream with the hash unchanged;
    2. the compose must reference its image by digest, never by tag, or the
       hash stays stable while the code beneath it changes;
    3. the compose must not set FUGAL_OPENROUTER_BASE itself (the default in
       runtime.py is the real endpoint, and a compose that sets it to
       anything else is approving a different upstream under the same name).

    The live validator refuses a bare base entry (one with no app identity),
    which is the fourth link; that one is checked by
    `check_live_requires_app_identity` below.
    """
    import json

    frozen = sorted((ROOT / "deploy" / "dstack").glob("app-compose*.json"))
    if not frozen:
        errors.append("deploy/dstack: no app-compose*.json to check the upstream binding on")
    for path in frozen:
        rel = path.relative_to(ROOT)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            errors.append(f"{rel}: not JSON ({e})")
            continue
        if doc.get("allowed_envs"):
            errors.append(
                f"{rel}: allowed_envs is {doc['allowed_envs']!r}; an allowed env "
                "arrives after attestation, unmeasured, so this must be [] (I8)"
            )
        compose = doc.get("docker_compose_file", "")
        for line in compose.splitlines():
            stripped = line.strip()
            if stripped.startswith("image:") and "@sha256:" not in stripped:
                errors.append(f"{rel}: image referenced by tag, not digest: {stripped} (I8)")
            if stripped.startswith("#"):
                continue
            # Both compose forms: a list entry `- FUGAL_OPENROUTER_BASE=...` and a
            # mapping entry `FUGAL_OPENROUTER_BASE: ...` (only the first was
            # matched until 2026-11-08).
            if re.search(r"\bFUGAL_OPENROUTER_BASE\s*[:=]", stripped):
                errors.append(
                    f"{rel}: compose sets FUGAL_OPENROUTER_BASE -- the upstream is "
                    "fixed by runtime.py's default and must not be overridden (I8)"
                )

    # TLS inside the proxy is the other half: a miner who controls the VM's
    # network reaches the same outcome by pointing DNS at themselves if the
    # proxy ever stops validating certificates.
    runtime = ROOT / "fugal_subnet" / "tee" / "runtime.py"
    text = runtime.read_text(encoding="utf-8")
    for needle in ("_create_unverified_context", "CERT_NONE", "check_hostname = False"):
        if needle in text:
            errors.append(
                f"fugal_subnet/tee/runtime.py: contains {needle!r}; the metering "
                "proxy must validate the upstream certificate (I8)"
            )


def check_live_composes_are_frozen(errors: list[str], root: Path | None = None) -> None:
    """Each live compose (deploy/dstack/docker-compose*.yaml) must be, byte for
    byte, the `docker_compose_file` of some frozen app-compose*.json.

    The compose hash validators approve is sha256 of the frozen app-compose,
    which embeds the compose text. A compose edited after its freeze yields a
    hash no entry names: a miner deploying it is refused, and nothing says
    why. Found 2026-11-05 (review F-074): both live composes gained
    FUGAL_POOL_CACHE_V2 after the 2026-11-03c freeze and matched none of the
    eight frozen files. Clear it by freezing the compose again
    (`dstack-cloud prepare`, commit the app-compose.json, publish the entry)
    or by reverting the compose change.
    """
    import json

    root = root or ROOT
    dstack = root / "deploy" / "dstack"
    frozen_texts: dict[str, str] = {}
    for path in sorted(dstack.glob("app-compose*.json")):
        try:
            frozen_texts[path.name] = json.loads(path.read_text(encoding="utf-8")).get(
                "docker_compose_file", "")
        except ValueError:
            continue  # reported by check_upstream_is_inside_the_measurement
    for live in sorted(dstack.glob("docker-compose*.yaml")):
        text = live.read_text(encoding="utf-8")
        if text not in frozen_texts.values():
            digest = next((ln.strip().split("@sha256:")[-1][:12] for ln in text.splitlines()
                           if "@sha256:" in ln), "?")
            errors.append(
                f"{live.relative_to(root)} (image {digest}...) is not the embedded compose of "
                f"any frozen app-compose*.json: a miner deploying it gets a compose hash no "
                "approved entry names. Freeze it (dstack-cloud prepare -> commit the "
                "app-compose.json, publish the entry) or revert the change (I8)"
            )


def check_release_composes_target_v2(errors: list[str], root: Path | None = None) -> None:
    """At a release, each live compose must pin its image by digest and carry
    a valid FUGAL_V2_ACTIVATION_BLOCK.

    The miner reads the activation block only from its compose, so a compose
    without one is a v1 miner, whatever image it names. `--release` passed a
    frozen compose with the v1 image and no activation block (Fable review
    2026-11-09, item 8): the gate checked that composes were frozen, not what
    they were frozen as. The block must be an epoch boundary at the compose's
    own FUGAL_EPOCH_INTERVAL (slicer.validate_activation_block).
    """
    import re

    from fugal_subnet.benchmarks.slicer import blocks_per_epoch, validate_activation_block

    root = root or ROOT
    for live in sorted((root / "deploy" / "dstack").glob("docker-compose*.yaml")):
        text = live.read_text(encoding="utf-8")
        name = live.relative_to(root)
        if not re.search(r"^\s*image:\s*\S+@sha256:[0-9a-f]{64}\s*$", text, re.M):
            errors.append(f"{name}: the image is not pinned by digest (@sha256:<64 hex>)")
        env = dict(re.findall(r"^\s*-?\s*(FUGAL_[A-Z0-9_]+)\s*[=:]\s*\"?([^\"\s#]*)", text, re.M))
        raw = env.get("FUGAL_V2_ACTIVATION_BLOCK", "")
        if not raw:
            errors.append(f"{name}: no FUGAL_V2_ACTIVATION_BLOCK: a compose without it is a v1 "
                          "miner (MINER_GUIDE, the flip); set the published block before freezing")
            continue
        try:
            interval = int(env.get("FUGAL_EPOCH_INTERVAL") or 3600)
            if int(raw) <= 0:
                raise ValueError("it must be a future block, above 0")
            validate_activation_block(int(raw), blocks_per_epoch(interval))
        except (ValueError, TypeError) as e:
            errors.append(f"{name}: FUGAL_V2_ACTIVATION_BLOCK={raw!r} is not valid: {e}")


# Blocks the MINER reads only from its compose's environment while the
# validator has them compiled in for finney. A finney compose that leaves one
# out makes every miner on it serve the older protocol after the block
# (v0.5.0: the first v3 composes set FUGAL_V2_ACTIVATION_BLOCK but not
# FUGAL_V3_ACTIVATION_BLOCK, so the image would have kept serving v2).
MINER_COMPOSE_BLOCKS = (
    ("FUGAL_V2_ACTIVATION_BLOCK", "FINNEY_V2_ACTIVATION_BLOCK"),
    ("FUGAL_V3_ACTIVATION_BLOCK", "FINNEY_V3_ACTIVATION_BLOCK"),
    ("FUGAL_V4_ACTIVATION_BLOCK", "FINNEY_V4_ACTIVATION_BLOCK"),
)


def check_finney_compose_carries_compiled_blocks(errors: list[str], root: Path | None = None,
                                                 cfg=None) -> None:
    """The finney live compose must set every block in MINER_COMPOSE_BLOCKS
    that the release compiles in, to exactly the compiled value. Always an
    error, not only at --release: a compose without it is a miner on the
    wrong protocol."""
    import re

    if cfg is None:
        from fugal_subnet import config as cfg
    root = root or ROOT
    for live in sorted((root / "deploy" / "dstack").glob("docker-compose*.yaml")):
        text = live.read_text(encoding="utf-8")
        network = re.search(r"-\s*--network\s*\n\s*-\s*(\S+)", text)
        if not network or network.group(1).strip("\"'") != "finney":
            continue
        env = dict(re.findall(r"^\s*-?\s*(FUGAL_[A-Z0-9_]+)\s*[=:]\s*\"?([^\"\s#]*)", text, re.M))
        for var, compiled_name in MINER_COMPOSE_BLOCKS:
            compiled = getattr(cfg, compiled_name, None)
            if compiled is None:
                continue
            have = env.get(var, "")
            if have != str(compiled):
                errors.append(
                    f"{live.relative_to(root)}: {var}={have or '(unset)'}, but the release "
                    f"compiles {compiled_name} = {compiled}. The miner reads this block only "
                    f"from its compose; set {var}={compiled} and re-freeze.")


def check_live_requires_app_identity(errors: list[str]) -> None:
    """A live validator must refuse a bare base approved entry at startup.

    Diligence 2026-10, finding 1. The refusal is `parse_approved(...,
    require_app=live)` in neurons/validator.py; it was a logger.warning before,
    and a warning about something the adversary controls is not a control.
    This check reads the AST so the call cannot quietly lose its keyword.
    """
    path = ROOT / "neurons" / "validator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "parse_approved"):
            continue
        for kw in node.keywords:
            if kw.arg != "require_app":
                continue
            v = kw.value
            if (isinstance(v, ast.Name) and v.id == "live") or (
                isinstance(v, ast.Constant) and v.value is True
            ):
                found = True
    if not found:
        errors.append(
            "neurons/validator.py: parse_approved is not called with "
            "require_app=live; a bare base entry would be accepted in live mode (I8)"
        )


def check_bootstrap_frame_pinned(errors: list[str]) -> None:
    """The shipped bootstrap reference frame is consensus state and is pinned.

    Diligence 2026-10, finding 3. The frame decides what the constant-policy
    frontier is valued at, and every live validator starts from the shipped
    file, so an unreviewed edit to it re-scores every miner on every
    validator at once -- the same hazard as data/models.json, and pinned the
    same way. A file with no pin, or a pin with no file, both fail: the first
    is an unreviewed frame, the second is a live validator that cannot start.
    """
    path = ROOT / "data" / "reference_frame.json"
    if PINNED_BOOTSTRAP_FRAME_SHA256 is None:
        if path.exists():
            errors.append(
                "data/reference_frame.json exists but PINNED_BOOTSTRAP_FRAME_SHA256 is "
                "None; pin it in the same commit (I1)"
            )
        return
    if not path.exists():
        errors.append(
            "data/reference_frame.json is pinned but missing; a live validator "
            "cannot start without it (I1)"
        )
        return
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != PINNED_BOOTSTRAP_FRAME_SHA256:
        errors.append(
            f"data/reference_frame.json: sha256 {digest[:16]}... does not match the "
            "pin. Restore it (git checkout), or rebuild with "
            "scripts/build_frame_from_calls.py --build-only; a deliberately new "
            "frame updates the pin in the same commit, saying why (I1)"
        )


def check_every_env_constant_classified(errors: list[str]) -> None:
    """Every FUGAL_ variable config.py reads is consensus, input, or operational.

    Diligence 2026-10, finding A2. The consensus digest only covers what
    someone remembered to add to it; this check makes forgetting impossible.
    A FUGAL_ name read by config.py must appear in exactly one of
    CONSENSUS_ENV_VARS, CONSENSUS_ENV_INPUTS or OPERATIONAL_ENV_VARS, and
    every CONSENSUS_ENV_VARS constant must be in the fingerprint's scoring
    block or epoch block, so the classification and the digest cannot drift.
    """
    import re
    import sys

    sys.path.insert(0, str(ROOT))
    from fugal_subnet import config, fingerprint

    source = (ROOT / "fugal_subnet" / "config.py").read_text(encoding="utf-8")
    # Every read anywhere in the shipped code, not only config.py: the pool's
    # skip list and the unpinned-pool switch lived in benchmarks/loader.py and
    # were consensus-affecting, undigested and unrefused on finney until the
    # scan was widened (review 2026-11-05, F-027).
    pattern = re.compile(r'os\.(?:getenv|environ\.get)\(\s*"(FUGAL_[A-Z0-9_]+)"'
                         r'|os\.environ\[\s*"(FUGAL_[A-Z0-9_]+)"\s*\](?!\s*=)')
    read: set[str] = set()
    for path in sorted(list((ROOT / "fugal_subnet").rglob("*.py")) + list((ROOT / "neurons").glob("*.py"))):
        for a, b in pattern.findall(path.read_text(encoding="utf-8")):
            read.add(a or b)
    lists = {
        "CONSENSUS_ENV_VARS": set(config.CONSENSUS_ENV_VARS),
        "CONSENSUS_ENV_INPUTS": set(config.CONSENSUS_ENV_INPUTS),
        "OPERATIONAL_ENV_VARS": set(config.OPERATIONAL_ENV_VARS),
    }
    for name in sorted(read):
        where = [k for k, v in lists.items() if name in v]
        if len(where) != 1:
            errors.append(
                f"{name} is read under fugal_subnet/ or neurons/ but listed in "
                f"{where or 'no classification list'}; it must be in exactly one "
                "of CONSENSUS_ENV_VARS, CONSENSUS_ENV_INPUTS, OPERATIONAL_ENV_VARS (I1)"
            )
    for k, v in lists.items():
        for name in sorted(v - read):
            errors.append(f"fugal_subnet/config.py: {k} lists {name}, which nothing reads")
    # Consensus variables read outside config.py are covered by the epoch
    # block, not by a module attribute; the attribute scan below is for the
    # ones config.py defines.
    outside = {"FUGAL_SKIP_BENCHMARKS", "FUGAL_POOL_UNPINNED", "FUGAL_OPENROUTER_BASE"}

    fp = fingerprint.environment_fingerprint()
    covered = set(fp["scoring"]) | {
        "SLICE_SIZE", "SLICE_SHARE_FACTOR", "EPOCH_INTERVAL", "EXPLORE_FRACTION",
        "EPOCH_COLLECT_FRACTION",
        # Not in the digest (routing_protocol hashes the shipped tables), but a
        # set FUGAL_MODEL_PRICES is a consensus override: refused on finney.
        "TEE_MODEL_PRICES_PATH",
    }
    for key in ("skip_benchmarks", "pool_unpinned", "pool_manifest_v1", "pool_manifest_v2",
                "openrouter_base"):
        if key not in fp["epoch"]:
            errors.append(f"fingerprint epoch block lacks {key} (I1)")
    for env_name in sorted(set(config.CONSENSUS_ENV_VARS) - outside):
        attr = re.search(
            rf'^([A-Z_]+)\s*=[^\n]*?os\.getenv\(\s*"{env_name}"', source, re.M | re.S,
        )
        if attr is None:
            # multi-line definitions: find the assignment line above the getenv
            idx = source.index(f'"{env_name}"')
            attr_name = re.findall(r"^([A-Z_]+)\s*=", source[:idx], re.M)[-1]
        else:
            attr_name = attr.group(1)
        if attr_name not in covered:
            errors.append(
                f"fugal_subnet/config.py: {attr_name} ({env_name}) is a consensus "
                "constant but is not in fingerprint.environment_fingerprint (I1)"
            )


def check_v4_constants_in_digest(errors: list[str]) -> None:
    """Every protocol v4 constant and pinned file is in the v4 section of the
    consensus digest, or named with the reason it decides nothing a v4 epoch
    publishes (fugal_subnet/consensus_v4.py; LIVE_MENU_V4.md 17, item 8). A
    constant a later package adds to a v4 module, or to a module v4 code
    reads, fails here until it is classified."""
    import sys

    sys.path.insert(0, str(ROOT))
    from fugal_subnet import consensus_v4

    errors.extend(consensus_v4.problems(ROOT))


# --- the protocol v4 release gate ----------------------------------------------------
#
# What a release that schedules protocol v4 on finney must have before it is
# cut (`--release-v4`; `--release` runs it too once FINNEY_V4_ACTIVATION_BLOCK
# is set, so a release that compiles the block cannot skip it). Each condition
# is named, so the gate run today lists exactly what is left. Lessons it
# keeps: a frozen compose is checked as frozen, by running the miner's own
# protocol selection on its environment (2026-11-17: a compose without a
# variable the miner read kept it on the protocol before); the image is
# checked to carry every file v4 reads.

_COMPOSE_ENV = re.compile(r"^\s*-?\s*([A-Z][A-Z0-9_]*)\s*[=:]\s*\"?([^\"\s#]*)", re.M)


def _compose_environment(compose_text: str) -> dict[str, str]:
    """The `environment:` entries of a compose (list or mapping form)."""
    lines, inside, indent = [], False, None
    for line in compose_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("environment:"):
            inside, indent = True, len(line) - len(line.lstrip())
            continue
        if inside:
            if stripped and not stripped.startswith("#") and len(line) - len(line.lstrip()) <= indent:
                inside = False
            elif stripped and not stripped.startswith("#"):
                lines.append(line)
    return dict(_COMPOSE_ENV.findall("\n".join(lines)))


def _chain_block_variables(config_source: str) -> list[tuple[str, str]]:
    """(compiled finney constant, variable) of every activation block config.py
    reads through `_chain_block` (the variables may only repeat the compiled
    value on finney)."""
    return re.findall(r'_chain_block\(\s*genesis_hash,\s*(FINNEY_[A-Z0-9_]+),\s*[A-Z0-9_]+,'
                      r'\s*"(FUGAL_[A-Z0-9_]+)"', config_source)


def _protocol_selection_variables(config_source: str,
                                  start: str = "protocol_for_chain") -> list[tuple[str, str]]:
    """(compiled finney constant or "", variable) of every variable the miner's
    protocol selection (`config.protocol_for_chain`, what miner.main calls)
    reads, found by walking its calls in config.py: the `_chain_block`
    variables of the functions it reaches, and the variable behind each module
    constant it reads (V2_ACTIVATION_BLOCK, which the miner reads only from
    its compose)."""
    tree = ast.parse(config_source)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assigns: dict[str, ast.AST] = {}
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Name):
                    assigns[t.id] = n.value
    compiled = set(assigns)

    def getenv_of(name: str, seen: set) -> set[str]:
        if name in seen or name not in assigns or assigns[name] is None:
            return set()
        seen.add(name)
        out = set()
        for node in ast.walk(assigns[name]):
            if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "getenv"
                    and node.args and isinstance(node.args[0], ast.Constant)):
                out.add(node.args[0].value)
            elif isinstance(node, ast.Name):
                out |= getenv_of(node.id, seen)
        return out

    found: set[str] = set()
    todo, done = [start], set()
    while todo:
        name = todo.pop()
        if name in done or name not in funcs:
            continue
        done.add(name)
        for node in ast.walk(funcs[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "_chain_block" and len(node.args) >= 4 \
                        and isinstance(node.args[3], ast.Constant):
                    found.add(node.args[3].value)
                todo.append(node.func.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                    and node.id.isupper() and node.id in assigns:
                found |= getenv_of(node.id, set())
    out = []
    for var in sorted(v for v in found if v.startswith("FUGAL_")):
        finney = "FINNEY_" + var[len("FUGAL_"):]
        out.append((finney if finney in compiled else "", var))
    return out


def _miner_operator_variables(miner_source: str) -> set[str]:
    """The FUGAL_V4_ variables the miner reads as one operator's own choice
    (its click options): a shared, hashed compose must not carry them."""
    return set(re.findall(r'os\.getenv\("(FUGAL_V4_[A-Z0-9_]+)"', miner_source))


# The miner's own v4 options that need no channel into a TD: their default
# inside the image is the one a TD runs with. Every other FUGAL_V4_ option
# must have a path into a TD over provisioning (`_td_option_gaps`), since a
# frozen compose cannot carry one operator's choice and has no allowed_envs.
V4_OPTIONS_WITH_IMAGE_DEFAULT = {
    "FUGAL_V4_STATE_DIR": "results/v4 under /app, on the image's results volume",
    "FUGAL_V4_ANCHOR": "a last resort for a miner outside a TD; a TD anchors on reveals",
}


_SELECTION_PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
out = {}
try:
    from fugal_subnet import config, fingerprint
    g = config.FINNEY_GENESIS_HASH
    fingerprint.assert_consensus_overrides_allowed(g)
    out["protocol"] = config.protocol_for_chain(g)
    out["v4_block"] = config.v4_activation_block(g)
    out["epoch_interval"] = config.EPOCH_INTERVAL
    out["root_hotkeys"] = [list(r) for r in config.root_hotkeys_v4(True)]
except Exception as e:  # noqa: BLE001 - reported
    out["error"] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
"""


def _provision_fields(root: Path) -> set[str]:
    """tee/provision.ALLOWED_FIELDS, read from the source (the literal set)."""
    path = root / "fugal_subnet" / "tee" / "provision.py"
    if not path.exists():
        return set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ALLOWED_FIELDS" for t in node.targets):
            return {n.value for n in ast.walk(node.value)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return set()


def _literal_dict(path: Path, name: str) -> dict:
    """A module-level dict literal of `path` (read from the source), or {}."""
    if not path.exists():
        return {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return {}
            return value if isinstance(value, dict) else {}
    return {}


def _function(path: Path, name: str):
    if not path.exists():
        return None
    return next((n for n in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _calls(node, name: str) -> bool:
    return node is not None and any(
        isinstance(n, ast.Call) and (getattr(n.func, "id", None) == name
                                     or getattr(n.func, "attr", None) == name)
        for n in ast.walk(node))


def _td_option_gaps(root: Path, variables) -> dict[str, list[str]]:
    """{option: what its path into a TD lacks} for each FUGAL_V4_ option of
    `variables` whose path is incomplete. The path, each link read from the
    release's source:

    1. tee/v4_options.FIELDS maps the option to a provisioning field;
    2. the field is in tee/provision.ALLOWED_FIELDS (the receiver takes it);
    3. tee/v4_options.PARSERS names a parser for the field, defined in that
       module (validate_payload checks the value with it in the TD);
    4. the miner hands the pushed fields on (`_await_provisioning` takes
       `v4_options`, and `main` calls `_v4_pushed_options` with them);
    5. `_v4_pushed_options` takes the option from the push
       (`v4_options.pushed_options`) and applies it: it names the option.
    """
    opts = root / "fugal_subnet" / "tee" / "v4_options.py"
    fields = _literal_dict(opts, "FIELDS")
    parsers = _literal_dict(opts, "PARSERS")
    allowed = _provision_fields(root)
    miner = root / "neurons" / "miner.py"
    main, wait = _function(miner, "main"), _function(miner, "_await_provisioning")
    apply = _function(miner, "_v4_pushed_options")
    handed = (wait is not None and "v4_options" in [a.arg for a in wait.args.args]
              and main is not None and _calls(main, "_v4_pushed_options")
              and any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_await_provisioning"
                      and "v4_options" in [k.arg for k in n.keywords] for n in ast.walk(main)))
    # `if "<option>" in given:` in _v4_pushed_options: the option is taken
    # from the push's options and applied.
    applied = {n.left.value for n in ast.walk(apply) if isinstance(n, ast.Compare)
               and isinstance(n.left, ast.Constant) and len(n.ops) == 1
               and isinstance(n.ops[0], ast.In)} if apply else set()
    out: dict[str, list[str]] = {}
    for var in sorted(variables):
        field = fields.get(var)
        gaps = []
        if not field:
            gaps.append("tee/v4_options.FIELDS names no provisioning field for it")
        else:
            if field not in allowed:
                gaps.append(f"provisioning (tee/provision.ALLOWED_FIELDS) has no field {field}")
            if parsers.get(field) is None or _function(opts, str(parsers.get(field))) is None:
                gaps.append(f"tee/v4_options.PARSERS has no parser for {field}")
        if not handed:
            gaps.append("the miner does not hand the pushed options on (_await_provisioning, "
                        "_v4_pushed_options)")
        elif not _calls(apply, "pushed_options") or var not in applied:
            gaps.append("the miner does not apply it from the push (_v4_pushed_options)")
        if gaps:
            out[var] = gaps
    return out


def _miner_selection(root: Path, env: dict[str, str]) -> dict:
    """What the miner's startup computes on finney under exactly `env`: the
    protocol and the v4 block (config.protocol_for_chain, as miner.main), and
    whether the consensus-override refusal passes. Run in a fresh interpreter
    so the compose's environment, and nothing of this one's, is what it reads."""
    import os
    import subprocess
    import sys

    clean = {k: v for k, v in os.environ.items()
             if not k.startswith("FUGAL_") and k not in ("PYTHONPATH",)}
    clean.update(env)
    proc = subprocess.run([sys.executable, "-c", _SELECTION_PROBE, str(root)], env=clean,
                          capture_output=True, text=True, timeout=120)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": (proc.stderr.strip().splitlines() or ["no output"])[-1]}


def _root_table_problems(table) -> list[str]:
    """What the root's hotkey table must be for its consumers
    (menu_publish.hotkey_for_epoch, neurons/validator._v4_inputs): a tuple of
    (first epoch, hotkey) rows, the first epochs integers from 0 and strictly
    increasing (the row in force is the last one at or before an epoch, so
    two rows at one epoch, or rows out of order, would pick a hotkey by
    position), each hotkey one canonical SS58 string (it is looked up as a
    key of the chain's commitments, so any other spelling reads nothing)."""
    if not isinstance(table, (tuple, list)) or not table:
        return ["is empty"]
    out = []
    starts = []
    for row in table:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            return [f"has a row that is not (first epoch, hotkey): {row!r}"]
        first, hotkey = row
        if type(first) is not int or first < 0:
            out.append(f"has a first epoch that is not a non-negative integer: {first!r}")
        else:
            starts.append(first)
        if not (isinstance(hotkey, str) and re.fullmatch(_SS58_ADDRESS, hotkey)):
            out.append(f"names {hotkey!r}, which is not one canonical SS58 hotkey")
    if any(b <= a for a, b in zip(starts, starts[1:])):
        out.append("is not strictly increasing by first epoch (two rows at one epoch, or out of "
                   "order)")
    return out


# Finney's epoch as every validator runs it (deploy/env/validator.env.example): the
# subnet's tempo of 360 blocks at 12 s a block.
FINNEY_EPOCH_INTERVAL_S = 4320

# An SS58 address of a 32-byte key (network prefix 42, as the root hotkey): 47 or 48
# characters of the base58 alphabet, no whitespace, no other character.
_SS58_ADDRESS = r"[1-9A-HJ-NP-Za-km-z]{47,48}"


def _seed_strand_blockers(root: Path, labs, seed_entries) -> list[str]:
    """The first signed list cannot strand the menu: the list the release
    ships (`labs`), or the draft under review until one ships
    (data/menu_allowlist.draft.json), admits every entry of the pinned seed,
    so it is adopted at the first step and never held back for leaving fewer
    than menu_state.MENU_MIN_ENTRIES (while it is held, the menu names no
    list and no proposal can join); and every seed entry's price is within
    the ceiling (menu_rules.check_price)."""
    from fugal_subnet import menu_rules as mr
    from fugal_subnet import menu_state as ms

    if seed_entries is None:
        return []
    out = []
    which, allow = "shipped list", labs
    draft = root / "data" / "menu_allowlist.draft.json"
    if allow is None and draft.exists():
        which = "draft list (data/menu_allowlist.draft.json)"
        try:
            allow = mr.parse_unsigned_list(draft.read_bytes(), mr.KIND_MENU_ALLOWLIST)
        except mr.ListRefused as e:
            out.append(f"list: the {which} does not parse ({e})")
    if allow is not None:
        refused = sorted(e.get("id", "?") for e in seed_entries
                         if mr.check_entry(e, allow) is not None)
        kept = len(seed_entries) - len(refused)
        if refused:
            out.append(f"list: the {which} refuses {len(refused)} of the pinned seed's "
                       f"{len(seed_entries)} entries (first: {refused[0]}), leaving {kept}"
                       + (f", fewer than MENU_MIN_ENTRIES ({ms.MENU_MIN_ENTRIES}): the menu step "
                          "would hold the list back and no proposal could join"
                          if kept < ms.MENU_MIN_ENTRIES else "")
                       + "; the first signed list must admit every seed entry")
    for e in seed_entries:
        try:
            refusal = mr.check_price(e.get("price_in"), e.get("price_out"))
        except ValueError as err:
            refusal = err
        if refusal is not None:
            out.append(f"list: the pinned seed's {e.get('id', '?')} is priced above the ceiling "
                       f"or not at all ({refusal})")
    return out


def v4_release_blockers(root: Path | None = None, cfg=None, *, signers: bytes | None = None,
                        probe=None, published_pin: str | None = None,
                        signers_pin: str | None = None) -> list[str]:
    """Why a release that schedules protocol v4 on finney cannot be cut, or []:

    1. block: FINNEY_V4_ACTIVATION_BLOCK is set, an epoch boundary at the
       frozen finney compose's epoch interval, and after every earlier
       compiled block;
    2. list: a signed lab and host list ships, verifies under the shipped
       (pinned) signers, is not the draft (the draft is removed), and every
       entry of the pinned seed passes it (the seed is rebuilt under the
       signed list, LIVE_MENU_V4.md 17, item 4; the seed itself names no
       list, or the one shipped); until a list ships, the draft admits every
       seed entry, so the first signed list cannot leave the menu under
       MENU_MIN_ENTRIES (`_seed_strand_blockers`); every seed entry is
       within the price ceiling;
    3. pinned: every value a v4 validator refuses to start without is pinned
       (the proposals' category floor, a summary window the menu step
       accepts, a commitment that fits), and the memorisation check's
       published-set seed ships, matches its pin, and is observed through an
       epoch no more than PUBLISHED_V4_SEED_MAX_AGE_EPOCHS before the
       activation epoch and no later than its read lag allows (the nodes
       fold the public reveals after it; see the pin's comment);
    4. compose: the newest frozen finney app-compose is what the live compose
       is, runs the miner on protocol v4 at the compiled block when its own
       environment is all the miner reads (the probe runs the miner's
       selection), sets every compiled activation-block variable the code
       reads to the compiled value, and sets nothing the compose checks
       forbid (no allowed_envs, the image by digest, no upstream, no
       operator's own v4 option), and every operator's own v4 option has a
       path into a TD over provisioning (`_td_option_gaps`);
    5. image: every file v4 reads (consensus_v4.PINNED_FILES, the shipped
       list) is in the package the image copies and not excluded from it;
    6. root: the root's hotkey table for v4 commitments covers the first v4
       epoch.

    `root`, `cfg`, `signers`, `probe`, `published_pin` and `signers_pin` exist for the tests."""
    import fnmatch
    import sys

    sys.path.insert(0, str(ROOT))
    from fugal_subnet import consensus_v4
    from fugal_subnet import menu_rules as mr
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch, validate_activation_block

    if cfg is None:
        from fugal_subnet import config as cfg
    root = ROOT if root is None else Path(root)
    probe = probe or _miner_selection
    out: list[str] = []
    package = root / "fugal_subnet"
    block = getattr(cfg, "FINNEY_V4_ACTIVATION_BLOCK", None)
    block_given = block
    if block is not None and type(block) is not int:
        out.append(f"block: FINNEY_V4_ACTIVATION_BLOCK={block!r} is not an integer block number")
        block = None

    # The release's frozen finney compose: the newest by its tag.
    frozen = sorted((root / "deploy" / "dstack").glob("app-compose.frozen-*-finney.json"))
    compose_doc, compose_env = None, {}
    if frozen:
        try:
            compose_doc = json.loads(frozen[-1].read_text(encoding="utf-8"))
            compose_env = _compose_environment(compose_doc.get("docker_compose_file", ""))
        except ValueError:
            compose_doc = None
    try:
        bpe = blocks_per_epoch(int(compose_env.get("FUGAL_EPOCH_INTERVAL")
                                   or getattr(cfg, "EPOCH_INTERVAL", 3600)))
    except (TypeError, ValueError):
        bpe = None
    first_epoch = block // bpe if block is not None and bpe else None

    # 1. the block
    if block is None and block_given is None:
        out.append("block: FINNEY_V4_ACTIVATION_BLOCK is not set (config.py); choose an epoch "
                   "boundary after every earlier block and compile it")
    elif block is None:
        pass                                       # named above: not an integer
    else:
        try:
            validate_activation_block(int(block), bpe)
        except (TypeError, ValueError) as e:
            out.append(f"block: FINNEY_V4_ACTIVATION_BLOCK={block} is not an epoch boundary ({e})")
        for finney_name in sorted({f for f, _v in _chain_block_variables(
                (package / "config.py").read_text(encoding="utf-8"))}
                | {"FINNEY_V2_ACTIVATION_BLOCK"}):
            earlier = getattr(cfg, finney_name, None)
            if finney_name != "FINNEY_V4_ACTIVATION_BLOCK" and earlier is not None \
                    and int(block) <= int(earlier):
                out.append(f"block: FINNEY_V4_ACTIVATION_BLOCK={block} is not after "
                           f"{finney_name}={earlier}")

    # 2. the signed lab and host list
    lists_dir = package / "signed_lists"
    name = mr.SHIPPED_LISTS[mr.KIND_MENU_ALLOWLIST]
    labs = None
    pinned_signers = mr.LIST_SIGNERS_SHA256 if signers_pin is None else signers_pin
    shipped_signers = lists_dir / "allowed_signers"
    if not shipped_signers.is_file() or hashlib.sha256(
            shipped_signers.read_bytes()).hexdigest() != pinned_signers:
        out.append("list: fugal_subnet/signed_lists/allowed_signers is not the signers file "
                   "menu_rules.LIST_SIGNERS_SHA256 pins: the signers are a trust root, so a list "
                   "verified under another file proves nothing")
    try:
        labs = mr.shipped_list(mr.KIND_MENU_ALLOWLIST, directory=lists_dir,
                               signers=signers if signers is not None
                               else (lists_dir / "allowed_signers").read_bytes())
    except FileNotFoundError:
        out.append(f"list: no signed lab and host list ships (fugal_subnet/signed_lists/{name} "
                   "and its .sig); sign the reviewed draft (scripts/signed_list.py) and pin it")
    except mr.ListRefused as e:
        out.append(f"list: fugal_subnet/signed_lists/{name} does not verify under the shipped "
                   f"signers ({e})")
    if (root / "data" / "menu_allowlist.draft.json").exists() and labs is not None:
        out.append("list: data/menu_allowlist.draft.json still ships beside the signed list")
    seed_path = package / "routing_data" / "menu_v4_seed.json"
    try:
        seed_doc = json.loads(seed_path.read_text(encoding="ascii"))
        seed_labs, seed_entries = seed_doc["allowlists"]["labs"], seed_doc["entries"]
    except (OSError, ValueError, KeyError, TypeError):
        seed_labs = seed_entries = None
        out.append("list: the pinned seed (routing_data/menu_v4_seed.json) cannot be read")
    if seed_labs is not None and seed_labs.get("serial") and (labs is None or (
            seed_labs.get("serial"), seed_labs.get("sha256")) != (labs.serial, labs.sha256)):
        out.append(f"list: the pinned seed names list serial {seed_labs.get('serial')}, which "
                   "this release does not ship")
    if labs is not None and seed_entries is not None:
        refusal = mr.check_menu(seed_entries, labs)
        if refusal is not None:
            out.append(f"list: the pinned seed is not checked under the shipped list "
                       f"({refusal.code}: {refusal.detail}); rebuild and re-pin it under the "
                       "signed list (LIVE_MENU_V4.md 17, item 4)")
    out.extend(_seed_strand_blockers(root, labs, seed_entries))

    # 3. what a v4 validator refuses to start without
    floor = getattr(cfg, "PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4", None)
    if not (type(floor) is int and 0 < floor <= 10_000):
        out.append(f"pinned: PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4={floor!r} is not a pinned "
                   "share in basis points; a v4 validator refuses to start without it")
    else:
        from fugal_subnet import epoch_plan, menu_inputs
        try:
            menu_inputs.check_summary_window(getattr(cfg, "CHAMPION_WINDOW_EPOCHS", 48),
                                             epoch_plan.menu_step_params_v4(floor))
        except ValueError as e:
            out.append(f"pinned: the menu step refuses the summary window ({e})")
    if first_epoch is not None:
        from fugal_subnet import menu_publish as mp
        from fugal_subnet.benchmarks.slicer import epoch_id_for_block
        for why in mp.startup_refusals(chain_hold=0, reveal_delay=0, root=False,
                                       root_hotkey="release", epoch_id=epoch_id_for_block(
                                           first_epoch)):
            out.append(f"pinned: {why}")
    published = package / "routing_data" / "published_v4.json"
    if not published.exists():
        out.append("pinned: the published-set seed (routing_data/published_v4.json, the "
                   "memorisation check) is not in this tree")
    else:
        pin = PINNED_PUBLISHED_V4_SHA256 if published_pin is None else published_pin
        raw = published.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            out.append("pinned: the published-set seed does not match PINNED_PUBLISHED_V4_SHA256; "
                       "rebuild it (scripts/build_published_v4.py) and re-pin it")
        from fugal_subnet import memo_v4
        through = None
        try:
            seed_doc = json.loads(raw)
            memo_v4.check_seed(seed_doc)             # what the validator's own reader accepts
            through = memo_v4.seed_through(seed_doc)
        except (ValueError, KeyError, TypeError) as e:
            out.append(f"pinned: the published-set seed is not a seed the validator reads ({e})")
        if through is None:
            pass
        elif first_epoch is not None:
            oldest = first_epoch - PUBLISHED_V4_SEED_MAX_AGE_EPOCHS
            newest = first_epoch - memo_v4.READ_LAG
            if through < oldest:
                out.append(f"pinned: the published-set seed is observed through e{through:08d}, "
                           f"{first_epoch - through} epochs before the activation epoch "
                           f"e{first_epoch:08d}, more than {PUBLISHED_V4_SEED_MAX_AGE_EPOCHS}; "
                           "re-observe, rebuild and re-pin it")
            elif through > newest:
                out.append(f"pinned: the published-set seed is observed through e{through:08d}, "
                           f"after e{newest:08d}, the newest reveal the first v4 epoch reads "
                           "(memo_v4.READ_LAG)")

    # 4. the release's frozen finney compose
    if compose_doc is None:
        out.append("compose: no readable deploy/dstack/app-compose.frozen-*-finney.json")
    else:
        rel = frozen[-1].relative_to(root)
        text = compose_doc.get("docker_compose_file", "")
        live = root / "deploy" / "dstack" / "docker-compose.yaml"
        if live.exists() and live.read_text(encoding="utf-8") != text:
            out.append(f"compose: deploy/dstack/docker-compose.yaml is not {rel}'s compose; "
                       "freeze the release's compose")
        if compose_doc.get("allowed_envs"):
            out.append(f"compose: {rel} has allowed_envs (unmeasured variables)")
        if not re.search(r"^\s*image:\s*\S+@sha256:[0-9a-f]{64}\s*$", text, re.M):
            out.append(f"compose: {rel} does not pin its image by digest")
        if "FUGAL_OPENROUTER_BASE" in compose_env:
            out.append(f"compose: {rel} sets FUGAL_OPENROUTER_BASE")
        if "FUGAL_EPOCH_INTERVAL" in compose_env \
                and compose_env["FUGAL_EPOCH_INTERVAL"] != str(FINNEY_EPOCH_INTERVAL_S):
            out.append(f"compose: {rel} sets FUGAL_EPOCH_INTERVAL="
                       f"{compose_env['FUGAL_EPOCH_INTERVAL']}, but finney's epoch is "
                       f"{FINNEY_EPOCH_INTERVAL_S} s (a 360-block tempo at 12 s a block, the "
                       "validators' own setting), so the miner and the validators would put "
                       "every epoch boundary at different blocks")
        operator = _miner_operator_variables(
            (root / "neurons" / "miner.py").read_text(encoding="utf-8"))
        for var in sorted(operator & set(compose_env)):
            out.append(f"compose: {rel} sets {var}, one operator's own choice, in the shared "
                       "compose every miner's hash names")
        gaps = _td_option_gaps(root, operator - set(V4_OPTIONS_WITH_IMAGE_DEFAULT))
        if gaps:
            out.append("compose: a miner in a TD cannot set " + ", ".join(sorted(gaps)) + ": the "
                       "frozen compose may not carry one operator's choice, and its path over "
                       "provisioning is incomplete (" + "; ".join(
                           f"{var}: {', '.join(why)}" for var, why in sorted(gaps.items())) + ")")
        config_source = (package / "config.py").read_text(encoding="utf-8")
        for finney_name, var in _protocol_selection_variables(config_source):
            compiled = getattr(cfg, finney_name, None) if finney_name else None
            if var not in compose_env:
                out.append(f"compose: {rel} does not set {var}, which the miner's protocol "
                           "selection reads" + (f" (the release compiles {finney_name}="
                                                f"{compiled})" if compiled is not None else ""))
            elif compiled is not None and compose_env[var] != str(compiled):
                out.append(f"compose: {rel} sets {var}={compose_env[var]}, but the release "
                           f"compiles {finney_name}={compiled}")
        for finney_name, var in _chain_block_variables(config_source):
            compiled = getattr(cfg, finney_name, None)
            if var in compose_env and compose_env[var] != str(compiled):
                out.append(f"compose: {rel} sets {var}={compose_env[var]}, but the release "
                           f"compiles {finney_name}={compiled} (the miner refuses it)")
        got = probe(root, compose_env)
        if "error" in got:
            out.append(f"compose: the miner refuses to start under {rel}'s environment "
                       f"({got['error']})")
        elif got.get("protocol") != 4 or (block is not None and got.get("v4_block") != block):
            out.append(f"compose: under {rel}'s environment the miner serves protocol "
                       f"{got.get('protocol')} (v4 block {got.get('v4_block')}), not protocol 4 "
                       f"from {block}")

    # 5. the files v4 reads, in the image
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8") if (
        root / "Dockerfile").exists() else ""
    if not re.search(r"^COPY\s+fugal_subnet/\s+fugal_subnet/", dockerfile, re.M):
        out.append("image: the Dockerfile does not copy fugal_subnet/ into the image")
    ignore = root / ".dockerignore"
    patterns = [ln.strip() for ln in (ignore.read_text(encoding="utf-8").splitlines()
                                      if ignore.exists() else ())
                if ln.strip() and not ln.strip().startswith("#")]
    wanted = {rel for rel, required in consensus_v4.PINNED_FILES.values() if required}
    wanted |= {f"signed_lists/{name}", f"signed_lists/{name}.sig",
               "routing_data/published_v4.json"}
    for rel in sorted(wanted):
        path = f"fugal_subnet/{rel}"
        if not (root / path).is_file():
            if rel.startswith("signed_lists/menu_allowlist") or rel.endswith("published_v4.json"):
                out.append(f"image: {path} is not in the tree, so not in the image")
            else:
                out.append(f"image: {path} (a pinned v4 file) is missing")
        elif any(fnmatch.fnmatch(path, p.rstrip("/")) or path.startswith(p.rstrip("/") + "/")
                 for p in patterns):
            out.append(f"image: {path} is excluded by .dockerignore")

    # 6. the root's hotkeys for the v4 commitments
    table = getattr(cfg, "FINNEY_ROOT_HOTKEYS_V4", ())
    why = _root_table_problems(table)
    if why:
        out.extend(f"root: FINNEY_ROOT_HOTKEYS_V4 {w}" for w in why)
    elif first_epoch is not None and table[0][0] > first_epoch:
        out.append(f"root: FINNEY_ROOT_HOTKEYS_V4 starts at epoch {table[0][0]}, after the first "
                   f"v4 epoch {first_epoch}")
    return out


def check_v4_release(errors: list[str]) -> None:
    errors.extend(f"v4 release gate: {why}" for why in v4_release_blockers())


def main(argv: list[str] | None = None) -> None:
    """Every check is a hard failure, except the launch gate: the live composes
    must match a frozen app-compose at launch (`--release`, the MAINNET_LAUNCH
    gate and the image release), but between releases a compose legitimately
    moves ahead of its freeze, so plain CI reports the drift as a warning
    instead of staying red until the next image is signed."""
    import argparse

    parser = argparse.ArgumentParser(description="Fugal safety invariants.")
    parser.add_argument("--release", action="store_true",
                        help="also fail on launch-gate drift (live composes not frozen)")
    parser.add_argument("--release-v4", action="store_true",
                        help="the gate for a release that schedules protocol v4 (implies "
                             "--release); --release runs it too once the v4 block is compiled")
    args = parser.parse_args(argv)
    import sys

    sys.path.insert(0, str(ROOT))
    from fugal_subnet import config as _config
    v4_gate = args.release_v4 or (args.release and _config.FINNEY_V4_ACTIVATION_BLOCK is not None)
    args.release = args.release or args.release_v4

    errors: list[str] = []
    launch_gate: list[str] = []
    check_np_load_calls(errors)
    check_runtime_annotations(errors)
    check_deserialize_contract(errors)
    check_pinned_grader(errors)
    check_pinned_code_grader(errors)
    check_code_questions_are_graded(errors)
    check_pinned_system_prompt_v3(errors)
    check_pool_v3_data_pinned(errors)
    check_pinned_arena_grader(errors)
    check_pool_v5_data_pinned(errors)
    check_routing_data_mirrors_data(errors)
    check_price_table_pinned(errors)
    check_price_table_v2_pinned(errors)
    check_menu_v3_pinned(errors)
    check_menu_v3b_pinned(errors)
    check_menu_v4_seed_pinned(errors)
    check_pool_mix_v4_pinned(errors)
    check_published_v4_pinned(errors)
    check_no_other_v4_menu_ships(errors)
    check_collateral_endpoint_explicit(errors)
    check_external_calls_bounded(errors)
    check_measured_registers_documented(errors)
    check_one_accuracy_definition_on_the_consensus_path(errors)
    check_tpm_trust_anchor(errors)
    check_tpm_dependency_pin(errors)
    check_signed_lists(errors)
    check_epoch_id_single_source(errors)
    check_paid_call_guards(errors)
    check_tee_safety(errors)
    check_upstream_is_inside_the_measurement(errors)
    check_live_requires_app_identity(errors)
    check_live_composes_are_frozen(errors if args.release else launch_gate)
    check_release_composes_target_v2(errors if args.release else launch_gate)
    check_finney_compose_carries_compiled_blocks(errors)
    check_v3_release_pins(errors if args.release else launch_gate)
    check_bootstrap_frame_pinned(errors)
    check_every_env_constant_classified(errors)
    check_v4_constants_in_digest(errors)
    check_documented_flags(errors)
    if v4_gate:
        check_v4_release(errors)
    if errors:
        raise SystemExit("Safety invariant check failed:\n- " + "\n- ".join(errors))
    for line in launch_gate:
        print(f"WARNING (launch gate, fails with --release): {line}")
    if not v4_gate:
        left = v4_release_blockers()
        print(f"Protocol v4 release gate: {len(left)} condition(s) open "
              "(not a failure; `--release-v4` lists them).")
    print("Safety invariants passed.")


if __name__ == "__main__":
    main()
