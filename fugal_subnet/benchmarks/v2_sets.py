"""Protocol v2 benchmark loaders (open licences only; no gated datasets).

Every v2 set is openly licensed and non-gated, because every miner embeds the pool text and every
validator rebuilds it from the pinned revisions; a gated set would need every operator to hold
access under terms that forbid redistribution. Licences are recorded per benchmark in
`V2_LICENCES` and written into the v2 pool manifest.

Categories (`config.BENCHMARK_CATEGORY_V2`): knowledge = MMLU-Pro + a 3,000-question MMLU subset;
math = MATH levels 3-5 + Omni-MATH integer items below difficulty 5; hard = SuperGPQA "hard"
(all 7,050) + AIME 1983-2025 + BeyondAIME + Omni-MATH integer items at difficulty >= 5,
weighted 50/50 between the multiple-choice set and the integer olympiad sets; instruction = IFEval (the mapped subset; the full verifier set is v2.1).

Deterministic: every subset is chosen by sha256 order of a stable id, never by load order.
"""
from __future__ import annotations

import hashlib
import re

from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS

LETTERS = "ABCDEFGHIJ"
MMLU_V2_SIZE = 3000
OMNI_HARD_DIFFICULTY = 5.0   # Omni-MATH's 1-10 scale; >= 5 is olympiad level (AIME and above)
INTEGER_HINT = "\n\nThe answer is an integer. End your response with the final answer."

V2_LICENCES = {
    "mmlupro": "MIT (TIGER-Lab/MMLU-Pro)",
    "mmlu": "MIT (cais/mmlu)",
    "math_l35": "MIT (EleutherAI/hendrycks_math)",
    "omnimath_mid": "Apache-2.0 (KbsdJames/Omni-MATH)",
    "supergpqa_hard": "ODC-BY (m-a-p/SuperGPQA)",
    "aime": "MIT (qq8933/AIME_1983_2024)",
    "aime25": "MIT (opencompass/AIME2025)",
    "beyondaime": "CC0-1.0 (ByteDance-Seed/BeyondAIME)",
    "omnimath_hard": "Apache-2.0 (KbsdJames/Omni-MATH)",
    "ifeval": "Apache-2.0 (google/IFEval)",
}


def _integer_gold(raw) -> str:
    """An AIME-style integer answer, tolerating decoration such as `336^\\circ`; anything that
    does not reduce to exactly one integer is refused rather than guessed."""
    found = re.findall(r"-?\d+", str(raw).replace(",", ""))
    if len(found) != 1:
        raise ValueError(f"not a single integer answer: {raw!r}")
    return str(int(found[0]))


# graders.integer_exact compares through float(), which is exact only up to
# 2**53: a larger gold can never be matched, by any reply.
_GRADEABLE_INTEGER = 2 ** 53


def _gradeable_integer(gold) -> bool:
    """A gold integer_exact can match: one plain integer within float range."""
    return bool(re.fullmatch(r"-?\d+", str(gold))) and abs(int(gold)) <= _GRADEABLE_INTEGER


def _by_hash(items: list[dict], key: str, n: int) -> list[dict]:
    """The first n items in sha256 order of `key`: a stable subset independent of load order."""
    return sorted(items, key=lambda it: hashlib.sha256(str(it[key]).encode("utf-8")).hexdigest())[:n]


def _mcq_prompt(question: str, options: list[str]) -> str:
    choices = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(options))
    return f"{question}\n\n{choices}\n\nAnswer with the letter only."


def load_mmlupro() -> list[dict]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", revision=DATASET_REVISIONS["TIGER-Lab/MMLU-Pro"])
    out = []
    for row in ds:
        opts = list(row["options"])[:10]
        if not opts or row["answer"] not in LETTERS[:len(opts)]:
            continue
        out.append({"prompt": _mcq_prompt(row["question"], opts), "gold": row["answer"],
                    "grader_id": "letter_mcq", "benchmark": "mmlupro",
                    "question_id": f"mmlupro_{int(row['question_id']):05d}",
                    "metadata": {"category": row.get("category", "")}})
    return out


def load_mmlu_v2() -> list[dict]:
    from fugal_subnet.benchmarks import mmlu
    return _by_hash(mmlu.load(), "question_id", MMLU_V2_SIZE)


def load_math_l35() -> list[dict]:
    from fugal_subnet.benchmarks import math_
    keep = {"Level 3", "Level 4", "Level 5"}
    return [{**q, "benchmark": "math_l35"} for q in math_.load() if q["metadata"].get("level") in keep]


def _omni_integer() -> list[dict]:
    ds = load_dataset("KbsdJames/Omni-MATH", split="test", revision=DATASET_REVISIONS["KbsdJames/Omni-MATH"])
    out = []
    for i, row in enumerate(ds):
        ans = str(row["answer"]).strip().strip("$").strip()
        if not re.fullmatch(r"-?\d+", ans):
            continue
        if abs(int(ans)) > _GRADEABLE_INTEGER:
            continue                          # integer_exact compares through float: never gradeable
        try:
            diff = float(row["difficulty"])
        except (TypeError, ValueError):
            continue
        out.append({"prompt": row["problem"] + INTEGER_HINT, "gold": str(int(ans)), "grader_id": "integer_exact",
                    "question_id": f"omnimath_{i:05d}", "metadata": {"difficulty": diff, "domain": str(row["domain"])[:80]}})
    return out


def load_omnimath_mid() -> list[dict]:
    return [{**q, "benchmark": "omnimath_mid"} for q in _omni_integer() if q["metadata"]["difficulty"] < OMNI_HARD_DIFFICULTY]


def load_omnimath_hard() -> list[dict]:
    return [{**q, "benchmark": "omnimath_hard"} for q in _omni_integer() if q["metadata"]["difficulty"] >= OMNI_HARD_DIFFICULTY]


def load_supergpqa_hard() -> list[dict]:
    ds = load_dataset("m-a-p/SuperGPQA", split="train", revision=DATASET_REVISIONS["m-a-p/SuperGPQA"])
    hard = []
    for row in ds:
        if row["difficulty"] != "hard":
            continue
        opts = list(row["options"])[:10]
        if row["answer_letter"] not in LETTERS[:len(opts)]:
            continue
        hard.append({"prompt": _mcq_prompt(row["question"], opts), "gold": row["answer_letter"],
                     "grader_id": "letter_mcq", "benchmark": "supergpqa_hard",
                     "question_id": f"supergpqa_{row['uuid']}", "metadata": {"field": row.get("field", "")}})
    return sorted(hard, key=lambda q: q["question_id"])   # all 7,050: the category weight, not pool size, sets its share


def load_aime_hist() -> list[dict]:
    """v1's AIME loader, keeping only golds the integer grader can match: one
    item's gold is "080 or 081 (both were accepted)", which no reply can
    equal. The v1 pool keeps it (it is pinned and must not change)."""
    from fugal_subnet.benchmarks import aime
    return [q for q in aime.load() if _gradeable_integer(q["gold"])]


def load_aime25() -> list[dict]:
    out = []
    for part in ("AIME2025-I", "AIME2025-II"):
        ds = load_dataset("opencompass/AIME2025", part, split="test", revision=DATASET_REVISIONS["opencompass/AIME2025"])
        for i, row in enumerate(ds):
            out.append({"prompt": row["question"], "gold": _integer_gold(row["answer"]),
                        "grader_id": "integer_exact", "benchmark": "aime25",
                        "question_id": f"aime25_{part[-2:].strip('-')}_{i + 1:02d}", "metadata": {"part": part}})
    return out


def load_beyondaime() -> list[dict]:
    ds = load_dataset("ByteDance-Seed/BeyondAIME", split="test", revision=DATASET_REVISIONS["ByteDance-Seed/BeyondAIME"])
    return [{"prompt": row["problem"] + INTEGER_HINT, "gold": _integer_gold(row["answer"]),
             "grader_id": "integer_exact", "benchmark": "beyondaime", "question_id": f"beyondaime_{i:03d}",
             "metadata": {}} for i, row in enumerate(ds)]


# IFEval constraints the immutable grader cannot judge the way IFEval means
# them (2026-11-08, measured on the real items): `bullet_lines` accepts only
# "- " bullets, but IFEval's number_bullet_lists prompts ask for markdown "* "
# bullets, which grade 0; `json_object` with an empty `required` passes only
# the literal "{}", so a real JSON answer grades 0. Items carrying either are
# questions every honest miner fails, so the v2 pool leaves them out; v1's pool
# is pinned and keeps them.
UNGRADEABLE_IFEVAL_KINDS = frozenset({"bullet_lines", "json_object"})


def _ifeval_gradeable(q: dict) -> bool:
    checker = (q.get("metadata") or {}).get("checker") or {}
    kinds = {c.get("kind") for c in (checker.get("params") or {}).get("constraints", [])}
    return not (kinds & UNGRADEABLE_IFEVAL_KINDS)


def load_ifeval() -> list[dict]:
    from fugal_subnet.benchmarks import ifeval
    return [q for q in ifeval.load() if _ifeval_gradeable(q)]


LOADERS_V2 = {
    "mmlupro": load_mmlupro, "mmlu": load_mmlu_v2,
    "math_l35": load_math_l35, "omnimath_mid": load_omnimath_mid,
    "supergpqa_hard": load_supergpqa_hard, "aime": load_aime_hist, "aime25": load_aime25,
    "beyondaime": load_beyondaime, "omnimath_hard": load_omnimath_hard,
    "ifeval": load_ifeval,
}
assert set(LOADERS_V2) == set(V2_LICENCES)
