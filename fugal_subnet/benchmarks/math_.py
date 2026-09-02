"""MATH benchmark loader — competition-level math problems (7 subjects).

The gold answer is the content of the LAST \\boxed{...} in the reference
solution (the Hendrycks MATH convention) — grading against the full solution
text would make the boxed_math string-match fallback unusable.
"""
from __future__ import annotations

from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS
from fugal_subnet.graders import last_boxed

_CONFIGS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def load() -> list[dict]:
    items = []
    idx = 0
    for config in _CONFIGS:
        ds = load_dataset(
            "EleutherAI/hendrycks_math", config, split="test",
            revision=DATASET_REVISIONS["EleutherAI/hendrycks_math"],
        )
        for row in ds:
            gold = last_boxed(row["solution"])
            if not gold:
                idx += 1
                continue
            items.append({
                "prompt": row["problem"],
                "gold": gold,
                "grader_id": "boxed_math",
                "benchmark": "math",
                "question_id": f"math_{config}_{idx:04d}",
                "metadata": {
                    "level": row.get("level", ""),
                    "type": row.get("type", config),
                },
            })
            idx += 1
    return items
