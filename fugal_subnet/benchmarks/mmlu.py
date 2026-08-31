"""MMLU benchmark loader — 57 subjects, 14,042 test questions."""
from __future__ import annotations
from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS

_CHOICES = ["A", "B", "C", "D"]


def load() -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split="test",
                      revision=DATASET_REVISIONS["cais/mmlu"])
    items = []
    for i, row in enumerate(ds):
        choices_text = "\n".join(
            f"{_CHOICES[j]}. {c}" for j, c in enumerate(row["choices"])
        )
        prompt = f"{row['question']}\n\n{choices_text}\n\nAnswer with the letter only."
        items.append({
            "prompt": prompt,
            "gold": _CHOICES[row["answer"]],
            "grader_id": "letter_mcq",
            "benchmark": "mmlu",
            "question_id": f"mmlu_{i:05d}",
            "metadata": {"subject": row.get("subject", "")},
        })
    return items
