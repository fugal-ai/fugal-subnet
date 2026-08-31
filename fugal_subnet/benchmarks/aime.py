"""AIME benchmark loader — competition math with integer answers (0-999)."""
from __future__ import annotations
from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS


def load() -> list[dict]:
    ds = load_dataset("qq8933/AIME_1983_2024", split="train",
                      revision=DATASET_REVISIONS["qq8933/AIME_1983_2024"])
    items = []
    for i, row in enumerate(ds):
        question = row.get("Question", "")
        answer = str(row.get("Answer", ""))
        if not question or not answer:
            continue
        year = row.get("Year", "")
        part = row.get("Part", "I")
        try:
            prob_num = int(row.get("Problem Number", 0) or 0)
        except (TypeError, ValueError):
            prob_num = 0
        items.append({
            "prompt": question,
            "gold": answer,
            "grader_id": "integer_exact",
            "benchmark": "aime",
            "question_id": f"aime_{year}_{part}_{prob_num:02d}_{i:04d}",
            "metadata": {
                "year": year,
                "part": row.get("Part", ""),
            },
        })
    return items
