"""GSM8K benchmark loader — 1,319 grade school math problems."""
from __future__ import annotations
import re
from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS


def _extract_answer(answer_text: str) -> str:
    m = re.search(r"####\s*(.+)", answer_text)
    if m:
        return m.group(1).strip().replace(",", "")
    return answer_text.strip()


def load() -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test",
                      revision=DATASET_REVISIONS["openai/gsm8k"])
    items = []
    for i, row in enumerate(ds):
        gold = _extract_answer(row["answer"])
        items.append({
            "prompt": row["question"],
            "gold": gold,
            "grader_id": "numeric_final",
            "benchmark": "gsm8k",
            "question_id": f"gsm8k_{i:04d}",
            "metadata": {},
        })
    return items
