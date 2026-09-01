"""GPQA-Diamond benchmark loader — 198 graduate-level science questions."""
from __future__ import annotations

import hashlib

from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS

_CHOICE_COLS = [
    "Correct Answer", "Incorrect Answer 1",
    "Incorrect Answer 2", "Incorrect Answer 3",
]
_LABELS = ["A", "B", "C", "D"]


def _shuffle_choices(choices: list[str], seed: int) -> tuple[list[str], int]:
    """Deterministically shuffle choices, return (shuffled, correct_index)."""
    n = len(choices)
    indices = list(range(n))
    # Fisher-Yates with deterministic seed per question
    state = seed
    for i in range(n - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    shuffled = [choices[idx] for idx in indices]
    correct_pos = indices.index(0)
    return shuffled, correct_pos


def load() -> list[dict]:
    # NOTE: gated dataset — requires HF login + accepted terms (see VALIDATOR_GUIDE).
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train",
                      revision=DATASET_REVISIONS["Idavidrein/gpqa"])
    items = []
    for i, row in enumerate(ds):
        choices = [row[c] for c in _CHOICE_COLS if row.get(c)]
        qid = f"gpqa_{i:03d}"
        seed = int(hashlib.md5(qid.encode()).hexdigest()[:8], 16)
        shuffled, correct_pos = _shuffle_choices(choices, seed)
        choices_text = "\n".join(f"{_LABELS[j]}. {c}" for j, c in enumerate(shuffled))
        prompt = f"{row['Question']}\n\n{choices_text}\n\nAnswer with the letter only."
        items.append({
            "prompt": prompt,
            "gold": _LABELS[correct_pos],
            "grader_id": "letter_mcq",
            "benchmark": "gpqa",
            "question_id": qid,
            "metadata": {},
        })
    return items
