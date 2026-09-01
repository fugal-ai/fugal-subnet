"""HumanEval benchmark loader — 164 code generation problems."""
from __future__ import annotations

from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS


def load() -> list[dict]:
    ds = load_dataset("openai/openai_humaneval", split="test",
                      revision=DATASET_REVISIONS["openai/openai_humaneval"])
    items = []
    for row in ds:
        task_id = row["task_id"]
        entry = row["entry_point"]
        prompt_code = row["prompt"]
        test_code = row["test"]
        prompt = (
            f"Write a Python function to solve the following problem.\n\n"
            f"{prompt_code}\n\n"
            f"Complete the function. Return only the code."
        )
        canonical = row.get("canonical_solution", "")
        test_inputs, test_outputs = _extract_io_from_tests(
            test_code, entry, prompt_code + canonical
        )
        if test_inputs and test_outputs:
            items.append({
                "prompt": prompt,
                "gold": test_outputs,
                "grader_id": "exec_io",
                "benchmark": "humaneval",
                "question_id": task_id.replace("/", "_"),
                "metadata": {
                    "checker": {
                        "id": "exec_io",
                        "params": {"func": entry, "inputs": test_inputs},
                    },
                    "entry_point": entry,
                    "stub": prompt_code,
                    "test": test_code,
                },
            })
        else:
            items.append({
                "prompt": prompt,
                "gold": "",
                "grader_id": "exec_unittest",
                "benchmark": "humaneval",
                "question_id": task_id.replace("/", "_"),
                "metadata": {
                    "entry_point": entry,
                    "stub": prompt_code,
                    "test": test_code,
                },
            })
    return items


def _extract_io_from_tests(test_code: str, entry: str, full_solution: str) -> tuple:
    """Best-effort extraction of I/O pairs from HumanEval assert-based tests.

    Falls back to empty lists if extraction fails — the loader will use
    exec_unittest instead of exec_io for those problems.
    """
    try:
        import ast
        tree = ast.parse(test_code)
    except SyntaxError:
        return [], []

    inputs, outputs = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            call = node.test.left
            if isinstance(call, ast.Call) and len(node.test.comparators) == 1:
                try:
                    args = [ast.literal_eval(a) for a in call.args]
                    expected = ast.literal_eval(node.test.comparators[0])
                    inputs.append(args)
                    outputs.append(expected)
                except (ValueError, TypeError):
                    continue
    return inputs, outputs
