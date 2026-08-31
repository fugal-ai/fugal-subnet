"""Static checks for repository safety invariants that must never regress."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FUTURE_IMPORTS = (
    ROOT / "neurons" / "miner.py",
    ROOT / "fugal_subnet" / "protocol.py",
)


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
    for path in FORBIDDEN_FUTURE_IMPORTS:
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
    path = ROOT / "fugal_subnet" / "protocol.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FugalSynapse":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "deserialize":
                    returns = [child for child in ast.walk(item) if isinstance(child, ast.Return)]
                    if len(returns) == 1 and isinstance(returns[0].value, ast.Name) and returns[0].value.id == "self":
                        return
                    errors.append(
                        "fugal_subnet/protocol.py: FugalSynapse.deserialize() must return only self"
                    )
                    return
    errors.append("fugal_subnet/protocol.py: FugalSynapse.deserialize() was not found")


def main() -> None:
    errors: list[str] = []
    check_np_load_calls(errors)
    check_runtime_annotations(errors)
    check_deserialize_contract(errors)
    if errors:
        raise SystemExit("Safety invariant check failed:\n- " + "\n- ".join(errors))
    print("Safety invariants passed.")


if __name__ == "__main__":
    main()
