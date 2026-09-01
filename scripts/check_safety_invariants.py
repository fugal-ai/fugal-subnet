"""Static checks for repository safety invariants that must never regress."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AXON_PROTOCOL_FILES = (
    ROOT / "neurons" / "miner.py",
    ROOT / "fugal_subnet" / "protocol.py",
    ROOT / "fugal_subnet" / "v2" / "protocol.py",
)
AXON_HANDLER_FILES = (ROOT / "fugal_subnet" / "v2" / "report_server.py",)
IMMUTABLE_V1_GRADER_SHA256 = (
    "895809dedf0d14c45d9ec046bcbec2f50a09fcf7d31d9996a178e35f3539c55f"
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
    for path in AXON_PROTOCOL_FILES + AXON_HANDLER_FILES:
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


def check_paid_call_guards(errors: list[str]) -> None:
    """Keep every production OpenRouter completion path explicit and tagged."""
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

            is_http_post = (
                isinstance(func, ast.Attribute)
                and func.attr == "post"
                and isinstance(func.value, ast.Name)
                and func.value.id == "httpx"
            )
            if is_http_post:
                lines = source.splitlines()
                context = "\n".join(lines[max(0, node.lineno - 5):node.lineno])
                if "[PAID ~$" not in context:
                    relative = path.relative_to(ROOT)
                    errors.append(
                        f"{relative}:{node.lineno}: paid HTTP call needs [PAID ~$X] annotation"
                    )


def check_immutable_v1_grader(errors: list[str]) -> None:
    path = ROOT / "fugal_subnet" / "graders.py"
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != IMMUTABLE_V1_GRADER_SHA256:
        errors.append(
            "fugal_subnet/graders.py: immutable v1 bytes changed; add a new grader "
            f"version instead (expected {IMMUTABLE_V1_GRADER_SHA256}, got {actual})"
        )


def check_inactive_v2_not_wired(errors: list[str]) -> None:
    """V2 package code must stay unreachable until a rollout changes this gate."""
    path = ROOT / "neurons" / "validator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "fugal_subnet.v2"
        ):
            errors.append(
                f"neurons/validator.py:{node.lineno}: inactive v2 code cannot be wired "
                "before a reviewed manifest activation"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("fugal_subnet.v2"):
                    errors.append(
                        f"neurons/validator.py:{node.lineno}: inactive v2 code cannot be "
                        "wired before a reviewed manifest activation"
                    )


def check_deserialize_contract(errors: list[str]) -> None:
    for path in AXON_PROTOCOL_FILES:
        if path.name == "miner.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        synapse_classes = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            is_synapse = any(
                (isinstance(base, ast.Attribute) and base.attr == "Synapse")
                or (isinstance(base, ast.Name) and base.id == "Synapse")
                for base in node.bases
            )
            if is_synapse:
                synapse_classes.append(node)
        if not synapse_classes:
            errors.append(f"{path.relative_to(ROOT)}: no bt.Synapse class found")
            continue
        for node in synapse_classes:
            found = False
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "deserialize":
                    found = True
                    returns = [child for child in ast.walk(item) if isinstance(child, ast.Return)]
                    if len(returns) == 1 and isinstance(returns[0].value, ast.Name) and returns[0].value.id == "self":
                        break
                    errors.append(
                        f"{path.relative_to(ROOT)}: {node.name}.deserialize() must return only self"
                    )
                    break
            if not found:
                errors.append(
                    f"{path.relative_to(ROOT)}: {node.name}.deserialize() was not found"
                )


def main() -> None:
    errors: list[str] = []
    check_np_load_calls(errors)
    check_runtime_annotations(errors)
    check_paid_call_guards(errors)
    check_immutable_v1_grader(errors)
    check_inactive_v2_not_wired(errors)
    check_deserialize_contract(errors)
    if errors:
        raise SystemExit("Safety invariant check failed:\n- " + "\n- ".join(errors))
    print("Safety invariants passed.")


if __name__ == "__main__":
    main()
