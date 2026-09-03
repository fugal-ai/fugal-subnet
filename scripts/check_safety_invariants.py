"""Static checks for repository safety invariants that must never regress."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AXON_PROTOCOL_FILES = (
    ROOT / "neurons" / "miner.py",
    ROOT / "fugal_subnet" / "protocol.py",
)
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
    path = ROOT / "fugal_subnet" / "protocol.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for synapse_name in ("FugalSynapse", "FugalProofSynapse"):
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


def check_immutable_v1_grader(errors: list[str]) -> None:
    path = ROOT / "fugal_subnet" / "graders.py"
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != IMMUTABLE_V1_GRADER_SHA256:
        errors.append(
            f"fugal_subnet/graders.py: grader hash {actual[:16]}... does not match "
            f"immutable v1 pin {IMMUTABLE_V1_GRADER_SHA256[:16]}... — "
            "a grader change is a consensus break"
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


def main() -> None:
    errors: list[str] = []
    check_np_load_calls(errors)
    check_runtime_annotations(errors)
    check_deserialize_contract(errors)
    check_immutable_v1_grader(errors)
    check_paid_call_guards(errors)
    check_documented_flags(errors)
    if errors:
        raise SystemExit("Safety invariant check failed:\n- " + "\n- ".join(errors))
    print("Safety invariants passed.")


if __name__ == "__main__":
    main()
