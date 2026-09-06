"""Static checks for repository safety invariants that must never regress."""

from __future__ import annotations

import ast
import hashlib
import json
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
# The price table is the consensus cost denominator: every validator prices
# every proof against it, so an unreviewed edit silently re-scores the whole
# subnet. Pinned for the same reason graders.py is. Changing prices is a
# deliberate act — update this hash in the same commit and say why.
PINNED_PRICE_TABLE_SHA256 = (
    "7bc8e332a43833bb4eaccef21e784f57ffde9e49e343275f822f35d9a6d19b27"
)

# Google's EK/AK CA root, vendored at fugal_subnet/tee/roots/. Verifying a TPM
# quote means trusting this CA in addition to Intel's — see docs/INVARIANTS.md.
PINNED_TPM_ROOT_SHA256 = (
    "594759594b9b61524f6c8ef668f7177f7b9066bc0674f2f49fe9e052be78beb2"
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
    """Every Axon-attached Synapse must define deserialize() returning self.

    Checked against whatever Synapse subclasses protocol.py actually declares,
    rather than a hardcoded list — a hardcoded list silently stops covering a
    synapse that gets renamed, and silently fails on one that gets deleted.
    """
    path = ROOT / "fugal_subnet" / "protocol.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    synapse_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(b, ast.Attribute) and b.attr == "Synapse")
            or (isinstance(b, ast.Name) and b.id == "Synapse")
            for b in node.bases
        )
    ]
    if not synapse_names:
        errors.append("fugal_subnet/protocol.py: no bt.Synapse subclass found")
        return

    for synapse_name in synapse_names:
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


def check_epoch_id_single_source(errors: list[str]) -> None:
    """Both neurons must derive epoch identity from the same helper.

    The nonce is sha256(f"{epoch_id}:{block_hash}"), so if the miner and the
    validator format the identifier differently they derive different nonces,
    select different question slices, and every proof fails the nonce check.
    That shipped: the miner used f"e_{block_hash[:16]}" and the validator used
    f"e{epoch_index:08d}", their slices overlapped 45/300, and the subnet could
    not have set weights.

    A behavioural test cannot catch this — a test naturally derives the id once
    and passes it to both sides, which is exactly the divergence it needs to
    detect. So the invariant is structural: neither neuron may build an
    epoch_id itself.
    """
    for name in ("miner.py", "validator.py"):
        path = ROOT / "neurons" / name
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        if "epoch_id_for_block" not in source:
            errors.append(
                f"neurons/{name}: must derive epoch identity from "
                "slicer.epoch_id_for_block, the single source of truth"
            )
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "epoch_id" not in targets:
                continue
            # A locally constructed string is the bug; a call is the fix.
            if isinstance(node.value, (ast.JoinedStr, ast.Constant, ast.BinOp)):
                errors.append(
                    f"neurons/{name}:{node.lineno}: epoch_id is built locally. "
                    "Call slicer.epoch_id_for_block instead — a second "
                    "formatting of the epoch id is a consensus break"
                )


def check_tpm_dependency_pin(errors: list[str]) -> None:
    """`cryptography` is pinned in two extras and both copies must agree.

    It is in `dev` as well as `tee` because the TPM tests need it, but writing
    `fugal-subnet[tee]` would drag dcap-qvl into the dev environment and make
    the `test` and `tee-verifier` CI jobs identical — they differ on purpose.
    The cost of that choice is a duplicated version string, and this is what
    keeps it from drifting.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pins = set(re.findall(r'"cryptography==([0-9][^"]*)"', text))
    if not pins:
        errors.append(
            "pyproject.toml no longer pins cryptography — the TPM verifier is a "
            "consensus rule for --live validators on GCP and its library is "
            "pinned exactly, like dcap-qvl"
        )
    elif len(pins) > 1:
        errors.append(
            f"pyproject.toml pins cryptography at {sorted(pins)} in different "
            "places — validators would disagree on TPM verification depending on "
            "which extra they installed"
        )


def check_tpm_trust_anchor(errors: list[str]) -> None:
    """The pinned Google EK/AK root is a trust anchor, so it gets a hash gate.

    A swapped anchor does not fail loudly on its own: every forged TPM quote
    signed under the substituted CA would simply verify. That is the same
    failure shape as an edited grader, so it gets the same treatment.
    """
    path = ROOT / "fugal_subnet" / "tee" / "roots" / "google_ek_ak_root.der"
    if not path.exists():
        errors.append(
            "fugal_subnet/tee/roots/google_ek_ak_root.der is missing — TPM quotes "
            "on GCP cannot be chained to any trust anchor"
        )
        return

    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != PINNED_TPM_ROOT_SHA256:
        errors.append(
            f"google_ek_ak_root.der hash {actual[:16]}... does not match pin "
            f"{PINNED_TPM_ROOT_SHA256[:16]}... — this is the trust anchor for "
            "every TPM quote the subnet accepts; changing it accepts a different "
            "CA's attestations"
        )

    # The module's own constant must agree, or the runtime check is checking
    # something the CI gate never saw.
    tpm_src = (ROOT / "fugal_subnet" / "tee" / "tpm.py").read_text(encoding="utf-8")
    if f'_ROOT_SHA256 = "{PINNED_TPM_ROOT_SHA256}"' not in tpm_src:
        errors.append(
            "fugal_subnet/tee/tpm.py:_ROOT_SHA256 disagrees with the pin in this "
            "script — two pins that can drift are worse than one"
        )


def check_price_table_pinned(errors: list[str]) -> None:
    """The consensus price table must match its pin, and be well-formed."""
    path = ROOT / "data" / "models.json"
    if not path.exists():
        errors.append("data/models.json is missing — validators cannot price proofs")
        return

    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != PINNED_PRICE_TABLE_SHA256:
        errors.append(
            f"data/models.json: price table hash {actual[:16]}... does not match "
            f"pin {PINNED_PRICE_TABLE_SHA256[:16]}... — a price change re-scores "
            "every miner, so update the pin deliberately"
        )

    try:
        models = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"data/models.json is not valid JSON: {e}")
        return

    seen = set()
    for entry in models:
        mid = entry.get("id")
        if not mid:
            errors.append("data/models.json: an entry has no 'id'")
            continue
        if mid in seen:
            errors.append(f"data/models.json: duplicate model id {mid!r}")
        seen.add(mid)
        for key in ("in", "out"):
            price = entry.get(key)
            if not isinstance(price, (int, float)) or price < 0:
                errors.append(
                    f"data/models.json: {mid!r} has non-numeric or negative {key!r} price"
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


def check_tee_safety(errors: list[str]) -> None:
    """TEE-specific safety invariants."""
    # Miner must not use deferred annotations
    miner_path = ROOT / "neurons" / "miner.py"
    tree = ast.parse(miner_path.read_text(encoding="utf-8"), filename=str(miner_path))
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
        ):
            errors.append(
                "neurons/miner.py: deferred annotations break bt.Axon.attach()"
            )

    # TEE harness must use the same graders.py (import check)
    harness_path = ROOT / "fugal_subnet" / "tee" / "harness.py"
    if harness_path.exists():
        source = harness_path.read_text(encoding="utf-8")
        if "fugal_subnet.graders" not in source:
            errors.append(
                "fugal_subnet/tee/harness.py: must import from fugal_subnet.graders "
                "(hash-pinned grader ensures TEE grades match validator)"
            )

    # Verify module must not import or call models
    verify_path = ROOT / "fugal_subnet" / "tee" / "verify.py"
    if verify_path.exists():
        vtree = ast.parse(verify_path.read_text(encoding="utf-8"), filename=str(verify_path))
        forbidden = {"build_matrix", "call_model", "compute_hidden_states"}
        for node in ast.walk(vtree):
            if isinstance(node, ast.Name) and node.id in forbidden:
                errors.append(
                    f"fugal_subnet/tee/verify.py: references {node.id} — "
                    "verification must never call models"
                )


def main() -> None:
    errors: list[str] = []
    check_np_load_calls(errors)
    check_runtime_annotations(errors)
    check_deserialize_contract(errors)
    check_immutable_v1_grader(errors)
    check_price_table_pinned(errors)
    check_tpm_trust_anchor(errors)
    check_tpm_dependency_pin(errors)
    check_epoch_id_single_source(errors)
    check_paid_call_guards(errors)
    check_tee_safety(errors)
    check_documented_flags(errors)
    if errors:
        raise SystemExit("Safety invariant check failed:\n- " + "\n- ".join(errors))
    print("Safety invariants passed.")


if __name__ == "__main__":
    main()
