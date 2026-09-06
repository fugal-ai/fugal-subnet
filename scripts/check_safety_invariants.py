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
_GUARDED = {"fugal_subnet", "neurons"}
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


def check_collateral_endpoint_explicit(errors: list[str]) -> None:
    """Every DCAP collateral fetch must name its endpoint.

    `get_collateral_and_verify(quote)` with no url resolves, inside dcap-qvl,
    to Phala's PCCS. That is how every --live validator acquired a third-party
    availability and privacy dependency nobody chose, and how INVARIANTS came
    to record the collateral as coming from Intel: nothing in this repo named
    the endpoint, so nothing contradicted the assumption.

    Restoring the one-argument call would restore that silence, and it would
    look like a simplification in review. So it is checked rather than trusted:
    the endpoint is this project's decision and must appear in this project.
    """
    src = (ROOT / "fugal_subnet" / "tee" / "attestation.py").read_text(encoding="utf-8")

    # Both entry points, because the code moved from one to the other once
    # already. A check naming only the function that happens to be called today
    # stops protecting anything the moment someone switches back, and it does so
    # silently -- it keeps passing, having found nothing to inspect.
    seen = 0
    for fn, min_args in (("get_collateral_and_verify", 2), ("get_collateral", 2)):
        for call in re.findall(rf"(?<![\w.]){fn}\(([^)]*)\)", src):
            seen += 1
            args = [a.strip() for a in call.split(",") if a.strip()]
            if len(args) < min_args:
                errors.append(
                    f"fugal_subnet/tee/attestation.py calls {fn} with "
                    f"{len(args)} argument(s) ({call.strip()!r}) — the PCCS url "
                    "must be passed explicitly, or dcap-qvl silently substitutes "
                    "its own default and the collateral source becomes invisible"
                )
    if seen == 0:
        errors.append(
            "fugal_subnet/tee/attestation.py makes no recognised collateral call — "
            "this check inspects call sites by name, so a rename turns it into a "
            "no-op that still reports success; teach it the new name"
        )

    cfg = (ROOT / "fugal_subnet" / "config.py").read_text(encoding="utf-8")
    if "TEE_PCCS_URL" not in cfg:
        errors.append(
            "config.TEE_PCCS_URL is missing — the collateral endpoint is a "
            "consensus-path dependency and must be pinned by this project "
            "rather than inherited from dcap-qvl's default"
        )


def check_external_calls_bounded(errors: list[str]) -> None:
    """I6. No external party may stop a validator completing an epoch.

    I6 used to say "no MINER behaviour", and that wording is what let the hang
    through: the collateral fetch is a call to a THIRD PARTY, so no reading of
    the old invariant was violated while a stalled PCCS blocked every validator
    in the field simultaneously -- no weights, and no log line either, because
    the code that records the failure sits downstream of the block.

    The lesson is that the invariant was wrong, not merely unenforced. A
    timeout on the one call that happened to be unbounded fixes today; this
    check is what stops the NEXT network call added inside the epoch loop from
    recreating it, and there will be one.

    Two things are enforced:

    1. Every `_run_coro` call passes a `timeout`. The helper runs a coroutine
       from sync code and defaults to unbounded, which is correct for a helper
       and fatal for a call on the epoch path.
    2. Every blocking network primitive passes a `timeout`. Each of these
       defaults to "wait forever" if the argument is omitted, and the omission
       is invisible at the call site -- it looks like every other call.

    Deliberately a syntactic check on argument presence, not a claim about the
    VALUE. A budget can be wrong; an absent budget is unbounded, and only the
    second kind produces a subnet that stops setting weights.
    """
    # Split by whether the NAME alone is evidence. `urlopen` and
    # `create_connection` mean one thing in Python; `get` and `post` are also
    # dict and mapping methods, so those need a receiver that looks like an
    # HTTP client before they count.
    #
    # They were one set for a commit, gated as a whole on the receiver. That
    # silently disabled the two unambiguous names in their normal
    # module-qualified form -- `urllib.request.urlopen(url)` has receiver
    # `request`, `socket.create_connection(addr)` has receiver `socket`, and
    # neither is in the client list. Recall measured at 2 of 6 planted calls.
    # The fix for a false positive had quietly created false negatives, which
    # is worse: the check kept reporting success while inspecting less.
    unambiguous_apis = {"urlopen", "create_connection"}
    ambiguous_apis = {
        "get", "post", "put", "head", "delete", "patch", "request",
    }
    unbounded_apis = unambiguous_apis | ambiguous_apis
    # Bare `get`/`post` are far too common as method names, so an attribute
    # call only counts when its receiver looks like an HTTP client.
    #
    # `s` was in this set for one commit, for `s = requests.Session()`. It
    # matched `s.get("accuracy", ...)` on a plain dict in epoch_logger instead.
    # Dropped deliberately: the cost of a false positive here is not noise, it
    # is that somebody eventually silences the whole check, and then the real
    # unbounded call lands unremarked. Precision over recall.
    #
    # KNOWN GAP, stated rather than papered over, and MEASURED rather than
    # guessed at. Against six deliberately planted unbounded calls this catches
    # four: `requests.get(url)`, `urlopen(url)`, `urllib.request.urlopen(url)`
    # and `socket.create_connection(addr)`. It misses the two that bind a
    # client to a name first -- `sess = requests.Session(); sess.get(url)` and
    # `httpx.Client().get(url)` -- because that needs assignment tracking.
    #
    # Four of six is worth having and is not worth mistaking for six of six.
    # I6 claims every call leaving the process on the epoch path is bounded and
    # names this check as the enforcement, so the distance between the claim
    # and the check belongs in writing, next to the check.
    http_receivers = {"requests", "httpx", "session", "client", "http"}

    for path in python_files():
        rel = path.relative_to(ROOT)
        if rel.parts[0] not in {"fugal_subnet", "neurons"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            has_timeout = any(k.arg == "timeout" for k in node.keywords)
            if has_timeout:
                continue

            if isinstance(node.func, ast.Name):
                name, receiver = node.func.id, None
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                receiver = (
                    node.func.value.id
                    if isinstance(node.func.value, ast.Name) else None
                )
            else:
                continue

            if name == "_run_coro":
                errors.append(
                    f"{rel}:{node.lineno} calls _run_coro without a timeout — "
                    "the helper is unbounded by default, so this is a call that "
                    "can block a validator's epoch forever (I6)"
                )
                continue
            if name not in unbounded_apis:
                continue
            if name in ambiguous_apis and isinstance(node.func, ast.Attribute) and (
                receiver is None or receiver.lower() not in http_receivers
            ):
                continue
            errors.append(
                f"{rel}:{node.lineno} calls {name}() with no timeout — a "
                "blocking network call on the validator's path waits forever by "
                "default, and one stalled upstream then halts every validator "
                "at once, because the collection point is a deterministic block "
                "(I6)"
            )

    cfg = (ROOT / "fugal_subnet" / "config.py").read_text(encoding="utf-8")
    if "TEE_COLLATERAL_TIMEOUT" not in cfg:
        errors.append(
            "config.TEE_COLLATERAL_TIMEOUT is missing — the collateral fetch is "
            "the one external call inside the epoch loop, and its bound is what "
            "keeps a stalled PCCS from halting the subnet (I6)"
        )


def check_measured_registers_documented(errors: list[str]) -> None:
    src = (ROOT / "fugal_subnet" / "tee" / "attestation.py").read_text(encoding="utf-8")
    body = re.search(
        r"def measurement_id\(.*?\n(.*?)(?=\n(?:def |class |@))", src, re.S)
    if body is None:
        errors.append("could not locate measurement_id() to derive its register set (I8)")
        return
    # The registers the CODE actually hashes.
    actual = {m.lower() for m in re.findall(r"quote\.(mrtd|rtmr\d)", body.group(1))}
    if not actual:
        errors.append("measurement_id() names no registers — cannot check docs against it (I8)")
        return

    # LIMITATION, found by this check firing on INVARIANTS.md prose that quoted
    # a wrong register set as an example of what not to write: it cannot tell an
    # assertion from an illustration. Documentation discussing a wrong set
    # should describe it rather than format it as a claim. Loosening the pattern
    # to fix that would cost more than it buys.
    #
    # Only claims of MEMBERSHIP are checked. Docs discuss RTMR0 and RTMR3 at
    # length to explain why they are EXCLUDED, and that prose must not trip.
    # Two assertion shapes, both anchored on MRTD appearing with the register
    # list, which is what a membership claim looks like:
    #   "sha256(MRTD || RTMR1 || RTMR2)"   explicit formula
    #   "(MRTD, RTMR0-2)"                  range form
    claim_re = re.compile(
        r"\(\s*MRTD\s*(?:[,‖|]|and)\s*RTMR\s*(\d)\s*(?:[-–]\s*(\d))?"
        r"(?:\s*(?:[,‖|]|and)\s*RTMR\s*(\d))?[^)]*\)", re.I)

    for rel in ("docs/MINER_GUIDE.md", "docs/INVARIANTS.md", "docs/VALIDATOR_GUIDE.md",
                "docs/design-decisions.md", "AGENTS.md", "README.md"):
        path = ROOT / rel
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in claim_re.finditer(line):
                lo, hi, extra = m.group(1), m.group(2), m.group(3)
                claimed = {"mrtd"}
                if hi:                      # a range: RTMR0-2
                    claimed |= {f"rtmr{i}" for i in range(int(lo), int(hi) + 1)}
                else:
                    claimed.add(f"rtmr{lo}")
                    if extra:
                        claimed.add(f"rtmr{extra}")
                if claimed != actual:
                    why = (
                        " RTMR0 is host-chosen, so a reader who believes it is "
                        "measured expects the approved list to fork by instance "
                        "size, and it does not."
                        if "rtmr0" in claimed - actual else ""
                    )
                    errors.append(
                        f"{rel}:{lineno} says the base measurement is "
                        f"{sorted(claimed)} but measurement_id() hashes "
                        f"{sorted(actual)} — a miner reading this predicts the "
                        f"wrong rejection.{why} (I8)"
                    )


def check_one_accuracy_definition_on_the_consensus_path(errors: list[str]) -> None:
    """I1. Every validator must divide by the same denominator.

    Two accuracy definitions exist in this repo and they differ by ~7 points:

      - `proof.accuracy` = `n_correct / len(scored_results)` — every scored
        question is in the denominator. THIS is what scoring reads, via
        `_proof_to_head_score`.
      - `head_eval.evaluate_head` drops questions no model in the pool answered
        correctly ("carry no routing signal", head_eval.py:203) before dividing.

    Both are defensible. Only one is consensus, and `evaluate_head` is not on
    that path at all — verified: it has zero callers under `fugal_subnet/` or
    `neurons/`, and every reference lives in tests, scripts or comments.

    So this guards a hazard rather than a bug. The trap is the NAME: it is the
    function you would reach for to reason about head scoring, and it is not
    the scoring function. Someone did reach for it and got a number 7% high
    with no warning — a reference model scoring 1.065 against itself, which the
    shipped definitions make impossible. A docstring would not have helped;
    docstrings are read by people already looking at the function, and the
    failure mode is someone grepping for a scoring function and finding one
    with the right name. Wiring it in would move every miner's apparent
    accuracy by ~7 points while reading as a refactor.

    NOTE FOR ANY RENAME. This check hardcodes the string "evaluate_head". If
    the function is renamed — `evaluate_head_offline` and
    `training_head_score` have both been suggested, and the name IS the trap —
    the string must move in the same commit, or this guard silently stops
    guarding. Deferred for now only because the rename would break a teammate's
    in-flight experiment, not because it is wrong.
    """
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.parts[0] not in _GUARDED or rel.parts[-1] == "head_eval.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            hit = (
                isinstance(node, ast.ImportFrom)
                and any(a.name == "evaluate_head" for a in node.names)
            ) or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "evaluate_head"
            ) or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "evaluate_head"
            )
            if hit:
                errors.append(
                    f"{rel}:{node.lineno} puts head_eval.evaluate_head on the "
                    f"consensus path. It uses a DIFFERENT accuracy denominator "
                    f"from proof.accuracy -- it drops questions no model "
                    f"answered, which inflated a measured score from 0.994 to "
                    f"1.065. Scoring must read accuracy from the proof "
                    f"(_proof_to_head_score), so that every validator divides "
                    f"by the same thing (I1)"
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
    check_collateral_endpoint_explicit(errors)
    check_external_calls_bounded(errors)
    check_measured_registers_documented(errors)
    check_one_accuracy_definition_on_the_consensus_path(errors)
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
