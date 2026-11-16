"""Fresh success-policy evidence namespace. No legacy proof/state admission."""
from __future__ import annotations

import json
from pathlib import Path

from fugal_subnet.vendor import success_contract as contract

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "routing_data" / "benchmark_tokens_v1.json"
PRICES_PATH = ROOT / "routing_data" / "models.json"
PRICES_PATH_V2 = ROOT / "routing_data" / "models_v2.json"
MANIFEST_PATH_V2 = ROOT / "routing_data" / "benchmark_tokens_protocol_v2.json"
# Protocol v3's menu and token manifest. Not shipped until the v3 release pins
# them: every v3 accessor below fails by name while they are absent, and a
# validator with a v3 block scheduled refuses to start without them.
PRICES_PATH_V3 = ROOT / "routing_data" / "models_v3.json"
MANIFEST_PATH_V3 = ROOT / "routing_data" / "benchmark_tokens_protocol_v3.json"
BENCHMARK_LAMBDA = 1.0  # Consensus constant; never a miner environment setting.


# "fugal-testnet-success-v1" is a protocol name, not a network name. It is
# hashed into every reveal and into the bootstrap frame built from testnet
# reveals, so renaming it for mainnet, or changing models.json or the manifest
# between the warm-up and the launch, is a new identity that rejects that
# frame. The network is --network / FUGAL_NETWORK, never this string.
def identity():
    return contract.digest({"protocol": "fugal-testnet-success-v1", "contract": contract.CONTRACT,
                            "profile": contract.PROFILE_ID, "lambda": BENCHMARK_LAMBDA,
                            "tokens": contract.file_hash(MANIFEST_PATH),
                            "prices": contract.file_hash(PRICES_PATH)})


def benchmark_costs(head, require_reviewed=False):
    if head.success is None:
        raise ValueError("legacy preference heads cannot enter the success benchmark")
    manifest = json.loads(MANIFEST_PATH.read_text())
    contract.check_manifest(head.success, manifest, deployable=require_reviewed)
    rates = {r["id"]: (r["in"] / 1e6, r["out"] / 1e6) for r in json.loads(PRICES_PATH.read_text())}
    return contract.estimated_cost(head.success, rates)


def require_identity(value):
    if value != identity():
        raise ValueError("incompatible routing evidence namespace; archive old state and start fresh")


# --- Protocol v2: one versioned accessor for the model list -----------------
#
# A release/v2 validator scores v1 epochs until V2_ACTIVATION_BLOCK and v2
# epochs after it, so both price tables ship side by side and every caller asks
# for the table of the epoch it is scoring, keyed on the ONE version switch:
# slicer.protocol_version_for_boundary(boundary_block, V2_ACTIVATION_BLOCK).
# The global model index space (sorted ids) comes from here too, so a window
# never mixes the two index spaces (the v2 window starts empty at the flip).
# identity() above is deliberately unchanged: it is hashed into every v1 reveal
# and the v1 frame, and folding the v2 table into it would reject both
# mid-run. The v2 table's hash is in the fingerprint's epoch block instead.


def prices_path(version: int) -> Path:
    if version == 1:
        return PRICES_PATH
    if version == 2:
        return PRICES_PATH_V2
    if version == 3:
        return PRICES_PATH_V3
    raise ValueError(f"unknown protocol version {version!r}")


def _table(version: int) -> list[dict]:
    return json.loads(prices_path(version).read_text())


def price_table(version: int) -> dict[str, tuple[float, float]]:
    """{model_id: (price per input token, price per output token)} for this protocol."""
    return {r["id"]: (r["in"] / 1e6, r["out"] / 1e6) for r in _table(version)}


def models_for_protocol(version: int) -> list[str]:
    """The global model index space for this protocol: the sorted model ids."""
    return sorted(r["id"] for r in _table(version))


def reasoning_for(version: int, model_id: str) -> dict | None:
    """The single OpenRouter `reasoning` object pinned for this model (v2 only).

    The unified parameter accepts `effort` OR `max_tokens`, never both, and each
    provider honours one of them; the table pins exactly one per model. v1 sends
    none.
    """
    if version == 1:
        return None
    for r in _table(version):
        if r["id"] == model_id:
            return dict(r["reasoning"]) if r.get("reasoning") else None
    raise KeyError(f"{model_id!r} is not in the v{version} model list")


def manifest_path(version: int) -> Path:
    if version == 1:
        return MANIFEST_PATH
    if version == 2:
        return MANIFEST_PATH_V2
    if version == 3:
        return MANIFEST_PATH_V3
    raise ValueError(f"unknown protocol version {version!r}")


def identity_v2():
    """The v2 evidence namespace: the v2 token manifest and price table, under a new protocol
    name. A v2 reveal or frame cannot be mistaken for a v1 one, and v1's identity() is
    untouched."""
    return contract.digest({"protocol": "fugal-success-v2", "contract": contract.CONTRACT,
                            "profile": contract.PROFILE_ID, "lambda": BENCHMARK_LAMBDA,
                            "tokens": contract.file_hash(MANIFEST_PATH_V2),
                            "prices": contract.file_hash(PRICES_PATH_V2)})


def benchmark_costs_for(head, version: int, require_reviewed: bool = False):
    """benchmark_costs for the epoch's protocol: its token manifest and price table."""
    if version == 1:
        return benchmark_costs(head, require_reviewed=require_reviewed)
    if head.success is None:
        raise ValueError("legacy preference heads cannot enter the success benchmark")
    manifest = json.loads(manifest_path(version).read_text())
    contract.check_manifest(head.success, manifest, deployable=require_reviewed)
    return contract.estimated_cost(head.success, price_table(version))


# --- Protocol v3: a menu entry is a way of calling a model -------------------
#
# From v3 an entry's id is `<model>@<effort>` when a model is offered at more
# than one reasoning effort (`openai/gpt-6.1-sol@high`), and the plain model id
# when it is offered at one. The id is what a head has a row for, what the
# price table and the token manifest are keyed on, and what every reveal
# names. What goes on the wire is the entry's REQUEST: the provider's model
# id, the reasoning setting, the answer budget and the providers it may be
# served by. The request is part of the pinned table, so the subnet's harness
# and anything that serves the winning head send the same bytes for an entry.

REQUEST_FIELDS = ("model", "reasoning", "max_tokens", "provider")


def identity_v3():
    """The v3 evidence namespace: the v3 menu and token manifest."""
    return contract.digest({"protocol": "fugal-success-v3", "contract": contract.CONTRACT,
                            "profile": contract.PROFILE_ID, "lambda": BENCHMARK_LAMBDA,
                            "tokens": contract.file_hash(MANIFEST_PATH_V3),
                            "prices": contract.file_hash(PRICES_PATH_V3)})


def identity_for(version: int):
    return {1: identity, 2: identity_v2, 3: identity_v3}[int(version)]()


def api_model_of(entry_id: str) -> str:
    """The provider's model id of a menu entry id: the part before `@`."""
    return str(entry_id).split("@", 1)[0]


def request_for(version: int, entry_id: str, default_max_tokens: int) -> dict:
    """The request fields the harness sends for this menu entry.

    v2: the model id as it is, the pinned `reasoning` if any, the harness's
    answer budget. v3: the entry's own `model` (default: the id before `@`),
    `reasoning`, `max_tokens` (default: the harness's budget) and `provider`
    (an OpenRouter provider preference: the pinned order, no fallbacks).
    """
    if version == 1:
        return {"model": entry_id, "max_tokens": int(default_max_tokens)}
    for r in _table(version):
        if r["id"] != entry_id:
            continue
        out = {"model": str(r.get("model") or api_model_of(entry_id)) if version >= 3 else entry_id,
               "max_tokens": int(r.get("max_tokens") or default_max_tokens) if version >= 3
               else int(default_max_tokens)}
        if r.get("reasoning"):
            out["reasoning"] = dict(r["reasoning"])
        if version >= 3 and r.get("provider"):
            out["provider"] = dict(r["provider"])
        return out
    raise KeyError(f"{entry_id!r} is not in the v{version} model list")


def validate_menu_v3(rows: list) -> None:
    """Refuse a v3 menu that a harness could not call or a head could not name.

    Every row: a unique id of at most 256 characters; an id with `@` whose
    part before it is the row's `model` (or the row has no `model`);
    non-negative prices; at most one of `effort` / `max_tokens` / `enabled`
    in `reasoning` (OpenRouter's unified parameter takes one); a positive
    `max_tokens` if given; and a `provider` that pins an order and forbids
    fallbacks, so two runs of an entry cannot be served by different hardware.
    """
    seen: set[str] = set()
    if not isinstance(rows, list) or not rows:
        raise ValueError("the v3 menu must be a non-empty list of entries")
    for r in rows:
        eid = r.get("id") if isinstance(r, dict) else None
        if not isinstance(eid, str) or not eid.strip() or len(eid) > 256 or eid in seen:
            raise ValueError(f"menu entry id {eid!r} is missing, too long or repeated")
        seen.add(eid)
        if eid.count("@") > 1 or eid.startswith("@") or eid.endswith("@"):
            raise ValueError(f"{eid}: at most one '@', with a model before it and an effort after")
        if "model" in r and r["model"] != api_model_of(eid):
            raise ValueError(f"{eid}: `model` {r['model']!r} is not the id before '@'")
        for key in ("in", "out"):
            v = r.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not v >= 0:
                raise ValueError(f"{eid}: `{key}` must be a non-negative price per million tokens")
        reasoning = r.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict) or len(
                    [k for k in ("effort", "max_tokens", "enabled") if k in reasoning]) != 1:
                raise ValueError(f"{eid}: `reasoning` must set exactly one of effort, "
                                 "max_tokens, enabled")
        mt = r.get("max_tokens")
        if mt is not None and (isinstance(mt, bool) or not isinstance(mt, int) or mt < 1):
            raise ValueError(f"{eid}: `max_tokens` must be a positive integer")
        provider = r.get("provider")
        if provider is not None:
            order = provider.get("order") if isinstance(provider, dict) else None
            if (not isinstance(order, list) or not order
                    or not all(isinstance(x, str) and x for x in order)
                    or provider.get("allow_fallbacks") is not False):
                raise ValueError(f"{eid}: `provider` must pin an `order` and set "
                                 "`allow_fallbacks` to false")
