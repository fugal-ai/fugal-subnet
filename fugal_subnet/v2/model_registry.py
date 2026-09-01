"""Strict loader for the inactive canonical v2 model and price registry."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fugal_subnet.consensus_manifest import canonical_json, canonical_network

RESOURCE = "model-registry-v2.json"
MAX_ACTIVE_MODELS = 8
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}/[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$")


class ModelRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    provider: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    context_length: int
    enabled: bool
    review_status: str
    output_terms_evidence: str


@dataclass(frozen=True)
class ModelRegistry:
    registry_id: str
    status: str
    verified_at: str
    routing_reference_input_tokens: int
    routing_reference_output_tokens: int
    models: tuple[ModelSpec, ...]
    sha256: str

    @property
    def active_models(self) -> tuple[ModelSpec, ...]:
        return tuple(model for model in self.models if model.enabled)

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model.model_id for model in self.active_models)

    @property
    def prices_per_token(self) -> dict[str, tuple[Decimal, Decimal]]:
        million = Decimal(1_000_000)
        return {
            model.model_id: (
                model.input_usd_per_million / million,
                model.output_usd_per_million / million,
            )
            for model in self.active_models
        }

    @property
    def route_costs(self) -> dict[str, Decimal]:
        million = Decimal(1_000_000)
        return {
            model.model_id: (
                model.input_usd_per_million * self.routing_reference_input_tokens
                + model.output_usd_per_million * self.routing_reference_output_tokens
            ) / million
            for model in self.active_models
        }


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelRegistryError(f"duplicate model registry key: {key}")
        result[key] = value
    return result


def load_model_registry(
    *,
    require_active: bool = False,
    network: str | None = None,
    override_path: str | os.PathLike[str] | None = None,
) -> ModelRegistry:
    """Load the packaged registry or an explicitly local/mock-only override."""
    configured_override = override_path or os.getenv("FUGAL_MODEL_REGISTRY")
    if configured_override is not None:
        if network is None or canonical_network(network) != "local":
            raise ModelRegistryError("model registry overrides are local/mock-only")
        raw_bytes = Path(configured_override).read_bytes()
    else:
        raw_bytes = importlib.resources.files("fugal_subnet").joinpath(RESOURCE).read_bytes()
    try:
        raw = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelRegistryError("model registry is not valid UTF-8 JSON") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version", "registry_id", "status", "openrouter_pricing_verified_at",
        "routing_reference_input_tokens", "routing_reference_output_tokens", "models"
    }:
        raise ModelRegistryError("model registry root schema differs")
    if raw["schema_version"] != 1 or not isinstance(raw["models"], list):
        raise ModelRegistryError("model registry version or models is invalid")
    models = []
    ids = set()
    expected_keys = {
        "id", "provider", "canonical_input_usd_per_million",
        "canonical_output_usd_per_million", "context_length", "enabled",
        "review_status", "output_terms_evidence",
    }
    for item in raw["models"]:
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise ModelRegistryError("model entry schema differs")
        model_id = item["id"]
        if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id) or model_id in ids:
            raise ModelRegistryError("model IDs are invalid or duplicated")
        ids.add(model_id)
        try:
            input_price = Decimal(item["canonical_input_usd_per_million"])
            output_price = Decimal(item["canonical_output_usd_per_million"])
        except (InvalidOperation, TypeError) as exc:
            raise ModelRegistryError(f"invalid canonical price for {model_id}") from exc
        if not input_price.is_finite() or not output_price.is_finite() or input_price < 0 or output_price < 0:
            raise ModelRegistryError(f"invalid canonical price for {model_id}")
        if type(item["enabled"]) is not bool or not isinstance(item["context_length"], int):
            raise ModelRegistryError(f"invalid enabled/context values for {model_id}")
        if item["context_length"] <= 0:
            raise ModelRegistryError(f"invalid context length for {model_id}")
        if item["enabled"] and item["review_status"] != "approved":
            raise ModelRegistryError(f"enabled model lacks approved terms review: {model_id}")
        models.append(ModelSpec(
            model_id=model_id,
            provider=item["provider"],
            input_usd_per_million=input_price,
            output_usd_per_million=output_price,
            context_length=item["context_length"],
            enabled=item["enabled"],
            review_status=item["review_status"],
            output_terms_evidence=item["output_terms_evidence"],
        ))
    active = [model for model in models if model.enabled]
    if len(active) > MAX_ACTIVE_MODELS:
        raise ModelRegistryError("active model registry exceeds eight models")
    if require_active and not active:
        raise ModelRegistryError("v2 model registry has no approved active models")
    for field in ("routing_reference_input_tokens", "routing_reference_output_tokens"):
        if not isinstance(raw[field], int) or isinstance(raw[field], bool) or raw[field] <= 0:
            raise ModelRegistryError(f"{field} must be a positive integer")
    return ModelRegistry(
        registry_id=raw["registry_id"],
        status=raw["status"],
        verified_at=raw["openrouter_pricing_verified_at"],
        routing_reference_input_tokens=raw["routing_reference_input_tokens"],
        routing_reference_output_tokens=raw["routing_reference_output_tokens"],
        models=tuple(models),
        sha256=hashlib.sha256(canonical_json(raw)).hexdigest(),
    )
