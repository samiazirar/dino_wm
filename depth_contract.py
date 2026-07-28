"""Fail-closed loading for the checkpoint-native DINOcular depth contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


NATIVE_DEPTH_SCHEMA = "dinocular-native-depth-contract-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DepthContractError(RuntimeError):
    """A required native-depth field, artifact, or hash is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    value = str(value).lower()
    if not SHA256_RE.fullmatch(value):
        raise DepthContractError(
            f"{label} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DepthContractError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DepthContractError(f"{label} must be a nonempty string")
    if value.strip().lower() in {"unknown", "todo", "tbd", "???"}:
        raise DepthContractError(f"{label} is unresolved")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DepthContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DepthContractError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class CacheToNativeBinding:
    environment: str
    producer_sha256: str
    wire_format_sha256: str
    wire_quantity: str
    wire_minimum: float
    wire_maximum: float
    raw_metric_scale: float
    raw_metric_offset: float
    raw_metric_minimum: float
    raw_metric_maximum: float
    interpolation: str
    payload_validation: str
    valid_value: float


@dataclass(frozen=True)
class NativeDepthContract:
    path: Path
    sha256: str
    manifest: Mapping[str, Any]
    checkpoint_sha256: str
    checkpoint_native_quantity: str
    checkpoint_native_units: str
    normalization_kind: str
    zero_intervention: str
    cache_bindings: tuple[CacheToNativeBinding, ...]

    def binding_for(
        self, producer_sha256: str, environment: str
    ) -> CacheToNativeBinding:
        producer_sha256 = require_sha256(
            producer_sha256, "selected cache producer SHA-256"
        )
        matches = [
            binding
            for binding in self.cache_bindings
            if binding.producer_sha256 == producer_sha256
            and binding.environment == str(environment)
        ]
        if len(matches) != 1:
            raise DepthContractError(
                "selected environment/cache producer has no unique wire-to-raw-metric binding"
            )
        return matches[0]


def load_native_depth_contract(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> NativeDepthContract:
    """Load and fully validate one immutable native student input manifest."""

    artifact = Path(path).expanduser().resolve()
    expected = require_sha256(expected_sha256, "native_depth_contract_sha256")
    if not artifact.is_file():
        raise DepthContractError(f"native depth contract does not exist: {artifact}")
    actual = sha256_file(artifact)
    if actual != expected:
        raise DepthContractError(
            f"native depth contract SHA-256 mismatch: expected {expected}, got {actual}"
        )
    try:
        manifest = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DepthContractError(f"cannot decode native depth contract: {exc}") from exc
    manifest = _object(manifest, "native depth contract")
    if manifest.get("schema") != NATIVE_DEPTH_SCHEMA:
        raise DepthContractError(
            f"unsupported native depth schema: {manifest.get('schema')!r}"
        )
    if manifest.get("status") != "complete":
        raise DepthContractError("native depth contract is incomplete")
    if manifest.get("self_acceptance_recorded") is not False:
        raise DepthContractError("native depth contract must not self-accept")
    if manifest.get("execution_authority_granted") is not False:
        raise DepthContractError("native depth contract must not grant execution authority")

    checkpoint = _object(manifest.get("checkpoint"), "checkpoint")
    checkpoint_sha256 = require_sha256(
        checkpoint.get("sha256"), "checkpoint.sha256"
    )
    for field in ("backend", "factory", "checkpoint_key", "state_prefix"):
        _text(checkpoint.get(field), f"checkpoint.{field}")
    if expected_checkpoint_sha256 is not None:
        configured = require_sha256(
            expected_checkpoint_sha256, "configured checkpoint_sha256"
        )
        if checkpoint_sha256 != configured:
            raise DepthContractError(
                "native contract checkpoint SHA-256 differs from the configured student"
            )

    producer = _object(manifest.get("producer"), "producer")
    for field in (
        "name",
        "model",
        "version",
        "code_commit",
        "temporal_mode",
        "raw_units",
    ):
        _text(producer.get(field), f"producer.{field}")
    require_sha256(producer.get("weight_sha256"), "producer.weight_sha256")
    for field in ("invocation", "preprocessing", "scale", "clipping"):
        value = _object(producer.get(field), f"producer.{field}")
        if not value:
            raise DepthContractError(f"producer.{field} must not be empty")

    encoder_input = _object(manifest.get("encoder_input"), "encoder_input")
    checkpoint_native = _object(
        encoder_input.get("checkpoint_native"), "encoder_input.checkpoint_native"
    )
    checkpoint_quantity = _text(
        checkpoint_native.get("quantity"), "checkpoint_native.quantity"
    )
    checkpoint_units = _text(
        checkpoint_native.get("units"), "checkpoint_native.units"
    )
    normalization = _object(
        checkpoint_native.get("normalization"),
        "encoder_input.checkpoint_native.normalization",
    )
    if normalization.get("kind") != "none_raw_metric":
        raise DepthContractError(
            "encoder_input.normalization.kind must be none_raw_metric"
        )
    if set(normalization) != {"kind"}:
        raise DepthContractError(
            "none_raw_metric normalization must not define mean/std or other transforms"
        )
    zero_depth = _object(encoder_input.get("zero_depth"), "encoder_input.zero_depth")
    if zero_depth.get("payload_validation") != "same_as_informative_depth":
        raise DepthContractError("zero-depth must validate the informative payload first")
    if zero_depth.get("intervention") != "exact_numeric_zero_at_encoder_boundary":
        raise DepthContractError("zero-depth intervention must be exact numeric zero")
    if zero_depth.get("learned_neutrality_claimed") is not False:
        raise DepthContractError("zero-depth must not claim learned neutrality")
    if zero_depth.get("rgb_equivalence_claimed") is not False:
        raise DepthContractError("zero-depth must not claim RGB equivalence")

    bindings_value = encoder_input.get("cache_bindings")
    if not isinstance(bindings_value, list) or not bindings_value:
        raise DepthContractError("encoder_input.cache_bindings must be a nonempty list")
    bindings = []
    for index, value in enumerate(bindings_value):
        label = f"encoder_input.cache_bindings[{index}]"
        value = _object(value, label)
        environment = _text(value.get("environment"), f"{label}.environment")
        if environment not in {"wall", "rope", "granular"}:
            raise DepthContractError(f"{label}.environment is unsupported")
        producer_sha = require_sha256(
            value.get("producer_sha256"), f"{label}.producer_sha256"
        )
        wire_sha = require_sha256(
            value.get("wire_format_sha256"), f"{label}.wire_format_sha256"
        )
        wire_quantity = _text(value.get("wire_quantity"), f"{label}.wire_quantity")
        wire_range = value.get("wire_range")
        if not isinstance(wire_range, list) or len(wire_range) != 2:
            raise DepthContractError(f"{label}.wire_range must contain two values")
        wire_minimum = _finite(wire_range[0], f"{label}.wire_range[0]")
        wire_maximum = _finite(wire_range[1], f"{label}.wire_range[1]")
        if not wire_maximum > wire_minimum:
            raise DepthContractError(f"{label}.wire_range is not increasing")
        affine = _object(
            value.get("affine_to_raw_metric"),
            f"{label}.affine_to_raw_metric",
        )
        if affine.get("operation") != "raw_metric=wire*scale+offset":
            raise DepthContractError(f"{label} has an unsupported affine operation")
        if affine.get("output_quantity") != checkpoint_quantity:
            raise DepthContractError(
                f"{label} affine output does not name the raw checkpoint quantity"
            )
        if affine.get("output_units") != checkpoint_units:
            raise DepthContractError(f"{label} affine output units differ")
        affine_scale = _finite(affine.get("scale"), f"{label}.affine.scale")
        affine_offset = _finite(affine.get("offset"), f"{label}.affine.offset")
        if affine_scale <= 0.0:
            raise DepthContractError(f"{label}.affine.scale must be positive")
        raw_minimum = affine_offset + wire_minimum * affine_scale
        raw_maximum = affine_offset + wire_maximum * affine_scale
        interpolation = _text(value.get("interpolation"), f"{label}.interpolation")
        if interpolation not in {
            "bilinear_align_corners_false",
            "identity_224x224",
        }:
            raise DepthContractError(f"{label}.interpolation is unsupported")
        validation = _object(
            value.get("payload_validation"), f"{label}.payload_validation"
        )
        validation_source = _text(
            validation.get("source"), f"{label}.payload_validation.source"
        )
        if validation_source != "reject_nonfinite_then_all_ones":
            raise DepthContractError(
                f"{label} uses unsupported payload validation {validation_source!r}"
            )
        valid_value = _finite(
            validation.get("valid_value"), f"{label}.payload_validation.valid_value"
        )
        if valid_value != 1.0:
            raise DepthContractError(f"{label} valid payload value must be exact 1")
        bindings.append(
            CacheToNativeBinding(
                environment=environment,
                producer_sha256=producer_sha,
                wire_format_sha256=wire_sha,
                wire_quantity=wire_quantity,
                wire_minimum=wire_minimum,
                wire_maximum=wire_maximum,
                raw_metric_scale=affine_scale,
                raw_metric_offset=affine_offset,
                raw_metric_minimum=raw_minimum,
                raw_metric_maximum=raw_maximum,
                interpolation=interpolation,
                payload_validation=validation_source,
                valid_value=valid_value,
            )
        )
    binding_keys = [
        (binding.environment, binding.producer_sha256) for binding in bindings
    ]
    if len(set(binding_keys)) != len(binding_keys):
        raise DepthContractError("cache bindings contain duplicate environment/producer pairs")

    return NativeDepthContract(
        path=artifact,
        sha256=actual,
        manifest=manifest,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_native_quantity=checkpoint_quantity,
        checkpoint_native_units=checkpoint_units,
        normalization_kind="none_raw_metric",
        zero_intervention="exact_numeric_zero_at_encoder_boundary",
        cache_bindings=tuple(bindings),
    )
