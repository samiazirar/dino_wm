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
    producer_sha256: str
    wire_format_sha256: str
    wire_quantity: str
    wire_minimum: float
    wire_maximum: float
    affine_scale: float
    affine_offset: float
    clip_minimum: float
    clip_maximum: float
    interpolation: str
    invalid_source: str
    valid_value: float
    invalid_value: float


@dataclass(frozen=True)
class NativeDepthContract:
    path: Path
    sha256: str
    manifest: Mapping[str, Any]
    checkpoint_sha256: str
    checkpoint_native_quantity: str
    checkpoint_native_units: str
    normalization_mean: float
    normalization_std: float
    neutral_normalized_depth: float
    neutral_validity_mask: float
    cache_bindings: tuple[CacheToNativeBinding, ...]

    def binding_for(self, producer_sha256: str) -> CacheToNativeBinding:
        producer_sha256 = require_sha256(
            producer_sha256, "selected cache producer SHA-256"
        )
        matches = [
            binding
            for binding in self.cache_bindings
            if binding.producer_sha256 == producer_sha256
        ]
        if len(matches) != 1:
            raise DepthContractError(
                "selected cache producer has no unique cache-wire to checkpoint-native binding"
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
    if manifest.get("status") != "complete" or manifest.get(
        "scientific_use_allowed"
    ) is not True:
        raise DepthContractError(
            "native depth contract is not complete and approved for scientific use"
        )

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
    if normalization.get("kind") != "affine_mean_std":
        raise DepthContractError(
            "encoder_input.normalization.kind must be affine_mean_std"
        )
    mean = _finite(normalization.get("mean"), "encoder_input.normalization.mean")
    std = _finite(normalization.get("std"), "encoder_input.normalization.std")
    if std <= 0:
        raise DepthContractError("encoder_input.normalization.std must be positive")
    neutral = _object(encoder_input.get("neutral"), "encoder_input.neutral")
    neutral_depth = _finite(
        neutral.get("normalized_depth"), "encoder_input.neutral.normalized_depth"
    )
    neutral_mask = _finite(
        neutral.get("validity_mask"), "encoder_input.neutral.validity_mask"
    )
    if neutral_mask not in {0.0, 1.0}:
        raise DepthContractError("neutral validity mask must be exactly 0 or 1")

    bindings_value = encoder_input.get("cache_bindings")
    if not isinstance(bindings_value, list) or not bindings_value:
        raise DepthContractError("encoder_input.cache_bindings must be a nonempty list")
    bindings = []
    for index, value in enumerate(bindings_value):
        label = f"encoder_input.cache_bindings[{index}]"
        value = _object(value, label)
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
            value.get("affine_to_checkpoint_native"),
            f"{label}.affine_to_checkpoint_native",
        )
        if affine.get("operation") != "checkpoint_native=wire*scale+offset":
            raise DepthContractError(f"{label} has an unsupported affine operation")
        if affine.get("output_quantity") != checkpoint_quantity:
            raise DepthContractError(
                f"{label} affine output does not name checkpoint-native quantity"
            )
        affine_scale = _finite(affine.get("scale"), f"{label}.affine.scale")
        affine_offset = _finite(affine.get("offset"), f"{label}.affine.offset")
        if affine_scale == 0.0:
            raise DepthContractError(f"{label}.affine.scale must be nonzero")
        clipping = _object(value.get("clipping"), f"{label}.clipping")
        if clipping.get("space") != "checkpoint_native_before_normalization":
            raise DepthContractError(f"{label}.clipping uses the wrong space")
        clip_minimum = _finite(clipping.get("minimum"), f"{label}.clipping.minimum")
        clip_maximum = _finite(clipping.get("maximum"), f"{label}.clipping.maximum")
        if not clip_maximum > clip_minimum:
            raise DepthContractError(f"{label}.clipping range is not increasing")
        interpolation = _text(value.get("interpolation"), f"{label}.interpolation")
        if interpolation not in {
            "bilinear_align_corners_false",
            "identity_224x224",
        }:
            raise DepthContractError(f"{label}.interpolation is unsupported")
        invalid = _object(value.get("invalid_mask"), f"{label}.invalid_mask")
        invalid_source = _text(invalid.get("source"), f"{label}.invalid_mask.source")
        if invalid_source != "all_finite_cache_values":
            raise DepthContractError(
                f"{label} uses unsupported invalid-mask source {invalid_source!r}"
            )
        valid_value = _finite(
            invalid.get("valid_value"), f"{label}.invalid_mask.valid_value"
        )
        invalid_value = _finite(
            invalid.get("invalid_value"), f"{label}.invalid_mask.invalid_value"
        )
        if valid_value != 1.0 or invalid_value != 0.0:
            raise DepthContractError(f"{label} validity values must be exact 1 and 0")
        bindings.append(
            CacheToNativeBinding(
                producer_sha256=producer_sha,
                wire_format_sha256=wire_sha,
                wire_quantity=wire_quantity,
                wire_minimum=wire_minimum,
                wire_maximum=wire_maximum,
                affine_scale=affine_scale,
                affine_offset=affine_offset,
                clip_minimum=clip_minimum,
                clip_maximum=clip_maximum,
                interpolation=interpolation,
                invalid_source=invalid_source,
                valid_value=valid_value,
                invalid_value=invalid_value,
            )
        )
    producer_hashes = [binding.producer_sha256 for binding in bindings]
    if len(set(producer_hashes)) != len(producer_hashes):
        raise DepthContractError("cache bindings contain duplicate producer hashes")

    return NativeDepthContract(
        path=artifact,
        sha256=actual,
        manifest=manifest,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_native_quantity=checkpoint_quantity,
        checkpoint_native_units=checkpoint_units,
        normalization_mean=mean,
        normalization_std=std,
        neutral_normalized_depth=neutral_depth,
        neutral_validity_mask=neutral_mask,
        cache_bindings=tuple(bindings),
    )
