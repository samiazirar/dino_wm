"""Fail-closed PushT empirical/lossy depth-consumption contract primitives."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol

import torch
import yaml

from depth_contract import (
    DepthContractError,
    load_native_depth_contract,
    require_sha256,
    sha256_file,
)


EMPIRICAL_CONTRACT_SCHEMA = "dinocular-empirical-cache-consumption-contract-v1"
CONSUMPTION_INDEX_SCHEMA = "dino-wm-depth-consumption-index-v2"
RUNTIME_RELEASE_SCHEMA = "dino-wm-empirical-runtime-release-v1"
RUNTIME_RELEASE_ACCEPTANCE_SCHEMA = (
    "dino-wm-empirical-runtime-release-acceptance-v1"
)
EMPIRICAL_ADAPTER_ID = "mapanything_pusht_empirical_lossy_v1"
EMPIRICAL_ASSUMPTION = "[ASSUMPTION: RECOVERED-CONTRACT]"
EMPIRICAL_PROXY_SCALE = 1.5746406149864196
EMPIRICAL_NON_EQUIVALENCE = (
    "This is a clipped later-producer empirical proxy and constant-zero numeric "
    "intervention, not original-producer recovery, checkpoint-native or neutral depth, "
    "RGB-only input, physical/metric depth, or a MapAnything-versus-Depth-Anything-3 claim."
)

ACCEPTED_IDENTITIES = {
    "checkpoint_sha256": "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc",
    "producer_sha256": "6996aa719531feb09f8dc858e66c2b703dc14393533f2d6dc073fd98937fe4e5",
    "manifest_sha256": "9f35d303a5c604d5870ebdc6aedcefe9860bd0ab2763648c75b11e3b4f50b691",
    "manifest_id": "b87fe658-4731-4e8e-8c88-38f4fac6344c",
    "validation_sha256": "ed3d63388a81581d20d560266a0d5e210b6672144fcb2cbb6a2e7ffef5736df9",
    "data_sha256": "68fe99e9f566eadcaa61e092b042962060fdc82c39ee8e71dcd7b9ef05247adb",
    "source_index_sha256": "ed6eecd62e455ffd35551074787f36b6c08039776b58a9dc6f22d17266ead6bd",
    "wire_format_sha256": "47a6d5944af9f3587ee7ef0b8154d157294e91db7e489c06b5b33bd3d9384ded",
    "calibration_key_sha256": "40eb3b62c3f77bee5d4399c46c2efba0672c05ed03ecbca2d91c5106f308115d",
}

_CANONICAL_ROOT = "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm"
ACCEPTED_CANONICAL_PATHS = {
    "checkpoint": f"{_CANONICAL_ROOT}/checkpoints/dinov2_depthembed_dropout_fullpr.pth",
    "cache_directory": f"{_CANONICAL_ROOT}/data/depth_cache_mapanything_singleton/pusht.lmdb",
    "manifest": f"{_CANONICAL_ROOT}/data/depth_cache_mapanything_singleton/pusht.lmdb/manifest.json",
    "validation": f"{_CANONICAL_ROOT}/data/depth_cache_mapanything_singleton/validation.json",
    "data": f"{_CANONICAL_ROOT}/data/depth_cache_mapanything_singleton/pusht.lmdb/data.mdb",
}
def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


CAPSULE_RUNTIME_PATHS = {
    "checkpoint": "/opt/dinocular/models/dinocular_student.pth",
    "cache_directory": "/opt/dinocular/depth/pusht.lmdb",
    "manifest": "/opt/dinocular/depth/pusht.lmdb/manifest.json",
    "validation": "/opt/dinocular/depth/validation.json",
    "data": "/opt/dinocular/depth/pusht.lmdb/data.mdb",
    "empirical_contract": "/opt/dinocular/contracts/pusht_mapanything_empirical_v1.json",
    "empirical_runtime_release": "/opt/dinocular/releases/pusht_empirical_runtime_release_v1.json",
}


@dataclass(frozen=True)
class EmpiricalCanonicalPaths:
    checkpoint: str
    cache_directory: str
    manifest: str
    validation: str
    data: str


@dataclass(frozen=True)
class EmpiricalRuntimePaths:
    checkpoint: Path
    cache_directory: Path
    manifest: Path
    validation: Path
    data: Path


class EmpiricalRuntimeResolver(Protocol):
    """Resolve and validate only the closed empirical runtime path set."""

    def resolve(self, canonical: EmpiricalCanonicalPaths) -> EmpiricalRuntimePaths: ...

    def validate_open_path(
        self, path: Path, *, artifact: str, expect_directory: bool = False
    ) -> None: ...


def _validate_no_alias_path(
    path: Path, *, artifact: str, expect_directory: bool = False
) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise EmpiricalDepthContractError(f"{artifact} runtime path is not closed")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            raise EmpiricalDepthContractError(
                f"{artifact} runtime path is absent: {current}"
            ) from None
        if stat.S_ISLNK(mode):
            raise EmpiricalDepthContractError(
                f"{artifact} runtime path contains a symlink: {current}"
            )
    final_mode = os.lstat(path).st_mode
    valid_type = stat.S_ISDIR(final_mode) if expect_directory else stat.S_ISREG(final_mode)
    if not valid_type:
        expected = "directory" if expect_directory else "regular file"
        raise EmpiricalDepthContractError(f"{artifact} runtime path is not a {expected}")


@dataclass(frozen=True)
class CanonicalEmpiricalRuntimeResolver:
    """Open the exact accepted host paths; no aliases or relocation are permitted."""

    def resolve(self, canonical: EmpiricalCanonicalPaths) -> EmpiricalRuntimePaths:
        return EmpiricalRuntimePaths(
            checkpoint=Path(canonical.checkpoint),
            cache_directory=Path(canonical.cache_directory),
            manifest=Path(canonical.manifest),
            validation=Path(canonical.validation),
            data=Path(canonical.data),
        )

    def validate_open_path(
        self, path: Path, *, artifact: str, expect_directory: bool = False
    ) -> None:
        _validate_no_alias_path(
            path, artifact=artifact, expect_directory=expect_directory
        )


@dataclass(frozen=True)
class CapsuleEmpiricalRuntimeResolver:
    """Closed capsule binding; runtime locations are fixed rather than caller-mapped."""

    def resolve(self, canonical: EmpiricalCanonicalPaths) -> EmpiricalRuntimePaths:
        expected = EmpiricalCanonicalPaths(
            checkpoint=ACCEPTED_CANONICAL_PATHS["checkpoint"],
            cache_directory=ACCEPTED_CANONICAL_PATHS["cache_directory"],
            manifest=ACCEPTED_CANONICAL_PATHS["manifest"],
            validation=ACCEPTED_CANONICAL_PATHS["validation"],
            data=ACCEPTED_CANONICAL_PATHS["data"],
        )
        if canonical != expected:
            raise EmpiricalDepthContractError("capsule resolver received an unbound canonical path")
        return EmpiricalRuntimePaths(
            checkpoint=Path("/opt/dinocular/models/dinocular_student.pth"),
            cache_directory=Path("/opt/dinocular/depth/pusht.lmdb"),
            manifest=Path("/opt/dinocular/depth/pusht.lmdb/manifest.json"),
            validation=Path("/opt/dinocular/depth/validation.json"),
            data=Path("/opt/dinocular/depth/pusht.lmdb/data.mdb"),
        )

    def validate_open_path(
        self, path: Path, *, artifact: str, expect_directory: bool = False
    ) -> None:
        _validate_no_alias_path(
            path, artifact=artifact, expect_directory=expect_directory
        )


REQUIRED_INVALID_CLAIMS = {
    "reproduction_of_original_student_training_depth",
    "equality_to_unrecovered_original_mapanything_invocation",
    "checkpoint_native_input_equivalence",
    "lossless_recovery_of_raw_depth_z",
    "metric_depth_or_physical_unit_recovery",
    "native_neutral_or_rgb_only_equivalence",
    "mapanything_versus_depth_anything_3",
}
REQUIRED_IRREVERSIBILITIES = {
    "clipping",
    "float16_quantization_after_normalization",
    "bilinear_resampling_and_downsampling",
    "original_invocation_and_scale_not_recovered",
}
FORBIDDEN_FALLBACKS = {
    "isolated_frame_producer": "forbidden",
    "dynamic_reproduction": "forbidden",
    "alternate_cache": "forbidden",
    "missing_depth_to_zero": "forbidden",
    "rgb_only_substitution": "forbidden",
}


class EmpiricalDepthContractError(DepthContractError):
    """The empirical contract, index, adapter, or provenance is invalid."""


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EmpiricalDepthContractError(f"{label} must be an object")
    return value


def _exact_list(value: Any, expected: list[Any], label: str) -> None:
    if value != expected:
        raise EmpiricalDepthContractError(f"{label} must be exactly {expected!r}")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EmpiricalDepthContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EmpiricalDepthContractError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class EmpiricalDepthContract:
    path: Path
    sha256: str
    manifest: Mapping[str, Any]
    checkpoint_sha256: str
    producer_sha256: str
    manifest_sha256: str
    manifest_id: str
    validation_sha256: str
    data_sha256: str
    source_index_sha256: str
    wire_format_sha256: str
    canonical_paths: EmpiricalCanonicalPaths
    checkpoint_path: Path
    cache_directory: Path
    manifest_path: Path
    validation_path: Path
    data_path: Path


@dataclass(frozen=True)
class EmpiricalRuntimeRelease:
    path: Path
    sha256: str
    independent_acceptance_path: Path
    independent_acceptance_sha256: str
    mode: str
    resolver: EmpiricalRuntimeResolver
    runtime_paths: Mapping[str, Path]
    capsule_record_path: Path | None = None
    capsule_record_sha256: str | None = None
    deployment_acceptance_path: Path | None = None
    deployment_acceptance_sha256: str | None = None


@dataclass(frozen=True)
class EmpiricalRuntimeBinding:
    mode: str
    runtime_paths: Mapping[str, str]
    capsule_record_path: str | None
    capsule_record_sha256: str | None
    deployment_acceptance_path: str | None
    deployment_acceptance_sha256: str | None


@dataclass(frozen=True)
class LoadedDepthConsumptionIndex(Mapping[str, Any]):
    """Immutable validated v2 document plus loader-derived runtime authority."""

    document: Mapping[str, Any]
    runtime_bindings: Mapping[str, EmpiricalRuntimeBinding]

    def __getitem__(self, key: str) -> Any:
        return self.document[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.document)

    def __len__(self) -> int:
        return len(self.document)

    def runtime_binding(self, entry_name: str) -> EmpiricalRuntimeBinding:
        try:
            return self.runtime_bindings[entry_name]
        except KeyError as exc:
            raise EmpiricalDepthContractError(
                f"empirical runtime binding is unavailable for {entry_name}"
            ) from exc


def load_empirical_depth_contract(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_checkpoint_sha256: str | None = None,
    runtime_resolver: EmpiricalRuntimeResolver | None = None,
) -> EmpiricalDepthContract:
    """Load the exact owner-authorized PushT empirical contract without fallback."""

    artifact = Path(path).expanduser()
    expected = require_sha256(expected_sha256, "empirical_depth_contract_sha256")
    _validate_no_alias_path(artifact, artifact="empirical depth contract")
    actual = sha256_file(artifact)
    if actual != expected:
        raise EmpiricalDepthContractError(
            f"empirical depth contract SHA-256 mismatch: expected {expected}, got {actual}"
        )
    try:
        value = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmpiricalDepthContractError(f"cannot decode empirical depth contract: {exc}") from exc
    manifest = _object(value, "empirical depth contract")
    if manifest.get("schema") != EMPIRICAL_CONTRACT_SCHEMA:
        raise EmpiricalDepthContractError(
            f"unsupported empirical depth schema: {manifest.get('schema')!r}"
        )
    if (
        manifest.get("status") != "complete"
        or manifest.get("empirical_use_allowed") is not True
        or manifest.get("native_equivalence_claimed") is not False
        or manifest.get("lossless_to_original_training_input") is not False
    ):
        raise EmpiricalDepthContractError("empirical contract approval/lossiness declarations are invalid")
    _exact_list(manifest.get("assumption_tags"), [EMPIRICAL_ASSUMPTION], "assumption_tags")

    authority = _object(manifest.get("authority"), "authority")
    if (
        authority.get("route") != "mapanything_primary_for_pusht"
        or authority.get("authorization_basis") != "owner_advisor_authorized_report_39"
        or authority.get("execution_authority_granted") is not False
    ):
        raise EmpiricalDepthContractError("empirical authority or execution boundary is invalid")

    scope = _object(manifest.get("scientific_scope"), "scientific_scope")
    invalid_claims = scope.get("invalid_claims")
    if not isinstance(invalid_claims, list) or set(invalid_claims) != REQUIRED_INVALID_CLAIMS:
        raise EmpiricalDepthContractError("scientific_scope.invalid_claims is incomplete")
    if scope.get("non_equivalence_statement") != EMPIRICAL_NON_EQUIVALENCE:
        raise EmpiricalDepthContractError("non-equivalence statement differs")

    checkpoint = _object(manifest.get("checkpoint"), "checkpoint")
    checkpoint_sha = require_sha256(checkpoint.get("sha256"), "checkpoint.sha256")
    if checkpoint_sha != ACCEPTED_IDENTITIES["checkpoint_sha256"]:
        raise EmpiricalDepthContractError("checkpoint differs from the exact accepted PushT identity")
    if expected_checkpoint_sha256 is not None and checkpoint_sha != require_sha256(
        expected_checkpoint_sha256, "configured checkpoint_sha256"
    ):
        raise EmpiricalDepthContractError("empirical contract selects a different checkpoint")
    expected_checkpoint_fields = {
        "backend": "df2_dino_rope_convs_de",
        "factory": "DFormerv2_S",
        "checkpoint_key": "student",
        "state_prefix": "module.backbone.",
        "feature_key": "x_norm_patchtokens",
        "input_shape": [1, 224, 224],
    }
    if any(checkpoint.get(key) != expected_value for key, expected_value in expected_checkpoint_fields.items()):
        raise EmpiricalDepthContractError("checkpoint runtime identity differs")

    cache = _object(manifest.get("cache"), "cache")
    if cache.get("environment") != "pusht":
        raise EmpiricalDepthContractError("empirical contract is PushT-only")
    identity_fields = {
        "manifest_sha256": "manifest_sha256",
        "manifest_id": "manifest_id",
        "data_sha256": "data_sha256",
        "producer_sha256": "producer_sha256",
        "source_index_sha256": "source_index_sha256",
        "wire_format_sha256": "wire_format_sha256",
    }
    for field, accepted_key in identity_fields.items():
        if cache.get(field) != ACCEPTED_IDENTITIES[accepted_key]:
            raise EmpiricalDepthContractError(
                f"cache.{field} differs from the exact accepted PushT identity"
            )
    validation = _object(cache.get("validation"), "cache.validation")
    if (
        validation.get("sha256") != ACCEPTED_IDENTITIES["validation_sha256"]
        or validation.get("schema") != "dinocular-mapanything-cache-validation-v1"
        or validation.get("state") != "PASS"
    ):
        raise EmpiricalDepthContractError("accepted MapAnything receipt identity differs")

    wire = _object(manifest.get("wire_semantics"), "wire_semantics")
    if (
        wire.get("stored_dtype") != "little_endian_float16"
        or wire.get("stored_shape") != [224, 224]
        or wire.get("stored_quantity")
        != "clipped_normalized_later_mapanything_depth_z_proxy"
        or wire.get("stored_range") != [0.0, 1.0]
        or wire.get("geometry") != "already_materialized_224x224_bilinear_depth"
        or set(wire.get("irreversibilities", [])) != REQUIRED_IRREVERSIBILITIES
    ):
        raise EmpiricalDepthContractError("wire semantics or lossiness declarations differ")
    calibration = _object(wire.get("calibration"), "wire_semantics.calibration")
    if (
        calibration.get("operation") != "wire=clip((later_depth_z-lo)/(hi-lo),0,1)"
        or _finite(calibration.get("lo"), "calibration.lo") != 0.0
        or _finite(calibration.get("hi"), "calibration.hi") != EMPIRICAL_PROXY_SCALE
        or calibration.get("key_sha256") != ACCEPTED_IDENTITIES["calibration_key_sha256"]
        or calibration.get("scope") != "pusht_training_only"
    ):
        raise EmpiricalDepthContractError("calibration differs from the exact accepted cache")

    adapter = _object(manifest.get("adapter"), "adapter")
    informative = _object(adapter.get("informative_mode"), "adapter.informative_mode")
    zero = _object(adapter.get("zero_intervention_mode"), "adapter.zero_intervention_mode")
    if (
        adapter.get("id") != EMPIRICAL_ADAPTER_ID
        or informative.get("decode") != "zstd_then_little_endian_float16"
        or informative.get("cast") != "float16_to_float32"
        or informative.get("reconstruction")
        != "proxy_depth_z=wire_float32*1.5746406149864196"
        or informative.get("spatial_operation") != "identity_on_materialized_224x224"
        or informative.get("additional_clipping") != "none"
        or informative.get("checkpoint_mean_std_normalization") != "none"
        or informative.get("invalid_policy") != "fail_on_missing_nonfinite_or_out_of_range"
        or informative.get("payload_presence_mask")
        != "all_ones_only_after_payload_validation"
        or informative.get("range_check")
        != "exact_closed_interval_[0,1]_no_tolerance"
        or zero.get("requires_same_cache_identity_and_coverage") is not True
        or zero.get("boundary_value") != 0.0
        or zero.get("boundary_mask") != 0.0
        or zero.get("neutrality_claimed") is not False
        or zero.get("rgb_only_claimed") is not False
        or zero.get("label") != "constant_zero_numeric_intervention_not_native_neutral"
        or adapter.get("fallback_policy") != FORBIDDEN_FALLBACKS
    ):
        raise EmpiricalDepthContractError("adapter, zero intervention, or no-fallback policy differs")

    disclosure = _object(manifest.get("result_disclosure"), "result_disclosure")
    if disclosure != {
        "required_assumption_tags": [EMPIRICAL_ASSUMPTION],
        "required_contract_sha256": True,
        "required_adapter_id": EMPIRICAL_ADAPTER_ID,
        "required_cache_identity_fields": True,
        "required_non_equivalence_statement": EMPIRICAL_NON_EQUIVALENCE,
    }:
        raise EmpiricalDepthContractError("result disclosure requirements differ")

    canonical_paths = EmpiricalCanonicalPaths(
        checkpoint=str(checkpoint.get("path")),
        cache_directory=str(cache.get("directory")),
        manifest=str(cache.get("manifest_path")),
        validation=str(validation.get("path")),
        data=str(cache.get("data_path")),
    )
    expected_paths = EmpiricalCanonicalPaths(
        checkpoint=ACCEPTED_CANONICAL_PATHS["checkpoint"],
        cache_directory=ACCEPTED_CANONICAL_PATHS["cache_directory"],
        manifest=ACCEPTED_CANONICAL_PATHS["manifest"],
        validation=ACCEPTED_CANONICAL_PATHS["validation"],
        data=ACCEPTED_CANONICAL_PATHS["data"],
    )
    if canonical_paths != expected_paths:
        raise EmpiricalDepthContractError("empirical contract canonical paths differ")
    resolver = runtime_resolver or CanonicalEmpiricalRuntimeResolver()
    runtime_paths = resolver.resolve(canonical_paths)
    if (
        not runtime_paths.checkpoint.is_absolute()
        or not runtime_paths.cache_directory.is_absolute()
        or runtime_paths.manifest.parent != runtime_paths.cache_directory
        or runtime_paths.manifest.name != "manifest.json"
        or runtime_paths.data.parent != runtime_paths.cache_directory
        or runtime_paths.data.name != "data.mdb"
        or not runtime_paths.validation.is_absolute()
    ):
        raise EmpiricalDepthContractError("empirical runtime resolver returned invalid closed paths")

    return EmpiricalDepthContract(
        path=artifact,
        sha256=actual,
        manifest=manifest,
        checkpoint_sha256=checkpoint_sha,
        producer_sha256=str(cache["producer_sha256"]),
        manifest_sha256=str(cache["manifest_sha256"]),
        manifest_id=str(cache["manifest_id"]),
        validation_sha256=str(validation["sha256"]),
        data_sha256=str(cache["data_sha256"]),
        source_index_sha256=str(cache["source_index_sha256"]),
        wire_format_sha256=str(cache["wire_format_sha256"]),
        canonical_paths=canonical_paths,
        checkpoint_path=runtime_paths.checkpoint,
        cache_directory=runtime_paths.cache_directory,
        manifest_path=runtime_paths.manifest,
        validation_path=runtime_paths.validation,
        data_path=runtime_paths.data,
    )


def load_empirical_runtime_release(
    path: str | Path,
    expected_sha256: str,
    *,
    contract: EmpiricalDepthContract,
) -> EmpiricalRuntimeRelease:
    """Load an independently accepted, hash-bound canonical/runtime release record."""

    artifact = Path(path).expanduser()
    expected = require_sha256(expected_sha256, "empirical runtime release SHA-256")
    _validate_no_alias_path(artifact, artifact="empirical runtime release")
    if sha256_file(artifact) != expected:
        raise EmpiricalDepthContractError("empirical runtime release identity differs")
    try:
        value = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmpiricalDepthContractError(
            f"cannot decode empirical runtime release: {exc}"
        ) from exc
    release = _object(value, "empirical runtime release")
    identities = {
        "checkpoint_sha256": contract.checkpoint_sha256,
        "producer_sha256": contract.producer_sha256,
        "manifest_sha256": contract.manifest_sha256,
        "manifest_id": contract.manifest_id,
        "validation_sha256": contract.validation_sha256,
        "data_sha256": contract.data_sha256,
        "source_index_sha256": contract.source_index_sha256,
        "wire_format_sha256": contract.wire_format_sha256,
    }
    canonical_paths = {
        "checkpoint": contract.canonical_paths.checkpoint,
        "cache_directory": contract.canonical_paths.cache_directory,
        "manifest": contract.canonical_paths.manifest,
        "validation": contract.canonical_paths.validation,
        "data": contract.canonical_paths.data,
    }
    if (
        release.get("schema") != RUNTIME_RELEASE_SCHEMA
        or release.get("state") != "INDEPENDENTLY_ACCEPTED"
        or release.get("empirical_contract_sha256") != contract.sha256
        or release.get("identities") != identities
        or release.get("canonical_paths") != canonical_paths
    ):
        raise EmpiricalDepthContractError("empirical runtime release bindings differ")
    mode = release.get("runtime_mode")
    acceptance_reference = _object(
        release.get("independent_acceptance"),
        "empirical runtime independent acceptance",
    )
    acceptance_path_value = acceptance_reference.get("path")
    acceptance_sha = require_sha256(
        acceptance_reference.get("sha256"),
        "empirical runtime independent acceptance SHA-256",
    )
    if not isinstance(acceptance_path_value, str):
        raise EmpiricalDepthContractError(
            "empirical runtime independent acceptance path is absent"
        )
    acceptance_path = Path(acceptance_path_value).expanduser()
    _validate_no_alias_path(
        acceptance_path, artifact="empirical runtime independent acceptance"
    )
    if sha256_file(acceptance_path) != acceptance_sha:
        raise EmpiricalDepthContractError(
            "empirical runtime independent acceptance identity differs"
        )
    try:
        acceptance_value = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmpiricalDepthContractError(
            f"cannot decode empirical runtime independent acceptance: {exc}"
        ) from exc
    acceptance = _object(
        acceptance_value, "empirical runtime independent acceptance"
    )
    if acceptance != {
        "schema": RUNTIME_RELEASE_ACCEPTANCE_SCHEMA,
        "state": "INDEPENDENTLY_ACCEPTED",
        "bindings": {
            "empirical_contract_sha256": contract.sha256,
            "identities": identities,
            "canonical_paths": canonical_paths,
            "runtime_mode": mode,
        },
    }:
        raise EmpiricalDepthContractError(
            "empirical runtime independent acceptance bindings differ"
        )
    capsule_record_path = None
    capsule_record_sha = None
    deployment_acceptance_path = None
    deployment_acceptance_sha = None
    if mode == "canonical_host_v1":
        if "capsule" in release or "deployment_acceptance" in release:
            raise EmpiricalDepthContractError(
                "canonical host release must not carry capsule evidence"
            )
        resolver: EmpiricalRuntimeResolver = CanonicalEmpiricalRuntimeResolver()
        resolved = resolver.resolve(contract.canonical_paths)
        runtime_paths = {
            "checkpoint": resolved.checkpoint,
            "cache_directory": resolved.cache_directory,
            "manifest": resolved.manifest,
            "validation": resolved.validation,
            "data": resolved.data,
            "empirical_contract": contract.path,
            "empirical_runtime_release": artifact,
        }
    elif mode == "capsule_v1":
        resolver = CapsuleEmpiricalRuntimeResolver()
        runtime_paths = {key: Path(value) for key, value in CAPSULE_RUNTIME_PATHS.items()}
        capsule_reference = _object(release.get("capsule"), "capsule evidence")
        deployment_reference = _object(
            release.get("deployment_acceptance"), "capsule deployment acceptance"
        )
        for reference, label in (
            (capsule_reference, "capsule evidence"),
            (deployment_reference, "capsule deployment acceptance"),
        ):
            if set(reference) != {"status", "path", "sha256"} or reference.get(
                "status"
            ) != "READY":
                raise EmpiricalDepthContractError(f"{label} is unavailable")
        capsule_record_path = Path(str(capsule_reference["path"])).expanduser()
        capsule_record_sha = require_sha256(
            capsule_reference.get("sha256"), "capsule evidence SHA-256"
        )
        _validate_no_alias_path(capsule_record_path, artifact="capsule evidence")
        if sha256_file(capsule_record_path) != capsule_record_sha:
            raise EmpiricalDepthContractError("capsule evidence identity differs")
        capsule_record = _object(
            json.loads(capsule_record_path.read_text(encoding="utf-8")),
            "capsule evidence",
        )
        expected_runtime_paths = {
            key: str(value) for key, value in runtime_paths.items()
        }
        if capsule_record != {
            "schema": "dino-wm-empirical-capsule-v1",
            "state": "ACCEPTED",
            "empirical_contract_sha256": contract.sha256,
            "identities": identities,
            "runtime_paths": expected_runtime_paths,
        }:
            raise EmpiricalDepthContractError("capsule evidence bindings differ")
        deployment_acceptance_path = Path(
            str(deployment_reference["path"])
        ).expanduser()
        deployment_acceptance_sha = require_sha256(
            deployment_reference.get("sha256"),
            "capsule deployment acceptance SHA-256",
        )
        _validate_no_alias_path(
            deployment_acceptance_path, artifact="capsule deployment acceptance"
        )
        if sha256_file(deployment_acceptance_path) != deployment_acceptance_sha:
            raise EmpiricalDepthContractError(
                "capsule deployment acceptance identity differs"
            )
        deployment = _object(
            json.loads(deployment_acceptance_path.read_text(encoding="utf-8")),
            "capsule deployment acceptance",
        )
        if deployment != {
            "schema": "dino-wm-empirical-capsule-deployment-acceptance-v1",
            "state": "INDEPENDENTLY_ACCEPTED",
            "verdict": "PASS",
            "bindings": {
                "empirical_contract_sha256": contract.sha256,
                "capsule_record_sha256": capsule_record_sha,
                "runtime_paths": expected_runtime_paths,
            },
        }:
            raise EmpiricalDepthContractError(
                "capsule deployment acceptance bindings differ"
            )
    else:
        raise EmpiricalDepthContractError("empirical runtime release mode is unsupported")
    return EmpiricalRuntimeRelease(
        path=artifact,
        sha256=expected,
        independent_acceptance_path=acceptance_path,
        independent_acceptance_sha256=acceptance_sha,
        mode=str(mode),
        resolver=resolver,
        runtime_paths=MappingProxyType(dict(runtime_paths)),
        capsule_record_path=capsule_record_path,
        capsule_record_sha256=capsule_record_sha,
        deployment_acceptance_path=deployment_acceptance_path,
        deployment_acceptance_sha256=deployment_acceptance_sha,
    )


def load_depth_consumption_index(path: str | Path) -> LoadedDepthConsumptionIndex:
    """Load immutable v2 entries and loader-derived runtime authority."""

    artifact = Path(path).expanduser()
    _validate_no_alias_path(artifact, artifact="depth consumption index")
    value = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    index = _object(value, "depth consumption index")
    if index.get("schema") != CONSUMPTION_INDEX_SCHEMA:
        raise EmpiricalDepthContractError("unsupported depth consumption index schema")
    if index.get("defaults") is not None:
        raise EmpiricalDepthContractError("depth consumption index defaults/fallbacks are forbidden")
    native_v1 = _object(index.get("native_v1"), "depth consumption native_v1")
    release = _object(
        index.get("empirical_runtime_release"),
        "depth consumption empirical_runtime_release",
    )
    if native_v1.get("delegated_environments") != ["wall", "rope", "granular"]:
        raise EmpiricalDepthContractError("mixed dispatcher native-v1 delegation differs")
    for record, path_field, sha_field, label in (
        (native_v1, "index_path", "index_sha256", "native-v1 index"),
        (release, "release_path", "release_sha256", "empirical runtime release"),
    ):
        if record.get("status") == "READY":
            require_sha256(record.get(sha_field), f"{label} SHA-256")
            if not isinstance(record.get(path_field), str) or not Path(
                str(record[path_field])
            ).is_absolute():
                raise EmpiricalDepthContractError(f"{label} path is invalid")
        elif record.get("status") == "BLOCKED_MISSING_ACCEPTED_ARTIFACT":
            if record.get(sha_field) is not None:
                raise EmpiricalDepthContractError(f"blocked {label} cannot claim an identity")
        else:
            raise EmpiricalDepthContractError(f"{label} readiness state is invalid")
    entries = _object(index.get("entries"), "depth consumption index entries")
    if not entries:
        raise EmpiricalDepthContractError("depth consumption index has no entries")
    runtime_bindings: dict[str, EmpiricalRuntimeBinding] = {}
    for name, raw_entry in entries.items():
        entry = _object(raw_entry, f"entries.{name}")
        kind = entry.get("contract_kind")
        contract_path = entry.get("contract_path")
        contract_sha = entry.get("contract_sha256")
        if kind == "empirical_lossy_cache":
            contract = load_empirical_depth_contract(
                str(contract_path), str(contract_sha), expected_checkpoint_sha256=entry.get("checkpoint_sha256")
            )
            if release.get("status") == "READY":
                runtime_release = load_empirical_runtime_release(
                    str(release.get("release_path")),
                    str(release.get("release_sha256")),
                    contract=contract,
                )
                runtime_bindings[str(name)] = EmpiricalRuntimeBinding(
                    mode=runtime_release.mode,
                    runtime_paths=MappingProxyType(
                        {
                            key: str(runtime_path)
                            for key, runtime_path in runtime_release.runtime_paths.items()
                        }
                    ),
                    capsule_record_path=(
                        str(runtime_release.capsule_record_path)
                        if runtime_release.capsule_record_path is not None
                        else None
                    ),
                    capsule_record_sha256=runtime_release.capsule_record_sha256,
                    deployment_acceptance_path=(
                        str(runtime_release.deployment_acceptance_path)
                        if runtime_release.deployment_acceptance_path is not None
                        else None
                    ),
                    deployment_acceptance_sha256=(
                        runtime_release.deployment_acceptance_sha256
                    ),
                )
            if (
                entry.get("environment") != "pusht"
                or entry.get("allowed_arms") != ["dinocular", "dinocular_zerodepth"]
                or entry.get("adapter_id") != EMPIRICAL_ADAPTER_ID
                or entry.get("cache_dir") != contract.canonical_paths.cache_directory
                or entry.get("validation_path") != contract.canonical_paths.validation
                or entry.get("data_path") != contract.canonical_paths.data
                or entry.get("checkpoint_path") != contract.canonical_paths.checkpoint
                or entry.get("producer_sha256") != contract.producer_sha256
                or entry.get("manifest_sha256") != contract.manifest_sha256
                or entry.get("manifest_id") != contract.manifest_id
                or entry.get("validation_sha256") != contract.validation_sha256
                or entry.get("validation_schema") != "dinocular-mapanything-cache-validation-v1"
                or entry.get("data_sha256") != contract.data_sha256
                or entry.get("source_index_sha256") != contract.source_index_sha256
                or entry.get("wire_format_sha256") != contract.wire_format_sha256
                or entry.get("assumption_tags") != [EMPIRICAL_ASSUMPTION]
                or entry.get("execution_authority_granted") is not False
            ):
                raise EmpiricalDepthContractError(f"entries.{name} differs from its empirical contract")
        elif kind == "native":
            try:
                load_native_depth_contract(str(contract_path), str(contract_sha))
            except DepthContractError as exc:
                if "unsupported native depth schema" in str(exc):
                    raise EmpiricalDepthContractError("contract_kind/schema mismatch") from exc
                raise
        else:
            raise EmpiricalDepthContractError(f"entries.{name} has unsupported contract_kind")
    return LoadedDepthConsumptionIndex(
        document=_deep_freeze(index),
        runtime_bindings=MappingProxyType(dict(runtime_bindings)),
    )


def validate_mapanything_receipt(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Validate the exact accepted receipt's branch-local identity semantics."""

    if receipt.get("schema") != "dinocular-mapanything-cache-validation-v1":
        raise EmpiricalDepthContractError("MapAnything receipt schema differs")
    if receipt.get("state") != "PASS":
        raise EmpiricalDepthContractError("MapAnything receipt top-level state is not PASS")
    if receipt.get("manifest_id") != manifest.get("manifest_id"):
        raise EmpiricalDepthContractError("MapAnything receipt manifest identity differs")
    if receipt.get("goal_gauge_gate") != "IDENTICAL_SINGLETON_PATH_BY_CONSTRUCTION":
        raise EmpiricalDepthContractError("MapAnything receipt goal-gauge identity differs")
    if receipt.get("prefix_invariance_gate") != "NOT_APPLICABLE_FRAMEWISE":
        raise EmpiricalDepthContractError("MapAnything receipt framewise prefix declaration differs")

    format_gate = _object(
        receipt.get("format_compatibility_gate"), "format_compatibility_gate"
    )
    if (
        format_gate.get("schema") != "dinocular-depth-cache-v1"
        or format_gate.get("producer_agnostic_fields_equal_to_da3") is not True
        or format_gate.get("wire_format") != manifest.get("wire_format")
    ):
        raise EmpiricalDepthContractError("MapAnything receipt wire-format gate differs")

    calibration = _object(manifest.get("calibration"), "manifest.calibration")
    calibration_gate = _object(receipt.get("calibration_gate"), "calibration_gate")
    if calibration_gate != {
        "scope": calibration.get("scope"),
        "lo": calibration.get("lo"),
        "hi": calibration.get("hi"),
        "keys": 128,
        "validation_keys": 0,
        "keys_sha256": calibration.get("keys_sha256"),
    }:
        raise EmpiricalDepthContractError("MapAnything receipt calibration gate differs")

    count_gate = _object(receipt.get("manifest_count_gate"), "manifest_count_gate")
    expected_counts = {
        "dataset_frames": manifest.get("frame_count"),
        "dataset_trajectories": manifest.get("trajectory_count"),
        "manifest_frames": manifest.get("frame_count"),
        "manifest_trajectories": manifest.get("trajectory_count"),
    }
    if count_gate != expected_counts:
        raise EmpiricalDepthContractError("MapAnything receipt count/coverage gate differs")
    range_gate = _object(receipt.get("range_gate"), "range_gate")
    if (
        range_gate.get("actual_keys") != manifest.get("frame_count")
        or range_gate.get("expected_keys") != manifest.get("frame_count")
    ):
        raise EmpiricalDepthContractError("MapAnything receipt range coverage differs")
    batch_gate = _object(
        receipt.get("independent_batch_equivalence_gate"),
        "independent_batch_equivalence_gate",
    )
    if batch_gate != {
        "frames": 3,
        "production_batch_size": 1,
        "max_absolute_error": 0.0,
        "threshold": 0.001,
    }:
        raise EmpiricalDepthContractError("MapAnything receipt batch-equivalence gate differs")
    spot_gate = _object(receipt.get("spot_recomputation_gate"), "spot_recomputation_gate")
    if (
        spot_gate.get("frames") != 3
        or spot_gate.get("max_absolute_error_after_wire_decode") != 0.000244140625
        or spot_gate.get("threshold") != 0.001
    ):
        raise EmpiricalDepthContractError("MapAnything receipt spot-recomputation gate differs")
    temporal = _object(receipt.get("temporal_gate"), "temporal_gate")
    if (
        temporal.get("state") != "FAIL"
        or temporal.get("acceptance") != "CHARACTERIZATION_ONLY_FRAMEWISE"
    ):
        raise EmpiricalDepthContractError("MapAnything receipt temporal characterization differs")


def apply_empirical_depth_adapter(
    wire: torch.Tensor,
    *,
    zero_intervention: bool,
    require_224: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply only float32 cast and the frozen affine, or the exact zero intervention."""

    if not isinstance(wire, torch.Tensor) or not torch.is_floating_point(wire):
        raise EmpiricalDepthContractError("empirical wire must be a floating-point tensor")
    if wire.ndim == 3:
        wire = wire.unsqueeze(1)
    if wire.ndim != 4 or wire.shape[1] != 1:
        raise EmpiricalDepthContractError("empirical wire must have shape [B,H,W] or [B,1,H,W]")
    if require_224 and tuple(wire.shape[-2:]) != (224, 224):
        raise EmpiricalDepthContractError("empirical wire requires exact 224x224 materialized geometry")
    wire_f32 = wire.to(dtype=torch.float32)
    if not torch.isfinite(wire_f32).all():
        raise EmpiricalDepthContractError("empirical wire contains nonfinite values")
    minimum = float(wire_f32.detach().amin())
    maximum = float(wire_f32.detach().amax())
    if minimum < 0.0 or maximum > 1.0:
        raise EmpiricalDepthContractError(
            f"empirical wire range [{minimum},{maximum}] is outside [0,1]"
        )
    saturated = int(torch.count_nonzero(wire_f32 == 1.0).item())
    count = int(wire_f32.numel())
    if zero_intervention:
        return (
            torch.zeros_like(wire_f32),
            torch.zeros_like(wire_f32),
            {
                "intervention": "exact_constant_zero_numeric",
                "neutrality_claimed": False,
                "rgb_only_claimed": False,
                "saturated_elements": saturated,
                "element_count": count,
                "saturation_fraction": saturated / count,
            },
        )
    proxy = wire_f32 * EMPIRICAL_PROXY_SCALE
    return (
        proxy,
        torch.ones_like(proxy),
        {
            "saturated_elements": saturated,
            "element_count": count,
            "saturation_fraction": saturated / count,
        },
    )


def validate_empirical_provenance(
    value: Mapping[str, Any], *, expected_contract_sha256: str
) -> None:
    """Require exact empirical identity and non-equivalence fields in downstream records."""

    expected = {
        "contract_kind": "empirical_lossy_cache",
        "empirical_contract_sha256": require_sha256(
            expected_contract_sha256, "expected empirical contract SHA-256"
        ),
        "assumption_tags": [EMPIRICAL_ASSUMPTION],
        "adapter_id": EMPIRICAL_ADAPTER_ID,
        "producer_sha256": ACCEPTED_IDENTITIES["producer_sha256"],
        "cache_manifest_sha256": ACCEPTED_IDENTITIES["manifest_sha256"],
        "manifest_id": ACCEPTED_IDENTITIES["manifest_id"],
        "validation_sha256": ACCEPTED_IDENTITIES["validation_sha256"],
        "validation_schema": "dinocular-mapanything-cache-validation-v1",
        "data_sha256": ACCEPTED_IDENTITIES["data_sha256"],
        "source_index_sha256": ACCEPTED_IDENTITIES["source_index_sha256"],
        "wire_format_sha256": ACCEPTED_IDENTITIES["wire_format_sha256"],
        "checkpoint_sha256": ACCEPTED_IDENTITIES["checkpoint_sha256"],
        "non_equivalence_statement": EMPIRICAL_NON_EQUIVALENCE,
        "execution_authority_granted": False,
        "neutrality_claimed": False,
        "rgb_only_claimed": False,
    }
    differing = [field for field, expected_value in expected.items() if value.get(field) != expected_value]
    if differing:
        raise EmpiricalDepthContractError(f"empirical provenance differs: {sorted(differing)}")
    if value.get("adapter_mode") not in {"proxy_depth_z", "exact_constant_zero_numeric"}:
        raise EmpiricalDepthContractError("empirical provenance has invalid adapter_mode")
