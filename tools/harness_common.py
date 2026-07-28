"""Shared fail-closed primitives for immutable study run cards."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from empirical_depth_contract import (  # noqa: E402
    EMPIRICAL_ADAPTER_ID,
    EMPIRICAL_ASSUMPTION,
    EMPIRICAL_NON_EQUIVALENCE,
    EmpiricalDepthContractError,
    LoadedDepthConsumptionIndex,
    load_depth_consumption_index,
    validate_empirical_provenance,
)


STUDY_SPEC_SCHEMA = "dino-wm-study-spec-v1"
CONTRACT_INDEX_SCHEMA = "dino-wm-depth-contract-index-v1"
MATRIX_SCHEMA = "dino-wm-run-matrix-v1"
RUN_CARD_SCHEMA = "dino-wm-run-card-v1"
LEGACY_RUN_CARD_SCHEMA = "dino-wm.legacy-run-card.v1"
LAUNCH_AUTHORIZATION_SCHEMA = "dino-wm-launch-authorization-v1"
AUTHORIZATION_PREREQUISITE_SCHEMAS = {
    "source_release": {
        "schema": "dino-wm-source-release-v1",
        "state": "ACCEPTED",
        "fields": {
            "schema",
            "state",
            "authorization_subject",
            "source_commit",
            "source_file_sha256",
        },
        "binding_fields": None,
    },
    "empirical_implementation_acceptance": {
        "schema": "dino-wm-empirical-implementation-acceptance-v1",
        "state": "INDEPENDENTLY_ACCEPTED",
        "verdict": "PASS",
        "fields": {
            "schema",
            "state",
            "verdict",
            "authorization_subject",
            "bindings",
        },
        "binding_fields": {
            "contracts_index_sha256",
            "empirical_contract_sha256",
            "source_release_sha256",
        },
    },
    "immutable_execution_acceptance": {
        "schema": "dino-wm-immutable-execution-acceptance-v1",
        "state": "INDEPENDENTLY_ACCEPTED",
        "verdict": "PASS",
        "fields": {
            "schema",
            "state",
            "verdict",
            "authorization_subject",
            "bindings",
        },
        "binding_fields": {
            "empirical_runtime_release_sha256",
            "source_release_sha256",
            "empirical_implementation_acceptance_sha256",
        },
    },
    "immutable_probe_acceptance": {
        "schema": "dino-wm-immutable-probe-acceptance-v1",
        "state": "INDEPENDENTLY_ACCEPTED",
        "verdict": "PASS",
        "fields": {
            "schema",
            "state",
            "verdict",
            "authorization_subject",
            "bindings",
        },
        "binding_fields": {
            "empirical_runtime_release_sha256",
            "immutable_execution_acceptance_sha256",
        },
    },
}
AUTHORIZATION_BINDING_FIELDS = {
    "contracts_index_sha256",
    "empirical_contract_sha256",
    "empirical_runtime_release_sha256",
    "source_commit",
    "source_release_sha256",
    "empirical_implementation_acceptance_sha256",
    "immutable_execution_acceptance_sha256",
    "immutable_probe_acceptance_sha256",
}
LAUNCH_OPERATIONS = frozenset(
    {"materialize", "submit", "run_matrix", "evaluation", "chain"}
)
LOCKED_ARMS = ("dino_pinned", "dinocular", "dinocular_zerodepth")
LOCKED_ENVS = ("pusht", "wall", "rope", "granular")
LOCKED_SEEDS = (1, 2, 3)
LOCKED_TARGETS = {"pusht": 123858, "wall": 143910, "rope": 53500, "granular": 53500}
LOCKED_FRAMESKIPS = {"pusht": 5, "wall": 5, "rope": 1, "granular": 1}
LOCKED_HORIZONS = {
    "pusht": [1, 5, 10, 25],
    "wall": [1, 5, 10],
    "rope": [1, 2, 3],
    "granular": [1, 2, 3],
}
LOCKED_NUM_HIST = {"pusht": 3, "wall": 1, "rope": 1, "granular": 1}
STUDENT_SHA256 = "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc"
RECOVERED_CONTRACT_ASSUMPTION = "[ASSUMPTION: RECOVERED-CONTRACT]"
TIMING_SUMMARY_SCHEMA = "dino-wm-p2-timing-summary-v1"
EVALUATION_DEPTH_PROVENANCE_FIELDS = (
    "depth_producer_sha256",
    "depth_cache_manifest_sha256",
    "depth_native_contract_sha256",
    "depth_validation_sha256",
    "depth_checkpoint_sha256",
)
EVALUATION_EMPIRICAL_PROVENANCE_FIELDS = (
    "depth_contract_kind",
    "depth_empirical_contract_sha256",
    "depth_adapter_id",
    "depth_adapter_mode",
    "depth_manifest_id",
    "depth_data_sha256",
    "depth_source_index_sha256",
    "depth_wire_format_sha256",
    "depth_validation_schema",
    "depth_non_equivalence_statement",
    "depth_neutrality_claimed",
    "depth_rgb_only_claimed",
    "depth_execution_authority_granted",
)
EVALUATION_IMMUTABLE_PROVENANCE_FIELDS = (
    "evaluation_run_card_sha256",
    "evaluation_run_card_file_sha256",
    "training_run_card_sha256",
    "training_run_card_file_sha256",
    "source_commit",
    "config_sha256",
    "container_sha256",
    "checkpoint_sha256",
    "manifest_sha256",
    "assumption_tags",
    *EVALUATION_DEPTH_PROVENANCE_FIELDS,
    *EVALUATION_EMPIRICAL_PROVENANCE_FIELDS,
)
EVALUATION_SHA256_PROVENANCE_FIELDS = tuple(
    field
    for field in EVALUATION_IMMUTABLE_PROVENANCE_FIELDS
    if field.endswith("_sha256")
)

BASE_ENVIRONMENT_FIELDS = frozenset({"DINOV2_REPO", "DINOV2_VITS14_WEIGHTS"})
NATIVE_ENVIRONMENT_FIELDS = BASE_ENVIRONMENT_FIELDS | frozenset(
    {
        "DINOCULAR_STUDENT_WEIGHTS",
        "DINOCULAR_NATIVE_DEPTH_CONTRACT",
        "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256",
        "DINOCULAR_CACHE_PRODUCER_SHA256",
        "DINOCULAR_CACHE_ENVIRONMENT",
    }
)
EMPIRICAL_ENVIRONMENT_FIELDS = BASE_ENVIRONMENT_FIELDS | frozenset(
    {
        "DINOCULAR_STUDENT_WEIGHTS",
        "DINOCULAR_DEPTH_INPUT_MODE",
        "DINOCULAR_EMPIRICAL_DEPTH_CONTRACT",
        "DINOCULAR_EMPIRICAL_DEPTH_CONTRACT_SHA256",
        "DINOCULAR_EMPIRICAL_RUNTIME_RELEASE",
        "DINOCULAR_EMPIRICAL_RUNTIME_RELEASE_SHA256",
        "DINOCULAR_EMPIRICAL_ADAPTER_ID",
        "DINOCULAR_EMPIRICAL_ADAPTER_MODE",
        "DINOCULAR_EMPIRICAL_ZERO_INTERVENTION",
    }
)
NATIVE_DEPTH_INPUT_FIELDS = frozenset(
    {
        "producer",
        "producer_sha256",
        "environment",
        "cache_dir",
        "cache_manifest_sha256",
        "validation_path",
        "validation_sha256",
        "native_contract_path",
        "native_contract_sha256",
        "checkpoint_sha256",
    }
)
EMPIRICAL_DEPTH_INPUT_FIELDS = frozenset(
    {
        "producer",
        "contract_kind",
        "empirical_contract_path",
        "empirical_contract_sha256",
        "empirical_runtime_release_path",
        "empirical_runtime_release_sha256",
        "runtime_mode",
        "runtime_paths",
        "capsule_record_path",
        "capsule_record_sha256",
        "deployment_acceptance_path",
        "deployment_acceptance_sha256",
        "producer_sha256",
        "cache_dir",
        "cache_manifest_sha256",
        "manifest_id",
        "validation_path",
        "validation_sha256",
        "validation_schema",
        "data_path",
        "data_sha256",
        "source_index_sha256",
        "wire_format_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "adapter_id",
        "adapter_mode",
        "assumption_tags",
        "non_equivalence_statement",
        "execution_authority_granted",
        "neutrality_claimed",
        "rgb_only_claimed",
    }
)
_CANDIDATE_AUTHORITY_FIELDS = frozenset(
    {"launch_authorization_subject", "launch_authorization", "authorization_bindings"}
)
_LEGACY_FORBIDDEN_KEY_PARTS = (
    "empirical",
    "runtime_path",
    "adapter_",
    "capsule",
    "deployment",
    "launch_authorization",
    "authorization_binding",
    "contract_kind",
)
_LEGACY_FORBIDDEN_VALUE_PARTS = (
    "empirical",
    "empirical_lossy_cache",
    "dinocular_pusht_empirical",
    "dinocular_zerodepth_pusht_empirical",
    "/opt/dinocular/",
)


@dataclass(frozen=True)
class LoadedMixedContractIndex(Mapping[str, Any]):
    empirical: LoadedDepthConsumptionIndex
    native_v1: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.empirical[key]

    def __iter__(self):
        return iter(self.empirical)

    def __len__(self) -> int:
        return len(self.empirical)


@dataclass(frozen=True)
class RunCardClassification:
    schema: str
    arm: str
    environment: str
    depth_kind: str
    expected_empirical_adapter_mode: str | None
    empirical_provenance: Mapping[str, Any] | None


class HarnessError(RuntimeError):
    """A locked matrix, path, artifact, hash, or gate is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def build_evaluation_provenance(
    card: Mapping[str, Any],
    training_card: Mapping[str, Any],
    *,
    evaluation_run_card_file_sha256: str,
    checkpoint_sha256: str,
    manifest_sha256: str,
    slurm_job_id: str | None,
) -> Mapping[str, Any]:
    depth = card.get("depth_inputs")
    if isinstance(depth, Mapping):
        depth_values = {
            "depth_producer_sha256": depth.get("producer_sha256"),
            "depth_cache_manifest_sha256": depth.get("cache_manifest_sha256"),
            "depth_native_contract_sha256": depth.get("native_contract_sha256"),
            "depth_validation_sha256": depth.get("validation_sha256"),
            "depth_checkpoint_sha256": depth.get("checkpoint_sha256"),
            "depth_contract_kind": depth.get("contract_kind"),
            "depth_empirical_contract_sha256": depth.get("empirical_contract_sha256"),
            "depth_adapter_id": depth.get("adapter_id"),
            "depth_adapter_mode": depth.get("adapter_mode"),
            "depth_manifest_id": depth.get("manifest_id"),
            "depth_data_sha256": depth.get("data_sha256"),
            "depth_source_index_sha256": depth.get("source_index_sha256"),
            "depth_wire_format_sha256": depth.get("wire_format_sha256"),
            "depth_validation_schema": depth.get("validation_schema"),
            "depth_non_equivalence_statement": depth.get("non_equivalence_statement"),
            "depth_neutrality_claimed": depth.get("neutrality_claimed"),
            "depth_rgb_only_claimed": depth.get("rgb_only_claimed"),
            "depth_execution_authority_granted": depth.get("execution_authority_granted"),
        }
    else:
        depth_values = {
            field: None
            for field in (
                *EVALUATION_DEPTH_PROVENANCE_FIELDS,
                *EVALUATION_EMPIRICAL_PROVENANCE_FIELDS,
            )
        }
    training_reference = card.get("training_run_card")
    container = card.get("container")
    if not isinstance(training_reference, Mapping) or not isinstance(
        container, Mapping
    ):
        raise HarnessError("evaluation card lacks training or container provenance")
    provenance = {
        "evaluation_run_card_sha256": card.get("run_card_sha256"),
        "evaluation_run_card_file_sha256": evaluation_run_card_file_sha256,
        "training_run_card_sha256": training_card.get("run_card_sha256"),
        "training_run_card_file_sha256": training_reference.get("file_sha256"),
        "source_commit": card.get("source_commit"),
        "config_sha256": card.get("config_sha256"),
        "container_sha256": container.get("sha256"),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "assumption_tags": list(card.get("assumption_tags", [])),
        "slurm_job_id": slurm_job_id,
        **depth_values,
    }
    validate_evaluation_provenance(
        provenance, requires_depth=isinstance(depth, Mapping)
    )
    return provenance


def validate_evaluation_provenance(
    value: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
    requires_depth: bool,
) -> None:
    empirical = value.get("depth_contract_kind") == "empirical_lossy_cache"
    for field in EVALUATION_SHA256_PROVENANCE_FIELDS:
        field_value = value.get(field)
        if not requires_depth and field in {
            *EVALUATION_DEPTH_PROVENANCE_FIELDS,
            *EVALUATION_EMPIRICAL_PROVENANCE_FIELDS,
        }:
            if field_value is not None:
                raise HarnessError("depth provenance is present for a depth-free run")
        elif field in EVALUATION_EMPIRICAL_PROVENANCE_FIELDS and not empirical:
            if field_value is not None:
                raise HarnessError("non-empirical evaluation carries empirical provenance")
        elif field == "depth_native_contract_sha256" and empirical:
            if field_value is not None:
                raise HarnessError("empirical evaluation cannot carry a native contract hash")
        elif field == "depth_empirical_contract_sha256" and not empirical:
            if field_value is not None:
                raise HarnessError("non-empirical evaluation carries an empirical contract hash")
        elif not _is_lower_hex(field_value, 64):
            raise HarnessError(f"evaluation provenance has invalid {field}")
    if not _is_lower_hex(value.get("source_commit"), 40):
        raise HarnessError("evaluation provenance has invalid source_commit")
    assumption_tags = value.get("assumption_tags")
    if not isinstance(assumption_tags, list) or any(
        item != RECOVERED_CONTRACT_ASSUMPTION for item in assumption_tags
    ):
        raise HarnessError("evaluation provenance has invalid assumption tags")
    if empirical:
        empirical_value = {
            "contract_kind": value.get("depth_contract_kind"),
            "empirical_contract_sha256": value.get("depth_empirical_contract_sha256"),
            "assumption_tags": value.get("assumption_tags"),
            "adapter_id": value.get("depth_adapter_id"),
            "adapter_mode": value.get("depth_adapter_mode"),
            "producer_sha256": value.get("depth_producer_sha256"),
            "cache_manifest_sha256": value.get("depth_cache_manifest_sha256"),
            "manifest_id": value.get("depth_manifest_id"),
            "validation_sha256": value.get("depth_validation_sha256"),
            "validation_schema": value.get("depth_validation_schema"),
            "data_sha256": value.get("depth_data_sha256"),
            "source_index_sha256": value.get("depth_source_index_sha256"),
            "wire_format_sha256": value.get("depth_wire_format_sha256"),
            "checkpoint_sha256": value.get("depth_checkpoint_sha256"),
            "non_equivalence_statement": value.get("depth_non_equivalence_statement"),
            "execution_authority_granted": value.get(
                "depth_execution_authority_granted"
            ),
            "neutrality_claimed": value.get("depth_neutrality_claimed"),
            "rgb_only_claimed": value.get("depth_rgb_only_claimed"),
        }
        try:
            validate_empirical_provenance(
                empirical_value,
                expected_contract_sha256=str(
                    value.get("depth_empirical_contract_sha256")
                ),
            )
        except EmpiricalDepthContractError as exc:
            raise HarnessError(f"evaluation empirical provenance is invalid: {exc}") from exc
    elif any(value.get(field) is not None for field in EVALUATION_EMPIRICAL_PROVENANCE_FIELDS):
        raise HarnessError("non-empirical evaluation carries empirical provenance")
    slurm_job_id = value.get("slurm_job_id")
    if not isinstance(slurm_job_id, str) or not slurm_job_id.isdigit():
        raise HarnessError("evaluation provenance has invalid slurm_job_id")
    if expected is not None:
        differing = [
            field
            for field in EVALUATION_IMMUTABLE_PROVENANCE_FIELDS
            if value.get(field) != expected.get(field)
        ]
        if differing:
            raise HarnessError(
                f"immutable evaluation provenance differs: {sorted(differing)}"
            )


def load_yaml(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise HarnessError(f"missing YAML file: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise HarnessError(f"YAML root must be an object: {path}")
    return value


def load_json(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise HarnessError(f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HarnessError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise HarnessError(f"JSON root must be an object: {path}")
    return value


def _require_no_alias_components(
    path: str | Path, label: str, *, allow_missing: bool
) -> tuple[Path, int | None]:
    artifact = Path(path).expanduser()
    if not artifact.is_absolute() or ".." in artifact.parts:
        raise HarnessError(f"{label} path is not closed")
    current = Path(artifact.anchor)
    final_mode = None
    for component in artifact.parts[1:]:
        current = current / component
        try:
            final_mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing:
                return artifact, None
            raise HarnessError(f"{label} is absent: {current}") from None
        if stat.S_ISLNK(final_mode):
            raise HarnessError(f"{label} path contains a symlink: {current}")
    return artifact, final_mode


def require_regular_file_no_alias(path: str | Path, label: str) -> Path:
    """Require an absolute regular file whose lexical path contains no symlink."""

    artifact, mode = _require_no_alias_components(path, label, allow_missing=False)
    if mode is None or not stat.S_ISREG(mode):
        raise HarnessError(f"{label} is not a regular file: {artifact}")
    return artifact


def require_directory_no_alias(
    path: str | Path, label: str, *, allow_missing: bool = False
) -> Path:
    """Require a lexical absolute directory path with no symlink component."""

    artifact, mode = _require_no_alias_components(
        path, label, allow_missing=allow_missing
    )
    if mode is not None and not stat.S_ISDIR(mode):
        raise HarnessError(f"{label} is not a directory: {artifact}")
    return artifact


def resolve_authorization_prerequisites(
    spec: Mapping[str, Any],
    *,
    contracts_index_sha256: str,
    empirical_contract_sha256: str,
    empirical_runtime_release_sha256: str,
    source_commit: str,
    source_file_sha256: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Resolve the complete acyclic pre-authorization evidence chain."""

    subject = spec.get("launch_authorization_subject")
    references = spec.get("authorization_prerequisites")
    if not isinstance(subject, str) or not subject or not isinstance(references, Mapping):
        raise HarnessError("launch authorization subject or prerequisites are absent")
    if set(references) != set(AUTHORIZATION_PREREQUISITE_SCHEMAS):
        raise HarnessError("launch authorization prerequisite set differs")
    resolved: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for name, closed_schema in AUTHORIZATION_PREREQUISITE_SCHEMAS.items():
        reference = references[name]
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"status", "path", "sha256"}
            or reference.get("status") != "READY"
        ):
            raise HarnessError(f"authorization prerequisite {name} is unavailable")
        path_value = reference.get("path")
        expected_sha = reference.get("sha256")
        if not isinstance(path_value, str) or not _is_lower_hex(expected_sha, 64):
            raise HarnessError(f"authorization prerequisite {name} identity is incomplete")
        path = require_regular_file_no_alias(
            Path(path_value), f"authorization prerequisite {name}"
        )
        if sha256_file(path) != expected_sha:
            raise HarnessError(f"authorization prerequisite {name} hash differs")
        record = load_json(path)
        if (
            set(record) != closed_schema["fields"]
            or record.get("schema") != closed_schema["schema"]
            or record.get("state") != closed_schema["state"]
            or (
                "verdict" in closed_schema
                and record.get("verdict") != closed_schema["verdict"]
            )
        ):
            raise HarnessError(
                f"authorization prerequisite {name} does not match its closed schema"
            )
        if record.get("authorization_subject") != subject:
            raise HarnessError(f"authorization prerequisite {name} subject differs")
        binding_fields = closed_schema["binding_fields"]
        if binding_fields is not None:
            bindings = record.get("bindings")
            if not isinstance(bindings, Mapping) or set(bindings) != binding_fields:
                raise HarnessError(
                    f"authorization prerequisite {name} binding schema differs"
                )
        resolved[name] = (str(expected_sha), record)
    source_release_sha, source_release = resolved["source_release"]
    if (
        source_release.get("source_commit") != source_commit
        or source_release.get("source_file_sha256") != source_file_sha256
    ):
        raise HarnessError("accepted source release differs from the executable source")
    expected_acceptance_bindings = {
        "empirical_implementation_acceptance": {
            "contracts_index_sha256": contracts_index_sha256,
            "empirical_contract_sha256": empirical_contract_sha256,
            "source_release_sha256": source_release_sha,
        },
        "immutable_execution_acceptance": {
            "empirical_runtime_release_sha256": empirical_runtime_release_sha256,
            "source_release_sha256": source_release_sha,
            "empirical_implementation_acceptance_sha256": resolved[
                "empirical_implementation_acceptance"
            ][0],
        },
        "immutable_probe_acceptance": {
            "empirical_runtime_release_sha256": empirical_runtime_release_sha256,
            "immutable_execution_acceptance_sha256": resolved[
                "immutable_execution_acceptance"
            ][0],
        },
    }
    for name, expected in expected_acceptance_bindings.items():
        if resolved[name][1].get("bindings") != expected:
            raise HarnessError(f"authorization prerequisite {name} bindings differ")
    return {
        "contracts_index_sha256": contracts_index_sha256,
        "empirical_contract_sha256": empirical_contract_sha256,
        "empirical_runtime_release_sha256": empirical_runtime_release_sha256,
        "source_commit": source_commit,
        "source_release_sha256": source_release_sha,
        "empirical_implementation_acceptance_sha256": resolved[
            "empirical_implementation_acceptance"
        ][0],
        "immutable_execution_acceptance_sha256": resolved[
            "immutable_execution_acceptance"
        ][0],
        "immutable_probe_acceptance_sha256": resolved[
            "immutable_probe_acceptance"
        ][0],
    }


def require_launch_authorization(
    value: Mapping[str, Any], *, operation: str
) -> Mapping[str, Any]:
    """Require a byte-hashed external explicit launch decision for execution work."""

    if operation not in LAUNCH_OPERATIONS:
        raise HarnessError(f"unsupported launch authorization operation {operation!r}")
    pointer = value.get("launch_authorization")
    if not isinstance(pointer, Mapping):
        raise HarnessError(f"{operation} is blocked: explicit launch authorization is absent")
    path_value = pointer.get("path")
    expected_sha = pointer.get("sha256")
    if pointer.get("status") not in {None, "AUTHORIZED"}:
        raise HarnessError(f"{operation} is blocked: launch authorization is not AUTHORIZED")
    if not isinstance(path_value, str) or not _is_lower_hex(expected_sha, 64):
        raise HarnessError(f"{operation} is blocked: launch authorization identity is incomplete")
    path = require_regular_file_no_alias(
        Path(path_value), "launch authorization record"
    )
    if sha256_file(path) != expected_sha:
        raise HarnessError(f"{operation} is blocked: launch authorization hash differs")
    record = load_json(path)
    if (
        record.get("schema") != LAUNCH_AUTHORIZATION_SCHEMA
        or record.get("state") != "AUTHORIZED"
        or record.get("execution_authority_granted") is not True
        or record.get("decision") != "EXPLICIT_LAUNCH_AUTHORIZED"
        or record.get("authorization_subject")
        != value.get("launch_authorization_subject")
    ):
        raise HarnessError(f"{operation} is blocked: launch authorization record is invalid")
    operations = record.get("authorized_operations")
    if not isinstance(operations, list) or set(operations) != LAUNCH_OPERATIONS:
        raise HarnessError("launch authorization does not bind the closed operation set")
    if operation not in operations:
        raise HarnessError(f"{operation} is not authorized")
    bindings = record.get("bindings")
    if not isinstance(bindings, Mapping):
        raise HarnessError("launch authorization bindings are absent")
    expected_bindings = value.get("authorization_bindings", {})
    if not isinstance(expected_bindings, Mapping) or bindings != expected_bindings:
        raise HarnessError("launch authorization is not hash-bound to this candidate")
    if set(bindings) != AUTHORIZATION_BINDING_FIELDS:
        raise HarnessError("launch authorization binding set differs")
    for name, identity in bindings.items():
        valid = (
            isinstance(identity, str)
            and (
                _is_lower_hex(identity, 64)
                or (name == "source_commit" and _is_lower_hex(identity, 40))
            )
        )
        if not valid:
            raise HarnessError("launch authorization candidate bindings are incomplete")
    return record


def _legacy_contains_forbidden_empirical_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in _LEGACY_FORBIDDEN_KEY_PARTS):
                return True
            if _legacy_contains_forbidden_empirical_value(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_legacy_contains_forbidden_empirical_value(item) for item in value)
    if isinstance(value, str):
        normalized = value.lower()
        return any(part in normalized for part in _LEGACY_FORBIDDEN_VALUE_PARTS)
    return False


def _require_exact_environment_schema(
    card: Mapping[str, Any], expected_fields: frozenset[str]
) -> None:
    environment = card.get("environment_variables")
    if not isinstance(environment, Mapping) or set(environment) != expected_fields:
        raise HarnessError("run-card environment schema differs")
    if any(value is None or not isinstance(value, str) or not value for value in environment.values()):
        raise HarnessError("run-card environment identity is null or absent")


def _require_exact_depth_schema(
    depth: Mapping[str, Any], expected_fields: frozenset[str], *, label: str
) -> None:
    if set(depth) != expected_fields:
        raise HarnessError(f"{label} depth schema differs")
    nullable = {
        "capsule_record_path",
        "capsule_record_sha256",
        "deployment_acceptance_path",
        "deployment_acceptance_sha256",
    }
    for key in expected_fields - nullable:
        if depth.get(key) is None:
            raise HarnessError(f"{label} depth identity is null or absent: {key}")
    if label == "empirical":
        mode = depth.get("runtime_mode")
        capsule_values = (
            depth.get("capsule_record_path"),
            depth.get("capsule_record_sha256"),
            depth.get("deployment_acceptance_path"),
            depth.get("deployment_acceptance_sha256"),
        )
        if mode == "canonical_host_v1" and any(value is not None for value in capsule_values):
            raise HarnessError("canonical empirical depth schema carries capsule evidence")
        if mode == "capsule_v1" and any(value is None for value in capsule_values):
            raise HarnessError("capsule empirical depth schema lacks accepted evidence")
        if mode not in {"canonical_host_v1", "capsule_v1"}:
            raise HarnessError("empirical runtime mode is invalid")
        runtime_paths = depth.get("runtime_paths")
        if not isinstance(runtime_paths, Mapping) or set(runtime_paths) != {
            "checkpoint",
            "cache_directory",
            "manifest",
            "validation",
            "data",
            "empirical_contract",
            "empirical_runtime_release",
        }:
            raise HarnessError("empirical runtime path schema differs")


def classify_run_card(card: Mapping[str, Any]) -> RunCardClassification:
    """Classify authority and receipt semantics from closed card/depth/env schemas."""

    schema = card.get("schema")
    arm = card.get("arm")
    environment = card.get("environment")
    if arm not in LOCKED_ARMS or environment not in LOCKED_ENVS:
        raise HarnessError("run-card schema has an invalid arm or environment")
    depth = card.get("depth_inputs")
    if schema == LEGACY_RUN_CARD_SCHEMA:
        if _legacy_contains_forbidden_empirical_value(card):
            raise HarnessError("legacy run card contains a forbidden empirical field family")
        if arm == "dino_pinned":
            if depth is not None:
                raise HarnessError("legacy dino_pinned card carries depth inputs")
            _require_exact_environment_schema(card, BASE_ENVIRONMENT_FIELDS)
            depth_kind = "none"
        else:
            if not isinstance(depth, Mapping):
                raise HarnessError("legacy run card lacks native depth inputs")
            _require_exact_depth_schema(depth, NATIVE_DEPTH_INPUT_FIELDS, label="legacy native")
            _require_exact_environment_schema(card, NATIVE_ENVIRONMENT_FIELDS)
            depth_kind = "native_v1"
        return RunCardClassification(
            schema=str(schema),
            arm=str(arm),
            environment=str(environment),
            depth_kind=depth_kind,
            expected_empirical_adapter_mode=None,
            empirical_provenance=None,
        )
    if schema != RUN_CARD_SCHEMA:
        raise HarnessError("unsupported run-card schema")
    if set(_CANDIDATE_AUTHORITY_FIELDS) - set(card):
        raise HarnessError("candidate run-card authority schema is incomplete")
    subject = card.get("launch_authorization_subject")
    pointer = card.get("launch_authorization")
    bindings = card.get("authorization_bindings")
    if (
        not isinstance(subject, str)
        or not subject
        or not isinstance(pointer, Mapping)
        or set(pointer) != {"status", "path", "sha256"}
        or pointer.get("status") != "AUTHORIZED"
        or not isinstance(pointer.get("path"), str)
        or not _is_lower_hex(pointer.get("sha256"), 64)
        or not isinstance(bindings, Mapping)
        or set(bindings) != AUTHORIZATION_BINDING_FIELDS
        or any(value is None for value in bindings.values())
    ):
        raise HarnessError("candidate run-card authority identity is null or absent")
    if arm == "dino_pinned":
        if depth is not None:
            raise HarnessError("candidate dino_pinned card carries depth inputs")
        _require_exact_environment_schema(card, BASE_ENVIRONMENT_FIELDS)
        return RunCardClassification(str(schema), str(arm), str(environment), "none", None, None)
    if not isinstance(depth, Mapping):
        raise HarnessError("candidate depth schema is absent")
    if set(depth) == NATIVE_DEPTH_INPUT_FIELDS:
        _require_exact_depth_schema(depth, NATIVE_DEPTH_INPUT_FIELDS, label="candidate native")
        _require_exact_environment_schema(card, NATIVE_ENVIRONMENT_FIELDS)
        return RunCardClassification(str(schema), str(arm), str(environment), "native_v1", None, None)
    _require_exact_depth_schema(depth, EMPIRICAL_DEPTH_INPUT_FIELDS, label="empirical")
    _require_exact_environment_schema(card, EMPIRICAL_ENVIRONMENT_FIELDS)
    if depth.get("contract_kind") != "empirical_lossy_cache":
        raise HarnessError("candidate empirical contract identity is null or absent")
    expected_mode = {
        "dinocular": "proxy_depth_z",
        "dinocular_zerodepth": "exact_constant_zero_numeric",
    }.get(str(arm))
    if environment != "pusht" or depth.get("adapter_mode") != expected_mode:
        raise HarnessError("candidate empirical schema differs from arm/environment")
    expected_encoder = {
        "dinocular": "encoder=dinocular_pusht_empirical",
        "dinocular_zerodepth": "encoder=dinocular_zerodepth_pusht_empirical",
    }[str(arm)]
    overrides = card.get("overrides")
    if not isinstance(overrides, list) or expected_encoder not in overrides:
        raise HarnessError("candidate empirical encoder contract differs")
    return RunCardClassification(
        schema=str(schema),
        arm=str(arm),
        environment=str(environment),
        depth_kind="empirical_lossy_cache",
        expected_empirical_adapter_mode=expected_mode,
        empirical_provenance=depth,
    )


def run_card_receipt_expectations(
    card: Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Derive receipt expectations only from the validated closed card schema."""

    classification = classify_run_card(card)
    return (
        classification.expected_empirical_adapter_mode,
        classification.empirical_provenance,
    )


def require_run_card_authorization(
    card: Mapping[str, Any], *, operation: str
) -> Mapping[str, Any] | None:
    """Require candidate authority while preserving exact disjoint native legacy cards."""

    classification = classify_run_card(card)
    if classification.schema == RUN_CARD_SCHEMA:
        return require_launch_authorization(card, operation=operation)
    return None


def require_real_marvin_path(path: str, label: str) -> str:
    root = "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm"
    if not isinstance(path, str) or not (path == root or path.startswith(root + "/")):
        raise HarnessError(f"{label} must be an absolute real Marvin project path")
    if "$" in path or "~" in path or ".." in Path(path).parts:
        raise HarnessError(f"{label} contains an unresolved or unsafe component")
    return path


def validate_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema") != STUDY_SPEC_SCHEMA:
        raise HarnessError("unsupported study spec schema")
    require_real_marvin_path(str(spec.get("study_root")), "study_root")
    require_real_marvin_path(str(spec.get("code_root")), "code_root")
    container = spec.get("container")
    if not isinstance(container, Mapping):
        raise HarnessError("container record is absent")
    require_real_marvin_path(str(container.get("path")), "container.path")
    if len(str(container.get("sha256"))) != 64:
        raise HarnessError("container SHA-256 is invalid")
    arms = spec.get("arms")
    if (
        not isinstance(arms, list)
        or tuple(item.get("id") for item in arms) != LOCKED_ARMS
    ):
        raise HarnessError("study arms must be exactly the locked three-arm order")
    environments = spec.get("environments")
    if not isinstance(environments, Mapping) or tuple(environments) != LOCKED_ENVS:
        raise HarnessError(
            "study environments must be exactly the locked four-env order"
        )
    for environment in LOCKED_ENVS:
        record = environments[environment]
        expected = (
            LOCKED_TARGETS[environment],
            LOCKED_FRAMESKIPS[environment],
            LOCKED_HORIZONS[environment],
            LOCKED_NUM_HIST[environment],
        )
        actual = (
            record.get("target_steps"),
            record.get("frameskip"),
            record.get("horizons"),
            record.get("num_hist"),
        )
        if actual != expected:
            raise HarnessError(
                f"locked steps/frameskip/horizons differ for {environment}: {actual}"
            )
    if tuple(spec.get("seeds", ())) != LOCKED_SEEDS:
        raise HarnessError("study seeds must be exactly 1,2,3")
    protocol = spec.get("shared_protocol")
    required_protocol = {
        "batch_size": 32,
        "predictor_lr": 0.00005,
        "num_pred": 1,
        "decoder": False,
        "train_encoder": False,
        "train_predictor": True,
        "strict_determinism": True,
        "num_workers": 0,
        "checkpoint_every_steps": 1000,
        "plan_during_training": False,
    }
    if protocol != required_protocol:
        raise HarnessError(
            "shared protocol differs from the locked decoder-off contract"
        )
    if spec.get("segment_sizing") != {
        "max_productive_hours": 8.0,
        "safety_margin_fraction": 0.2,
        "quantum_steps": 1000,
    }:
        raise HarnessError(
            "segment sizing policy differs from the locked safety contract"
        )
    if spec.get("p3_completion") != {
        "heldout_selection": "all_validation_examples",
        "percent_rounding": "ceil_target_times_percent_over_100",
        "progress_points": 100,
        "early_window": [76, 77, 78, 79, 80],
        "late_window": [96, 97, 98, 99, 100],
        "plateau_relative_threshold": 0.02,
        "final_receipt": "final_acceptance.json",
    }:
        raise HarnessError("P3 completion policy differs from the locked contract")
    p2a = spec.get("p2a")
    if not isinstance(p2a, Mapping):
        raise HarnessError("P2a spec is absent")
    if (
        p2a.get("environment") != "pusht"
        or p2a.get("seed") != 1
        or p2a.get("encoder") != "dinocular"
        or p2a.get("target_steps") != 123858
        or p2a.get("producers")
        != ["da3_giant_video", "mapanything_recovered_framewise"]
        or p2a.get("decision_horizons") != [5, 10]
        or p2a.get("tie_tolerance") != 0.000001
    ):
        raise HarnessError("P2a spec differs from the locked two-cell pilot")


def verify_hashed_path(record: Mapping[str, Any], label: str) -> dict[str, str]:
    path = require_real_marvin_path(str(record.get("path")), f"{label}.path")
    expected = str(record.get("sha256"))
    if len(expected) != 64:
        raise HarnessError(f"{label}.sha256 is invalid")
    artifact = require_regular_file_no_alias(Path(path), f"pinned {label}")
    actual = sha256_file(artifact)
    if actual != expected:
        raise HarnessError(
            f"pinned {label} hash mismatch: expected {expected}, got {actual}"
        )
    return {"path": path, "sha256": actual}


def load_timing_summary(path: str | Path) -> Mapping[str, Any]:
    summary = load_json(path)
    if summary.get("schema") != TIMING_SUMMARY_SCHEMA or summary.get("state") != "PASS":
        raise HarnessError("P2 timing summary is absent, failed, or incompatible")
    rates = summary.get("rates")
    expected = {
        f"{arm}/{environment}" for arm in LOCKED_ARMS for environment in LOCKED_ENVS
    }
    if not isinstance(rates, Mapping) or set(rates) != expected:
        raise HarnessError(
            "P2 timing summary must contain exactly 12 arm/environment rates"
        )
    for key, record in rates.items():
        rate = (
            record.get("optimizer_steps_per_second")
            if isinstance(record, Mapping)
            else None
        )
        if (
            not isinstance(record, Mapping)
            or record.get("state") != "PASS"
            or not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not math.isfinite(float(rate))
            or float(rate) <= 0.0
        ):
            raise HarnessError(f"invalid accepted timing rate for {key}")
    return summary


def derive_segment_sizing(
    *,
    summary_path: str | Path,
    arm: str,
    environment: str,
    target_steps: int,
    policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    summary_path = require_regular_file_no_alias(
        summary_path, "segment sizing timing summary"
    )
    summary = load_timing_summary(summary_path)
    key = f"{arm}/{environment}"
    rate = float(summary["rates"][key]["optimizer_steps_per_second"])
    max_hours = float(policy["max_productive_hours"])
    margin = float(policy["safety_margin_fraction"])
    quantum = int(policy["quantum_steps"])
    if max_hours <= 0 or not 0 < margin < 1 or quantum <= 0:
        raise HarnessError("invalid segment sizing policy")
    safe_capacity = rate * max_hours * 3600.0 * (1.0 - margin)
    derived = int(safe_capacity // quantum) * quantum
    if derived < quantum:
        raise HarnessError(f"accepted rate for {key} cannot fit one sizing quantum")
    derived = min(int(target_steps), derived)
    record = {
        "timing_summary_path": str(summary_path),
        "timing_summary_sha256": sha256_file(summary_path),
        "timing_matrix_sha256": summary.get("matrix_sha256"),
        "timing_source_commit": summary.get("source_commit"),
        "rate_key": key,
        "optimizer_steps_per_second": rate,
        "max_productive_hours": max_hours,
        "safety_margin_fraction": margin,
        "quantum_steps": quantum,
        "derived_segment_steps": derived,
    }
    validate_segment_sizing(
        record, arm=arm, environment=environment, target_steps=target_steps
    )
    return record


def validate_segment_sizing(
    record: Mapping[str, Any], *, arm: str, environment: str, target_steps: int
) -> None:
    if (
        not isinstance(record, Mapping)
        or record.get("rate_key") != f"{arm}/{environment}"
    ):
        raise HarnessError("segment sizing rate key differs from the run card")
    rate = record.get("optimizer_steps_per_second")
    max_hours = record.get("max_productive_hours")
    margin = record.get("safety_margin_fraction")
    quantum = record.get("quantum_steps")
    derived = record.get("derived_segment_steps")
    if (
        not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isfinite(float(rate))
        or float(rate) <= 0
        or not isinstance(max_hours, (int, float))
        or float(max_hours) != 8.0
        or not isinstance(margin, (int, float))
        or float(margin) != 0.2
        or not isinstance(quantum, int)
        or quantum != 1000
        or not isinstance(derived, int)
        or not isinstance(record.get("timing_source_commit"), str)
        or len(record["timing_source_commit"]) != 40
    ):
        raise HarnessError("segment sizing record has invalid rate or policy fields")
    expected = min(
        int(target_steps),
        int(
            (float(rate) * float(max_hours) * 3600.0 * (1.0 - float(margin))) // quantum
        )
        * quantum,
    )
    if expected < quantum or derived != expected:
        raise HarnessError(
            "segment steps were not derived from the accepted rate and margin"
        )


def source_evidence(
    spec: Mapping[str, Any], local_code_root: Path
) -> Mapping[str, Any]:
    local_code_root = local_code_root.expanduser()
    commit = subprocess.check_output(
        ["git", "-C", str(local_code_root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(local_code_root), "status", "--porcelain"], text=True
    )
    if status:
        raise HarnessError("run cards cannot be materialized from a dirty source tree")
    files = spec.get("source_hash_files")
    if not isinstance(files, list) or not files:
        raise HarnessError("source_hash_files is empty")
    hashes = {}
    for relative in files:
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise HarnessError(f"invalid source hash path: {relative!r}")
        path = require_regular_file_no_alias(
            local_code_root / relative, f"run-card source file {relative}"
        )
        hashes[relative] = sha256_file(path)
    artifacts = {
        name: verify_hashed_path(record, name)
        for name, record in spec.get("artifacts", {}).items()
    }
    container = verify_hashed_path(spec["container"], "container")
    return {
        "source_commit": commit,
        "source_file_sha256": hashes,
        "artifacts": artifacts,
        "container": container,
    }


def load_native_contract_index(path: str | Path) -> Mapping[str, Any]:
    """Load only an accepted native-v1 index without touching empirical readiness."""

    index_path = require_regular_file_no_alias(Path(path), "native/mixed index")
    index = load_yaml(index_path)
    if index.get("schema") == CONTRACT_INDEX_SCHEMA:
        return load_contract_index(index_path)
    if index.get("schema") != "dino-wm-depth-consumption-index-v2":
        raise HarnessError("unsupported depth contract index schema")
    native = index.get("native_v1")
    if not isinstance(native, Mapping) or native.get("status") != "READY":
        raise HarnessError("accepted native-v1 index artifact is not ready")
    if native.get("delegated_environments") != ["wall", "rope", "granular"]:
        raise HarnessError("native-v1 delegation scope differs")
    native_sha = native.get("index_sha256")
    if not _is_lower_hex(native_sha, 64):
        raise HarnessError("native-v1 index identity is unavailable")
    native_path = require_regular_file_no_alias(
        Path(str(native.get("index_path"))), "native-v1 index reference"
    )
    if sha256_file(native_path) != native_sha:
        raise HarnessError("native-v1 index SHA-256 differs")
    loaded = load_contract_index(native_path)
    if loaded.get("schema") != CONTRACT_INDEX_SCHEMA:
        raise HarnessError("native-v1 reference does not identify a native index")
    return loaded


def load_contract_index(path: str | Path) -> Mapping[str, Any]:
    index_path = require_regular_file_no_alias(Path(path), "depth contract index")
    index = load_yaml(index_path)
    if index.get("schema") == "dino-wm-depth-consumption-index-v2":
        native_record = index.get("native_v1")
        release_record = index.get("empirical_runtime_release")
        if not isinstance(native_record, Mapping) or not isinstance(
            release_record, Mapping
        ):
            raise HarnessError("mixed v2 index lacks native-v1 or empirical release readiness")
        if native_record.get("status") != "READY":
            raise HarnessError(
                "mixed dispatcher code is ready but accepted native-v1 index artifact is not ready"
            )
        if release_record.get("status") != "READY":
            raise HarnessError(
                "mixed dispatcher code is ready but accepted empirical runtime release is not ready"
            )
        native_sha = native_record.get("index_sha256")
        if not _is_lower_hex(native_sha, 64):
            raise HarnessError("mixed dispatcher native-v1 index identity is unavailable")
        native_path = require_regular_file_no_alias(
            Path(str(native_record.get("index_path"))), "mixed dispatcher native-v1 index"
        )
        if sha256_file(native_path) != native_sha:
            raise HarnessError("mixed dispatcher native-v1 index SHA-256 differs")
        release_sha = release_record.get("release_sha256")
        if not _is_lower_hex(release_sha, 64):
            raise HarnessError("mixed dispatcher empirical release identity is unavailable")
        release_path = require_regular_file_no_alias(
            Path(str(release_record.get("release_path"))),
            "mixed dispatcher empirical release",
        )
        if sha256_file(release_path) != release_sha:
            raise HarnessError("mixed dispatcher empirical release SHA-256 differs")
        try:
            loaded = load_depth_consumption_index(index_path)
        except EmpiricalDepthContractError as exc:
            raise HarnessError(str(exc)) from exc
        native_index = load_contract_index(native_path)
        if native_index.get("schema") != CONTRACT_INDEX_SCHEMA:
            raise HarnessError("mixed dispatcher must delegate to an unchanged native-v1 index")
        return LoadedMixedContractIndex(empirical=loaded, native_v1=native_index)
    if index.get("schema") != CONTRACT_INDEX_SCHEMA:
        raise HarnessError("unsupported depth contract index schema")
    native = index.get("native_contract")
    if not isinstance(native, Mapping):
        raise HarnessError("native contract record is absent")
    verify_hashed_path(native, "native contract")
    if native.get("checkpoint_sha256") != STUDENT_SHA256:
        raise HarnessError("native contract selects the wrong student checkpoint")
    producers = index.get("producers")
    if not isinstance(producers, Mapping):
        raise HarnessError("depth producer index is absent")
    required_producers = {"da3_giant_video"}
    if set(producers) != required_producers:
        raise HarnessError(
            "contract index must contain exactly the two locked producers"
        )
    for producer_name, producer in producers.items():
        if (
            not isinstance(producer, Mapping)
            or len(str(producer.get("producer_sha256"))) != 64
        ):
            raise HarnessError(f"invalid producer record for {producer_name}")
        caches = producer.get("caches")
        if not isinstance(caches, Mapping):
            raise HarnessError(f"cache records are absent for {producer_name}")
        for environment, cache in caches.items():
            if environment not in LOCKED_ENVS or not isinstance(cache, Mapping):
                raise HarnessError(
                    f"invalid cache record {producer_name}/{environment}"
                )
            require_real_marvin_path(
                str(cache.get("cache_dir")), f"{producer_name}/{environment}.cache_dir"
            )
            manifest_record = {
                "path": str(Path(str(cache["cache_dir"])) / "manifest.json"),
                "sha256": cache.get("manifest_sha256"),
            }
            verify_hashed_path(
                manifest_record, f"{producer_name}/{environment} manifest"
            )
            validation = verify_hashed_path(
                {
                    "path": cache.get("validation_path"),
                    "sha256": cache.get("validation_sha256"),
                },
                f"{producer_name}/{environment} validation",
            )
            report = load_json(validation["path"])
            result = report.get("results", {}).get(environment, {})
            if report.get("state") != "PASS" or result.get("state") != "PASS":
                raise HarnessError(
                    f"depth cache validation is not PASS for {producer_name}/{environment}"
                )
    return index


def legacy_depth_inputs(
    index: Mapping[str, Any], producer_name: str, environment: str
) -> Mapping[str, Any]:
    """Select only the unchanged native-v1 route for legacy comparisons."""

    if isinstance(index, LoadedMixedContractIndex):
        return depth_inputs(index.native_v1, producer_name, environment)
    if index.get("schema") == "dino-wm-depth-consumption-index-v2":
        raise HarnessError("legacy native-v1 depth index is unavailable")
    return depth_inputs(index, producer_name, environment)


def depth_inputs(
    index: Mapping[str, Any], producer_name: str, environment: str
) -> Mapping[str, Any]:
    mixed: LoadedDepthConsumptionIndex | None = None
    native_index: Mapping[str, Any] | None = None
    if isinstance(index, LoadedMixedContractIndex):
        mixed = index.empirical
        native_index = index.native_v1
    elif isinstance(index, LoadedDepthConsumptionIndex):
        mixed = index
    elif index.get("schema") == "dino-wm-depth-consumption-index-v2":
        raise HarnessError("v2 depth inputs require immutable typed loader output")
    if mixed is not None:
        entry = mixed.get("entries", {}).get(f"{environment}/{producer_name}")
        if not isinstance(entry, Mapping):
            native_record = mixed.get("native_v1")
            delegated = (
                native_record.get("delegated_environments")
                if isinstance(native_record, Mapping)
                else None
            )
            if (
                not isinstance(delegated, (list, tuple))
                or tuple(delegated) != ("wall", "rope", "granular")
                or environment not in delegated
            ):
                raise HarnessError(
                    f"depth consumption entry {environment}/{producer_name} is absent"
                )
            if not isinstance(native_index, Mapping):
                raise HarnessError(
                    f"depth consumption entry {environment}/{producer_name} is absent"
                )
            return depth_inputs(native_index, producer_name, environment)
        if entry.get("contract_kind") == "empirical_lossy_cache":
            mode = "proxy_depth_z"
            release = mixed.get("empirical_runtime_release")
            try:
                runtime = mixed.runtime_binding(f"{environment}/{producer_name}")
            except EmpiricalDepthContractError as exc:
                raise HarnessError(str(exc)) from exc
            runtime_paths = runtime.runtime_paths
            if (
                not isinstance(release, Mapping)
                or release.get("status") != "READY"
                or runtime.mode not in {"canonical_host_v1", "capsule_v1"}
                or set(runtime_paths)
                != {
                    "checkpoint",
                    "cache_directory",
                    "manifest",
                    "validation",
                    "data",
                    "empirical_contract",
                    "empirical_runtime_release",
                }
            ):
                raise HarnessError(
                    "empirical runtime release or accepted deployment evidence is not ready"
                )
            return {
                "producer": producer_name,
                "contract_kind": "empirical_lossy_cache",
                "empirical_contract_path": entry["contract_path"],
                "empirical_contract_sha256": entry["contract_sha256"],
                "empirical_runtime_release_path": release["release_path"],
                "empirical_runtime_release_sha256": release["release_sha256"],
                "runtime_mode": runtime.mode,
                "runtime_paths": dict(runtime_paths),
                "capsule_record_path": runtime.capsule_record_path,
                "capsule_record_sha256": runtime.capsule_record_sha256,
                "deployment_acceptance_path": runtime.deployment_acceptance_path,
                "deployment_acceptance_sha256": runtime.deployment_acceptance_sha256,
                "producer_sha256": entry["producer_sha256"],
                "cache_dir": entry["cache_dir"],
                "cache_manifest_sha256": entry["manifest_sha256"],
                "manifest_id": entry["manifest_id"],
                "validation_path": entry["validation_path"],
                "validation_sha256": entry["validation_sha256"],
                "validation_schema": entry["validation_schema"],
                "data_path": entry["data_path"],
                "data_sha256": entry["data_sha256"],
                "source_index_sha256": entry["source_index_sha256"],
                "wire_format_sha256": entry["wire_format_sha256"],
                "checkpoint_path": entry["checkpoint_path"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "adapter_id": entry["adapter_id"],
                "adapter_mode": mode,
                "assumption_tags": list(entry["assumption_tags"]),
                "non_equivalence_statement": EMPIRICAL_NON_EQUIVALENCE,
                "execution_authority_granted": False,
                "neutrality_claimed": False,
                "rgb_only_claimed": False,
            }
        raise HarnessError("v2 native entries require the unchanged native-v1 dispatcher")
    producer = index["producers"].get(producer_name)
    if not isinstance(producer, Mapping):
        raise HarnessError(f"producer {producer_name!r} is absent")
    cache = producer.get("caches", {}).get(environment)
    if not isinstance(cache, Mapping):
        raise HarnessError(f"cache {producer_name}/{environment} is absent")
    native = index["native_contract"]
    return {
        "environment": environment,
        "producer": producer_name,
        "producer_sha256": (
            cache["producer_sha256"]
            if "producer_sha256" in cache
            else producer["producer_sha256"]
        ),
        "cache_dir": cache["cache_dir"],
        "cache_manifest_sha256": cache["manifest_sha256"],
        "validation_path": cache["validation_path"],
        "validation_sha256": cache["validation_sha256"],
        "native_contract_path": native["path"],
        "native_contract_sha256": native["sha256"],
        "checkpoint_sha256": native["checkpoint_sha256"],
    }


def depth_artifact_records(value: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """Return the exact discriminated artifact set for live verification."""

    records: dict[str, Mapping[str, Any]]
    if value.get("contract_kind") == "empirical_lossy_cache":
        records = {
            "depth cache manifest": {
                "path": str(Path(str(value["cache_dir"])) / "manifest.json"),
                "sha256": value["cache_manifest_sha256"],
            },
            "depth validation": {
                "path": value["validation_path"],
                "sha256": value["validation_sha256"],
            },
            "empirical depth contract": {
                "path": value["empirical_contract_path"],
                "sha256": value["empirical_contract_sha256"],
            },
            "empirical runtime release": {
                "path": value["empirical_runtime_release_path"],
                "sha256": value["empirical_runtime_release_sha256"],
            },
            "empirical checkpoint": {
                "path": value["checkpoint_path"],
                "sha256": value["checkpoint_sha256"],
            },
        }
        if value.get("runtime_mode") == "capsule_v1":
            records.update(
                {
                    "empirical capsule record": {
                        "path": value["capsule_record_path"],
                        "sha256": value["capsule_record_sha256"],
                    },
                    "empirical deployment acceptance": {
                        "path": value["deployment_acceptance_path"],
                        "sha256": value["deployment_acceptance_sha256"],
                    },
                }
            )
    else:
        records = {
            "depth cache manifest": {
                "path": str(Path(str(value["cache_dir"])) / "manifest.json"),
                "sha256": value["cache_manifest_sha256"],
            },
            "depth validation": {
                "path": value["validation_path"],
                "sha256": value["validation_sha256"],
            },
            "native depth contract": {
                "path": value["native_contract_path"],
                "sha256": value["native_contract_sha256"],
            },
        }
    return records


def validate_empirical_depth_inputs(
    value: Mapping[str, Any], *, arm: str, environment: str
) -> None:
    """Gate the amended PushT cells without changing any other environment."""

    if arm == "dino_pinned":
        raise HarnessError("dino_pinned must not carry an empirical depth contract")
    if environment != "pusht":
        raise HarnessError("empirical MapAnything consumption is authorized for PushT only")
    if arm not in {"dinocular", "dinocular_zerodepth"}:
        raise HarnessError(f"unsupported empirical PushT arm {arm!r}")
    contract_sha = value.get("empirical_contract_sha256")
    if not _is_lower_hex(contract_sha, 64):
        raise HarnessError("empirical depth inputs lack a contract SHA-256")
    try:
        validate_empirical_provenance(
            value, expected_contract_sha256=str(contract_sha)
        )
    except EmpiricalDepthContractError as exc:
        raise HarnessError(f"invalid empirical depth provenance: {exc}") from exc
    expected_mode = (
        "exact_constant_zero_numeric"
        if arm == "dinocular_zerodepth"
        else "proxy_depth_z"
    )
    if value.get("adapter_mode") != expected_mode:
        raise HarnessError("empirical adapter mode differs from the locked PushT arm")
    if value.get("adapter_id") != EMPIRICAL_ADAPTER_ID:
        raise HarnessError("empirical adapter identity differs")
    if value.get("assumption_tags") != [EMPIRICAL_ASSUMPTION]:
        raise HarnessError("empirical recovered-contract assumption differs")
    if value.get("non_equivalence_statement") != EMPIRICAL_NON_EQUIVALENCE:
        raise HarnessError("empirical non-equivalence disclosure differs")


def run_card_digest(card: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(card))
    value.pop("run_card_sha256", None)
    return sha256_bytes(canonical_json_bytes(value))


def finalize_run_card(card: Mapping[str, Any]) -> Mapping[str, Any]:
    result = copy.deepcopy(dict(card))
    result["run_card_sha256"] = run_card_digest(result)
    return result


def validate_run_card(card: Mapping[str, Any]) -> None:
    classification = classify_run_card(card)
    if card.get("run_card_sha256") != run_card_digest(card):
        raise HarnessError(f"run-card content hash mismatch for {card.get('run_id')}")
    require_real_marvin_path(str(card.get("run_dir")), "run_dir")
    if card.get("source_commit") is None or len(str(card["source_commit"])) != 40:
        raise HarnessError("run card has no full source commit")
    if not isinstance(card.get("source_file_sha256"), Mapping):
        raise HarnessError("run card source hashes are absent")
    if card.get("decoder") is not False:
        raise HarnessError("decoder must remain off")
    producer = card.get("producer_pilot", {}).get("producer")
    if producer == "mapanything_recovered_framewise":
        if card.get("assumption_tags") != [RECOVERED_CONTRACT_ASSUMPTION]:
            raise HarnessError("MapAnything card lacks recovered-contract provenance")
    elif card.get("kind") in {"p2a-producer-pilot", "p2a-open-loop"}:
        if card.get("assumption_tags", []) != []:
            raise HarnessError("DA3 P2a card has unexpected assumption provenance")
    depth_inputs_value = card.get("depth_inputs")
    if isinstance(depth_inputs_value, Mapping) and depth_inputs_value.get(
        "contract_kind"
    ) == "empirical_lossy_cache":
        validate_empirical_depth_inputs(
            depth_inputs_value,
            arm=str(card.get("arm")),
            environment=str(card.get("environment")),
        )
        if card.get("assumption_tags") != [EMPIRICAL_ASSUMPTION]:
            raise HarnessError("empirical PushT card lacks exact assumption provenance")
    if card.get("kind") in {"p3-training", "p4-open-loop"}:
        heldout = card.get("heldout_loss_manifest")
        if (
            not isinstance(heldout, Mapping)
            or heldout.get("selection") != "all_validation_examples"
            or not isinstance(heldout.get("target_steps"), int)
            or isinstance(heldout.get("target_steps"), bool)
            or heldout.get("target_steps") != card.get("target_steps")
            or heldout.get("rounding_rule") != "ceil(target_steps*percent/100)"
            or not isinstance(heldout.get("entry_count"), int)
            or isinstance(heldout.get("entry_count"), bool)
            or heldout.get("entry_count") <= 0
            or not all(
                _is_lower_hex(heldout.get(field), 64)
                for field in (
                    "sha256",
                    "metadata_sha256",
                    "data_manifest_sha256",
                    "split_sha256",
                )
            )
        ):
            raise HarnessError("P3/P4 card has no immutable held-out loss manifest")
        initialization = card.get("initialization_policy")
        if initialization != {
            "predictor": "fresh_seeded",
            "action_encoder": "fresh_seeded",
            "proprio_encoder": "fresh_seeded",
            "seed": card.get("seed"),
            "encoder": "frozen",
        }:
            raise HarnessError("P3/P4 initialization policy differs")
        if (
            card.get("optimizer_policy")
            != {
                "predictor": "adamw",
                "predictor_lr": 0.00005,
                "action_proprio": "adamw",
                "action_proprio_lr": 0.0005,
            }
            or card.get("schedule_policy") != "fixed_learning_rates"
        ):
            raise HarnessError("P3/P4 optimizer or schedule policy differs")
        empirical_pusht = (
            isinstance(card.get("depth_inputs"), Mapping)
            and card["depth_inputs"].get("contract_kind")
            == "empirical_lossy_cache"
        )
        if empirical_pusht:
            expected_boundary = {
                "dinocular": "empirical_proxy_depth_and_payload_presence_mask",
                "dinocular_zerodepth": "exact_constant_zero_numeric_and_audit_mask",
            }.get(card.get("arm"))
        else:
            expected_boundary = {
                "dino_pinned": "not_applicable",
                "dinocular": "informative_depth_and_mask",
                "dinocular_zerodepth": "manifest_neutral_depth_and_mask",
            }.get(card.get("arm"))
        if card.get("encoder_boundary") != expected_boundary:
            raise HarnessError("P3/P4 encoder boundary policy differs")
    if card.get("kind") == "p4-open-loop":
        completion = card.get("training_completion_receipt")
        if (
            not isinstance(completion, Mapping)
            or completion.get("schema") != "dino-wm.p3-final-acceptance.v1"
            or completion.get("training_run_card_sha256")
            != card.get("training_run_card", {}).get("run_card_sha256")
        ):
            raise HarnessError("P4 card is not bound to a P3 completion receipt")


def verify_evaluation_bindings(
    card: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if card.get("kind") not in {"p4-open-loop", "p2a-open-loop"}:
        raise HarnessError("evaluation bindings require a P4 or P2a evaluation card")
    fixed = card.get("fixed_manifest")
    if not isinstance(fixed, Mapping):
        raise HarnessError("evaluation card has no fixed manifest record")
    manifest_path = require_regular_file_no_alias(
        Path(str(fixed.get("path"))), "evaluation fixed manifest"
    )
    metadata_path = require_regular_file_no_alias(
        Path(str(fixed.get("metadata_path"))), "evaluation fixed manifest metadata"
    )
    if (
        sha256_file(manifest_path) != fixed.get("sha256")
        or sha256_file(metadata_path) != fixed.get("metadata_sha256")
    ):
        raise HarnessError(
            "fixed manifest or metadata hash differs from evaluation card"
        )
    metadata = load_json(metadata_path)
    environment = str(card.get("environment"))
    if (
        metadata.get("schema") != "dino-wm-open-loop-manifest-v1"
        or metadata.get("manifest_sha256") != fixed.get("sha256")
        or metadata.get("environment") != environment
        or metadata.get("frameskip") != LOCKED_FRAMESKIPS[environment]
        or metadata.get("horizons") != LOCKED_HORIZONS[environment]
    ):
        raise HarnessError("fixed manifest metadata differs from evaluation card")

    reference = card.get("training_run_card")
    if not isinstance(reference, Mapping):
        raise HarnessError("evaluation card has no hashed training run-card reference")
    training_path = require_regular_file_no_alias(
        Path(str(reference.get("path"))), "evaluation training run card"
    )
    if sha256_file(training_path) != reference.get("file_sha256"):
        raise HarnessError("training run-card file hash differs from evaluation card")
    training = load_yaml(training_path)
    validate_run_card(training)
    if training.get("run_card_sha256") != reference.get("run_card_sha256"):
        raise HarnessError(
            "training run-card content hash differs from evaluation card"
        )
    expected_kind = (
        "p3-training" if card.get("kind") == "p4-open-loop" else "p2a-producer-pilot"
    )
    if (
        training.get("kind") != expected_kind
        or training.get("run_id") != card.get("training_run_id")
        or training.get("run_dir") != card.get("training_run_dir")
        or training.get("arm") != card.get("arm")
        or training.get("environment") != card.get("environment")
        or training.get("seed") != card.get("seed")
        or training.get("source_commit") != card.get("source_commit")
        or training.get("config_sha256") != card.get("config_sha256")
        or training.get("target_steps") != card.get("target_steps")
        or training.get("overrides") != card.get("overrides")
        or training.get("depth_inputs") != card.get("depth_inputs")
        or training.get("environment_variables") != card.get("environment_variables")
        or training.get("assumption_tags", []) != card.get("assumption_tags", [])
        or training.get("heldout_loss_manifest") != card.get("heldout_loss_manifest")
        or training.get("initialization_policy") != card.get("initialization_policy")
        or training.get("optimizer_policy") != card.get("optimizer_policy")
        or training.get("schedule_policy") != card.get("schedule_policy")
        or training.get("encoder_boundary") != card.get("encoder_boundary")
    ):
        raise HarnessError("evaluation card is not exactly bound to its training card")
    if card.get("kind") == "p4-open-loop":
        completion = card["training_completion_receipt"]
        training_run_dir = require_directory_no_alias(
            Path(str(training["run_dir"])), "evaluation training run directory"
        )
        completion_path = require_regular_file_no_alias(
            Path(str(completion.get("path"))), "evaluation completion reference"
        )
        if completion_path != training_run_dir / "final_acceptance.json" or completion.get(
            "training_run_card_sha256"
        ) != training.get("run_card_sha256"):
            raise HarnessError("P4 completion receipt reference differs from P3")
    return training, metadata


def immutable_yaml(path: str | Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    text = yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise HarnessError(
                f"immutable YAML already exists with different bytes: {path}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    return sha256_file(path)


def write_matrix(
    output: Path,
    *,
    kind: str,
    cards: Sequence[Mapping[str, Any]],
    source_commit: str,
) -> Mapping[str, Any]:
    cards_dir = output.with_suffix("").with_name(output.stem + "_cards")
    references = []
    seen = set()
    for card in cards:
        validate_run_card(card)
        run_id = str(card["run_id"])
        if run_id in seen:
            raise HarnessError(f"duplicate run ID {run_id}")
        seen.add(run_id)
        card_path = cards_dir / f"{run_id}.yaml"
        file_sha = immutable_yaml(card_path, card)
        references.append(
            {
                "run_id": run_id,
                "path": str(card_path.absolute()),
                "file_sha256": file_sha,
                "run_card_sha256": card["run_card_sha256"],
            }
        )
    matrix = {
        "schema": MATRIX_SCHEMA,
        "kind": kind,
        "source_commit": source_commit,
        "card_count": len(references),
        "cards": references,
    }
    matrix["matrix_sha256"] = sha256_bytes(canonical_json_bytes(matrix))
    immutable_yaml(output, matrix)
    return matrix


def load_matrix(path: str | Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    matrix_path = require_regular_file_no_alias(path, "run matrix")
    matrix = load_yaml(matrix_path)
    if matrix.get("schema") != MATRIX_SCHEMA:
        raise HarnessError("unsupported run matrix schema")
    expected = copy.deepcopy(dict(matrix))
    digest = expected.pop("matrix_sha256", None)
    if digest != sha256_bytes(canonical_json_bytes(expected)):
        raise HarnessError("matrix content hash mismatch")
    cards = []
    seen = set()
    for reference in matrix.get("cards", []):
        reference_id = str(reference.get("run_id"))
        if reference_id in seen:
            raise HarnessError(f"duplicate matrix run ID: {reference_id}")
        seen.add(reference_id)
        path_value = require_regular_file_no_alias(
            Path(str(reference.get("path"))), f"matrix run card {reference_id}"
        )
        if sha256_file(path_value) != reference.get("file_sha256"):
            raise HarnessError(f"run-card file hash mismatch: {path_value}")
        card = load_yaml(path_value)
        validate_run_card(card)
        if (
            card.get("run_id") != reference_id
            or card.get("run_card_sha256") != reference.get("run_card_sha256")
            or card.get("source_commit") != matrix.get("source_commit")
        ):
            raise HarnessError(f"run-card reference hash mismatch: {path_value}")
        cards.append(card)
    if len(cards) != matrix.get("card_count"):
        raise HarnessError("matrix card count differs from loaded cards")
    return matrix, cards
