"""Shared fail-closed primitives for immutable study run cards."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import yaml


STUDY_SPEC_SCHEMA = "dino-wm-study-spec-v1"
CONTRACT_INDEX_SCHEMA = "dino-wm-depth-contract-index-v1"
MATRIX_SCHEMA = "dino-wm-run-matrix-v1"
RUN_CARD_SCHEMA = "dino-wm-run-card-v1"
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
TIMING_SUMMARY_SCHEMA = "dino-wm-p2-timing-summary-v1"
EVALUATION_DEPTH_PROVENANCE_FIELDS = (
    "depth_producer_sha256",
    "depth_cache_manifest_sha256",
    "depth_native_contract_sha256",
    "depth_validation_sha256",
    "depth_checkpoint_sha256",
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
    *EVALUATION_DEPTH_PROVENANCE_FIELDS,
)
EVALUATION_SHA256_PROVENANCE_FIELDS = tuple(
    field
    for field in EVALUATION_IMMUTABLE_PROVENANCE_FIELDS
    if field != "source_commit"
)


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
        }
    else:
        depth_values = {field: None for field in EVALUATION_DEPTH_PROVENANCE_FIELDS}
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
    for field in EVALUATION_SHA256_PROVENANCE_FIELDS:
        field_value = value.get(field)
        if field in EVALUATION_DEPTH_PROVENANCE_FIELDS and not requires_depth:
            if field_value is not None:
                raise HarnessError("depth provenance is present for a depth-free run")
        elif not _is_lower_hex(field_value, 64):
            raise HarnessError(f"evaluation provenance has invalid {field}")
    if not _is_lower_hex(value.get("source_commit"), 40):
        raise HarnessError("evaluation provenance has invalid source_commit")
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
    artifact = Path(path)
    if not artifact.is_file():
        raise HarnessError(f"missing pinned {label}: {artifact}")
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
    summary_path = Path(summary_path).resolve()
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
    local_code_root = local_code_root.resolve()
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
        path = local_code_root / relative
        if not path.is_file():
            raise HarnessError(f"missing run-card source file: {path}")
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


def load_contract_index(path: str | Path) -> Mapping[str, Any]:
    index = load_yaml(path)
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
    required_producers = {"da3_giant_video", "mapanything_recovered_framewise"}
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


def depth_inputs(
    index: Mapping[str, Any], producer_name: str, environment: str
) -> Mapping[str, Any]:
    producer = index["producers"].get(producer_name)
    if not isinstance(producer, Mapping):
        raise HarnessError(f"producer {producer_name!r} is absent")
    cache = producer.get("caches", {}).get(environment)
    if not isinstance(cache, Mapping):
        raise HarnessError(f"cache {producer_name}/{environment} is absent")
    native = index["native_contract"]
    return {
        "producer": producer_name,
        "producer_sha256": producer["producer_sha256"],
        "cache_dir": cache["cache_dir"],
        "cache_manifest_sha256": cache["manifest_sha256"],
        "validation_path": cache["validation_path"],
        "validation_sha256": cache["validation_sha256"],
        "native_contract_path": native["path"],
        "native_contract_sha256": native["sha256"],
        "checkpoint_sha256": native["checkpoint_sha256"],
    }


def run_card_digest(card: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(card))
    value.pop("run_card_sha256", None)
    return sha256_bytes(canonical_json_bytes(value))


def finalize_run_card(card: Mapping[str, Any]) -> Mapping[str, Any]:
    result = copy.deepcopy(dict(card))
    result["run_card_sha256"] = run_card_digest(result)
    return result


def validate_run_card(card: Mapping[str, Any]) -> None:
    if card.get("schema") != RUN_CARD_SCHEMA:
        raise HarnessError("unsupported run-card schema")
    if card.get("run_card_sha256") != run_card_digest(card):
        raise HarnessError(f"run-card content hash mismatch for {card.get('run_id')}")
    require_real_marvin_path(str(card.get("run_dir")), "run_dir")
    if card.get("source_commit") is None or len(str(card["source_commit"])) != 40:
        raise HarnessError("run card has no full source commit")
    if not isinstance(card.get("source_file_sha256"), Mapping):
        raise HarnessError("run card source hashes are absent")
    if card.get("decoder") is not False:
        raise HarnessError("decoder must remain off")


def verify_evaluation_bindings(
    card: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if card.get("kind") not in {"p4-open-loop", "p2a-open-loop"}:
        raise HarnessError("evaluation bindings require a P4 or P2a evaluation card")
    fixed = card.get("fixed_manifest")
    if not isinstance(fixed, Mapping):
        raise HarnessError("evaluation card has no fixed manifest record")
    manifest_path = Path(str(fixed.get("path")))
    metadata_path = Path(str(fixed.get("metadata_path")))
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != fixed.get("sha256")
        or not metadata_path.is_file()
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
    training_path = Path(str(reference.get("path")))
    if not training_path.is_file() or sha256_file(training_path) != reference.get(
        "file_sha256"
    ):
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
    ):
        raise HarnessError("evaluation card is not exactly bound to its training card")
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
                "path": str(card_path.resolve()),
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
    matrix = load_yaml(path)
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
        path_value = Path(str(reference.get("path")))
        if not path_value.is_file() or sha256_file(path_value) != reference.get(
            "file_sha256"
        ):
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
