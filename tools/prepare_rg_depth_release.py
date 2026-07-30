#!/usr/bin/env python3
"""Bind validated Rope/Granular raw MapAnything caches to the frozen student."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CHECKPOINT_SHA256 = (
    "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc"
)
RAW_WIRE_MAXIMUM = 65504.0
ENVIRONMENTS = ("rope", "granular")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("frozen DINOcular checkpoint identity differs")

    manifests: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    for environment in ENVIRONMENTS:
        cache = args.cache_root / f"{environment}.lmdb"
        manifest_path = cache / "manifest.json"
        validation_path = args.cache_root / environment / "validation.json"
        manifest = json.loads(manifest_path.read_text())
        validation = json.loads(validation_path.read_text())
        if (
            manifest.get("schema") != "dinocular-depth-cache-v1"
            or manifest.get("environment") != environment
            or manifest.get("calibration", {}).get("wire_mode")
            != "raw_depth_z_float16"
            or manifest.get("wire_format", {}).get("normalization")
            != "none_raw_depth_z_float16"
            or not manifest.get("closed_before_hash")
            or sha256_file(cache / "data.mdb") != manifest.get("data_mdb_sha256")
        ):
            raise RuntimeError(f"{environment} raw cache identity is invalid")
        if (
            validation.get("schema")
            != "dinocular-mapanything-cache-validation-v2"
            or validation.get("environment") != environment
            or validation.get("state") != "PASS"
            or validation.get("manifest_id") != manifest.get("manifest_id")
        ):
            raise RuntimeError(f"{environment} producer validation is invalid")
        manifests[environment] = manifest
        validations[environment] = validation

    producer = manifests["rope"]["producer"]
    if manifests["granular"]["producer"] != producer:
        raise RuntimeError("Rope and Granular producer identities differ")
    producer_sha256 = sha256_bytes(canonical_bytes(producer))
    settings = producer["settings"]
    artifacts = producer["artifacts"]
    model_artifact = artifacts["model.safetensors"]
    contract_producer = {
        "name": "mapanything_recovered_framewise_raw_depth_z",
        "model": "facebook_map-anything",
        "version": producer["model_revision"],
        "code_commit": producer["commit"],
        "temporal_mode": settings["temporal_mode"],
        "raw_units": "declared_metric_Z_depth_nonphysical_later_checkpoint_proxy",
        "weight_sha256": model_artifact["sha256"],
        "invocation": {
            "output": "pred['depth_z']",
            "apply_mask": settings["apply_mask"],
            "batch_semantics": settings["batch_semantics"],
        },
        "preprocessing": {
            "api": settings["preprocessing_api"],
            "resize_mode": settings["resize_mode"],
            "resolution_set": settings["resolution_set"],
            "world_model_crop": "resize_short_side_224_center_crop_224",
        },
        "scale": {
            "operation": "identity_raw_depth_z",
            "calibration": "none",
            "compatibility": (
                "original_student_loader_np_load_float32_without_depth_normalization"
            ),
        },
        "clipping": {
            "operation": "none",
            "float16_representable_range": [0.0, RAW_WIRE_MAXIMUM],
        },
        "invalid_policy": {
            "producer": settings["invalid_policy"],
            "custom_hole_fill": settings["custom_hole_fill"],
            "cache_reader": "reject_nonfinite_then_all_ones",
        },
        "recovery_status": producer["recovery_status"],
    }
    bindings = []
    reader_receipts = {}
    for environment in ENVIRONMENTS:
        manifest = manifests[environment]
        wire_sha256 = sha256_bytes(canonical_bytes(manifest["wire_format"]))
        bindings.append(
            {
                "environment": environment,
                "producer_sha256": producer_sha256,
                "wire_format_sha256": wire_sha256,
                "wire_quantity": "raw_mapanything_depth_z_float16",
                "wire_range": [0.0, RAW_WIRE_MAXIMUM],
                "affine_to_raw_metric": {
                    "operation": "raw_metric=wire*scale+offset",
                    "output_quantity": "later_pinned_MapAnything_depth_z_proxy",
                    "output_units": (
                        "declared_metric_Z_depth_nonphysical_later_checkpoint_proxy"
                    ),
                    "scale": 1.0,
                    "offset": 0.0,
                },
                "interpolation": "identity_224x224",
                "payload_validation": {
                    "source": "reject_nonfinite_then_all_ones",
                    "valid_value": 1.0,
                },
            }
        )
        reader_receipt = {
            "schema": "dinocular-depth-cache-validation-v1",
            "created_utc": validation["created_utc"],
            "state": "PASS",
            "results": {
                environment: {
                    "environment": environment,
                    "state": "PASS",
                    "manifest_id": manifest["manifest_id"],
                    "data_mdb_sha256": manifest["data_mdb_sha256"],
                    "producer_validation_path": str(
                        args.cache_root / environment / "validation.json"
                    ),
                    "producer_validation_sha256": sha256_file(
                        args.cache_root / environment / "validation.json"
                    ),
                }
            },
        }
        receipt_path = args.output / "reader-validation" / f"{environment}.json"
        write_json(receipt_path, reader_receipt)
        reader_receipts[environment] = {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
        }

    contract = {
        "schema": "dinocular-native-depth-contract-v1",
        "status": "complete",
        "self_acceptance_recorded": False,
        "execution_authority_granted": False,
        "checkpoint": {
            "name": args.checkpoint.name,
            "sha256": CHECKPOINT_SHA256,
            "backend": "df2_dino_rope_convs_de",
            "factory": "DFormerv2_S",
            "checkpoint_key": "student",
            "state_prefix": "module.backbone.",
        },
        "producer": contract_producer,
        "encoder_input": {
            "checkpoint_native": {
                "quantity": "later_pinned_MapAnything_depth_z_proxy",
                "units": (
                    "declared_metric_Z_depth_nonphysical_later_checkpoint_proxy"
                ),
                "normalization": {"kind": "none_raw_metric"},
            },
            "zero_depth": {
                "payload_validation": "same_as_informative_depth",
                "intervention": "exact_numeric_zero_at_encoder_boundary",
                "learned_neutrality_claimed": False,
                "rgb_equivalence_claimed": False,
            },
            "cache_bindings": bindings,
        },
    }
    contract_path = args.output / "native_depth_contract_rg_mapanything_v1.json"
    write_json(contract_path, contract)
    release = {
        "schema": "dinocular.rg-mapanything-depth-release.v1",
        "state": "READY_FOR_END_TO_END_VALIDATION",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": CHECKPOINT_SHA256,
        },
        "producer_sha256": producer_sha256,
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "environments": {
            environment: {
                "cache": str(args.cache_root / f"{environment}.lmdb"),
                "manifest_sha256": sha256_file(
                    args.cache_root / f"{environment}.lmdb" / "manifest.json"
                ),
                "data_mdb_sha256": manifests[environment]["data_mdb_sha256"],
                "reader_validation": reader_receipts[environment],
            }
            for environment in ENVIRONMENTS
        },
    }
    write_json(args.output / "release.json", release)
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
