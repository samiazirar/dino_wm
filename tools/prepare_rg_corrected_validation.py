#!/usr/bin/env python3
"""Freeze one corrected Rope/Granular cache validation without rebuilding it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RAW_CALIBRATION = {
    "wire_mode": "raw_depth_z_float16",
    "checkpoint_compatibility": (
        "original_student_loader_np_load_float32_without_depth_normalization"
    ),
    "quantity": "later_pinned_MapAnything_pred_depth_z_proxy",
    "units": "declared_metric_Z_depth_nonphysical_later_checkpoint_proxy",
    "invalid_policy": "reject_nonfinite_or_negative_preserve_upstream_masked_exact_zero",
    "calibration": "none",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path: Path, document: dict[str, Any]) -> None:
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"immutable validation card differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def prepare_card(
    *,
    cache_dir: Path,
    environment: str,
    source_commit: str,
    source_tool: Path,
    expected_build_tool_sha256: str,
    expected_manifest_sha256: str,
    expected_data_sha256: str,
    max_trajectories: int,
    spot_frames: int,
    batch_size: int,
) -> dict[str, Any]:
    if environment not in {"rope", "granular"}:
        raise RuntimeError("environment must be rope or granular")
    if cache_dir.name != f"{environment}.lmdb":
        raise RuntimeError("cache directory does not match environment")
    if max_trajectories <= 0 or spot_frames <= 0 or batch_size <= 0:
        raise RuntimeError("frozen validation criteria must be positive")
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise RuntimeError("source commit must be a lowercase SHA-1")
    manifest_path = cache_dir / "manifest.json"
    data_path = cache_dir / "data.mdb"
    if not manifest_path.is_file() or not data_path.is_file() or not source_tool.is_file():
        raise RuntimeError("cache manifest, cache bytes, or staged source is missing")
    manifest_sha256 = sha256_file(manifest_path)
    data_sha256 = sha256_file(data_path)
    if manifest_sha256 != expected_manifest_sha256 or data_sha256 != expected_data_sha256:
        raise RuntimeError("completed cache bytes differ from the frozen build receipt")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "dinocular-depth-cache-v1"
        or manifest.get("environment") != environment
        or manifest.get("calibration") != RAW_CALIBRATION
        or manifest.get("wire_format", {}).get("dtype") != "<f2"
        or manifest.get("wire_format", {}).get("normalization")
        != "none_raw_depth_z_float16"
        or not manifest.get("closed_before_hash")
        or manifest.get("data_mdb_sha256") != data_sha256
        or manifest.get("tool_sha256") != expected_build_tool_sha256
    ):
        raise RuntimeError("completed cache manifest identity is invalid")
    source_tool_sha256 = sha256_file(source_tool)
    return {
        "schema": "dinocular.rg-corrected-validation-card.v1",
        "state": "READY",
        "stage": "validate_only_no_rebuild",
        "environment": environment,
        "cache": {
            "path": str(cache_dir),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "data_mdb_sha256": data_sha256,
            "manifest_id": manifest["manifest_id"],
            "build_tool_sha256": expected_build_tool_sha256,
            "source_index_sha256": manifest["source_index_sha256"],
        },
        "corrected_validator": {
            "source_commit": source_commit,
            "source_tool": str(source_tool),
            "source_tool_sha256": source_tool_sha256,
        },
        "frozen_criteria": {
            "max_trajectories": max_trajectories,
            "spot_frames": spot_frames,
            "batch_size": batch_size,
            "producer_gate": "pinned_MapAnything_identity_must_match_manifest",
            "cache_mutation": "forbidden",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tool", type=Path, required=True)
    parser.add_argument("--expected-build-tool-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--max-trajectories", type=int, default=1)
    parser.add_argument("--spot-frames", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    card = prepare_card(
        cache_dir=args.cache,
        environment=args.environment,
        source_commit=args.source_commit,
        source_tool=args.source_tool,
        expected_build_tool_sha256=args.expected_build_tool_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_data_sha256=args.expected_data_sha256,
        max_trajectories=args.max_trajectories,
        spot_frames=args.spot_frames,
        batch_size=args.batch_size,
    )
    write_immutable(args.output, card)
    print(json.dumps(card, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
