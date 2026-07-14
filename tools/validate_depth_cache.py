#!/usr/bin/env python3
"""Validate fixed DINOcular full-trajectory depth caches, without mutating them."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Supports both ``python tools/...`` and imports from the repository root.
    from tools.precompute_depth import (
        ARTIFACTS,
        CALIBRATION_PER_ENV,
        CHUNK_SIZE,
        MAP_SIZE,
        OUTPUT_SHAPE,
        OVERLAP,
        PROCESS_RES,
        SELECTED_ENVIRONMENTS,
        WIRE_DTYPE,
        ZSTD_LEVEL,
        ContractError,
        OfficialDA3StreamingProducer,
        Trajectory,
        TrajectoryDepthProducer,
        canonical_json_bytes,
        crop_producer_result,
        crop_rgb_for_world_model,
        decode_depth_value,
        decode_trajectory,
        enumerate_environment,
        enumerate_selected,
        normalize_depth,
        producer_identity,
        select_calibration_keys,
        sha256_bytes,
        sha256_file,
        streaming_chunks,
        utc_now,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from precompute_depth import (  # type: ignore[no-redef]
        ARTIFACTS,
        CALIBRATION_PER_ENV,
        CHUNK_SIZE,
        MAP_SIZE,
        OUTPUT_SHAPE,
        OVERLAP,
        PROCESS_RES,
        SELECTED_ENVIRONMENTS,
        WIRE_DTYPE,
        ZSTD_LEVEL,
        ContractError,
        OfficialDA3StreamingProducer,
        Trajectory,
        TrajectoryDepthProducer,
        canonical_json_bytes,
        crop_producer_result,
        crop_rgb_for_world_model,
        decode_depth_value,
        decode_trajectory,
        enumerate_environment,
        enumerate_selected,
        normalize_depth,
        producer_identity,
        select_calibration_keys,
        sha256_bytes,
        sha256_file,
        streaming_chunks,
        utc_now,
    )


RANGE_SAMPLE_FRAMES = 4096
TEMPORAL_SAMPLE_PAIRS = 4096
RECOMPUTE_FRACTION = 0.01
RECOMPUTE_MAX_ABS = 1e-3


def _require_lmdb_zstd() -> tuple[Any, Any]:
    try:
        import lmdb
        import zstandard
    except Exception as exc:
        raise ContractError(
            "the pinned image must provide Python packages lmdb and zstandard"
        ) from exc
    return lmdb, zstandard


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise ContractError(f"missing cache manifest {path}")
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:
        raise ContractError(f"cannot decode cache manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ContractError(f"cache manifest is not an object: {path}")
    return manifest


def _source_index_hash(trajectories: Sequence[Trajectory]) -> str:
    index = [
        {
            "key": trajectory.trajectory_key,
            "frames": trajectory.frame_count,
            "source": trajectory.source_relpath,
            "sha256": trajectory.source_sha256,
        }
        for trajectory in trajectories
    ]
    return sha256_bytes(canonical_json_bytes(index))


def _validate_producer_provenance(producer: Mapping[str, Any]) -> None:
    settings = producer.get("settings", {})
    expected_settings = {
        "precision": "bfloat16",
        "process_res": PROCESS_RES,
        "process_res_method": "upper_bound_resize",
        "ref_view_strategy": "saddle_balanced",
        "loop_closure": True,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "overlap_policy": "discard_duplicated_tail_no_blend",
    }
    if settings != expected_settings:
        raise ContractError(f"producer settings differ from fixed contract: {settings}")
    artifacts = producer.get("artifacts", {})
    for name, expected in ARTIFACTS.items():
        record = artifacts.get(name, {})
        if (
            record.get("bytes") != expected["bytes"]
            or record.get("sha256") != expected["sha256"]
        ):
            raise ContractError(f"manifest producer artifact mismatch for {name}")


def _validate_calibration(calibration: Mapping[str, Any]) -> tuple[float, float]:
    keys = calibration.get("keys")
    if calibration.get("per_environment") != CALIBRATION_PER_ENV or not isinstance(
        keys, list
    ):
        raise ContractError(
            "cache does not use 128 calibration keys per selected environment"
        )
    if len(keys) != CALIBRATION_PER_ENV * len(SELECTED_ENVIRONMENTS) or len(
        set(keys)
    ) != len(keys):
        raise ContractError("global calibration key set is not exactly 512 unique keys")
    for environment in SELECTED_ENVIRONMENTS:
        if (
            sum(key.startswith(f"{environment}/") for key in keys)
            != CALIBRATION_PER_ENV
        ):
            raise ContractError(
                f"calibration is not stratified to 128 keys for {environment}"
            )
    if calibration.get("keys_sha256") != sha256_bytes(canonical_json_bytes(keys)):
        raise ContractError("calibration key-list hash mismatch")
    if calibration.get("sample") != "cropped_metric_depth[::8,::8]":
        raise ContractError("calibration sampling contract differs")
    if calibration.get("percentiles") != [2.0, 98.0]:
        raise ContractError("calibration percentiles differ from global 2/98")
    lo, hi = calibration.get("lo"), calibration.get("hi")
    if (
        not isinstance(lo, (int, float))
        or not isinstance(hi, (int, float))
        or not math.isfinite(lo)
        or not math.isfinite(hi)
        or not hi > lo + 1e-6
    ):
        raise ContractError(f"invalid calibration lo/hi: {lo!r}, {hi!r}")
    return float(lo), float(hi)


def validate_manifest_contract(
    cache_dir: Path,
    environment: str,
    trajectories: Sequence[Trajectory],
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if manifest.get("schema") != "dinocular-depth-cache-v1":
        raise ContractError("unsupported depth-cache manifest schema")
    if manifest.get("environment") != environment:
        raise ContractError("cache environment does not match manifest")
    if manifest.get("trajectory_count") != len(trajectories):
        raise ContractError("manifest trajectory count differs from the dataset")
    expected_frame_count = sum(item.frame_count for item in trajectories)
    if manifest.get("frame_count") != expected_frame_count:
        raise ContractError("manifest frame count differs from the dataset")
    if manifest.get("source_index_sha256") != _source_index_hash(trajectories):
        raise ContractError("source index hash differs from the current dataset")
    _validate_producer_provenance(manifest.get("producer", {}))
    _validate_calibration(manifest.get("calibration", {}))
    wire = manifest.get("wire_format", {})
    expected_wire = {
        "physical_key": "<split>/<episode:05d>/<frame:06d>",
        "dtype": "<f2",
        "shape": list(OUTPUT_SHAPE),
        "order": "C",
        "compressor": "zstd",
        "compressor_level": ZSTD_LEVEL,
        "map_size": MAP_SIZE,
        "normalization": "clip((depth_m-lo)/(hi-lo),0,1)",
        "inverted": False,
    }
    if wire != expected_wire:
        raise ContractError(f"LMDB wire contract differs: {wire}")
    if not manifest.get("closed_before_hash"):
        raise ContractError(
            "manifest does not attest that LMDB was closed before hashing"
        )
    current_builder = Path(inspect.getsourcefile(streaming_chunks) or "")
    if not current_builder.is_file() or manifest.get("tool_sha256") != sha256_file(
        current_builder
    ):
        raise ContractError("cache was not built by this exact precompute_depth.py")
    data_path = cache_dir / "data.mdb"
    if not data_path.is_file() or sha256_file(data_path) != manifest.get(
        "data_mdb_sha256"
    ):
        raise ContractError("closed data.mdb SHA-256 mismatch")

    records = manifest.get("trajectories")
    if not isinstance(records, list) or len(records) != len(trajectories):
        raise ContractError("manifest trajectory index is incomplete")
    by_key: dict[str, Mapping[str, Any]] = {}
    expected_by_key = {
        trajectory.trajectory_key: trajectory for trajectory in trajectories
    }
    for record in records:
        if not isinstance(record, dict):
            raise ContractError("manifest trajectory record is not an object")
        key = record.get("trajectory_key")
        if key in by_key or key not in expected_by_key:
            raise ContractError(f"duplicate or unexpected manifest trajectory {key!r}")
        trajectory = expected_by_key[key]
        if (
            record.get("source_path") != trajectory.source_relpath
            or record.get("source_video_sha256") != trajectory.source_sha256
            or record.get("ordered_frame_count") != trajectory.frame_count
        ):
            raise ContractError(f"manifest source record differs for {key}")
        expected_keys = [
            trajectory.physical_key(frame) for frame in range(trajectory.frame_count)
        ]
        if record.get("ordered_output_keys") != expected_keys:
            raise ContractError(f"manifest output-key order differs for {key}")
        chunks = record.get("chunk_boundaries")
        if not isinstance(chunks, list) or chunks != streaming_chunks(
            trajectory.frame_count
        ):
            raise ContractError(f"manifest 120/60 chunk schedule differs for {key}")
        if "alignment_results" not in record or not isinstance(
            record["alignment_results"], list
        ):
            raise ContractError(f"manifest does not record alignments for {key}")
        by_key[key] = record
    if set(by_key) != set(expected_by_key):
        raise ContractError("manifest trajectory key set differs from the dataset")
    return by_key


def _smallest_hash(items: Sequence[Any], key: Any, limit: int) -> list[Any]:
    return sorted(
        items, key=lambda item: (sha256_bytes(key(item).encode()), key(item))
    )[:limit]


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    count = int(histogram.sum())
    if count == 0:
        raise ContractError("cannot take quantile of an empty range sample")
    rank = int(math.floor(quantile * (count - 1)))
    index = int(np.searchsorted(np.cumsum(histogram, dtype=np.uint64), rank + 1))
    return float(np.array([index], dtype=np.uint16).view(WIRE_DTYPE)[0])


def validate_key_set_and_range(
    database: Any,
    environment: str,
    trajectories: Sequence[Trajectory],
) -> dict[str, Any]:
    _, zstandard = _require_lmdb_zstd()
    decompressor = zstandard.ZstdDecompressor()
    expected = [
        trajectory.physical_key(frame)
        for trajectory in trajectories
        for frame in range(trajectory.frame_count)
    ]
    expected_set = set(expected)
    with database.begin(write=False) as transaction:
        actual = [key.decode("ascii") for key, _ in transaction.cursor()]
    if len(actual) != len(expected) or set(actual) != expected_set:
        missing = sorted(expected_set - set(actual))[:10]
        extra = sorted(set(actual) - expected_set)[:10]
        raise ContractError(
            f"LMDB key set differs: expected={len(expected)}, actual={len(actual)}, "
            f"missing={missing}, extra={extra}"
        )

    sampled = _smallest_hash(
        expected, lambda physical: f"{environment}/{physical}", RANGE_SAMPLE_FRAMES
    )
    sampled_set = set(sampled)
    histogram = np.zeros(1 << 16, dtype=np.uint64)
    saturated = 0
    sampled_values = 0
    low_std_maps = 0
    with database.begin(write=False) as transaction:
        for physical_key in actual:
            payload = transaction.get(physical_key.encode("ascii"))
            if payload is None:
                raise ContractError(f"LMDB cursor/get disagreement for {physical_key}")
            depth = decode_depth_value(payload, decompressor)
            if depth.shape != OUTPUT_SHAPE or depth.dtype.str != "<f2":
                raise ContractError(
                    f"wire decode differs for {physical_key}: {depth.shape}/{depth.dtype.str}"
                )
            if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(depth > 1):
                raise ContractError(f"non-finite/out-of-range depth at {physical_key}")
            if physical_key in sampled_set:
                bits = depth.reshape(-1).view(np.uint16)
                histogram += np.bincount(bits, minlength=1 << 16).astype(np.uint64)
                saturated += int(np.count_nonzero((depth == 0) | (depth == 1)))
                sampled_values += depth.size
                low_std_maps += int(float(np.std(depth, dtype=np.float64)) <= 1e-4)
    q10 = _histogram_quantile(histogram, 0.10)
    q90 = _histogram_quantile(histogram, 0.90)
    saturation_fraction = saturated / sampled_values
    low_std_fraction = low_std_maps / len(sampled)
    if not q10 < q90:
        raise ContractError(f"range gate failed: q10={q10} is not less than q90={q90}")
    if not saturation_fraction < 0.50:
        raise ContractError(
            f"range gate failed: saturation={saturation_fraction:.6f} is not <0.50"
        )
    if not low_std_fraction < 0.01:
        raise ContractError(
            f"range gate failed: low-std maps={low_std_fraction:.6f} is not <0.01"
        )
    return {
        "expected_keys": len(expected),
        "actual_keys": len(actual),
        "sampled_frames": len(sampled),
        "q10": q10,
        "q90": q90,
        "saturation_fraction": saturation_fraction,
        "low_std_map_fraction": low_std_fraction,
    }


def dilate_changed_mask(changed: np.ndarray) -> np.ndarray:
    changed = np.asarray(changed, dtype=bool)
    if changed.shape != OUTPUT_SHAPE:
        raise ContractError(f"temporal RGB mask has shape {changed.shape}")
    padded = np.pad(changed, 2, mode="constant", constant_values=False)
    result = np.zeros_like(changed)
    for row in range(5):
        for column in range(5):
            result |= padded[
                row : row + changed.shape[0], column : column + changed.shape[1]
            ]
    return result


def temporal_pair_delta(
    rgb_left: np.ndarray,
    rgb_right: np.ndarray,
    depth_left: np.ndarray,
    depth_right: np.ndarray,
) -> tuple[float, float | None]:
    if rgb_left.shape != (*OUTPUT_SHAPE, 3) or rgb_right.shape != (*OUTPUT_SHAPE, 3):
        raise ContractError("cropped RGB is not [224,224,3]")
    changed = np.mean(np.abs(rgb_right - rgb_left), axis=-1) > (4.0 / 255.0)
    static = ~dilate_changed_mask(changed)
    static_fraction = float(np.mean(static))
    if static_fraction < 0.20:
        return static_fraction, None
    median = float(
        np.median(
            np.abs(depth_right.astype(np.float32) - depth_left.astype(np.float32))[
                static
            ]
        )
    )
    return static_fraction, median


def _temporal_pairs(
    environment: str,
    trajectories: Sequence[Trajectory],
    records: Mapping[str, Mapping[str, Any]],
) -> list[tuple[Trajectory, int, bool]]:
    all_pairs: list[tuple[Trajectory, int, bool]] = []
    boundary_pairs: dict[tuple[str, int], tuple[Trajectory, int, bool]] = {}
    for trajectory in trajectories:
        for left in range(trajectory.frame_count - 1):
            all_pairs.append((trajectory, left, False))
        for chunk in records[trajectory.trajectory_key]["chunk_boundaries"]:
            pair = chunk.get("boundary_pair")
            if pair is not None:
                left, right = pair
                if right != left + 1:
                    raise ContractError(f"non-consecutive chunk-boundary pair {pair}")
                boundary_pairs[(trajectory.trajectory_key, left)] = (
                    trajectory,
                    left,
                    True,
                )

    def pair_key(item: tuple[Trajectory, int, bool]) -> str:
        trajectory, left, _ = item
        return f"{environment}/{trajectory.trajectory_key}/{left:06d}-{left + 1:06d}"

    selected = _smallest_hash(all_pairs, pair_key, TEMPORAL_SAMPLE_PAIRS)
    selected_by_key = {(item[0].trajectory_key, item[1]): item for item in selected}
    selected_by_key.update(boundary_pairs)
    return list(selected_by_key.values())


def validate_temporal_gate(
    database: Any,
    environment: str,
    trajectories: Sequence[Trajectory],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _, zstandard = _require_lmdb_zstd()
    decompressor = zstandard.ZstdDecompressor()
    pairs = _temporal_pairs(environment, trajectories, records)
    grouped: dict[str, list[tuple[Trajectory, int, bool]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair[0].trajectory_key].append(pair)

    eligible_values: list[float] = []
    eligible_count = 0
    boundary_count = 0
    boundary_values: list[float] = []
    minimum_static = 1.0
    with database.begin(write=False) as transaction:
        for trajectory_key in sorted(grouped):
            trajectory = grouped[trajectory_key][0][0]
            frames = decode_trajectory(trajectory)
            cropped_rgb: dict[int, np.ndarray] = {}
            for _, left, boundary in grouped[trajectory_key]:
                if left not in cropped_rgb:
                    cropped_rgb[left] = crop_rgb_for_world_model(frames[left])
                if left + 1 not in cropped_rgb:
                    cropped_rgb[left + 1] = crop_rgb_for_world_model(frames[left + 1])
                payload_left = transaction.get(
                    trajectory.physical_key(left).encode("ascii")
                )
                payload_right = transaction.get(
                    trajectory.physical_key(left + 1).encode("ascii")
                )
                if payload_left is None or payload_right is None:
                    raise ContractError(
                        f"temporal pair is absent for {trajectory_key}/{left}"
                    )
                depth_left = decode_depth_value(payload_left, decompressor)
                depth_right = decode_depth_value(payload_right, decompressor)
                static_fraction, median = temporal_pair_delta(
                    cropped_rgb[left], cropped_rgb[left + 1], depth_left, depth_right
                )
                minimum_static = min(minimum_static, static_fraction)
                boundary_count += int(boundary)
                if median is not None:
                    eligible_count += 1
                    eligible_values.append(median)
                    if boundary:
                        boundary_values.append(median)
                elif boundary:
                    raise ContractError(
                        f"chunk-boundary pair {trajectory_key}/{left}-{left + 1} retains "
                        f"only {static_fraction:.6f} static pixels"
                    )
    eligible_fraction = eligible_count / len(pairs) if pairs else 1.0
    if eligible_fraction < 0.95:
        raise ContractError(
            f"temporal gate failed: only {eligible_fraction:.6f} pairs retain >=20% static pixels"
        )
    if not eligible_values:
        raise ContractError("temporal gate has no eligible consecutive pairs")
    median = float(np.median(eligible_values))
    q95 = float(np.percentile(eligible_values, 95.0, method="linear"))
    max_boundary = max(boundary_values) if boundary_values else None
    if median > 0.01:
        raise ContractError(
            f"temporal gate failed: median static delta {median:.6f} > 0.01"
        )
    if q95 > 0.03:
        raise ContractError(f"temporal gate failed: q95 static delta {q95:.6f} > 0.03")
    if max_boundary is not None and max_boundary > 0.03:
        raise ContractError(
            f"temporal seam gate failed: max boundary delta {max_boundary:.6f} > 0.03"
        )
    return {
        "selected_pairs_including_boundaries": len(pairs),
        "eligible_pairs": eligible_count,
        "eligible_fraction": eligible_fraction,
        "minimum_static_fraction": minimum_static,
        "median_static_depth_delta": median,
        "q95_static_depth_delta": q95,
        "chunk_boundary_pairs": boundary_count,
        "max_chunk_boundary_static_depth_delta": max_boundary,
    }


def validate_recomputation(
    database: Any,
    environment: str,
    trajectories: Sequence[Trajectory],
    manifest: Mapping[str, Any],
    producer: TrajectoryDepthProducer,
    fraction: float = RECOMPUTE_FRACTION,
) -> dict[str, Any]:
    if not 0 < fraction <= 1:
        raise ContractError("whole-trajectory recomputation fraction must be in (0,1]")
    _, zstandard = _require_lmdb_zstd()
    decompressor = zstandard.ZstdDecompressor()
    count = max(1, int(math.ceil(len(trajectories) * fraction)))
    selected = _smallest_hash(
        list(trajectories),
        lambda trajectory: f"{environment}/{trajectory.trajectory_key}",
        count,
    )
    lo, hi = _validate_calibration(manifest["calibration"])
    maximum = 0.0
    compared_frames = 0
    with database.begin(write=False) as transaction:
        for trajectory in selected:
            frames = decode_trajectory(trajectory)
            result = producer.infer_trajectory(frames, trajectory)
            if list(result.chunk_boundaries) != streaming_chunks(
                trajectory.frame_count
            ):
                raise ContractError(
                    f"recomputed chunk schedule differs for {trajectory.trajectory_key}"
                )
            cropped, _ = crop_producer_result(result, frames)
            for frame in range(trajectory.frame_count):
                payload = transaction.get(
                    trajectory.physical_key(frame).encode("ascii")
                )
                if payload is None:
                    raise ContractError(
                        f"recomputed frame is absent: {trajectory.physical_key(frame)}"
                    )
                cached = decode_depth_value(payload, decompressor).astype(np.float32)
                recomputed = normalize_depth(cropped[frame], lo, hi)
                error = float(np.max(np.abs(recomputed - cached)))
                maximum = max(maximum, error)
                compared_frames += 1
    if maximum > RECOMPUTE_MAX_ABS:
        raise ContractError(
            f"whole-trajectory recomputation failed: max abs {maximum:.8f} > {RECOMPUTE_MAX_ABS}"
        )
    return {
        "selection": "smallest_sha256_whole_trajectory_keys",
        "fraction": fraction,
        "trajectories": len(selected),
        "trajectory_keys": [trajectory.trajectory_key for trajectory in selected],
        "frames": compared_frames,
        "max_absolute_error": maximum,
        "threshold": RECOMPUTE_MAX_ABS,
    }


def validate_environment_cache(
    *,
    root: Path,
    cache_root: Path,
    environment: str,
    producer: TrajectoryDepthProducer,
    recompute_fraction: float = RECOMPUTE_FRACTION,
    trajectories: Sequence[Trajectory] | None = None,
) -> dict[str, Any]:
    lmdb, _ = _require_lmdb_zstd()
    trajectories = (
        list(trajectories)
        if trajectories is not None
        else enumerate_environment(root, environment)
    )
    cache_dir = cache_root / f"{environment}.lmdb"
    manifest = load_manifest(cache_dir)
    records = validate_manifest_contract(cache_dir, environment, trajectories, manifest)
    database = lmdb.open(
        str(cache_dir),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=32,
    )
    try:
        range_result = validate_key_set_and_range(database, environment, trajectories)
        recomputation_result = validate_recomputation(
            database,
            environment,
            trajectories,
            manifest,
            producer,
            recompute_fraction,
        )
        temporal_result = validate_temporal_gate(
            database, environment, trajectories, records
        )
    finally:
        database.close()
    return {
        "environment": environment,
        "state": "PASS",
        "manifest_id": manifest["manifest_id"],
        "data_mdb_sha256": manifest["data_mdb_sha256"],
        "range_gate": range_result,
        "recomputation_gate": recomputation_result,
        "temporal_gate": temporal_result,
    }


def _parse_environments(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result or len(set(result)) != len(result):
        raise ContractError("--environments must contain unique selected environments")
    unsupported = sorted(set(result) - set(SELECTED_ENVIRONMENTS))
    if unsupported:
        raise ContractError(f"unsupported environments: {unsupported}")
    return result


def _producer_from_manifest(
    manifest: Mapping[str, Any],
    da3_root: Path | None,
    model_dir: Path | None,
    work_root: Path,
) -> OfficialDA3StreamingProducer:
    provenance = manifest.get("producer", {})
    recorded_root = provenance.get("da3_root")
    if da3_root is None and not recorded_root:
        raise ContractError("manifest does not record the pinned DA3 checkout path")
    resolved_root = da3_root or Path(recorded_root)
    if model_dir is None:
        artifact_path = (
            provenance.get("artifacts", {}).get("model.safetensors", {}).get("path")
        )
        if not artifact_path:
            raise ContractError(
                "manifest does not record the pinned producer artifact path"
            )
        model_dir = Path(artifact_path).parent
    producer = OfficialDA3StreamingProducer(
        resolved_root, model_dir, work_root=work_root
    )
    if producer_identity(producer.provenance) != producer_identity(provenance):
        raise ContractError(
            "live pinned producer provenance differs from the cache manifest"
        )
    return producer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="root containing extracted datasets"
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="depth_cache directory"
    )
    parser.add_argument(
        "--environments",
        default=",".join(SELECTED_ENVIRONMENTS),
        help="comma-separated caches",
    )
    parser.add_argument(
        "--recompute-trajectory-fraction", type=float, default=RECOMPUTE_FRACTION
    )
    parser.add_argument(
        "--da3-root", type=Path, help="override recorded pinned DA3 checkout path"
    )
    parser.add_argument(
        "--model-dir", type=Path, help="override recorded pinned artifact directory"
    )
    parser.add_argument("--json", type=Path, help="write an external validation report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.recompute_trajectory_fraction != RECOMPUTE_FRACTION:
        raise ContractError(
            "the fixed validation gate requires exactly 1% trajectory recomputation"
        )
    environments = _parse_environments(args.environments)
    first_manifest = load_manifest(args.cache / f"{environments[0]}.lmdb")
    calibration = first_manifest.get("calibration", {})
    trajectories_by_env = enumerate_selected(args.root)
    if calibration.get("keys") != select_calibration_keys(trajectories_by_env):
        raise ContractError(
            "manifest calibration keys are not the current dataset's exact stratified "
            "smallest-SHA256 selection"
        )
    producer = _producer_from_manifest(
        first_manifest,
        args.da3_root,
        args.model_dir,
        work_root=args.cache / ".validate-da3-work",
    )
    results: dict[str, Any] = {}
    failed = False
    for environment in environments:
        try:
            manifest = load_manifest(args.cache / f"{environment}.lmdb")
            if producer_identity(manifest.get("producer", {})) != producer_identity(
                first_manifest.get("producer", {})
            ):
                raise ContractError(
                    "environment caches use different producer provenance"
                )
            if manifest.get("calibration") != first_manifest.get("calibration"):
                raise ContractError(
                    "environment caches do not share one exact global calibration"
                )
            results[environment] = validate_environment_cache(
                root=args.root,
                cache_root=args.cache,
                environment=environment,
                producer=producer,
                recompute_fraction=args.recompute_trajectory_fraction,
                trajectories=trajectories_by_env[environment],
            )
        except ContractError as exc:
            failed = True
            results[environment] = {
                "environment": environment,
                "state": "FAIL",
                "error": str(exc),
            }
    report = {
        "schema": "dinocular-depth-cache-validation-v1",
        "created_utc": utc_now(),
        "state": "FAIL" if failed else "PASS",
        "results": results,
    }
    if args.json:
        # Validation reports are external evidence; cache directories remain immutable.
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"DEPTH CACHE VALIDATION CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
