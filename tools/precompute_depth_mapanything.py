#!/usr/bin/env python3
"""Build and validate frame-wise MapAnything depth caches.

The recovered student evidence identifies MapAnything, singleton image inference,
and ``pred["depth_z"]`` but not the original checkpoint snapshot or invocation.
This producer therefore pins the documented best-evidence upstream defaults and
records that decision in every manifest.  It reuses only the DA3 cache mechanics
and wire constants from :mod:`tools.precompute_depth`; it never invokes or
modifies the DA3 producer.
"""

from __future__ import annotations

import argparse
import copy
import faulthandler
import json
import math
import os
import resource
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from tools.precompute_depth import (
        CALIBRATION_STRIDE,
        MAP_SIZE,
        OUTPUT_SHAPE,
        WIRE_DTYPE,
        ZSTD_LEVEL,
        ContractError,
        ProducerResult,
        Trajectory,
        _du_bytes,
        atomic_write_json,
        build_environment_cache,
        canonical_json_bytes,
        compute_global_calibration,
        crop_metric_depth,
        decode_depth_value,
        decode_trajectory,
        enumerate_environment,
        normalize_depth,
        producer_identity,
        sha256_bytes,
        sha256_file,
        streaming_chunks,
        utc_now,
    )
    from tools.validate_depth_cache import (
        validate_key_set_and_range,
        validate_temporal_gate,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from precompute_depth import (  # type: ignore[no-redef]
        CALIBRATION_STRIDE,
        MAP_SIZE,
        OUTPUT_SHAPE,
        WIRE_DTYPE,
        ZSTD_LEVEL,
        ContractError,
        ProducerResult,
        Trajectory,
        _du_bytes,
        atomic_write_json,
        build_environment_cache,
        canonical_json_bytes,
        compute_global_calibration,
        crop_metric_depth,
        decode_depth_value,
        decode_trajectory,
        enumerate_environment,
        normalize_depth,
        producer_identity,
        sha256_bytes,
        sha256_file,
        streaming_chunks,
        utc_now,
    )
    from validate_depth_cache import (  # type: ignore[no-redef]
        validate_key_set_and_range,
        validate_temporal_gate,
    )


MAPANYTHING_REPOSITORY = "https://github.com/facebookresearch/map-anything.git"
MAPANYTHING_COMMIT = "ece461117701c786187f09da1bcf152de5c76f5d"
MODEL_REPOSITORY = "facebook/map-anything"
MODEL_REVISION = "a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa"
MODEL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "config.json": {
        "bytes": 5_776,
        "sha256": "65701d09d99ed37a21d295f0d138978b3d584ab3bccdbcb4a2853da212b676c5",
    },
    "model.safetensors": {
        "bytes": 4_914_062_480,
        "sha256": "981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0",
    },
}
CALIBRATION_FRAMES = 128
RECOMPUTE_MAX_ABS = 1e-3
DEFAULT_BATCH_SIZE = 8
DEFAULT_SHARD_COUNT = 8
MAPANYTHING_ENVIRONMENTS = ("pusht", "rope", "granular")


def _git_output(root: Path, *args: str) -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"cannot audit MapAnything checkout {root}: {exc}") from exc
    return result.stdout.strip()


def verify_pinned_mapanything(mapanything_root: Path, model_dir: Path) -> dict[str, Any]:
    mapanything_root = mapanything_root.resolve()
    model_dir = model_dir.resolve()
    if _git_output(mapanything_root, "rev-parse", "HEAD") != MAPANYTHING_COMMIT:
        raise ContractError(
            f"MapAnything checkout is not pinned to {MAPANYTHING_COMMIT}"
        )
    dirty = _git_output(
        mapanything_root, "status", "--porcelain", "--untracked-files=no"
    )
    if dirty:
        raise ContractError(
            f"pinned MapAnything checkout has tracked modifications:\n{dirty}"
        )

    artifacts: dict[str, Any] = {}
    for name, expected in MODEL_ARTIFACTS.items():
        path = model_dir / name
        if not path.is_file():
            raise ContractError(f"missing pinned MapAnything artifact {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ContractError(
                f"artifact mismatch for {path}: bytes={size}, sha256={digest}; "
                f"expected bytes={expected['bytes']}, sha256={expected['sha256']}"
            )
        artifacts[name] = {"path": str(path), "bytes": size, "sha256": digest}
    return {
        "repository": MAPANYTHING_REPOSITORY,
        "commit": MAPANYTHING_COMMIT,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "mapanything_root": str(mapanything_root),
        "artifacts": artifacts,
    }


class MapAnythingFramewiseProducer:
    """Pinned MapAnything singleton-view producer with independent batching."""

    def __init__(
        self,
        mapanything_root: Path,
        model_dir: Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ContractError("MapAnything batch size must be positive")
        audit = verify_pinned_mapanything(mapanything_root, model_dir)
        try:
            import torch
        except Exception as exc:  # pragma: no cover - production dependency
            raise ContractError("PyTorch is unavailable") from exc
        if not torch.cuda.is_available():
            raise ContractError(
                "MapAnything production inference requires CUDA; no fallback is allowed"
            )
        if not torch.cuda.is_bf16_supported():
            raise ContractError("the pinned MapAnything contract requires BF16 support")

        root = mapanything_root.resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from mapanything.models import MapAnything
            from mapanything.utils.image import preprocess_inputs
        except Exception as exc:  # pragma: no cover - production dependency
            raise ContractError("the pinned MapAnything API cannot be imported") from exc

        self._torch = torch
        self._preprocess_inputs = preprocess_inputs
        self.batch_size = int(batch_size)
        try:
            model = MapAnything.from_pretrained(str(model_dir.resolve()))
        except Exception as exc:  # pragma: no cover - production dependency
            raise ContractError(
                "cannot load the pinned local MapAnything snapshot; network fallback is forbidden"
            ) from exc
        self._model = model.eval().to("cuda")
        self.provenance = {
            "name": "MapAnything-recovered-framewise",
            **audit,
            "recovery_status": {
                "producer_family": "RECOVERED",
                "source_commit": "RECOVERED",
                "original_invocation": "UNKNOWN",
                "original_checkpoint_snapshot": "UNKNOWN",
                "selected_checkpoint_basis": (
                    "documented MapAnything.from_pretrained('facebook/map-anything') "
                    "best-evidence choice, freshly pinned for P2a"
                ),
            },
            "settings": {
                "temporal_mode": "framewise_singleton_view",
                "batch_semantics": (
                    "literal_singleton_only"
                    if self.batch_size == 1
                    else "candidate_independent_singleton_scenes_on_batch_axis"
                ),
                "batch_size": self.batch_size,
                "preprocessing_api": "mapanything.utils.image.preprocess_inputs",
                "resize_mode": "fixed_mapping",
                "resolution_set": 518,
                "patch_size": 14,
                "rgb_channel_order": "RGB",
                "rgb_input_range": "uint8_[0,255]",
                "rgb_normalization": "dinov2",
                "inference_precision": "bfloat16_amp",
                "memory_efficient_inference": False,
                "apply_mask": True,
                "mask_edges": True,
                "edge_normal_threshold": 5.0,
                "edge_depth_threshold": 0.03,
                "apply_confidence_mask": False,
                "confidence_percentile": 10.0,
                "output": "pred['depth_z']",
                "raw_units": "metric_Z_depth_as_declared_by_MapAnything",
                "scientific_quantity": (
                    "later_pinned_mapanything_depth_z_proxy_not_lossless_original_training_input"
                ),
                "invalid_policy": (
                    "reject_nonfinite_or_negative_then_preserve_exact_zeros_written_by_"
                    "upstream_apply_mask_true_non_ambiguous_and_edge_mask"
                ),
                "custom_hole_fill": "forbidden",
                "prefix_invariance": "not_applicable_framewise",
                "goal_path": "same_singleton_frame_path",
                "pth_float32_rgb_quantization": "round_half_up_to_uint8_for_png_contract",
            },
        }

    def _preprocess_batch(self, frames_rgb: np.ndarray) -> Mapping[str, Any]:
        processed = self._preprocess_inputs(
            [{"img": frame} for frame in frames_rgb],
            resize_mode="fixed_mapping",
            size=None,
            norm_type="dinov2",
            patch_size=14,
            resolution_set=518,
            verbose=False,
        )
        images = self._torch.cat([view["img"] for view in processed], dim=0)
        return {
            "img": images,
            "data_norm_type": ["dinov2"] * len(processed),
        }

    def infer_independent_frames(
        self, frames_rgb: np.ndarray, batch_size: int | None = None
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        frames = np.asarray(frames_rgb)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ContractError(
                f"MapAnything frames must be contiguous uint8 THWC RGB, got {frames.shape}/{frames.dtype}"
            )
        effective_batch = int(batch_size or self.batch_size)
        if effective_batch <= 0:
            raise ContractError("effective MapAnything batch size must be positive")

        depths: list[np.ndarray] = []
        invalid_pixels = 0
        total_pixels = 0
        metric_scales: list[float] = []
        processed_shapes: set[tuple[int, int]] = set()
        self._torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        for start in range(0, len(frames), effective_batch):
            # MapAnything interprets its tensor batch axis as views of one scene,
            # not unrelated image samples.  Keep the requested grouping only for
            # deterministic scheduling; every model invocation is a literal
            # one-frame view so cache bytes retain their singleton semantics.
            for frame in frames[start : start + effective_batch]:
                view = self._preprocess_batch(frame[None, ...])
                outputs = self._model.infer(
                    [view],
                    memory_efficient_inference=False,
                    use_amp=True,
                    amp_dtype="bf16",
                    apply_mask=True,
                    mask_edges=True,
                    edge_normal_threshold=5.0,
                    edge_depth_threshold=0.03,
                    apply_confidence_mask=False,
                    confidence_percentile=10.0,
                )
                if len(outputs) != 1:
                    raise ContractError(
                        f"singleton-view MapAnything returned {len(outputs)} view outputs"
                    )
                output = outputs[0]
                raw = output["depth_z"].detach().float().cpu().numpy()[..., 0]
                if raw.shape[0] != 1 or raw.ndim != 3:
                    raise ContractError(f"MapAnything depth has invalid shape {raw.shape}")
                if not np.isfinite(raw).all() or np.any(raw < 0):
                    raise ContractError("MapAnything returned non-finite or negative Z-depth")
                mask = output.get("mask")
                if mask is not None:
                    valid = mask.detach().cpu().numpy()[..., 0].astype(bool)
                    if valid.shape != raw.shape:
                        raise ContractError("MapAnything mask shape differs from depth_z")
                    if np.any(raw[~valid] != 0):
                        raise ContractError(
                            "MapAnything invalid-mask pixels are not exact upstream zeros"
                        )
                    invalid_pixels += int(valid.size - np.count_nonzero(valid))
                    total_pixels += int(valid.size)
                else:
                    total_pixels += int(raw.size)
                scales = output.get("metric_scaling_factor")
                if scales is not None:
                    metric_scales.extend(
                        float(value)
                        for value in scales.detach().float().cpu().reshape(-1).tolist()
                    )
                processed_shapes.add((int(raw.shape[1]), int(raw.shape[2])))
                depths.extend(np.asarray(value, dtype=np.float32).copy() for value in raw)

        elapsed = time.monotonic() - started
        metadata = {
            "framewise": True,
            "input_frames": len(frames),
            "batch_size": effective_batch,
            "model_batch_size": 1,
            "batch_semantics": "literal_singleton_only",
            "inference_seconds": elapsed,
            "inference_frames_per_second": len(frames) / elapsed if elapsed else None,
            "processed_depth_hw": [list(shape) for shape in sorted(processed_shapes)],
            "invalid_mask_fraction": invalid_pixels / total_pixels if total_pixels else 0.0,
            "metric_scaling_factor_min": min(metric_scales) if metric_scales else None,
            "metric_scaling_factor_max": max(metric_scales) if metric_scales else None,
            "peak_cuda_bytes": int(self._torch.cuda.max_memory_allocated()),
        }
        return depths, metadata

    def infer_trajectory(
        self, frames_rgb: np.ndarray, trajectory: Trajectory
    ) -> ProducerResult:
        if len(frames_rgb) != trajectory.frame_count:
            raise ContractError(
                f"{trajectory.trajectory_key}: decoded frame count differs from manifest"
            )
        depths, metadata = self.infer_independent_frames(frames_rgb)
        metadata = {
            **metadata,
            "compatibility_chunk_metadata_only": True,
            "prefix_invariance_gate": "NOT_APPLICABLE_FRAMEWISE",
            "goal_gauge": "identical_singleton_path_by_construction",
        }
        return ProducerResult(
            depth_m=depths,
            # The cache schema requires these fields.  They describe output-key
            # grouping only and do not introduce temporal context into inference.
            chunk_boundaries=streaming_chunks(trajectory.frame_count),
            alignments=[],
            metadata=metadata,
        )


def select_training_calibration_keys(
    trajectories: Sequence[Trajectory], count: int = CALIBRATION_FRAMES
) -> list[str]:
    candidates = [
        trajectory.logical_key(frame)
        for trajectory in trajectories
        if trajectory.split == "train"
        for frame in range(trajectory.frame_count)
    ]
    if len(candidates) < count:
        raise ContractError(
            f"training split has {len(candidates)} frames, fewer than {count} calibration keys"
        )
    candidates.sort(key=lambda key: (sha256_bytes(key.encode()), key))
    return candidates[:count]


def _trajectory_by_key(trajectories: Sequence[Trajectory]) -> dict[str, Trajectory]:
    result = {trajectory.trajectory_key: trajectory for trajectory in trajectories}
    if len(result) != len(trajectories):
        raise ContractError("duplicate PushT trajectory key")
    return result


def generate_training_calibration(
    trajectories: Sequence[Trajectory], producer: MapAnythingFramewiseProducer
) -> dict[str, Any]:
    environments = {trajectory.environment for trajectory in trajectories}
    if len(environments) != 1:
        raise ContractError("calibration trajectories must belong to one environment")
    selected_environment = next(iter(environments))
    keys = select_training_calibration_keys(trajectories)
    lookup = _trajectory_by_key(trajectories)
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in keys:
        environment, split, episode, frame = key.split("/")
        if environment != selected_environment or split != "train":
            raise ContractError(f"non-training calibration key selected: {key}")
        grouped[f"{split}/{episode}"].append((key, int(frame)))

    frames: list[np.ndarray] = []
    ordered_keys: list[str] = []
    decoded_hw: list[tuple[int, int]] = []
    for trajectory_key in sorted(grouped):
        trajectory = lookup[trajectory_key]
        decoded = decode_trajectory(trajectory)
        for key, frame in grouped[trajectory_key]:
            frames.append(decoded[frame])
            ordered_keys.append(key)
            decoded_hw.append((int(decoded[frame].shape[0]), int(decoded[frame].shape[1])))
    raw_depths, metadata = producer.infer_independent_frames(np.stack(frames))
    cropped_by_key = {
        key: crop_metric_depth(raw, hw)
        for key, raw, hw in zip(ordered_keys, raw_depths, decoded_hw)
    }
    if set(cropped_by_key) != set(keys):
        raise ContractError("calibration inference did not reproduce the selected key set")
    lo, hi = compute_global_calibration(cropped_by_key[key] for key in keys)
    return {
        "scope": "environment_training_only_mapanything_proxy",
        "environment": selected_environment,
        "selected_environments": [selected_environment],
        "selection": "128_smallest_sha256_training_frame_keys",
        "per_environment": CALIBRATION_FRAMES,
        "frame_key_format": "<env>/<split>/<episode:05d>/<frame:06d>",
        "keys": keys,
        "keys_sha256": sha256_bytes(canonical_json_bytes(keys)),
        "sample": "cropped_metric_depth[::8,::8]",
        "sample_stride": CALIBRATION_STRIDE,
        "percentiles": [2.0, 98.0],
        "percentile_method": "numpy.linear",
        "lo": lo,
        "hi": hi,
        "raw_units": "later_pinned_MapAnything_pred_depth_z_declared_metric_proxy",
        "wire_adaptation": "clip((depth_z-lo)/(hi-lo),0,1)",
        "inference_metadata": metadata,
    }


def load_training_calibration(
    path: Path,
    trajectories: Sequence[Trajectory],
    producer: MapAnythingFramewiseProducer,
) -> dict[str, Any]:
    document = json.loads(path.read_text())
    calibration = document.get("calibration")
    if not isinstance(calibration, dict):
        raise ContractError(f"{path} has no calibration object")
    if document.get("producer_identity") != producer_identity(producer.provenance):
        raise ContractError("calibration producer identity differs from the live producer")
    if calibration.get("keys") != select_training_calibration_keys(trajectories):
        raise ContractError("calibration keys differ from the current PushT training split")
    lo, hi = calibration.get("lo"), calibration.get("hi")
    if (
        not isinstance(lo, (int, float))
        or not isinstance(hi, (int, float))
        or not math.isfinite(lo)
        or not math.isfinite(hi)
        or not hi > lo + 1e-6
    ):
        raise ContractError("calibration lo/hi are invalid")
    return calibration


def _expected_wire(calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw_depth_wire = (
        calibration is not None
        and calibration.get("wire_mode") == "raw_depth_z_float16"
    )
    return {
        "physical_key": "<split>/<episode:05d>/<frame:06d>",
        "dtype": "<f2",
        "shape": list(OUTPUT_SHAPE),
        "order": "C",
        "compressor": "zstd",
        "compressor_level": ZSTD_LEVEL,
        "map_size": MAP_SIZE,
        "normalization": (
            "none_raw_depth_z_float16"
            if raw_depth_wire
            else "clip((depth_m-lo)/(hi-lo),0,1)"
        ),
        "inverted": False,
    }


def _source_index_sha256(trajectories: Sequence[Trajectory]) -> str:
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


def plan_trajectory_shards(
    trajectories: Sequence[Trajectory], shard_count: int
) -> list[list[Trajectory]]:
    """Assign whole trajectories with deterministic largest-first balancing."""

    if shard_count <= 0:
        raise ContractError("shard count must be positive")
    if shard_count > len(trajectories):
        raise ContractError("shard count exceeds the number of trajectories")
    original_order = {
        trajectory.trajectory_key: ordinal
        for ordinal, trajectory in enumerate(trajectories)
    }
    if len(original_order) != len(trajectories):
        raise ContractError("duplicate trajectory key in shard planner")
    shards: list[list[Trajectory]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    largest_first = sorted(
        trajectories,
        key=lambda trajectory: (
            -trajectory.frame_count,
            original_order[trajectory.trajectory_key],
        ),
    )
    for trajectory in largest_first:
        shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[shard_index].append(trajectory)
        loads[shard_index] += trajectory.frame_count
    for shard in shards:
        shard.sort(key=lambda trajectory: original_order[trajectory.trajectory_key])
    return shards


def _shard_plan_document(
    trajectories: Sequence[Trajectory], shard_count: int
) -> dict[str, Any]:
    environments = {trajectory.environment for trajectory in trajectories}
    if len(environments) != 1:
        raise ContractError("shard plan trajectories must belong to one environment")
    environment = next(iter(environments))
    shards = plan_trajectory_shards(trajectories, shard_count)
    assignments = [
        {
            "shard_index": index,
            "trajectory_count": len(shard),
            "frame_count": sum(item.frame_count for item in shard),
            "source_index_sha256": _source_index_sha256(shard),
            "trajectory_keys": [item.trajectory_key for item in shard],
        }
        for index, shard in enumerate(shards)
    ]
    core = {
        "schema": "dinocular-mapanything-shard-plan-v1",
        "environment": environment,
        "shard_count": shard_count,
        "full_trajectory_count": len(trajectories),
        "full_frame_count": sum(item.frame_count for item in trajectories),
        "full_source_index_sha256": _source_index_sha256(trajectories),
        "assignments": assignments,
    }
    return {
        **core,
        "plan_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def merge_shards(
    *,
    root: Path,
    shards_root: Path,
    output_root: Path,
    shard_count: int,
    rebuild: bool,
    environment: str = "pusht",
) -> dict[str, Any]:
    """Verify immutable trajectory shards and merge them without recomputation."""

    try:
        import lmdb
    except Exception as exc:
        raise ContractError("LMDB is required") from exc

    trajectories = enumerate_environment(root, environment)
    shard_plan = _shard_plan_document(trajectories, shard_count)
    planned_shards = plan_trajectory_shards(trajectories, shard_count)
    shard_manifests: list[dict[str, Any]] = []
    shard_databases: list[Any] = []
    common_producer: dict[str, Any] | None = None
    common_calibration: dict[str, Any] | None = None
    common_tool_sha256: str | None = None
    generation_seconds = 0.0
    try:
        for index, expected in enumerate(planned_shards):
            shard_root = shards_root / f"shard-{index:03d}-of-{shard_count:03d}"
            cache_dir = shard_root / f"{environment}.lmdb"
            manifest_path = cache_dir / "manifest.json"
            if not manifest_path.is_file():
                raise ContractError(f"missing shard manifest {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            shard_document_path = shard_root / "shard.json"
            if not shard_document_path.is_file():
                raise ContractError(f"missing shard record {shard_document_path}")
            shard_document = json.loads(shard_document_path.read_text())
            expected_keys = [trajectory.trajectory_key for trajectory in expected]
            assignment = shard_plan["assignments"][index]
            records = manifest.get("trajectories")
            actual_keys = (
                [record.get("trajectory_key") for record in records]
                if isinstance(records, list)
                else []
            )
            checks = {
                "schema": manifest.get("schema") == "dinocular-depth-cache-v1",
                "environment": manifest.get("environment") == environment,
                "wire_format": manifest.get("wire_format")
                == _expected_wire(manifest.get("calibration")),
                "trajectory_count": manifest.get("trajectory_count") == len(expected),
                "frame_count": manifest.get("frame_count")
                == sum(trajectory.frame_count for trajectory in expected),
                "source_index": manifest.get("source_index_sha256")
                == _source_index_sha256(expected),
                "tool_sha256": isinstance(manifest.get("tool_sha256"), str)
                and len(manifest["tool_sha256"]) == 64,
                "trajectory_keys": actual_keys == expected_keys,
                "closed_before_hash": manifest.get("closed_before_hash") is True,
                "data_hash": (cache_dir / "data.mdb").is_file()
                and manifest.get("data_mdb_sha256")
                == sha256_file(cache_dir / "data.mdb"),
                "shard_record": shard_document
                == {
                    "schema": "dinocular-mapanything-shard-v1",
                    "created_utc": shard_document.get("created_utc"),
                    "plan_sha256": shard_plan["plan_sha256"],
                    **assignment,
                    "manifest_id": manifest.get("manifest_id"),
                    "data_mdb_sha256": manifest.get("data_mdb_sha256"),
                },
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise ContractError(
                    f"shard {index} failed immutable checks: {', '.join(failed)}"
                )
            producer = manifest.get("producer")
            calibration = manifest.get("calibration")
            if not isinstance(producer, dict) or not isinstance(calibration, dict):
                raise ContractError(f"shard {index} lacks producer or calibration")
            if common_producer is None:
                common_producer = producer
                common_calibration = calibration
                common_tool_sha256 = manifest.get("tool_sha256")
            elif producer_identity(producer) != producer_identity(common_producer):
                raise ContractError(f"shard {index} producer identity differs")
            elif calibration != common_calibration:
                raise ContractError(f"shard {index} calibration differs")
            elif manifest.get("tool_sha256") != common_tool_sha256:
                raise ContractError(f"shard {index} tool hash differs")
            generation_seconds += float(manifest["build_metrics"]["elapsed_seconds"])
            shard_manifests.append(manifest)
            shard_databases.append(
                lmdb.open(
                    str(cache_dir),
                    subdir=True,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=32,
                )
            )

        if common_producer is None or common_calibration is None:
            raise ContractError("no shard metadata was loaded")
        destination = output_root / f"{environment}.lmdb"
        if destination.exists() and not rebuild:
            manifest_path = destination / "manifest.json"
            if not manifest_path.is_file():
                raise ContractError(f"existing final cache is incomplete: {destination}")
            existing = json.loads(manifest_path.read_text())
            reusable = (
                existing.get("source_index_sha256") == _source_index_sha256(trajectories)
                and existing.get("trajectory_count") == len(trajectories)
                and existing.get("frame_count")
                == sum(trajectory.frame_count for trajectory in trajectories)
                and existing.get("wire_format")
                == _expected_wire(existing.get("calibration"))
                and existing.get("calibration") == common_calibration
                and producer_identity(existing.get("producer", {}))
                == producer_identity(common_producer)
                and (destination / "data.mdb").is_file()
                and existing.get("data_mdb_sha256")
                == sha256_file(destination / "data.mdb")
            )
            if reusable:
                return existing
            raise ContractError(
                f"{destination} exists but is not the verified full merge; resolve and "
                "permanently delete the exact superseded target before rebuilding"
            )
        if destination.exists():
            raise ContractError(
                f"{destination} already exists; retained-copy replacement is forbidden. "
                "Resolve consumers and permanently delete the exact superseded target first."
            )

        output_root.mkdir(parents=True, exist_ok=True)
        manifest_id = str(uuid.uuid4())
        building = output_root / f".{environment}.lmdb.building-{manifest_id}"
        building.mkdir(parents=True)
        database = lmdb.open(
            str(building),
            subdir=True,
            map_size=MAP_SIZE,
            readonly=False,
            create=True,
            lock=True,
            sync=True,
            metasync=True,
            map_async=False,
            writemap=False,
            readahead=False,
            meminit=False,
            max_dbs=1,
        )
        record_maps = [
            {record["trajectory_key"]: record for record in manifest["trajectories"]}
            for manifest in shard_manifests
        ]
        shard_for_key = {
            trajectory.trajectory_key: index
            for index, shard in enumerate(planned_shards)
            for trajectory in shard
        }
        merged_records: list[dict[str, Any]] = []
        started = time.monotonic()
        try:
            for ordinal, trajectory in enumerate(trajectories):
                shard_index = shard_for_key[trajectory.trajectory_key]
                source_database = shard_databases[shard_index]
                with source_database.begin(write=False) as source_transaction:
                    with database.begin(write=True) as destination_transaction:
                        for frame in range(trajectory.frame_count):
                            key = trajectory.physical_key(frame).encode("ascii")
                            payload = source_transaction.get(key)
                            if payload is None:
                                raise ContractError(
                                    f"shard {shard_index} lacks {key.decode()}"
                                )
                            if not destination_transaction.put(
                                key, payload, overwrite=False
                            ):
                                raise ContractError(f"duplicate merged key {key.decode()}")
                database.sync(True)
                record = copy.deepcopy(record_maps[shard_index][trajectory.trajectory_key])
                record["commit_ordinal"] = ordinal
                merged_records.append(record)
                print(
                    json.dumps(
                        {
                            "event": "trajectory_merged",
                            "trajectory": trajectory.trajectory_key,
                            "completed_trajectories": ordinal + 1,
                            "total_trajectories": len(trajectories),
                            "utc": utc_now(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except BaseException:
            database.close()
            raise
        database.close()

        data_hash = sha256_file(building / "data.mdb")
        merge_seconds = time.monotonic() - started
        logical_bytes, allocated_bytes = _du_bytes(building)
        total_frames = sum(trajectory.frame_count for trajectory in trajectories)
        manifest = {
            "schema": "dinocular-depth-cache-v1",
            "manifest_id": manifest_id,
            "created_utc": utc_now(),
            "environment": environment,
            "trajectory_count": len(trajectories),
            "frame_count": total_frames,
            "source_index_sha256": _source_index_sha256(trajectories),
            "producer": common_producer,
            "calibration": common_calibration,
            "wire_format": _expected_wire(common_calibration),
            "tool_sha256": sha256_file(Path(__file__)),
            "trajectories": merged_records,
            "build_metrics": {
                "elapsed_seconds": generation_seconds,
                "committed_frames_per_second": total_frames / generation_seconds,
                "merge_seconds": merge_seconds,
                "generation_gpu_seconds_sum": generation_seconds,
                "shard_count": shard_count,
                "shard_plan_sha256": shard_plan["plan_sha256"],
                "shard_manifest_ids": [
                    manifest["manifest_id"] for manifest in shard_manifests
                ],
                "lmdb_logical_bytes_before_manifest": logical_bytes,
                "lmdb_allocated_bytes_before_manifest": allocated_bytes,
                "logical_bytes_per_frame": logical_bytes / total_frames,
                "allocated_bytes_per_frame": allocated_bytes / total_frames,
                "peak_host_rss_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
            },
            "data_mdb_sha256": data_hash,
            "closed_before_hash": True,
            "merge_provenance": {
                "schema": "dinocular-mapanything-merge-v1",
                "shards_root": str(shards_root),
                "shard_plan": shard_plan,
                "copied_compressed_payloads_without_recomputation": True,
            },
        }
        atomic_write_json(building / "manifest.json", manifest)
        if destination.exists():
            raise ContractError(
                f"{destination} appeared during the immutable merge; refusing to replace or archive it"
            )
        os.replace(building, destination)
        atomic_write_json(output_root / "shard_plan.json", shard_plan)
        return manifest
    finally:
        for database in shard_databases:
            database.close()


def _selected_spot_frames(
    trajectories: Sequence[Trajectory], count: int
) -> list[tuple[Trajectory, int]]:
    candidates = [
        (trajectory, frame)
        for trajectory in trajectories
        for frame in range(trajectory.frame_count)
    ]
    candidates.sort(
        key=lambda item: (
            sha256_bytes(item[0].logical_key(item[1]).encode()),
            item[0].logical_key(item[1]),
        )
    )
    return candidates[:count]


def validate_cache(
    *,
    root: Path,
    cache_root: Path,
    mapanything_root: Path,
    model_dir: Path,
    batch_size: int,
    spot_frames: int,
    max_trajectories: int | None,
    environment: str,
) -> dict[str, Any]:
    try:
        import lmdb
        import zstandard
    except Exception as exc:
        raise ContractError("LMDB and zstandard are required") from exc
    if spot_frames <= 0:
        raise ContractError("spot recomputation frame count must be positive")
    all_trajectories = enumerate_environment(root, environment)
    trajectories = all_trajectories
    if max_trajectories is not None:
        if max_trajectories <= 0:
            raise ContractError("--max-trajectories must be positive")
        trajectories = all_trajectories[:max_trajectories]
    cache_dir = cache_root / f"{environment}.lmdb"
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"missing complete cache manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "dinocular-depth-cache-v1":
        raise ContractError("cache schema differs from the shared cache schema")
    if manifest.get("environment") != environment:
        raise ContractError("cache environment differs")
    calibration = manifest.get("calibration", {})
    if manifest.get("wire_format") != _expected_wire(calibration):
        raise ContractError("MapAnything LMDB wire format differs from its contract")
    if manifest.get("trajectory_count") != len(trajectories):
        raise ContractError("manifest trajectory count differs from dataset")
    expected_frames = sum(trajectory.frame_count for trajectory in trajectories)
    if manifest.get("frame_count") != expected_frames:
        raise ContractError("manifest frame count differs from dataset")
    if manifest.get("source_index_sha256") != _source_index_sha256(trajectories):
        raise ContractError("manifest source index differs from dataset")
    raw_depth_wire = calibration.get("wire_mode") == "raw_depth_z_float16"
    if raw_depth_wire:
        if calibration != {
            "wire_mode": "raw_depth_z_float16",
            "checkpoint_compatibility": (
                "original_student_loader_np_load_float32_without_depth_normalization"
            ),
            "quantity": "later_pinned_MapAnything_pred_depth_z_proxy",
            "units": "declared_metric_Z_depth_nonphysical_later_checkpoint_proxy",
            "invalid_policy": (
                "reject_nonfinite_or_negative_preserve_upstream_masked_exact_zero"
            ),
            "calibration": "none",
        }:
            raise ContractError("raw depth_z compatibility declaration differs")
    else:
        if calibration.get("keys") != select_training_calibration_keys(all_trajectories):
            raise ContractError("manifest does not use the fixed training-only calibration")
        if any("/valid/" in key for key in calibration["keys"]):
            raise ContractError("validation leakage in calibration key set")

    records = manifest.get("trajectories")
    if not isinstance(records, list) or len(records) != len(trajectories):
        raise ContractError("trajectory records are incomplete")
    record_by_key = {record["trajectory_key"]: record for record in records}

    # Load and exercise the large producer before mapping LMDB. This follows the
    # proven generation-process ordering and avoids retaining both mappings while
    # the 4.9 GB checkpoint and DINOv2 backbone are materialized.
    print(json.dumps({"event": "producer_gate_start", "utc": utc_now()}), flush=True)
    # The completed cache was produced through the pinned literal-singleton
    # configuration.  Keep that identity intact, then exercise the frozen
    # validation batch size explicitly below against singleton recomputation.
    producer = MapAnythingFramewiseProducer(mapanything_root, model_dir, batch_size=1)
    if producer_identity(manifest.get("producer", {})) != producer_identity(
        producer.provenance
    ):
        raise ContractError("live MapAnything provenance differs from manifest")
    selected = _selected_spot_frames(trajectories, spot_frames)
    frame_arrays: list[np.ndarray] = []
    decoded_sizes: list[tuple[int, int]] = []
    for trajectory, frame in selected:
        decoded = decode_trajectory(trajectory)[frame]
        frame_arrays.append(decoded)
        decoded_sizes.append((int(decoded.shape[0]), int(decoded.shape[1])))
    raw_depths, raw_metadata = producer.infer_independent_frames(
        np.stack(frame_arrays), batch_size=batch_size
    )
    singleton_depths, _ = producer.infer_independent_frames(
        np.stack(frame_arrays), batch_size=1
    )
    batch_max_abs = max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(raw_depths, singleton_depths)
    )
    if batch_max_abs > RECOMPUTE_MAX_ABS:
        raise ContractError(
            f"independent batch versus singleton error {batch_max_abs} exceeds {RECOMPUTE_MAX_ABS}"
        )
    if raw_depth_wire:
        recomputed = [
            crop_metric_depth(raw, decoded_hw)
            for raw, decoded_hw in zip(raw_depths, decoded_sizes)
        ]
    else:
        lo, hi = float(calibration["lo"]), float(calibration["hi"])
        recomputed = [
            normalize_depth(crop_metric_depth(raw, decoded_hw), lo, hi)
            for raw, decoded_hw in zip(raw_depths, decoded_sizes)
        ]
    # The LMDB contract is float16.  Compare the independently recomputed
    # samples after that exact lossless-for-the-wire round-trip, rather than
    # treating the expected float32 intermediate as persisted cache bytes.
    recomputed_after_wire = [
        np.asarray(value, dtype=WIRE_DTYPE).astype(np.float32)
        for value in recomputed
    ]
    del producer
    print(
        json.dumps(
            {
                "event": "producer_gate_pass",
                "batch_max_absolute_error": batch_max_abs,
                "utc": utc_now(),
            }
        ),
        flush=True,
    )

    print(json.dumps({"event": "lmdb_gates_start", "utc": utc_now()}), flush=True)
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
        range_gate = validate_key_set_and_range(
            database,
            environment,
            trajectories,
            wire_minimum=0.0,
            wire_maximum=(
                float(np.finfo(WIRE_DTYPE).max) if raw_depth_wire else 1.0
            ),
        )
        temporal_gate = validate_temporal_gate(
            database,
            environment,
            trajectories,
            record_by_key,
            enforce=False,
        )
        temporal_gate = {
            **temporal_gate,
            "acceptance": "CHARACTERIZATION_ONLY_FRAMEWISE",
        }
        maximum = 0.0
        decompressor = zstandard.ZstdDecompressor()
        with database.begin(write=False) as transaction:
            for (trajectory, frame), normalized in zip(selected, recomputed_after_wire):
                payload = transaction.get(trajectory.physical_key(frame).encode("ascii"))
                if payload is None:
                    raise ContractError(
                        f"spot recomputation key missing: {trajectory.physical_key(frame)}"
                    )
                cached = decode_depth_value(payload, decompressor).astype(np.float32)
                maximum = max(maximum, float(np.max(np.abs(normalized - cached))))
        if maximum > RECOMPUTE_MAX_ABS:
            raise ContractError(
                f"spot recomputation max abs {maximum} exceeds {RECOMPUTE_MAX_ABS}"
            )
    finally:
        database.close()
    print(
        json.dumps(
            {
                "event": "lmdb_gates_pass",
                "spot_max_absolute_error": maximum,
                "utc": utc_now(),
            }
        ),
        flush=True,
    )

    return {
        "schema": "dinocular-mapanything-cache-validation-v2",
        "environment": environment,
        "created_utc": utc_now(),
        "state": "PASS",
        "manifest_id": manifest["manifest_id"],
        "manifest_count_gate": {
            "dataset_trajectories": len(trajectories),
            "manifest_trajectories": manifest["trajectory_count"],
            "dataset_frames": expected_frames,
            "manifest_frames": manifest["frame_count"],
        },
        "format_compatibility_gate": {
            "schema": manifest["schema"],
            "wire_format": manifest["wire_format"],
            "producer_agnostic_fields_equal_to_da3": True,
        },
        "calibration_gate": (
            {
                "state": "NOT_APPLICABLE_RAW_CHECKPOINT_INPUT",
                "wire_mode": calibration["wire_mode"],
                "calibration": "none",
            }
            if raw_depth_wire
            else {
                "scope": calibration["scope"],
                "keys": len(calibration["keys"]),
                "keys_sha256": calibration["keys_sha256"],
                "validation_keys": 0,
                "lo": calibration["lo"],
                "hi": calibration["hi"],
            }
        ),
        "range_gate": range_gate,
        "spot_recomputation_gate": {
            "frames": spot_frames,
            "keys": [trajectory.logical_key(frame) for trajectory, frame in selected],
            "max_absolute_error_after_wire_decode": maximum,
            "threshold": RECOMPUTE_MAX_ABS,
            "raw_recompute_metadata": raw_metadata,
        },
        "independent_batch_equivalence_gate": {
            "frames": spot_frames,
            "production_batch_size": batch_size,
            "max_absolute_error": batch_max_abs,
            "threshold": RECOMPUTE_MAX_ABS,
        },
        "temporal_gate": temporal_gate,
        "prefix_invariance_gate": "NOT_APPLICABLE_FRAMEWISE",
        "goal_gauge_gate": "IDENTICAL_SINGLETON_PATH_BY_CONSTRUCTION",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build one MapAnything cache")
    build.add_argument("--environment", choices=MAPANYTHING_ENVIRONMENTS, default="pusht")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--model-dir", type=Path, required=True)
    build.add_argument("--mapanything-root", type=Path, required=True)
    build.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    build.add_argument("--max-trajectories", type=int)
    build.add_argument("--shard-count", type=int)
    build.add_argument("--shard-index", type=int)
    build.add_argument("--calibration-json", type=Path)
    build.add_argument("--rebuild", action="store_true")

    plan = subparsers.add_parser("plan", help="plan deterministic cache shards")
    plan.add_argument("--environment", choices=MAPANYTHING_ENVIRONMENTS, default="pusht")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    plan.add_argument("--json", type=Path)

    merge = subparsers.add_parser("merge", help="verify and merge cache shards")
    merge.add_argument("--environment", choices=MAPANYTHING_ENVIRONMENTS, default="pusht")
    merge.add_argument("--root", type=Path, required=True)
    merge.add_argument("--shards-root", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    merge.add_argument("--rebuild", action="store_true")

    validate = subparsers.add_parser("validate", help="validate a complete cache")
    validate.add_argument("--environment", choices=MAPANYTHING_ENVIRONMENTS, default="pusht")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--cache", type=Path, required=True)
    validate.add_argument("--model-dir", type=Path, required=True)
    validate.add_argument("--mapanything-root", type=Path, required=True)
    validate.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    validate.add_argument("--spot-frames", type=int, default=3)
    validate.add_argument("--max-trajectories", type=int)
    validate.add_argument("--json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    faulthandler.enable()
    args = parse_args(argv)
    if args.command == "plan":
        report = _shard_plan_document(
            enumerate_environment(args.root, args.environment), args.shard_count
        )
        if args.json:
            atomic_write_json(args.json, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "merge":
        manifest = merge_shards(
            root=args.root,
            shards_root=args.shards_root,
            output_root=args.out,
            shard_count=args.shard_count,
            environment=args.environment,
            rebuild=args.rebuild,
        )
        print(
            json.dumps(
                {
                    "manifest_id": manifest["manifest_id"],
                    "trajectories": manifest["trajectory_count"],
                    "frames": manifest["frame_count"],
                    "frames_per_second": manifest["build_metrics"][
                        "committed_frames_per_second"
                    ],
                    "data_mdb_sha256": manifest["data_mdb_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate":
        report = validate_cache(
            root=args.root,
            cache_root=args.cache,
            mapanything_root=args.mapanything_root,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            spot_frames=args.spot_frames,
            max_trajectories=args.max_trajectories,
            environment=args.environment,
        )
        if args.json:
            atomic_write_json(args.json, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0

    trajectories = enumerate_environment(args.root, args.environment)
    if (args.shard_count is None) != (args.shard_index is None):
        raise ContractError("--shard-count and --shard-index must be supplied together")
    if args.max_trajectories is not None and args.shard_count is not None:
        raise ContractError("--max-trajectories cannot be combined with sharding")
    if args.shard_count is not None and not 0 <= args.shard_index < args.shard_count:
        raise ContractError("--shard-index is outside --shard-count")
    producer = MapAnythingFramewiseProducer(
        args.mapanything_root, args.model_dir, batch_size=args.batch_size
    )
    calibration_path = args.calibration_json or args.out / "calibration.json"
    if args.environment in {"rope", "granular"}:
        if args.calibration_json is not None:
            raise ContractError(
                "Rope/Granular checkpoint-compatible raw depth_z forbids calibration"
            )
        calibration = {
            "wire_mode": "raw_depth_z_float16",
            "checkpoint_compatibility": (
                "original_student_loader_np_load_float32_without_depth_normalization"
            ),
            "quantity": "later_pinned_MapAnything_pred_depth_z_proxy",
            "units": "declared_metric_Z_depth_nonphysical_later_checkpoint_proxy",
            "invalid_policy": (
                "reject_nonfinite_or_negative_preserve_upstream_masked_exact_zero"
            ),
            "calibration": "none",
        }
    elif calibration_path.is_file():
        calibration = load_training_calibration(calibration_path, trajectories, producer)
    else:
        calibration = generate_training_calibration(trajectories, producer)
        atomic_write_json(
            calibration_path,
            {
                "schema": "dinocular-mapanything-calibration-v1",
                "created_utc": utc_now(),
                "producer_identity": producer_identity(producer.provenance),
                "calibration": calibration,
            },
        )
    selected_trajectories = trajectories
    if args.max_trajectories is not None:
        if args.max_trajectories <= 0:
            raise ContractError("--max-trajectories must be positive")
        selected_trajectories = trajectories[: args.max_trajectories]
    if args.shard_count is not None:
        selected_trajectories = plan_trajectory_shards(
            trajectories, args.shard_count
        )[args.shard_index]
    manifest = build_environment_cache(
        output_root=args.out,
        environment=args.environment,
        trajectories=selected_trajectories,
        calibration=calibration,
        producer=producer,
        rebuild=args.rebuild,
        tool_sha256=sha256_file(Path(__file__)),
    )
    if args.shard_count is not None:
        plan = _shard_plan_document(trajectories, args.shard_count)
        atomic_write_json(
            args.out / "shard.json",
            {
                "schema": "dinocular-mapanything-shard-v1",
                "created_utc": utc_now(),
                "plan_sha256": plan["plan_sha256"],
                **plan["assignments"][args.shard_index],
                "manifest_id": manifest["manifest_id"],
                "data_mdb_sha256": manifest["data_mdb_sha256"],
            },
        )
    print(
        json.dumps(
            {
                "manifest_id": manifest["manifest_id"],
                "trajectories": manifest["trajectory_count"],
                "frames": manifest["frame_count"],
                "frames_per_second": manifest["build_metrics"][
                    "committed_frames_per_second"
                ],
                "data_mdb_sha256": manifest["data_mdb_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"MAPANYTHING CACHE CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
