#!/usr/bin/env python3
"""Measure source RGB motion on the frozen Rope/Granular admission pairs.

The final depth-admission receipts define four episodes and eight selected
frames per episode.  This command reads those identities from the receipts,
checks the receipt-bound replacement manifests and released RGB manifests,
then measures RGB motion on the same 28 adjacent selected-frame pairs per
environment.  It does not read or modify depth payloads.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import posixpath
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ENVIRONMENTS = ("rope", "granular")
RGB_MOVING_THRESHOLD_UINT8 = 5.0
FROZEN_EPISODES = (0, 1, 3, 4)
FROZEN_FRAMES = (0, 2, 5, 8, 10, 13, 16, 19)
FROZEN_SELECTION_RULE = (
    "episodes_0_1_3_4_frames_0_2_5_8_10_13_16_19_frozen_before_values"
)
EXPECTED_PAIRS_PER_ENVIRONMENT = len(FROZEN_EPISODES) * (len(FROZEN_FRAMES) - 1)


class MeasurementError(RuntimeError):
    """Raised when an input is not the exact frozen measurement input."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise MeasurementError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def require_regular_file(path: Path, label: str) -> None:
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise MeasurementError(f"{label} is unavailable at {path}: {exc}") from exc
    if not stat_result.st_mode & 0o170000 == 0o100000:
        raise MeasurementError(f"{label} is not a regular file: {path}")
    if path.is_symlink():
        raise MeasurementError(f"{label} must not be a symlink: {path}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    require_regular_file(path, label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"{label} is not strict JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MeasurementError(f"{label} must be a JSON object: {path}")
    return value


def receipt_sha256(receipt: dict[str, Any]) -> str:
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def _relative_to_source_root(source_path: str, source_root: Path) -> str:
    expected_prefix = posixpath.join(source_root.as_posix().rstrip("/"), "")
    normalized = posixpath.normpath(source_path)
    if not normalized.startswith(expected_prefix):
        raise MeasurementError(
            "receipt RGB source path is outside the bound source root: "
            f"{source_path} vs {source_root}"
        )
    relative = normalized[len(expected_prefix) :]
    if not relative or relative.startswith("/") or relative == ".." or relative.startswith("../"):
        raise MeasurementError(f"invalid receipt RGB source relative path: {source_path}")
    return relative


def _selection(receipt: dict[str, Any], environment: str) -> list[dict[str, Any]]:
    if receipt.get("schema") != "dinocular.depth-admission-validation.v1":
        raise MeasurementError(f"{environment} receipt schema differs")
    if receipt.get("environment") != environment:
        raise MeasurementError(f"{environment} receipt environment differs")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or claimed != receipt_sha256(receipt):
        raise MeasurementError(f"{environment} receipt self-hash is invalid")

    fixed = receipt.get("fixed_selection")
    if not isinstance(fixed, dict):
        raise MeasurementError(f"{environment} receipt lacks fixed selection")
    if fixed.get("rule") != FROZEN_SELECTION_RULE:
        raise MeasurementError(f"{environment} frozen selection rule differs")
    identities = fixed.get("frame_identities")
    expected = [
        f"train/{episode:05d}/{frame:06d}"
        for episode in FROZEN_EPISODES
        for frame in FROZEN_FRAMES
    ]
    if identities != expected:
        raise MeasurementError(f"{environment} frozen frame identities differ")

    alignment = receipt.get("checks", {}).get("rgb_depth_frame_alignment")
    if not isinstance(alignment, dict) or alignment.get("state") != "PASS":
        raise MeasurementError(f"{environment} receipt RGB/depth alignment is not PASS")
    episodes = alignment.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(FROZEN_EPISODES):
        raise MeasurementError(f"{environment} RGB alignment episode coverage differs")
    by_key: dict[str, dict[str, Any]] = {}
    for row in episodes:
        if not isinstance(row, dict):
            raise MeasurementError(f"{environment} RGB alignment row is malformed")
        key = row.get("trajectory_key")
        if not isinstance(key, str) or key in by_key:
            raise MeasurementError(f"{environment} RGB alignment trajectory identity differs")
        by_key[key] = row

    grouped: list[dict[str, Any]] = []
    for episode in FROZEN_EPISODES:
        key = f"train/{episode:05d}"
        row = by_key.get(key)
        if row is None:
            raise MeasurementError(f"{environment} receipt lacks RGB alignment for {key}")
        if row.get("selected_frames") != list(FROZEN_FRAMES):
            raise MeasurementError(f"{environment} receipt selected frames differ for {key}")
        if row.get("decoded_source_frames") != 20 or row.get("decoded_source_hw") != [224, 224]:
            raise MeasurementError(f"{environment} receipt source geometry differs for {key}")
        if row.get("depth_hw") != [224, 224] or row.get("encoder_rgb_hw") != [224, 224]:
            raise MeasurementError(f"{environment} receipt aligned geometry differs for {key}")
        if row.get("orientation_transform") != "none" or row.get("resize_hw") != [224, 224]:
            raise MeasurementError(f"{environment} receipt RGB transform differs for {key}")
        source_path = row.get("source_path")
        source_sha256 = row.get("source_sha256")
        manifest_source_sha256 = row.get("manifest_source_sha256")
        if not isinstance(source_path, str) or not isinstance(source_sha256, str):
            raise MeasurementError(f"{environment} RGB source binding is malformed for {key}")
        if source_sha256 != manifest_source_sha256:
            raise MeasurementError(f"{environment} receipt RGB source/manifest hash differs for {key}")
        grouped.append(
            {
                "trajectory_key": key,
                "episode": episode,
                "frames": list(FROZEN_FRAMES),
                "source_path": source_path,
                "source_sha256": source_sha256,
            }
        )
    return grouped


def _manifest_index(manifest: dict[str, Any], environment: str) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != "dinocular-depth-cache-v1":
        raise MeasurementError(f"{environment} replacement manifest schema differs")
    if manifest.get("environment") != environment:
        raise MeasurementError(f"{environment} replacement manifest environment differs")
    if manifest.get("trajectory_count") != 1000 or manifest.get("frame_count") != 20000:
        raise MeasurementError(f"{environment} replacement manifest coverage differs")
    trajectories = manifest.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 1000:
        raise MeasurementError(f"{environment} replacement manifest trajectories differ")
    by_key: dict[str, dict[str, Any]] = {}
    for trajectory in trajectories:
        if not isinstance(trajectory, dict):
            raise MeasurementError(f"{environment} replacement trajectory is malformed")
        trajectory_key = trajectory.get("trajectory_key")
        source_path = trajectory.get("source_path")
        source_sha256 = trajectory.get("source_video_sha256")
        ordered_keys = trajectory.get("ordered_output_keys")
        if (
            not isinstance(trajectory_key, str)
            or not isinstance(source_path, str)
            or not isinstance(source_sha256, str)
            or not isinstance(ordered_keys, list)
        ):
            raise MeasurementError(f"{environment} replacement trajectory binding is malformed")
        for output_key in ordered_keys:
            if not isinstance(output_key, str) or output_key in by_key:
                raise MeasurementError(f"{environment} replacement output-key identity differs")
            by_key[output_key] = trajectory
    if len(by_key) != 20000:
        raise MeasurementError(f"{environment} replacement manifest key coverage differs")
    return by_key


def _rgb_manifest_index(manifest: dict[str, Any], environment: str) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != "dino-wm.rgb-dataset-manifest.v1":
        raise MeasurementError(f"{environment} RGB manifest schema differs")
    if manifest.get("environment") != environment:
        raise MeasurementError(f"{environment} RGB manifest environment differs")
    if manifest.get("file_count") != 1002:
        raise MeasurementError(f"{environment} RGB manifest file coverage differs")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 1002:
        raise MeasurementError(f"{environment} RGB manifest files differ")
    by_relative_path: dict[str, dict[str, Any]] = {}
    for file_record in files:
        if not isinstance(file_record, dict):
            raise MeasurementError(f"{environment} RGB manifest file record is malformed")
        relative_path = file_record.get("relative_path")
        if not isinstance(relative_path, str) or relative_path in by_relative_path:
            raise MeasurementError(f"{environment} RGB manifest path identity differs")
        by_relative_path[relative_path] = file_record
    return by_relative_path


def _load_rgb(path: Path, expected_sha256: str, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    require_regular_file(path, label)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise MeasurementError(
            f"{label} hash differs: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        stored = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch raises several pickle/storage exception types.
        raise MeasurementError(f"cannot decode {label}: {path}: {exc}") from exc
    array = np.asarray(stored)
    if array.ndim != 4:
        raise MeasurementError(f"{label} source shape is not 4D: {array.shape}")
    if array.shape[-1] == 3:
        layout = "THWC"
        frames = array
    elif array.shape[1] == 3:
        layout = "TCHW"
        frames = np.transpose(array, (0, 2, 3, 1))
    else:
        raise MeasurementError(f"{label} RGB channel axis is ambiguous: {array.shape}")
    if tuple(frames.shape[1:]) != (224, 224, 3):
        raise MeasurementError(f"{label} RGB geometry differs: {frames.shape}")
    rounded = np.rint(frames)
    if not np.array_equal(frames, rounded):
        raise MeasurementError(f"{label} RGB values are not integer-valued")
    if float(rounded.min()) < 0.0 or float(rounded.max()) > 255.0:
        raise MeasurementError(f"{label} RGB range differs")
    decoded = rounded.astype(np.uint8, copy=False)
    return decoded, {
        "path": str(path),
        "sha256": actual_sha256,
        "stored_shape": list(array.shape),
        "stored_dtype": array.dtype.str,
        "stored_layout": layout,
        "decoded_shape": list(decoded.shape),
        "decoded_dtype": decoded.dtype.str,
        "decoded_range": [int(decoded.min()), int(decoded.max())],
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        raise MeasurementError("cannot summarize an empty distribution")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise MeasurementError("RGB motion distribution contains a non-finite value")
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _receipt_aligned_rgb(decoded: np.ndarray, frames: list[int]) -> np.ndarray:
    """Reproduce the receipt's RGB alignment transform exactly."""
    raw = torch.from_numpy(decoded[np.asarray(frames, dtype=np.int64)])
    if raw.ndim != 4 or raw.shape[-1] != 3:
        raise MeasurementError(f"decoded RGB selection has unexpected shape: {tuple(raw.shape)}")
    rgb = raw.permute(0, 3, 1, 2).float() / 255.0
    height, width = rgb.shape[-2:]
    scale = 224.0 / min(height, width)
    resized = (int(round(height * scale)), int(round(width * scale)))
    rgb = F.interpolate(rgb, size=resized, mode="bilinear", align_corners=False)
    top = (resized[0] - 224) // 2
    left = (resized[1] - 224) // 2
    rgb = rgb[:, :, top : top + 224, left : left + 224]
    rgb = (rgb - 0.5) / 0.5
    return rgb.permute(0, 2, 3, 1).numpy()


def _measure_pairs(
    environment: str,
    groups: list[dict[str, Any]],
    rgb_by_path: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for group in groups:
        source_path = group["source_path"]
        frames = group["frames"]
        rgb = rgb_by_path[source_path]
        for left_frame, right_frame in zip(frames, frames[1:]):
            left = rgb[frames.index(left_frame)]
            right = rgb[frames.index(right_frame)]
            channel_delta = (
                np.abs(right.astype(np.float32) - left.astype(np.float32)) * 127.5
            )
            pixel_delta = channel_delta.mean(axis=-1)
            moving = pixel_delta >= RGB_MOVING_THRESHOLD_UINT8
            static = ~moving
            if not bool(moving.any()) or not bool(static.any()):
                raise MeasurementError(
                    f"{environment} pair lacks both moving and static RGB regions: "
                    f"{group['trajectory_key']}/{left_frame}->{right_frame}"
                )
            moving_values = pixel_delta[moving].astype(np.float64)
            static_values = pixel_delta[static].astype(np.float64)
            moving_mean = float(moving_values.mean())
            static_mean = float(static_values.mean())
            ratio = moving_mean / static_mean if static_mean > 0.0 else None
            if ratio is None or not np.isfinite(ratio):
                raise MeasurementError(
                    f"{environment} pair has no finite RGB moving/static ratio: "
                    f"{group['trajectory_key']}/{left_frame}->{right_frame}"
                )
            pairs.append(
                {
                    "trajectory_key": group["trajectory_key"],
                    "episode": group["episode"],
                    "left_frame": left_frame,
                    "right_frame": right_frame,
                    "left_identity": f"{group['trajectory_key']}/{left_frame:06d}",
                    "right_identity": f"{group['trajectory_key']}/{right_frame:06d}",
                    "source_path": source_path,
                    "rgb_motion": {
                        "mean_absolute_channel_delta_uint8": float(channel_delta.mean()),
                        "moving_region_mean_absolute_delta_uint8": moving_mean,
                        "static_region_mean_absolute_delta_uint8": static_mean,
                        "moving_to_static_delta_ratio": float(ratio),
                        "moving_pixel_fraction_at_5_uint8": float(moving.mean()),
                        "moving_pixel_count": int(moving.sum()),
                        "static_pixel_count": int(static.sum()),
                        "moving_region_pixel_delta_median_uint8": float(np.median(moving_values)),
                        "static_region_pixel_delta_median_uint8": float(np.median(static_values)),
                    },
                }
            )

    if len(pairs) != EXPECTED_PAIRS_PER_ENVIRONMENT:
        raise MeasurementError(
            f"{environment} pair count is {len(pairs)}, "
            f"not {EXPECTED_PAIRS_PER_ENVIRONMENT}"
        )
    metrics = [pair["rgb_motion"] for pair in pairs]
    moving_values = [float(row["moving_region_mean_absolute_delta_uint8"]) for row in metrics]
    static_values = [float(row["static_region_mean_absolute_delta_uint8"]) for row in metrics]
    ratio_values = [float(row["moving_to_static_delta_ratio"]) for row in metrics]
    full_values = [float(row["mean_absolute_channel_delta_uint8"]) for row in metrics]
    fraction_values = [float(row["moving_pixel_fraction_at_5_uint8"]) for row in metrics]
    moving_medians = float(np.median(np.asarray(moving_values, dtype=np.float64)))
    static_medians = float(np.median(np.asarray(static_values, dtype=np.float64)))
    ratio_median = float(np.median(np.asarray(ratio_values, dtype=np.float64)))
    return pairs, {
        "pair_count": len(pairs),
        "episode_count": len(groups),
        "selected_frame_count": sum(len(group["frames"]) for group in groups),
        "pairs_with_both_regions": sum(
            int(row["moving_pixel_count"] > 0 and row["static_pixel_count"] > 0)
            for row in metrics
        ),
        "moving_pixel_count_total": sum(int(row["moving_pixel_count"]) for row in metrics),
        "static_pixel_count_total": sum(int(row["static_pixel_count"]) for row in metrics),
        "moving_pixel_fraction_median": float(np.median(np.asarray(fraction_values))),
        "moving_region_mean_median_uint8": moving_medians,
        "static_region_mean_median_uint8": static_medians,
        "median_pair_moving_to_static_ratio": ratio_median,
        "ratio_of_moving_to_static_region_medians": (
            moving_medians / static_medians if static_medians > 0.0 else None
        ),
        "pair_fraction_moving_to_static_ge_2": float(
            np.mean(np.asarray(ratio_values) >= 2.0)
        ),
        "pair_fraction_moving_to_static_ge_3": float(
            np.mean(np.asarray(ratio_values) >= 3.0)
        ),
        "distributions": {
            "mean_absolute_channel_delta_uint8": _distribution(full_values),
            "moving_region_mean_absolute_delta_uint8": _distribution(moving_values),
            "static_region_mean_absolute_delta_uint8": _distribution(static_values),
            "moving_to_static_delta_ratio": _distribution(ratio_values),
            "moving_pixel_fraction_at_5_uint8": _distribution(fraction_values),
        },
    }


def _task_measurement(
    environment: str,
    bundle_root: Path,
    source_root: Path,
    replacement_root: Path,
    rgb_manifest_root: Path,
) -> dict[str, Any]:
    receipt_path = bundle_root / "final" / environment / "admission.json"
    receipt = load_json(receipt_path, f"{environment} final admission receipt")
    groups = _selection(receipt, environment)
    receipt_hash = receipt["receipt_sha256"]

    replacement_manifest_path = replacement_root / f"{environment}.lmdb" / "manifest.json"
    replacement_manifest = load_json(
        replacement_manifest_path, f"{environment} receipt-bound replacement manifest"
    )
    replacement_manifest_hash = sha256_file(replacement_manifest_path)
    expected_manifest_hash = receipt["artifacts"].get("cache_manifest_sha256")
    if replacement_manifest_hash != expected_manifest_hash:
        raise MeasurementError(
            f"{environment} replacement manifest hash differs from final receipt: "
            f"expected {expected_manifest_hash}, got {replacement_manifest_hash}"
        )
    producer_hash = sha256_bytes(canonical_json_bytes(replacement_manifest["producer"]))
    if producer_hash != receipt["artifacts"].get("producer_sha256"):
        raise MeasurementError(f"{environment} replacement producer hash differs from receipt")
    replacement_index = _manifest_index(replacement_manifest, environment)

    rgb_manifest_path = rgb_manifest_root / f"dataset_manifest_{environment}.json"
    rgb_manifest = load_json(rgb_manifest_path, f"{environment} released RGB manifest")
    rgb_manifest_hash = sha256_file(rgb_manifest_path)
    rgb_index = _rgb_manifest_index(rgb_manifest, environment)

    rgb_by_path: dict[str, np.ndarray] = {}
    source_records: list[dict[str, Any]] = []
    for group in groups:
        trajectory_key = group["trajectory_key"]
        replacement_record = replacement_index.get(trajectory_key + "/000000")
        if replacement_record is None:
            raise MeasurementError(
                f"{environment} replacement manifest lacks {trajectory_key}/000000"
            )
        replacement_source_path = replacement_record["source_path"]
        if replacement_source_path != _relative_to_source_root(
            group["source_path"], source_root
        ):
            raise MeasurementError(f"{environment} receipt/replacement RGB path differs for {trajectory_key}")
        if replacement_record["source_video_sha256"] != group["source_sha256"]:
            raise MeasurementError(f"{environment} receipt/replacement RGB hash differs for {trajectory_key}")
        rgb_record = rgb_index.get(replacement_source_path)
        if rgb_record is None:
            raise MeasurementError(f"{environment} released RGB manifest lacks {replacement_source_path}")
        if (
            rgb_record.get("split") != "train"
            or rgb_record.get("episode") != group["episode"]
            or rgb_record.get("type") != "obses"
            or rgb_record.get("sha256") != group["source_sha256"]
            or rgb_record.get("relative_path") != replacement_source_path
        ):
            raise MeasurementError(f"{environment} released RGB manifest binding differs for {trajectory_key}")
        source_path = source_root / replacement_source_path
        if not source_path.is_absolute():
            raise MeasurementError(f"{environment} source root is not absolute")
        decoded, source_info = _load_rgb(
            source_path,
            group["source_sha256"],
            f"{environment} {trajectory_key} RGB source",
        )
        if decoded.shape[0] != 20:
            raise MeasurementError(f"{environment} RGB frame count differs for {trajectory_key}")
        rgb_by_path[group["source_path"]] = _receipt_aligned_rgb(
            decoded, group["frames"]
        )
        source_info["aligned_rgb_shape"] = [len(group["frames"]), 224, 224, 3]
        source_info["aligned_rgb_value_range"] = [
            float(rgb_by_path[group["source_path"]].min()),
            float(rgb_by_path[group["source_path"]].max()),
        ]
        actual_bytes = source_path.stat().st_size
        if actual_bytes != rgb_record["bytes"]:
            raise MeasurementError(
                f"{environment} released RGB byte count differs for {trajectory_key}"
            )
        source_records.append(
            {
                **source_info,
                "trajectory_key": trajectory_key,
                "relative_path": replacement_source_path,
                "manifest_sha256": rgb_record["sha256"],
                "manifest_bytes": rgb_record["bytes"],
                "actual_bytes": actual_bytes,
                "receipt_source_sha256": group["source_sha256"],
                "replacement_manifest_source_sha256": replacement_record[
                    "source_video_sha256"
                ],
                "hash_checks": {
                    "state": "PASS",
                    "receipt_source_sha256_match": True,
                    "replacement_manifest_sha256_match": True,
                    "rgb_manifest_sha256_match": True,
                    "actual_source_sha256_match": True,
                    "actual_bytes_match": True,
                },
            }
        )

    pairs, rgb_summary = _measure_pairs(environment, groups, rgb_by_path)
    depth_check = receipt.get("checks", {}).get("moving_object_depth_correlation")
    if not isinstance(depth_check, dict):
        raise MeasurementError(f"{environment} receipt lacks depth motion comparison")
    depth_replacement = depth_check.get("replacement")
    depth_defective = depth_check.get("defective")
    depth_criteria = depth_check.get("criteria")
    if not isinstance(depth_replacement, dict) or not isinstance(depth_defective, dict):
        raise MeasurementError(f"{environment} receipt depth motion comparison is malformed")
    if depth_replacement.get("pair_count") != EXPECTED_PAIRS_PER_ENVIRONMENT:
        raise MeasurementError(f"{environment} receipt depth pair count differs")
    receipt_pairs = depth_replacement.get("pairs")
    if not isinstance(receipt_pairs, list) or len(receipt_pairs) != EXPECTED_PAIRS_PER_ENVIRONMENT:
        raise MeasurementError(f"{environment} receipt depth pair rows differ")
    mask_fraction_differences = []
    for index, (rgb_pair, depth_pair) in enumerate(zip(pairs, receipt_pairs)):
        receipt_fraction = depth_pair.get("moving_pixel_fraction")
        if not isinstance(receipt_fraction, (int, float)) or isinstance(receipt_fraction, bool):
            raise MeasurementError(f"{environment} receipt moving-mask fraction is malformed at pair {index}")
        difference = abs(
            float(rgb_pair["rgb_motion"]["moving_pixel_fraction_at_5_uint8"])
            - float(receipt_fraction)
        )
        if difference > 1e-12:
            raise MeasurementError(
                f"{environment} RGB moving mask differs from receipt at pair {index}: "
                f"difference {difference}"
            )
        mask_fraction_differences.append(difference)
    depth_ratio = float(depth_replacement["moving_to_static_ratio_median"])
    rgb_ratio = float(rgb_summary["median_pair_moving_to_static_ratio"])
    ratio_threshold = float(
        depth_criteria["moving_region_spatial_concentration"]["minimum_median_ratio"]
    )
    pair_fraction_threshold = float(
        depth_criteria["moving_region_spatial_concentration"][
            "minimum_pair_fraction_ratio_ge_2"
        ]
    )
    if rgb_ratio >= ratio_threshold and depth_ratio < ratio_threshold:
        conclusion_class = "depth_remains_specifically_deficient"
        conclusion = (
            f"{environment.capitalize()} source RGB contains strong same-pair "
            f"moving/static separation (median RGB ratio {rgb_ratio:.6g}, "
            f"with pair fraction {rgb_summary['pair_fraction_moving_to_static_ge_2']:.6g} "
            f"at ratio >= 2), while corrected depth is only {depth_ratio:.6g} "
            f"against the unchanged admission median threshold {ratio_threshold:.6g}. "
            "The low corrected-depth separation is therefore not explained by weak "
            "source-video motion; depth remains specifically deficient."
        )
    elif rgb_ratio < ratio_threshold and depth_ratio < ratio_threshold:
        conclusion_class = "source_video_motion_can_explain_low_depth_separation"
        conclusion = (
            f"{environment.capitalize()} source RGB has low same-pair moving/static "
            f"separation (median RGB ratio {rgb_ratio:.6g}) and corrected depth is "
            f"{depth_ratio:.6g}, both below the unchanged admission median threshold "
            f"{ratio_threshold:.6g}. Source-video motion can explain the low depth "
            "separation on these frozen pairs; this does not change the failed admission."
        )
    else:
        conclusion_class = "source_and_depth_separation_are_not_decisively_distinguished"
        conclusion = (
            f"{environment.capitalize()} source RGB median moving/static separation "
            f"is {rgb_ratio:.6g}; corrected depth is {depth_ratio:.6g}; the unchanged "
            f"admission median threshold is {ratio_threshold:.6g}. The frozen-pair "
            "comparison does not support a stronger source-versus-depth conclusion."
        )

    return {
        "environment": environment,
        "final_receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
            "receipt_sha256": receipt_hash,
            "state": receipt["state"],
            "fixed_selection": copy.deepcopy(receipt["fixed_selection"]),
        },
        "replacement_manifest": {
            "path": str(replacement_manifest_path),
            "sha256": replacement_manifest_hash,
            "manifest_id": replacement_manifest["manifest_id"],
            "data_mdb_sha256": replacement_manifest["data_mdb_sha256"],
            "producer_sha256": producer_hash,
            "receipt_bound_sha256": expected_manifest_hash,
        },
        "rgb_manifest": {
            "path": str(rgb_manifest_path),
            "sha256": rgb_manifest_hash,
            "source_commit": rgb_manifest.get("source_commit"),
            "file_count": rgb_manifest.get("file_count"),
            "total_bytes": rgb_manifest.get("total_bytes"),
        },
        "source_files": source_records,
        "rgb_input_hash_coverage": {
            "state": "PASS",
            "selected_source_file_count": len(source_records),
            "all_selected_sources_passed_receipt_and_manifest_hashes": all(
                record["hash_checks"]["state"] == "PASS" for record in source_records
            ),
        },
        "rgb_motion_rule": {
            "pair_set": "adjacent selected frames within each receipt-defined episode group",
            "alignment": "same float32 divide-by-255, resize/crop, and [-1,1] transform as final receipt",
            "pixel_motion_value": "mean absolute aligned RGB channel delta multiplied by 127.5",
            "moving_region": "pixel motion value >= 5.0 uint8-equivalent",
            "static_region": "complement of moving region",
            "threshold_uint8": RGB_MOVING_THRESHOLD_UINT8,
            "no_future_frames": True,
            "no_substitution": True,
        },
        "rgb_motion_summary": rgb_summary,
        "pairs": pairs,
        "receipt_mask_alignment": {
            "state": "PASS",
            "pair_count": len(mask_fraction_differences),
            "max_abs_moving_pixel_fraction_difference": max(mask_fraction_differences),
            "basis": "direct RGB 5-uint8 mask fraction equals each receipt replacement pair row",
        },
        "depth_comparison_from_receipt": {
            "scientific_basis": depth_check.get("scientific_basis"),
            "defective": copy.deepcopy(depth_defective),
            "replacement": copy.deepcopy(depth_replacement),
            "unchanged_admission_criteria": copy.deepcopy(depth_criteria),
        },
        "source_vs_depth_separation": {
            "rgb_median_pair_moving_to_static_ratio": rgb_ratio,
            "replacement_depth_median_pair_moving_to_static_ratio": depth_ratio,
            "rgb_to_replacement_depth_separation_ratio": rgb_ratio / depth_ratio,
            "unchanged_depth_median_ratio_threshold": ratio_threshold,
            "unchanged_depth_pair_fraction_ratio_ge_2_threshold": pair_fraction_threshold,
            "conclusion_class": conclusion_class,
            "plain_conclusion": conclusion,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--rgb-manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-code-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path, label in (
        (args.bundle_root, "bundle root"),
        (args.source_root, "RGB source root"),
        (args.replacement_root, "replacement root"),
        (args.rgb_manifest_root, "RGB manifest root"),
    ):
        if not path.is_absolute():
            raise MeasurementError(f"{label} must be absolute: {path}")
        if not path.is_dir():
            raise MeasurementError(f"{label} is not a directory: {path}")

    tasks = [
        _task_measurement(
            environment,
            args.bundle_root,
            args.source_root,
            args.replacement_root,
            args.rgb_manifest_root,
        )
        for environment in ENVIRONMENTS
    ]
    by_environment = {task["environment"]: task for task in tasks}
    observed_pair_count = sum(task["rgb_motion_summary"]["pair_count"] for task in tasks)
    result: dict[str, Any] = {
        "schema": "dinocular.rg-rgb-motion-measurement.v1",
        "read_only_inputs": True,
        "inputs_unchanged": True,
        "measurement_code_sha256": args.measurement_code_sha256,
        "invocation_argv": sys.argv,
        "frozen_contract": {
            "environments": list(ENVIRONMENTS),
            "selection_rule": FROZEN_SELECTION_RULE,
            "episodes": list(FROZEN_EPISODES),
            "frames": list(FROZEN_FRAMES),
            "pairs_per_environment": EXPECTED_PAIRS_PER_ENVIRONMENT,
            "total_pairs": EXPECTED_PAIRS_PER_ENVIRONMENT * len(ENVIRONMENTS),
            "rgb_motion_threshold_uint8": RGB_MOVING_THRESHOLD_UINT8,
            "depth_admission_values_and_thresholds_preserved": True,
            "depth_admission_receipts_not_rewritten": True,
        },
        "expected_reconciliation": {
            environment: {
                "selected_frames": len(FROZEN_EPISODES) * len(FROZEN_FRAMES),
                "adjacent_pairs": EXPECTED_PAIRS_PER_ENVIRONMENT,
            }
            for environment in ENVIRONMENTS
        },
        "observed_reconciliation": {
            environment: {
                "selected_frames": task["rgb_motion_summary"]["selected_frame_count"],
                "adjacent_pairs": task["rgb_motion_summary"]["pair_count"],
                "source_files": len(task["source_files"]),
            }
            for environment, task in by_environment.items()
        },
        "coverage_pass": observed_pair_count
        == EXPECTED_PAIRS_PER_ENVIRONMENT * len(ENVIRONMENTS),
        "tasks": tasks,
        "plain_conclusions": {
            environment: task["source_vs_depth_separation"]["plain_conclusion"]
            for environment, task in by_environment.items()
        },
    }
    result["measurement_sha256"] = sha256_bytes(canonical_json_bytes(result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "measurement_sha256": result["measurement_sha256"],
                "pair_counts": {
                    environment: task["rgb_motion_summary"]["pair_count"]
                    for environment, task in by_environment.items()
                },
                "conclusion_classes": {
                    environment: task["source_vs_depth_separation"]["conclusion_class"]
                    for environment, task in by_environment.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MeasurementError as exc:
        raise SystemExit(f"MEASUREMENT_ERROR: {exc}")
