#!/usr/bin/env python3
"""Read-only evidence collector for the Wall low-standard-deviation range gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tools.precompute_depth import (
    OUTPUT_SHAPE,
    crop_rgb_for_world_model,
    decode_depth_value,
    decode_trajectory,
    enumerate_environment,
)


RANGE_SAMPLE_FRAMES = 4096
LOW_STD_THRESHOLD = 1e-4
RATE_CANARY_TRAJECTORIES = (
    "train/00000",
    "train/00002",
    "train/00004",
    "train/00005",
    "train/00007",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _deciles(values: list[int], denominator: int) -> list[int]:
    result = [0] * 10
    for value in values:
        result[min(9, 10 * value // denominator)] += 1
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--montage", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import lmdb
    import zstandard
    from PIL import Image, ImageDraw

    trajectories = enumerate_environment(args.root, "wall")
    trajectory_by_key = {item.trajectory_key: item for item in trajectories}
    expected = [
        (trajectory, frame, trajectory.physical_key(frame))
        for trajectory in trajectories
        for frame in range(trajectory.frame_count)
    ]
    range_sample = sorted(
        expected,
        key=lambda item: (
            _sha256_text(f"wall/{item[2]}"),
            f"wall/{item[2]}",
        ),
    )[:RANGE_SAMPLE_FRAMES]
    range_keys = {item[2] for item in range_sample}
    canary_keys = {
        trajectory_by_key[key].physical_key(frame)
        for key in RATE_CANARY_TRAJECTORIES
        for frame in range(trajectory_by_key[key].frame_count)
    }

    manifest = json.loads((args.cache / "manifest.json").read_text())
    database = lmdb.open(
        str(args.cache),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=8,
    )
    decompressor = zstandard.ZstdDecompressor()
    all_flagged: list[dict[str, object]] = []
    range_metrics: list[dict[str, object]] = []
    canary_metrics: list[dict[str, object]] = []
    histogram = np.zeros(1 << 16, dtype=np.uint64)
    sampled_saturated = 0
    sampled_values = 0
    with database.begin(write=False) as transaction:
        for trajectory, frame, physical_key in expected:
            payload = transaction.get(physical_key.encode("ascii"))
            if payload is None:
                raise RuntimeError(f"missing cache key {physical_key}")
            depth = decode_depth_value(payload, decompressor)
            if depth.shape != OUTPUT_SHAPE or depth.dtype.str != "<f2":
                raise RuntimeError(f"bad wire value at {physical_key}")
            std = float(np.std(depth, dtype=np.float64))
            record = {
                "logical_key": f"wall/{physical_key}",
                "physical_key": physical_key,
                "trajectory": trajectory.trajectory_key,
                "episode": trajectory.episode,
                "frame": frame,
                "depth_mean": float(np.mean(depth, dtype=np.float64)),
                "depth_std": std,
                "depth_min": float(np.min(depth)),
                "depth_max": float(np.max(depth)),
                "depth_saturation_fraction": float(
                    np.mean((depth == 0) | (depth == 1))
                ),
            }
            if std <= LOW_STD_THRESHOLD:
                all_flagged.append(record)
            if physical_key in range_keys:
                range_metrics.append(record)
                bits = depth.reshape(-1).view(np.uint16)
                histogram += np.bincount(bits, minlength=1 << 16).astype(np.uint64)
                sampled_saturated += int(np.count_nonzero((depth == 0) | (depth == 1)))
                sampled_values += depth.size
            if physical_key in canary_keys:
                canary_metrics.append(record)
    database.close()

    range_flagged = [
        item for item in range_metrics if item["depth_std"] <= LOW_STD_THRESHOLD
    ]
    canary_flagged = [
        item for item in canary_metrics if item["depth_std"] <= LOW_STD_THRESHOLD
    ]

    by_trajectory: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in all_flagged:
        by_trajectory[str(item["trajectory"])].append(item)
    distinct_sample = []
    for trajectory_key in sorted(
        by_trajectory,
        key=lambda key: (_sha256_text(key), key),
    ):
        item = min(
            by_trajectory[trajectory_key],
            key=lambda row: (_sha256_text(str(row["logical_key"])), row["logical_key"]),
        )
        distinct_sample.append(dict(item))
        if len(distinct_sample) == args.sample_count:
            break
    if len(distinct_sample) < args.sample_count:
        raise RuntimeError(
            f"only {len(distinct_sample)} distinct flagged trajectories available"
        )

    decoded_by_trajectory: dict[str, np.ndarray] = {}
    depth_by_key: dict[str, np.ndarray] = {}
    database = lmdb.open(
        str(args.cache),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=8,
    )
    with database.begin(write=False) as transaction:
        for item in distinct_sample:
            trajectory_key = str(item["trajectory"])
            trajectory = trajectory_by_key[trajectory_key]
            if trajectory_key not in decoded_by_trajectory:
                decoded_by_trajectory[trajectory_key] = decode_trajectory(trajectory)
            frame = int(item["frame"])
            rgb = crop_rgb_for_world_model(decoded_by_trajectory[trajectory_key][frame])
            rgb_u8 = np.floor(np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
            colors, counts = np.unique(
                rgb_u8.reshape(-1, 3), axis=0, return_counts=True
            )
            item.update(
                {
                    "source_path": trajectory.source_relpath,
                    "source_frame": frame,
                    "rgb_mean": float(np.mean(rgb, dtype=np.float64)),
                    "rgb_std": float(np.std(rgb, dtype=np.float64)),
                    "rgb_channel_std": [
                        float(np.std(rgb[..., channel], dtype=np.float64))
                        for channel in range(3)
                    ],
                    "rgb_unique_uint8_colors": int(len(colors)),
                    "rgb_dominant_color_fraction": float(
                        np.max(counts) / rgb_u8.shape[0] / rgb_u8.shape[1]
                    ),
                }
            )
            payload = transaction.get(str(item["physical_key"]).encode("ascii"))
            if payload is None:
                raise RuntimeError(f"missing sampled key {item['physical_key']}")
            depth_by_key[str(item["physical_key"])] = decode_depth_value(
                payload, decompressor
            )
    database.close()

    tile_width, tile_height = 224, 468
    columns = 6
    rows = (len(distinct_sample) + columns - 1) // columns
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(montage)
    for index, item in enumerate(distinct_sample):
        trajectory_key = str(item["trajectory"])
        frame = int(item["frame"])
        rgb = crop_rgb_for_world_model(decoded_by_trajectory[trajectory_key][frame])
        rgb_u8 = np.floor(np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        depth = depth_by_key[str(item["physical_key"])]
        depth_u8 = np.floor(np.clip(depth, 0, 1) * 255 + 0.5).astype(np.uint8)
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        montage.paste(Image.fromarray(rgb_u8), (x, y))
        montage.paste(Image.fromarray(depth_u8, mode="L").convert("RGB"), (x, y + 224))
        draw.text(
            (x + 3, y + 450),
            f"{trajectory_key} f{frame:02d} std={item['depth_std']:.2g}",
            fill="black",
        )
    args.montage.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.montage)

    histogram_cumulative = np.cumsum(histogram, dtype=np.uint64)
    total_histogram = int(histogram_cumulative[-1])

    def histogram_quantile(quantile: float) -> float:
        rank = int(np.floor(quantile * (total_histogram - 1)))
        index = int(np.searchsorted(histogram_cumulative, rank + 1))
        return float(np.array([index], dtype=np.uint16).view("<f2")[0])

    flags_per_trajectory = Counter(str(item["trajectory"]) for item in all_flagged)
    flags_per_frame = Counter(int(item["frame"]) for item in all_flagged)
    flags_per_split = Counter(
        str(item["trajectory"]).split("/")[0] for item in all_flagged
    )
    flagged_episodes = [int(item["episode"]) for item in all_flagged]
    result = {
        "schema": "dinocular-wall-range-gate-diagnostic-v1",
        "read_only_cache": str(args.cache),
        "manifest": {
            "manifest_id": manifest["manifest_id"],
            "data_mdb_sha256": manifest["data_mdb_sha256"],
            "tool_sha256": manifest["tool_sha256"],
            "trajectory_count": manifest["trajectory_count"],
            "frame_count": manifest["frame_count"],
            "calibration": {
                key: value
                for key, value in manifest["calibration"].items()
                if key != "keys"
            },
        },
        "threshold": LOW_STD_THRESHOLD,
        "validator_range_sample": {
            "sampled_frames": len(range_metrics),
            "flagged_frames": len(range_flagged),
            "low_std_map_fraction": len(range_flagged) / len(range_metrics),
            "q10": histogram_quantile(0.10),
            "q90": histogram_quantile(0.90),
            "saturation_fraction": sampled_saturated / sampled_values,
            "flagged_distinct_trajectories": len(
                {str(item["trajectory"]) for item in range_flagged}
            ),
        },
        "full_cache_distribution": {
            "frames": len(expected),
            "flagged_frames": len(all_flagged),
            "low_std_map_fraction": len(all_flagged) / len(expected),
            "flagged_distinct_trajectories": len(flags_per_trajectory),
            "trajectory_fraction": len(flags_per_trajectory) / len(trajectories),
            "flags_per_trajectory_histogram": dict(
                sorted(Counter(flags_per_trajectory.values()).items())
            ),
            "maximum_flags_in_one_trajectory": max(flags_per_trajectory.values()),
            "split_counts": dict(sorted(flags_per_split.items())),
            "episode_decile_counts": _deciles(flagged_episodes, len(trajectories)),
            "frame_decile_counts": _deciles(
                [int(item["frame"]) for item in all_flagged], 50
            ),
            "top_trajectories": flags_per_trajectory.most_common(20),
            "frame_position_counts": dict(sorted(flags_per_frame.items())),
        },
        "rate_canary_reconstruction": {
            "qualification": (
                "The rate canary retained no validation receipt or depth maps. "
                "These are the exact five canary trajectories read from the immutable "
                "full cache under the corrected Wall calibration."
            ),
            "trajectories": list(RATE_CANARY_TRAJECTORIES),
            "frames": len(canary_metrics),
            "flagged_frames": len(canary_flagged),
            "low_std_map_fraction": len(canary_flagged) / len(canary_metrics),
            "per_trajectory": {
                key: {
                    "frames": sum(item["trajectory"] == key for item in canary_metrics),
                    "flagged_frames": sum(
                        item["trajectory"] == key for item in canary_flagged
                    ),
                }
                for key in RATE_CANARY_TRAJECTORIES
            },
        },
        "distinct_flagged_samples": distinct_sample,
        "montage": str(args.montage),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "json": str(args.json),
                "montage": str(args.montage),
                "validator_range_sample": result["validator_range_sample"],
                "full_cache_distribution": result["full_cache_distribution"],
                "rate_canary_reconstruction": result["rate_canary_reconstruction"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
