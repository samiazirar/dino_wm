#!/usr/bin/env python3
"""Localize Rope/Granular depth variation loss on fixed accepted episodes.

The command is read-only.  It compares released RGB and particle state, the
aligned PyFlex HDF5 renderer depth, immutable LMDB bytes, decoded wire arrays,
and the final affine depth presented at the DINOcular encoder boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import lmdb
import numpy as np
import torch
import zstandard

from tools.precompute_depth import decode_depth_value


EPISODES = (0, 1, 3, 4)
FRAMES = (0, 2, 5, 8, 10, 13, 16, 19)
ENVIRONMENTS = ("rope", "granular")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "sha256": sha256_bytes(array.tobytes()),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "finite": bool(np.isfinite(array).all()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def pair_record(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_state: np.ndarray,
    right_state: np.ndarray,
    left_renderer: np.ndarray,
    right_renderer: np.ndarray,
    left_cache: np.ndarray,
    right_cache: np.ndarray,
) -> dict[str, Any]:
    rgb_delta = np.abs(right_rgb.astype(np.float32) - left_rgb.astype(np.float32))
    moving = rgb_delta.mean(axis=-1) >= 5.0
    renderer_delta = np.abs(right_renderer.astype(np.float32) - left_renderer.astype(np.float32))
    cache_delta = np.abs(right_cache.astype(np.float32) - left_cache.astype(np.float32))

    def spatial(delta: np.ndarray) -> dict[str, Any]:
        moving_mean = float(delta[moving].mean()) if moving.any() else 0.0
        static_mean = float(delta[~moving].mean()) if (~moving).any() else 0.0
        return {
            "mean_absolute_delta": float(delta.mean()),
            "maximum_absolute_delta": float(delta.max()),
            "moving_rgb_region_mean_absolute_delta": moving_mean,
            "static_rgb_region_mean_absolute_delta": static_mean,
            "moving_to_static_delta_ratio": (
                moving_mean / static_mean if static_mean > 0.0 else None
            ),
        }

    return {
        "rgb_mean_absolute_delta_uint8": float(rgb_delta.mean()),
        "rgb_moving_pixel_fraction_at_5_uint8": float(moving.mean()),
        "particle_state_rms_delta": float(
            np.sqrt(np.mean((right_state.astype(np.float64) - left_state.astype(np.float64)) ** 2))
        ),
        "raw_renderer_depth_m": spatial(renderer_delta),
        "cache_final_depth": spatial(cache_delta),
    }


def diagnose_environment(root: Path, cache_root: Path, environment: str) -> dict[str, Any]:
    base = root / "deformable" / environment
    cache_dir = cache_root / f"{environment}.lmdb"
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lo = float(manifest["calibration"]["lo"])
    hi = float(manifest["calibration"]["hi"])
    states = torch.load(base / "states.pth", map_location="cpu", mmap=True, weights_only=False)
    database = lmdb.open(
        str(cache_dir),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=4,
    )
    decompressor = zstandard.ZstdDecompressor()
    episodes: list[dict[str, Any]] = []
    try:
        for episode in EPISODES:
            observations = torch.load(
                base / f"{episode:06d}" / "obses.pth",
                map_location="cpu",
                weights_only=False,
            ).numpy()
            samples: list[dict[str, Any]] = []
            arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            with database.begin(write=False) as transaction:
                for frame in FRAMES:
                    rgb = np.asarray(observations[frame])
                    state = np.asarray(states[episode, frame])
                    hdf5_path = base / f"{episode:06d}" / f"{frame:02d}.h5"
                    with h5py.File(hdf5_path, "r") as handle:
                        renderer_rgb = np.asarray(handle["observations/color/cam_1"][0])
                        renderer_state = np.asarray(handle["positions"][0])
                        renderer_depth_mm = np.asarray(handle["observations/depth/cam_1"][0])
                    if not np.array_equal(rgb, renderer_rgb):
                        raise RuntimeError(f"{environment}/{episode}/{frame}: RGB alignment failed")
                    if not np.array_equal(state, renderer_state):
                        raise RuntimeError(f"{environment}/{episode}/{frame}: state alignment failed")
                    physical_key = f"train/{episode:05d}/{frame:06d}"
                    payload = transaction.get(physical_key.encode("ascii"))
                    if payload is None:
                        raise RuntimeError(f"missing cache key {environment}/{physical_key}")
                    wire = decode_depth_value(payload, decompressor).astype(np.float32)
                    final = wire * (hi - lo) + lo
                    renderer = renderer_depth_mm.astype(np.float32) / 1000.0
                    samples.append(
                        {
                            "identity": f"{environment}/{physical_key}",
                            "rgb": array_record(rgb),
                            "particle_state": array_record(state),
                            "raw_renderer_depth_uint16_mm": array_record(renderer_depth_mm),
                            "cache_key": physical_key,
                            "serialized_payload_sha256": sha256_bytes(payload),
                            "serialized_payload_bytes": len(payload),
                            "decoded_wire": array_record(wire),
                            "final_affine_depth": array_record(final),
                            "wire_to_final_operation": f"final=wire*{hi - lo!r}+{lo!r}",
                        }
                    )
                    arrays.append((rgb, state, renderer, final))
            pairs = []
            for index in range(1, len(arrays)):
                left_rgb, left_state, left_renderer, left_cache = arrays[index - 1]
                right_rgb, right_state, right_renderer, right_cache = arrays[index]
                pairs.append(
                    {
                        "left_frame": FRAMES[index - 1],
                        "right_frame": FRAMES[index],
                        **pair_record(
                            left_rgb,
                            right_rgb,
                            left_state,
                            right_state,
                            left_renderer,
                            right_renderer,
                            left_cache,
                            right_cache,
                        ),
                    }
                )
            episodes.append(
                {
                    "episode": episode,
                    "samples": samples,
                    "adjacent_selected_frame_pairs": pairs,
                    "unique_hash_counts": {
                        "rgb": len({sample["rgb"]["sha256"] for sample in samples}),
                        "particle_state": len(
                            {sample["particle_state"]["sha256"] for sample in samples}
                        ),
                        "raw_renderer_depth": len(
                            {
                                sample["raw_renderer_depth_uint16_mm"]["sha256"]
                                for sample in samples
                            }
                        ),
                        "serialized_payload": len(
                            {sample["serialized_payload_sha256"] for sample in samples}
                        ),
                        "decoded_wire": len(
                            {sample["decoded_wire"]["sha256"] for sample in samples}
                        ),
                        "final_affine_depth": len(
                            {sample["final_affine_depth"]["sha256"] for sample in samples}
                        ),
                    },
                }
            )
    finally:
        database.close()

    pair_records = [
        pair_record
        for episode in episodes
        for pair_record in episode["adjacent_selected_frame_pairs"]
    ]

    def mean(path: tuple[str, ...]) -> float:
        values: list[float] = []
        for record in pair_records:
            value: Any = record
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        return float(np.mean(values)) if values else 0.0

    summary = {
        "fixed_episode_count": len(episodes),
        "fixed_frame_count": len(episodes) * len(FRAMES),
        "rgb_mean_absolute_delta_uint8": mean(("rgb_mean_absolute_delta_uint8",)),
        "particle_state_rms_delta": mean(("particle_state_rms_delta",)),
        "raw_renderer_depth_mean_absolute_delta_m": mean(
            ("raw_renderer_depth_m", "mean_absolute_delta")
        ),
        "cache_final_depth_mean_absolute_delta": mean(
            ("cache_final_depth", "mean_absolute_delta")
        ),
        "raw_renderer_moving_to_static_delta_ratio_mean": mean(
            ("raw_renderer_depth_m", "moving_to_static_delta_ratio")
        ),
        "cache_moving_to_static_delta_ratio_mean": mean(
            ("cache_final_depth", "moving_to_static_delta_ratio")
        ),
    }
    return {
        "environment": environment,
        "manifest_path": str(manifest_path),
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "producer": manifest["producer"],
        "calibration": manifest["calibration"],
        "selection": {
            "episodes": list(EPISODES),
            "frames": list(FRAMES),
            "selection_fixed_before_values": True,
        },
        "summary": summary,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "dinocular-rg-depth-defect-localization-v1",
        "read_only": True,
        "environments": [
            diagnose_environment(args.root, args.cache_root, environment)
            for environment in ENVIRONMENTS
        ],
        "stage_order": [
            "released_rgb_and_particle_state",
            "aligned_pyflex_hdf5_renderer_depth",
            "cache_key_selection",
            "serialized_lmdb_payload",
            "decoded_float16_wire",
            "final_affine_depth",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "sha256": sha256_bytes(args.output.read_bytes())}))


if __name__ == "__main__":
    main()
