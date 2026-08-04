#!/usr/bin/env python3
"""Materialize and validate the immutable Rope camera-1 HDF5 depth binding.

The released extractor owns the LMDB wire format and its source alignment
checks.  This task-level wrapper adds the native raw-metric contract and a
closed validation receipt, then reopens the cache through the training
reader, so the artifacts consumed by the seed-one run are the artifacts that
were directly checked here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from depth_contract import canonical_json_bytes, sha256_bytes, sha256_file
from datasets.depth_cache import DepthCacheReader
from tools import extract_native_depth as extractor


ENVIRONMENT = "rope"
CHECKPOINT_BACKEND = "df2_dino_rope_convs_de"
CHECKPOINT_FACTORY = "DFormerv2_S"
CHECKPOINT_KEY = "student"
CHECKPOINT_STATE_PREFIX = "module.backbone."


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise RuntimeError(f"missing immutable cache manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("cache manifest is not an object")
    return value


def validate_source_records(
    *, root: Path, manifest: dict[str, Any], release: Any
) -> dict[str, int]:
    """Recheck every HDF5 record, action transition, RGB, and state join."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - pinned Marvin image
        raise RuntimeError("h5py is required for direct HDF5 validation") from exc

    source = manifest.get("source_provenance")
    if not isinstance(source, dict):
        raise RuntimeError("cache manifest lacks native source provenance")
    episodes = source.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1000:
        raise RuntimeError("native source provenance does not contain 1000 episodes")

    trajectories = {
        int(item.episode): item for item in release.trajectories
    }
    counters = {
        "hdf5_frames": 0,
        "depth_frames": 0,
        "action_aligned_frames": 0,
        "rgb_aligned_model_frames": 0,
        "state_aligned_model_frames": 0,
    }
    for episode_record in episodes:
        episode = int(episode_record.get("episode", -1))
        trajectory = trajectories.get(episode)
        if trajectory is None:
            raise RuntimeError(f"source provenance has unknown episode {episode}")
        observation = extractor._load_observations(release, trajectory)
        hdf5_records = episode_record.get("hdf5")
        if not isinstance(hdf5_records, list) or len(hdf5_records) != 21:
            raise RuntimeError(f"episode {episode}: expected 21 HDF5 records")
        for frame, record in enumerate(hdf5_records):
            relative = record.get("path") if isinstance(record, dict) else None
            expected_relative = f"{episode:06d}/{frame:02d}.h5"
            require_equal(relative, expected_relative, f"episode {episode} frame {frame} path")
            path = release.base / relative
            actual_sha = sha256_file(path)
            require_equal(
                actual_sha,
                record.get("sha256"),
                f"episode {episode} frame {frame} HDF5 SHA-256",
            )
            with h5py.File(path, "r") as handle:
                extractor._require_hdf5_schema(
                    handle, path=path, particle_count=release.particle_count
                )
                depth = extractor.decode_uint16_millimetres(
                    np.asarray(handle[extractor.DEPTH_DATASET][0])
                )
                if not np.isfinite(depth).all():
                    raise RuntimeError(f"episode {episode} frame {frame}: nonfinite depth")
                action = np.asarray(handle["action"][()])
                expected_action = np.zeros_like(action) if frame == 0 else release.actions[
                    episode, min(frame - 1, 19)
                ].detach().cpu().numpy()
                if not np.array_equal(action, expected_action):
                    raise RuntimeError(
                        f"episode {episode} frame {frame}: action is not aligned to the released transition"
                    )
                counters["hdf5_frames"] += 1
                counters["depth_frames"] += 1
                counters["action_aligned_frames"] += 1
            if frame < 20:
                extractor._read_aligned_depth(
                    release, trajectory, observation, frame
                )
                counters["rgb_aligned_model_frames"] += 1
                counters["state_aligned_model_frames"] += 1
    return counters


def make_contract(
    *, manifest: dict[str, Any], checkpoint_sha256: str
) -> dict[str, Any]:
    producer = manifest.get("producer")
    wire = manifest.get("wire_format")
    calibration = manifest.get("calibration")
    if not isinstance(producer, dict) or not isinstance(wire, dict):
        raise RuntimeError("cache manifest lacks producer or wire format")
    if not isinstance(calibration, dict):
        raise RuntimeError("cache manifest lacks calibration")
    producer_sha256 = sha256_bytes(canonical_json_bytes(producer))
    wire_sha256 = sha256_bytes(canonical_json_bytes(wire))
    lo = float(calibration["lo"])
    hi = float(calibration["hi"])
    if not hi > lo:
        raise RuntimeError("Rope calibration range is not increasing")
    source_index_sha256 = str(manifest["native_source_index_sha256"])
    tool_sha256 = str(manifest["tool_sha256"])
    return {
        "schema": "dinocular-native-depth-contract-v1",
        "status": "complete",
        "self_acceptance_recorded": False,
        "execution_authority_granted": False,
        "checkpoint": {
            "name": "dinov2_depthembed_dropout_fullpr.pth",
            "sha256": checkpoint_sha256,
            "backend": CHECKPOINT_BACKEND,
            "factory": CHECKPOINT_FACTORY,
            "checkpoint_key": CHECKPOINT_KEY,
            "state_prefix": CHECKPOINT_STATE_PREFIX,
        },
        "producer": {
            "name": "released_pyflex_hdf5_camera_1",
            "model": "released_pyflex_frame_records",
            "version": "uint16_millimetre_camera_1",
            "code_commit": tool_sha256,
            "code_commit_kind": "extractor_sha256",
            "temporal_mode": "frame_records",
            "raw_units": "metric_meters",
            "weight_sha256": source_index_sha256,
            "weight_sha256_kind": "native_source_index_sha256_compatibility_field",
            "source_artifact_sha256": source_index_sha256,
            "invocation": {
                "route": "released_hdf5_frame_records",
                "camera": "cam_1",
                "depth_dataset": "observations/depth/cam_1",
            },
            "preprocessing": {
                "operations": [
                    "read_uint16_millimetres",
                    "decode_float32_metres",
                    "identity_224x224",
                ]
            },
            "scale": {
                "operation": "wire=clip((depth_m-lo)/(hi-lo),0,1)",
                "calibration_scope": "rope_training_only",
                "lo": lo,
                "hi": hi,
            },
            "clipping": {
                "space": "wire",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "encoder_input": {
            "checkpoint_native": {
                "quantity": "raw_metric_depth",
                "units": "metric_meters",
                "normalization": {"kind": "none_raw_metric"},
            },
            "zero_depth": {
                "payload_validation": "same_as_informative_depth",
                "intervention": "exact_numeric_zero_at_encoder_boundary",
                "learned_neutrality_claimed": False,
                "rgb_equivalence_claimed": False,
            },
            "cache_bindings": [
                {
                    "environment": ENVIRONMENT,
                    "producer_sha256": producer_sha256,
                    "wire_format_sha256": wire_sha256,
                    "wire_quantity": "clip_normalized_metric_depth_in_unit_interval",
                    "wire_range": [0.0, 1.0],
                    "affine_to_raw_metric": {
                        "operation": "raw_metric=wire*scale+offset",
                        "output_quantity": "raw_metric_depth",
                        "output_units": "metric_meters",
                        "scale": hi - lo,
                        "offset": lo,
                    },
                    "interpolation": "identity_224x224",
                    "payload_validation": {
                        "source": "reject_nonfinite_then_all_ones",
                        "valid_value": 1.0,
                    },
                }
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()

    cache_dir = args.cache_root.resolve() / f"{ENVIRONMENT}.lmdb"
    if cache_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing GT cache: {cache_dir}")
    manifest = extractor.build_native_cache(
        root=args.raw_root.resolve(),
        output_root=args.cache_root.resolve(),
        environment=ENVIRONMENT,
        contract=extractor.released_contract(ENVIRONMENT),
    )
    cache_dir = args.cache_root.resolve() / f"{ENVIRONMENT}.lmdb"
    manifest_path = cache_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    data_sha256 = sha256_file(cache_dir / "data.mdb")
    release = extractor.inspect_release(
        args.raw_root.resolve(), ENVIRONMENT, extractor.released_contract(ENVIRONMENT)
    )
    source_counts = validate_source_records(
        root=args.raw_root.resolve(), manifest=manifest, release=release
    )
    contract = make_contract(
        manifest=manifest, checkpoint_sha256=args.checkpoint_sha256
    )
    atomic_json(args.contract_path.resolve(), contract)
    contract_sha256 = sha256_file(args.contract_path.resolve())
    validation = {
        "schema": "dinocular-depth-cache-validation-v1",
        "state": "PASS",
        "validation_kind": "direct_full_rope_hdf5_camera_1_binding",
        "environment": ENVIRONMENT,
        "source": {
            "kind": "released_simulator_hdf5_frame_records",
            "camera": "cam_1",
            "depth_dataset": "observations/depth/cam_1",
            "source_root": str(release.base),
            "source_index_sha256": manifest["native_source_index_sha256"],
            "hdf5_inventory": "1000 episodes x 21 exact files",
        },
        "results": {
            ENVIRONMENT: {
                "state": "PASS",
                "manifest_id": manifest["manifest_id"],
                "manifest_sha256": manifest_sha256,
                "data_mdb_sha256": data_sha256,
                "producer_sha256": sha256_bytes(
                    canonical_json_bytes(manifest["producer"])
                ),
                "wire_format_sha256": sha256_bytes(
                    canonical_json_bytes(manifest["wire_format"])
                ),
                "trajectory_count": manifest["trajectory_count"],
                "frame_count": manifest["frame_count"],
                "train_episodes": sum(
                    record["trajectory_key"].startswith("train/")
                    for record in manifest["trajectories"]
                ),
                "valid_episodes": sum(
                    record["trajectory_key"].startswith("valid/")
                    for record in manifest["trajectories"]
                ),
                "model_observations": manifest["frame_count"],
                **source_counts,
                "contract_sha256": contract_sha256,
            }
        },
    }
    atomic_json(args.validation_path.resolve(), validation)
    validation_sha256 = sha256_file(args.validation_path.resolve())

    reader = DepthCacheReader(
        environment=ENVIRONMENT,
        source_root=args.raw_root.resolve(),
        cache_dir=cache_dir,
        cache_manifest_sha256=manifest_sha256,
        validation_path=args.validation_path.resolve(),
        validation_sha256=validation_sha256,
        native_contract_path=args.contract_path.resolve(),
        native_contract_sha256=contract_sha256,
        expected_producer_sha256=validation["results"][ENVIRONMENT]["producer_sha256"],
        expected_checkpoint_sha256=args.checkpoint_sha256,
    )
    episodes = [
        (record["trajectory_key"].split("/")[0], int(record["trajectory_key"].split("/")[1]), 20)
        for record in manifest["trajectories"]
    ]
    reader.assert_dataset_coverage(episodes)
    read_frames = 0
    for split, episode, _count in episodes:
        depth, validity = reader.read(
            split=split, episode=episode, frames=range(20)
        )
        if tuple(depth.shape) != (20, 224, 224) or not bool(np.isfinite(depth.numpy()).all()):
            raise RuntimeError(f"cache read validation failed for {split}/{episode:05d}")
        if not bool(np.allclose(validity.numpy(), 1.0)):
            raise RuntimeError(f"cache validity validation failed for {split}/{episode:05d}")
        read_frames += int(depth.shape[0])
    require_equal(read_frames, 20000, "cache read frame count")
    validation["results"][ENVIRONMENT].update(
        {
            "reader_contract_reopen": "PASS",
            "reader_exact_coverage": "PASS",
            "reader_finite_wire_values": read_frames,
            "reader_validity_one_values": read_frames,
        }
    )
    atomic_json(args.validation_path.resolve(), validation)
    validation_sha256 = sha256_file(args.validation_path.resolve())
    print(
        json.dumps(
            {
                "state": "PASS",
                "environment": ENVIRONMENT,
                "cache_dir": str(cache_dir),
                "cache_manifest_sha256": manifest_sha256,
                "cache_manifest_id": manifest["manifest_id"],
                "data_mdb_sha256": data_sha256,
                "producer_sha256": validation["results"][ENVIRONMENT]["producer_sha256"],
                "native_contract_path": str(args.contract_path.resolve()),
                "native_contract_sha256": contract_sha256,
                "validation_path": str(args.validation_path.resolve()),
                "validation_sha256": validation_sha256,
                "model_observations": read_frames,
                "source_hdf5_frames": source_counts["hdf5_frames"],
                "action_aligned_frames": source_counts["action_aligned_frames"],
                "rgb_aligned_model_frames": source_counts["rgb_aligned_model_frames"],
                "state_aligned_model_frames": source_counts["state_aligned_model_frames"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
