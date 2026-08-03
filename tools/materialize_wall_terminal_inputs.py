#!/usr/bin/env python3
"""Materialize and directly check the fixed Wall terminal observation inputs.

The accepted Wall cache intentionally contains frames 0..49.  This command
does not alter that cache.  It binds the fixed 192-route manifest to the
released 51-frame RGB PTH files, invokes the exact producer contract used for
the accepted cache on each complete 51-frame trajectory, selects only producer
output frame 50, and writes a small immutable terminal-input bundle.

The producer module is imported from the source checkout supplied at runtime.
The command therefore fails closed when a later producer changes the admitted
identity or settings (for example by adding a new deterministic-execution
setting to the provenance).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.precompute_depth import (  # noqa: E402
    ARTIFACTS,
    CHUNK_SIZE,
    ContractError,
    OfficialDA3StreamingProducer,
    OUTPUT_SHAPE,
    OVERLAP,
    PROCESS_RES,
    Trajectory,
    WIRE_DTYPE,
    _jsonable,
    _torch_load,
    canonical_json_bytes,
    crop_producer_result,
    decode_trajectory,
    normalize_depth,
    producer_identity,
    sha256_bytes,
    sha256_file,
    streaming_chunks,
)


SCHEMA = "dinocular.wall-terminal-input-bundle.v1"
FIXED_MANIFEST_SCHEMA = "dino-wm-open-loop-manifest-v1"
EXPECTED_ROUTE_COUNT = 192
EXPECTED_OBSERVATION_COUNT = 51
EXPECTED_ACTION_COUNT = 50
EXPECTED_HORIZONS = [1, 5, 10]
EXPECTED_FRAMESKIP = 5
EXPECTED_NUM_HIST = 1
EXPECTED_SELECTION_SEED = 20260714
EXPECTED_PRODUCER_SOURCE_COMMIT = (
    "a820029d6ba2f1b579efdfcd0a5e3fdf30fdab16"
)
EXPECTED_CACHE_MANIFEST_SHA256 = (
    "b6db2c8b495ed2a3a142973cdc3622ab193a5a38d51147515d2f38678214819d"
)
EXPECTED_CACHE_DATA_SHA256 = (
    "e4107dc52f50dc206df6bc5c5e8b1655c3b699fa58d6d3196d5329d63c0dc467"
)
EXPECTED_CACHE_MANIFEST_ID = "54bab01a-8c49-4df5-a1d9-3a76e32e5234"
EXPECTED_VALIDATION_EVIDENCE_SHA256 = (
    "22771f9c2d032115373c62010e594853fb33fd5c62448d35df9f34571e3234a0"
)
EXPECTED_VALIDATION_RECEIPT_SHA256 = (
    "b2404b80ba54f7b9b9cd857c4032bf515a91d1a3295f62167c7c9c65f122fab4"
)

EXPECTED_PRODUCER_SETTINGS = {
    "chunk_size": 120,
    "loop_closure": True,
    "overlap": 60,
    "overlap_policy": "discard_duplicated_tail_no_blend",
    "precision": "bfloat16",
    "process_res": 504,
    "process_res_method": "upper_bound_resize",
    "pth_float32_rgb_quantization": "round_half_up_to_uint8_for_png",
    "ref_view_strategy": "saddle_balanced",
    "wall_terminal_observation_policy": (
        "drop_post_action_frame_not_selected_by_WallDataset"
    ),
}
EXPECTED_WIRE_FORMAT = {
    "compressor": "zstd",
    "compressor_level": 3,
    "dtype": "<f2",
    "inverted": False,
    "map_size": 1 << 40,
    "normalization": "clip((depth_m-lo)/(hi-lo),0,1)",
    "order": "C",
    "physical_key": "<split>/<episode:05d>/<frame:06d>",
    "shape": [224, 224],
}
EXPECTED_AUXILIARY = {
    "actions": {
        "relative_path": "wall_single/actions.pth",
        "sha256": "c9620d6d5029f50b452d31fd04add070591379269903f67f7a4a38c24f718b07",
        "shape": [1920, 50, 2],
    },
    "states": {
        "relative_path": "wall_single/states.pth",
        "sha256": "4fd7e0b44dcd37e5adcc32a7f8f310cd0d9bc224cad1c150fe146240d0d6f004",
        "shape": [1920, 50, 2],
    },
    "door_locations": {
        "relative_path": "wall_single/door_locations.pth",
        "sha256": "1d6920542c0a986a68677145a91cc837b8de98bec7f30cd7e95be5a75f4da354",
        "shape": [1920, 50, 1],
    },
    "wall_locations": {
        "relative_path": "wall_single/wall_locations.pth",
        "sha256": "cc6a714d4599b067d4178e417b130f8ce7fcefbb75c90dbea1b2003268ac8473",
        "shape": [1920, 50, 1],
    },
}
EXPECTED_PLANNING_SOURCE_FILES = {
    "states.pth": EXPECTED_AUXILIARY["states"]["sha256"],
    "actions.pth": EXPECTED_AUXILIARY["actions"]["sha256"],
    "door_locations.pth": EXPECTED_AUXILIARY["door_locations"]["sha256"],
    "wall_locations.pth": EXPECTED_AUXILIARY["wall_locations"]["sha256"],
}
EXPECTED_PLANNING_PROTOCOL = {
    "goal_H": 5,
    "goal_distance_model_steps": 5,
    "goal_source": "random_state",
    "frameskip": 5,
    "split_fraction": 0.9,
    "split_seed": 42,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fail(message: str) -> None:
    raise ContractError(message)


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSON at {path}:{line_number}: {exc}")
            if not isinstance(row, dict):
                fail(f"manifest row {line_number} is not an object")
            rows.append(row)
    return rows


def validate_fixed_manifest(
    manifest_path: Path, meta_path: Path
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    meta = json.loads(meta_path.read_text())
    if meta.get("schema") != FIXED_MANIFEST_SCHEMA:
        fail(f"unexpected fixed manifest schema {meta.get('schema')!r}")
    if sha256_file(manifest_path) != meta.get("manifest_sha256"):
        fail("fixed manifest file hash does not match its metadata")
    if meta.get("environment") != "wall":
        fail("fixed manifest is not Wall")
    for name, expected in (
        ("eligible_count", EXPECTED_ROUTE_COUNT),
        ("selected_count", EXPECTED_ROUTE_COUNT),
        ("observation_count", EXPECTED_OBSERVATION_COUNT),
        ("action_count", EXPECTED_ACTION_COUNT),
        ("frameskip", EXPECTED_FRAMESKIP),
        ("num_hist", EXPECTED_NUM_HIST),
        ("selection_seed", EXPECTED_SELECTION_SEED),
    ):
        if meta.get(name) != expected:
            fail(f"fixed manifest metadata {name}={meta.get(name)!r}, expected {expected!r}")
    if meta.get("horizons") != EXPECTED_HORIZONS:
        fail(f"fixed manifest horizons changed: {meta.get('horizons')!r}")
    if len(rows) != EXPECTED_ROUTE_COUNT:
        fail(f"fixed manifest has {len(rows)} rows, expected {EXPECTED_ROUTE_COUNT}")
    keys = []
    expected_raw_actions = {
        str(index): list(range(index * EXPECTED_FRAMESKIP, (index + 1) * EXPECTED_FRAMESKIP))
        for index in range(10)
    }
    for row in rows:
        split = row.get("split")
        episode = row.get("episode")
        if row.get("schema") != "dino-wm-open-loop-start-v1":
            fail(f"unexpected route schema for {row.get('key')!r}")
        if row.get("environment") != "wall" or split != "valid":
            fail(f"route is outside fixed Wall valid split: {row.get('key')!r}")
        if not isinstance(episode, int) or not 0 <= episode < 1920:
            fail(f"invalid route episode {episode!r}")
        expected_key = f"wall/{split}/{episode:05d}/000000"
        if row.get("key") != expected_key or row.get("start") != 0:
            fail(f"route key/start mismatch for episode {episode}")
        if row.get("history_frames") != [0]:
            fail(f"route history changed for {expected_key}")
        if row.get("horizons") != EXPECTED_HORIZONS or row.get("frameskip") != EXPECTED_FRAMESKIP:
            fail(f"route protocol changed for {expected_key}")
        if row.get("target_frames") != {"1": 5, "5": 25, "10": 50}:
            fail(f"route target frames changed for {expected_key}")
        if row.get("raw_action_indices") != expected_raw_actions:
            fail(f"route raw action coverage changed for {expected_key}")
        source_path = row.get("source_path")
        if not isinstance(source_path, str) or Path(source_path).is_absolute():
            fail(f"invalid source path for {expected_key}")
        source_sha256 = row.get("source_sha256")
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            fail(f"invalid source hash for {expected_key}")
        keys.append(row["key"])
    if len(set(keys)) != EXPECTED_ROUTE_COUNT:
        fail("fixed manifest contains duplicate route keys")
    expected_keys_sha = sha256_bytes(canonical_json_bytes(keys))
    if meta.get("keys_sha256") != expected_keys_sha:
        fail("fixed manifest key-list hash mismatch")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "meta_path": str(meta_path),
        "meta_sha256": sha256_file(meta_path),
        "meta": meta,
        "rows": rows,
        "keys": keys,
    }


def validate_cache_contract(
    cache_manifest_path: Path, validation_evidence_path: Path
) -> dict[str, Any]:
    cache_sha = sha256_file(cache_manifest_path)
    if cache_sha != EXPECTED_CACHE_MANIFEST_SHA256:
        fail(f"accepted Wall cache manifest hash changed: {cache_sha}")
    cache = json.loads(cache_manifest_path.read_text())
    data_path = cache_manifest_path.parent / "data.mdb"
    data_sha = sha256_file(data_path)
    if data_sha != EXPECTED_CACHE_DATA_SHA256:
        fail(f"accepted Wall cache data hash changed: {data_sha}")
    evidence_sha = sha256_file(validation_evidence_path)
    if evidence_sha != EXPECTED_VALIDATION_EVIDENCE_SHA256:
        fail(f"accepted Wall validation evidence hash changed: {evidence_sha}")
    evidence = json.loads(validation_evidence_path.read_text())
    expected_evidence = {
        "schema": "dinocular-da3-wall-recovery-evidence-v1",
        "manifest_id": EXPECTED_CACHE_MANIFEST_ID,
        "manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "data_mdb_sha256": EXPECTED_CACHE_DATA_SHA256,
        "source_commit": EXPECTED_PRODUCER_SOURCE_COMMIT,
        "validation_receipt_sha256": EXPECTED_VALIDATION_RECEIPT_SHA256,
    }
    for key, expected in expected_evidence.items():
        if evidence.get(key) != expected:
            fail(f"accepted Wall validation evidence {key} is not pinned")
    if cache.get("schema") != "dinocular-depth-cache-v1":
        fail("unexpected Wall depth-cache schema")
    if cache.get("manifest_id") != EXPECTED_CACHE_MANIFEST_ID:
        fail("accepted Wall cache manifest id changed")
    if cache.get("environment") != "wall" or cache.get("trajectory_count") != 1920:
        fail("accepted Wall cache trajectory contract changed")
    if cache.get("frame_count") != 96000:
        fail("accepted Wall cache frame count changed")
    if cache.get("producer", {}).get("settings") != EXPECTED_PRODUCER_SETTINGS:
        fail("accepted Wall producer settings changed")
    producer = cache["producer"]
    expected_identity_fields = {
        "name": "DA3NESTED-GIANT-LARGE-1.1/DA3-Streaming",
        "repository": "https://github.com/ByteDance-Seed/Depth-Anything-3.git",
        "commit": "e74fd796e96b7e781a5506fd8503b6bd7232513c",
        "salad_commit": "6aede13a3f6c25750bf7fde10209c06cb73060bb",
        "model_revision": "b2359bdf726fb44ef62acca04d629dcf158053e7",
        "effective_config_sha256": "547bb79de9c42bc4b81b92e5fb9cc081c2f3ed2ebcd54c2897c8b89c8c20f9c8",
    }
    for key, expected in expected_identity_fields.items():
        if producer.get(key) != expected:
            fail(f"accepted Wall producer identity {key} changed")
    for artifact_name, artifact_contract in ARTIFACTS.items():
        artifact = producer.get("artifacts", {}).get(artifact_name, {})
        if artifact.get("bytes") != artifact_contract["bytes"] or artifact.get("sha256") != artifact_contract["sha256"]:
            fail(f"accepted Wall artifact contract changed for {artifact_name}")
    if cache.get("wire_format") != EXPECTED_WIRE_FORMAT:
        fail("accepted Wall cache wire format changed")
    calibration = cache.get("calibration", {})
    if calibration.get("lo") != 0.6311277413368225 or calibration.get("hi") != 1.145139434337616:
        fail("accepted Wall calibration changed")
    trajectories = cache.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 1920:
        fail("accepted Wall cache does not contain all 1920 trajectories")
    trajectory_by_key = {entry.get("trajectory_key"): entry for entry in trajectories}
    if len(trajectory_by_key) != 1920:
        fail("accepted Wall cache trajectory keys are not unique")
    return {
        "path": str(cache_manifest_path),
        "sha256": cache_sha,
        "data_path": str(data_path),
        "data_mdb_sha256": data_sha,
        "validation_evidence_path": str(validation_evidence_path),
        "validation_evidence_sha256": evidence_sha,
        "validation_evidence": evidence,
        "manifest": cache,
        "trajectory_by_key": trajectory_by_key,
    }


def validate_auxiliary_sources(root: Path) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    sources: dict[str, Any] = {}
    for name, expected in EXPECTED_AUXILIARY.items():
        path = root / expected["relative_path"]
        if not path.is_file():
            fail(f"missing original Wall auxiliary source {path}")
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            fail(f"Wall auxiliary source changed for {name}: {digest}")
        array = as_numpy(_torch_load(path))
        if list(array.shape) != expected["shape"]:
            fail(f"Wall auxiliary shape changed for {name}: {array.shape}")
        if not np.issubdtype(array.dtype, np.floating):
            fail(f"Wall auxiliary dtype is not numeric for {name}: {array.dtype}")
        arrays[name] = np.ascontiguousarray(array)
        sources[name] = {
            "relative_path": expected["relative_path"],
            "path": str(path),
            "sha256": digest,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "available_episode_indices": [0, int(array.shape[0]) - 1],
            "available_observation_indices": [0, int(array.shape[1]) - 1],
            "terminal_observation_index_50": "absent",
        }
    return {
        "sources": sources,
        "arrays": arrays,
        "terminal_observation_index": 50,
        "terminal_auxiliary_source_status": "no_recorded_state_or_location_at_index_50",
        "fixed_prediction_scored_target": "visual_embedding_only",
        "required_terminal_inputs": [
            "original_rgb_frame_50",
            "admitted_depth_frame_50",
            "derived_all_finite_depth_validity_mask",
        ],
        "not_required_by_fixed_scored_target": [
            "proprio",
            "state",
            "door_location",
            "wall_location",
        ],
        "proprio_source_rule": (
            "no frame-49 clamp or fabricated frame-50 value; the released state and "
            "location tensors end at observation index 49"
        ),
    }


def validate_planning_targets(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing fixed Wall planning target manifest {path}")
    digest = sha256_file(path)
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        fail("fixed Wall planning target manifest is not a dictionary")
    if value.get("schema") != "dinocular.fixed-planning-targets.v1":
        fail("fixed Wall planning target schema changed")
    if value.get("environment") != "wall" or value.get("selection_seed") != EXPECTED_SELECTION_SEED:
        fail("fixed Wall planning target selection changed")
    if value.get("count") != 50 or value.get("goal_H") != 5:
        fail("fixed Wall planning target count or horizon changed")
    if value.get("protocol") != EXPECTED_PLANNING_PROTOCOL:
        fail("fixed Wall planning target protocol changed")
    if len(value.get("target_ids", [])) != 50 or len(value.get("target_records", [])) != 50:
        fail("fixed Wall planning target records changed")
    for name, array in (("state_0", value.get("state_0")), ("state_g", value.get("state_g"))):
        array = np.asarray(array)
        if array.shape != (50, 2) or not np.isfinite(array).all():
            fail(f"fixed Wall planning target {name} changed")
    source_files = value.get("source_files", [])
    observed_source_hashes = {
        Path(item.get("path", "")).name: item.get("sha256")
        for item in source_files
        if isinstance(item, dict)
    }
    if observed_source_hashes != EXPECTED_PLANNING_SOURCE_FILES:
        fail("fixed Wall planning target source hashes changed")
    return {
        "path": str(path),
        "sha256": digest,
        "schema": value["schema"],
        "environment": value["environment"],
        "selection_seed": value["selection_seed"],
        "count": value["count"],
        "goal_H": value["goal_H"],
        "protocol": value["protocol"],
        "target_ids_sha256": sha256_bytes(canonical_json_bytes(value["target_ids"])),
        "source_files": source_files,
    }


def validate_contract(
    *,
    root: Path,
    fixed_manifest_path: Path,
    fixed_meta_path: Path,
    cache_manifest_path: Path,
    validation_evidence_path: Path,
    planning_targets_path: Path,
) -> dict[str, Any]:
    fixed = validate_fixed_manifest(fixed_manifest_path, fixed_meta_path)
    cache = validate_cache_contract(cache_manifest_path, validation_evidence_path)
    aux = validate_auxiliary_sources(root)
    planning = validate_planning_targets(planning_targets_path)
    return {"fixed": fixed, "cache": cache, "auxiliary": aux, "planning": planning}


def route_trajectory(row: Mapping[str, Any], root: Path) -> Trajectory:
    source_path = root / str(row["source_path"])
    return Trajectory(
        environment="wall",
        split=str(row["split"]),
        episode=int(row["episode"]),
        frame_count=EXPECTED_OBSERVATION_COUNT,
        source_path=source_path,
        source_relpath=str(row["source_path"]),
        source_kind="wall_pth",
        source_sha256=str(row["source_sha256"]),
    )


def source_frame_audit(
    row: Mapping[str, Any], root: Path
) -> tuple[Trajectory, np.ndarray, np.ndarray, dict[str, Any]]:
    trajectory = route_trajectory(row, root)
    if not trajectory.source_path.is_file():
        fail(f"missing route RGB source {trajectory.source_path}")
    source_file_sha = sha256_file(trajectory.source_path)
    if source_file_sha != trajectory.source_sha256:
        fail(f"route RGB source hash changed for {trajectory.trajectory_key}")
    raw = as_numpy(_torch_load(trajectory.source_path))
    if raw.dtype != np.float32 or raw.shape != (51, 3, 224, 224):
        fail(
            f"route RGB source must be float32 [51,3,224,224], got "
            f"{raw.dtype} {raw.shape} for {trajectory.trajectory_key}"
        )
    if not np.isfinite(raw).all() or float(raw.min()) < 0 or float(raw.max()) > 255:
        fail(f"route RGB source has invalid values for {trajectory.trajectory_key}")
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    raw49 = np.ascontiguousarray(raw[49], dtype=np.float32)
    raw50 = np.ascontiguousarray(raw[50], dtype=np.float32)
    raw49_hash = sha256_bytes(raw49.tobytes(order="C"))
    raw50_hash = sha256_bytes(raw50.tobytes(order="C"))
    if raw49_hash == raw50_hash or np.array_equal(raw49, raw50):
        fail(f"route RGB frame 50 duplicates frame 49 for {trajectory.trajectory_key}")
    decoded = decode_trajectory(trajectory)
    if decoded.shape != (51, 224, 224, 3) or decoded.dtype != np.uint8:
        fail(f"decoded route RGB geometry changed for {trajectory.trajectory_key}")
    expected_uint8 = np.floor(np.transpose(raw50, (1, 2, 0)) + 0.5).astype(np.uint8)
    expected_uint8 = np.ascontiguousarray(expected_uint8)
    if not np.array_equal(decoded[50], expected_uint8):
        fail(f"producer RGB quantization does not use source frame 50 for {trajectory.trajectory_key}")
    audit = {
        "source_file_sha256": source_file_sha,
        "source_frame_count": int(raw.shape[0]),
        "source_frame_index": 50,
        "source_frame_49_f32_chw_sha256": raw49_hash,
        "source_frame_50_f32_chw_sha256": raw50_hash,
        "producer_input_frame_count": int(decoded.shape[0]),
        "producer_input_frame_indices": [0, 50],
        "producer_input_frame_50_uint8_hwc_sha256": sha256_bytes(
            decoded[50].tobytes(order="C")
        ),
    }
    return trajectory, raw50, decoded, audit


def episode_auxiliary_audit(
    episode: int, auxiliary: Mapping[str, Any]
) -> dict[str, Any]:
    arrays = auxiliary["arrays"]
    result: dict[str, Any] = {}
    for name, array in arrays.items():
        episode_array = np.ascontiguousarray(array[episode])
        result[name] = {
            "episode_index": episode,
            "available_observation_indices": [0, int(episode_array.shape[0]) - 1],
            "terminal_observation_index_50": "absent",
            "episode_values_sha256": sha256_bytes(episode_array.tobytes(order="C")),
        }
    return result


def expected_cache_route(
    row: Mapping[str, Any], cache: Mapping[str, Any]
) -> Mapping[str, Any]:
    trajectory_key = f"{row['split']}/{int(row['episode']):05d}"
    entry = cache["trajectory_by_key"].get(trajectory_key)
    if entry is None:
        fail(f"selected route has no accepted cache trajectory {trajectory_key}")
    expected_outputs = [f"{trajectory_key}/{frame:06d}" for frame in range(50)]
    if entry.get("source_path") != row.get("source_path"):
        fail(f"accepted cache source differs for {trajectory_key}")
    if entry.get("source_video_sha256") != row.get("source_sha256"):
        fail(f"accepted cache source hash differs for {trajectory_key}")
    if entry.get("ordered_frame_count") != 50 or entry.get("ordered_output_keys") != expected_outputs:
        fail(f"accepted cache has incomplete or misaligned frames for {trajectory_key}")
    if entry.get("chunk_boundaries") != streaming_chunks(50):
        fail(f"accepted cache chunk contract changed for {trajectory_key}")
    producer_metadata = entry.get("producer_metadata", {})
    if producer_metadata.get("input_frame_count") != 50 or not producer_metadata.get("trajectory_only_call"):
        fail(f"accepted cache producer input contract changed for {trajectory_key}")
    return entry


def terminal_route_record(
    *,
    row: Mapping[str, Any],
    trajectory: Trajectory,
    raw50: np.ndarray,
    decoded: np.ndarray,
    source_audit: Mapping[str, Any],
    result: Any,
    cropped: np.ndarray,
    resize_records: Sequence[Mapping[str, Any]],
    depth16: np.ndarray,
    auxiliary: Mapping[str, Any],
    cache_entry: Mapping[str, Any],
) -> dict[str, Any]:
    raw_depth = np.ascontiguousarray(np.squeeze(np.asarray(result.depth_m[50], dtype=np.float32)))
    if raw_depth.shape != (PROCESS_RES, PROCESS_RES):
        fail(f"terminal producer depth shape changed for {trajectory.trajectory_key}")
    if cropped.shape != (EXPECTED_OBSERVATION_COUNT, *OUTPUT_SHAPE):
        fail(f"terminal cropped depth shape changed for {trajectory.trajectory_key}")
    metadata = _jsonable(result.metadata)
    if metadata.get("official_class") != "da3_streaming.DA3_Streaming":
        fail(f"terminal producer class changed for {trajectory.trajectory_key}")
    if metadata.get("trajectory_only_call") is not True or metadata.get("input_frame_count") != 51:
        fail(f"terminal producer did not process one complete 51-frame trajectory for {trajectory.trajectory_key}")
    if _jsonable(result.chunk_boundaries) != streaming_chunks(51):
        fail(f"terminal producer chunk boundaries changed for {trajectory.trajectory_key}")
    if len(result.depth_m) != 51:
        fail(f"terminal producer omitted an output frame for {trajectory.trajectory_key}")
    if len(resize_records) != 51:
        fail(f"terminal resize records omitted an output frame for {trajectory.trajectory_key}")
    if resize_records[50] != {
        "decoded_hw": [224, 224],
        "producer_depth_hw": [504, 504],
        "producer_to_decoded_scale_yx": [0.4444444444444444, 0.4444444444444444],
        "world_model_resize": "short_edge_224_bilinear_antialias_true",
        "world_model_crop": "center_224",
    }:
        fail(f"terminal crop/resize settings changed for {trajectory.trajectory_key}")
    return {
        "key": row["key"],
        "environment": "wall",
        "split": row["split"],
        "episode": int(row["episode"]),
        "source_path": row["source_path"],
        "source_sha256": row["source_sha256"],
        **dict(source_audit),
        "terminal_output_selection": {
            "producer_output_frame_index": 50,
            "source_rgb_frame_index": 50,
            "source_frame_count": 51,
            "producer_input_frame_count": 51,
            "producer_output_frame_indices": [0, 50],
            "selection": "direct_frame_50_from_complete_original_sequence",
            "copy_interpolate_pad_wrap_duplicate_or_other_route": False,
        },
        "producer_depth_504_f32_hw_sha256": sha256_bytes(raw_depth.tobytes(order="C")),
        "terminal_depth_f16_hw_sha256": sha256_bytes(depth16.tobytes(order="C")),
        "producer_metadata": metadata,
        "chunk_boundaries": _jsonable(result.chunk_boundaries),
        "alignments": _jsonable(result.alignments),
        "terminal_resize_record": _jsonable(resize_records[50]),
        "accepted_cache_route": {
            "trajectory_key": cache_entry["trajectory_key"],
            "ordered_frame_count": cache_entry["ordered_frame_count"],
            "ordered_output_first": cache_entry["ordered_output_keys"][0],
            "ordered_output_last": cache_entry["ordered_output_keys"][-1],
            "accepted_producer_metadata_input_frame_count": cache_entry[
                "producer_metadata"
            ]["input_frame_count"],
        },
        "auxiliary_episode_sources": episode_auxiliary_audit(
            int(row["episode"]), auxiliary
        ),
        "raw_action_indices": row["raw_action_indices"],
        "target_frames": row["target_frames"],
    }


def array_descriptor(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": np.dtype(array.dtype).str,
        "order": "C",
    }


def write_bundle(
    *,
    out: Path,
    contract: Mapping[str, Any],
    root: Path,
    model_dir: Path,
    da3_root: Path,
    work_root: Path | None,
    producer_source_commit: str,
    materializer_commit: str,
) -> None:
    if out.exists():
        fail(f"refusing to overwrite existing terminal-input bundle {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    building = out.parent / f".{out.name}.building-{uuid.uuid4().hex}"
    building.mkdir(parents=True)
    try:
        cache_manifest = contract["cache"]["manifest"]
        producer = OfficialDA3StreamingProducer(
            da3_root,
            model_dir,
            work_root=(work_root or (building / ".da3-work")),
        )
        if producer_source_commit != EXPECTED_PRODUCER_SOURCE_COMMIT:
            fail("terminal producer source commit is not the accepted Wall producer commit")
        if producer_identity(producer.provenance) != producer_identity(cache_manifest["producer"]):
            fail("live terminal producer identity/settings do not exactly match accepted Wall cache")

        route_count = len(contract["fixed"]["rows"])
        rgb_f32 = np.empty((route_count, 3, 224, 224), dtype=np.float32)
        rgb_u8 = np.empty((route_count, 224, 224, 3), dtype=np.uint8)
        depth_f16 = np.empty((route_count, *OUTPUT_SHAPE), dtype=WIRE_DTYPE)
        depth_validity = np.ones((route_count, *OUTPUT_SHAPE), dtype=np.uint8)
        route_records: list[dict[str, Any]] = []
        lo = float(cache_manifest["calibration"]["lo"])
        hi = float(cache_manifest["calibration"]["hi"])
        for index, row in enumerate(contract["fixed"]["rows"]):
            trajectory, raw50, decoded, source_audit = source_frame_audit(row, root)
            cache_entry = expected_cache_route(row, contract["cache"])
            result = producer.infer_trajectory(decoded, trajectory)
            cropped, resize_records = crop_producer_result(result, decoded)
            normalized = normalize_depth(cropped[50], lo, hi)
            if normalized.shape != OUTPUT_SHAPE or not np.isfinite(normalized).all():
                fail(f"terminal normalized depth is invalid for {trajectory.trajectory_key}")
            if float(normalized.min()) < 0 or float(normalized.max()) > 1:
                fail(f"terminal normalized depth left [0,1] for {trajectory.trajectory_key}")
            rgb_f32[index] = raw50
            rgb_u8[index] = decoded[50]
            depth_f16[index] = np.asarray(normalized, dtype=WIRE_DTYPE)
            route_records.append(
                terminal_route_record(
                    row=row,
                    trajectory=trajectory,
                    raw50=raw50,
                    decoded=decoded,
                    source_audit=source_audit,
                    result=result,
                    cropped=cropped,
                    resize_records=resize_records,
                    depth16=depth_f16[index],
                    auxiliary=contract["auxiliary"],
                    cache_entry=cache_entry,
                )
            )

        rgb_f32_path = building / "terminal_rgb_f32_chw.npy"
        rgb_u8_path = building / "terminal_rgb_uint8_hwc.npy"
        depth_path = building / "terminal_depth_normalized_f16_hw.npy"
        validity_path = building / "terminal_depth_validity_u8_hw.npy"
        atomic_write_npy(rgb_f32_path, rgb_f32)
        atomic_write_npy(rgb_u8_path, rgb_u8)
        atomic_write_npy(depth_path, depth_f16)
        atomic_write_npy(validity_path, depth_validity)
        receipt = {
            "schema": SCHEMA,
            "state": "PASS",
            "created_utc": utc_now(),
            "environment": "wall",
            "route_count": route_count,
            "terminal_observation_index": 50,
            "fixed_protocol": {
                "observation_count": 51,
                "action_count": 50,
                "horizons": EXPECTED_HORIZONS,
                "frameskip": 5,
                "num_hist": 1,
                "held_out_split": "valid",
                "selected_prediction_episodes": route_count,
                "planning_target_protocol_unchanged": True,
            },
            "fixed_manifest": {
                "path": contract["fixed"]["path"],
                "sha256": contract["fixed"]["sha256"],
                "meta_path": contract["fixed"]["meta_path"],
                "meta_sha256": contract["fixed"]["meta_sha256"],
                "keys_sha256": contract["fixed"]["meta"]["keys_sha256"],
            },
            "accepted_depth_cache": {
                "manifest_path": contract["cache"]["path"],
                "manifest_sha256": contract["cache"]["sha256"],
                "data_mdb_path": contract["cache"]["data_path"],
                "data_mdb_sha256": contract["cache"]["data_mdb_sha256"],
                "manifest_id": cache_manifest["manifest_id"],
                "existing_frame_contract": "0..49_only",
            },
            "validation_evidence": {
                "path": contract["cache"]["validation_evidence_path"],
                "sha256": contract["cache"]["validation_evidence_sha256"],
                "source_commit": producer_source_commit,
                "validation_receipt_sha256": EXPECTED_VALIDATION_RECEIPT_SHA256,
            },
            "producer": _jsonable(producer.provenance),
            "producer_source_commit": producer_source_commit,
            "materializer": {
                "path": str(Path(__file__).resolve()),
                "source_sha256": sha256_file(Path(__file__).resolve()),
                "git_commit": materializer_commit,
                "invocation": "one OfficialDA3StreamingProducer.infer_trajectory call per complete 51-frame route",
                "chunk_schedule": streaming_chunks(51),
                "selection": "producer depth output frame 50 only",
                "prohibited_operations": [
                    "copy",
                    "interpolate",
                    "pad",
                    "wrap",
                    "duplicate",
                    "substitute_other_route",
                    "frame_49_clamp",
                ],
            },
            "preprocessing": {
                "source_pth_layout": "TCHW_float32",
                "source_rgb_range": [0, 255],
                "producer_rgb_quantization": "floor(float32_rgb+0.5).astype(uint8)",
                "process_res": PROCESS_RES,
                "process_res_method": "upper_bound_resize",
                "producer_depth_restore": "bilinear_align_corners_false_no_antialias",
                "world_model_resize": "short_edge_224_bilinear_antialias_true",
                "world_model_crop": "center_224",
                "normalization": "clip((depth_m-lo)/(hi-lo),0,1)",
                "normalization_lo": lo,
                "normalization_hi": hi,
                "wire_dtype": WIRE_DTYPE.str,
            },
            "auxiliary_sources": {
                key: value
                for key, value in contract["auxiliary"].items()
                if key != "arrays"
            },
            "planning_targets": contract["planning"],
            "arrays": {
                "terminal_rgb_f32_chw": array_descriptor(rgb_f32_path, rgb_f32),
                "terminal_rgb_uint8_hwc": array_descriptor(rgb_u8_path, rgb_u8),
                "terminal_depth_normalized_f16_hw": array_descriptor(depth_path, depth_f16),
                "terminal_depth_validity_u8_hw": array_descriptor(validity_path, depth_validity),
            },
            "routes": route_records,
        }
        atomic_write_json(building / "terminal_input_receipt.json", receipt)
        os.replace(building, out)
    except Exception:
        for path in sorted(building.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        building.rmdir()
        raise


def check_bundle(
    *,
    bundle: Path,
    contract: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    receipt_path = bundle / "terminal_input_receipt.json"
    if not receipt_path.is_file():
        fail(f"missing terminal-input receipt {receipt_path}")
    receipt_sha = sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("state") != "PASS":
        fail("terminal-input receipt is not a PASS bundle")
    if receipt.get("route_count") != EXPECTED_ROUTE_COUNT:
        fail("terminal-input route count changed")
    if receipt.get("terminal_observation_index") != 50:
        fail("terminal observation index changed")
    protocol = receipt.get("fixed_protocol", {})
    expected_protocol = {
        "observation_count": 51,
        "action_count": 50,
        "horizons": [1, 5, 10],
        "frameskip": 5,
        "num_hist": 1,
        "held_out_split": "valid",
        "selected_prediction_episodes": 192,
        "planning_target_protocol_unchanged": True,
    }
    if protocol != expected_protocol:
        fail("terminal-input receipt changed the frozen evaluation protocol")
    if receipt.get("fixed_manifest", {}).get("sha256") != contract["fixed"]["sha256"]:
        fail("terminal-input receipt is bound to a different fixed manifest")
    if receipt.get("accepted_depth_cache", {}).get("manifest_sha256") != contract["cache"]["sha256"]:
        fail("terminal-input receipt is bound to a different accepted depth cache")
    if receipt.get("producer_source_commit") != EXPECTED_PRODUCER_SOURCE_COMMIT:
        fail("terminal-input receipt does not use the accepted producer source commit")
    if receipt.get("producer", {}).get("settings") != EXPECTED_PRODUCER_SETTINGS:
        fail("terminal-input receipt producer settings changed")
    if producer_identity(receipt.get("producer", {})) != producer_identity(
        contract["cache"]["manifest"]["producer"]
    ):
        fail("terminal-input receipt producer identity differs from accepted cache")
    array_values: dict[str, np.ndarray] = {}
    for name, descriptor in receipt.get("arrays", {}).items():
        path = bundle / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
            fail(f"terminal array hash mismatch for {name}")
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != descriptor.get("shape") or np.dtype(array.dtype).str != descriptor.get("dtype"):
            fail(f"terminal array descriptor mismatch for {name}")
        array_values[name] = array
    required_arrays = {
        "terminal_rgb_f32_chw",
        "terminal_rgb_uint8_hwc",
        "terminal_depth_normalized_f16_hw",
        "terminal_depth_validity_u8_hw",
    }
    if set(array_values) != required_arrays:
        fail("terminal bundle array set changed")
    rgb_f32 = array_values["terminal_rgb_f32_chw"]
    rgb_u8 = array_values["terminal_rgb_uint8_hwc"]
    depth = array_values["terminal_depth_normalized_f16_hw"]
    validity = array_values["terminal_depth_validity_u8_hw"]
    if rgb_f32.shape != (192, 3, 224, 224) or rgb_f32.dtype != np.float32:
        fail("terminal source RGB array geometry/dtype changed")
    if rgb_u8.shape != (192, 224, 224, 3) or rgb_u8.dtype != np.uint8:
        fail("terminal producer RGB array geometry/dtype changed")
    if depth.shape != (192, 224, 224) or np.dtype(depth.dtype).str != "<f2":
        fail("terminal depth array geometry/dtype changed")
    if validity.shape != (192, 224, 224) or validity.dtype != np.uint8 or not np.all(validity == 1):
        fail("terminal depth validity is not the all-finite producer mask")
    if not np.isfinite(rgb_f32).all() or float(rgb_f32.min()) < 0 or float(rgb_f32.max()) > 255:
        fail("terminal source RGB contains invalid values")
    if not np.isfinite(depth).all() or float(depth.min()) < 0 or float(depth.max()) > 1:
        fail("terminal normalized depth contains invalid values")
    routes = receipt.get("routes", [])
    if len(routes) != EXPECTED_ROUTE_COUNT:
        fail("terminal receipt route records are incomplete")
    route_by_key = {route.get("key"): route for route in routes}
    if len(route_by_key) != EXPECTED_ROUTE_COUNT:
        fail("terminal receipt route keys are not unique")
    for index, row in enumerate(contract["fixed"]["rows"]):
        key = row["key"]
        route = route_by_key.get(key)
        if route is None:
            fail(f"terminal receipt is missing route {key}")
        trajectory, raw50, decoded, source_audit = source_frame_audit(row, root)
        expected_cache_route(row, contract["cache"])
        if route.get("source_sha256") != row["source_sha256"] or route.get("source_path") != row["source_path"]:
            fail(f"terminal receipt source binding changed for {key}")
        if route.get("source_frame_count") != 51 or route.get("source_frame_index") != 50:
            fail(f"terminal receipt terminal source frame changed for {key}")
        if route.get("source_frame_50_f32_chw_sha256") != source_audit["source_frame_50_f32_chw_sha256"]:
            fail(f"terminal receipt source frame 50 hash mismatch for {key}")
        if route.get("source_frame_49_f32_chw_sha256") != source_audit["source_frame_49_f32_chw_sha256"]:
            fail(f"terminal receipt source frame 49 hash mismatch for {key}")
        selection = route.get("terminal_output_selection", {})
        if selection.get("producer_output_frame_index") != 50 or selection.get("source_rgb_frame_index") != 50:
            fail(f"terminal output selection changed for {key}")
        if selection.get("source_frame_count") != 51 or selection.get("producer_input_frame_count") != 51:
            fail(f"terminal producer input count changed for {key}")
        if selection.get("copy_interpolate_pad_wrap_duplicate_or_other_route") is not False:
            fail(f"terminal output provenance permits an invalid substitution for {key}")
        if route.get("producer_metadata", {}).get("input_frame_count") != 51:
            fail(f"terminal producer metadata was truncated for {key}")
        if route.get("producer_metadata", {}).get("trajectory_only_call") is not True:
            fail(f"terminal producer call was not trajectory-only for {key}")
        if route.get("chunk_boundaries") != streaming_chunks(51):
            fail(f"terminal chunk schedule changed for {key}")
        if route.get("terminal_resize_record") != {
            "decoded_hw": [224, 224],
            "producer_depth_hw": [504, 504],
            "producer_to_decoded_scale_yx": [0.4444444444444444, 0.4444444444444444],
            "world_model_resize": "short_edge_224_bilinear_antialias_true",
            "world_model_crop": "center_224",
        }:
            fail(f"terminal preprocessing settings changed for {key}")
        if not np.array_equal(rgb_f32[index], raw50):
            fail(f"terminal RGB array is not the exact original frame 50 for {key}")
        if not np.array_equal(rgb_u8[index], decoded[50]):
            fail(f"terminal producer RGB input is not the exact quantized frame 50 for {key}")
        expected_uint8 = np.floor(np.transpose(raw50, (1, 2, 0)) + 0.5).astype(np.uint8)
        if not np.array_equal(rgb_u8[index], expected_uint8):
            fail(f"terminal RGB quantization changed for {key}")
        if route.get("terminal_depth_f16_hw_sha256") != sha256_bytes(
            np.ascontiguousarray(depth[index]).tobytes(order="C")
        ):
            fail(f"terminal depth row hash mismatch for {key}")
        for name, episode_audit in route.get("auxiliary_episode_sources", {}).items():
            episode_array = np.ascontiguousarray(contract["auxiliary"]["arrays"][name][int(row["episode"])])
            if episode_audit.get("terminal_observation_index_50") != "absent":
                fail(f"terminal auxiliary source was extended for {name} on {key}")
            if episode_audit.get("episode_values_sha256") != sha256_bytes(episode_array.tobytes(order="C")):
                fail(f"terminal auxiliary source hash mismatch for {name} on {key}")
    check = {
        "schema": "dinocular.wall-terminal-input-check.v1",
        "state": "PASS",
        "checked_utc": utc_now(),
        "bundle": str(bundle),
        "receipt_sha256": receipt_sha,
        "route_count": EXPECTED_ROUTE_COUNT,
        "route_coverage": {
            "observations": "0..50_exactly",
            "actions": "0..49_exactly",
            "terminal_depth_source": "each_route_original_rgb_frame_50",
            "terminal_rgb_source": "each_route_original_rgb_frame_50",
            "producer_identity": "accepted_Wall_DA3_Streaming_exact",
            "auxiliary_source": "original_recorded_tensors_only; no_index_50_state_or_location_exists",
            "planning_targets": "fixed_wall_50_manifest_unchanged",
            "loss_or_substitution": False,
        },
        "fixed_manifest_sha256": contract["fixed"]["sha256"],
        "accepted_cache_manifest_sha256": contract["cache"]["sha256"],
        "accepted_cache_data_mdb_sha256": contract["cache"]["data_mdb_sha256"],
        "planning_targets_sha256": contract["planning"]["sha256"],
        "source_frame_50_count": EXPECTED_ROUTE_COUNT,
        "terminal_depth_count": EXPECTED_ROUTE_COUNT,
        "terminal_auxiliary_index_50": {
            name: "absent_in_original_source"
            for name in EXPECTED_AUXILIARY
        },
    }
    return check


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("materialize", "check"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--fixed-meta", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--validation-evidence", type=Path, required=True)
    parser.add_argument("--planning-targets", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, help="existing bundle for --mode check")
    parser.add_argument("--out", type=Path, help="new bundle for --mode materialize")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--da3-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--producer-source-commit", default=EXPECTED_PRODUCER_SOURCE_COMMIT)
    parser.add_argument("--materializer-commit", default="unknown")
    parser.add_argument("--check-report", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "materialize":
        for name in ("out", "model_dir", "da3_root"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required for materialize")
    elif args.bundle is None:
        parser.error("--bundle is required for check")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contract = validate_contract(
            root=args.root,
            fixed_manifest_path=args.fixed_manifest,
            fixed_meta_path=args.fixed_meta,
            cache_manifest_path=args.cache_manifest,
            validation_evidence_path=args.validation_evidence,
            planning_targets_path=args.planning_targets,
        )
        if args.mode == "materialize":
            write_bundle(
                out=args.out,
                contract=contract,
                root=args.root,
                model_dir=args.model_dir,
                da3_root=args.da3_root,
                work_root=args.work_root,
                producer_source_commit=args.producer_source_commit,
                materializer_commit=args.materializer_commit,
            )
            check = check_bundle(bundle=args.out, contract=contract, root=args.root)
        else:
            check = check_bundle(bundle=args.bundle, contract=contract, root=args.root)
        if args.check_report is not None:
            atomic_write_json(args.check_report, check)
        print(json.dumps(check, sort_keys=True))
        return 0
    except ContractError as exc:
        print(json.dumps({"schema": "dinocular.wall-terminal-input-check.v1", "state": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
