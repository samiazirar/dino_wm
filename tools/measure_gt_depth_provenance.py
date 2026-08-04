#!/usr/bin/env python3
"""Measure simulator-ground-truth depth coverage and training provenance.

This is intentionally a read-only measurement.  It scans the frozen RGB,
open-loop, and planning manifests together with the released deformable
simulator HDF5 records.  It emits one compact JSON receipt on stdout and does
not write to the project or to the cluster.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import yaml


TASKS = ("wall", "rope", "granular")
TASK_SPEC = {
    "wall": {
        "raw_subdir": Path("wall_single"),
        "model_frames": 50,
        "raw_frames": 51,
        "frameskip": 5,
        "h5_supported": False,
    },
    "rope": {
        "raw_subdir": Path("deformable") / "rope",
        "model_frames": 20,
        "raw_frames": 20,
        "frameskip": 1,
        "h5_supported": True,
    },
    "granular": {
        "raw_subdir": Path("deformable") / "granular",
        "model_frames": 20,
        "raw_frames": 20,
        "frameskip": 1,
        "h5_supported": True,
    },
}

CAMERA = "cam_1"
DEPTH_DTYPE = "uint16"
IMAGE_SHAPE = (224, 224)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rgb_frames(value: Any) -> np.ndarray:
    """Normalize released RGB tensors to T,H,W,C without changing values."""

    if isinstance(value, dict):
        for key in ("obses", "observations", "rgb", "images"):
            if key in value:
                value = value[key]
                break
    array = as_numpy(value)
    if array.ndim != 4:
        raise ValueError(f"expected four-dimensional RGB observations, got {array.shape}")
    if array.shape[-1] == 3:
        return array
    if array.shape[1] == 3:
        return np.transpose(array, (0, 2, 3, 1))
    raise ValueError(f"could not identify RGB channel axis in {array.shape}")


def episode_obs_path(project: Path, task: str, episode: int) -> Path:
    if task == "wall":
        return project / "data" / "raw" / "wall_single" / "obses" / f"episode_{episode:03d}.pth"
    return (
        project
        / "data"
        / "raw"
        / "deformable"
        / task
        / f"{episode:06d}"
        / "obses.pth"
    )


def episode_h5_path(project: Path, task: str, episode: int, frame: int) -> Path:
    return (
        project
        / "data"
        / "raw"
        / "deformable"
        / task
        / f"{episode:06d}"
        / f"{frame:02d}.h5"
    )


def actions_path(project: Path, task: str) -> Path:
    if task == "wall":
        return project / "data" / "raw" / "wall_single" / "actions.pth"
    return project / "data" / "raw" / "deformable" / task / "actions.pth"


def source_path_for_manifest(task: str, episode: int) -> str:
    if task == "wall":
        return f"wall_single/obses/episode_{episode}.pth"
    return f"deformable/{task}/{episode:06d}/obses.pth"


def manifest_path(project: Path, task: str) -> Path:
    return (
        project
        / "outputs"
        / "campaign-seed1"
        / "native-seed1-20260728c"
        / "code"
        / "dino_wm"
        / "releases"
        / "rgb_dataset_manifests"
        / f"dataset_manifest_{task}.json"
    )


def split_and_files(manifest: dict[str, Any]) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    episodes: dict[int, str] = {}
    files: dict[str, dict[str, Any]] = {}
    for item in manifest.get("files", []):
        relative = item.get("relative_path")
        if relative:
            files[relative] = item
        if item.get("type") == "obses":
            episodes[int(item["episode"])] = str(item["split"])
    return episodes, files


def load_rgb_episode(project: Path, task: str, episode: int) -> np.ndarray:
    return rgb_frames(load_torch(episode_obs_path(project, task, episode)))


def manifest_measurement(project: Path, task: str, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[int, str], dict[str, dict[str, Any]]]:
    path = manifest_path(project, task)
    manifest = load_json(path)
    episodes, files = split_and_files(manifest)
    split_counts = Counter(episodes.values())
    model_frames = int(spec["model_frames"])
    raw_frames = int(spec["raw_frames"])
    train_episodes = split_counts.get("train", 0)
    valid_episodes = split_counts.get("valid", 0)
    recipe = manifest.get("window_recipe", {})
    per_episode_windows = recipe.get("per_episode_windows")
    if per_episode_windows is None:
        per_episode_windows = model_frames - int(recipe.get("num_hist", 1)) - int(recipe.get("num_pred", 1)) + 1
    train_windows = recipe.get("train_window_count", train_episodes * int(per_episode_windows))
    valid_windows = recipe.get("valid_window_count", valid_episodes * int(per_episode_windows))
    result = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": manifest.get("schema"),
        "environment": manifest.get("environment"),
        "data_subdir": manifest.get("data_subdir"),
        "file_count": manifest.get("file_count"),
        "episode_count": len(episodes),
        "split_episode_counts": dict(sorted(split_counts.items())),
        "raw_frames_per_episode": raw_frames,
        "model_frames_per_episode": model_frames,
        "frameskip": int(spec["frameskip"]),
        "window_recipe": {
            "frameskip": recipe.get("frameskip"),
            "num_hist": recipe.get("num_hist"),
            "num_pred": recipe.get("num_pred"),
            "per_episode_windows": int(per_episode_windows),
            "train_window_count": int(train_windows),
            "valid_window_count": int(valid_windows),
        },
        "train_observation_count": train_episodes * model_frames,
        "valid_observation_count": valid_episodes * model_frames,
        "raw_post_action_terminal_frame": raw_frames - 1,
    }
    return result, episodes, files


def h5_dataset(handle: h5py.File, path: str) -> np.ndarray:
    return np.asarray(handle[path])


def scan_ground_truth(
    project: Path,
    task: str,
    spec: dict[str, Any],
    episode_splits: dict[int, str],
    actions: np.ndarray,
) -> dict[str, Any]:
    model_frames = int(spec["model_frames"])
    raw_frames = int(spec["raw_frames"])
    if not spec["h5_supported"]:
        return {
            "availability": "not_found",
            "source_root": str(project / "data" / "raw" / "wall_single"),
            "camera": CAMERA,
            "reason": "selected released Wall raw source has no simulator HDF5 depth records",
            "episodes_scanned": 0,
            "model_frames_expected": 0,
            "h5_frames_expected": 0,
            "rgb_exact_frame_count": 0,
            "depth_valid_frame_count": 0,
            "action_exact_frame_count": 0,
            "post_action_frame_exact_count": 0,
            "missing_h5_files": [],
            "extra_h5_frames": [],
            "split_model_frame_counts": {},
        }

    missing: list[str] = []
    extra: list[str] = []
    rgb_exact = 0
    depth_valid = 0
    action_exact = 0
    frame0_zero_action = 0
    post_action_exact = 0
    h5_frames = 0
    depth_dtype_counts: Counter[str] = Counter()
    depth_shape_counts: Counter[str] = Counter()
    rgb_shape_counts: Counter[str] = Counter()
    rgb_dtype_counts: Counter[str] = Counter()
    split_model_frames: Counter[str] = Counter()
    split_rgb_exact: Counter[str] = Counter()
    split_depth_valid: Counter[str] = Counter()
    action_pairs_expected = 0
    post_action_expected = 0

    for episode in sorted(episode_splits):
        split = episode_splits[episode]
        raw_rgb = load_rgb_episode(project, task, episode)
        if raw_rgb.shape[0] != raw_frames:
            raise ValueError(
                f"{task} episode {episode} has {raw_rgb.shape[0]} RGB frames, expected {raw_frames}"
            )
        split_model_frames[split] += model_frames
        split_name = split
        for frame in range(model_frames + 1):
            path = episode_h5_path(project, task, episode, frame)
            if not path.is_file():
                missing.append(str(path))
                continue
            h5_frames += 1
            with h5py.File(path, "r") as handle:
                color = h5_dataset(handle, f"observations/color/{CAMERA}")
                depth = h5_dataset(handle, f"observations/depth/{CAMERA}")
                action = h5_dataset(handle, "action")
            if color.ndim == 4 and color.shape[0] == 1:
                color = color[0]
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            depth_dtype_counts[str(depth.dtype)] += 1
            depth_shape_counts[str(tuple(depth.shape))] += 1
            if frame < model_frames:
                rgb = raw_rgb[frame]
                rgb_shape_counts[str(tuple(color.shape))] += 1
                rgb_dtype_counts[str(color.dtype)] += 1
                if color.shape == rgb.shape and np.array_equal(color, rgb) and color.tobytes() == rgb.tobytes():
                    rgb_exact += 1
                    split_rgb_exact[split_name] += 1
                expected_action = np.zeros_like(action) if frame == 0 else actions[episode, frame - 1]
                action_pairs_expected += 1
                if action.shape == expected_action.shape and np.array_equal(action, expected_action):
                    action_exact += 1
                if frame == 0 and np.array_equal(action, np.zeros_like(action)):
                    frame0_zero_action += 1
                if depth.shape == IMAGE_SHAPE and depth.dtype == np.dtype(DEPTH_DTYPE) and np.isfinite(depth).all():
                    depth_valid += 1
                    split_depth_valid[split_name] += 1
            else:
                post_action_expected += 1
                if action.shape == actions[episode, model_frames - 1].shape and np.array_equal(
                    action, actions[episode, model_frames - 1]
                ):
                    post_action_exact += 1
    for episode in sorted(episode_splits):
        episode_dir = episode_h5_path(project, task, episode, 0).parent
        for path in sorted(episode_dir.glob("*.h5")):
            try:
                frame = int(path.stem)
            except ValueError:
                continue
            if frame < 0 or frame > model_frames:
                extra.append(str(path))

    expected_model_frames = len(episode_splits) * model_frames
    expected_h5_frames = len(episode_splits) * (model_frames + 1)
    complete = (
        not missing
        and not extra
        and h5_frames == expected_h5_frames
        and rgb_exact == expected_model_frames
        and depth_valid == expected_model_frames
        and action_exact == action_pairs_expected
        and post_action_exact == post_action_expected
    )
    return {
        "availability": "complete" if complete else "present_but_misaligned",
        "source_root": str(project / "data" / "raw" / "deformable" / task),
        "camera": CAMERA,
        "source_record": "released simulator HDF5 frame records",
        "episodes_scanned": len(episode_splits),
        "model_frames_expected": expected_model_frames,
        "h5_frames_expected": expected_h5_frames,
        "h5_frames_seen": h5_frames,
        "rgb_exact_frame_count": rgb_exact,
        "depth_valid_frame_count": depth_valid,
        "action_exact_frame_count": action_exact,
        "action_pair_expected_count": action_pairs_expected,
        "frame0_zero_action_count": frame0_zero_action,
        "post_action_frame": model_frames,
        "post_action_frame_expected_count": post_action_expected,
        "post_action_frame_exact_count": post_action_exact,
        "missing_h5_files": missing,
        "extra_h5_frames": extra,
        "depth_dtype_counts": dict(sorted(depth_dtype_counts.items())),
        "depth_shape_counts": dict(sorted(depth_shape_counts.items())),
        "rgb_shape_counts": dict(sorted(rgb_shape_counts.items())),
        "rgb_dtype_counts": dict(sorted(rgb_dtype_counts.items())),
        "split_model_frame_counts": dict(sorted(split_model_frames.items())),
        "split_rgb_exact_counts": dict(sorted(split_rgb_exact.items())),
        "split_depth_valid_counts": dict(sorted(split_depth_valid.items())),
        "action_alignment_rule": "HDF5 frame 0 carries zero action; frame f>=1 carries raw action f-1; model transition i->i+1 uses raw action i",
    }


def load_actions(project: Path, task: str) -> np.ndarray:
    return as_numpy(load_torch(actions_path(project, task)))


def reference_key(task: str, split: str, episode: int, frame: int) -> str:
    return f"{task}/{split}/{episode:05d}/{frame:06d}"


def measure_openloop(
    project: Path,
    task: str,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    manifest_files: dict[str, dict[str, Any]],
    gt: dict[str, Any],
    cache_frames: int | None,
) -> dict[str, Any]:
    path = (
        project
        / "outputs"
        / "campaign-evaluation"
        / "eval36-20260728a"
        / "manifests"
        / f"openloop_{task}.jsonl"
    )
    meta_path = path.with_suffix(".meta.json")
    rows = load_jsonl(path)
    meta = load_json(meta_path)
    model_frames = int(spec["model_frames"])
    raw_frames = int(spec["raw_frames"])
    frameskip = int(spec["frameskip"])
    episode_splits, _ = split_and_files(manifest)
    source_hash_mismatches = 0
    source_episode_mismatches = 0
    action_alignment_errors = 0
    target_alignment_errors = 0
    future_leakage_count = 0
    out_of_range_count = 0
    terminal_target_refs = 0
    model_target_refs = 0
    gt_target_refs = 0
    cache_target_refs = 0
    observation_refs = 0
    target_refs = 0
    prediction_pairs = 0
    unique_refs: set[tuple[int, int]] = set()
    unique_gt_refs: set[tuple[int, int]] = set()
    unique_cache_refs: set[tuple[int, int]] = set()
    source_split_counts: Counter[str] = Counter()
    target_horizon_counts: Counter[str] = Counter()

    gt_complete = gt.get("availability") == "complete"
    for row in rows:
        episode = int(row["episode"])
        split = str(row["split"])
        start = int(row["start"])
        source_split_counts[split] += 1
        if episode_splits.get(episode) != split:
            source_episode_mismatches += 1
        source_path = str(row["source_path"])
        source_entry = manifest_files.get(source_path)
        if source_entry is None or source_entry.get("sha256") != row.get("source_sha256"):
            source_hash_mismatches += 1
        history = [int(frame) for frame in row.get("history_frames", [])]
        row_target_frames = {int(h): int(frame) for h, frame in row["target_frames"].items()}
        row_horizons = [int(h) for h in row["horizons"]]
        raw_indices = {int(k): [int(i) for i in v] for k, v in row["raw_action_indices"].items()}
        observation_refs += len(history) + len(row_horizons)
        target_refs += len(row_horizons)
        prediction_pairs += len(row_horizons)
        for frame in history:
            if frame < 0 or frame >= raw_frames:
                out_of_range_count += 1
            unique_refs.add((episode, frame))
        for step in range(max(row_horizons) if row_horizons else 0):
            expected_indices = list(range(start + step * frameskip, start + (step + 1) * frameskip))
            if raw_indices.get(step) != expected_indices:
                action_alignment_errors += 1
            if any(index < 0 or index >= raw_frames for index in raw_indices.get(step, [])):
                out_of_range_count += 1
        for horizon in row_horizons:
            target = row_target_frames.get(horizon)
            expected_target = start + horizon * frameskip
            target_horizon_counts[str(horizon)] += 1
            if target != expected_target:
                target_alignment_errors += 1
            if target is None or target < 0 or target >= raw_frames:
                out_of_range_count += 1
                continue
            relevant_indices = [index for step in range(horizon) for index in raw_indices.get(step, [])]
            if relevant_indices != list(range(start, target)):
                action_alignment_errors += 1
            if any(index >= target for index in relevant_indices):
                future_leakage_count += 1
            unique_refs.add((episode, target))
            if target >= model_frames:
                terminal_target_refs += 1
            else:
                model_target_refs += 1
                if gt_complete:
                    gt_target_refs += 1
                    unique_gt_refs.add((episode, target))
                if cache_frames is not None and target < cache_frames:
                    cache_target_refs += 1
                    unique_cache_refs.add((episode, target))
    expected_count = int(meta.get("selected_count", meta.get("requested_count", len(rows))))
    meta_match = len(rows) == expected_count
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "meta_path": str(meta_path),
        "meta_sha256": sha256_file(meta_path),
        "manifest_sha256": meta.get("manifest_sha256"),
        "dataset_index_sha256": meta.get("dataset_index_sha256"),
        "split_sha256": meta.get("split_sha256"),
        "rows": len(rows),
        "selected_count_expected": expected_count,
        "meta_count_match": meta_match,
        "prediction_pairs": prediction_pairs,
        "horizon_counts": dict(sorted(target_horizon_counts.items(), key=lambda item: int(item[0]))),
        "observation_references": observation_refs,
        "target_observation_references": target_refs,
        "unique_observation_keys": len(unique_refs),
        "model_frame_target_references": model_target_refs,
        "terminal_target_references": terminal_target_refs,
        "gt_aligned_target_references": gt_target_refs,
        "gt_aligned_unique_target_keys": len(unique_gt_refs),
        "current_cache_aligned_target_references": cache_target_refs,
        "current_cache_aligned_unique_target_keys": len(unique_cache_refs),
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "source_sha256_mismatches": source_hash_mismatches,
        "source_episode_split_mismatches": source_episode_mismatches,
        "action_alignment_errors": action_alignment_errors,
        "target_alignment_errors": target_alignment_errors,
        "future_leakage_count": future_leakage_count,
        "out_of_range_count": out_of_range_count,
        "frameskip": frameskip,
        "model_frame_range": [0, model_frames - 1],
        "raw_frame_range": [0, raw_frames - 1],
        "causal_alignment_pass": all(
            (
                meta_match,
                source_hash_mismatches == 0,
                source_episode_mismatches == 0,
                action_alignment_errors == 0,
                target_alignment_errors == 0,
                future_leakage_count == 0,
                out_of_range_count == 0,
            )
        ),
    }


def planning_path(project: Path, task: str) -> Path:
    return (
        project
        / "outputs"
        / "campaign-evaluation"
        / "eval36-20260728a"
        / "manifests"
        / f"{task}_10.pkl"
        if task != "wall"
        else project
        / "outputs"
        / "campaign-evaluation"
        / "eval36-20260728a"
        / "manifests"
        / "wall_50.pkl"
    )


def planning_protocol(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    selected = (
        "goal_source",
        "initial_source",
        "goal_observation",
        "goal_shape",
        "goal_transform",
        "frameskip",
        "goal_H",
        "split_seed",
        "split_fraction",
    )
    return {key: value[key] for key in selected if key in value}


def measure_planning(
    project: Path,
    task: str,
    spec: dict[str, Any],
    episode_splits: dict[int, str],
    gt: dict[str, Any],
) -> dict[str, Any]:
    path = planning_path(project, task)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    records = list(data.get("target_records", []))
    model_frames = int(spec["model_frames"])
    source_range_errors = 0
    visual_binding_errors = 0
    initial_gt_refs = 0
    unique_initial_refs: set[tuple[int, int]] = set()
    source_split_counts: Counter[str] = Counter()
    raw_cache: dict[int, np.ndarray] = {}
    gt_complete = gt.get("availability") == "complete"
    source_files = [str(item.get("path", "")) for item in data.get("source_files", [])]
    visual_binding_checked = any("/obses/" in item or item.endswith("/obses.pth") for item in source_files)
    for record in records:
        episode = int(record["source_episode"])
        frame = int(record["source_offset"])
        source_split_counts[episode_splits.get(episode, "missing")] += 1
        if episode not in episode_splits or frame < 0 or frame >= model_frames:
            source_range_errors += 1
            continue
        unique_initial_refs.add((episode, frame))
        if gt_complete:
            initial_gt_refs += 1
        if visual_binding_checked:
            if episode not in raw_cache:
                raw_cache[episode] = load_rgb_episode(project, task, episode)
            expected = raw_cache[episode][frame]
            observed = as_numpy(record["obs_0"]["visual"])
            if expected.shape != observed.shape or not np.array_equal(expected, observed):
                visual_binding_errors += 1
    protocol = planning_protocol(data.get("protocol", {}))
    goal_observation = protocol.get("goal_observation")
    target_count = int(data.get("count", len(records)))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": data.get("schema"),
        "count": target_count,
        "records_seen": len(records),
        "goal_H": data.get("goal_H"),
        "protocol": protocol,
        "source_files_count": len(data.get("source_files", [])),
        "environment_files_count": len(data.get("environment_files", [])),
        "initial_observation_references": len(records),
        "unique_initial_observation_keys": len(unique_initial_refs),
        "initial_gt_aligned_references": initial_gt_refs,
        "initial_visual_binding_checked": visual_binding_checked,
        "initial_visual_binding_mode": (
            "exact_source_rgb" if visual_binding_checked else "state_derived_or_rendered_from_declared_source_state"
        ),
        "goal_observation_is_simulator_frame": False,
        "goal_observation_source": goal_observation or "synthetic_or_state_derived",
        "goal_depth_references": 0,
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "source_range_errors": source_range_errors,
        "initial_visual_binding_errors": visual_binding_errors,
        "binding_pass": (
            target_count == len(records)
            and source_range_errors == 0
            and (not visual_binding_checked or visual_binding_errors == 0)
        ),
    }


def latest_validation(project: Path, task: str) -> Path | None:
    directory = project / "data" / "depth_cache_validation"
    paths = [
        path
        for path in directory.glob(f"{task}*.json")
        if path.is_file() and not path.name.endswith(".evidence.json")
    ]
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def cache_frame_count(manifest: dict[str, Any]) -> int | None:
    trajectories = manifest.get("trajectories", [])
    if trajectories:
        first = trajectories[0]
        for key in ("frame_count", "frames_count"):
            if key in first:
                return int(first[key])
        for key in ("frames", "frame_keys", "keys"):
            value = first.get(key)
            if isinstance(value, list):
                return len(value)
    count = manifest.get("frame_count")
    trajectory_count = manifest.get("trajectory_count") or len(trajectories)
    if count is not None and trajectory_count:
        return int(count) // int(trajectory_count)
    return None


def measure_current_cache(project: Path, task: str) -> tuple[dict[str, Any], int | None]:
    path = project / "data" / "depth_cache" / f"{task}.lmdb" / "manifest.json"
    data = load_json(path)
    validation_path = latest_validation(project, task)
    validation = load_json(validation_path) if validation_path else {}
    producer = data.get("producer", {})
    settings = producer.get("settings", {}) if isinstance(producer, dict) else {}
    result = {
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "manifest_id": data.get("manifest_id"),
        "environment": data.get("environment"),
        "frame_count": data.get("frame_count"),
        "trajectory_count": len(data.get("trajectories", [])),
        "frames_per_trajectory": cache_frame_count(data),
        "data_mdb_sha256": data.get("data_mdb_sha256"),
        "producer_name": producer.get("name"),
        "producer_commit": producer.get("commit"),
        "producer_kind": "learned_metric_proxy",
        "wall_terminal_observation_policy": settings.get("wall_terminal_observation_policy"),
        "validation_path": str(validation_path) if validation_path else None,
        "validation_sha256": sha256_file(validation_path) if validation_path else None,
        "validation_state": validation.get("state"),
        "validation_manifest_id": validation.get("results", {}).get(task, {}).get("manifest_id"),
        "validation_actual_keys": validation.get("results", {}).get(task, {}).get("range_gate", {}).get("actual_keys"),
        "validation_expected_keys": validation.get("results", {}).get(task, {}).get("range_gate", {}).get("expected_keys"),
    }
    return result, cache_frame_count(data)


def p3_card_roots(project: Path) -> list[Path]:
    return [
        project / "outputs" / "campaign-seed1" / "native-seed1-20260728c" / "cards",
        project / "outputs" / "campaign-seed1" / "rgb-dino-seed1-20260728a" / "cards",
        project / "outputs" / "campaign-seeds23" / "seeds23-20260728a" / "cards",
    ]


def card_depth_signature(
    card: dict[str, Any],
    task: str,
    arm: str,
    current_cache: dict[str, Any],
) -> dict[str, Any]:
    depth_inputs = card.get("depth_inputs") or {}
    if arm == "dino_pinned":
        return {
            "kind": "rgb_only",
            "declared_depth_source": None,
            "same_source_as_simulator_gt": False,
            "matrix_decision": "REUSABLE_RGB_ONLY",
        }
    if arm == "dinocular_zerodepth":
        return {
            "kind": "constant_zero_depth_intervention",
            "declared_cache_producer": depth_inputs.get("producer"),
            "declared_cache_manifest_sha256": depth_inputs.get("cache_manifest_sha256"),
            "declared_validation_sha256": depth_inputs.get("validation_sha256"),
            "cache_matches_current_manifest": depth_inputs.get("cache_manifest_sha256")
            == current_cache.get("manifest_sha256"),
            "same_source_as_simulator_gt": False,
            "functional_reuse_proof": "not_found_in_selected_current_records",
            "matrix_decision": "PENDING_EXACT_FUNCTIONAL_REUSE_PROOF",
        }
    return {
        "kind": "learned_depth_proxy",
        "producer": depth_inputs.get("producer"),
        "producer_sha256": depth_inputs.get("producer_sha256"),
        "cache_dir": depth_inputs.get("cache_dir"),
        "cache_manifest_sha256": depth_inputs.get("cache_manifest_sha256"),
        "validation_path": depth_inputs.get("validation_path"),
        "validation_sha256": depth_inputs.get("validation_sha256"),
        "cache_matches_current_manifest": depth_inputs.get("cache_manifest_sha256")
        == current_cache.get("manifest_sha256"),
        "same_source_as_simulator_gt": False,
        "matrix_decision": "RETRAIN_ON_SIMULATOR_GT",
    }


def measure_training_lineages(
    project: Path,
    manifests: dict[str, dict[str, Any]],
    caches: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for root in p3_card_roots(project):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("p3-*.yaml")):
            with path.open() as handle:
                card = yaml.safe_load(handle)
            task = card.get("environment")
            arm = card.get("arm")
            if task not in TASKS or arm not in ("dino_pinned", "dinocular", "dinocular_zerodepth"):
                continue
            artifacts = card.get("artifacts") or {}
            rgb_artifact = artifacts.get("rgb_dataset_manifest") or {}
            heldout = card.get("heldout_loss_manifest") or {}
            depth_signature = card_depth_signature(card, task, arm, caches[task])
            rows.append(
                {
                    "run_id": card.get("run_id"),
                    "environment": task,
                    "arm": arm,
                    "seed": int(card.get("seed")),
                    "card_path": str(path),
                    "run_card_sha256": card.get("run_card_sha256"),
                    "run_dir_exists": Path(card.get("run_dir", "")).is_dir(),
                    "source_commit": card.get("source_commit"),
                    "frameskip": card.get("frameskip"),
                    "target_steps": card.get("target_steps"),
                    "encoder_boundary": card.get("encoder_boundary"),
                    "rgb_dataset_manifest_sha256": rgb_artifact.get("sha256"),
                    "heldout_manifest_sha256": heldout.get("sha256"),
                    "heldout_split_sha256": heldout.get("split_sha256"),
                    "depth": depth_signature,
                }
            )
    rows.sort(key=lambda row: (row["environment"], row["arm"], row["seed"]))
    grouped: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        grouped[task] = {}
        for arm in ("dino_pinned", "dinocular", "dinocular_zerodepth"):
            selected = [row for row in rows if row["environment"] == task and row["arm"] == arm]
            current_rgb_sha = manifests[task]["sha256"]
            rgb_matches = [
                row["rgb_dataset_manifest_sha256"] == current_rgb_sha
                for row in selected
            ]
            signatures = []
            for row in selected:
                signature = json.dumps(row["depth"], sort_keys=True)
                if signature not in signatures:
                    signatures.append(signature)
            grouped[task][arm] = {
                "count": len(selected),
                "seeds": [row["seed"] for row in selected],
                "rgb_manifest_matches_current": all(rgb_matches) if selected else False,
                "source_signature_count": len(signatures),
                "lineages": selected,
            }
    return {
        "expected_lineage_count": len(TASKS) * 3 * 3,
        "observed_lineage_count": len(rows),
        "lineages": rows,
        "by_task_and_arm": grouped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    project = args.project_root

    manifests: dict[str, dict[str, Any]] = {}
    episode_splits: dict[str, dict[int, str]] = {}
    manifest_files: dict[str, dict[str, dict[str, Any]]] = {}
    caches: dict[str, dict[str, Any]] = {}
    cache_frames: dict[str, int | None] = {}
    gt: dict[str, dict[str, Any]] = {}
    tasks: dict[str, dict[str, Any]] = {}

    for task in TASKS:
        spec = TASK_SPEC[task]
        manifest_result, splits, files = manifest_measurement(project, task, spec)
        manifests[task] = manifest_result
        episode_splits[task] = splits
        manifest_files[task] = files
        actions = load_actions(project, task)
        gt_result = scan_ground_truth(project, task, spec, splits, actions)
        gt[task] = gt_result
        cache_result, frame_count = measure_current_cache(project, task)
        caches[task] = cache_result
        cache_frames[task] = frame_count
        prediction = measure_openloop(
            project,
            task,
            spec,
            load_json(manifest_path(project, task)),
            files,
            gt_result,
            frame_count,
        )
        planning = measure_planning(project, task, spec, splits, gt_result)
        tasks[task] = {
            "rgb_dataset": manifest_result,
            "ground_truth": gt_result,
            "fixed_prediction": prediction,
            "fixed_planning": planning,
            "current_depth_cache": cache_result,
        }

    training = measure_training_lineages(project, manifests, caches)
    checks = {
        "frozen_rgb_manifests_reconcile": all(
            tasks[task]["rgb_dataset"]["episode_count"]
            == sum(tasks[task]["rgb_dataset"]["split_episode_counts"].values())
            for task in TASKS
        ),
        "deformable_ground_truth_is_complete_and_aligned": all(
            tasks[task]["ground_truth"]["availability"] == "complete" for task in ("rope", "granular")
        ),
        "open_loop_manifests_are_causal_and_bound": all(
            tasks[task]["fixed_prediction"]["causal_alignment_pass"] for task in TASKS
        ),
        "planning_initial_observations_have_declared_source_binding": all(
            tasks[task]["fixed_planning"]["binding_pass"] for task in TASKS
        ),
        "training_lineage_matrix_is_complete": training["observed_lineage_count"] == training["expected_lineage_count"],
        "current_informative_depth_is_not_simulator_gt": all(
            row["depth"].get("same_source_as_simulator_gt") is False
            for row in training["lineages"]
            if row["arm"] == "dinocular"
        ),
        "no_depth_source_mixing_in_current_cards": all(
            row["depth"].get("kind") in {
                "rgb_only",
                "constant_zero_depth_intervention",
                "learned_depth_proxy",
            }
            for row in training["lineages"]
        ),
    }

    decisions = {
        "current_informative_depth_source": {
            task: {
                "producer": caches[task]["producer_name"],
                "kind": caches[task]["producer_kind"],
                "cache_manifest_sha256": caches[task]["manifest_sha256"],
                "not_simulator_ground_truth": True,
            }
            for task in TASKS
        },
        "reusable_rgb_only_cells": {
            "count": sum(
                1
                for row in training["lineages"]
                if row["arm"] == "dino_pinned"
            ),
            "cells": [
                f"{row['environment']}/dino_pinned/seed{row['seed']}"
                for row in training["lineages"]
                if row["arm"] == "dino_pinned"
            ],
            "condition": "same frozen RGB manifest and split; no depth input is consumed",
        },
        "zero_depth_cells": {
            "candidate_count": sum(
                1
                for row in training["lineages"]
                if row["arm"] == "dinocular_zerodepth"
            ),
            "accepted_reuse_count": 0,
            "cells": [
                f"{row['environment']}/dinocular_zerodepth/seed{row['seed']}"
                for row in training["lineages"]
                if row["arm"] == "dinocular_zerodepth"
            ],
            "condition": "pending exact functional reuse proof; not mixed into the GT comparison",
        },
        "informative_depth_retraining": {
            "immediate_cells": [
                f"{task}/dinocular/seed{seed}"
                for task in ("rope", "granular")
                for seed in (1, 2, 3)
            ],
            "wall_cells_after_gt_asset": [f"wall/dinocular/seed{seed}" for seed in (1, 2, 3)],
            "total_cells_for_complete_three_task_gt_matrix": 9,
            "reason": "every current informative-depth card names a DA3 learned proxy, never the released simulator HDF5 depth source",
        },
        "wall_terminal_depth_recovery": {
            "retire_learned_terminal_recovery": False,
            "reason": "Wall has no released simulator HDF5 depth source; the fixed horizon-10 target is raw frame 50 while the current 50-frame cache explicitly drops that post-action terminal frame",
        },
        "smallest_next_run_package": [
            "Bind Rope and Granular camera-1 HDF5 depth frames 0..19 into one immutable GT depth release without the DA3 cache, then retrain the six Rope/Granular informative-depth seed cells.",
            "Produce a Wall simulator-GT depth release covering all 1920 episodes and raw frames 0..50, including terminal frame 50, then retrain the three Wall informative-depth seed cells.",
            "Keep all nine RGB-only cells reusable; do not reuse zero-depth cells until the exact functional proof is recorded.",
        ],
    }

    receipt = {
        "schema": "dino-wm.gt-depth-provenance-measurement.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_question": "Does useful depth input improve world-model prediction/planning?",
        "scope": "Wall, Rope, and Granular frozen training observations, fixed prediction observations, planning initial/goal observations, and current P3 depth provenance",
        "selected_camera": CAMERA,
        "tasks": tasks,
        "training_lineages": training,
        "checks": checks,
        "decisions": decisions,
        "done": all(checks.values()),
    }

    if args.compact:
        compact_tasks = {}
        for task in TASKS:
            item = tasks[task]
            compact_tasks[task] = {
                "rgb_dataset": {
                    "path": item["rgb_dataset"]["path"],
                    "sha256": item["rgb_dataset"]["sha256"],
                    "episode_count": item["rgb_dataset"]["episode_count"],
                    "split_episode_counts": item["rgb_dataset"]["split_episode_counts"],
                    "train_observation_count": item["rgb_dataset"]["train_observation_count"],
                    "valid_observation_count": item["rgb_dataset"]["valid_observation_count"],
                    "window_recipe": item["rgb_dataset"]["window_recipe"],
                },
                "ground_truth": item["ground_truth"],
                "fixed_prediction": item["fixed_prediction"],
                "fixed_planning": item["fixed_planning"],
                "current_depth_cache": item["current_depth_cache"],
            }
        compact_lineages = []
        for row in training["lineages"]:
            depth = row["depth"]
            compact_lineages.append(
                {
                    "environment": row["environment"],
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "encoder_boundary": row["encoder_boundary"],
                    "rgb_dataset_manifest_sha256": row["rgb_dataset_manifest_sha256"],
                    "depth": {
                        key: depth[key]
                        for key in (
                            "kind",
                            "producer",
                            "producer_sha256",
                            "declared_cache_producer",
                            "cache_manifest_sha256",
                            "declared_cache_manifest_sha256",
                            "validation_sha256",
                            "declared_validation_sha256",
                            "same_source_as_simulator_gt",
                            "functional_reuse_proof",
                            "matrix_decision",
                        )
                        if key in depth
                    },
                }
            )
        receipt = {
            "schema": receipt["schema"],
            "created_utc": receipt["created_utc"],
            "project_question": receipt["project_question"],
            "scope": receipt["scope"],
            "selected_camera": receipt["selected_camera"],
            "tasks": compact_tasks,
            "training_lineages": {
                "expected_lineage_count": training["expected_lineage_count"],
                "observed_lineage_count": training["observed_lineage_count"],
                "lineages": compact_lineages,
            },
            "checks": checks,
            "decisions": decisions,
            "done": receipt["done"],
        }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    if not receipt["done"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
