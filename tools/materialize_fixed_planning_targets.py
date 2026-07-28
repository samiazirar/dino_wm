#!/usr/bin/env python3
"""Materialize the four immutable fixed planning-target manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import decord
import numpy as np
import torch


SELECTION_SEED = 20260714
GOAL_H = 5
COUNTS = {"pusht": 50, "wall": 50, "rope": 10, "granular": 10}
FILENAMES = {
    "pusht": "pusht_50.pkl",
    "wall": "wall_50.pkl",
    "rope": "rope_10.pkl",
    "granular": "granular_10.pkl",
}
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])


class TargetMaterializationError(RuntimeError):
    """A fixed source or immutable target artifact is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, np.ndarray):
            return {
                "dtype": str(item.dtype),
                "shape": list(item.shape),
                "sha256": hashlib.sha256(item.tobytes(order="C")).hexdigest(),
            }
        if isinstance(item, torch.Tensor):
            return normalize(item.detach().cpu().numpy())
        return item

    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _target_id(environment: str, ordinal: int, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        b"|".join(
            (
                str(SELECTION_SEED).encode(),
                environment.encode(),
                str(ordinal).encode(),
                _canonical(payload),
            )
        )
    ).hexdigest()
    return f"{environment}-{ordinal:03d}-{digest[:16]}"


def _write_pickle_immutable(path: Path, value: Mapping[str, Any]) -> None:
    data = pickle.dumps(value, protocol=4)
    if path.exists():
        if path.read_bytes() != data:
            raise TargetMaterializationError(f"immutable target differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _stack_observations(values: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    keys = set(values[0])
    if any(set(value) != keys for value in values):
        raise TargetMaterializationError("observation keys differ across targets")
    return {
        key: np.stack([np.asarray(value[key]) for value in values], axis=0)[:, None]
        for key in sorted(keys)
    }


def _base_manifest(
    environment: str,
    source_files: list[Path],
    protocol: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    target_ids = [
        _target_id(environment, ordinal, record)
        for ordinal, record in enumerate(records)
    ]
    if len(records) != COUNTS[environment] or len(set(target_ids)) != len(target_ids):
        raise TargetMaterializationError(f"{environment} target cardinality mismatch")
    manifest = {
        "schema": "dinocular.fixed-planning-targets.v1",
        "environment": environment,
        "selection_seed": SELECTION_SEED,
        "count": len(records),
        "target_ids": target_ids,
        "source_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in source_files
        ],
        "environment_files": [],
        "protocol": dict(protocol),
        "target_records": records,
        "obs_0": _stack_observations([record["obs_0"] for record in records]),
        "obs_g": _stack_observations([record["obs_g"] for record in records]),
        "state_0": np.stack([np.asarray(record["state_0"]) for record in records]),
        "state_g": np.stack([np.asarray(record["state_g"]) for record in records]),
        "gt_actions": None,
        "goal_H": GOAL_H,
    }
    return manifest


def _pusht(source_root: Path) -> dict[str, Any]:
    root = source_root / "pusht_noise" / "val"
    state_path = root / "states.pth"
    velocity_path = root / "velocities.pth"
    action_path = root / "rel_actions.pth"
    lengths_path = root / "seq_lengths.pkl"
    states = torch.load(state_path, map_location="cpu").float()
    velocities = torch.load(velocity_path, map_location="cpu").float()
    actions = torch.load(action_path, map_location="cpu").float() / 100.0
    actions = (actions - ACTION_MEAN) / ACTION_STD
    with lengths_path.open("rb") as handle:
        lengths = pickle.load(handle)
    rng = random.Random(SELECTION_SEED)
    records = []
    selected_videos: set[Path] = set()
    for _ in range(COUNTS["pusht"]):
        episode = rng.randint(0, len(lengths) - 1)
        max_offset = int(lengths[episode]) - (5 * GOAL_H + 1)
        if max_offset < 0:
            raise TargetMaterializationError("short PushT validation trajectory")
        offset = rng.randint(0, max_offset)
        goal = offset + 5 * GOAL_H
        video_path = root / "obses" / f"episode_{episode:03d}.mp4"
        selected_videos.add(video_path)
        reader = decord.VideoReader(str(video_path), num_threads=1)
        frames = reader.get_batch([offset, goal]).asnumpy()
        state = torch.cat((states[episode], velocities[episode]), dim=-1)
        raw_actions = actions[episode, offset:goal]
        records.append(
            {
                "source_episode": episode,
                "source_offset": offset,
                "goal_offset": goal,
                "obs_0": {
                    "visual": frames[0],
                    "proprio": state[offset, [0, 1, 5, 6]].numpy(),
                },
                "obs_g": {
                    "visual": frames[1],
                    "proprio": state[goal, [0, 1, 5, 6]].numpy(),
                },
                "state_0": state[offset].numpy(),
                "state_g": state[goal].numpy(),
                "normalized_actions": raw_actions.reshape(GOAL_H, 10).numpy(),
            }
        )
    manifest = _base_manifest(
        "pusht",
        [state_path, velocity_path, action_path, lengths_path, *sorted(selected_videos)],
        {"goal_source": "dset", "frameskip": 5, "goal_H": GOAL_H},
        records,
    )
    manifest["gt_actions"] = np.stack(
        [record["normalized_actions"] for record in records]
    )
    return manifest


def _wall(source_root: Path, code_root: Path) -> dict[str, Any]:
    if "env" not in sys.modules:
        package = types.ModuleType("env")
        package.__path__ = [str(code_root / "env")]
        sys.modules["env"] = package
    from env.wall.wall_env_wrapper import WallEnvWrapper

    root = source_root / "wall_single"
    state_path = root / "states.pth"
    action_path = root / "actions.pth"
    door_path = root / "door_locations.pth"
    wall_path = root / "wall_locations.pth"
    states = torch.load(state_path, map_location="cpu")
    doors = torch.load(door_path, map_location="cpu")
    walls = torch.load(wall_path, map_location="cpu")
    permutation = torch.randperm(len(states), generator=torch.Generator().manual_seed(42))
    validation = permutation[int(0.9 * len(states)) :].tolist()
    rng = random.Random(SELECTION_SEED)
    records = []
    for ordinal in range(COUNTS["wall"]):
        episode = validation[rng.randint(0, len(validation) - 1)]
        offset = rng.randint(0, int(states.shape[1]) - 2)
        eval_seed = SELECTION_SEED * ordinal + 1
        env = WallEnvWrapper(device="cpu")
        env.update_env(
            {
                "fix_door_location": doors[episode, offset],
                "fix_wall_location": walls[episode, offset],
            }
        )
        initial, goal = env.sample_random_init_goal_states(eval_seed)
        obs_0, prepared_0 = env.prepare(eval_seed, initial)
        obs_g, prepared_g = env.prepare(eval_seed, goal)
        records.append(
            {
                "source_episode": episode,
                "source_offset": offset,
                "eval_seed": eval_seed,
                "environment_info": {
                    "fix_door_location": float(doors[episode, offset]),
                    "fix_wall_location": float(walls[episode, offset]),
                },
                "obs_0": {key: np.asarray(value) for key, value in obs_0.items()},
                "obs_g": {key: np.asarray(value) for key, value in obs_g.items()},
                "state_0": np.asarray(prepared_0),
                "state_g": np.asarray(prepared_g),
            }
        )
        env.close()
    return _base_manifest(
        "wall",
        [state_path, action_path, door_path, wall_path],
        {
            "goal_source": "random_state",
            "goal_distance_model_steps": GOAL_H,
            "frameskip": 5,
            "goal_H": GOAL_H,
            "split_seed": 42,
            "split_fraction": 0.9,
        },
        records,
    )


def _transform_particles(
    state: np.ndarray, scale: float, theta_degrees: float, delta: float, rng: np.random.RandomState
) -> np.ndarray:
    result = np.array(state, copy=True)
    result[:, 0] *= scale
    result[:, 2] *= scale
    theta = math.radians(theta_degrees)
    rotation = np.array(
        [
            [math.cos(theta), 0.0, math.sin(theta)],
            [0.0, 1.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta)],
        ],
        dtype=result.dtype,
    )
    result[:, :3] = result[:, :3] @ rotation.T
    result[:, 0] += delta * rng.choice([-1, 1])
    result[:, 2] += delta * rng.choice([-1, 1])
    return result


def _affine_goal_image(
    image: np.ndarray, scale: float, theta_degrees: float, delta: float
) -> np.ndarray:
    from torchvision.transforms.functional import affine

    tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1)
    translated = int(round(delta * 12.0))
    result = affine(
        tensor,
        angle=-theta_degrees,
        translate=[translated, translated],
        scale=scale,
        shear=[0.0, 0.0],
    )
    return result.permute(1, 2, 0).numpy()


def _deformable(source_root: Path, environment: str) -> dict[str, Any]:
    root = source_root / "deformable" / environment
    state_path = root / "states.pth"
    action_path = root / "actions.pth"
    states = torch.load(state_path, map_location="cpu").float()
    permutation = torch.randperm(len(states), generator=torch.Generator().manual_seed(42))
    validation = permutation[int(0.9 * len(states)) :].tolist()
    chooser = random.Random(SELECTION_SEED)
    records = []
    selected_observations: set[Path] = set()
    for ordinal in range(COUNTS[environment]):
        episode = validation[chooser.randint(0, len(validation) - 1)]
        offset = chooser.randint(0, int(states.shape[1]) - 2)
        eval_seed = SELECTION_SEED * ordinal + 1
        rng = np.random.RandomState(eval_seed)
        initial = states[episode, offset].numpy()
        if environment == "rope":
            scale = 1.0
            theta = float(rng.uniform(0.0, 90.0))
            delta = float(rng.uniform(-1.0, 1.0))
        else:
            scale = float(rng.uniform(0.6, 0.9))
            theta = 0.0
            delta = float(rng.uniform(-1.0, 1.0))
        goal = _transform_particles(initial, scale, theta, delta, rng)
        obs_path = root / f"{episode:06d}" / "obses.pth"
        selected_observations.add(obs_path)
        observations = torch.load(obs_path, map_location="cpu")
        initial_image = observations[offset].numpy().astype(np.uint8)
        goal_image = _affine_goal_image(initial_image, scale, theta, delta)
        records.append(
            {
                "source_episode": episode,
                "source_offset": offset,
                "eval_seed": eval_seed,
                "goal_transform": {
                    "scale": scale,
                    "orientation_degrees": theta,
                    "displacement": delta,
                },
                "obs_0": {
                    "visual": initial_image,
                    "proprio": np.zeros(1, dtype=np.float32),
                },
                "obs_g": {
                    "visual": goal_image,
                    "proprio": np.zeros(1, dtype=np.float32),
                },
                "state_0": initial,
                "state_g": goal,
            }
        )
    return _base_manifest(
        environment,
        [state_path, action_path, *sorted(selected_observations)],
        {
            "goal_source": "random_state",
            "initial_source": "validation",
            "goal_shape": "rope" if environment == "rope" else "fixed_square",
            "goal_transform": (
                "random_orientation_and_displacement"
                if environment == "rope"
                else "random_location_and_scale"
            ),
            "goal_observation": "deterministic_source-view_affine_of_goal_state_transform",
            "frameskip": 1,
            "goal_H": GOAL_H,
            "split_seed": 42,
            "split_fraction": 0.9,
        },
        records,
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    manifests = {
        "pusht": _pusht(args.source_root),
        "wall": _wall(args.source_root, args.code_root),
        "rope": _deformable(args.source_root, "rope"),
        "granular": _deformable(args.source_root, "granular"),
    }
    receipt = {
        "schema": "dinocular.fixed-planning-target-materialization.v1",
        "selection_seed": SELECTION_SEED,
        "manifests": {},
    }
    for environment, manifest in manifests.items():
        environment_files = []
        for relative in (
            f"conf/env/{environment}.yaml",
            "env/deformable_env/FlexEnvWrapper.py"
            if environment in ("rope", "granular")
            else (
                "env/pusht/pusht_wrapper.py"
                if environment == "pusht"
                else "env/wall/wall_env_wrapper.py"
            ),
        ):
            path = args.code_root / relative
            environment_files.append({"path": str(path), "sha256": _sha256(path)})
        manifest["environment_files"] = environment_files
        path = args.out_root / FILENAMES[environment]
        _write_pickle_immutable(path, manifest)
        receipt["manifests"][environment] = {
            "path": str(path),
            "count": manifest["count"],
            "first_target_id": manifest["target_ids"][0],
            "last_target_id": manifest["target_ids"][-1],
            "sha256": _sha256(path),
        }
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    materialize(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
