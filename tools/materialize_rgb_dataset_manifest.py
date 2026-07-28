#!/usr/bin/env python3
"""Deterministic read-only RGB/state/action dataset manifest materializer.

Materializes exactly the immutable dataset manifests required to make the Wall,
Rope, and Granular DINOv2 seed-1 cells countable. The manifests enumerate every
file the current dataset loaders can read (``datasets.wall_dset.WallDataset`` and
``datasets.deformable_env_dset.DeformDataset``) and record, per file, its
canonical Marvin path, repository-relative path, loader file type, size in
bytes, file mode, SHA-256, environment, split, and episode identity.

The materializer is fail-closed and read-only. It rejects:

* aliases -- any symlink in a traversed path component or file;
* missing files -- any loader-readable source that is absent;
* duplicate identities -- episode indices that collide or episode file sets with
  gaps, extras, or repeated indices;
* data changes during traversal -- any change to a file's size, mtime, inode, or
  mode between discovery, hashing, and a final re-enumeration.

Only loader-readable sources are included. Depth caches, ``.h5`` camera dumps,
``property_params.pkl``, ``.gitattributes`` and dataset config pickles are
intentionally out of scope because the RGB/state/action loaders never read them.

The split mirrors the frozen seed-1 recipe used by the held-out materializer:
``split_traj_datasets`` with ``train_fraction=0.9`` and
``torch.Generator().manual_seed(42)`` (see ``datasets/traj_dset.py``). Per-episode
observation files are attributed to the ``train`` or ``valid`` partition by the
global episode index; shared metadata tensors (``states.pth``/``actions.pth``
and the Wall location tensors) are attributed to ``shared``.

torch is imported lazily so the module compiles without a torch installation;
torch is only required to read tensor shapes and reproduce the exact seeded
``randperm`` split that the loaders use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

MANIFEST_SCHEMA = "dino-wm.rgb-dataset-manifest.v1"
MATERIALIZER_NAME = "tools/materialize_rgb_dataset_manifest.py"
MARVIN_PROJECT_ROOT = "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm"

SPLIT_MODE = "random"
SPLIT_FRACTION = 0.9
SPLIT_SEED = 42
NUM_PRED = 1

# Locked seed-1 recipes (conf/study_matrix.yaml). The held-out materializer and
# the loader reproduction both consume exactly these values.
ENVIRONMENT_RECIPES: dict[str, dict[str, Any]] = {
    "wall": {
        "loader_module": "datasets.wall_dset",
        "loader_class": "WallDataset",
        "loader_function": "load_wall_slice_train_val",
        "data_subdir": "wall_single",
        "object_name": None,
        "num_hist": 1,
        "frameskip": 5,
        "shared_files": {
            "states.pth": "states",
            "actions.pth": "actions",
            "door_locations.pth": "door_locations",
            "wall_locations.pth": "wall_locations",
        },
        "episode_pattern": re.compile(r"^episode_(\d+)\.pth$"),
        "episode_file": "episode_{idx:03d}.pth",
        "episode_subdir": "obses",
        "episode_index_format": "03d",
        "trajectory_length_source": "actions",
        "trajectory_length_axis": 1,
    },
    "rope": {
        "loader_module": "datasets.deformable_env_dset",
        "loader_class": "DeformDataset",
        "loader_function": "load_deformable_dset_slice_train_val",
        "data_subdir": "deformable/rope",
        "object_name": "rope",
        "num_hist": 1,
        "frameskip": 1,
        "shared_files": {
            "states.pth": "states",
            "actions.pth": "actions",
        },
        "episode_pattern": re.compile(r"^(\d{6})$"),
        "episode_file": "obses.pth",
        "episode_subdir": "{idx:06d}",
        "episode_index_format": "06d",
        "trajectory_length_source": "states",
        "trajectory_length_axis": 1,
    },
    "granular": {
        "loader_module": "datasets.deformable_env_dset",
        "loader_class": "DeformDataset",
        "loader_function": "load_deformable_dset_slice_train_val",
        "data_subdir": "deformable/granular",
        "object_name": "granular",
        "num_hist": 1,
        "frameskip": 1,
        "shared_files": {
            "states.pth": "states",
            "actions.pth": "actions",
        },
        "episode_pattern": re.compile(r"^(\d{6})$"),
        "episode_file": "obses.pth",
        "episode_subdir": "{idx:06d}",
        "episode_index_format": "06d",
        "trajectory_length_source": "states",
        "trajectory_length_axis": 1,
    },
}


class ManifestError(RuntimeError):
    """A closed manifest contract, path, identity, or hash is invalid."""


def is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def require_marvin_path(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"{label} is not a closed absolute path: {path}")
    if "$" in str(path) or "~" in str(path):
        raise ManifestError(f"{label} contains an unresolved component: {path}")
    if str(path) != MARVIN_PROJECT_ROOT and not str(path).startswith(
        MARVIN_PROJECT_ROOT + "/"
    ):
        raise ManifestError(f"{label} must live under the Marvin project root: {path}")
    return path


def _lstat_regular(path: Path, label: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ManifestError(f"{label} is absent: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ManifestError(f"{label} is a symlink and is rejected as an alias: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise ManifestError(f"{label} is not a regular file: {path}")
    return info


def _assert_path_components_real(path: Path, label: str) -> None:
    """Reject any symlink component in a closed absolute path (alias guard)."""

    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise ManifestError(f"{label} component is absent: {current}") from exc
        if stat.S_ISLNK(mode):
            raise ManifestError(
                f"{label} path contains a symlink component: {current}"
            )


def _require_closed_directory(path: Path, label: str) -> None:
    _assert_path_components_real(path, label)
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ManifestError(f"{label} directory is absent: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ManifestError(f"{label} directory is a symlink: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise ManifestError(f"{label} is not a directory: {path}")


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_size, info.st_mtime_ns, info.st_ino, info.st_mode)


def _hash_with_mutation_guard(
    path: Path, label: str, before: os.stat_result
) -> str:
    after_lstat = _lstat_regular(path, label)
    if _identity(after_lstat) != _identity(before):
        raise ManifestError(
            f"{label} changed between discovery and hashing: {path}"
        )
    # Hash through an open descriptor and fstat it in-flight so a writer that
    # truncates or extends the file mid-read is detected rather than silently
    # producing a stable-but-wrong digest.
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
        open_info = os.fstat(handle.fileno())
    try:
        post_lstat = _lstat_regular(path, label)
    except FileNotFoundError as exc:
        raise ManifestError(f"{label} vanished during hashing: {path}") from exc
    if open_info.st_size != before.st_size or post_lstat.st_size != before.st_size:
        raise ManifestError(f"{label} size changed during hashing: {path}")
    if _identity(post_lstat) != _identity(before):
        raise ManifestError(f"{label} changed during hashing: {path}")
    return digest.hexdigest()


def _load_episode_count(
    environment: str, env_root: Path, recipe: Mapping[str, Any]
) -> tuple[int, int]:
    """Return (episode_count, trajectory_length) from the loader's own sources.

    ``episode_count`` is ``len(states)`` exactly as the loader computes it; the
    trajectory length is read from the same tensor the loader uses
    (``actions.shape[1]`` for Wall, ``states.shape[1]`` for deformable).
    """

    import torch  # local import: only required at materialization time

    states_path = env_root / "states.pth"
    actions_path = env_root / "actions.pth"
    _assert_path_components_real(states_path, f"{environment} states")
    _assert_path_components_real(actions_path, f"{environment} actions")
    states_tensor = torch.load(states_path, map_location="cpu")
    if not isinstance(states_tensor, type(torch.empty(0))):
        raise ManifestError(f"{environment} states.pth is not a torch tensor")
    episode_count = int(states_tensor.shape[0])
    if recipe["trajectory_length_source"] == "actions":
        actions_tensor = torch.load(actions_path, map_location="cpu")
        trajectory_length = int(actions_tensor.shape[1])
    else:
        trajectory_length = int(states_tensor.shape[1])
    if episode_count <= 0 or trajectory_length <= 0:
        raise ManifestError(
            f"{environment} reported non-positive episode count or trajectory length"
        )
    return episode_count, trajectory_length


def _seeded_split(episode_count: int) -> tuple[list[int], list[int]]:
    """Reproduce ``split_traj_datasets`` with the frozen seed-1 recipe."""

    import torch  # local import: mirrors the loader's generator exactly

    train_len = int(SPLIT_FRACTION * episode_count)
    lengths = [train_len, episode_count - train_len]
    if sum(lengths) != episode_count or min(lengths) <= 0:
        raise ManifestError("seeded split lengths do not cover the dataset")
    permutation = torch.randperm(
        episode_count, generator=torch.Generator().manual_seed(SPLIT_SEED)
    ).tolist()
    train_episodes = sorted(permutation[:train_len])
    valid_episodes = sorted(permutation[train_len:episode_count])
    if (
        sorted(permutation) != list(range(episode_count))
        or len(set(train_episodes)) != len(train_episodes)
        or len(set(valid_episodes)) != len(valid_episodes)
        or set(train_episodes) & set(valid_episodes)
    ):
        raise ManifestError("seeded split produced duplicate or leaking episodes")
    return train_episodes, valid_episodes


def _discover_episode_indices(
    environment: str, env_root: Path, recipe: Mapping[str, Any], episode_count: int
) -> dict[int, Path]:
    """Discover the closed per-episode loader-readable file set."""

    if environment == "wall":
        container = env_root / "obses"
        _require_closed_directory(container, f"{environment} obses directory")
        present: dict[int, Path] = {}
        for name in sorted(os.listdir(container)):
            match = recipe["episode_pattern"].match(name)
            if not match:
                continue
            index = int(match.group(1))
            if index in present:
                raise ManifestError(
                    f"{environment} obses directory duplicates episode index {index}"
                )
            present[index] = container / name
    else:
        present = {}
        for name in sorted(os.listdir(env_root)):
            match = recipe["episode_pattern"].match(name)
            if not match:
                continue
            index = int(match.group(1))
            if index in present:
                raise ManifestError(
                    f"{environment} episode directory duplicates index {index}"
                )
            present[index] = env_root / name / recipe["episode_file"]
    expected = set(range(episode_count))
    actual = set(present)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ManifestError(
            f"{environment} loader-readable episode set differs from "
            f"len(states): missing={missing[:8]} extra={extra[:8]}"
        )
    return present


def _build_file_records(
    environment: str,
    data_root: Path,
    env_root: Path,
    recipe: Mapping[str, Any],
    episode_count: int,
    split_lookup: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Two-pass enumeration + hashing with a final re-enumeration guard."""

    records: list[dict[str, Any]] = []
    snapshots: dict[Path, tuple[str, os.stat_result]] = {}

    def snapshot(label: str, relative: str, file_type: str, episode: int | None) -> None:
        path = env_root / relative
        _assert_path_components_real(path, f"{environment} {label}")
        info = _lstat_regular(path, f"{environment} {label}")
        if path in snapshots:
            raise ManifestError(f"{environment} duplicate loader-readable path: {path}")
        snapshots[path] = (file_type, info)
        records.append(
            {"_path": path, "_relative": relative, "_type": file_type, "_episode": episode}
        )

    for name, file_type in sorted(recipe["shared_files"].items()):
        snapshot(name + " shared tensor", name, file_type, None)

    episode_paths = _discover_episode_indices(
        environment, env_root, recipe, episode_count
    )
    for index in sorted(episode_paths):
        if environment == "wall":
            relative = f"obses/episode_{index:03d}.pth"
        else:
            relative = f"{index:06d}/obses.pth"
        snapshot(
            f"episode {index} observation tensor",
            relative,
            "obses",
            index,
        )
        # Confirm the discovered path equals the closed relative path.
        if episode_paths[index] != env_root / relative:
            raise ManifestError(
                f"{environment} episode {index} discovery path mismatch"
            )

    # Pass B: hash each file with mutation guards.
    finalized: list[dict[str, Any]] = []
    for entry in records:
        path: Path = entry["_path"]
        relative: str = entry["_relative"]
        file_type: str = entry["_type"]
        episode: int | None = entry["_episode"]
        info = snapshots[path][1]
        digest = _hash_with_mutation_guard(path, f"{environment} {relative}", info)
        if episode is None:
            split = "shared"
        else:
            split = split_lookup[episode]
        finalized.append(
            {
                "relative_path": str(Path(recipe["data_subdir"]) / relative),
                "canonical_path": str(path),
                "type": file_type,
                "episode": episode,
                "split": split,
                "bytes": int(info.st_size),
                "mode": format(info.st_mode, "06o"),
                "sha256": digest,
            }
        )

    # Pass C: final re-enumeration; identities must be unchanged and the closed
    # file set must be byte-for-byte the same.
    for path, (_file_type, info) in snapshots.items():
        current = _lstat_regular(path, f"{environment} re-enumeration {path.name}")
        if _identity(current) != _identity(info):
            raise ManifestError(
                f"{environment} file changed during traversal: {path}"
            )
    rediscovered = _discover_episode_indices(
        environment, env_root, recipe, episode_count
    )
    if set(rediscovered) != set(episode_paths):
        raise ManifestError(
            f"{environment} episode set changed during traversal"
        )

    finalized.sort(key=lambda item: item["relative_path"])
    seen = {item["relative_path"] for item in finalized}
    if len(seen) != len(finalized):
        raise ManifestError(f"{environment} manifest contains duplicate relative paths")
    return finalized


def materialize_manifest(
    *,
    environment: str,
    data_root: Path,
    source_commit: str,
    out_path: Path,
) -> dict[str, Any]:
    if environment not in ENVIRONMENT_RECIPES:
        raise ManifestError(f"unsupported environment: {environment}")
    if not is_lower_hex(source_commit, 40):
        raise ManifestError("source commit must be a 40-character lower-hex string")
    recipe = ENVIRONMENT_RECIPES[environment]
    data_root = require_marvin_path(data_root, "data root")
    _require_closed_directory(data_root, "data root")
    env_root = require_marvin_path(
        data_root / recipe["data_subdir"], f"{environment} data subtree"
    )
    _require_closed_directory(env_root, f"{environment} data subtree")

    episode_count, trajectory_length = _load_episode_count(
        environment, env_root, recipe
    )
    train_episodes, valid_episodes = _seeded_split(episode_count)
    split_lookup: dict[int, str] = {}
    for idx in train_episodes:
        split_lookup[idx] = "train"
    for idx in valid_episodes:
        split_lookup[idx] = "valid"

    num_hist = int(recipe["num_hist"])
    frameskip = int(recipe["frameskip"])
    num_frames = num_hist + NUM_PRED
    per_episode_windows = trajectory_length - num_frames * frameskip + 1
    if per_episode_windows <= 0:
        raise ManifestError(
            f"{environment} trajectory too short for the locked window recipe"
        )
    train_window_count = per_episode_windows * len(train_episodes)
    valid_window_count = per_episode_windows * len(valid_episodes)

    files = _build_file_records(
        environment,
        data_root,
        env_root,
        recipe,
        episode_count,
        split_lookup,
    )
    total_bytes = sum(int(item["bytes"]) for item in files)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "environment": environment,
        "source_commit": source_commit,
        "materializer": {
            "name": MATERIALIZER_NAME,
            "read_only": True,
            "rejects": [
                "aliases",
                "missing_files",
                "duplicate_identities",
                "data_changes_during_traversal",
            ],
        },
        "data_root": str(data_root),
        "data_subdir": recipe["data_subdir"],
        "object_name": recipe["object_name"],
        "loader": {
            "module": recipe["loader_module"],
            "class": recipe["loader_class"],
            "function": recipe["loader_function"],
        },
        "trajectory": {
            "episode_count": episode_count,
            "trajectory_length": trajectory_length,
            "episode_index_format": recipe["episode_index_format"],
        },
        "split": {
            "mode": SPLIT_MODE,
            "train_fraction": SPLIT_FRACTION,
            "random_seed": SPLIT_SEED,
            "generator": "torch.Generator().manual_seed(seed)",
            "train_episode_count": len(train_episodes),
            "valid_episode_count": len(valid_episodes),
            "train_episodes": train_episodes,
            "valid_episodes": valid_episodes,
        },
        "window_recipe": {
            "num_hist": num_hist,
            "num_pred": NUM_PRED,
            "num_frames": num_frames,
            "frameskip": frameskip,
            "per_episode_windows": per_episode_windows,
            "train_window_count": train_window_count,
            "valid_window_count": valid_window_count,
        },
        "files": files,
        "file_count": len(files),
        "total_bytes": total_bytes,
    }
    text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Marvin released raw data root (e.g. .../dinocular-wm/data/raw)",
    )
    parser.add_argument(
        "--environment",
        choices=sorted(ENVIRONMENT_RECIPES),
        required=True,
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="40-character lower-hex source commit the manifest is bound to",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = materialize_manifest(
        environment=args.environment,
        data_root=args.data_root,
        source_commit=args.source_commit,
        out_path=args.out,
    )
    summary = {
        "schema": MANIFEST_SCHEMA,
        "environment": manifest["environment"],
        "out_path": str(Path(args.out).resolve()),
        "sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "train_window_count": manifest["window_recipe"]["train_window_count"],
        "valid_window_count": manifest["window_recipe"]["valid_window_count"],
        "train_episode_count": manifest["split"]["train_episode_count"],
        "valid_episode_count": manifest["split"]["valid_episode_count"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as exc:
        print(f"DATASET MANIFEST CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
