#!/usr/bin/env python3
"""Minimum loader reproduction for the RGB/state/action dataset manifests.

Calls the actual frozen dataset loaders (``datasets.wall_dset`` and
``datasets.deformable_env_dset``) with the exact seed-1 recipe that the held-out
materializer uses (``tools/materialize_p3_heldout._datasets``) and confirms that
the loader-produced train/validation split, episode identity sets, and locked
training/validation window counts match the manifest objects emitted by
``tools/materialize_rgb_dataset_manifest.py``.

This is the minimum reproduction required to prove the manifest's episode/split
identities and the locked window counts (Wall 70,848 train; Rope/Granular 17,100
train each). It loads only tensor metadata and rebuilds slice indices; it never
reads RGB frames and never touches depth caches.

Run it on Marvin inside the project container so the loader modules and their
dependencies (torch/decord/einops/yaml) match the frozen source commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

MANIFEST_SCHEMA = "dino-wm.rgb-dataset-manifest.v1"
REPRODUCTION_SCHEMA = "dino-wm.rgb-dataset-manifest-reproduction.v1"
SOURCE_COMMIT = "00e46eb0ded3b8c379ec6ebdb74c8991604f4b8f"

LOCKED_TRAIN_WINDOW_COUNTS = {"wall": 70848, "rope": 17100, "granular": 17100}
ENVIRONMENT_OBJECT = {"wall": None, "rope": "rope", "granular": "granular"}


class ReproductionError(RuntimeError):
    """A manifest/loader reproduction invariant is invalid."""


def _load_manifest(path: Path, environment: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ReproductionError(f"manifest is absent: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReproductionError(f"manifest is invalid JSON: {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ReproductionError(f"manifest root is not an object: {path}")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("environment") != environment
        or manifest.get("source_commit") != SOURCE_COMMIT
    ):
        raise ReproductionError(
            f"manifest schema/environment/source commit differs for {environment}: {path}"
        )
    return manifest


def _split_from_loader(
    datasets: Mapping[str, Any], environment: str
) -> tuple[set[int], set[int]]:
    """Recover the global train/valid episode sets from the loaded slices."""

    train_slicer = datasets["train"]
    valid_slicer = datasets["valid"]
    train_base = getattr(train_slicer, "dataset", None)
    valid_base = getattr(valid_slicer, "dataset", None)
    train_indices = getattr(train_base, "indices", None)
    valid_indices = getattr(valid_base, "indices", None)
    if train_indices is None or valid_indices is None:
        raise ReproductionError(
            f"{environment} loader did not expose a random TrajSubset split"
        )
    return set(int(i) for i in train_indices), set(int(i) for i in valid_indices)


def _load_with_frozen_recipe(environment: str, data_root: Path) -> Mapping[str, Any]:
    """Mirror ``tools/materialize_p3_heldout._datasets`` exactly for wall/rope/granular."""

    from datasets.deformable_env_dset import load_deformable_dset_slice_train_val
    from datasets.wall_dset import load_wall_slice_train_val

    common = {
        "n_rollout": None,
        "transform": None,
        "num_pred": 1,
    }
    if environment == "wall":
        datasets, _ = load_wall_slice_train_val(
            data_path=str(data_root / "wall_single"),
            normalize_action=False,
            split_ratio=0.9,
            split_mode="random",
            num_hist=1,
            frameskip=5,
            **common,
        )
    else:
        datasets, _ = load_deformable_dset_slice_train_val(
            data_path=str(data_root / "deformable"),
            object_name=environment,
            normalize_action=False,
            split_ratio=0.9,
            num_hist=1,
            frameskip=1,
            **common,
        )
    return datasets


def reproduce_environment(
    *, environment: str, data_root: Path, manifest_path: Path
) -> dict[str, Any]:
    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)

    manifest = _load_manifest(manifest_path, environment)
    expected_train_eps = set(manifest["split"]["train_episodes"])
    expected_valid_eps = set(manifest["split"]["valid_episodes"])
    expected_train_windows = int(manifest["window_recipe"]["train_window_count"])
    expected_valid_windows = int(manifest["window_recipe"]["valid_window_count"])
    locked_train_windows = LOCKED_TRAIN_WINDOW_COUNTS[environment]

    datasets = _load_with_frozen_recipe(environment, data_root)
    train_window_count = int(len(datasets["train"]))
    valid_window_count = int(len(datasets["valid"]))
    train_eps, valid_eps = _split_from_loader(datasets, environment)

    checks: dict[str, Any] = {
        "train_window_count_match_manifest": train_window_count
        == expected_train_windows,
        "valid_window_count_match_manifest": valid_window_count
        == expected_valid_windows,
        "train_window_count_match_locked_recipe": train_window_count
        == locked_train_windows,
        "train_episode_set_match_manifest": train_eps == expected_train_eps,
        "valid_episode_set_match_manifest": valid_eps == expected_valid_eps,
        "train_valid_episodes_disjoint": not (train_eps & valid_eps),
        "train_valid_episodes_partition": (train_eps | valid_eps)
        == set(range(manifest["trajectory"]["episode_count"])),
    }
    for name, passed in checks.items():
        if not passed:
            raise ReproductionError(
                f"{environment} reproduction check failed: {name} "
                f"(train_windows={train_window_count}, valid_windows={valid_window_count}, "
                f"train_eps={len(train_eps)}, valid_eps={len(valid_eps)})"
            )

    return {
        "environment": environment,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "data_root": str(data_root),
        "loader_train_window_count": train_window_count,
        "loader_valid_window_count": valid_window_count,
        "locked_train_window_count": locked_train_windows,
        "loader_train_episode_count": len(train_eps),
        "loader_valid_episode_count": len(valid_eps),
        "checks": {name: bool(passed) for name, passed in checks.items()},
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--environments",
        default="wall,rope,granular",
        help="Comma-separated subset of wall/rope/granular to reproduce",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    environments = [name.strip() for name in args.environments.split(",") if name.strip()]
    if not environments or any(name not in LOCKED_TRAIN_WINDOW_COUNTS for name in environments):
        raise ReproductionError("--environments must be a subset of wall,rope,granular")
    results = [
        reproduce_environment(
            environment=name,
            data_root=args.data_root,
            manifest_path=args.manifest_dir / f"dataset_manifest_{name}.json",
        )
        for name in environments
    ]
    payload = {
        "schema": REPRODUCTION_SCHEMA,
        "state": "PASS",
        "manifest_schema": MANIFEST_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "data_root": str(args.data_root.resolve()),
        "environments": environments,
        "results": {item["environment"]: item for item in results},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReproductionError as exc:
        print(f"DATASET MANIFEST REPRODUCTION FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
