#!/usr/bin/env python3
"""Extract released Rope/Granular PyFlex depth without simulator replay.

This is an explicit upper-bound ablation producer.  It reads the depth already
stored in the released per-frame HDF5 files, proves that each selected HDF5
record is exactly aligned with the released RGB and particle state, converts
the stored uint16 millimetres to metres, and writes the existing immutable LMDB
wire format.  It does not change any training configuration or depth default.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from tools.precompute_depth import (
        CALIBRATION_PER_ENV,
        CALIBRATION_STRIDE,
        MAP_SIZE,
        OUTPUT_SHAPE,
        ZSTD_LEVEL,
        ContractError,
        Trajectory,
        _source_index_hash,
        atomic_write_json,
        canonical_json_bytes,
        compute_global_calibration,
        encode_depth_value,
        enumerate_environment,
        normalize_depth,
        select_calibration_keys,
        sha256_bytes,
        sha256_file,
        streaming_chunks,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from precompute_depth import (  # type: ignore[no-redef]
        CALIBRATION_PER_ENV,
        CALIBRATION_STRIDE,
        MAP_SIZE,
        OUTPUT_SHAPE,
        ZSTD_LEVEL,
        ContractError,
        Trajectory,
        _source_index_hash,
        atomic_write_json,
        canonical_json_bytes,
        compute_global_calibration,
        encode_depth_value,
        enumerate_environment,
        normalize_depth,
        select_calibration_keys,
        sha256_bytes,
        sha256_file,
        streaming_chunks,
    )


SUPPORTED_ENVIRONMENTS = ("rope", "granular")
OUTPUT_FRAMES_PER_EPISODE = 20
RELEASED_EPISODES = 1000
HDF5_FILES_PER_EPISODE = 21
EXPECTED_KEYS_PER_ENVIRONMENT = RELEASED_EPISODES * OUTPUT_FRAMES_PER_EPISODE
CAMERA_INDEX = 1
DEPTH_DATASET = f"observations/depth/cam_{CAMERA_INDEX}"
RGB_DATASET = f"observations/color/cam_{CAMERA_INDEX}"
POSITIONS_DATASET = "positions"
MILLIMETRES_PER_METRE = 1000

RELEASE_HASHES = {
    "rope": {
        "states.pth": "cabaa66a583b89416339b2430a7da9fbac73b1f17ed98dfc57f9cb1c858d4c3a",
        "actions.pth": "fd73ab44cc933ed2c68aecd2e3c70efdc639c4284de3ad328b7217fb5ec5d711",
    },
    "granular": {
        "states.pth": "b8a56120e71471d7621c63d5700ee3bc1292324a53177843d7c63f1bd157e6a2",
        "actions.pth": "ce7567cf1653e217f42aaed34471cea9bbed1c74085312db10c6590030da8f3f",
    },
}
CAMERA_HASHES = {
    "intrinsic.npy": "b40bfb9115a92b6d59ccafa31a643d3437485931bb8f254b7e2941d121aefeb6",
    "extrinsic.npy": "4b7170f398a9fc0a3e6914f5daf2fcc43e73e11de0654c7563f432c694b1bd85",
}

EXPECTED_HDF5_DATASETS = {
    "action",
    "eef_states",
    "info/n_cams",
    "info/n_particles",
    "info/timestamp",
    *(f"observations/color/cam_{index}" for index in range(4)),
    *(f"observations/depth/cam_{index}" for index in range(4)),
    "positions",
}


@dataclasses.dataclass(frozen=True)
class ReleaseContract:
    episodes: int
    output_frames: int
    hdf5_files: int
    calibration_frames: int
    expected_release_hashes: Mapping[str, str] | None
    expected_camera_hashes: Mapping[str, str] | None

    @property
    def expected_keys(self) -> int:
        return self.episodes * self.output_frames


def released_contract(environment: str) -> ReleaseContract:
    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ContractError(f"native depth is unsupported for {environment!r}")
    return ReleaseContract(
        episodes=RELEASED_EPISODES,
        output_frames=OUTPUT_FRAMES_PER_EPISODE,
        hdf5_files=HDF5_FILES_PER_EPISODE,
        calibration_frames=CALIBRATION_PER_ENV,
        expected_release_hashes=RELEASE_HASHES[environment],
        expected_camera_hashes=CAMERA_HASHES,
    )


def _torch_load(path: Path, *, mmap: bool = False) -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - pinned-container dependency
        raise ContractError("the pinned image must provide PyTorch") from exc
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except Exception as exc:
        raise ContractError(f"cannot load released tensor {path}: {exc}") from exc


def decode_uint16_millimetres(depth_mm: np.ndarray) -> np.ndarray:
    """Decode the producer's uint16 millimetre storage as float32 metres."""

    value = np.asarray(depth_mm)
    if value.dtype != np.dtype("uint16"):
        raise ContractError(f"native depth must be uint16 millimetres, got {value.dtype}")
    if value.shape != OUTPUT_SHAPE:
        raise ContractError(
            f"native depth has shape {value.shape}, expected {OUTPUT_SHAPE}"
        )
    return value.astype(np.float32) / np.float32(MILLIMETRES_PER_METRE)


def _dataset_names(handle: Any) -> set[str]:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - pinned-container dependency
        raise ContractError("the pinned image must provide h5py") from exc
    names: set[str] = set()

    def visitor(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            names.add(name)

    handle.visititems(visitor)
    return names


def _require_hdf5_schema(
    handle: Any,
    *,
    path: Path,
    particle_count: int,
) -> None:
    names = _dataset_names(handle)
    if names != EXPECTED_HDF5_DATASETS:
        missing = sorted(EXPECTED_HDF5_DATASETS - names)
        extra = sorted(names - EXPECTED_HDF5_DATASETS)
        raise ContractError(
            f"{path}: HDF5 dataset set differs; missing={missing}, extra={extra}"
        )
    expected = {
        "action": ((4,), np.dtype("float64")),
        "eef_states": ((1, 1, 14), np.dtype("float64")),
        "info/n_cams": ((), np.dtype("int64")),
        "info/n_particles": ((), np.dtype("int64")),
        "info/timestamp": ((), np.dtype("int64")),
        "positions": ((1, particle_count, 4), np.dtype("float32")),
    }
    for index in range(4):
        expected[f"observations/color/cam_{index}"] = (
            (1, *OUTPUT_SHAPE, 3),
            np.dtype("float32"),
        )
        expected[f"observations/depth/cam_{index}"] = (
            (1, *OUTPUT_SHAPE),
            np.dtype("uint16"),
        )
    for name, (shape, dtype) in expected.items():
        dataset = handle[name]
        if dataset.shape != shape or dataset.dtype != dtype:
            raise ContractError(
                f"{path}::{name} has {dataset.shape}/{dataset.dtype}, "
                f"expected {shape}/{dtype}"
            )
    if int(handle["info/n_cams"][()]) != 4:
        raise ContractError(f"{path}: info/n_cams is not exactly 4")
    if int(handle["info/n_particles"][()]) != particle_count:
        raise ContractError(f"{path}: info/n_particles differs from released state")


def _environment_root(root: Path, environment: str) -> Path:
    return root.resolve() / "deformable" / environment


def _require_exact_inventory(
    root: Path,
    environment: str,
    contract: ReleaseContract,
) -> Path:
    base = _environment_root(root, environment)
    if not base.is_dir():
        raise ContractError(f"missing released {environment} directory: {base}")
    expected_entries = {
        "actions.pth",
        "states.pth",
        "cameras",
        *(f"{episode:06d}" for episode in range(contract.episodes)),
    }
    actual_entries = {entry.name for entry in base.iterdir()}
    if actual_entries != expected_entries:
        missing = sorted(expected_entries - actual_entries)
        extra = sorted(actual_entries - expected_entries)
        raise ContractError(
            f"{base}: released inventory differs; missing={missing[:5]}, extra={extra[:5]}"
        )
    expected_episode_entries = {
        "obses.pth",
        "property_params.pkl",
        *(f"{frame:02d}.h5" for frame in range(contract.hdf5_files)),
    }
    for episode in range(contract.episodes):
        episode_root = base / f"{episode:06d}"
        actual = {entry.name for entry in episode_root.iterdir()}
        if actual != expected_episode_entries:
            missing = sorted(expected_episode_entries - actual)
            extra = sorted(actual - expected_episode_entries)
            raise ContractError(
                f"{episode_root}: episode inventory differs; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    camera_root = base / "cameras"
    actual_camera_entries = {entry.name for entry in camera_root.iterdir()}
    if actual_camera_entries != set(CAMERA_HASHES):
        raise ContractError(
            f"{camera_root}: camera inventory differs: {sorted(actual_camera_entries)}"
        )
    return base


def _hash_and_require(path: Path, expected: str | None, label: str) -> str:
    if not path.is_file():
        raise ContractError(f"missing {label}: {path}")
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ContractError(
            f"{label} SHA-256 differs: expected {expected}, got {digest}"
        )
    return digest


@dataclasses.dataclass
class InspectedRelease:
    root: Path
    base: Path
    environment: str
    contract: ReleaseContract
    trajectories: list[Trajectory]
    states: Any
    actions: Any
    particle_count: int
    global_hashes: dict[str, str]
    camera_records: dict[str, dict[str, Any]]


def inspect_release(
    root: Path,
    environment: str,
    contract: ReleaseContract | None = None,
) -> InspectedRelease:
    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ContractError(f"native depth is unsupported for {environment!r}")
    contract = contract or released_contract(environment)
    base = _require_exact_inventory(root, environment, contract)
    expected_globals = contract.expected_release_hashes or {}
    global_hashes = {
        name: _hash_and_require(base / name, expected_globals.get(name), name)
        for name in ("states.pth", "actions.pth")
    }
    states = _torch_load(base / "states.pth", mmap=True)
    actions = _torch_load(base / "actions.pth", mmap=True)
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ContractError("the pinned image must provide PyTorch") from exc
    if not isinstance(states, torch.Tensor) or states.dtype != torch.float32:
        raise ContractError("states.pth must be a torch.float32 tensor")
    if not isinstance(actions, torch.Tensor) or actions.dtype != torch.float64:
        raise ContractError("actions.pth must be a torch.float64 tensor")
    if states.ndim != 4 or tuple(states.shape[:2]) != (
        contract.episodes,
        contract.output_frames,
    ) or int(states.shape[-1]) != 4:
        raise ContractError(
            f"states.pth shape {tuple(states.shape)} does not satisfy "
            f"({contract.episodes},{contract.output_frames},P,4)"
        )
    if tuple(actions.shape) != (
        contract.episodes,
        contract.output_frames,
        4,
    ):
        raise ContractError(
            f"actions.pth shape {tuple(actions.shape)} does not satisfy "
            f"({contract.episodes},{contract.output_frames},4)"
        )
    if not bool(torch.isfinite(states).all()) or not bool(torch.isfinite(actions).all()):
        raise ContractError("released states/actions contain non-finite values")

    trajectories = enumerate_environment(root, environment)
    if len(trajectories) != contract.episodes:
        raise ContractError(
            f"{environment} has {len(trajectories)} trajectories, "
            f"expected {contract.episodes}"
        )
    if {trajectory.episode for trajectory in trajectories} != set(
        range(contract.episodes)
    ):
        raise ContractError(f"{environment} episode index is not exact and contiguous")
    if any(trajectory.frame_count != contract.output_frames for trajectory in trajectories):
        raise ContractError(
            f"{environment} does not have exactly {contract.output_frames} frames per episode"
        )

    expected_camera_hashes = contract.expected_camera_hashes or {}
    camera_records: dict[str, dict[str, Any]] = {}
    for name, shape in (("intrinsic.npy", (4, 4)), ("extrinsic.npy", (4, 4, 4))):
        path = base / "cameras" / name
        digest = _hash_and_require(path, expected_camera_hashes.get(name), name)
        value = np.load(path, allow_pickle=False)
        if value.shape != shape or value.dtype != np.dtype("float64"):
            raise ContractError(
                f"{path} has {value.shape}/{value.dtype}, expected {shape}/float64"
            )
        if not np.isfinite(value).all():
            raise ContractError(f"{path} contains non-finite values")
        camera_records[name] = {
            "sha256": digest,
            "shape": list(shape),
            "dtype": "float64",
        }
    return InspectedRelease(
        root=root.resolve(),
        base=base,
        environment=environment,
        contract=contract,
        trajectories=trajectories,
        states=states,
        actions=actions,
        particle_count=int(states.shape[2]),
        global_hashes=global_hashes,
        camera_records=camera_records,
    )


def _load_observations(release: InspectedRelease, trajectory: Trajectory) -> np.ndarray:
    value = _torch_load(trajectory.source_path)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    observations = np.asarray(value)
    expected = (release.contract.output_frames, *OUTPUT_SHAPE, 3)
    if observations.shape != expected or observations.dtype != np.dtype("float32"):
        raise ContractError(
            f"{trajectory.source_path} has {observations.shape}/{observations.dtype}, "
            f"expected {expected}/float32"
        )
    if not np.isfinite(observations).all():
        raise ContractError(f"{trajectory.source_path} contains non-finite RGB")
    return observations


def _read_aligned_depth(
    release: InspectedRelease,
    trajectory: Trajectory,
    observations: np.ndarray,
    frame: int,
    *,
    validate_schema: bool = True,
) -> np.ndarray:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - pinned-container dependency
        raise ContractError("the pinned image must provide h5py") from exc
    path = release.base / f"{trajectory.episode:06d}" / f"{frame:02d}.h5"
    try:
        with h5py.File(path, "r") as handle:
            if validate_schema:
                _require_hdf5_schema(
                    handle, path=path, particle_count=release.particle_count
                )
            hdf5_rgb = np.asarray(handle[RGB_DATASET][0])
            if not np.array_equal(observations[frame], hdf5_rgb):
                raise ContractError(
                    f"{path}: cam_1 RGB is not exactly aligned with obses.pth frame {frame}"
                )
            hdf5_state = np.asarray(handle[POSITIONS_DATASET][0])
            released_state = (
                release.states[trajectory.episode, frame].detach().cpu().numpy()
            )
            if not np.array_equal(released_state, hdf5_state):
                raise ContractError(
                    f"{path}: positions are not exactly aligned with states.pth "
                    f"episode {trajectory.episode} frame {frame}"
                )
            return decode_uint16_millimetres(np.asarray(handle[DEPTH_DATASET][0]))
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"cannot read released HDF5 {path}: {exc}") from exc


def _calibration(
    release: InspectedRelease,
) -> dict[str, Any]:
    trajectories_by_env = {release.environment: release.trajectories}
    keys = select_calibration_keys(
        trajectories_by_env,
        per_environment=release.contract.calibration_frames,
        environments=[release.environment],
    )
    trajectory_by_episode = {
        trajectory.episode: trajectory for trajectory in release.trajectories
    }
    selected: dict[int, list[int]] = defaultdict(list)
    for key in keys:
        environment, _split, episode, frame = key.split("/")
        if environment != release.environment:
            raise ContractError(f"foreign native calibration key {key}")
        selected[int(episode)].append(int(frame))
    samples_by_key: dict[str, np.ndarray] = {}
    for episode in sorted(selected):
        trajectory = trajectory_by_episode[episode]
        observations = _load_observations(release, trajectory)
        for frame in selected[episode]:
            samples_by_key[trajectory.logical_key(frame)] = _read_aligned_depth(
                release, trajectory, observations, frame
            )
    if set(samples_by_key) != set(keys):
        raise ContractError("native calibration did not reproduce its exact key set")
    lo, hi = compute_global_calibration(samples_by_key[key] for key in keys)
    return {
        "scope": "environment_training_only_native_upper_bound",
        "environment": release.environment,
        "selected_environments": [release.environment],
        "selection": "smallest_sha256_training_keys_for_environment",
        "per_environment": release.contract.calibration_frames,
        "frame_key_format": "<env>/<split>/<episode:05d>/<frame:06d>",
        "keys": keys,
        "keys_sha256": sha256_bytes(canonical_json_bytes(keys)),
        "sample": "metric_depth_metres[::8,::8]",
        "sample_stride": CALIBRATION_STRIDE,
        "percentiles": [2.0, 98.0],
        "percentile_method": "numpy.linear",
        "lo": lo,
        "hi": hi,
        "raw_units": "metres",
    }


def _producer_provenance(contract: ReleaseContract) -> dict[str, Any]:
    return {
        "name": "released-pyflex-native-depth-upper-bound",
        "source": "released AdaptiGraph HDF5",
        "scientific_role": "explicit_upper_bound_ablation_only",
        "simulator_replay": False,
        "camera": "cam_1",
        "rgb_alignment": "numpy.array_equal(obses.pth[t], color/cam_1[0])",
        "state_alignment": "numpy.array_equal(states.pth[episode,t], positions[0])",
        "storage_dtype": "uint16",
        "storage_units": "millimetres",
        "metric_decode": "float32(depth_uint16) / 1000",
        "output_frames_per_episode": contract.output_frames,
        "released_hdf5_files_per_episode": contract.hdf5_files,
        "excluded_hdf5_frames": list(
            range(contract.output_frames, contract.hdf5_files)
        ),
        "training_defaults_modified": False,
        "neutral_depth_boundary_modified": False,
    }


def _wire_format() -> dict[str, Any]:
    return {
        "physical_key": "<split>/<episode:05d>/<frame:06d>",
        "dtype": "<f2",
        "shape": list(OUTPUT_SHAPE),
        "order": "C",
        "compressor": "zstd",
        "compressor_level": ZSTD_LEVEL,
        "map_size": MAP_SIZE,
        "normalization": "clip((depth_m-lo)/(hi-lo),0,1)",
        "inverted": False,
    }


def _validate_extra_hdf5(
    release: InspectedRelease,
    trajectory: Trajectory,
    frame: int,
) -> None:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ContractError("the pinned image must provide h5py") from exc
    path = release.base / f"{trajectory.episode:06d}" / f"{frame:02d}.h5"
    try:
        with h5py.File(path, "r") as handle:
            _require_hdf5_schema(
                handle, path=path, particle_count=release.particle_count
            )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"cannot read released HDF5 {path}: {exc}") from exc


def _native_source_hash(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _existing_destination_error(destination: Path) -> ContractError:
    detail = ""
    manifest = destination / "manifest.json"
    data = destination / "data.mdb"
    if manifest.is_file() and data.is_file():
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
            recorded = document.get("data_mdb_sha256")
            actual = sha256_file(data)
            detail = f" (recorded data hash={recorded}, actual={actual})"
        except Exception as exc:
            detail = f" (existing cache cannot be verified: {exc})"
    return ContractError(
        f"destination already exists and will not be overwritten: {destination}{detail}"
    )


def build_native_cache(
    *,
    root: Path,
    output_root: Path,
    environment: str,
    contract: ReleaseContract | None = None,
) -> dict[str, Any]:
    """Build one cache and publish it only after every source/key/hash gate passes."""

    try:
        import lmdb
        import zstandard
    except Exception as exc:  # pragma: no cover - pinned-container dependency
        raise ContractError("the pinned image must provide lmdb and zstandard") from exc

    release = inspect_release(root, environment, contract)
    calibration = _calibration(release)
    destination = output_root.resolve() / f"{environment}.lmdb"
    if destination.exists():
        raise _existing_destination_error(destination)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{environment}.lmdb.building-{uuid.uuid4().hex}"
    staging.mkdir()
    database = lmdb.open(
        str(staging),
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
        # Native caches are intended to be byte-reproducible.  LMDB's
        # meminit=False optimization can persist nondeterministic unused page
        # bytes, so zero-initialize every new page before it is hashed.
        meminit=True,
        max_dbs=1,
    )
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    records: list[dict[str, Any]] = []
    source_episode_records: list[dict[str, Any]] = []
    total_keys = 0
    try:
        for ordinal, trajectory in enumerate(release.trajectories):
            observations = _load_observations(release, trajectory)
            episode_root = release.base / f"{trajectory.episode:06d}"
            hdf5_records: list[dict[str, str]] = []
            depths: list[np.ndarray] = []
            for frame in range(release.contract.hdf5_files):
                path = episode_root / f"{frame:02d}.h5"
                hdf5_records.append(
                    {
                        "path": path.relative_to(release.root).as_posix(),
                        "sha256": _hash_and_require(
                            path, None, f"HDF5 frame {trajectory.episode}/{frame}"
                        ),
                    }
                )
                if frame < release.contract.output_frames:
                    depths.append(
                        _read_aligned_depth(
                            release, trajectory, observations, frame
                        )
                    )
                else:
                    _validate_extra_hdf5(release, trajectory, frame)
            ordered_keys = [
                trajectory.physical_key(frame)
                for frame in range(release.contract.output_frames)
            ]
            with database.begin(write=True) as transaction:
                for key, depth_m in zip(ordered_keys, depths):
                    value = normalize_depth(
                        depth_m,
                        lo=float(calibration["lo"]),
                        hi=float(calibration["hi"]),
                    )
                    if not transaction.put(
                        key.encode("ascii"),
                        encode_depth_value(value, compressor),
                        overwrite=False,
                    ):
                        raise ContractError(f"duplicate native depth key {key}")
            database.sync(True)
            total_keys += len(ordered_keys)
            property_path = episode_root / "property_params.pkl"
            property_hash = _hash_and_require(
                property_path,
                None,
                f"property parameters for episode {trajectory.episode}",
            )
            records.append(
                {
                    "trajectory_key": trajectory.trajectory_key,
                    "source_path": trajectory.source_relpath,
                    "source_kind": trajectory.source_kind,
                    "source_video_sha256": trajectory.source_sha256,
                    "ordered_frame_count": release.contract.output_frames,
                    "ordered_output_keys": ordered_keys,
                    "chunk_boundaries": streaming_chunks(
                        release.contract.output_frames
                    ),
                    "alignment_results": [
                        {
                            "kind": "released_hdf5_exact_alignment",
                            "camera": "cam_1",
                            "rgb": "array_equal",
                            "state": "array_equal",
                            "frames": release.contract.output_frames,
                        }
                    ],
                    "commit_ordinal": ordinal,
                    "alignment": {
                        "rgb": "exact_array_equal_cam_1",
                        "state": "exact_array_equal_positions",
                        "aligned_frames": release.contract.output_frames,
                    },
                    "native_depth": {
                        "dataset": DEPTH_DATASET,
                        "stored_dtype": "uint16",
                        "stored_units": "millimetres",
                        "decoded_dtype": "float32",
                        "decoded_units": "metres",
                        "scale": 0.001,
                    },
                }
            )
            source_episode_records.append(
                {
                    "episode": trajectory.episode,
                    "obses": {
                        "path": trajectory.source_relpath,
                        "sha256": trajectory.source_sha256,
                    },
                    "property_params": {
                        "path": property_path.relative_to(release.root).as_posix(),
                        "sha256": property_hash,
                    },
                    "hdf5": hdf5_records,
                }
            )
        entries = int(database.stat()["entries"])
        if total_keys != release.contract.expected_keys or entries != total_keys:
            raise ContractError(
                f"native key count differs: expected {release.contract.expected_keys}, "
                f"committed={total_keys}, lmdb={entries}"
            )
    except BaseException:
        database.close()
        raise
    database.close()

    data_hash = sha256_file(staging / "data.mdb")
    native_sources = {
        "environment": environment,
        "global": release.global_hashes,
        "cameras": release.camera_records,
        "episodes": source_episode_records,
    }
    native_source_index_sha256 = _native_source_hash(native_sources)
    tool_sha256 = sha256_file(Path(__file__))
    manifest_basis: dict[str, Any] = {
        "schema": "dinocular-depth-cache-v1",
        "environment": environment,
        "trajectory_count": len(release.trajectories),
        "frame_count": total_keys,
        "source_index_sha256": _source_index_hash(release.trajectories),
        "native_source_index_sha256": native_source_index_sha256,
        "source_provenance": native_sources,
        "producer": _producer_provenance(release.contract),
        "calibration": calibration,
        "wire_format": _wire_format(),
        "tool_sha256": tool_sha256,
        "trajectories": records,
        "data_mdb_sha256": data_hash,
        "closed_before_hash": True,
        "determinism": {
            "manifest_excludes_wall_clock_and_host_fields": True,
            "manifest_id_is_sha256_of_manifest_basis": True,
            "insertion_order": "split_then_episode_then_frame",
            "expected_keys": release.contract.expected_keys,
        },
    }
    manifest_id = sha256_bytes(canonical_json_bytes(manifest_basis))
    manifest = {
        **manifest_basis,
        "manifest_id": manifest_id,
        "deterministic_manifest_basis_sha256": manifest_id,
    }
    atomic_write_json(staging / "manifest.json", manifest)

    if destination.exists():
        raise _existing_destination_error(destination)
    os.rename(staging, destination)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--environment", choices=SUPPORTED_ENVIRONMENTS, required=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_native_cache(
        root=args.root,
        output_root=args.out,
        environment=args.environment,
        contract=released_contract(args.environment),
    )
    destination = args.out.resolve() / f"{args.environment}.lmdb"
    result = {
        "state": "PASS",
        "scientific_role": "explicit_upper_bound_ablation_only",
        "environment": args.environment,
        "cache": str(destination),
        "manifest_sha256": sha256_file(destination / "manifest.json"),
        "manifest_id": manifest["manifest_id"],
        "data_mdb_sha256": manifest["data_mdb_sha256"],
        "keys": manifest["frame_count"],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"NATIVE DEPTH CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
