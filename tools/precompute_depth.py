#!/usr/bin/env python3
"""Build the fixed full-trajectory DA3-Streaming depth caches.

The command-line path is intentionally fail closed: it accepts only the pinned
producer and the fixed 120/60/504 contract.  The small public functions in this
module accept an injected trajectory producer so that cache mechanics can be
tested on CPU without importing (or pretending to run) DA3.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import pickle
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np


DA3_REPOSITORY = "https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_COMMIT = "e74fd796e96b7e781a5506fd8503b6bd7232513c"
SALAD_COMMIT = "6aede13a3f6c25750bf7fde10209c06cb73060bb"
MODEL_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
ARTIFACTS: dict[str, dict[str, Any]] = {
    "model.safetensors": {
        "bytes": 6_759_558_100,
        "sha256": "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c",
    },
    "config.json": {
        "bytes": 3_113,
        "sha256": "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468",
    },
    "dino_salad.ckpt": {
        "bytes": 352_040_378,
        "sha256": "6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada",
    },
}
SELECTED_ENVIRONMENTS = ("pusht", "wall", "rope", "granular")
DEFAULT_BUILD_ORDER = ("wall", "pusht", "rope", "granular")
CHUNK_SIZE = 120
OVERLAP = 60
PROCESS_RES = 504
CALIBRATION_PER_ENV = 128
CALIBRATION_STRIDE = 8
MAP_SIZE = 1 << 40  # sparse 1 TiB
OUTPUT_SHAPE = (224, 224)
WIRE_DTYPE = np.dtype("<f2")
ZSTD_LEVEL = 3


class ContractError(RuntimeError):
    """A pinned producer, dataset, cache, or invocation violated the contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def producer_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return relocation-independent producer identity for transferred caches."""

    identity = copy.deepcopy(dict(provenance))
    identity.pop("da3_root", None)
    for artifact in identity.get("artifacts", {}).values():
        if isinstance(artifact, dict):
            artifact.pop("path", None)
    return identity


@dataclasses.dataclass(frozen=True)
class Trajectory:
    environment: str
    split: str
    episode: int
    frame_count: int
    source_path: Path
    source_relpath: str
    source_kind: str
    source_sha256: str

    @property
    def trajectory_key(self) -> str:
        return f"{self.split}/{self.episode:05d}"

    def physical_key(self, frame: int) -> str:
        if not 0 <= frame < self.frame_count:
            raise IndexError(frame)
        return f"{self.trajectory_key}/{frame:06d}"

    def logical_key(self, frame: int) -> str:
        return f"{self.environment}/{self.physical_key(frame)}"


@dataclasses.dataclass
class ProducerResult:
    """Uncropped metric depth and auditable streaming metadata for one trajectory."""

    depth_m: Sequence[np.ndarray]
    chunk_boundaries: Sequence[Mapping[str, Any]]
    alignments: Sequence[Mapping[str, Any]]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


class TrajectoryDepthProducer(Protocol):
    provenance: Mapping[str, Any]

    def infer_trajectory(
        self, frames_rgb: np.ndarray, trajectory: Trajectory
    ) -> ProducerResult: ...


def _torch_load(path: Path) -> Any:
    import torch

    # Explicitly retain the DINO-WM dataset behavior under newer PyTorch, whose
    # default changed to weights_only=True.
    return torch.load(path, map_location="cpu", weights_only=False)


def _split_indices(
    n: int, train_fraction: float = 0.9, seed: int = 42
) -> dict[int, str]:
    import torch

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    train_count = int(train_fraction * n)
    result = {idx: "train" for idx in indices[:train_count]}
    result.update({idx: "valid" for idx in indices[train_count:]})
    if len(result) != n:
        raise ContractError("deterministic train/valid split lost an episode")
    return result


def _make_trajectory(
    root: Path,
    environment: str,
    split: str,
    episode: int,
    frame_count: int,
    source_path: Path,
    source_kind: str,
) -> Trajectory:
    if frame_count <= 0:
        raise ContractError(
            f"non-positive frame count for {environment}/{split}/{episode}"
        )
    if not source_path.is_file():
        raise ContractError(f"missing trajectory RGB source: {source_path}")
    return Trajectory(
        environment=environment,
        split=split,
        episode=episode,
        frame_count=int(frame_count),
        source_path=source_path,
        source_relpath=source_path.relative_to(root).as_posix(),
        source_kind=source_kind,
        source_sha256=sha256_file(source_path),
    )


def enumerate_environment(root: Path, environment: str) -> list[Trajectory]:
    """Enumerate the official DINO-WM RGB sources without importing datasets."""

    root = root.resolve()
    trajectories: list[Trajectory] = []
    if environment == "pusht":
        base = root / "pusht_noise"
        for split, source_split in (("train", "train"), ("valid", "val")):
            split_root = base / source_split
            with (split_root / "seq_lengths.pkl").open("rb") as handle:
                lengths = pickle.load(handle)
            for episode, frame_count in enumerate(lengths):
                trajectories.append(
                    _make_trajectory(
                        root,
                        environment,
                        split,
                        episode,
                        int(frame_count),
                        split_root / "obses" / f"episode_{episode:03d}.mp4",
                        "mp4",
                    )
                )
    elif environment == "wall":
        base = root / "wall_single"
        actions = _torch_load(base / "actions.pth")
        if getattr(actions, "ndim", None) < 2:
            raise ContractError(f"invalid wall actions tensor: {base / 'actions.pth'}")
        split_for = _split_indices(int(actions.shape[0]))
        for episode in range(int(actions.shape[0])):
            trajectories.append(
                _make_trajectory(
                    root,
                    environment,
                    split_for[episode],
                    episode,
                    int(actions.shape[1]),
                    base / "obses" / f"episode_{episode:03d}.pth",
                    "wall_pth",
                )
            )
    elif environment in ("rope", "granular"):
        base = root / "deformable" / environment
        # actions.pth carries the same N,T axes but is dramatically smaller
        # than the particle-state tensor, so enumeration does not inflate host
        # memory before DA3 starts.
        actions = _torch_load(base / "actions.pth")
        if getattr(actions, "ndim", None) < 2:
            raise ContractError(
                f"invalid {environment} actions tensor: {base / 'actions.pth'}"
            )
        split_for = _split_indices(int(actions.shape[0]))
        for episode in range(int(actions.shape[0])):
            trajectories.append(
                _make_trajectory(
                    root,
                    environment,
                    split_for[episode],
                    episode,
                    int(actions.shape[1]),
                    base / f"{episode:06d}" / "obses.pth",
                    "deformable_pth",
                )
            )
    else:
        raise ContractError(f"unsupported environment {environment!r}")

    trajectories.sort(key=lambda item: (item.split, item.episode))
    if not trajectories:
        raise ContractError(f"no trajectories found for {environment}")
    return trajectories


def enumerate_selected(root: Path) -> dict[str, list[Trajectory]]:
    return {
        environment: enumerate_environment(root, environment)
        for environment in SELECTED_ENVIRONMENTS
    }


def _as_uint8_rgb(frames: Any, trajectory: Trajectory) -> np.ndarray:
    if hasattr(frames, "detach"):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ContractError(
            f"{trajectory.source_relpath}: expected a rank-4 RGB trajectory, got {frames.shape}"
        )
    if trajectory.source_kind == "wall_pth":
        if frames.shape[1] not in (3, 4):
            raise ContractError(f"wall RGB must be TCHW, got {frames.shape}")
        frames = np.transpose(frames[:, :3], (0, 2, 3, 1))
    elif trajectory.source_kind in ("deformable_pth", "mp4"):
        if frames.shape[-1] not in (3, 4):
            raise ContractError(f"RGB must be THWC, got {frames.shape}")
        frames = frames[..., :3]
    else:
        raise ContractError(f"unknown source kind {trajectory.source_kind!r}")

    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating):
            if frames.dtype != np.float32 or not np.isfinite(frames).all():
                raise ContractError(
                    f"{trajectory.source_relpath}: released PTH RGB floats must be finite float32"
                )
            if frames.size and (frames.min() < 0 or frames.max() > 255):
                raise ContractError(
                    f"{trajectory.source_relpath}: RGB values outside [0,255]"
                )
            # The released Wall/Rope/Granular tensors preserve renderer
            # antialiasing as continuous float32 values in [0,255].  Official
            # DA3-Streaming consumes image files, so make the required PNG
            # quantization explicit and deterministic instead of relying on a
            # library cast.
            frames = np.floor(frames + np.float32(0.5)).astype(np.uint8)
        elif not np.issubdtype(frames.dtype, np.integer):
            raise ContractError(
                f"{trajectory.source_relpath}: source RGB must be integer or released float32 [0,255], got {frames.dtype}"
            )
        else:
            if frames.size and (frames.min() < 0 or frames.max() > 255):
                raise ContractError(
                    f"{trajectory.source_relpath}: RGB values outside [0,255]"
                )
            frames = frames.astype(np.uint8)
    if (
        trajectory.source_kind == "wall_pth"
        and len(frames) == trajectory.frame_count + 1
    ):
        # WallDataset defines episode length from actions.shape[1] and indexes
        # exactly those frames. The released PTH additionally stores the final
        # post-action observation, which is outside the official model sample.
        frames = frames[: trajectory.frame_count]
    frames = np.ascontiguousarray(frames)
    if len(frames) != trajectory.frame_count:
        raise ContractError(
            f"{trajectory.source_relpath}: decoded {len(frames)} frames, expected "
            f"{trajectory.frame_count}"
        )
    return frames


def decode_trajectory(trajectory: Trajectory) -> np.ndarray:
    """Decode every source frame in chronological order, without subsampling."""

    if trajectory.source_kind == "mp4":
        try:
            from decord import VideoReader
        except Exception as exc:  # pragma: no cover - exercised in the pinned image
            raise ContractError(
                "decord is required to decode PushT trajectories"
            ) from exc
        reader = VideoReader(str(trajectory.source_path), num_threads=1)
        if len(reader) != trajectory.frame_count:
            raise ContractError(
                f"{trajectory.source_relpath}: video has {len(reader)} frames, expected "
                f"{trajectory.frame_count}"
            )
        decoded = reader.get_batch(list(range(len(reader))))
        if hasattr(decoded, "asnumpy"):
            decoded = decoded.asnumpy()
        return _as_uint8_rgb(decoded, trajectory)
    return _as_uint8_rgb(_torch_load(trajectory.source_path), trajectory)


def _resize_bilinear(
    array: np.ndarray, output_hw: tuple[int, int], *, antialias: bool
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    value = np.array(array, dtype=np.float32, copy=True, order="C")
    if value.ndim == 2:
        tensor = torch.from_numpy(value)[None, None]
        channels_last = False
    elif value.ndim == 3 and value.shape[-1] in (1, 3, 4):
        tensor = torch.from_numpy(value).permute(2, 0, 1)[None]
        channels_last = True
    else:
        raise ContractError(f"resize expects HW or HWC, got {value.shape}")
    kwargs = dict(size=output_hw, mode="bilinear", align_corners=False)
    # Torch 2.3 (the pinned image) supports fixed antialias for bilinear resize.
    result = functional.interpolate(tensor, antialias=antialias, **kwargs)[0]
    if channels_last:
        return result.permute(1, 2, 0).cpu().numpy()
    return result[0].cpu().numpy()


def world_model_resize_crop(array: np.ndarray) -> np.ndarray:
    """Apply Resize(short edge=224), CenterCrop(224) with bilinear resize."""

    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        raise ContractError(f"empty image geometry {array.shape}")
    if height <= width:
        output_h, output_w = 224, int(224 * width / height)
    else:
        output_h, output_w = int(224 * height / width), 224
    resized = _resize_bilinear(array, (output_h, output_w), antialias=True)
    # torchvision.transforms.CenterCrop uses round rather than floor here.
    top = int(round((output_h - 224) / 2.0))
    left = int(round((output_w - 224) / 2.0))
    cropped = resized[top : top + 224, left : left + 224]
    if cropped.shape[:2] != OUTPUT_SHAPE:
        raise ContractError(f"world-model crop produced {cropped.shape}")
    return np.ascontiguousarray(cropped, dtype=np.float32)


def crop_rgb_for_world_model(frame_rgb: np.ndarray) -> np.ndarray:
    return np.clip(
        world_model_resize_crop(np.asarray(frame_rgb, dtype=np.float32) / 255.0), 0, 1
    )


def crop_metric_depth(depth_m: np.ndarray, decoded_hw: tuple[int, int]) -> np.ndarray:
    """Map DA3 output back to decoded geometry, then apply the RGB crop."""

    depth = np.asarray(depth_m, dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ContractError(f"DA3 depth must be HW after squeeze, got {depth.shape}")
    if not np.isfinite(depth).all():
        raise ContractError("DA3 returned non-finite metric depth")
    # Mapping to decoded geometry is explicitly bilinear with align_corners=False.
    restored = _resize_bilinear(depth, decoded_hw, antialias=False)
    cropped = world_model_resize_crop(restored)
    if not np.isfinite(cropped).all():
        raise ContractError("cropped metric depth is non-finite")
    return cropped


def crop_producer_result(
    result: ProducerResult, frames_rgb: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(result.depth_m) != len(frames_rgb):
        raise ContractError(
            f"producer emitted {len(result.depth_m)} depths for {len(frames_rgb)} decoded frames"
        )
    cropped: list[np.ndarray] = []
    resize_records: list[dict[str, Any]] = []
    for frame, depth in zip(frames_rgb, result.depth_m):
        decoded_hw = (int(frame.shape[0]), int(frame.shape[1]))
        raw_depth = np.squeeze(np.asarray(depth))
        if raw_depth.ndim != 2:
            raise ContractError(f"producer depth has invalid shape {raw_depth.shape}")
        cropped.append(crop_metric_depth(raw_depth, decoded_hw))
        resize_records.append(
            {
                "decoded_hw": list(decoded_hw),
                "producer_depth_hw": [int(raw_depth.shape[0]), int(raw_depth.shape[1])],
                "producer_to_decoded_scale_yx": [
                    decoded_hw[0] / raw_depth.shape[0],
                    decoded_hw[1] / raw_depth.shape[1],
                ],
                "world_model_resize": "short_edge_224_bilinear_antialias_true",
                "world_model_crop": "center_224",
            }
        )
    return np.stack(cropped).astype(np.float32, copy=False), resize_records


def streaming_chunks(frame_count: int) -> list[dict[str, Any]]:
    """The official 120/60 ranges and overlap-tail discard schedule."""

    if frame_count <= 0:
        raise ContractError("trajectory must contain at least one frame")
    if frame_count <= CHUNK_SIZE:
        ranges = [(0, frame_count)]
    else:
        step = CHUNK_SIZE - OVERLAP
        count = (frame_count - OVERLAP + step - 1) // step
        ranges = [
            (idx * step, min(idx * step + CHUNK_SIZE, frame_count))
            for idx in range(count)
        ]
    chunks: list[dict[str, Any]] = []
    for idx, (start, end) in enumerate(ranges):
        emitted_end = end if idx == len(ranges) - 1 else end - OVERLAP
        chunks.append(
            {
                "chunk_index": idx,
                "start": start,
                "end": end,
                "emitted_start": start,
                "emitted_end": emitted_end,
                "discarded_overlap_tail": 0 if idx == len(ranges) - 1 else OVERLAP,
                "boundary_pair": None if idx == 0 else [start - 1, start],
            }
        )
    emitted = [
        index
        for chunk in chunks
        for index in range(chunk["emitted_start"], chunk["emitted_end"])
    ]
    if emitted != list(range(frame_count)):
        raise ContractError(
            "120/60 overlap discard did not emit each frame exactly once"
        )
    return chunks


def select_calibration_keys(
    trajectories_by_env: Mapping[str, Sequence[Trajectory]],
    per_environment: int = CALIBRATION_PER_ENV,
) -> list[str]:
    selected: list[str] = []
    for environment in SELECTED_ENVIRONMENTS:
        trajectories = trajectories_by_env.get(environment, ())
        keys = [
            trajectory.logical_key(frame)
            for trajectory in trajectories
            for frame in range(trajectory.frame_count)
        ]
        if len(keys) < per_environment:
            raise ContractError(
                f"{environment} has {len(keys)} frames, fewer than the required {per_environment} "
                "global-calibration keys"
            )
        keys.sort(key=lambda key: (sha256_bytes(key.encode()), key))
        selected.extend(keys[:per_environment])
    expected = per_environment * len(SELECTED_ENVIRONMENTS)
    if len(selected) != expected or len(set(selected)) != expected:
        raise ContractError(
            "calibration selection is not exactly stratified and unique"
        )
    return selected


def compute_global_calibration(samples: Iterable[np.ndarray]) -> tuple[float, float]:
    flattened: list[np.ndarray] = []
    for sample in samples:
        depth = np.asarray(sample, dtype=np.float32)
        if depth.shape != OUTPUT_SHAPE:
            raise ContractError(
                f"calibration depth has shape {depth.shape}, expected {OUTPUT_SHAPE}"
            )
        strided = depth[::CALIBRATION_STRIDE, ::CALIBRATION_STRIDE].reshape(-1)
        if not np.isfinite(strided).all():
            raise ContractError("calibration contains non-finite depth")
        flattened.append(strided)
    if not flattened:
        raise ContractError("calibration has no samples")
    values = np.concatenate(flattened)
    lo, hi = np.percentile(values, [2.0, 98.0], method="linear")
    lo, hi = float(lo), float(hi)
    if not math.isfinite(lo) or not math.isfinite(hi) or not hi > lo + 1e-6:
        raise ContractError(f"invalid global depth calibration lo={lo!r}, hi={hi!r}")
    return lo, hi


def normalize_depth(depth_m: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not math.isfinite(lo) or not math.isfinite(hi) or not hi > lo + 1e-6:
        raise ContractError(f"invalid fixed calibration lo={lo!r}, hi={hi!r}")
    value = np.asarray(depth_m, dtype=np.float32)
    if value.shape != OUTPUT_SHAPE or not np.isfinite(value).all():
        raise ContractError(f"invalid cropped metric depth {value.shape}")
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def _require_lmdb_zstd() -> tuple[Any, Any]:
    try:
        import lmdb
        import zstandard
    except Exception as exc:
        raise ContractError(
            "the pinned image must provide Python packages lmdb and zstandard"
        ) from exc
    return lmdb, zstandard


def encode_depth_value(gray: np.ndarray, compressor: Any | None = None) -> bytes:
    _, zstandard = _require_lmdb_zstd()
    value = np.asarray(gray)
    if value.shape != OUTPUT_SHAPE or not np.isfinite(value).all():
        raise ContractError(f"cannot encode invalid depth {value.shape}")
    if value.size and (value.min() < 0 or value.max() > 1):
        raise ContractError("cannot encode depth outside [0,1]")
    wire = np.asarray(value, dtype=WIRE_DTYPE, order="C")
    if wire.dtype.str != "<f2":
        raise ContractError(
            f"wire dtype is not explicitly little endian: {wire.dtype.str}"
        )
    compressor = compressor or zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    return compressor.compress(wire.tobytes(order="C"))


def decode_depth_value(payload: bytes, decompressor: Any | None = None) -> np.ndarray:
    _, zstandard = _require_lmdb_zstd()
    decompressor = decompressor or zstandard.ZstdDecompressor()
    raw = decompressor.decompress(
        payload, max_output_size=int(np.prod(OUTPUT_SHAPE)) * 2
    )
    expected = int(np.prod(OUTPUT_SHAPE)) * WIRE_DTYPE.itemsize
    if len(raw) != expected:
        raise ContractError(
            f"depth value has {len(raw)} raw bytes, expected {expected}"
        )
    return np.frombuffer(raw, dtype=WIRE_DTYPE).reshape(OUTPUT_SHAPE).copy()


def _source_index_hash(trajectories: Sequence[Trajectory]) -> str:
    index = [
        {
            "key": trajectory.trajectory_key,
            "frames": trajectory.frame_count,
            "source": trajectory.source_relpath,
            "sha256": trajectory.source_sha256,
        }
        for trajectory in trajectories
    ]
    return sha256_bytes(canonical_json_bytes(index))


def _du_bytes(path: Path) -> tuple[int, int]:
    logical = 0
    allocated = 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            stat = candidate.stat()
            logical += stat.st_size
            allocated += stat.st_blocks * 512
    return logical, allocated


def _manifest_is_reusable(
    destination: Path,
    trajectories: Sequence[Trajectory],
    calibration: Mapping[str, Any],
    producer: TrajectoryDepthProducer,
) -> bool:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    return (
        manifest.get("source_index_sha256") == _source_index_hash(trajectories)
        and manifest.get("calibration", {}).get("keys_sha256")
        == calibration.get("keys_sha256")
        and manifest.get("calibration", {}).get("lo") == calibration.get("lo")
        and manifest.get("calibration", {}).get("hi") == calibration.get("hi")
        and producer_identity(manifest.get("producer", {}))
        == producer_identity(_jsonable(producer.provenance))
        and manifest.get("data_mdb_sha256") == sha256_file(destination / "data.mdb")
    )


def build_environment_cache(
    *,
    output_root: Path,
    environment: str,
    trajectories: Sequence[Trajectory],
    calibration: Mapping[str, Any],
    producer: TrajectoryDepthProducer,
    calibration_raw: Mapping[str, tuple[Path, Mapping[str, Any]]] | None = None,
    rebuild: bool = False,
    tool_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one immutable LMDB, committing each trajectory in one transaction."""

    lmdb, zstandard = _require_lmdb_zstd()
    if environment not in SELECTED_ENVIRONMENTS:
        raise ContractError(f"unsupported environment {environment!r}")
    if any(trajectory.environment != environment for trajectory in trajectories):
        raise ContractError(
            "environment cache received a trajectory from another environment"
        )
    if not trajectories:
        raise ContractError(f"cannot build empty {environment} cache")
    lo, hi = float(calibration["lo"]), float(calibration["hi"])
    if not hi > lo + 1e-6:
        raise ContractError("cache build received invalid global calibration")

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{environment}.lmdb"
    if destination.exists() and not rebuild:
        if _manifest_is_reusable(destination, trajectories, calibration, producer):
            return json.loads((destination / "manifest.json").read_text())
        raise ContractError(
            f"{destination} already exists but does not match this immutable build; use --rebuild "
            "to mint a new manifest and quarantine the old cache"
        )

    manifest_id = str(uuid.uuid4())
    building = output_root / f".{environment}.lmdb.building-{manifest_id}"
    if building.exists():
        raise ContractError(f"unexpected existing build directory {building}")
    building.mkdir(parents=True)
    database = lmdb.open(
        str(building),
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
        meminit=False,
        max_dbs=1,
    )
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    trajectory_records: list[dict[str, Any]] = []
    total_frames = 0
    started = time.monotonic()
    try:
        for ordinal, trajectory in enumerate(trajectories):
            trajectory_started = time.monotonic()
            decode_started = time.monotonic()
            frames = decode_trajectory(trajectory)
            decode_seconds = time.monotonic() - decode_started
            staged = (calibration_raw or {}).get(
                f"{trajectory.environment}/{trajectory.trajectory_key}"
            )
            if staged is not None:
                raw_path, staged_metadata = staged
                cropped_metric = np.load(raw_path, allow_pickle=False)
                producer_metadata = copy.deepcopy(dict(staged_metadata))
                resize_records = producer_metadata.pop("resize_records")
                chunks = producer_metadata.pop("chunk_boundaries")
                alignments = producer_metadata.pop("alignments")
            else:
                inference_started = time.monotonic()
                result = producer.infer_trajectory(frames, trajectory)
                inference_seconds = time.monotonic() - inference_started
                cropped_metric, resize_records = crop_producer_result(result, frames)
                chunks = _jsonable(result.chunk_boundaries)
                alignments = _jsonable(result.alignments)
                producer_metadata = _jsonable(result.metadata)
                producer_metadata["inference_seconds"] = inference_seconds
            producer_metadata["source_decode_seconds"] = decode_seconds
            if cropped_metric.shape != (trajectory.frame_count, *OUTPUT_SHAPE):
                raise ContractError(
                    f"{trajectory.trajectory_key}: cropped result has shape {cropped_metric.shape}"
                )
            expected_chunks = streaming_chunks(trajectory.frame_count)
            compact_chunks = [
                {key: chunk.get(key) for key in expected.keys()}
                for chunk, expected in zip(chunks, expected_chunks)
            ]
            if compact_chunks != expected_chunks or len(chunks) != len(expected_chunks):
                raise ContractError(
                    f"{trajectory.trajectory_key}: producer chunk schedule differs from fixed 120/60"
                )

            ordered_keys = [
                trajectory.physical_key(frame)
                for frame in range(trajectory.frame_count)
            ]
            # One and only one write transaction is used for this trajectory.
            normalize_compress_commit_started = time.monotonic()
            with database.begin(write=True) as transaction:
                for frame, physical_key in enumerate(ordered_keys):
                    gray = normalize_depth(cropped_metric[frame], lo, hi)
                    inserted = transaction.put(
                        physical_key.encode("ascii"),
                        encode_depth_value(gray, compressor),
                        overwrite=False,
                    )
                    if not inserted:
                        raise ContractError(f"duplicate LMDB key {physical_key}")
            database.sync(True)
            producer_metadata["normalize_compress_commit_seconds"] = (
                time.monotonic() - normalize_compress_commit_started
            )
            total_frames += trajectory.frame_count
            trajectory_records.append(
                {
                    "trajectory_key": trajectory.trajectory_key,
                    "source_path": trajectory.source_relpath,
                    "source_kind": trajectory.source_kind,
                    "source_video_sha256": trajectory.source_sha256,
                    "ordered_frame_count": trajectory.frame_count,
                    "chunk_boundaries": chunks,
                    "alignment_results": alignments,
                    "ordered_output_keys": ordered_keys,
                    "resize_records": resize_records,
                    "producer_metadata": producer_metadata,
                    "commit_ordinal": ordinal,
                    "trajectory_seconds": time.monotonic() - trajectory_started,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "trajectory_committed",
                        "environment": environment,
                        "trajectory": trajectory.trajectory_key,
                        "frames": trajectory.frame_count,
                        "completed_trajectories": ordinal + 1,
                        "total_trajectories": len(trajectories),
                        "utc": utc_now(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except BaseException:
        database.close()
        raise
    database.close()

    data_path = building / "data.mdb"
    data_hash = sha256_file(data_path)
    elapsed = time.monotonic() - started
    logical_bytes, allocated_bytes = _du_bytes(building)
    manifest: dict[str, Any] = {
        "schema": "dinocular-depth-cache-v1",
        "manifest_id": manifest_id,
        "created_utc": utc_now(),
        "environment": environment,
        "trajectory_count": len(trajectories),
        "frame_count": total_frames,
        "source_index_sha256": _source_index_hash(trajectories),
        "producer": _jsonable(producer.provenance),
        "calibration": _jsonable(calibration),
        "wire_format": {
            "physical_key": "<split>/<episode:05d>/<frame:06d>",
            "dtype": "<f2",
            "shape": list(OUTPUT_SHAPE),
            "order": "C",
            "compressor": "zstd",
            "compressor_level": ZSTD_LEVEL,
            "map_size": MAP_SIZE,
            "normalization": "clip((depth_m-lo)/(hi-lo),0,1)",
            "inverted": False,
        },
        "tool_sha256": tool_sha256 or sha256_file(Path(__file__)),
        "trajectories": trajectory_records,
        "build_metrics": {
            "elapsed_seconds": elapsed,
            "committed_frames_per_second": total_frames / elapsed if elapsed else None,
            "lmdb_logical_bytes_before_manifest": logical_bytes,
            "lmdb_allocated_bytes_before_manifest": allocated_bytes,
            "logical_bytes_per_frame": logical_bytes / total_frames,
            "allocated_bytes_per_frame": allocated_bytes / total_frames,
            "peak_host_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "data_mdb_sha256": data_hash,
        "closed_before_hash": True,
    }
    atomic_write_json(building / "manifest.json", manifest)

    if destination.exists():
        quarantine = output_root / (
            f"{environment}.lmdb.quarantine-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        os.replace(destination, quarantine)
    os.replace(building, destination)
    return manifest


def _git_output(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"cannot audit pinned git checkout {root}: {exc}") from exc
    return result.stdout.strip()


def verify_pinned_da3(da3_root: Path, model_dir: Path) -> dict[str, Any]:
    da3_root = da3_root.resolve()
    model_dir = model_dir.resolve()
    if _git_output(da3_root, "rev-parse", "HEAD") != DA3_COMMIT:
        raise ContractError(f"DA3 checkout is not pinned to {DA3_COMMIT}")
    dirty = _git_output(da3_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ContractError(f"pinned DA3 checkout has tracked modifications:\n{dirty}")
    salad_root = da3_root / "da3_streaming" / "loop_utils" / "salad"
    if (
        not salad_root.is_dir()
        or _git_output(salad_root, "rev-parse", "HEAD") != SALAD_COMMIT
    ):
        raise ContractError(
            f"salad submodule is absent or not pinned to {SALAD_COMMIT}"
        )

    verified: dict[str, Any] = {}
    for name, expected in ARTIFACTS.items():
        path = model_dir / name
        if not path.is_file():
            raise ContractError(f"missing pinned artifact {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ContractError(
                f"pinned artifact mismatch for {path}: bytes={size}, sha256={digest}; "
                f"expected bytes={expected['bytes']}, sha256={expected['sha256']}"
            )
        verified[name] = {"path": str(path), "bytes": size, "sha256": digest}
    return {
        "repository": DA3_REPOSITORY,
        "commit": DA3_COMMIT,
        "salad_commit": SALAD_COMMIT,
        "model_revision": MODEL_REVISION,
        "da3_root": str(da3_root),
        "artifacts": verified,
    }


class OfficialDA3StreamingProducer:
    """Pinned official DA3-Streaming runner with one shared model instance.

    Only model initialization is shared.  Each call constructs fresh official
    streaming state and calls ``DA3_Streaming.run()`` exactly once on one image
    directory containing exactly one complete trajectory.
    """

    def __init__(self, da3_root: Path, model_dir: Path, work_root: Path | None = None):
        audit = verify_pinned_da3(da3_root, model_dir)
        try:
            import torch
        except Exception as exc:  # pragma: no cover - production dependency
            raise ContractError(
                "PyTorch is unavailable for the pinned DA3 producer"
            ) from exc
        if not torch.cuda.is_available():
            raise ContractError(
                "official DA3-Streaming requires CUDA; no CPU/fake fallback is allowed"
            )
        if not torch.cuda.is_bf16_supported():
            raise ContractError("fixed DA3 contract requires BF16 on H100/Blackwell")

        da3_root = da3_root.resolve()
        model_dir = model_dir.resolve()
        sys.path.insert(0, str(da3_root / "src"))
        sys.path.insert(0, str(da3_root / "da3_streaming"))
        try:
            import inspect
            import yaml
            import da3_streaming as streaming
        except Exception as exc:  # pragma: no cover - production dependency
            raise ContractError(
                "the official pinned da3_streaming API could not be imported; no alternate producer "
                "will be substituted"
            ) from exc

        signature = inspect.signature(streaming.DepthAnything3.inference)
        if (
            signature.parameters["process_res"].default != PROCESS_RES
            or signature.parameters["process_res_method"].default
            != "upper_bound_resize"
        ):
            raise ContractError(
                "pinned DA3 inference defaults no longer satisfy 504 upper-bound resize"
            )
        with (
            da3_root / "da3_streaming" / "configs" / "base_config.yaml"
        ).open() as handle:
            config = yaml.safe_load(handle)
        config["Weights"] = {
            "DA3": str(model_dir / "model.safetensors"),
            "DA3_CONFIG": str(model_dir / "config.json"),
            "SALAD": str(model_dir / "dino_salad.ckpt"),
        }
        config["Model"]["chunk_size"] = CHUNK_SIZE
        config["Model"]["overlap"] = OVERLAP
        config["Model"]["loop_enable"] = True
        config["Model"]["ref_view_strategy"] = "saddle_balanced"
        config["Model"]["ref_view_strategy_loop"] = "saddle_balanced"
        config["Model"]["save_depth_conf_result"] = True
        config["Model"]["save_debug_info"] = True
        config["Model"]["delete_temp_files"] = False

        with (model_dir / "config.json").open() as handle:
            model_config = json.load(handle)
        model = streaming.DepthAnything3(**model_config)
        state_dict = streaming.load_file(str(model_dir / "model.safetensors"))
        incompatible = model.load_state_dict(state_dict, strict=False)
        model.eval()
        self._model = model.to("cuda")
        self._streaming = streaming
        self._torch = torch
        self._config = config
        self._work_root = work_root.resolve() if work_root is not None else None
        if self._work_root is not None:
            self._work_root.mkdir(parents=True, exist_ok=True)
        contract_config = copy.deepcopy(config)
        contract_config["Weights"] = {
            name: ARTIFACTS[artifact]["sha256"]
            for name, artifact in (
                ("DA3", "model.safetensors"),
                ("DA3_CONFIG", "config.json"),
                ("SALAD", "dino_salad.ckpt"),
            )
        }
        self.provenance = {
            "name": "DA3NESTED-GIANT-LARGE-1.1/DA3-Streaming",
            **audit,
            "effective_config_sha256": sha256_bytes(
                canonical_json_bytes(contract_config)
            ),
            "settings": {
                "precision": "bfloat16",
                "process_res": PROCESS_RES,
                "process_res_method": "upper_bound_resize",
                "ref_view_strategy": "saddle_balanced",
                "loop_closure": True,
                "chunk_size": CHUNK_SIZE,
                "overlap": OVERLAP,
                "overlap_policy": "discard_duplicated_tail_no_blend",
                "pth_float32_rgb_quantization": "round_half_up_to_uint8_for_png",
                "wall_terminal_observation_policy": "drop_post_action_frame_not_selected_by_WallDataset",
            },
            "official_non_strict_load_audit": {
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
            },
        }

    def _new_engine(self, image_dir: Path, output_dir: Path) -> Any:
        """Initialize the pinned official class state while reusing its model."""

        streaming = self._streaming
        torch = self._torch
        config = copy.deepcopy(self._config)
        engine = streaming.DA3_Streaming.__new__(streaming.DA3_Streaming)
        engine.config = config
        engine.chunk_size = config["Model"]["chunk_size"]
        engine.overlap = config["Model"]["overlap"]
        engine.overlap_s = 0
        engine.overlap_e = engine.overlap
        engine.conf_threshold = 1.5
        engine.seed = 42
        engine.device = "cuda"
        engine.dtype = torch.bfloat16
        engine.img_dir = str(image_dir)
        engine.img_list = None
        engine.output_dir = str(output_dir)
        engine.result_unaligned_dir = str(output_dir / "_tmp_results_unaligned")
        engine.result_aligned_dir = str(output_dir / "_tmp_results_aligned")
        engine.result_loop_dir = str(output_dir / "_tmp_results_loop")
        engine.result_output_dir = str(output_dir / "results_output")
        engine.pcd_dir = str(output_dir / "pcd")
        for directory in (
            engine.result_unaligned_dir,
            engine.result_aligned_dir,
            engine.result_loop_dir,
            engine.pcd_dir,
        ):
            os.makedirs(directory, exist_ok=True)
        engine.all_camera_poses = []
        engine.all_camera_intrinsics = []
        engine.delete_temp_files = False
        engine.model = self._model
        engine.skyseg_session = None
        engine.chunk_indices = None
        engine.loop_list = []
        engine.loop_optimizer = streaming.Sim3LoopOptimizer(config)
        engine.sim3_list = []
        engine.loop_sim3_list = []
        engine.loop_predict_list = []
        engine.loop_enable = True
        loop_info = str(output_dir / "loop_closures.txt")
        engine.loop_detector = streaming.LoopDetector(
            image_dir=str(image_dir), output=loop_info, config=config
        )
        engine.loop_detector.load_model()
        return engine

    def infer_trajectory(
        self, frames_rgb: np.ndarray, trajectory: Trajectory
    ) -> ProducerResult:
        if len(frames_rgb) != trajectory.frame_count:
            raise ContractError(
                "official producer was not given one complete trajectory"
            )
        from PIL import Image

        with tempfile.TemporaryDirectory(
            prefix="dinocular-da3-trajectory-", dir=self._work_root
        ) as temporary:
            temporary_path = Path(temporary)
            image_dir = temporary_path / "images"
            output_dir = temporary_path / "output"
            image_dir.mkdir()
            output_dir.mkdir()
            for index, frame in enumerate(frames_rgb):
                Image.fromarray(frame, mode="RGB").save(
                    image_dir / f"frame_{index:06d}.png", format="PNG", compress_level=1
                )
            engine = self._new_engine(image_dir, output_dir)
            self._torch.cuda.reset_peak_memory_stats()
            if trajectory.frame_count <= CHUNK_SIZE:
                # At the pinned commit, the optional camera-pose text exporter
                # subtracts a 60-frame tail even for a single chunk and then
                # dereferences empty pose slots.  Disable only that irrelevant
                # export; the official run, joint prediction, loop detection,
                # and chunk result remain unchanged and are audited below.
                engine.save_camera_poses = lambda: None
            sequential_alignments: list[dict[str, Any]] = []
            original_align = engine.align_2pcds

            def recorded_align(*args: Any, **kwargs: Any) -> Any:
                scale, rotation, translation = original_align(*args, **kwargs)
                sequential_alignments.append(
                    {
                        "kind": "dense_confidence_weighted_sequential_sim3",
                        "scale": _jsonable(scale),
                        "rotation": _jsonable(rotation),
                        "translation": _jsonable(translation),
                    }
                )
                return scale, rotation, translation

            engine.align_2pcds = recorded_align
            started = time.monotonic()
            engine.run()  # exactly one official full-trajectory streaming invocation
            elapsed = time.monotonic() - started
            expected_chunks = streaming_chunks(trajectory.frame_count)
            official_ranges = [tuple(pair) for pair in engine.chunk_indices]
            if official_ranges != [
                (chunk["start"], chunk["end"]) for chunk in expected_chunks
            ]:
                raise ContractError(
                    "official DA3 chunk boundaries differ from fixed 120/60 schedule"
                )

            depths: list[np.ndarray] = []
            result_dir = output_dir / "results_output"
            output_files = [
                result_dir / f"frame_{index}.npz"
                for index in range(trajectory.frame_count)
            ]
            if all(path.is_file() for path in output_files):
                for path in output_files:
                    with np.load(path, allow_pickle=False) as record:
                        depths.append(
                            np.asarray(record["depth"], dtype=np.float32).copy()
                        )
            elif len(expected_chunks) == 1:
                # The pinned official one-chunk path leaves its prediction in the
                # official unaligned chunk file and does not populate results_output.
                chunk = np.load(
                    output_dir / "_tmp_results_unaligned" / "chunk_0.npy",
                    allow_pickle=True,
                ).item()
                chunk_depth = np.asarray(chunk.depth)
                if chunk_depth.ndim == 2:
                    chunk_depth = chunk_depth[None]
                depths = [
                    np.asarray(value, dtype=np.float32).copy() for value in chunk_depth
                ]
            else:
                missing = [str(path) for path in output_files if not path.is_file()]
                raise ContractError(
                    f"official DA3-Streaming omitted output frames: {missing[:5]}"
                )

            final_alignments = []
            for index, transform in enumerate(engine.sim3_list):
                scale, rotation, translation = transform
                final_alignments.append(
                    {
                        "kind": "loop_optimized_accumulated_sim3",
                        "target_chunk": index + 1,
                        "scale": _jsonable(scale),
                        "rotation": _jsonable(rotation),
                        "translation": _jsonable(translation),
                    }
                )
            loop_alignments = []
            for left, right, transform in engine.loop_sim3_list:
                scale, rotation, translation = transform
                loop_alignments.append(
                    {
                        "kind": "loop_sim3",
                        "left_chunk": int(left),
                        "right_chunk": int(right),
                        "scale": _jsonable(scale),
                        "rotation": _jsonable(rotation),
                        "translation": _jsonable(translation),
                    }
                )
            metadata = {
                "official_class": "da3_streaming.DA3_Streaming",
                "trajectory_only_call": True,
                "input_frame_count": len(frames_rgb),
                "elapsed_seconds": elapsed,
                "loop_pairs": _jsonable(engine.loop_list),
                "peak_cuda_bytes": int(self._torch.cuda.max_memory_allocated()),
                "processed_depth_hw": list(np.squeeze(depths[0]).shape),
                "single_chunk_pose_export_workaround": trajectory.frame_count
                <= CHUNK_SIZE,
            }
            temporary_logical, temporary_allocated = _du_bytes(temporary_path)
            metadata["temporary_logical_bytes_before_cleanup"] = temporary_logical
            metadata["temporary_allocated_bytes_before_cleanup"] = temporary_allocated
            engine.model = None
            del engine
            self._torch.cuda.empty_cache()
            return ProducerResult(
                depth_m=depths,
                chunk_boundaries=expected_chunks,
                alignments=[
                    *sequential_alignments,
                    *final_alignments,
                    *loop_alignments,
                ],
                metadata=metadata,
            )


def _trajectory_lookup(
    trajectories_by_env: Mapping[str, Sequence[Trajectory]],
) -> dict[str, Trajectory]:
    result: dict[str, Trajectory] = {}
    for trajectories in trajectories_by_env.values():
        for trajectory in trajectories:
            key = f"{trajectory.environment}/{trajectory.trajectory_key}"
            if key in result:
                raise ContractError(f"duplicate trajectory key {key}")
            result[key] = trajectory
    return result


def generate_global_calibration(
    *,
    trajectories_by_env: Mapping[str, Sequence[Trajectory]],
    producer: TrajectoryDepthProducer,
    staging_root: Path,
    per_environment: int = CALIBRATION_PER_ENV,
) -> tuple[dict[str, Any], dict[str, tuple[Path, Mapping[str, Any]]]]:
    keys = select_calibration_keys(trajectories_by_env, per_environment)
    trajectory_lookup = _trajectory_lookup(trajectories_by_env)
    selected_frames: dict[str, list[int]] = defaultdict(list)
    for logical_key in keys:
        environment, split, episode, frame = logical_key.split("/")
        selected_frames[f"{environment}/{split}/{episode}"].append(int(frame))

    raw_by_trajectory: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    samples_by_key: dict[str, np.ndarray] = {}
    environment_order = {name: index for index, name in enumerate(DEFAULT_BUILD_ORDER)}
    calibration_parents = sorted(
        selected_frames,
        key=lambda key: (environment_order[key.split("/", 1)[0]], key),
    )
    for trajectory_key in calibration_parents:
        trajectory = trajectory_lookup[trajectory_key]
        frames = decode_trajectory(trajectory)
        result = producer.infer_trajectory(frames, trajectory)
        cropped, resize_records = crop_producer_result(result, frames)
        raw_path = (
            staging_root
            / trajectory.environment
            / trajectory.split
            / f"{trajectory.episode:05d}.npy"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_path, cropped, allow_pickle=False)
        metadata = {
            "resize_records": resize_records,
            "chunk_boundaries": _jsonable(result.chunk_boundaries),
            "alignments": _jsonable(result.alignments),
            **_jsonable(result.metadata),
            "calibration_parent": True,
        }
        raw_by_trajectory[trajectory_key] = (raw_path, metadata)
        for frame in selected_frames[trajectory_key]:
            samples_by_key[trajectory.logical_key(frame)] = cropped[frame]
    if set(samples_by_key) != set(keys):
        raise ContractError(
            "calibration inference did not produce every selected frame key"
        )
    lo, hi = compute_global_calibration(samples_by_key[key] for key in keys)
    calibration = {
        "scope": "global_across_selected_environments",
        "selected_environments": list(SELECTED_ENVIRONMENTS),
        "selection": "128_smallest_sha256_keys_per_environment",
        "per_environment": per_environment,
        "frame_key_format": "<env>/<split>/<episode:05d>/<frame:06d>",
        "keys": keys,
        "keys_sha256": sha256_bytes(canonical_json_bytes(keys)),
        "sample": "cropped_metric_depth[::8,::8]",
        "sample_stride": CALIBRATION_STRIDE,
        "percentiles": [2.0, 98.0],
        "percentile_method": "numpy.linear",
        "lo": lo,
        "hi": hi,
    }
    return calibration, raw_by_trajectory


def load_calibration_manifest(
    path: Path, producer: TrajectoryDepthProducer
) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ContractError(f"{path} has no calibration object")
    if producer_identity(manifest.get("producer", {})) != producer_identity(
        _jsonable(producer.provenance)
    ):
        raise ContractError(
            "calibration manifest producer differs from the pinned current producer"
        )
    if calibration.get("per_environment") != CALIBRATION_PER_ENV:
        raise ContractError(
            "calibration manifest is not the fixed 128-key-per-environment transform"
        )
    if len(calibration.get("keys", [])) != CALIBRATION_PER_ENV * len(
        SELECTED_ENVIRONMENTS
    ):
        raise ContractError(
            "calibration manifest does not contain exactly 512 frame keys"
        )
    expected_hash = sha256_bytes(canonical_json_bytes(calibration["keys"]))
    if calibration.get("keys_sha256") != expected_hash:
        raise ContractError("calibration key-list hash mismatch")
    lo, hi = calibration.get("lo"), calibration.get("hi")
    if (
        not isinstance(lo, (int, float))
        or not isinstance(hi, (int, float))
        or not hi > lo + 1e-6
    ):
        raise ContractError("calibration manifest contains invalid lo/hi")
    return calibration


def _resolve_da3_root(argument: Path | None, model_dir: Path) -> Path:
    if argument is not None:
        return argument
    environment = os.environ.get("DA3_ROOT")
    if environment:
        return Path(environment)
    planned = model_dir.resolve().parent.parent / "code" / "depth-anything-3"
    if planned.is_dir():
        return planned
    raise ContractError(
        "the pinned Depth-Anything-3 checkout is required; pass --da3-root or set DA3_ROOT"
    )


def _parse_environments(value: str) -> list[str]:
    environments = [part.strip() for part in value.split(",") if part.strip()]
    if not environments or len(set(environments)) != len(environments):
        raise ContractError(
            "--build-environments must contain unique selected environments"
        )
    unsupported = sorted(set(environments) - set(SELECTED_ENVIRONMENTS))
    if unsupported:
        raise ContractError(f"unsupported build environments: {unsupported}")
    return environments


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="root containing four extracted datasets",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="depth_cache output directory"
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="pinned DA3 model directory"
    )
    parser.add_argument(
        "--da3-root", type=Path, help="pinned Depth-Anything-3 git checkout"
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument("--process-res", type=int, default=PROCESS_RES)
    parser.add_argument(
        "--build-environments",
        default=",".join(DEFAULT_BUILD_ORDER),
        help="comma-separated caches to build; calibration still spans all four selected environments",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        help="reuse the exact global calibration from an already-built cache manifest",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="quarantine and replace nonmatching LMDBs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.chunk_size, args.overlap, args.process_res) != (
        CHUNK_SIZE,
        OVERLAP,
        PROCESS_RES,
    ):
        raise ContractError(
            "this works-proof tool permits only the fixed DA3-Streaming contract: "
            "--chunk-size 120 --overlap 60 --process-res 504"
        )
    build_environments = _parse_environments(args.build_environments)
    args.out.mkdir(parents=True, exist_ok=True)
    da3_root = _resolve_da3_root(args.da3_root, args.model_dir)
    producer = OfficialDA3StreamingProducer(
        da3_root, args.model_dir, work_root=args.out / ".da3-work"
    )
    trajectories_by_env = enumerate_selected(args.root)
    with tempfile.TemporaryDirectory(
        prefix=".depth-calibration-", dir=args.out
    ) as temporary:
        if args.calibration_manifest:
            calibration = load_calibration_manifest(args.calibration_manifest, producer)
            expected_keys = select_calibration_keys(trajectories_by_env)
            if calibration.get("keys") != expected_keys:
                raise ContractError(
                    "reused calibration keys are not the current dataset's exact stratified "
                    "smallest-SHA256 selection"
                )
            calibration_raw: dict[str, tuple[Path, Mapping[str, Any]]] = {}
        else:
            calibration, calibration_raw = generate_global_calibration(
                trajectories_by_env=trajectories_by_env,
                producer=producer,
                staging_root=Path(temporary),
            )
        summaries = {}
        for environment in build_environments:
            summaries[environment] = build_environment_cache(
                output_root=args.out,
                environment=environment,
                trajectories=trajectories_by_env[environment],
                calibration=calibration,
                producer=producer,
                calibration_raw=calibration_raw,
                rebuild=args.rebuild,
            )
    print(
        json.dumps(
            {
                environment: {
                    "manifest_id": manifest["manifest_id"],
                    "frames": manifest["frame_count"],
                    "frames_per_second": manifest["build_metrics"][
                        "committed_frames_per_second"
                    ],
                    "data_mdb_sha256": manifest["data_mdb_sha256"],
                }
                for environment, manifest in summaries.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"DEPTH CACHE CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
