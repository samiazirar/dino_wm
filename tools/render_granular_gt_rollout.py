#!/usr/bin/env python3
"""Render one exact Granular RGB/metric-simulator-depth rollout.

The renderer reads only the released RGB ``obses.pth`` and matching simulator
HDF5 frame records.  It validates the complete selected episode before making
the figure, including HDF5 RGB identity, uint16 metric-depth records, and the
causal action carried by each frame.  The depth image is a monotone display of
the source millimetre values; it is not a predicted or reconstructed panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec


CAMERA = "cam_1"
MODEL_FRAMES = 20
IMAGE_SHAPE = (224, 224)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_rgb_frames(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        for key in ("obses", "observations", "rgb", "images"):
            if key in value:
                value = value[key]
                break
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 4:
        raise ValueError(f"expected four-dimensional RGB observations, got {array.shape}")
    if array.shape[-1] == 3:
        return array
    if array.shape[1] == 3:
        return np.transpose(array, (0, 2, 3, 1))
    raise ValueError(f"could not identify RGB channel axis in {array.shape}")


def rgb_source_path(project: Path, episode: int) -> Path:
    return project / "data" / "raw" / "deformable" / "granular" / f"{episode:06d}" / "obses.pth"


def h5_source_path(project: Path, episode: int, frame: int) -> Path:
    return (
        project
        / "data"
        / "raw"
        / "deformable"
        / "granular"
        / f"{episode:06d}"
        / f"{frame:02d}.h5"
    )


def _single_frame(dataset: h5py.Dataset) -> np.ndarray:
    value = np.asarray(dataset)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value


def compact_action(action: np.ndarray) -> str:
    values = np.asarray(action).reshape(-1)
    return "[" + ", ".join(f"{float(value):.3g}" for value in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--accepted-measurement-sha256", required=True)
    return parser.parse_args()


def validate_and_load(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.frame < MODEL_FRAMES:
        raise ValueError(f"anchor frame must be in [0, {MODEL_FRAMES}), got {args.frame}")

    manifest_sha256 = sha256_file(args.manifest)
    if manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError(
            "accepted RGB manifest hash mismatch: "
            f"expected {args.expected_manifest_sha256}, got {manifest_sha256}"
        )
    manifest = json.loads(args.manifest.read_text())
    relative = f"deformable/granular/{args.episode:06d}/obses.pth"
    manifest_item = next(
        (item for item in manifest.get("files", []) if item.get("relative_path") == relative),
        None,
    )
    if manifest_item is None:
        raise ValueError(f"episode is absent from the accepted RGB manifest: {relative}")

    rgb_path = rgb_source_path(args.project_root, args.episode)
    actions_path = args.project_root / "data" / "raw" / "deformable" / "granular" / "actions.pth"
    rgb_sha256 = sha256_file(rgb_path)
    if manifest_item.get("sha256") and rgb_sha256 != manifest_item["sha256"]:
        raise ValueError(
            "released RGB file hash mismatch: "
            f"manifest {manifest_item['sha256']}, observed {rgb_sha256}"
        )
    actions_item = next(
        (item for item in manifest.get("files", []) if item.get("relative_path") == "deformable/granular/actions.pth"),
        None,
    )
    actions_sha256 = sha256_file(actions_path)
    if actions_item is None or actions_item.get("sha256") != actions_sha256:
        raise ValueError(
            "released action source hash mismatch: "
            f"manifest {actions_item.get('sha256') if actions_item else None}, observed {actions_sha256}"
        )
    rgb_frames = as_rgb_frames(load_torch(rgb_path))
    actions = load_torch(actions_path)
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions)
    if rgb_frames.shape != (MODEL_FRAMES, *IMAGE_SHAPE, 3):
        raise ValueError(f"unexpected RGB shape for {relative}: {rgb_frames.shape}")
    if not np.isfinite(rgb_frames).all() or rgb_frames.min() < 0 or rgb_frames.max() > 255:
        raise ValueError("RGB source has non-finite or out-of-range values")

    records: list[dict[str, Any]] = []
    depths: list[np.ndarray] = []
    for frame in range(MODEL_FRAMES + 1):
        path = h5_source_path(args.project_root, args.episode, frame)
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            color = _single_frame(handle[f"observations/color/{CAMERA}"])
            depth = _single_frame(handle[f"observations/depth/{CAMERA}"])
            action = np.asarray(handle["action"])
        if frame < MODEL_FRAMES:
            if color.shape != rgb_frames[frame].shape or not np.array_equal(color, rgb_frames[frame]):
                raise ValueError(f"HDF5 RGB differs from released RGB at frame {frame}")
            if color.tobytes() != rgb_frames[frame].tobytes():
                raise ValueError(f"HDF5 RGB bytes differ from released RGB at frame {frame}")
            if depth.shape != IMAGE_SHAPE or depth.dtype != np.dtype("uint16"):
                raise ValueError(f"invalid metric depth record at frame {frame}: {depth.shape} {depth.dtype}")
            if not np.isfinite(depth).all():
                raise ValueError(f"non-finite metric depth at frame {frame}")
            depths.append(depth)
        expected_action = np.zeros_like(action) if frame == 0 else actions[args.episode, frame - 1]
        if action.shape != expected_action.shape or not np.array_equal(action, expected_action):
            raise ValueError(
                f"causal action mismatch at HDF5 frame {frame}: "
                f"expected {expected_action.tolist()}, got {action.tolist()}"
            )
        records.append(
            {
                "frame": frame,
                "h5_path": str(path),
                "h5_sha256": sha256_file(path),
                "action": action.tolist(),
            }
        )

    depth_stack = np.stack(depths)
    return {
        "manifest": manifest,
        "manifest_item": manifest_item,
        "manifest_sha256": manifest_sha256,
        "rgb_path": rgb_path,
        "rgb_sha256": rgb_sha256,
        "actions_path": actions_path,
        "actions_sha256": actions_sha256,
        "rgb_frames": rgb_frames,
        "depths": depths,
        "records": records,
        "depth_min_mm": int(depth_stack.min()),
        "depth_max_mm": int(depth_stack.max()),
        "rgb_exact_count": MODEL_FRAMES,
        "depth_valid_count": MODEL_FRAMES,
        "action_exact_count": MODEL_FRAMES + 1,
    }


def render(args: argparse.Namespace, data: dict[str, Any]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = data["rgb_frames"]
    depths = data["depths"]
    frame = args.frame
    display_min = data["depth_min_mm"]
    display_max = data["depth_max_mm"]
    if display_min == display_max:
        display_max = display_min + 1

    figure = plt.figure(figsize=(19, 11), facecolor="white")
    layout = gridspec.GridSpec(
        3,
        1,
        figure=figure,
        height_ratios=(1.55, 0.04, 1.0),
        hspace=0.32,
        top=0.82,
        bottom=0.08,
        left=0.035,
        right=0.965,
    )
    main = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=layout[0], wspace=0.08)
    ax_rgb = figure.add_subplot(main[0, 0])
    ax_depth = figure.add_subplot(main[0, 1])
    ax_rgb.imshow(np.clip(np.rint(rgb_frames[frame]), 0, 255).astype(np.uint8))
    ax_rgb.set_title(f"RGB · released source · camera {CAMERA}\nobservation frame f={frame}", fontsize=14)
    ax_rgb.axis("off")
    depth_image = ax_depth.imshow(
        depths[frame],
        cmap="magma",
        vmin=display_min,
        vmax=display_max,
        interpolation="nearest",
    )
    ax_depth.set_title(
        f"Metric simulator ground-truth depth · {CAMERA}\n"
        "source uint16 values displayed linearly in millimetres",
        fontsize=14,
    )
    ax_depth.axis("off")
    colorbar = figure.colorbar(depth_image, ax=ax_depth, fraction=0.046, pad=0.02)
    colorbar.set_label("depth (mm); monotone display only", fontsize=11)

    context_frames = min(MODEL_FRAMES, max(frame + 3, 8))
    context = gridspec.GridSpecFromSubplotSpec(
        1,
        2 * context_frames - 1,
        subplot_spec=layout[2],
        width_ratios=[1 if i % 2 == 0 else 0.42 for i in range(2 * context_frames - 1)],
        wspace=0.02,
    )
    for idx in range(context_frames):
        axis = figure.add_subplot(context[0, 2 * idx])
        axis.imshow(np.clip(np.rint(rgb_frames[idx]), 0, 255).astype(np.uint8))
        axis.axis("off")
        color = "#d62728" if idx == frame else "#333333"
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2.5 if idx == frame else 0.8)
        axis.set_title(f"f={idx}", fontsize=9, color=color, pad=3)
        if idx < context_frames - 1:
            arrow = figure.add_subplot(context[0, 2 * idx + 1])
            arrow.axis("off")
            action = np.asarray(data["records"][idx + 1]["action"])
            arrow.text(
                0.5,
                0.56,
                "→",
                ha="center",
                va="center",
                fontsize=14,
                color="#555555",
            )
            arrow.text(
                0.5,
                0.25,
                f"a{idx}\n{compact_action(action)}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="#333333",
            )

    figure.suptitle(
        "Granular · exact accepted RGB + simulator ground-truth depth · causal rollout",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    split = data["manifest_item"].get("split", "unknown")
    figure.text(
        0.5,
        0.94,
        f"episode {args.episode:06d} ({split}) · anchor f={frame} · incoming action a{max(frame - 1, 0)} is stored on HDF5 frame f={frame} · displayed causal context f=0..{context_frames - 1} · no proxy panel",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )
    figure.text(
        0.5,
        0.040,
        "Source: released obses.pth = HDF5 observations/color/cam_1 for f=0..19 · "
        "HDF5 f>=1 stores action a(f−1); f=20 stores a19.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#444444",
    )
    figure.text(
        0.5,
        0.018,
        f"Depth: source uint16 displayed monotonically in mm · selected-episode range "
        f"{data['depth_min_mm']:,}–{data['depth_max_mm']:,} mm · no predicted or inferred depth.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#444444",
    )
    figure.savefig(args.output, dpi=170, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    data = validate_and_load(args)
    render(args, data)
    print(
        json.dumps(
            {
                "status": "success",
                "artifact": {
                    "path": str(args.output),
                    "sha256": sha256_file(args.output),
                    "bytes": args.output.stat().st_size,
                },
                "episode": args.episode,
                "anchor_frame": args.frame,
                "split": data["manifest_item"].get("split"),
                "camera": CAMERA,
                "rgb_source": {
                    "path": str(data["rgb_path"]),
                    "sha256": data["rgb_sha256"],
                    "manifest_sha256": data["manifest_sha256"],
                    "manifest_item": data["manifest_item"],
                    "h5_rgb_exact_count": data["rgb_exact_count"],
                },
                "depth_source": {
                    "record_pattern": str(h5_source_path(args.project_root, args.episode, 0).parent / "{00..20}.h5"),
                    "dtype": "uint16",
                    "units": "millimetres; simulator meters multiplied by 1000 at release",
                    "valid_count": data["depth_valid_count"],
                    "selected_episode_range_mm": [data["depth_min_mm"], data["depth_max_mm"]],
                    "anchor_h5_sha256": data["records"][args.frame]["h5_sha256"],
                },
                "causal_alignment": {
                    "rule": "HDF5 frame f=0 carries zero action; frame f>=1 carries raw action a(f-1)",
                    "checked_h5_frames": MODEL_FRAMES + 1,
                    "exact_action_count": data["action_exact_count"],
                    "actions_path": str(data["actions_path"]),
                    "actions_sha256": data["actions_sha256"],
                    "future_leakage": 0,
                },
                "accepted_depth_measurement_sha256": args.accepted_measurement_sha256,
                "proxy_panel": "omitted; no proxy source is displayed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
