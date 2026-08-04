#!/usr/bin/env python3
"""Render one exact PushT RGB/MapAnything-proxy validation observation.

The figure is built from the released PushT validation video and the exact
framewise proxy payload read by the active DINOcular cache.  The cache values
are displayed as their normalized wire values; they are an estimated
MapAnything proxy and are not simulator, metric, or physical ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import decord
import lmdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import zstandard
from matplotlib import gridspec


IMAGE_SHAPE = (224, 224)
WIRE_MINIMUM = 0.0
WIRE_MAXIMUM = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compact_action(action: np.ndarray) -> str:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{float(value):.3g}" for value in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-validation", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--expected-cache-validation-sha256", required=True)
    parser.add_argument("--expected-cache-data-sha256", required=True)
    parser.add_argument("--expected-source-index-sha256", required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    return parser.parse_args()


def source_video_path(project_root: Path, split: str, episode: int) -> Path:
    raw_split = "val" if split == "valid" else split
    return project_root / "data" / "raw" / "pusht_noise" / raw_split / "obses" / f"episode_{episode:03d}.mp4"


def action_path(project_root: Path, split: str) -> Path:
    raw_split = "val" if split == "valid" else split
    return project_root / "data" / "raw" / "pusht_noise" / raw_split / "rel_actions.pth"


def trajectory_key(split: str, episode: int) -> str:
    return f"{split}/{episode:05d}"


def validate_contract(args: argparse.Namespace) -> dict[str, Any]:
    actual = sha256_file(args.contract)
    if actual != args.expected_contract_sha256:
        raise ValueError(f"accepted proxy contract hash mismatch: {actual} != {args.expected_contract_sha256}")
    contract = load_json(args.contract)
    if contract.get("schema") != "dinocular-empirical-cache-consumption-contract-v1":
        raise ValueError("unexpected PushT proxy contract schema")
    cache = contract.get("cache")
    if not isinstance(cache, dict):
        raise ValueError("proxy contract has no cache binding")
    if cache.get("manifest_sha256") != args.expected_cache_manifest_sha256:
        raise ValueError("proxy contract/cache manifest hash mismatch")
    if cache.get("validation", {}).get("sha256") != args.expected_cache_validation_sha256:
        raise ValueError("proxy contract/cache validation hash mismatch")
    if cache.get("data_sha256") != args.expected_cache_data_sha256:
        raise ValueError("proxy contract/cache data hash mismatch")
    if cache.get("source_index_sha256") != args.expected_source_index_sha256:
        raise ValueError("proxy contract/cache source-index hash mismatch")
    if contract.get("status") != "complete" or contract.get("empirical_use_allowed") is not True:
        raise ValueError("PushT proxy contract is not an accepted empirical contract")
    return contract


def validate_cache(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = args.cache_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != args.expected_cache_manifest_sha256:
        raise ValueError(f"accepted cache manifest hash mismatch: {manifest_sha256} != {args.expected_cache_manifest_sha256}")
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "dinocular-depth-cache-v1" or manifest.get("environment") != "pusht":
        raise ValueError("unexpected PushT cache manifest")
    if manifest.get("source_index_sha256") != args.expected_source_index_sha256:
        raise ValueError("cache source-index hash mismatch")
    data_path = args.cache_dir / "data.mdb"
    data_sha256 = sha256_file(data_path)
    if data_sha256 != args.expected_cache_data_sha256 or manifest.get("data_mdb_sha256") != data_sha256:
        raise ValueError("cache data.mdb hash mismatch")

    validation_sha256 = sha256_file(args.cache_validation)
    if validation_sha256 != args.expected_cache_validation_sha256:
        raise ValueError(f"accepted cache validation hash mismatch: {validation_sha256} != {args.expected_cache_validation_sha256}")
    validation = load_json(args.cache_validation)
    if validation.get("schema") != "dinocular-mapanything-cache-validation-v1":
        raise ValueError("unexpected cache validation schema")
    if validation.get("state") != "PASS" or validation.get("manifest_id") != manifest.get("manifest_id"):
        raise ValueError("cache validation does not accept this exact cache")

    key = trajectory_key(args.split, args.episode)
    record = next((item for item in manifest.get("trajectories", []) if item.get("trajectory_key") == key), None)
    if not isinstance(record, dict):
        raise ValueError(f"cache trajectory is absent: {key}")
    expected_source = f"pusht_noise/{'val' if args.split == 'valid' else args.split}/obses/episode_{args.episode:03d}.mp4"
    if record.get("source_path") != expected_source:
        raise ValueError(f"cache source path mismatch: {record.get('source_path')} != {expected_source}")
    return manifest, validation, record


def load_source_and_actions(args: argparse.Namespace, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    video_path = source_video_path(args.project_root, args.split, args.episode)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    video_sha256 = sha256_file(video_path)
    if video_sha256 != record.get("source_video_sha256"):
        raise ValueError(f"released RGB source hash mismatch: {video_sha256} != {record.get('source_video_sha256')}")
    reader = decord.VideoReader(str(video_path), num_threads=1)
    frame_count = len(reader)
    expected_count = int(record.get("ordered_frame_count", -1))
    if frame_count != expected_count:
        raise ValueError(f"released RGB frame count mismatch: {frame_count} != {expected_count}")
    if not 0 <= args.frame < frame_count:
        raise ValueError(f"anchor frame must be in [0, {frame_count}), got {args.frame}")
    if args.context_frames < 2:
        raise ValueError("context must contain at least two frames")
    context_start = max(0, args.frame - args.context_frames + 1)
    context_ids = list(range(context_start, args.frame + 1))
    rgb_frames = reader.get_batch(context_ids).asnumpy()
    if rgb_frames.shape != (len(context_ids), *IMAGE_SHAPE, 3) or rgb_frames.dtype != np.uint8:
        raise ValueError(f"unexpected released RGB shape/dtype: {rgb_frames.shape} {rgb_frames.dtype}")

    actions_source = action_path(args.project_root, args.split)
    actions_sha256 = sha256_file(actions_source)
    actions = load_torch(actions_source)
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions)
    if actions.ndim != 3 or actions.shape[0] <= args.episode or actions.shape[2] != 2:
        raise ValueError(f"unexpected released action shape: {actions.shape}")
    if actions.shape[1] < frame_count or not np.isfinite(actions[args.episode, :frame_count]).all():
        raise ValueError("released action source does not cover the selected RGB frames")
    source = {
        "path": str(video_path),
        "sha256": video_sha256,
        "frame_count": frame_count,
        "context_frame_ids": context_ids,
        "context_frame_sha256": [sha256_bytes(frame.tobytes()) for frame in rgb_frames],
        "actions_path": str(actions_source),
        "actions_sha256": actions_sha256,
    }
    return rgb_frames, actions[args.episode], source


def read_proxy(args: argparse.Namespace, record: dict[str, Any], frame_ids: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
    key_prefix = record["trajectory_key"]
    expected_keys = [f"{key_prefix}/{frame:06d}" for frame in frame_ids]
    values: list[np.ndarray] = []
    payload_records: list[dict[str, Any]] = []
    decompressor = zstandard.ZstdDecompressor()
    database = lmdb.open(str(args.cache_dir), readonly=True, lock=False, readahead=False, meminit=False, max_readers=64)
    try:
        with database.begin(write=False) as transaction:
            for physical_key in expected_keys:
                payload = transaction.get(physical_key.encode("ascii"))
                if payload is None:
                    raise ValueError(f"missing exact proxy cache key: {physical_key}")
                raw = decompressor.decompress(payload)
                value = np.frombuffer(raw, dtype=np.dtype("<f2"))
                if value.size != IMAGE_SHAPE[0] * IMAGE_SHAPE[1]:
                    raise ValueError(f"wrong proxy payload size at {physical_key}: {value.size}")
                value = value.reshape(IMAGE_SHAPE)
                if not np.isfinite(value).all() or value.min() < WIRE_MINIMUM or value.max() > WIRE_MAXIMUM:
                    raise ValueError(f"invalid proxy payload range at {physical_key}")
                values.append(value.astype(np.float32, copy=True))
                payload_records.append(
                    {
                        "key": physical_key,
                        "compressed_sha256": sha256_bytes(payload),
                        "wire_sha256": sha256_bytes(raw),
                        "wire_dtype": "<f2",
                        "wire_shape": list(IMAGE_SHAPE),
                        "wire_min": float(value.min()),
                        "wire_max": float(value.max()),
                    }
                )
    finally:
        database.close()
    return np.stack(values), {"keys": expected_keys, "payloads": payload_records}


def render(args: argparse.Namespace, rgb_frames: np.ndarray, actions: np.ndarray, proxy_values: np.ndarray, source: dict[str, Any], cache: dict[str, Any], record: dict[str, Any]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context_ids = source["context_frame_ids"]
    anchor = len(context_ids) - 1
    figure = plt.figure(figsize=(19, 11), facecolor="white")
    layout = gridspec.GridSpec(2, 1, figure=figure, height_ratios=(1.55, 1.0), hspace=0.33, top=0.81, bottom=0.10, left=0.035, right=0.965)
    main = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=layout[0], wspace=0.08)
    ax_rgb = figure.add_subplot(main[0, 0])
    ax_proxy = figure.add_subplot(main[0, 1])
    ax_rgb.imshow(rgb_frames[anchor], interpolation="nearest")
    ax_rgb.set_title(f"Released RGB source · validation episode {args.episode:05d}\nobservation frame f={args.frame}", fontsize=14)
    ax_rgb.axis("off")
    proxy_image = ax_proxy.imshow(proxy_values[anchor], cmap="magma", vmin=WIRE_MINIMUM, vmax=WIRE_MAXIMUM, interpolation="nearest")
    ax_proxy.set_title("Estimated MapAnything proxy depth consumed by DINOcular\nnormalized cache wire value; no physical/metric units", fontsize=14)
    ax_proxy.axis("off")
    colorbar = figure.colorbar(proxy_image, ax=ax_proxy, fraction=0.046, pad=0.02)
    colorbar.set_label("accepted proxy cache value [0, 1]", fontsize=11)

    context = gridspec.GridSpecFromSubplotSpec(1, 2 * len(context_ids) - 1, subplot_spec=layout[1], width_ratios=[1 if i % 2 == 0 else 0.48 for i in range(2 * len(context_ids) - 1)], wspace=0.02)
    for index, frame_id in enumerate(context_ids):
        axis = figure.add_subplot(context[0, 2 * index])
        axis.imshow(rgb_frames[index], interpolation="nearest")
        axis.axis("off")
        color = "#d62728" if index == anchor else "#333333"
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2.5 if index == anchor else 0.8)
        axis.set_title(f"f={frame_id}", fontsize=9, color=color, pad=3)
        if index < len(context_ids) - 1:
            arrow = figure.add_subplot(context[0, 2 * index + 1])
            arrow.axis("off")
            arrow.text(0.5, 0.58, "→", ha="center", va="center", fontsize=14, color="#555555")
            arrow.text(0.5, 0.23, f"a{frame_id}\n{compact_action(actions[frame_id])}", ha="center", va="center", fontsize=7.2, color="#333333")

    figure.suptitle("PushT · exact released RGB + estimated MapAnything proxy depth · causal validation rollout", fontsize=20, fontweight="bold", y=0.975)
    figure.text(0.5, 0.925, f"{record['trajectory_key']} · anchor f={args.frame} · displayed RGB context f={context_ids[0]}..{args.frame} · actions a{context_ids[0]}..a{args.frame - 1} only · no future observation or action", ha="center", va="center", fontsize=11, color="#333333")
    figure.text(0.5, 0.052, "Depth panel: exact accepted framewise MapAnything-derived proxy payload consumed by the current DINOcular condition; normalized cache wire values only.", ha="center", va="bottom", fontsize=8.5, color="#444444")
    figure.text(0.5, 0.030, "This proxy is estimated and recovered-contract qualified; it is not simulator, metric, or physical ground-truth depth, and no pixels or depth were synthesized.", ha="center", va="bottom", fontsize=8.5, color="#444444")
    figure.savefig(args.output, dpi=170, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.episode < 0 or args.frame < 0:
        raise ValueError("episode and frame must be non-negative")
    contract = validate_contract(args)
    manifest, validation, record = validate_cache(args)
    rgb_frames, actions, source = load_source_and_actions(args, record)
    frame_ids = source["context_frame_ids"]
    proxy_values, proxy = read_proxy(args, record, frame_ids)
    render(args, rgb_frames, actions, proxy_values, source, manifest, record)
    output_sha256 = sha256_file(args.output)
    print(json.dumps({
        "status": "success",
        "artifact": {"path": str(args.output), "sha256": output_sha256, "bytes": args.output.stat().st_size},
        "selected": {"split": args.split, "episode": args.episode, "anchor_frame": args.frame, "trajectory_key": record["trajectory_key"]},
        "rgb_source": source,
        "proxy_source": {
            "cache_directory": str(args.cache_dir),
            "manifest_id": manifest.get("manifest_id"),
            "manifest_sha256": args.expected_cache_manifest_sha256,
            "validation_sha256": args.expected_cache_validation_sha256,
            "data_mdb_sha256": args.expected_cache_data_sha256,
            "source_index_sha256": args.expected_source_index_sha256,
            "contract_sha256": args.expected_contract_sha256,
            "cache_record_source_path": record.get("source_path"),
            "cache_record_source_video_sha256": record.get("source_video_sha256"),
            "checked_keys": proxy["keys"],
            "payloads": proxy["payloads"],
            "displayed_quantity": "normalized little-endian float16 cache wire value cast to float32 for display",
            "physical_or_metric_depth_claim": False,
            "simulator_ground_truth_claim": False,
        },
        "causal_alignment": {
            "action_rule": "released rel_actions[episode, f] is shown on the arrow from RGB observation f to f+1",
            "context_frame_ids": frame_ids,
            "displayed_action_ids": list(range(frame_ids[0], args.frame)),
            "future_observation_count": 0,
            "future_action_count": 0,
        },
        "contract_disclosure": contract.get("scientific_scope", {}).get("non_equivalence_statement"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
