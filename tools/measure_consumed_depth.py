#!/usr/bin/env python3
"""Measure accepted consumed depth and paired frozen DINOcular sensitivity.

This command is intentionally read-only with respect to source caches and run
lineages.  It selects the first four sorted training trajectories and eight
fixed, uniformly spaced frames per trajectory before inspecting any values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from decord import VideoReader
from einops import rearrange

from datasets.depth_cache import DepthCacheReader, EmpiricalDepthCacheReader
from models.dinocular import DinocularEncoder


SAMPLES_PER_EPISODE = 8
EPISODES = 4
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hash(path: str | Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


def check_card(path: Path, expected_semantic_sha256: str, task: str) -> str:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual_semantic = value.get("run_card_sha256")
    if actual_semantic != expected_semantic_sha256:
        raise RuntimeError(
            f"{task} immutable card semantic hash mismatch: "
            f"expected {expected_semantic_sha256}, got {actual_semantic}"
        )
    return sha256_file(path)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fixed_frames(count: int) -> list[int]:
    if count < SAMPLES_PER_EPISODE:
        raise RuntimeError(f"trajectory has only {count} frames")
    return np.linspace(0, count - 1, SAMPLES_PER_EPISODE, dtype=np.int64).tolist()


def rgb_for(source: Path, frames: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    if source.suffix == ".pth":
        stored = torch.load(source, map_location="cpu")
        raw = stored[frames]
        decoded_frames = len(stored)
        if raw.ndim != 4:
            raise RuntimeError(f"unexpected RGB tensor shape {tuple(raw.shape)} for {source}")
        if raw.shape[1] == 3:
            rgb = raw.float() / 255.0
            stored_layout = "TCHW"
        elif raw.shape[-1] == 3:
            rgb = raw.permute(0, 3, 1, 2).float() / 255.0
            stored_layout = "THWC"
        else:
            raise RuntimeError(f"RGB channel axis is ambiguous in {tuple(raw.shape)} for {source}")
    else:
        reader = VideoReader(str(source), num_threads=1)
        raw = reader.get_batch(frames).asnumpy()
        if raw.ndim != 4 or raw.shape[-1] != 3:
            raise RuntimeError(f"unexpected RGB video shape {raw.shape} for {source}")
        rgb = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0
        decoded_frames = len(reader)
        stored_layout = "THWC video"
    source_shape = list(rgb.shape[-2:])
    # torchvision Resize(int) followed by CenterCrop(224), exactly as default_transform.
    h, w = source_shape
    scale = 224.0 / min(h, w)
    resized = (int(round(h * scale)), int(round(w * scale)))
    rgb = F.interpolate(rgb, size=resized, mode="bilinear", align_corners=False)
    top = (resized[0] - 224) // 2
    left = (resized[1] - 224) // 2
    rgb = rgb[:, :, top : top + 224, left : left + 224]
    rgb = (rgb - 0.5) / 0.5
    return rgb, {
        "decoded_source_frames": decoded_frames,
        "stored_layout": stored_layout,
        "decoded_source_hw": source_shape,
        "resize_hw": list(resized),
        "center_crop_tlbr": [top, left, top + 224, left + 224],
        "orientation_transform": "none",
        "encoder_rgb_hw": [224, 224],
    }


def encoder_from_hydra(config: dict[str, Any]) -> DinocularEncoder:
    values = dict(config["encoder"])
    values.pop("_target_", None)
    return DinocularEncoder(**values)


def effective_rank(matrix: torch.Tensor) -> float:
    centered = matrix.double() - matrix.double().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    power = singular.square()
    if float(power.sum()) == 0.0:
        return 0.0
    probability = power / power.sum()
    probability = probability[probability > 0]
    return float(torch.exp(-(probability * probability.log()).sum()))


def aggregate_depth(boundary: torch.Tensor, mask: torch.Tensor, groups: list[slice]) -> dict[str, Any]:
    flat = boundary.double().flatten()
    spatial_var = boundary.double().flatten(1).var(dim=1, unbiased=False)
    temporal = [
        boundary[group].double().var(dim=0, unbiased=False).mean() for group in groups
    ]
    span = boundary.double().flatten(1).amax(1) - boundary.double().flatten(1).amin(1)
    return {
        "finite_min": float(flat.min()),
        "finite_max": float(flat.max()),
        "quantiles": {
            f"{q:g}": float(torch.quantile(flat, q)) for q in QUANTILES
        },
        "nonfinite_fraction": float((~torch.isfinite(boundary)).float().mean()),
        "invalid_fraction": float((mask != 1).float().mean()),
        "constant_frame_fraction": float((span == 0).double().mean()),
        "near_constant_frame_fraction_std_le_1e-6": float(
            (spatial_var.sqrt() <= 1e-6).double().mean()
        ),
        "per_frame_spatial_variance": {
            "mean": float(spatial_var.mean()),
            "min": float(spatial_var.min()),
            "median": float(spatial_var.median()),
            "max": float(spatial_var.max()),
        },
        "sequence_pixelwise_temporal_variance": {
            "mean": float(torch.stack(temporal).mean()),
            "per_episode": [float(value) for value in temporal],
        },
    }


def aggregate_features(
    real: torch.Tensor, zero: torch.Tensor, groups: list[slice]
) -> dict[str, Any]:
    real_flat = real.double().flatten(1)
    zero_flat = zero.double().flatten(1)
    delta = real_flat - zero_flat
    cosine = F.cosine_similarity(real_flat, zero_flat, dim=1)
    ratio = real_flat.norm(dim=1) / zero_flat.norm(dim=1).clamp_min(1e-15)
    temporal_real = [
        real_flat[group].var(dim=0, unbiased=False).mean() for group in groups
    ]
    temporal_zero = [
        zero_flat[group].var(dim=0, unbiased=False).mean() for group in groups
    ]
    return {
        "elementwise_absolute_difference_mean": float(delta.abs().mean()),
        "elementwise_absolute_difference_max": float(delta.abs().max()),
        "elementwise_rms_difference": float(delta.square().mean().sqrt()),
        "per_frame_cosine_similarity": {
            "mean": float(cosine.mean()),
            "min": float(cosine.min()),
            "max": float(cosine.max()),
        },
        "per_frame_real_to_zero_norm_ratio": {
            "mean": float(ratio.mean()),
            "min": float(ratio.min()),
            "max": float(ratio.max()),
        },
        "temporal_variance": {
            "real_mean": float(torch.stack(temporal_real).mean()),
            "zero_mean": float(torch.stack(temporal_zero).mean()),
            "real_per_episode": [float(x) for x in temporal_real],
            "zero_per_episode": [float(x) for x in temporal_zero],
        },
        "effective_rank": {
            "real": effective_rank(real_flat),
            "zero": effective_rank(zero_flat),
            "difference": effective_rank(delta),
            "maximum_from_sample_count": real.shape[0] - 1,
        },
        "numerically_indistinguishable_at_1e-7": bool(float(delta.abs().max()) <= 1e-7),
    }


def save_montage(path: Path, rgb: torch.Tensor, depth: torch.Tensor, labels: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(12, rgb.shape[0])
    figure, axes = plt.subplots(count, 2, figsize=(7, 2.35 * count), squeeze=False)
    lo, hi = torch.quantile(
        depth.double().flatten(), torch.tensor([0.01, 0.99], dtype=torch.float64)
    ).tolist()
    if not hi > lo:
        hi = lo + 1.0
    for index in range(count):
        image = ((rgb[index] * 0.5 + 0.5).clamp(0, 1)).permute(1, 2, 0)
        axes[index, 0].imshow(image.cpu())
        axes[index, 0].set_title(labels[index])
        axes[index, 1].imshow(depth[index, 0].cpu(), cmap="viridis", vmin=lo, vmax=hi)
        axes[index, 1].set_title(f"consumed depth [{lo:.3g}, {hi:.3g}]")
        for axis in axes[index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run_task(task: str, hydra_path: Path, card_path: Path, card_sha: str, output: Path) -> dict[str, Any]:
    card_file_sha = check_card(card_path, card_sha, task)
    config = yaml.safe_load(hydra_path.read_text(encoding="utf-8"))
    dataset = config["env"]["dataset"]
    checkpoint = config["encoder"]["checkpoint_path"]
    check_hash(checkpoint, config["encoder"]["checkpoint_sha256"], f"{task} student")
    source_root = Path(dataset["data_path"]).resolve().parent
    if task == "pusht":
        reader = EmpiricalDepthCacheReader(
            environment=task,
            source_root=source_root,
            empirical_contract_path=dataset["empirical_depth_contract_path"],
            empirical_contract_sha256=dataset["empirical_depth_contract_sha256"],
            empirical_runtime_release_path=dataset["empirical_runtime_release_path"],
            empirical_runtime_release_sha256=dataset["empirical_runtime_release_sha256"],
            expected_checkpoint_sha256=dataset["depth_checkpoint_sha256"],
        )
    else:
        reader = DepthCacheReader(
            environment=task,
            source_root=source_root,
            cache_dir=dataset["depth_cache_dir"],
            cache_manifest_sha256=dataset["depth_cache_manifest_sha256"],
            validation_path=dataset["depth_validation_path"],
            validation_sha256=dataset["depth_validation_sha256"],
            native_contract_path=dataset["native_depth_contract_path"],
            native_contract_sha256=dataset["native_depth_contract_sha256"],
            expected_producer_sha256=dataset["depth_cache_producer_sha256"],
            expected_checkpoint_sha256=dataset["depth_checkpoint_sha256"],
        )

    records = sorted(
        (record for record in reader._records.values() if str(record["trajectory_key"]).startswith("train/")),
        key=lambda record: str(record["trajectory_key"]),
    )[:EPISODES]
    if len(records) != EPISODES:
        raise RuntimeError(f"{task} has only {len(records)} training trajectories")
    rgb_parts, wire_parts, mask_parts, identities, alignments = [], [], [], [], []
    groups, cursor = [], 0
    for record in records:
        split, episode_text = str(record["trajectory_key"]).split("/")
        episode = int(episode_text)
        frames = fixed_frames(int(record["ordered_frame_count"]))
        wire, mask = reader.read(split=split, episode=episode, frames=frames)
        source = (source_root / str(record["source_path"])).resolve()
        check_hash(source, str(record["source_video_sha256"]), f"{task} RGB source")
        rgb, alignment = rgb_for(source, frames)
        if tuple(wire.shape[-2:]) != tuple(rgb.shape[-2:]):
            raise RuntimeError(f"{task} RGB/depth encoder shape mismatch")
        rgb_parts.append(rgb)
        wire_parts.append(wire)
        mask_parts.append(mask)
        identities.extend(f"{record['trajectory_key']}/{frame:06d}" for frame in frames)
        alignments.append({
            "trajectory_key": record["trajectory_key"],
            "source_path": str(source),
            "source_video_sha256": record["source_video_sha256"],
            "manifest_ordered_frame_count": record["ordered_frame_count"],
            "selected_frames": frames,
            **alignment,
            "depth_encoder_hw": list(wire.shape[-2:]),
            "same_episode_and_frame_index": True,
            "resize_crop_orientation_contract_consistent": True,
        })
        groups.append(slice(cursor, cursor + len(frames)))
        cursor += len(frames)

    rgb = torch.cat(rgb_parts)
    wire = torch.cat(wire_parts)
    mask = torch.cat(mask_parts)
    encoder = encoder_from_hydra(config).cuda().eval()
    rgb_gpu, wire_gpu, mask_gpu = rgb.cuda(), wire.cuda(), mask.cuda()
    with torch.inference_mode():
        boundary, boundary_mask = encoder.prepare_depth_encoder_input(wire_gpu, mask_gpu)
        real = encoder(rgb_gpu, wire_gpu, mask_gpu)
        if task == "pusht":
            encoder.empirical_zero_intervention = True
        else:
            encoder.neutralize_depth_at_encoder_input = True
        zero = encoder(rgb_gpu, wire_gpu, mask_gpu)
        zero_boundary, zero_mask = encoder.prepare_depth_encoder_input(wire_gpu, mask_gpu)
    if torch.count_nonzero(zero_boundary).item() != 0:
        raise RuntimeError(f"{task} zero boundary is not exact zero")
    boundary_cpu = boundary.cpu()
    result = {
        "schema": "dinocular-consumed-depth-validity-v1",
        "task": task,
        "deterministic_selection_rule": (
            "first four lexicographically sorted training trajectories; "
            "eight integer linspace frames from 0 through final frame"
        ),
        "sample_count": len(identities),
        "sample_frame_identities": identities,
        "provenance": {
            "immutable_card": str(card_path),
            "immutable_card_semantic_sha256": card_sha,
            "immutable_card_file_sha256": card_file_sha,
            "hydra_config": str(hydra_path),
            "hydra_config_sha256": sha256_file(hydra_path),
            "cache_manifest": str(reader.manifest_path),
            "cache_manifest_sha256": sha256_file(reader.manifest_path),
            "validation_receipt": str(reader.validation_path),
            "validation_receipt_sha256": sha256_file(reader.validation_path),
            "checkpoint": checkpoint,
            "checkpoint_sha256": config["encoder"]["checkpoint_sha256"],
        },
        "alignment": {
            "state": "PASS",
            "same_manifest_source_episode_frame_resize_crop_orientation_shape": True,
            "episodes": alignments,
        },
        "wire_tensor": aggregate_depth(wire.unsqueeze(1), mask.unsqueeze(1), groups),
        "encoder_boundary_tensor": aggregate_depth(boundary_cpu, boundary_mask.cpu(), groups),
        "zero_boundary": {
            "exact_zero": True,
            "mask_unique": sorted(float(x) for x in torch.unique(zero_mask.cpu())),
        },
        "feature_sensitivity": aggregate_features(real.cpu(), zero.cpu(), groups),
    }
    if task == "pusht":
        result["contract"] = {
            "kind": "recovered_empirical_lossy_proxy",
            "quantity": "later MapAnything depth-z proxy",
            "units": "nonphysical proxy units",
            "wire_range": [0.0, 1.0],
            "scale_to_boundary": 1.5746406149864196,
            "normalization": "none after frozen affine reconstruction",
            "clipping": "wire clipped to [0,1] before float16 storage",
            "mask": "all ones after payload validation; zero intervention mask is zero",
            "contract_sha256": dataset["empirical_depth_contract_sha256"],
        }
    else:
        binding = reader.binding
        result["contract"] = {
            "kind": "native_depth_v1",
            "quantity": reader.native_contract.checkpoint_native_quantity,
            "units": reader.native_contract.checkpoint_native_units,
            "wire_range": [binding.wire_minimum, binding.wire_maximum],
            "scale_to_boundary": binding.raw_metric_scale,
            "offset_to_boundary": binding.raw_metric_offset,
            "normalization": reader.native_contract.normalization_kind,
            "clipping": reader.native_contract.manifest["producer"]["clipping"],
            "mask": binding.payload_validation,
            "interpolation": binding.interpolation,
            "contract_sha256": dataset["native_depth_contract_sha256"],
        }
    task_json = output / f"{task}.json"
    montage = output / f"{task}_rgb_depth.png"
    json_dump(task_json, result)
    save_montage(montage, rgb, boundary_cpu, identities)
    result["outputs"] = {
        "measurement": str(task_json),
        "visualization": str(montage),
        "visualization_sha256": sha256_file(montage),
    }
    json_dump(task_json, result)
    result["outputs"]["measurement_sha256"] = sha256_file(task_json)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task",
        action="append",
        nargs=4,
        metavar=("NAME", "HYDRA", "CARD", "CARD_SHA256"),
        required=True,
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = [
        run_task(name, Path(hydra), Path(card), card_sha, args.output)
        for name, hydra, card, card_sha in args.task
    ]
    summary = {
        "schema": "dinocular-consumed-depth-validity-summary-v1",
        "tasks": {
            result["task"]: {
                "sample_count": result["sample_count"],
                "encoder_boundary_tensor": result["encoder_boundary_tensor"],
                "feature_sensitivity": result["feature_sensitivity"],
                "contract": result["contract"],
                "alignment_state": result["alignment"]["state"],
                "outputs": result["outputs"],
            }
            for result in results
        },
    }
    json_dump(args.output / "summary.json", summary)
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    json_dump(args.output / "sha256.json", hashes)
    print(json.dumps({"output": str(args.output), "hashes": hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
