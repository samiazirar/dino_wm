#!/usr/bin/env python3
"""Emit the controller admission receipt for corrected Rope/Granular depth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zstandard

from datasets.depth_cache import DepthCacheReader
from models.dinocular import DinocularEncoder
from tools.measure_consumed_depth import rgb_for, sha256_file
from tools.precompute_depth import decode_depth_value


CHECKPOINT_SHA256 = (
    "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc"
)
FIXED_EPISODES = (0, 1, 3, 4)
FIXED_FRAMES = (0, 2, 5, 8, 10, 13, 16, 19)
RGB_MOVING_THRESHOLD_UINT8 = 5.0

# Frozen before inspecting replacement values. These are deliberately relative
# to the immutable defective cache on the identical frame pairs because the old
# DA3 and recovered MapAnything proxies do not have a trustworthy common
# absolute scale.
MAX_EFFECTIVELY_REPEATED_FRACTION = 0.25
MIN_REPEATED_FRACTION_REDUCTION = 0.25
MIN_MEDIAN_NORMALIZED_MOVING_GAIN = 2.0
MIN_MEAN_NORMALIZED_MOVING_GAIN = 1.5
MIN_MEDIAN_MOVING_TO_STATIC_RATIO = 3.0
MIN_PAIR_FRACTION_MOVING_TO_STATIC_GE_2 = 0.75
MIN_ABSOLUTE_FEATURE_RMS = 1e-4
MIN_RELATIVE_FEATURE_RMS = 1e-3
MIN_CHANGED_FEATURE_FRACTION = 0.95


def canonical_sha256(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def payload_hash_rate(cache_dir: Path) -> tuple[float, int, int]:
    database = lmdb.open(
        str(cache_dir), readonly=True, lock=False, readahead=False, meminit=False
    )
    try:
        with database.begin(write=False) as transaction:
            hashes = [
                hashlib.sha256(payload).digest()
                for _, payload in transaction.cursor()
            ]
    finally:
        database.close()
    repeated = len(hashes) - len(set(hashes))
    return repeated / len(hashes), repeated, len(hashes)


def read_defective_depth(
    cache_dir: Path, selections: list[tuple[str, int, list[int]]]
) -> torch.Tensor:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    calibration = manifest.get("calibration", {})
    lo, hi = float(calibration["lo"]), float(calibration["hi"])
    database = lmdb.open(
        str(cache_dir), readonly=True, lock=False, readahead=False, meminit=False
    )
    decompressor = zstandard.ZstdDecompressor()
    parts: list[torch.Tensor] = []
    try:
        with database.begin(write=False) as transaction:
            for split, episode, frames in selections:
                arrays = []
                for frame in frames:
                    key = f"{split}/{episode:05d}/{frame:06d}".encode("ascii")
                    payload = transaction.get(key)
                    if payload is None:
                        raise RuntimeError(
                            f"defective comparison cache lacks {key.decode()}"
                        )
                    wire = decode_depth_value(payload, decompressor).astype(np.float32)
                    arrays.append(wire * (hi - lo) + lo)
                parts.append(torch.from_numpy(np.stack(arrays)))
    finally:
        database.close()
    return torch.cat(parts)


def comparative_motion_metrics(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    episode_slices: list[slice],
) -> dict[str, Any]:
    pair_rows: list[dict[str, float]] = []
    for group in episode_slices:
        indices = list(range(group.start, group.stop))
        for left, right in zip(indices, indices[1:]):
            rgb_delta_uint8 = (
                (rgb[right] - rgb[left]).abs().mean(0).double() * 127.5
            )
            moving = rgb_delta_uint8 >= RGB_MOVING_THRESHOLD_UINT8
            if not bool(moving.any()) or not bool((~moving).any()):
                raise RuntimeError("fixed pair lacks both moving and static RGB regions")
            delta = (depth[right] - depth[left]).abs().double()
            moving_mean = float(delta[moving].mean())
            static_mean = float(delta[~moving].mean())
            positive = torch.cat((depth[left].flatten(), depth[right].flatten()))
            positive = positive[positive > 0].double()
            if positive.numel() == 0:
                raise RuntimeError("fixed pair has no positive depth scale")
            scale = float(positive.median())
            if not scale > 0:
                raise RuntimeError("fixed pair depth scale is not positive")
            pair_rows.append(
                {
                    "mean_absolute_delta": float(delta.mean()),
                    "moving_mean_absolute_delta": moving_mean,
                    "static_mean_absolute_delta": static_mean,
                    "moving_to_static_ratio": (
                        moving_mean / static_mean
                        if static_mean > 0
                        else moving_mean / np.finfo(np.float64).tiny
                    ),
                    "normalized_moving_delta": moving_mean / scale,
                    "positive_depth_median": scale,
                    "moving_pixel_fraction": float(moving.double().mean()),
                }
            )
    normalized = np.asarray(
        [row["normalized_moving_delta"] for row in pair_rows], dtype=np.float64
    )
    ratios = np.asarray(
        [row["moving_to_static_ratio"] for row in pair_rows], dtype=np.float64
    )
    return {
        "pairs": pair_rows,
        "pair_count": len(pair_rows),
        "normalized_moving_delta_mean": float(normalized.mean()),
        "normalized_moving_delta_median": float(np.median(normalized)),
        "moving_to_static_ratio_median": float(np.median(ratios)),
        "pair_fraction_moving_to_static_ge_2": float(np.mean(ratios >= 2.0)),
    }


def comparative_acceptance(
    defective: dict[str, Any], replacement: dict[str, Any]
) -> dict[str, Any]:
    repeat_threshold = defective["normalized_moving_delta_median"]
    defective_repeated = float(
        np.mean(
            [
                row["normalized_moving_delta"] <= repeat_threshold
                for row in defective["pairs"]
            ]
        )
    )
    replacement_repeated = float(
        np.mean(
            [
                row["normalized_moving_delta"] <= repeat_threshold
                for row in replacement["pairs"]
            ]
        )
    )
    repeated_reduction = defective_repeated - replacement_repeated
    median_gain = (
        replacement["normalized_moving_delta_median"]
        / defective["normalized_moving_delta_median"]
    )
    mean_gain = (
        replacement["normalized_moving_delta_mean"]
        / defective["normalized_moving_delta_mean"]
    )
    checks = {
        "effective_repeated_pair_reduction": {
            "state": "PASS"
            if replacement_repeated <= MAX_EFFECTIVELY_REPEATED_FRACTION
            and repeated_reduction >= MIN_REPEATED_FRACTION_REDUCTION
            else "FAIL",
            "definition": (
                "pair normalized moving-region delta <= immutable defective "
                "median on identical fixed pairs"
            ),
            "threshold": repeat_threshold,
            "defective_fraction": defective_repeated,
            "replacement_fraction": replacement_repeated,
            "absolute_reduction": repeated_reduction,
            "maximum_replacement_fraction": MAX_EFFECTIVELY_REPEATED_FRACTION,
            "minimum_absolute_reduction": MIN_REPEATED_FRACTION_REDUCTION,
        },
        "moving_region_temporal_gain": {
            "state": "PASS"
            if median_gain >= MIN_MEDIAN_NORMALIZED_MOVING_GAIN
            and mean_gain >= MIN_MEAN_NORMALIZED_MOVING_GAIN
            else "FAIL",
            "normalization": "moving_region_mean_absolute_delta/pair_positive_depth_median",
            "defective_median": defective["normalized_moving_delta_median"],
            "replacement_median": replacement[
                "normalized_moving_delta_median"
            ],
            "median_gain": median_gain,
            "minimum_median_gain": MIN_MEDIAN_NORMALIZED_MOVING_GAIN,
            "defective_mean": defective["normalized_moving_delta_mean"],
            "replacement_mean": replacement["normalized_moving_delta_mean"],
            "mean_gain": mean_gain,
            "minimum_mean_gain": MIN_MEAN_NORMALIZED_MOVING_GAIN,
        },
        "moving_region_spatial_concentration": {
            "state": "PASS"
            if replacement["moving_to_static_ratio_median"]
            >= MIN_MEDIAN_MOVING_TO_STATIC_RATIO
            and replacement["pair_fraction_moving_to_static_ge_2"]
            >= MIN_PAIR_FRACTION_MOVING_TO_STATIC_GE_2
            else "FAIL",
            "replacement_median_moving_to_static_ratio": replacement[
                "moving_to_static_ratio_median"
            ],
            "minimum_median_ratio": MIN_MEDIAN_MOVING_TO_STATIC_RATIO,
            "replacement_pair_fraction_ratio_ge_2": replacement[
                "pair_fraction_moving_to_static_ge_2"
            ],
            "minimum_pair_fraction_ratio_ge_2": (
                MIN_PAIR_FRACTION_MOVING_TO_STATIC_GE_2
            ),
        },
    }
    return checks


def encoder(
    checkpoint: Path,
    contract: Path,
    contract_sha256: str,
    producer_sha256: str,
    environment: str,
    *,
    zero: bool,
) -> DinocularEncoder:
    return DinocularEncoder(
        name="dinocular_student_dropout_fullpr",
        backend="df2_dino_rope_convs_de",
        factory="DFormerv2_S",
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=CHECKPOINT_SHA256,
        checkpoint_key="student",
        state_prefix="module.backbone.",
        allowed_outside_prefixes=("module.dino_head.", "module.ibot_head."),
        allowed_missing_keys=(),
        feature_key="x_norm_patchtokens",
        input_size=224,
        num_patches=49,
        emb_dim=512,
        frozen=True,
        depth_contract_status="complete",
        native_depth_contract_path=str(contract),
        native_depth_contract_sha256=contract_sha256,
        selected_cache_producer_sha256=producer_sha256,
        selected_cache_environment=environment,
        neutralize_depth_at_encoder_input=zero,
    ).cuda().eval()


def montage(
    output: Path,
    rgb: torch.Tensor,
    defective_depth: torch.Tensor,
    depth: torch.Tensor,
    identities: list[str],
) -> None:
    count = len(identities)
    figure, axes = plt.subplots(count, 5, figsize=(16, 2.35 * count))
    if count == 1:
        axes = axes[None, :]
    for index in range(count):
        image = ((rgb[index] + 1.0) * 0.5).clamp(0, 1).permute(1, 2, 0)
        axes[index, 0].imshow(image)
        axes[index, 0].set_title(identities[index])
        axes[index, 1].imshow(defective_depth[index], cmap="viridis")
        axes[index, 1].set_title("defective DA3 depth")
        axes[index, 2].imshow(depth[index], cmap="viridis")
        axes[index, 2].set_title("replacement raw depth_z")
        if index:
            defective_heat = (
                defective_depth[index] - defective_depth[index - 1]
            ).abs()
            replacement_heat = (depth[index] - depth[index - 1]).abs()
        else:
            defective_heat = torch.zeros_like(defective_depth[index])
            replacement_heat = torch.zeros_like(depth[index])
        axes[index, 3].imshow(defective_heat, cmap="magma")
        axes[index, 3].set_title("defective temporal delta")
        axes[index, 4].imshow(replacement_heat, cmap="magma")
        axes[index, 4].set_title("replacement temporal delta")
        for axis in axes[index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("rope", "granular"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--producer-validation", type=Path, required=True)
    parser.add_argument("--defective-cache-dir", type=Path, required=True)
    parser.add_argument("--defect-localization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    producer_validation = json.loads(args.producer_validation.read_text())
    if producer_validation.get("state") != "PASS":
        raise RuntimeError("producer validation did not pass")
    repeated_rate, repeated, payloads = payload_hash_rate(args.cache_dir)
    reader = DepthCacheReader(
        environment=args.environment,
        source_root=args.source_root,
        cache_dir=args.cache_dir,
        cache_manifest_sha256=args.manifest_sha256,
        validation_path=args.validation,
        validation_sha256=args.validation_sha256,
        native_contract_path=args.contract,
        native_contract_sha256=args.contract_sha256,
        expected_producer_sha256=args.producer_sha256,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
    )
    records_by_episode = {
        int(str(record["trajectory_key"]).split("/")[1]): record
        for record in reader._records.values()
        if str(record["trajectory_key"]).startswith("train/")
    }
    if any(episode not in records_by_episode for episode in FIXED_EPISODES):
        raise RuntimeError("fixed training episode coverage is incomplete")
    records = [records_by_episode[episode] for episode in FIXED_EPISODES]

    rgb_parts, depth_parts, mask_parts = [], [], []
    identities: list[str] = []
    selections: list[tuple[str, int, list[int]]] = []
    episode_slices: list[slice] = []
    alignment_rows = []
    cursor = 0
    for record in records:
        split, episode_text = str(record["trajectory_key"]).split("/")
        if int(record["ordered_frame_count"]) <= max(FIXED_FRAMES):
            raise RuntimeError("fixed training episode is shorter than frame 19")
        frames = list(FIXED_FRAMES)
        depth, mask = reader.read(
            split=split, episode=int(episode_text), frames=frames
        )
        source = (args.source_root / str(record["source_path"])).resolve()
        rgb, alignment = rgb_for(source, frames)
        rgb_parts.append(rgb)
        depth_parts.append(depth)
        mask_parts.append(mask)
        identities.extend(
            f"{record['trajectory_key']}/{frame:06d}" for frame in frames
        )
        episode_slices.append(slice(cursor, cursor + len(frames)))
        selections.append((split, int(episode_text), frames))
        cursor += len(frames)
        alignment_rows.append(
            {
                "trajectory_key": record["trajectory_key"],
                "source_path": str(source),
                "source_sha256": sha256_file(source),
                "manifest_source_sha256": record["source_video_sha256"],
                "selected_frames": frames,
                "depth_hw": list(depth.shape[-2:]),
                **alignment,
            }
        )

    rgb = torch.cat(rgb_parts)
    depth = torch.cat(depth_parts)
    mask = torch.cat(mask_parts)
    defective_depth = read_defective_depth(args.defective_cache_dir, selections)
    if not torch.isfinite(depth).all() or not torch.all(mask == 1):
        raise RuntimeError("fixed depth payload is invalid")
    if defective_depth.shape != depth.shape or not torch.isfinite(
        defective_depth
    ).all():
        raise RuntimeError("defective comparison depth is invalid or misaligned")
    spatial = depth.double().flatten(1).var(1, unbiased=False)
    temporal = [
        depth[group].double().var(0, unbiased=False).mean()
        for group in episode_slices
    ]
    defective_motion = comparative_motion_metrics(
        rgb, defective_depth, episode_slices
    )
    replacement_motion = comparative_motion_metrics(rgb, depth, episode_slices)
    comparative_checks = comparative_acceptance(
        defective_motion, replacement_motion
    )
    localization = json.loads(args.defect_localization.read_text())
    localization_environment = next(
        (
            item
            for item in localization.get("environments", [])
            if item.get("environment") == args.environment
        ),
        None,
    )
    if localization_environment is None:
        raise RuntimeError("defect-localization receipt lacks this environment")
    if localization_environment.get("selection") != {
        "episodes": list(FIXED_EPISODES),
        "frames": list(FIXED_FRAMES),
        "selection_fixed_before_values": True,
    }:
        raise RuntimeError("defect-localization fixed selection differs")

    real_encoder = encoder(
        args.checkpoint,
        args.contract,
        args.contract_sha256,
        args.producer_sha256,
        args.environment,
        zero=False,
    )
    with torch.inference_mode():
        real = real_encoder(rgb.cuda(), depth.cuda(), mask.cuda())
        real_encoder.neutralize_depth_at_encoder_input = True
        zero = real_encoder(rgb.cuda(), depth.cuda(), mask.cuda())
        zero_boundary, _ = real_encoder.prepare_depth_encoder_input(
            depth.cuda(), mask.cuda()
        )
    if torch.count_nonzero(zero_boundary).item() != 0:
        raise RuntimeError("zero intervention is not exact")
    feature_rms = float(
        (real.double() - zero.double()).square().mean().sqrt().cpu()
    )
    zero_feature_rms = float(zero.double().square().mean().sqrt().cpu())
    relative_feature_rms = (
        feature_rms / zero_feature_rms if zero_feature_rms > 0 else 0.0
    )
    feature_absolute_delta = (real.double() - zero.double()).abs()
    changed_feature_fraction = float(
        (feature_absolute_delta >= MIN_ABSOLUTE_FEATURE_RMS)
        .double()
        .mean()
        .cpu()
    )
    observed_minimum = float(depth.min())
    observed_maximum = float(depth.max())
    exact_alignment = all(
        row["source_sha256"] == row["manifest_source_sha256"]
        and row["selected_frames"] == sorted(row["selected_frames"])
        and row["depth_hw"] == [224, 224]
        for row in alignment_rows
    )
    checks = {
        "repeated_frame_hash_rate": {
            "state": "PASS" if repeated == 0 else "FAIL",
            "value": repeated_rate,
            "repeated_payloads": repeated,
            "payloads": payloads,
            "acceptance": "zero exact duplicate payloads over the cache",
        },
        "finite_invalid_fraction": {
            "state": "PASS",
            "nonfinite_fraction": 0.0,
            "invalid_fraction": float((mask != 1).double().mean()),
            "producer_invalid_policy": (
                "reject_nonfinite_or_negative_preserve_upstream_masked_exact_zero"
            ),
        },
        "spatial_variance": {
            "state": "PASS" if float(spatial.min()) > 0 else "FAIL",
            "minimum": float(spatial.min()),
            "values": [float(value) for value in spatial],
        },
        "temporal_variance": {
            "state": "PASS" if min(map(float, temporal)) > 0 else "FAIL",
            "minimum": min(map(float, temporal)),
            "per_episode": list(map(float, temporal)),
        },
        "moving_object_depth_correlation": {
            "state": (
                "PASS"
                if all(
                    check["state"] == "PASS"
                    for check in comparative_checks.values()
                )
                else "FAIL"
            ),
            "scientific_basis": (
                "identical fixed moving episodes; RGB-change masks and "
                "moving-to-static measurements match diagnose_rg_depth.py; "
                "depth deltas are normalized by each pair's positive median "
                "because producer proxies lack a common trusted absolute scale"
            ),
            "defective": defective_motion,
            "replacement": replacement_motion,
            "criteria": comparative_checks,
        },
        "rgb_depth_frame_alignment": {
            "state": "PASS" if exact_alignment else "FAIL",
            "exact": exact_alignment,
            "episodes": alignment_rows,
        },
        "expected_scale_range": {
            "state": "PASS"
            if 0.0 <= observed_minimum < observed_maximum <= 65504.0
            else "FAIL",
            "observed_minimum": observed_minimum,
            "observed_maximum": observed_maximum,
            "expected_minimum": 0.0,
            "expected_maximum": 65504.0,
            "quantity": "later_pinned_MapAnything_depth_z_proxy",
            "units": "declared_metric_Z_depth_nonphysical_later_checkpoint_proxy",
        },
        "real_zero_feature_response": {
            "state": "PASS"
            if feature_rms >= MIN_ABSOLUTE_FEATURE_RMS
            and relative_feature_rms >= MIN_RELATIVE_FEATURE_RMS
            and changed_feature_fraction >= MIN_CHANGED_FEATURE_FRACTION
            else "FAIL",
            "absolute_rms_delta": feature_rms,
            "minimum_absolute_rms_delta": MIN_ABSOLUTE_FEATURE_RMS,
            "zero_feature_rms": zero_feature_rms,
            "relative_rms_delta": relative_feature_rms,
            "minimum_relative_rms_delta": MIN_RELATIVE_FEATURE_RMS,
            "changed_feature_fraction_at_absolute_1e-4": changed_feature_fraction,
            "minimum_changed_feature_fraction": MIN_CHANGED_FEATURE_FRACTION,
            "fixed_frame_count": len(identities),
        },
    }
    state = "PASS" if all(value["state"] == "PASS" for value in checks.values()) else "FAIL"
    visualization = args.output / f"{args.environment}_rgb_depth_delta.png"
    montage(visualization, rgb, defective_depth, depth, identities)
    receipt = {
        "schema": "dinocular.depth-admission-validation.v1",
        "state": state,
        "environment": args.environment,
        "fixed_selection": {
            "rule": (
                "episodes_0_1_3_4_frames_0_2_5_8_10_13_16_19_frozen_before_values"
            ),
            "frame_identities": identities,
        },
        "artifacts": {
            "cache_manifest_sha256": args.manifest_sha256,
            "reader_validation_sha256": args.validation_sha256,
            "producer_validation_sha256": sha256_file(args.producer_validation),
            "native_contract_sha256": args.contract_sha256,
            "producer_sha256": args.producer_sha256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "defective_cache_manifest_sha256": sha256_file(
                args.defective_cache_dir / "manifest.json"
            ),
            "defect_localization_sha256": sha256_file(args.defect_localization),
            "visualization": str(visualization),
            "visualization_sha256": sha256_file(visualization),
        },
        "checks": checks,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(args.output / "admission.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if state != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
