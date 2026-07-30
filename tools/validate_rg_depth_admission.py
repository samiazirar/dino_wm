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

from datasets.depth_cache import DepthCacheReader
from models.dinocular import DinocularEncoder
from tools.measure_consumed_depth import fixed_frames, rgb_for, sha256_file


CHECKPOINT_SHA256 = (
    "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc"
)
FIXED_EPISODES = 4


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
    depth: torch.Tensor,
    identities: list[str],
) -> None:
    count = min(12, len(identities))
    figure, axes = plt.subplots(count, 3, figsize=(10, 2.5 * count))
    if count == 1:
        axes = axes[None, :]
    for index in range(count):
        image = ((rgb[index] + 1.0) * 0.5).clamp(0, 1).permute(1, 2, 0)
        axes[index, 0].imshow(image)
        axes[index, 0].set_title(identities[index])
        axes[index, 1].imshow(depth[index], cmap="viridis")
        axes[index, 1].set_title("raw depth_z proxy")
        if index:
            heat = (depth[index] - depth[index - 1]).abs()
        else:
            heat = torch.zeros_like(depth[index])
        axes[index, 2].imshow(heat, cmap="magma")
        axes[index, 2].set_title("absolute temporal delta")
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    producer_validation = json.loads(args.producer_validation.read_text())
    if producer_validation.get("state") != "PASS":
        raise RuntimeError("producer validation did not pass")
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
    records = sorted(
        (
            record
            for record in reader._records.values()
            if str(record["trajectory_key"]).startswith("train/")
        ),
        key=lambda record: str(record["trajectory_key"]),
    )[:FIXED_EPISODES]
    if len(records) != FIXED_EPISODES:
        raise RuntimeError("fixed training episode coverage is incomplete")

    rgb_parts, depth_parts, mask_parts = [], [], []
    identities: list[str] = []
    episode_slices: list[slice] = []
    alignment_rows = []
    cursor = 0
    for record in records:
        split, episode_text = str(record["trajectory_key"]).split("/")
        frames = fixed_frames(int(record["ordered_frame_count"]))
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
    if not torch.isfinite(depth).all() or not torch.all(mask == 1):
        raise RuntimeError("fixed depth payload is invalid")
    spatial = depth.double().flatten(1).var(1, unbiased=False)
    temporal = [
        depth[group].double().var(0, unbiased=False).mean()
        for group in episode_slices
    ]
    depth_deltas = []
    correlations = []
    for group in episode_slices:
        indices = list(range(group.start, group.stop))
        for left, right in zip(indices, indices[1:]):
            rgb_delta = (rgb[right] - rgb[left]).abs().mean(0).double().flatten()
            depth_delta = (depth[right] - depth[left]).abs().double().flatten()
            depth_deltas.append(float(depth_delta.mean()))
            if float(rgb_delta.std(unbiased=False)) > 0 and float(
                depth_delta.std(unbiased=False)
            ) > 0:
                correlation = torch.corrcoef(
                    torch.stack((rgb_delta, depth_delta))
                )[0, 1]
                correlations.append(abs(float(correlation)))
    if not depth_deltas or not correlations:
        raise RuntimeError("fixed motion sample has no measurable variation")

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
    repeated_rate, repeated, payloads = payload_hash_rate(args.cache_dir)
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
            "state": "PASS" if repeated_rate < 1.0 else "FAIL",
            "value": repeated_rate,
            "repeated_payloads": repeated,
            "payloads": payloads,
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
            "state": "PASS"
            if min(depth_deltas) > 0 and min(correlations) > 0
            else "FAIL",
            "minimum_depth_delta": min(depth_deltas),
            "minimum_absolute_correlation": min(correlations),
            "mean_depth_delta": float(np.mean(depth_deltas)),
            "mean_absolute_correlation": float(np.mean(correlations)),
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
            "state": "PASS" if feature_rms > 0 else "FAIL",
            "minimum_rms_delta": feature_rms,
            "fixed_frame_count": len(identities),
        },
    }
    state = "PASS" if all(value["state"] == "PASS" for value in checks.values()) else "FAIL"
    visualization = args.output / f"{args.environment}_rgb_depth_delta.png"
    montage(visualization, rgb, depth, identities)
    receipt = {
        "schema": "dinocular.depth-admission-validation.v1",
        "state": state,
        "environment": args.environment,
        "fixed_selection": {
            "rule": (
                "first_four_lexicographic_training_episodes_eight_linspace_frames"
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
