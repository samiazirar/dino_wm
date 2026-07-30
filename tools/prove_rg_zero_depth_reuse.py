#!/usr/bin/env python3
"""Prove exact zero-depth reuse across old and corrected cache identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf
import torch

from datasets.depth_cache import DepthCacheReader
from models.dinocular import DinocularEncoder
from tools.measure_consumed_depth import fixed_frames, rgb_for, sha256_file


CHECKPOINT_SHA256 = (
    "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc"
)


def decision_sha256(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("decision_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def encoder(
    checkpoint: Path,
    contract: Path,
    contract_sha256: str,
    producer_sha256: str,
    environment: str,
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
        neutralize_depth_at_encoder_input=True,
    ).cuda().eval()


def reader(
    *,
    environment: str,
    source_root: Path,
    cache: Path,
    manifest_sha256: str,
    validation: Path,
    validation_sha256: str,
    contract: Path,
    contract_sha256: str,
    producer_sha256: str,
) -> DepthCacheReader:
    return DepthCacheReader(
        environment=environment,
        source_root=source_root,
        cache_dir=cache,
        cache_manifest_sha256=manifest_sha256,
        validation_path=validation,
        validation_sha256=validation_sha256,
        native_contract_path=contract,
        native_contract_sha256=contract_sha256,
        expected_producer_sha256=producer_sha256,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
    )


def manifest_order(value: DepthCacheReader) -> list[dict[str, Any]]:
    return [
        {
            "trajectory_key": record["trajectory_key"],
            "source_path": record["source_path"],
            "source_video_sha256": record["source_video_sha256"],
            "ordered_frame_count": record["ordered_frame_count"],
            "ordered_output_keys": record["ordered_output_keys"],
            "commit_ordinal": record["commit_ordinal"],
        }
        for record in value.manifest["trajectories"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("rope", "granular"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--hydra", type=Path, required=True)
    for prefix in ("old", "new"):
        parser.add_argument(f"--{prefix}-cache", type=Path, required=True)
        parser.add_argument(f"--{prefix}-manifest-sha256", required=True)
        parser.add_argument(f"--{prefix}-validation", type=Path, required=True)
        parser.add_argument(f"--{prefix}-validation-sha256", required=True)
        parser.add_argument(f"--{prefix}-contract", type=Path, required=True)
        parser.add_argument(f"--{prefix}-contract-sha256", required=True)
        parser.add_argument(f"--{prefix}-producer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("zero-depth proof output already exists")
    args.output.mkdir(parents=True)

    old = reader(
        environment=args.environment,
        source_root=args.source_root,
        cache=args.old_cache,
        manifest_sha256=args.old_manifest_sha256,
        validation=args.old_validation,
        validation_sha256=args.old_validation_sha256,
        contract=args.old_contract,
        contract_sha256=args.old_contract_sha256,
        producer_sha256=args.old_producer_sha256,
    )
    new = reader(
        environment=args.environment,
        source_root=args.source_root,
        cache=args.new_cache,
        manifest_sha256=args.new_manifest_sha256,
        validation=args.new_validation,
        validation_sha256=args.new_validation_sha256,
        contract=args.new_contract,
        contract_sha256=args.new_contract_sha256,
        producer_sha256=args.new_producer_sha256,
    )
    data_order_exact = manifest_order(old) == manifest_order(new)
    records = sorted(
        (
            record
            for record in new._records.values()
            if str(record["trajectory_key"]).startswith("train/")
        ),
        key=lambda record: str(record["trajectory_key"]),
    )[:4]
    rgb_parts, old_parts, new_parts, mask_parts, identities = [], [], [], [], []
    for record in records:
        split, episode_text = str(record["trajectory_key"]).split("/")
        frames = fixed_frames(int(record["ordered_frame_count"]))
        old_depth, old_mask = old.read(
            split=split, episode=int(episode_text), frames=frames
        )
        new_depth, new_mask = new.read(
            split=split, episode=int(episode_text), frames=frames
        )
        source = (args.source_root / str(record["source_path"])).resolve()
        rgb, _ = rgb_for(source, frames)
        rgb_parts.append(rgb)
        old_parts.append(old_depth)
        new_parts.append(new_depth)
        if not torch.equal(old_mask, new_mask):
            raise RuntimeError("old/new payload-presence masks differ")
        mask_parts.append(new_mask)
        identities.extend(
            f"{record['trajectory_key']}/{frame:06d}" for frame in frames
        )
    rgb = torch.cat(rgb_parts).cuda()
    old_depth = torch.cat(old_parts).cuda()
    new_depth = torch.cat(new_parts).cuda()
    mask = torch.cat(mask_parts).cuda()
    old_encoder = encoder(
        args.checkpoint,
        args.old_contract,
        args.old_contract_sha256,
        args.old_producer_sha256,
        args.environment,
    )
    new_encoder = encoder(
        args.checkpoint,
        args.new_contract,
        args.new_contract_sha256,
        args.new_producer_sha256,
        args.environment,
    )
    with torch.inference_mode():
        old_boundary, old_boundary_mask = old_encoder.prepare_depth_encoder_input(
            old_depth, mask
        )
        new_boundary, new_boundary_mask = new_encoder.prepare_depth_encoder_input(
            new_depth, mask
        )
        old_features = old_encoder(rgb, old_depth, mask)
        new_features = new_encoder(rgb, new_depth, mask)
    boundary_exact = (
        torch.equal(old_boundary, new_boundary)
        and torch.count_nonzero(old_boundary).item() == 0
        and torch.equal(old_boundary_mask, new_boundary_mask)
    )
    feature_exact = torch.equal(old_features, new_features)

    config = OmegaConf.load(args.hydra)
    predictor = hydra.utils.instantiate(config.predictor).cuda().eval()
    checkpoint = torch.load(
        args.world_model_checkpoint, map_location="cpu", weights_only=False
    )
    predictor.load_state_dict(checkpoint["models"]["predictor"], strict=True)
    zeros = torch.zeros(
        old_features.shape[0], 2, old_features.shape[-1], device="cuda"
    )
    with torch.inference_mode():
        old_downstream = predictor(torch.cat((old_features, zeros), dim=1))
        new_downstream = predictor(torch.cat((new_features, zeros), dim=1))
    downstream_exact = torch.equal(old_downstream, new_downstream)

    state = (
        "ACCEPTED"
        if data_order_exact and boundary_exact and feature_exact and downstream_exact
        else "RERUN_REQUIRED"
    )
    receipt = {
        "schema": "dinocular.zero-depth-reuse-decision.v1",
        "state": state,
        "environment": args.environment,
        "arm": "dinocular_zerodepth",
        "fixed_frame_identities": identities,
        "checks": {
            "data_order_and_rgb_source_identity_exact": data_order_exact,
            "encoder_input_bytes_exact_zero": boundary_exact,
            "frozen_feature_bytes_exact": feature_exact,
            "accepted_predictor_execution_bytes_exact": downstream_exact,
            "payload_presence_mask_bytes_exact": torch.equal(
                old_boundary_mask, new_boundary_mask
            ),
            "model_checkpoint_selection_exact": (
                sha256_file(args.checkpoint) == CHECKPOINT_SHA256
            ),
            "world_model_checkpoint": {
                "path": str(args.world_model_checkpoint),
                "sha256": sha256_file(args.world_model_checkpoint),
                "global_step": checkpoint["global_step"],
            },
        },
        "old_identity": {
            "cache_manifest_sha256": args.old_manifest_sha256,
            "validation_sha256": args.old_validation_sha256,
            "contract_sha256": args.old_contract_sha256,
            "producer_sha256": args.old_producer_sha256,
        },
        "corrected_identity": {
            "cache_manifest_sha256": args.new_manifest_sha256,
            "validation_sha256": args.new_validation_sha256,
            "contract_sha256": args.new_contract_sha256,
            "producer_sha256": args.new_producer_sha256,
        },
    }
    receipt["decision_sha256"] = decision_sha256(receipt)
    path = args.output / "reuse_decision.json"
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if state != "ACCEPTED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
