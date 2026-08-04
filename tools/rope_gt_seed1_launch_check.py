#!/usr/bin/env python3
"""Run one non-training GPU forward through the frozen Rope GT input path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir

from utils import seed


def move_to_device(data, device: torch.device):
    obs, action, state = data
    obs = {key: value.to(device, non_blocking=True) for key, value in obs.items()}
    return obs, action.to(device, non_blocking=True), state.to(device, non_blocking=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--environment", default="rope")
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--validation-sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--producer-sha256", required=True)
    args = parser.parse_args()
    if args.environment != "rope":
        raise ValueError("this launch check is locked to Rope")
    if not torch.cuda.is_available():
        raise RuntimeError("Rope GT launch check requires CUDA")

    code_root = args.code_root.resolve()
    overrides = [
        "env=deformable_env",
        "encoder=dinocular",
        "training.seed=1",
        "training.batch_size=32",
        "training.predictor_lr=5e-5",
        "training.strict_determinism=true",
        "frameskip=1",
        "num_hist=1",
        "num_pred=1",
        "has_decoder=false",
        "model.train_encoder=false",
        "model.train_predictor=true",
        "model.train_decoder=false",
        "env.num_workers=0",
        "plan_settings.plan_cfg_path=null",
        "env.dataset.object_name=rope",
        "env.kwargs.object_name=rope",
        "+env.dataset.depth_cache_dir="
        + os.environ["ROPE_GT_CACHE_DIR"],
        "+env.dataset.depth_cache_manifest_sha256=" + args.cache_manifest_sha256,
        "+env.dataset.depth_validation_path="
        + os.environ["ROPE_GT_VALIDATION_PATH"],
        "+env.dataset.depth_validation_sha256=" + args.validation_sha256,
        "+env.dataset.native_depth_contract_path="
        + os.environ["ROPE_GT_CONTRACT_PATH"],
        "+env.dataset.native_depth_contract_sha256=" + args.contract_sha256,
        "+env.dataset.depth_cache_producer_sha256=" + args.producer_sha256,
        "+env.dataset.depth_checkpoint_sha256=" + os.environ["ROPE_GT_CHECKPOINT_SHA256"],
    ]
    with initialize_config_dir(
        version_base=None, config_dir=str((code_root / "conf").resolve())
    ):
        cfg = compose(config_name="train", overrides=overrides)

    seed(cfg.training.seed)
    datasets, _ = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    train_dataset = datasets["train"]
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=None,
    )
    data = next(iter(loader))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    encoder.requires_grad_(False)
    encoder.eval()
    proprio_encoder = hydra.utils.instantiate(
        cfg.proprio_encoder,
        in_chans=train_dataset.proprio_dim,
        emb_dim=cfg.proprio_emb_dim,
    ).to(device)
    action_encoder = hydra.utils.instantiate(
        cfg.action_encoder,
        in_chans=train_dataset.action_dim,
        emb_dim=cfg.action_emb_dim,
    ).to(device)
    num_patches = int(encoder.num_patches) + (2 if cfg.concat_dim == 0 else 0)
    predictor_dim = int(encoder.emb_dim) + (
        int(proprio_encoder.emb_dim) * int(cfg.num_proprio_repeat)
        + int(action_encoder.emb_dim) * int(cfg.num_action_repeat)
    ) * int(cfg.concat_dim)
    predictor = hydra.utils.instantiate(
        cfg.predictor,
        num_patches=num_patches,
        num_frames=cfg.num_hist,
        dim=predictor_dim,
    ).to(device)
    model = hydra.utils.instantiate(
        cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        decoder=None,
        proprio_dim=proprio_encoder.emb_dim,
        action_dim=action_encoder.emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
    ).to(device)
    obs, action, _state = move_to_device(data, device)
    with torch.no_grad():
        _prediction, _target, _features, loss, _images = model(obs, action)
    torch.cuda.synchronize(device)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"launch-check loss is nonfinite: {loss.item()}")
    print(
        json.dumps(
            {
                "state": "PASS",
                "environment": args.environment,
                "arm": "dinocular",
                "seed": 1,
                "batch_size": int(action.shape[0]),
                "depth_cache_manifest_sha256": args.cache_manifest_sha256,
                "depth_validation_sha256": args.validation_sha256,
                "native_depth_contract_sha256": args.contract_sha256,
                "depth_producer_sha256": args.producer_sha256,
                "gpu": torch.cuda.get_device_name(device),
                "source_commit": subprocess.check_output(
                    ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
                ).strip(),
                "finite_loss": float(loss.detach().item()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
