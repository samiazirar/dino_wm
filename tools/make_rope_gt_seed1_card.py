#!/usr/bin/env python3
"""Create the immutable legacy P3 card for the Rope HDF5 seed-one lineage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import hydra
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tools.harness_common import finalize_run_card, validate_run_card
from training_resume import json_sha256


PROJECT = Path(
    "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm"
)
OLD_CARD = PROJECT / "outputs/campaign-seed1/native-seed1-20260728c/cards/p3-rope-dinocular-s1.yaml"
RUN_ID = "p3-rope-dinocular-gt-hdf5-cam1-s1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_config_hash(
    *, code_root: Path, cache_dir: Path, validation_path: Path, contract_path: Path,
    manifest_sha256: str, validation_sha256: str, contract_sha256: str,
    producer_sha256: str, checkpoint_sha256: str,
) -> str:
    os.environ["DATASET_DIR"] = str(PROJECT / "data/raw")
    os.environ["DINOV2_REPO"] = str(PROJECT / "code/dinov2")
    os.environ["DINOV2_VITS14_WEIGHTS"] = str(
        PROJECT / "models/dinov2_vits14_pretrain.pth"
    )
    os.environ["DINOCULAR_STUDENT_WEIGHTS"] = str(
        PROJECT / "checkpoints/dinov2_depthembed_dropout_fullpr.pth"
    )
    os.environ["DINOCULAR_NATIVE_DEPTH_CONTRACT"] = str(contract_path)
    os.environ["DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256"] = contract_sha256
    os.environ["DINOCULAR_CACHE_PRODUCER_SHA256"] = producer_sha256
    os.environ["DINOCULAR_CACHE_ENVIRONMENT"] = "rope"

    overrides = [
        "env=deformable_env",
        "encoder=dinocular",
        "training.seed=1",
        "training.predictor_lr=5e-5",
        "training.strict_determinism=true",
        "training.resume_from=auto",
        "training.batch_size=32",
        "env.num_workers=0",
        "frameskip=1",
        "num_hist=1",
        "num_pred=1",
        "has_decoder=false",
        "model.train_encoder=false",
        "model.train_predictor=true",
        "model.train_decoder=false",
        "plan_settings.plan_cfg_path=null",
        "env.dataset.object_name=rope",
        "env.kwargs.object_name=rope",
        f"+env.dataset.depth_cache_dir={cache_dir}",
        f"+env.dataset.depth_cache_manifest_sha256={manifest_sha256}",
        f"+env.dataset.depth_validation_path={validation_path}",
        f"+env.dataset.depth_validation_sha256={validation_sha256}",
        f"+env.dataset.native_depth_contract_path={contract_path}",
        f"+env.dataset.native_depth_contract_sha256={contract_sha256}",
        f"+env.dataset.depth_cache_producer_sha256={producer_sha256}",
        f"+env.dataset.depth_checkpoint_sha256={checkpoint_sha256}",
        "training.p3_completion_enabled=true",
        f"training.p3_heldout_manifest={PROJECT / 'outputs/campaign-seed1/native-seed1-20260728c/heldout/heldout_rope.jsonl'}",
        f"training.p3_heldout_manifest_sha256={OLD_HELDOUT_SHA}",
        f"training.p3_heldout_metadata={PROJECT / 'outputs/campaign-seed1/native-seed1-20260728c/heldout/heldout_rope.meta.json'}",
        f"training.p3_heldout_metadata_sha256={OLD_HELDOUT_META_SHA}",
        f"training.p3_data_manifest_sha256={RGB_MANIFEST_SHA}",
        f"training.p3_split_sha256={SPLIT_SHA}",
        "training.target_steps=53500",
        "training.segment_steps=1800",
        "training.checkpoint_every_steps=500",
        "training.test_signal_after_step=null",
    ]
    with initialize_config_dir(
        version_base=None, config_dir=str((code_root / "conf").resolve())
    ):
        cfg = compose(config_name="train", overrides=overrides)
    config = OmegaConf.to_container(cfg, resolve=True)
    for key in (
        "hydra",
        "ckpt_base_path",
        "saved_folder",
        "wandb_run_id",
        "effective_batch_size",
        "gpu_batch_size",
    ):
        config.pop(key, None)
    training = config["training"]
    for key in (
        "target_steps",
        "segment_steps",
        "resume_from",
        "checkpoint_every_steps",
        "test_signal_after_step",
        "timing_output",
        "timing_run_card",
        "timing_warmup_steps",
        "timing_measured_steps",
        "timing_projection_target_steps",
        "final_acceptance",
    ):
        training.pop(key, None)
    return json_sha256(config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--validation-sha256", required=True)
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-card", type=Path, required=True)
    args = parser.parse_args()

    old = yaml.safe_load(OLD_CARD.read_text(encoding="utf-8"))
    card = copy.deepcopy(old)
    source_commit = subprocess.check_output(
        ["git", "-C", str(args.code_root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_hashes = {
        relative: sha256_file(args.code_root / relative)
        for relative in old["source_file_sha256"]
    }
    config_sha256 = semantic_config_hash(
        code_root=args.code_root,
        cache_dir=args.cache_dir.resolve(),
        validation_path=args.validation_path.resolve(),
        contract_path=args.contract_path.resolve(),
        manifest_sha256=args.manifest_sha256,
        validation_sha256=args.validation_sha256,
        contract_sha256=args.contract_sha256,
        producer_sha256=args.producer_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    card.update(
        {
            "run_id": RUN_ID,
            "run_dir": str(
                PROJECT
                / "outputs/campaign-seed1/native-seed1-20260804c/rope/dinocular_gt_hdf5_cam1/run"
            ),
            "code_root": str(args.code_root.resolve()),
            "source_commit": source_commit,
            "source_file_sha256": source_hashes,
            "config_sha256": config_sha256,
            "environment_variables": {
                "DINOV2_REPO": str(PROJECT / "code/dinov2"),
                "DINOV2_VITS14_WEIGHTS": str(
                    PROJECT / "models/dinov2_vits14_pretrain.pth"
                ),
                "DINOCULAR_STUDENT_WEIGHTS": str(
                    PROJECT / "checkpoints/dinov2_depthembed_dropout_fullpr.pth"
                ),
                "DINOCULAR_NATIVE_DEPTH_CONTRACT": str(
                    args.contract_path.resolve()
                ),
                "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256": args.contract_sha256,
                "DINOCULAR_CACHE_PRODUCER_SHA256": args.producer_sha256,
                "DINOCULAR_CACHE_ENVIRONMENT": "rope",
            },
            "depth_inputs": {
                "environment": "rope",
                "producer": "released_pyflex_hdf5_camera_1",
                "producer_sha256": args.producer_sha256,
                "cache_dir": str(args.cache_dir.resolve()),
                "cache_manifest_sha256": args.manifest_sha256,
                "validation_path": str(args.validation_path.resolve()),
                "validation_sha256": args.validation_sha256,
                "native_contract_path": str(args.contract_path.resolve()),
                "native_contract_sha256": args.contract_sha256,
                "checkpoint_sha256": args.checkpoint_sha256,
            },
        }
    )
    card["artifacts"] = {
        key: value
        for key, value in old["artifacts"].items()
        if key
        in {
            "dinov2",
            "dinocular_student",
            "rgb_dataset_manifest",
            "rgb_dataset_manifest_acceptance",
            "container",
        }
    }
    card["artifacts"]["native_depth_contract"] = {
        "path": str(args.contract_path.resolve()),
        "sha256": args.contract_sha256,
    }
    card["overrides"] = [
        "env=deformable_env",
        "encoder=dinocular",
        "training.seed=1",
        "training.predictor_lr=5e-5",
        "training.strict_determinism=true",
        "training.resume_from=auto",
        "training.batch_size=32",
        "env.num_workers=0",
        "frameskip=1",
        "num_hist=1",
        "num_pred=1",
        "has_decoder=false",
        "model.train_encoder=false",
        "model.train_predictor=true",
        "model.train_decoder=false",
        "plan_settings.plan_cfg_path=null",
        "env.dataset.object_name=rope",
        "env.kwargs.object_name=rope",
        f"+env.dataset.depth_cache_dir={args.cache_dir.resolve()}",
        f"+env.dataset.depth_cache_manifest_sha256={args.manifest_sha256}",
        f"+env.dataset.depth_validation_path={args.validation_path.resolve()}",
        f"+env.dataset.depth_validation_sha256={args.validation_sha256}",
        f"+env.dataset.native_depth_contract_path={args.contract_path.resolve()}",
        f"+env.dataset.native_depth_contract_sha256={args.contract_sha256}",
        f"+env.dataset.depth_cache_producer_sha256={args.producer_sha256}",
        f"+env.dataset.depth_checkpoint_sha256={args.checkpoint_sha256}",
        "training.p3_completion_enabled=true",
        f"training.p3_heldout_manifest={PROJECT / 'outputs/campaign-seed1/native-seed1-20260728c/heldout/heldout_rope.jsonl'}",
        f"training.p3_heldout_manifest_sha256={OLD_HELDOUT_SHA}",
        f"training.p3_heldout_metadata={PROJECT / 'outputs/campaign-seed1/native-seed1-20260728c/heldout/heldout_rope.meta.json'}",
        f"training.p3_heldout_metadata_sha256={OLD_HELDOUT_META_SHA}",
        f"training.p3_data_manifest_sha256={RGB_MANIFEST_SHA}",
        f"training.p3_split_sha256={SPLIT_SHA}",
    ]
    card.pop("producer_decision", None)
    card.pop("native_depth_acceptance", None)
    card = dict(finalize_run_card(card))
    validate_run_card(card)
    args.output_card.parent.mkdir(parents=True, exist_ok=True)
    args.output_card.write_text(
        yaml.safe_dump(card, sort_keys=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "state": "PASS",
                "card": str(args.output_card.resolve()),
                "run_card_sha256": card["run_card_sha256"],
                "run_card_file_sha256": sha256_file(args.output_card.resolve()),
                "config_sha256": config_sha256,
                "source_commit": source_commit,
            },
            sort_keys=True,
        )
    )
    return 0


# Frozen release/card identities are copied from the already accepted Rope
# seed-one card; only the depth source and run identity change.
OLD_HELDOUT_SHA = "82672d804b198c31258105e294237c2716dd31e34c07ba05fdc9f5002ef0c607"
OLD_HELDOUT_META_SHA = "280f11df2ac40011fed3d16b9299d48b550452e30875fd24287d3fbd9248b8c2"
RGB_MANIFEST_SHA = "f7e51a2a87b9c197a7c4b5416a765f7a5b73ffb67b8dde1d7614e772cfab194f"
SPLIT_SHA = "2d7a406813f78b6cbdfd0777c579d6a423e548badaebb56281c4dbf48cccf751"


if __name__ == "__main__":
    raise SystemExit(main())
