#!/usr/bin/env python3
"""Run a real PushT straight-versus-signal-resume determinism proof."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

from training_resume import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-steps", type=int, default=3)
    parser.add_argument("--resume-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_training(
    code_root: Path,
    run_dir: Path,
    target_steps: int,
    batch_size: int,
    signal_after_step: int | None,
    label: str,
) -> tuple[dict, float]:
    run_dir.mkdir(parents=True, exist_ok=False)
    signal_override = (
        "null" if signal_after_step is None else str(signal_after_step)
    )
    command = [
        sys.executable,
        str(code_root / "train.py"),
        "env=pusht",
        "encoder=dino_pinned",
        "training.seed=1",
        f"training.target_steps={target_steps}",
        f"training.segment_steps={target_steps}",
        "training.resume_from=auto",
        "training.checkpoint_every_steps=0",
        "training.strict_determinism=true",
        f"training.test_signal_after_step={signal_override}",
        f"training.batch_size={batch_size}",
        "training.predictor_lr=5e-5",
        "env.num_workers=0",
        "frameskip=5",
        "num_hist=3",
        "num_pred=1",
        "has_decoder=false",
        "model.train_encoder=false",
        "model.train_predictor=true",
        "model.train_decoder=false",
        "plan_settings.plan_cfg_path=null",
        "debug=true",
        f"ckpt_base_path={run_dir}",
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=true",
        "hydra.output_subdir=null",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "WANDB_MODE": "disabled",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    started = time.perf_counter()
    with (run_dir / f"{label}.log").open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=code_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} training failed with exit {completed.returncode}; "
            f"see {run_dir / f'{label}.log'}"
        )
    return read_json(run_dir / "progress.json"), elapsed


def resume_training(
    code_root: Path,
    run_dir: Path,
    target_steps: int,
    batch_size: int,
    signal_after_step: int,
) -> tuple[dict, float]:
    command = [
        sys.executable,
        str(code_root / "train.py"),
        "env=pusht",
        "encoder=dino_pinned",
        "training.seed=1",
        f"training.target_steps={target_steps}",
        f"training.segment_steps={target_steps}",
        "training.resume_from=auto",
        "training.checkpoint_every_steps=0",
        "training.strict_determinism=true",
        f"training.test_signal_after_step={signal_after_step}",
        f"training.batch_size={batch_size}",
        "training.predictor_lr=5e-5",
        "env.num_workers=0",
        "frameskip=5",
        "num_hist=3",
        "num_pred=1",
        "has_decoder=false",
        "model.train_encoder=false",
        "model.train_predictor=true",
        "model.train_decoder=false",
        "plan_settings.plan_cfg_path=null",
        "debug=true",
        f"ckpt_base_path={run_dir}",
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=true",
        "hydra.output_subdir=null",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "WANDB_MODE": "disabled",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    started = time.perf_counter()
    with (run_dir / "resume.log").open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=code_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"resume training failed with exit {completed.returncode}; "
            f"see {run_dir / 'resume.log'}"
        )
    return read_json(run_dir / "progress.json"), elapsed


def main() -> None:
    args = parse_args()
    if args.first_steps <= 0 or args.resume_steps <= 0 or args.batch_size <= 0:
        raise ValueError("step counts and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Determinism proof requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"Proof output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    code_root = Path(__file__).resolve().parents[1]
    target_steps = args.first_steps + args.resume_steps
    straight, straight_seconds = run_training(
        code_root,
        args.output_dir / "straight",
        target_steps,
        args.batch_size,
        None,
        "straight",
    )
    split_first, split_first_seconds = run_training(
        code_root,
        args.output_dir / "split",
        target_steps,
        args.batch_size,
        args.first_steps,
        "signal_stop",
    )
    shutil.copy2(
        args.output_dir / "split" / "progress.json",
        args.output_dir / "split_after_signal.json",
    )
    if split_first["status"] != "SIGNAL_CHECKPOINTED":
        raise AssertionError(f"Signal path did not checkpoint: {split_first}")
    if split_first["global_step"] != args.first_steps:
        raise AssertionError(
            f"Signal checkpoint step is {split_first['global_step']}, "
            f"expected {args.first_steps}"
        )

    resumed, resume_seconds = resume_training(
        code_root,
        args.output_dir / "split",
        target_steps,
        args.batch_size,
        args.first_steps,
    )
    if straight["status"] != "TARGET_REACHED" or resumed["status"] != "TARGET_REACHED":
        raise AssertionError("Straight or resumed run did not reach the target")

    exact_fields = [
        "last_step_loss",
        "parameter_sha256",
        "optimizer_sha256",
        "scheduler_sha256",
        "global_step",
    ]
    mismatches = {
        field: {"straight": straight[field], "resumed": resumed[field]}
        for field in exact_fields
        if straight[field] != resumed[field]
    }
    if mismatches:
        raise AssertionError(f"Determinism mismatch: {mismatches}")

    result = {
        "schema": "dino-wm.resume-determinism-proof.v1",
        "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "code_commit": subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "first_steps": args.first_steps,
        "resume_steps": args.resume_steps,
        "target_steps": target_steps,
        "batch_size": args.batch_size,
        "straight_seconds": straight_seconds,
        "signal_segment_seconds": split_first_seconds,
        "resume_segment_seconds": resume_seconds,
        "straight_final_loss": straight["last_step_loss"],
        "resumed_final_loss": resumed["last_step_loss"],
        "parameter_sha256": straight["parameter_sha256"],
        "optimizer_sha256": straight["optimizer_sha256"],
        "scheduler_sha256": straight["scheduler_sha256"],
        "signal_checkpoint_sha256": split_first["checkpoint_sha256"],
        "straight_checkpoint_sha256": straight["checkpoint_sha256"],
        "resumed_checkpoint_sha256": resumed["checkpoint_sha256"],
        "exact_fields": exact_fields,
    }
    atomic_write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    print("DETERMINISM_PROOF=PASS")


if __name__ == "__main__":
    main()
