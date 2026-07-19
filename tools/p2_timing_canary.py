#!/usr/bin/env python3
"""Measure a single-GPU P2 training timing card for the RGB-only arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import threading
import time
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils import seed


ENVIRONMENT_OVERRIDES = {
    "pusht": ("pusht", 3, []),
    "wall": ("wall", 1, []),
    "rope": ("deformable_env", 1, ["env.dataset.object_name=rope"]),
    "granular": ("deformable_env", 1, ["env.dataset.object_name=granular"]),
}


class NvidiaSmiMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.peak_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                values = [
                    int(line.strip())
                    for line in completed.stdout.splitlines()
                    if line.strip().isdigit()
                ]
                if values:
                    self.peak_mib = max(self.peak_mib, sum(values))
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=sorted(ENVIRONMENT_OVERRIDES), required=True)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measured-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def move_to_device(data, device: torch.device):
    obs, act, state = data
    obs = {key: value.to(device, non_blocking=True) for key, value in obs.items()}
    return obs, act.to(device, non_blocking=True), state.to(device, non_blocking=True)


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 1 or args.measured_steps < 1:
        raise ValueError("warmup and measured step counts must both be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("P2 timing requires CUDA")

    code_root = Path(__file__).resolve().parents[1]
    env_config, num_hist, extra_overrides = ENVIRONMENT_OVERRIDES[args.env]
    overrides = [
        f"env={env_config}",
        "encoder=dino_pinned",
        "training.seed=1",
        "training.epochs=100",
        "training.batch_size=32",
        "training.predictor_lr=5e-5",
        "frameskip=5",
        f"num_hist={num_hist}",
        "num_pred=1",
        "has_decoder=false",
        "model.train_encoder=false",
        "model.train_predictor=true",
        "model.train_decoder=false",
        "plan_settings.plan_cfg_path=null",
    ] + extra_overrides
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
    train_windows = len(train_dataset)
    batch_size = int(cfg.training.batch_size)
    steps_per_epoch = math.ceil(train_windows / batch_size)
    required_steps = args.warmup_steps + args.measured_steps
    if steps_per_epoch < required_steps:
        raise RuntimeError(
            f"Timing window needs {required_steps} steps but {args.env} has only "
            f"{steps_per_epoch} steps per epoch"
        )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.env.num_workers),
        collate_fn=None,
    )

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
    predictor_optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=float(cfg.training.predictor_lr)
    )
    auxiliary_optimizer = torch.optim.AdamW(
        list(action_encoder.parameters()) + list(proprio_encoder.parameters()),
        lr=float(cfg.training.action_encoder_lr),
    )

    monitor = NvidiaSmiMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    measured_start = None
    measured_end = None
    measured_samples = 0
    final_loss = None
    iterator = iter(loader)
    try:
        for step_index in range(required_steps):
            data = next(iterator)
            obs, act, _ = move_to_device(data, device)
            model.train()
            _, _, _, loss, _ = model(obs, act)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite loss at step {step_index + 1}: {loss.item()}"
                )
            predictor_optimizer.zero_grad()
            auxiliary_optimizer.zero_grad()
            loss.backward()
            predictor_optimizer.step()
            auxiliary_optimizer.step()
            final_loss = float(loss.detach().item())

            completed_steps = step_index + 1
            if completed_steps == args.warmup_steps:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                measured_start = time.perf_counter()
            elif completed_steps > args.warmup_steps:
                measured_samples += int(act.shape[0])
            if completed_steps == required_steps:
                torch.cuda.synchronize(device)
                measured_end = time.perf_counter()
    finally:
        monitor.stop()

    if measured_start is None or measured_end is None or final_loss is None:
        raise RuntimeError("timing window did not complete")
    measured_seconds = measured_end - measured_start
    steps_per_second = args.measured_steps / measured_seconds
    samples_per_second = measured_samples / measured_seconds
    epochs_per_hour = 3600.0 * samples_per_second / train_windows
    seconds_per_epoch = 3600.0 / epochs_per_hour
    result = {
        "schema": "dinocular-wm.p2-timing.v1",
        "status": "MEASURED",
        "arm": "dinov2_vits14",
        "environment": args.env,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "code_commit": subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "global_batch_size": batch_size,
        "warmup_steps_excluded": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "measured_samples": measured_samples,
        "measured_seconds": measured_seconds,
        "steps_per_second": steps_per_second,
        "samples_per_second": samples_per_second,
        "train_windows": train_windows,
        "steps_per_epoch": steps_per_epoch,
        "epochs_per_hour": epochs_per_hour,
        "epoch_equivalent_seconds_from_measured_rate": seconds_per_epoch,
        "complete_epoch_validation": "NOT_RUN_FIXED_STEP_CANARY",
        "peak_torch_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_torch_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "peak_nvidia_smi_process_mib": monitor.peak_mib,
        "nvidia_smi_samples": monitor.samples,
        "final_loss": final_loss,
        "assumption_tags": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
