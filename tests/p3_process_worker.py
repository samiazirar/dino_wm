#!/usr/bin/env python3
"""Tiny two-process Trainer fixture for P3 final-acceptance integration tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from train import Trainer
from training_resume import SerializableConstantScheduler


class TinyP3Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.slices = np.asarray(
            [[0, index, index + 1] for index in range(4)], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int):
        value = float(index + 1)
        return (
            torch.tensor([value], dtype=torch.float32),
            torch.tensor([0.25], dtype=torch.float32),
            torch.tensor([value], dtype=torch.float32),
        )


class TinyP3Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2]))
        self.train_encoder = True
        self.train_decoder = False
        self.train_predictor = False
        self.concat_dim = 0
        self.action_dim = 1

    def forward(self, obs, _act):
        batch = int(obs.shape[0])
        value = obs.reshape(batch, -1).mean(dim=1) * self.weight
        z_pred = value.reshape(batch, 1, 1, 1).expand(batch, 1, 3, 1)
        loss = (z_pred[:, :, :-1, :] - 0.5).square().mean()
        return z_pred, None, None, loss, {"loss": loss}


class CpuAccelerator:
    is_main_process = True

    @staticmethod
    def prepare(value):
        return value

    @staticmethod
    def unwrap_model(value):
        return value

    @staticmethod
    def backward(loss):
        loss.backward()

    @staticmethod
    def gather_for_metrics(value):
        return value


def _config(run_dir: Path, *, final_acceptance: bool):
    return OmegaConf.create(
        {
            "saved_folder": str(run_dir),
            "gpu_batch_size": 2,
            "has_decoder": False,
            "has_predictor": False,
            "training": {
                "target_steps": 100,
                "segment_steps": 100,
                "resume_from": "auto" if final_acceptance else None,
                "checkpoint_every_steps": 0,
                "test_signal_after_step": None,
                "timing_output": None,
                "timing_run_card": None,
                "timing_warmup_steps": 0,
                "timing_measured_steps": 0,
                "timing_projection_target_steps": 100,
                "final_acceptance": final_acceptance,
                "p3_completion_enabled": True,
                "p3_heldout_manifest_sha256": "5" * 64,
                "p3_data_manifest_sha256": "6" * 64,
                "p3_split_sha256": "7" * 64,
                "seed": 17,
            },
            "env": {"num_workers": 0},
        }
    )


def build_trainer(run_dir: Path, *, final_acceptance: bool) -> Trainer:
    torch.manual_seed(17)
    np.random.seed(17)
    trainer = object.__new__(Trainer)
    trainer.cfg = _config(run_dir, final_acceptance=final_acceptance)
    trainer.accelerator = CpuAccelerator()
    trainer.datasets = {"train": TinyP3Dataset()}
    trainer.model = TinyP3Model()
    trainer.encoder = trainer.model
    trainer.action_encoder = torch.nn.Identity()
    trainer.proprio_encoder = torch.nn.Identity()
    trainer.predictor = None
    trainer.decoder = None
    trainer.encoder_optimizer = torch.optim.AdamW(
        trainer.model.parameters(), lr=5e-4
    )
    trainer.schedulers = {
        "encoder": SerializableConstantScheduler(trainer.encoder_optimizer)
    }
    trainer.p3_validation_loader = torch.utils.data.DataLoader(
        TinyP3Dataset(), batch_size=2, shuffle=False, num_workers=0
    )
    trainer.p3_heldout_rows = [{"key": "synthetic/validation/000000"}]
    trainer.p3_heldout_metadata = {
        "data_manifest_sha256": "6" * 64,
        "split_sha256": "7" * 64,
    }
    trainer.p3_completion_enabled = True
    trainer.final_acceptance = final_acceptance
    trainer.step_mode = True
    trainer.global_step = 0
    trainer.epoch = 0
    trainer.last_step_loss = None
    trainer._resume_rng_state = None
    trainer._stop_requested = False
    trainer._stop_signal = None
    trainer._last_saved_step = None
    trainer._last_checkpoint_path = None
    trainer._last_checkpoint_sha256 = None
    trainer._last_checkpoint_history_record = None
    trainer._last_checkpoint_state_hashes = None
    trainer._loaded_checkpoint_metadata = None
    trainer.source_commit = "f" * 40
    Trainer._initialize_step_resume(trainer)
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "final"))
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    trainer = build_trainer(
        args.run_dir.resolve(), final_acceptance=args.mode == "final"
    )
    if args.mode == "train":
        Trainer.run_steps(trainer)
        print("SYNTHETIC_P3_TRAIN=PASS")
    else:
        Trainer._run_p3_final_acceptance(trainer)


if __name__ == "__main__":
    main()
