import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from train import Trainer
from tools.submit_p3_chain import verify_progress_evidence
from training_resume import (
    CHECKPOINT_SCHEMA,
    SerializableConstantScheduler,
    StepBatchSampler,
    StepCheckpointManager,
    capture_rng_state,
    file_sha256,
    parameter_sha256,
    restore_rng_state,
)


class TinyStepDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.tensor([index]), torch.tensor([0.0]), torch.tensor([index])


@pytest.mark.parametrize(
    (
        "start_step",
        "segment_steps",
        "checkpoint_every",
        "expected_saves",
        "expected_status",
        "expected_stop",
    ),
    [
        (0, 4, 0, [3, 4], "SEGMENT_COMPLETE", 4),
        (0, 7, 5, [3, 5, 6, 7], "TARGET_REACHED", 7),
        (2, 3, 0, [3, 5], "SEGMENT_COMPLETE", 5),
    ],
)
def test_run_steps_saves_every_epoch_without_changing_segment_semantics(
    tmp_path,
    start_step,
    segment_steps,
    checkpoint_every,
    expected_saves,
    expected_status,
    expected_stop,
):
    immutable_run_card_sha256 = "a" * 64
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = SerializableConstantScheduler(optimizer)
    trainer = object.__new__(Trainer)
    trainer.cfg = OmegaConf.create(
        {
            "saved_folder": str(tmp_path),
            "gpu_batch_size": 4,
            "training": {
                "target_steps": 7,
                "segment_steps": segment_steps,
                "checkpoint_every_steps": checkpoint_every,
                "test_signal_after_step": None,
                "timing_output": None,
                "seed": 1,
            },
            "env": {"num_workers": 0},
        }
    )
    trainer.datasets = {"train": TinyStepDataset()}
    trainer.p3_completion_enabled = False
    trainer.accelerator = SimpleNamespace(prepare=lambda loader: loader)
    trainer.global_step = start_step
    trainer.epoch = start_step // 3
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
    trainer.resume_config_sha256 = "b" * 64
    trainer.immutable_run_card_sha256 = immutable_run_card_sha256
    trainer.dataset_order_sha256 = "c" * 64
    trainer.checkpoint_manager = StepCheckpointManager(
        tmp_path / "checkpoints" / "steps"
    )
    trainer.schedulers = {"model": scheduler}
    trainer._model_components = lambda: {"model": model}
    trainer._optimizers = lambda: {"model": optimizer}
    trainer._train_one_step = lambda _data: float(trainer.global_step + 1)

    saved_payloads = []

    def save_and_record(reasons=("LEGACY_CALL",)):
        previous_step = trainer._last_saved_step
        path, digest = Trainer.save_step_checkpoint(trainer, reasons)
        if trainer._last_saved_step != previous_step:
            assert file_sha256(path) == digest
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert payload["immutable_run_card_sha256"] == immutable_run_card_sha256
            assert payload["global_step"] == trainer.global_step
            assert payload["sampler"]["next_step"] == trainer.global_step
            saved_payloads.append(payload)
        return path, digest

    trainer.save_step_checkpoint = save_and_record
    Trainer.run_steps(trainer)

    assert [payload["global_step"] for payload in saved_payloads] == expected_saves
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["status"] == expected_status
    assert progress["global_step"] == expected_stop
    assert progress["segment_start_step"] == start_step
    assert progress["segment_stop_step"] == expected_stop
    assert progress["completed_segment_steps"] == segment_steps
    assert progress["immutable_run_card_sha256"] == immutable_run_card_sha256
    final_checkpoint = torch.load(
        progress["checkpoint"], map_location="cpu", weights_only=False
    )
    assert progress["checkpoint_sha256"] == file_sha256(Path(progress["checkpoint"]))
    manifest = {
        "run_dir": str(tmp_path),
        "run_card_sha256": immutable_run_card_sha256,
    }
    assert (
        verify_progress_evidence(manifest, progress)
        == Path(progress["checkpoint"]).resolve()
    )
    assert final_checkpoint["immutable_run_card_sha256"] == immutable_run_card_sha256
    assert final_checkpoint["global_step"] == expected_stop

    changed_progress = dict(progress)
    changed_progress["immutable_run_card_sha256"] = "d" * 64
    with pytest.raises(
        RuntimeError, match="Progress differs from the immutable run card"
    ):
        verify_progress_evidence(manifest, changed_progress)
    changed_progress = dict(progress)
    changed_progress["checkpoint_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="checkpoint SHA-256 differs"):
        verify_progress_evidence(manifest, changed_progress)

    trainer.immutable_run_card_sha256 = "d" * 64
    with pytest.raises(RuntimeError, match="Immutable run card differs"):
        Trainer._load_step_checkpoint(
            trainer,
            trainer._last_checkpoint_path,
            trainer._last_checkpoint_sha256,
        )


def test_step_sampler_split_matches_straight():
    straight = list(StepBatchSampler(10, 4, 0, 7))
    split = list(StepBatchSampler(10, 4, 0, 4)) + list(StepBatchSampler(10, 4, 4, 7))
    assert straight == split
    assert straight == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
        [0, 1, 2, 3],
    ]
    state = StepBatchSampler(10, 4, 7, 7).state_dict(7)
    assert state == {
        "dataset_size": 10,
        "batch_size": 4,
        "steps_per_epoch": 3,
        "next_step": 7,
        "completed_epochs": 2,
        "next_batch_in_epoch": 1,
        "next_sample_in_epoch": 4,
    }


def test_rng_round_trip_restores_python_numpy_and_torch():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.random()
    np.random.rand()
    torch.rand(3)
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_constant_scheduler_state_is_exact():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    scheduler = SerializableConstantScheduler(optimizer)
    scheduler.step()
    scheduler.step()
    state = scheduler.state_dict()

    restored = SerializableConstantScheduler(optimizer)
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_checkpoint_manager_hash_falls_back_one_checkpoint(tmp_path):
    manager = StepCheckpointManager(tmp_path)
    first, first_digest = manager.save(
        {"schema": CHECKPOINT_SCHEMA, "global_step": 1}, 1, reasons=("COMPLETE_EPOCH",)
    )
    second, _second_digest = manager.save(
        {"schema": CHECKPOINT_SCHEMA, "global_step": 2}, 2, reasons=("EXACT_TARGET",)
    )
    second.write_bytes(b"corrupt")
    resolved, digest = manager.resolve("auto")
    assert resolved == first
    assert digest == first_digest


def test_parameter_hash_covers_parameters_and_buffers():
    model = torch.nn.BatchNorm1d(3)
    before = parameter_sha256({"model": model})
    model.running_mean[0] = 1
    after = parameter_sha256({"model": model})
    assert before != after
