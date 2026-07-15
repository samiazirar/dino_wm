import random

import numpy as np
import torch

from training_resume import (
    CHECKPOINT_SCHEMA,
    SerializableConstantScheduler,
    StepBatchSampler,
    StepCheckpointManager,
    capture_rng_state,
    parameter_sha256,
    restore_rng_state,
)


def test_step_sampler_split_matches_straight():
    straight = list(StepBatchSampler(10, 4, 0, 7))
    split = list(StepBatchSampler(10, 4, 0, 4)) + list(
        StepBatchSampler(10, 4, 4, 7)
    )
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
        {"schema": CHECKPOINT_SCHEMA, "global_step": 1}, 1
    )
    second, _second_digest = manager.save(
        {"schema": CHECKPOINT_SCHEMA, "global_step": 2}, 2
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
