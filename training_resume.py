"""Deterministic step-level training state and checkpoint utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from p3_completion import (
    P3CompletionError,
    append_checkpoint_history,
    load_checkpoint_history,
)


CHECKPOINT_SCHEMA = "dino-wm.step-checkpoint.v1"
INDEX_SCHEMA = "dino-wm.step-checkpoint-index.v1"
PROGRESS_SCHEMA = "dino-wm.step-progress.v1"


def configure_strict_determinism(enabled: bool) -> None:
    """Enable deterministic PyTorch execution or fail before training starts."""
    if not enabled:
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_available() and workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "Strict resume requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 "
            "to make CUDA matrix multiplications deterministic"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng_state() -> dict[str, Any]:
    """Capture every process RNG used by the training path."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != expected:
        raise RuntimeError(
            f"RNG checkpoint keys differ: expected {sorted(expected)}, got {sorted(state)}"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA RNG device count differs between checkpoint and resume: "
                f"{len(cuda_states)} versus {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states:
        raise RuntimeError("Checkpoint has CUDA RNG state but CUDA is unavailable")


class StepBatchSampler(Sampler[list[int]]):
    """Map absolute optimizer steps to the legacy, unshuffled batch sequence."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        start_step: int,
        stop_step: int,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if start_step < 0 or stop_step < start_step:
            raise ValueError("invalid absolute step interval")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.steps_per_epoch = math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        for absolute_step in range(self.start_step, self.stop_step):
            batch_in_epoch = absolute_step % self.steps_per_epoch
            start = batch_in_epoch * self.batch_size
            stop = min(start + self.batch_size, self.dataset_size)
            yield list(range(start, stop))

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def state_dict(self, next_step: int) -> dict[str, int]:
        if next_step < 0:
            raise ValueError("next_step must be non-negative")
        batch_in_epoch = next_step % self.steps_per_epoch
        next_sample = min(batch_in_epoch * self.batch_size, self.dataset_size)
        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "next_step": int(next_step),
            "completed_epochs": int(next_step // self.steps_per_epoch),
            "next_batch_in_epoch": int(batch_in_epoch),
            "next_sample_in_epoch": int(next_sample),
        }


class SerializableConstantScheduler:
    """A stateful no-op scheduler for the repository's fixed learning rates."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.completed_steps = 0

    def step(self) -> None:
        self.completed_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "constant",
            "base_lrs": list(self.base_lrs),
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("kind") != "constant":
            raise RuntimeError(f"Unsupported scheduler state: {state.get('kind')!r}")
        checkpoint_lrs = [float(value) for value in state["base_lrs"]]
        current_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        if checkpoint_lrs != self.base_lrs or current_lrs != self.base_lrs:
            raise RuntimeError(
                "Scheduler learning rates differ between checkpoint and current optimizer"
            )
        self.completed_steps = int(state["completed_steps"])


def dataset_order_sha256(dataset: Any) -> str:
    """Hash the exact TrajSlicerDataset order used by the sampler."""
    if not hasattr(dataset, "slices"):
        raise TypeError("Step resume requires a dataset with an explicit slices array")
    slices = np.asarray(dataset.slices, dtype=np.int64)
    if slices.ndim != 2 or slices.shape[1] != 3:
        raise ValueError(f"Expected slice array [N,3], got {slices.shape}")
    digest = hashlib.sha256()
    digest.update(str(slices.shape).encode("ascii"))
    digest.update(slices.tobytes(order="C"))
    return digest.hexdigest()


def _update_tensor_hash(digest: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().contiguous().cpu()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))


def parameter_sha256(components: Mapping[str, torch.nn.Module]) -> str:
    """Hash all model parameters and buffers in a stable component/key order."""
    digest = hashlib.sha256()
    for component_name in sorted(components):
        state = components[component_name].state_dict()
        for tensor_name in sorted(state):
            value = state[tensor_name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Non-tensor model state at {component_name}.{tensor_name}: {type(value)}"
                )
            _update_tensor_hash(digest, f"{component_name}.{tensor_name}", value)
    return digest.hexdigest()


def nested_state_sha256(value: Any) -> str:
    """Hash tensor-heavy optimizer and scheduler state without serialization noise."""
    digest = hashlib.sha256()

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            _update_tensor_hash(digest, path, item)
        elif isinstance(item, Mapping):
            for key in sorted(item, key=lambda entry: str(entry)):
                visit(item[key], f"{path}/{key}")
        elif isinstance(item, (list, tuple)):
            for index, entry in enumerate(item):
                visit(entry, f"{path}/{index}")
        elif isinstance(item, np.ndarray):
            digest.update(path.encode("utf-8"))
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(str(item.shape).encode("ascii"))
            digest.update(item.tobytes(order="C"))
        else:
            digest.update(path.encode("utf-8"))
            digest.update(repr(item).encode("utf-8"))

    visit(value, "root")
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class StepCheckpointManager:
    """Write hashed checkpoints atomically and retain a one-checkpoint rollback."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.index_path = self.directory / "step_latest.json"
        self.history_path = self.directory / "checkpoint_history.json"
        self.last_history_record: Mapping[str, Any] | None = None

    def _read_index(self) -> dict[str, Any] | None:
        if not self.index_path.exists():
            return None
        with self.index_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("schema") != INDEX_SCHEMA or not isinstance(
            value.get("checkpoints"), list
        ):
            raise RuntimeError(f"Malformed checkpoint index: {self.index_path}")
        return value

    def history_record(self, step: int) -> Mapping[str, Any]:
        matches = [
            record
            for record in load_checkpoint_history(self.history_path)
            if record["step"] == int(step)
        ]
        if not matches:
            raise RuntimeError(f"Checkpoint history has no record for step {step}")
        self.last_history_record = matches[-1]
        return matches[-1]

    def save(
        self,
        payload: Mapping[str, Any],
        step: int,
        reasons: Sequence[str],
    ) -> tuple[Path, str]:
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Refusing to save an unknown checkpoint schema")
        self.directory.mkdir(parents=True, exist_ok=True)
        filename = f"step_{step:09d}.pth"
        final_path = self.directory / filename
        temporary = self.directory / f".{filename}.tmp.{os.getpid()}"
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = file_sha256(temporary)
        content_digest = nested_state_sha256(dict(payload))
        existing_history = next(
            (
                record
                for record in load_checkpoint_history(self.history_path)
                if record["step"] == int(step)
            ),
            None,
        )
        if existing_history is not None and (
            existing_history.get("filename") != filename
            or existing_history.get("checkpoint_sha256") != digest
            or existing_history.get("content_sha256") != content_digest
        ):
            temporary.unlink(missing_ok=True)
            raise P3CompletionError(
                f"checkpoint step {step} would create a divergent duplicate"
            )
        os.replace(temporary, final_path)

        history_record = append_checkpoint_history(
            self.history_path,
            step=int(step),
            filename=filename,
            checkpoint_sha256=digest,
            content_sha256=content_digest,
            reasons=reasons,
            source_commit=payload.get("source_commit"),
            immutable_run_card_sha256=payload.get("immutable_run_card_sha256"),
            dataset_order_sha256=payload.get("dataset_order_sha256"),
        )
        self.last_history_record = history_record

        previous_index = self._read_index()
        previous_entries = (
            [] if previous_index is None else previous_index["checkpoints"]
        )
        entries = [
            {
                "step": int(step),
                "file": filename,
                "sha256": digest,
                "history_record_sha256": history_record["record_sha256"],
            },
            *[entry for entry in previous_entries if entry.get("file") != filename],
        ][:2]
        atomic_write_json(
            self.index_path,
            {"schema": INDEX_SCHEMA, "checkpoints": entries},
        )
        retained = {entry["file"] for entry in entries}
        for stale in self.directory.glob("step_*.pth"):
            if stale.name not in retained:
                stale.unlink()
        return final_path, digest

    def resolve(self, requested: str | Path | None) -> tuple[Path | None, str | None]:
        if requested is None or str(requested).lower() in {"", "none", "null"}:
            return None, None
        if str(requested) != "auto":
            explicit = Path(requested)
            if not explicit.is_file():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {explicit}")
            return explicit, file_sha256(explicit)

        index = self._read_index()
        if index is None:
            unindexed = (
                list(self.directory.glob("step_*.pth"))
                if self.directory.exists()
                else []
            )
            if unindexed:
                raise RuntimeError(
                    f"Checkpoint files exist without a valid index in {self.directory}"
                )
            return None, None
        failures = []
        for entry in index["checkpoints"]:
            candidate = self.directory / entry["file"]
            if not candidate.is_file():
                failures.append(f"missing {candidate.name}")
                continue
            actual = file_sha256(candidate)
            if actual != entry["sha256"]:
                failures.append(f"hash mismatch {candidate.name}")
                continue
            return candidate, actual
        raise RuntimeError(
            "No valid indexed step checkpoint remains: " + "; ".join(failures)
        )


def exact_key_check(label: str, expected: Iterable[str], actual: Iterable[str]) -> None:
    expected_set = set(expected)
    actual_set = set(actual)
    if expected_set != actual_set:
        raise RuntimeError(
            f"{label} keys differ: missing={sorted(expected_set - actual_set)}, "
            f"unexpected={sorted(actual_set - expected_set)}"
        )
