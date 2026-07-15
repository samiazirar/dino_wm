"""Timing instrumentation for the production deterministic step loop."""

from __future__ import annotations

import math
import os
import platform
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from omegaconf import OmegaConf

from training_resume import atomic_write_json, file_sha256, json_sha256


TIMING_SCHEMA = "dino-wm.strict-p2-timing.v1"
RUN_CARD_SCHEMA = "dino-wm.strict-p2-run-card.v1"


def _atomic_write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(value), handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_commit(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _file_hash_from_environment(name: str) -> str | None:
    path = os.environ.get(name)
    if not path:
        return None
    candidate = Path(path)
    return file_sha256(candidate) if candidate.is_file() else None


class NvidiaSmiMonitor:
    """Sample visible compute-process memory without profiling the model."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = float(interval_seconds)
        self.pid = os.getpid()
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
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                values = []
                for line in completed.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
                        if int(fields[0]) == self.pid:
                            values.append(int(fields[1]))
                if values:
                    self.peak_mib = max(self.peak_mib, sum(values))
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)


class StrictTimingWindow:
    """Measure fixed production optimizer steps after an excluded warmup."""

    @classmethod
    def from_trainer(
        cls,
        trainer: Any,
        *,
        sampler: Any,
        segment_start: int,
        segment_stop: int,
        checkpoint_every: int,
    ) -> "StrictTimingWindow | None":
        output = trainer.cfg.training.timing_output
        if output is None:
            return None
        return cls(
            trainer,
            sampler=sampler,
            segment_start=segment_start,
            segment_stop=segment_stop,
            checkpoint_every=checkpoint_every,
        )

    def __init__(
        self,
        trainer: Any,
        *,
        sampler: Any,
        segment_start: int,
        segment_stop: int,
        checkpoint_every: int,
    ) -> None:
        cfg = trainer.cfg
        self.output_path = Path(str(cfg.training.timing_output))
        run_card = cfg.training.timing_run_card
        if run_card is None:
            raise ValueError("training.timing_run_card is required for timing")
        self.run_card_path = Path(str(run_card))
        self.warmup_steps = int(cfg.training.timing_warmup_steps)
        self.measured_steps = int(cfg.training.timing_measured_steps)
        self.projection_target_steps = int(
            cfg.training.timing_projection_target_steps
        )
        self.required_steps = self.warmup_steps + self.measured_steps
        if self.warmup_steps <= 0 or self.measured_steps <= 0:
            raise ValueError("timing warmup and measured steps must be positive")
        if self.projection_target_steps <= 0:
            raise ValueError("timing projection target must be positive")
        if not bool(cfg.training.strict_determinism):
            raise RuntimeError("strict timing requires training.strict_determinism=true")
        if int(cfg.env.num_workers) != 0:
            raise RuntimeError("strict timing requires env.num_workers=0")
        if trainer.accelerator.num_processes != 1:
            raise RuntimeError("strict timing requires exactly one process")
        if not torch.cuda.is_available():
            raise RuntimeError("strict timing requires CUDA")
        if segment_start != 0 or trainer.global_step != 0:
            raise RuntimeError("strict timing must start from optimizer step zero")
        if segment_stop != self.required_steps:
            raise RuntimeError(
                "strict timing segment must equal warmup plus measured steps"
            )
        if int(cfg.training.target_steps) != self.required_steps:
            raise RuntimeError(
                "strict timing target must equal warmup plus measured steps"
            )
        if checkpoint_every != 0:
            raise RuntimeError("checkpointing must be disabled in the timing segment")
        if cfg.training.resume_from is not None:
            raise RuntimeError("strict timing does not resume from a checkpoint")
        if cfg.training.test_signal_after_step is not None:
            raise RuntimeError("strict timing does not permit a test signal")

        self.device = trainer.device
        self.train_windows = len(trainer.datasets["train"])
        self.steps_per_epoch = int(sampler.steps_per_epoch)
        if self.steps_per_epoch < self.required_steps:
            raise RuntimeError("timing interval must fit within the first epoch")
        self.measured_samples = 0
        self.measured_start: float | None = None
        self.measured_end: float | None = None
        self.peak_allocated_mib: float | None = None
        self.peak_reserved_mib: float | None = None
        self.monitor = NvidiaSmiMonitor()
        self.monitor.start()
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.config_sha256 = json_sha256(self.config)
        self.environment = os.environ.get("STRICT_P2_ENV") or str(cfg.env.name)
        self.artifacts = self._artifacts(trainer)
        self._card = self._initial_card(trainer)
        _atomic_write_yaml(self.run_card_path, self._card)

    def _artifacts(self, trainer: Any) -> dict[str, Any]:
        code_root = str(trainer.base_path)
        timing_path = Path(__file__).resolve()
        return {
            "source_commit": trainer.source_commit,
            "source_base_commit": os.environ.get("STRICT_P2_BASE_COMMIT"),
            "train_py_sha256": file_sha256(Path(code_root) / "train.py"),
            "training_resume_py_sha256": file_sha256(
                Path(code_root) / "training_resume.py"
            ),
            "training_timing_py_sha256": file_sha256(timing_path),
            "slurm_wrapper_sha256": _file_hash_from_environment(
                "STRICT_P2_SLURM_WRAPPER"
            ),
            "container_path": os.environ.get("STRICT_P2_CONTAINER"),
            "container_sha256": os.environ.get("STRICT_P2_CONTAINER_SHA256"),
            "dinov2_repo_commit": _git_commit(os.environ.get("DINOV2_REPO")),
            "dinov2_weights_path": os.environ.get("DINOV2_VITS14_WEIGHTS"),
            "dinov2_weights_sha256": os.environ.get(
                "DINOV2_VITS14_WEIGHTS_SHA256"
            ),
            "dataset_order_sha256": trainer.dataset_order_sha256,
            "semantic_config_sha256": trainer.resume_config_sha256,
            "resolved_config_sha256": self.config_sha256,
        }

    def _initial_card(self, trainer: Any) -> dict[str, Any]:
        return {
            "schema": RUN_CARD_SCHEMA,
            "status": "RUNNING",
            "claim": "strict production-path DINOv2 single-A100 timing",
            "measurement_status": "MEASURED after completion; projections labeled separately",
            "environment": self.environment,
            "arm": "dinov2_vits14",
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "job_name": os.environ.get("SLURM_JOB_NAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "node": socket.gethostname(),
                "tasks": os.environ.get("SLURM_NTASKS"),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "protocol": {
                "processes": trainer.accelerator.num_processes,
                "global_batch_size": int(trainer.cfg.training.batch_size),
                "num_workers": int(trainer.cfg.env.num_workers),
                "frame_skip": int(trainer.cfg.frameskip),
                "warmup_steps_excluded": self.warmup_steps,
                "measured_optimizer_steps": self.measured_steps,
                "projection_target_steps": self.projection_target_steps,
                "checkpointing_in_measured_window": False,
                "evaluation_in_measured_window": False,
                "profiler_in_measured_window": False,
            },
            "artifacts": self.artifacts,
            "resolved_config": self.config,
            "result_path": str(self.output_path),
        }

    def after_step(self, *, completed_step: int, batch_samples: int) -> None:
        if completed_step == self.warmup_steps:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.measured_start = time.perf_counter()
            return
        if completed_step <= self.warmup_steps:
            return
        if self.measured_start is None:
            raise RuntimeError("measured steps began before the warmup boundary")
        self.measured_samples += int(batch_samples)
        if completed_step == self.required_steps:
            torch.cuda.synchronize(self.device)
            self.measured_end = time.perf_counter()
            self.peak_allocated_mib = (
                torch.cuda.max_memory_allocated(self.device) / 2**20
            )
            self.peak_reserved_mib = (
                torch.cuda.max_memory_reserved(self.device) / 2**20
            )
            self.monitor.stop()

    def abort(self, error: BaseException) -> None:
        self.monitor.stop()
        self._card["status"] = "FAILED"
        self._card["failure"] = f"{type(error).__name__}: {error}"
        _atomic_write_yaml(self.run_card_path, self._card)

    def finalize(self, trainer: Any, *, sampler: Any, status: str) -> None:
        self.monitor.stop()
        if status != "TARGET_REACHED":
            raise RuntimeError(f"timing target was not reached: {status}")
        if self.measured_start is None or self.measured_end is None:
            raise RuntimeError("timing window did not complete")
        if self.measured_samples <= 0 or trainer.last_step_loss is None:
            raise RuntimeError("timing result is missing samples or final loss")
        if not math.isfinite(float(trainer.last_step_loss)):
            raise FloatingPointError("timing final loss is nonfinite")
        measured_seconds = self.measured_end - self.measured_start
        steps_per_second = self.measured_steps / measured_seconds
        samples_per_second = self.measured_samples / measured_seconds
        projection_seconds = self.projection_target_steps / steps_per_second
        result = {
            "schema": TIMING_SCHEMA,
            "status": "MEASURED_PASS",
            "arm": "dinov2_vits14",
            "environment": self.environment,
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device),
            "gpu_count": torch.cuda.device_count(),
            "artifacts": self.artifacts,
            "config": self.config,
            "strict_settings": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
                "processes": trainer.accelerator.num_processes,
                "num_workers": int(trainer.cfg.env.num_workers),
            },
            "global_batch_size": int(trainer.cfg.training.batch_size),
            "frame_skip": int(trainer.cfg.frameskip),
            "warmup_steps_excluded": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "measured_samples": self.measured_samples,
            "measured_seconds": measured_seconds,
            "steps_per_second": steps_per_second,
            "samples_per_second": samples_per_second,
            "train_windows": self.train_windows,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs_per_hour_projected": (
                3600.0 * samples_per_second / self.train_windows
            ),
            "epoch_equivalent_seconds_projected": (
                self.train_windows / samples_per_second
            ),
            "projection_target_steps": self.projection_target_steps,
            "target_wall_seconds_projected": projection_seconds,
            "target_wall_hours_projected": projection_seconds / 3600.0,
            "peak_torch_allocated_mib": self.peak_allocated_mib,
            "peak_torch_reserved_mib": self.peak_reserved_mib,
            "peak_nvidia_smi_process_mib": self.monitor.peak_mib,
            "nvidia_smi_samples": self.monitor.samples,
            "final_loss": float(trainer.last_step_loss),
            "checkpoint_after_measured_window": str(trainer._last_checkpoint_path),
            "checkpoint_after_measured_window_sha256": (
                trainer._last_checkpoint_sha256
            ),
            "sampler_state_after_window": sampler.state_dict(trainer.global_step),
            "run_card": str(self.run_card_path),
            "assumption_tags": [],
        }
        atomic_write_json(self.output_path, result)
        self._card["status"] = "PASSED"
        self._card["measurements"] = {
            key: result[key]
            for key in [
                "measured_seconds",
                "steps_per_second",
                "samples_per_second",
                "train_windows",
                "peak_torch_reserved_mib",
                "peak_nvidia_smi_process_mib",
                "final_loss",
            ]
        }
        self._card["projection"] = {
            "target_steps": self.projection_target_steps,
            "wall_seconds": projection_seconds,
            "wall_hours": projection_seconds / 3600.0,
            "status": "PROJECTED_FROM_MEASURED_DINOV2_RATE",
        }
        self._card["result_sha256"] = file_sha256(self.output_path)
        _atomic_write_yaml(self.run_card_path, self._card)
        print(
            "STRICT_P2_TIMING "
            f"status=PASS env={self.environment} steps_per_s={steps_per_second} "
            f"samples_per_s={samples_per_second} loss={trainer.last_step_loss}"
        )
