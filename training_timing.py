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
IMMUTABLE_RUN_CARD_SCHEMA = "dino-wm-run-card-v1"
LEGACY_IMMUTABLE_RUN_CARD_SCHEMA = "dino-wm.legacy-run-card.v1"
IMMUTABLE_RUN_CARD_SCHEMAS = frozenset(
    {IMMUTABLE_RUN_CARD_SCHEMA, LEGACY_IMMUTABLE_RUN_CARD_SCHEMA}
)
DINOCULAR_ARMS = frozenset({"dinocular", "dinocular_zerodepth"})
LOCKED_ARMS = frozenset({"dino_pinned", *DINOCULAR_ARMS})
LOCKED_FRAMESKIPS = {"pusht": 5, "wall": 5, "rope": 1, "granular": 1}
LOCKED_PROJECTION_TARGETS = {
    "pusht": 123_858,
    "wall": 143_910,
    "rope": 53_500,
    "granular": 53_500,
}
RELEASED_TRAIN_WINDOWS = {
    "pusht": 1_981_721,
    "wall": 70_848,
    "rope": 17_100,
    "granular": 17_100,
}
ARM_SPECIFIC_PROJECTION_STATUS = "PROJECTED_FROM_MEASURED_ARM_SPECIFIC_RATE"


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        self.projection_target_steps = int(cfg.training.timing_projection_target_steps)
        self.required_steps = self.warmup_steps + self.measured_steps
        if self.warmup_steps <= 0 or self.measured_steps <= 0:
            raise ValueError("timing warmup and measured steps must be positive")
        if self.projection_target_steps <= 0:
            raise ValueError("timing projection target must be positive")
        if not bool(cfg.training.strict_determinism):
            raise RuntimeError(
                "strict timing requires training.strict_determinism=true"
            )
        if int(cfg.env.num_workers) != 0:
            raise RuntimeError("strict timing requires env.num_workers=0")
        if trainer.accelerator.num_processes != 1:
            raise RuntimeError("strict timing requires exactly one process")
        if not torch.cuda.is_available():
            raise RuntimeError("strict timing requires CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("strict timing requires exactly one visible GPU")
        if "A100" not in torch.cuda.get_device_name(trainer.device):
            raise RuntimeError("strict timing requires one A100 GPU")
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
        self.environment = os.environ.get("STRICT_P2_ENV") or str(cfg.env.name)
        self.arm = os.environ.get("STRICT_P2_ARM") or str(cfg.encoder.name)
        if (
            self.arm not in LOCKED_ARMS
            or self.environment not in RELEASED_TRAIN_WINDOWS
        ):
            raise RuntimeError("strict timing arm or environment is not locked")
        if int(cfg.training.batch_size) != 32:
            raise RuntimeError("strict timing requires global batch size 32")
        if int(cfg.frameskip) != LOCKED_FRAMESKIPS[self.environment]:
            raise RuntimeError("strict timing frame skip differs from the locked card")
        if self.projection_target_steps != LOCKED_PROJECTION_TARGETS[self.environment]:
            raise RuntimeError(
                "strict timing projection target differs from the released target"
            )
        self.train_windows = len(trainer.datasets["train"])
        expected_windows = RELEASED_TRAIN_WINDOWS[self.environment]
        if self.train_windows != expected_windows:
            raise RuntimeError(
                "strict timing released train-window count differs: "
                f"{self.train_windows} versus {expected_windows}"
            )
        self.steps_per_epoch = int(sampler.steps_per_epoch)
        expected_steps_per_epoch = math.ceil(self.train_windows / 32)
        if self.steps_per_epoch != expected_steps_per_epoch:
            raise RuntimeError(
                "timing sampler epoch size differs from released windows"
            )
        if self.steps_per_epoch < self.required_steps:
            raise RuntimeError("timing interval must fit within the first epoch")
        self.measured_samples = 0
        self.measured_start: float | None = None
        self.measured_end: float | None = None
        self.peak_allocated_mib: float | None = None
        self.peak_reserved_mib: float | None = None
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.config_sha256 = json_sha256(self.config)
        self.immutable_card, self.immutable_run_card = self._load_immutable_card(
            trainer
        )
        self.artifacts = self._artifacts(trainer)
        self.monitor = NvidiaSmiMonitor()
        self.monitor.start()
        self.component_timing_enabled = (
            os.environ.get("STRICT_P2_COMPONENT_TIMING", "0") == "1"
        )
        self._component_active = False
        self._batch_cpu_started: float | None = None
        self._train_cpu_started: float | None = None
        self._batch_cpu_seconds = 0.0
        self._train_cpu_seconds = 0.0
        self._batch_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._train_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._encoder_cuda_events: list[
            tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._encoder_started: torch.cuda.Event | None = None
        self._encoder_module = None
        self._encoder_original_forward = None
        if self.component_timing_enabled:
            self._encoder_module = trainer.model.encoder
            self._encoder_original_forward = self._encoder_module.forward
            self._encoder_module.forward = self._timed_encoder_forward
        self._card = self._initial_card(trainer)
        _atomic_write_yaml(self.run_card_path, self._card)

    def _load_immutable_card(
        self, trainer: Any
    ) -> tuple[Mapping[str, Any], dict[str, Any]]:
        raw_path = os.environ.get("STRICT_P2_IMMUTABLE_RUN_CARD")
        expected_digest = os.environ.get("STRICT_P2_IMMUTABLE_RUN_CARD_SHA256")
        if not raw_path or not _is_sha256(expected_digest):
            raise RuntimeError(
                "strict timing requires an immutable run-card path and content hash"
            )
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"immutable timing run card is missing: {path}")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise RuntimeError("immutable timing run card is not an object")
        card = dict(value)
        embedded_digest = card.get("run_card_sha256")
        digest_value = dict(card)
        digest_value.pop("run_card_sha256", None)
        computed_digest = json_sha256(digest_value)
        if embedded_digest != expected_digest or computed_digest != expected_digest:
            raise RuntimeError("immutable timing run-card content hash differs")
        timing = card.get("timing")
        if (
            card.get("schema") not in IMMUTABLE_RUN_CARD_SCHEMAS
            or card.get("kind") != "p2-timing"
            or card.get("gate_mode") != "timing"
            or card.get("arm") != self.arm
            or card.get("environment") != self.environment
            or card.get("source_commit") != trainer.source_commit
            or card.get("batch_size") != 32
            or card.get("frameskip") != LOCKED_FRAMESKIPS[self.environment]
            or card.get("target_steps") != self.required_steps
            or card.get("segment_steps") != self.required_steps
            or timing
            != {
                "fixed_steps": self.measured_steps,
                "warmup_steps": self.warmup_steps,
            }
            or Path(str(card.get("run_dir"))).resolve()
            != self.output_path.parent.resolve()
        ):
            raise RuntimeError(
                "runtime timing configuration differs from the immutable P2 card"
            )
        return card, {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "run_card_sha256": expected_digest,
            "run_id": card.get("run_id"),
        }

    def _verified_file_identity(
        self, record: Any, environment_name: str, label: str
    ) -> dict[str, str]:
        if not isinstance(record, Mapping):
            raise RuntimeError(f"immutable timing card has no {label} record")
        expected_path = record.get("path")
        expected_digest = record.get("sha256")
        runtime_path = os.environ.get(environment_name)
        if (
            not runtime_path
            or runtime_path != expected_path
            or not _is_sha256(expected_digest)
        ):
            raise RuntimeError(f"runtime {label} identity differs from immutable card")
        path = Path(runtime_path)
        if not path.is_file() or file_sha256(path) != expected_digest:
            raise RuntimeError(f"runtime {label} file hash differs from immutable card")
        return {"path": runtime_path, "sha256": str(expected_digest)}

    def _artifacts(self, trainer: Any) -> dict[str, Any]:
        code_root = str(trainer.base_path)
        timing_path = Path(__file__).resolve()
        card_artifacts = self.immutable_card.get("artifacts")
        if not isinstance(card_artifacts, Mapping):
            raise RuntimeError("immutable timing card has no artifact records")
        dinov2 = self._verified_file_identity(
            card_artifacts.get("dinov2"),
            "DINOV2_VITS14_WEIGHTS",
            "DINOv2 weights",
        )
        artifacts = {
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
            "dinov2_weights_path": dinov2["path"],
            "dinov2_weights_sha256": dinov2["sha256"],
            "dataset_order_sha256": trainer.dataset_order_sha256,
            "semantic_config_sha256": trainer.resume_config_sha256,
            "resolved_config_sha256": self.config_sha256,
        }
        source_hashes = self.immutable_card.get("source_file_sha256")
        expected_source_hashes = {
            "train_py_sha256": "train.py",
            "training_resume_py_sha256": "training_resume.py",
            "training_timing_py_sha256": "training_timing.py",
            "slurm_wrapper_sha256": "tools/p3_step_segment.sbatch",
        }
        if not isinstance(source_hashes, Mapping) or any(
            artifacts[field] != source_hashes.get(relative)
            for field, relative in expected_source_hashes.items()
        ):
            raise RuntimeError(
                "runtime timing source hashes differ from immutable card"
            )
        container = self.immutable_card.get("container")
        if (
            not isinstance(container, Mapping)
            or artifacts["container_path"] != container.get("path")
            or artifacts["container_sha256"] != container.get("sha256")
        ):
            raise RuntimeError("runtime timing container differs from immutable card")
        if self.arm in DINOCULAR_ARMS:
            student = self._verified_file_identity(
                card_artifacts.get("dinocular_student"),
                "DINOCULAR_STUDENT_WEIGHTS",
                "DINOcular student",
            )
            depth_inputs = self.immutable_card.get("depth_inputs")
            if not isinstance(depth_inputs, Mapping):
                raise RuntimeError("DINOcular timing card has no depth-input record")
            native = self._verified_file_identity(
                {
                    "path": depth_inputs.get("native_contract_path"),
                    "sha256": depth_inputs.get("native_contract_sha256"),
                },
                "DINOCULAR_NATIVE_DEPTH_CONTRACT",
                "DINOcular native contract",
            )
            selected_producer = os.environ.get("DINOCULAR_CACHE_PRODUCER_SHA256")
            if (
                not _is_sha256(selected_producer)
                or selected_producer != depth_inputs.get("producer_sha256")
                or os.environ.get("DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256")
                != native["sha256"]
                or depth_inputs.get("checkpoint_sha256") != student["sha256"]
            ):
                raise RuntimeError(
                    "DINOcular student, native contract, or producer identity differs"
                )
            artifacts["dinocular_identity"] = {
                "student": student,
                "native_depth_contract": native,
                "selected_producer_sha256": selected_producer,
            }
        else:
            artifacts["dinocular_identity"] = None
        return artifacts

    def _initial_card(self, trainer: Any) -> dict[str, Any]:
        claim = f"strict production-path {self.arm} single-A100 timing"
        return {
            "schema": RUN_CARD_SCHEMA,
            "status": "RUNNING",
            "claim": claim,
            "measurement_status": "MEASURED_ARM_SPECIFIC_AFTER_COMPLETION",
            "environment": self.environment,
            "arm": self.arm,
            "immutable_run_card": self.immutable_run_card,
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
                "component_timing_enabled": self.component_timing_enabled,
            },
            "artifacts": self.artifacts,
            "resolved_config": self.config,
            "result_path": str(self.output_path),
        }

    @staticmethod
    def _cuda_event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    def before_batch(self, *, completed_step: int) -> None:
        if (
            not self.component_timing_enabled
            or completed_step < self.warmup_steps
            or completed_step >= self.required_steps
        ):
            return
        self._batch_cpu_started = time.perf_counter()
        started = self._cuda_event()
        finished = self._cuda_event()
        started.record()
        self._batch_cuda_events.append((started, finished))

    def after_batch(self, *, completed_step: int) -> None:
        if self._batch_cpu_started is None:
            return
        self._batch_cuda_events[-1][1].record()
        self._batch_cpu_seconds += time.perf_counter() - self._batch_cpu_started
        self._batch_cpu_started = None

    def before_train_step(self, *, completed_step: int) -> None:
        if (
            not self.component_timing_enabled
            or completed_step < self.warmup_steps
            or completed_step >= self.required_steps
        ):
            return
        self._train_cpu_started = time.perf_counter()
        self._component_active = True
        started = self._cuda_event()
        finished = self._cuda_event()
        started.record()
        self._train_cuda_events.append((started, finished))

    def after_train_step(self) -> None:
        if self._train_cpu_started is None:
            return
        self._train_cuda_events[-1][1].record()
        self._component_active = False
        self._train_cpu_seconds += time.perf_counter() - self._train_cpu_started
        self._train_cpu_started = None

    def _before_encoder(self) -> None:
        if not self._component_active:
            return
        if self._encoder_started is not None:
            raise RuntimeError("nested encoder forward is unsupported in strict timing")
        self._encoder_started = self._cuda_event()
        self._encoder_started.record()

    def _after_encoder(self) -> None:
        if not self._component_active:
            return
        if self._encoder_started is None:
            raise RuntimeError("encoder timing ended without a matching start")
        finished = self._cuda_event()
        finished.record()
        self._encoder_cuda_events.append((self._encoder_started, finished))
        self._encoder_started = None

    def _timed_encoder_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._encoder_original_forward is None:
            raise RuntimeError("strict timing encoder wrapper is not initialized")
        self._before_encoder()
        try:
            return self._encoder_original_forward(*args, **kwargs)
        finally:
            self._after_encoder()

    def _remove_component_hooks(self) -> None:
        if self._encoder_module is not None and self._encoder_original_forward is not None:
            self._encoder_module.forward = self._encoder_original_forward
        self._encoder_module = None
        self._encoder_original_forward = None

    @staticmethod
    def _elapsed_seconds(
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> float:
        return sum(start.elapsed_time(end) for start, end in events) / 1000.0

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
            self.peak_reserved_mib = torch.cuda.max_memory_reserved(self.device) / 2**20
            self.monitor.stop()

    def abort(self, error: BaseException) -> None:
        self.monitor.stop()
        self._remove_component_hooks()
        self._card["status"] = "FAILED"
        self._card["failure"] = f"{type(error).__name__}: {error}"
        _atomic_write_yaml(self.run_card_path, self._card)

    def finalize(self, trainer: Any, *, sampler: Any, status: str) -> None:
        self.monitor.stop()
        self._remove_component_hooks()
        if status != "TARGET_REACHED":
            raise RuntimeError(f"timing target was not reached: {status}")
        if self.measured_start is None or self.measured_end is None:
            raise RuntimeError("timing window did not complete")
        if self.measured_samples <= 0 or trainer.last_step_loss is None:
            raise RuntimeError("timing result is missing samples or final loss")
        expected_samples = self.measured_steps * 32
        if self.measured_samples != expected_samples:
            raise RuntimeError(
                "timing measured sample count differs: "
                f"{self.measured_samples} versus {expected_samples}"
            )
        if not math.isfinite(float(trainer.last_step_loss)):
            raise FloatingPointError("timing final loss is nonfinite")
        measured_seconds = self.measured_end - self.measured_start
        steps_per_second = self.measured_steps / measured_seconds
        samples_per_second = self.measured_samples / measured_seconds
        projection_seconds = self.projection_target_steps / steps_per_second
        component_timings = None
        if self.component_timing_enabled:
            if (
                len(self._batch_cuda_events) != self.measured_steps
                or len(self._train_cuda_events) != self.measured_steps
                or len(self._encoder_cuda_events) != self.measured_steps
            ):
                raise RuntimeError(
                    "component timing event counts differ from measured steps"
                )
            batch_cuda_seconds = self._elapsed_seconds(self._batch_cuda_events)
            train_cuda_seconds = self._elapsed_seconds(self._train_cuda_events)
            encoder_cuda_seconds = self._elapsed_seconds(self._encoder_cuda_events)
            component_timings = {
                "method": "non-synchronizing CUDA events plus CPU perf_counter",
                "measured_steps": self.measured_steps,
                "batch_cpu_seconds": self._batch_cpu_seconds,
                "batch_cuda_interval_seconds": batch_cuda_seconds,
                "train_cpu_seconds": self._train_cpu_seconds,
                "train_cuda_seconds": train_cuda_seconds,
                "encoder_cuda_seconds": encoder_cuda_seconds,
                "downstream_cuda_seconds": train_cuda_seconds
                - encoder_cuda_seconds,
            }
        result = {
            "schema": TIMING_SCHEMA,
            "status": "MEASURED_PASS",
            "arm": self.arm,
            "environment": self.environment,
            "measurement_claim": self._card["claim"],
            "projection_status": ARM_SPECIFIC_PROJECTION_STATUS,
            "immutable_run_card": self.immutable_run_card,
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
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
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
            "component_timings": component_timings,
            "final_loss": float(trainer.last_step_loss),
            "final_parameter_sha256": trainer._last_checkpoint_state_hashes.get(
                "parameter_sha256"
            ),
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
            "status": ARM_SPECIFIC_PROJECTION_STATUS,
        }
        self._card["result_sha256"] = file_sha256(self.output_path)
        _atomic_write_yaml(self.run_card_path, self._card)
        print(
            "STRICT_P2_TIMING "
            f"status=PASS env={self.environment} steps_per_s={steps_per_second} "
            f"samples_per_s={samples_per_second} loss={trainer.last_step_loss}"
        )
