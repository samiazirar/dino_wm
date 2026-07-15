#!/usr/bin/env python3
"""Collect the exact 12-card P2 timing matrix into a submission rate summary."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_common import (  # noqa: E402
    HarnessError,
    LOCKED_ARMS,
    LOCKED_ENVS,
    LOCKED_FRAMESKIPS,
    LOCKED_TARGETS,
    TIMING_SUMMARY_SCHEMA,
    load_matrix,
    sha256_file,
)


RELEASED_TRAIN_WINDOWS = {
    "pusht": 1_981_721,
    "wall": 70_848,
    "rope": 17_100,
    "granular": 17_100,
}
MEASURED_SAMPLES = 6_400
PROJECTION_STATUS = "PROJECTED_FROM_MEASURED_ARM_SPECIFIC_RATE"
STRICT_SETTINGS = {
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cublas_workspace_config": ":4096:8",
    "processes": 1,
    "num_workers": 0,
}


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise HarnessError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise HarnessError(f"{label} is not an object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise HarnessError(f"nonfinite timing field {label}")
    return float(value)


def _require_exact_int(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise HarnessError(f"timing field {label} must be integer {expected}")


def _require_runtime_bindings(
    *,
    card: Mapping[str, Any],
    reference: Mapping[str, Any],
    result: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    run_id = str(card["run_id"])
    expected_immutable = {
        "path": str(Path(str(reference["path"])).resolve()),
        "file_sha256": reference["file_sha256"],
        "run_card_sha256": reference["run_card_sha256"],
        "run_id": run_id,
    }
    if (
        result.get("immutable_run_card") != expected_immutable
        or runtime.get("immutable_run_card") != expected_immutable
    ):
        raise HarnessError(
            f"timing outputs are not bound to the exact run card for {run_id}"
        )
    result_artifacts = result.get("artifacts")
    runtime_artifacts = runtime.get("artifacts")
    if (
        not isinstance(result_artifacts, Mapping)
        or result_artifacts != runtime_artifacts
    ):
        raise HarnessError(
            f"timing result/runtime artifact identities differ for {run_id}"
        )
    expected_dinov2 = card.get("artifacts", {}).get("dinov2", {})
    if result_artifacts.get("dinov2_weights_path") != expected_dinov2.get(
        "path"
    ) or result_artifacts.get("dinov2_weights_sha256") != expected_dinov2.get("sha256"):
        raise HarnessError(
            f"DINOv2 identity differs from the immutable card for {run_id}"
        )
    if card["arm"] in {"dinocular", "dinocular_zerodepth"}:
        depth = card.get("depth_inputs")
        student = card.get("artifacts", {}).get("dinocular_student")
        if not isinstance(depth, Mapping) or not isinstance(student, Mapping):
            raise HarnessError(f"DINOcular card identities are absent for {run_id}")
        expected_dinocular = {
            "student": {
                "path": student.get("path"),
                "sha256": student.get("sha256"),
            },
            "native_depth_contract": {
                "path": depth.get("native_contract_path"),
                "sha256": depth.get("native_contract_sha256"),
            },
            "selected_producer_sha256": depth.get("producer_sha256"),
        }
    else:
        expected_dinocular = None
    if result_artifacts.get("dinocular_identity") != expected_dinocular:
        raise HarnessError(
            f"DINOcular student/native-contract/producer identity differs for {run_id}"
        )


def _require_strict_configuration(
    result: Mapping[str, Any], runtime: Mapping[str, Any], card: Mapping[str, Any]
) -> None:
    run_id = str(card["run_id"])
    strict = result.get("strict_settings")
    if not isinstance(strict, Mapping) or any(
        strict.get(key) != value for key, value in STRICT_SETTINGS.items()
    ):
        raise HarnessError(f"strict deterministic runtime settings differ for {run_id}")
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise HarnessError(f"resolved timing config is absent for {run_id}")
    training = config.get("training")
    environment = config.get("env")
    model = config.get("model")
    if (
        not isinstance(training, Mapping)
        or not isinstance(environment, Mapping)
        or not isinstance(model, Mapping)
        or training.get("seed") != card.get("seed")
        or training.get("predictor_lr") != card.get("predictor_lr")
        or training.get("strict_determinism") is not True
        or training.get("target_steps") != 220
        or training.get("segment_steps") != 220
        or training.get("checkpoint_every_steps") != 0
        or training.get("resume_from") is not None
        or training.get("batch_size") != 32
        or training.get("timing_warmup_steps") != 20
        or training.get("timing_measured_steps") != 200
        or training.get("timing_projection_target_steps")
        != LOCKED_TARGETS[str(card["environment"])]
        or environment.get("num_workers") != 0
        or config.get("frameskip") != card.get("frameskip")
        or config.get("has_decoder") is not False
        or model.get("train_encoder") is not False
        or model.get("train_predictor") is not True
        or model.get("train_decoder") is not False
    ):
        raise HarnessError(f"resolved timing configuration differs for {run_id}")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise HarnessError(f"immutable timing summary differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def collect(args: argparse.Namespace) -> None:
    matrix, cards = load_matrix(args.matrix)
    if matrix.get("kind") != "p2-timing" or len(cards) != 12:
        raise HarnessError(
            "timing collector requires the exact 12-card P2 timing matrix"
        )
    expected_axes = {
        (arm, environment) for arm in LOCKED_ARMS for environment in LOCKED_ENVS
    }
    actual_axes = {(str(card["arm"]), str(card["environment"])) for card in cards}
    if actual_axes != expected_axes:
        raise HarnessError("P2 timing matrix axes differ from the locked 3x4 set")
    references = {str(item["run_id"]): item for item in matrix["cards"]}
    rates = {}
    evidence = {}
    for card in cards:
        arm = str(card["arm"])
        environment = str(card["environment"])
        run_id = str(card["run_id"])
        reference = references[run_id]
        if (
            card.get("gate_mode") != "timing"
            or card.get("timing") != {"fixed_steps": 200, "warmup_steps": 20}
            or card.get("target_steps") != 220
            or card.get("segment_steps") != 220
            or card.get("frameskip") != LOCKED_FRAMESKIPS[environment]
        ):
            raise HarnessError(f"P2 timing card settings differ for {run_id}")
        run_dir = (
            args.results_root / run_id
            if args.results_root is not None
            else Path(str(card["run_dir"]))
        )
        gate_path = run_dir / "timing_gate.json"
        gate = _load_json(gate_path, "timing gate")
        if (
            gate.get("schema") != "dino-wm-p2-timing-gate-v1"
            or gate.get("state") != "PASS"
            or gate.get("run_id") != run_id
        ):
            raise HarnessError(f"timing gate did not pass for {run_id}")
        result_path = Path(str(gate.get("result_path")))
        runtime_path = Path(str(gate.get("runtime_card_path")))
        if (
            result_path != run_dir / "timing_result.json"
            or runtime_path != run_dir / "timing_runtime_card.yaml"
        ):
            raise HarnessError(
                f"timing gate paths escape the immutable run directory for {run_id}"
            )
        if sha256_file(result_path) != gate.get("result_sha256") or sha256_file(
            runtime_path
        ) != gate.get("runtime_card_sha256"):
            raise HarnessError(f"timing output hash mismatch for {run_id}")
        result = _load_json(result_path, "timing result")
        runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
        if not isinstance(runtime, Mapping):
            raise HarnessError(f"timing runtime card is not an object for {run_id}")
        protocol = runtime.get("protocol", {})
        artifacts = runtime.get("artifacts", {})
        if (
            result.get("schema") != "dino-wm.strict-p2-timing.v1"
            or result.get("status") != "MEASURED_PASS"
            or result.get("arm") != arm
            or result.get("environment") != environment
            or result.get("warmup_steps_excluded") != 20
            or result.get("measured_steps") != 200
            or result.get("global_batch_size") != 32
            or result.get("frame_skip") != LOCKED_FRAMESKIPS[environment]
            or result.get("projection_target_steps") != LOCKED_TARGETS[environment]
            or result.get("projection_status") != PROJECTION_STATUS
            or result.get("measurement_claim")
            != f"strict production-path {arm} single-A100 timing"
            or result.get("assumption_tags") != []
            or runtime.get("schema") != "dino-wm.strict-p2-run-card.v1"
            or runtime.get("status") != "PASSED"
            or runtime.get("arm") != arm
            or runtime.get("environment") != environment
            or runtime.get("claim")
            != f"strict production-path {arm} single-A100 timing"
            or runtime.get("measurement_status")
            != "MEASURED_ARM_SPECIFIC_AFTER_COMPLETION"
            or runtime.get("projection", {}).get("status") != PROJECTION_STATUS
            or runtime.get("projection", {}).get("target_steps")
            != LOCKED_TARGETS[environment]
            or protocol.get("global_batch_size") != 32
            or protocol.get("num_workers") != 0
            or protocol.get("processes") != 1
            or protocol.get("frame_skip") != LOCKED_FRAMESKIPS[environment]
            or protocol.get("warmup_steps_excluded") != 20
            or protocol.get("measured_optimizer_steps") != 200
            or protocol.get("projection_target_steps") != LOCKED_TARGETS[environment]
            or protocol.get("checkpointing_in_measured_window") is not False
            or protocol.get("evaluation_in_measured_window") is not False
            or protocol.get("profiler_in_measured_window") is not False
            or artifacts.get("source_commit") != card.get("source_commit")
            or artifacts.get("container_sha256")
            != card.get("container", {}).get("sha256")
        ):
            raise HarnessError(f"timing result/runtime settings differ for {run_id}")
        _require_runtime_bindings(
            card=card,
            reference=reference,
            result=result,
            runtime=runtime,
        )
        _require_strict_configuration(result, runtime, card)
        _require_exact_int(
            result.get("measured_samples"),
            MEASURED_SAMPLES,
            f"{run_id}.measured_samples",
        )
        _require_exact_int(result.get("gpu_count"), 1, f"{run_id}.gpu_count")
        if "A100" not in str(result.get("gpu")) or not result.get("slurm_job_id"):
            raise HarnessError(
                f"timing card did not record one scheduled A100 for {run_id}"
            )
        expected_windows = RELEASED_TRAIN_WINDOWS[environment]
        _require_exact_int(
            result.get("train_windows"), expected_windows, f"{run_id}.train_windows"
        )
        expected_steps_per_epoch = math.ceil(expected_windows / 32)
        _require_exact_int(
            result.get("steps_per_epoch"),
            expected_steps_per_epoch,
            f"{run_id}.steps_per_epoch",
        )
        sampler = result.get("sampler_state_after_window")
        if not isinstance(sampler, Mapping):
            raise HarnessError(f"timing sampler state is absent for {run_id}")
        for key, expected_value in {
            "dataset_size": expected_windows,
            "batch_size": 32,
            "steps_per_epoch": expected_steps_per_epoch,
            "next_step": 220,
        }.items():
            _require_exact_int(
                sampler.get(key), expected_value, f"{run_id}.sampler.{key}"
            )
        rate = _finite(result.get("steps_per_second"), f"{run_id}.steps_per_second")
        if rate <= 0:
            raise HarnessError(f"nonpositive timing rate for {run_id}")
        for name in (
            "samples_per_second",
            "measured_seconds",
            "final_loss",
            "peak_torch_reserved_mib",
        ):
            _finite(result.get(name), f"{run_id}.{name}")
        measured_seconds = float(result["measured_seconds"])
        samples_per_second = float(result["samples_per_second"])
        if (
            measured_seconds <= 0
            or samples_per_second <= 0
            or not math.isclose(rate, 200.0 / measured_seconds, rel_tol=1e-12)
            or not math.isclose(
                samples_per_second, MEASURED_SAMPLES / measured_seconds, rel_tol=1e-12
            )
            or not math.isclose(samples_per_second, rate * 32.0, rel_tol=1e-12)
        ):
            raise HarnessError(
                f"timing rates are inconsistent with measured counts for {run_id}"
            )
        key = f"{arm}/{environment}"
        rates[key] = {
            "state": "PASS",
            "optimizer_steps_per_second": rate,
            "samples_per_second": float(result["samples_per_second"]),
            "train_windows": result["train_windows"],
            "measured_samples": result["measured_samples"],
            "gpu": result["gpu"],
            "gpu_count": result["gpu_count"],
            "peak_torch_reserved_mib": float(result["peak_torch_reserved_mib"]),
            "result_sha256": gate["result_sha256"],
            "runtime_card_sha256": gate["runtime_card_sha256"],
        }
        evidence[key] = {
            "run_id": run_id,
            "run_card_path": reference["path"],
            "run_card_file_sha256": reference["file_sha256"],
            "run_card_sha256": reference["run_card_sha256"],
            "timing_gate_path": str(gate_path),
            "timing_gate_sha256": sha256_file(gate_path),
            "result_path": str(result_path),
            "runtime_card_path": str(runtime_path),
        }
    output = {
        "schema": TIMING_SUMMARY_SCHEMA,
        "state": "PASS",
        "source_commit": matrix["source_commit"],
        "matrix_path": str(args.matrix.resolve()),
        "matrix_sha256": matrix["matrix_sha256"],
        "card_count": 12,
        "rates": rates,
        "throughput_evidence": evidence,
    }
    _write_immutable(args.out, output)
    print(json.dumps(output, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    collect(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HarnessError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"P2 TIMING COLLECTION FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
