#!/usr/bin/env python3
"""Validate four strict P2 cards and build the authoritative summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


EXPECTED = {
    "pusht": {"frame_skip": 5, "windows": 1_981_721, "target_steps": 123_858},
    "wall": {"frame_skip": 5, "windows": 70_848, "target_steps": 143_910},
    "rope": {"frame_skip": 1, "windows": 17_100, "target_steps": 53_500},
    "granular": {"frame_skip": 1, "windows": 17_100, "target_steps": 53_500},
}
TIMING_SCHEMA = "dino-wm.strict-p2-timing.v1"
RUN_CARD_SCHEMA = "dino-wm.strict-p2-run-card.v1"
SUMMARY_SCHEMA = "dino-wm.strict-p2-timing-summary.v1"
WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
SIF_SHA256 = "6992ca7aa544434f80cfb375523ae96a1e41656078b5c2d7e89128288c2aecae"
BASE_COMMIT = "12374e46f27cb9074afe5e3973b78c5561f05657"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def verify_result(path: Path) -> tuple[str, dict[str, Any]]:
    value = json.loads(path.read_text())
    environment = value.get("environment")
    require(environment in EXPECTED, f"unknown environment in {path}: {environment}")
    expected = EXPECTED[environment]
    require(value.get("schema") == TIMING_SCHEMA, f"bad schema in {path}")
    require(value.get("status") == "MEASURED_PASS", f"failed card in {path}")
    require(value.get("arm") == "dinov2_vits14", f"wrong arm in {path}")
    require(value.get("slurm_job_id"), f"missing job id in {path}")
    require("A100" in str(value.get("gpu")), f"job did not record an A100 in {path}")
    require(value.get("gpu_count") == 1, f"job did not expose one GPU in {path}")
    require(value.get("global_batch_size") == 32, f"wrong batch in {path}")
    require(value.get("warmup_steps_excluded") == 20, f"wrong warmup in {path}")
    require(value.get("measured_steps") == 200, f"wrong step count in {path}")
    require(value.get("measured_samples") == 6_400, f"wrong sample count in {path}")
    require(value.get("frame_skip") == expected["frame_skip"], f"wrong frame skip in {path}")
    require(value.get("train_windows") == expected["windows"], f"wrong windows in {path}")
    require(
        value.get("projection_target_steps") == expected["target_steps"],
        f"wrong projection target in {path}",
    )
    require(math.isfinite(float(value.get("final_loss"))), f"nonfinite loss in {path}")
    require(float(value.get("steps_per_second")) > 0, f"invalid steps/s in {path}")
    require(float(value.get("samples_per_second")) > 0, f"invalid samples/s in {path}")
    require(float(value.get("measured_seconds")) > 0, f"invalid wall in {path}")
    require(value.get("assumption_tags") == [], f"unexpected assumption tag in {path}")

    strict = value["strict_settings"]
    require(strict.get("deterministic_algorithms") is True, f"determinism off in {path}")
    require(strict.get("cudnn_benchmark") is False, f"cuDNN benchmark on in {path}")
    require(strict.get("cudnn_deterministic") is True, f"cuDNN nondeterministic in {path}")
    require(strict.get("cuda_matmul_allow_tf32") is False, f"matmul TF32 on in {path}")
    require(strict.get("cudnn_allow_tf32") is False, f"cuDNN TF32 on in {path}")
    require(strict.get("cublas_workspace_config") == ":4096:8", f"bad CUBLAS config in {path}")
    require(strict.get("processes") == 1, f"wrong process count in {path}")
    require(strict.get("num_workers") == 0, f"wrong worker count in {path}")

    config = value["config"]
    require(config["training"]["seed"] == 1, f"wrong seed in {path}")
    require(config["training"]["predictor_lr"] == 5e-5, f"wrong predictor LR in {path}")
    require(config["training"]["checkpoint_every_steps"] == 0, f"checkpointing enabled in {path}")
    require(config["has_decoder"] is False, f"decoder enabled in {path}")
    require(config["model"]["train_encoder"] is False, f"encoder training enabled in {path}")
    require(config["model"]["train_predictor"] is True, f"predictor training off in {path}")
    require(config["model"]["train_decoder"] is False, f"decoder training on in {path}")
    require(config["predictor"]["depth"] == 6, f"wrong predictor depth in {path}")
    require(config["predictor"]["heads"] == 16, f"wrong predictor heads in {path}")
    require(config["predictor"]["mlp_dim"] == 2048, f"wrong predictor MLP in {path}")

    artifacts = value["artifacts"]
    require(artifacts.get("source_base_commit") == BASE_COMMIT, f"wrong base commit in {path}")
    require(artifacts.get("container_sha256") == SIF_SHA256, f"wrong SIF in {path}")
    require(artifacts.get("dinov2_weights_sha256") == WEIGHTS_SHA256, f"wrong weights in {path}")
    for field in [
        "source_commit",
        "train_py_sha256",
        "training_resume_py_sha256",
        "training_timing_py_sha256",
        "slurm_wrapper_sha256",
        "dinov2_repo_commit",
        "dataset_order_sha256",
        "semantic_config_sha256",
        "resolved_config_sha256",
    ]:
        require(artifacts.get(field), f"missing artifact {field} in {path}")

    card_path = Path(value["run_card"])
    require(card_path.is_file(), f"missing run card {card_path}")
    card = yaml.safe_load(card_path.read_text())
    require(card.get("schema") == RUN_CARD_SCHEMA, f"bad run card schema {card_path}")
    require(card.get("status") == "PASSED", f"run card did not pass {card_path}")
    require(card.get("result_sha256") == file_sha256(path), f"result hash mismatch {path}")
    return environment, value


def write_figure(summary: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for environment in EXPECTED:
        rate = summary["environments"][environment]["samples_per_second"]
        axis.scatter([1], [rate], s=65, label=environment)
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.8, 1.25)
    axis.set_xticks([1], labels=["1"])
    axis.set_xlabel("A100 GPUs per job")
    axis.set_ylabel("Measured samples/s")
    axis.set_title("Strict production-path DINOv2 throughput")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    require(len(args.result) == 4, "exactly four result paths are required")
    verified: dict[str, Any] = {}
    result_paths: dict[str, Path] = {}
    source_commits = set()
    for path in args.result:
        environment, value = verify_result(path)
        require(environment not in verified, f"duplicate environment {environment}")
        verified[environment] = value
        result_paths[environment] = path
        source_commits.add(value["artifacts"]["source_commit"])
    require(set(verified) == set(EXPECTED), "the four environment cards are incomplete")
    require(len(source_commits) == 1, "cards used different source commits")

    environments = {}
    for environment in EXPECTED:
        value = verified[environment]
        environments[environment] = {
            "slurm_job_id": value["slurm_job_id"],
            "frame_skip": value["frame_skip"],
            "train_windows": value["train_windows"],
            "window_count_verification": (
                "released-dataset slicer audit" if environment in {"pusht", "wall"}
                else "900 training trajectories * (20 - 2 * 1 + 1) = 17100"
            ),
            "measured_seconds": value["measured_seconds"],
            "steps_per_second": value["steps_per_second"],
            "samples_per_second": value["samples_per_second"],
            "peak_torch_reserved_mib": value["peak_torch_reserved_mib"],
            "peak_nvidia_smi_process_mib": value["peak_nvidia_smi_process_mib"],
            "final_loss": value["final_loss"],
            "projection_target_steps": value["projection_target_steps"],
            "target_wall_seconds_projected": value["target_wall_seconds_projected"],
            "target_wall_hours_projected": value["target_wall_hours_projected"],
            "result_path": str(result_paths[environment]),
            "result_sha256": file_sha256(result_paths[environment]),
        }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "PASS",
        "claim": "four measured strict-path DINOv2 timing cards",
        "source_commit": next(iter(source_commits)),
        "source_base_commit": BASE_COMMIT,
        "protocol": {
            "gpu_per_job": 1,
            "global_batch_size": 32,
            "num_workers": 0,
            "warmup_steps_excluded": 20,
            "measured_optimizer_steps": 200,
            "deterministic_algorithms": True,
            "tf32": False,
            "cublas_workspace_config": ":4096:8",
        },
        "configuration_mismatch_tags": [],
        "dinocular_rate_status": "NOT_MEASURED; projections remain assumptions",
        "environments": environments,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "timing_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)
    write_figure(summary, args.output_dir / "throughput_vs_gpu.png")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
