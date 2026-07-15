#!/usr/bin/env python3
"""Submit and extend an unattended P3 step-training SLURM chain."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "dino-wm.p3-slurm-chain.v1"
DEFAULT_OVERRIDES = [
    "env=pusht",
    "encoder=dino_pinned",
    "training.seed=1",
    "training.predictor_lr=5e-5",
    "training.strict_determinism=true",
    "training.resume_from=auto",
    "frameskip=5",
    "num_hist=3",
    "num_pred=1",
    "has_decoder=false",
    "model.train_encoder=false",
    "model.train_predictor=true",
    "model.train_decoder=false",
    "plan_settings.plan_cfg_path=null",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(f"Unknown chain manifest schema: {path}")
    return manifest


def submit_job(manifest_path: Path, manifest: dict, dependency: str | None) -> str:
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={manifest['job_name']}",
        f"--partition={manifest['partition']}",
        f"--time={manifest['time_limit']}",
        f"--export=ALL,P3_CHAIN_MANIFEST={manifest_path}",
    ]
    if dependency is not None:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(manifest["sbatch_script"])
    output = subprocess.check_output(command, text=True).strip()
    job_id = output.split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job id from {output!r}")
    return job_id


def start(args: argparse.Namespace) -> None:
    if args.target_steps <= 0 or args.segment_steps <= 0:
        raise ValueError("target and segment steps must be positive")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    code_root = Path(__file__).resolve().parents[1]
    manifest_path = run_dir / "chain.json"
    overrides = DEFAULT_OVERRIDES + list(args.override)
    manifest = {
        "schema": SCHEMA,
        "status": "SUBMITTING",
        "created_at": now(),
        "updated_at": now(),
        "run_dir": str(run_dir),
        "target_steps": args.target_steps,
        "segment_steps": args.segment_steps,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "partition": args.partition,
        "time_limit": args.time_limit,
        "job_name": args.job_name,
        "sbatch_script": str((code_root / "tools" / "p3_step_segment.sbatch").resolve()),
        "code_root": str(code_root),
        "overrides": overrides,
        "jobs": [],
        "events": [],
    }
    atomic_json(manifest_path, manifest)
    job_id = submit_job(manifest_path, manifest, None)
    manifest["status"] = "SUBMITTED"
    manifest["jobs"].append(
        {"job_id": job_id, "dependency": None, "submitted_at": now()}
    )
    manifest["updated_at"] = now()
    atomic_json(manifest_path, manifest)
    print(job_id)


def continue_chain(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    progress_path = Path(manifest["run_dir"]) / "progress.json"
    with progress_path.open("r", encoding="utf-8") as handle:
        progress = json.load(handle)
    if int(progress["target_steps"]) != int(manifest["target_steps"]):
        raise RuntimeError("Progress target differs from chain target")
    if progress.get("source_commit") != subprocess.check_output(
        ["git", "-C", manifest["code_root"], "rev-parse", "HEAD"], text=True
    ).strip():
        raise RuntimeError("Progress source commit differs from current branch")

    prior = [event for event in manifest["events"] if event["job_id"] == args.parent_job]
    if prior:
        event = prior[-1]
        if event.get("child_job_id"):
            print(event["child_job_id"])
        else:
            print("TARGET_REACHED")
        return

    event = {
        "job_id": args.parent_job,
        "recorded_at": now(),
        "progress_status": progress["status"],
        "global_step": int(progress["global_step"]),
        "completed_segment_steps": int(progress["completed_segment_steps"]),
        "last_step_loss": progress["last_step_loss"],
        "parameter_sha256": progress["parameter_sha256"],
        "checkpoint_sha256": progress["checkpoint_sha256"],
    }
    if int(progress["global_step"]) == int(manifest["target_steps"]):
        if progress["status"] != "TARGET_REACHED":
            raise RuntimeError("Target step was reached without TARGET_REACHED status")
        manifest["status"] = "PASSED"
        manifest["completed_at"] = now()
        manifest["final_progress"] = progress
        manifest["events"].append(event)
        manifest["updated_at"] = now()
        atomic_json(manifest_path, manifest)
        print("TARGET_REACHED")
        return
    if int(progress["global_step"]) > int(manifest["target_steps"]):
        raise RuntimeError("Training exceeded the chain target")
    if progress["status"] not in {"SEGMENT_COMPLETE", "SIGNAL_CHECKPOINTED"}:
        raise RuntimeError(f"Refusing to continue from status {progress['status']}")

    child_job_id = submit_job(manifest_path, manifest, args.parent_job)
    event["child_job_id"] = child_job_id
    manifest["events"].append(event)
    manifest["jobs"].append(
        {
            "job_id": child_job_id,
            "dependency": f"afterok:{args.parent_job}",
            "submitted_at": now(),
        }
    )
    manifest["status"] = "CHAINED"
    manifest["updated_at"] = now()
    atomic_json(manifest_path, manifest)
    print(child_job_id)


def emit(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest.resolve())
    values = {
        "run_dir": manifest["run_dir"],
        "target_steps": str(manifest["target_steps"]),
        "segment_steps": str(manifest["segment_steps"]),
        "checkpoint_every_steps": str(manifest["checkpoint_every_steps"]),
        "code_root": manifest["code_root"],
    }
    if args.field == "overrides":
        for override in manifest["overrides"]:
            print(override)
    else:
        print(values[args.field])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--run-dir", type=Path, required=True)
    start_parser.add_argument("--target-steps", type=int, required=True)
    start_parser.add_argument("--segment-steps", type=int, required=True)
    start_parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    start_parser.add_argument(
        "--partition",
        choices=["sgpu_devel", "sgpu_short", "sgpu_medium"],
        default="sgpu_short",
    )
    start_parser.add_argument("--time-limit", default="07:55:00")
    start_parser.add_argument("--job-name", default="p3-resume")
    start_parser.add_argument("--override", action="append", default=[])
    start_parser.set_defaults(function=start)

    continue_parser = subparsers.add_parser("continue")
    continue_parser.add_argument("--manifest", type=Path, required=True)
    continue_parser.add_argument("--parent-job", required=True)
    continue_parser.set_defaults(function=continue_chain)

    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("--manifest", type=Path, required=True)
    emit_parser.add_argument(
        "--field",
        choices=[
            "run_dir",
            "target_steps",
            "segment_steps",
            "checkpoint_every_steps",
            "code_root",
            "overrides",
        ],
        required=True,
    )
    emit_parser.set_defaults(function=emit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
