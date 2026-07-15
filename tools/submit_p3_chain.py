#!/usr/bin/env python3
"""Submit and extend an unattended P3 step-training SLURM chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from harness_common import validate_run_card


SCHEMA = "dino-wm.p3-slurm-chain.v1"


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
    existing_manifest_path = run_dir / "chain.json"
    if run_dir.exists():
        if not existing_manifest_path.is_file():
            raise FileExistsError(
                f"run directory exists without an immutable chain manifest: {run_dir}"
            )
        existing = load_manifest(existing_manifest_path)
        expected = {
            "target_steps": args.target_steps,
            "segment_steps": args.segment_steps,
            "checkpoint_every_steps": args.checkpoint_every_steps,
            "partition": args.partition,
            "time_limit": args.time_limit,
            "job_name": args.job_name,
            "run_card": str(args.run_card.resolve()),
            "run_card_file_sha256": args.run_card_file_sha256,
            "run_card_sha256": args.run_card_sha256,
        }
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches or list(existing.get("overrides", [])) != list(args.override):
            raise RuntimeError(
                f"idempotent start differs from existing chain: {mismatches}"
            )
        if not existing.get("jobs"):
            raise RuntimeError("existing chain has no submitted job record")
        print(existing["jobs"][-1]["job_id"])
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    code_root = Path(__file__).resolve().parents[1]
    manifest_path = run_dir / "chain.json"
    run_card_path = args.run_card.resolve()
    if not run_card_path.is_file():
        raise FileNotFoundError(f"run card does not exist: {run_card_path}")
    run_card_file_sha256 = hashlib.sha256(run_card_path.read_bytes()).hexdigest()
    if run_card_file_sha256 != args.run_card_file_sha256:
        raise RuntimeError(
            "run-card file SHA-256 differs from the immutable matrix reference"
        )
    run_card = yaml.safe_load(run_card_path.read_text(encoding="utf-8"))
    if not isinstance(run_card, dict) or run_card.get("run_card_sha256") != args.run_card_sha256:
        raise RuntimeError("run-card content SHA-256 differs from the matrix reference")
    validate_run_card(run_card)
    if (
        Path(run_card.get("run_dir", "")).resolve() != run_dir
        or int(run_card.get("target_steps", -1)) != args.target_steps
        or int(run_card.get("segment_steps", -1)) != args.segment_steps
    ):
        raise RuntimeError("chain settings differ from the immutable run card")
    if list(args.override) != list(run_card.get("overrides", [])):
        raise RuntimeError("chain overrides differ from the immutable run card")
    environment_variables = run_card.get("environment_variables")
    if not isinstance(environment_variables, dict):
        raise RuntimeError("run card has no immutable environment variables")
    for key, expected in environment_variables.items():
        if os.environ.get(str(key)) != str(expected):
            raise RuntimeError(f"launch environment differs from run card for {key}")
    overrides = list(args.override)
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
        "run_card": str(run_card_path),
        "run_card_file_sha256": run_card_file_sha256,
        "run_card_sha256": args.run_card_sha256,
        "run_card_kind": run_card.get("kind"),
        "environment": run_card.get("environment"),
        "arm": run_card.get("arm"),
        "source_commit": run_card.get("source_commit"),
        "source_file_sha256": run_card.get("source_file_sha256"),
        "artifacts": run_card.get("artifacts"),
        "container": run_card.get("container"),
        "depth_inputs": run_card.get("depth_inputs"),
        "environment_variables": environment_variables,
        "jobs": [],
        "events": [],
    }
    atomic_json(manifest_path, manifest)
    job_id = submit_job(manifest_path, manifest, args.dependency)
    manifest["status"] = "SUBMITTED"
    manifest["jobs"].append(
        {"job_id": job_id, "dependency": args.dependency, "submitted_at": now()}
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
        "run_card": manifest["run_card"],
        "run_card_sha256": manifest["run_card_sha256"],
        "kind": str(manifest.get("run_card_kind")),
        "environment": str(manifest.get("environment")),
        "arm": str(manifest.get("arm")),
        "container_sha256": str(manifest.get("container", {}).get("sha256")),
    }
    if args.field == "overrides":
        for override in manifest["overrides"]:
            print(override)
    else:
        print(values[args.field])


def verify(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest.resolve())
    code_root = Path(manifest["code_root"])
    current_commit = subprocess.check_output(
        ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if current_commit != manifest.get("source_commit"):
        raise RuntimeError("live source commit differs from the immutable run card")
    dirty = subprocess.check_output(
        ["git", "-C", str(code_root), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("live source tree is dirty; refusing chain execution")
    for relative, expected in manifest.get("source_file_sha256", {}).items():
        path = code_root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"live source file hash differs: {relative}")
    records = {
        **manifest.get("artifacts", {}),
        "container": manifest.get("container", {}),
    }
    depth = manifest.get("depth_inputs")
    if isinstance(depth, dict):
        records.update(
            {
                "depth cache manifest": {
                    "path": str(Path(depth["cache_dir"]) / "manifest.json"),
                    "sha256": depth["cache_manifest_sha256"],
                },
                "depth validation": {
                    "path": depth["validation_path"],
                    "sha256": depth["validation_sha256"],
                },
                "native depth contract": {
                    "path": depth["native_contract_path"],
                    "sha256": depth["native_contract_sha256"],
                },
            }
        )
    for label, record in records.items():
        path = Path(record.get("path", ""))
        digest = hashlib.sha256()
        if not path.is_file():
            raise RuntimeError(f"missing pinned {label}: {path}")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != record.get("sha256"):
            raise RuntimeError(f"live pinned artifact hash differs: {label}")
    for key, expected in manifest.get("environment_variables", {}).items():
        if os.environ.get(str(key)) != str(expected):
            raise RuntimeError(f"live environment differs from run card for {key}")
    run_card_path = Path(manifest["run_card"])
    if hashlib.sha256(run_card_path.read_bytes()).hexdigest() != manifest.get(
        "run_card_file_sha256"
    ):
        raise RuntimeError("live run-card file hash differs")
    print("CHAIN_PREFLIGHT=PASS")


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
    start_parser.add_argument("--run-card", type=Path, required=True)
    start_parser.add_argument("--run-card-file-sha256", required=True)
    start_parser.add_argument("--run-card-sha256", required=True)
    start_parser.add_argument("--dependency")
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
            "run_card",
            "run_card_sha256",
            "kind",
            "environment",
            "arm",
            "container_sha256",
            "overrides",
        ],
        required=True,
    )
    emit_parser.set_defaults(function=emit)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.set_defaults(function=verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
