#!/usr/bin/env python3
"""Submit and extend an unattended P3 step-training SLURM chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

sys_path = str(Path(__file__).resolve().parents[1])
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from p3_completion import (  # noqa: E402
    P3CompletionError,
    canonical_first_heldout_manifest_key,
    is_process_id,
    load_final_receipt,
    validate_final_sampler,
)

try:
    from .harness_common import (
        HarnessError,
        depth_artifact_records,
        require_directory_no_alias,
        require_run_card_authorization,
        require_regular_file_no_alias,
        run_card_receipt_expectations,
        validate_run_card,
    )
except ImportError:
    from harness_common import (
        HarnessError,
        depth_artifact_records,
        require_directory_no_alias,
        require_run_card_authorization,
        require_regular_file_no_alias,
        run_card_receipt_expectations,
        validate_run_card,
    )


SCHEMA = "dino-wm.p3-slurm-chain.v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_candidate_launch_authorization(run_card: dict[str, Any]) -> None:
    """Require candidate authority or the disjoint native legacy schema."""

    try:
        require_run_card_authorization(run_card, operation="chain")
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict:
    try:
        path = require_regular_file_no_alias(path, "chain manifest")
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(f"Unknown chain manifest schema: {path}")
    return manifest


def load_and_validate_run_card(
    path: Path, *, expected_file_sha256: str, expected_content_sha256: str
) -> dict[str, Any]:
    try:
        path = require_regular_file_no_alias(path, "chain run card")
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise RuntimeError("run-card file SHA-256 differs from the immutable chain reference")
    run_card = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(run_card, dict):
        raise RuntimeError("chain run card is invalid")
    try:
        validate_run_card(run_card)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    if run_card.get("run_card_sha256") != expected_content_sha256:
        raise RuntimeError("run-card content SHA-256 differs from the chain reference")
    require_candidate_launch_authorization(run_card)
    return run_card


def verify_progress_evidence(manifest: dict, progress: dict) -> Path:
    expected_run_card_sha256 = manifest.get("run_card_sha256")
    if (
        not isinstance(expected_run_card_sha256, str)
        or len(expected_run_card_sha256) != 64
        or progress.get("immutable_run_card_sha256") != expected_run_card_sha256
    ):
        raise RuntimeError("Progress differs from the immutable run card")
    if not is_process_id(progress.get("training_process_id")):
        raise RuntimeError("Progress has no valid segment training process ID")
    global_step = int(progress["global_step"])
    try:
        run_dir = require_directory_no_alias(manifest["run_dir"], "chain run directory")
        checkpoint = require_regular_file_no_alias(
            Path(str(progress.get("checkpoint"))), "progress checkpoint"
        )
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc
    expected_directory = run_dir / "checkpoints" / "steps"
    expected_name = f"step_{global_step:09d}.pth"
    if checkpoint.parent != expected_directory or checkpoint.name != expected_name:
        raise RuntimeError("Progress checkpoint path differs from the completed step")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != progress.get("checkpoint_sha256"):
        raise RuntimeError("Progress checkpoint SHA-256 differs from the saved file")
    sampler = progress.get("sampler")
    if not isinstance(sampler, dict) or sampler.get("next_step") != global_step:
        raise RuntimeError("Progress sampler cursor differs from the completed step")
    try:
        validate_final_sampler(
            sampler,
            target_steps=global_step,
            dataset_order_sha256=str(sampler.get("dataset_order_sha256")),
        )
    except P3CompletionError as exc:
        raise RuntimeError(str(exc)) from exc
    return checkpoint


def submit_job(manifest_path: Path, manifest: dict, dependency: str | None) -> str:
    try:
        sbatch_script = require_regular_file_no_alias(
            Path(str(manifest["sbatch_script"])), "chain sbatch script"
        )
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc
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
    command.append(str(sbatch_script))
    output = subprocess.check_output(command, text=True).strip()
    job_id = output.split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job id from {output!r}")
    return job_id


def start(args: argparse.Namespace) -> None:
    if args.target_steps <= 0 or args.segment_steps <= 0:
        raise ValueError("target and segment steps must be positive")
    try:
        run_dir = require_directory_no_alias(
            args.run_dir, "chain run directory", allow_missing=True
        )
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc
    run_card_path = args.run_card.expanduser()
    run_card = load_and_validate_run_card(
        run_card_path,
        expected_file_sha256=args.run_card_file_sha256,
        expected_content_sha256=args.run_card_sha256,
    )
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
            "run_card": str(run_card_path),
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
    code_root = Path(__file__).resolve().parents[1]
    manifest_path = run_dir / "chain.json"
    run_card_file_sha256 = args.run_card_file_sha256
    try:
        card_run_dir = require_directory_no_alias(
            Path(str(run_card.get("run_dir", ""))),
            "run-card run directory",
            allow_missing=True,
        )
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc
    if (
        card_run_dir != run_dir
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
    run_dir.mkdir(parents=True, exist_ok=False)
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
        "sbatch_script": str(code_root / "tools" / "p3_step_segment.sbatch"),
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
    manifest_path = args.manifest.expanduser()
    manifest = load_manifest(manifest_path)
    run_card = load_and_validate_run_card(
        Path(manifest["run_card"]),
        expected_file_sha256=str(manifest.get("run_card_file_sha256")),
        expected_content_sha256=str(manifest.get("run_card_sha256")),
    )
    try:
        run_dir = require_directory_no_alias(
            Path(str(manifest["run_dir"])), "chain run directory"
        )
        code_root = require_directory_no_alias(
            Path(str(manifest["code_root"])), "chain code root"
        )
        progress_path = require_regular_file_no_alias(
            run_dir / "progress.json", "chain progress"
        )
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc
    with progress_path.open("r", encoding="utf-8") as handle:
        progress = json.load(handle)
    if int(progress["target_steps"]) != int(manifest["target_steps"]):
        raise RuntimeError("Progress target differs from chain target")
    if (
        progress.get("source_commit")
        != subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
        ).strip()
    ):
        raise RuntimeError("Progress source commit differs from current branch")
    checkpoint = verify_progress_evidence(manifest, progress)

    prior = [
        event for event in manifest["events"] if event["job_id"] == args.parent_job
    ]
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
        "immutable_run_card_sha256": progress["immutable_run_card_sha256"],
        "training_process_id": progress["training_process_id"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": progress["checkpoint_sha256"],
    }
    if int(progress["global_step"]) == int(manifest["target_steps"]):
        if progress["status"] != "TARGET_REACHED":
            raise RuntimeError("Target step was reached without TARGET_REACHED status")
        if manifest.get("run_card_kind") == "p3-training":
            completion = progress.get("p3_completion")
            if not isinstance(completion, dict):
                raise RuntimeError("P3 target progress lacks completion evidence")
            heldout = run_card.get("heldout_loss_manifest")
            if not isinstance(heldout, dict) or any(
                completion.get(completion_field) != heldout.get(card_field)
                for completion_field, card_field in (
                    ("heldout_manifest_sha256", "sha256"),
                    ("data_manifest_sha256", "data_manifest_sha256"),
                    ("split_sha256", "split_sha256"),
                )
            ):
                raise RuntimeError("P3 completion held-out provenance differs")
            try:
                validation_manifest_key = canonical_first_heldout_manifest_key(
                    heldout,
                    target_steps=int(manifest["target_steps"]),
                )
            except (KeyError, P3CompletionError) as exc:
                raise RuntimeError(str(exc)) from exc
            expected_receipt = {
                "slurm_job_id": str(args.parent_job),
                "source_commit": manifest["source_commit"],
                "immutable_run_card_sha256": manifest["run_card_sha256"],
                "config_sha256": run_card["config_sha256"],
                "container_sha256": manifest["container"]["sha256"],
                "target_steps": int(manifest["target_steps"]),
                "global_step": int(manifest["target_steps"]),
                "training_process_id": progress["training_process_id"],
                "checkpoint_sha256": progress["checkpoint_sha256"],
                "parameter_sha256": progress["parameter_sha256"],
                "optimizer_sha256": progress["optimizer_sha256"],
                "scheduler_sha256": progress["scheduler_sha256"],
                "manifest_sha256": heldout["sha256"],
                "data_manifest_sha256": heldout["data_manifest_sha256"],
                "split_sha256": heldout["split_sha256"],
                "training_ledger_sha256": completion["training_ledger_sha256"],
                "validation_ledger_sha256": completion["validation_ledger_sha256"],
                "checkpoint_history_sha256": completion["checkpoint_history_sha256"],
                "dataset_order_sha256": progress["sampler"]["dataset_order_sha256"],
                "sampler": progress["sampler"],
                "validation_batch.manifest_key": validation_manifest_key,
            }
            depth = run_card.get("depth_inputs")
            expected_empirical_adapter_mode, empirical_provenance = (
                run_card_receipt_expectations(run_card)
            )
            expected_receipt.update(
                {
                    "depth_producer_sha256": depth.get("producer_sha256")
                    if depth
                    else None,
                    "depth_cache_manifest_sha256": depth.get("cache_manifest_sha256")
                    if depth
                    else None,
                    "depth_native_contract_sha256": depth.get("native_contract_sha256")
                    if depth
                    else None,
                    "depth_empirical_contract_sha256": depth.get("empirical_contract_sha256")
                    if depth
                    else None,
                    "depth_empirical_provenance": (
                        dict(empirical_provenance)
                        if empirical_provenance is not None
                        else None
                    ),
                    "depth_validation_sha256": depth.get("validation_sha256")
                    if depth
                    else None,
                    "depth_checkpoint_sha256": depth.get("checkpoint_sha256")
                    if depth
                    else None,
                }
            )
            try:
                require_regular_file_no_alias(
                    run_dir / "final_acceptance.json",
                    "final acceptance receipt",
                )
                receipt, receipt_path, receipt_sha256 = load_final_receipt(
                    run_dir,
                    expected_empirical_adapter_mode=expected_empirical_adapter_mode,
                    expected=expected_receipt,
                )
            except (HarnessError, P3CompletionError) as exc:
                raise RuntimeError(str(exc)) from exc
            event["final_acceptance_receipt"] = str(receipt_path)
            event["final_acceptance_receipt_sha256"] = receipt_sha256
            event["final_acceptance_process_id"] = receipt["process_id"]
            manifest["training_process_id"] = progress["training_process_id"]
            manifest["final_acceptance_process_id"] = receipt["process_id"]
            manifest["final_acceptance_receipt"] = str(receipt_path)
            manifest["final_acceptance_receipt_sha256"] = receipt_sha256
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
    manifest = load_manifest(args.manifest.expanduser())
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
    manifest = load_manifest(args.manifest.expanduser())
    try:
        require_directory_no_alias(
            Path(str(manifest["run_dir"])), "chain run directory"
        )
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc
    run_card = load_and_validate_run_card(
        Path(manifest["run_card"]),
        expected_file_sha256=str(manifest.get("run_card_file_sha256")),
        expected_content_sha256=str(manifest.get("run_card_sha256")),
    )
    try:
        code_root = require_directory_no_alias(
            Path(str(manifest["code_root"])), "chain code root"
        )
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc
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
        try:
            path = require_regular_file_no_alias(
                code_root / relative, f"live source file {relative}"
            )
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"live source file hash differs: {relative}")
    records = {
        **manifest.get("artifacts", {}),
        "container": manifest.get("container", {}),
    }
    depth = manifest.get("depth_inputs")
    if isinstance(depth, dict):
        records.update(depth_artifact_records(depth))
    for label, record in records.items():
        try:
            path = require_regular_file_no_alias(
                Path(record.get("path", "")), f"live pinned artifact {label}"
            )
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != record.get("sha256"):
            raise RuntimeError(f"live pinned artifact hash differs: {label}")
    for key, expected in manifest.get("environment_variables", {}).items():
        if os.environ.get(str(key)) != str(expected):
            raise RuntimeError(f"live environment differs from run card for {key}")
    try:
        run_card_path = require_regular_file_no_alias(
            Path(str(manifest["run_card"])), "live chain run card"
        )
    except HarnessError as exc:
        raise RuntimeError(str(exc)) from exc
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
