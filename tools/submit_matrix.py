#!/usr/bin/env python3
"""Validate immutable matrices and dry-run or submit their Marvin commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_completion import (
    P3CompletionError,
    canonical_first_heldout_manifest_key,
    is_process_id,
    load_final_receipt,
    validate_final_sampler,
)

if __package__:
    from .harness_common import (
        HarnessError,
        depth_artifact_records,
        LOCKED_ARMS,
        LOCKED_ENVS,
        LOCKED_FRAMESKIPS,
        LOCKED_HORIZONS,
        LOCKED_SEEDS,
        LOCKED_TARGETS,
        load_json,
        load_matrix,
        load_timing_summary,
        load_yaml,
        require_directory_no_alias,
        require_run_card_authorization,
        require_real_marvin_path,
        run_card_receipt_expectations,
        require_regular_file_no_alias,
        sha256_file,
        validate_segment_sizing,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness_common import (  # noqa: E402
        HarnessError,
        depth_artifact_records,
        LOCKED_ARMS,
        LOCKED_ENVS,
        LOCKED_FRAMESKIPS,
        LOCKED_HORIZONS,
        LOCKED_SEEDS,
        LOCKED_TARGETS,
        load_json,
        load_matrix,
        load_timing_summary,
        load_yaml,
        require_directory_no_alias,
        require_run_card_authorization,
        require_real_marvin_path,
        run_card_receipt_expectations,
        require_regular_file_no_alias,
        sha256_file,
        validate_segment_sizing,
    )


MODE_KINDS = {
    "canary": "p2-geometry",
    "timing": "p2-timing",
    "producer-pilot": "p2a-producer-pilot",
    "producer-pilot-eval": "p2a-open-loop",
    "train": "p3-training",
    "open-loop": "p4-open-loop",
}
LOCKED_P2_ARMS = "dino_pinned,dinocular,dinocular_zerodepth"
LOCKED_P2_ENVS = "pusht,wall,rope,granular"
LOCKED_P2_FRAMESKIPS = "pusht=5,wall=5,rope=1,granular=1"
CHAIN_KINDS = {"p2-geometry", "p2-timing", "p2a-producer-pilot", "p3-training"}
LOCKED_P2A_PRODUCERS = {"da3_giant_video", "mapanything_recovered_framewise"}


def _matrix_default(mode: str) -> Path:
    root = os.environ.get("STUDY_ROOT")
    if not root:
        raise HarnessError("--matrix or STUDY_ROOT is required")
    names = {
        "canary": "p2_geometry.yaml",
        "timing": "p2_timing.yaml",
        "producer-pilot": "p2a_pusht_producers.yaml",
        "producer-pilot-eval": "p2a_pusht_open_loop.yaml",
        "train": "matrix.yaml",
        "open-loop": "open_loop.yaml",
    }
    return Path(root) / "runcards" / names[mode]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_live_card(card: Mapping[str, Any]) -> None:
    code_root = require_directory_no_alias(
        require_real_marvin_path(str(card.get("code_root")), "code_root"),
        "live code root",
    )
    commit = subprocess.check_output(
        ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != card.get("source_commit"):
        raise HarnessError(f"live source commit differs for {card['run_id']}")
    dirty = subprocess.check_output(
        ["git", "-C", str(code_root), "status", "--porcelain"], text=True
    )
    if dirty:
        raise HarnessError("live Marvin source tree is dirty; sbatch is forbidden")
    for relative, expected in card["source_file_sha256"].items():
        path = require_regular_file_no_alias(
            code_root / relative, f"live source file {relative}"
        )
        if _file_hash(path) != expected:
            raise HarnessError(f"live source file hash differs: {relative}")
    records = dict(card.get("artifacts", {}))
    records["container"] = card["container"]
    depth = card.get("depth_inputs")
    if isinstance(depth, Mapping):
        records.update(depth_artifact_records(depth))
    for label, record in records.items():
        path = require_regular_file_no_alias(
            require_real_marvin_path(str(record.get("path")), f"{label}.path"),
            f"live pinned artifact {label}",
        )
        if _file_hash(path) != record.get("sha256"):
            raise HarnessError(f"live pinned artifact hash differs: {label}")


def _card_paths(matrix: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    return {str(reference["run_id"]): reference for reference in matrix["cards"]}


def _chain_command(
    card: Mapping[str, Any],
    reference: Mapping[str, Any],
    dependency: str | None,
) -> list[str]:
    if card["kind"] == "p2-geometry":
        partition, time_limit = "sgpu_devel", "00:30:00"
    elif card["kind"] == "p2-timing":
        partition, time_limit = "sgpu_devel", "00:50:00"
    else:
        partition, time_limit = "sgpu_short", "07:55:00"
    command = [
        "python3",
        f"{card['code_root']}/tools/submit_p3_chain.py",
        "start",
        "--run-dir",
        str(card["run_dir"]),
        "--target-steps",
        str(card["target_steps"]),
        "--segment-steps",
        str(card["segment_steps"]),
        "--checkpoint-every-steps",
        "1000",
        "--partition",
        partition,
        "--time-limit",
        time_limit,
        "--job-name",
        str(card["run_id"])[:64],
        "--run-card",
        str(reference["path"]),
        "--run-card-file-sha256",
        str(reference["file_sha256"]),
        "--run-card-sha256",
        str(reference["run_card_sha256"]),
    ]
    for override in card["overrides"]:
        command.extend(["--override", str(override)])
    if dependency is not None:
        command.extend(["--dependency", dependency])
    return command


def _batch_command(
    card: Mapping[str, Any], reference: Mapping[str, Any], dependency: str | None
) -> list[str]:
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={str(card['run_id'])[:64]}",
        "--partition=sgpu_short",
        "--time=07:55:00",
        "--export=ALL,"
        f"RUN_CARD={reference['path']},"
        f"RUN_CARD_FILE_SHA256={reference['file_sha256']},"
        f"RUN_CARD_SHA256={reference['run_card_sha256']}",
    ]
    if dependency is not None:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(f"{card['code_root']}/tools/matrix_job.sbatch")
    return command


def _load_dependencies(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    path = require_regular_file_no_alias(path, "dependency index")
    value = load_json(path)
    jobs = value.get("jobs")
    if not isinstance(jobs, Mapping):
        raise HarnessError("dependency index has no jobs mapping")
    result = {str(key): str(job) for key, job in jobs.items()}
    if any(not job.isdigit() for job in result.values()):
        raise HarnessError("dependency index contains a nonnumeric job ID")
    return result


def verify_evaluation_training_dependencies(
    cards: Sequence[Mapping[str, Any]], dependency_jobs: Mapping[str, str]
) -> None:
    for card in cards:
        run_id = str(card["run_id"])
        training_run_id = str(card.get("training_run_id"))
        if card.get("depends_on") != [training_run_id]:
            raise HarnessError(f"evaluation dependency differs for {run_id}")
        indexed_job_id = dependency_jobs.get(training_run_id)
        if indexed_job_id is None:
            raise HarnessError(
                f"evaluation execution has no dependency job for {training_run_id}"
            )
        training_reference = card.get("training_run_card")
        if not isinstance(training_reference, Mapping):
            raise HarnessError(
                f"evaluation card has no training reference for {run_id}"
            )
        training_run_card = require_regular_file_no_alias(
            Path(str(training_reference.get("path"))),
            f"training run card for {run_id}",
        )
        if sha256_file(training_run_card) != training_reference.get("file_sha256"):
            raise HarnessError(f"training run-card file hash differs for {run_id}")
        training_card = load_yaml(training_run_card)
        training_run_dir = require_directory_no_alias(
            Path(str(card.get("training_run_dir"))),
            f"training run directory for {run_id}",
        )
        chain_path = require_regular_file_no_alias(
            training_run_dir / "chain.json", f"training chain for {run_id}"
        )
        progress_path = require_regular_file_no_alias(
            training_run_dir / "progress.json", f"training progress for {run_id}"
        )
        chain = load_json(chain_path)
        progress = load_json(progress_path)
        expected_run_card_sha256 = training_reference.get("run_card_sha256")
        target_steps = int(card["target_steps"])
        chain_run_dir = require_directory_no_alias(
            Path(str(chain.get("run_dir"))), f"chain run directory for {run_id}"
        )
        chain_run_card = require_regular_file_no_alias(
            Path(str(chain.get("run_card"))), f"chain run card for {run_id}"
        )
        if (
            chain.get("schema") != "dino-wm.p3-slurm-chain.v1"
            or chain.get("status") != "PASSED"
            or chain_run_dir != training_run_dir
            or int(chain.get("target_steps", -1)) != target_steps
            or chain.get("run_card_sha256") != expected_run_card_sha256
            or chain_run_card != training_run_card
            or chain.get("run_card_file_sha256")
            != training_reference.get("file_sha256")
            or chain.get("source_commit") != card.get("source_commit")
        ):
            raise HarnessError(f"training chain is not exact and PASSED for {run_id}")
        jobs = chain.get("jobs")
        if not isinstance(jobs, list) or not jobs or not isinstance(jobs[-1], Mapping):
            raise HarnessError(f"training chain has no tail job for {run_id}")
        tail_job_id = str(jobs[-1].get("job_id"))
        if not tail_job_id.isdigit() or indexed_job_id != tail_job_id:
            raise HarnessError(
                f"dependency job for {training_run_id} is not chain tail {tail_job_id}"
            )
        checkpoint = require_regular_file_no_alias(
            Path(str(progress.get("checkpoint"))),
            f"training checkpoint for {run_id}",
        )
        expected_checkpoint = (
            training_run_dir / "checkpoints" / "steps" / f"step_{target_steps:09d}.pth"
        )
        sampler = progress.get("sampler")
        if (
            progress.get("status") != "TARGET_REACHED"
            or int(progress.get("global_step", -1)) != target_steps
            or int(progress.get("target_steps", -1)) != target_steps
            or progress.get("source_commit") != card.get("source_commit")
            or progress.get("immutable_run_card_sha256") != expected_run_card_sha256
            or not is_process_id(progress.get("training_process_id"))
            or checkpoint != expected_checkpoint
            or sha256_file(checkpoint) != progress.get("checkpoint_sha256")
            or not isinstance(sampler, Mapping)
            or sampler.get("next_step") != target_steps
        ):
            raise HarnessError(
                f"training progress/checkpoint is not exact TARGET_REACHED for {run_id}"
            )
        if chain.get("final_progress") != progress:
            raise HarnessError(
                f"chain final progress differs from progress file for {run_id}"
            )
        events = chain.get("events")
        if (
            not isinstance(events, list)
            or not events
            or not isinstance(events[-1], Mapping)
        ):
            raise HarnessError(f"training chain has no final event for {run_id}")
        final_event = events[-1]
        final_event_checkpoint = require_regular_file_no_alias(
            Path(str(final_event.get("checkpoint"))),
            f"training final-event checkpoint for {run_id}",
        )
        if (
            str(final_event.get("job_id")) != tail_job_id
            or final_event.get("progress_status") != "TARGET_REACHED"
            or int(final_event.get("global_step", -1)) != target_steps
            or final_event.get("immutable_run_card_sha256") != expected_run_card_sha256
            or final_event.get("training_process_id") != progress["training_process_id"]
            or final_event_checkpoint != checkpoint
            or final_event.get("checkpoint_sha256") != progress.get("checkpoint_sha256")
        ):
            raise HarnessError(f"training chain final event differs for {run_id}")
        if card.get("kind") == "p4-open-loop":
            try:
                validate_final_sampler(
                    sampler,
                    target_steps=target_steps,
                    dataset_order_sha256=str(sampler.get("dataset_order_sha256")),
                    expected_batch_size=int(card["batch_size"]),
                )
            except P3CompletionError as exc:
                raise HarnessError(str(exc)) from exc
            completion_reference = card.get("training_completion_receipt")
            completion = progress.get("p3_completion")
            if not isinstance(completion_reference, Mapping) or not isinstance(
                completion, Mapping
            ):
                raise HarnessError(f"P4 lacks P3 completion evidence for {run_id}")
            heldout = card.get("heldout_loss_manifest")
            if not isinstance(heldout, Mapping) or any(
                completion.get(completion_field) != heldout.get(card_field)
                for completion_field, card_field in (
                    ("heldout_manifest_sha256", "sha256"),
                    ("data_manifest_sha256", "data_manifest_sha256"),
                    ("split_sha256", "split_sha256"),
                )
            ):
                raise HarnessError(
                    f"P4 completion held-out provenance differs for {run_id}"
                )
            expected_receipt_path = require_regular_file_no_alias(
                training_run_dir / "final_acceptance.json",
                f"final acceptance receipt for {run_id}",
            )
            completion_reference_path = require_regular_file_no_alias(
                Path(str(completion_reference.get("path"))),
                f"P4 completion receipt reference for {run_id}",
            )
            if completion_reference_path != expected_receipt_path:
                raise HarnessError(f"P4 completion receipt path differs for {run_id}")
            try:
                validation_manifest_key = canonical_first_heldout_manifest_key(
                    heldout, target_steps=target_steps
                )
            except (KeyError, P3CompletionError) as exc:
                raise HarnessError(str(exc)) from exc
            expected_receipt = {
                "slurm_job_id": tail_job_id,
                "source_commit": card["source_commit"],
                "immutable_run_card_sha256": expected_run_card_sha256,
                "config_sha256": card["config_sha256"],
                "container_sha256": card["container"]["sha256"],
                "target_steps": target_steps,
                "global_step": target_steps,
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
                "dataset_order_sha256": sampler["dataset_order_sha256"],
                "sampler": sampler,
                "validation_batch.manifest_key": validation_manifest_key,
            }
            depth = card.get("depth_inputs")
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
                    "depth_validation_sha256": depth.get("validation_sha256")
                    if depth
                    else None,
                    "depth_checkpoint_sha256": depth.get("checkpoint_sha256")
                    if depth
                    else None,
                }
            )
            try:
                expected_empirical_adapter_mode, _empirical_provenance = (
                    run_card_receipt_expectations(training_card)
                )
                receipt, receipt_path, receipt_sha256 = load_final_receipt(
                    training_run_dir,
                    expected_empirical_adapter_mode=expected_empirical_adapter_mode,
                    expected=expected_receipt,
                )
            except P3CompletionError as exc:
                raise HarnessError(str(exc)) from exc
            chain_receipt_path = require_regular_file_no_alias(
                Path(str(chain.get("final_acceptance_receipt"))),
                f"chain final acceptance receipt for {run_id}",
            )
            event_receipt_path = require_regular_file_no_alias(
                Path(str(final_event.get("final_acceptance_receipt"))),
                f"final-event acceptance receipt for {run_id}",
            )
            if (
                receipt_path != expected_receipt_path
                or chain.get("training_process_id") != progress["training_process_id"]
                or chain.get("final_acceptance_process_id") != receipt["process_id"]
                or final_event.get("final_acceptance_process_id")
                != receipt["process_id"]
                or chain_receipt_path != receipt_path
                or chain.get("final_acceptance_receipt_sha256") != receipt_sha256
                or event_receipt_path != receipt_path
                or final_event.get("final_acceptance_receipt_sha256") != receipt_sha256
            ):
                raise HarnessError(
                    f"P4 is not bound to the accepted P3 receipt for {run_id}"
                )


def _write_submission_state(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_rates(
    path: Path, cards: Sequence[Mapping[str, Any]], max_hours: float
) -> None:
    path = require_regular_file_no_alias(path, "timing summary")
    if float(max_hours) != 8.0:
        raise HarnessError("productive wall limit must remain exactly 8 hours")
    rates = load_timing_summary(path)
    accepted = rates.get("rates")
    summary_sha256 = sha256_file(path)
    for card in cards:
        key = f"{card['arm']}/{card['environment']}"
        record = accepted.get(key)
        if not isinstance(record, Mapping) or record.get("state") != "PASS":
            raise HarnessError(f"no accepted strict timing rate for {key}")
        rate = record.get("optimizer_steps_per_second")
        if not isinstance(rate, (int, float)) or rate <= 0:
            raise HarnessError(f"invalid strict timing rate for {key}")
        sizing = card.get("segment_sizing")
        validate_segment_sizing(
            sizing,
            arm=str(card["arm"]),
            environment=str(card["environment"]),
            target_steps=int(card["target_steps"]),
        )
        if (
            sizing.get("timing_summary_sha256") != summary_sha256
            or sizing.get("optimizer_steps_per_second") != float(rate)
            or int(card["segment_steps"]) != sizing.get("derived_segment_steps")
        ):
            raise HarnessError(
                f"run card is not bound to the supplied timing summary for {key}"
            )
        projected = float(card["segment_steps"]) / float(rate) / 3600.0
        safe_hours = max_hours * (1.0 - float(sizing["safety_margin_fraction"]))
        if projected > safe_hours:
            raise HarnessError(
                f"segment for {key} projects to {projected:.6f} h, above safe {safe_hours} h"
            )


def _validate_locked_cards(kind: str, cards: Sequence[Mapping[str, Any]]) -> None:
    for card in cards:
        environment = card.get("environment")
        if environment not in LOCKED_ENVS:
            raise HarnessError("run card contains an unlocked environment")
        if (
            card.get("frameskip") != LOCKED_FRAMESKIPS[environment]
            or card.get("horizons") != LOCKED_HORIZONS[environment]
            or card.get("batch_size") != 32
            or card.get("predictor_lr") != 0.00005
            or card.get("decoder") is not False
            or card.get("strict_resume") is not True
        ):
            raise HarnessError(f"locked run settings differ for {card.get('run_id')}")
    if kind in {"p3-training", "p4-open-loop"}:
        expected_axes = {
            (arm, environment, seed)
            for arm in LOCKED_ARMS
            for environment in LOCKED_ENVS
            for seed in LOCKED_SEEDS
        }
        actual_axes = {
            (card.get("arm"), card.get("environment"), card.get("seed"))
            for card in cards
        }
        if actual_axes != expected_axes:
            raise HarnessError("P3/P4 card axes differ from the locked 3x4x3 matrix")
        for card in cards:
            environment = str(card["environment"])
            if card.get("target_steps") != LOCKED_TARGETS[environment]:
                raise HarnessError(f"locked step settings differ for {card['run_id']}")
            validate_segment_sizing(
                card.get("segment_sizing"),
                arm=str(card["arm"]),
                environment=environment,
                target_steps=int(card["target_steps"]),
            )
            if (
                card.get("segment_steps")
                != card["segment_sizing"]["derived_segment_steps"]
            ):
                raise HarnessError(
                    "run card segment differs from its accepted sizing record"
                )
    elif kind in {"p2-geometry", "p2-timing"}:
        expected_axes = {
            (arm, environment, 1) for arm in LOCKED_ARMS for environment in LOCKED_ENVS
        }
        actual_axes = {
            (card.get("arm"), card.get("environment"), card.get("seed"))
            for card in cards
        }
        if actual_axes != expected_axes:
            raise HarnessError("P2 card axes differ from the locked 3x4 seed-1 matrix")
        expected_steps = 1 if kind == "p2-geometry" else 220
        if any(
            card.get("target_steps") != expected_steps
            or card.get("segment_steps") != expected_steps
            for card in cards
        ):
            raise HarnessError("P2 card steps differ from the selected gate mode")
    elif kind == "p2a-producer-pilot":
        producers = {card.get("producer_pilot", {}).get("producer") for card in cards}
        if producers != LOCKED_P2A_PRODUCERS:
            raise HarnessError("P2a cards differ from the locked two producers")
        if any(
            card.get("arm") != "dinocular"
            or card.get("environment") != "pusht"
            or card.get("seed") != 1
            or card.get("target_steps") != 123858
            for card in cards
        ):
            raise HarnessError("P2a cards differ from the locked PushT pilot")
        for card in cards:
            producer = card.get("producer_pilot", {}).get("producer")
            expected_assumptions = (
                ["[ASSUMPTION: RECOVERED-CONTRACT]"]
                if producer == "mapanything_recovered_framewise"
                else []
            )
            if card.get("assumption_tags") != expected_assumptions:
                raise HarnessError("P2a assumption provenance differs")
            validate_segment_sizing(
                card.get("segment_sizing"),
                arm="dinocular",
                environment="pusht",
                target_steps=123858,
            )
            if (
                card.get("segment_steps")
                != card["segment_sizing"]["derived_segment_steps"]
            ):
                raise HarnessError(
                    "P2a segment differs from its accepted sizing record"
                )
    elif kind == "p2a-open-loop":
        producers = {card.get("producer_pilot", {}).get("producer") for card in cards}
        if producers != LOCKED_P2A_PRODUCERS or any(
            card.get("arm") != "dinocular"
            or card.get("environment") != "pusht"
            or card.get("seed") != 1
            or not isinstance(card.get("training_run_dir"), str)
            or not isinstance(card.get("training_run_card"), Mapping)
            for card in cards
        ):
            raise HarnessError(
                "P2a evaluation cards differ from the locked paired pilot"
            )
        for card in cards:
            producer = card.get("producer_pilot", {}).get("producer")
            expected_assumptions = (
                ["[ASSUMPTION: RECOVERED-CONTRACT]"]
                if producer == "mapanything_recovered_framewise"
                else []
            )
            if card.get("assumption_tags") != expected_assumptions:
                raise HarnessError("P2a evaluation assumption provenance differs")


def build_dry_run(
    matrix: Mapping[str, Any],
    cards: Sequence[Mapping[str, Any]],
    dependency_jobs: Mapping[str, str],
) -> list[Mapping[str, Any]]:
    references = _card_paths(matrix)
    result = []
    for card in cards:
        reference = references[str(card["run_id"])]
        dependencies = list(card.get("depends_on", []))
        missing = [item for item in dependencies if item not in dependency_jobs]
        if missing:
            dependency = ":".join(f"<{item}>" for item in missing)
        else:
            dependency = (
                ":".join(dependency_jobs[item] for item in dependencies) or None
            )
        if card["kind"] in CHAIN_KINDS:
            command = _chain_command(card, reference, dependency)
        else:
            command = _batch_command(card, reference, dependency)
        result.append(
            {
                "run_id": card["run_id"],
                "environment_variables": card["environment_variables"],
                "dependencies": dependencies,
                "command": command,
            }
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=sorted(MODE_KINDS))
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-live", action="store_true")
    parser.add_argument("--dependency-index", type=Path)
    parser.add_argument("--rates", type=Path)
    parser.add_argument("--max-productive-hours", type=float, default=8.0)
    parser.add_argument("--encoders")
    parser.add_argument("--envs")
    parser.add_argument("--seeds")
    parser.add_argument("--minutes", type=int)
    parser.add_argument("--fixed-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--frameskips")
    args = parser.parse_args(argv)
    matrix_path = require_regular_file_no_alias(
        args.matrix or _matrix_default(args.mode), "submission matrix"
    )
    matrix, cards = load_matrix(matrix_path)
    expected_kind = MODE_KINDS[args.mode]
    if matrix.get("kind") != expected_kind or any(
        card.get("kind") != expected_kind for card in cards
    ):
        raise HarnessError(f"{args.mode} command received the wrong matrix kind")
    if expected_kind in {"p2-geometry", "p2-timing"} and len(cards) != 12:
        raise HarnessError("P2 matrix must contain exactly 12 cards")
    if expected_kind in {"p2-geometry", "p2-timing"}:
        if (
            args.encoders != LOCKED_P2_ARMS
            or args.envs != LOCKED_P2_ENVS
            or args.seeds != "1"
            or args.frameskips != LOCKED_P2_FRAMESKIPS
        ):
            raise HarnessError(
                "P2 encoder, environment, seed, or frameskip CLI locks differ"
            )
        if expected_kind == "p2-geometry":
            if (
                args.minutes != 30
                or args.fixed_steps is not None
                or args.warmup_steps is not None
            ):
                raise HarnessError("P2 geometry must use the locked 30-minute gate")
            if any(card.get("gate_mode") != "geometry" for card in cards):
                raise HarnessError("P2 geometry card does not invoke the geometry gate")
        else:
            if args.minutes != 50 or args.fixed_steps != 200 or args.warmup_steps != 20:
                raise HarnessError(
                    "P2 timing must use locked 50-minute 20+200 settings"
                )
            if any(
                card.get("gate_mode") != "timing"
                or card.get("timing") != {"fixed_steps": 200, "warmup_steps": 20}
                for card in cards
            ):
                raise HarnessError(
                    "P2 timing card does not invoke the strict timing gate"
                )
    if expected_kind == "p2a-producer-pilot" and len(cards) != 2:
        raise HarnessError("P2a matrix must contain exactly two cards")
    if expected_kind == "p2a-open-loop" and len(cards) != 2:
        raise HarnessError("P2a evaluation matrix must contain exactly two cards")
    if expected_kind in {"p3-training", "p4-open-loop"} and len(cards) != 36:
        raise HarnessError("P3/P4 matrix must contain exactly 36 cards")
    _validate_locked_cards(expected_kind, cards)
    if args.rates is not None:
        _validate_rates(args.rates, cards, args.max_productive_hours)
    elif args.execute and expected_kind in {"p2a-producer-pilot", "p3-training"}:
        raise HarnessError("P2a/P3 execution requires accepted per-arm timing rates")
    if args.execute:
        for card in cards:
            require_run_card_authorization(card, operation="submit")
    if args.verify_live or args.execute:
        for card in cards:
            verify_live_card(card)
    dependencies = _load_dependencies(args.dependency_index)
    if args.execute and expected_kind in {"p2a-open-loop", "p4-open-loop"}:
        verify_evaluation_training_dependencies(cards, dependencies)
    dry_run = build_dry_run(matrix, cards, dependencies)
    if not args.execute:
        print(
            json.dumps(
                {
                    "schema": "dino-wm-submit-dry-run-v1",
                    "state": "PASS",
                    "mode": args.mode,
                    "matrix_sha256": matrix["matrix_sha256"],
                    "job_count": len(dry_run),
                    "jobs": dry_run,
                    "sbatch_calls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if any(job["dependencies"] for job in dry_run):
        missing = [
            dependency
            for job in dry_run
            for dependency in job["dependencies"]
            if dependency not in dependencies
        ]
        if missing:
            raise HarnessError(
                f"execution lacks dependency job IDs: {sorted(set(missing))}"
            )
    state_path = matrix_path.with_suffix(".submissions.json")
    if state_path.exists():
        state_path = require_regular_file_no_alias(
            state_path, "submission state"
        )
        state = load_json(state_path)
        if state.get("matrix_sha256") != matrix["matrix_sha256"]:
            raise HarnessError("submission state belongs to a different matrix hash")
        submitted = {
            str(key): str(value) for key, value in state.get("jobs", {}).items()
        }
    else:
        state = {
            "schema": "dino-wm-matrix-submissions-v1",
            "matrix_sha256": matrix["matrix_sha256"],
            "jobs": {},
        }
        submitted = {}
    for job, card in zip(dry_run, cards):
        run_id = str(card["run_id"])
        if run_id in submitted:
            if not submitted[run_id].isdigit():
                raise HarnessError(f"invalid recorded job ID for {run_id}")
            continue
        environment = os.environ.copy()
        environment.update(
            {str(k): str(v) for k, v in job["environment_variables"].items()}
        )
        output = subprocess.check_output(
            job["command"], text=True, env=environment
        ).strip()
        job_id = output.split(";")[0]
        if not job_id.isdigit():
            raise HarnessError(f"cannot parse submitted job ID from {output!r}")
        submitted[run_id] = job_id
        state["jobs"] = dict(submitted)
        _write_submission_state(state_path, state)
    print(json.dumps({"state": "SUBMITTED", "jobs": submitted}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as exc:
        print(f"HARNESS CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
