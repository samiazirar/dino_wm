#!/usr/bin/env python3
"""Materialize fail-closed planning launch artifacts for the fixed 36 lineages."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat
from typing import Any, Mapping, Sequence


EVALUATION_SCHEMA = "dinocular.fixed-evaluation-launch-artifacts.v1"
PLANNING_SCHEMA = "dinocular.fixed-planning-launch-artifacts.v1"
CARD_SCHEMA = "dinocular.fixed-planning-card.v1"
RECEIPT_SCHEMA = "dinocular.planning-materialization-receipt.v1"
SELECTION_SEED = 20260714
ENVIRONMENTS = ("pusht", "wall", "rope", "granular")
ARMS = ("dino_pinned", "dinocular", "dinocular_zerodepth")
SEEDS = (1, 2, 3)
TARGET_COUNTS = {"pusht": 50, "wall": 50, "rope": 10, "granular": 10}
TARGET_FILENAMES = {
    "pusht": "pusht_50.pkl",
    "wall": "wall_50.pkl",
    "rope": "rope_10.pkl",
    "granular": "granular_10.pkl",
}
PLANNER = {
    "name": "mpc_cem",
    "horizon": 5,
    "n_taken_actions": 5,
    "num_samples": 100,
    "opt_steps": 10,
    "topk": 30,
    "objective_mode": "last",
    "objective_alpha": 1,
    "action_bounds": "repository_unchanged",
    "base_stream": "sha256(20260714,environment,target_id,replan_index)",
}


class MaterializationError(RuntimeError):
    """The fixed evaluation manifest or an immutable output is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MaterializationError(f"expected JSON object at {path}")
    return value


def _write_immutable(path: Path, text: str, executable: bool = False) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise MaterializationError(f"immutable artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def _target_ids(path: Path, environment: str) -> list[str]:
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception as exc:
        raise MaterializationError(
            f"{environment} fixed target manifest is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise MaterializationError(
            f"{environment} fixed target manifest is not a mapping: {path}"
        )
    ids = value.get("target_ids")
    if (
        not isinstance(ids, (list, tuple))
        or len(ids) != TARGET_COUNTS[environment]
        or any(not isinstance(item, str) or not item for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise MaterializationError(
            f"{environment} target_ids must contain exactly "
            f"{TARGET_COUNTS[environment]} unique nonempty strings"
        )
    return list(ids)


def _validated_launch(path: Path) -> Mapping[str, Any]:
    launch = _object(path)
    expected = {
        f"{environment}/{arm}/s{seed}"
        for environment in ENVIRONMENTS
        for arm in ARMS
        for seed in SEEDS
    }
    records = launch.get("records")
    if (
        launch.get("schema") != EVALUATION_SCHEMA
        or launch.get("state") != "READY"
        or launch.get("selection_seed") != SELECTION_SEED
        or launch.get("lineage_count") != 36
        or not isinstance(records, Mapping)
        or set(records) != expected
    ):
        raise MaterializationError(
            "evaluation launch manifest is not the fixed 36-lineage campaign"
        )
    return launch


def _missing(path: Path, input_name: str) -> Mapping[str, Any]:
    return {"input": input_name, "expected_path": str(path), "reason": "ABSENT"}


def _lineage_evidence(record: Mapping[str, Any]) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    missing = []
    training_dir = Path(str(record["training_run_dir"]))
    chain_path = training_dir / "chain.json"
    receipt_path = training_dir / "final_acceptance.json"
    prediction_path = Path(str(record["evaluation_run_dir"])) / "episode_errors.jsonl"
    for path, name in (
        (chain_path, "accepted_training_chain"),
        (receipt_path, "training_final_acceptance"),
        (prediction_path, "held_out_prediction_result"),
    ):
        if not path.is_file():
            missing.append(_missing(path, name))
    if missing:
        return {}, missing
    chain = _object(chain_path)
    receipt = _object(receipt_path)
    progress = chain.get("final_progress")
    jobs = chain.get("jobs")
    if (
        chain.get("schema") != "dino-wm.p3-slurm-chain.v1"
        or chain.get("status") != "PASSED"
        or not isinstance(progress, Mapping)
        or progress.get("status") != "TARGET_REACHED"
        or not isinstance(jobs, list)
        or not jobs
    ):
        raise MaterializationError(f"accepted training chain is invalid: {chain_path}")
    checkpoint = Path(str(progress.get("checkpoint", "")))
    if not checkpoint.is_file():
        return {}, [_missing(checkpoint, "final_training_checkpoint")]
    return {
        "chain": {"path": str(chain_path), "sha256": _sha256(chain_path)},
        "final_acceptance": {
            "path": str(receipt_path),
            "sha256": _sha256(receipt_path),
            "schema": receipt.get("schema"),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "prediction_result": {
            "path": str(prediction_path),
            "sha256": _sha256(prediction_path),
        },
        "training_tail_job_id": str(jobs[-1].get("job_id")),
    }, []


def _wrapper(card_path: Path, card: Mapping[str, Any], card_file_sha256: str) -> str:
    command = " ".join(json.dumps(str(item)) for item in card["plan_command"])
    return f"""#!/bin/bash
# Immutable no-submit wrapper for {card["lineage_id"]}.
set -euo pipefail
CARD={json.dumps(str(card_path))}
test "$(sha256sum "$CARD" | awk '{{print $1}}')" = {card_file_sha256}
cd {json.dumps(str(card["code_root"]))}
{command}
"""


def materialize(args: argparse.Namespace) -> Mapping[str, Any]:
    evaluation = _validated_launch(args.evaluation_launch_manifest)
    launch_root = args.evaluation_launch_manifest.parent
    target_root = args.target_root or launch_root / "manifests"
    records: dict[str, Any] = {}
    expected_outputs: dict[str, Any] = {}
    missing_tasks: dict[str, Any] = {}
    missing_lineages: dict[str, Any] = {}
    task_targets: dict[str, Any] = {}

    for environment in ENVIRONMENTS:
        target_path = target_root / TARGET_FILENAMES[environment]
        task_missing: list[Mapping[str, Any]] = []
        if not target_path.is_file():
            task_missing.append(_missing(target_path, "fixed_target_manifest"))
        if environment in ("rope", "granular"):
            task_missing.append(
                {
                    "input": "finite_mpc_iteration_cap",
                    "expected_path": None,
                    "reason": "UNKNOWN_IN_FIXED_EVIDENCE",
                }
            )
        target_ids: list[str] = []
        if not task_missing:
            try:
                target_ids = _target_ids(target_path, environment)
            except MaterializationError as exc:
                task_missing.append(
                    {
                        "input": "fixed_target_manifest_contract",
                        "expected_path": str(target_path),
                        "reason": str(exc),
                    }
                )
        task_targets[environment] = {
            "expected_count": TARGET_COUNTS[environment],
            "target_manifest": (
                {"path": str(target_path), "sha256": _sha256(target_path)}
                if target_path.is_file()
                else {"path": str(target_path), "sha256": None}
            ),
            "target_ids": target_ids,
        }
        if task_missing:
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "environment": environment,
                "state": "MISSING_FIXED_INPUT",
                "missing_inputs": task_missing,
            }
            receipt_path = args.out_root / "missing" / f"{environment}.json"
            _write_immutable(
                receipt_path,
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            )
            missing_tasks[environment] = {
                **receipt,
                "receipt": str(receipt_path),
                "receipt_sha256": _sha256(receipt_path),
            }

        for arm in ARMS:
            for seed in SEEDS:
                lineage = f"{environment}/{arm}/s{seed}"
                output_path = (
                    args.out_root
                    / "lineages"
                    / f"seed{seed}"
                    / environment
                    / arm
                    / "planning_results.jsonl"
                )
                expected_outputs[lineage] = {
                    "path": str(output_path),
                    "expected_count": TARGET_COUNTS[environment],
                }
                if task_missing:
                    continue
                source = evaluation["records"][lineage]
                evidence, lineage_missing = _lineage_evidence(source)
                if lineage_missing:
                    missing_lineages[lineage] = lineage_missing
                    continue
                evaluation_card = _object(Path(source["evaluation_card"]))
                card = {
                    "schema": CARD_SCHEMA,
                    "lineage_id": lineage,
                    "environment": environment,
                    "arm": arm,
                    "seed": seed,
                    "target_ids": target_ids,
                    "target_count": len(target_ids),
                    "target_manifest": task_targets[environment]["target_manifest"],
                    "planner": {**PLANNER, "max_iter": 60},
                    "result_contract": {
                        "schema": "dinocular.planning-target-result.v1",
                        "path": str(output_path),
                        "exactly_once_per_target": True,
                        "endpoint": (
                            ["success", "terminal_state_error"]
                            if environment in ("pusht", "wall")
                            else ["chamfer_distance"]
                        ),
                    },
                    "evaluation_launch_manifest": {
                        "path": str(args.evaluation_launch_manifest),
                        "sha256": _sha256(args.evaluation_launch_manifest),
                    },
                    "evaluation_record": copy.deepcopy(source),
                    "accepted_training_evidence": evidence,
                    "source_commit": evaluation_card.get("source_commit"),
                    "code_root": evaluation_card.get("code_root"),
                    "container": evaluation_card.get("container"),
                    "model_data_depth_identities": {
                        key: evaluation_card.get(key)
                        for key in (
                            "source_file_sha256",
                            "artifacts",
                            "config_sha256",
                            "depth_inputs",
                            "fixed_manifest",
                        )
                    },
                    "plan_command": [
                        "python",
                        "plan.py",
                        "--config-name",
                        f"plan_{environment}.yaml"
                        if environment in ("pusht", "wall")
                        else "plan.yaml",
                        "planner=mpc_cem",
                        "goal_source=file",
                        f"goal_file_path={target_path}",
                        f"n_evals={len(target_ids)}",
                        f"seed={SELECTION_SEED}",
                        "goal_H=5",
                        "planner.max_iter=60",
                        "planner.n_taken_actions=5",
                        "planner.sub_planner.horizon=5",
                        "planner.sub_planner.num_samples=100",
                        "planner.sub_planner.opt_steps=10",
                        "planner.sub_planner.topk=30",
                        "objective.mode=last",
                        "objective.alpha=1",
                    ],
                }
                card_path = args.out_root / "cards" / (
                    f"planning-{environment}-{arm}-s{seed}.json"
                )
                card["card_sha256"] = hashlib.sha256(_canonical(card)).hexdigest()
                text = json.dumps(card, indent=2, sort_keys=True) + "\n"
                _write_immutable(card_path, text)
                card_file_sha256 = _sha256(card_path)
                wrapper_path = (
                    output_path.parent.parent / "direct_planning_no_submit.sbatch"
                )
                _write_immutable(
                    wrapper_path,
                    _wrapper(card_path, card, card_file_sha256),
                    True,
                )
                records[lineage] = {
                    "environment": environment,
                    "arm": arm,
                    "seed": seed,
                    "card": str(card_path),
                    "card_sha256": _sha256(card_path),
                    "wrapper": str(wrapper_path),
                    "wrapper_sha256": _sha256(wrapper_path),
                    "result_path": str(output_path),
                    "target_count": len(target_ids),
                }

    result = {
        "schema": PLANNING_SCHEMA,
        "state": "READY" if len(records) == 36 else "PARTIAL_MISSING_INPUTS",
        "selection_seed": SELECTION_SEED,
        "lineage_count": len(records),
        "expected_lineage_count": 36,
        "card_count": len(records),
        "wrapper_count": len(records),
        "expected_result_count": sum(
            TARGET_COUNTS[environment] * 9 for environment in ENVIRONMENTS
        ),
        "materialized_result_count": sum(
            int(record["target_count"]) for record in records.values()
        ),
        "planner": PLANNER,
        "tasks": task_targets,
        "missing_tasks": missing_tasks,
        "missing_lineages": missing_lineages,
        "expected_outputs": expected_outputs,
        "records": records,
    }
    manifest_path = args.out_root / "planning_launch_manifest.json"
    _write_immutable(
        manifest_path, json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "status": result["state"],
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "card_count": len(records),
        "wrapper_count": len(records),
        "expected_result_count": result["expected_result_count"],
        "materialized_result_count": result["materialized_result_count"],
        "missing_tasks": {
            task: value["missing_inputs"]
            for task, value in missing_tasks.items()
        },
        "missing_lineage_count": len(missing_lineages),
    }
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-launch-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    materialize(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MaterializationError as exc:
        print(json.dumps({"status": "FAILED", "reason": str(exc)}, sort_keys=True))
        raise SystemExit(2)
