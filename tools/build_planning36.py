#!/usr/bin/env python3
"""Materialize fail-closed planning launch artifacts for the fixed 36 lineages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat
import textwrap
from typing import Any, Mapping, Sequence

import yaml


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
TASK_PLANNER = {
    "pusht": {"max_iter": 60},
    "wall": {"max_iter": 60},
    "rope": {
        "max_iter": 4,
        "stop_on_success": False,
        "evaluation_cap": "predeclared_20_high_level_actions",
        "replan_count": 4,
        "executed_actions_per_replan": 5,
        "executed_action_count": 20,
        "endpoint": "terminal_state_chamfer_after_action_20",
        "best_so_far": False,
    },
    "granular": {
        "max_iter": 4,
        "stop_on_success": False,
        "evaluation_cap": "predeclared_20_high_level_actions",
        "replan_count": 4,
        "executed_actions_per_replan": 5,
        "executed_action_count": 20,
        "endpoint": "terminal_state_chamfer_after_action_20",
        "best_so_far": False,
    },
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


def _evaluation_card(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MaterializationError(
            f"cannot read YAML evaluation card {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise MaterializationError(f"expected YAML object at {path}")
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


def _wrapper(card_path: Path, card: Mapping[str, Any], card_file_sha256: str) -> str:
    resolver = textwrap.dedent(
        r"""
        import hashlib
        import json
        import os
        from pathlib import Path

        def fail(message):
            raise SystemExit(f"planning runtime evidence refused: {message}")

        def sha256(path):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        def object_at(path, label):
            if not path.is_file():
                fail(f"{label} is absent: {path}")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                fail(f"{label} is unreadable: {exc}")
            if not isinstance(value, dict):
                fail(f"{label} is not an object")
            return value

        card_path = Path(os.environ["PLANNING_CARD"])
        expected_card_sha256 = os.environ["PLANNING_CARD_SHA256"]
        if not card_path.is_file() or sha256(card_path) != expected_card_sha256:
            fail("planning card file hash differs")
        card = object_at(card_path, "planning card")
        contract = card["runtime_evidence_contract"]
        chain_path = Path(contract["accepted_training_chain"]["path"])
        receipt_path = Path(contract["final_acceptance"]["path"])
        prediction_path = Path(contract["held_out_prediction"]["path"])
        evaluation_card_path = Path(card["evaluation_card"]["path"])
        launch_manifest_path = Path(card["evaluation_launch_manifest"]["path"])
        target_manifest_path = Path(card["target_manifest"]["path"])
        if (
            not evaluation_card_path.is_file()
            or sha256(evaluation_card_path) != card["evaluation_card"]["sha256"]
        ):
            fail("evaluation card hash differs")
        if (
            not launch_manifest_path.is_file()
            or sha256(launch_manifest_path)
            != card["evaluation_launch_manifest"]["sha256"]
        ):
            fail("evaluation launch manifest hash differs")
        if (
            not target_manifest_path.is_file()
            or sha256(target_manifest_path) != card["target_manifest"]["sha256"]
        ):
            fail("fixed target manifest hash differs")
        chain = object_at(chain_path, "accepted training chain")
        receipt = object_at(receipt_path, "final acceptance")
        progress = chain.get("final_progress")
        jobs = chain.get("jobs")
        events = chain.get("events")
        if (
            chain.get("schema") != contract["accepted_training_chain"]["schema"]
            or chain.get("status") != "PASSED"
            or not isinstance(progress, dict)
            or progress.get("status") != "TARGET_REACHED"
            or not isinstance(jobs, list)
            or not jobs
            or not isinstance(events, list)
            or not events
            or not isinstance(events[-1], dict)
        ):
            fail("training chain is not an accepted completed chain")
        if receipt.get("schema") != contract["final_acceptance"]["schema"]:
            fail("final acceptance schema differs")
        checkpoint_text = progress.get("checkpoint")
        checkpoint_sha256 = progress.get("checkpoint_sha256")
        if not isinstance(checkpoint_text, str) or not checkpoint_text:
            fail("final checkpoint path is absent from accepted chain")
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
            fail("final checkpoint hash is absent from accepted chain")
        checkpoint_path = Path(checkpoint_text)
        if not checkpoint_path.is_file() or sha256(checkpoint_path) != checkpoint_sha256:
            fail("final checkpoint is absent or differs from accepted chain")
        receipt_sha256 = sha256(receipt_path)
        final_event = events[-1]
        if (
            chain.get("final_acceptance_receipt") != str(receipt_path)
            or chain.get("final_acceptance_receipt_sha256") != receipt_sha256
            or final_event.get("final_acceptance_receipt") != str(receipt_path)
            or final_event.get("final_acceptance_receipt_sha256") != receipt_sha256
            or final_event.get("checkpoint") != str(checkpoint_path)
            or final_event.get("checkpoint_sha256") != checkpoint_sha256
        ):
            fail("chain, final event, acceptance, and checkpoint do not hash-bind")
        if not prediction_path.is_file() or prediction_path.stat().st_size == 0:
            fail(f"held-out prediction output is absent or empty: {prediction_path}")
        prediction_sha256 = sha256(prediction_path)
        binding = {
            "schema": "dinocular.planning-runtime-evidence-binding.v1",
            "lineage_id": card["lineage_id"],
            "planning_card": {
                "path": str(card_path),
                "sha256": expected_card_sha256,
            },
            "accepted_training_chain": {
                "path": str(chain_path),
                "sha256": sha256(chain_path),
            },
            "final_acceptance": {
                "path": str(receipt_path),
                "sha256": receipt_sha256,
            },
            "final_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            },
            "held_out_prediction": {
                "path": str(prediction_path),
                "sha256": prediction_sha256,
            },
        }
        binding_path = Path(contract["binding_receipt_path"])
        text = json.dumps(binding, indent=2, sort_keys=True) + "\n"
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        if binding_path.exists():
            if binding_path.read_text(encoding="utf-8") != text:
                fail("immutable runtime evidence binding differs")
        else:
            temporary = binding_path.with_name(
                f".{binding_path.name}.tmp.{os.getpid()}"
            )
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, binding_path)

        checkpoints_dir = checkpoint_path.parent
        model_dir = checkpoints_dir.parent
        if checkpoints_dir.name != "checkpoints" or not checkpoint_path.name.startswith("model_") or checkpoint_path.suffix != ".pth":
            fail("final checkpoint does not use the planning loader layout")
        outputs_dir = model_dir
        while outputs_dir.name != "outputs" and outputs_dir != outputs_dir.parent:
            outputs_dir = outputs_dir.parent
        if outputs_dir.name != "outputs":
            fail("final checkpoint has no outputs model root")
        model_name = str(model_dir.relative_to(outputs_dir))
        model_epoch = checkpoint_path.name[len("model_") : -len(".pth")]
        command = list(card["plan_command"])
        command.extend(
            [
                f"ckpt_base_path={outputs_dir.parent}",
                f"model_name={model_name}",
                f"model_epoch={model_epoch}",
            ]
        )
        os.chdir(card["code_root"])
        os.execvp(command[0], command)
        """
    ).strip()
    return f"""#!/bin/bash
# Immutable no-submit wrapper for {card["lineage_id"]}.
set -euo pipefail
export PLANNING_CARD={json.dumps(str(card_path))}
export PLANNING_CARD_SHA256={json.dumps(card_file_sha256)}
python - <<'PY'
{resolver}
PY
"""


def materialize(args: argparse.Namespace) -> Mapping[str, Any]:
    evaluation = _validated_launch(args.evaluation_launch_manifest)
    launch_root = args.evaluation_launch_manifest.parent
    target_root = args.target_root or launch_root / "manifests"
    records: dict[str, Any] = {}
    expected_outputs: dict[str, Any] = {}
    missing_tasks: dict[str, Any] = {}
    task_targets: dict[str, Any] = {}

    for environment in ENVIRONMENTS:
        target_path = target_root / TARGET_FILENAMES[environment]
        task_missing: list[Mapping[str, Any]] = []
        if not target_path.is_file():
            task_missing.append(_missing(target_path, "fixed_target_manifest"))
        target_ids: list[str] = []
        if target_path.is_file():
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
                evaluation_card = _evaluation_card(Path(source["evaluation_card"]))
                task_planner = TASK_PLANNER[environment]
                training_dir = Path(str(source["training_run_dir"]))
                prediction_path = (
                    Path(str(source["evaluation_run_dir"])) / "episode_errors.jsonl"
                )
                card = {
                    "schema": CARD_SCHEMA,
                    "lineage_id": lineage,
                    "environment": environment,
                    "arm": arm,
                    "seed": seed,
                    "target_ids": target_ids,
                    "target_count": len(target_ids),
                    "target_manifest": task_targets[environment]["target_manifest"],
                    "planner": {**PLANNER, **task_planner},
                    "protocol_interpretation": (
                        "predeclared evaluation cap"
                        if environment in ("rope", "granular")
                        else "repository unchanged"
                    ),
                    "result_contract": {
                        "schema": "dinocular.planning-target-result.v1",
                        "path": str(output_path),
                        "exactly_once_per_target": True,
                        "endpoint": (
                            ["success", "terminal_state_error"]
                            if environment in ("pusht", "wall")
                            else ["terminal_state_chamfer_after_action_20"]
                        ),
                    },
                    "evaluation_launch_manifest": {
                        "path": str(args.evaluation_launch_manifest),
                        "sha256": _sha256(args.evaluation_launch_manifest),
                    },
                    "evaluation_card": {
                        "path": str(source["evaluation_card"]),
                        "sha256": _sha256(Path(source["evaluation_card"])),
                    },
                    "runtime_evidence_contract": {
                        "resolution": "strict_at_wrapper_launch",
                        "accepted_training_chain": {
                            "path": str(training_dir / "chain.json"),
                            "schema": "dino-wm.p3-slurm-chain.v1",
                            "required_status": "PASSED",
                            "required_progress_status": "TARGET_REACHED",
                        },
                        "final_acceptance": {
                            "path": str(training_dir / "final_acceptance.json"),
                            "schema": "dino-wm.p3-final-acceptance.v1",
                        },
                        "final_checkpoint": {
                            "path_source": "accepted_training_chain.final_progress.checkpoint",
                            "sha256_source": "accepted_training_chain.final_progress.checkpoint_sha256",
                        },
                        "held_out_prediction": {
                            "path": str(prediction_path),
                            "sha256_resolution": "compute_at_wrapper_launch",
                        },
                        "binding_receipt_path": str(
                            args.out_root
                            / "runtime_evidence"
                            / f"seed{seed}"
                            / environment
                            / arm
                            / "binding.json"
                        ),
                        "launch_policy": "refuse_unless_all_evidence_exists_and_hash_binds",
                    },
                    "source_commit": evaluation_card.get("source_commit"),
                    "code_root": evaluation_card.get("code_root"),
                    "container": evaluation_card.get("container"),
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
                        f"planner.max_iter={task_planner['max_iter']}",
                        "planner.n_taken_actions=5",
                        "planner.sub_planner.horizon=5",
                        "planner.sub_planner.num_samples=100",
                        "planner.sub_planner.opt_steps=10",
                        "planner.sub_planner.topk=30",
                        "objective.mode=last",
                        "objective.alpha=1",
                    ],
                }
                if environment in ("rope", "granular"):
                    card["plan_command"].append("+planner.stop_on_success=false")
                serialized = _canonical(card).decode("utf-8")
                if (
                    "null" in serialized
                    or "PENDING" in serialized
                    or "UNKNOWN_IN_FIXED_EVIDENCE" in serialized
                ):
                    raise MaterializationError(
                        f"{lineage} planning card contains an unresolved binding"
                    )
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
        "missing_lineages": {},
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
        "lineage_count": result["lineage_count"],
        "card_count": len(records),
        "wrapper_count": len(records),
        "expected_lineage_count": result["expected_lineage_count"],
        "expected_result_count": result["expected_result_count"],
        "materialized_result_count": result["materialized_result_count"],
        "missing_tasks": {
            task: value["missing_inputs"]
            for task, value in missing_tasks.items()
        },
        "missing_lineage_count": 0,
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
