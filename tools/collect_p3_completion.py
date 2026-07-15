#!/usr/bin/env python3
"""Collect and audit exact completion evidence for the locked 36-cell P3 matrix."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_completion import (  # noqa: E402
    P3CompletionError,
    canonical_first_heldout_manifest_key,
    comparison_verdict,
    is_process_id,
    load_checkpoint_history,
    load_final_receipt,
    load_jsonl,
    plateau_verdict,
    sha256_file,
    validate_checkpoint_evidence_bindings,
    validate_final_sampler,
    validate_training_records,
    validate_validation_records,
    verify_training_tail_index,
)
from tools.harness_common import (  # noqa: E402
    HarnessError,
    LOCKED_ARMS,
    LOCKED_ENVS,
    LOCKED_SEEDS,
    LOCKED_TARGETS,
    canonical_json_bytes,
    load_json,
    load_matrix,
    sha256_bytes,
)


_PAIRED_TOP_LEVEL_DIFFERENCES = frozenset(
    {
        "arm",
        "depth_inputs",
        "encoder_boundary",
        "run_card_sha256",
        "run_dir",
        "run_id",
        "segment_sizing",
        "segment_steps",
    }
)
_PAIRED_ARM_ENVIRONMENT_VARIABLES = frozenset(
    {
        "DINOCULAR_CACHE_PRODUCER_SHA256",
        "DINOCULAR_NATIVE_DEPTH_CONTRACT",
        "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256",
        "DINOCULAR_STUDENT_WEIGHTS",
    }
)


def _normalized_overrides(card: Mapping[str, Any]) -> list[str]:
    arm_specific = (
        "encoder=",
        "+env.dataset.depth_",
        "+env.dataset.native_depth_",
    )
    return [
        str(value)
        for value in card["overrides"]
        if not str(value).startswith(arm_specific)
    ]


def _normalized_paired_card(card: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove only reviewed arm identity/material and rate-derived differences."""
    overrides = [str(value) for value in card["overrides"]]
    if card.get("config_sha256") != sha256_bytes(canonical_json_bytes(overrides)):
        raise HarnessError("paired card config_sha256 differs from its overrides")
    normalized = copy.deepcopy(dict(card))
    for field in _PAIRED_TOP_LEVEL_DIFFERENCES:
        normalized.pop(field, None)
    normalized_overrides = _normalized_overrides(card)
    normalized["overrides"] = normalized_overrides
    normalized["config_sha256"] = sha256_bytes(
        canonical_json_bytes(normalized_overrides)
    )
    if "environment_variables" in normalized:
        environment_variables = normalized["environment_variables"]
        if not isinstance(environment_variables, Mapping):
            raise HarnessError("paired card environment_variables must be a mapping")
        normalized["environment_variables"] = {
            str(key): value
            for key, value in environment_variables.items()
            if str(key) not in _PAIRED_ARM_ENVIRONMENT_VARIABLES
        }
    return normalized


def _load_cell(card: Mapping[str, Any]) -> Mapping[str, Any]:
    run_dir = Path(str(card["run_dir"]))
    progress = load_json(run_dir / "progress.json")
    chain = load_json(run_dir / "chain.json")
    target = int(card["target_steps"])
    jobs = chain.get("jobs")
    tail_job = (
        str(jobs[-1].get("job_id"))
        if isinstance(jobs, list) and jobs and isinstance(jobs[-1], Mapping)
        else ""
    )
    if (
        chain.get("status") != "PASSED"
        or not tail_job.isdigit()
        or progress.get("status") != "TARGET_REACHED"
        or progress.get("global_step") != target
        or progress.get("target_steps") != target
        or progress.get("immutable_run_card_sha256") != card["run_card_sha256"]
        or not is_process_id(progress.get("training_process_id"))
        or chain.get("run_card_sha256") != card["run_card_sha256"]
        or chain.get("final_progress") != progress
    ):
        raise HarnessError(f"cell is not exact TARGET_REACHED/PASSED: {card['run_id']}")
    checkpoint = Path(str(progress["checkpoint"]))
    if (
        checkpoint != run_dir / "checkpoints" / "steps" / f"step_{target:09d}.pth"
        or not checkpoint.is_file()
        or sha256_file(checkpoint) != progress.get("checkpoint_sha256")
    ):
        raise HarnessError(f"cell final checkpoint differs: {card['run_id']}")
    completion = progress.get("p3_completion")
    sampler = progress.get("sampler")
    if not isinstance(completion, Mapping) or not isinstance(sampler, Mapping):
        raise HarnessError(f"cell lacks completion/sampler evidence: {card['run_id']}")
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
            f"cell completion held-out provenance differs: {card['run_id']}"
        )
    try:
        validate_final_sampler(
            sampler,
            target_steps=target,
            dataset_order_sha256=str(sampler.get("dataset_order_sha256")),
            expected_batch_size=int(card["batch_size"]),
        )
        validation_manifest_key = canonical_first_heldout_manifest_key(
            heldout, target_steps=target
        )
    except P3CompletionError as exc:
        raise HarnessError(str(exc)) from exc
    expected_receipt = {
        "slurm_job_id": tail_job,
        "source_commit": card["source_commit"],
        "immutable_run_card_sha256": card["run_card_sha256"],
        "config_sha256": card["config_sha256"],
        "container_sha256": card["container"]["sha256"],
        "target_steps": target,
        "global_step": target,
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
    try:
        receipt, receipt_path, receipt_sha256 = load_final_receipt(
            run_dir, expected=expected_receipt
        )
    except P3CompletionError as exc:
        raise HarnessError(str(exc)) from exc
    event = chain.get("events", [])[-1]
    if (
        event.get("job_id") != tail_job
        or event.get("training_process_id") != progress["training_process_id"]
        or event.get("final_acceptance_process_id") != receipt["process_id"]
        or chain.get("training_process_id") != progress["training_process_id"]
        or chain.get("final_acceptance_process_id") != receipt["process_id"]
        or event.get("final_acceptance_receipt") != str(receipt_path)
        or event.get("final_acceptance_receipt_sha256") != receipt_sha256
        or chain.get("final_acceptance_receipt") != str(receipt_path)
        or chain.get("final_acceptance_receipt_sha256") != receipt_sha256
    ):
        raise HarnessError(
            f"cell receipt is not bound to the chain tail: {card['run_id']}"
        )
    training_path = Path(str(completion["training_ledger"]))
    validation_path = Path(str(completion["validation_ledger"]))
    history_path = Path(str(completion["checkpoint_history"]))
    if (
        training_path != run_dir / "training_steps.jsonl"
        or validation_path != run_dir / "heldout_loss.jsonl"
        or history_path != run_dir / "checkpoints" / "steps" / "checkpoint_history.json"
        or sha256_file(training_path) != completion["training_ledger_sha256"]
        or sha256_file(validation_path) != completion["validation_ledger_sha256"]
        or sha256_file(history_path) != completion["checkpoint_history_sha256"]
    ):
        raise HarnessError(f"cell completion file hash differs: {card['run_id']}")
    training_rows = load_jsonl(training_path)
    validation_rows = load_jsonl(validation_path)
    validate_training_records(
        training_rows,
        source_commit=card["source_commit"],
        immutable_run_card_sha256=card["run_card_sha256"],
        dataset_order_sha256=sampler["dataset_order_sha256"],
        target_steps=target,
        expected_dataset_size=int(sampler["dataset_size"]),
        expected_batch_size=int(card["batch_size"]),
        require_complete=True,
    )
    verify_training_tail_index(
        training_path,
        training_rows,
        source_commit=card["source_commit"],
        immutable_run_card_sha256=card["run_card_sha256"],
        dataset_order_sha256=sampler["dataset_order_sha256"],
        config_sha256=card["config_sha256"],
        dataset_size=int(sampler["dataset_size"]),
        batch_size=int(card["batch_size"]),
        target_steps=target,
    )
    if any(row.get("config_sha256") != card["config_sha256"] for row in training_rows):
        raise HarnessError(f"cell training config provenance differs: {card['run_id']}")
    validate_validation_records(
        validation_rows,
        target_steps=target,
        immutable_run_card_sha256=card["run_card_sha256"],
        manifest_sha256=card["heldout_loss_manifest"]["sha256"],
        require_complete=True,
    )
    depth = card.get("depth_inputs")
    expected_depth = {
        "depth_producer_sha256": depth.get("producer_sha256") if depth else None,
        "depth_cache_manifest_sha256": depth.get("cache_manifest_sha256")
        if depth
        else None,
        "depth_native_contract_sha256": depth.get("native_contract_sha256")
        if depth
        else None,
        "depth_validation_sha256": depth.get("validation_sha256") if depth else None,
        "depth_checkpoint_sha256": depth.get("checkpoint_sha256") if depth else None,
    }
    expected_validation = {
        "source_commit": card["source_commit"],
        "config_sha256": card["config_sha256"],
        "container_sha256": card["container"]["sha256"],
        "manifest_sha256": card["heldout_loss_manifest"]["sha256"],
        "data_manifest_sha256": card["heldout_loss_manifest"]["data_manifest_sha256"],
        "split_sha256": card["heldout_loss_manifest"]["split_sha256"],
        "immutable_run_card_sha256": card["run_card_sha256"],
        **expected_depth,
    }
    for row in validation_rows:
        if any(row.get(key) != value for key, value in expected_validation.items()):
            raise HarnessError(
                f"cell held-out provenance differs from its run card: {card['run_id']}"
            )
    if any(receipt.get(key) != value for key, value in expected_depth.items()):
        raise HarnessError(
            f"cell final receipt depth provenance differs: {card['run_id']}"
        )
    history = load_checkpoint_history(history_path)
    validate_checkpoint_evidence_bindings(
        history,
        directory=history_path.parent,
        source_commit=card["source_commit"],
        immutable_run_card_sha256=card["run_card_sha256"],
        dataset_order_sha256=sampler["dataset_order_sha256"],
        training_rows=training_rows,
        validation_rows=validation_rows,
        final_receipt=receipt,
    )
    reasons = {reason for record in history for reason in record["reasons"]}
    required_reasons = {
        "CONFIGURED_INTERVAL",
        "COMPLETE_EPOCH",
        "SEGMENT_BOUNDARY",
        "EXACT_TARGET",
        "INTEGER_PERCENT",
    }
    if not required_reasons.issubset(reasons) or history[-1]["step"] != target:
        raise HarnessError(f"cell checkpoint reason coverage differs: {card['run_id']}")
    verdict = plateau_verdict(validation_rows)
    return {
        "run_id": card["run_id"],
        "arm": card["arm"],
        "environment": card["environment"],
        "seed": int(card["seed"]),
        "target_steps": target,
        "tail_job_id": tail_job,
        "training_process_id": progress["training_process_id"],
        "final_acceptance_process_id": receipt["process_id"],
        "checkpoint_sha256": progress["checkpoint_sha256"],
        "receipt_sha256": receipt_sha256,
        "dataset_order_sha256": sampler["dataset_order_sha256"],
        "heldout_manifest_sha256": card["heldout_loss_manifest"]["sha256"],
        "depth_producer_sha256": receipt.get("depth_producer_sha256"),
        "depth_cache_manifest_sha256": receipt.get("depth_cache_manifest_sha256"),
        "depth_native_contract_sha256": receipt.get("depth_native_contract_sha256"),
        "depth_validation_sha256": receipt.get("depth_validation_sha256"),
        "depth_checkpoint_sha256": receipt.get("depth_checkpoint_sha256"),
        "curve": [
            {
                "percent": int(row["percent"]),
                "global_step": int(row["global_step"]),
                "mean_loss": float(row["mean_loss"]),
            }
            for row in validation_rows
        ],
        "plateau": verdict,
    }


def _paired_audit(
    cards: Sequence[Mapping[str, Any]],
    cells: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_axis = {
        (str(card["environment"]), str(card["arm"]), int(card["seed"])): card
        for card in cards
    }
    comparisons = []
    for environment in LOCKED_ENVS:
        for seed in LOCKED_SEEDS:
            group_cards = [by_axis[(environment, arm, seed)] for arm in LOCKED_ARMS]
            group_cells = [cells[(environment, arm, seed)] for arm in LOCKED_ARMS]
            normalized_cards = [_normalized_paired_card(card) for card in group_cards]
            normalized_bytes = [canonical_json_bytes(card) for card in normalized_cards]
            if any(value != normalized_bytes[0] for value in normalized_bytes[1:]):
                all_fields = set().union(*(card.keys() for card in normalized_cards))
                differing = sorted(
                    field
                    for field in all_fields
                    if any(
                        (field in card, card.get(field))
                        != (
                            field in normalized_cards[0],
                            normalized_cards[0].get(field),
                        )
                        for card in normalized_cards[1:]
                    )
                )
                raise HarnessError(
                    f"paired non-arm settings differ for {environment}/s{seed}: {differing}"
                )
            dataset_orders = [cell["dataset_order_sha256"] for cell in group_cells]
            if any(value != dataset_orders[0] for value in dataset_orders[1:]):
                raise HarnessError(
                    f"paired dataset order differs for {environment}/s{seed}"
                )
            informative = by_axis[(environment, "dinocular", seed)]
            neutral = by_axis[(environment, "dinocular_zerodepth", seed)]
            if (
                informative.get("depth_inputs") != neutral.get("depth_inputs")
                or informative.get("artifacts", {}).get("dinocular_student")
                != neutral.get("artifacts", {}).get("dinocular_student")
                or informative.get("environment_variables")
                != neutral.get("environment_variables")
                or informative.get("encoder_boundary") != "informative_depth_and_mask"
                or neutral.get("encoder_boundary") != "manifest_neutral_depth_and_mask"
            ):
                raise HarnessError(
                    f"DINOcular/zero pairing differs beyond neutral boundary for {environment}/s{seed}"
                )
            informative_cell = cells[(environment, "dinocular", seed)]
            neutral_cell = cells[(environment, "dinocular_zerodepth", seed)]
            for field in (
                "depth_producer_sha256",
                "depth_cache_manifest_sha256",
                "depth_native_contract_sha256",
                "depth_validation_sha256",
                "depth_checkpoint_sha256",
            ):
                if informative_cell[field] != neutral_cell[field]:
                    raise HarnessError(
                        f"DINOcular/zero completion pairing differs at {field}"
                    )
            for left_index, left in enumerate(LOCKED_ARMS):
                for right in LOCKED_ARMS[left_index + 1 :]:
                    left_cell = cells[(environment, left, seed)]
                    right_cell = cells[(environment, right, seed)]
                    comparisons.append(
                        {
                            "environment": environment,
                            "seed": seed,
                            "left": left,
                            "right": right,
                            "optimization_status": comparison_verdict(
                                left_cell["plateau"], right_cell["plateau"]
                            ),
                        }
                    )
    return comparisons


def collect(args: argparse.Namespace) -> None:
    matrix, cards = load_matrix(args.matrix)
    if matrix.get("kind") != "p3-training" or len(cards) != 36:
        raise HarnessError("P3 completion collector requires the exact 36-cell matrix")
    expected_axes = {
        (environment, arm, seed)
        for environment in LOCKED_ENVS
        for arm in LOCKED_ARMS
        for seed in LOCKED_SEEDS
    }
    actual_axes = {
        (str(card["environment"]), str(card["arm"]), int(card["seed"]))
        for card in cards
    }
    if actual_axes != expected_axes or any(
        card["target_steps"] != LOCKED_TARGETS[str(card["environment"])]
        for card in cards
    ):
        raise HarnessError("P3 completion axes or targets differ")
    cells = {}
    for card in cards:
        axis = (str(card["environment"]), str(card["arm"]), int(card["seed"]))
        cells[axis] = _load_cell(card)
    comparisons = _paired_audit(cards, cells)
    output = {
        "schema": "dino-wm.p3-completion-summary.v1",
        "state": "PASS",
        "matrix_sha256": matrix["matrix_sha256"],
        "source_commit": matrix["source_commit"],
        "cell_count": len(cells),
        "plateau_rule": {
            "early": [76, 77, 78, 79, 80],
            "late": [96, 97, 98, 99, 100],
            "relative_absolute_threshold": 0.02,
        },
        "cells": [cells[axis] for axis in sorted(cells)],
        "comparisons": comparisons,
        "non_plateaued_cells": sorted(
            cell["run_id"]
            for cell in cells.values()
            if not cell["plateau"]["plateaued"]
        ),
        "p4_blocked_by_plateau": False,
    }
    output["summary_sha256"] = sha256_bytes(canonical_json_bytes(output))
    text = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out.exists() and args.out.read_text(encoding="utf-8") != text:
        raise HarnessError("immutable P3 completion summary already differs")
    if not args.out.exists():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, args.out)
    print(json.dumps({"state": "PASS", "cells": 36, "out": str(args.out)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    collect(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HarnessError, P3CompletionError, KeyError, ValueError) as exc:
        print(f"P3 COMPLETION CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
