from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import p3_completion
from p3_completion import (
    CHECKPOINT_HISTORY_SCHEMA,
    FINAL_RECEIPT_SCHEMA,
    HELDOUT_ENTRY_SCHEMA,
    P3CompletionError,
    TRAINING_RECORD_SCHEMA,
    VALIDATION_RECORD_SCHEMA,
    append_training_record,
    append_validation_record,
    comparison_verdict,
    load_checkpoint_history,
    load_final_receipt,
    materialize_heldout_manifest,
    percent_step,
    percent_step_map,
    plateau_verdict,
    sha256_file,
    validate_checkpoint_evidence_bindings,
    validate_training_records,
    validate_validation_records,
    write_final_receipt,
)
from tools.collect_p3_completion import _paired_audit
from tools.harness_common import (
    HarnessError,
    LOCKED_ARMS,
    LOCKED_ENVS,
    LOCKED_SEEDS,
    canonical_json_bytes,
    sha256_bytes,
)
from tools.submit_matrix import verify_evaluation_training_dependencies
from training_resume import CHECKPOINT_SCHEMA, StepCheckpointManager


def _write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def test_percent_mapping_uses_exact_ceiling_rule_and_unique_target_points():
    mapping = percent_step_map(123_858)
    assert mapping[1] == 1_239
    assert mapping[50] == 61_929
    assert mapping[100] == 123_858
    assert len(set(mapping.values())) == 100
    assert percent_step(53_500, 1) == 535
    with pytest.raises(P3CompletionError, match="at least 100"):
        percent_step_map(99)


def test_checkpoint_history_records_every_reason_and_duplicate_is_idempotent(tmp_path):
    manager = StepCheckpointManager(tmp_path)
    required = [
        "INITIAL_STATE",
        "CONFIGURED_INTERVAL",
        "COMPLETE_EPOCH",
        "SEGMENT_BOUNDARY",
        "USR1",
        "SIGNAL_STOP",
        "EXACT_TARGET",
        "INTEGER_PERCENT",
        "LOADER_STOP",
    ]
    for step, reason in enumerate(required, 1):
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "global_step": step,
            "source_commit": "f" * 40,
            "immutable_run_card_sha256": "a" * 64,
            "dataset_order_sha256": "b" * 64,
            "state": torch.tensor([step]),
        }
        path, digest = manager.save(payload, step, reasons=(reason,))
        assert sha256_file(path) == digest
        if step == len(required):
            repeated_path, repeated_digest = manager.save(
                payload, step, reasons=(reason,)
            )
            assert (repeated_path, repeated_digest) == (path, digest)
    history = load_checkpoint_history(manager.history_path)
    assert {reason for row in history for reason in row["reasons"]} == set(required)
    same_path, same_digest = manager.save(payload, len(required), reasons=("USR1",))
    assert (same_path, same_digest) == (path, digest)
    same_step = [
        row
        for row in load_checkpoint_history(manager.history_path)
        if row["step"] == len(required)
    ]
    assert [row["reasons"] for row in same_step] == [
        ["LOADER_STOP"],
        ["LOADER_STOP", "USR1"],
    ]

    coalesced_step = len(required) + 1
    coalesced_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "global_step": coalesced_step,
        "source_commit": "f" * 40,
        "immutable_run_card_sha256": "a" * 64,
        "dataset_order_sha256": "b" * 64,
        "state": torch.tensor([coalesced_step]),
    }
    coalesced_reasons = (
        "SEGMENT_BOUNDARY",
        "EXACT_TARGET",
        "INTEGER_PERCENT",
    )
    coalesced_path, coalesced_digest = manager.save(
        coalesced_payload,
        coalesced_step,
        reasons=coalesced_reasons,
    )
    repeated_path, repeated_digest = manager.save(
        coalesced_payload,
        coalesced_step,
        reasons=tuple(reversed(coalesced_reasons)),
    )
    assert (repeated_path, repeated_digest) == (
        coalesced_path,
        coalesced_digest,
    )
    coalesced_records = [
        row
        for row in load_checkpoint_history(manager.history_path)
        if row["step"] == coalesced_step
    ]
    assert len(coalesced_records) == 1
    assert coalesced_records[0]["reasons"] == sorted(coalesced_reasons)
    assert (
        json.loads(manager.history_path.read_text())["schema"]
        == CHECKPOINT_HISTORY_SCHEMA
    )
    assert len(list(tmp_path.glob("step_*.pth"))) == 2
    divergent = dict(coalesced_payload)
    divergent["state"] = torch.tensor([-1])
    with pytest.raises(P3CompletionError, match="divergent duplicate"):
        manager.save(divergent, coalesced_step, reasons=coalesced_reasons)


def _entry(environment: str, partition: str, episode: int, start: int):
    return {
        "schema": HELDOUT_ENTRY_SCHEMA,
        "environment": environment,
        "key": f"{environment}/{partition}/{episode:06d}/{start:06d}-000002-f1",
        "source_partition": partition,
        "episode": episode,
        "start": start,
        "end": 2,
        "frameskip": 1,
        "num_frames": 2,
    }


def test_heldout_manifest_is_all_validation_fixed_and_rejects_leakage(tmp_path):
    data = tmp_path / "DATASET_MANIFEST.json"
    _write_json(data, {"released": True})
    record = materialize_heldout_manifest(
        environment="rope",
        training_entries=[_entry("rope", "shared", 1, 0)],
        validation_entries=[_entry("rope", "shared", 2, 0)],
        data_manifest_path=data,
        source_commit="f" * 40,
        out_path=tmp_path / "heldout_rope.jsonl",
    )
    metadata = json.loads(Path(record["metadata_path"]).read_text())
    assert metadata["selection"] == "all_validation_examples"
    assert metadata["entry_count"] == 1
    assert record["sha256"] == sha256_file(record["path"])
    with pytest.raises(P3CompletionError, match="episodes leak"):
        materialize_heldout_manifest(
            environment="rope",
            training_entries=[_entry("rope", "shared", 2, 1)],
            validation_entries=[_entry("rope", "shared", 2, 0)],
            data_manifest_path=data,
            source_commit="f" * 40,
            out_path=tmp_path / "leaking.jsonl",
        )


def _training_row(step: int, *, source="f" * 40):
    return {
        "schema": TRAINING_RECORD_SCHEMA,
        "source_commit": source,
        "immutable_run_card_sha256": "a" * 64,
        "dataset_order_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "global_step": step,
        "completed_epochs": 0,
        "sampler": {"next_step": step},
        "loss": 1.0 / step,
        "parameter_sha256": "d" * 64,
        "optimizer_sha256": "e" * 64,
        "scheduler_sha256": "1" * 64,
        "slurm_job_id": "12345",
        "checkpoint": {
            "step": step,
            "path": f"/tmp/step_{step:09d}.pth",
            "checkpoint_sha256": "2" * 64,
            "history_record_sha256": "3" * 64,
        },
    }


def test_training_ledger_is_exactly_once_and_rejects_gap_duplicate_or_drift(tmp_path):
    path = tmp_path / "training_steps.jsonl"
    for step in range(1, 4):
        append_training_record(path, _training_row(step), target_steps=100)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    validate_training_records(
        rows,
        source_commit="f" * 40,
        immutable_run_card_sha256="a" * 64,
        dataset_order_sha256="b" * 64,
        target_steps=100,
    )
    assert append_training_record(path, _training_row(2), target_steps=100) == rows[1]
    changed = _training_row(2)
    changed["loss"] = 9.0
    with pytest.raises(P3CompletionError, match="differs"):
        append_training_record(path, changed, target_steps=100)
    with pytest.raises(P3CompletionError, match="provenance drift"):
        append_training_record(
            path, _training_row(4, source="0" * 40), target_steps=100
        )
    with pytest.raises(P3CompletionError, match="gap"):
        append_training_record(path, _training_row(5), target_steps=100)
    drift = [dict(row) for row in rows]
    drift[1]["source_commit"] = "0" * 40
    with pytest.raises(P3CompletionError, match="provenance drift"):
        validate_training_records(
            drift,
            source_commit="f" * 40,
            immutable_run_card_sha256="a" * 64,
            dataset_order_sha256="b" * 64,
            target_steps=100,
        )
    with pytest.raises(P3CompletionError, match="cover every"):
        validate_training_records(
            rows,
            source_commit="f" * 40,
            immutable_run_card_sha256="a" * 64,
            dataset_order_sha256="b" * 64,
            target_steps=100,
            require_complete=True,
        )


def test_consecutive_training_appends_do_not_full_scan_or_revalidate(
    tmp_path, monkeypatch
):
    path = tmp_path / "training_steps.jsonl"

    def unexpected_full_scan(*_args, **_kwargs):
        raise AssertionError("normal consecutive append performed a full scan")

    monkeypatch.setattr(p3_completion, "load_jsonl", unexpected_full_scan)
    monkeypatch.setattr(
        p3_completion, "validate_training_records", unexpected_full_scan
    )
    records = [
        append_training_record(path, _training_row(step), target_steps=100)
        for step in range(1, 11)
    ]
    marker = json.loads(
        (tmp_path / "training_steps.tail.json").read_text(encoding="utf-8")
    )
    assert marker["next_step"] == 11
    assert marker["tail_record_sha256"] == records[-1]["record_sha256"]
    assert marker["ledger_size_bytes"] == path.stat().st_size
    assert len(path.read_text(encoding="utf-8").splitlines()) == 10


def test_training_replay_uses_validated_direct_index(tmp_path, monkeypatch):
    path = tmp_path / "training_steps.jsonl"
    for step in range(1, 4):
        append_training_record(path, _training_row(step), target_steps=100)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    class OnePassRows(list):
        def __init__(self, values):
            super().__init__(values)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("replay lookup scanned the validated rows")
            return super().__iter__()

    guarded_rows = OnePassRows(rows)
    monkeypatch.setattr(p3_completion, "load_jsonl", lambda _path: guarded_rows)

    assert append_training_record(
        path, _training_row(2), target_steps=100
    ) == rows[1]
    assert guarded_rows.iterations == 1


def test_ledgers_resolve_exact_checkpoint_history_records(tmp_path):
    manager = StepCheckpointManager(tmp_path / "steps")
    checkpoint_path, checkpoint_sha256 = manager.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "global_step": 1,
            "source_commit": "f" * 40,
            "immutable_run_card_sha256": "a" * 64,
            "dataset_order_sha256": "b" * 64,
        },
        1,
        reasons=("INTEGER_PERCENT",),
    )
    history = load_checkpoint_history(manager.history_path)
    training = _training_row(1)
    training["checkpoint"] = {
        "step": 1,
        "path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "history_record_sha256": history[0]["record_sha256"],
    }
    training_path = tmp_path / "training.jsonl"
    append_training_record(training_path, training, target_steps=100)
    training_rows = [
        json.loads(line) for line in training_path.read_text().splitlines()
    ]
    validation = _validation_row(1, 1.0)
    validation["checkpoint_sha256"] = checkpoint_sha256
    validation["checkpoint_history_record_sha256"] = history[0]["record_sha256"]
    validation_path = tmp_path / "validation.jsonl"
    append_validation_record(validation_path, validation, target_steps=100)
    validation_rows = [
        json.loads(line) for line in validation_path.read_text().splitlines()
    ]
    validate_checkpoint_evidence_bindings(
        history,
        directory=manager.directory,
        source_commit="f" * 40,
        immutable_run_card_sha256="a" * 64,
        dataset_order_sha256="b" * 64,
        training_rows=training_rows,
        validation_rows=validation_rows,
    )
    changed = [dict(training_rows[0])]
    changed[0]["checkpoint"] = dict(changed[0]["checkpoint"])
    changed[0]["checkpoint"]["history_record_sha256"] = "0" * 64
    with pytest.raises(P3CompletionError, match="unknown checkpoint history"):
        validate_checkpoint_evidence_bindings(
            history,
            directory=manager.directory,
            source_commit="f" * 40,
            immutable_run_card_sha256="a" * 64,
            dataset_order_sha256="b" * 64,
            training_rows=changed,
        )


def _validation_row(percent: int, mean: float):
    return {
        "schema": VALIDATION_RECORD_SCHEMA,
        "percent": percent,
        "global_step": percent,
        "target_steps": 100,
        "rounding_rule": "ceil(target_steps*percent/100)",
        "loss_numerator": mean * 10,
        "element_count": 10,
        "mean_loss": mean,
        "source_commit": "f" * 40,
        "config_sha256": "a" * 64,
        "container_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "checkpoint_history_record_sha256": "e" * 64,
        "manifest_sha256": "1" * 64,
        "data_manifest_sha256": "2" * 64,
        "split_sha256": "3" * 64,
        "immutable_run_card_sha256": "4" * 64,
        "slurm_job_id": "12345",
        "state_restored": True,
    }


def _plateau_rows(early: float, late: float):
    return [
        _validation_row(percent, late if percent >= 96 else early)
        for percent in range(1, 101)
    ]


def test_validation_resume_coverage_and_inconclusive(tmp_path):
    path = tmp_path / "heldout_loss.jsonl"
    for percent in range(1, 101):
        mean = 1.02 if percent >= 96 else 1.0
        append_validation_record(path, _validation_row(percent, mean), target_steps=100)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    validate_validation_records(
        rows,
        target_steps=100,
        immutable_run_card_sha256="4" * 64,
        manifest_sha256="1" * 64,
        require_complete=True,
    )
    with pytest.raises(P3CompletionError, match=r"exact 1\.\.100 coverage"):
        validate_validation_records(
            rows[:-1],
            target_steps=100,
            immutable_run_card_sha256="4" * 64,
            manifest_sha256="1" * 64,
            require_complete=True,
        )
    duplicate = rows[:2] + [dict(rows[1])] + rows[2:]
    with pytest.raises(P3CompletionError, match="duplicates a percent"):
        validate_validation_records(
            duplicate,
            target_steps=100,
            immutable_run_card_sha256="4" * 64,
            manifest_sha256="1" * 64,
        )
    assert comparison_verdict(
        {"plateaued": True}, {"plateaued": False}
    ) == "optimization-inconclusive"
    nonfinite = _validation_row(1, 1.0)
    nonfinite["mean_loss"] = float("nan")
    bad_path = tmp_path / "bad.jsonl"
    with pytest.raises(P3CompletionError, match="finite JSON"):
        append_validation_record(bad_path, nonfinite, target_steps=100)
    assert not bad_path.exists()


def test_plateau_exact_boundary_is_inclusive():
    boundary = plateau_verdict(_plateau_rows(1.0, 1.02))
    assert boundary["relative_absolute_change"] == 0.02
    assert boundary["plateaued"] is True


def test_plateau_just_above_boundary_is_rejected():
    above = plateau_verdict(_plateau_rows(10.0, 10.200000000000001))
    assert above["relative_absolute_change"] == 0.0200000000000001
    assert above["plateaued"] is False


def _receipt(**updates):
    value = {
        "schema": FINAL_RECEIPT_SCHEMA,
        "state": "PASS",
        "fresh_model_process": True,
        "process_id": 123,
        "training_process_id": 456,
        "slurm_job_id": "222",
        "source_commit": "f" * 40,
        "immutable_run_card_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "container_sha256": "c" * 64,
        "target_steps": 100,
        "global_step": 100,
        "sampler": {"next_step": 100},
        "checkpoint": "/tmp/step_000000100.pth",
        "checkpoint_sha256": "d" * 64,
        "checkpoint_history_record_sha256": "e" * 64,
        "parameter_sha256": "1" * 64,
        "optimizer_sha256": "2" * 64,
        "scheduler_sha256": "3" * 64,
        "rng_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "data_manifest_sha256": "6" * 64,
        "split_sha256": "7" * 64,
        "training_ledger_sha256": "8" * 64,
        "validation_ledger_sha256": "9" * 64,
        "checkpoint_history_sha256": "0" * 64,
        "dataset_order_sha256": "a" * 64,
        "validation_batch": {
            "manifest_key": "first",
            "loss_numerator": 2.0,
            "element_count": 2,
            "mean_loss": 1.0,
        },
        "depth_producer_sha256": None,
        "depth_cache_manifest_sha256": None,
        "depth_native_contract_sha256": None,
        "depth_validation_sha256": None,
        "depth_checkpoint_sha256": None,
    }
    value.update(updates)
    return value


def test_final_fresh_load_receipt_rejects_nonfinite_and_provenance_drift(tmp_path):
    path = tmp_path / "final_acceptance.json"
    digest = write_final_receipt(path, _receipt())
    loaded, loaded_path, loaded_digest = load_final_receipt(
        tmp_path, expected={"checkpoint_sha256": "d" * 64, "slurm_job_id": "222"}
    )
    assert loaded["fresh_model_process"] is True
    assert (loaded_path, loaded_digest) == (path, digest)
    with pytest.raises(P3CompletionError, match="differs"):
        load_final_receipt(tmp_path, expected={"checkpoint_sha256": "f" * 64})
    bad = _receipt()
    bad["validation_batch"] = dict(bad["validation_batch"], mean_loss=float("inf"))
    with pytest.raises(P3CompletionError, match="validation loss"):
        write_final_receipt(tmp_path / "bad_receipt.json", bad)


def test_final_receipt_rejects_equal_training_and_acceptance_process_ids(tmp_path):
    with pytest.raises(P3CompletionError, match="must differ"):
        write_final_receipt(
            tmp_path / "same_process.json",
            _receipt(process_id=456, training_process_id=456),
        )


def _paired_fixture():
    cards = []
    cells = {}
    heldout = {"sha256": "5" * 64, "split_sha256": "6" * 64}
    for environment in LOCKED_ENVS:
        for arm in LOCKED_ARMS:
            for seed in LOCKED_SEEDS:
                depth = None
                if arm != "dino_pinned":
                    depth = {
                        "producer_sha256": "1" * 64,
                        "cache_manifest_sha256": f"{LOCKED_ENVS.index(environment) + 2}"
                        * 64,
                        "native_contract_sha256": "7" * 64,
                        "validation_sha256": "8" * 64,
                        "checkpoint_sha256": "9" * 64,
                    }
                card = {
                    "schema": "dino-wm.run-card.v1",
                    "kind": "p3-training",
                    "run_id": f"p3-{environment}-{arm}-s{seed}",
                    "run_dir": f"/study/outputs/p3-training/{environment}-{arm}-s{seed}",
                    "code_root": "/study/code/dino_wm",
                    "source_commit": "f" * 40,
                    "source_file_sha256": {"train.py": "e" * 64},
                    "environment": environment,
                    "arm": arm,
                    "seed": seed,
                    "target_steps": 100,
                    "frameskip": 1,
                    "horizons": [1],
                    "batch_size": 32,
                    "predictor_lr": 0.00005,
                    "decoder": False,
                    "heldout_loss_manifest": heldout,
                    "initialization_policy": {
                        "predictor": "fresh_seeded",
                        "action_encoder": "fresh_seeded",
                        "proprio_encoder": "fresh_seeded",
                        "seed": seed,
                        "encoder": "frozen",
                    },
                    "optimizer_policy": {"locked": True},
                    "schedule_policy": "fixed_learning_rates",
                    "overrides": [f"env={environment}", f"encoder={arm}"],
                    "depth_inputs": depth,
                    "artifacts": {
                        "dinov2": {
                            "path": "/study/models/dinov2.pth",
                            "sha256": "4" * 64,
                        },
                        "dinocular_student": {
                            "path": "/study/checkpoints/student.pth",
                            "sha256": "9" * 64,
                        },
                    },
                    "container": {
                        "path": "/study/container.sif",
                        "sha256": "c" * 64,
                    },
                    "strict_resume": True,
                    "depends_on": [],
                    "environment_variables": {
                        "DINOV2_REPO": "/study/code/dinov2",
                        "DINOV2_VITS14_WEIGHTS": "/study/models/dinov2.pth",
                    },
                    "encoder_boundary": {
                        "dino_pinned": "not_applicable",
                        "dinocular": "informative_depth_and_mask",
                        "dinocular_zerodepth": "manifest_neutral_depth_and_mask",
                    }[arm],
                    "segment_sizing": {
                        "derived_segment_steps": 100 - LOCKED_ARMS.index(arm),
                        "measured_steps_per_second": 1.0 + LOCKED_ARMS.index(arm),
                        "timing_source_commit": "f" * 40,
                    },
                    "segment_steps": 100 - LOCKED_ARMS.index(arm),
                    "producer_decision": {
                        "path": "/study/results/producer_decision.json",
                        "sha256": "d" * 64,
                        "winner": "da3_giant_video",
                    },
                    "run_card_sha256": str(LOCKED_ARMS.index(arm) + 4) * 64,
                }
                if arm != "dino_pinned":
                    card["environment_variables"].update(
                        {
                            "DINOCULAR_STUDENT_WEIGHTS": "/study/checkpoints/student.pth",
                            "DINOCULAR_NATIVE_DEPTH_CONTRACT": "/study/contracts/native.json",
                            "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256": "7" * 64,
                            "DINOCULAR_CACHE_PRODUCER_SHA256": "1" * 64,
                        }
                    )
                card["config_sha256"] = sha256_bytes(
                    canonical_json_bytes(card["overrides"])
                )
                cards.append(card)
                depth_fields = {
                    "depth_producer_sha256": None,
                    "depth_cache_manifest_sha256": None,
                    "depth_native_contract_sha256": None,
                    "depth_validation_sha256": None,
                    "depth_checkpoint_sha256": None,
                }
                if depth is not None:
                    depth_fields = {
                        f"depth_{key}": value for key, value in depth.items()
                    }
                cells[(environment, arm, seed)] = {
                    "dataset_order_sha256": "a" * 64,
                    "plateau": {"plateaued": arm != "dinocular_zerodepth"},
                    **depth_fields,
                }
    return cards, cells


def test_paired_config_audit_and_inconclusive_propagation():
    cards, cells = _paired_fixture()
    comparisons = _paired_audit(cards, cells)
    assert len(comparisons) == 36
    assert any(
        row["optimization_status"] == "optimization-inconclusive" for row in comparisons
    )
    changed = [dict(card) for card in cards]
    changed[1]["optimizer_policy"] = {"locked": False}
    with pytest.raises(HarnessError, match="non-arm settings differ"):
        _paired_audit(changed, cells)


def test_paired_config_audit_rejects_unallowlisted_container_drift():
    cards, cells = _paired_fixture()
    changed = copy.deepcopy(cards)
    changed[1]["container"]["sha256"] = "0" * 64
    with pytest.raises(HarnessError, match=r"non-arm settings differ.*container"):
        _paired_audit(changed, cells)


def test_p4_execute_gate_requires_exact_final_receipt_and_tail(tmp_path):
    run_dir = tmp_path / "training"
    checkpoint = run_dir / "checkpoints" / "steps" / "step_000000100.pth"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"target": 100}, checkpoint)
    training_card = tmp_path / "training.yaml"
    training_card.write_text("immutable: true\n", encoding="utf-8")
    progress = {
        "status": "TARGET_REACHED",
        "global_step": 100,
        "target_steps": 100,
        "source_commit": "f" * 40,
        "immutable_run_card_sha256": "a" * 64,
        "training_process_id": 456,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameter_sha256": "1" * 64,
        "optimizer_sha256": "2" * 64,
        "scheduler_sha256": "3" * 64,
        "sampler": {"next_step": 100, "dataset_order_sha256": "a" * 64},
        "p3_completion": {
            "heldout_manifest_sha256": "5" * 64,
            "data_manifest_sha256": "6" * 64,
            "split_sha256": "7" * 64,
            "training_ledger_sha256": "8" * 64,
            "validation_ledger_sha256": "9" * 64,
            "checkpoint_history_sha256": "0" * 64,
        },
    }
    _write_json(run_dir / "progress.json", progress)
    receipt = _receipt(checkpoint_sha256=progress["checkpoint_sha256"])
    receipt_path = run_dir / "final_acceptance.json"
    receipt_sha = write_final_receipt(receipt_path, receipt)
    event = {
        "job_id": "222",
        "progress_status": "TARGET_REACHED",
        "global_step": 100,
        "immutable_run_card_sha256": "a" * 64,
        "training_process_id": 456,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": progress["checkpoint_sha256"],
        "final_acceptance_receipt": str(receipt_path),
        "final_acceptance_receipt_sha256": receipt_sha,
        "final_acceptance_process_id": 123,
    }
    chain = {
        "schema": "dino-wm.p3-slurm-chain.v1",
        "status": "PASSED",
        "run_dir": str(run_dir),
        "target_steps": 100,
        "run_card_sha256": "a" * 64,
        "run_card": str(training_card),
        "run_card_file_sha256": sha256_file(training_card),
        "source_commit": "f" * 40,
        "jobs": [{"job_id": "111"}, {"job_id": "222"}],
        "events": [event],
        "final_progress": progress,
        "training_process_id": 456,
        "final_acceptance_process_id": 123,
        "final_acceptance_receipt": str(receipt_path),
        "final_acceptance_receipt_sha256": receipt_sha,
    }
    _write_json(run_dir / "chain.json", chain)
    card = {
        "kind": "p4-open-loop",
        "run_id": "p4-pusht-dino_pinned-s1",
        "training_run_id": "p3-pusht-dino_pinned-s1",
        "training_run_dir": str(run_dir),
        "training_run_card": {
            "path": str(training_card),
            "file_sha256": sha256_file(training_card),
            "run_card_sha256": "a" * 64,
        },
        "training_completion_receipt": {
            "path": str(receipt_path),
            "schema": FINAL_RECEIPT_SCHEMA,
            "training_run_card_sha256": "a" * 64,
        },
        "depends_on": ["p3-pusht-dino_pinned-s1"],
        "target_steps": 100,
        "source_commit": "f" * 40,
        "config_sha256": "b" * 64,
        "container": {"sha256": "c" * 64},
    }
    verify_evaluation_training_dependencies([card], {"p3-pusht-dino_pinned-s1": "222"})

    event["training_process_id"] = 999
    _write_json(run_dir / "chain.json", chain)
    with pytest.raises(HarnessError, match="final event differs"):
        verify_evaluation_training_dependencies(
            [card], {"p3-pusht-dino_pinned-s1": "222"}
        )
    event["training_process_id"] = 456

    equal_pid_receipt = _receipt(
        checkpoint_sha256=progress["checkpoint_sha256"],
        process_id=456,
        training_process_id=456,
    )
    equal_pid_receipt_sha = _write_json(receipt_path, equal_pid_receipt)
    event["final_acceptance_process_id"] = 456
    event["final_acceptance_receipt_sha256"] = equal_pid_receipt_sha
    chain["final_acceptance_process_id"] = 456
    chain["final_acceptance_receipt_sha256"] = equal_pid_receipt_sha
    _write_json(run_dir / "chain.json", chain)
    with pytest.raises(HarnessError, match="must differ"):
        verify_evaluation_training_dependencies(
            [card], {"p3-pusht-dino_pinned-s1": "222"}
        )

    receipt_path.unlink()
    with pytest.raises(HarnessError, match="final_acceptance"):
        verify_evaluation_training_dependencies(
            [card], {"p3-pusht-dino_pinned-s1": "222"}
        )


def test_fresh_process_wrapper_and_heldout_materializer_are_zero_submit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "tools/p3_step_segment.sbatch").read_text(encoding="utf-8")
    assert "P3_FINAL_ACCEPTANCE_PROCESS=1" in wrapper
    assert '"training.final_acceptance=true"' in wrapper
    completed = subprocess.run(
        [
            "python3",
            str(root / "tools/materialize_p3_heldout.py"),
            "--data-root",
            str(tmp_path / "data"),
            "--data-manifest",
            str(tmp_path / "DATASET_MANIFEST.json"),
            "--out-dir",
            str(tmp_path / "heldout"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["state"] == "PASS"
    assert result["environments"] == list(LOCKED_ENVS)
    assert result["sbatch_calls"] == 0


@pytest.mark.skipif(
    importlib.util.find_spec("accelerate") is None,
    reason="the host environment lacks the pinned Trainer dependencies",
)
def test_fresh_trainer_process_loads_final_state_and_rejects_tampered_metadata(
    tmp_path,
):
    root = Path(__file__).resolve().parents[1]
    worker = root / "tests" / "p3_process_worker.py"
    run_dir = tmp_path / "synthetic-p3"
    run_dir.mkdir()
    run_card = tmp_path / "p3-run-card.json"
    run_card.write_text(
        json.dumps(
            {
                "kind": "p3-training",
                "run_card_sha256": "a" * 64,
                "source_commit": "f" * 40,
                "config_sha256": "b" * 64,
                "container": {"sha256": "c" * 64},
                "heldout_loss_manifest": {
                    "sha256": "5" * 64,
                    "data_manifest_sha256": "6" * 64,
                    "split_sha256": "7" * 64,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(root),
            "SLURM_JOB_ID": "424242",
            "STRICT_P2_IMMUTABLE_RUN_CARD": str(run_card),
            "STRICT_P2_IMMUTABLE_RUN_CARD_SHA256": "a" * 64,
            "STRICT_P2_CONTAINER_SHA256": "c" * 64,
        }
    )
    training = subprocess.run(
        [sys.executable, str(worker), "train", str(run_dir)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert training.returncode == 0, training.stdout + training.stderr
    assert "SYNTHETIC_P3_TRAIN=PASS" in training.stdout
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "TARGET_REACHED"
    assert progress["global_step"] == 100
    assert len((run_dir / "training_steps.jsonl").read_text().splitlines()) == 100
    assert len((run_dir / "heldout_loss.jsonl").read_text().splitlines()) == 100

    final_environment = dict(environment, P3_FINAL_ACCEPTANCE_PROCESS="1")
    final = subprocess.run(
        [sys.executable, str(worker), "final", str(run_dir)],
        env=final_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert final.returncode == 0, final.stdout + final.stderr
    assert "P3_FINAL_ACCEPTANCE=PASS" in final.stdout
    receipt_path = run_dir / "final_acceptance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["training_process_id"] == progress["training_process_id"]
    assert receipt["process_id"] != progress["training_process_id"]
    assert receipt["global_step"] == receipt["sampler"]["next_step"] == 100
    assert receipt["optimizer_sha256"] == progress["optimizer_sha256"]
    assert receipt["scheduler_sha256"] == progress["scheduler_sha256"]
    assert receipt["checkpoint_sha256"] == progress["checkpoint_sha256"]
    assert receipt["validation_batch"]["element_count"] > 0

    receipt_path.unlink()
    checkpoint_path = Path(progress["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["sampler"] = dict(checkpoint["sampler"])
    checkpoint["sampler"]["next_step"] = 99
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    index_path = checkpoint_path.parent / "step_latest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    target_entry = next(
        entry for entry in index["checkpoints"] if entry["file"] == checkpoint_path.name
    )
    target_entry["sha256"] = checkpoint_sha256
    _write_json(index_path, index)

    rejected = subprocess.run(
        [sys.executable, str(worker), "final", str(run_dir)],
        env=final_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "Sampler cursor differs" in rejected.stdout + rejected.stderr
    assert not receipt_path.exists()
