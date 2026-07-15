from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import torch

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
    validate_training_records,
    validate_validation_records,
    write_final_receipt,
)
from tools.collect_p3_completion import _paired_audit
from tools.harness_common import HarnessError, LOCKED_ARMS, LOCKED_ENVS, LOCKED_SEEDS
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
        "CONFIGURED_INTERVAL",
        "COMPLETE_EPOCH",
        "SEGMENT_BOUNDARY",
        "USR1",
        "EXACT_TARGET",
        "INTEGER_PERCENT",
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
    assert [row["reasons"] for row in same_step] == [["INTEGER_PERCENT"], ["USR1"]]
    assert (
        json.loads(manager.history_path.read_text())["schema"]
        == CHECKPOINT_HISTORY_SCHEMA
    )
    assert len(list(tmp_path.glob("step_*.pth"))) == 2
    divergent = dict(payload)
    divergent["state"] = torch.tensor([-1])
    with pytest.raises(P3CompletionError, match="divergent duplicate"):
        manager.save(divergent, len(required), reasons=(required[-1],))


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


def test_validation_resume_coverage_plateau_boundary_and_inconclusive(tmp_path):
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
    boundary = plateau_verdict(rows)
    assert boundary["relative_absolute_change"] == pytest.approx(0.02)
    assert boundary["plateaued"] is True
    rows[-1] = dict(rows[-1])
    rows[-1]["mean_loss"] = 1.03
    rows[-1]["loss_numerator"] = 10.3
    above = plateau_verdict(rows)
    assert above["plateaued"] is False
    assert comparison_verdict(boundary, above) == "optimization-inconclusive"
    nonfinite = _validation_row(1, 1.0)
    nonfinite["mean_loss"] = float("nan")
    bad_path = tmp_path / "bad.jsonl"
    with pytest.raises(P3CompletionError, match="finite JSON"):
        append_validation_record(bad_path, nonfinite, target_steps=100)
    assert not bad_path.exists()


def _receipt(**updates):
    value = {
        "schema": FINAL_RECEIPT_SCHEMA,
        "state": "PASS",
        "fresh_model_process": True,
        "process_id": 123,
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
                    "artifacts": {"dinocular_student": {"sha256": "9" * 64}},
                    "encoder_boundary": {
                        "dino_pinned": "not_applicable",
                        "dinocular": "informative_depth_and_mask",
                        "dinocular_zerodepth": "manifest_neutral_depth_and_mask",
                    }[arm],
                }
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
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": progress["checkpoint_sha256"],
        "final_acceptance_receipt": str(receipt_path),
        "final_acceptance_receipt_sha256": receipt_sha,
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
