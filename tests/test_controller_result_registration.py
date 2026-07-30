import hashlib
import json

from tools.controller_result_registration import (
    ADMISSION_QUARANTINE_REASON,
    DEPTH_VALIDATION_SCHEMA,
    EVALUATION_MARKER_SCHEMA,
    EVALUATION_SCHEMA,
    PLANNING_SCHEMA,
    ZERO_DEPTH_REUSE_SCHEMA,
    admission_decision,
    apply_admission_policy,
    result_registration_outcome,
    validate_evaluation_summary,
    validate_planning_summary,
)


def test_evaluation_requires_exact_schema_coverage_and_terminal_job():
    summary = {
        "records": 100,
        "unique_episodes": 100,
        "schemas": [EVALUATION_SCHEMA],
        "environments": ["rope"],
        "arms": ["dinocular"],
        "seeds": [1],
        "slurm_job_ids": ["42"],
        "marker_schema": EVALUATION_MARKER_SCHEMA,
        "manifest_hashes": ["a" * 64],
        "marker_manifest_sha256": "a" * 64,
        "marker_manifest_file_sha256": "a" * 64,
        "finite": True,
    }
    assert (
        validate_evaluation_summary(
            summary,
            environment="rope",
            arm="dinocular",
            seed=1,
            job_id="42",
        )
        is None
    )
    summary["slurm_job_ids"] = ["failed-job"]
    assert (
        validate_evaluation_summary(
            summary,
            environment="rope",
            arm="dinocular",
            seed=1,
            job_id="42",
        )
        == "evaluation_terminal_job"
    )


def test_planning_requires_all_ten_ordered_fixed_targets():
    summary = {
        "records": 10,
        "unique_targets": 10,
        "schemas": [PLANNING_SCHEMA],
        "environments": ["granular"],
        "arms": ["dino_pinned"],
        "seeds": [1],
        "manifest_hashes": ["b" * 64],
        "expected_manifest_sha256": "b" * 64,
        "target_ids": [
            f"granular-{index:03d}-{'c' * 16}" for index in range(10)
        ],
        "finite": True,
    }
    assert (
        validate_planning_summary(
            summary,
            environment="granular",
            arm="dino_pinned",
            seed=1,
        )
        is None
    )
    summary["target_ids"][-1] = f"granular-008-{'d' * 16}"
    assert (
        validate_planning_summary(
            summary,
            environment="granular",
            arm="dino_pinned",
            seed=1,
        )
        == "planning_target_order"
    )


def _depth_receipt(environment: str) -> dict:
    receipt = {
        "schema": DEPTH_VALIDATION_SCHEMA,
        "state": "PASS",
        "environment": environment,
        "checks": {
            "repeated_frame_hash_rate": {"state": "PASS", "value": 0.0},
            "finite_invalid_fraction": {
                "state": "PASS",
                "nonfinite_fraction": 0.0,
                "invalid_fraction": 0.0,
            },
            "spatial_variance": {"state": "PASS", "minimum": 0.1},
            "temporal_variance": {"state": "PASS", "minimum": 0.1},
            "moving_object_depth_correlation": {
                "state": "PASS",
                "minimum_depth_delta": 0.1,
                "minimum_absolute_correlation": 0.1,
            },
            "rgb_depth_frame_alignment": {"state": "PASS", "exact": True},
            "expected_scale_range": {
                "state": "PASS",
                "observed_minimum": 0.1,
                "observed_maximum": 0.9,
                "expected_minimum": 0.0,
                "expected_maximum": 1.0,
            },
            "real_zero_feature_response": {
                "state": "PASS",
                "minimum_rms_delta": 0.1,
            },
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return receipt


def test_admission_is_exact_fail_closed_and_receipt_gated():
    for environment in ("rope", "granular"):
        blocked = admission_decision(environment, "dinocular")
        assert blocked == {
            "status": "QUARANTINED",
            "reason": ADMISSION_QUARANTINE_REASON,
            "detail": "depth_validation_receipt_missing",
        }
        incomplete = _depth_receipt(environment)
        del incomplete["checks"]["moving_object_depth_correlation"]
        assert (
            admission_decision(
                environment, "dinocular", depth_receipt=incomplete
            )["status"]
            == "QUARANTINED"
        )
        assert (
            admission_decision(
                environment,
                "dinocular",
                depth_receipt=_depth_receipt(environment),
            )["status"]
            == "ELIGIBLE"
        )
        zero_receipt = {
            "schema": ZERO_DEPTH_REUSE_SCHEMA,
            "state": "ACCEPTED",
            "environment": environment,
            "arm": "dinocular_zerodepth",
            "decision_sha256": "d" * 64,
        }
        assert (
            admission_decision(
                environment,
                "dinocular_zerodepth",
                zero_depth_reuse_receipt=zero_receipt,
            )["status"]
            == "ELIGIBLE"
        )

    for environment, arm in (
        ("rope", "dino_pinned"),
        ("granular", "dino_pinned"),
        ("wall", "dinocular"),
        ("pusht", "dinocular_zerodepth"),
    ):
        assert admission_decision(environment, arm)["status"] == "ELIGIBLE"


def test_policy_marks_historical_results_without_rewriting_identity():
    state = {
        "chains": {
            "rope/dinocular": {
                "environment": "rope",
                "arm": "dinocular",
                "lineages": {
                    "1": {
                        "seed": 1,
                        "evaluation_complete": True,
                        "evaluation_result": {"arms": ["dinocular"], "sha256": "a" * 64},
                        "training_chain_receipt": {"chain": "rope/dinocular"},
                    }
                },
            },
            "wall/dinocular": {
                "environment": "wall",
                "arm": "dinocular",
                "lineages": {"1": {"seed": 1, "evaluation_complete": True}},
            },
        },
        "result_pipeline": {},
    }
    original_result = dict(
        state["chains"]["rope/dinocular"]["lineages"]["1"]["evaluation_result"]
    )
    original_receipt = dict(
        state["chains"]["rope/dinocular"]["lineages"]["1"][
            "training_chain_receipt"
        ]
    )
    apply_admission_policy(state)
    held = state["chains"]["rope/dinocular"]
    assert held["admission"]["status"] == "QUARANTINED"
    assert (
        held["lineages"]["1"]["result_admission"]["status"]
        == "EXCLUDED_QUARANTINED"
    )
    assert state["result_pipeline"]["excluded_lineages"] == [
        "rope/dinocular/s1"
    ]
    assert held["lineages"]["1"]["evaluation_result"] == original_result
    assert held["lineages"]["1"]["training_chain_receipt"] == original_receipt
    assert state["chains"]["wall/dinocular"]["admission"]["status"] == "ELIGIBLE"


def test_surprising_valid_result_is_accepted_and_flagged_but_malformed_rejects():
    accepted = result_registration_outcome(
        None,
        stage="prediction",
        environment="rope",
        arm="dinocular",
        seed=1,
        surprising_behavior="arm_ranking",
    )
    assert accepted["accepted"] is True
    assert accepted["events"] == [
        {
            "type": "result_correctness_check_requested",
            "stage": "prediction",
            "chain": "rope/dinocular",
            "seed": 1,
            "trigger": "arm_ranking",
        }
    ]
    malformed = result_registration_outcome(
        "evaluation_schema",
        stage="prediction",
        environment="rope",
        arm="dinocular",
        seed=1,
        surprising_behavior="arm_ranking",
    )
    assert malformed == {
        "accepted": False,
        "error": "evaluation_schema",
        "events": [],
    }
