from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import build_open_loop_result_artifacts as builder
from tools.controller_result_registration import (
    ADMISSION_POLICY_SCHEMA,
    ADMISSION_QUARANTINE_REASON,
    DEPTH_VALIDATION_SCHEMA,
    QUARANTINED_TARGETS,
    ZERO_DEPTH_REUSE_SCHEMA,
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


def _policy() -> dict:
    return {
        "schema": ADMISSION_POLICY_SCHEMA,
        "reason": ADMISSION_QUARANTINE_REASON,
        "targets": sorted(QUARANTINED_TARGETS),
        "depth_validation_receipts": {
            environment: _depth_receipt(environment)
            for environment in ("rope", "granular")
        },
        "zero_depth_reuse_receipts": {
            environment: {
                "schema": ZERO_DEPTH_REUSE_SCHEMA,
                "state": "ACCEPTED",
                "environment": environment,
                "arm": "dinocular_zerodepth",
                "decision_sha256": environment[0] * 64,
            }
            for environment in ("rope", "granular")
        },
    }


def _inputs(tmp_path: Path) -> dict[str, Path]:
    inputs = {}
    for chain in sorted(QUARANTINED_TARGETS):
        for seed in builder.SEEDS:
            lineage = f"{chain}/s{seed}"
            path = tmp_path / lineage.replace("/", "_")
            path.write_text(lineage, encoding="utf-8")
            inputs[lineage] = path
    return inputs


def test_admission_records_bind_exact_validated_receipts_and_lineages(tmp_path):
    policy = _policy()
    inputs = _inputs(tmp_path)

    records = builder._admission_records(policy, inputs)

    assert [record["chain"] for record in records] == sorted(QUARANTINED_TARGETS)
    for record in records:
        environment, arm = record["chain"].split("/")
        receipt = (
            policy["depth_validation_receipts"][environment]
            if arm == "dinocular"
            else policy["zero_depth_reuse_receipts"][environment]
        )
        assert record["required_admission_kind"] == (
            "depth_validation" if arm == "dinocular" else "zero_depth_reuse"
        )
        assert record["decision"] == {
            "status": "ELIGIBLE",
            "reason": None,
            "detail": None,
        }
        assert record["receipt"] == {
            "embedded_record": receipt,
            "sha256": hashlib.sha256(
                builder._canonical_json_bytes(receipt)
            ).hexdigest(),
        }
        assert record["result_lineages"] == [
            {
                "lineage": f"{record['chain']}/s{seed}",
                "episode_errors_sha256": builder._sha256(
                    inputs[f"{record['chain']}/s{seed}"]
                ),
            }
            for seed in builder.SEEDS
        ]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda policy: policy["depth_validation_receipts"].pop("rope"),
            "rope/dinocular: depth_validation_receipt_missing",
        ),
        (
            lambda policy: policy["depth_validation_receipts"]["granular"].update(
                state="FAIL"
            ),
            "granular/dinocular: depth_validation_receipt_identity",
        ),
        (
            lambda policy: policy["depth_validation_receipts"]["rope"].update(
                environment="granular"
            ),
            "rope/dinocular: depth_validation_receipt_identity",
        ),
        (
            lambda policy: policy["depth_validation_receipts"]["rope"].update(
                receipt_sha256="0" * 64
            ),
            "rope/dinocular: depth_validation_receipt_hash",
        ),
        (
            lambda policy: policy["zero_depth_reuse_receipts"]["granular"].update(
                decision_sha256="stale"
            ),
            "granular/dinocular_zerodepth: zero_depth_reuse_receipt_invalid",
        ),
    ],
)
def test_admission_records_fail_closed_for_missing_failed_or_mismatched_receipts(
    tmp_path, mutate, match
):
    policy = copy.deepcopy(_policy())
    mutate(policy)

    with pytest.raises(builder.BuildError, match=match):
        builder._admission_records(policy, _inputs(tmp_path))


def test_result_builder_requires_admission_policy_argument():
    with pytest.raises(SystemExit):
        builder.build_parser().parse_args(
            [
                "--launch-manifest",
                "/tmp/launch.json",
                "--completion-summary",
                "/tmp/completion.json",
                "--out-dir",
                "/tmp/bundle",
            ]
        )
