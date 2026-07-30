from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping


EVALUATION_SCHEMA = "dino-wm-open-loop-episode-errors-v1"
EVALUATION_MARKER_SCHEMA = "dino-wm-evaluation-run-marker-v1"
PLANNING_SCHEMA = "dinocular.planning-target-result.v1"
EVALUATION_EXPECTED_RECORDS = {
    "pusht": 100,
    "rope": 100,
    "granular": 100,
    "wall": 192,
}

ADMISSION_POLICY_SCHEMA = "dinocular.campaign-admission-policy.v1"
DEPTH_VALIDATION_SCHEMA = "dinocular.depth-admission-validation.v1"
ZERO_DEPTH_REUSE_SCHEMA = "dinocular.zero-depth-reuse-decision.v1"
ADMISSION_QUARANTINE_REASON = (
    "rope_granular_depth_validation_and_zero_depth_reuse_pending"
)
QUARANTINED_TARGETS = frozenset(
    {
        "rope/dinocular",
        "rope/dinocular_zerodepth",
        "granular/dinocular",
        "granular/dinocular_zerodepth",
    }
)
DEPTH_CHECKS = frozenset(
    {
        "repeated_frame_hash_rate",
        "finite_invalid_fraction",
        "spatial_variance",
        "temporal_variance",
        "moving_object_depth_correlation",
        "rgb_depth_frame_alignment",
        "expected_scale_range",
        "real_zero_feature_response",
    }
)
SURPRISING_RESULT_TRIGGERS = frozenset(
    {
        "metric_sign",
        "arm_ranking",
        "real_zero_near_tie",
        "cross_seed_reversal",
    }
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_depth_admission_receipt(
    receipt: Any, *, environment: str
) -> str | None:
    """Require a complete, immutable end-to-end depth validation receipt."""
    if not isinstance(receipt, Mapping):
        return "depth_validation_receipt_missing"
    if (
        receipt.get("schema") != DEPTH_VALIDATION_SCHEMA
        or receipt.get("state") != "PASS"
        or receipt.get("environment") != environment
    ):
        return "depth_validation_receipt_identity"
    try:
        expected_hash = _canonical_sha256(receipt)
    except (TypeError, ValueError):
        return "depth_validation_receipt_noncanonical"
    if receipt.get("receipt_sha256") != expected_hash:
        return "depth_validation_receipt_hash"
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != DEPTH_CHECKS:
        return "depth_validation_receipt_coverage"
    if any(
        not isinstance(checks[name], Mapping)
        or checks[name].get("state") != "PASS"
        for name in DEPTH_CHECKS
    ):
        return "depth_validation_receipt_failed_check"

    repeated = checks["repeated_frame_hash_rate"].get("value")
    finite = checks["finite_invalid_fraction"]
    spatial = checks["spatial_variance"].get("minimum")
    temporal = checks["temporal_variance"].get("minimum")
    motion = checks["moving_object_depth_correlation"]
    alignment = checks["rgb_depth_frame_alignment"].get("exact")
    scale = checks["expected_scale_range"]
    response = checks["real_zero_feature_response"].get("minimum_rms_delta")
    numeric_values = (
        repeated,
        finite.get("nonfinite_fraction"),
        finite.get("invalid_fraction"),
        spatial,
        temporal,
        motion.get("minimum_depth_delta"),
        motion.get("minimum_absolute_correlation"),
        scale.get("observed_minimum"),
        scale.get("observed_maximum"),
        scale.get("expected_minimum"),
        scale.get("expected_maximum"),
        response,
    )
    if not all(_finite_number(value) for value in numeric_values):
        return "depth_validation_receipt_measurements"
    if (
        not 0.0 <= float(repeated) < 1.0
        or float(finite["nonfinite_fraction"]) != 0.0
        or not 0.0 <= float(finite["invalid_fraction"]) < 1.0
        or float(spatial) <= 0.0
        or float(temporal) <= 0.0
        or float(motion["minimum_depth_delta"]) <= 0.0
        or float(motion["minimum_absolute_correlation"]) <= 0.0
        or alignment is not True
        or float(scale["observed_minimum"]) < float(scale["expected_minimum"])
        or float(scale["observed_maximum"]) > float(scale["expected_maximum"])
        or float(scale["observed_maximum"]) <= float(scale["observed_minimum"])
        or float(response) <= 0.0
    ):
        return "depth_validation_receipt_threshold"
    return None


def validate_zero_depth_reuse_receipt(
    receipt: Any, *, environment: str
) -> str | None:
    if not isinstance(receipt, Mapping):
        return "zero_depth_reuse_receipt_missing"
    if (
        receipt.get("schema") != ZERO_DEPTH_REUSE_SCHEMA
        or receipt.get("state") != "ACCEPTED"
        or receipt.get("environment") != environment
        or receipt.get("arm") != "dinocular_zerodepth"
        or not isinstance(receipt.get("decision_sha256"), str)
        or len(receipt["decision_sha256"]) != 64
    ):
        return "zero_depth_reuse_receipt_invalid"
    return None


def admission_decision(
    environment: str,
    arm: str,
    *,
    depth_receipt: Any = None,
    zero_depth_reuse_receipt: Any = None,
) -> dict[str, Any]:
    chain = f"{environment}/{arm}"
    if chain not in QUARANTINED_TARGETS:
        return {"status": "ELIGIBLE", "reason": None, "detail": None}
    detail = (
        validate_depth_admission_receipt(depth_receipt, environment=environment)
        if arm == "dinocular"
        else validate_zero_depth_reuse_receipt(
            zero_depth_reuse_receipt, environment=environment
        )
    )
    return {
        "status": "ELIGIBLE" if detail is None else "QUARANTINED",
        "reason": None if detail is None else ADMISSION_QUARANTINE_REASON,
        "detail": detail,
    }


def apply_admission_policy(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply admission state without changing lineage identities or receipts."""
    policy = state.setdefault(
        "admission_policy",
        {
            "schema": ADMISSION_POLICY_SCHEMA,
            "reason": ADMISSION_QUARANTINE_REASON,
            "targets": sorted(QUARANTINED_TARGETS),
            "depth_validation_receipts": {},
            "zero_depth_reuse_receipts": {},
        },
    )
    policy.setdefault("depth_validation_receipts", {})
    policy.setdefault("zero_depth_reuse_receipts", {})
    pipeline = state.setdefault("result_pipeline", {})
    excluded = set(pipeline.setdefault("excluded_lineages", []))
    pending_reuse = set(pipeline.setdefault("pending_reuse_lineages", []))
    events: list[dict[str, Any]] = []

    for chain_key, chain in state["chains"].items():
        environment = str(chain["environment"])
        arm = str(chain["arm"])
        decision = admission_decision(
            environment,
            arm,
            depth_receipt=policy["depth_validation_receipts"].get(environment),
            zero_depth_reuse_receipt=policy["zero_depth_reuse_receipts"].get(
                environment
            ),
        )
        previous = chain.get("admission")
        chain["admission"] = decision
        if previous != decision and chain_key in QUARANTINED_TARGETS:
            events.append(
                {
                    "type": "admission_quarantine_changed",
                    "chain": chain_key,
                    **decision,
                }
            )
        for seed_text, lineage in chain["lineages"].items():
            if chain_key not in QUARANTINED_TARGETS or not lineage.get(
                "evaluation_complete"
            ):
                continue
            lineage_id = f"{chain_key}/s{seed_text}"
            if arm == "dinocular":
                lineage.setdefault(
                    "result_admission",
                    {
                        "status": "EXCLUDED_QUARANTINED",
                        "reason": ADMISSION_QUARANTINE_REASON,
                    },
                )
                excluded.add(lineage_id)
            else:
                lineage.setdefault(
                    "result_admission",
                    {
                        "status": "HISTORICAL_PENDING_REUSE_PROOF",
                        "reason": ADMISSION_QUARANTINE_REASON,
                    },
                )
                pending_reuse.add(lineage_id)
    pipeline["excluded_lineages"] = sorted(excluded)
    pipeline["pending_reuse_lineages"] = sorted(pending_reuse)
    return events


def result_registration_outcome(
    validation_error: str | None,
    *,
    stage: str,
    environment: str,
    arm: str,
    seed: int,
    surprising_behavior: str | None = None,
) -> dict[str, Any]:
    """Keep correctness review separate from schema/provenance acceptance."""
    if validation_error is not None:
        return {"accepted": False, "error": validation_error, "events": []}
    events = []
    if surprising_behavior in SURPRISING_RESULT_TRIGGERS:
        events.append(
            {
                "type": "result_correctness_check_requested",
                "stage": stage,
                "chain": f"{environment}/{arm}",
                "seed": seed,
                "trigger": surprising_behavior,
            }
        )
    return {"accepted": True, "error": None, "events": events}


def validate_evaluation_summary(
    summary: Any,
    *,
    environment: str,
    arm: str,
    seed: int,
    job_id: str,
    expected_records: int | None = None,
) -> str | None:
    if not isinstance(summary, dict):
        return "evaluation_summary_absent"
    if expected_records is None:
        expected_records = EVALUATION_EXPECTED_RECORDS.get(environment)
    if not isinstance(expected_records, int) or expected_records <= 0:
        return "evaluation_expected_coverage"
    if summary.get("records") != expected_records:
        return "evaluation_record_count"
    if summary.get("unique_episodes") != expected_records:
        return "evaluation_episode_coverage"
    if summary.get("schemas") != [EVALUATION_SCHEMA]:
        return "evaluation_schema"
    if summary.get("environments") != [environment]:
        return "evaluation_environment"
    if summary.get("arms") != [arm]:
        return "evaluation_arm"
    if summary.get("seeds") != [seed]:
        return "evaluation_seed"
    if summary.get("slurm_job_ids") != [str(job_id)]:
        return "evaluation_terminal_job"
    if summary.get("marker_schema") != EVALUATION_MARKER_SCHEMA:
        return "evaluation_marker_schema"
    manifest_hashes = summary.get("manifest_hashes")
    if (
        not isinstance(manifest_hashes, list)
        or len(manifest_hashes) != 1
        or manifest_hashes != [summary.get("marker_manifest_sha256")]
        or summary.get("marker_manifest_file_sha256")
        != summary.get("marker_manifest_sha256")
    ):
        return "evaluation_manifest_binding"
    if not summary.get("finite"):
        return "evaluation_nonfinite"
    return None


def validate_planning_summary(
    summary: Any,
    *,
    environment: str,
    arm: str,
    seed: int,
    expected_records: int = 10,
) -> str | None:
    if not isinstance(summary, dict):
        return "planning_summary_absent"
    if summary.get("records") != expected_records:
        return "planning_record_count"
    if summary.get("unique_targets") != expected_records:
        return "planning_target_coverage"
    if summary.get("schemas") != [PLANNING_SCHEMA]:
        return "planning_schema"
    if summary.get("environments") != [environment]:
        return "planning_environment"
    if summary.get("arms") != [arm]:
        return "planning_arm"
    if summary.get("seeds") != [seed]:
        return "planning_seed"
    if summary.get("manifest_hashes") != [
        summary.get("expected_manifest_sha256")
    ]:
        return "planning_manifest_binding"
    target_ids = summary.get("target_ids")
    if not isinstance(target_ids, list) or len(target_ids) != expected_records:
        return "planning_target_ids"
    pattern = re.compile(
        rf"^{re.escape(environment)}-(\d{{3}})-[0-9a-f]{{16}}$"
    )
    indices: list[int] = []
    for target_id in target_ids:
        match = pattern.fullmatch(target_id) if isinstance(target_id, str) else None
        if match is None:
            return "planning_target_id_schema"
        indices.append(int(match.group(1)))
    if indices != list(range(expected_records)):
        return "planning_target_order"
    if not summary.get("finite"):
        return "planning_nonfinite"
    return None
