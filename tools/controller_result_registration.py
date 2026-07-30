from __future__ import annotations

import re
from typing import Any


EVALUATION_SCHEMA = "dino-wm-open-loop-episode-errors-v1"
EVALUATION_MARKER_SCHEMA = "dino-wm-evaluation-run-marker-v1"
PLANNING_SCHEMA = "dinocular.planning-target-result.v1"
EVALUATION_EXPECTED_RECORDS = {
    "pusht": 100,
    "rope": 100,
    "granular": 100,
    "wall": 192,
}


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
