from tools.controller_result_registration import (
    EVALUATION_MARKER_SCHEMA,
    EVALUATION_SCHEMA,
    PLANNING_SCHEMA,
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
