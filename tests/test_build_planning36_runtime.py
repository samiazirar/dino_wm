from pathlib import Path

from tools.build_planning36 import _planner_override, _wrapper


def test_depth_wrapper_exports_fixed_student_weights_before_env_binding():
    card = {
        "lineage_id": "rope/dinocular/s1",
        "environment": "rope",
        "arm": "dinocular",
        "seed": 1,
        "code_root": "/runtime",
        "container": {"path": "/project/containers/runtime.sif", "sha256": "a" * 64},
        "result_contract": {"path": "/results/planning_results.jsonl"},
    }
    wrapper = _wrapper(Path("/cards/planning.json"), card, "b" * 64)
    export = (
        'export DINOCULAR_STUDENT_WEIGHTS='
        '"$PROJECT/checkpoints/dinov2_depthembed_dropout_fullpr.pth"'
    )
    assert export in wrapper
    assert wrapper.index(export) < wrapper.index(
        'source "$CODE_ROOT/tools/dinocular_container_env.sh"'
    )


def test_wall_uses_additive_planner_override_only_where_defaults_are_absent():
    assert _planner_override("wall") == "+planner=mpc_cem"
    assert _planner_override("rope") == "planner=mpc_cem"
    assert _planner_override("granular") == "planner=mpc_cem"
