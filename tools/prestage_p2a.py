#!/usr/bin/env python3
"""Create and validate two no-submit, fail-closed P2a pilot card templates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

try:
    from tools.harness_common import (
        HarnessError,
        RUN_CARD_SCHEMA,
        canonical_json_bytes,
        finalize_run_card,
        load_yaml,
        sha256_bytes,
        validate_spec,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness_common import (  # type: ignore[no-redef]
        HarnessError,
        RUN_CARD_SCHEMA,
        canonical_json_bytes,
        finalize_run_card,
        load_yaml,
        sha256_bytes,
        validate_spec,
    )


ASSUMPTION = "[ASSUMPTION: RECOVERED-CONTRACT]"
PRODUCERS = ("da3_giant_video", "mapanything_recovered_framewise")
PENDING_PREFIX = "PENDING_"
ACCEPTED_RATE = 0.48803113065953124
ACCEPTED_TIMING_SHA256 = "771ac83fe23773f390ca3b653f06825675ebe2d37267e837fd7f53e7a2b19167"
ACCEPTED_TIMING_COMMIT = "33970180f64d45672aa26d40e2ccf95cf5a95550"
ACCEPTED_SEGMENT_STEPS = 11_000
FIXED_MANIFEST_PATH = (
    "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/"
    "manifests/openloop_pusht.jsonl"
)
PRODUCER_DECISION_PATH = (
    "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/"
    "results/p2a/producer_decision.json"
)


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _committed_source_hashes(root: Path, files: Sequence[str]) -> Mapping[str, str]:
    result = {}
    for relative in files:
        payload = subprocess.check_output(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"]
        )
        result[relative] = hashlib.sha256(payload).hexdigest()
    return result


def _placeholder(label: str) -> str:
    return f"{PENDING_PREFIX}{label}"


def _depth_slots(producer: str) -> Mapping[str, Any]:
    token = "DA3" if producer == PRODUCERS[0] else "MAPANYTHING"
    return {
        "producer": producer,
        "producer_sha256": _placeholder(f"{token}_PRODUCER_SHA256"),
        "cache_dir": _placeholder(f"{token}_PUSHT_CACHE_DIR"),
        "cache_sha256": _placeholder(f"{token}_PUSHT_CACHE_SHA256"),
        "cache_manifest_sha256": _placeholder(f"{token}_PUSHT_CACHE_MANIFEST_SHA256"),
        "validation_path": _placeholder(f"{token}_PUSHT_VALIDATION_RECEIPT_PATH"),
        "validation_sha256": _placeholder(f"{token}_PUSHT_VALIDATION_RECEIPT_SHA256"),
        "native_contract_path": _placeholder("RECOVERED_NATIVE_CONTRACT_PATH"),
        "native_contract_sha256": _placeholder("RECOVERED_NATIVE_CONTRACT_SHA256"),
        "checkpoint_sha256": "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc",
    }


def _card(
    *, spec: Mapping[str, Any], commit: str, source_hashes: Mapping[str, str], producer: str
) -> Mapping[str, Any]:
    run_id = f"p2a-pusht-dinocular-s1-{producer}"
    depth = _depth_slots(producer)
    assumptions = [ASSUMPTION] if producer == PRODUCERS[1] else []
    overrides = [
        "env=pusht",
        "encoder=dinocular",
        "training.seed=1",
        "training.predictor_lr=5e-5",
        "training.strict_determinism=true",
        "training.resume_from=auto",
        "training.batch_size=32",
        "env.num_workers=0",
        "frameskip=5",
        "num_hist=3",
        "num_pred=1",
        "has_decoder=false",
        "model.train_encoder=false",
        "model.train_predictor=true",
        "model.train_decoder=false",
        "plan_settings.plan_cfg_path=null",
        f"+env.dataset.depth_cache_dir={depth['cache_dir']}",
        f"+env.dataset.depth_cache_manifest_sha256={depth['cache_manifest_sha256']}",
        f"+env.dataset.depth_validation_path={depth['validation_path']}",
        f"+env.dataset.depth_validation_sha256={depth['validation_sha256']}",
        f"+env.dataset.native_depth_contract_path={depth['native_contract_path']}",
        f"+env.dataset.native_depth_contract_sha256={depth['native_contract_sha256']}",
        f"+env.dataset.depth_cache_producer_sha256={depth['producer_sha256']}",
        f"+env.dataset.depth_checkpoint_sha256={depth['checkpoint_sha256']}",
    ]
    card = {
        "schema": RUN_CARD_SCHEMA,
        "kind": "p2a-producer-pilot",
        "prestage_state": "BLOCKED_MISSING_EVIDENCE",
        "run_id": run_id,
        "environment": "pusht",
        "arm": "dinocular",
        "seed": 1,
        "run_dir": f"{spec['study_root']}/outputs/p2a-producer-pilot/{run_id}",
        "code_root": spec["code_root"],
        "source_commit": commit,
        "source_file_sha256": copy.deepcopy(dict(source_hashes)),
        "artifacts": copy.deepcopy(dict(spec["artifacts"])),
        "container": copy.deepcopy(dict(spec["container"])),
        "dataset_identity": {
            "archive_sha256": "442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08",
            "train_seq_lengths_sha256": "60b98f3e84b05717f088948b895a8c37debc93a0f46f16887bd749fbb87681a1",
            "validation_seq_lengths_sha256": "ed0218e14c639d254f318419e314799cc063499e6984cd4c080555a004fba40f",
            "trajectory_count": 18_706,
            "frame_count": 2_339_250,
            "selection": "all_released_train_and_validation_trajectories_unfiltered",
        },
        "target_steps": 123_858,
        "segment_steps": ACCEPTED_SEGMENT_STEPS,
        "segment_sizing": {
            "evidence": "DONE-26 accepted strict DINOv2 PushT sizing baseline",
            "timing_summary_path": (
                f"{spec['study_root']}/results/p2/strict_dinov2_vits14/timing_summary.json"
            ),
            "timing_summary_sha256": ACCEPTED_TIMING_SHA256,
            "timing_source_commit": ACCEPTED_TIMING_COMMIT,
            "rate_key": "dinocular/pusht",
            "rate_basis": "accepted_conservative_DINOv2_baseline_pending_arm_specific_refresh",
            "optimizer_steps_per_second": ACCEPTED_RATE,
            "max_productive_hours": 8.0,
            "safety_margin_fraction": 0.2,
            "quantum_steps": 1000,
            "derived_segment_steps": ACCEPTED_SEGMENT_STEPS,
        },
        "frameskip": 5,
        "horizons": [1, 5, 10, 25],
        "batch_size": 32,
        "predictor_lr": 0.00005,
        "decoder": False,
        "strict_resume": True,
        "strict_determinism": {
            "processes": 1,
            "num_workers": 0,
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "tf32": False,
            "cublas_workspace_config": ":4096:8",
            "checkpoint_every_steps": 1000,
            "resume_state": ["model", "optimizers", "schedulers", "sampler", "rng"],
        },
        "depth_inputs": depth,
        "assumption_tags": assumptions,
        "done_28_execution_gate": False,
        "fixed_evaluation_manifest": {
            "path": FIXED_MANIFEST_PATH,
            "sha256": _placeholder("FIXED_PUSHT_OPEN_LOOP_MANIFEST_SHA256"),
            "metadata_path": FIXED_MANIFEST_PATH.replace(".jsonl", ".meta.json"),
            "metadata_sha256": _placeholder("FIXED_PUSHT_OPEN_LOOP_METADATA_SHA256"),
            "seed": 20260714,
            "horizons": [1, 5, 10, 25],
        },
        "producer_pilot": {
            "producer": producer,
            "decision_horizons": [5, 10],
            "score": "0.5*(pooled_NRE_h5+pooled_NRE_h10)",
            "paired_manifest_required": True,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20260714,
            "bootstrap_unit": "held_out_trajectory",
            "decision_delta": "S_DA3-S_MapAnything",
            "selection_rule": "strictly_lower_locked_point_estimate",
            "tie_tolerance": 0.000001,
            "block_on": ["tie", "missing_key", "nonfinite", "provenance_failure", "gate_failure"],
        },
        "producer_decision_output": {
            "path": PRODUCER_DECISION_PATH,
            "sha256": _placeholder("P2A_PRODUCER_DECISION_SHA256"),
            "required_before_p2a_launch": False,
            "required_before_p3_p4_p5": True,
        },
        "depends_on": [
            f"p2-timing-{arm}-{environment}-s1"
            for arm in ("dino_pinned", "dinocular", "dinocular_zerodepth")
            for environment in ("pusht", "wall", "rope", "granular")
        ],
        "environment_variables": {
            "DINOV2_REPO": f"{spec['study_root']}/code/dinov2",
            "DINOV2_VITS14_WEIGHTS": spec["artifacts"]["dinov2"]["path"],
            "DINOCULAR_STUDENT_WEIGHTS": spec["artifacts"]["dinocular_student"]["path"],
            "DINOCULAR_NATIVE_DEPTH_CONTRACT": depth["native_contract_path"],
            "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256": depth["native_contract_sha256"],
            "DINOCULAR_CACHE_PRODUCER_SHA256": depth["producer_sha256"],
        },
        "overrides": overrides,
        "config_sha256": sha256_bytes(canonical_json_bytes(overrides)),
    }
    return finalize_run_card(card)


def _pending_values(value: Any, path: str = "") -> list[str]:
    result = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.extend(_pending_values(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_pending_values(item, f"{path}[{index}]"))
    elif isinstance(value, str) and PENDING_PREFIX in value:
        result.append(path)
    return result


def _substitute(value: Any, identities: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {key: _substitute(item, identities) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, identities) for item in value]
    if isinstance(value, str):
        result = value
        for pending, replacement in identities.items():
            result = result.replace(pending, replacement)
        return result
    return value


def validate_cards(cards: Sequence[Mapping[str, Any]], identities: Mapping[str, str] | None = None) -> Mapping[str, Any]:
    if len(cards) != 2 or tuple(card.get("producer_pilot", {}).get("producer") for card in cards) != PRODUCERS:
        raise HarnessError("prestage must contain exactly the two locked producers in order")
    materialized = [_substitute(card, identities or {}) for card in cards]
    for card in materialized:
        if (
            card.get("environment") != "pusht"
            or card.get("arm") != "dinocular"
            or card.get("seed") != 1
            or card.get("target_steps") != 123858
            or card.get("segment_steps") != ACCEPTED_SEGMENT_STEPS
            or card.get("done_28_execution_gate") is not False
        ):
            raise HarnessError("prestage card differs from the locked P2a execution contract")
    if materialized[0]["fixed_evaluation_manifest"] != materialized[1]["fixed_evaluation_manifest"]:
        raise HarnessError("P2a cards do not share the identical fixed PushT manifest")
    if materialized[1].get("assumption_tags") != [ASSUMPTION]:
        raise HarnessError("MapAnything prestage card lacks the recovered-contract tag")
    pending = sorted({item for card in materialized for item in _pending_values(card)})
    if pending:
        raise HarnessError("pending immutable evidence: " + ", ".join(pending))
    return {
        "schema": "dino-wm-p2a-prestage-validation-v1",
        "state": "STRUCTURALLY_LAUNCHABLE",
        "synthetic_identities": identities is not None,
        "card_count": 2,
        "run_card_sha256": [card["run_card_sha256"] for card in cards],
        "sbatch_calls": 0,
    }


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")


def create(args: argparse.Namespace) -> Mapping[str, Any]:
    spec = load_yaml(args.spec)
    validate_spec(spec)
    root = args.local_code_root.resolve()
    commit = _git_commit(root)
    hashes = _committed_source_hashes(root, spec["source_hash_files"])
    cards = [_card(spec=spec, commit=commit, source_hashes=hashes, producer=item) for item in PRODUCERS]
    references = []
    for card in cards:
        path = args.out / f"{card['run_id']}.yaml"
        _write_yaml(path, card)
        references.append({"run_id": card["run_id"], "path": str(path.resolve()), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "run_card_sha256": card["run_card_sha256"]})
    matrix = {"schema": "dino-wm-p2a-prestage-matrix-v1", "state": "BLOCKED_MISSING_EVIDENCE", "card_count": 2, "cards": references}
    matrix["matrix_sha256"] = sha256_bytes(canonical_json_bytes(matrix))
    _write_yaml(args.out / "p2a_pusht_producers.prestage.yaml", matrix)
    return matrix


def validate(args: argparse.Namespace) -> Mapping[str, Any]:
    matrix = load_yaml(args.matrix)
    cards = [load_yaml(Path(item["path"])) for item in matrix["cards"]]
    identities = json.loads(args.synthetic_identities.read_text()) if args.synthetic_identities else None
    if identities is not None and not isinstance(identities, Mapping):
        raise HarnessError("synthetic identities must be a JSON object")
    return validate_cards(cards, identities)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--spec", type=Path, default=Path(__file__).resolve().parents[1] / "conf" / "study_matrix.yaml")
    create_parser.add_argument("--local-code-root", type=Path, default=Path(__file__).resolve().parents[1])
    create_parser.add_argument("--out", type=Path, required=True)
    create_parser.set_defaults(function=create)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--matrix", type=Path, required=True)
    validate_parser.add_argument("--synthetic-identities", type=Path)
    validate_parser.set_defaults(function=validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.function(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as exc:
        print(f"P2A PRESTAGE CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
