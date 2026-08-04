#!/usr/bin/env python3
"""Measure exact zero-depth reuse at the frozen DINOcular encoder boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ENVIRONMENTS = ("wall", "rope", "granular")
SEEDS = (1, 2, 3)
EXPECTED_TARGETS = {"wall": 143910, "rope": 53500, "granular": 53500}
FUNCTIONALLY_REUSABLE = "functionally_reusable"
NOT_REUSABLE = "not_reusable"
NOT_YET = "not_yet_trained_or_established"


class MeasurementError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(tensor.numpy().tobytes())


def tensor_max_abs(left: Any, right: Any) -> float:
    if tuple(left.shape) != tuple(right.shape):
        return float("inf")
    if left.numel() == 0:
        return 0.0
    torch = __import__("torch")
    return float(
        (left.to(dtype=torch.float64) - right.to(dtype=torch.float64)).abs().max()
    )


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MeasurementError(f"JSON object required: {path}")
    return value


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MeasurementError(f"cannot load YAML {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MeasurementError(f"YAML object required: {path}")
    return value


def git_blob_sha256(repo: Path, commit: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise MeasurementError(
            f"source blob is unavailable for {commit}:{relative}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return sha256_bytes(result.stdout)


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise MeasurementError(
            f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def verify_implementation(
    card: Mapping[str, Any], implementation_root: Path
) -> Mapping[str, Any]:
    code_root = Path(str(card["code_root"]))
    commit = str(card["source_commit"])
    source_files = card.get("source_file_sha256")
    if not isinstance(source_files, Mapping) or not source_files:
        raise MeasurementError(f"source file map is absent for {card['run_id']}")
    if not code_root.is_dir():
        raise MeasurementError(f"card code root is absent: {code_root}")
    head = git_value(code_root, "rev-parse", "HEAD")
    dirty = bool(git_value(code_root, "status", "--porcelain"))
    committed_mismatches: dict[str, Any] = {}
    archive_mismatches: dict[str, Any] = {}
    for relative, expected_value in source_files.items():
        relative = str(relative)
        expected = str(expected_value)
        committed = git_blob_sha256(code_root, commit, relative)
        archive_path = implementation_root / relative
        archived = sha256_file(archive_path) if archive_path.is_file() else None
        if committed != expected:
            committed_mismatches[relative] = {"card": expected, "commit": committed}
        if archived != expected:
            archive_mismatches[relative] = {"card": expected, "archive": archived}
    if committed_mismatches or archive_mismatches:
        raise MeasurementError(
            f"immutable implementation differs for {card['run_id']}: "
            f"committed={committed_mismatches} archive={archive_mismatches}"
        )
    return {
        "code_root": str(code_root),
        "source_commit": commit,
        "source_commit_is_head": head == commit,
        "source_code_root_dirty": dirty,
        "source_file_sha256": dict(sorted((str(k), str(v)) for k, v in source_files.items())),
        "source_file_map_sha256": sha256_bytes(canonical_json(dict(sorted(source_files.items())))),
        "measurement_implementation_root": str(implementation_root),
    }


def verify_card_artifacts(card: Mapping[str, Any], checked: dict[str, str]) -> Mapping[str, Any]:
    records: dict[str, Mapping[str, Any]] = {}
    artifacts = card.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        records.update({str(k): v for k, v in artifacts.items() if isinstance(v, Mapping)})
    container = card.get("container")
    if isinstance(container, Mapping):
        records["container"] = container
    depth = card.get("depth_inputs", {})
    if isinstance(depth, Mapping):
        records["depth_cache_manifest"] = {
            "path": str(Path(str(depth["cache_dir"])) / "manifest.json"),
            "sha256": str(depth["cache_manifest_sha256"]),
        }
        records["depth_validation"] = {
            "path": str(depth["validation_path"]),
            "sha256": str(depth["validation_sha256"]),
        }
        records["native_depth_contract"] = {
            "path": str(depth["native_contract_path"]),
            "sha256": str(depth["native_contract_sha256"]),
        }
    verified: dict[str, Any] = {}
    for label, record in sorted(records.items()):
        path = Path(str(record.get("path")))
        expected = str(record.get("sha256"))
        if not path.is_file():
            raise MeasurementError(f"artifact is absent for {card['run_id']}: {label} {path}")
        if label == "container":
            verified[label] = {"path": str(path), "sha256": expected, "hash_checked_by_wrapper": True}
            continue
        actual = checked.get(str(path))
        if actual is None:
            actual = sha256_file(path)
            checked[str(path)] = actual
        if actual != expected:
            raise MeasurementError(
                f"artifact hash differs for {card['run_id']} {label}: {actual} != {expected}"
            )
        verified[label] = {"path": str(path), "sha256": actual}
    return verified


def run_state(card: Mapping[str, Any]) -> Mapping[str, Any]:
    run_dir = Path(str(card["run_dir"]))
    progress_path = run_dir / "progress.json"
    if not progress_path.is_file():
        return {
            "run_dir": str(run_dir),
            "trained_to_target": False,
            "reason": "progress_missing",
        }
    progress = load_json(progress_path)
    expected_target = int(card["target_steps"])
    final_path = run_dir / "final_acceptance.json"
    final = load_json(final_path) if final_path.is_file() else None
    target_reached = (
        progress.get("status") == "TARGET_REACHED"
        and int(progress.get("global_step", -1)) == expected_target
        and int(progress.get("target_steps", -1)) == expected_target
    )
    final_pass = (
        isinstance(final, Mapping)
        and final.get("state") == "PASS"
        and int(final.get("global_step", -1)) == expected_target
        and int(final.get("target_steps", -1)) == expected_target
        and final.get("source_commit") == card.get("source_commit")
        and final.get("immutable_run_card_sha256") == card.get("run_card_sha256")
    )
    checkpoint = Path(str(final.get("checkpoint"))) if isinstance(final, Mapping) and final.get("checkpoint") else None
    checkpoint_present = checkpoint is not None and checkpoint.is_file()
    trained = target_reached and final_pass and checkpoint_present
    reasons = []
    if not target_reached:
        reasons.append("target_not_reached")
    if not final_pass:
        reasons.append("final_acceptance_not_PASS_or_not_bound")
    if not checkpoint_present:
        reasons.append("final_checkpoint_missing")
    return {
        "run_dir": str(run_dir),
        "progress_path": str(progress_path),
        "progress_sha256": sha256_file(progress_path),
        "progress_status": progress.get("status"),
        "global_step": progress.get("global_step"),
        "target_steps": progress.get("target_steps"),
        "checkpoint_sha256": progress.get("checkpoint_sha256"),
        "immutable_run_card_sha256": progress.get("immutable_run_card_sha256"),
        "source_commit": progress.get("source_commit"),
        "final_acceptance_path": str(final_path) if final_path.is_file() else None,
        "final_acceptance_sha256": sha256_file(final_path) if final_path.is_file() else None,
        "final_acceptance_state": final.get("state") if isinstance(final, Mapping) else None,
        "final_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "final_checkpoint_present": checkpoint_present,
        "trained_to_target": trained,
        "reason": ";".join(reasons) if reasons else None,
    }


def load_payload_pair(card: Mapping[str, Any], dataset_dir: Path, torch: Any) -> Mapping[str, Any]:
    from datasets.img_transforms import default_transform

    environment = str(card["environment"])
    depth = card["depth_inputs"]
    common = {
        "transform": default_transform(img_size=224),
        "n_rollout": None,
        "normalize_action": True,
        "split_ratio": 0.9,
        "num_hist": 1,
        "num_pred": 1,
        "frameskip": int(card["frameskip"]),
        "depth_cache_dir": str(depth["cache_dir"]),
        "depth_cache_manifest_sha256": str(depth["cache_manifest_sha256"]),
        "depth_validation_path": str(depth["validation_path"]),
        "depth_validation_sha256": str(depth["validation_sha256"]),
        "native_depth_contract_path": str(depth["native_contract_path"]),
        "native_depth_contract_sha256": str(depth["native_contract_sha256"]),
        "depth_cache_producer_sha256": str(depth["producer_sha256"]),
        "depth_checkpoint_sha256": str(depth["checkpoint_sha256"]),
    }
    if environment == "wall":
        from datasets.wall_dset import load_wall_slice_train_val

        _, trajectory_sets = load_wall_slice_train_val(
            data_path=str(dataset_dir / "wall_single"),
            split_mode="random",
            **common,
        )
    else:
        from datasets.deformable_env_dset import load_deformable_dset_slice_train_val

        _, trajectory_sets = load_deformable_dset_slice_train_val(
            data_path=str(dataset_dir / "deformable"),
            object_name=environment,
            **common,
        )
    train = trajectory_sets["train"]
    for trajectory_position in range(min(len(train), 32)):
        observation, _, _, _ = train[trajectory_position]
        visual = observation["visual"]
        depths = observation["depth"]
        masks = observation["depth_validity_mask"]
        if depths.shape[0] < 2:
            continue
        for frame_b in range(1, int(depths.shape[0])):
            depth_a = depths[0:1]
            depth_b = depths[frame_b : frame_b + 1]
            if torch.equal(depth_a, depth_b):
                continue
            indices = getattr(train, "indices", None)
            episode = int(indices[trajectory_position]) if indices is not None else trajectory_position
            reader = getattr(train, "depth_reader", None)
            trajectory_key = None
            source_path = None
            if reader is not None:
                record = reader._resolve_record(None, episode)
                trajectory_key = str(record["trajectory_key"])
                source_path = str(record["source_path"])
            return {
                "trajectory_position": trajectory_position,
                "episode": episode,
                "trajectory_key": trajectory_key,
                "source_path": source_path,
                "frame_a": 0,
                "frame_b": frame_b,
                "payload_a": f"{trajectory_key}/{0:06d}" if trajectory_key else None,
                "payload_b": f"{trajectory_key}/{frame_b:06d}" if trajectory_key else None,
                "rgb": visual[0:1].detach(),
                "depth_a": depth_a.detach(),
                "depth_b": depth_b.detach(),
                "mask_a": masks[0:1].detach(),
                "mask_b": masks[frame_b : frame_b + 1].detach(),
            }
    raise MeasurementError(
        f"could not find two distinct valid depth payloads in the first training trajectories for {environment}"
    )


def measure_cell(
    card: Mapping[str, Any], implementation_root: Path, dataset_dir: Path, torch: Any
) -> Mapping[str, Any]:
    for key, value in card.get("environment_variables", {}).items():
        os.environ[str(key)] = str(value)
    os.environ["DATASET_DIR"] = str(dataset_dir)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    payload = load_payload_pair(card, dataset_dir, torch)
    config = OmegaConf.load(str(implementation_root / "conf/encoder/dinocular_zerodepth.yaml"))
    encoder = instantiate(config).to("cuda")
    encoder.eval()
    boundary: list[tuple[Any, Any]] = []

    def capture(_module: Any, depth: Any, mask: Any) -> None:
        boundary.append((depth.detach().cpu().clone(), mask.detach().cpu().clone()))

    remove = encoder.register_encoder_boundary_hook(capture)
    try:
        with torch.inference_mode():
            output_a = encoder(
                payload["rgb"].to("cuda"),
                payload["depth_a"].to("cuda"),
                payload["mask_a"].to("cuda"),
            ).detach().cpu()
            output_b = encoder(
                payload["rgb"].to("cuda"),
                payload["depth_b"].to("cuda"),
                payload["mask_b"].to("cuda"),
            ).detach().cpu()
    finally:
        remove()
    if len(boundary) != 2:
        raise MeasurementError(f"expected two encoder-boundary captures, got {len(boundary)}")
    boundary_depth_a, boundary_mask_a = boundary[0]
    boundary_depth_b, boundary_mask_b = boundary[1]
    raw_depth_distinct = not torch.equal(payload["depth_a"], payload["depth_b"])
    boundary_depth_exact = torch.equal(boundary_depth_a, boundary_depth_b)
    boundary_mask_exact = torch.equal(boundary_mask_a, boundary_mask_b)
    boundary_zero = bool(
        torch.count_nonzero(boundary_depth_a).item() == 0
        and torch.count_nonzero(boundary_depth_b).item() == 0
    )
    output_exact = torch.equal(output_a, output_b)
    passed = (
        raw_depth_distinct
        and boundary_depth_exact
        and boundary_mask_exact
        and boundary_zero
        and output_exact
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "device": "cuda",
        "rgb_sha256": sha256_tensor(payload["rgb"]),
        "raw_payloads": {
            "trajectory_key": payload["trajectory_key"],
            "source_path": payload["source_path"],
            "a": {
                "physical_key": payload["payload_a"],
                "depth_sha256": sha256_tensor(payload["depth_a"]),
                "validity_mask_sha256": sha256_tensor(payload["mask_a"]),
                "validity_all_ones": bool(torch.all(payload["mask_a"] == 1.0)),
                "depth_min": float(payload["depth_a"].min()),
                "depth_max": float(payload["depth_a"].max()),
            },
            "b": {
                "physical_key": payload["payload_b"],
                "depth_sha256": sha256_tensor(payload["depth_b"]),
                "validity_mask_sha256": sha256_tensor(payload["mask_b"]),
                "validity_all_ones": bool(torch.all(payload["mask_b"] == 1.0)),
                "depth_min": float(payload["depth_b"].min()),
                "depth_max": float(payload["depth_b"].max()),
            },
            "distinct": raw_depth_distinct,
        },
        "encoder_boundary": {
            "depth_a_sha256": sha256_tensor(boundary_depth_a),
            "depth_b_sha256": sha256_tensor(boundary_depth_b),
            "mask_a_sha256": sha256_tensor(boundary_mask_a),
            "mask_b_sha256": sha256_tensor(boundary_mask_b),
            "depth_exact": boundary_depth_exact,
            "mask_exact": boundary_mask_exact,
            "depth_is_exact_zero": boundary_zero,
            "depth_max_abs": max(float(boundary_depth_a.abs().max()), float(boundary_depth_b.abs().max())),
            "mask_unique_a": sorted(float(value) for value in torch.unique(boundary_mask_a)),
            "mask_unique_b": sorted(float(value) for value in torch.unique(boundary_mask_b)),
        },
        "output": {
            "shape": list(output_a.shape),
            "dtype": str(output_a.dtype),
            "a_sha256": sha256_tensor(output_a),
            "b_sha256": sha256_tensor(output_b),
            "exact": output_exact,
            "max_abs_difference": tensor_max_abs(output_a, output_b),
        },
        "pass_conditions": {
            "distinct_valid_raw_payloads": raw_depth_distinct,
            "exact_zero_boundary_depth": boundary_zero and boundary_depth_exact,
            "exact_boundary_mask": boundary_mask_exact,
            "exact_encoder_output": output_exact,
        },
    }


def classify_cell(
    implementation: Mapping[str, Any] | None,
    state: Mapping[str, Any],
    measurement: Mapping[str, Any] | None,
    error: str | None,
) -> tuple[str, str | None]:
    if not state.get("trained_to_target"):
        return NOT_YET, state.get("reason")
    if implementation is None:
        return NOT_REUSABLE, error or "immutable_implementation_not_bound"
    if measurement is None:
        return NOT_REUSABLE, error or "measurement_missing"
    if measurement.get("status") != "PASS":
        return NOT_REUSABLE, "direct_encoder_boundary_equality_failed"
    return FUNCTIONALLY_REUSABLE, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", action="append", type=Path, required=True)
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cards = [load_yaml(path) for path in args.card]
    expected_keys = {(environment, seed) for environment in ENVIRONMENTS for seed in SEEDS}
    actual_keys = {(str(card.get("environment")), int(card.get("seed", -1))) for card in cards}
    if actual_keys != expected_keys or len(cards) != len(expected_keys):
        raise MeasurementError(f"exact nine cards required: expected={expected_keys}, actual={actual_keys}")
    if not args.implementation_root.is_dir():
        raise MeasurementError(f"implementation root is absent: {args.implementation_root}")
    sys.path.insert(0, str(args.implementation_root))
    import torch

    checked_files: dict[str, str] = {}
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    implementation_records: dict[str, Mapping[str, Any]] = {}
    for card in cards:
        environment = str(card["environment"])
        seed = int(card["seed"])
        if card.get("arm") != "dinocular_zerodepth" or card.get("kind") != "p3-training":
            raise MeasurementError(f"card is not the locked zero-depth training card: {card.get('run_id')}")
        if int(card.get("target_steps", -1)) != EXPECTED_TARGETS[environment]:
            raise MeasurementError(f"target steps differ for {card['run_id']}")
        card_path = next(path for path, value in zip(args.card, cards) if value is card)
        state = run_state(card)
        card_file_sha = sha256_file(card_path)
        implementation: Mapping[str, Any] | None = None
        binding_error: str | None = None
        try:
            implementation = verify_implementation(card, args.implementation_root)
            implementation_records[environment] = implementation
            artifacts = verify_card_artifacts(card, checked_files)
        except MeasurementError as exc:
            binding_error = str(exc)
            artifacts = None
        measurement: Mapping[str, Any] | None = None
        measurement_error: str | None = None
        if state.get("trained_to_target") and implementation is not None and artifacts is not None:
            try:
                measurement = measure_cell(card, args.implementation_root, args.dataset_dir, torch)
            except (MeasurementError, OSError, RuntimeError, ValueError) as exc:
                measurement_error = str(exc)
        classification, reason = classify_cell(
            implementation, state, measurement, measurement_error or binding_error
        )
        if reason is None and measurement_error is not None:
            reason = measurement_error
        entries[(environment, seed)] = {
            "cell": f"{environment}/seed{seed}",
            "environment": environment,
            "seed": seed,
            "classification": classification,
            "reason": reason,
            "run_card": {
                "path": str(card_path),
                "file_sha256": card_file_sha,
                "run_card_sha256": str(card["run_card_sha256"]),
                "run_id": str(card["run_id"]),
            },
            "source": implementation,
            "artifacts": artifacts,
            "run_state": state,
            "measurement": measurement,
        }
    matrix = {
        environment: {
            f"seed{seed}": entries[(environment, seed)]["classification"] for seed in SEEDS
        }
        for environment in ENVIRONMENTS
    }
    reusable = [
        entry for entry in entries.values() if entry["classification"] == FUNCTIONALLY_REUSABLE
    ]
    not_reusable = [entry for entry in entries.values() if entry["classification"] == NOT_REUSABLE]
    receipt = {
        "schema": "dino-wm.zero-depth-functional-equivalence.v1",
        "status": "PASS" if not not_reusable else "FAIL",
        "project_question": "Does a learned 3D-aware visual representation help a robot world model predict and plan?",
        "measurement_question": "Are completed Wall/Rope/Granular zero-depth lineages exactly source independent at the encoder boundary?",
        "contract": {
            "source_payloads": "two distinct valid payloads from each card-pinned cache, same RGB held fixed",
            "boundary": "manifest_neutral_depth_and_mask",
            "reuse_requires": [
                "immutable card and committed implementation binding",
                "target-reached PASS final acceptance",
                "distinct valid raw payloads",
                "exact zero boundary depth and exact boundary mask",
                "bitwise-identical frozen-encoder outputs",
            ],
            "proxy_lineages_relabelled": False,
            "jobs_weights_settings_thresholds_data_modified": False,
        },
        "matrix": matrix,
        "cells": [entries[(environment, seed)] for environment in ENVIRONMENTS for seed in SEEDS],
        "implementation_bindings": implementation_records,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_bytes(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": receipt["status"], "matrix": matrix, "output": str(args.output)}, sort_keys=True))
    return 0 if not not_reusable else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MeasurementError as exc:
        print(f"MEASUREMENT CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
