#!/usr/bin/env python3
"""Create fixed open-loop manifests and evaluate accepted step checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from p3_completion import (
    P3CompletionError,
    canonical_first_heldout_manifest_key,
    is_process_id,
    load_final_receipt,
    validate_final_sampler,
)

from tools.harness_common import (
    HarnessError,
    build_evaluation_provenance,
    require_directory_no_alias,
    require_regular_file_no_alias,
    require_run_card_authorization,
    run_card_receipt_expectations,
    validate_run_card,
    validate_evaluation_provenance,
    verify_evaluation_bindings,
)


MANIFEST_SCHEMA = "dino-wm-open-loop-start-v1"
META_SCHEMA = "dino-wm-open-loop-manifest-v1"
RESULT_SCHEMA = "dino-wm-open-loop-episode-errors-v1"
HORIZONS = {
    "pusht": [1, 5, 10, 25],
    "wall": [1, 5, 10],
    "rope": [1, 2, 3],
    "granular": [1, 2, 3],
}
FRAMESKIPS = {"pusht": 5, "wall": 5, "rope": 1, "granular": 1}
SELECTION_SEED = 20260714
NUM_HIST = {"pusht": 3, "wall": 1, "rope": 1, "granular": 1}


class EvaluationContractError(RuntimeError):
    """A fixed-manifest, checkpoint, configuration, or coverage gate failed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _valid_indices(count: int) -> list[int]:
    import torch

    order = torch.randperm(count, generator=torch.Generator().manual_seed(42)).tolist()
    return sorted(order[int(0.9 * count) :])


def _episode_index(root: Path, environment: str) -> list[Mapping[str, Any]]:
    root = root.resolve()
    result = []
    if environment == "pusht":
        base = root / "pusht_noise" / "val"
        with (base / "seq_lengths.pkl").open("rb") as handle:
            lengths = pickle.load(handle)
        for episode, count in enumerate(lengths):
            source = base / "obses" / f"episode_{episode:03d}.mp4"
            result.append(
                {
                    "episode": episode,
                    "frame_count": int(count),
                    "source_path": source.relative_to(root).as_posix(),
                    "source_sha256": sha256_file(source),
                }
            )
    elif environment == "wall":
        base = root / "wall_single"
        actions = _torch_load(base / "actions.pth")
        for episode in _valid_indices(int(actions.shape[0])):
            source = base / "obses" / f"episode_{episode:03d}.pth"
            result.append(
                {
                    "episode": episode,
                    "frame_count": int(actions.shape[1]),
                    "source_path": source.relative_to(root).as_posix(),
                    "source_sha256": sha256_file(source),
                }
            )
    elif environment in {"rope", "granular"}:
        base = root / "deformable" / environment
        actions = _torch_load(base / "actions.pth")
        for episode in _valid_indices(int(actions.shape[0])):
            source = base / f"{episode:06d}" / "obses.pth"
            result.append(
                {
                    "episode": episode,
                    "frame_count": int(actions.shape[1]),
                    "source_path": source.relative_to(root).as_posix(),
                    "source_sha256": sha256_file(source),
                }
            )
    else:
        raise EvaluationContractError(f"unsupported environment {environment!r}")
    if not result:
        raise EvaluationContractError(f"no held-out episodes found for {environment}")
    return result


def _write_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise EvaluationContractError(
                f"immutable manifest already exists with different bytes: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def make_manifest(args: argparse.Namespace) -> None:
    environment = args.env
    if args.seed != SELECTION_SEED:
        raise EvaluationContractError(f"manifest seed must be exactly {SELECTION_SEED}")
    if args.n != 1000:
        raise EvaluationContractError(
            "fixed open-loop manifest size must be exactly 1000"
        )
    expected_horizons = HORIZONS.get(environment)
    if args.horizons != expected_horizons:
        raise EvaluationContractError(
            f"horizons for {environment} must be exactly {expected_horizons}"
        )
    expected_frameskip = FRAMESKIPS[environment]
    frameskip = args.frameskip if args.frameskip is not None else expected_frameskip
    if frameskip != expected_frameskip:
        raise EvaluationContractError(
            f"frameskip for {environment} must be exactly {expected_frameskip}"
        )
    root = args.root or Path(os.environ.get("DATASET_DIR", ""))
    if not str(root):
        raise EvaluationContractError("--root or DATASET_DIR is required")
    index = _episode_index(root, environment)
    dataset_index_sha256 = sha256_bytes(canonical_json_bytes(index))
    split = [record["episode"] for record in index]
    split_sha256 = sha256_bytes(canonical_json_bytes(split))
    maximum_horizon = max(expected_horizons)
    num_hist = NUM_HIST[environment]
    candidates = []
    for episode in index:
        maximum_start = (
            int(episode["frame_count"])
            - 1
            - (num_hist - 1 + maximum_horizon) * frameskip
        )
        for start in range(maximum_start + 1):
            key = f"{environment}/valid/{episode['episode']:05d}/{start:06d}"
            action_blocks = {
                str(step): list(
                    range(
                        start + step * frameskip,
                        start + (step + 1) * frameskip,
                    )
                )
                for step in range(num_hist + maximum_horizon - 1)
            }
            record = {
                "schema": MANIFEST_SCHEMA,
                "key": key,
                "environment": environment,
                "split": "valid",
                "episode": int(episode["episode"]),
                "start": start,
                "num_hist": num_hist,
                "frameskip": frameskip,
                "horizons": expected_horizons,
                "history_frames": [
                    start + step * frameskip for step in range(num_hist)
                ],
                "target_frames": {
                    str(horizon): start + (num_hist - 1 + horizon) * frameskip
                    for horizon in expected_horizons
                },
                "raw_action_indices": action_blocks,
                "dataset_index_sha256": dataset_index_sha256,
                "split_sha256": split_sha256,
                "source_path": episode["source_path"],
                "source_sha256": episode["source_sha256"],
            }
            candidates.append(record)
    selected = sorted(
        candidates,
        key=lambda record: (sha256_bytes(record["key"].encode("utf-8")), record["key"]),
    )[: args.n]
    if not selected:
        raise EvaluationContractError("no eligible fixed open-loop starts")
    text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in selected
    )
    _write_immutable(args.out, text)
    manifest_sha256 = sha256_file(args.out)
    metadata = {
        "schema": META_SCHEMA,
        "environment": environment,
        "selection": "smallest_sha256_start_keys",
        "selection_seed": SELECTION_SEED,
        "requested_count": 1000,
        "eligible_count": len(candidates),
        "selected_count": len(selected),
        "frameskip": frameskip,
        "horizons": expected_horizons,
        "num_hist": num_hist,
        "dataset_index_sha256": dataset_index_sha256,
        "split_sha256": split_sha256,
        "manifest_sha256": manifest_sha256,
        "keys_sha256": sha256_bytes(
            canonical_json_bytes([record["key"] for record in selected])
        ),
    }
    meta_path = args.out.with_suffix(".meta.json")
    _write_immutable(
        meta_path,
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise EvaluationContractError(
                    f"blank JSONL line at {path}:{line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationContractError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, Mapping):
                raise EvaluationContractError(
                    f"non-object JSONL record at line {line_number}"
                )
            records.append(value)
    return records


def _load_eval_model(training_run_dir: Path, checkpoint_path: Path, device: str):
    import hydra
    from omegaconf import OmegaConf
    import torch

    cfg_path = training_run_dir / "hydra.yaml"
    if not cfg_path.is_file():
        raise EvaluationContractError(f"training Hydra config is absent: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if (
        int(cfg.training.batch_size) != 32
        or float(cfg.training.predictor_lr) != 0.00005
        or bool(cfg.has_decoder)
        or int(cfg.num_hist) <= 0
        or int(cfg.num_pred) != 1
    ):
        raise EvaluationContractError("accepted training config violates P3 protocol")
    datasets, traj_dsets = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    encoder = hydra.utils.instantiate(cfg.encoder)
    proprio = hydra.utils.instantiate(
        cfg.proprio_encoder,
        in_chans=datasets["train"].proprio_dim,
        emb_dim=cfg.proprio_emb_dim,
    )
    action = hydra.utils.instantiate(
        cfg.action_encoder,
        in_chans=datasets["train"].action_dim,
        emb_dim=cfg.action_emb_dim,
    )
    num_patches = int(encoder.num_patches) + (2 if int(cfg.concat_dim) == 0 else 0)
    predictor = hydra.utils.instantiate(
        cfg.predictor,
        num_patches=num_patches,
        num_frames=cfg.num_hist,
        dim=encoder.emb_dim
        + (
            int(cfg.proprio_emb_dim) * int(cfg.num_proprio_repeat)
            + int(cfg.action_emb_dim) * int(cfg.num_action_repeat)
        )
        * int(cfg.concat_dim),
    )
    model = hydra.utils.instantiate(
        cfg.model,
        encoder=encoder,
        proprio_encoder=proprio,
        action_encoder=action,
        predictor=predictor,
        decoder=None,
        proprio_dim=cfg.proprio_emb_dim,
        action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    components = {
        "encoder": encoder,
        "proprio_encoder": proprio,
        "action_encoder": action,
        "predictor": predictor,
    }
    states = checkpoint.get("models")
    if not isinstance(states, Mapping) or set(states) != set(components):
        raise EvaluationContractError("step checkpoint model components differ")
    for name, component in components.items():
        component.load_state_dict(states[name], strict=True)
    model.to(device)
    model.eval()
    return cfg, model, traj_dsets["valid"]


def _base_traj_dataset(dataset):
    return getattr(dataset, "dataset", dataset)


def _tensor_obs(obs: Mapping[str, Any], device: str) -> Mapping[str, Any]:
    return {key: value.unsqueeze(0).to(device) for key, value in obs.items()}


def _load_target_observations(dataset, environment: str, episode: int, frames):
    frames = [int(frame) for frame in frames]
    trajectory_length = int(dataset.get_seq_length(episode))
    if environment != "wall" or max(frames) < trajectory_length:
        observations, _act, _state, _info = dataset.get_frames(episode, frames)
        return observations
    if min(frames) < 0 or max(frames) > trajectory_length:
        raise EvaluationContractError("Wall target frame exceeds the released rollout")
    if dataset.depth_reader is not None:
        raise EvaluationContractError(
            "Wall terminal target observation has no accepted depth-cache frame"
        )
    images = _torch_load(
        dataset.data_path / "obses" / f"episode_{episode:03d}.pth"
    )
    if int(images.shape[0]) != trajectory_length + 1:
        raise EvaluationContractError(
            "Wall terminal target requires one post-action observation"
        )
    visual = images[frames] / 255
    if dataset.transform:
        visual = dataset.transform(visual)
    proprio_frames = [min(frame, trajectory_length - 1) for frame in frames]
    return {
        "visual": visual,
        "proprio": dataset.proprios[episode, proprio_frames],
    }


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _verify_training_completion(
    card: Mapping[str, Any],
    training_card: Mapping[str, Any],
    training_run_dir: Path,
) -> tuple[Mapping[str, Any], Path]:
    import torch

    try:
        training_run_dir = require_directory_no_alias(
            training_run_dir, "evaluation training directory"
        )
        progress_path = require_regular_file_no_alias(
            training_run_dir / "progress.json", "evaluation training progress"
        )
    except HarnessError as exc:
        raise EvaluationContractError(str(exc)) from exc
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("status") != "TARGET_REACHED" or int(
        progress.get("global_step", -1)
    ) != int(card["target_steps"]):
        raise EvaluationContractError("training run has not reached its exact target")
    try:
        checkpoint_path = require_regular_file_no_alias(
            Path(progress["checkpoint"]), "evaluation training checkpoint"
        )
    except HarnessError as exc:
        raise EvaluationContractError(str(exc)) from exc
    if sha256_file(checkpoint_path) != progress.get("checkpoint_sha256"):
        raise EvaluationContractError("final training checkpoint hash mismatch")
    if progress.get("source_commit") != card.get("source_commit"):
        raise EvaluationContractError("training/evaluation source commit mismatch")
    if not is_process_id(progress.get("training_process_id")):
        raise EvaluationContractError("training progress has no valid process ID")
    if progress.get("immutable_run_card_sha256") != training_card.get(
        "run_card_sha256"
    ):
        raise EvaluationContractError(
            "training progress belongs to a different run card"
        )
    checkpoint_metadata = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint_metadata.get("immutable_run_card_sha256") != training_card.get(
        "run_card_sha256"
    ):
        raise EvaluationContractError(
            "training checkpoint belongs to a different run card"
        )
    if card.get("kind") == "p4-open-loop":
        sampler = progress.get("sampler")
        if (
            not isinstance(sampler, Mapping)
            or checkpoint_metadata.get("sampler") != sampler
        ):
            raise EvaluationContractError(
                "training progress sampler differs from the final checkpoint"
            )
        try:
            validate_final_sampler(
                sampler,
                target_steps=int(card["target_steps"]),
                dataset_order_sha256=str(sampler.get("dataset_order_sha256")),
                expected_batch_size=int(training_card["batch_size"]),
            )
        except P3CompletionError as exc:
            raise EvaluationContractError(str(exc)) from exc
        try:
            chain_path = require_regular_file_no_alias(
                training_run_dir / "chain.json", "evaluation training chain"
            )
        except HarnessError as exc:
            raise EvaluationContractError(str(exc)) from exc
        chain = json.loads(chain_path.read_text(encoding="utf-8"))
        jobs = chain.get("jobs")
        tail_job_id = (
            str(jobs[-1].get("job_id"))
            if isinstance(jobs, list) and jobs and isinstance(jobs[-1], Mapping)
            else ""
        )
        completion = progress.get("p3_completion")
        if (
            chain.get("status") != "PASSED"
            or not tail_job_id.isdigit()
            or not isinstance(completion, Mapping)
        ):
            raise EvaluationContractError("P3 chain has no accepted completion tail")
        heldout = card.get("heldout_loss_manifest")
        if not isinstance(heldout, Mapping) or any(
            completion.get(completion_field) != heldout.get(card_field)
            for completion_field, card_field in (
                ("heldout_manifest_sha256", "sha256"),
                ("data_manifest_sha256", "data_manifest_sha256"),
                ("split_sha256", "split_sha256"),
            )
        ):
            raise EvaluationContractError("P3 completion held-out provenance differs")
        try:
            validation_manifest_key = canonical_first_heldout_manifest_key(
                heldout,
                target_steps=int(card["target_steps"]),
            )
        except (KeyError, P3CompletionError) as exc:
            raise EvaluationContractError(str(exc)) from exc
        expected = {
            "slurm_job_id": tail_job_id,
            "source_commit": card["source_commit"],
            "immutable_run_card_sha256": training_card["run_card_sha256"],
            "config_sha256": card["config_sha256"],
            "container_sha256": card["container"]["sha256"],
            "target_steps": int(card["target_steps"]),
            "global_step": int(card["target_steps"]),
            "training_process_id": progress["training_process_id"],
            "checkpoint_sha256": progress["checkpoint_sha256"],
            "parameter_sha256": progress["parameter_sha256"],
            "optimizer_sha256": progress["optimizer_sha256"],
            "scheduler_sha256": progress["scheduler_sha256"],
            "manifest_sha256": heldout["sha256"],
            "data_manifest_sha256": heldout["data_manifest_sha256"],
            "split_sha256": heldout["split_sha256"],
            "training_ledger_sha256": completion["training_ledger_sha256"],
            "validation_ledger_sha256": completion["validation_ledger_sha256"],
            "checkpoint_history_sha256": completion["checkpoint_history_sha256"],
            "dataset_order_sha256": progress["sampler"]["dataset_order_sha256"],
            "sampler": progress["sampler"],
            "validation_batch.manifest_key": validation_manifest_key,
        }
        depth = card.get("depth_inputs")
        expected_empirical_adapter_mode, empirical_provenance = (
            run_card_receipt_expectations(training_card)
        )
        expected.update(
            {
                "depth_producer_sha256": depth.get("producer_sha256")
                if depth
                else None,
                "depth_cache_manifest_sha256": depth.get("cache_manifest_sha256")
                if depth
                else None,
                "depth_native_contract_sha256": depth.get("native_contract_sha256")
                if depth
                else None,
                "depth_empirical_contract_sha256": depth.get("empirical_contract_sha256")
                if depth
                else None,
                "depth_empirical_provenance": (
                    dict(empirical_provenance)
                    if empirical_provenance is not None
                    else None
                ),
                "depth_validation_sha256": depth.get("validation_sha256")
                if depth
                else None,
                "depth_checkpoint_sha256": depth.get("checkpoint_sha256")
                if depth
                else None,
            }
        )
        try:
            require_regular_file_no_alias(
                training_run_dir / "final_acceptance.json",
                "evaluation final acceptance receipt",
            )
            receipt, receipt_path, receipt_sha256 = load_final_receipt(
                training_run_dir,
                expected_empirical_adapter_mode=expected_empirical_adapter_mode,
                expected=expected,
            )
        except (HarnessError, P3CompletionError) as exc:
            raise EvaluationContractError(str(exc)) from exc
        final_event = chain.get("events", [])[-1]
        if (
            chain.get("training_process_id") != progress["training_process_id"]
            or chain.get("final_acceptance_process_id") != receipt["process_id"]
            or final_event.get("training_process_id") != progress["training_process_id"]
            or final_event.get("final_acceptance_process_id") != receipt["process_id"]
            or chain.get("final_acceptance_receipt") != str(receipt_path)
            or chain.get("final_acceptance_receipt_sha256") != receipt_sha256
            or final_event.get("final_acceptance_receipt") != str(receipt_path)
            or final_event.get("final_acceptance_receipt_sha256") != receipt_sha256
        ):
            raise EvaluationContractError(
                "P4 checkpoint lacks accepted P3 receipt binding"
            )
    return progress, checkpoint_path


def _result_provenance(
    card: Mapping[str, Any],
    training_card: Mapping[str, Any],
    *,
    run_card_path: Path,
    checkpoint_sha256: str,
    manifest_sha256: str,
    slurm_job_id: str | None,
) -> Mapping[str, Any]:
    try:
        return build_evaluation_provenance(
            card,
            training_card,
            evaluation_run_card_file_sha256=sha256_file(run_card_path),
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=manifest_sha256,
            slurm_job_id=slurm_job_id,
        )
    except HarnessError as exc:
        raise EvaluationContractError(str(exc)) from exc


def _validate_existing_result_provenance(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    requires_depth: bool,
) -> None:
    try:
        validate_evaluation_provenance(
            row, expected=expected, requires_depth=requires_depth
        )
    except HarnessError as exc:
        raise EvaluationContractError(
            f"existing evaluator output provenance differs: {exc}"
        ) from exc


def evaluate(args: argparse.Namespace) -> None:
    import torch

    try:
        manifest_path = require_regular_file_no_alias(
            args.manifest, "evaluation fixed manifest"
        )
        run_card_path = require_regular_file_no_alias(
            args.run_card, "evaluation run card"
        )
        training_run_dir = require_directory_no_alias(
            args.training_run_dir, "evaluation training directory"
        )
    except HarnessError as exc:
        raise EvaluationContractError(str(exc)) from exc
    if sha256_file(manifest_path) != args.manifest_sha256:
        raise EvaluationContractError("fixed manifest SHA-256 mismatch")
    records = _load_jsonl(manifest_path)
    keys = [str(record.get("key")) for record in records]
    if len(keys) != len(set(keys)) or not keys:
        raise EvaluationContractError("fixed manifest has duplicate or empty coverage")
    card = __import__("yaml").safe_load(run_card_path.read_text(encoding="utf-8"))
    if not isinstance(card, Mapping) or card.get("kind") not in {
        "p4-open-loop",
        "p2a-open-loop",
    }:
        raise EvaluationContractError(
            "evaluator requires an immutable P4 or P2a run card"
        )
    try:
        validate_run_card(card)
        require_run_card_authorization(card, operation="evaluation")
        training_card, _metadata = verify_evaluation_bindings(card)
    except HarnessError as exc:
        raise EvaluationContractError(str(exc)) from exc
    try:
        card_training_run_dir = require_directory_no_alias(
            Path(card["training_run_dir"]), "run-card training directory"
        )
    except HarnessError as exc:
        raise EvaluationContractError(str(exc)) from exc
    if training_run_dir != card_training_run_dir:
        raise EvaluationContractError("CLI training directory differs from run card")
    if card.get("fixed_manifest", {}).get("sha256") != args.manifest_sha256:
        raise EvaluationContractError("run card identifies a different fixed manifest")
    environment = str(card["environment"])
    if any(
        record.get("schema") != MANIFEST_SCHEMA
        or record.get("environment") != environment
        or record.get("frameskip") != FRAMESKIPS[environment]
        or record.get("horizons") != HORIZONS[environment]
        for record in records
    ):
        raise EvaluationContractError(
            "fixed manifest records violate environment contract"
        )

    progress, checkpoint_path = _verify_training_completion(
        card, training_card, training_run_dir
    )
    result_provenance = _result_provenance(
        card,
        training_card,
        run_card_path=run_card_path,
        checkpoint_sha256=str(progress["checkpoint_sha256"]),
        manifest_sha256=args.manifest_sha256,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )

    cfg, model, trajectory_dataset = _load_eval_model(
        training_run_dir, checkpoint_path, args.device
    )
    if (
        str(cfg.env.name) not in {environment, "deformable_env"}
        or int(cfg.frameskip) != FRAMESKIPS[environment]
        or int(cfg.num_hist) != NUM_HIST[environment]
        or int(cfg.training.seed) != int(card["seed"])
    ):
        raise EvaluationContractError("training config does not match evaluation card")
    base_dataset = _base_traj_dataset(trajectory_dataset)
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_episode[int(record["episode"])].append(record)

    completed = {}
    if args.out.exists():
        for row in _load_jsonl(args.out):
            _validate_existing_result_provenance(
                row,
                result_provenance,
                requires_depth=isinstance(card.get("depth_inputs"), Mapping),
            )
            if (
                row.get("schema") != RESULT_SCHEMA
                or row.get("run_id") != card["run_id"]
                or row.get("arm") != card["arm"]
                or row.get("producer") != card.get("depth_inputs", {}).get("producer")
                or int(row.get("seed", -1)) != int(card["seed"])
                or row.get("environment") != environment
                or row.get("manifest_sha256") != args.manifest_sha256
                or row.get("checkpoint_sha256") != progress["checkpoint_sha256"]
                or set(row.get("horizons", {}))
                != {str(value) for value in HORIZONS[environment]}
            ):
                raise EvaluationContractError(
                    "existing evaluator output differs from the immutable evaluation run"
                )
            episode = int(row["episode"])
            if episode in completed:
                raise EvaluationContractError(
                    "existing evaluator output duplicates an episode"
                )
            completed[episode] = row
    expected_episodes = set(by_episode)
    if not set(completed).issubset(expected_episodes):
        raise EvaluationContractError(
            "existing evaluator output contains an extra episode"
        )
    for episode, row in completed.items():
        expected_keys = {str(record["key"]) for record in by_episode[episode]}
        if set(row.get("manifest_keys", [])) != expected_keys:
            raise EvaluationContractError(
                "existing evaluator output has different fixed-manifest keys"
            )

    with torch.inference_mode():
        for episode in sorted(expected_episodes - set(completed)):
            totals = {
                horizon: {"model": 0.0, "persistence": 0.0, "elements": 0}
                for horizon in HORIZONS[environment]
            }
            episode_keys = []
            for record in sorted(by_episode[episode], key=lambda item: item["key"]):
                history_frames = [int(value) for value in record["history_frames"]]
                target_frames = [
                    int(record["target_frames"][str(horizon)])
                    for horizon in HORIZONS[environment]
                ]
                history_obs, _act, _state, _info = base_dataset.get_frames(
                    episode, history_frames
                )
                target_obs = _load_target_observations(
                    base_dataset, environment, episode, target_frames
                )
                maximum_horizon = max(HORIZONS[environment])
                num_hist = NUM_HIST[environment]
                action_blocks = []
                for step in range(num_hist + maximum_horizon - 1):
                    raw_indices = record["raw_action_indices"][str(step)]
                    action_blocks.append(
                        base_dataset.actions[episode, raw_indices].reshape(-1)
                    )
                actions = torch.stack(action_blocks).unsqueeze(0).to(args.device)
                history_obs = _tensor_obs(history_obs, args.device)
                target_obs = _tensor_obs(target_obs, args.device)
                target_z = model.encode_obs(target_obs)["visual"]
                rollout_z, _all_z = model.rollout(history_obs, actions)
                visual_rollout = rollout_z["visual"]
                persistence = visual_rollout[:, num_hist - 1]
                for target_index, horizon in enumerate(HORIZONS[environment]):
                    prediction = visual_rollout[:, num_hist - 1 + horizon]
                    target = target_z[:, target_index]
                    model_error = (prediction - target).to(torch.float64).square()
                    persistence_error = (
                        (persistence - target).to(torch.float64).square()
                    )
                    count = int(model_error.numel())
                    totals[horizon]["model"] += float(model_error.sum().cpu())
                    totals[horizon]["persistence"] += float(
                        persistence_error.sum().cpu()
                    )
                    totals[horizon]["elements"] += count
                episode_keys.append(record["key"])
            horizon_records = {}
            raw_mse = []
            for horizon in HORIZONS[environment]:
                values = totals[horizon]
                if (
                    not np.isfinite(values["model"])
                    or not np.isfinite(values["persistence"])
                    or values["elements"] <= 0
                ):
                    raise EvaluationContractError(
                        f"nonfinite or empty episode totals for episode {episode}"
                    )
                mse = values["model"] / values["elements"]
                raw_mse.append(mse)
                horizon_records[str(horizon)] = {
                    "model_squared_error": values["model"],
                    "persistence_squared_error": values["persistence"],
                    "element_count": values["elements"],
                    "raw_mse": mse,
                }
            auc = float(np.trapz(raw_mse, x=HORIZONS[environment]))
            output = {
                "schema": RESULT_SCHEMA,
                "run_id": card["run_id"],
                "arm": card["arm"],
                "producer": card.get("depth_inputs", {}).get("producer"),
                "seed": int(card["seed"]),
                "environment": environment,
                "episode": episode,
                **result_provenance,
                "manifest_keys": episode_keys,
                "manifest_key_count": len(episode_keys),
                "horizons": horizon_records,
                "horizon_auc_raw_mse": auc,
            }
            _append_jsonl(args.out, output)

    final = _load_jsonl(args.out)
    coverage = [key for row in final for key in row["manifest_keys"]]
    if len(coverage) != len(set(coverage)) or set(coverage) != set(keys):
        raise EvaluationContractError(
            "evaluator output coverage differs from fixed manifest"
        )
    print(
        json.dumps(
            {
                "state": "PASS",
                "episodes": len(final),
                "manifest_keys": len(coverage),
                "output_sha256": sha256_file(args.out),
            },
            sort_keys=True,
        )
    )


def _parse_horizons(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "horizons must be unique comma-separated integers"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--env", choices=sorted(HORIZONS), required=True)
    manifest.add_argument("--root", type=Path)
    manifest.add_argument("--n", type=int, required=True)
    manifest.add_argument("--seed", type=int, required=True)
    manifest.add_argument("--horizons", type=_parse_horizons, required=True)
    manifest.add_argument("--frameskip", type=int)
    manifest.add_argument("--out", type=Path, required=True)
    manifest.set_defaults(function=make_manifest)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--run-card", type=Path, required=True)
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--manifest-sha256", required=True)
    evaluate_parser.add_argument("--training-run-dir", type=Path, required=True)
    evaluate_parser.add_argument("--out", type=Path, required=True)
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.set_defaults(function=evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationContractError as exc:
        print(f"EVALUATION CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
