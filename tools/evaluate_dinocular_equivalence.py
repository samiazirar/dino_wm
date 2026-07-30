#!/usr/bin/env python3
"""Compare the legacy and direct DINOcular paths on one accepted Rope batch."""

from __future__ import annotations

import copy
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import hydra
import torch
from omegaconf import OmegaConf

from train import Trainer
from training_resume import (
    StepBatchSampler,
    atomic_write_json,
    capture_rng_state,
    dataset_order_sha256,
    nested_state_sha256,
    parameter_sha256,
    restore_rng_state,
)


PARENT_OUTPUT_KIND = "dino_feature_dict"
CANDIDATE_OUTPUT_KIND = "dino_encoder_feature_map"


def _clone_state(components: Mapping[str, torch.nn.Module]) -> dict[str, Any]:
    return {
        name: {
            key: value.detach().cpu().clone()
            for key, value in component.state_dict().items()
        }
        for name, component in components.items()
    }


def _restore_state(
    components: Mapping[str, torch.nn.Module], state: Mapping[str, Any]
) -> None:
    for name, component in components.items():
        component.load_state_dict(state[name], strict=True)


def _capture_gradients(
    components: Mapping[str, torch.nn.Module],
) -> dict[str, torch.Tensor]:
    result = {}
    for component_name in sorted(components):
        for parameter_name, parameter in components[component_name].named_parameters():
            if parameter.grad is not None:
                result[f"{component_name}.{parameter_name}"] = (
                    parameter.grad.detach().cpu().clone()
                )
    return result


def _tensor_comparison(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    left_names = set(left)
    right_names = set(right)
    common = sorted(left_names & right_names)
    unequal = [name for name in common if not torch.equal(left[name], right[name])]
    max_abs = 0.0
    for name in common:
        if left[name].numel():
            max_abs = max(
                max_abs,
                float(
                    (left[name].to(torch.float64) - right[name].to(torch.float64))
                    .abs()
                    .max()
                ),
            )
    return {
        "exact": not unequal and left_names == right_names,
        "left_count": len(left_names),
        "right_count": len(right_names),
        "missing_from_left": sorted(right_names - left_names),
        "missing_from_right": sorted(left_names - right_names),
        "unequal_names": unequal,
        "max_abs_difference": max_abs,
    }


def _run_path(
    trainer: Trainer,
    data: Any,
    *,
    output_kind: str,
) -> dict[str, Any]:
    encoder = trainer.accelerator.unwrap_model(trainer.encoder)
    encoder.backend_spec = replace(encoder.backend_spec, output_kind=output_kind)
    captured_features: list[torch.Tensor] = []
    encoder_module = trainer.model.encoder
    original_forward = encoder_module.forward

    def capture_feature(*args: Any, **kwargs: Any) -> torch.Tensor:
        output = original_forward(*args, **kwargs)
        captured_features.append(output.detach().cpu().clone())
        return output

    encoder_module.forward = capture_feature
    try:
        loss = trainer._train_one_step(data)
    finally:
        encoder_module.forward = original_forward
    if len(captured_features) != 1:
        raise RuntimeError(
            f"expected one encoder output for {output_kind}, got {len(captured_features)}"
        )
    patch_tokens = captured_features[0]
    components = trainer._model_components()
    optimizer_states = {
        name: optimizer.state_dict() for name, optimizer in trainer._optimizers().items()
    }
    return {
        "loss": loss,
        "features": {
            "x_norm_patchtokens": patch_tokens,
            "x_norm_clstoken": patch_tokens.mean(dim=1),
        },
        "gradients": _capture_gradients(components),
        "post_parameter_sha256": parameter_sha256(components),
        "post_optimizer_sha256": nested_state_sha256(optimizer_states),
    }


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: OmegaConf) -> None:
    output = os.environ.get("DINOCULAR_EQUIVALENCE_OUTPUT")
    if not output:
        raise RuntimeError("DINOCULAR_EQUIVALENCE_OUTPUT is required")
    if str(cfg.env.dataset.object_name) != "rope":
        raise RuntimeError("equivalence evaluation requires the accepted Rope dataset")
    if cfg.training.timing_output is not None:
        raise RuntimeError("equivalence evaluation must not enable timing instrumentation")

    trainer = Trainer(cfg)
    sampler = StepBatchSampler(
        dataset_size=len(trainer.datasets["train"]),
        batch_size=int(cfg.gpu_batch_size),
        start_step=0,
        stop_step=1,
    )
    loader = torch.utils.data.DataLoader(
        trainer.datasets["train"],
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=None,
        generator=torch.Generator().manual_seed(int(cfg.training.seed)),
    )
    data = next(iter(trainer.accelerator.prepare(loader)))
    components = trainer._model_components()
    initial_components = _clone_state(components)
    initial_optimizers = copy.deepcopy(
        {
            name: optimizer.state_dict()
            for name, optimizer in trainer._optimizers().items()
        }
    )
    initial_schedulers = copy.deepcopy(
        {name: scheduler.state_dict() for name, scheduler in trainer.schedulers.items()}
    )
    initial_rng = capture_rng_state()

    parent = _run_path(trainer, data, output_kind=PARENT_OUTPUT_KIND)

    _restore_state(components, initial_components)
    for name, optimizer in trainer._optimizers().items():
        optimizer.load_state_dict(initial_optimizers[name])
    for name, scheduler in trainer.schedulers.items():
        scheduler.load_state_dict(initial_schedulers[name])
    restore_rng_state(initial_rng)

    candidate = _run_path(trainer, data, output_kind=CANDIDATE_OUTPUT_KIND)

    feature_comparison = _tensor_comparison(parent["features"], candidate["features"])
    gradient_comparison = _tensor_comparison(
        parent["gradients"], candidate["gradients"]
    )
    loss_exact = parent["loss"] == candidate["loss"]
    post_parameter_exact = (
        parent["post_parameter_sha256"] == candidate["post_parameter_sha256"]
    )
    post_optimizer_exact = (
        parent["post_optimizer_sha256"] == candidate["post_optimizer_sha256"]
    )
    passed = all(
        [
            feature_comparison["exact"],
            gradient_comparison["exact"],
            loss_exact,
            post_parameter_exact,
            post_optimizer_exact,
        ]
    )
    result = {
        "schema": "dino-wm.dinocular-equivalence.v1",
        "status": "PASS" if passed else "FAIL",
        "source_commit": trainer.source_commit,
        "comparison": {
            "parent_commit": "799370ac6e4e5eda4922640ebc9624c33bc4858e",
            "parent_output_kind": PARENT_OUTPUT_KIND,
            "candidate_commit": "918c3e422a8bad663d9484a6105cb6acb16aff7c",
            "candidate_output_kind": CANDIDATE_OUTPUT_KIND,
        },
        "accepted_inputs": {
            "environment": "rope",
            "dataset_order_sha256": dataset_order_sha256(trainer.datasets["train"]),
            "batch_sha256": nested_state_sha256(data),
            "depth_cache_manifest_sha256": os.environ.get(
                "EQUIVALENCE_DEPTH_CACHE_MANIFEST_SHA256"
            ),
            "depth_validation_sha256": os.environ.get(
                "EQUIVALENCE_DEPTH_VALIDATION_SHA256"
            ),
            "native_depth_contract_sha256": os.environ.get(
                "DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256"
            ),
            "producer_sha256": os.environ.get("DINOCULAR_CACHE_PRODUCER_SHA256"),
            "student_checkpoint_sha256": str(cfg.encoder.checkpoint_sha256),
        },
        "feature_comparison": feature_comparison,
        "loss_comparison": {
            "exact": loss_exact,
            "parent": parent["loss"],
            "candidate": candidate["loss"],
            "absolute_difference": abs(parent["loss"] - candidate["loss"]),
        },
        "gradient_comparison": gradient_comparison,
        "post_step_comparison": {
            "parameter_exact": post_parameter_exact,
            "parent_parameter_sha256": parent["post_parameter_sha256"],
            "candidate_parameter_sha256": candidate["post_parameter_sha256"],
            "optimizer_exact": post_optimizer_exact,
            "parent_optimizer_sha256": parent["post_optimizer_sha256"],
            "candidate_optimizer_sha256": candidate["post_optimizer_sha256"],
        },
    }
    atomic_write_json(Path(output), result)
    print(f"DINOCULAR_EQUIVALENCE={result['status']} output={output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
