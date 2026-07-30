#!/usr/bin/env python3
"""Profile the two frozen encoders and their accepted downstream modules.

This tool is intentionally forward-only.  It loads the exact local checkpoints,
counts unique model state, records module-hook traversal, reports only FLOPs
supported by PyTorch's dispatch profiler (plus every observed unsupported
operator), and measures synchronized single-image latency on one CUDA device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


DINOV2_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINOCULAR_SHA256 = "decc7c73283bf46f66dbbedec6fb065ad0a943fb2ddb1c49317fafb0319f5dcc"

TASKS = {
    "pusht": {"num_hist": 3, "action_input_dim": 10, "proprio_input_dim": 4},
    "wall": {"num_hist": 1, "action_input_dim": 10, "proprio_input_dim": 4},
    "rope": {"num_hist": 1, "action_input_dim": 4, "proprio_input_dim": 1},
    "granular": {"num_hist": 1, "action_input_dim": 4, "proprio_input_dim": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--dinov2-repo", type=Path, required=True)
    parser.add_argument("--dinov2-weights", type=Path, required=True)
    parser.add_argument("--dinocular-weights", type=Path, required=True)
    parser.add_argument("--native-depth-contract", type=Path, required=True)
    parser.add_argument("--native-depth-contract-sha256", required=True)
    parser.add_argument("--cache-producer-sha256", required=True)
    parser.add_argument("--cache-environment", default="wall")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_named_tensors(
    rows: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    seen: set[int] = set()
    unique = []
    for name, tensor in rows:
        identity = id(tensor)
        if identity not in seen:
            seen.add(identity)
            unique.append((name, tensor))
    return unique


def state_counts(model: torch.nn.Module) -> dict[str, Any]:
    params = unique_named_tensors(list(model.named_parameters(remove_duplicate=False)))
    buffers = unique_named_tensors(list(model.named_buffers(remove_duplicate=False)))
    return {
        "parameters": sum(tensor.numel() for _, tensor in params),
        "trainable_parameters": sum(
            tensor.numel() for _, tensor in params if tensor.requires_grad
        ),
        "frozen_parameters": sum(
            tensor.numel() for _, tensor in params if not tensor.requires_grad
        ),
        "parameter_tensors": len(params),
        "buffers": sum(tensor.numel() for _, tensor in buffers),
        "buffer_tensors": len(buffers),
    }


def forward_activity(
    model: torch.nn.Module, forward: Callable[[], torch.Tensor]
) -> tuple[torch.Tensor, dict[str, Any]]:
    active_parameter_ids: set[int] = set()
    active_buffer_ids: set[int] = set()
    traversed_modules: set[str] = set()
    module_names = {id(module): name or "<root>" for name, module in model.named_modules()}
    handles = []
    params = unique_named_tensors(list(model.named_parameters(remove_duplicate=False)))
    buffers = unique_named_tensors(list(model.named_buffers(remove_duplicate=False)))
    tracked_tensor_ids = {
        id(tensor)
        for _, tensor in [*params, *buffers]
    }

    class DirectTensorUseMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: Any = None,
        ) -> Any:
            flat, _ = tree_flatten((args, kwargs or {}))
            for value in flat:
                if torch.is_tensor(value) and id(value) in tracked_tensor_ids:
                    if isinstance(value, torch.nn.Parameter):
                        active_parameter_ids.add(id(value))
                    else:
                        active_buffer_ids.add(id(value))
            return func(*args, **(kwargs or {}))

    def mark(module: torch.nn.Module, _inputs: Any) -> None:
        traversed_modules.add(module_names[id(module)])
        for parameter in module.parameters(recurse=False):
            active_parameter_ids.add(id(parameter))
        for buffer in module.buffers(recurse=False):
            active_buffer_ids.add(id(buffer))

    for module in model.modules():
        handles.append(module.register_forward_pre_hook(mark))
    try:
        with torch.inference_mode(), DirectTensorUseMode():
            output = forward()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    inactive_parameters = [
        name for name, tensor in params if id(tensor) not in active_parameter_ids
    ]
    inactive_buffers = [
        name for name, tensor in buffers if id(tensor) not in active_buffer_ids
    ]
    return output, {
        "method": (
            "A parameter or buffer is active when its owning module's forward-pre-hook "
            "fires or the exact tensor is observed by TorchDispatch (covering direct "
            "forward_features method calls); shared tensors are counted once by Python "
            "object identity."
        ),
        "traversed_module_count": len(traversed_modules),
        "active_parameters": sum(
            tensor.numel() for _, tensor in params if id(tensor) in active_parameter_ids
        ),
        "active_parameter_tensors": sum(
            id(tensor) in active_parameter_ids for _, tensor in params
        ),
        "active_buffers": sum(
            tensor.numel() for _, tensor in buffers if id(tensor) in active_buffer_ids
        ),
        "active_buffer_tensors": sum(
            id(tensor) in active_buffer_ids for _, tensor in buffers
        ),
        "inactive_parameter_names": inactive_parameters,
        "inactive_buffer_names": inactive_buffers,
    }


def flop_counts(forward: Callable[[], torch.Tensor]) -> dict[str, Any]:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.inference_mode(), torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
    ) as profiler:
        output = forward()
    torch.cuda.synchronize()
    events = profiler.key_averages()
    counted = {
        event.key: int(event.flops)
        for event in sorted(events, key=lambda event: event.key)
        if int(event.flops) > 0
    }
    unsupported = sorted(
        event.key
        for event in events
        if event.key.startswith(("aten::", "xformers::")) and int(event.flops) == 0
    )
    return {
        "supported_flops": sum(counted.values()),
        "counted_operator_flops": counted,
        "unsupported_observed_operators": unsupported,
        "unsupported_disclosure": (
            "supported_flops is not an exact total: torch.profiler estimates only "
            "supported matrix-multiplication and 2-D-convolution operations. Observed "
            "ATen/xFormers operators with zero profiler FLOPs are listed and contribute zero."
        ),
        "output_shape": list(output.shape),
    }


def latency(
    forward: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            forward()
        torch.cuda.synchronize()
        elapsed_ms = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            forward()
            torch.cuda.synchronize()
            elapsed_ms.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(elapsed_ms)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "warmup": warmup,
        "repeats": repeats,
        "synchronization": "torch.cuda.synchronize after warmup and every measured forward",
        "mean_ms": statistics.fmean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "p05_ms": percentile(0.05),
        "p95_ms": percentile(0.95),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
    }


def profile_encoder(
    model: torch.nn.Module,
    forward: Callable[[], torch.Tensor],
    *,
    input_contract: dict[str, Any],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    model.eval()
    output, activity = forward_activity(model, forward)
    return {
        "input_contract": input_contract,
        "output_shape": list(output.shape),
        "state": state_counts(model),
        "forward_activity": activity,
        "flops": flop_counts(forward),
        "latency": latency(forward, warmup, repeats),
    }


def downstream_counts(
    predictor_type: type[torch.nn.Module],
    proprio_type: type[torch.nn.Module],
    *,
    device: torch.device,
) -> dict[str, Any]:
    arms = {
        "dino_pinned": {"num_patches": 196, "encoder_dim": 384},
        "dinocular": {"num_patches": 49, "encoder_dim": 512},
    }
    results: dict[str, Any] = {}
    for task_name, task in TASKS.items():
        task_rows: dict[str, Any] = {}
        for arm, arm_spec in arms.items():
            predictor_dim = arm_spec["encoder_dim"] + 10 + 10
            predictor = predictor_type(
                num_patches=arm_spec["num_patches"],
                num_frames=task["num_hist"],
                dim=predictor_dim,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dim_head=64,
                dropout=0.1,
                emb_dropout=0.0,
                pool="mean",
            ).to(device)
            action_encoder = proprio_type(
                num_frames=1,
                tubelet_size=1,
                in_chans=task["action_input_dim"],
                emb_dim=10,
                use_3d_pos=False,
            ).to(device)
            proprio_encoder = proprio_type(
                num_frames=1,
                tubelet_size=1,
                in_chans=task["proprio_input_dim"],
                emb_dim=10,
                use_3d_pos=False,
            ).to(device)
            predictor_state = state_counts(predictor)
            action_state = state_counts(action_encoder)
            proprio_state = state_counts(proprio_encoder)
            task_rows[arm] = {
                "num_hist": task["num_hist"],
                "visual_tokens_per_frame": arm_spec["num_patches"],
                "visual_width": arm_spec["encoder_dim"],
                "predictor_input_width": predictor_dim,
                "action_input_dim": task["action_input_dim"],
                "proprio_input_dim": task["proprio_input_dim"],
                "predictor_parameters": predictor_state["parameters"],
                "action_encoder_parameters": action_state["parameters"],
                "proprio_state_encoder_parameters": proprio_state["parameters"],
                "total_trainable_downstream_parameters": (
                    predictor_state["parameters"]
                    + action_state["parameters"]
                    + proprio_state["parameters"]
                ),
            }
            del predictor, action_encoder, proprio_encoder
        results[task_name] = task_rows
    return results


def nvidia_smi() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    name, uuid, driver = [item.strip() for item in completed.stdout.strip().split(",")]
    return {"name": name, "uuid": uuid, "driver_version": driver}


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.repeats < 2 or args.batch_size < 1:
        raise ValueError("warmup >= 1, repeats >= 2, and batch-size >= 1 are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for same-runtime latency measurement")

    code_root = args.code_root.resolve()
    sys.path.insert(0, str(code_root))
    from models.dino import DinoV2Encoder
    from models.dinocular import DinocularEncoder
    from models.proprio import ProprioceptiveEmbedding
    from models.vit import ViTPredictor

    artifact_hashes = {
        "dinov2_weights": sha256_file(args.dinov2_weights),
        "dinocular_weights": sha256_file(args.dinocular_weights),
        "native_depth_contract": sha256_file(args.native_depth_contract),
    }
    if artifact_hashes["dinov2_weights"] != DINOV2_SHA256:
        raise RuntimeError("DINOv2 checkpoint hash mismatch")
    if artifact_hashes["dinocular_weights"] != DINOCULAR_SHA256:
        raise RuntimeError("DINOcular checkpoint hash mismatch")
    if artifact_hashes["native_depth_contract"] != args.native_depth_contract_sha256:
        raise RuntimeError("native depth contract hash mismatch")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = torch.float32
    dino = DinoV2Encoder(
        name="dinov2_vits14",
        feature_key="x_norm_patchtokens",
        repo_dir=str(args.dinov2_repo),
        weights_path=str(args.dinov2_weights),
        weights_sha256=DINOV2_SHA256,
        frozen=True,
    ).to(device=device, dtype=dtype)
    dinocular = DinocularEncoder(
        name="dinocular_student_dropout_fullpr",
        backend="df2_dino_rope_convs_de",
        factory="DFormerv2_S",
        checkpoint_path=str(args.dinocular_weights),
        checkpoint_sha256=DINOCULAR_SHA256,
        checkpoint_key="student",
        state_prefix="module.backbone.",
        allowed_outside_prefixes=("module.dino_head.", "module.ibot_head."),
        allowed_missing_keys=(),
        feature_key="x_norm_patchtokens",
        input_size=224,
        num_patches=49,
        emb_dim=512,
        frozen=True,
        depth_contract_status="complete",
        native_depth_contract_path=str(args.native_depth_contract),
        native_depth_contract_sha256=args.native_depth_contract_sha256,
        selected_cache_producer_sha256=args.cache_producer_sha256,
        selected_cache_environment=args.cache_environment,
        neutralize_depth_at_encoder_input=False,
    ).to(device=device, dtype=dtype)

    dino_rgb = torch.zeros(
        args.batch_size, 3, 196, 196, device=device, dtype=dtype
    )
    dinocular_rgb = torch.zeros(
        args.batch_size, 3, 224, 224, device=device, dtype=dtype
    )
    dinocular_depth = torch.full(
        (args.batch_size, 1, 224, 224), 0.5, device=device, dtype=dtype
    )
    dinocular_validity = torch.ones_like(dinocular_depth)
    dino_forward = lambda: dino(dino_rgb)
    dinocular_forward = lambda: dinocular(
        dinocular_rgb, dinocular_depth, dinocular_validity
    )

    dino_profile = profile_encoder(
        dino,
        dino_forward,
        input_contract={
            "channels": {"rgb": 3, "depth": 0},
            "resolution": [196, 196],
            "batch_size": args.batch_size,
            "dtype": str(dtype),
        },
        warmup=args.warmup,
        repeats=args.repeats,
    )
    dinocular_profile = profile_encoder(
        dinocular,
        dinocular_forward,
        input_contract={
            "channels": {"rgb": 3, "depth": 1, "depth_validity_mask": 1},
            "resolution": [224, 224],
            "batch_size": args.batch_size,
            "dtype": str(dtype),
        },
        warmup=args.warmup,
        repeats=args.repeats,
    )
    downstream = downstream_counts(
        ViTPredictor, ProprioceptiveEmbedding, device=device
    )
    downstream_ratios = {
        task: rows["dinocular"]["total_trainable_downstream_parameters"]
        / rows["dino_pinned"]["total_trainable_downstream_parameters"]
        for task, rows in downstream.items()
    }
    ratios = {
        "loaded_encoder_parameters_dinocular_over_dino": (
            dinocular_profile["state"]["parameters"]
            / dino_profile["state"]["parameters"]
        ),
        "active_encoder_parameters_dinocular_over_dino": (
            dinocular_profile["forward_activity"]["active_parameters"]
            / dino_profile["forward_activity"]["active_parameters"]
        ),
        "supported_flops_dinocular_over_dino": (
            dinocular_profile["flops"]["supported_flops"]
            / dino_profile["flops"]["supported_flops"]
        ),
        "median_latency_dinocular_over_dino": (
            dinocular_profile["latency"]["median_ms"]
            / dino_profile["latency"]["median_ms"]
        ),
        "downstream_parameters_dinocular_over_dino": downstream_ratios,
    }
    capacity_ratios = [
        ratios["loaded_encoder_parameters_dinocular_over_dino"],
        ratios["active_encoder_parameters_dinocular_over_dino"],
        ratios["supported_flops_dinocular_over_dino"],
        *downstream_ratios.values(),
    ]
    lower_tolerance, upper_tolerance = 2.0 / 3.0, 1.5
    defensible = all(
        lower_tolerance <= ratio <= upper_tolerance for ratio in capacity_ratios
    )
    result = {
        "schema": "dinocular.encoder-capacity-profile.v1",
        "command": {
            "argv": sys.argv,
            "batch_size": args.batch_size,
            "dtype": str(dtype),
            "forward_only": True,
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": nvidia_smi(),
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "container": os.environ.get("PROFILE_CONTAINER"),
            "container_sha256": os.environ.get("PROFILE_CONTAINER_SHA256"),
            "code_root": str(code_root),
            "source_commit": subprocess.run(
                ["git", "-C", str(code_root), "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip(),
        },
        "artifacts": {
            "paths": {
                "dinov2_repo": str(args.dinov2_repo),
                "dinov2_weights": str(args.dinov2_weights),
                "dinocular_weights": str(args.dinocular_weights),
                "native_depth_contract": str(args.native_depth_contract),
            },
            "sha256": artifact_hashes,
        },
        "encoders": {
            "dino_pinned": dino_profile,
            "dinocular": dinocular_profile,
        },
        "downstream": downstream,
        "comparison": {
            "ratios": ratios,
            "fairness_tolerance": {
                "lower_ratio": lower_tolerance,
                "upper_ratio": upper_tolerance,
                "applies_to": [
                    "loaded encoder parameters",
                    "module-hook active encoder parameters",
                    "profiler-supported encoder FLOPs",
                    "task-specific total trainable downstream parameters",
                ],
                "justification": (
                    "A symmetric 1.5x capacity band is strict enough to reject a "
                    "different model-size tier while allowing token/width trade-offs "
                    "between unlike frozen architectures."
                ),
                "latency_treatment": (
                    "Same-GPU latency is an implementation-efficiency diagnostic, not "
                    "a representational-capacity pass criterion. It is reported because "
                    "it determines operational cost; supported FLOPs disclose arithmetic "
                    "coverage and unsupported operators explicitly."
                ),
            },
            "verdict": (
                "DEFEND_DINOV2_VITS14_PRIMARY_BASELINE"
                if defensible
                else "SUPPLEMENTAL_RGB_COMPARATOR_REQUIRED"
            ),
            "supplemental_rgb_comparator": None,
            "supplemental_training_cells": 0 if defensible else 12,
            "reason": (
                "All loaded/active encoder, supported-FLOP, and task-specific "
                "downstream-parameter ratios fall inside the declared 2/3-to-1.5 "
                "capacity band. The large latency ratio is retained as an operational "
                "cost difference and does not add trainable capacity."
                if defensible
                else "At least one capacity or supported-compute ratio is outside the "
                "declared 2/3-to-1.5 band."
            ),
        },
        "dinocular_load_audit": {
            key: getattr(dinocular.load_audit, key)
            for key in dinocular.load_audit.__dataclass_fields__
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
