#!/usr/bin/env python3
"""Post-train geometry and timing gates for immutable P2 run cards."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_common import HarnessError, sha256_file, validate_run_card  # noqa: E402


def _load(path: Path, mode: str) -> Mapping[str, Any]:
    card = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(card, Mapping):
        raise HarnessError("P2 run card is not an object")
    validate_run_card(card)
    if card.get("kind") != f"p2-{mode}" or card.get("gate_mode") != mode:
        raise HarnessError(f"P2 run card does not invoke the {mode} gate")
    return card


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_cem_horizon_five(
    model: Any, obs: Mapping[str, Any], *, num_hist: int, action_dim: int
) -> Any:
    import torch

    from planning.cem import CEMPlanner

    class IdentityPreprocessor:
        @staticmethod
        def transform_obs(value):
            return value

    class NullRun:
        @staticmethod
        def log(_value):
            return None

    def objective(imagined, goal):
        difference = imagined["visual"][:, -1] - goal["visual"][:, -1]
        return difference.flatten(1).square().mean(dim=1)

    history = {key: value[:, :num_hist] for key, value in obs.items()}
    goal = {key: value[:, num_hist - 1 : num_hist] for key, value in obs.items()}
    torch.manual_seed(0)
    planner = CEMPlanner(
        horizon=5,
        topk=2,
        num_samples=4,
        var_scale=1.0,
        opt_steps=1,
        eval_every=1,
        wm=model,
        action_dim=action_dim,
        objective_fn=objective,
        preprocessor=IdentityPreprocessor(),
        evaluator=None,
        wandb_run=NullRun(),
        log_filename=None,
    )
    actions, _lengths = planner.plan(history, goal)
    if tuple(actions.shape) != (1, 5, action_dim) or not torch.isfinite(actions).all():
        raise HarnessError(
            f"CEM horizon-5 output is invalid: shape={tuple(actions.shape)}"
        )
    return actions


def geometry(args: argparse.Namespace) -> None:
    import gym
    import hydra
    from omegaconf import OmegaConf
    import torch

    import env  # noqa: F401
    from eval_encoder_swap import _load_eval_model

    card = _load(args.run_card, "geometry")
    progress = json.loads((args.run_dir / "progress.json").read_text(encoding="utf-8"))
    if (
        progress.get("status") != "TARGET_REACHED"
        or progress.get("global_step") != 1
        or not math.isfinite(float(progress.get("last_step_loss")))
    ):
        raise HarnessError("geometry optimizer/checkpoint gate did not reach one finite step")
    checkpoint = Path(progress["checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != progress.get(
        "checkpoint_sha256"
    ):
        raise HarnessError("geometry final checkpoint hash mismatch")
    cfg, model, _trajectory_dataset = _load_eval_model(args.run_dir, checkpoint, "cuda")
    raw_encoder = model.encoder
    expected_geometry = (
        (196, 384, 196)
        if card["arm"] == "dino_pinned"
        else (49, 512, 224)
    )
    actual_geometry = (
        int(raw_encoder.num_patches),
        int(raw_encoder.emb_dim),
        int(raw_encoder.input_size),
    )
    if actual_geometry != expected_geometry:
        raise HarnessError(
            f"encoder geometry differs: {actual_geometry} versus {expected_geometry}"
        )
    if card["arm"] != "dino_pinned":
        expected_neutral = card["arm"] == "dinocular_zerodepth"
        if bool(raw_encoder.neutralize_depth_at_encoder_input) != expected_neutral:
            raise HarnessError("DINOcular neutral-boundary switch differs from arm")
        if raw_encoder.input_metadata["checkpoint_sha256"] != card["artifacts"][
            "dinocular_student"
        ]["sha256"]:
            raise HarnessError("DINOcular checkpoint metadata differs from run card")
        if raw_encoder.input_metadata["native_depth_contract_sha256"] != card[
            "depth_inputs"
        ]["native_contract_sha256"]:
            raise HarnessError("DINOcular native contract metadata differs from run card")

    datasets, _traj = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    expected_action_dim = 10 if card["environment"] in {"pusht", "wall"} else 4
    if int(datasets["train"].action_dim) != expected_action_dim:
        raise HarnessError("geometry action dimension differs from frame-skip contract")
    loader = torch.utils.data.DataLoader(
        datasets["valid"], batch_size=1, shuffle=False, num_workers=0
    )
    obs, act, _state = next(iter(loader))
    obs = {key: value.cuda() for key, value in obs.items()}
    act = act.cuda()
    boundary_sentinel = []
    remove_boundary_hook = None
    if card["arm"] == "dinocular_zerodepth":
        expected_depth = float(raw_encoder.neutral_normalized_depth)
        expected_mask = float(raw_encoder.neutral_validity_mask)

        def observe_boundary(_encoder, depth_value, mask_value):
            boundary_sentinel.append(
                {
                    "depth": bool(torch.all(depth_value == expected_depth)),
                    "mask": bool(torch.all(mask_value == expected_mask)),
                }
            )

        remove_boundary_hook = raw_encoder.register_encoder_boundary_hook(
            observe_boundary
        )
    try:
        with torch.inference_mode():
            output = model(obs, act)
            loss = output[3]
            rollout_obs, rollout_all = model.rollout(
                {key: value[:, : int(cfg.num_hist)] for key, value in obs.items()},
                act[:, : int(cfg.num_hist)],
            )
            cem_actions = None
            if card["environment"] in {"rope", "granular"}:
                cem_actions = _run_cem_horizon_five(
                    model,
                    obs,
                    num_hist=int(cfg.num_hist),
                    action_dim=4,
                )
    finally:
        if remove_boundary_hook is not None:
            remove_boundary_hook()
    if not torch.isfinite(loss) or not torch.isfinite(rollout_obs["visual"]).all():
        raise HarnessError("geometry forward or one-step open-loop result is nonfinite")
    if rollout_all.shape[1] != int(cfg.num_hist) + 1:
        raise HarnessError("one-step open-loop output has wrong temporal geometry")
    if card["arm"] == "dinocular_zerodepth" and (
        not boundary_sentinel
        or not all(item["depth"] and item["mask"] for item in boundary_sentinel)
    ):
        raise HarnessError(
            "zero-depth boundary did not apply the manifest neutral depth and fixed mask"
        )

    kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True)
    planning_env = gym.make(str(cfg.env.name), *list(cfg.env.args), **kwargs)
    try:
        reset_value = planning_env.reset()
        action_value = planning_env.action_space.sample()
        raw_action_dim = int(__import__("numpy").asarray(action_value).reshape(-1).shape[0])
        expected_raw_action_dim = 4 if card["environment"] in {"rope", "granular"} else 2
        if raw_action_dim != expected_raw_action_dim:
            raise HarnessError(
                f"planning raw action dimension differs: {raw_action_dim} versus {expected_raw_action_dim}"
            )
        step_value = planning_env.step(action_value)
        if reset_value is None or step_value is None:
            raise HarnessError("planning reset/step returned no value")
        if card["environment"] == "pusht":
            rendered = planning_env.render(mode="rgb_array")
            if rendered is None:
                raise HarnessError("PushT planning render returned no frame")
        elif card["environment"] in {"rope", "granular"}:
            rendered = planning_env.render()
            if rendered is None:
                raise HarnessError("deformable planning render returned no frame")
        else:
            reset_obs = reset_value[0] if isinstance(reset_value, tuple) else reset_value
            if not isinstance(reset_obs, Mapping) or "visual" not in reset_obs:
                raise HarnessError("Wall planning reset did not render a visual observation")
    finally:
        planning_env.close()

    result = {
        "schema": "dino-wm-p2-geometry-gate-v1",
        "state": "PASS",
        "run_id": card["run_id"],
        "checkpoint_sha256": progress["checkpoint_sha256"],
        "encoder_geometry": list(actual_geometry),
        "action_dim": expected_action_dim,
        "raw_action_dim": expected_raw_action_dim,
        "optimizer_steps": 1,
        "finite_loss": float(loss.cpu()),
        "open_loop_steps": 1,
        "planning_reset_render_step": "PASS",
        "cem_horizon_5": (
            "PASS" if card["environment"] in {"rope", "granular"} else "NOT_APPLICABLE"
        ),
        "zero_depth_boundary_forwards": len(boundary_sentinel),
    }
    _atomic_json(args.run_dir / "geometry_gate.json", result)
    print(json.dumps(result, sort_keys=True))


def timing(args: argparse.Namespace) -> None:
    card = _load(args.run_card, "timing")
    result_path = args.run_dir / "timing_result.json"
    runtime_card_path = args.run_dir / "timing_runtime_card.yaml"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    runtime_card = yaml.safe_load(runtime_card_path.read_text(encoding="utf-8"))
    expected = card["timing"]
    if (
        result.get("schema") != "dino-wm.strict-p2-timing.v1"
        or result.get("status") != "MEASURED_PASS"
        or result.get("arm") != card["arm"]
        or result.get("environment") != card["environment"]
        or result.get("warmup_steps_excluded") != expected["warmup_steps"]
        or result.get("measured_steps") != expected["fixed_steps"]
        or result.get("global_batch_size") != 32
        or result.get("frame_skip") != card["frameskip"]
        or runtime_card.get("status") != "PASSED"
    ):
        raise HarnessError("strict timing output differs from the immutable P2 card")
    for key in (
        "steps_per_second",
        "samples_per_second",
        "measured_seconds",
        "final_loss",
        "peak_torch_reserved_mib",
    ):
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise HarnessError(f"strict timing result {key} is nonfinite")
    gate = {
        "schema": "dino-wm-p2-timing-gate-v1",
        "state": "PASS",
        "run_id": card["run_id"],
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "runtime_card_path": str(runtime_card_path),
        "runtime_card_sha256": sha256_file(runtime_card_path),
        "measured_steps": expected["fixed_steps"],
        "warmup_steps": expected["warmup_steps"],
    }
    _atomic_json(args.run_dir / "timing_gate.json", gate)
    print(json.dumps(gate, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name, function in (("geometry", geometry), ("timing", timing)):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--run-card", type=Path, required=True)
        subparser.add_argument("--run-dir", type=Path, required=True)
        subparser.set_defaults(function=function)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as exc:
        print(f"P2 HARNESS GATE FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
