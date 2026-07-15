#!/usr/bin/env python3
"""Collect the exact 12-card P2 timing matrix into a submission rate summary."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_common import (  # noqa: E402
    HarnessError,
    LOCKED_ARMS,
    LOCKED_ENVS,
    LOCKED_FRAMESKIPS,
    LOCKED_TARGETS,
    TIMING_SUMMARY_SCHEMA,
    load_matrix,
    sha256_file,
)


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise HarnessError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise HarnessError(f"{label} is not an object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise HarnessError(f"nonfinite timing field {label}")
    return float(value)


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise HarnessError(f"immutable timing summary differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def collect(args: argparse.Namespace) -> None:
    matrix, cards = load_matrix(args.matrix)
    if matrix.get("kind") != "p2-timing" or len(cards) != 12:
        raise HarnessError("timing collector requires the exact 12-card P2 timing matrix")
    expected_axes = {
        (arm, environment) for arm in LOCKED_ARMS for environment in LOCKED_ENVS
    }
    actual_axes = {(str(card["arm"]), str(card["environment"])) for card in cards}
    if actual_axes != expected_axes:
        raise HarnessError("P2 timing matrix axes differ from the locked 3x4 set")
    references = {str(item["run_id"]): item for item in matrix["cards"]}
    rates = {}
    evidence = {}
    for card in cards:
        arm = str(card["arm"])
        environment = str(card["environment"])
        run_id = str(card["run_id"])
        if (
            card.get("gate_mode") != "timing"
            or card.get("timing") != {"fixed_steps": 200, "warmup_steps": 20}
            or card.get("target_steps") != 220
            or card.get("segment_steps") != 220
            or card.get("frameskip") != LOCKED_FRAMESKIPS[environment]
        ):
            raise HarnessError(f"P2 timing card settings differ for {run_id}")
        run_dir = (
            args.results_root / run_id
            if args.results_root is not None
            else Path(str(card["run_dir"]))
        )
        gate_path = run_dir / "timing_gate.json"
        gate = _load_json(gate_path, "timing gate")
        if (
            gate.get("schema") != "dino-wm-p2-timing-gate-v1"
            or gate.get("state") != "PASS"
            or gate.get("run_id") != run_id
        ):
            raise HarnessError(f"timing gate did not pass for {run_id}")
        result_path = Path(str(gate.get("result_path")))
        runtime_path = Path(str(gate.get("runtime_card_path")))
        if result_path != run_dir / "timing_result.json" or runtime_path != run_dir / "timing_runtime_card.yaml":
            raise HarnessError(f"timing gate paths escape the immutable run directory for {run_id}")
        if (
            sha256_file(result_path) != gate.get("result_sha256")
            or sha256_file(runtime_path) != gate.get("runtime_card_sha256")
        ):
            raise HarnessError(f"timing output hash mismatch for {run_id}")
        result = _load_json(result_path, "timing result")
        runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
        if not isinstance(runtime, Mapping):
            raise HarnessError(f"timing runtime card is not an object for {run_id}")
        protocol = runtime.get("protocol", {})
        artifacts = runtime.get("artifacts", {})
        if (
            result.get("schema") != "dino-wm.strict-p2-timing.v1"
            or result.get("status") != "MEASURED_PASS"
            or result.get("arm") != arm
            or result.get("environment") != environment
            or result.get("warmup_steps_excluded") != 20
            or result.get("measured_steps") != 200
            or result.get("global_batch_size") != 32
            or result.get("frame_skip") != LOCKED_FRAMESKIPS[environment]
            or result.get("projection_target_steps") != LOCKED_TARGETS[environment]
            or runtime.get("status") != "PASSED"
            or runtime.get("arm") != arm
            or runtime.get("environment") != environment
            or protocol.get("global_batch_size") != 32
            or protocol.get("num_workers") != 0
            or protocol.get("frame_skip") != LOCKED_FRAMESKIPS[environment]
            or protocol.get("warmup_steps_excluded") != 20
            or protocol.get("measured_optimizer_steps") != 200
            or protocol.get("projection_target_steps") != LOCKED_TARGETS[environment]
            or artifacts.get("source_commit") != card.get("source_commit")
            or artifacts.get("container_sha256") != card.get("container", {}).get("sha256")
        ):
            raise HarnessError(f"timing result/runtime settings differ for {run_id}")
        rate = _finite(result.get("steps_per_second"), f"{run_id}.steps_per_second")
        if rate <= 0:
            raise HarnessError(f"nonpositive timing rate for {run_id}")
        for name in (
            "samples_per_second",
            "measured_seconds",
            "final_loss",
            "peak_torch_reserved_mib",
            "train_windows",
        ):
            _finite(result.get(name), f"{run_id}.{name}")
        key = f"{arm}/{environment}"
        rates[key] = {
            "state": "PASS",
            "optimizer_steps_per_second": rate,
            "samples_per_second": float(result["samples_per_second"]),
            "train_windows": int(result["train_windows"]),
            "peak_torch_reserved_mib": float(result["peak_torch_reserved_mib"]),
            "result_sha256": gate["result_sha256"],
            "runtime_card_sha256": gate["runtime_card_sha256"],
        }
        reference = references[run_id]
        evidence[key] = {
            "run_id": run_id,
            "run_card_path": reference["path"],
            "run_card_file_sha256": reference["file_sha256"],
            "run_card_sha256": reference["run_card_sha256"],
            "timing_gate_path": str(gate_path),
            "timing_gate_sha256": sha256_file(gate_path),
            "result_path": str(result_path),
            "runtime_card_path": str(runtime_path),
        }
    output = {
        "schema": TIMING_SUMMARY_SCHEMA,
        "state": "PASS",
        "source_commit": matrix["source_commit"],
        "matrix_path": str(args.matrix.resolve()),
        "matrix_sha256": matrix["matrix_sha256"],
        "card_count": 12,
        "rates": rates,
        "throughput_evidence": evidence,
    }
    _write_immutable(args.out, output)
    print(json.dumps(output, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    collect(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HarnessError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"P2 TIMING COLLECTION FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
