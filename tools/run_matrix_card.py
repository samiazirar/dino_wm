#!/usr/bin/env python3
"""Verify or execute one immutable non-training matrix run card."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_common import (  # noqa: E402
    HarnessError,
    canonical_json_bytes,
    load_yaml,
    require_run_card_authorization,
    sha256_file,
    validate_run_card,
    verify_evaluation_bindings,
)
from submit_matrix import verify_live_card  # noqa: E402


def load_verified(path: Path):
    if not path.is_file():
        raise HarnessError(f"run card is absent: {path}")
    expected_file = os.environ.get("RUN_CARD_FILE_SHA256")
    expected_content = os.environ.get("RUN_CARD_SHA256")
    if sha256_file(path) != expected_file:
        raise HarnessError("run-card file SHA-256 differs from sbatch export")
    card = load_yaml(path)
    validate_run_card(card)
    if card["run_card_sha256"] != expected_content:
        raise HarnessError("run-card content SHA-256 differs from sbatch export")
    return card


def verify(args: argparse.Namespace) -> None:
    card = load_verified(args.run_card)
    require_run_card_authorization(card, operation="run_matrix")
    verify_live_card(card)
    verify_evaluation_bindings(card)
    for key, expected in card["environment_variables"].items():
        if os.environ.get(key) != str(expected):
            raise HarnessError(f"exported environment differs from run card for {key}")
    print("RUN_CARD_PREFLIGHT=PASS")


def emit(args: argparse.Namespace) -> None:
    card = load_verified(args.run_card)
    values = {
        "container": card["container"]["path"],
        "code_root": card["code_root"],
        "run_dir": card["run_dir"],
        "arm": card["arm"],
    }
    print(values[args.field])


def evaluation_command(card, run_card: Path) -> list[str]:
    if card["kind"] not in {"p4-open-loop", "p2a-open-loop"}:
        raise HarnessError(f"generic matrix runner does not implement {card['kind']}")
    run_dir = Path(card["run_dir"])
    return [
        "python",
        f"{card['code_root']}/eval_encoder_swap.py",
        "evaluate",
        "--run-card",
        str(run_card),
        "--manifest",
        str(card["fixed_manifest"]["path"]),
        "--manifest-sha256",
        str(card["fixed_manifest"]["sha256"]),
        "--training-run-dir",
        str(card["training_run_dir"]),
        "--out",
        str(run_dir / "episode_errors.jsonl"),
    ]


def _prepare_run_dir(card) -> Path:
    run_dir = Path(card["run_dir"])
    marker_path = run_dir / "run_marker.json"
    marker = {
        "schema": "dino-wm-evaluation-run-marker-v1",
        "evaluation_run_card_sha256": card["run_card_sha256"],
        "training_run_card": card["training_run_card"],
        "fixed_manifest": card["fixed_manifest"],
        "training_run_dir": card["training_run_dir"],
    }
    text = json.dumps(marker, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if run_dir.exists():
        if not marker_path.is_file() or marker_path.read_text(encoding="utf-8") != text:
            raise HarnessError(
                "evaluation run directory exists without the matching immutable marker"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        temporary = marker_path.with_name(f".{marker_path.name}.tmp.{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker_path)
    if canonical_json_bytes(
        json.loads(marker_path.read_text(encoding="utf-8"))
    ) != canonical_json_bytes(marker):
        raise HarnessError("evaluation run marker content differs")
    return run_dir


def execute(args: argparse.Namespace) -> None:
    card = load_verified(args.run_card)
    require_run_card_authorization(card, operation="run_matrix")
    for key, expected in card["environment_variables"].items():
        if os.environ.get(key) != str(expected):
            raise HarnessError(f"runtime environment differs from run card for {key}")
    verify_evaluation_bindings(card)
    _prepare_run_dir(card)
    command = evaluation_command(card, args.run_card)
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--run-card", type=Path, required=True)
    verify_parser.set_defaults(function=verify)
    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("--run-card", type=Path, required=True)
    emit_parser.add_argument(
        "--field", choices=["container", "code_root", "run_dir", "arm"], required=True
    )
    emit_parser.set_defaults(function=emit)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--run-card", type=Path, required=True)
    execute_parser.set_defaults(function=execute)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as exc:
        print(f"HARNESS CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
