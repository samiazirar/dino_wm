#!/usr/bin/env python3
"""Materialize immutable all-validation P3 held-out loss manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_completion import (  # noqa: E402
    P3CompletionError,
    materialize_heldout_manifest,
    runtime_slice_entries,
)
from tools.harness_common import (  # noqa: E402
    HarnessError,
    LOCKED_ENVS,
    validate_spec,
)


def _datasets(
    environment: str, data_root: Path, spec: Mapping[str, Any]
) -> Mapping[str, Any]:
    from datasets.deformable_env_dset import load_deformable_dset_slice_train_val
    from datasets.pusht_dset import load_pusht_slice_train_val
    from datasets.wall_dset import load_wall_slice_train_val

    record = spec["environments"][environment]
    common = {
        "n_rollout": None,
        "transform": None,
        "num_hist": int(record["num_hist"]),
        "num_pred": 1,
        "frameskip": int(record["frameskip"]),
    }
    if environment == "pusht":
        datasets, _trajectory = load_pusht_slice_train_val(
            data_path=str(data_root / "pusht_noise"),
            normalize_action=True,
            split_ratio=0.9,
            with_velocity=True,
            **common,
        )
    elif environment == "wall":
        datasets, _trajectory = load_wall_slice_train_val(
            data_path=str(data_root / "wall_single"),
            normalize_action=False,
            split_ratio=0.9,
            split_mode="random",
            **common,
        )
    else:
        datasets, _trajectory = load_deformable_dset_slice_train_val(
            data_path=str(data_root / "deformable"),
            object_name=environment,
            normalize_action=False,
            split_ratio=0.9,
            **common,
        )
    return datasets


def materialize(args: argparse.Namespace) -> None:
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    validate_spec(spec)
    outputs = {
        environment: str((args.out_dir / f"heldout_{environment}.jsonl").resolve())
        for environment in LOCKED_ENVS
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "dino-wm-p3-heldout-dry-run-v1",
                    "state": "PASS",
                    "selection": "all_validation_examples",
                    "environments": list(LOCKED_ENVS),
                    "outputs": outputs,
                    "sbatch_calls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    source_commit = subprocess.check_output(
        ["git", "-C", str(args.code_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.code_root), "status", "--porcelain"], text=True
    )
    if dirty:
        raise HarnessError("held-out materialization requires a clean source tree")
    if not args.data_manifest.is_file():
        raise HarnessError(f"released data manifest is absent: {args.data_manifest}")
    result = {}
    for environment in LOCKED_ENVS:
        datasets = _datasets(environment, args.data_root, spec)
        training = runtime_slice_entries(
            datasets["train"], environment=environment, partition="train"
        )
        validation = runtime_slice_entries(
            datasets["valid"], environment=environment, partition="valid"
        )
        result[environment] = materialize_heldout_manifest(
            environment=environment,
            training_entries=training,
            validation_entries=validation,
            data_manifest_path=args.data_manifest,
            source_commit=source_commit,
            out_path=Path(outputs[environment]),
        )
    print(
        json.dumps(
            {
                "schema": "dino-wm-p3-heldout-materialization-v1",
                "state": "PASS",
                "source_commit": source_commit,
                "selection": "all_validation_examples",
                "manifests": result,
                "sbatch_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "conf" / "study_matrix.yaml",
    )
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    materialize(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HarnessError, P3CompletionError) as exc:
        print(f"P3 COMPLETION CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
