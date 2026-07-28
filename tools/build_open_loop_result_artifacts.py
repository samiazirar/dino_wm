#!/usr/bin/env python3
"""Build the immutable paper artifacts for the fixed 36-run open-loop campaign."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

LAUNCH_SCHEMA = "dinocular.fixed-evaluation-launch-artifacts.v1"
BUNDLE_SCHEMA = "dinocular.combined-result-bundle.v1"
PLANNING_LAUNCH_SCHEMA = "dinocular.fixed-planning-launch-artifacts.v1"
ENVIRONMENTS = ("pusht", "wall", "rope", "granular")
ARMS = ("dino_pinned", "dinocular", "dinocular_zerodepth")
SEEDS = (1, 2, 3)
HORIZONS = {
    "pusht": (1, 5, 10, 25),
    "wall": (1, 5, 10),
    "rope": (1, 2, 3),
    "granular": (1, 2, 3),
}
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260714


class BuildError(RuntimeError):
    """The launch manifest or generated bundle violates the fixed contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BuildError(f"expected a JSON object at {path}")
    return value


def _discover(manifest_path: Path) -> tuple[Mapping[str, Any], dict[str, Path]]:
    manifest = _load_object(manifest_path)
    records = manifest.get("records")
    if (
        manifest.get("schema") != LAUNCH_SCHEMA
        or manifest.get("state") != "READY"
        or manifest.get("lineage_count") != 36
        or manifest.get("selection_seed") != BOOTSTRAP_SEED
        or not isinstance(records, Mapping)
        or len(records) != 36
    ):
        raise BuildError("launch manifest does not declare the fixed 36-run campaign")

    expected = {
        f"{environment}/{arm}/s{seed}"
        for environment in ENVIRONMENTS
        for arm in ARMS
        for seed in SEEDS
    }
    if set(records) != expected:
        raise BuildError("launch manifest task/system/seed coverage is not exact")

    inputs: dict[str, Path] = {}
    for lineage in sorted(expected):
        record = records[lineage]
        if not isinstance(record, Mapping):
            raise BuildError(f"invalid launch record for {lineage}")
        environment, arm, seed_text = lineage.split("/")
        if (
            record.get("environment") != environment
            or record.get("arm") != arm
            or record.get("seed") != int(seed_text[1:])
        ):
            raise BuildError(f"launch record identity mismatch for {lineage}")
        run_dir = record.get("evaluation_run_dir")
        if not isinstance(run_dir, str) or not Path(run_dir).is_absolute():
            raise BuildError(f"invalid evaluation_run_dir for {lineage}")
        inputs[lineage] = Path(run_dir) / "episode_errors.jsonl"
    if len(set(inputs.values())) != 36:
        raise BuildError("duplicate episode-error output paths in launch manifest")
    return manifest, inputs


def _planning_discover(path: Path) -> tuple[Mapping[str, Any], dict[str, tuple[Path, int]]]:
    manifest = _load_object(path)
    outputs = manifest.get("expected_outputs")
    if (
        manifest.get("schema") != PLANNING_LAUNCH_SCHEMA
        or manifest.get("expected_lineage_count") != 36
        or manifest.get("expected_result_count") != 1080
        or not isinstance(outputs, Mapping)
        or len(outputs) != 36
    ):
        raise BuildError("planning launch manifest is not the fixed 36-lineage campaign")
    result = {}
    for lineage, record in outputs.items():
        if not isinstance(record, Mapping):
            raise BuildError(f"invalid planning output record for {lineage}")
        result[str(lineage)] = (
            Path(str(record["path"])),
            int(record["expected_count"]),
        )
    return manifest, result


def _planning_available(outputs: Mapping[str, tuple[Path, int]]) -> tuple[int, list[str]]:
    available = 0
    missing = []
    for lineage, (path, expected) in sorted(outputs.items()):
        count = 0
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                count = sum(1 for line in handle if line.strip())
        available += min(count, expected)
        if count != expected:
            missing.append(f"{lineage}:{expected - min(count, expected)}")
    return available, missing


def _readiness(
    inputs: Mapping[str, Path],
    planning_outputs: Mapping[str, tuple[Path, int]] | None = None,
) -> Mapping[str, Any]:
    available = sorted(key for key, path in inputs.items() if path.is_file())
    missing = sorted(set(inputs) - set(available))
    planning_available, planning_missing = (
        _planning_available(planning_outputs) if planning_outputs is not None else (0, [])
    )
    planning_expected = (
        sum(value[1] for value in planning_outputs.values())
        if planning_outputs is not None
        else 0
    )
    return {
        "status": (
            "WAITING_FOR_PREDICTION_AND_PLANNING"
            if planning_outputs is not None
            else "WAITING_FOR_36_EVALUATIONS"
        ),
        "prediction_expected_count": 36,
        "prediction_available_count": len(available),
        "prediction_missing_count": len(missing),
        "available_lineages": available,
        "missing_lineages": missing,
        "planning_expected_count": planning_expected,
        "planning_available_count": planning_available,
        "planning_missing_count": planning_expected - planning_available,
        "planning_missing_lineages": planning_missing,
    }


def _pooled_rows(collection: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = collection.get("summaries")
    contrasts = collection.get("paired_contrasts")
    expected_summaries = {
        f"{environment}/{arm}/s{seed}"
        for environment in ENVIRONMENTS
        for arm in ARMS
        for seed in SEEDS
    }
    if (
        collection.get("schema") != "dino-wm-open-loop-collection-v1"
        or collection.get("state") != "PASS"
        or not isinstance(summaries, Mapping)
        or set(summaries) != expected_summaries
        or not isinstance(contrasts, Mapping)
    ):
        raise BuildError("collector output does not match the fixed campaign")
    for value in contrasts.values():
        if (
            not isinstance(value, Mapping)
            or value.get("bootstrap_seed") != BOOTSTRAP_SEED
            or value.get("replicates") != BOOTSTRAP_REPLICATES
            or value.get("resampling") != "crossed_training_seed_and_episode"
        ):
            raise BuildError("collector bootstrap contract mismatch")

    rows: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        for arm in ARMS:
            seed_summaries = [
                summaries[f"{environment}/{arm}/s{seed}"] for seed in SEEDS
            ]
            horizons = set(seed_summaries[0]["horizons"])
            if any(set(value["horizons"]) != horizons for value in seed_summaries):
                raise BuildError(f"collector horizon mismatch for {environment}/{arm}")
            for horizon in sorted(horizons, key=int):
                records = [value["horizons"][horizon] for value in seed_summaries]
                model = sum(float(value["model_squared_error"]) for value in records)
                persistence = sum(
                    float(value["persistence_squared_error"]) for value in records
                )
                elements = sum(int(value["element_count"]) for value in records)
                if (
                    not math.isfinite(model)
                    or not math.isfinite(persistence)
                    or model < 0
                    or persistence <= 1e-12
                    or elements <= 0
                ):
                    raise BuildError(
                        f"nonfinite value or degenerate denominator for "
                        f"{environment}/{arm}/h{horizon}"
                    )
                nre = model / persistence
                if not math.isfinite(nre):
                    raise BuildError(
                        f"nonfinite pooled NRE for {environment}/{arm}/h{horizon}"
                    )
                rows.append(
                    {
                        "environment": environment,
                        "arm": arm,
                        "horizon": int(horizon),
                        "normalized_rollout_error": nre,
                        "model_squared_error": model,
                        "persistence_squared_error": persistence,
                        "element_count": elements,
                    }
                )
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_latex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    by_key = {
        (row["environment"], row["arm"], row["horizon"]): row[
            "normalized_rollout_error"
        ]
        for row in rows
    }
    lines = [
        r"\begin{tabular}{ll" + "r" * max(map(len, HORIZONS.values())) + "}",
        r"\toprule",
        r"Task & System & \multicolumn{4}{c}{Normalized rollout error by horizon} \\",
        r"\midrule",
    ]
    labels = {
        "dino_pinned": "DINO",
        "dinocular": "DINOcular",
        "dinocular_zerodepth": r"DINOcular (zero depth)",
    }
    for environment in ENVIRONMENTS:
        horizons = HORIZONS[environment]
        for arm in ARMS:
            values = " & ".join(
                f"{by_key[(environment, arm, horizon)]:.4f}"
                for horizon in horizons
            )
            padding = " & --" * (4 - len(horizons))
            lines.append(
                f"{environment.title()} & {labels[arm]} & {values}{padding} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_figures(directory: Path, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    colors = ("#767676", "#0072B2", "#D55E00")
    for environment in ENVIRONMENTS:
        task_rows = [row for row in rows if row["environment"] == environment]
        horizons = list(HORIZONS[environment])
        figure, axis = plt.subplots(figsize=(4.6, 2.8))
        for arm, color in zip(ARMS, colors):
            arm_rows = {row["horizon"]: row for row in task_rows if row["arm"] == arm}
            axis.plot(
                horizons,
                [arm_rows[h]["normalized_rollout_error"] for h in horizons],
                marker="o",
                linewidth=1.8,
                label=arm.replace("_", " "),
                color=color,
            )
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set(xlabel="Rollout horizon", ylabel="Normalized rollout error")
        axis.set_xticks(horizons)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=7)
        figure.tight_layout()
        path = directory / f"open_loop_{environment}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        figures.append(path)
    return figures


def _render_planning_figures(
    directory: Path, rows: Sequence[Mapping[str, Any]]
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    for environment in ENVIRONMENTS:
        endpoints = sorted(
            {row["endpoint"] for row in rows if row["environment"] == environment}
        )
        figure, axes = plt.subplots(
            1, len(endpoints), figsize=(4.2 * len(endpoints), 2.8), squeeze=False
        )
        for axis, endpoint in zip(axes[0], endpoints):
            for arm in ARMS:
                values = [
                    float(row["mean"])
                    for row in rows
                    if row["environment"] == environment
                    and row["endpoint"] == endpoint
                    and row["arm"] == arm
                ]
                axis.plot(SEEDS, values, marker="o", label=arm.replace("_", " "))
            axis.set(
                xlabel="Training seed",
                ylabel=endpoint.replace("_", " "),
                title=environment.title(),
            )
            axis.set_xticks(SEEDS)
            axis.grid(axis="y", alpha=0.25)
        axes[0][-1].legend(frameon=False, fontsize=7)
        figure.tight_layout()
        path = directory / f"planning_{environment}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        figures.append(path)
    return figures


def _build(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    inputs: Mapping[str, Path],
    out_dir: Path,
    planning_manifest_path: Path | None = None,
    planning_manifest: Mapping[str, Any] | None = None,
    planning_outputs: Mapping[str, tuple[Path, int]] | None = None,
) -> Mapping[str, Any]:
    try:
        from . import collect_runs
    except ImportError:
        import collect_runs  # type: ignore[no-redef]

    if (
        collect_runs.P4_ARMS != ARMS
        or collect_runs.P4_SEEDS != SEEDS
        or collect_runs.P4_HORIZONS != HORIZONS
        or collect_runs.BOOTSTRAP_REPLICATES != BOOTSTRAP_REPLICATES
        or collect_runs.BOOTSTRAP_SEED != BOOTSTRAP_SEED
    ):
        raise BuildError("local collector differs from the locked artifact contract")
    if out_dir.exists():
        raise BuildError(f"immutable result bundle already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        collection_path = stage / "open_loop_collection.json"
        collector_args = argparse.Namespace(
            inputs=[inputs[key] for key in sorted(inputs)],
            bootstrap=BOOTSTRAP_REPLICATES,
            paired=True,
            seed=BOOTSTRAP_SEED,
            out=collection_path,
        )
        try:
            with redirect_stdout(io.StringIO()):
                collect_runs.open_loop(collector_args)
        except collect_runs.CollectionError as exc:
            raise BuildError(f"collector rejected fixed outputs: {exc}") from exc
        collection = _load_object(collection_path)
        rows = _pooled_rows(collection)
        planning_collection_path = None
        planning_collection = None
        planning_rows: list[dict[str, Any]] = []
        if planning_outputs is not None:
            planning_collection_path = stage / "planning_collection.json"
            planning_args = argparse.Namespace(
                inputs=[planning_outputs[key][0] for key in sorted(planning_outputs)],
                bootstrap=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED,
                out=planning_collection_path,
            )
            try:
                with redirect_stdout(io.StringIO()):
                    collect_runs.planning(planning_args)
            except collect_runs.CollectionError as exc:
                raise BuildError(f"planning collector rejected fixed outputs: {exc}") from exc
            planning_collection = _load_object(planning_collection_path)
            for key, value in sorted(planning_collection["summaries"].items()):
                environment, endpoint, arm, seed = key.split("/")
                planning_rows.append(
                    {
                        "environment": environment,
                        "endpoint": endpoint,
                        "arm": arm,
                        "seed": seed,
                        "target_count": value["target_count"],
                        "mean": value["mean"],
                    }
                )

        table_json = stage / "open_loop_table.json"
        table_csv = stage / "open_loop_table.csv"
        table_tex = stage / "open_loop_table.tex"
        _write_json(
            table_json,
            {
                "schema": "dinocular.open-loop-pooled-table.v1",
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "rows": rows,
            },
        )
        _render_csv(table_csv, rows)
        _render_latex(table_tex, rows)
        figures = _render_figures(stage, rows)
        planning_artifacts: list[Path] = []
        if planning_collection_path is not None:
            planning_json = stage / "planning_table.json"
            planning_csv = stage / "planning_table.csv"
            planning_tex = stage / "planning_table.tex"
            _write_json(
                planning_json,
                {
                    "schema": "dinocular.planning-paper-table.v1",
                    "rows": planning_rows,
                    "paired_contrasts": planning_collection["paired_contrasts"],
                },
            )
            _render_csv(planning_csv, planning_rows)
            planning_tex.write_text(
                "\n".join(
                    [
                        r"\begin{tabular}{llllr}",
                        r"\toprule",
                        r"Task & Endpoint & System & Seed & Mean \\",
                        r"\midrule",
                        *[
                            f"{row['environment'].title()} & "
                            f"{row['endpoint'].replace('_', ' ')} & "
                            f"{row['arm'].replace('_', ' ')} & {row['seed']} & "
                            f"{float(row['mean']):.4f} \\\\"
                            for row in planning_rows
                        ],
                        r"\bottomrule",
                        r"\end{tabular}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            planning_artifacts = [
                planning_collection_path,
                planning_json,
                planning_csv,
                planning_tex,
                *_render_planning_figures(stage, planning_rows),
            ]

        artifacts = [
            collection_path,
            table_json,
            table_csv,
            table_tex,
            *figures,
            *planning_artifacts,
        ]
        bundle_manifest = {
            "schema": BUNDLE_SCHEMA,
            "state": "PASS",
            "launch_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "schema": manifest["schema"],
            },
            "planning_launch_manifest": (
                {
                    "path": str(planning_manifest_path),
                    "sha256": _sha256(planning_manifest_path),
                    "schema": planning_manifest["schema"],
                }
                if planning_manifest_path is not None
                and planning_manifest is not None
                else None
            ),
            "collector": {
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "paired": True,
            },
            "inputs": {
                key: {"path": str(path), "sha256": _sha256(path)}
                for key, path in sorted(inputs.items())
            },
            "planning_inputs": (
                {
                    key: {"path": str(value[0]), "sha256": _sha256(value[0])}
                    for key, value in sorted(planning_outputs.items())
                }
                if planning_outputs is not None
                else {}
            ),
            "artifacts": {
                path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in artifacts
            },
        }
        _write_json(stage / "result_bundle_manifest.json", bundle_manifest)
        os.replace(stage, out_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "status": "RESULT_BUNDLE_READY",
        "output_directory": str(out_dir),
        "artifact_count": len(bundle_manifest["artifacts"]) + 1,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--planning-launch-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, inputs = _discover(args.launch_manifest)
    planning_manifest = None
    planning_outputs = None
    if args.planning_launch_manifest is not None:
        planning_manifest, planning_outputs = _planning_discover(
            args.planning_launch_manifest
        )
    readiness = _readiness(inputs, planning_outputs)
    if (
        readiness["prediction_missing_count"]
        or readiness["planning_missing_count"]
    ):
        print(json.dumps(readiness, sort_keys=True))
        return 0
    result = _build(
        args.launch_manifest,
        manifest,
        inputs,
        args.out_dir,
        args.planning_launch_manifest,
        planning_manifest,
        planning_outputs,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(json.dumps({"status": "FAILED", "reason": str(exc)}, sort_keys=True))
        raise SystemExit(2)
