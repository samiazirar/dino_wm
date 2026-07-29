#!/usr/bin/env python3
"""Build a fail-closed qualitative DINOcular results page from campaign outputs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


EVALUATION_LAUNCH_SCHEMA = "dinocular.fixed-evaluation-launch-artifacts.v1"
PLANNING_LAUNCH_SCHEMA = "dinocular.fixed-planning-launch-artifacts.v1"
RESULT_BUNDLE_SCHEMA = "dinocular.combined-result-bundle.v1"
EVALUATION_RESULT_SCHEMA = "dino-wm-open-loop-episode-errors-v1"
PLANNING_RESULT_SCHEMA = "dinocular.planning-target-result.v1"
QUALITATIVE_EVALUATION_SCHEMA = "dinocular.qualitative-evaluation-output.v1"
QUALITATIVE_PLANNING_SCHEMA = "dinocular.qualitative-planning-output.v1"
ENVIRONMENTS = ("pusht", "wall", "rope", "granular")
ARMS = ("dino_pinned", "dinocular", "dinocular_zerodepth")
SEEDS = (1, 2, 3)
PLANNING_TARGET_COUNTS = {"pusht": 50, "wall": 50, "rope": 10, "granular": 10}
DISPLAY_NAMES = {
    "dino_pinned": "DINOv2",
    "dinocular": "DINOcular real depth",
    "dinocular_zerodepth": "Zero-depth DINOcular",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".webm"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BuildError(RuntimeError):
    """An input violates the immutable qualitative-results contract."""


class MissingInput(RuntimeError):
    """A real output needed for a qualitative comparison is not present yet."""


def _snapshot_key(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BuildError(f"{label} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute():
        raise BuildError(f"{label} must be absolute")
    return path


def _require_child(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BuildError(f"{label} must stay under {root}") from exc


def _snapshot_bytes(path: Path, label: str) -> tuple[bytes, str, tuple[int, int, int, int, int]]:
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise MissingInput(f"{label} is absent: {path}") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise BuildError(f"{label} must be a regular file, not an alias: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            content = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise BuildError(f"cannot read {label}: {path}: {exc}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or _snapshot_key(path_stat) != _snapshot_key(before)
        or _snapshot_key(before) != _snapshot_key(after)
        or len(content) != before.st_size
    ):
        raise BuildError(f"{label} changed while being read: {path}")
    return content, hashlib.sha256(content).hexdigest(), _snapshot_key(after)


def _read_object(path: Path, label: str) -> tuple[Mapping[str, Any], str]:
    content, digest, _snapshot = _snapshot_bytes(path, label)
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"{label} is not a JSON object: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BuildError(f"{label} is not a JSON object: {path}")
    return value, digest


def _read_jsonl(path: Path, label: str) -> tuple[list[Mapping[str, Any]], str]:
    content, digest, _snapshot = _snapshot_bytes(path, label)
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            raise BuildError(f"blank JSONL line in {label}: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BuildError(f"invalid JSONL in {label}: {path}:{line_number}") from exc
        if not isinstance(row, Mapping):
            raise BuildError(f"non-object JSONL row in {label}: {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise BuildError(f"{label} is empty: {path}")
    return rows, digest


def _expected_axes() -> set[tuple[str, str, int]]:
    return {
        (environment, arm, seed)
        for environment in ENVIRONMENTS
        for arm in ARMS
        for seed in SEEDS
    }


def _lineage(environment: str, arm: str, seed: int) -> str:
    return f"{environment}/{arm}/s{seed}"


def _load_evaluation_launch(path: Path) -> Mapping[str, Path]:
    launch, _digest = _read_object(path, "evaluation launch manifest")
    records = launch.get("records")
    if (
        launch.get("schema") != EVALUATION_LAUNCH_SCHEMA
        or launch.get("state") != "READY"
        or launch.get("lineage_count") != 36
        or not isinstance(records, Mapping)
        or len(records) != 36
    ):
        raise BuildError("evaluation launch manifest is not the fixed ready 36-lineage campaign")
    result: dict[str, Path] = {}
    for environment, arm, seed in sorted(_expected_axes()):
        lineage = _lineage(environment, arm, seed)
        record = records.get(lineage)
        if not isinstance(record, Mapping):
            raise BuildError(f"evaluation launch record is absent: {lineage}")
        if (
            record.get("environment") != environment
            or record.get("arm") != arm
            or record.get("seed") != seed
        ):
            raise BuildError(f"evaluation launch identity differs: {lineage}")
        run_dir = _require_absolute_path(
            record.get("evaluation_run_dir"), f"evaluation_run_dir for {lineage}"
        )
        result[lineage] = run_dir
    return result


def _load_planning_launch(path: Path) -> Mapping[str, tuple[Path, int]]:
    launch, _digest = _read_object(path, "planning launch manifest")
    outputs = launch.get("expected_outputs")
    if (
        launch.get("schema") != PLANNING_LAUNCH_SCHEMA
        or launch.get("expected_lineage_count") != 36
        or launch.get("expected_result_count") != 1080
        or not isinstance(outputs, Mapping)
        or len(outputs) != 36
    ):
        raise BuildError("planning launch manifest is not the fixed 36-lineage campaign")
    result: dict[str, tuple[Path, int]] = {}
    for environment, arm, seed in sorted(_expected_axes()):
        lineage = _lineage(environment, arm, seed)
        record = outputs.get(lineage)
        if not isinstance(record, Mapping):
            raise BuildError(f"planning launch output is absent: {lineage}")
        path_value = _require_absolute_path(
            record.get("path"), f"planning output path for {lineage}"
        )
        expected_count = record.get("expected_count")
        if expected_count != PLANNING_TARGET_COUNTS[environment]:
            raise BuildError(f"planning target count differs: {lineage}")
        result[lineage] = (path_value, int(expected_count))
    return result


def _asset(record: Any, label: str, root: Path, suffixes: set[str]) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise BuildError(f"{label} must contain exactly path and sha256")
    path = _require_absolute_path(record.get("path"), f"{label} path")
    _require_child(path, root, f"{label} path")
    suffix = path.suffix.lower()
    if suffix not in suffixes:
        raise BuildError(f"{label} has unsupported media type: {path}")
    claimed_digest = record.get("sha256")
    if not isinstance(claimed_digest, str) or not SHA256_RE.fullmatch(claimed_digest):
        raise BuildError(f"{label} has an invalid sha256")
    return {"path": path, "sha256": claimed_digest, "suffix": suffix}


def _check_asset(asset: Mapping[str, Any], label: str) -> None:
    _content, digest, _snapshot = _snapshot_bytes(Path(asset["path"]), label)
    if digest != asset["sha256"]:
        raise BuildError(f"{label} hash differs from its qualitative output record")


def _evaluation_rows(
    path: Path, environment: str, arm: str, seed: int
) -> tuple[set[str], str]:
    rows, digest = _read_jsonl(path, f"evaluation output for {_lineage(environment, arm, seed)}")
    keys: set[str] = set()
    for row in rows:
        if (
            row.get("schema") != EVALUATION_RESULT_SCHEMA
            or row.get("environment") != environment
            or row.get("arm") != arm
            or row.get("seed") != seed
        ):
            raise BuildError(f"evaluation output identity differs: {_lineage(environment, arm, seed)}")
        row_keys = row.get("manifest_keys")
        if not isinstance(row_keys, list) or not row_keys:
            raise BuildError(f"evaluation output lacks manifest keys: {_lineage(environment, arm, seed)}")
        for key in row_keys:
            if not isinstance(key, str) or not key or key in keys:
                raise BuildError(f"evaluation output has duplicate manifest coverage: {_lineage(environment, arm, seed)}")
            keys.add(key)
    return keys, digest


def _planning_rows(
    path: Path, environment: str, arm: str, seed: int, expected_count: int
) -> tuple[Mapping[str, Mapping[str, Any]], str]:
    rows, digest = _read_jsonl(path, f"planning output for {_lineage(environment, arm, seed)}")
    if len(rows) != expected_count:
        raise BuildError(f"planning output count differs: {_lineage(environment, arm, seed)}")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        target_id = row.get("target_id")
        if (
            row.get("schema") != PLANNING_RESULT_SCHEMA
            or row.get("environment") != environment
            or row.get("arm") != arm
            or row.get("seed") != seed
            or not isinstance(target_id, str)
            or not target_id
            or target_id in indexed
            or not isinstance(row.get("success"), bool)
        ):
            raise BuildError(f"planning output identity or outcome differs: {_lineage(environment, arm, seed)}")
        indexed[target_id] = row
    return indexed, digest


def _evaluation_qualitative(
    run_dir: Path, environment: str, arm: str, seed: int
) -> Mapping[str, Any]:
    source_path = run_dir / "episode_errors.jsonl"
    output_path = run_dir / "qualitative" / "qualitative_evaluation.json"
    record, _digest = _read_object(output_path, f"qualitative evaluation output for {_lineage(environment, arm, seed)}")
    required = {
        "schema",
        "environment",
        "arm",
        "seed",
        "manifest_key",
        "evaluation_output",
        "dataset_example",
        "goal_frame",
        "ground_truth_rollout",
        "imagined_rollout",
    }
    if set(record) != required or record.get("schema") != QUALITATIVE_EVALUATION_SCHEMA:
        raise BuildError(f"qualitative evaluation schema differs: {_lineage(environment, arm, seed)}")
    if (
        record.get("environment") != environment
        or record.get("arm") != arm
        or record.get("seed") != seed
        or not isinstance(record.get("manifest_key"), str)
        or not record["manifest_key"]
    ):
        raise BuildError(f"qualitative evaluation identity differs: {_lineage(environment, arm, seed)}")
    binding = record.get("evaluation_output")
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise BuildError(f"qualitative evaluation source binding differs: {_lineage(environment, arm, seed)}")
    bound_path = _require_absolute_path(binding.get("path"), "qualitative evaluation source")
    if bound_path != source_path:
        raise BuildError(f"qualitative evaluation source path differs: {_lineage(environment, arm, seed)}")
    keys, source_digest = _evaluation_rows(source_path, environment, arm, seed)
    if binding.get("sha256") != source_digest:
        raise BuildError(f"qualitative evaluation source hash differs: {_lineage(environment, arm, seed)}")
    if record["manifest_key"] not in keys:
        raise BuildError(f"qualitative evaluation key is not in its validated output: {_lineage(environment, arm, seed)}")
    assets = {
        "dataset_example": _asset(record["dataset_example"], "dataset example", run_dir, IMAGE_SUFFIXES),
        "goal_frame": _asset(record["goal_frame"], "goal frame", run_dir, IMAGE_SUFFIXES),
        "ground_truth_rollout": _asset(record["ground_truth_rollout"], "ground-truth rollout", run_dir, VIDEO_SUFFIXES),
        "imagined_rollout": _asset(record["imagined_rollout"], "imagined rollout", run_dir, VIDEO_SUFFIXES),
    }
    for name, asset in assets.items():
        _check_asset(asset, f"{name} for {_lineage(environment, arm, seed)}")
    return {
        "manifest_key": record["manifest_key"],
        "source_keys": keys,
        "source_path": source_path,
        "source_sha256": source_digest,
        **assets,
    }


def _planning_qualitative(
    result_path: Path, expected_count: int, environment: str, arm: str, seed: int
) -> Mapping[str, Any]:
    output_path = result_path.parent / "qualitative" / "qualitative_planning.json"
    record, _digest = _read_object(output_path, f"qualitative planning output for {_lineage(environment, arm, seed)}")
    required = {
        "schema",
        "environment",
        "arm",
        "seed",
        "target_id",
        "planning_output",
        "rollout",
    }
    if set(record) != required or record.get("schema") != QUALITATIVE_PLANNING_SCHEMA:
        raise BuildError(f"qualitative planning schema differs: {_lineage(environment, arm, seed)}")
    if (
        record.get("environment") != environment
        or record.get("arm") != arm
        or record.get("seed") != seed
        or not isinstance(record.get("target_id"), str)
        or not record["target_id"]
    ):
        raise BuildError(f"qualitative planning identity differs: {_lineage(environment, arm, seed)}")
    binding = record.get("planning_output")
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise BuildError(f"qualitative planning source binding differs: {_lineage(environment, arm, seed)}")
    bound_path = _require_absolute_path(binding.get("path"), "qualitative planning source")
    if bound_path != result_path:
        raise BuildError(f"qualitative planning source path differs: {_lineage(environment, arm, seed)}")
    rows, source_digest = _planning_rows(result_path, environment, arm, seed, expected_count)
    if binding.get("sha256") != source_digest:
        raise BuildError(f"qualitative planning source hash differs: {_lineage(environment, arm, seed)}")
    target_id = record["target_id"]
    if target_id not in rows:
        raise BuildError(f"qualitative planning target is not in its validated output: {_lineage(environment, arm, seed)}")
    rollout = _asset(record["rollout"], "planning rollout", result_path.parent, VIDEO_SUFFIXES)
    _check_asset(rollout, f"planning rollout for {_lineage(environment, arm, seed)}")
    return {
        "target_id": target_id,
        "source_targets": rows,
        "source_path": result_path,
        "source_sha256": source_digest,
        "outcome": rows[target_id]["success"],
        "rollout": rollout,
    }


def _candidate(
    evaluation: Mapping[str, Path],
    planning: Mapping[str, tuple[Path, int]],
    environment: str,
    seed: int,
) -> Mapping[str, Any]:
    if environment not in ("pusht", "wall"):
        raise BuildError("qualitative planning success/failure is defined only for PushT and Wall")
    evaluations = {}
    planning_outputs = {}
    for arm in ARMS:
        lineage = _lineage(environment, arm, seed)
        evaluations[arm] = _evaluation_qualitative(evaluation[lineage], environment, arm, seed)
        result_path, expected_count = planning[lineage]
        planning_outputs[arm] = _planning_qualitative(
            result_path, expected_count, environment, arm, seed
        )

    shared_keys = set.intersection(*(set(value["source_keys"]) for value in evaluations.values()))
    if not shared_keys or any(set(value["source_keys"]) != shared_keys for value in evaluations.values()):
        raise BuildError(f"evaluation coverage is not exactly matched: {environment}/s{seed}")
    expected_manifest_key = min(shared_keys)
    if any(value["manifest_key"] != expected_manifest_key for value in evaluations.values()):
        raise BuildError(f"qualitative evaluation selection is not the fixed first shared key: {environment}/s{seed}")

    for field in ("dataset_example", "goal_frame", "ground_truth_rollout"):
        hashes = {str(value[field]["sha256"]) for value in evaluations.values()}
        if len(hashes) != 1:
            raise BuildError(f"qualitative {field} differs across the matched triplet: {environment}/s{seed}")

    shared_targets = set.intersection(*(set(value["source_targets"]) for value in planning_outputs.values()))
    if not shared_targets or any(set(value["source_targets"]) != shared_targets for value in planning_outputs.values()):
        raise BuildError(f"planning target coverage is not exactly matched: {environment}/s{seed}")
    expected_target_id = min(shared_targets)
    if any(value["target_id"] != expected_target_id for value in planning_outputs.values()):
        raise BuildError(f"qualitative planning selection is not the fixed first shared target: {environment}/s{seed}")
    return {
        "environment": environment,
        "seed": seed,
        "manifest_key": expected_manifest_key,
        "target_id": expected_target_id,
        "evaluations": evaluations,
        "planning": planning_outputs,
    }


def _bundle(path: Path) -> tuple[Mapping[str, Any], str] | None:
    try:
        record, digest = _read_object(path, "combined result bundle manifest")
    except MissingInput:
        return None
    if record.get("schema") != RESULT_BUNDLE_SCHEMA or record.get("state") != "PASS":
        raise BuildError("combined result bundle is not a passing immutable result bundle")
    return record, digest


def _validate_bundle_bindings(
    bundle: Mapping[str, Any],
    evaluation_launch_manifest: Path,
    planning_launch_manifest: Path,
    candidate: Mapping[str, Any],
) -> None:
    expected_manifests = (
        ("launch_manifest", evaluation_launch_manifest),
        ("planning_launch_manifest", planning_launch_manifest),
    )
    for field, path in expected_manifests:
        binding = bundle.get(field)
        _content, digest, _snapshot = _snapshot_bytes(path, field.replace("_", " "))
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != str(path)
            or binding.get("sha256") != digest
        ):
            raise BuildError(f"combined result bundle does not bind this {field.replace('_', ' ')}")
    inputs = bundle.get("inputs")
    planning_inputs = bundle.get("planning_inputs")
    if not isinstance(inputs, Mapping) or not isinstance(planning_inputs, Mapping):
        raise BuildError("combined result bundle lacks evaluation or planning input bindings")
    for arm in ARMS:
        lineage = _lineage(candidate["environment"], arm, candidate["seed"])
        evaluation = candidate["evaluations"][arm]
        planning = candidate["planning"][arm]
        for collection, source, label in (
            (inputs, evaluation, "evaluation"),
            (planning_inputs, planning, "planning"),
        ):
            binding = collection.get(lineage)
            if (
                not isinstance(binding, Mapping)
                or binding.get("path") != str(source["source_path"])
                or binding.get("sha256") != source["source_sha256"]
            ):
                raise BuildError(f"combined result bundle does not bind selected {label} output: {lineage}")


def _safe_paper_url(value: str | None) -> str:
    if not isinstance(value, str) or not value:
        raise BuildError("--paper-url is required before a results page can be published")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise BuildError("--paper-url must be an https URL")
    return value


def _copy_asset(asset: Mapping[str, Any], directory: Path) -> str:
    source = Path(asset["path"])
    target_name = f"{asset['sha256']}{asset['suffix']}"
    target = directory / target_name
    if target.exists():
        return f"media/{target_name}"
    try:
        source_stat = source.lstat()
    except OSError as exc:
        raise BuildError(f"media source disappeared before copy: {source}") from exc
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise BuildError(f"media source is no longer a regular file: {source}")
    digest = hashlib.sha256()
    try:
        source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        target_descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(source_descriptor, "rb") as input_handle, os.fdopen(target_descriptor, "wb") as output_handle:
            before = os.fstat(input_handle.fileno())
            for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                digest.update(block)
                output_handle.write(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
            after = os.fstat(input_handle.fileno())
    except OSError as exc:
        raise BuildError(f"cannot copy media source {source}: {exc}") from exc
    if (
        _snapshot_key(source_stat) != _snapshot_key(before)
        or _snapshot_key(before) != _snapshot_key(after)
        or digest.hexdigest() != asset["sha256"]
    ):
        raise BuildError(f"media source changed before copy: {source}")
    return f"media/{target_name}"


def _media_tag(url: str, label: str, video: bool) -> str:
    escaped_url = html.escape(url, quote=True)
    escaped_label = html.escape(label, quote=True)
    if video:
        return f'<video controls preload="metadata" src="{escaped_url}" aria-label="{escaped_label}"></video>'
    return f'<img src="{escaped_url}" alt="{escaped_label}">'


def _render_page(candidate: Mapping[str, Any], urls: Mapping[str, Mapping[str, str]], paper_url: str) -> str:
    task = str(candidate["environment"]).title()
    seed = candidate["seed"]
    shared = urls[ARMS[0]]
    cards = []
    for arm in ARMS:
        evaluation = candidate["evaluations"][arm]
        planning = candidate["planning"][arm]
        arm_urls = urls[arm]
        outcome = "Success" if planning["outcome"] else "Failure"
        cards.append(
            "\n".join(
                (
                    '<article class="system">',
                    f"<h3>{html.escape(DISPLAY_NAMES[arm])}</h3>",
                    '<div class="row"><span>Ground truth</span>'
                    + _media_tag(arm_urls["ground_truth_rollout"], "Ground-truth rollout", True)
                    + "</div>",
                    '<div class="row"><span>Imagined rollout</span>'
                    + _media_tag(arm_urls["imagined_rollout"], "Imagined rollout", True)
                    + "</div>",
                    f'<div class="row planning"><span>Planning: {outcome}</span>'
                    + _media_tag(arm_urls["planning_rollout"], f"Planning {outcome.lower()} rollout", True)
                    + "</div>",
                    "</article>",
                )
            )
        )
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DINOcular qualitative results</title>
<style>
body { background: #f8f8f8; color: #202124; font-family: Arial, sans-serif; margin: 0; }
main { margin: auto; max-width: 1240px; padding: 28px 20px 48px; }
h1, h2, h3 { margin: 0 0 10px; } p { line-height: 1.45; }
a { color: #0759a5; } .links { display: flex; gap: 16px; margin: 14px 0 28px; }
.reference, .systems { display: grid; gap: 16px; }
.reference { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-bottom: 22px; }
.systems { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.reference figure, .system { background: #fff; border: 1px solid #ddd; border-radius: 8px; margin: 0; padding: 12px; }
figcaption { font-weight: 600; margin-bottom: 8px; } img, video { background: #111; display: block; max-width: 100%%; width: 100%%; }
.row { border-top: 1px solid #e5e5e5; margin-top: 12px; padding-top: 12px; }
.row span { display: block; font-size: .9rem; font-weight: 600; margin-bottom: 6px; }
@media (max-width: 850px) { .systems { grid-template-columns: 1fr; } }
@media (max-width: 560px) { .reference { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
<h1>DINOcular qualitative results</h1>
<p>Matched %s comparison, seed %s. The same fixed evaluation example and fixed planning target are shown for DINOv2, DINOcular real depth, and zero-depth DINOcular.</p>
<nav class="links"><a href="%s">Empirical paper</a><a href="result_bundle_manifest.json">Result bundle manifest</a></nav>
<section class="reference">
<figure><figcaption>Dataset example</figcaption>%s</figure>
<figure><figcaption>Goal frame</figcaption>%s</figure>
</section>
<section class="systems">
%s
</section>
</main>
</body>
</html>
""" % (
        html.escape(task),
        seed,
        html.escape(paper_url, quote=True),
        _media_tag(shared["dataset_example"], "Dataset example", False),
        _media_tag(shared["goal_frame"], "Goal frame", False),
        "\n".join(cards),
    )


def _build(
    candidate: Mapping[str, Any],
    bundle_path: Path,
    expected_bundle_digest: str,
    paper_url: str,
    out_dir: Path,
) -> Mapping[str, Any]:
    if out_dir.exists():
        raise BuildError(f"immutable results-page output already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        media_dir = stage / "media"
        media_dir.mkdir()
        urls: dict[str, Mapping[str, str]] = {}
        for arm in ARMS:
            evaluation = candidate["evaluations"][arm]
            planning = candidate["planning"][arm]
            urls[arm] = {
                "dataset_example": _copy_asset(evaluation["dataset_example"], media_dir),
                "goal_frame": _copy_asset(evaluation["goal_frame"], media_dir),
                "ground_truth_rollout": _copy_asset(evaluation["ground_truth_rollout"], media_dir),
                "imagined_rollout": _copy_asset(evaluation["imagined_rollout"], media_dir),
                "planning_rollout": _copy_asset(planning["rollout"], media_dir),
            }
        bundle_content, current_bundle_digest, _snapshot = _snapshot_bytes(bundle_path, "combined result bundle manifest")
        if not bundle_content or current_bundle_digest != expected_bundle_digest:
            raise BuildError("combined result bundle changed before page creation")
        (stage / "result_bundle_manifest.json").write_bytes(bundle_content)
        (stage / "index.html").write_text(_render_page(candidate, urls, paper_url), encoding="utf-8")
        os.replace(stage, out_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "status": "RESULTS_PAGE_READY",
        "output_directory": str(out_dir),
        "task": candidate["environment"],
        "seed": candidate["seed"],
        "manifest_key": candidate["manifest_key"],
        "target_id": candidate["target_id"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-launch-manifest", type=Path, required=True)
    parser.add_argument("--planning-launch-manifest", type=Path, required=True)
    parser.add_argument("--result-bundle-manifest", type=Path, required=True)
    parser.add_argument("--paper-url")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation = _load_evaluation_launch(args.evaluation_launch_manifest)
    planning = _load_planning_launch(args.planning_launch_manifest)
    candidates = []
    waiting: list[str] = []
    for environment in ("pusht", "wall"):
        for seed in SEEDS:
            try:
                candidates.append(_candidate(evaluation, planning, environment, seed))
            except MissingInput as exc:
                waiting.append(f"{environment}/s{seed}: {exc}")
    bundle = _bundle(args.result_bundle_manifest)
    if not candidates or bundle is None:
        receipt = {
            "status": "WAITING_FOR_MATCHED_QUALITATIVE_TRIPLET",
            "complete_triplet_count": len(candidates),
            "result_bundle": {
                "state": "READY" if bundle is not None else "MISSING",
                "path": str(args.result_bundle_manifest),
            },
            "waiting_on": waiting,
        }
        print(json.dumps(receipt, sort_keys=True))
        return 0
    candidate = candidates[0]
    _validate_bundle_bindings(
        bundle[0],
        args.evaluation_launch_manifest,
        args.planning_launch_manifest,
        candidate,
    )
    paper_url = _safe_paper_url(args.paper_url)
    result = _build(
        candidate,
        args.result_bundle_manifest,
        bundle[1],
        paper_url,
        args.out_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(json.dumps({"status": "FAILED", "reason": str(exc)}, sort_keys=True))
        raise SystemExit(2)
