#!/usr/bin/env python3
"""Plan, build, and merge deterministic whole-trajectory DA3 PushT shards.

This tool never submits scheduler jobs. It reuses the accepted DA3 producer and
cache writer, keeps every trajectory intact, and refuses to replace an existing
canonical cache. The external repository validator remains the only promotion
gate after merge.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import uuid
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.precompute_depth import (  # noqa: E402
    MAP_SIZE,
    OUTPUT_SHAPE,
    ZSTD_LEVEL,
    ContractError,
    OfficialDA3StreamingProducer,
    Trajectory,
    _jsonable,
    _require_lmdb_zstd,
    _source_index_hash,
    atomic_write_json,
    build_environment_cache,
    canonical_json_bytes,
    enumerate_environment,
    generate_environment_calibration,
    load_calibration_manifest,
    producer_identity,
    sha256_bytes,
    sha256_file,
    select_calibration_keys,
    utc_now,
)


SCHEMA = "dinocular-da3-pusht-shard-plan-v1"
SHARD_SCHEMA = "dinocular-da3-pusht-shard-v1"
MERGE_SCHEMA = "dinocular-da3-pusht-merge-v1"
SHARD_COUNT = 8
EXPECTED_TRAJECTORIES = 18_706
EXPECTED_FRAMES = 2_339_250
EXPECTED_MIN_LOAD = 292_391
EXPECTED_MAX_LOAD = 292_440
CALIBRATION_PATH = Path("/workspace/data/depth_cache_calibration/pusht.json")
SHARDS_ROOT = Path("/workspace/data/depth_cache_da3_pusht_shards")
CANONICAL_CACHE_ROOT = Path("/workspace/data/depth_cache")


def _canonical_trajectory_index(trajectories: Sequence[Trajectory]) -> list[dict[str, Any]]:
    return [
        {
            "trajectory_key": item.trajectory_key,
            "frame_count": item.frame_count,
            "source_path": item.source_relpath,
            "source_sha256": item.source_sha256,
        }
        for item in trajectories
    ]


def plan_trajectory_shards(
    trajectories: Sequence[Trajectory], shard_count: int = SHARD_COUNT
) -> list[list[Trajectory]]:
    """Apply the accepted stable LPT whole-trajectory assignment."""

    if shard_count != SHARD_COUNT:
        raise ContractError(f"DA3 PushT reroute requires exactly {SHARD_COUNT} shards")
    if len({item.trajectory_key for item in trajectories}) != len(trajectories):
        raise ContractError("duplicate PushT trajectory key")
    original_order = {
        item.trajectory_key: ordinal for ordinal, item in enumerate(trajectories)
    }
    ordered = sorted(
        trajectories,
        key=lambda item: (
            -item.frame_count,
            sha256_bytes(item.trajectory_key.encode("ascii")),
            item.trajectory_key,
        ),
    )
    shards: list[list[Trajectory]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for trajectory in ordered:
        shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[shard_index].append(trajectory)
        loads[shard_index] += trajectory.frame_count
    for shard in shards:
        shard.sort(key=lambda item: original_order[item.trajectory_key])
    return shards


def _load_calibration_source(
    path: Path, trajectories: Sequence[Trajectory]
) -> tuple[Mapping[str, Any], str]:
    if path != CALIBRATION_PATH:
        raise ContractError(f"sole calibration source must be {CALIBRATION_PATH}")
    if not path.is_file():
        raise ContractError(f"missing PushT calibration manifest: {path}")
    digest = sha256_file(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid PushT calibration manifest {path}: {exc}") from exc
    calibration = manifest.get("calibration")
    expected_keys = select_calibration_keys(
        {"pusht": trajectories}, environments=["pusht"]
    )
    calibration_keys = calibration.get("keys") if isinstance(calibration, Mapping) else None
    lo = calibration.get("lo") if isinstance(calibration, Mapping) else None
    hi = calibration.get("hi") if isinstance(calibration, Mapping) else None
    if (
        manifest.get("schema") != "dinocular-depth-calibration-v1"
        or manifest.get("environment") != "pusht"
        or manifest.get("source_index_sha256") != _source_index_hash(trajectories)
        or not isinstance(manifest.get("producer"), Mapping)
        or not isinstance(calibration, Mapping)
        or calibration.get("scope") != "environment_training_only"
        or calibration.get("environment") != "pusht"
        or calibration.get("selected_environments") != ["pusht"]
        or calibration.get("selection")
        != "128_smallest_sha256_training_keys_for_environment"
        or calibration.get("per_environment") != 128
        or calibration_keys != expected_keys
        or calibration.get("keys_sha256")
        != sha256_bytes(canonical_json_bytes(expected_keys))
        or calibration.get("sample") != "cropped_metric_depth[::8,::8]"
        or calibration.get("sample_stride") != 8
        or calibration.get("percentiles") != [2.0, 98.0]
        or calibration.get("percentile_method") != "numpy.linear"
        or not isinstance(lo, (int, float))
        or not isinstance(hi, (int, float))
        or not hi > lo + 1e-6
    ):
        raise ContractError("PushT calibration source violates the immutable contract")
    return manifest, digest


def create_calibration(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.json != CALIBRATION_PATH:
        raise ContractError(f"calibration output must be {CALIBRATION_PATH}")
    if args.json.exists():
        raise ContractError(f"calibration output already exists: {args.json}")
    trajectories = enumerate_environment(args.root, "pusht")
    producer = _producer(args, args.json.parent / ".da3-work")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pusht-calibration-", dir=args.json.parent
    ) as staging:
        calibration, _raw = generate_environment_calibration(
            environment="pusht",
            trajectories_by_env={"pusht": trajectories},
            producer=producer,
            staging_root=Path(staging),
        )
    document = {
        "schema": "dinocular-depth-calibration-v1",
        "created_utc": utc_now(),
        "environment": "pusht",
        "source_index_sha256": _source_index_hash(trajectories),
        "producer": _jsonable(producer.provenance),
        "calibration": calibration,
    }
    atomic_write_json(args.json, document)
    return document


def make_plan(
    *, trajectories: Sequence[Trajectory], calibration_manifest: Path
) -> Mapping[str, Any]:
    source, calibration_sha256 = _load_calibration_source(
        calibration_manifest, trajectories
    )
    if (
        len(trajectories) != EXPECTED_TRAJECTORIES
        or sum(item.frame_count for item in trajectories) != EXPECTED_FRAMES
    ):
        raise ContractError("PushT inventory differs from 18,706 trajectories / 2,339,250 frames")
    shards = plan_trajectory_shards(trajectories)
    assignments = []
    for index, shard in enumerate(shards):
        assignments.append(
            {
                "shard_index": index,
                "root": str(SHARDS_ROOT / f"shard-{index:03d}-of-{SHARD_COUNT:03d}"),
                "trajectory_count": len(shard),
                "frame_count": sum(item.frame_count for item in shard),
                "source_index_sha256": _source_index_hash(shard),
                "trajectory_keys": [item.trajectory_key for item in shard],
            }
        )
    loads = [item["frame_count"] for item in assignments]
    if min(loads) != EXPECTED_MIN_LOAD or max(loads) != EXPECTED_MAX_LOAD:
        raise ContractError(
            f"stable LPT load range differs: {min(loads)}..{max(loads)}"
        )
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "shard_count": SHARD_COUNT,
        "assignment_policy": (
            "stable_lpt_descending_frame_count_then_sha256_trajectory_key_then_key_"
            "least_loaded_shard_then_lowest_index"
        ),
        "full_trajectory_count": len(trajectories),
        "full_frame_count": sum(item.frame_count for item in trajectories),
        "full_source_index_sha256": _source_index_hash(trajectories),
        "trajectory_index_sha256": sha256_bytes(
            canonical_json_bytes(_canonical_trajectory_index(trajectories))
        ),
        "calibration_manifest": str(calibration_manifest),
        "calibration_manifest_sha256": calibration_sha256,
        "calibration_object_sha256": sha256_bytes(
            canonical_json_bytes(source["calibration"])
        ),
        "assignments": assignments,
    }
    document["plan_sha256"] = sha256_bytes(canonical_json_bytes(document))
    return document


def _load_and_verify_plan(path: Path, trajectories: Sequence[Trajectory]) -> Mapping[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing immutable DA3 shard plan: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = make_plan(
        trajectories=trajectories,
        calibration_manifest=Path(str(value.get("calibration_manifest"))),
    )
    if value != expected:
        raise ContractError("DA3 shard plan bytes or current dataset assignment differ")
    return value


def _producer(args: argparse.Namespace, work_root: Path) -> OfficialDA3StreamingProducer:
    return OfficialDA3StreamingProducer(
        args.da3_root, args.model_dir, work_root=work_root
    )


def build_shard(args: argparse.Namespace) -> Mapping[str, Any]:
    trajectories = enumerate_environment(args.root, "pusht")
    plan = _load_and_verify_plan(args.plan, trajectories)
    index = args.shard_index
    if not 0 <= index < SHARD_COUNT:
        raise ContractError("shard index must be in 0..7")
    assignment = plan["assignments"][index]
    shard_root = Path(str(assignment["root"]))
    expected_root = SHARDS_ROOT / f"shard-{index:03d}-of-{SHARD_COUNT:03d}"
    if shard_root != expected_root or args.shards_root != SHARDS_ROOT:
        raise ContractError("shard root differs from the locked distinct-root contract")
    if shard_root.exists():
        raise ContractError(f"shard root must be clean before build: {shard_root}")
    by_key = {item.trajectory_key: item for item in trajectories}
    selected = [by_key[key] for key in assignment["trajectory_keys"]]
    producer = _producer(args, shard_root / ".da3-work")
    calibration = load_calibration_manifest(CALIBRATION_PATH, producer, "pusht")
    source, source_sha256 = _load_calibration_source(CALIBRATION_PATH, trajectories)
    if (
        calibration != source["calibration"]
        or source_sha256 != plan["calibration_manifest_sha256"]
        or sha256_bytes(canonical_json_bytes(calibration))
        != plan["calibration_object_sha256"]
    ):
        raise ContractError("shard calibration differs from the immutable plan")
    manifest = build_environment_cache(
        output_root=shard_root,
        environment="pusht",
        trajectories=selected,
        calibration=calibration,
        producer=producer,
        tool_sha256=sha256_file(Path(__file__).with_name("precompute_depth.py")),
    )
    from tools.validate_depth_cache import validate_environment_cache

    validation = validate_environment_cache(
        root=args.root,
        cache_root=shard_root,
        environment="pusht",
        producer=producer,
        recompute_fraction=0.01,
        trajectories=selected,
    )
    if validation.get("state") != "PASS":
        raise ContractError(f"shard {index} validation is not PASS")
    validation_path = shard_root / "validation.json"
    atomic_write_json(
        validation_path,
        {
            "schema": "dinocular-depth-cache-validation-v1",
            "created_utc": utc_now(),
            "state": "PASS",
            "scope": "selected_whole_trajectory_shard",
            "plan_sha256": plan["plan_sha256"],
            "shard_index": index,
            "results": {"pusht": validation},
        },
    )
    shard_record = {
        "schema": SHARD_SCHEMA,
        "plan_path": str(args.plan),
        "plan_sha256": plan["plan_sha256"],
        **copy.deepcopy(assignment),
        "calibration_manifest_sha256": source_sha256,
        "calibration_object_sha256": plan["calibration_object_sha256"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": sha256_file(shard_root / "pusht.lmdb" / "manifest.json"),
        "data_mdb_sha256": manifest["data_mdb_sha256"],
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
    }
    atomic_write_json(shard_root / "shard.json", shard_record)
    return shard_record


def _expected_wire() -> Mapping[str, Any]:
    return {
        "physical_key": "<split>/<episode:05d>/<frame:06d>",
        "dtype": "<f2",
        "shape": list(OUTPUT_SHAPE),
        "order": "C",
        "compressor": "zstd",
        "compressor_level": ZSTD_LEVEL,
        "map_size": MAP_SIZE,
        "normalization": "clip((depth_m-lo)/(hi-lo),0,1)",
        "inverted": False,
    }


def validate_shard_record_for_merge(
    *,
    shard: Mapping[str, Any],
    assignment: Mapping[str, Any],
    root: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    validation_path: Path,
    validation_sha256: str,
) -> None:
    expected = {
        "schema": SHARD_SCHEMA,
        "plan_path": str(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "shard_index": assignment["shard_index"],
        "root": str(root),
        "trajectory_count": assignment["trajectory_count"],
        "frame_count": assignment["frame_count"],
        "source_index_sha256": assignment["source_index_sha256"],
        "trajectory_keys": assignment["trajectory_keys"],
        "calibration_manifest_sha256": plan["calibration_manifest_sha256"],
        "calibration_object_sha256": plan["calibration_object_sha256"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha256,
        "data_mdb_sha256": manifest["data_mdb_sha256"],
        "validation_path": str(validation_path),
        "validation_sha256": validation_sha256,
    }
    differing = sorted(key for key, value in expected.items() if shard.get(key) != value)
    if differing or set(shard) != set(expected):
        raise ContractError(
            f"shard {assignment['shard_index']} record differs from plan/evidence: {differing}"
        )


def merge_shards(args: argparse.Namespace) -> Mapping[str, Any]:
    trajectories = enumerate_environment(args.root, "pusht")
    plan = _load_and_verify_plan(args.plan, trajectories)
    if args.shards_root != SHARDS_ROOT or args.out != CANONICAL_CACHE_ROOT:
        raise ContractError("merge roots differ from the locked canonical contract")
    destination = args.out / "pusht.lmdb"
    if destination.exists():
        raise ContractError(f"canonical PushT cache already exists: {destination}")
    lmdb, _zstandard = _require_lmdb_zstd()
    manifests = []
    databases = []
    common_producer = common_calibration = common_tool = None
    try:
        for assignment in plan["assignments"]:
            index = assignment["shard_index"]
            root = SHARDS_ROOT / f"shard-{index:03d}-of-{SHARD_COUNT:03d}"
            cache = root / "pusht.lmdb"
            manifest_path = cache / "manifest.json"
            shard_path = root / "shard.json"
            validation_path = root / "validation.json"
            if (
                not manifest_path.is_file()
                or not shard_path.is_file()
                or not validation_path.is_file()
            ):
                raise ContractError(f"missing closed shard evidence for shard {index}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            records = manifest.get("trajectories")
            manifest_sha256 = sha256_file(manifest_path)
            validation_sha256 = sha256_file(validation_path)
            validate_shard_record_for_merge(
                shard=shard,
                assignment=assignment,
                root=root,
                plan=plan,
                plan_path=args.plan,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                validation_path=validation_path,
                validation_sha256=validation_sha256,
            )
            if (
                validation.get("schema") != "dinocular-depth-cache-validation-v1"
                or validation.get("state") != "PASS"
                or validation.get("scope") != "selected_whole_trajectory_shard"
                or validation.get("plan_sha256") != plan["plan_sha256"]
                or validation.get("shard_index") != index
                or validation.get("results", {}).get("pusht", {}).get("state") != "PASS"
                or manifest.get("schema") != "dinocular-depth-cache-v1"
                or manifest.get("environment") != "pusht"
                or manifest.get("wire_format") != _expected_wire()
                or manifest.get("trajectory_count") != assignment["trajectory_count"]
                or manifest.get("frame_count") != assignment["frame_count"]
                or manifest.get("source_index_sha256") != assignment["source_index_sha256"]
                or [item.get("trajectory_key") for item in records or []]
                != assignment["trajectory_keys"]
                or manifest.get("closed_before_hash") is not True
                or not (cache / "data.mdb").is_file()
                or sha256_file(cache / "data.mdb") != manifest.get("data_mdb_sha256")
            ):
                raise ContractError(f"shard {index} fails immutable merge checks")
            identity = producer_identity(manifest.get("producer", {}))
            calibration = manifest.get("calibration")
            tool = manifest.get("tool_sha256")
            if common_producer is None:
                common_producer, common_calibration, common_tool = identity, calibration, tool
            elif (
                identity != common_producer
                or calibration != common_calibration
                or tool != common_tool
            ):
                raise ContractError("producer, calibration, or tool differs across shards")
            if (
                sha256_bytes(canonical_json_bytes(calibration))
                != plan["calibration_object_sha256"]
            ):
                raise ContractError("shard calibration differs from the plan")
            manifests.append((manifest, manifest_sha256, shard))
            databases.append(
                lmdb.open(str(cache), subdir=True, readonly=True, lock=False, readahead=False)
            )

        args.out.mkdir(parents=True, exist_ok=True)
        manifest_id = str(uuid.uuid4())
        building = args.out / f".pusht.lmdb.building-{manifest_id}"
        building.mkdir()
        output = lmdb.open(
            str(building), subdir=True, map_size=MAP_SIZE, readonly=False,
            create=True, lock=True, sync=True, metasync=True, map_async=False,
            writemap=False, readahead=False, meminit=False, max_dbs=1,
        )
        started = time.monotonic()
        records_by_key = {
            item["trajectory_key"]: (index, item)
            for index, (manifest, _manifest_sha, _shard) in enumerate(manifests)
            for item in manifest["trajectories"]
        }
        shard_for_key = {
            key: assignment["shard_index"]
            for assignment in plan["assignments"]
            for key in assignment["trajectory_keys"]
        }
        merged_records = []
        written = 0
        try:
            for ordinal, trajectory in enumerate(trajectories):
                shard_index = shard_for_key[trajectory.trajectory_key]
                source = databases[shard_index]
                with source.begin() as read_txn, output.begin(write=True) as write_txn:
                    for frame in range(trajectory.frame_count):
                        key = trajectory.physical_key(frame).encode("ascii")
                        payload = read_txn.get(key)
                        if payload is None or not write_txn.put(key, payload, overwrite=False):
                            raise ContractError(f"missing or duplicate merged key {key.decode()}")
                        written += 1
                record = copy.deepcopy(records_by_key[trajectory.trajectory_key][1])
                record["commit_ordinal"] = ordinal
                record["source_shard_index"] = shard_index
                merged_records.append(record)
            output.sync(True)
        finally:
            output.close()
        if written != EXPECTED_FRAMES or len(merged_records) != EXPECTED_TRAJECTORIES:
            raise ContractError("merged key or trajectory count differs from the full inventory")
        data_sha256 = sha256_file(building / "data.mdb")
        elapsed = time.monotonic() - started
        full_manifest = {
            "schema": "dinocular-depth-cache-v1",
            "manifest_id": manifest_id,
            "created_utc": utc_now(),
            "environment": "pusht",
            "trajectory_count": EXPECTED_TRAJECTORIES,
            "frame_count": EXPECTED_FRAMES,
            "source_index_sha256": plan["full_source_index_sha256"],
            "producer": manifests[0][0]["producer"],
            "calibration": common_calibration,
            "wire_format": _expected_wire(),
            "tool_sha256": common_tool,
            "trajectories": merged_records,
            "build_metrics": {
                "elapsed_seconds": elapsed,
                "committed_frames_per_second": EXPECTED_FRAMES / elapsed,
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "merge_only": True,
            },
            "data_mdb_sha256": data_sha256,
            "closed_before_hash": True,
            "merge_provenance": {
                "schema": MERGE_SCHEMA,
                "plan_path": str(args.plan),
                "plan_sha256": plan["plan_sha256"],
                "calibration_manifest": plan["calibration_manifest"],
                "calibration_manifest_sha256": plan["calibration_manifest_sha256"],
                "calibration_object_sha256": plan["calibration_object_sha256"],
                "shards": [
                    {
                        "shard_index": index,
                        "manifest_id": manifest["manifest_id"],
                        "manifest_sha256": manifest_sha,
                        "data_mdb_sha256": manifest["data_mdb_sha256"],
                        "validation_sha256": _shard["validation_sha256"],
                        "shard_record_sha256": sha256_file(
                            SHARDS_ROOT / f"shard-{index:03d}-of-{SHARD_COUNT:03d}" / "shard.json"
                        ),
                    }
                    for index, (manifest, manifest_sha, _shard) in enumerate(manifests)
                ],
            },
        }
        atomic_write_json(building / "manifest.json", full_manifest)
        os.replace(building, destination)
        return full_manifest
    finally:
        for database in databases:
            database.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--root", type=Path, required=True)
    calibrate.add_argument("--model-dir", type=Path, required=True)
    calibrate.add_argument("--da3-root", type=Path, required=True)
    calibrate.add_argument("--json", type=Path, default=CALIBRATION_PATH)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--calibration-manifest", type=Path, default=CALIBRATION_PATH)
    plan.add_argument("--json", type=Path, required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--shards-root", type=Path, default=SHARDS_ROOT)
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--shard-index", type=int, required=True)
    build.add_argument("--model-dir", type=Path, required=True)
    build.add_argument("--da3-root", type=Path, required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--root", type=Path, required=True)
    merge.add_argument("--shards-root", type=Path, default=SHARDS_ROOT)
    merge.add_argument("--out", type=Path, default=CANONICAL_CACHE_ROOT)
    merge.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        value = create_calibration(args)
    elif args.command == "plan":
        value = make_plan(
            trajectories=enumerate_environment(args.root, "pusht"),
            calibration_manifest=args.calibration_manifest,
        )
        atomic_write_json(args.json, value)
    elif args.command == "build":
        value = build_shard(args)
    else:
        value = merge_shards(args)
    print(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"DA3 SHARD CONTRACT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
