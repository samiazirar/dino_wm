from __future__ import annotations

import json
from pathlib import Path

import lmdb

from tools.precompute_depth import Trajectory, sha256_bytes
from tools.precompute_depth_mapanything import (
    CALIBRATION_FRAMES,
    _expected_wire,
    _shard_plan_document,
    merge_shards,
    plan_trajectory_shards,
    select_training_calibration_keys,
)


def _trajectory(split: str, episode: int, frames: int) -> Trajectory:
    return Trajectory(
        environment="pusht",
        split=split,
        episode=episode,
        frame_count=frames,
        source_path=Path(f"/{split}-{episode}.mp4"),
        source_relpath=f"pusht_noise/{split}/{episode}.mp4",
        source_kind="mp4",
        source_sha256=f"{episode:064x}",
    )


def test_calibration_is_smallest_hash_training_only() -> None:
    trajectories = [
        _trajectory("train", 0, 80),
        _trajectory("train", 1, 80),
        _trajectory("valid", 0, 200),
    ]
    selected = select_training_calibration_keys(trajectories)
    candidates = [
        trajectory.logical_key(frame)
        for trajectory in trajectories
        if trajectory.split == "train"
        for frame in range(trajectory.frame_count)
    ]
    expected = sorted(
        candidates, key=lambda key: (sha256_bytes(key.encode()), key)
    )[:CALIBRATION_FRAMES]
    assert selected == expected
    assert all("/train/" in key for key in selected)


def test_wire_contract_matches_da3_cache_layout() -> None:
    assert _expected_wire() == {
        "physical_key": "<split>/<episode:05d>/<frame:06d>",
        "dtype": "<f2",
        "shape": [224, 224],
        "order": "C",
        "compressor": "zstd",
        "compressor_level": 3,
        "map_size": 1 << 40,
        "normalization": "clip((depth_m-lo)/(hi-lo),0,1)",
        "inverted": False,
    }


def test_shards_are_deterministic_balanced_and_trajectory_atomic() -> None:
    trajectories = [
        _trajectory("train", episode, frames)
        for episode, frames in enumerate([100, 90, 80, 70, 60, 50, 40, 30])
    ]
    first = plan_trajectory_shards(trajectories, 3)
    second = plan_trajectory_shards(list(trajectories), 3)
    assert [[item.trajectory_key for item in shard] for shard in first] == [
        [item.trajectory_key for item in shard] for shard in second
    ]
    assert sorted(item.trajectory_key for shard in first for item in shard) == sorted(
        item.trajectory_key for item in trajectories
    )
    loads = [sum(item.frame_count for item in shard) for shard in first]
    assert max(loads) - min(loads) <= max(item.frame_count for item in trajectories)
    document = _shard_plan_document(trajectories, 3)
    assert document["full_frame_count"] == sum(item.frame_count for item in trajectories)
    assert len(document["plan_sha256"]) == 64


def test_merge_verifies_shards_and_restores_dataset_order(
    tmp_path: Path, monkeypatch
) -> None:
    trajectories = [
        _trajectory("train", episode, frames)
        for episode, frames in enumerate([3, 2, 4, 1])
    ]
    shards = plan_trajectory_shards(trajectories, 2)
    shards_root = tmp_path / "shards"
    producer = {"name": "fake-mapanything", "settings": {"batch_size": 8}}
    calibration = {"lo": 0.0, "hi": 1.0, "keys": [], "keys_sha256": "0" * 64}
    from tools import precompute_depth_mapanything as module

    monkeypatch.setattr(module, "enumerate_environment", lambda root, env: trajectories)
    for index, shard in enumerate(shards):
        cache_dir = shards_root / f"shard-{index:03d}-of-002" / "pusht.lmdb"
        cache_dir.mkdir(parents=True)
        database = lmdb.open(str(cache_dir), map_size=1 << 20, subdir=True)
        records = []
        with database.begin(write=True) as transaction:
            for ordinal, trajectory in enumerate(shard):
                for frame in range(trajectory.frame_count):
                    key = trajectory.physical_key(frame).encode("ascii")
                    transaction.put(key, b"payload:" + key, overwrite=False)
                records.append(
                    {
                        "trajectory_key": trajectory.trajectory_key,
                        "commit_ordinal": ordinal,
                    }
                )
        database.close()
        manifest = {
            "schema": "dinocular-depth-cache-v1",
            "manifest_id": f"shard-{index}",
            "environment": "pusht",
            "trajectory_count": len(shard),
            "frame_count": sum(item.frame_count for item in shard),
            "source_index_sha256": module._source_index_sha256(shard),
            "producer": producer,
            "calibration": calibration,
            "wire_format": _expected_wire(),
            "tool_sha256": "1" * 64,
            "trajectories": records,
            "build_metrics": {"elapsed_seconds": 2.0 + index},
            "data_mdb_sha256": module.sha256_file(cache_dir / "data.mdb"),
            "closed_before_hash": True,
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))
        plan = _shard_plan_document(trajectories, 2)
        (cache_dir.parent / "shard.json").write_text(
            json.dumps(
                {
                    "schema": "dinocular-mapanything-shard-v1",
                    "created_utc": "2026-07-15T00:00:00Z",
                    "plan_sha256": plan["plan_sha256"],
                    **plan["assignments"][index],
                    "manifest_id": manifest["manifest_id"],
                    "data_mdb_sha256": manifest["data_mdb_sha256"],
                }
            )
        )

    output_root = tmp_path / "merged"
    manifest = merge_shards(
        root=tmp_path / "raw",
        shards_root=shards_root,
        output_root=output_root,
        shard_count=2,
        rebuild=False,
    )
    assert manifest["trajectory_count"] == 4
    assert manifest["frame_count"] == 10
    assert manifest["build_metrics"]["generation_gpu_seconds_sum"] == 5.0
    assert [record["trajectory_key"] for record in manifest["trajectories"]] == [
        trajectory.trajectory_key for trajectory in trajectories
    ]
    assert [record["commit_ordinal"] for record in manifest["trajectories"]] == [
        0,
        1,
        2,
        3,
    ]
    database = lmdb.open(
        str(output_root / "pusht.lmdb"), readonly=True, lock=False, subdir=True
    )
    with database.begin(write=False) as transaction:
        for trajectory in trajectories:
            for frame in range(trajectory.frame_count):
                key = trajectory.physical_key(frame).encode("ascii")
                assert transaction.get(key) == b"payload:" + key
    database.close()
