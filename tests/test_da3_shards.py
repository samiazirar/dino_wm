from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tools.precompute_depth import Trajectory
from tools import precompute_depth_da3_shards as sharding
import pytest


def _trajectory(index: int, frames: int) -> Trajectory:
    return Trajectory(
        environment="pusht",
        split="train",
        episode=index,
        frame_count=frames,
        source_path=Path(f"/synthetic/{index}.mp4"),
        source_relpath=f"pusht/train/{index}.mp4",
        source_kind="mp4",
        source_sha256=f"{index + 1:064x}"[-64:],
    )


def test_stable_lpt_keeps_whole_trajectories() -> None:
    trajectories = [_trajectory(index, 50 + (index % 7)) for index in range(80)]
    first = sharding.plan_trajectory_shards(trajectories)
    second = sharding.plan_trajectory_shards(trajectories)
    assert [[item.trajectory_key for item in shard] for shard in first] == [
        [item.trajectory_key for item in shard] for shard in second
    ]
    keys = [item.trajectory_key for shard in first for item in shard]
    assert len(first) == 8
    assert len(keys) == len(set(keys)) == len(trajectories)
    loads = [sum(item.frame_count for item in shard) for shard in first]
    assert max(loads) - min(loads) <= max(item.frame_count for item in trajectories)


def test_build_shard_writes_and_hashes_validation_receipt(tmp_path, monkeypatch) -> None:
    shards_root = tmp_path / "shards"
    calibration = tmp_path / "wall.lmdb" / "manifest.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}", encoding="utf-8")
    selected = [_trajectory(0, 50), _trajectory(1, 51)]
    assignment = {
        "shard_index": 0,
        "root": str(shards_root / "shard-000-of-008"),
        "trajectory_count": 2,
        "frame_count": 101,
        "source_index_sha256": "1" * 64,
        "trajectory_keys": [item.trajectory_key for item in selected],
    }
    plan = {
        "plan_sha256": "2" * 64,
        "calibration_manifest_sha256": "3" * 64,
        "calibration_object_sha256": "4" * 64,
        "assignments": [assignment],
    }
    monkeypatch.setattr(sharding, "SHARDS_ROOT", shards_root)
    monkeypatch.setattr(sharding, "CALIBRATION_PATH", calibration)
    monkeypatch.setattr(sharding, "enumerate_environment", lambda *_: selected)
    monkeypatch.setattr(sharding, "_load_and_verify_plan", lambda *_: plan)
    monkeypatch.setattr(sharding, "_producer", lambda *_: object())
    monkeypatch.setattr(sharding, "load_calibration_manifest", lambda *_: {"lo": 0, "hi": 1})
    monkeypatch.setattr(
        sharding,
        "_load_wall_calibration",
        lambda *_: ({"calibration": {"lo": 0, "hi": 1}}, "3" * 64),
    )
    monkeypatch.setattr(
        sharding,
        "sha256_bytes",
        lambda value: "4" * 64 if b'"hi":1' in value else "5" * 64,
    )

    def fake_build(**kwargs):
        cache = kwargs["output_root"] / "pusht.lmdb"
        cache.mkdir(parents=True)
        (cache / "manifest.json").write_text("{}", encoding="utf-8")
        return {"manifest_id": "synthetic", "data_mdb_sha256": "6" * 64}

    monkeypatch.setattr(sharding, "build_environment_cache", fake_build)
    import tools.validate_depth_cache as validator

    monkeypatch.setattr(validator, "validate_environment_cache", lambda **_: {"state": "PASS"})
    args = argparse.Namespace(
        root=tmp_path,
        shards_root=shards_root,
        plan=tmp_path / "plan.json",
        shard_index=0,
        model_dir=tmp_path,
        da3_root=tmp_path,
    )
    record = sharding.build_shard(args)
    receipt = shards_root / "shard-000-of-008" / "validation.json"
    assert receipt.is_file()
    assert json.loads(receipt.read_text())["state"] == "PASS"
    assert record["validation_sha256"] == sharding.sha256_file(receipt)


@pytest.mark.parametrize("corruption", ["trajectory_keys", "validation_sha256"])
def test_merge_rejects_shard_assignment_or_receipt_drift(tmp_path, corruption) -> None:
    root = tmp_path / "shard-000-of-008"
    validation_path = root / "validation.json"
    assignment = {
        "shard_index": 0,
        "root": str(root),
        "trajectory_count": 1,
        "frame_count": 50,
        "source_index_sha256": "1" * 64,
        "trajectory_keys": ["train/00000"],
    }
    plan = {
        "plan_sha256": "2" * 64,
        "calibration_manifest_sha256": "3" * 64,
        "calibration_object_sha256": "4" * 64,
    }
    manifest = {"manifest_id": "manifest", "data_mdb_sha256": "5" * 64}
    record = {
        "schema": sharding.SHARD_SCHEMA,
        "plan_path": str(tmp_path / "plan.json"),
        "plan_sha256": plan["plan_sha256"],
        **assignment,
        "calibration_manifest_sha256": plan["calibration_manifest_sha256"],
        "calibration_object_sha256": plan["calibration_object_sha256"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": "6" * 64,
        "data_mdb_sha256": manifest["data_mdb_sha256"],
        "validation_path": str(validation_path),
        "validation_sha256": "7" * 64,
    }
    corrupted = copy.deepcopy(record)
    if corruption == "trajectory_keys":
        corrupted[corruption] = ["train/99999"]
    else:
        corrupted[corruption] = "8" * 64
    with pytest.raises(sharding.ContractError, match="record differs"):
        sharding.validate_shard_record_for_merge(
            shard=corrupted,
            assignment=assignment,
            root=root,
            plan=plan,
            plan_path=tmp_path / "plan.json",
            manifest=manifest,
            manifest_sha256="6" * 64,
            validation_path=validation_path,
            validation_sha256="7" * 64,
        )
