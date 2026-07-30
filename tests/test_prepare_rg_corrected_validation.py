from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.prepare_rg_corrected_validation import RAW_CALIBRATION, prepare_card


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_cache(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    cache = tmp_path / "rope.lmdb"
    cache.mkdir()
    data = cache / "data.mdb"
    data.write_bytes(b"completed-cache-bytes")
    source_tool = tmp_path / "precompute_depth_mapanything.py"
    source_tool.write_text("# corrected validator\n")
    hashes = {
        "data": sha256(data),
        "tool": "a" * 64,
        "source": sha256(source_tool),
    }
    manifest = {
        "schema": "dinocular-depth-cache-v1",
        "environment": "rope",
        "calibration": RAW_CALIBRATION,
        "wire_format": {"dtype": "<f2", "normalization": "none_raw_depth_z_float16"},
        "closed_before_hash": True,
        "data_mdb_sha256": hashes["data"],
        "tool_sha256": hashes["tool"],
        "manifest_id": "cache-id",
        "source_index_sha256": "b" * 64,
    }
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    hashes["manifest"] = sha256(manifest_path)
    return cache, source_tool, hashes


def test_prepare_card_freezes_existing_cache_and_corrected_source(tmp_path: Path) -> None:
    cache, source_tool, hashes = make_cache(tmp_path)

    card = prepare_card(
        cache_dir=cache,
        environment="rope",
        source_commit="c" * 40,
        source_tool=source_tool,
        expected_build_tool_sha256=hashes["tool"],
        expected_manifest_sha256=hashes["manifest"],
        expected_data_sha256=hashes["data"],
        max_trajectories=1,
        spot_frames=32,
        batch_size=8,
    )

    assert card["stage"] == "validate_only_no_rebuild"
    assert card["cache"]["data_mdb_sha256"] == hashes["data"]
    assert card["corrected_validator"]["source_tool_sha256"] == hashes["source"]


def test_prepare_card_rejects_changed_completed_bytes(tmp_path: Path) -> None:
    cache, source_tool, hashes = make_cache(tmp_path)
    (cache / "data.mdb").write_bytes(b"modified-cache-bytes")

    with pytest.raises(RuntimeError, match="completed cache bytes differ"):
        prepare_card(
            cache_dir=cache,
            environment="rope",
            source_commit="c" * 40,
            source_tool=source_tool,
            expected_build_tool_sha256=hashes["tool"],
            expected_manifest_sha256=hashes["manifest"],
            expected_data_sha256=hashes["data"],
            max_trajectories=1,
            spot_frames=32,
            batch_size=8,
        )
