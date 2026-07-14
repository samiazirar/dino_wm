from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools.precompute_depth import (
    ARTIFACTS,
    MAP_SIZE,
    SELECTED_ENVIRONMENTS,
    ContractError,
    ProducerResult,
    Trajectory,
    build_environment_cache,
    canonical_json_bytes,
    decode_depth_value,
    encode_depth_value,
    enumerate_environment,
    normalize_depth,
    select_calibration_keys,
    sha256_bytes,
    streaming_chunks,
    world_model_resize_crop,
)
from tools.validate_depth_cache import (
    dilate_changed_mask,
    temporal_pair_delta,
    validate_environment_cache,
)


def _fake_provenance() -> dict:
    return {
        "name": "test-only-injected-trajectory-producer",
        "repository": "injected-by-unit-test",
        "commit": "0" * 40,
        "salad_commit": "1" * 40,
        "model_revision": "test",
        "da3_root": "/nonexistent/test-only",
        "artifacts": {
            name: {"path": f"/nonexistent/{name}", **expected}
            for name, expected in ARTIFACTS.items()
        },
        "effective_config_sha256": "2" * 64,
        "settings": {
            "precision": "bfloat16",
            "process_res": 504,
            "process_res_method": "upper_bound_resize",
            "ref_view_strategy": "saddle_balanced",
            "loop_closure": True,
            "chunk_size": 120,
            "overlap": 60,
            "overlap_policy": "discard_duplicated_tail_no_blend",
        },
        "official_non_strict_load_audit": {"missing_keys": [], "unexpected_keys": []},
    }


class DeterministicTrajectoryProducer:
    provenance = _fake_provenance()

    def __init__(self, fail_episode: int | None = None):
        self.calls: list[tuple[str, int]] = []
        self.fail_episode = fail_episode

    def infer_trajectory(
        self, frames_rgb: np.ndarray, trajectory: Trajectory
    ) -> ProducerResult:
        assert len(frames_rgb) == trajectory.frame_count
        self.calls.append((trajectory.trajectory_key, len(frames_rgb)))
        if trajectory.episode == self.fail_episode:
            raise ContractError("injected trajectory failure")
        height, width = frames_rgb.shape[1:3]
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        depths = [
            0.25 + 0.25 * x + 0.15 * y + frame * 0.001
            for frame in range(len(frames_rgb))
        ]
        return ProducerResult(
            depth_m=depths,
            chunk_boundaries=streaming_chunks(len(frames_rgb)),
            alignments=[],
            metadata={"injected": True},
        )


def _calibration() -> dict:
    keys = [
        f"{environment}/train/{episode:05d}/{frame:06d}"
        for environment in SELECTED_ENVIRONMENTS
        for episode in range(16)
        for frame in range(8)
    ]
    assert len(keys) == 512
    return {
        "scope": "global_across_selected_environments",
        "selected_environments": list(SELECTED_ENVIRONMENTS),
        "selection": "128_smallest_sha256_keys_per_environment",
        "per_environment": 128,
        "frame_key_format": "<env>/<split>/<episode:05d>/<frame:06d>",
        "keys": keys,
        "keys_sha256": sha256_bytes(canonical_json_bytes(keys)),
        "sample": "cropped_metric_depth[::8,::8]",
        "sample_stride": 8,
        "percentiles": [2.0, 98.0],
        "percentile_method": "numpy.linear",
        "lo": 0.0,
        "hi": 1.0,
    }


def _make_wall_dataset(root: Path, episodes: int = 2, frames: int = 4) -> None:
    wall = root / "wall_single"
    (wall / "obses").mkdir(parents=True)
    torch.save(
        torch.zeros((episodes, frames, 2), dtype=torch.float32), wall / "actions.pth"
    )
    for episode in range(episodes):
        # Static RGB makes the temporal static-background gate unambiguous.
        rgb = torch.full((frames, 3, 32, 48), 64 + episode, dtype=torch.uint8)
        torch.save(rgb, wall / "obses" / f"episode_{episode:03d}.pth")


def _trajectory(environment: str, split: str, episode: int, frames: int) -> Trajectory:
    return Trajectory(
        environment=environment,
        split=split,
        episode=episode,
        frame_count=frames,
        source_path=Path(f"/{environment}-{episode}"),
        source_relpath=f"{environment}-{episode}",
        source_kind="test",
        source_sha256=f"{episode:064x}",
    )


def test_streaming_120_60_discards_only_duplicate_tails() -> None:
    chunks = streaming_chunks(300)
    assert [(chunk["start"], chunk["end"]) for chunk in chunks] == [
        (0, 120),
        (60, 180),
        (120, 240),
        (180, 300),
    ]
    assert [chunk["boundary_pair"] for chunk in chunks] == [
        None,
        [59, 60],
        [119, 120],
        [179, 180],
    ]
    emitted = [
        frame
        for chunk in chunks
        for frame in range(chunk["emitted_start"], chunk["emitted_end"])
    ]
    assert emitted == list(range(300))
    assert streaming_chunks(120)[0]["emitted_end"] == 120


def test_calibration_keys_are_stratified_smallest_sha256() -> None:
    trajectories = {
        environment: [
            _trajectory(environment, "train", 0, 5),
            _trajectory(environment, "valid", 1, 4),
        ]
        for environment in SELECTED_ENVIRONMENTS
    }
    selected = select_calibration_keys(trajectories, per_environment=2)
    assert len(selected) == 8
    for environment in SELECTED_ENVIRONMENTS:
        candidates = [
            item.logical_key(frame)
            for item in trajectories[environment]
            for frame in range(item.frame_count)
        ]
        expected = sorted(
            candidates, key=lambda key: (sha256_bytes(key.encode()), key)
        )[:2]
        assert [
            key for key in selected if key.startswith(f"{environment}/")
        ] == expected


def test_wire_is_zstd3_little_endian_float16() -> None:
    gray = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(224, 224)
    payload = encode_depth_value(gray)
    decoded = decode_depth_value(payload)
    assert decoded.shape == (224, 224)
    assert decoded.dtype.str == "<f2"
    np.testing.assert_array_equal(decoded, gray.astype("<f2"))


def test_world_model_short_edge_resize_then_center_crop() -> None:
    image = np.broadcast_to(np.arange(200, dtype=np.float32)[None, :], (100, 200))
    cropped = world_model_resize_crop(image)
    assert cropped.shape == (224, 224)
    # The center crop must remove both long-edge sides, not stretch 100x200 to a square.
    assert 45 < float(cropped[:, 0].mean()) < 55
    assert 145 < float(cropped[:, -1].mean()) < 155


def test_changed_mask_uses_5x5_dilation_and_static_median() -> None:
    changed = np.zeros((224, 224), dtype=bool)
    changed[100, 100] = True
    dilated = dilate_changed_mask(changed)
    assert int(dilated.sum()) == 25
    rgb_left = np.zeros((224, 224, 3), dtype=np.float32)
    rgb_right = rgb_left.copy()
    rgb_right[100, 100] = 1.0
    depth_left = np.full((224, 224), 0.25, dtype="<f2")
    depth_right = np.full((224, 224), 0.255, dtype="<f2")
    static_fraction, median = temporal_pair_delta(
        rgb_left, rgb_right, depth_left, depth_right
    )
    assert static_fraction == pytest.approx(1 - 25 / (224 * 224))
    assert median == pytest.approx(float(np.float16(0.255) - np.float16(0.25)))


def test_build_and_all_validation_gates_with_injected_full_trajectory_producer(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    _make_wall_dataset(data_root)
    trajectories = enumerate_environment(data_root, "wall")
    producer = DeterministicTrajectoryProducer()
    manifest = build_environment_cache(
        output_root=cache_root,
        environment="wall",
        trajectories=trajectories,
        calibration=_calibration(),
        producer=producer,
    )
    cache = cache_root / "wall.lmdb"
    assert cache.is_dir()
    assert manifest["wire_format"]["map_size"] == MAP_SIZE
    assert manifest["frame_count"] == 8
    assert [call[1] for call in producer.calls] == [4, 4]
    assert manifest["data_mdb_sha256"]
    assert manifest["trajectories"][0]["ordered_output_keys"] == [
        f"{trajectories[0].trajectory_key}/{frame:06d}" for frame in range(4)
    ]

    result = validate_environment_cache(
        root=data_root,
        cache_root=cache_root,
        environment="wall",
        producer=producer,
        recompute_fraction=0.01,
    )
    assert result["state"] == "PASS"
    assert result["range_gate"]["expected_keys"] == 8
    assert result["recomputation_gate"]["max_absolute_error"] <= 1e-3
    assert result["temporal_gate"]["eligible_fraction"] == 1.0


def test_failed_later_trajectory_never_publishes_partial_environment(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    _make_wall_dataset(data_root)
    trajectories = enumerate_environment(data_root, "wall")
    producer = DeterministicTrajectoryProducer(fail_episode=trajectories[1].episode)
    with pytest.raises(ContractError, match="injected trajectory failure"):
        build_environment_cache(
            output_root=cache_root,
            environment="wall",
            trajectories=trajectories,
            calibration=_calibration(),
            producer=producer,
        )
    assert not (cache_root / "wall.lmdb").exists()
    building = list(cache_root.glob(".wall.lmdb.building-*"))
    assert (
        len(building) == 1
    )  # recovery evidence is retained; no in-place continuation occurs.


def test_normalization_is_global_clip_without_inversion() -> None:
    depth = np.linspace(1.0, 3.0, 224 * 224, dtype=np.float32).reshape(224, 224)
    gray = normalize_depth(depth, lo=1.5, hi=2.5)
    assert gray[0, 0] == 0.0
    assert gray[-1, -1] == 1.0
    assert gray[100, 100] < gray[120, 120]
