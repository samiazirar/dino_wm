from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools import precompute_depth as depth_module
from tools.precompute_depth import (
    ARTIFACTS,
    CUBLAS_WORKSPACE_CONFIG,
    MAP_SIZE,
    SELECTED_ENVIRONMENTS,
    ContractError,
    ProducerResult,
    Trajectory,
    build_environment_cache,
    canonical_json_bytes,
    configure_deterministic_producer,
    decode_trajectory,
    decode_depth_value,
    encode_depth_value,
    enumerate_environment,
    normalize_depth,
    select_calibration_keys,
    sha256_bytes,
    sha256_file,
    streaming_chunks,
    load_loop_detector_model_offline,
    verify_local_dinov2_hub,
    world_model_resize_crop,
)
from tools.validate_depth_cache import (
    dilate_changed_mask,
    temporal_pair_delta,
    validate_environment_cache,
    validate_key_set_and_range,
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
            "pth_float32_rgb_quantization": "round_half_up_to_uint8_for_png",
            "wall_terminal_observation_policy": "drop_post_action_frame_not_selected_by_WallDataset",
            "deterministic_execution": {
                "seed": 42,
                "cublas_workspace_config": ":4096:8",
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "allow_tf32": False,
                "reset_before_each_trajectory": True,
            },
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


def test_da3_determinism_is_reset_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = {
        "enabled": torch.are_deterministic_algorithms_enabled(),
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ContractError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_deterministic_producer(torch)

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    try:
        first_provenance = configure_deterministic_producer(torch)
        first = (np.random.random(), torch.rand(1).item())
        torch.backends.cudnn.benchmark = True
        second_provenance = configure_deterministic_producer(torch)
        second = (np.random.random(), torch.rand(1).item())

        assert first == second
        assert first_provenance == second_provenance
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.use_deterministic_algorithms(prior["enabled"])
        torch.backends.cudnn.benchmark = prior["benchmark"]
        torch.backends.cudnn.deterministic = prior["deterministic"]
        torch.backends.cudnn.allow_tf32 = prior["cudnn_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = prior["matmul_tf32"]


def _calibration(environment: str = "wall") -> dict:
    keys = [
        f"{environment}/train/{episode:05d}/{frame:06d}"
        for episode in range(16)
        for frame in range(8)
    ]
    assert len(keys) == 128
    return {
        "scope": "environment_training_only",
        "environment": environment,
        "selected_environments": [environment],
        "selection": "128_smallest_sha256_training_keys_for_environment",
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


def test_released_pth_float32_rgb_quantization_is_explicit(tmp_path: Path) -> None:
    _make_wall_dataset(tmp_path, episodes=1, frames=2)
    source = tmp_path / "wall_single" / "obses" / "episode_000.pth"
    released = torch.load(source).to(torch.float32)
    terminal = torch.full_like(released[:1], 200)
    released = torch.cat((released, terminal), dim=0)
    torch.save(released, source)
    trajectory = enumerate_environment(tmp_path, "wall")[0]
    decoded = decode_trajectory(trajectory)
    assert decoded.dtype == np.uint8
    assert decoded.shape == (2, 32, 48, 3)
    assert np.all(decoded == 64)

    released[0, 0, 0, 0] = 64.5
    released[0, 1, 0, 0] = 64.49
    torch.save(released, source)
    decoded = decode_trajectory(trajectory)
    assert decoded[0, 0, 0, 0] == 65
    assert decoded[0, 0, 0, 1] == 64

    released[0, 0, 0, 0] = torch.nan
    torch.save(released, source)
    with pytest.raises(ContractError, match="must be finite float32"):
        decode_trajectory(trajectory)


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
            if item.split == "train"
            for frame in range(item.frame_count)
        ]
        expected = sorted(
            candidates, key=lambda key: (sha256_bytes(key.encode()), key)
        )[:2]
        assert [
            key for key in selected if key.startswith(f"{environment}/")
        ] == expected
        assert all("/train/" in key for key in selected)


def test_wire_is_zstd3_little_endian_float16() -> None:
    gray = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(224, 224)
    payload = encode_depth_value(gray)
    decoded = decode_depth_value(payload)
    assert decoded.shape == (224, 224)
    assert decoded.dtype.str == "<f2"
    np.testing.assert_array_equal(decoded, gray.astype("<f2"))


def test_low_std_is_characterization_only_for_wall_and_hard_elsewhere(
    tmp_path: Path,
) -> None:
    lmdb = pytest.importorskip("lmdb")

    def constant_cache(environment: str):
        trajectory = _trajectory(environment, "train", 0, 4)
        cache = tmp_path / environment
        database = lmdb.open(str(cache), map_size=64 << 20, subdir=True)
        with database.begin(write=True) as transaction:
            for frame, value in enumerate((0.1, 0.3, 0.6, 0.9)):
                payload = encode_depth_value(
                    np.full((224, 224), value, dtype=np.float32)
                )
                transaction.put(trajectory.physical_key(frame).encode("ascii"), payload)
        return database, trajectory

    wall_database, wall_trajectory = constant_cache("wall")
    try:
        result = validate_key_set_and_range(wall_database, "wall", [wall_trajectory])
    finally:
        wall_database.close()
    assert result["low_std_map_fraction"] == 1.0
    assert (
        result["low_std_map_fraction_acceptance"]
        == "CHARACTERIZATION_ONLY_FLAT_SCENE_CONTROL"
    )
    assert result["low_std_map_fraction_reference_threshold"] == 0.01

    rope_database, rope_trajectory = constant_cache("rope")
    try:
        with pytest.raises(ContractError, match="low-std maps=1.000000"):
            validate_key_set_and_range(rope_database, "rope", [rope_trajectory])
    finally:
        rope_database.close()


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


def test_rebuild_never_replaces_or_quarantines_existing_cache(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    _make_wall_dataset(data_root)
    trajectories = enumerate_environment(data_root, "wall")
    build_environment_cache(
        output_root=cache_root,
        environment="wall",
        trajectories=trajectories,
        calibration=_calibration(),
        producer=DeterministicTrajectoryProducer(),
    )
    original_manifest = (cache_root / "wall.lmdb" / "manifest.json").read_bytes()
    with pytest.raises(ContractError, match="retained-copy replacement is forbidden"):
        build_environment_cache(
            output_root=cache_root,
            environment="wall",
            trajectories=trajectories,
            calibration=_calibration(),
            producer=DeterministicTrajectoryProducer(),
            rebuild=True,
        )
    assert (cache_root / "wall.lmdb" / "manifest.json").read_bytes() == original_manifest
    assert not list(cache_root.glob("*.quarantine-*"))


def test_normalization_is_global_clip_without_inversion() -> None:
    depth = np.linspace(1.0, 3.0, 224 * 224, dtype=np.float32).reshape(224, 224)
    gray = normalize_depth(depth, lo=1.5, hi=2.5)
    assert gray[0, 0] == 0.0
    assert gray[-1, -1] == 1.0
    assert gray[100, 100] < gray[120, 120]


def _fake_torch_hub(tmp_path: Path):
    repo = tmp_path / "facebookresearch_dinov2_main"
    repo.mkdir()
    hubconf = repo / "hubconf.py"
    hubconf.write_text("# pinned test hubconf\n", encoding="utf-8")
    calls: list[tuple[str, str, tuple, dict]] = []

    def load(repo_or_dir: str, model: str, *args, **kwargs):
        calls.append((repo_or_dir, model, args, kwargs))
        return model

    hub = type("FakeHub", (), {})()
    hub.get_dir = lambda: str(tmp_path)
    hub.load = load
    torch_module = type("FakeTorch", (), {})()
    torch_module.hub = hub
    return torch_module, hubconf, calls, load


def test_dinov2_hub_redirect_is_exact_and_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch_module, hubconf, calls, original_load = _fake_torch_hub(tmp_path)
    monkeypatch.setattr(
        depth_module, "DINOV2_HUBCONF_SHA256", sha256_file(hubconf)
    )

    class Detector:
        def load_model(self) -> None:
            torch_module.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", verbose=False
            )
            torch_module.hub.load("other/repository", "other_model", source="github")

    provenance = verify_local_dinov2_hub(torch_module)
    load_loop_detector_model_offline(Detector(), torch_module)

    assert provenance == {
        "requested_repository": "facebookresearch/dinov2",
        "repo_or_dir": str(tmp_path / "facebookresearch_dinov2_main"),
        "source": "local",
        "hubconf_sha256": sha256_file(hubconf),
    }
    assert calls[0] == (
        str(tmp_path / "facebookresearch_dinov2_main"),
        "dinov2_vitb14",
        (),
        {"verbose": False, "source": "local"},
    )
    assert calls[1] == (
        "other/repository",
        "other_model",
        (),
        {"source": "github"},
    )
    assert torch_module.hub.load is original_load


def test_dinov2_hub_redirect_fails_closed_on_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch_module, _hubconf, calls, original_load = _fake_torch_hub(tmp_path)
    monkeypatch.setattr(depth_module, "DINOV2_HUBCONF_SHA256", "0" * 64)

    class Detector:
        def load_model(self) -> None:
            raise AssertionError("hash gate must run before model loading")

    with pytest.raises(ContractError, match="hubconf mismatch"):
        load_loop_detector_model_offline(Detector(), torch_module)
    assert calls == []
    assert torch_module.hub.load is original_load


def test_dinov2_hub_redirect_restores_after_load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch_module, hubconf, _calls, original_load = _fake_torch_hub(tmp_path)
    monkeypatch.setattr(
        depth_module, "DINOV2_HUBCONF_SHA256", sha256_file(hubconf)
    )

    class Detector:
        def load_model(self) -> None:
            torch_module.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            raise RuntimeError("injected load failure")

    with pytest.raises(RuntimeError, match="injected load failure"):
        load_loop_detector_model_offline(Detector(), torch_module)
    assert torch_module.hub.load is original_load


# Every DA3 PushT sbatch wrapper that constructs OfficialDA3StreamingProducer
# (build shard; merge-validate recompute) MUST hand CUBLAS_WORKSPACE_CONFIG=:4096:8
# into the container, or configure_deterministic_producer() fail-closes at
# producer __init__ with ContractError (exit 2) before any depth is written.
# Regression for the 2026-07-17 eight-shard 2:0 failure: the wrappers were
# authored against the pre-determinism-fix producer and omitted this --env.
DA3_PUSHT_PRODUCER_SBATCH = (
    "da3_pusht_shard.sbatch",
    "da3_pusht_merge_validate.sbatch",
)


@pytest.mark.parametrize("script_name", DA3_PUSHT_PRODUCER_SBATCH)
def test_da3_pusht_producer_sbatch_passes_cublas_workspace_config(
    script_name: str,
) -> None:
    script = Path(__file__).resolve().parents[1] / "tools" / script_name
    text = script.read_text(encoding="utf-8")
    assert (
        f"--env CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG}" in text
    ), (
        f"{script_name} must pass --env CUBLAS_WORKSPACE_CONFIG="
        f"{CUBLAS_WORKSPACE_CONFIG} into apptainer; the pinned producer "
        "fail-closes without it"
    )
