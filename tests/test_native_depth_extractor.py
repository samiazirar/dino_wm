from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import h5py
import lmdb
import numpy as np
import pytest
import torch

from tools.extract_native_depth import (
    CAMERA_INDEX,
    EXPECTED_KEYS_PER_ENVIRONMENT,
    HDF5_FILES_PER_EPISODE,
    OUTPUT_FRAMES_PER_EPISODE,
    RELEASED_EPISODES,
    ReleaseContract,
    build_native_cache,
    decode_uint16_millimetres,
)
from tools.precompute_depth import ContractError, decode_depth_value, sha256_file


def _test_contract(*, episodes: int = 2, frames: int = 3) -> ReleaseContract:
    return ReleaseContract(
        episodes=episodes,
        output_frames=frames,
        hdf5_files=frames + 1,
        calibration_frames=2,
        expected_release_hashes=None,
        expected_camera_hashes=None,
    )


def _write_hdf5(
    path: Path,
    *,
    rgb: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    depth_mm: int,
) -> None:
    particle_count = state.shape[0]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=action.astype(np.float64))
        handle.create_dataset(
            "eef_states", data=np.zeros((1, 1, 14), dtype=np.float64)
        )
        handle.create_dataset("info/n_cams", data=np.int64(4))
        handle.create_dataset("info/n_particles", data=np.int64(particle_count))
        handle.create_dataset("info/timestamp", data=np.int64(0))
        for camera in range(4):
            handle.create_dataset(
                f"observations/color/cam_{camera}",
                data=rgb[None].astype(np.float32),
            )
            handle.create_dataset(
                f"observations/depth/cam_{camera}",
                data=np.full(
                    (1, 224, 224), depth_mm + camera, dtype=np.uint16
                ),
            )
        handle.create_dataset("positions", data=state[None].astype(np.float32))


def _make_release(
    root: Path,
    *,
    environment: str = "rope",
    contract: ReleaseContract | None = None,
) -> tuple[ReleaseContract, Path]:
    contract = contract or _test_contract()
    base = root / "deformable" / environment
    (base / "cameras").mkdir(parents=True)
    np.save(base / "cameras" / "intrinsic.npy", np.eye(4, dtype=np.float64))
    np.save(
        base / "cameras" / "extrinsic.npy",
        np.broadcast_to(np.eye(4), (4, 4, 4)).astype(np.float64),
    )
    particle_count = 5
    actions = torch.zeros(
        (contract.episodes, contract.output_frames, 4), dtype=torch.float64
    )
    states = torch.empty(
        (contract.episodes, contract.output_frames, particle_count, 4),
        dtype=torch.float32,
    )
    for episode in range(contract.episodes):
        for frame in range(contract.output_frames):
            actions[episode, frame] = episode * 10 + frame
            states[episode, frame] = (
                torch.arange(particle_count * 4, dtype=torch.float32).reshape(
                    particle_count, 4
                )
                + episode * 100
                + frame
            )
    torch.save(actions, base / "actions.pth")
    torch.save(states, base / "states.pth")

    for episode in range(contract.episodes):
        episode_root = base / f"{episode:06d}"
        episode_root.mkdir()
        observations = np.empty(
            (contract.output_frames, 224, 224, 3), dtype=np.float32
        )
        for frame in range(contract.output_frames):
            observations[frame] = 50 + episode * 10 + frame
        torch.save(torch.from_numpy(observations), episode_root / "obses.pth")
        (episode_root / "property_params.pkl").write_bytes(
            f"property-{environment}-{episode}".encode()
        )
        for frame in range(contract.hdf5_files):
            source_frame = min(frame, contract.output_frames - 1)
            _write_hdf5(
                episode_root / f"{frame:02d}.h5",
                rgb=observations[source_frame],
                state=states[episode, source_frame].numpy(),
                action=actions[episode, source_frame].numpy(),
                depth_mm=1000 + episode * 100 + frame * 10,
            )
    return contract, base


def test_released_production_contract_has_exact_key_count() -> None:
    assert RELEASED_EPISODES == 1000
    assert OUTPUT_FRAMES_PER_EPISODE == 20
    assert HDF5_FILES_PER_EPISODE == 21
    assert EXPECTED_KEYS_PER_ENVIRONMENT == 20_000


def test_uint16_millimetres_decode_to_float32_metres_exactly() -> None:
    value = np.zeros((224, 224), dtype=np.uint16)
    value[0, :5] = [0, 1, 999, 1000, 65535]
    decoded = decode_uint16_millimetres(value)
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(
        decoded[0, :5],
        np.array([0.0, 0.001, 0.999, 1.0, 65.535], dtype=np.float32),
        rtol=0,
        atol=4e-6,
    )
    with pytest.raises(ContractError, match="uint16 millimetres"):
        decode_uint16_millimetres(value.astype(np.int32))


def test_build_is_exact_aligned_hashed_and_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    contract, _ = _make_release(root)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = build_native_cache(
        root=root,
        output_root=first_root,
        environment="rope",
        contract=contract,
    )
    second = build_native_cache(
        root=root,
        output_root=second_root,
        environment="rope",
        contract=contract,
    )

    assert first == second
    assert first["frame_count"] == contract.expected_keys == 6
    assert first["trajectory_count"] == contract.episodes
    assert first["producer"]["scientific_role"] == "explicit_upper_bound_ablation_only"
    assert first["producer"]["simulator_replay"] is False
    assert first["producer"]["training_defaults_modified"] is False
    assert first["producer"]["neutral_depth_boundary_modified"] is False
    assert len(first["native_source_index_sha256"]) == 64
    assert len(first["tool_sha256"]) == 64
    assert len(first["data_mdb_sha256"]) == 64
    first_manifest = first_root / "rope.lmdb" / "manifest.json"
    second_manifest = second_root / "rope.lmdb" / "manifest.json"
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert sha256_file(first_manifest) == sha256_file(second_manifest)
    assert sha256_file(first_root / "rope.lmdb" / "data.mdb") == sha256_file(
        second_root / "rope.lmdb" / "data.mdb"
    )

    expected_keys = [
        key
        for record in first["trajectories"]
        for key in record["ordered_output_keys"]
    ]
    assert len(expected_keys) == len(set(expected_keys)) == contract.expected_keys
    database = lmdb.open(
        str(first_root / "rope.lmdb"), readonly=True, lock=False, subdir=True
    )
    try:
        with database.begin(write=False) as transaction:
            actual_keys = [key.decode("ascii") for key, _ in transaction.cursor()]
            assert actual_keys == sorted(expected_keys)
            payload = transaction.get(expected_keys[0].encode("ascii"))
            decoded = decode_depth_value(payload)
            assert decoded.shape == (224, 224)
            assert np.isfinite(decoded).all()
            assert 0 <= float(decoded.min()) <= float(decoded.max()) <= 1
    finally:
        database.close()

    manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
    assert manifest == first
    for episode in manifest["source_provenance"]["episodes"]:
        assert len(episode["hdf5"]) == contract.hdf5_files
        assert all(len(record["sha256"]) == 64 for record in episode["hdf5"])


Mutation = Callable[[Path, ReleaseContract, Path], None]


def _rgb_mismatch(root: Path, contract: ReleaseContract, base: Path) -> None:
    path = base / f"{contract.episodes - 1:06d}" / "02.h5"
    with h5py.File(path, "r+") as handle:
        handle[f"observations/color/cam_{CAMERA_INDEX}"][0, 0, 0, 0] += 1


def _state_mismatch(root: Path, contract: ReleaseContract, base: Path) -> None:
    path = base / f"{contract.episodes - 1:06d}" / "02.h5"
    with h5py.File(path, "r+") as handle:
        handle["positions"][0, 0, 0] += 1


def _wrong_depth_dtype(root: Path, contract: ReleaseContract, base: Path) -> None:
    path = base / "000000" / "00.h5"
    with h5py.File(path, "r+") as handle:
        value = handle[f"observations/depth/cam_{CAMERA_INDEX}"][()]
        del handle[f"observations/depth/cam_{CAMERA_INDEX}"]
        handle.create_dataset(
            f"observations/depth/cam_{CAMERA_INDEX}",
            data=value.astype(np.float32),
        )


def _missing_hdf5(root: Path, contract: ReleaseContract, base: Path) -> None:
    (base / "000000" / "01.h5").unlink()


def _extra_hdf5(root: Path, contract: ReleaseContract, base: Path) -> None:
    (base / "000000" / "99.h5").write_bytes(b"unexpected")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_rgb_mismatch, "RGB is not exactly aligned"),
        (_state_mismatch, "positions are not exactly aligned"),
        (_wrong_depth_dtype, "expected .*uint16"),
        (_missing_hdf5, "episode inventory differs"),
        (_extra_hdf5, "episode inventory differs"),
    ],
)
def test_alignment_schema_and_inventory_fail_closed(
    tmp_path: Path,
    mutation: Mutation,
    message: str,
) -> None:
    root = tmp_path / "raw"
    contract, base = _make_release(root)
    mutation(root, contract, base)
    output = tmp_path / "cache"
    with pytest.raises(ContractError, match=message):
        build_native_cache(
            root=root,
            output_root=output,
            environment="rope",
            contract=contract,
        )
    assert not (output / "rope.lmdb").exists()


def test_late_failure_retains_only_unpublished_staging_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    contract, base = _make_release(root)
    _rgb_mismatch(root, contract, base)
    output = tmp_path / "cache"
    with pytest.raises(ContractError, match="RGB is not exactly aligned"):
        build_native_cache(
            root=root,
            output_root=output,
            environment="rope",
            contract=contract,
        )
    assert not (output / "rope.lmdb").exists()
    staging = list(output.glob(".rope.lmdb.building-*"))
    assert len(staging) == 1
    assert (staging[0] / "data.mdb").is_file()


def test_existing_destination_is_never_replaced(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    contract, _ = _make_release(root)
    output = tmp_path / "cache"
    build_native_cache(
        root=root,
        output_root=output,
        environment="rope",
        contract=contract,
    )
    manifest = output / "rope.lmdb" / "manifest.json"
    original = manifest.read_bytes()
    with pytest.raises(ContractError, match="will not be overwritten"):
        build_native_cache(
            root=root,
            output_root=output,
            environment="rope",
            contract=contract,
        )
    assert manifest.read_bytes() == original


def test_exact_episode_and_frame_count_is_required_before_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    contract, base = _make_release(root)
    actions = torch.load(base / "actions.pth", weights_only=False)
    torch.save(actions[:, :-1], base / "actions.pth")
    output = tmp_path / "cache"
    with pytest.raises(ContractError, match="actions.pth shape"):
        build_native_cache(
            root=root,
            output_root=output,
            environment="rope",
            contract=contract,
        )
    assert not (output / "rope.lmdb").exists()
