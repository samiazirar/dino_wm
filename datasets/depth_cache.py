"""Read immutable cache-aligned trajectory depth under pinned manifests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from depth_contract import (
    DepthContractError,
    canonical_json_bytes,
    load_native_depth_contract,
    require_sha256,
    sha256_bytes,
    sha256_file,
)
from empirical_depth_contract import (
    EmpiricalDepthContractError,
    load_empirical_depth_contract,
    load_empirical_runtime_release,
    validate_mapanything_receipt,
)


CACHE_SCHEMA = "dinocular-depth-cache-v1"
VALIDATION_SCHEMA = "dinocular-depth-cache-validation-v1"
WIRE_STORAGE_FORMAT = {
    "physical_key": "<split>/<episode:05d>/<frame:06d>",
    "dtype": "<f2",
    "shape": [224, 224],
    "order": "C",
    "compressor": "zstd",
    "compressor_level": 3,
    "map_size": 1 << 40,
}


class DepthCacheError(DepthContractError):
    """The configured cache is incomplete, mismatched, or corrupt."""


def _load_hashed_json(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    expected = require_sha256(expected_sha256, f"{label} SHA-256")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise DepthCacheError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise DepthCacheError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DepthCacheError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DepthCacheError(f"{label} must be a JSON object")
    return value


class DepthCacheReader:
    """Hash-audited, read-only LMDB reader shared by train and validation sets."""

    def __init__(
        self,
        *,
        environment: str,
        source_root: str | Path,
        cache_dir: str | Path,
        cache_manifest_sha256: str,
        validation_path: str | Path,
        validation_sha256: str,
        native_contract_path: str | Path,
        native_contract_sha256: str,
        expected_producer_sha256: str,
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        self.environment = str(environment)
        self.source_root = Path(source_root).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.native_contract = load_native_depth_contract(
            native_contract_path,
            native_contract_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        self.expected_producer_sha256 = require_sha256(
            expected_producer_sha256, "expected cache producer SHA-256"
        )
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = _load_hashed_json(
            self.manifest_path, cache_manifest_sha256, "cache manifest"
        )
        self.validation_path = Path(validation_path).expanduser().resolve()
        self.validation = _load_hashed_json(
            self.validation_path, validation_sha256, "cache validation report"
        )
        self._database = None
        self._decompressor = None
        self._records: dict[tuple[str, int], Mapping[str, Any]] = {}
        self._records_by_episode: dict[int, list[Mapping[str, Any]]] = {}
        self.binding = None
        self._validate()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_database"] = None
        state["_decompressor"] = None
        return state

    def _validate(self) -> None:
        manifest = self.manifest
        if manifest.get("schema") != CACHE_SCHEMA:
            raise DepthCacheError("unsupported cache manifest schema")
        if manifest.get("environment") != self.environment:
            raise DepthCacheError("cache manifest environment mismatch")
        wire = manifest.get("wire_format")
        if not isinstance(wire, Mapping):
            raise DepthCacheError("cache wire format is absent")
        storage_mismatches = {
            key: (wire.get(key), expected)
            for key, expected in WIRE_STORAGE_FORMAT.items()
            if wire.get(key) != expected
        }
        if storage_mismatches:
            raise DepthCacheError(
                f"cache storage format differs from the supported reader: {storage_mismatches}"
            )
        if manifest.get("closed_before_hash") is not True:
            raise DepthCacheError("cache was not closed before data.mdb hashing")
        producer = manifest.get("producer")
        if not isinstance(producer, Mapping) or not producer:
            raise DepthCacheError("cache producer provenance is absent")
        producer_sha256 = sha256_bytes(canonical_json_bytes(producer))
        if producer_sha256 != self.expected_producer_sha256:
            raise DepthCacheError(
                "cache producer differs from the run's selected producer SHA-256"
            )
        self.binding = self.native_contract.binding_for(producer_sha256)
        wire_sha256 = sha256_bytes(canonical_json_bytes(wire))
        if wire_sha256 != self.binding.wire_format_sha256:
            raise DepthCacheError(
                "cache wire format is not the exact producer-bound format approved "
                "by the native student manifest"
            )

        data_path = self.cache_dir / "data.mdb"
        expected_data_sha = require_sha256(
            manifest.get("data_mdb_sha256"), "cache data_mdb_sha256"
        )
        if not data_path.is_file() or sha256_file(data_path) != expected_data_sha:
            raise DepthCacheError("closed data.mdb SHA-256 mismatch")

        report = self.validation
        if report.get("schema") != VALIDATION_SCHEMA or report.get("state") != "PASS":
            raise DepthCacheError("cache validation report is not PASS")
        result = report.get("results", {}).get(self.environment)
        if not isinstance(result, Mapping) or result.get("state") != "PASS":
            raise DepthCacheError(
                f"cache validation has no PASS result for {self.environment}"
            )
        if (
            result.get("manifest_id") != manifest.get("manifest_id")
            or result.get("data_mdb_sha256") != expected_data_sha
        ):
            raise DepthCacheError("validation report does not identify this exact cache")

        records = manifest.get("trajectories")
        if not isinstance(records, list) or len(records) != manifest.get(
            "trajectory_count"
        ):
            raise DepthCacheError("cache trajectory index is incomplete")
        total_frames = 0
        source_index = []
        for record in records:
            if not isinstance(record, Mapping):
                raise DepthCacheError("cache trajectory record is not an object")
            key = record.get("trajectory_key")
            if not isinstance(key, str) or key.count("/") != 1:
                raise DepthCacheError(f"invalid trajectory key: {key!r}")
            split, episode_text = key.split("/")
            if split not in {"train", "valid"} or not episode_text.isdigit():
                raise DepthCacheError(f"invalid trajectory key: {key!r}")
            episode = int(episode_text)
            count = record.get("ordered_frame_count")
            if not isinstance(count, int) or count <= 0:
                raise DepthCacheError(f"invalid frame count for {key}")
            expected_keys = [f"{key}/{frame:06d}" for frame in range(count)]
            if record.get("ordered_output_keys") != expected_keys:
                raise DepthCacheError(f"physical key order differs for {key}")
            source_relpath = record.get("source_path")
            if not isinstance(source_relpath, str) or Path(source_relpath).is_absolute():
                raise DepthCacheError(f"invalid source path for {key}")
            source = (self.source_root / source_relpath).resolve()
            try:
                source.relative_to(self.source_root)
            except ValueError as exc:
                raise DepthCacheError(f"source path escapes dataset root: {source}") from exc
            expected_source_sha = require_sha256(
                record.get("source_video_sha256"), f"source SHA-256 for {key}"
            )
            if not source.is_file() or sha256_file(source) != expected_source_sha:
                raise DepthCacheError(f"source hash differs for {key}: {source}")
            record_key = (split, episode)
            if record_key in self._records:
                raise DepthCacheError(f"duplicate cache trajectory {key}")
            self._records[record_key] = record
            self._records_by_episode.setdefault(episode, []).append(record)
            total_frames += count
            source_index.append(
                {
                    "key": key,
                    "frames": count,
                    "source": source_relpath,
                    "sha256": expected_source_sha,
                }
            )
        if total_frames != manifest.get("frame_count"):
            raise DepthCacheError("cache frame count differs from trajectory records")
        if sha256_bytes(canonical_json_bytes(source_index)) != manifest.get(
            "source_index_sha256"
        ):
            raise DepthCacheError("cache source index hash is invalid")

    def assert_dataset_coverage(
        self, episodes: Iterable[tuple[str | None, int, int]]
    ) -> None:
        """Check every dataset episode and frame count before training starts."""

        episode_list = list(episodes)
        expected = set()
        for split, episode, frame_count in episode_list:
            record = self._resolve_record(split, episode)
            key = str(record["trajectory_key"])
            if int(record["ordered_frame_count"]) != int(frame_count):
                raise DepthCacheError(
                    f"dataset/cache frame-count mismatch for {key}: "
                    f"{frame_count} versus {record['ordered_frame_count']}"
                )
            expected.add(key)
        if len(expected) != len(episode_list):
            raise DepthCacheError("dataset episode mapping contains duplicates")
        if expected != {str(record["trajectory_key"]) for record in self._records.values()}:
            raise DepthCacheError(
                "dataset/cache trajectory coverage differs; partial caches are forbidden"
            )

    def _resolve_record(self, split: str | None, episode: int) -> Mapping[str, Any]:
        if split is not None:
            record = self._records.get((str(split), int(episode)))
            if record is None:
                raise DepthCacheError(f"missing cache trajectory {split}/{episode:05d}")
            return record
        matches = self._records_by_episode.get(int(episode), [])
        if len(matches) != 1:
            raise DepthCacheError(
                f"episode {episode} maps to {len(matches)} cache trajectories; split is required"
            )
        return matches[0]

    def _open(self) -> None:
        if self._database is not None:
            return
        try:
            import lmdb
            import zstandard
        except ImportError as exc:
            raise DepthCacheError(
                "pinned container must provide lmdb and zstandard"
            ) from exc
        self._database = lmdb.open(
            str(self.cache_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=64,
        )
        self._decompressor = zstandard.ZstdDecompressor()

    def _validate_wire_range(self, value: np.ndarray, physical_key: str) -> None:
        minimum = float(value.min())
        maximum = float(value.max())
        binding = self.binding
        if (
            minimum < binding.wire_minimum - 1e-4
            or maximum > binding.wire_maximum + 1e-4
        ):
            raise DepthCacheError(
                f"depth range [{minimum},{maximum}] violates approved wire range "
                f"at {physical_key}"
            )

    def read(
        self,
        *,
        split: str | None,
        episode: int,
        frames: Sequence[int] | range | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        record = self._resolve_record(split, int(episode))
        frame_ids = [int(value) for value in frames]
        count = int(record["ordered_frame_count"])
        if len(set(frame_ids)) != len(frame_ids):
            raise DepthCacheError("depth frame selection contains duplicates")
        if any(frame < 0 or frame >= count for frame in frame_ids):
            raise DepthCacheError(
                f"depth frame selection is outside [0,{count}) for {record['trajectory_key']}"
            )
        self._open()
        values = []
        with self._database.begin(write=False) as transaction:
            for frame in frame_ids:
                physical_key = f"{record['trajectory_key']}/{frame:06d}"
                payload = transaction.get(physical_key.encode("ascii"))
                if payload is None:
                    raise DepthCacheError(f"missing physical depth key {physical_key}")
                try:
                    raw = self._decompressor.decompress(payload)
                except Exception as exc:
                    raise DepthCacheError(
                        f"cannot decompress physical depth key {physical_key}"
                    ) from exc
                value = np.frombuffer(raw, dtype=np.dtype("<f2"))
                if value.size != 224 * 224:
                    raise DepthCacheError(
                        f"wrong payload element count at {physical_key}: {value.size}"
                    )
                value = value.reshape(224, 224)
                if not np.isfinite(value).all():
                    raise DepthCacheError(f"nonfinite depth at {physical_key}")
                self._validate_wire_range(value, physical_key)
                values.append(value.astype(np.float32, copy=True))
        depth = torch.from_numpy(np.stack(values, axis=0))
        validity = torch.ones_like(depth, dtype=torch.float32)
        return depth, validity


class EmpiricalDepthCacheReader(DepthCacheReader):
    """Exact MapAnything PushT cache reader for the empirical/lossy branch only."""

    contract_kind = "empirical_lossy_cache"
    validation_schema = "dinocular-mapanything-cache-validation-v1"

    def _validate_wire_range(self, value: np.ndarray, physical_key: str) -> None:
        minimum = float(value.min())
        maximum = float(value.max())
        if minimum < 0.0 or maximum > 1.0:
            raise DepthCacheError(
                f"depth range [{minimum},{maximum}] violates approved wire range "
                f"at {physical_key}"
            )

    def __init__(
        self,
        *,
        environment: str,
        source_root: str | Path,
        empirical_contract_path: str | Path,
        empirical_contract_sha256: str,
        empirical_runtime_release_path: str | Path,
        empirical_runtime_release_sha256: str,
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        if str(environment) != "pusht":
            raise DepthCacheError("empirical MapAnything cache reader is PushT-only")
        self.environment = "pusht"
        self.source_root = Path(source_root).expanduser().resolve()
        try:
            canonical_contract = load_empirical_depth_contract(
                empirical_contract_path,
                empirical_contract_sha256,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
            )
            self.empirical_runtime_release = load_empirical_runtime_release(
                empirical_runtime_release_path,
                empirical_runtime_release_sha256,
                contract=canonical_contract,
            )
            self.empirical_contract = load_empirical_depth_contract(
                empirical_contract_path,
                empirical_contract_sha256,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                runtime_resolver=self.empirical_runtime_release.resolver,
            )
            resolver = self.empirical_runtime_release.resolver
            resolver.validate_open_path(
                self.empirical_contract.cache_directory,
                artifact="empirical cache directory",
                expect_directory=True,
            )
            resolver.validate_open_path(
                self.empirical_contract.manifest_path,
                artifact="empirical cache manifest",
            )
            resolver.validate_open_path(
                self.empirical_contract.validation_path,
                artifact="empirical validation receipt",
            )
            resolver.validate_open_path(
                self.empirical_contract.data_path,
                artifact="empirical data.mdb",
            )
        except EmpiricalDepthContractError as exc:
            raise DepthCacheError(str(exc)) from exc
        self.cache_dir = self.empirical_contract.cache_directory
        self.manifest_path = self.empirical_contract.manifest_path
        self.validation_path = self.empirical_contract.validation_path
        self.manifest = _load_hashed_json(
            self.manifest_path,
            self.empirical_contract.manifest_sha256,
            "empirical cache manifest",
        )
        self.validation = _load_hashed_json(
            self.validation_path,
            self.empirical_contract.validation_sha256,
            "empirical cache validation receipt",
        )
        self._database = None
        self._decompressor = None
        self._records: dict[tuple[str, int], Mapping[str, Any]] = {}
        self._records_by_episode: dict[int, list[Mapping[str, Any]]] = {}
        self.binding = SimpleNamespace(wire_minimum=0.0, wire_maximum=1.0)
        self._validate_empirical()

    def _validate_empirical(self) -> None:
        manifest = self.manifest
        contract = self.empirical_contract
        if manifest.get("schema") != CACHE_SCHEMA or manifest.get("environment") != "pusht":
            raise DepthCacheError("unsupported empirical PushT cache manifest")
        if manifest.get("manifest_id") != contract.manifest_id:
            raise DepthCacheError("empirical cache manifest ID differs from contract")
        wire = manifest.get("wire_format")
        if not isinstance(wire, Mapping):
            raise DepthCacheError("empirical cache wire format is absent")
        if sha256_bytes(canonical_json_bytes(wire)) != contract.wire_format_sha256:
            raise DepthCacheError("empirical cache canonical wire-format hash differs")
        storage_mismatches = {
            key: (wire.get(key), expected)
            for key, expected in WIRE_STORAGE_FORMAT.items()
            if wire.get(key) != expected
        }
        if storage_mismatches:
            raise DepthCacheError(
                f"empirical cache storage format differs: {storage_mismatches}"
            )
        if manifest.get("closed_before_hash") is not True:
            raise DepthCacheError("empirical cache was not closed before hashing")
        producer = manifest.get("producer")
        if not isinstance(producer, Mapping) or sha256_bytes(
            canonical_json_bytes(producer)
        ) != contract.producer_sha256:
            raise DepthCacheError("empirical cache producer identity differs")
        if manifest.get("source_index_sha256") != contract.source_index_sha256:
            raise DepthCacheError("empirical cache source-index identity differs")
        if manifest.get("data_mdb_sha256") != contract.data_sha256:
            raise DepthCacheError("empirical cache data identity differs")
        if not contract.data_path.is_file():
            raise DepthCacheError("accepted empirical data.mdb is absent")
        try:
            validate_mapanything_receipt(self.validation, manifest)
        except EmpiricalDepthContractError as exc:
            raise DepthCacheError(str(exc)) from exc

        calibration = manifest.get("calibration")
        expected_calibration = contract.manifest["wire_semantics"]["calibration"]
        if not isinstance(calibration, Mapping) or (
            calibration.get("scope") != expected_calibration.get("scope")
            or calibration.get("lo") != expected_calibration.get("lo")
            or calibration.get("hi") != expected_calibration.get("hi")
            or calibration.get("keys_sha256")
            != expected_calibration.get("key_sha256")
        ):
            raise DepthCacheError("empirical cache calibration differs from contract")

        records = manifest.get("trajectories")
        if not isinstance(records, list) or len(records) != manifest.get("trajectory_count"):
            raise DepthCacheError("empirical cache trajectory index is incomplete")
        total_frames = 0
        source_index = []
        for record in records:
            if not isinstance(record, Mapping):
                raise DepthCacheError("empirical trajectory record is not an object")
            key = record.get("trajectory_key")
            if not isinstance(key, str) or key.count("/") != 1:
                raise DepthCacheError(f"invalid empirical trajectory key: {key!r}")
            split, episode_text = key.split("/")
            if split not in {"train", "valid"} or not episode_text.isdigit():
                raise DepthCacheError(f"invalid empirical trajectory key: {key!r}")
            episode = int(episode_text)
            count = record.get("ordered_frame_count")
            if not isinstance(count, int) or count <= 0:
                raise DepthCacheError(f"invalid empirical frame count for {key}")
            if record.get("ordered_output_keys") != [
                f"{key}/{frame:06d}" for frame in range(count)
            ]:
                raise DepthCacheError(f"empirical physical key order differs for {key}")
            source_relpath = record.get("source_path")
            if not isinstance(source_relpath, str) or Path(source_relpath).is_absolute():
                raise DepthCacheError(f"invalid empirical source path for {key}")
            source = (self.source_root / source_relpath).resolve()
            try:
                source.relative_to(self.source_root)
            except ValueError as exc:
                raise DepthCacheError(f"empirical source path escapes dataset root: {source}") from exc
            expected_source_sha = require_sha256(
                record.get("source_video_sha256"), f"empirical source SHA-256 for {key}"
            )
            if not source.is_file():
                raise DepthCacheError(f"accepted empirical source is absent for {key}: {source}")
            record_key = (split, episode)
            if record_key in self._records:
                raise DepthCacheError(f"duplicate empirical cache trajectory {key}")
            self._records[record_key] = record
            self._records_by_episode.setdefault(episode, []).append(record)
            total_frames += count
            source_index.append(
                {
                    "key": key,
                    "frames": count,
                    "source": source_relpath,
                    "sha256": expected_source_sha,
                }
            )
        if total_frames != manifest.get("frame_count"):
            raise DepthCacheError("empirical frame count differs from trajectory records")
        if sha256_bytes(canonical_json_bytes(source_index)) != contract.source_index_sha256:
            raise DepthCacheError("empirical source-index canonical hash is invalid")


def require_complete_depth_arguments(**values: Any) -> bool:
    """Return whether depth is configured, rejecting every partial configuration."""

    present = {name: value is not None for name, value in values.items()}
    if any(present.values()) and not all(present.values()):
        missing = sorted(name for name, is_present in present.items() if not is_present)
        raise DepthCacheError(
            f"partial depth-cache configuration is forbidden; missing {missing}"
        )
    return all(present.values())
