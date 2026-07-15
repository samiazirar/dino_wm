"""Fail-closed P3 training completion and convergence contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CHECKPOINT_HISTORY_SCHEMA = "dino-wm.step-checkpoint-history.v1"
TRAINING_RECORD_SCHEMA = "dino-wm.p3-training-step.v1"
VALIDATION_RECORD_SCHEMA = "dino-wm.p3-heldout-loss.v1"
HELDOUT_ENTRY_SCHEMA = "dino-wm.p3-heldout-example.v1"
HELDOUT_METADATA_SCHEMA = "dino-wm.p3-heldout-manifest.v1"
FINAL_RECEIPT_SCHEMA = "dino-wm.p3-final-acceptance.v1"
PLATEAU_THRESHOLD = 0.02


class P3CompletionError(RuntimeError):
    """A P3 checkpoint, ledger, held-out, or final receipt is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise P3CompletionError(f"value is not canonical finite JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def is_source_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def require_job_id(value: Any, label: str = "SLURM job ID") -> str:
    if not isinstance(value, str) or not value.isdigit():
        raise P3CompletionError(f"{label} must be a numeric string")
    return value


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def immutable_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise P3CompletionError(
                f"immutable file already exists with different bytes: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P3CompletionError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise P3CompletionError(f"JSON root is not an object: {path}")
    return value


def load_jsonl(
    path: str | Path, *, allow_missing: bool = False
) -> list[Mapping[str, Any]]:
    path = Path(path)
    if not path.exists():
        if allow_missing:
            return []
        raise P3CompletionError(f"missing JSONL file: {path}")
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise P3CompletionError(
                    f"unterminated JSONL record at {path}:{line_number}"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise P3CompletionError(
                    f"invalid JSONL record at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise P3CompletionError(
                    f"JSONL record is not an object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise P3CompletionError(f"short JSONL append to {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def percent_step(target_steps: int, percent: int) -> int:
    if target_steps < 100:
        raise P3CompletionError(
            "target_steps must be at least 100 for unique integer-percent points"
        )
    if percent < 1 or percent > 100:
        raise P3CompletionError("progress percent must be in 1..100")
    return (int(target_steps) * int(percent) + 99) // 100


def percent_step_map(target_steps: int) -> Mapping[int, int]:
    result = {percent: percent_step(target_steps, percent) for percent in range(1, 101)}
    if len(set(result.values())) != 100:
        raise P3CompletionError("integer-percent mapping contains duplicate steps")
    if result[100] != int(target_steps):
        raise P3CompletionError("100 percent does not map to the exact target")
    return result


def percents_at_step(target_steps: int, global_step: int) -> list[int]:
    return [
        percent
        for percent, step in percent_step_map(target_steps).items()
        if step == int(global_step)
    ]


def _record_digest(record: Mapping[str, Any]) -> str:
    value = dict(record)
    value.pop("record_sha256", None)
    return sha256_bytes(canonical_json_bytes(value))


def load_checkpoint_history(path: str | Path) -> list[Mapping[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    value = load_json(path)
    if value.get("schema") != CHECKPOINT_HISTORY_SCHEMA or not isinstance(
        value.get("records"), list
    ):
        raise P3CompletionError(f"malformed checkpoint history: {path}")
    records = value["records"]
    prior = None
    for sequence, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise P3CompletionError("checkpoint history record is not an object")
        if record.get("sequence") != sequence:
            raise P3CompletionError("checkpoint history sequence is not contiguous")
        step = record.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise P3CompletionError("checkpoint history has an invalid step")
        if sequence and step < int(records[sequence - 1]["step"]):
            raise P3CompletionError("checkpoint history step order moved backwards")
        if sequence and step == int(records[sequence - 1]["step"]):
            previous = records[sequence - 1]
            for field in (
                "filename",
                "checkpoint_sha256",
                "content_sha256",
                "source_commit",
                "immutable_run_card_sha256",
                "dataset_order_sha256",
            ):
                if record.get(field) != previous.get(field):
                    raise P3CompletionError(
                        "same-step checkpoint trigger has divergent content"
                    )
        if record.get("previous_record_sha256") != prior:
            raise P3CompletionError("checkpoint history hash chain is broken")
        if record.get("record_sha256") != _record_digest(record):
            raise P3CompletionError("checkpoint history record hash differs")
        if not is_sha256(record.get("checkpoint_sha256")) or not is_sha256(
            record.get("content_sha256")
        ):
            raise P3CompletionError("checkpoint history has an invalid hash")
        reasons = record.get("reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or reasons != sorted(set(reasons))
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise P3CompletionError("checkpoint history reasons are invalid")
        prior = record["record_sha256"]
    return records


def append_checkpoint_history(
    path: str | Path,
    *,
    step: int,
    filename: str,
    checkpoint_sha256: str,
    content_sha256: str,
    reasons: Sequence[str],
    source_commit: str | None,
    immutable_run_card_sha256: str | None,
    dataset_order_sha256: str | None,
) -> Mapping[str, Any]:
    path = Path(path)
    records = load_checkpoint_history(path)
    normalized_reasons = sorted(set(str(reason) for reason in reasons))
    if not normalized_reasons or any(not reason for reason in normalized_reasons):
        raise P3CompletionError("checkpoint save reasons must be nonempty")
    same_step = [record for record in records if record["step"] == int(step)]
    for existing in same_step:
        if any(
            existing.get(field) != expected
            for field, expected in (
                ("filename", filename),
                ("checkpoint_sha256", checkpoint_sha256),
                ("content_sha256", content_sha256),
                ("source_commit", source_commit),
                ("immutable_run_card_sha256", immutable_run_card_sha256),
                ("dataset_order_sha256", dataset_order_sha256),
            )
        ):
            raise P3CompletionError(
                f"checkpoint step {step} would create a divergent duplicate"
            )
        if existing.get("reasons") == normalized_reasons:
            return existing
    if records and int(step) < int(records[-1]["step"]):
        raise P3CompletionError("checkpoint history step order moved backwards")
    record = {
        "sequence": len(records),
        "step": int(step),
        "filename": filename,
        "checkpoint_sha256": checkpoint_sha256,
        "content_sha256": content_sha256,
        "reasons": normalized_reasons,
        "source_commit": source_commit,
        "immutable_run_card_sha256": immutable_run_card_sha256,
        "dataset_order_sha256": dataset_order_sha256,
        "previous_record_sha256": records[-1]["record_sha256"] if records else None,
    }
    record["record_sha256"] = _record_digest(record)
    records.append(record)
    atomic_write_json(
        path,
        {"schema": CHECKPOINT_HISTORY_SCHEMA, "records": records},
    )
    return record


def checkpoint_reference(
    record: Mapping[str, Any], directory: str | Path
) -> Mapping[str, Any]:
    return {
        "step": int(record["step"]),
        "path": str(Path(directory) / str(record["filename"])),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "history_record_sha256": record["record_sha256"],
    }


def _validate_checkpoint_reference(value: Any, *, global_step: int) -> None:
    if not isinstance(value, Mapping):
        raise P3CompletionError("training record has no checkpoint reference")
    step = value.get("step")
    if (
        not isinstance(step, int)
        or isinstance(step, bool)
        or not 0 <= step <= global_step
    ):
        raise P3CompletionError("training checkpoint reference step is invalid")
    if not is_sha256(value.get("checkpoint_sha256")) or not is_sha256(
        value.get("history_record_sha256")
    ):
        raise P3CompletionError("training checkpoint reference hash is invalid")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise P3CompletionError("training checkpoint reference path is invalid")


def _stable_training_record(value: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(value)
    result.pop("slurm_job_id", None)
    result.pop("record_sha256", None)
    result.pop("previous_record_sha256", None)
    return result


def validate_training_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    immutable_run_card_sha256: str,
    dataset_order_sha256: str,
    target_steps: int,
    require_complete: bool = False,
) -> None:
    prior = None
    seen = set()
    expected_next = 1
    percent_steps = set(percent_step_map(target_steps).values())
    config_sha256 = rows[0].get("config_sha256") if rows else None
    for row in rows:
        if row.get("schema") != TRAINING_RECORD_SCHEMA:
            raise P3CompletionError("training ledger has an unknown schema")
        step = row.get("global_step")
        if not isinstance(step, int) or isinstance(step, bool) or step != expected_next:
            raise P3CompletionError("training ledger has a gap or out-of-order step")
        if step in seen:
            raise P3CompletionError("training ledger duplicates a step ID")
        seen.add(step)
        expected_next += 1
        if (
            row.get("source_commit") != source_commit
            or row.get("immutable_run_card_sha256") != immutable_run_card_sha256
            or row.get("dataset_order_sha256") != dataset_order_sha256
        ):
            raise P3CompletionError("training ledger provenance drift")
        if (
            not is_sha256(row.get("config_sha256"))
            or row.get("config_sha256") != config_sha256
        ):
            raise P3CompletionError("training ledger configuration provenance drift")
        if not isinstance(row.get("loss"), (int, float)) or not math.isfinite(
            float(row["loss"])
        ):
            raise P3CompletionError("training ledger contains nonfinite loss")
        require_job_id(row.get("slurm_job_id"))
        sampler = row.get("sampler")
        if not isinstance(sampler, Mapping) or sampler.get("next_step") != step:
            raise P3CompletionError("training ledger sampler cursor differs")
        _validate_checkpoint_reference(row.get("checkpoint"), global_step=step)
        if step in percent_steps and row["checkpoint"]["step"] != step:
            raise P3CompletionError("percent training step lacks its exact checkpoint")
        for field in ("parameter_sha256", "optimizer_sha256", "scheduler_sha256"):
            value = row.get(field)
            if value is not None and not is_sha256(value):
                raise P3CompletionError(f"training ledger has invalid {field}")
        if row.get("previous_record_sha256") != prior:
            raise P3CompletionError("training ledger hash chain is broken")
        if row.get("record_sha256") != _record_digest(row):
            raise P3CompletionError("training ledger record hash differs")
        prior = row["record_sha256"]
    if len(rows) > target_steps:
        raise P3CompletionError("training ledger exceeds target steps")
    if require_complete and len(rows) != target_steps:
        raise P3CompletionError("training ledger does not cover every target step")


def append_training_record(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    target_steps: int,
    known_rows: list[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    path = Path(path)
    rows = load_jsonl(path, allow_missing=True) if known_rows is None else known_rows
    step = int(value.get("global_step", -1))
    existing = next((row for row in rows if row.get("global_step") == step), None)
    if existing is not None:
        if _stable_training_record(existing) != _stable_training_record(value):
            raise P3CompletionError(
                f"replayed optimizer step {step} differs from its exactly-once record"
            )
        return existing
    if step != len(rows) + 1:
        raise P3CompletionError("new training record would create a gap")
    if value.get("schema") != TRAINING_RECORD_SCHEMA:
        raise P3CompletionError("new training record has an unknown schema")
    if not is_source_commit(value.get("source_commit")) or not all(
        is_sha256(value.get(field))
        for field in (
            "immutable_run_card_sha256",
            "dataset_order_sha256",
            "config_sha256",
        )
    ):
        raise P3CompletionError("new training record has invalid provenance")
    if not isinstance(value.get("loss"), (int, float)) or not math.isfinite(
        float(value["loss"])
    ):
        raise P3CompletionError("new training record contains nonfinite loss")
    require_job_id(value.get("slurm_job_id"))
    if value.get("sampler", {}).get("next_step") != step:
        raise P3CompletionError("new training record sampler cursor differs")
    _validate_checkpoint_reference(value.get("checkpoint"), global_step=step)
    if (
        step in set(percent_step_map(target_steps).values())
        and value["checkpoint"]["step"] != step
    ):
        raise P3CompletionError("new percent step lacks its exact checkpoint")
    for field in ("parameter_sha256", "optimizer_sha256", "scheduler_sha256"):
        if value.get(field) is not None and not is_sha256(value[field]):
            raise P3CompletionError(f"new training record has invalid {field}")
    if rows:
        first = rows[0]
        for field in (
            "source_commit",
            "immutable_run_card_sha256",
            "dataset_order_sha256",
            "config_sha256",
        ):
            if value.get(field) != first.get(field):
                raise P3CompletionError("new training record has provenance drift")
    record = dict(value)
    record["previous_record_sha256"] = rows[-1]["record_sha256"] if rows else None
    record["record_sha256"] = _record_digest(record)
    append_jsonl(path, record)
    if known_rows is not None:
        known_rows.append(record)
    return record


def _slice_source_episode(
    sliced_dataset: Any, local_episode: int, partition: str
) -> tuple[str, int]:
    base = sliced_dataset.dataset
    indices = getattr(base, "indices", None)
    if indices is None:
        return partition, int(local_episode)
    return "shared", int(indices[int(local_episode)])


def runtime_slice_entries(
    sliced_dataset: Any,
    *,
    environment: str,
    partition: str,
) -> list[Mapping[str, Any]]:
    entries = []
    frameskip = int(sliced_dataset.frameskip)
    num_frames = int(sliced_dataset.num_frames)
    for dataset_index, raw_slice in enumerate(sliced_dataset.slices):
        local_episode, start, end = (int(value) for value in raw_slice)
        namespace, episode = _slice_source_episode(
            sliced_dataset, local_episode, partition
        )
        key = (
            f"{environment}/{namespace}/{episode:06d}/"
            f"{start:06d}-{end:06d}-f{frameskip}"
        )
        entries.append(
            {
                "schema": HELDOUT_ENTRY_SCHEMA,
                "environment": environment,
                "key": key,
                "source_partition": namespace,
                "episode": episode,
                "start": start,
                "end": end,
                "frameskip": frameskip,
                "num_frames": num_frames,
                "dataset_index": dataset_index,
            }
        )
    return sorted(entries, key=lambda item: str(item["key"]))


def _manifest_entry(value: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(value)
    result.pop("dataset_index", None)
    return result


def _unique_entries(
    entries: Iterable[Mapping[str, Any]], label: str
) -> list[Mapping[str, Any]]:
    result = [_manifest_entry(entry) for entry in entries]
    keys = [entry.get("key") for entry in result]
    if not result or any(not isinstance(key, str) or not key for key in keys):
        raise P3CompletionError(f"{label} split has an empty or invalid key")
    if len(keys) != len(set(keys)):
        raise P3CompletionError(f"{label} split duplicates an example key")
    return sorted(result, key=lambda item: str(item["key"]))


def materialize_heldout_manifest(
    *,
    environment: str,
    training_entries: Iterable[Mapping[str, Any]],
    validation_entries: Iterable[Mapping[str, Any]],
    data_manifest_path: str | Path,
    source_commit: str,
    out_path: str | Path,
) -> Mapping[str, Any]:
    if not is_source_commit(source_commit):
        raise P3CompletionError("held-out manifest source commit is invalid")
    data_manifest_path = Path(data_manifest_path).resolve()
    if not data_manifest_path.is_file():
        raise P3CompletionError(f"data manifest is absent: {data_manifest_path}")
    train = _unique_entries(training_entries, "training")
    valid = _unique_entries(validation_entries, "validation")
    if any(entry.get("environment") != environment for entry in [*train, *valid]):
        raise P3CompletionError("split entry environment differs from the manifest")
    train_keys = {str(entry["key"]) for entry in train}
    valid_keys = {str(entry["key"]) for entry in valid}
    if train_keys & valid_keys:
        raise P3CompletionError("held-out validation keys leak into training")
    train_episodes = {
        (entry["source_partition"], int(entry["episode"])) for entry in train
    }
    valid_episodes = {
        (entry["source_partition"], int(entry["episode"])) for entry in valid
    }
    if train_episodes & valid_episodes:
        raise P3CompletionError("held-out validation episodes leak into training")
    out_path = Path(out_path)
    text = "".join(
        canonical_json_bytes(entry).decode("utf-8") + "\n" for entry in valid
    )
    immutable_write_text(out_path, text)
    manifest_sha256 = sha256_file(out_path)
    split_contract = {
        "training_keys": sorted(train_keys),
        "validation_keys": sorted(valid_keys),
    }
    metadata = {
        "schema": HELDOUT_METADATA_SCHEMA,
        "environment": environment,
        "selection": "all_validation_examples",
        "entry_count": len(valid),
        "manifest_path": str(out_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "data_manifest_path": str(data_manifest_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "split_sha256": sha256_bytes(canonical_json_bytes(split_contract)),
        "training_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(train_keys))),
        "validation_keys_sha256": sha256_bytes(
            canonical_json_bytes(sorted(valid_keys))
        ),
        "source_commit": source_commit,
        "rounding_rule": "ceil(target_steps*percent/100)",
    }
    metadata_path = out_path.with_suffix(".meta.json")
    immutable_write_text(
        metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return {
        "path": str(out_path.resolve()),
        "sha256": manifest_sha256,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "data_manifest_sha256": metadata["data_manifest_sha256"],
        "split_sha256": metadata["split_sha256"],
        "selection": metadata["selection"],
    }


def validate_runtime_heldout_manifest(
    record: Mapping[str, Any],
    *,
    environment: str,
    source_commit: str,
    training_entries: Iterable[Mapping[str, Any]],
    validation_entries: Iterable[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[int], Mapping[str, Any]]:
    path = Path(str(record.get("path")))
    metadata_path = Path(str(record.get("metadata_path")))
    if (
        not path.is_file()
        or sha256_file(path) != record.get("sha256")
        or not metadata_path.is_file()
        or sha256_file(metadata_path) != record.get("metadata_sha256")
    ):
        raise P3CompletionError("held-out manifest or metadata hash differs")
    metadata = load_json(metadata_path)
    rows = load_jsonl(path)
    train = _unique_entries(training_entries, "runtime training")
    runtime_valid = list(validation_entries)
    valid = _unique_entries(runtime_valid, "runtime validation")
    manifest_rows = _unique_entries(rows, "held-out manifest")
    if manifest_rows != valid:
        raise P3CompletionError(
            "held-out manifest differs from the exact validation split"
        )
    train_episodes = {
        (entry["source_partition"], int(entry["episode"])) for entry in train
    }
    valid_episodes = {
        (entry["source_partition"], int(entry["episode"])) for entry in valid
    }
    if train_episodes & valid_episodes:
        raise P3CompletionError("runtime held-out validation leaks into training")
    split_contract = {
        "training_keys": sorted(str(entry["key"]) for entry in train),
        "validation_keys": sorted(str(entry["key"]) for entry in valid),
    }
    if (
        metadata.get("schema") != HELDOUT_METADATA_SCHEMA
        or metadata.get("environment") != environment
        or metadata.get("selection") != "all_validation_examples"
        or metadata.get("source_commit") != source_commit
        or metadata.get("manifest_sha256") != record.get("sha256")
        or metadata.get("entry_count") != len(valid)
        or metadata.get("split_sha256")
        != sha256_bytes(canonical_json_bytes(split_contract))
        or metadata.get("data_manifest_sha256") != record.get("data_manifest_sha256")
        or metadata.get("split_sha256") != record.get("split_sha256")
    ):
        raise P3CompletionError("held-out manifest metadata differs from runtime")
    index_by_key = {
        str(entry["key"]): int(entry["dataset_index"]) for entry in runtime_valid
    }
    indices = [index_by_key[str(row["key"])] for row in manifest_rows]
    return manifest_rows, indices, metadata


def _stable_validation_record(value: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(value)
    result.pop("slurm_job_id", None)
    result.pop("record_sha256", None)
    result.pop("previous_record_sha256", None)
    return result


def validate_validation_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_steps: int,
    immutable_run_card_sha256: str,
    manifest_sha256: str,
    require_complete: bool = False,
) -> None:
    mapping = percent_step_map(target_steps)
    seen = set()
    prior = None
    immutable_fields = (
        "source_commit",
        "config_sha256",
        "container_sha256",
        "manifest_sha256",
        "data_manifest_sha256",
        "split_sha256",
        "immutable_run_card_sha256",
        "depth_producer_sha256",
        "depth_cache_manifest_sha256",
        "depth_native_contract_sha256",
        "depth_validation_sha256",
        "depth_checkpoint_sha256",
    )
    immutable = (
        {field: rows[0].get(field) for field in immutable_fields} if rows else {}
    )
    for row in rows:
        if row.get("schema") != VALIDATION_RECORD_SCHEMA:
            raise P3CompletionError("held-out loss ledger has an unknown schema")
        percent = row.get("percent")
        if (
            not isinstance(percent, int)
            or isinstance(percent, bool)
            or percent not in mapping
        ):
            raise P3CompletionError("held-out loss ledger has an invalid percent")
        if percent in seen:
            raise P3CompletionError("held-out loss ledger duplicates a percent")
        seen.add(percent)
        if row.get("global_step") != mapping[percent]:
            raise P3CompletionError(
                "held-out loss uses the wrong percent-to-step mapping"
            )
        if (
            row.get("immutable_run_card_sha256") != immutable_run_card_sha256
            or row.get("manifest_sha256") != manifest_sha256
        ):
            raise P3CompletionError("held-out loss provenance drift")
        if any(row.get(field) != immutable[field] for field in immutable_fields):
            raise P3CompletionError("held-out loss immutable provenance drift")
        numerator = row.get("loss_numerator")
        count = row.get("element_count")
        mean = row.get("mean_loss")
        if (
            not isinstance(numerator, (int, float))
            or not math.isfinite(float(numerator))
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or not isinstance(mean, (int, float))
            or not math.isfinite(float(mean))
            or not math.isclose(
                float(mean), float(numerator) / count, rel_tol=1e-12, abs_tol=1e-15
            )
        ):
            raise P3CompletionError(
                "held-out loss is nonfinite, empty, or inconsistent"
            )
        require_job_id(row.get("slurm_job_id"))
        for field in (
            "source_commit",
            "config_sha256",
            "container_sha256",
            "model_sha256",
            "checkpoint_sha256",
            "data_manifest_sha256",
            "split_sha256",
        ):
            value = row.get(field)
            if field == "source_commit":
                if not is_source_commit(value):
                    raise P3CompletionError(f"held-out loss has invalid {field}")
            elif not is_sha256(value):
                raise P3CompletionError(f"held-out loss has invalid {field}")
        if not is_sha256(row.get("checkpoint_history_record_sha256")):
            raise P3CompletionError(
                "held-out loss has invalid checkpoint history reference"
            )
        depth_values = [
            row.get(field) for field in immutable_fields if field.startswith("depth_")
        ]
        if any(value is not None for value in depth_values) and not all(
            is_sha256(value) for value in depth_values
        ):
            raise P3CompletionError("held-out loss has incomplete depth provenance")
        if row.get("previous_record_sha256") != prior:
            raise P3CompletionError("held-out loss ledger hash chain is broken")
        if row.get("record_sha256") != _record_digest(row):
            raise P3CompletionError("held-out loss record hash differs")
        prior = row["record_sha256"]
    if require_complete and seen != set(range(1, 101)):
        raise P3CompletionError("held-out loss ledger lacks exact 1..100 coverage")


def append_validation_record(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    target_steps: int,
) -> Mapping[str, Any]:
    path = Path(path)
    rows = load_jsonl(path, allow_missing=True)
    percent = int(value.get("percent", -1))
    existing = next((row for row in rows if row.get("percent") == percent), None)
    if existing is not None:
        if _stable_validation_record(existing) != _stable_validation_record(value):
            raise P3CompletionError(
                f"resumed held-out percent {percent} differs from existing evidence"
            )
        return existing
    expected_missing = sorted(
        set(range(1, 101)) - {int(row["percent"]) for row in rows}
    )
    if not expected_missing or percent != expected_missing[0]:
        raise P3CompletionError(
            "held-out loss append would create missing progress points"
        )
    record = dict(value)
    record["previous_record_sha256"] = rows[-1]["record_sha256"] if rows else None
    record["record_sha256"] = _record_digest(record)
    validate_validation_records(
        [*rows, record],
        target_steps=target_steps,
        immutable_run_card_sha256=str(record["immutable_run_card_sha256"]),
        manifest_sha256=str(record["manifest_sha256"]),
    )
    append_jsonl(path, record)
    return record


def validate_final_receipt(
    receipt: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> None:
    if (
        receipt.get("schema") != FINAL_RECEIPT_SCHEMA
        or receipt.get("state") != "PASS"
        or receipt.get("fresh_model_process") is not True
    ):
        raise P3CompletionError("final acceptance receipt is not a fresh-process PASS")
    pid = receipt.get("process_id")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise P3CompletionError("final acceptance process ID is invalid")
    require_job_id(receipt.get("slurm_job_id"))
    target = receipt.get("target_steps")
    if (
        not isinstance(target, int)
        or isinstance(target, bool)
        or receipt.get("global_step") != target
        or receipt.get("sampler", {}).get("next_step") != target
    ):
        raise P3CompletionError("final acceptance target or sampler is not exact")
    for field in (
        "immutable_run_card_sha256",
        "config_sha256",
        "container_sha256",
        "checkpoint_sha256",
        "parameter_sha256",
        "optimizer_sha256",
        "scheduler_sha256",
        "rng_sha256",
        "manifest_sha256",
        "data_manifest_sha256",
        "split_sha256",
        "training_ledger_sha256",
        "validation_ledger_sha256",
        "checkpoint_history_sha256",
        "dataset_order_sha256",
    ):
        if not is_sha256(receipt.get(field)):
            raise P3CompletionError(f"final acceptance has invalid {field}")
    if not is_sha256(receipt.get("checkpoint_history_record_sha256")):
        raise P3CompletionError(
            "final acceptance has invalid checkpoint history record"
        )
    if not isinstance(receipt.get("checkpoint"), str) or not receipt["checkpoint"]:
        raise P3CompletionError("final acceptance checkpoint path is invalid")
    depth_fields = (
        "depth_producer_sha256",
        "depth_cache_manifest_sha256",
        "depth_native_contract_sha256",
        "depth_validation_sha256",
        "depth_checkpoint_sha256",
    )
    depth_values = [receipt.get(field) for field in depth_fields]
    if any(value is not None for value in depth_values) and not all(
        is_sha256(value) for value in depth_values
    ):
        raise P3CompletionError("final acceptance has incomplete depth provenance")
    if not is_source_commit(receipt.get("source_commit")):
        raise P3CompletionError("final acceptance source commit is invalid")
    validation = receipt.get("validation_batch")
    if not isinstance(validation, Mapping):
        raise P3CompletionError("final acceptance validation batch is absent")
    numerator = validation.get("loss_numerator")
    count = validation.get("element_count")
    mean = validation.get("mean_loss")
    if (
        not isinstance(numerator, (int, float))
        or not math.isfinite(float(numerator))
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or not isinstance(mean, (int, float))
        or not math.isfinite(float(mean))
        or not math.isclose(
            float(mean), float(numerator) / count, rel_tol=1e-12, abs_tol=1e-15
        )
    ):
        raise P3CompletionError("final acceptance validation loss is invalid")
    if expected is not None:
        differing = [
            key for key, value in expected.items() if receipt.get(key) != value
        ]
        if differing:
            raise P3CompletionError(
                f"final acceptance receipt differs: {sorted(differing)}"
            )


def write_final_receipt(path: str | Path, receipt: Mapping[str, Any]) -> str:
    validate_final_receipt(receipt)
    text = json.dumps(dict(receipt), indent=2, sort_keys=True, allow_nan=False) + "\n"
    immutable_write_text(path, text)
    return sha256_file(path)


def load_final_receipt(
    run_dir: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Path, str]:
    path = Path(run_dir) / "final_acceptance.json"
    receipt = load_json(path)
    validate_final_receipt(receipt, expected=expected)
    return receipt, path, sha256_file(path)


def plateau_verdict(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_percent = {int(row["percent"]): float(row["mean_loss"]) for row in rows}
    if set(by_percent) != set(range(1, 101)):
        raise P3CompletionError("plateau audit requires exact 1..100 loss coverage")
    early = sum(by_percent[percent] for percent in range(76, 81)) / 5.0
    late = sum(by_percent[percent] for percent in range(96, 101)) / 5.0
    if not math.isfinite(early) or not math.isfinite(late) or early <= 0.0:
        raise P3CompletionError("plateau audit requires a finite positive early mean")
    relative_change = abs(late - early) / early
    plateaued = relative_change <= PLATEAU_THRESHOLD or math.isclose(
        relative_change, PLATEAU_THRESHOLD, rel_tol=0.0, abs_tol=1e-15
    )
    return {
        "early_mean_76_80": early,
        "late_mean_96_100": late,
        "relative_absolute_change": relative_change,
        "threshold": PLATEAU_THRESHOLD,
        "plateaued": plateaued,
    }


def comparison_verdict(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    return (
        "optimization-conclusive"
        if left.get("plateaued") is True and right.get("plateaued") is True
        else "optimization-inconclusive"
    )
