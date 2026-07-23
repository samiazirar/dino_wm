"""Fail-closed P3 training completion and convergence contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from empirical_depth_contract import (
    EmpiricalDepthContractError,
    validate_empirical_provenance,
)


CHECKPOINT_HISTORY_SCHEMA = "dino-wm.step-checkpoint-history.v1"
TRAINING_RECORD_SCHEMA = "dino-wm.p3-training-step.v1"
TRAINING_TAIL_INDEX_SCHEMA = "dino-wm.p3-training-tail-index.v3"
VALIDATION_RECORD_SCHEMA = "dino-wm.p3-heldout-loss.v1"
HELDOUT_ENTRY_SCHEMA = "dino-wm.p3-heldout-example.v1"
HELDOUT_METADATA_SCHEMA = "dino-wm.p3-heldout-manifest.v1"
FINAL_RECEIPT_SCHEMA = "dino-wm.p3-final-acceptance.v1"
PLATEAU_THRESHOLD = 0.02
HELDOUT_ROUNDING_RULE = "ceil(target_steps*percent/100)"


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


def is_process_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def require_job_id(value: Any, label: str = "SLURM job ID") -> str:
    if not isinstance(value, str) or not value.isdigit():
        raise P3CompletionError(f"{label} must be a numeric string")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    _fsync_directory(path.parent)


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
    flags = os.O_WRONLY | os.O_APPEND
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
    except FileExistsError:
        descriptor = os.open(path, flags)
        created = False
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise P3CompletionError(f"short JSONL append to {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if created:
        _fsync_directory(path.parent)


def percent_step(target_steps: int, percent: int) -> int:
    if not isinstance(target_steps, int) or isinstance(target_steps, bool):
        raise P3CompletionError("target_steps must be an integer")
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
    if same_step and same_step[-1].get("reasons") == normalized_reasons:
        return same_step[-1]
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


def validate_checkpoint_evidence_bindings(
    history: Sequence[Mapping[str, Any]],
    *,
    directory: str | Path,
    source_commit: str,
    immutable_run_card_sha256: str,
    dataset_order_sha256: str,
    training_rows: Sequence[Mapping[str, Any]] = (),
    validation_rows: Sequence[Mapping[str, Any]] = (),
    final_receipt: Mapping[str, Any] | None = None,
) -> None:
    by_record_sha256 = {record["record_sha256"]: record for record in history}
    if len(by_record_sha256) != len(history):
        raise P3CompletionError("checkpoint history duplicates a record hash")
    for record in history:
        if (
            record.get("source_commit") != source_commit
            or record.get("immutable_run_card_sha256") != immutable_run_card_sha256
            or record.get("dataset_order_sha256") != dataset_order_sha256
        ):
            raise P3CompletionError("checkpoint history provenance drift")

    directory = Path(directory).resolve()

    def resolve_reference(
        history_record_sha256: Any,
        checkpoint_sha256: Any,
        *,
        expected_step: int,
        expected_path: Any = None,
    ) -> Mapping[str, Any]:
        record = by_record_sha256.get(history_record_sha256)
        if record is None:
            raise P3CompletionError(
                "evidence references an unknown checkpoint history record"
            )
        if (
            record.get("step") != expected_step
            or record.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise P3CompletionError(
                "checkpoint evidence differs from its history record"
            )
        if (
            expected_path is not None
            and Path(str(expected_path)).resolve()
            != (directory / str(record["filename"])).resolve()
        ):
            raise P3CompletionError("checkpoint evidence path differs from history")
        return record

    for row in training_rows:
        reference = row["checkpoint"]
        resolve_reference(
            reference["history_record_sha256"],
            reference["checkpoint_sha256"],
            expected_step=int(reference["step"]),
            expected_path=reference["path"],
        )
    for row in validation_rows:
        resolve_reference(
            row["checkpoint_history_record_sha256"],
            row["checkpoint_sha256"],
            expected_step=int(row["global_step"]),
        )
    if final_receipt is not None:
        resolve_reference(
            final_receipt["checkpoint_history_record_sha256"],
            final_receipt["checkpoint_sha256"],
            expected_step=int(final_receipt["global_step"]),
            expected_path=final_receipt["checkpoint"],
        )


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


_TRAINING_SAMPLER_FIELDS = frozenset(
    {
        "dataset_size",
        "batch_size",
        "steps_per_epoch",
        "next_step",
        "completed_epochs",
        "next_batch_in_epoch",
        "next_sample_in_epoch",
        "dataset_order_sha256",
    }
)


def _validate_training_cursor(
    row: Mapping[str, Any],
    *,
    step: int,
    dataset_order_sha256: str,
    expected_dataset_size: int | None = None,
    expected_batch_size: int | None = None,
) -> tuple[int, int, int]:
    sampler = row.get("sampler")
    if not isinstance(sampler, Mapping) or set(sampler) != _TRAINING_SAMPLER_FIELDS:
        raise P3CompletionError("training ledger sampler fields differ")
    dataset_size = sampler.get("dataset_size")
    batch_size = sampler.get("batch_size")
    steps_per_epoch = sampler.get("steps_per_epoch")
    if (
        not isinstance(dataset_size, int)
        or isinstance(dataset_size, bool)
        or dataset_size <= 0
        or not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or not isinstance(steps_per_epoch, int)
        or isinstance(steps_per_epoch, bool)
        or steps_per_epoch != (dataset_size + batch_size - 1) // batch_size
        or (expected_dataset_size is not None and dataset_size != expected_dataset_size)
        or (expected_batch_size is not None and batch_size != expected_batch_size)
    ):
        raise P3CompletionError("training ledger dataset geometry differs")
    completed_epochs = step // steps_per_epoch
    next_batch = step % steps_per_epoch
    next_sample = min(next_batch * batch_size, dataset_size)
    cursor_values = (
        row.get("completed_epochs"),
        sampler.get("next_step"),
        sampler.get("completed_epochs"),
        sampler.get("next_batch_in_epoch"),
        sampler.get("next_sample_in_epoch"),
    )
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in cursor_values
        )
        or row.get("completed_epochs") != completed_epochs
        or sampler.get("next_step") != step
        or sampler.get("completed_epochs") != completed_epochs
        or sampler.get("next_batch_in_epoch") != next_batch
        or sampler.get("next_sample_in_epoch") != next_sample
        or sampler.get("dataset_order_sha256") != dataset_order_sha256
    ):
        raise P3CompletionError("training ledger epoch or sampler cursor differs")
    return dataset_size, batch_size, steps_per_epoch


def validate_final_sampler(
    sampler: Mapping[str, Any],
    *,
    target_steps: int,
    dataset_order_sha256: str,
    expected_dataset_size: int | None = None,
    expected_batch_size: int | None = None,
) -> None:
    """Validate an exact terminal sampler cursor against immutable geometry."""
    _validate_training_cursor(
        {"sampler": sampler, "completed_epochs": sampler.get("completed_epochs")},
        step=target_steps,
        dataset_order_sha256=dataset_order_sha256,
        expected_dataset_size=expected_dataset_size,
        expected_batch_size=expected_batch_size,
    )


def validate_training_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    immutable_run_card_sha256: str,
    dataset_order_sha256: str,
    target_steps: int,
    expected_dataset_size: int | None = None,
    expected_batch_size: int | None = None,
    require_complete: bool = False,
) -> None:
    if (expected_dataset_size is None) != (expected_batch_size is None):
        raise P3CompletionError("training ledger expected geometry is incomplete")
    if expected_dataset_size is not None and (
        not isinstance(expected_dataset_size, int)
        or isinstance(expected_dataset_size, bool)
        or expected_dataset_size <= 0
        or not isinstance(expected_batch_size, int)
        or isinstance(expected_batch_size, bool)
        or expected_batch_size <= 0
    ):
        raise P3CompletionError("training ledger expected geometry is invalid")
    prior = None
    seen = set()
    expected_next = 1
    percent_steps = set(percent_step_map(target_steps).values())
    config_sha256 = rows[0].get("config_sha256") if rows else None
    dataset_size = expected_dataset_size
    batch_size = expected_batch_size
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
        if not is_finite_number(row.get("loss")):
            raise P3CompletionError("training ledger contains nonfinite loss")
        require_job_id(row.get("slurm_job_id"))
        row_dataset_size, row_batch_size, _steps_per_epoch = _validate_training_cursor(
            row,
            step=step,
            dataset_order_sha256=dataset_order_sha256,
            expected_dataset_size=dataset_size,
            expected_batch_size=batch_size,
        )
        dataset_size = row_dataset_size
        batch_size = row_batch_size
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


def training_tail_index_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}.tail.json")


def _training_tail_index_record(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    immutable_run_card_sha256: str,
    dataset_order_sha256: str,
    config_sha256: str,
    dataset_size: int,
    batch_size: int,
    target_steps: int,
) -> Mapping[str, Any]:
    size = path.stat().st_size if path.exists() else 0
    tail_bytes = canonical_json_bytes(rows[-1]) + b"\n" if rows else None
    tail_size = len(tail_bytes) if tail_bytes is not None else None
    tail_offset = size - tail_size if tail_size is not None else None
    if tail_bytes is not None:
        if tail_offset is None or tail_offset < 0:
            raise P3CompletionError("training ledger is shorter than its tail record")
        with path.open("rb") as handle:
            handle.seek(tail_offset)
            if handle.read(tail_size) != tail_bytes:
                raise P3CompletionError(
                    "training ledger tail bytes are not canonical or differ"
                )
    return {
        "schema": TRAINING_TAIL_INDEX_SCHEMA,
        "ledger_path": str(path.resolve()),
        "ledger_size_bytes": size,
        "record_count": len(rows),
        "next_step": len(rows) + 1,
        "tail_record_sha256": rows[-1]["record_sha256"] if rows else None,
        "tail_offset_bytes": tail_offset,
        "tail_size_bytes": tail_size,
        "tail_line_sha256": sha256_bytes(tail_bytes)
        if tail_bytes is not None
        else None,
        "source_commit": source_commit,
        "immutable_run_card_sha256": immutable_run_card_sha256,
        "dataset_order_sha256": dataset_order_sha256,
        "config_sha256": config_sha256,
        "dataset_size": int(dataset_size),
        "batch_size": int(batch_size),
        "steps_per_epoch": (int(dataset_size) + int(batch_size) - 1) // int(batch_size),
        "target_steps": int(target_steps),
    }


def initialize_training_tail_index(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    immutable_run_card_sha256: str,
    dataset_order_sha256: str,
    config_sha256: str,
    dataset_size: int,
    batch_size: int,
    target_steps: int,
) -> Mapping[str, Any]:
    """Full-validate a scanned ledger, then atomically rebuild its O(1) marker."""
    path = Path(path)
    validate_training_records(
        rows,
        source_commit=source_commit,
        immutable_run_card_sha256=immutable_run_card_sha256,
        dataset_order_sha256=dataset_order_sha256,
        target_steps=target_steps,
        expected_dataset_size=dataset_size,
        expected_batch_size=batch_size,
    )
    if rows and rows[0].get("config_sha256") != config_sha256:
        raise P3CompletionError("training tail index configuration drift")
    record = _training_tail_index_record(
        path,
        rows,
        source_commit=source_commit,
        immutable_run_card_sha256=immutable_run_card_sha256,
        dataset_order_sha256=dataset_order_sha256,
        config_sha256=config_sha256,
        dataset_size=dataset_size,
        batch_size=batch_size,
        target_steps=target_steps,
    )
    atomic_write_json(training_tail_index_path(path), record)
    return record


def verify_training_tail_index(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    immutable_run_card_sha256: str,
    dataset_order_sha256: str,
    config_sha256: str,
    dataset_size: int,
    batch_size: int,
    target_steps: int,
) -> Mapping[str, Any]:
    """Bind a fully scanned ledger to its atomic tail marker."""
    path = Path(path)
    if rows and rows[0].get("config_sha256") != config_sha256:
        raise P3CompletionError("training tail index configuration drift")
    actual = load_json(training_tail_index_path(path))
    expected = _training_tail_index_record(
        path,
        rows,
        source_commit=source_commit,
        immutable_run_card_sha256=immutable_run_card_sha256,
        dataset_order_sha256=dataset_order_sha256,
        config_sha256=config_sha256,
        dataset_size=dataset_size,
        batch_size=batch_size,
        target_steps=target_steps,
    )
    if actual != expected:
        raise P3CompletionError("training tail index differs from the full ledger")
    return actual


def _load_incremental_training_tail_index(
    path: Path, *, target_steps: int
) -> Mapping[str, Any]:
    index = load_json(training_tail_index_path(path))
    next_step = index.get("next_step")
    record_count = index.get("record_count")
    ledger_size = index.get("ledger_size_bytes")
    tail = index.get("tail_record_sha256")
    tail_offset = index.get("tail_offset_bytes")
    tail_size = index.get("tail_size_bytes")
    tail_line_sha256 = index.get("tail_line_sha256")
    dataset_size = index.get("dataset_size")
    batch_size = index.get("batch_size")
    steps_per_epoch = index.get("steps_per_epoch")
    if (
        index.get("schema") != TRAINING_TAIL_INDEX_SCHEMA
        or index.get("ledger_path") != str(path.resolve())
        or index.get("target_steps") != int(target_steps)
        or not isinstance(next_step, int)
        or isinstance(next_step, bool)
        or not 1 <= next_step <= int(target_steps) + 1
        or not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count != next_step - 1
        or not isinstance(ledger_size, int)
        or isinstance(ledger_size, bool)
        or ledger_size < 0
        or not isinstance(dataset_size, int)
        or isinstance(dataset_size, bool)
        or dataset_size <= 0
        or not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or not isinstance(steps_per_epoch, int)
        or isinstance(steps_per_epoch, bool)
        or steps_per_epoch != (dataset_size + batch_size - 1) // batch_size
        or (tail is not None and not is_sha256(tail))
        or (tail_line_sha256 is not None and not is_sha256(tail_line_sha256))
        or (record_count == 0) != (tail is None)
        or (record_count == 0) != (tail_offset is None)
        or (record_count == 0) != (tail_size is None)
        or (record_count == 0) != (tail_line_sha256 is None)
        or (record_count == 0) != (ledger_size == 0)
        or (
            record_count > 0
            and (
                not isinstance(tail_offset, int)
                or isinstance(tail_offset, bool)
                or tail_offset < 0
                or not isinstance(tail_size, int)
                or isinstance(tail_size, bool)
                or tail_size <= 0
                or tail_offset + tail_size != ledger_size
            )
        )
        or not is_source_commit(index.get("source_commit"))
        or not all(
            is_sha256(index.get(field))
            for field in (
                "immutable_run_card_sha256",
                "dataset_order_sha256",
                "config_sha256",
            )
        )
    ):
        raise P3CompletionError("training tail index is invalid")
    actual_size = path.stat().st_size if path.exists() else 0
    if actual_size != ledger_size:
        raise P3CompletionError("training ledger size differs from its tail index")
    if record_count > 0:
        with path.open("rb") as handle:
            handle.seek(tail_offset)
            tail_bytes = handle.read(tail_size)
        if len(tail_bytes) != tail_size or sha256_bytes(tail_bytes) != tail_line_sha256:
            raise P3CompletionError("training ledger tail bytes differ from its index")
        try:
            tail_row = json.loads(tail_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise P3CompletionError(
                "training ledger indexed tail is invalid JSON"
            ) from exc
        if (
            not isinstance(tail_row, Mapping)
            or canonical_json_bytes(tail_row) + b"\n" != tail_bytes
            or tail_row.get("global_step") != record_count
            or tail_row.get("record_sha256") != tail
            or tail_row.get("record_sha256") != _record_digest(tail_row)
        ):
            raise P3CompletionError("training ledger indexed tail record differs")
        for field in (
            "source_commit",
            "immutable_run_card_sha256",
            "dataset_order_sha256",
            "config_sha256",
        ):
            if tail_row.get(field) != index.get(field):
                raise P3CompletionError(
                    "training ledger tail has immutable provenance drift"
                )
        _validate_training_cursor(
            tail_row,
            step=record_count,
            dataset_order_sha256=str(index["dataset_order_sha256"]),
            expected_dataset_size=dataset_size,
            expected_batch_size=batch_size,
        )
    return index


def _validate_new_training_record(
    value: Mapping[str, Any],
    *,
    target_steps: int,
    provenance: Mapping[str, Any],
) -> int:
    step = value.get("global_step")
    if value.get("schema") != TRAINING_RECORD_SCHEMA:
        raise P3CompletionError("new training record has an unknown schema")
    if (
        not isinstance(step, int)
        or isinstance(step, bool)
        or step < 1
        or step > int(target_steps)
    ):
        raise P3CompletionError("new training record step is outside 1..target_steps")
    if not is_source_commit(value.get("source_commit")) or not all(
        is_sha256(value.get(field))
        for field in (
            "immutable_run_card_sha256",
            "dataset_order_sha256",
            "config_sha256",
        )
    ):
        raise P3CompletionError("new training record has invalid provenance")
    for field in (
        "source_commit",
        "immutable_run_card_sha256",
        "dataset_order_sha256",
        "config_sha256",
    ):
        if value.get(field) != provenance.get(field):
            raise P3CompletionError("new training record has provenance drift")
    _validate_training_cursor(
        value,
        step=step,
        dataset_order_sha256=str(provenance["dataset_order_sha256"]),
        expected_dataset_size=provenance.get("dataset_size"),
        expected_batch_size=provenance.get("batch_size"),
    )
    if not is_finite_number(value.get("loss")):
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
    return step


def append_training_record(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    target_steps: int,
) -> Mapping[str, Any]:
    path = Path(path)
    index_path = training_tail_index_path(path)
    if not index_path.exists():
        if path.exists() and path.stat().st_size != 0:
            raise P3CompletionError(
                "nonempty training ledger requires a full-scan tail-index rebuild"
            )
        provenance = {
            field: value.get(field)
            for field in (
                "source_commit",
                "immutable_run_card_sha256",
                "dataset_order_sha256",
                "config_sha256",
            )
        }
        sampler = value.get("sampler")
        if not isinstance(sampler, Mapping):
            raise P3CompletionError("new training record sampler is absent")
        provenance["dataset_size"] = sampler.get("dataset_size")
        provenance["batch_size"] = sampler.get("batch_size")
        _validate_new_training_record(
            value, target_steps=target_steps, provenance=provenance
        )
        initialize_training_tail_index(
            path,
            [],
            source_commit=str(value["source_commit"]),
            immutable_run_card_sha256=str(value["immutable_run_card_sha256"]),
            dataset_order_sha256=str(value["dataset_order_sha256"]),
            config_sha256=str(value["config_sha256"]),
            dataset_size=int(provenance["dataset_size"]),
            batch_size=int(provenance["batch_size"]),
            target_steps=target_steps,
        )
    index = _load_incremental_training_tail_index(path, target_steps=target_steps)
    provenance = {
        field: index[field]
        for field in (
            "source_commit",
            "immutable_run_card_sha256",
            "dataset_order_sha256",
            "config_sha256",
            "dataset_size",
            "batch_size",
        )
    }
    step = _validate_new_training_record(
        value, target_steps=target_steps, provenance=provenance
    )
    next_step = int(index["next_step"])
    if step < next_step:
        rows = load_jsonl(path)
        validate_training_records(
            rows,
            source_commit=str(index["source_commit"]),
            immutable_run_card_sha256=str(index["immutable_run_card_sha256"]),
            dataset_order_sha256=str(index["dataset_order_sha256"]),
            target_steps=target_steps,
            expected_dataset_size=int(index["dataset_size"]),
            expected_batch_size=int(index["batch_size"]),
        )
        verify_training_tail_index(
            path,
            rows,
            source_commit=str(index["source_commit"]),
            immutable_run_card_sha256=str(index["immutable_run_card_sha256"]),
            dataset_order_sha256=str(index["dataset_order_sha256"]),
            config_sha256=str(index["config_sha256"]),
            dataset_size=int(index["dataset_size"]),
            batch_size=int(index["batch_size"]),
            target_steps=target_steps,
        )
        if step < 1 or step > len(rows):
            raise P3CompletionError(
                f"replayed optimizer step {step} is outside the validated ledger"
            )
        existing = rows[step - 1]
        if existing.get("global_step") != step:
            raise P3CompletionError(
                f"replayed optimizer step {step} differs from its validated index"
            )
        if _stable_training_record(existing) != _stable_training_record(value):
            raise P3CompletionError(
                f"replayed optimizer step {step} differs from its exactly-once record"
            )
        return existing
    if step != next_step:
        raise P3CompletionError("new training record would create a gap")
    record = dict(value)
    record["previous_record_sha256"] = index["tail_record_sha256"]
    record["record_sha256"] = _record_digest(record)
    append_jsonl(path, record)
    previous_size = int(index["ledger_size_bytes"])
    tail_bytes = canonical_json_bytes(record) + b"\n"
    ledger_size = path.stat().st_size
    if ledger_size != previous_size + len(tail_bytes):
        raise P3CompletionError("training ledger append size differs")
    updated_index = dict(index)
    updated_index["ledger_size_bytes"] = ledger_size
    updated_index["record_count"] = step
    updated_index["next_step"] = step + 1
    updated_index["tail_record_sha256"] = record["record_sha256"]
    updated_index["tail_offset_bytes"] = previous_size
    updated_index["tail_size_bytes"] = len(tail_bytes)
    updated_index["tail_line_sha256"] = sha256_bytes(tail_bytes)
    atomic_write_json(index_path, updated_index)
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
    target_steps: int,
    out_path: str | Path,
) -> Mapping[str, Any]:
    if not is_source_commit(source_commit):
        raise P3CompletionError("held-out manifest source commit is invalid")
    percent_step_map(target_steps)
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
        "target_steps": int(target_steps),
        "rounding_rule": HELDOUT_ROUNDING_RULE,
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
        "entry_count": metadata["entry_count"],
        "target_steps": metadata["target_steps"],
        "rounding_rule": metadata["rounding_rule"],
    }


def validate_runtime_heldout_manifest(
    record: Mapping[str, Any],
    *,
    environment: str,
    source_commit: str,
    target_steps: int,
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
        or not isinstance(metadata.get("target_steps"), int)
        or isinstance(metadata.get("target_steps"), bool)
        or metadata.get("target_steps") != target_steps
        or metadata.get("rounding_rule") != HELDOUT_ROUNDING_RULE
        or not isinstance(record.get("target_steps"), int)
        or isinstance(record.get("target_steps"), bool)
        or record.get("target_steps") != target_steps
        or record.get("rounding_rule") != HELDOUT_ROUNDING_RULE
        or metadata.get("manifest_sha256") != record.get("sha256")
        or not isinstance(metadata.get("entry_count"), int)
        or isinstance(metadata.get("entry_count"), bool)
        or metadata.get("entry_count") <= 0
        or metadata.get("entry_count") != len(valid)
        or not isinstance(record.get("entry_count"), int)
        or isinstance(record.get("entry_count"), bool)
        or record.get("entry_count") <= 0
        or record.get("entry_count") != len(valid)
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


def canonical_first_heldout_manifest_key(
    record: Mapping[str, Any], *, target_steps: int
) -> str:
    """Hash-verify a held-out manifest and return its canonical first key."""
    path = Path(str(record.get("path")))
    metadata_path = Path(str(record.get("metadata_path")))
    if (
        not path.is_file()
        or sha256_file(path) != record.get("sha256")
        or not metadata_path.is_file()
        or sha256_file(metadata_path) != record.get("metadata_sha256")
    ):
        raise P3CompletionError("held-out manifest or metadata hash differs")
    rows = load_jsonl(path)
    if not rows or any(row.get("schema") != HELDOUT_ENTRY_SCHEMA for row in rows):
        raise P3CompletionError("held-out manifest entries are absent or invalid")
    canonical_rows = _unique_entries(rows, "held-out manifest")
    if [_manifest_entry(row) for row in rows] != canonical_rows:
        raise P3CompletionError("held-out manifest entries are not in canonical order")
    metadata = load_json(metadata_path)
    if (
        record.get("selection") != "all_validation_examples"
        or not isinstance(record.get("target_steps"), int)
        or isinstance(record.get("target_steps"), bool)
        or record.get("target_steps") != target_steps
        or record.get("rounding_rule") != HELDOUT_ROUNDING_RULE
        or not isinstance(record.get("entry_count"), int)
        or isinstance(record.get("entry_count"), bool)
        or record.get("entry_count") <= 0
        or record.get("entry_count") != len(rows)
        or metadata.get("schema") != HELDOUT_METADATA_SCHEMA
        or metadata.get("selection") != "all_validation_examples"
        or metadata.get("manifest_sha256") != record.get("sha256")
        or metadata.get("data_manifest_sha256") != record.get("data_manifest_sha256")
        or metadata.get("split_sha256") != record.get("split_sha256")
        or not isinstance(metadata.get("entry_count"), int)
        or isinstance(metadata.get("entry_count"), bool)
        or metadata.get("entry_count") <= 0
        or metadata.get("entry_count") != len(rows)
        or not isinstance(metadata.get("target_steps"), int)
        or isinstance(metadata.get("target_steps"), bool)
        or metadata.get("target_steps") != target_steps
        or metadata.get("rounding_rule") != HELDOUT_ROUNDING_RULE
    ):
        raise P3CompletionError("held-out manifest acceptance metadata differs")
    return str(rows[0]["key"])


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
    expected_empirical_adapter_mode: str | None = None,
) -> None:
    mapping = percent_step_map(target_steps)
    seen = set()
    expected_percent = 1
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
        "depth_empirical_contract_sha256",
        "depth_empirical_provenance",
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
        if percent != expected_percent:
            raise P3CompletionError(
                "held-out loss ledger is not in canonical percent order"
            )
        seen.add(percent)
        expected_percent += 1
        global_step = row.get("global_step")
        if (
            not isinstance(global_step, int)
            or isinstance(global_step, bool)
            or global_step != mapping[percent]
        ):
            raise P3CompletionError(
                "held-out loss uses the wrong percent-to-step mapping"
            )
        row_target = row.get("target_steps")
        if (
            not isinstance(row_target, int)
            or isinstance(row_target, bool)
            or row_target != target_steps
            or row.get("rounding_rule") != HELDOUT_ROUNDING_RULE
        ):
            raise P3CompletionError("held-out loss target or rounding rule differs")
        if (
            row.get("immutable_run_card_sha256") != immutable_run_card_sha256
            or row.get("manifest_sha256") != manifest_sha256
        ):
            raise P3CompletionError("held-out loss provenance drift")
        if any(row.get(field) != immutable[field] for field in immutable_fields):
            raise P3CompletionError("held-out loss immutable provenance drift")
        if row.get("state_restored") is not True:
            raise P3CompletionError("held-out loss did not restore exact state")
        numerator = row.get("loss_numerator")
        count = row.get("element_count")
        mean = row.get("mean_loss")
        if (
            not is_finite_number(numerator)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or not is_finite_number(mean)
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
        depth_fields = {
            field: row.get(field)
            for field in immutable_fields
            if field.startswith("depth_")
        }
        empirical_provenance = depth_fields["depth_empirical_provenance"]
        if isinstance(empirical_provenance, Mapping):
            scalar_fields = (
                "depth_producer_sha256",
                "depth_cache_manifest_sha256",
                "depth_empirical_contract_sha256",
                "depth_validation_sha256",
                "depth_checkpoint_sha256",
            )
            if depth_fields["depth_native_contract_sha256"] is not None or not all(
                is_sha256(depth_fields[field]) for field in scalar_fields
            ):
                raise P3CompletionError("held-out loss has incomplete empirical depth provenance")
            try:
                validate_empirical_provenance(
                    empirical_provenance,
                    expected_contract_sha256=str(
                        depth_fields["depth_empirical_contract_sha256"]
                    ),
                )
            except EmpiricalDepthContractError as exc:
                raise P3CompletionError(
                    f"held-out loss empirical provenance differs: {exc}"
                ) from exc
            aligned = {
                "producer_sha256": "depth_producer_sha256",
                "cache_manifest_sha256": "depth_cache_manifest_sha256",
                "empirical_contract_sha256": "depth_empirical_contract_sha256",
                "validation_sha256": "depth_validation_sha256",
                "checkpoint_sha256": "depth_checkpoint_sha256",
            }
            if any(
                empirical_provenance.get(field) != depth_fields[scalar]
                for field, scalar in aligned.items()
            ):
                raise P3CompletionError("held-out loss empirical provenance is not scalar-aligned")
            if (
                expected_empirical_adapter_mode is not None
                and empirical_provenance.get("adapter_mode")
                != expected_empirical_adapter_mode
            ):
                raise P3CompletionError(
                    "held-out loss empirical adapter mode differs from the run-card arm"
                )
        elif empirical_provenance is not None:
            raise P3CompletionError("held-out loss empirical provenance must be an object")
        else:
            native_fields = (
                "depth_producer_sha256",
                "depth_cache_manifest_sha256",
                "depth_native_contract_sha256",
                "depth_validation_sha256",
                "depth_checkpoint_sha256",
            )
            any_depth = any(
                value is not None
                for field, value in depth_fields.items()
                if field != "depth_empirical_provenance"
            )
            if any_depth and (
                depth_fields["depth_empirical_contract_sha256"] is not None
                or not all(is_sha256(depth_fields[field]) for field in native_fields)
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
    expected_empirical_adapter_mode: str | None = None,
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
        expected_empirical_adapter_mode=expected_empirical_adapter_mode,
    )
    append_jsonl(path, record)
    return record


def validate_final_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_empirical_adapter_mode: str | None,
    expected: Mapping[str, Any] | None = None,
) -> None:
    if (
        receipt.get("schema") != FINAL_RECEIPT_SCHEMA
        or receipt.get("state") != "PASS"
        or receipt.get("fresh_model_process") is not True
    ):
        raise P3CompletionError("final acceptance receipt is not a fresh-process PASS")
    pid = receipt.get("process_id")
    training_pid = receipt.get("training_process_id")
    if not is_process_id(pid) or not is_process_id(training_pid):
        raise P3CompletionError("final acceptance process IDs are invalid")
    if pid == training_pid:
        raise P3CompletionError(
            "final acceptance process ID must differ from training process ID"
        )
    require_job_id(receipt.get("slurm_job_id"))
    target = receipt.get("target_steps")
    sampler = receipt.get("sampler")
    if (
        not isinstance(target, int)
        or isinstance(target, bool)
        or not isinstance(receipt.get("global_step"), int)
        or isinstance(receipt.get("global_step"), bool)
        or receipt.get("global_step") != target
        or not isinstance(sampler, Mapping)
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
    depth_identity_fields = (
        "depth_producer_sha256",
        "depth_cache_manifest_sha256",
        "depth_validation_sha256",
        "depth_checkpoint_sha256",
    )
    depth_values = [receipt.get(field) for field in depth_identity_fields]
    has_depth = any(value is not None for value in depth_values)
    if has_depth and not all(is_sha256(value) for value in depth_values):
        raise P3CompletionError("final acceptance has incomplete depth provenance")
    native_contract = receipt.get("depth_native_contract_sha256")
    empirical_contract = receipt.get("depth_empirical_contract_sha256")
    empirical_provenance = receipt.get("depth_empirical_provenance")
    if has_depth:
        if is_sha256(native_contract):
            if empirical_contract is not None or empirical_provenance is not None:
                raise P3CompletionError("native final acceptance carries empirical provenance")
        elif is_sha256(empirical_contract):
            if native_contract is not None or not isinstance(empirical_provenance, Mapping):
                raise P3CompletionError("empirical final acceptance provenance is incomplete")
            try:
                validate_empirical_provenance(
                    empirical_provenance,
                    expected_contract_sha256=str(empirical_contract),
                )
            except EmpiricalDepthContractError as exc:
                raise P3CompletionError(
                    f"final acceptance empirical provenance differs: {exc}"
                ) from exc
            aligned = {
                "producer_sha256": "depth_producer_sha256",
                "cache_manifest_sha256": "depth_cache_manifest_sha256",
                "empirical_contract_sha256": "depth_empirical_contract_sha256",
                "validation_sha256": "depth_validation_sha256",
                "checkpoint_sha256": "depth_checkpoint_sha256",
            }
            if any(
                empirical_provenance.get(field) != receipt.get(scalar)
                for field, scalar in aligned.items()
            ):
                raise P3CompletionError(
                    "final acceptance empirical provenance is not scalar-aligned"
                )
            if (
                expected_empirical_adapter_mode is None
                or empirical_provenance.get("adapter_mode")
                != expected_empirical_adapter_mode
            ):
                raise P3CompletionError(
                    "final acceptance empirical adapter mode differs"
                )
        else:
            raise P3CompletionError("final acceptance has no discriminated depth contract")
    elif any(
        value is not None
        for value in (native_contract, empirical_contract, empirical_provenance)
    ):
        raise P3CompletionError("depth-free final acceptance carries depth provenance")
    if expected_empirical_adapter_mode is not None and not is_sha256(empirical_contract):
        raise P3CompletionError(
            "final acceptance empirical adapter mode differs"
        )
    if not is_source_commit(receipt.get("source_commit")):
        raise P3CompletionError("final acceptance source commit is invalid")
    validate_final_sampler(
        sampler,
        target_steps=target,
        dataset_order_sha256=str(receipt["dataset_order_sha256"]),
    )
    validation = receipt.get("validation_batch")
    if not isinstance(validation, Mapping):
        raise P3CompletionError("final acceptance validation batch is absent")
    if not isinstance(validation.get("manifest_key"), str) or not validation.get(
        "manifest_key"
    ):
        raise P3CompletionError("final acceptance validation manifest key is invalid")
    numerator = validation.get("loss_numerator")
    count = validation.get("element_count")
    mean = validation.get("mean_loss")
    if (
        not is_finite_number(numerator)
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or not is_finite_number(mean)
        or not math.isclose(
            float(mean), float(numerator) / count, rel_tol=1e-12, abs_tol=1e-15
        )
    ):
        raise P3CompletionError("final acceptance validation loss is invalid")
    if expected is not None:
        differing = []
        for key, value in expected.items():
            if key == "validation_batch.manifest_key":
                actual = validation.get("manifest_key")
            else:
                actual = receipt.get(key)
            if actual != value:
                differing.append(key)
        if differing:
            raise P3CompletionError(
                f"final acceptance receipt differs: {sorted(differing)}"
            )


def write_final_receipt(
    path: str | Path,
    receipt: Mapping[str, Any],
    *,
    expected_empirical_adapter_mode: str | None,
) -> str:
    validate_final_receipt(
        receipt,
        expected_empirical_adapter_mode=expected_empirical_adapter_mode,
    )
    text = json.dumps(dict(receipt), indent=2, sort_keys=True, allow_nan=False) + "\n"
    immutable_write_text(path, text)
    return sha256_file(path)


def load_final_receipt(
    run_dir: str | Path,
    *,
    expected_empirical_adapter_mode: str | None,
    expected: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Path, str]:
    path = Path(run_dir) / "final_acceptance.json"
    receipt = load_json(path)
    validate_final_receipt(
        receipt,
        expected_empirical_adapter_mode=expected_empirical_adapter_mode,
        expected=expected,
    )
    return receipt, path, sha256_file(path)


def plateau_verdict(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_percent = {}
    for row in rows:
        try:
            mean = Decimal(canonical_json_bytes(row["mean_loss"]).decode("utf-8"))
        except (InvalidOperation, KeyError) as exc:
            raise P3CompletionError(
                "plateau audit requires canonical numeric mean losses"
            ) from exc
        if not mean.is_finite():
            raise P3CompletionError(
                "plateau audit requires canonical numeric mean losses"
            )
        by_percent[int(row["percent"])] = mean
    if set(by_percent) != set(range(1, 101)):
        raise P3CompletionError("plateau audit requires exact 1..100 loss coverage")
    early = sum(
        (by_percent[percent] for percent in range(76, 81)), Decimal(0)
    ) / Decimal(5)
    late = sum(
        (by_percent[percent] for percent in range(96, 101)), Decimal(0)
    ) / Decimal(5)
    if early <= Decimal(0):
        raise P3CompletionError("plateau audit requires a finite positive early mean")
    relative_change = abs(late - early) / early
    threshold = Decimal("0.02")
    return {
        "early_mean_76_80": float(early),
        "late_mean_96_100": float(late),
        "relative_absolute_change": float(relative_change),
        "threshold": PLATEAU_THRESHOLD,
        "plateaued": relative_change <= threshold,
    }


def comparison_verdict(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    return (
        "optimization-conclusive"
        if left.get("plateaued") is True and right.get("plateaued") is True
        else "optimization-inconclusive"
    )
