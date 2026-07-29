#!/usr/bin/env python3
"""Fail-open W&B telemetry for the immutable DINOcular campaign outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable


DEFAULT_PROJECT = "dinocular-wm-campaign"
DEFAULT_REMOTE = "marvin"
SCHEMA = "dinocular.wandb-campaign.v1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def remote_text(remote: str, path: str) -> str:
    result = subprocess.run(
        ["ssh", remote, f"cat -- {shlex.quote(path)}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote_read_failed:{path}:{result.stderr.strip()}")
    return result.stdout


def remote_json(remote: str, path: str) -> Any:
    return json.loads(remote_text(remote, path))


def remote_jsonl(remote: str, path: str) -> Iterable[dict[str, Any]]:
    process = subprocess.Popen(
        ["ssh", remote, f"cat -- {shlex.quote(path)}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    for number, line in enumerate(process.stdout, 1):
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                process.kill()
                raise RuntimeError(f"invalid_jsonl:{path}:{number}") from exc
            if not isinstance(value, dict):
                process.kill()
                raise RuntimeError(f"invalid_jsonl_record:{path}:{number}")
            yield value
    stderr = process.stderr.read() if process.stderr is not None else ""
    if process.wait() != 0:
        raise RuntimeError(f"remote_read_failed:{path}:{stderr.strip()}")


def resolved_config(remote: str, run_dir: str) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(remote_text(remote, f"{run_dir}/hydra.yaml"))
    if not isinstance(value, dict) or not isinstance(value.get("training"), dict):
        raise RuntimeError(f"invalid_hydra_config:{run_dir}")
    return value


def wrapper_resources(remote: str, wrapper: str) -> dict[str, Any]:
    text = remote_text(remote, wrapper)

    def directive(name: str) -> str | None:
        match = re.search(rf"^#SBATCH\s+--{re.escape(name)}(?:=|\s+)(\S+)", text, re.M)
        return match.group(1) if match else None

    gpu_request = directive("gpus") or directive("gres") or "not_recorded"
    return {
        "cluster": remote,
        "partition": directive("partition") or "not_recorded",
        "gpu_request": gpu_request,
        "gpu_model": "not_recorded",
    }


def lineage_id(chain: str, seed: int) -> str:
    return f"{chain}/seed{seed}"


def wandb_run_id(identity: str) -> str:
    return "dc-" + hashlib.sha256(identity.encode()).hexdigest()[:20]


def select_lineages(
    state: dict[str, Any], requested: set[str] | None
) -> Iterable[tuple[str, dict[str, Any], dict[str, Any]]]:
    for chain_name, chain in sorted(state["chains"].items()):
        for seed_text, lineage in sorted(
            chain["lineages"].items(), key=lambda item: int(item[0])
        ):
            identity = lineage_id(chain_name, int(seed_text))
            if requested is None or identity in requested:
                yield identity, chain, lineage


def event_lineages(payload: dict[str, Any], state: dict[str, Any]) -> set[str]:
    selected: set[str] = set()
    for event in payload.get("events", []):
        if not isinstance(event, dict) or not isinstance(event.get("chain"), str):
            continue
        chain = event["chain"]
        if isinstance(event.get("seed"), int):
            selected.add(lineage_id(chain, event["seed"]))
        elif chain in state["chains"]:
            selected.update(
                lineage_id(chain, int(seed))
                for seed in state["chains"][chain]["lineages"]
            )
    return selected


def learning_rates(config: dict[str, Any]) -> dict[str, float]:
    training = config["training"]
    values: dict[str, float] = {}
    for name in ("encoder_lr", "decoder_lr", "predictor_lr", "action_encoder_lr"):
        value = training.get(name)
        if isinstance(value, (int, float)):
            values[f"learning_rate/{name.removesuffix('_lr')}"] = float(value)
    return values


def prior_summary(api: Any, path: str) -> dict[str, Any]:
    try:
        return dict(api.run(path).summary)
    except Exception:
        return {}


def imported_rows(
    remote: str,
    lineage: dict[str, Any],
    progress: dict[str, Any],
    rates: dict[str, float],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completion = progress.get("p3_completion", {})
    training_path = completion.get(
        "training_ledger", f"{lineage['run_dir']}/training_steps.jsonl"
    )
    validation_path = completion.get(
        "validation_ledger", f"{lineage['run_dir']}/heldout_loss.jsonl"
    )
    history_path = completion.get(
        "checkpoint_history",
        f"{lineage['run_dir']}/checkpoints/steps/checkpoint_history.json",
    )
    last_training = int(summary.get("telemetry/last_training_step", -1))
    last_validation = int(summary.get("telemetry/last_validation_step", -1))
    last_checkpoint = int(summary.get("telemetry/last_checkpoint_sequence", -1))
    segments = {
        str(job_id): index for index, job_id in enumerate(lineage.get("segments", []))
    }
    rows: list[dict[str, Any]] = []

    for record in remote_jsonl(remote, training_path):
        step = int(record["global_step"])
        if step <= last_training:
            continue
        job_id = str(record.get("slurm_job_id", "not_recorded"))
        segment_index = segments.get(job_id)
        rows.append(
            {
                "provenance": "imported",
                "event_type": "training_step",
                "global_step": step,
                "training_loss": float(record["loss"]),
                "segment_job_id": job_id,
                "segment_index": segment_index,
                "resume_segment": bool(segment_index and segment_index > 0),
                **rates,
            }
        )
        last_training = step

    for record in remote_jsonl(remote, validation_path):
        step = int(record["global_step"])
        if step <= last_validation:
            continue
        rows.append(
            {
                "provenance": "imported",
                "event_type": "heldout_loss",
                "global_step": step,
                "heldout_loss": float(record["mean_loss"]),
                "segment_job_id": str(record.get("slurm_job_id", "not_recorded")),
                **rates,
            }
        )
        last_validation = step

    history = remote_json(remote, history_path)
    for record in history.get("records", []):
        sequence = int(record["sequence"])
        if sequence <= last_checkpoint:
            continue
        rows.append(
            {
                "provenance": "imported",
                "event_type": "checkpoint",
                "global_step": int(record["step"]),
                "checkpoint_event": True,
                "checkpoint_reasons": ",".join(record.get("reasons", [])),
                "checkpoint_sha256": record.get("checkpoint_sha256"),
                **rates,
            }
        )
        last_checkpoint = sequence

    progress_fingerprint = fingerprint(progress)
    if summary.get("telemetry/progress_fingerprint") != progress_fingerprint:
        rows.append(
            {
                "provenance": "imported",
                "event_type": "terminal_progress",
                "global_step": int(progress["global_step"]),
                "training_loss": progress.get("last_step_loss"),
                "runtime_seconds": progress.get("elapsed_seconds"),
                "terminal_status": progress.get("status"),
                "segment_job_id": str(progress.get("slurm_job_id", "not_recorded")),
                "checkpoint_event": bool(progress.get("checkpoint_sha256")),
                "checkpoint_sha256": progress.get("checkpoint_sha256"),
                **rates,
            }
        )

    rows.sort(
        key=lambda row: (
            int(row.get("global_step", -1)),
            {"training_step": 0, "heldout_loss": 1, "checkpoint": 2}.get(
                str(row["event_type"]), 3
            ),
        )
    )
    cursor = {
        "last_training_step": last_training,
        "last_validation_step": last_validation,
        "last_checkpoint_sequence": last_checkpoint,
        "progress_fingerprint": progress_fingerprint,
    }
    return rows, cursor


def live_rows(
    payload: dict[str, Any] | None,
    identity: str,
    progress: dict[str, Any],
    rates: dict[str, float],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    prior = list(summary.get("telemetry/live_event_fingerprints", []))
    seen = set(prior)
    rows: list[dict[str, Any]] = []
    if payload is None:
        return rows, prior
    chain, seed_text = identity.rsplit("/seed", 1)
    matching = [
        event
        for event in payload.get("events", [])
        if isinstance(event, dict)
        and event.get("chain") == chain
        and (event.get("seed") is None or event.get("seed") == int(seed_text))
    ]
    if not matching:
        return rows, prior
    batch_fingerprint = fingerprint(
        {"timestamp": payload.get("timestamp"), "events": matching}
    )
    if batch_fingerprint in seen:
        return rows, prior
    rows.append(
        {
            "provenance": "live",
            "event_type": "controller_event",
            "controller_event_types": ",".join(
                str(event.get("type", "unknown")) for event in matching
            ),
            "controller_events": matching,
            "global_step": max(
                (
                    int(event["global_step"])
                    for event in matching
                    if isinstance(event.get("global_step"), int)
                ),
                default=int(progress["global_step"]),
            ),
            "runtime_seconds": progress.get("elapsed_seconds"),
            "terminal_status": matching[-1].get(
                "terminal_state", matching[-1].get("state", matching[-1].get("type"))
            ),
            **rates,
        }
    )
    return rows, (prior + [batch_fingerprint])[-256:]


def sync_lineage(
    *,
    wandb: Any,
    api: Any,
    remote: str,
    entity: str,
    project: str,
    identity: str,
    chain: dict[str, Any],
    lineage: dict[str, Any],
    payload: dict[str, Any] | None,
    dry_run: bool,
) -> dict[str, Any]:
    progress = remote_json(remote, f"{lineage['run_dir']}/progress.json")
    config = resolved_config(remote, lineage["run_dir"])
    resources = wrapper_resources(remote, lineage["wrapper"])
    run_id = wandb_run_id(identity)
    path = f"{entity}/{project}/{run_id}"
    summary = {} if dry_run else prior_summary(api, path)
    rates = learning_rates(config)
    imported, cursor = imported_rows(remote, lineage, progress, rates, summary)
    live, live_fingerprints = live_rows(payload, identity, progress, rates, summary)
    rows = imported + live
    if dry_run:
        return {
            "lineage": identity,
            "run_id": run_id,
            "imported_records": len(imported),
            "live_records": len(live),
            "status": "dry_run",
        }

    run = wandb.init(
        dir="/tmp/dinocular-wandb",
        entity=entity,
        project=project,
        id=run_id,
        name=identity.replace("/", "-"),
        group=chain["arm"],
        job_type="campaign-telemetry",
        resume="allow",
        config={
            "telemetry_schema": SCHEMA,
            "task": chain["environment"],
            "system": chain["arm"],
            "seed": lineage["seed"],
            **resources,
            **rates,
        },
        settings=wandb.Settings(
            silent=True, console="off", disable_git=True, save_code=False
        ),
    )
    sequence = int(summary.get("telemetry/last_sequence", -1))
    for row in rows:
        sequence += 1
        run.log({"telemetry_sequence": sequence, **resources, **row}, step=sequence)
    run.summary.update(
        {
            "telemetry/schema": SCHEMA,
            "telemetry/last_sequence": sequence,
            "telemetry/last_training_step": cursor["last_training_step"],
            "telemetry/last_validation_step": cursor["last_validation_step"],
            "telemetry/last_checkpoint_sequence": cursor["last_checkpoint_sequence"],
            "telemetry/progress_fingerprint": cursor["progress_fingerprint"],
            "telemetry/live_event_fingerprints": live_fingerprints,
            "telemetry/last_sync_at": utc_now(),
            "telemetry/provenance_labels": ["imported", "live"],
        }
    )
    run.finish()
    visible = api.run(path)
    return {
        "lineage": identity,
        "run_id": run_id,
        "run_url": visible.url,
        "imported_records": len(imported),
        "live_records": len(live),
        "status": "success",
    }


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--event-file", type=Path)
    parser.add_argument("--lineage", action="append")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "timestamp": utc_now(),
        "status": "fail_open",
        "project": args.project,
    }
    try:
        state = json.loads(args.state.read_text())
        payload = (
            json.loads(args.event_file.read_text())
            if args.event_file and args.event_file.exists()
            else None
        )
        requested = set(args.lineage) if args.lineage else None
        if requested is None and payload is not None:
            requested = event_lineages(payload, state)
        if requested == set():
            raise RuntimeError("controller_event_has_no_lineage")

        import wandb

        api = wandb.Api(timeout=30)
        entity = args.entity or api.default_entity
        if not entity:
            raise RuntimeError("missing_wandb_entity")
        receipt.update(
            {
                "entity": entity,
                "project_identity": f"{entity}/{args.project}",
                "project_url": f"https://wandb.ai/{entity}/{args.project}",
            }
        )
        results = [
            sync_lineage(
                wandb=wandb,
                api=api,
                remote=args.remote,
                entity=entity,
                project=args.project,
                identity=identity,
                chain=chain,
                lineage=lineage,
                payload=payload,
                dry_run=args.dry_run,
            )
            for identity, chain, lineage in select_lineages(state, requested)
        ]
        if not results:
            raise RuntimeError("requested_lineage_not_found")
        receipt.update(
            {
                "status": "dry_run" if args.dry_run else "success",
                "lineages": results,
                "controller_hook": (
                    f"{sys.executable} {Path(__file__).resolve()} "
                    "--state /tmp/dinocular_resume_controller_state.json "
                    "--event-file /tmp/dinocular_resume_controller.event "
                    f"--project {shlex.quote(args.project)} "
                    f"--receipt {shlex.quote(str(args.receipt))}"
                ),
            }
        )
    except Exception as exc:
        receipt["missing_input_or_error"] = f"{type(exc).__name__}:{exc}"
    write_receipt(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
