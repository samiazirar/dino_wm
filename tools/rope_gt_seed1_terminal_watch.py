#!/usr/bin/env python3
"""One external terminal/max-silence owner for the Rope GT seed-one chain."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}
FAILED_STATES = TERMINAL_STATES - {"COMPLETED"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def remote_snapshot(project: str, finalizer_job: str) -> dict[str, Any]:
    script = r'''
import json
import pathlib
import subprocess
import sys

project = pathlib.Path(sys.argv[1])
finalizer = str(sys.argv[2])
chain_path = project / "outputs/campaign-seed1/native-seed1-20260804c/rope/dinocular_gt_hdf5_cam1/run/chain.json"
chain = None
try:
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    pass
job_ids = [finalizer]
if isinstance(chain, dict):
    for item in chain.get("jobs", []):
        if isinstance(item, dict) and str(item.get("job_id", "")).isdigit():
            job_ids.append(str(item["job_id"]))
job_ids = sorted(set(job_ids), key=int)
states = {}
if job_ids:
    joined = ",".join(job_ids)
    queue = subprocess.run(
        ["squeue", "-h", "-j", joined, "-o", "%i|%T|%R"],
        text=True, capture_output=True, check=False,
    )
    if queue.returncode == 0:
        for line in queue.stdout.splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                states[parts[0]] = {"state": parts[1].strip().split("+", 1)[0], "reason": parts[2]}
    missing = [job for job in job_ids if job not in states]
    if missing:
        account = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", ",".join(missing),
             "-o", "JobIDRaw,State,ExitCode,End"],
            text=True, capture_output=True, check=False,
        )
        if account.returncode == 0:
            for line in account.stdout.splitlines():
                parts = line.split("|")
                if len(parts) >= 4 and parts[0] in missing:
                    states[parts[0]] = {
                        "state": parts[1].strip().split("+", 1)[0],
                        "exit_code": parts[2],
                        "end": parts[3],
                    }
progress_path = project / "outputs/campaign-seed1/native-seed1-20260804c/rope/dinocular_gt_hdf5_cam1/run/progress.json"
progress = None
progress_mtime_ns = None
try:
    progress_mtime_ns = progress_path.stat().st_mtime_ns
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    pass
compact_progress = None
if isinstance(progress, dict):
    compact_progress = {
        key: progress.get(key)
        for key in ("status", "global_step", "target_steps", "completed_segment_steps")
    }
print(json.dumps({
    "observed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "finalizer_job": finalizer,
    "chain_path": str(chain_path),
    "chain_status": chain.get("status") if isinstance(chain, dict) else None,
    "chain_jobs": job_ids,
    "states": states,
    "progress": compact_progress,
    "progress_mtime_ns": progress_mtime_ns,
}, sort_keys=True))
'''
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "marvin", "python3", "-", project, finalizer_job],
        input=script,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"remote scheduler inspection failed rc={result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[-400:]}"
        )
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError, TypeError) as exc:
        raise RuntimeError(f"remote scheduler inspection returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("remote scheduler inspection did not return an object")
    return value


def send_operations(event: str, reason: str, snapshot: dict[str, Any], *, pid: int) -> None:
    payload = {
        "service": "rope-gt-seed1-terminal-max-silence-owner",
        "event": event,
        "reason": reason,
        "owner_pid": pid,
        "observed_at": now(),
        "measured": {
            "finalizer_job": snapshot.get("finalizer_job"),
            "chain_status": snapshot.get("chain_status"),
            "chain_jobs": snapshot.get("chain_jobs", []),
            "states": snapshot.get("states", {}),
            "progress": snapshot.get("progress"),
        },
    }
    helper = os.environ.get(
        "HERDR_ROLE_MESSAGE", "/home/user/azirar/.local/bin/herdr-role-message"
    )
    result = subprocess.run(
        [helper, "operations", json.dumps(payload, sort_keys=True, separators=(",", ":"))],
        env={**os.environ, "HERDR_ENV": "1"},
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Operations delivery failed rc={result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[-400:]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalizer-job", required=True)
    parser.add_argument("--project", default="/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm")
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--max-silence-seconds", type=int, default=28800)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.max_silence_seconds <= 0:
        raise ValueError("poll and maximum-silence intervals must be positive")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    last_fingerprint = None
    last_activity = time.monotonic()
    with args.log.open("a", encoding="utf-8") as log:
        log.write(
            f"{now()} START finalizer={args.finalizer_job} "
            f"poll_seconds={args.poll_seconds} max_silence_seconds={args.max_silence_seconds} pid={os.getpid()}\n"
        )
        while True:
            try:
                snapshot = remote_snapshot(args.project, args.finalizer_job)
            except Exception as exc:
                log.write(f"{now()} INSPECTION_ERROR {exc}\n")
                log.flush()
                if time.monotonic() - last_activity >= args.max_silence_seconds:
                    send_operations("MAX_SILENCE", str(exc), {"finalizer_job": args.finalizer_job}, pid=os.getpid())
                    return 0
                time.sleep(args.poll_seconds)
                continue
            fingerprint_value = {
                "chain_status": snapshot.get("chain_status"),
                "chain_jobs": snapshot.get("chain_jobs"),
                "states": snapshot.get("states"),
                "progress": snapshot.get("progress"),
                "progress_mtime_ns": snapshot.get("progress_mtime_ns"),
            }
            fingerprint = digest(fingerprint_value)
            if fingerprint != last_fingerprint:
                last_fingerprint = fingerprint
                last_activity = time.monotonic()
                log.write(f"{now()} EVENT {json.dumps(snapshot, sort_keys=True)}\n")
                log.flush()

            chain_status = snapshot.get("chain_status")
            if chain_status == "PASSED":
                send_operations("TERMINAL", "Rope GT seed-one chain reached PASSED", snapshot, pid=os.getpid())
                log.write(f"{now()} STOP terminal=PASSED\n")
                return 0
            states = snapshot.get("states", {})
            failed = {
                job: record
                for job, record in states.items()
                if record.get("state") in FAILED_STATES
            }
            if failed:
                send_operations("TERMINAL_FAILURE", "Rope GT seed-one chain has a failed scheduler job", snapshot, pid=os.getpid())
                log.write(f"{now()} STOP failed_jobs={json.dumps(failed, sort_keys=True)}\n")
                return 0
            if (
                snapshot.get("chain_status") is None
                and states.get(str(args.finalizer_job), {}).get("state") in FAILED_STATES
            ):
                send_operations("TERMINAL_FAILURE", "Rope GT finalizer failed before chain materialization", snapshot, pid=os.getpid())
                log.write(f"{now()} STOP finalizer_failure\n")
                return 0
            if time.monotonic() - last_activity >= args.max_silence_seconds:
                send_operations("MAX_SILENCE", "no terminal or progress event within the maximum silence interval", snapshot, pid=os.getpid())
                log.write(f"{now()} STOP maximum_silence\n")
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"rope GT terminal owner failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
