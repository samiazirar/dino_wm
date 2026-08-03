#!/usr/bin/env python3
"""One bounded, service-side progress route for the Dinocular campaign.

The route is deliberately not a model watcher. It reads a fixed set of
durable Wall/PushT markers, stores the last measured point outside the
repository, and wakes Operations only for a positive productive delta, a
confirmed failure, or maximum silence.

The marker/receipt/Operations delivery shape follows the existing
rg_depth_event_watch.sh service. This file adds the campaign-wide productive
counters and a singleton lock; it does not submit, cancel, requeue, signal, or
otherwise modify a job.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA = "dinocular.operations-productive-silence.v1"
REMOTE_ROOT = os.environ.get(
    "OPS_PRODUCTIVE_REMOTE_ROOT",
    "/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm",
)
REMOTE_HOST = os.environ.get("OPS_PRODUCTIVE_REMOTE_HOST", "marvin")
POLL_SECONDS = int(os.environ.get("OPS_PRODUCTIVE_POLL_SECONDS", "600"))
MAX_SILENCE_SECONDS = int(
    os.environ.get("OPS_PRODUCTIVE_MAX_SILENCE_SECONDS", "10800")
)
RETRY_LIMIT = int(os.environ.get("OPS_PRODUCTIVE_RETRY_LIMIT", "3"))
RUN_ONCE = os.environ.get("OPS_PRODUCTIVE_RUN_ONCE", "0") == "1"
DISABLE_REMOTE = os.environ.get("OPS_PRODUCTIVE_DISABLE_REMOTE", "0") == "1"
HELPER = os.environ.get(
    "OPS_PRODUCTIVE_HELPER", "/home/user/azirar/.local/bin/herdr-role-message"
)
WORKSPACE_ID = os.environ.get("HERDR_WORKSPACE_ID", "w3")

STATE_DIR = Path(
    os.environ.get(
        "OPS_PRODUCTIVE_STATE_DIR",
        "/home/user/azirar/.local/state/dinocular-operations-productive-silence",
    )
)
STATE_PATH = Path(
    os.environ.get("OPS_PRODUCTIVE_STATE", str(STATE_DIR / "state.json"))
)
RECEIPT_PATH = Path(
    os.environ.get("OPS_PRODUCTIVE_RECEIPT", str(STATE_DIR / "current-event.json"))
)
LOG_PATH = Path(
    os.environ.get("OPS_PRODUCTIVE_LOG", str(STATE_DIR / "watch.log"))
)
LOCK_PATH = Path(
    os.environ.get("OPS_PRODUCTIVE_LOCK", str(STATE_DIR / "singleton.lock"))
)
LOCAL_CAUSAL_MARKER = Path(
    os.environ.get(
        "OPS_PRODUCTIVE_CAUSAL_MARKER",
        "/home/user/azirar/dinocular-wm/scratch/operations_productive_markers/causal-result.json",
    )
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_iso()} {message}\n")


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def marker_file(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False, "count": 0}

    result: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if path.suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, dict):
            result["status"] = str(
                payload.get("state", payload.get("status", ""))
            ).upper()
            for key in (
                "accepted_count",
                "completed_count",
                "count",
                "episode_count",
                "planning_target_count",
                "routes_accepted",
                "terminal_inputs_accepted",
                "version",
                "result_id",
            ):
                if key in payload:
                    result[key] = payload[key]
    return result


def local_causal_marker() -> dict[str, Any]:
    marker = marker_file(LOCAL_CAUSAL_MARKER)
    if not marker.get("exists"):
        return {
            "version": None,
            "completed": 0,
            "status": "MISSING",
            "source": str(LOCAL_CAUSAL_MARKER),
        }
    payload = read_json(LOCAL_CAUSAL_MARKER)
    if not isinstance(payload, dict):
        return {
            "version": None,
            "completed": 0,
            "status": "INVALID",
            "source": str(LOCAL_CAUSAL_MARKER),
        }
    status = str(payload.get("state", payload.get("status", ""))).upper()
    accepted = status in {"PASS", "ACCEPTED", "COMPLETE", "COMPLETED"} or (
        payload.get("accepted") is True
    )
    try:
        completed = int(payload.get("completed_count", payload.get("count", 1)))
    except (TypeError, ValueError):
        completed = 0
    return {
        "version": payload.get("version", payload.get("result_id")),
        "completed": completed if accepted else 0,
        "status": status or ("ACCEPTED" if accepted else "UNACCEPTED"),
        "source": str(LOCAL_CAUSAL_MARKER),
    }


REMOTE_QUERY = r'''import json
import pathlib
import re
import subprocess

root = pathlib.Path(__ROOT__)
terminal_candidates = [
    root / "outputs/campaign-evaluation/eval36-20260728a/manifests/wall_terminal_inputs.accepted.json",
    root / "outputs/campaign-evaluation/eval36-20260728a/manifests/wall_terminal_input_recovery.receipt.json",
    root / "outputs/campaign-evaluation/eval36-20260728a/manifests/wall_50.acceptance.json",
    root / "outputs/campaign-evaluation/eval36-20260728a/manifests/wall_50.receipt.json",
    root / "evidence/wall_terminal_inputs.accepted.json",
    root / "evidence/wall_terminal_input_recovery.receipt.json",
]

prediction_names = (
    "prediction_results.jsonl",
    "prediction_episodes.jsonl",
    "open_loop_results.jsonl",
    "wall_prediction_episodes.jsonl",
)
planning_names = (
    "planning_results.jsonl",
    "planning_targets.jsonl",
    "planning_completion.json",
    "planning_results.json",
)
prediction_candidates = []
planning_candidates = []
for seed in ("1", "2", "3"):
    for arm in ("dino_pinned", "dinocular", "dinocular_zerodepth"):
        base = root / f"outputs/campaign-evaluation/eval36-20260728a/lineages/seed{seed}/wall/{arm}"
        for name in prediction_names:
            prediction_candidates.extend((base / name, base / "run" / name, base / "results" / name))
        for name in planning_names:
            planning_candidates.extend((base / name, base / "run" / name, base / "results" / name))
for base in (
    root / "outputs/campaign-evaluation/eval36-20260728a/results",
    root / "outputs/campaign-evaluation/eval36-20260728a/predictions",
    root / "outputs/campaign-planning/planning36-20260728a/results",
    root / "outputs/campaign-planning/planning36-20260728a/completed",
):
    prediction_candidates.extend(base / name for name in prediction_names)
    planning_candidates.extend(base / name for name in planning_names)

controller_path = root / "outputs/campaign-seeds23/seeds23-20260728a/resume_controller_state.json"
pusht_paths = {
    "dinocular": root / "outputs/campaign-seed1/pusht-seed1-countable-20260728a/dinocular/run/progress.json",
    "dinocular_zerodepth": root / "outputs/campaign-seed1/pusht-seed1-countable-20260728a/dinocular_zerodepth/run/progress.json",
}
terminal_states = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
    "SPECIAL_EXIT",
}

def info(path):
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

def load(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None

def integer(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    return None

def accepted_count(path):
    payload = load(path)
    if not isinstance(payload, dict):
        return 0
    state = str(payload.get("state", payload.get("status", ""))).upper()
    accepted = payload.get("accepted") is True or state in {"PASS", "ACCEPTED", "COMPLETE", "COMPLETED"}
    if not accepted:
        return 0
    for key in ("accepted_count", "terminal_inputs_accepted", "routes_accepted", "completed_count", "count"):
        value = integer(payload.get(key))
        if value is not None:
            return value
    return 1

def completed_count(path):
    if path.suffix == ".jsonl":
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError:
            return 0
    payload = load(path)
    if isinstance(payload, dict):
        for key in ("completed_count", "episode_count", "planning_target_count", "count", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
            parsed = integer(value)
            if parsed is not None:
                return parsed
    return 0

terminal_record = {
    "accepted_count": 0,
    "source": None,
    "artifact": info(root / "outputs/campaign-evaluation/eval36-20260728a/manifests/wall_50.pkl"),
}
for candidate in terminal_candidates:
    record = info(candidate)
    if record["exists"]:
        terminal_record = {
            "accepted_count": accepted_count(candidate),
            "source": str(candidate),
            "marker": record,
        }
        break

prediction_files = [path for path in prediction_candidates if info(path)["exists"]]
planning_files = [path for path in planning_candidates if info(path)["exists"]]
prediction_count = sum(completed_count(path) for path in prediction_files)
planning_count = sum(completed_count(path) for path in planning_files)

controller = load(controller_path)
chains = controller.get("chains", {}) if isinstance(controller, dict) else {}
arm_records = {}
job_ids = []
for arm in ("dinocular", "dinocular_zerodepth"):
    chain_name = f"pusht/{arm}"
    chain = chains.get(chain_name) if isinstance(chains, dict) else None
    lineage = chain.get("lineages", {}).get("1", {}) if isinstance(chain, dict) else {}
    progress = lineage.get("last_progress") if isinstance(lineage, dict) else None
    progress = progress if isinstance(progress, dict) else {}
    job_id = str(lineage.get("job_id", "")) if isinstance(lineage, dict) else ""
    if job_id:
        job_ids.append(job_id)
    arm_records[arm] = {
        "controller_present": isinstance(chain, dict),
        "phase": chain.get("phase") if isinstance(chain, dict) else None,
        "job_id": job_id or None,
        "accepted_step": integer(progress.get("global_step")) or 0,
        "checkpoint": progress.get("checkpoint"),
        "checkpoint_sha256": progress.get("checkpoint_sha256"),
        "training_complete": bool(lineage.get("training_complete")) if isinstance(lineage, dict) else False,
        "failure": lineage.get("failure") if isinstance(lineage, dict) else None,
        "progress_source": str(pusht_paths[arm]),
        "progress_file": info(pusht_paths[arm]),
    }

scheduler = {}
if job_ids:
    joined = ",".join(sorted(set(job_ids)))
    queue = subprocess.run(["squeue", "-h", "-j", joined, "-o", "%i|%T|%R"], text=True, capture_output=True)
    if queue.returncode == 0:
        for line in queue.stdout.splitlines():
            parts = line.split("|", 2)
            if len(parts) >= 2:
                scheduler[parts[0]] = parts[1].strip().split("+", 1)[0]
    else:
        scheduler["__error__"] = f"squeue_rc={queue.returncode}"
    missing = [job for job in sorted(set(job_ids)) if job not in scheduler]
    if missing and "__error__" not in scheduler:
        account = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(missing), "-o", "JobIDRaw,State"], text=True, capture_output=True)
        if account.returncode == 0:
            for line in account.stdout.splitlines():
                parts = line.split("|", 1)
                if len(parts) == 2 and parts[0] in missing:
                    scheduler[parts[0]] = parts[1].strip().split("+", 1)[0]
        else:
            scheduler["__error__"] = f"sacct_rc={account.returncode}"

for arm, record in arm_records.items():
    job_id = record["job_id"]
    state = scheduler.get(job_id, "UNKNOWN") if job_id else "MISSING"
    record["continuation"] = state
    record["legitimately_running_or_queued"] = state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}

print(json.dumps({
    "schema": "dinocular.operations-productive-remote-markers.v1",
    "controller": {
        "path": str(controller_path),
        "exists": isinstance(controller, dict),
        "updated_at": controller.get("updated_at") if isinstance(controller, dict) else None,
    },
    "productive": {
        "wall_terminal_inputs_accepted": terminal_record["accepted_count"],
        "wall_prediction_episodes_completed": prediction_count,
        "wall_planning_targets_completed": planning_count,
        "pusht_dinocular_accepted_step": arm_records["dinocular"]["accepted_step"],
        "pusht_dinocular_zerodepth_accepted_step": arm_records["dinocular_zerodepth"]["accepted_step"],
    },
    "health": {
        "wall_terminal_marker": terminal_record,
        "wall_prediction_sources": [str(path) for path in prediction_files[:12]],
        "wall_planning_sources": [str(path) for path in planning_files[:12]],
        "pusht": arm_records,
        "scheduler_error": scheduler.get("__error__"),
    },
}, sort_keys=True))
'''.replace("__ROOT__", json.dumps(REMOTE_ROOT))


def remote_snapshot() -> dict[str, Any]:
    if DISABLE_REMOTE:
        return {"disabled": True, "productive": {}, "health": {}}
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                REMOTE_HOST,
                "python3",
                "-",
            ],
            input=REMOTE_QUERY,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": f"remote_query_exception={error}"}
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")
        return {"error": f"remote_query_rc={result.returncode}:{detail[:400]}"}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError, TypeError) as error:
        detail = result.stdout.strip().replace("\n", " ")
        return {"error": f"remote_query_invalid_json={error}:{detail[:400]}"}


def fake_snapshot() -> dict[str, Any] | None:
    raw = os.environ.get("OPS_PRODUCTIVE_SNAPSHOT_JSON")
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except ValueError as error:
        return {"error": f"fake_snapshot_invalid_json={error}"}
    if not isinstance(value, dict) or not isinstance(value.get("productive"), dict):
        return {"error": "fake_snapshot_requires_object_with_productive"}
    value.setdefault("health", {})
    value.setdefault("schema", "dinocular.operations-productive-fake-markers.v1")
    return value


def collect_snapshot() -> dict[str, Any]:
    override = fake_snapshot()
    if override is not None:
        return override
    remote = remote_snapshot()
    if "error" in remote:
        return remote
    productive = dict(remote.get("productive", {}))
    health = dict(remote.get("health", {}))
    causal = local_causal_marker()
    productive["causal_result_completed"] = causal["completed"]
    productive["causal_result_version"] = causal["version"]
    health["causal_result"] = causal
    return {
        "schema": SCHEMA,
        "observed_at": now_iso(),
        "productive": productive,
        "health": health,
    }


COUNTER_KEYS = {
    "wall_terminal_inputs_accepted",
    "wall_prediction_episodes_completed",
    "wall_planning_targets_completed",
    "pusht_dinocular_accepted_step",
    "pusht_dinocular_zerodepth_accepted_step",
    "causal_result_completed",
}


def productive_delta(
    previous: dict[str, Any], current: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    delta: dict[str, Any] = {}
    positive = False
    regression = False
    for key in sorted(COUNTER_KEYS):
        old = previous.get(key, 0)
        new = current.get(key, 0)
        if isinstance(old, bool) or not isinstance(old, (int, float)):
            old = 0
        if isinstance(new, bool) or not isinstance(new, (int, float)):
            new = 0
        change = new - old
        if change:
            delta[key] = change
            if change > 0:
                positive = True
            else:
                regression = True
    old_version = previous.get("causal_result_version")
    new_version = current.get("causal_result_version")
    if new_version != old_version and new_version is not None:
        delta["causal_result_version"] = {"from": old_version, "to": new_version}
        positive = True
    return {"values": delta, "positive": positive, "regression": regression}, positive


def failure_reason(snapshot: dict[str, Any]) -> str | None:
    health = snapshot.get("health", {})
    if not isinstance(health, dict):
        return None
    scheduler_error = health.get("scheduler_error")
    if scheduler_error:
        return f"bounded scheduler inspection failed: {scheduler_error}"
    causal = health.get("causal_result", {})
    if isinstance(causal, dict) and causal.get("status") in {
        "FAILED",
        "ERROR",
        "INVALID",
    }:
        return f"causal marker status={causal.get('status')}"
    pusht = health.get("pusht", {})
    if isinstance(pusht, dict):
        for arm in ("dinocular", "dinocular_zerodepth"):
            record = pusht.get(arm, {})
            if not isinstance(record, dict):
                return f"missing PushT marker record for {arm}"
            if not record.get("controller_present"):
                return f"resume controller has no seed-one chain for {arm}"
            if record.get("failure"):
                return f"resume controller failure for {arm}: {record['failure']}"
            if record.get("continuation") in {
                "FAILED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "CANCELLED",
                "REVOKED",
            }:
                return f"PushT continuation {arm} is {record['continuation']}"
    return None


def stream_actions(snapshot: dict[str, Any]) -> dict[str, str]:
    productive = snapshot.get("productive", {})
    health = snapshot.get("health", {})
    actions: dict[str, str] = {}
    terminal = int(productive.get("wall_terminal_inputs_accepted", 0) or 0)
    predictions = int(productive.get("wall_prediction_episodes_completed", 0) or 0)
    planning = int(productive.get("wall_planning_targets_completed", 0) or 0)
    if terminal < 192:
        actions["Wall terminal inputs"] = (
            f"Continue authentic Wall terminal-input recovery for the remaining {192 - terminal} routes; "
            "accept only each route's original frame-50 source and do not fabricate depth."
        )
    if predictions < 1728:
        actions["Wall prediction episodes"] = (
            f"After accepted terminal inputs, run the fixed Wall prediction evaluations; {1728 - predictions} episodes remain."
        )
    if planning < 450:
        actions["Wall planning targets"] = (
            f"After the fixed prediction inputs exist, materialize/run the fixed Wall planning targets; {450 - planning} targets remain."
        )
    pusht = health.get("pusht", {}) if isinstance(health, dict) else {}
    for arm in ("dinocular", "dinocular_zerodepth"):
        record = pusht.get(arm, {}) if isinstance(pusht, dict) else {}
        state = record.get("continuation", "UNKNOWN")
        failure = record.get("failure")
        label = f"PushT {arm}"
        if failure or state in {
            "FAILED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "CANCELLED",
        }:
            actions[label] = (
                "Read the controller event and exact failed-process receipt once; "
                "repair the causal failure before any same-lineage continuation."
            )
        elif state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            actions[label] = (
                f"Preserve the one controller-owned continuation ({state}); do not submit, "
                "cancel, requeue, signal, or duplicate it."
            )
        else:
            actions[label] = (
                "Inspect the controller state and one exact scheduler receipt, then restore "
                "only the same unique continuation through the controller."
            )
    if not productive.get("causal_result_completed"):
        actions["Causal repair/result"] = (
            "Keep the current Wall/PushT productive package moving; record this marker only "
            "when a causal repair or scientific result is actually complete."
        )
    return actions


def bounded_payload(
    event: str,
    snapshot: dict[str, Any],
    delta: dict[str, Any],
    state: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    health = snapshot.get("health", {})
    pusht = health.get("pusht", {}) if isinstance(health, dict) else {}
    compact_pusht = {}
    for arm in ("dinocular", "dinocular_zerodepth"):
        record = pusht.get(arm, {}) if isinstance(pusht, dict) else {}
        compact_pusht[arm] = {
            "accepted_step": snapshot.get("productive", {}).get(
                f"pusht_{arm}_accepted_step", 0
            ),
            "job_id": record.get("job_id"),
            "continuation": record.get("continuation", "UNKNOWN"),
            "phase": record.get("phase"),
            "training_complete": record.get("training_complete", False),
            "failure": record.get("failure"),
            "legitimately_running_or_queued": record.get(
                "legitimately_running_or_queued", False
            ),
        }
    productive = snapshot.get("productive", {})
    return {
        "service": "dinocular-operations-productive-silence",
        "schema": SCHEMA,
        "owner": "Operations Lead",
        "event": event,
        "reason": reason,
        "observed_at": now_iso(),
        "maximum_silence_seconds": MAX_SILENCE_SECONDS,
        "productive_delta_since_last_route": delta,
        "measured": {
            "Wall": {
                "terminal_inputs_accepted": productive.get(
                    "wall_terminal_inputs_accepted", 0
                ),
                "prediction_episodes_completed": productive.get(
                    "wall_prediction_episodes_completed", 0
                ),
                "planning_targets_completed": productive.get(
                    "wall_planning_targets_completed", 0
                ),
            },
            "PushT": compact_pusht,
            "causal_result_completed": productive.get("causal_result_completed", 0),
            "causal_result_version": productive.get("causal_result_version"),
        },
        "next_productive_actions": stream_actions(snapshot),
        "marker_sources": {
            "controller": (health.get("pusht", {}) or {})
            .get("dinocular", {})
            .get("progress_source"),
            "causal_result": (health.get("causal_result", {}) or {}).get("source"),
        },
        "prior_measured_at": state.get("last_productive_at"),
    }


def invoke_helper(payload: dict[str, Any]) -> dict[str, Any]:
    message = canonical(payload)
    try:
        result = subprocess.run(
            [HELPER, "operations", message],
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
            env={
                **os.environ,
                "HERDR_ENV": "1",
                "HERDR_WORKSPACE_ID": WORKSPACE_ID,
            },
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": f"helper_exception={error}"}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-400:],
        "stderr": result.stderr[-400:],
    }


def deliver(
    event: str,
    snapshot: dict[str, Any],
    delta: dict[str, Any],
    state: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    payload = bounded_payload(event, snapshot, delta, state, reason)
    delivery = invoke_helper(payload)
    receipt = {
        "schema": SCHEMA,
        "service": payload["service"],
        "owner": payload["owner"],
        "event": event,
        "receipt_at": now_iso(),
        "payload": payload,
        "delivery": delivery,
    }
    atomic_json_write(RECEIPT_PATH, receipt)
    log(
        f"ROUTE event={event} reason={reason} delivery_ok={delivery.get('ok')} "
        f"receipt={RECEIPT_PATH}"
    )
    return receipt


def load_state() -> dict[str, Any]:
    payload = read_json(STATE_PATH)
    return payload if isinstance(payload, dict) else {}


def save_state(state: dict[str, Any]) -> None:
    atomic_json_write(STATE_PATH, state)


def process_once() -> None:
    state = load_state()
    snapshot = collect_snapshot()
    if "error" in snapshot:
        failures = int(state.get("consecutive_collection_failures", 0)) + 1
        state["consecutive_collection_failures"] = failures
        if failures >= RETRY_LIMIT:
            fingerprint = digest({"event": "MONITOR_ERROR", "error": snapshot["error"]})
            if fingerprint != state.get("last_failure_fingerprint"):
                deliver(
                    "MONITOR_ERROR",
                    {
                        "productive": state.get("last_event_productive", {}),
                        "health": {"collection_error": snapshot["error"]},
                    },
                    {"collection_error": snapshot["error"]},
                    state,
                    "bounded marker collection failed after retry limit",
                )
                state["last_failure_fingerprint"] = fingerprint
        save_state(state)
        log(f"COLLECTION_ERROR failures={failures} detail={snapshot['error']}")
        return

    state["consecutive_collection_failures"] = 0
    current_productive = snapshot.get("productive", {})
    prior_productive = state.get("last_event_productive")
    if not isinstance(prior_productive, dict):
        state.update(
            {
                "schema": SCHEMA,
                "last_event_productive": current_productive,
                "last_productive_at": time.time(),
                "last_silence_at": 0,
                "last_snapshot": snapshot,
            }
        )
        save_state(state)
        log("BASELINE_WRITTEN no_route")
        return

    failure = failure_reason(snapshot)
    if failure:
        fingerprint = digest({"event": "LONG_PROCESS_FAILURE", "reason": failure})
        if fingerprint != state.get("last_failure_fingerprint"):
            deliver(
                "LONG_PROCESS_FAILURE",
                snapshot,
                {"failure": failure},
                state,
                failure,
            )
            state["last_failure_fingerprint"] = fingerprint
    else:
        state["last_failure_fingerprint"] = None

    comparison, positive = productive_delta(prior_productive, current_productive)
    if comparison["regression"]:
        fingerprint = digest(
            {"event": "MARKER_REGRESSION", "productive": current_productive}
        )
        if fingerprint != state.get("last_failure_fingerprint"):
            deliver(
                "MARKER_REGRESSION",
                snapshot,
                comparison["values"],
                state,
                "a bounded productive marker moved backwards; inspect the marker source before acting",
            )
            state["last_failure_fingerprint"] = fingerprint
    elif positive:
        fingerprint = digest(current_productive)
        if fingerprint != state.get("last_delta_fingerprint"):
            deliver(
                "PRODUCTIVE_DELTA",
                snapshot,
                comparison["values"],
                state,
                "positive bounded Wall/PushT result delta",
            )
            state["last_delta_fingerprint"] = fingerprint
        state["last_event_productive"] = current_productive
        state["last_productive_at"] = time.time()
        state["last_silence_at"] = 0
    else:
        last_productive_at = float(state.get("last_productive_at", time.time()))
        last_silence_at = float(state.get("last_silence_at", 0))
        silence_due = time.time() - last_productive_at >= MAX_SILENCE_SECONDS
        silence_repeat_due = time.time() - last_silence_at >= MAX_SILENCE_SECONDS
        if silence_due and silence_repeat_due:
            deliver(
                "MAX_SILENCE",
                snapshot,
                {"since_last_productive_delta": comparison["values"]},
                state,
                f"no productive delta for {MAX_SILENCE_SECONDS} seconds",
            )
            state["last_silence_at"] = time.time()

    state["last_snapshot"] = snapshot
    save_state(state)


def main() -> int:
    if POLL_SECONDS < 0 or MAX_SILENCE_SECONDS <= 0 or RETRY_LIMIT <= 0:
        print("invalid non-positive service interval/retry configuration", file=sys.stderr)
        return 2
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("DUPLICATE_SUPPRESSED existing singleton lock")
        return 0

    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log(
        f"START schema={SCHEMA} poll_seconds={POLL_SECONDS} "
        f"max_silence_seconds={MAX_SILENCE_SECONDS} owner=Operations "
        f"remote={not DISABLE_REMOTE}"
    )
    while not stop:
        process_once()
        if RUN_ONCE:
            break
        for _ in range(POLL_SECONDS):
            if stop:
                break
            time.sleep(1)
    log("STOP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

