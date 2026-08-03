#!/usr/bin/env bash
# Adapter for exactly one Wall materializer job. The trusted watcher is not
# modified; its scheduler receipt is captured, then the exact terminal output
# is checked and one final event is sent to Ada.
set -uo pipefail

JOB_ID=${WALL_MATERIALIZER_JOB_ID:-26814624}
BUNDLE_PATH=${WALL_MATERIALIZER_BUNDLE_PATH:-/lustre/mlnvme/data/sazirar_hpc-marvin-ssd/projects/dinocular-wm/outputs/campaign-evaluation/eval36-20260728a/inputs/wall-terminal}
RECEIPT_PATH=${WALL_MATERIALIZER_RECEIPT_PATH:-$BUNDLE_PATH/terminal_input_receipt.json}
TRUSTED_WATCHER=${WALL_TERMINAL_TRUSTED_WATCHER:-/home/user/azirar/dinocular-wm-worktrees/selene-watcher-route-repair/tools/rg_depth_event_watch.sh}
EVENT_PATH=${WALL_TERMINAL_EVENT_PATH:-/tmp/dinocular-wall-materializer-${JOB_ID}.event.json}
RAW_EVENT_PATH=${WALL_TERMINAL_RAW_EVENT_PATH:-/tmp/dinocular-wall-materializer-${JOB_ID}.trusted.event.json}
LOG_PATH=${WALL_TERMINAL_LOG_PATH:-/tmp/dinocular-wall-materializer-${JOB_ID}.watch.log}
RAW_LOG_PATH=${WALL_TERMINAL_RAW_LOG_PATH:-/tmp/dinocular-wall-materializer-${JOB_ID}.trusted.watch.log}
PID_PATH=${WALL_TERMINAL_PID_PATH:-/tmp/dinocular-wall-materializer-${JOB_ID}.trusted.pid}
STATE_PATH=${WALL_TERMINAL_STATE_PATH:-/tmp/dinocular-wall-materializer-${JOB_ID}.state}
LOCK_PATH=${WALL_TERMINAL_LOCK_PATH:-/tmp/dinocular-wall-materializer-${JOB_ID}.lock}
MAX_SILENCE_SECONDS=${WALL_TERMINAL_MAX_SILENCE_SECONDS:-10800}
POLL_SECONDS=${WALL_TERMINAL_POLL_SECONDS:-600}
RETRY_LIMIT=${WALL_TERMINAL_RETRY_LIMIT:-3}
RESTART_SECONDS=${WALL_TERMINAL_RESTART_SECONDS:-60}
SSH_BIN=${WALL_TERMINAL_SSH_BIN:-ssh}
ROLE_MESSAGE=${WALL_TERMINAL_ROLE_MESSAGE:-/home/user/azirar/.local/bin/herdr-role-message}
WORKSPACE_ID=${HERDR_WORKSPACE_ID:-w3}
ADA_TARGET='Suborchestrator · Ada Wall Fixed Evaluations'

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$EVENT_PATH")" "$(dirname "$STATE_PATH")"

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >> "$LOG_PATH" 2>&1 || true
}

write_state() {
    local status=$1
    local temporary
    temporary=$(mktemp "${STATE_PATH}.tmp.XXXXXX") || return 1
    {
        printf 'route_id=wall-materializer-%s\n' "$JOB_ID"
        printf 'job_id=%s\n' "$JOB_ID"
        printf 'trusted_watcher=%s\n' "$TRUSTED_WATCHER"
        printf 'trusted_watcher_commit=8475e3865159324ef1ce5a7c3903bc685cb53a2c\n'
        printf 'owner=%s\n' "$ADA_TARGET"
        printf 'bundle=%s\n' "$BUNDLE_PATH"
        printf 'receipt=%s\n' "$RECEIPT_PATH"
        printf 'event=%s\n' "$EVENT_PATH"
        printf 'raw_event=%s\n' "$RAW_EVENT_PATH"
        printf 'maximum_silence_seconds=%s\n' "$MAX_SILENCE_SECONDS"
        printf 'poll_seconds=%s\n' "$POLL_SECONDS"
        printf 'scheduler_mutation=false\n'
        printf 'evaluation_launch=false\n'
        printf 'model_polling=false\n'
        printf 'status=%s\n' "$status"
        printf 'started_at=%s\n' "${STARTED_AT:-unknown}"
        printf 'deadline_epoch=%s\n' "${DEADLINE_EPOCH:-unknown}"
    } > "$temporary" || {
        rm -f "$temporary"
        return 1
    }
    chmod 600 "$temporary"
    mv -f "$temporary" "$STATE_PATH"
}

raw_field() {
    local key=$1
    local default=${2:-}
    [[ -r "$RAW_EVENT_PATH" ]] || {
        printf '%s' "$default"
        return 0
    }
    python3 - "$RAW_EVENT_PATH" "$key" "$default" <<'PY'
import json
import sys
try:
    payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
    value = payload.get(sys.argv[2], sys.argv[3])
except (OSError, ValueError, TypeError):
    value = sys.argv[3]
print(sys.argv[3] if value is None else str(value))
PY
}

final_event_field() {
    local key=$1
    local default=${2:-}
    [[ -r "$EVENT_PATH" ]] || {
        printf '%s' "$default"
        return 0
    }
    python3 - "$EVENT_PATH" "$key" "$default" <<'PY'
import json
import sys
try:
    payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
    value = payload.get(sys.argv[2], sys.argv[3])
except (OSError, ValueError, TypeError):
    value = sys.argv[3]
print(sys.argv[3] if value is None else str(value))
PY
}

all_jobs_completed() {
    [[ -r "$RAW_EVENT_PATH" ]] || {
        printf '0'
        return 0
    }
    python3 - "$RAW_EVENT_PATH" <<'PY'
import json
import sys
try:
    payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
    jobs = payload.get("jobs", [])
except (OSError, ValueError, TypeError):
    jobs = []
print("1" if jobs and all(str(row.get("state", "")).split("+", 1)[0] == "COMPLETED" for row in jobs) else "0")
PY
}

probe_output() {
    local probe_file=$1
    local state route_count receipt_sha terminal_index
    : > "$probe_file" || return 1
    if ! "$SSH_BIN" -o BatchMode=yes -o ConnectTimeout=15 marvin bash -s -- \
        "$BUNDLE_PATH" "$RECEIPT_PATH" > "$probe_file" 2>&1 <<'REMOTE'
set -u
bundle=$1
receipt=$2
if [[ ! -d "$bundle" ]]; then
    printf 'OUTPUT_STATE=MISSING_BUNDLE\n'
    exit 0
fi
if [[ ! -f "$receipt" ]]; then
    printf 'OUTPUT_STATE=MISSING_RECEIPT\n'
    exit 0
fi
python3 - "$receipt" <<'PY'
import hashlib
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    print("OUTPUT_STATE=INVALID_RECEIPT")
    raise SystemExit(0)
schema_ok = payload.get("schema") == "dinocular.wall-terminal-input-bundle.v1"
state_ok = payload.get("state") == "PASS"
route_count = payload.get("route_count")
terminal_index = payload.get("terminal_observation_index")
if schema_ok and state_ok and route_count == 192 and terminal_index == 50:
    print("OUTPUT_STATE=PASS")
    print(f"OUTPUT_ROUTE_COUNT={route_count}")
    print(f"OUTPUT_TERMINAL_INDEX={terminal_index}")
    print(f"OUTPUT_RECEIPT_SHA256={hashlib.sha256(path.read_bytes()).hexdigest()}")
else:
    print("OUTPUT_STATE=INVALID_RECEIPT")
    print(f"OUTPUT_ROUTE_COUNT={route_count}")
    print(f"OUTPUT_TERMINAL_INDEX={terminal_index}")
PY
REMOTE
    then
        printf 'OUTPUT_STATE=QUERY_ERROR\n' >> "$probe_file"
    fi
    state=$(sed -n 's/^OUTPUT_STATE=//p' "$probe_file" | tail -n 1)
    route_count=$(sed -n 's/^OUTPUT_ROUTE_COUNT=//p' "$probe_file" | tail -n 1)
    receipt_sha=$(sed -n 's/^OUTPUT_RECEIPT_SHA256=//p' "$probe_file" | tail -n 1)
    terminal_index=$(sed -n 's/^OUTPUT_TERMINAL_INDEX=//p' "$probe_file" | tail -n 1)
    [[ -n "$state" ]] || state=QUERY_ERROR
    case "$state" in
        PASS) reason="bundle_and_receipt_PASS route_count=$route_count terminal_index=$terminal_index" ;;
        MISSING_BUNDLE) reason="expected bundle missing at $BUNDLE_PATH" ;;
        MISSING_RECEIPT) reason="expected receipt missing at $RECEIPT_PATH" ;;
        INVALID_RECEIPT) reason="expected receipt is not the PASS 192-route terminal bundle" ;;
        *) reason="bounded output probe failed: $state" ;;
    esac
    OUTPUT_STATE=$state
    OUTPUT_REASON=$reason
    OUTPUT_ROUTE_COUNT=$route_count
    OUTPUT_RECEIPT_SHA=$receipt_sha
}

write_receipt() {
    local event=$1 reason=$2 output_state=$3 output_reason=$4
    local output_route_count=$5 output_receipt_sha=$6 temporary
    temporary=$(mktemp "${EVENT_PATH}.tmp.XXXXXX") || return 1
    if ! python3 - "$temporary" "$event" "$reason" "$output_state" "$output_reason" \
        "$output_route_count" "$output_receipt_sha" "$RAW_EVENT_PATH" "$JOB_ID" \
        "$BUNDLE_PATH" "$RECEIPT_PATH" "$TRUSTED_WATCHER" "$EVENT_PATH" \
        "$MAX_SILENCE_SECONDS" "$STARTED_AT" "$DEADLINE_EPOCH" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
(
    temporary, event, reason, output_state, output_reason, output_route_count,
    output_receipt_sha, raw_event_path, job_id, bundle_path, receipt_path,
    trusted_watcher, event_path, maximum_silence, started_at, deadline_epoch,
) = sys.argv[1:]
try:
    raw = json.loads(pathlib.Path(raw_event_path).read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    raw = {}
try:
    raw_sha = hashlib.sha256(pathlib.Path(raw_event_path).read_bytes()).hexdigest()
except OSError:
    raw_sha = None
try:
    route_count = int(output_route_count)
except (TypeError, ValueError):
    route_count = None
payload = {
    "schema": "dinocular.wall-materializer-terminal-route.v1",
    "route_id": f"wall-materializer-{job_id}",
    "job_id": int(job_id),
    "owner": "Suborchestrator · Ada Wall Fixed Evaluations",
    "event": event,
    "reason": reason,
    "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "maximum_silence_seconds": int(maximum_silence),
    "started_at": started_at,
    "deadline_epoch": int(deadline_epoch),
    "scheduler": {
        "raw_receipt_path": raw_event_path,
        "raw_receipt_sha256": raw_sha,
        "kind": raw.get("kind"),
        "reason": raw.get("reason"),
        "expected_job_ids": raw.get("expected_job_ids", []),
        "jobs": raw.get("jobs", []),
    },
    "output": {
        "bundle_path": bundle_path,
        "receipt_path": receipt_path,
        "state": output_state,
        "reason": output_reason,
        "route_count": route_count,
        "receipt_sha256": output_receipt_sha or None,
    },
    "trusted_watcher": {
        "path": trusted_watcher,
        "commit": "8475e3865159324ef1ce5a7c3903bc685cb53a2c",
        "unchanged": True,
    },
    "restrictions": {
        "scheduler_mutation": False,
        "evaluation_launched": False,
        "model_polling": False,
    },
    "delivery": {"state": "PENDING"},
}
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, event_path)
PY
    then
        rm -f "$temporary"
        log "RECEIPT_WRITE_ERROR event=$event"
        return 1
    fi
    log "RECEIPT_WRITTEN event=$event path=$EVENT_PATH"
}

send_to_ada() {
    local event=$1 reason=$2 output_state=$3 output_reason=$4
    local message returncode=1
    message="Wall materializer route event=$event job=$JOB_ID owner=$ADA_TARGET reason=$reason output_state=$output_state output_reason=$output_reason receipt=$EVENT_PATH. On TERMINAL_SUCCESS, read this receipt once and begin the fresh acceptance package only after its bundle/receipt fields pass. On TERMINAL_FAILURE, name the scheduler/output cause and repair only this materializer. On MAX_SILENCE, perform one fresh bounded read-only scheduler/output snapshot through Luna; do not submit, cancel, requeue, signal, or model-poll."
    for delay in 0 1 2; do
        (( delay > 0 )) && sleep "$delay"
        if HERDR_ENV=1 HERDR_WORKSPACE_ID="$WORKSPACE_ID" HERDR_WAKE_NO_DISARM=1 \
            "$ROLE_MESSAGE" named "$ADA_TARGET" "$message" >> "$LOG_PATH" 2>&1; then
            returncode=0
            break
        fi
    done
    if [[ "$returncode" -eq 0 ]]; then
        log "DELIVERY_OK event=$event target=$ADA_TARGET"
    else
        log "DELIVERY_ERROR event=$event target=$ADA_TARGET receipt=$EVENT_PATH"
    fi
    return "$returncode"
}

finalize_event() {
    local event=$1 reason=$2 output_state=${3:-NOT_PROBED}
    local output_reason=${4:-not probed for silence}
    local output_route_count=${5:-} output_receipt_sha=${6:-}
    write_state "$event" || true
    write_receipt "$event" "$reason" "$output_state" "$output_reason" \
        "$output_route_count" "$output_receipt_sha" || return 1
    send_to_ada "$event" "$reason" "$output_state" "$output_reason" || true
}

[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { printf 'job scope must be a numeric Slurm job id\n' >&2; exit 2; }
[[ "$MAX_SILENCE_SECONDS" =~ ^[1-9][0-9]*$ && "$POLL_SECONDS" =~ ^[0-9]+$ && "$RETRY_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    printf 'invalid route interval configuration\n' >&2
    exit 2
}
[[ -x "$TRUSTED_WATCHER" ]] || { printf 'trusted watcher is not executable: %s\n' "$TRUSTED_WATCHER" >&2; exit 2; }

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    log "DUPLICATE_SUPPRESSED lock=$LOCK_PATH job=$JOB_ID"
    exit 0
fi
if [[ -r "$EVENT_PATH" ]]; then
    prior_event=$(final_event_field event)
    case "$prior_event" in
        TERMINAL_SUCCESS|TERMINAL_FAILURE|MAX_SILENCE)
            log "ALREADY_TERMINATED event=$prior_event job=$JOB_ID"
            exit 0
            ;;
    esac
fi

STARTED_EPOCH=$(date +%s)
STARTED_AT=$(date --iso-8601=seconds)
DEADLINE_EPOCH=$((STARTED_EPOCH + MAX_SILENCE_SECONDS))
write_state ACTIVE || true
log "START route_id=wall-materializer-$JOB_ID job=$JOB_ID trusted_watcher=$TRUSTED_WATCHER trusted_commit=8475e3865159324ef1ce5a7c3903bc685cb53a2c owner=$ADA_TARGET maximum_silence_seconds=$MAX_SILENCE_SECONDS poll_seconds=$POLL_SECONDS"

CAPTURE_DIR=$(mktemp -d "/tmp/dinocular-wall-materializer-${JOB_ID}.capture.XXXXXX") || exit 2
BASH_ENV_FILE="$CAPTURE_DIR/bash-env"
CAPTURE_FILE="$CAPTURE_DIR/messages.log"
printf '%s\n' \
    'herdr-role-message() {' \
    '    printf "%s\n" "$*" >> "$WALL_TERMINAL_CAPTURE_FILE"' \
    '    return 0' \
    '}' > "$BASH_ENV_FILE"
export WALL_TERMINAL_CAPTURE_FILE="$CAPTURE_FILE"
cleanup() {
    rm -f "$BASH_ENV_FILE" "$CAPTURE_FILE" "$PID_PATH"
    rmdir "$CAPTURE_DIR" 2>/dev/null || true
}
trap cleanup EXIT

while :; do
    now=$(date +%s)
    remaining=$((DEADLINE_EPOCH - now))
    if (( remaining <= 0 )); then
        finalize_event MAX_SILENCE \
            "no terminal scheduler/output result for $MAX_SILENCE_SECONDS seconds; Ada must perform one fresh bounded scheduler/output snapshot" \
            NOT_PROBED "silence recovery is delegated to Ada through Luna" || true
        exit 0
    fi

    rm -f "$RAW_EVENT_PATH"
    run_status=0
    /usr/bin/timeout --foreground --signal=TERM --kill-after=30 "$remaining" \
        env BASH_ENV="$BASH_ENV_FILE" HERDR_ENV=1 HERDR_WORKSPACE_ID="$WORKSPACE_ID" \
        HERDR_WAKE_NO_DISARM=1 RG_DEPTH_WATCH_LOG="$RAW_LOG_PATH" \
        RG_DEPTH_WATCH_PID_FILE="$PID_PATH" INTERVAL_SECONDS="$POLL_SECONDS" \
        RETRY_LIMIT="$RETRY_LIMIT" RG_DEPTH_WATCH_INITIAL_DELAY_SECONDS=0 \
        RG_DEPTH_WATCH_MAX_POLLS=0 \
        bash "$TRUSTED_WATCHER" "$JOB_ID" "$RAW_EVENT_PATH" || run_status=$?

    raw_kind=$(raw_field kind)
    if [[ "$raw_kind" == TERMINAL ]]; then
        probe_file=$(mktemp "/tmp/dinocular-wall-materializer-${JOB_ID}.probe.XXXXXX") || exit 2
        OUTPUT_STATE=QUERY_ERROR
        OUTPUT_REASON='bounded output probe did not run'
        OUTPUT_ROUTE_COUNT=
        OUTPUT_RECEIPT_SHA=
        probe_output "$probe_file" || true
        rm -f "$probe_file"
        if [[ "$(all_jobs_completed)" == 1 && "$OUTPUT_STATE" == PASS ]]; then
            finalize_event TERMINAL_SUCCESS \
                "scheduler=COMPLETED and authentic terminal bundle/receipt is PASS for all 192 routes" \
                "$OUTPUT_STATE" "$OUTPUT_REASON" "$OUTPUT_ROUTE_COUNT" "$OUTPUT_RECEIPT_SHA" || true
        else
            finalize_event TERMINAL_FAILURE \
                "scheduler=$(raw_field reason unknown); output=$OUTPUT_REASON" \
                "$OUTPUT_STATE" "$OUTPUT_REASON" "$OUTPUT_ROUTE_COUNT" "$OUTPUT_RECEIPT_SHA" || true
        fi
        exit 0
    fi

    now=$(date +%s)
    remaining=$((DEADLINE_EPOCH - now))
    if (( remaining <= 0 )); then
        finalize_event MAX_SILENCE \
            "no terminal scheduler/output result for $MAX_SILENCE_SECONDS seconds; Ada must perform one fresh bounded scheduler/output snapshot" \
            NOT_PROBED "silence recovery is delegated to Ada through Luna" || true
        exit 0
    fi
    log "TRUSTED_WATCHER_RETURN status=$run_status raw_kind=${raw_kind:-none} restarting_after=$RESTART_SECONDS"
    sleep_for=$RESTART_SECONDS
    (( sleep_for > remaining )) && sleep_for=$remaining
    sleep "$sleep_for"
done
