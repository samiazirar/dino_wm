#!/bin/bash

# Poll a fixed set of Marvin jobs and always leave an atomic receipt before
# attempting to notify Operations.  This deliberately avoids `set -e`: a
# transient SSH/sacct failure is an observation to record, not a reason to
# disappear without a receipt.
set -uo pipefail

JOBS=${1:?comma-separated jobs required}
EVENT=${2:?event path required}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-60}
RETRY_LIMIT=${RETRY_LIMIT:-3}
MAX_POLLS=${RG_DEPTH_WATCH_MAX_POLLS:-0} # Test-only bounded loop; zero means forever.
LOG=${RG_DEPTH_WATCH_LOG:-"${EVENT%.event.json}.watch.log"}

IFS=, read -r -a EXPECTED_JOB_IDS <<< "$JOBS"
if [ "${#EXPECTED_JOB_IDS[@]}" -eq 0 ]; then
    printf 'no jobs supplied\n' >&2
    exit 2
fi
for job in "${EXPECTED_JOB_IDS[@]}"; do
    case "$job" in
        ''|*[!0-9]*)
            printf 'invalid job ID: %s\n' "$job" >&2
            exit 2
            ;;
    esac
done
case "$RETRY_LIMIT" in
    ''|*[!0-9]*|0)
        printf 'RETRY_LIMIT must be a positive integer\n' >&2
        exit 2
        ;;
esac
case "$MAX_POLLS" in
    ''|*[!0-9]*)
        printf 'RG_DEPTH_WATCH_MAX_POLLS must be a nonnegative integer\n' >&2
        exit 2
        ;;
esac

SNAPSHOT_FILE=$(mktemp "${EVENT}.snapshot.XXXXXX") || exit 2
PARSED_FILE=$(mktemp "${EVENT}.parsed.XXXXXX") || {
    rm -f "$SNAPSHOT_FILE"
    exit 2
}
cleanup() {
    rm -f "$SNAPSHOT_FILE" "$PARSED_FILE"
}
trap cleanup EXIT

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >> "$LOG" 2>&1 || true
}

trim() {
    local value=$1
    value=${value#"${value%%[![:space:]]*}"}
    value=${value%"${value##*[![:space:]]}"}
    printf '%s' "$value"
}

write_receipt() {
    local kind=$1
    local reason=$2
    local failures=$3
    local temporary

    temporary=$(mktemp "${EVENT}.tmp.XXXXXX") || {
        log "RECEIPT_WRITE_ERROR kind=$kind reason=$reason stage=mktemp"
        return 1
    }
    if ! python3 - "$temporary" "$kind" "$reason" "$failures" "$JOBS" \
        "$SNAPSHOT_FILE" "$PARSED_FILE" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import sys

temporary, kind, reason, failures, jobs, snapshot_path, parsed_path = sys.argv[1:]
expected = jobs.split(",")
snapshot = pathlib.Path(snapshot_path).read_text(errors="replace")
records = {}
for line in pathlib.Path(parsed_path).read_text(errors="replace").splitlines():
    job_id, state, exit_code, elapsed_raw = line.split("|", 3)
    records[job_id] = {
        "job_id": job_id,
        "state": state,
        "exit_code": exit_code,
        "elapsed_raw": elapsed_raw,
    }

receipt = {
    "schema": "dinocular.rg-depth-watch-receipt.v1",
    "kind": kind,
    "reason": reason,
    "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "expected_job_ids": expected,
    "jobs": [records[job_id] for job_id in expected if job_id in records],
    "consecutive_failures": int(failures),
    "snapshot_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
    "snapshot": snapshot,
}
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
    then
        rm -f "$temporary"
        log "RECEIPT_WRITE_ERROR kind=$kind reason=$reason stage=json"
        return 1
    fi
    if ! mv -f "$temporary" "$EVENT"; then
        rm -f "$temporary"
        log "RECEIPT_WRITE_ERROR kind=$kind reason=$reason stage=rename"
        return 1
    fi
    log "RECEIPT_WRITTEN kind=$kind reason=$reason path=$EVENT"
}

notify_operations() {
    local kind=$1
    if herdr-role-message operations \
        "MATERIAL ROPE/GRANULAR DEPTH WATCH EVENT ($kind). Read $EVENT once and inspect only the four corrected canary jobs and named repair outputs. Continue the same task with the concrete correction or dependent stage; do not model-poll." \
        >> "$LOG" 2>&1; then
        log "DELIVERY_OK kind=$kind"
    else
        # The receipt has already been atomically renamed into place, so a
        # stale messenger or temporary Herdr failure cannot erase the event.
        log "DELIVERY_ERROR kind=$kind receipt=$EVENT"
    fi
}

capture_snapshot() {
    : > "$SNAPSHOT_FILE" || return 1
    ssh -o BatchMode=yes -o ConnectTimeout=15 marvin \
        "sacct -X -n -P -j '$JOBS' -o JobIDRaw,State,ExitCode,ElapsedRaw" \
        > "$SNAPSHOT_FILE" 2>&1
}

PARSE_ERROR=
parse_snapshot() {
    local job state exit_code elapsed_raw ignored
    local -A seen=()
    local -a missing=()

    PARSE_ERROR=
    : > "$PARSED_FILE" || {
        PARSE_ERROR=PARSED_RECEIPT_TEMPORARY_WRITE_FAILED
        return 1
    }
    while IFS='|' read -r job state exit_code elapsed_raw ignored; do
        job=$(trim "$job")
        case ",$JOBS," in
            *",$job,"*) ;;
            *) continue ;;
        esac
        state=$(trim "${state%%+*}")
        exit_code=$(trim "$exit_code")
        elapsed_raw=$(trim "$elapsed_raw")
        if [ -z "$state" ]; then
            PARSE_ERROR="EMPTY_STATE_FOR_JOB_$job"
            return 1
        fi
        if [[ -v "seen[$job]" ]]; then
            PARSE_ERROR="DUPLICATE_JOB_ROW_$job"
            return 1
        fi
        seen[$job]=1
        printf '%s|%s|%s|%s\n' "$job" "$state" "$exit_code" "$elapsed_raw" >> "$PARSED_FILE" || {
            PARSE_ERROR=PARSED_RECEIPT_TEMPORARY_WRITE_FAILED
            return 1
        }
    done < "$SNAPSHOT_FILE"

    for job in "${EXPECTED_JOB_IDS[@]}"; do
        if [[ ! -v "seen[$job]" ]]; then
            missing+=("$job")
        fi
    done
    if [ "${#missing[@]}" -ne 0 ]; then
        PARSE_ERROR="MISSING_EXPECTED_JOB_ROWS_$(IFS=,; printf '%s' "${missing[*]}")"
        return 1
    fi
}

terminal_state() {
    case "$1" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

FAILURES=0
POLLS=0
while :; do
    POLLS=$((POLLS + 1))
    PARSE_ERROR=
    if ! capture_snapshot; then
        FAILURES=$((FAILURES + 1))
        log "POLL_ERROR kind=SSH_OR_SACCT_FAILED failures=$FAILURES"
    elif ! parse_snapshot; then
        FAILURES=$((FAILURES + 1))
        log "POLL_ERROR kind=$PARSE_ERROR failures=$FAILURES"
    else
        FAILURES=0
        terminal=1
        failed=0
        while IFS='|' read -r job state exit_code elapsed_raw; do
            if ! terminal_state "$state"; then
                terminal=0
            elif [ "$state" != COMPLETED ]; then
                failed=1
            fi
        done < "$PARSED_FILE"
        if [ "$terminal" -eq 1 ]; then
            if ! write_receipt TERMINAL "ALL_EXPECTED_JOBS_TERMINAL_failed=$failed" 0; then
                exit 2
            fi
            notify_operations TERMINAL
            exit 0
        fi
        log "POLL_OK kind=NONTERMINAL"
    fi

    if [ "$FAILURES" -ge "$RETRY_LIMIT" ]; then
        if ! write_receipt MONITOR_ERROR "${PARSE_ERROR:-SSH_OR_SACCT_FAILED}" "$FAILURES"; then
            exit 2
        fi
        notify_operations MONITOR_ERROR
        exit 0
    fi
    if [ "$MAX_POLLS" -gt 0 ] && [ "$POLLS" -ge "$MAX_POLLS" ]; then
        if ! write_receipt MONITOR_ERROR MAX_POLLS_REACHED_WITH_NONTERMINAL_JOBS 0; then
            exit 2
        fi
        notify_operations MONITOR_ERROR
        exit 0
    fi
    sleep "$INTERVAL_SECONDS" || log "SLEEP_ERROR interval=$INTERVAL_SECONDS"
done
