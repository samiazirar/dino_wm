#!/bin/bash

# Run the watcher in the foreground of a dedicated ordinary Herdr command pane.
# Do not background it from a model tool call: command-runner cgroups can still
# reclaim such descendants despite nohup or setsid.  The watcher writes/removes
# its own PID file atomically, so the PID always names the foreground process.
set -uo pipefail

JOBS=${1:?comma-separated jobs required}
EVENT=${2:?event path required}
PID_FILE=${3:?PID path required}
LOG=${4:?log path required}
WATCHER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WATCHER="$WATCHER_DIR/rg_depth_event_watch.sh"
if [ -r "$PID_FILE" ]; then
    old_pid=$(tr -d '[:space:]' < "$PID_FILE")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        old_args=$(ps -o args= -p "$old_pid" 2>/dev/null || true)
        if [[ "$old_args" == *"rg_depth_event_watch.sh"*"$JOBS"*"$EVENT"* ]]; then
            printf 'watcher already live with PID %s\n' "$old_pid"
            exit 0
        fi
    fi
    rm -f "$PID_FILE"
fi

exec env \
    RG_DEPTH_WATCH_LOG="$LOG" \
    RG_DEPTH_WATCH_PID_FILE="$PID_FILE" \
    INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}" \
    RETRY_LIMIT="${RETRY_LIMIT:-3}" \
    RG_DEPTH_WATCH_INITIAL_DELAY_SECONDS="${RG_DEPTH_WATCH_INITIAL_DELAY_SECONDS:-0}" \
    bash "$WATCHER" "$JOBS" "$EVENT"
