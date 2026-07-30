#!/bin/bash
set -euo pipefail

JOBS=${1:?comma-separated jobs required}
EVENT=${2:?event path required}
TARGET=${3:?Herdr target required}

while true; do
    snapshot=$(ssh marvin "sacct -X -n -P -j '$JOBS' -o JobIDRaw,State,ExitCode,ElapsedRaw")
    terminal=1
    failed=0
    while IFS='|' read -r job state exit_code elapsed; do
        case "$job" in
            267*) ;;
            *) continue ;;
        esac
        state=${state%%+*}
        case "$state" in
            COMPLETED) ;;
            FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
                failed=1
                ;;
            *) terminal=0 ;;
        esac
    done <<< "$snapshot"
    if [ "$terminal" = 1 ]; then
        {
            printf 'jobs=%s\n' "$JOBS"
            printf 'failed=%s\n' "$failed"
            printf '%s\n' "$snapshot"
        } > "$EVENT"
        herdr pane run "$TARGET" \
            "MATERIAL ROPE/GRANULAR DEPTH EVENT. Read $EVENT once and inspect only the named repair outputs/logs. Continue the same task with concrete correction or the next dependent stage; do not model-poll."
        exit 0
    fi
    sleep 60
done
