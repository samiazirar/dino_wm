#!/bin/bash

set -uo pipefail

JOB_ID=${1:?lmdort job ID is required}
TARGET=${2:?Operations pane ID is required}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-60}
EVENT=/tmp/dinocular_lmdort_rope_dinocular_seed1_watch.event
LOG=/tmp/dinocular_lmdort_rope_dinocular_seed1_watch.log
ROOT=/cephfs/users/azirar/projects/dinocular-wm-lmdort-rope-dino-s1
PREVIOUS=

snapshot() {
    ssh -o BatchMode=yes -o ConnectTimeout=15 lmdort \
        "sacct -X -n -P -j '$JOB_ID' -o JobIDRaw,JobName,State,ExitCode,Start,End,NodeList,Reason" \
        2>&1
}

while :; do
    CURRENT=$(snapshot)
    STATUS=$?
    if [ "$STATUS" -ne 0 ]; then
        printf '%s ACCESS_ERROR status=%s\n%s\n' \
            "$(date --iso-8601=seconds)" "$STATUS" "$CURRENT" >> "$LOG"
        sleep "$INTERVAL_SECONDS"
        continue
    fi

    STATE=$(printf '%s\n' "$CURRENT" | awk -F'|' -v job="$JOB_ID" '
        $1 == job {sub(/[+].*$/, "", $3); print $3; exit}
    ')
    if [ -z "$PREVIOUS" ]; then
        PREVIOUS=$CURRENT
        printf '%s BASELINE state=%s\n%s\n' \
            "$(date --iso-8601=seconds)" "$STATE" "$CURRENT" >> "$LOG"
    elif [ "$CURRENT" != "$PREVIOUS" ]; then
        {
            printf 'OBSERVED_AT=%s\n' "$(date --iso-8601=seconds)"
            printf 'JOB_ID=%s\n' "$JOB_ID"
            printf '%s\n' "$CURRENT"
        } > "$EVENT"
        case "$STATE" in
            COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)
                DETAIL=$(ssh -o BatchMode=yes -o ConnectTimeout=15 lmdort \
                    "srun --partition=CPU --ntasks=1 --cpus-per-task=2 --mem=4G --time=00:05:00 bash -c 'set -u; \
                    printf \"PROGRESS\\\\n\"; sed -n \"1,260p\" \"$ROOT/outputs/campaign-seed1/native-seed1-20260728c/rope/dinocular/run/progress.json\" 2>/dev/null || true; \
                    printf \"STDOUT_TAIL\\\\n\"; tail -n 120 \"$ROOT/lmdort/logs/dinocular.segment.$JOB_ID.out\" 2>/dev/null || true; \
                    printf \"STDERR_TAIL\\\\n\"; tail -n 120 \"$ROOT/lmdort/logs/dinocular.segment.$JOB_ID.err\" 2>/dev/null || true; \
                    printf \"OUTPUT_FILES\\\\n\"; find \"$ROOT/outputs/campaign-seed1/native-seed1-20260728c/rope/dinocular/run\" -maxdepth 4 -type f -printf \"%s %p\\\\n\" | sort'" \
                    2>&1)
                {
                    printf '\nTERMINAL_DETAIL\n%s\n' "$DETAIL"
                } >> "$EVENT"
                ;;
        esac
        printf '%s MATERIAL_STATE_CHANGE state=%s\n%s\n' \
            "$(date --iso-8601=seconds)" "$STATE" "$CURRENT" >> "$LOG"
        herdr pane run "$TARGET" \
            "MATERIAL LMDORT ROPE DINOcular SEED-ONE EVENT. Read $EVENT once, verify job $JOB_ID and the lmdort receipt/log paths it names, integrate only the changed technical state, and do not model-poll. Do not message Hannah from this worker route." \
            >> "$LOG" 2>&1 || true
        PREVIOUS=$CURRENT
    fi

    case "$STATE" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)
            printf '%s STOPPED state=%s\n' \
                "$(date --iso-8601=seconds)" "$STATE" >> "$LOG"
            exit 0
            ;;
    esac

    sleep "$INTERVAL_SECONDS"
done
