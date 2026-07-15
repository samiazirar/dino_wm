#!/bin/bash

build_dinocular_container_env() {
    if [ "$#" -ne 1 ]; then
        printf 'build_dinocular_container_env requires exactly one arm\n' >&2
        return 2
    fi
    local arm=$1
    : "${DINOCULAR_STUDENT_WEIGHTS:?DINOcular student weights are required}"
    DINOCULAR_CONTAINER_ENV=(
        --env DINOCULAR_STUDENT_WEIGHTS="$DINOCULAR_STUDENT_WEIGHTS"
    )
    case "$arm" in
        dinocular|dinocular_zerodepth)
            : "${DINOCULAR_NATIVE_DEPTH_CONTRACT:?DINOcular native contract is required}"
            : "${DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256:?DINOcular native contract hash is required}"
            : "${DINOCULAR_CACHE_PRODUCER_SHA256:?DINOcular producer hash is required}"
            DINOCULAR_CONTAINER_ENV+=(
                --env DINOCULAR_NATIVE_DEPTH_CONTRACT="$DINOCULAR_NATIVE_DEPTH_CONTRACT"
                --env DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256="$DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256"
                --env DINOCULAR_CACHE_PRODUCER_SHA256="$DINOCULAR_CACHE_PRODUCER_SHA256"
            )
            ;;
        dino_pinned) ;;
        *)
            printf 'unsupported encoder arm for container environment: %s\n' "$arm" >&2
            return 2
            ;;
    esac
}
