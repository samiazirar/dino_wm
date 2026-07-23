#!/bin/bash

build_dinocular_container_env() {
    if [ "$#" -ne 1 ]; then
        printf 'build_dinocular_container_env requires exactly one arm\n' >&2
        return 2
    fi
    local arm=$1
    local mode=${DINOCULAR_DEPTH_INPUT_MODE:-native_v1}
    DINOCULAR_CONTAINER_ENV=()
    case "$arm" in
        dinocular|dinocular_zerodepth)
            : "${DINOCULAR_STUDENT_WEIGHTS:?DINOcular student weights are required}"
            case "$mode" in
                native_v1)
                    if [ -n "${DINOCULAR_EMPIRICAL_DEPTH_CONTRACT:-}" ] || \
                       [ -n "${DINOCULAR_EMPIRICAL_DEPTH_CONTRACT_SHA256:-}" ] || \
                       [ -n "${DINOCULAR_EMPIRICAL_RUNTIME_RELEASE:-}" ] || \
                       [ -n "${DINOCULAR_EMPIRICAL_RUNTIME_RELEASE_SHA256:-}" ] || \
                       [ -n "${DINOCULAR_EMPIRICAL_ADAPTER_ID:-}" ] || \
                       [ -n "${DINOCULAR_EMPIRICAL_ADAPTER_MODE:-}" ] || \
                       [ -n "${DINOCULAR_EMPIRICAL_ZERO_INTERVENTION:-}" ]; then
                        printf 'native depth mode must not include empirical depth inputs\n' >&2
                        return 2
                    fi
                    : "${DINOCULAR_NATIVE_DEPTH_CONTRACT:?DINOcular native contract is required}"
                    : "${DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256:?DINOcular native contract hash is required}"
                    : "${DINOCULAR_CACHE_PRODUCER_SHA256:?DINOcular producer hash is required}"
                    DINOCULAR_CONTAINER_ENV+=(
                        --env DINOCULAR_STUDENT_WEIGHTS="$DINOCULAR_STUDENT_WEIGHTS"
                        --env DINOCULAR_NATIVE_DEPTH_CONTRACT="$DINOCULAR_NATIVE_DEPTH_CONTRACT"
                        --env DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256="$DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256"
                        --env DINOCULAR_CACHE_PRODUCER_SHA256="$DINOCULAR_CACHE_PRODUCER_SHA256"
                    )
                    ;;
                empirical_lossy_cache_v1)
                    if [ -n "${DINOCULAR_NATIVE_DEPTH_CONTRACT:-}" ] || \
                       [ -n "${DINOCULAR_NATIVE_DEPTH_CONTRACT_SHA256:-}" ] || \
                       [ -n "${DINOCULAR_CACHE_PRODUCER_SHA256:-}" ]; then
                        printf 'empirical depth mode must not include native depth inputs\n' >&2
                        return 2
                    fi
                    : "${DINOCULAR_EMPIRICAL_DEPTH_CONTRACT:?DINOcular empirical contract is required}"
                    : "${DINOCULAR_EMPIRICAL_DEPTH_CONTRACT_SHA256:?DINOcular empirical contract hash is required}"
                    : "${DINOCULAR_EMPIRICAL_RUNTIME_RELEASE:?DINOcular empirical runtime release is required}"
                    : "${DINOCULAR_EMPIRICAL_RUNTIME_RELEASE_SHA256:?DINOcular empirical runtime release hash is required}"
                    : "${DINOCULAR_EMPIRICAL_ADAPTER_ID:?DINOcular empirical adapter ID is required}"
                    : "${DINOCULAR_EMPIRICAL_ADAPTER_MODE:?DINOcular empirical adapter mode is required}"
                    : "${DINOCULAR_EMPIRICAL_ZERO_INTERVENTION:?DINOcular empirical zero intervention is required}"
                    if [ "$DINOCULAR_EMPIRICAL_ADAPTER_ID" != "mapanything_pusht_empirical_lossy_v1" ]; then
                        printf 'empirical adapter ID differs from the exact contract\n' >&2
                        return 2
                    fi
                    if { [ "$arm" = "dinocular" ] && \
                         { [ "$DINOCULAR_EMPIRICAL_ADAPTER_MODE" != "proxy_depth_z" ] || \
                           [ "$DINOCULAR_EMPIRICAL_ZERO_INTERVENTION" != "false" ]; }; } || \
                       { [ "$arm" = "dinocular_zerodepth" ] && \
                         { [ "$DINOCULAR_EMPIRICAL_ADAPTER_MODE" != "exact_constant_zero_numeric" ] || \
                           [ "$DINOCULAR_EMPIRICAL_ZERO_INTERVENTION" != "true" ]; }; }; then
                        printf 'empirical adapter mode differs from arm\n' >&2
                        return 2
                    fi
                    DINOCULAR_CONTAINER_ENV+=(
                        --env DINOCULAR_STUDENT_WEIGHTS="$DINOCULAR_STUDENT_WEIGHTS"
                        --env DINOCULAR_DEPTH_INPUT_MODE="$DINOCULAR_DEPTH_INPUT_MODE"
                        --env DINOCULAR_EMPIRICAL_DEPTH_CONTRACT="$DINOCULAR_EMPIRICAL_DEPTH_CONTRACT"
                        --env DINOCULAR_EMPIRICAL_DEPTH_CONTRACT_SHA256="$DINOCULAR_EMPIRICAL_DEPTH_CONTRACT_SHA256"
                        --env DINOCULAR_EMPIRICAL_RUNTIME_RELEASE="$DINOCULAR_EMPIRICAL_RUNTIME_RELEASE"
                        --env DINOCULAR_EMPIRICAL_RUNTIME_RELEASE_SHA256="$DINOCULAR_EMPIRICAL_RUNTIME_RELEASE_SHA256"
                        --env DINOCULAR_EMPIRICAL_ADAPTER_ID="$DINOCULAR_EMPIRICAL_ADAPTER_ID"
                        --env DINOCULAR_EMPIRICAL_ADAPTER_MODE="$DINOCULAR_EMPIRICAL_ADAPTER_MODE"
                        --env DINOCULAR_EMPIRICAL_ZERO_INTERVENTION="$DINOCULAR_EMPIRICAL_ZERO_INTERVENTION"
                    )
                    ;;
                *)
                    printf 'unsupported DINOcular depth input mode: %s\n' "$mode" >&2
                    return 2
                    ;;
            esac
            ;;
        dino_pinned) ;;
        *)
            printf 'unsupported encoder arm for container environment: %s\n' "$arm" >&2
            return 2
            ;;
    esac
}
