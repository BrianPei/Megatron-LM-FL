#!/bin/bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

set -euo pipefail

# ==============================================================================
# Resolve Platform Image
# ==============================================================================
# This script determines which image to use for a platform:
# 1. If PR built candidate image, use it
# 2. Otherwise, use stable image from platform config
# ==============================================================================

PLATFORM="${1:-}"
CONFIG_FILE="${2:-.github/configs/${PLATFORM}.yml}"

if [[ -z "${PLATFORM}" ]]; then
    echo "ERROR: Platform not specified" >&2
    echo "Usage: $0 <platform> [config_file]" >&2
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# Check if we're in a PR with candidate images
CANDIDATE_IMAGE=""
if [[ -n "${GITHUB_EVENT_NAME:-}" ]] && [[ "${GITHUB_EVENT_NAME}" == "pull_request" ]]; then
    PR_NUMBER="${GITHUB_PR_NUMBER:-}"

    if [[ -n "${PR_NUMBER}" ]]; then
        # Check if candidate image exists for this PR
        CANDIDATE_TAG="candidate-pr${PR_NUMBER}-${PLATFORM}"
        CANDIDATE_IMAGE="harbor.baai.ac.cn/megatron-ci/${PLATFORM}:${CANDIDATE_TAG}"

        # Verify candidate image exists
        if docker manifest inspect "${CANDIDATE_IMAGE}" >/dev/null 2>&1; then
            echo "Using candidate image from PR#${PR_NUMBER}: ${CANDIDATE_IMAGE}"
            echo "${CANDIDATE_IMAGE}"
            exit 0
        fi
    fi
fi

# Fallback to stable image from config
STABLE_IMAGE=$(yq '.ci_image' "${CONFIG_FILE}")

if [[ -z "${STABLE_IMAGE}" ]] || [[ "${STABLE_IMAGE}" == "null" ]]; then
    echo "ERROR: ci_image not configured in ${CONFIG_FILE}" >&2
    exit 1
fi

echo "Using stable image: ${STABLE_IMAGE}"
echo "${STABLE_IMAGE}"
