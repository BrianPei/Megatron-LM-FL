#!/bin/bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

set -euo pipefail

# ==============================================================================
# Megatron-LM-FL Docker Image Build Script
# ==============================================================================
# Builds platform-specific CI images with fixed TE-FL versions.
# Usage: PLATFORM=musa TARGET=ci ./build.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Default configuration ----
PLATFORM="${PLATFORM:-cuda}"
TARGET="${TARGET:-ci}"
VERSIONS_FILE="${VERSIONS_FILE:-${SCRIPT_DIR}/versions.yaml}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-harbor.baai.ac.cn}"
IMAGE_REPO="${IMAGE_REPO:-megatron-ci}"
NO_CACHE="${NO_CACHE:-}"
EXTRA_BUILD_ARGS=()

# ---- Helper functions ----
err() {
    printf "ERROR: %s\n" "$1" >&2
    exit 1
}

msg() {
    printf ">>> %s\n" "$1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || err "$1 is required but not found in PATH"
}

# Parse versions.yaml using Python (yq alternative)
parse_yaml() {
    local key="$1"
    python3 - <<EOF
import sys, yaml
try:
    with open("${VERSIONS_FILE}") as f:
        cfg = yaml.safe_load(f)
    keys = "${key}".split(".")
    val = cfg
    for k in keys:
        val = val[k]
    print(val if val is not None else "")
except Exception as e:
    print(f"ERROR: Failed to parse {key}: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

validate_versions_file() {
    msg "Validating versions.yaml"

    require_command python3
    python3 -c "import yaml" 2>/dev/null || err "Python yaml module required: pip install pyyaml"

    [[ -f "${VERSIONS_FILE}" ]] || err "versions.yaml not found: ${VERSIONS_FILE}"

    # Validate TE-FL commit
    TE_FL_COMMIT=$(parse_yaml "te_fl.commit")
    if [[ -z "${TE_FL_COMMIT}" ]]; then
        err "TE-FL commit is empty in versions.yaml. Run update_te_lock.py first."
    fi

    if ! [[ "${TE_FL_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
        err "TE-FL commit must be 40-char hex SHA: got '${TE_FL_COMMIT}'"
    fi

    # Validate platform exists
    PLATFORM_EXISTS=$(parse_yaml "platforms.${PLATFORM}" 2>/dev/null || echo "")
    if [[ -z "${PLATFORM_EXISTS}" ]]; then
        err "Platform '${PLATFORM}' not found in versions.yaml"
    fi

    msg "Versions file validated"
}

load_platform_config() {
    msg "Loading configuration for platform: ${PLATFORM}"

    BASE_IMAGE=$(parse_yaml "platforms.${PLATFORM}.base_image")
    TORCH_VERSION=$(parse_yaml "platforms.${PLATFORM}.torch_version")
    PYTHON_VERSION=$(parse_yaml "platforms.${PLATFORM}.python_version")
    VENDOR_SDK=$(parse_yaml "platforms.${PLATFORM}.vendor_sdk")
    CUDA_ARCH=$(parse_yaml "platforms.${PLATFORM}.cuda_arch")
    COMPILER=$(parse_yaml "platforms.${PLATFORM}.compiler")
    TE_FL_ARTIFACT=$(parse_yaml "platforms.${PLATFORM}.te_fl_artifact")

    # Validate required fields
    [[ -n "${BASE_IMAGE}" ]] || err "base_image not configured for ${PLATFORM}"
    [[ -n "${TORCH_VERSION}" ]] || err "torch_version not configured for ${PLATFORM}"
    [[ -n "${PYTHON_VERSION}" ]] || err "python_version not configured for ${PLATFORM}"

    # TE_FL_ARTIFACT is optional for initial setup, but warn if empty
    if [[ -z "${TE_FL_ARTIFACT}" ]]; then
        msg "WARNING: te_fl_artifact not configured for ${PLATFORM}"
        msg "         Build will fail unless Dockerfile has fallback logic"
    fi

    msg "Configuration loaded:"
    msg "  Base image:     ${BASE_IMAGE}"
    msg "  Torch:          ${TORCH_VERSION}"
    msg "  Python:         ${PYTHON_VERSION}"
    msg "  Vendor SDK:     ${VENDOR_SDK}"
    msg "  TE-FL artifact: ${TE_FL_ARTIFACT:-<not set>}"
}

generate_build_metadata() {
    local output_file="$1"

    cat > "${output_file}" <<EOF
{
  "platform": "${PLATFORM}",
  "target": "${TARGET}",
  "te_fl_commit": "${TE_FL_COMMIT}",
  "base_image": "${BASE_IMAGE}",
  "torch_version": "${TORCH_VERSION}",
  "python_version": "${PYTHON_VERSION}",
  "vendor_sdk": "${VENDOR_SDK}",
  "cuda_arch": "${CUDA_ARCH}",
  "compiler": "${COMPILER}",
  "te_fl_artifact": "${TE_FL_ARTIFACT}",
  "build_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_commit": "$(cd "${PROJECT_ROOT}" && git rev-parse HEAD 2>/dev/null || echo "unknown")",
  "git_branch": "$(cd "${PROJECT_ROOT}" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"
}
EOF
}

build_image() {
    local dockerfile="${SCRIPT_DIR}/${PLATFORM}/Dockerfile"
    [[ -f "${dockerfile}" ]] || err "Dockerfile not found: ${dockerfile}"

    # Generate image tag
    local short_commit="${TE_FL_COMMIT:0:8}"
    local timestamp=$(date +%Y%m%d-%H%M%S)
    IMAGE_TAG="${PLATFORM}-te${short_commit}-${TARGET}-${timestamp}"
    IMAGE_FULL="${IMAGE_REGISTRY}/${IMAGE_REPO}/${PLATFORM}:${IMAGE_TAG}"

    msg "Building image: ${IMAGE_FULL}"
    msg "Dockerfile: ${dockerfile}"

    # Prepare build context metadata
    local build_metadata="${SCRIPT_DIR}/.build-metadata-${PLATFORM}.json"
    generate_build_metadata "${build_metadata}"

    # Build arguments
    local build_args=(
        --file "${dockerfile}"
        --target "${TARGET}"
        --build-arg "BASE_IMAGE=${BASE_IMAGE}"
        --build-arg "TE_FL_COMMIT=${TE_FL_COMMIT}"
        --build-arg "TE_FL_ARTIFACT=${TE_FL_ARTIFACT}"
        --build-arg "TORCH_VERSION=${TORCH_VERSION}"
        --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
        --build-arg "VENDOR_SDK=${VENDOR_SDK}"
        --build-arg "CUDA_ARCH=${CUDA_ARCH}"
        --build-arg "COMPILER=${COMPILER}"
        --label "com.flagos.megatron.platform=${PLATFORM}"
        --label "com.flagos.megatron.te_fl_commit=${TE_FL_COMMIT}"
        --label "com.flagos.megatron.build_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        --tag "${IMAGE_FULL}"
    )

    # Add OCI annotations
    build_args+=(
        --label "org.opencontainers.image.source=https://github.com/FlagOpen/Megatron-LM-FL"
        --label "org.opencontainers.image.vendor=FlagOS"
        --label "org.opencontainers.image.title=Megatron-LM-FL CI (${PLATFORM})"
    )

    [[ -n "${NO_CACHE}" ]] && build_args+=(--no-cache)

    # Add extra build args
    for arg in "${EXTRA_BUILD_ARGS[@]}"; do
        build_args+=("${arg}")
    done

    # Build
    docker build "${build_args[@]}" "${SCRIPT_DIR}"

    msg "Build complete: ${IMAGE_FULL}"

    # Output manifest
    local manifest="${SCRIPT_DIR}/.manifest-${PLATFORM}-${TARGET}.json"
    cat > "${manifest}" <<EOF
{
  "image": "${IMAGE_FULL}",
  "platform": "${PLATFORM}",
  "target": "${TARGET}",
  "te_fl_commit": "${TE_FL_COMMIT}",
  "base_image": "${BASE_IMAGE}",
  "torch_version": "${TORCH_VERSION}",
  "python_version": "${PYTHON_VERSION}",
  "vendor_sdk": "${VENDOR_SDK}",
  "build_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_commit": "$(cd "${PROJECT_ROOT}" && git rev-parse HEAD 2>/dev/null || echo "unknown")"
}
EOF

    msg "Manifest written: ${manifest}"
    echo "${IMAGE_FULL}"
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build Megatron-LM-FL Docker images with fixed TE-FL versions.

OPTIONS:
    --platform PLATFORM    Platform: cuda, musa, metax, ascend, hygon, kunlunxin, enflame
                           (default: ${PLATFORM})
    --target TARGET        Build target: ci, dev (default: ${TARGET})
    --no-cache             Build without cache
    --build-arg ARG=VAL    Pass additional build argument to docker
    --help                 Show this help

ENVIRONMENT VARIABLES:
    PLATFORM              Same as --platform
    TARGET                Same as --target
    VERSIONS_FILE         Path to versions.yaml (default: docker/versions.yaml)
    IMAGE_REGISTRY        Docker registry (default: harbor.baai.ac.cn)
    IMAGE_REPO            Image repository (default: megatron-ci)

EXAMPLES:
    # Build MUSA CI image
    PLATFORM=musa TARGET=ci ./build.sh

    # Build MetaX dev image without cache
    ./build.sh --platform metax --target dev --no-cache

    # Build with custom base image
    ./build.sh --platform cuda --build-arg BASE_IMAGE=custom:tag
EOF
}

# ==============================================================================
# Main
# ==============================================================================

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="1"
            shift
            ;;
        --build-arg)
            EXTRA_BUILD_ARGS+=("--build-arg" "$2")
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            err "Unknown option: $1. Use --help for usage."
            ;;
    esac
done

# Validate environment
require_command docker
require_command python3

# Validate and load configuration
validate_versions_file
TE_FL_COMMIT=$(parse_yaml "te_fl.commit")
load_platform_config

# Build image
build_image

msg "Done!"
