#!/bin/bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

set -euo pipefail

# ==============================================================================
# TE-FL Installation Script
# ==============================================================================
# Installs TransformerEngine-FL from artifact or source.
# Should be called from Dockerfile with appropriate build args.
# ==============================================================================

err() {
    printf "ERROR: %s\n" "$1" >&2
    exit 1
}

msg() {
    printf ">>> %s\n" "$1"
}

# ==============================================================================
# Parse arguments
# ==============================================================================
TE_FL_ARTIFACT="${TE_FL_ARTIFACT:-}"
TE_FL_COMMIT="${TE_FL_COMMIT:-}"
TE_FL_REPOSITORY="${TE_FL_REPOSITORY:-https://github.com/FlagOpen/TransformerEngine-FL.git}"
PLATFORM="${PLATFORM:-unknown}"
INSTALL_MODE="${INSTALL_MODE:-artifact}"  # "artifact" or "source"

while [[ $# -gt 0 ]]; do
    case $1 in
        --artifact)
            TE_FL_ARTIFACT="$2"
            INSTALL_MODE="artifact"
            shift 2
            ;;
        --commit)
            TE_FL_COMMIT="$2"
            shift 2
            ;;
        --repository)
            TE_FL_REPOSITORY="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --source)
            INSTALL_MODE="source"
            shift
            ;;
        *)
            err "Unknown option: $1"
            ;;
    esac
done

# ==============================================================================
# Validate inputs
# ==============================================================================
[[ -n "${TE_FL_COMMIT}" ]] || err "TE_FL_COMMIT is required"
[[ "${TE_FL_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || err "TE_FL_COMMIT must be 40-char hex SHA"
[[ -n "${PLATFORM}" ]] || err "PLATFORM is required"

msg "TE-FL Installation Configuration:"
msg "  Mode: ${INSTALL_MODE}"
msg "  Platform: ${PLATFORM}"
msg "  Commit: ${TE_FL_COMMIT}"
msg "  Repository: ${TE_FL_REPOSITORY}"

# ==============================================================================
# Install from artifact (preferred)
# ==============================================================================
if [[ "${INSTALL_MODE}" == "artifact" ]]; then
    if [[ -z "${TE_FL_ARTIFACT}" ]]; then
        err "TE_FL_ARTIFACT is required for artifact mode"
    fi

    msg "Installing TE-FL from artifact: ${TE_FL_ARTIFACT}"

    # Check if artifact is URL or local path
    if [[ "${TE_FL_ARTIFACT}" =~ ^https?:// ]]; then
        # Download from URL
        TEMP_WHEEL="/tmp/te_fl_${PLATFORM}_${TE_FL_COMMIT:0:8}.whl"
        msg "Downloading: ${TE_FL_ARTIFACT} -> ${TEMP_WHEEL}"

        curl -fsSL -o "${TEMP_WHEEL}" "${TE_FL_ARTIFACT}" || \
            err "Failed to download TE-FL artifact"

        WHEEL_PATH="${TEMP_WHEEL}"
    elif [[ -f "${TE_FL_ARTIFACT}" ]]; then
        # Local file
        WHEEL_PATH="${TE_FL_ARTIFACT}"
        msg "Using local artifact: ${WHEEL_PATH}"
    else
        err "TE_FL_ARTIFACT not found: ${TE_FL_ARTIFACT}"
    fi

    # Install wheel
    msg "Installing wheel: ${WHEEL_PATH}"
    pip install --no-deps --no-cache-dir --force-reinstall "${WHEEL_PATH}" || \
        err "Failed to install TE-FL wheel"

    # Cleanup
    [[ "${WHEEL_PATH}" == /tmp/* ]] && rm -f "${WHEEL_PATH}"

    msg "TE-FL installed from artifact"

# ==============================================================================
# Install from source (fallback)
# ==============================================================================
elif [[ "${INSTALL_MODE}" == "source" ]]; then
    msg "Installing TE-FL from source (commit: ${TE_FL_COMMIT})"
    msg "WARNING: Source installation will compile native code"
    msg "         This is slow and should only be used for development"

    WORK_DIR="/tmp/te-fl-build-${TE_FL_COMMIT:0:8}"
    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}"

    msg "Cloning repository: ${TE_FL_REPOSITORY}"
    git clone "${TE_FL_REPOSITORY}" te-fl || err "Failed to clone TE-FL repository"
    cd te-fl
    git checkout "${TE_FL_COMMIT}" || err "Failed to checkout commit ${TE_FL_COMMIT}"

    msg "Building TE-FL (this may take 10-30 minutes)..."
    pip install --no-cache-dir --no-deps . || err "Failed to build TE-FL from source"

    cd /
    rm -rf "${WORK_DIR}"

    msg "TE-FL installed from source"
else
    err "Unknown INSTALL_MODE: ${INSTALL_MODE}"
fi

# ==============================================================================
# Set environment variables for provenance tracking
# ==============================================================================
msg "Setting TE-FL provenance environment variables"

cat >> /etc/environment <<EOF
TE_FL_COMMIT=${TE_FL_COMMIT}
TE_FL_PLATFORM=${PLATFORM}
TE_FL_REPOSITORY=${TE_FL_REPOSITORY}
EOF

# Also set in current shell for immediate verification
export TE_FL_COMMIT="${TE_FL_COMMIT}"
export TE_FL_PLATFORM="${PLATFORM}"
export TE_FL_REPOSITORY="${TE_FL_REPOSITORY}"

# ==============================================================================
# Verification
# ==============================================================================
msg "Verifying installation"

python3 - <<'PY' || err "TE-FL verification failed"
import transformer_engine
import os

te_file = transformer_engine.__file__
te_version = getattr(transformer_engine, "__version__", "unknown")
te_commit = os.getenv("TE_FL_COMMIT")
te_platform = os.getenv("TE_FL_PLATFORM")

print(f"OK: transformer_engine imported")
print(f"  Path: {te_file}")
print(f"  Version: {te_version}")
print(f"  Commit: {te_commit}")
print(f"  Platform: {te_platform}")

# Verify native module
try:
    import transformer_engine_torch
    print(f"OK: Native module: transformer_engine_torch")
except ImportError:
    try:
        import transformer_engine.pytorch
        print(f"OK: Native module: transformer_engine.pytorch")
    except ImportError:
        print("WARNING: No native module found")
        import sys
        sys.exit(1)
PY

msg "SUCCESS: TE-FL installation complete and verified"
