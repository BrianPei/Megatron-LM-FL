#!/bin/bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

set -euo pipefail

# ==============================================================================
# TE-FL Runtime Verification
# ==============================================================================
# This script runs INSIDE the test container after platform setup to verify:
# 1. The resolved container image
# 2. transformer_engine installation and path
# 3. Native extension presence
# 4. TE-FL commit match (if expected)
# 5. Platform/vendor implementation registration
# 6. Actual operator backend selection
# ==============================================================================

EXIT_CODE=0

msg() {
    printf ">>> %s\n" "$1"
}

err() {
    printf "ERROR: %s\n" "$1" >&2
    EXIT_CODE=1
}

msg "TE-FL Runtime Verification"
msg "==========================================="

# ==============================================================================
# 1. Report resolved container image
# ==============================================================================
msg "1. Resolved Container Image"

# Try to read from various sources
if [[ -n "${MEGATRON_CI_IMAGE:-}" ]]; then
    msg "   Input image: ${MEGATRON_CI_IMAGE}"
fi

if [[ -f /etc/hostname ]]; then
    HOSTNAME=$(cat /etc/hostname)
    msg "   Hostname: ${HOSTNAME}"
fi

# Check for OCI labels if available
if command -v skopeo &>/dev/null && [[ -n "${MEGATRON_CI_IMAGE:-}" ]]; then
    IMAGE_DIGEST=$(skopeo inspect --no-creds "docker://${MEGATRON_CI_IMAGE}" 2>/dev/null | jq -r '.Digest // "unknown"' || echo "unknown")
    msg "   Digest: ${IMAGE_DIGEST}"
fi

# ==============================================================================
# 2. Verify transformer_engine installation
# ==============================================================================
msg "2. transformer_engine Installation"

python3 - <<'PY' || { err "transformer_engine not found or failed to import"; }
import sys
try:
    import transformer_engine as te
    print(f"   Path: {te.__file__}")
    version = getattr(te, "__version__", "unknown")
    print(f"   Version: {version}")
except ImportError as e:
    print(f"ERROR: Failed to import transformer_engine: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ==============================================================================
# 3. Verify native extension
# ==============================================================================
msg "3. Native Extension"

python3 - <<'PY' || { err "Native extension not found"; }
import sys

native_found = False
native_path = None

# Try common native module names
try:
    import transformer_engine_torch
    native_found = True
    native_path = transformer_engine_torch.__file__
    print(f"   Module: transformer_engine_torch")
    print(f"   Path: {native_path}")
except ImportError:
    pass

if not native_found:
    try:
        import transformer_engine.pytorch
        native_found = True
        native_path = transformer_engine.pytorch.__file__
        print(f"   Module: transformer_engine.pytorch")
        print(f"   Path: {native_path}")
    except ImportError:
        pass

if not native_found:
    print("ERROR: No native extension found", file=sys.stderr)
    print("   Tried: transformer_engine_torch, transformer_engine.pytorch", file=sys.stderr)
    sys.exit(1)
PY

# ==============================================================================
# 4. Check TE-FL commit (if expected value provided)
# ==============================================================================
msg "4. TE-FL Commit Verification"

EXPECTED_COMMIT="${EXPECTED_TE_FL_COMMIT:-}"

if [[ -n "${EXPECTED_COMMIT}" ]]; then
    msg "   Expected commit: ${EXPECTED_COMMIT}"

    # Check environment variable
    ACTUAL_COMMIT="${TE_FL_COMMIT:-}"

    if [[ -z "${ACTUAL_COMMIT}" ]]; then
        err "TE_FL_COMMIT environment variable not set"
    elif [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
        err "TE-FL commit mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}"
    else
        msg "   Actual commit: ${ACTUAL_COMMIT} (MATCH)"
    fi
else
    msg "   No expected commit provided (skipping verification)"
    if [[ -n "${TE_FL_COMMIT:-}" ]]; then
        msg "   Found TE_FL_COMMIT: ${TE_FL_COMMIT}"
    fi
fi

# ==============================================================================
# 5. Check platform/vendor implementation registration
# ==============================================================================
msg "5. Platform Implementation Registration"

python3 - <<'PY' || { err "Failed to check platform registration"; }
import sys
import os

platform = os.getenv("TE_FL_PLATFORM", "unknown")
print(f"   TE_FL_PLATFORM: {platform}")

# Try to introspect available backends
try:
    import transformer_engine.pytorch as te

    # Check for common platform-specific attributes or modules
    has_cuda = hasattr(te, 'fp8_autocast') or hasattr(te, 'DotProductAttention')
    print(f"   Has TE operators: {has_cuda}")

    # Try to detect vendor backend
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
            print(f"   Torch CUDA available: True")
            print(f"   Device count: {device_count}")
            print(f"   Device name: {device_name}")
        else:
            print(f"   Torch CUDA available: False")
    except Exception as e:
        print(f"   Warning: Could not detect torch device: {e}")

except Exception as e:
    print(f"WARNING: Could not fully introspect platform: {e}", file=sys.stderr)
PY

# ==============================================================================
# 6. Verify operator backend selection
# ==============================================================================
msg "6. Operator Backend Selection"

python3 - <<'PY' || { err "Failed to verify operator backend"; }
import sys
import os

try:
    from transformer_engine.pytorch import Linear

    print(f"   Linear operator imported successfully")

    # Try to instantiate (doesn't allocate device memory in constructor)
    try:
        layer = Linear(4, 8)
        print(f"   Linear layer instantiated: in_features=4, out_features=8")

        # Check if this is using vendor backend or falling back
        layer_type = type(layer).__name__
        layer_module = type(layer).__module__
        print(f"   Layer type: {layer_module}.{layer_type}")

        # Detect if using reference implementation
        if "reference" in layer_module.lower() or "torch" in layer_module.lower():
            print(f"   WARNING: May be using reference/torch fallback")

    except Exception as e:
        print(f"   Warning: Could not instantiate layer: {e}")

except ImportError as e:
    print(f"ERROR: Failed to import Linear operator: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ==============================================================================
# 7. Platform-specific checks
# ==============================================================================
msg "7. Platform-Specific Checks"

PLATFORM="${TE_FL_PLATFORM:-unknown}"
msg "   Platform: ${PLATFORM}"

case "${PLATFORM}" in
    cuda)
        python3 -c "import torch; print(f'   CUDA version: {torch.version.cuda}')" 2>/dev/null || true
        ;;
    musa)
        python3 -c "import torch_musa; print(f'   MUSA available: {torch_musa.is_available()}')" 2>/dev/null || msg "   (torch_musa not available)"
        ;;
    ascend)
        python3 -c "import torch_npu; print(f'   NPU available: {torch_npu.npu.is_available()}')" 2>/dev/null || msg "   (torch_npu not available)"
        ;;
    metax)
        msg "   MetaX platform (checks TBD)"
        ;;
    *)
        msg "   Platform: ${PLATFORM} (no specific checks)"
        ;;
esac

# ==============================================================================
# Summary
# ==============================================================================
msg "==========================================="

if [[ ${EXIT_CODE} -eq 0 ]]; then
    msg "SUCCESS: TE-FL runtime verification passed"
else
    msg "FAILED: TE-FL runtime verification failed"
fi

exit ${EXIT_CODE}
