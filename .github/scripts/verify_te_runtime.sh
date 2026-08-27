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
# 3. Native extension presence (actual .so files)
# 4. TE-FL commit match with provenance validation
# 5. Platform/vendor implementation registration (explicit check)
# 6. Actual operator backend selection (execution verification)
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

if [[ -n "${MEGATRON_CI_IMAGE:-}" ]]; then
    msg "   Input image: ${MEGATRON_CI_IMAGE}"
else
    err "MEGATRON_CI_IMAGE not set"
fi

if [[ -f /etc/hostname ]]; then
    HOSTNAME=$(cat /etc/hostname)
    msg "   Hostname: ${HOSTNAME}"
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
# 3. Verify native extension (STRICT: must be actual .so file)
# ==============================================================================
msg "3. Native Extension (strict .so check)"

python3 - <<'PY' || { err "Native extension (.so) not found or invalid"; }
import sys
import os

native_found = False
native_path = None
native_module_name = None

# Try transformer_engine_torch first
try:
    import transformer_engine_torch
    native_path = transformer_engine_torch.__file__
    native_module_name = "transformer_engine_torch"
    native_found = True
except ImportError:
    pass

# Try transformer_engine.pytorch as fallback
if not native_found:
    try:
        import transformer_engine.pytorch as te_pytorch
        native_path = te_pytorch.__file__
        native_module_name = "transformer_engine.pytorch"
        # Verify it's not just a Python __init__.py
        if native_path and os.path.isfile(native_path):
            native_found = True
    except (ImportError, AttributeError):
        pass

if not native_found or not native_path:
    print("ERROR: No native extension module found", file=sys.stderr)
    sys.exit(1)

# STRICT: Verify the path points to a native library (.so, .pyd, .dylib)
native_extensions = ('.so', '.pyd', '.dylib')
if not any(native_path.endswith(ext) for ext in native_extensions):
    print(f"ERROR: Module path is not a native library: {native_path}", file=sys.stderr)
    print(f"  Expected extensions: {native_extensions}", file=sys.stderr)
    sys.exit(1)

if not os.path.isfile(native_path):
    print(f"ERROR: Native library file does not exist: {native_path}", file=sys.stderr)
    sys.exit(1)

print(f"   Module: {native_module_name}")
print(f"   Path: {native_path}")
print(f"   Verified: native library file exists")
PY

# ==============================================================================
# 4. TE-FL Commit Verification with Provenance
# ==============================================================================
msg "4. TE-FL Commit Verification"

if [[ -n "${EXPECTED_TE_FL_COMMIT:-}" ]]; then
    # Expected commit provided, must verify

    if [[ ! "${EXPECTED_TE_FL_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
        err "EXPECTED_TE_FL_COMMIT invalid format: ${EXPECTED_TE_FL_COMMIT} (expected 40 hex chars)"
    fi

    # Read commit from environment
    ACTUAL_COMMIT="${TE_FL_COMMIT:-}"

    if [[ -z "$ACTUAL_COMMIT" ]]; then
        err "TE_FL_COMMIT environment variable not set, cannot verify commit"
    elif [[ ! "$ACTUAL_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
        err "TE_FL_COMMIT invalid format: ${ACTUAL_COMMIT}"
    else
        msg "   Expected commit: ${EXPECTED_TE_FL_COMMIT}"
        msg "   Actual commit (env): ${ACTUAL_COMMIT}"

        if [[ "$ACTUAL_COMMIT" == "$EXPECTED_TE_FL_COMMIT" ]]; then
            msg "   Status: MATCH"
        else
            err "Commit mismatch: expected ${EXPECTED_TE_FL_COMMIT}, got ${ACTUAL_COMMIT}"
        fi
    fi

    # Additional provenance check: verify TE-FL package metadata if available
    python3 - <<'PY' || msg "   Warning: Could not read package metadata"
import sys
try:
    import importlib.metadata
    te_metadata = importlib.metadata.metadata('transformer-engine')
    version = te_metadata.get('Version', 'unknown')
    print(f"   Package version: {version}")
except Exception:
    # metadata not available, not fatal
    pass
PY

else
    msg "   No expected commit provided, skipping commit verification"
    msg "   Note: TE_FL_COMMIT=${TE_FL_COMMIT:-not_set}"
fi

# ==============================================================================
# 5. Platform/Vendor Implementation Registration (EXPLICIT CHECK)
# ==============================================================================
msg "5. Platform/Vendor Implementation Registration"

python3 - <<'PY' || { err "Platform implementation not properly registered"; }
import sys
import os

# Check TE_FL_PLATFORM environment variable
platform = os.getenv("TE_FL_PLATFORM", "")
if not platform:
    print("ERROR: TE_FL_PLATFORM environment variable not set", file=sys.stderr)
    sys.exit(1)

print(f"   TE_FL_PLATFORM: {platform}")

# Import TE and check for vendor backend registration
try:
    import transformer_engine as te

    # Check if TE has implementation registry/manager
    has_registry = False
    registry_info = []

    # Try to access TE-FL's implementation manager
    if hasattr(te, 'impl_manager') or hasattr(te, 'implementation_manager'):
        manager = getattr(te, 'impl_manager', None) or getattr(te, 'implementation_manager', None)
        if manager:
            has_registry = True
            registry_info.append(f"Implementation manager found: {type(manager).__name__}")

            # Try to get registered implementations
            if hasattr(manager, 'get_registered_implementations'):
                impls = manager.get_registered_implementations()
                registry_info.append(f"Registered implementations: {impls}")
            elif hasattr(manager, 'list_implementations'):
                impls = manager.list_implementations()
                registry_info.append(f"Registered implementations: {impls}")

    # Alternative: check for backend/device manager
    if hasattr(te, 'backend'):
        backend = te.backend
        registry_info.append(f"Backend module found: {backend}")
        has_registry = True

    # Check torch device availability as secondary indicator
    import torch
    device_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if device_available else 0

    print(f"   Has implementation registry: {has_registry}")
    if registry_info:
        for info in registry_info:
            print(f"   {info}")

    print(f"   Torch device available: {device_available}")
    print(f"   Device count: {device_count}")

    # For non-CUDA platforms, require explicit vendor implementation
    if platform.lower() != "cuda":
        if not has_registry and not device_available:
            print(f"ERROR: Platform '{platform}' requires vendor implementation, but no registry or devices found", file=sys.stderr)
            sys.exit(1)
        elif device_count == 0:
            print(f"ERROR: Platform '{platform}' has no available devices", file=sys.stderr)
            sys.exit(1)

except ImportError as e:
    print(f"ERROR: Failed to check implementation registration: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ==============================================================================
# 6. Operator Backend Selection (EXECUTION VERIFICATION)
# ==============================================================================
msg "6. Operator Backend Selection (execution verification)"

python3 - <<'PY' || { err "Operator backend selection failed or fallback to reference"; }
import sys
import os

try:
    from transformer_engine.pytorch import Linear
    print("   Linear operator imported successfully")

    # Instantiate the operator
    layer = Linear(in_features=4, out_features=8)
    print(f"   Linear layer instantiated: in_features=4, out_features=8")

    # Check the layer type and module
    layer_type = type(layer).__name__
    layer_module = type(layer).__module__
    print(f"   Layer type: {layer_type}")
    print(f"   Layer module: {layer_module}")

    # CRITICAL: Detect reference.torch fallback
    if 'reference' in layer_module.lower() or 'torch' in layer_module.lower():
        # Check if this is actually a vendor implementation or just torch fallback
        platform = os.getenv("TE_FL_PLATFORM", "").lower()

        if platform and platform != "cuda":
            # Non-CUDA platform should NOT use reference/torch backend
            print(f"ERROR: Platform '{platform}' is using reference/torch backend: {layer_module}", file=sys.stderr)
            print(f"  This indicates vendor implementation is not active", file=sys.stderr)
            sys.exit(1)

    # Try to execute a forward pass to confirm backend is functional
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        try:
            layer = layer.to(device)
            x = torch.randn(2, 4, device=device)
            y = layer(x)
            print(f"   Forward pass successful: input shape {tuple(x.shape)} -> output shape {tuple(y.shape)}")
            print("   Backend execution verified")
        except Exception as e:
            print(f"ERROR: Forward pass failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("   Warning: CUDA not available, skipping forward pass execution test")

except ImportError as e:
    print(f"ERROR: Failed to import Linear operator: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Operator instantiation or execution failed: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ==============================================================================
# 7. Platform-Specific Checks
# ==============================================================================
msg "7. Platform-Specific Checks"

PLATFORM="${TE_FL_PLATFORM:-unknown}"
msg "   Platform: ${PLATFORM}"

case "${PLATFORM,,}" in
    cuda)
        python3 - <<'PY' || msg "   Warning: CUDA check failed"
import torch
if torch.cuda.is_available():
    print(f"   CUDA available: True")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Device count: {torch.cuda.device_count()}")
else:
    print("   CUDA available: False")
PY
        ;;
    musa)
        python3 - <<'PY' || msg "   Warning: MUSA check failed"
import torch
if hasattr(torch, 'musa') and torch.musa.is_available():
    print(f"   MUSA available: True")
    print(f"   MUSA device count: {torch.musa.device_count()}")
else:
    print("   MUSA available: False (or torch.musa not found)")
PY
        ;;
    *)
        msg "   Generic platform, skipping specific device checks"
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
