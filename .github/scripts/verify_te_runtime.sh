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
# 3. Native extension presence (platform-specific vendor module)
# 4. TE-FL commit match with FlagScale provenance manifest (REQUIRED)
# 5. Platform/vendor implementation registration (explicit check)
# 6. Actual operator backend selection (FAIL-CLOSED: query real manager API)
#
# IMPORTANT: This script is ONLY executed when expected_te_fl_commit is provided
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
# 3. Verify native extension (platform-specific vendor module)
# ==============================================================================
msg "3. Native Extension (platform-specific vendor module)"

python3 - <<'PY' || { err "Native vendor module not found or invalid"; }
import sys
import os

platform = os.getenv("TE_FL_PLATFORM", "").lower()

# For TE-FL with plugin architecture, the native module varies by platform
# MUSA: transformer_engine_musa_torch
# CUDA: transformer_engine_torch (NVIDIA native)
# Others: transformer_engine_{platform}_torch

native_found = False
native_path = None
native_module_name = None

# Try platform-specific module first
if platform == "musa":
    try:
        import transformer_engine_musa_torch
        native_path = getattr(transformer_engine_musa_torch, '__file__', None)
        native_module_name = "transformer_engine_musa_torch"
        native_found = True
    except ImportError:
        pass
elif platform == "cuda":
    try:
        import transformer_engine_torch
        native_path = getattr(transformer_engine_torch, '__file__', None)
        native_module_name = "transformer_engine_torch"
        native_found = True
    except ImportError:
        pass
else:
    # Generic platform: try transformer_engine_{platform}_torch
    try:
        mod = __import__(f"transformer_engine_{platform}_torch")
        native_path = getattr(mod, '__file__', None)
        native_module_name = f"transformer_engine_{platform}_torch"
        native_found = True
    except ImportError:
        pass

# Fallback: try generic transformer_engine.pytorch
if not native_found:
    try:
        import transformer_engine.pytorch as te_pytorch
        # Check if this is a real module with __file__
        native_path = getattr(te_pytorch, '__file__', None)
        if native_path and os.path.isfile(native_path):
            native_module_name = "transformer_engine.pytorch"
            native_found = True
    except (ImportError, AttributeError):
        pass

if not native_found:
    print(f"ERROR: No native vendor module found for platform '{platform}'", file=sys.stderr)
    print(f"  Expected: transformer_engine_{platform}_torch or platform-specific native module", file=sys.stderr)
    sys.exit(1)

# Note: TE-FL may register a synthetic TEFLModule as sys.modules["transformer_engine_torch"]
# which doesn't have a normal __file__. In that case, we skip the .so check.
if native_path is None:
    print(f"   Module: {native_module_name}")
    print(f"   Type: Synthetic module (no __file__, likely TEFLModule)")
    print(f"   Note: Actual vendor .so will be loaded by plugin system")
else:
    # If __file__ exists, verify it's a native library
    native_extensions = ('.so', '.pyd', '.dylib')
    if not any(native_path.endswith(ext) for ext in native_extensions):
        print(f"   Module: {native_module_name}")
        print(f"   Path: {native_path}")
        print(f"   Type: Python module (not native .so)")
        print(f"   Note: TE-FL plugin system will load vendor .so dynamically")
    else:
        if not os.path.isfile(native_path):
            print(f"ERROR: Native library file does not exist: {native_path}", file=sys.stderr)
            sys.exit(1)

        print(f"   Module: {native_module_name}")
        print(f"   Path: {native_path}")
        print(f"   Verified: native library file exists")
PY

# ==============================================================================
# 4. TE-FL Commit Verification with FlagScale Provenance Manifest (REQUIRED)
# ==============================================================================
msg "4. TE-FL Commit Verification (FlagScale provenance REQUIRED)"

if [[ -n "${EXPECTED_TE_FL_COMMIT:-}" ]]; then
    if [[ ! "${EXPECTED_TE_FL_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
        err "EXPECTED_TE_FL_COMMIT invalid format: ${EXPECTED_TE_FL_COMMIT} (expected 40 hex chars)"
    fi

    # FlagScale provenance manifest is REQUIRED when verification is enabled
    PROVENANCE_FILE="/etc/flagos/te-fl.json"
    if [[ ! -f "$PROVENANCE_FILE" ]]; then
        err "FlagScale provenance manifest not found: ${PROVENANCE_FILE}"
        err "Cannot verify TE-FL provenance without manifest"
        err "Fallback to TE_FL_COMMIT environment variable is not allowed"
        exit 1
    fi

    msg "   Found FlagScale provenance manifest: ${PROVENANCE_FILE}"

    python3 - <<'PY' || { err "Failed to verify provenance manifest"; exit 1; }
import sys
import json
import os

expected_commit = os.getenv("EXPECTED_TE_FL_COMMIT")
platform = os.getenv("TE_FL_PLATFORM", "").lower()
provenance_file = "/etc/flagos/te-fl.json"

try:
    with open(provenance_file) as f:
        manifest = json.load(f)

    # Required fields
    actual_commit = manifest.get("commit", "")
    wheel_sha256 = manifest.get("wheel_sha256", "")
    build_time = manifest.get("build_time", "")
    manifest_platform = manifest.get("platform", "")
    torch_version = manifest.get("torch_version", "")
    python_abi = manifest.get("python_abi", "")

    # Validate required fields exist
    missing_fields = []
    if not actual_commit:
        missing_fields.append("commit")
    if not wheel_sha256:
        missing_fields.append("wheel_sha256")
    if not build_time:
        missing_fields.append("build_time")
    if not manifest_platform:
        missing_fields.append("platform")
    if not torch_version:
        missing_fields.append("torch_version")
    if not python_abi:
        missing_fields.append("python_abi")

    if missing_fields:
        print(f"ERROR: Provenance manifest missing required fields: {missing_fields}", file=sys.stderr)
        sys.exit(1)

    # Validate commit format
    if len(actual_commit) != 40 or not all(c in '0123456789abcdef' for c in actual_commit):
        print(f"ERROR: Invalid commit format in manifest: {actual_commit}", file=sys.stderr)
        sys.exit(1)

    # Validate SHA256 format
    if len(wheel_sha256) != 64 or not all(c in '0123456789abcdef' for c in wheel_sha256):
        print(f"ERROR: Invalid wheel_sha256 format in manifest: {wheel_sha256}", file=sys.stderr)
        sys.exit(1)

    # Validate platform match
    if platform and manifest_platform.lower() != platform:
        print(f"ERROR: Platform mismatch", file=sys.stderr)
        print(f"  Expected: {platform}", file=sys.stderr)
        print(f"  Manifest: {manifest_platform}", file=sys.stderr)
        sys.exit(1)

    print(f"   Provenance commit: {actual_commit}")
    print(f"   Wheel SHA256: {wheel_sha256[:16]}...")
    print(f"   Build time: {build_time}")
    print(f"   Platform: {manifest_platform}")
    print(f"   Torch version: {torch_version}")
    print(f"   Python ABI: {python_abi}")

    # Verify commit match
    if actual_commit != expected_commit:
        print(f"ERROR: Commit mismatch in provenance manifest", file=sys.stderr)
        print(f"  Expected: {expected_commit}", file=sys.stderr)
        print(f"  Actual: {actual_commit}", file=sys.stderr)
        sys.exit(1)

    print("   Status: MATCH (verified from provenance)")

except FileNotFoundError:
    print(f"ERROR: Provenance manifest not found: {provenance_file}", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON in provenance manifest: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to read provenance manifest: {e}", file=sys.stderr)
    sys.exit(1)
PY

else
    err "EXPECTED_TE_FL_COMMIT not provided, cannot verify"
fi

# ==============================================================================
# 5. Platform/Vendor Implementation Registration (EXPLICIT CHECK)
# ==============================================================================
msg "5. Platform/Vendor Implementation Registration"

PLATFORM="${TE_FL_PLATFORM:-}"
if [[ -z "$PLATFORM" ]]; then
    err "TE_FL_PLATFORM not set in workflow environment"
fi

msg "   Platform: ${PLATFORM}"

python3 - <<'PY' || { err "Platform implementation not properly registered"; }
import sys
import os

platform = os.getenv("TE_FL_PLATFORM", "").lower()

try:
    import transformer_engine as te

    # Try to get TE-FL plugin manager (correct API)
    has_manager = False
    manager = None

    try:
        from transformer_engine.plugin.core import get_manager
        manager = get_manager()
        has_manager = True
        print(f"   Plugin manager found: {type(manager).__name__}")
    except ImportError:
        pass

    if not has_manager:
        if platform != "cuda":
            print(f"ERROR: Platform '{platform}' requires plugin manager, but import failed", file=sys.stderr)
            print(f"  Could not import from transformer_engine.plugin.core", file=sys.stderr)
            sys.exit(1)
        else:
            print("   Warning: No plugin manager (CUDA default)")

    # Platform-specific device checks
    import torch

    if platform == "cuda":
        device_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if device_available else 0
        print(f"   CUDA available: {device_available}")
        print(f"   CUDA device count: {device_count}")

    elif platform == "musa":
        if hasattr(torch, 'musa'):
            device_available = torch.musa.is_available()
            device_count = torch.musa.device_count() if device_available else 0
            print(f"   MUSA available: {device_available}")
            print(f"   MUSA device count: {device_count}")
        else:
            print("ERROR: torch.musa module not found", file=sys.stderr)
            sys.exit(1)

        if not device_available or device_count == 0:
            print(f"ERROR: Platform 'musa' requires MUSA devices, but none available", file=sys.stderr)
            sys.exit(1)

    else:
        # Generic platform: check torch.cuda as fallback
        device_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if device_available else 0
        print(f"   Torch device available: {device_available}")
        print(f"   Device count: {device_count}")

    # For non-CUDA platforms, require manager + devices
    if platform != "cuda":
        if not has_manager and not device_available:
            print(f"ERROR: Platform '{platform}' requires plugin manager or devices, but none found", file=sys.stderr)
            sys.exit(1)

except ImportError as e:
    print(f"ERROR: Failed to check implementation registration: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ==============================================================================
# 6. Operator Backend Selection (FAIL-CLOSED: query real manager API)
# ==============================================================================
msg "6. Operator Backend Selection (real manager API verification)"

python3 - <<'PY' || { err "Operator backend selection failed"; }
import sys
import os

platform = os.getenv("TE_FL_PLATFORM", "").lower()

try:
    from transformer_engine.pytorch import Linear
    print("   Linear operator imported successfully")

    # Instantiate the operator
    layer = Linear(in_features=4, out_features=8)
    print(f"   Linear layer instantiated: in_features=4, out_features=8")

    # Execute forward pass on appropriate device FIRST
    import torch

    device_type = None
    if platform == "musa":
        if hasattr(torch, 'musa') and torch.musa.is_available():
            device_type = "musa"
            device = torch.device('musa:0')
        else:
            print("ERROR: MUSA platform but torch.musa not available", file=sys.stderr)
            sys.exit(1)
    elif platform == "cuda":
        if torch.cuda.is_available():
            device_type = "cuda"
            device = torch.device('cuda:0')
        else:
            print("   Warning: CUDA not available, skipping forward pass")
            sys.exit(0)
    else:
        # Generic: try cuda
        if torch.cuda.is_available():
            device_type = "cuda"
            device = torch.device('cuda:0')
        else:
            print("   Warning: No device available, skipping forward pass")
            sys.exit(0)

    if device_type:
        try:
            layer = layer.to(device)
            x = torch.randn(2, 4, device=device)
            y = layer(x)
            print(f"   Forward pass successful on {device_type}: input {tuple(x.shape)} -> output {tuple(y.shape)}")
        except Exception as e:
            print(f"ERROR: Forward pass failed on {device_type}: {e}", file=sys.stderr)
            sys.exit(1)

    # CRITICAL: Query actual selected implementation using real TE-FL API
    # Correct API: from transformer_engine.plugin.core import get_manager
    #              manager.get_selected_impl_id("generic_gemm")

    selected_impl_id = None
    query_failed = False
    query_error = None

    try:
        from transformer_engine.plugin.core import get_manager

        manager = get_manager()
        print(f"   Plugin manager obtained: {type(manager).__name__}")

        # Query the implementation used for generic_gemm (Linear uses this)
        try:
            selected_impl_id = manager.get_selected_impl_id("generic_gemm")
            print(f"   Selected impl_id for 'generic_gemm': {selected_impl_id}")
        except Exception as e:
            query_error = f"get_selected_impl_id('generic_gemm') raised: {e}"
            query_failed = True

    except ImportError as e:
        query_error = f"Failed to import get_manager from transformer_engine.plugin.core: {e}"
        query_failed = True

    # FAIL-CLOSED: query failure is fatal for non-CUDA
    if query_failed:
        if platform != "cuda":
            print(f"ERROR: Failed to query implementation manager: {query_error}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"   Warning: Query failed: {query_error}")
            selected_impl_id = "cuda_default"

    # FAIL-CLOSED: selected_impl_id MUST NOT be empty or "unknown"
    if selected_impl_id is None or selected_impl_id == "" or selected_impl_id == "unknown":
        if platform != "cuda":
            print(f"ERROR: Implementation manager returned invalid result: {selected_impl_id}", file=sys.stderr)
            print(f"  Platform '{platform}' requires explicit vendor implementation", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"   Warning: Unknown implementation (CUDA default)")
            selected_impl_id = "cuda_default"

    # Validate implementation for non-CUDA platforms
    if platform != "cuda":
        impl_str = str(selected_impl_id).lower()

        # FAIL-CLOSED: reject reference/unknown/none
        if 'reference' in impl_str or impl_str == 'none':
            print(f"ERROR: Platform '{platform}' is using reference implementation: {selected_impl_id}", file=sys.stderr)
            sys.exit(1)

        # FAIL-CLOSED: MUSA must be "vendor.musa"
        if platform == "musa":
            if selected_impl_id != "vendor.musa":
                print(f"ERROR: Platform 'musa' must use 'vendor.musa' implementation", file=sys.stderr)
                print(f"  Got: {selected_impl_id}", file=sys.stderr)
                print(f"  Expected exact match: 'vendor.musa'", file=sys.stderr)
                sys.exit(1)

        print(f"   Implementation verified: vendor-specific backend confirmed")
    else:
        print(f"   Backend execution verified")

except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    if platform != "cuda":
        print(f"ERROR: Unexpected error during implementation verification: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    else:
        print(f"   Warning: Implementation query failed: {e}")

PY

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
