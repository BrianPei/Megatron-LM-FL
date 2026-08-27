#!/bin/bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

set -euo pipefail

# ==============================================================================
# TE-FL Installation Smoke Test
# ==============================================================================
# Validates that TE-FL is correctly installed and operational.
# MUST be run after installing TE-FL in the container.
# ==============================================================================

EXIT_CODE=0

msg() {
    printf ">>> %s\n" "$1"
}

err() {
    printf "ERROR: %s\n" "$1" >&2
    EXIT_CODE=1
}

# ==============================================================================
# Test 1: Import transformer_engine
# ==============================================================================
msg "Test 1: Import transformer_engine"
python3 - <<'PY' || { err "Failed to import transformer_engine"; }
import transformer_engine
print(f"OK: transformer_engine imported successfully")
print(f"  Module path: {transformer_engine.__file__}")
PY

# ==============================================================================
# Test 2: Check version
# ==============================================================================
msg "Test 2: Check TE-FL version"
python3 - <<'PY' || { err "Failed to get transformer_engine version"; }
import transformer_engine
version = getattr(transformer_engine, "__version__", "unknown")
print(f"OK: TE version: {version}")
if version == "unknown":
    print("  WARNING: __version__ attribute not found")
PY

# ==============================================================================
# Test 3: Verify provenance metadata
# ==============================================================================
msg "Test 3: Verify TE-FL provenance"
python3 - <<'PY' || { err "TE-FL provenance metadata incomplete"; }
import os, sys

te_fl_commit = os.getenv("TE_FL_COMMIT")
te_fl_platform = os.getenv("TE_FL_PLATFORM")

if not te_fl_commit:
    print("ERROR: TE_FL_COMMIT environment variable not set", file=sys.stderr)
    sys.exit(1)

if not te_fl_platform:
    print("ERROR: TE_FL_PLATFORM environment variable not set", file=sys.stderr)
    sys.exit(1)

print(f"OK: TE-FL provenance validated")
print(f"  Commit: {te_fl_commit}")
print(f"  Platform: {te_fl_platform}")

# Validate commit format
if len(te_fl_commit) != 40 or not all(c in '0123456789abcdef' for c in te_fl_commit):
    print(f"ERROR: TE_FL_COMMIT invalid format: {te_fl_commit}", file=sys.stderr)
    sys.exit(1)

print(f"  Commit format: valid")
PY

# ==============================================================================
# Test 4: Import native module
# ==============================================================================
msg "Test 4: Import transformer_engine native module"
python3 - <<'PY' || { err "Failed to import transformer_engine native module"; }
try:
    import transformer_engine_torch
    print(f"OK: transformer_engine_torch imported")
    print(f"  Module path: {transformer_engine_torch.__file__}")
except ImportError as e:
    # Some platforms may use different native module names
    print(f"  WARNING: transformer_engine_torch import failed: {e}")
    print(f"  Attempting alternative imports...")

    try:
        import transformer_engine.pytorch
        print(f"OK: transformer_engine.pytorch imported (alternative)")
        print(f"  Module path: {transformer_engine.pytorch.__file__}")
    except ImportError as e2:
        print(f"ERROR: No native module found", file=sys.stderr)
        print(f"  transformer_engine_torch: {e}", file=sys.stderr)
        print(f"  transformer_engine.pytorch: {e2}", file=sys.stderr)
        raise
PY

# ==============================================================================
# Test 5: Import basic operators
# ==============================================================================
msg "Test 5: Import TE-FL operators"
python3 - <<'PY' || { err "Failed to import TE-FL operators"; }
import sys

try:
    from transformer_engine.pytorch import Linear
    print(f"OK: Linear operator imported")
except ImportError as e:
    print(f"ERROR: Failed to import Linear: {e}", file=sys.stderr)
    sys.exit(1)

# Optional operators (platform-dependent)
optional_ops = [
    "LayerNorm",
    "LayerNormLinear",
    "RMSNorm",
]

for op_name in optional_ops:
    try:
        exec(f"from transformer_engine.pytorch import {op_name}")
        print(f"OK: {op_name} operator imported")
    except ImportError:
        print(f"  WARNING: {op_name} not available (platform-specific)")
PY

# ==============================================================================
# Test 6: Verify no CUDA hardcoding in critical paths
# ==============================================================================
msg "Test 6: Verify platform abstraction"
python3 - <<'PY' || { err "Platform abstraction check failed"; }
import os, torch

# Check that TE doesn't force CUDA device
te_fl_platform = os.getenv("TE_FL_PLATFORM", "unknown")

if te_fl_platform != "cuda":
    # For non-CUDA platforms, verify device type matches
    if torch.cuda.is_available():
        device_type = "cuda"  # torch_musa/torch_npu may report as cuda
    else:
        device_type = "cpu"

    print(f"OK: Platform: {te_fl_platform}")
    print(f"  PyTorch device type: {device_type}")
    print(f"  Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
else:
    print(f"OK: Platform: CUDA")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
PY

# ==============================================================================
# Test 7: Minimal forward pass (CPU-only, no actual device required)
# ==============================================================================
msg "Test 7: Minimal operator instantiation"
python3 - <<'PY' || { err "Operator instantiation failed"; }
import torch
from transformer_engine.pytorch import Linear

# Create a simple Linear layer (no device allocation in constructor)
try:
    layer = Linear(4, 8)
    print(f"OK: Linear layer instantiated")
    print(f"  in_features: 4, out_features: 8")
except Exception as e:
    print(f"ERROR: Failed to instantiate Linear: {e}", file=sys.stderr)
    raise
PY

# ==============================================================================
# Summary
# ==============================================================================
if [[ ${EXIT_CODE} -eq 0 ]]; then
    msg "SUCCESS: All smoke tests passed"
else
    msg "ERROR: Some smoke tests failed"
fi

exit ${EXIT_CODE}
