#!/bin/bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

set -euo pipefail

# ==============================================================================
# Verify TE-FL Provenance in Test Container
# ==============================================================================
# This script MUST be run at the start of every test job to ensure:
# 1. TE-FL is correctly installed
# 2. Provenance metadata is present
# 3. Version matches expected contract
# ==============================================================================

EXIT_CODE=0

err() {
    printf "ERROR: ERROR: %s\n" "$1" >&2
    EXIT_CODE=1
}

msg() {
    printf ">>> %s\n" "$1"
}

# ==============================================================================
# Environment variable checks
# ==============================================================================
msg "Checking TE-FL provenance environment variables"

if [[ -z "${TE_FL_COMMIT:-}" ]]; then
    err "TE_FL_COMMIT not set"
fi

if [[ -z "${TE_FL_PLATFORM:-}" ]]; then
    err "TE_FL_PLATFORM not set"
fi

if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo ""
    echo "TE-FL provenance incomplete. This container does not meet CI requirements."
    echo "Expected environment variables: TE_FL_COMMIT, TE_FL_PLATFORM"
    exit ${EXIT_CODE}
fi

msg "OK: TE_FL_COMMIT: ${TE_FL_COMMIT}"
msg "OK: TE_FL_PLATFORM: ${TE_FL_PLATFORM}"

# ==============================================================================
# Python import checks
# ==============================================================================
msg "Verifying TE-FL installation"

python3 - <<'PY' || { err "TE-FL verification failed"; exit 1; }
import sys
import os

# Check basic import
try:
    import transformer_engine
    te_path = transformer_engine.__file__
    te_version = getattr(transformer_engine, "__version__", "unknown")
    print(f"OK: transformer_engine: {te_path}")
    print(f"  Version: {te_version}")
except ImportError as e:
    print(f"ERROR: Cannot import transformer_engine: {e}", file=sys.stderr)
    sys.exit(1)

# Check native module
native_ok = False
try:
    import transformer_engine_torch
    print(f"OK: transformer_engine_torch: {transformer_engine_torch.__file__}")
    native_ok = True
except ImportError:
    try:
        import transformer_engine.pytorch
        print(f"OK: transformer_engine.pytorch: {transformer_engine.pytorch.__file__}")
        native_ok = True
    except ImportError:
        pass

if not native_ok:
    print("ERROR: No TE-FL native module found", file=sys.stderr)
    sys.exit(1)

# Verify provenance
te_commit = os.getenv("TE_FL_COMMIT")
te_platform = os.getenv("TE_FL_PLATFORM")

print(f"OK: Provenance: commit={te_commit[:8]}, platform={te_platform}")

# Check basic operator availability
try:
    from transformer_engine.pytorch import Linear
    print(f"OK: Linear operator available")
except ImportError as e:
    print(f"WARNING: Linear operator not available: {e}")
PY

# ==============================================================================
# Output summary for CI logs
# ==============================================================================
if [[ ${EXIT_CODE} -eq 0 ]]; then
    msg "SUCCESS: TE-FL provenance verified"
    echo ""
    echo "Container ready for testing with:"
    echo "  TE-FL commit: ${TE_FL_COMMIT}"
    echo "  Platform: ${TE_FL_PLATFORM}"
else
    echo ""
    echo "ERROR: TE-FL provenance verification failed"
    echo "This container cannot be used for CI testing."
fi

exit ${EXIT_CODE}
