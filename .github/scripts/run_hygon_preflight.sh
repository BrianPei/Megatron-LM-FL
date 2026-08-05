#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! [[ "${CI_NPROC_PER_NODE:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::CI_NPROC_PER_NODE must be a positive integer"
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "::error::GNU timeout is required for Hygon distributed preflight"
  exit 1
fi

probe_exit_code=0
echo "Running the Hygon device and RCCL preflight."
timeout \
  --signal=TERM \
  --kill-after=15s \
  180s \
  python3 -u -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="$CI_NPROC_PER_NODE" \
    --rdzv-backend=c10d \
    --rdzv-endpoint=127.0.0.1:29500 \
    --rdzv-id=hygon-preflight \
    "$PROJECT_ROOT/.github/scripts/probe_hygon_distributed.py" ||
  probe_exit_code=$?

if [ "$probe_exit_code" -ne 0 ]; then
  if [ "$probe_exit_code" -eq 124 ] || [ "$probe_exit_code" -eq 137 ]; then
    echo "::error::Hygon distributed preflight timed out. The host device or KFD state must be recovered before rerunning CI."
  else
    echo "::error::Hygon distributed preflight failed with exit code $probe_exit_code."
  fi
  exit "$probe_exit_code"
fi
