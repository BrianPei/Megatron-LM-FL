#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

ci_require_env CI_TEST_SUITE
if [ "$CI_TEST_SUITE" != unit ]; then
  echo "::error::PPU currently supports CI_TEST_SUITE=unit only" >&2
  exit 1
fi
ci_require_env CI_NPROC_PER_NODE
if ! [[ "$CI_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::CI_NPROC_PER_NODE must be a positive integer" >&2
  exit 1
fi

ci_activate_python_environment
ci_export_env CUDA_VISIBLE_DEVICES 0,1,2,3,4,5,6,7
ci_export_env CUDA_DEVICE_MAX_CONNECTIONS 1
ci_export_env OMP_NUM_THREADS 1
ci_export_env NCCL_DEBUG WARN
ci_export_env NCCL_MAX_NCHANNELS 1
ci_export_env NCCL_NVLS_ENABLE 0
# The manual tests use the vendor NCCL/PCCL interface, not the FlagCX plugin.
ci_export_env DISTRIBUTED_BACKEND nccl

"$CI_PYTHON_BIN" -c \
  'import torch; assert torch.cuda.is_available(); print(f"Torch: {torch.__version__}")'
device_count=$("$CI_PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())' |
  awk '/^[0-9]+$/ { count = $0 } END { print count }')
ci_validate_device_capacity "$device_count"

for asset in datasets tokenizers; do
  if [ ! -d "/opt/data/$asset" ] || [ -z "$(ls -A "/opt/data/$asset")" ]; then
    echo "::error::Missing /opt/data/$asset; build the completed PPU CI image first" >&2
    exit 1
  fi
done

# Install only the checked-out Megatron source. Never resolve the vendor stack here.
ci_install_project
