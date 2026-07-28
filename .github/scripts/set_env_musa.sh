#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

configure_musa_runtime() {
  ci_export_env DISTRIBUTED_BACKEND mccl
  ci_export_env TORCHDYNAMO_DISABLE 1
  ci_export_env TORCH_COMPILE_DISABLE 1
  ci_export_env LD_LIBRARY_PATH "/usr/local/musa-4.3.4/lib:${LD_LIBRARY_PATH:-}"
}

validate_musa_capacity() {
  local device_count
  device_count=$(python3 -c "import torch; print(torch.musa.device_count())")
  ci_validate_device_capacity "$device_count"
}

install_musa_project() {
  cd "$CI_PROJECT_ROOT"
  python3 -m pip install -e . \
    --no-deps \
    --no-build-isolation \
    --no-cache-dir \
    --ignore-requires-python
}

setup_unit_environment() {
  ci_activate_python_environment
  configure_musa_runtime
  ci_ensure_curl

  local test_dependencies=(
    boto3
    mock
    pytest-mock
    coverage
    pytest-asyncio
    anyio
    wandb
    openai
    httpx
    nltk
  )
  python3 -m pip install "${test_dependencies[@]}" --no-cache-dir
  python3 -m pip install fastapi uvicorn --no-cache-dir

  echo "Skipping NVIDIA CUPTI and Emerging-Optimizers dependencies on MUSA."
  install_musa_project
  validate_musa_capacity
}

setup_build_environment() {
  ci_activate_python_environment
  configure_musa_runtime
  install_musa_project
  validate_musa_capacity
}

setup_functional_environment() {
  configure_musa_runtime
  ci_setup_functional_environment --ignore-requires-python

  # Keep the image-provided torch/torch_musa pair intact. torchada is a
  # pure-Python compatibility layer that redirects CUDA APIs and device
  # strings to MUSA.
  python3 -m pip install \
    torchada==0.1.40 \
    --no-deps \
    --no-cache-dir
  validate_musa_capacity

  mkdir -p /tmp/musa-ci-site
  cat > /tmp/musa-ci-site/sitecustomize.py <<'SITEEOF'
import torchada  # noqa: F401
import torch

from megatron.plugin.platform import get_platform


# torchada intentionally leaves this probe unchanged for platform detection.
# Select and cache MUSA first, then satisfy Megatron's legacy CUDA assertion.
if get_platform().device_name() == "musa":
    torch.cuda.is_available = torch.musa.is_available
SITEEOF
  ci_export_env PYTHONPATH "/tmp/musa-ci-site:${PYTHONPATH:-}"
}

ci_require_env CI_TEST_SUITE
case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    setup_functional_environment
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
