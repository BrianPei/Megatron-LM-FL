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

ci_require_env CI_TEST_SUITE
case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    configure_musa_runtime

    # torch_musa exposes torch.musa but, unlike torch_npu's transfer_to_npu or
    # torch_txda's transfer_to_txda, ships no layer that remaps torch.cuda onto
    # the device.  get_platform() therefore selects PlatformCUDA (initialize.py
    # asserts torch.cuda.is_available() before distributed init, so it has to
    # report True) and every one of PlatformCUDA's ~40 torch.cuda.* calls hits a
    # CPU-only torch build.  Forward the whole namespace instead of chasing them
    # one at a time: copy each public torch.musa attribute onto its torch.cuda
    # namesake, so PlatformCUDA transparently runs on MUSA.
    mkdir -p /tmp/musa-ci-site
    cat > /tmp/musa-ci-site/sitecustomize.py <<'SITEEOF'
import torch
if hasattr(torch, "musa") and torch.musa.is_available():
    # Only override names that exist on both sides; MUSA-only attributes such as
    # MUSAGraph must not leak into the torch.cuda namespace.
    for _name in dir(torch.musa):
        if _name.startswith("_"):
            continue
        if hasattr(torch.cuda, _name):
            try:
                setattr(torch.cuda, _name, getattr(torch.musa, _name))
            except (AttributeError, TypeError):
                pass

    # The remaining overrides must come after the bulk copy: these are wrappers,
    # not plain forwards, and the loop would replace them with the raw
    # torch.musa versions.
    def _musa_device_index(device=None):
        if device is None:
            return torch.musa.current_device()
        if isinstance(device, (str, torch.device)):
            index = torch.device(device).index
            return torch.musa.current_device() if index is None else index
        return device

    def _musa_device_capability(device=None):
        properties = torch.musa.get_device_properties(_musa_device_index(device))
        return properties.major, properties.minor

    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_properties = (
        lambda device=None: torch.musa.get_device_properties(_musa_device_index(device))
    )
    torch.cuda.get_device_capability = _musa_device_capability
SITEEOF
    ci_export_env PYTHONPATH "/tmp/musa-ci-site:${PYTHONPATH:-}"

    # Shared functional test toolchain and Python 3.10-compatible project install.
    ci_setup_functional_environment --ignore-requires-python
    validate_musa_capacity
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
