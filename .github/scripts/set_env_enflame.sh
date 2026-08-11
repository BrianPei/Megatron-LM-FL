#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

validate_enflame_torch() {
  python3 -c \
    "import torch, torch_gcu; print(f'Torch: {torch.__version__}, torch_gcu: {torch_gcu.__version__}')"
}

validate_enflame_capacity() {
  local device_count
  device_count=$(python3 -c \
    "import torch, torch_gcu; print(torch.gcu.device_count())" |
    awk '/^[0-9]+$/ { count = $0 } END { print count }')
  ci_validate_device_capacity "$device_count"

  python3 -c \
    "import torch, torch_gcu; assert torch.gcu.is_available(); print(f'GCU devices: {torch.gcu.device_count()}')"
}

configure_enflame_runtime() {
  validate_enflame_torch
  validate_enflame_capacity
}

disable_unavailable_test_asset_downloads() {
  local data_dir=/opt/data
  mkdir -p "$data_dir"

  # The Enflame unit runner does not mount the NVIDIA unit-test release assets.
  # Asset-dependent tests are excluded in enflame.yml; this marker prevents the
  # session fixture from downloading the same archives in every matrix job.
  if [ -z "$(find "$data_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    touch "$data_dir/.enflame-ci-assets-unavailable"
  fi
}

setup_unit_environment() {
  ci_activate_python_environment
  ci_ensure_curl
  validate_enflame_torch

  # Clean up stale .pth file from earlier image builds (commit 539097af3 → 271405d79).
  # The image may still contain fix_coverage_enflame.pth which causes SyntaxError
  # at Python startup. Remove both the .pth and the module it imports.
  python3 -c "
import site, os, glob
for sp in [site.getusersitepackages(), site.getsitepackages()[0]]:
    for pattern in ['fix_coverage_enflame.pth', '_fix_coverage_enflame.py']:
        for path in glob.glob(os.path.join(sp, pattern)):
            os.remove(path)
            print(f'Removed stale coverage patch: {path}')
" || true

  local test_dependencies=(
    mock
    pytest-mock
    coverage
    pytest-asyncio
    anyio
    wandb
    openai
    httpx
    nltk
    msgpack
  )
  local pip_index_args=(
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    --timeout 300
    --retries 10
    --no-cache-dir
    --break-system-packages
  )

  # boto3 is intentionally omitted: S3 unit tests provide a local mock, while
  # botocore downloads have been unreliable through the CI proxy.
  python3 -m pip install ninja "${test_dependencies[@]}" "${pip_index_args[@]}"
  echo "Ninja: $(ninja --version)"

  # Collection-only dependencies are installed without dependencies to preserve
  # the torch, protobuf, and numpy versions validated in the Enflame image.
  python3 -m pip install fastapi starlette uvicorn griffe \
    --no-deps "${pip_index_args[@]}"

  echo "Skipping NVIDIA CUPTI dependencies and Emerging-Optimizers on Enflame."

  # Workaround: torch_gcu registers _OpNamespace objects in sys.modules whose
  # __path__ attribute is not a sequence (no __len__). When coverage.py scans
  # already-imported modules at startup it calls len() on every module's
  # __path__, which raises TypeError and crashes all ranks before any test
  # runs. Other platforms (ascend/metax) don't hit this because their torch
  # backends don't inject non-sequence __path__ objects into sys.modules.
  #
  # Use a pytest plugin instead of a .pth file: .pth runs at Python startup
  # before coverage is installed; pytest_configure runs after all deps load.
  local site_dir=/tmp/enflame-ci-site
  mkdir -p "$site_dir"
  cat > "$site_dir/enflame_ci_pytest.py" <<'PYTESTEOF'
def pytest_configure(config):
    import coverage.inorout
    orig = coverage.inorout.InOrOut.warn_already_imported_files
    def safe_warn(self):
        try:
            orig(self)
        except TypeError:
            pass
    coverage.inorout.InOrOut.warn_already_imported_files = safe_warn
PYTESTEOF
  ci_export_env PYTHONPATH "$site_dir:${PYTHONPATH:-}"
  ci_export_env PYTEST_ADDOPTS "${PYTEST_ADDOPTS:-} -p enflame_ci_pytest"

  ci_install_project --break-system-packages
  configure_enflame_runtime
  disable_unavailable_test_asset_downloads
}

setup_build_environment() {
  ci_activate_python_environment
  validate_enflame_torch
  ci_install_project --break-system-packages
  configure_enflame_runtime
}

ci_require_env CI_TEST_SUITE
case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    validate_enflame_torch
    ci_setup_functional_environment
    configure_enflame_runtime
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
