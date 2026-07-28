#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

configure_hygon_runtime() {
  # DTK exposes BW1000 through PyTorch's CUDA-compatible API. FlagGems uses
  # the AMD Triton backend in the documented image.
  local dtk_home="${DTK_HOME:-/opt/dtk}"
  local hip_path="${HIP_PATH:-$dtk_home/hip}"
  local hsa_path="${HSA_PATH:-$dtk_home/hsa}"
  local dtk_path="$dtk_home/bin:$hip_path/bin:$dtk_home/llvm/bin"
  local dtk_library_path="/opt/hyhal/lib/criu:/opt/hyhal/lib/rocprofiler:/opt/hyhal/lib:$hip_path/lib:$dtk_home/lib:$dtk_home/llvm/lib:$dtk_home/dcc/lib:$dtk_home/aillvm/lib:$hsa_path/lib"

  ci_export_env GEMS_VENDOR amd
  ci_export_env CUDA_VISIBLE_DEVICES "0,1,2,3,4,5,6,7"
  # RCCL warns about this at init and hangs in ncclCommInitRank without it,
  # so every 8-card collective on BW1000 needs it set before torchrun starts.
  ci_export_env HSA_FORCE_FINE_GRAIN_PCIE 1
  # Preserve warnings and errors without printing every process-group setup
  # from all eight ranks.
  ci_export_env NCCL_DEBUG WARN
  ci_export_env TORCH_CPP_LOG_LEVEL WARNING
  # PyTorch otherwise starts 32 Inductor workers per rank in this image.
  ci_export_env TORCHINDUCTOR_COMPILE_THREADS 1
  ci_export_env TE_FL_SKIP_CUDA 1
  ci_export_env TE_FL_PREFER flagos
  ci_export_env NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD 1
  ci_export_env NVTE_FUSED_ATTN 0
  ci_export_env NVTE_UNFUSED_ATTN 0
  ci_export_env DTK_HOME "$dtk_home"
  ci_export_env ROCM_PATH "$dtk_home"
  ci_export_env HIP_PATH "$hip_path"
  ci_export_env HSA_PATH "$hsa_path"
  ci_export_env HIP_CLANG_PATH "$dtk_home/llvm/bin"
  ci_export_env DEVICE_LIB_PATH "$dtk_home/amdgcn/bitcode"
  ci_export_env PATH "$dtk_path:$PATH"
  ci_export_env LD_LIBRARY_PATH "$dtk_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  test -d /opt/hyhal
  test -e /dev/kfd
  test -e /dev/mkfd
  test -d /dev/dri
}

# Catch a stale or mis-tagged image before the tests start.
verify_hygon_software_stack() {
  python3 - <<'PY'
import flag_gems
import transformer_engine
import transformer_engine.pytorch

print(f"FlagGems import passed: {flag_gems.__file__}")
print(f"TransformerEngine-FL import passed: {transformer_engine.__file__}")
PY
}

install_hygon_project() {
  cd "$CI_PROJECT_ROOT"
  git config --global --add safe.directory "$CI_PROJECT_ROOT"
  # The validated DTK image currently provides Python 3.10 while the project
  # metadata requires 3.12. The documented training stack is source-compatible
  # with 3.10, so bypass only the metadata gate and preserve the image runtime.
  python3 -m pip install -e . \
    --ignore-requires-python \
    --no-deps \
    --no-build-isolation \
    --no-cache-dir
}

validate_hygon_capacity() {
  python3 - <<'PY'
import torch

print(f"Torch: {torch.__version__}")
print(f"HIP runtime: {torch.version.hip}")
print(f"CUDA-compatible API available: {torch.cuda.is_available()}")
print(f"Visible BW1000 devices: {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count():
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
PY

  local device_count
  device_count=$(python3 -c \
    "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
  ci_validate_device_capacity "$device_count"
}

setup_unit_environment() {
  ci_activate_python_environment
  configure_hygon_runtime
  ci_ensure_curl

  echo "Preserving the PyTorch and DTK packages supplied by the BW1000 image."
  echo "Skipping NVIDIA CUPTI, NVRx, and Emerging Optimizers dependencies."
  verify_hygon_software_stack
  install_hygon_project
  validate_hygon_capacity
}

setup_functional_environment() {
  ci_activate_python_environment
  configure_hygon_runtime

  local python_bin
  python_bin=$(command -v python3)
  if [ "$python_bin" != "/usr/bin/python3" ] && \
     [ "$python_bin" != "/usr/local/bin/python3" ]; then
    ln -sf "$python_bin" /usr/local/bin/python3
  fi

  ci_install_yq
  ci_install_envsubst
  ci_install_uv_compatibility_shim
  verify_hygon_software_stack
  install_hygon_project
  validate_hygon_capacity
}

setup_build_environment() {
  ci_activate_python_environment
  configure_hygon_runtime
  verify_hygon_software_stack
  install_hygon_project
  validate_hygon_capacity
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
