#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

configure_kunlunxin_runtime() {
  # KunLunXin P800 uses XMLIR to expose XPU as a CUDA-compatible device.
  # FlagCx is the collective communication library (KunLunXin's equivalent
  # of NCCL).  TE_FL_SKIP_CUDA tells TransformerEngine-FL not to probe the
  # CUDA vendor backend so it falls through to the kunlunxin vendor path.
  ci_export_env XPU 1
  ci_export_env DISTRIBUTED_BACKEND flagcx
  ci_export_env TE_FL_SKIP_CUDA 1
  ci_export_env KLX_USE_AUTOTUNE 0
}

validate_kunlunxin_capacity() {
  local device_count
  device_count=$(python3 -c \
    "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
  ci_validate_device_capacity "$device_count"
}

setup_unit_environment() {
  ci_activate_python_environment
  ci_install_project
  configure_kunlunxin_runtime
  validate_kunlunxin_capacity
}

setup_build_environment() {
  ci_activate_python_environment
  ci_install_project
  configure_kunlunxin_runtime
  validate_kunlunxin_capacity
}

ci_require_env CI_TEST_SUITE
case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    ci_setup_functional_environment
    configure_kunlunxin_runtime
    validate_kunlunxin_capacity
    ;;
  build)
    setup_build_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
