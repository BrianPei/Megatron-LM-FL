#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

# Override the common install helper: pyproject.toml declares
# requires-python>=3.12, but the KunLunXin torch environment ships
# Python 3.10. --ignore-requires-python bypasses that version gate;
# the code runs correctly with the XPU stack on 3.10.
ci_install_project() {
  cd "$CI_PROJECT_ROOT"
  python3 -m pip install -e . --no-deps --no-build-isolation \
      --no-cache-dir --ignore-requires-python
}

activate_kunlunxin_python_environment() {
  # The KunLunXin CI image installs conda under /root/miniconda, not
  # /opt/conda, so the common ci_activate_python_environment() guard
  # never fires. Activate the PyTorch/XPU environment directly.
  source /root/miniconda/etc/profile.d/conda.sh
  conda activate python310_torch29_cuda
  ci_export_env PATH "$PATH"
  echo "Python: $(command -v python3) ($(python3 --version 2>&1))"
}

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
  activate_kunlunxin_python_environment
  # data preprocessing tests fork multiprocessing workers whose initializer
  # calls nltk.load('tokenizers/punkt/PY3/english.pickle'); pre-download the
  # punkt model via curl (uses the CI proxy) instead of the interactive
  # nltk.downloader which prompts for retry and fails with EOFError.
  mkdir -p /usr/local/share/nltk_data/tokenizers
  curl -fsSL \
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip" \
    -o /tmp/punkt.zip && \
  unzip -o /tmp/punkt.zip -d /usr/local/share/nltk_data/tokenizers/
  ci_install_project
  configure_kunlunxin_runtime
  validate_kunlunxin_capacity
}

setup_build_environment() {
  activate_kunlunxin_python_environment
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
