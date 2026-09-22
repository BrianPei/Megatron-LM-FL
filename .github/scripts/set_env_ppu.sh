#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

configure_ppu_runtime() {
  ci_export_env CUDA_VISIBLE_DEVICES 0,1,2,3,4,5,6,7
  ci_export_env CUDA_DEVICE_MAX_CONNECTIONS 1
  ci_export_env OMP_NUM_THREADS 1
  ci_export_env NCCL_DEBUG WARN
  ci_export_env NCCL_MAX_NCHANNELS 1
  ci_export_env NCCL_NVLS_ENABLE 0
  # The manual tests use the vendor NCCL/PCCL interface, not the FlagCX plugin.
  ci_export_env DISTRIBUTED_BACKEND nccl
}

validate_ppu_capacity() {
  "$CI_PYTHON_BIN" -c \
    'import torch; assert torch.cuda.is_available(); print(f"Torch: {torch.__version__}")'
  local device_count
  device_count=$("$CI_PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())' |
    awk '/^[0-9]+$/ { count = $0 } END { print count }')
  ci_validate_device_capacity "$device_count"
}

setup_unit_environment() {
  ci_activate_python_environment
  configure_ppu_runtime
  validate_ppu_capacity

  # Validate the mounted fixtures before pytest; nonempty directories are insufficient.
  "$CI_PYTHON_BIN" - /opt/data <<'PY'
import sys
from pathlib import Path

from transformers import AutoTokenizer

root = Path(sys.argv[1])
required_files = (
    "datasets/fim/fim_text_document.bin",
    "datasets/fim/fim_text_document.idx",
    "tokenizers/sentencepiece/tokenizer.model",
    "tokenizers/megatron/gpt2-vocab.json",
    "tokenizers/megatron/gpt2-merges.txt",
    "tokenizers/tiktoken/tiktoken.vocab.json",
)
missing = [
    str(root / name)
    for name in required_files
    if not (root / name).is_file() or (root / name).stat().st_size == 0
]
if missing:
    raise SystemExit("::error::Missing or empty PPU test assets: " + ", ".join(missing))

for name in ("huggingface", "multimodal"):
    path = root / "tokenizers" / name
    if not path.is_dir():
        raise SystemExit(f"::error::Missing PPU tokenizer fixture: {path}")
    try:
        AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except Exception as error:
        raise SystemExit(f"::error::Cannot load local PPU tokenizer fixture {path}: {error}")

print(f"PPU test assets validated: {root}")
PY

  # Install only the checked-out Megatron source. Never resolve the vendor stack here.
  ci_install_project
}

setup_functional_environment() {
  configure_ppu_runtime
  ci_setup_functional_environment
  ci_install_local_tokenizer_dependencies
  ci_validate_qwen_assets /home/gitlab-runner/data /home/gitlab-runner/tokenizers
  validate_ppu_capacity
}

ci_require_env CI_TEST_SUITE
ci_require_env CI_NPROC_PER_NODE
if ! [[ "$CI_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::CI_NPROC_PER_NODE must be a positive integer" >&2
  exit 1
fi

case "$CI_TEST_SUITE" in
  unit)
    setup_unit_environment
    ;;
  functional)
    setup_functional_environment
    ;;
  *)
    echo "::error::Unsupported CI_TEST_SUITE: $CI_TEST_SUITE"
    exit 1
    ;;
esac
