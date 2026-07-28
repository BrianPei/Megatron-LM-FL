#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-dtk26.04-py3.10}"
TARGET_IMAGE="${TARGET_IMAGE:-harbor.baai.ac.cn/flagos-dev/megatron-lm-fl:manual-20260728-hygon-dev}"
BUILD_CONTAINER="${BUILD_CONTAINER:-megatron-hygon-image-build}"
DIGEST_OUTPUT="${DIGEST_OUTPUT:-/home/secure/cicd_dev/megatron-lm-fl/hygon-image-digest.txt}"

if [ "$(id -un)" != "secure" ]; then
  echo "Run this script as the secure user."
  exit 1
fi

if [ -n "$(docker ps --filter "ancestor=$BASE_IMAGE" --format '{{.ID}}')" ]; then
  echo "An Actions container is still using $BASE_IMAGE. Wait for it to finish first."
  exit 1
fi

proxy_status=$(docker run --rm --pull=never "$BASE_IMAGE" bash -lc '
  if [ -n "${https_proxy:-${HTTPS_PROXY:-}}" ]; then
    echo set
  else
    echo unset
  fi
')
if [ "$proxy_status" != "set" ]; then
  echo "The secure user's Docker proxy configuration is not reaching containers."
  exit 1
fi
echo "Container proxy configuration: set"

source_ref=$(docker run --rm \
  --pull=never \
  --volume "$ROOT_DIR:/repo:ro" \
  "$BASE_IMAGE" \
  git -c safe.directory=/repo -C /repo rev-parse HEAD)
echo "Megatron source: $source_ref"

docker rm -f "$BUILD_CONTAINER" >/dev/null 2>&1 || true

docker run -d \
  --name "$BUILD_CONTAINER" \
  --pull=never \
  --ipc=host \
  --shm-size=32g \
  --user root \
  --ulimit nofile=65535:65535 \
  --volume /opt/hyhal:/opt/hyhal:ro \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  "$BASE_IMAGE" \
  bash -lc 'sleep infinity'

docker exec "$BUILD_CONTAINER" mkdir -p /workspace/repo
docker cp "$ROOT_DIR/." "$BUILD_CONTAINER:/workspace/repo"

docker exec \
  --env CI_TEST_SUITE=unit \
  --env CI_PLATFORM=hygon \
  --env CI_DEVICE=bw1000 \
  --env CI_TEST_GROUP=image-build \
  --env CI_NPROC_PER_NODE=8 \
  "$BUILD_CONTAINER" \
  bash /workspace/repo/.github/scripts/set_env_hygon.sh

docker exec \
  --env MEGATRON_SOURCE_REF="$source_ref" \
  "$BUILD_CONTAINER" \
  bash -lc '
    set -euo pipefail

    source /workspace/repo/.github/scripts/set_env_common.sh

    for attempt in 1 2 3 4 5; do
      rm -f /usr/local/bin/yq
      if ci_install_yq; then
        break
      fi
      test "$attempt" -lt 5
      sleep $((attempt * 5))
    done

    ci_install_envsubst
    python3 -m pip install \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --retries 10 \
      --timeout 60 \
      "uv<0.9.29" \
      "tensorboard<2.18" \
      --no-cache-dir

    source_root=$(find /tmp -maxdepth 1 -type d \
      -name "megatron-hygon-dependencies.*" -print -quit)
    test -n "$source_root"
    test -d "$source_root/FlagGems"
    test -d "$source_root/TransformerEngine-FL"

    python3 -m pip uninstall -y flag-gems transformer-engine megatron-core
    python3 -m pip install "$source_root/FlagGems" \
      --no-deps --no-build-isolation --no-cache-dir
    python3 -m pip install "$source_root/TransformerEngine-FL" \
      --no-deps --no-build-isolation --no-cache-dir

    git config --global --unset-all safe.directory || true

    mkdir -p /opt/flagos-ci
    {
      echo "base_image=harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-dtk26.04-py3.10"
      echo "megatron_build_ref=$MEGATRON_SOURCE_REF"
      echo "flaggems_ref=66a4ddb3656bf2fc4d305f610a5c49c26192bb04"
      echo "te_fl_ref=b7f65d1b4a4c73b554e5b8f5ce0547eab0c3c35a"
      echo "architecture=$(uname -m)"
      echo "python=$(python3 --version 2>&1)"
      echo "torch=$(python3 -c "import torch; print(torch.__version__)")"
      echo "tensorboard=$(python3 -c "import tensorboard; print(tensorboard.__version__)")"
      echo "yq=$(yq --version)"
      echo "uv=$(uv --version)"
    } > /opt/flagos-ci/manifest.txt
    python3 -m pip freeze > /opt/flagos-ci/python-freeze.txt

    rm -rf \
      "$source_root" \
      /workspace/repo \
      /root/.cache/pip \
      /root/.bash_history \
      /var/lib/apt/lists/*

    python3 - <<"PY"
import importlib.metadata
import pathlib
import shutil

import flag_gems
import tensorboard
import torch
import transformer_engine
import transformer_engine.pytorch

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 8
assert shutil.which("yq")
assert shutil.which("envsubst")
assert shutil.which("uv")
assert "/tmp/" not in str(pathlib.Path(flag_gems.__file__))
assert "/tmp/" not in str(pathlib.Path(transformer_engine.__file__))

try:
    importlib.metadata.version("megatron-core")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError("Megatron must not be included in the CI image")

print("torch:", torch.__version__)
print("devices:", torch.cuda.device_count())
print("flag_gems:", flag_gems.__file__)
print("transformer_engine:", transformer_engine.__file__)
print("tensorboard:", tensorboard.__version__)
PY
  '

docker commit \
  --change 'ENV http_proxy=' \
  --change 'ENV https_proxy=' \
  --change 'ENV HTTP_PROXY=' \
  --change 'ENV HTTPS_PROXY=' \
  --change 'ENV all_proxy=' \
  --change 'ENV ALL_PROXY=' \
  --change 'ENV ftp_proxy=' \
  --change 'ENV FTP_PROXY=' \
  --change 'ENV no_proxy=' \
  --change 'ENV NO_PROXY=' \
  --change 'WORKDIR /workspace' \
  --change 'CMD ["/bin/bash"]' \
  --change 'LABEL org.opencontainers.image.title=megatron-lm-fl-hygon-ci' \
  --change 'LABEL org.opencontainers.image.version=manual-20260728-hygon-dev' \
  --change "LABEL org.opencontainers.image.revision=$source_ref" \
  "$BUILD_CONTAINER" \
  "$TARGET_IMAGE"

docker rm -f "$BUILD_CONTAINER"

image_env=$(docker image inspect "$TARGET_IMAGE" --format '{{json .Config.Env}}')
if printf '%s\n' "$image_env" | grep -Eq 'https?://[^" ]+@'; then
  echo "Authenticated URL found in image environment; refusing to push."
  exit 1
fi

docker run --rm \
  --pull=never \
  --ipc=host \
  --shm-size=32g \
  --env GEMS_VENDOR=amd \
  --env TE_FL_SKIP_CUDA=1 \
  --env TE_FL_PREFER=flagos \
  --env LD_LIBRARY_PATH=/opt/hyhal/lib/criu:/opt/hyhal/lib/rocprofiler:/opt/hyhal/lib:/opt/dtk/hip/lib:/opt/dtk/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/lib:/opt/dtk/aillvm/lib:/opt/dtk/hsa/lib \
  --volume /opt/hyhal:/opt/hyhal:ro \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  "$TARGET_IMAGE" \
  bash -lc '
    cat /opt/flagos-ci/manifest.txt
    python3 -c "import torch, flag_gems, tensorboard, transformer_engine.pytorch; assert torch.cuda.device_count() >= 8; print(torch.__version__, torch.cuda.device_count())"
  '

docker push "$TARGET_IMAGE"
docker pull "$TARGET_IMAGE"
docker image inspect "$TARGET_IMAGE" --format '{{json .RepoDigests}}' | tee "$DIGEST_OUTPUT"
