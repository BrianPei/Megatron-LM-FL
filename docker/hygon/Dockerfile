# syntax=docker/dockerfile:1.4
# Hygon DCU Platform Dockerfile
ARG BASE_IMAGE
ARG PYTHON_VERSION=3.10

FROM ${BASE_IMAGE} AS platform-runtime
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

FROM platform-runtime AS te-fl-native
ARG TE_FL_ARTIFACT
ARG TE_FL_COMMIT
ARG TE_FL_REPOSITORY=https://github.com/FlagOpen/TransformerEngine-FL.git
ARG PLATFORM=hygon
COPY common/install_te_fl.sh /tmp/install_te_fl.sh
RUN chmod +x /tmp/install_te_fl.sh && \
    if [ -z "${TE_FL_ARTIFACT}" ]; then echo "ERROR: TE_FL_ARTIFACT required" >&2; exit 1; fi && \
    /tmp/install_te_fl.sh --artifact "${TE_FL_ARTIFACT}" --commit "${TE_FL_COMMIT}" \
        --repository "${TE_FL_REPOSITORY}" --platform "${PLATFORM}"
ENV TE_FL_COMMIT=${TE_FL_COMMIT} TE_FL_PLATFORM=${PLATFORM} TE_FL_REPOSITORY=${TE_FL_REPOSITORY}
LABEL com.flagos.te_fl.commit="${TE_FL_COMMIT}" com.flagos.te_fl.platform="${PLATFORM}"

FROM te-fl-native AS ci
ARG TORCH_VERSION
ARG VENDOR_SDK
RUN pip install --no-cache-dir pytest==8.3.3 pytest-cov==5.0.0 pytest-timeout==2.3.1 \
    pytest-metadata==3.1.1 pytest-json-report==1.5.0 nltk==3.9.1 zarr==2.18.3 tensorstore==0.1.67
COPY common/smoke_test.sh /usr/local/bin/smoke_test.sh
RUN chmod +x /usr/local/bin/smoke_test.sh && smoke_test.sh
WORKDIR /workspace
LABEL org.opencontainers.image.title="Megatron-LM-FL CI (Hygon)" \
    com.flagos.megatron.platform="hygon" com.flagos.megatron.torch_version="${TORCH_VERSION}"

FROM ci AS dev
RUN pip install --no-cache-dir ipython==8.28.0 jupyter==1.1.1 pre-commit==4.0.1 ruff==0.7.4 black==24.10.0
RUN apt-get update && apt-get install -y --no-install-recommends vim tmux htop && rm -rf /var/lib/apt/lists/*
