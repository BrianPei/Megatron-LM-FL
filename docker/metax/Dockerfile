# syntax=docker/dockerfile:1.4
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

# ==============================================================================
# Megatron-LM-FL CI Image - MetaX Platform
# ==============================================================================
# Multi-stage build for MetaX MACA platform
# ==============================================================================

ARG BASE_IMAGE
ARG PYTHON_VERSION=3.12

FROM ${BASE_IMAGE} AS platform-runtime

ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

FROM platform-runtime AS te-fl-native

ARG TE_FL_ARTIFACT
ARG TE_FL_COMMIT
ARG TE_FL_REPOSITORY=https://github.com/FlagOpen/TransformerEngine-FL.git
ARG PLATFORM=metax

COPY common/install_te_fl.sh /tmp/install_te_fl.sh
RUN chmod +x /tmp/install_te_fl.sh

RUN if [ -z "${TE_FL_ARTIFACT}" ]; then \
        echo "ERROR: TE_FL_ARTIFACT not specified for MetaX platform" >&2; \
        exit 1; \
    fi && \
    /tmp/install_te_fl.sh \
        --artifact "${TE_FL_ARTIFACT}" \
        --commit "${TE_FL_COMMIT}" \
        --repository "${TE_FL_REPOSITORY}" \
        --platform "${PLATFORM}"

ENV TE_FL_COMMIT=${TE_FL_COMMIT}
ENV TE_FL_PLATFORM=${PLATFORM}
ENV TE_FL_REPOSITORY=${TE_FL_REPOSITORY}

LABEL com.flagos.te_fl.commit="${TE_FL_COMMIT}"
LABEL com.flagos.te_fl.platform="${PLATFORM}"
LABEL com.flagos.te_fl.artifact="${TE_FL_ARTIFACT}"

FROM te-fl-native AS ci

ARG TORCH_VERSION
ARG VENDOR_SDK

RUN pip install --no-cache-dir \
    pytest==8.3.3 \
    pytest-cov==5.0.0 \
    pytest-timeout==2.3.1 \
    pytest-metadata==3.1.1 \
    pytest-json-report==1.5.0 \
    nltk==3.9.1 \
    zarr==2.18.3 \
    tensorstore==0.1.67

COPY common/smoke_test.sh /usr/local/bin/smoke_test.sh
RUN chmod +x /usr/local/bin/smoke_test.sh

RUN smoke_test.sh

WORKDIR /workspace

ENV MACA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LABEL org.opencontainers.image.title="Megatron-LM-FL CI (MetaX)"
LABEL org.opencontainers.image.description="CI environment for Megatron-LM-FL on MetaX MACA"
LABEL org.opencontainers.image.vendor="FlagOS"
LABEL com.flagos.megatron.platform="metax"
LABEL com.flagos.megatron.torch_version="${TORCH_VERSION}"
LABEL com.flagos.megatron.vendor_sdk="${VENDOR_SDK}"

FROM ci AS dev

RUN pip install --no-cache-dir \
    ipython==8.28.0 \
    jupyter==1.1.1 \
    pre-commit==4.0.1 \
    ruff==0.7.4 \
    black==24.10.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="Megatron-LM-FL Dev (MetaX)"
LABEL org.opencontainers.image.description="Development environment for Megatron-LM-FL on MetaX MACA"
