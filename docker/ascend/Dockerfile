# syntax=docker/dockerfile:1.4
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

# ==============================================================================
# Megatron-LM-FL CI Image - Ascend Platform
# ==============================================================================
# Multi-stage build for Huawei Ascend NPU platform
# ==============================================================================

ARG BASE_IMAGE
ARG PYTHON_VERSION=3.10

# ==============================================================================
# Stage 1: platform-runtime
# ==============================================================================
FROM ${BASE_IMAGE} AS platform-runtime

ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive

# Ascend base images usually come with CANN runtime pre-installed
# Only install missing build tools

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is up to date
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ==============================================================================
# Stage 2: te-fl-native
# ==============================================================================
FROM platform-runtime AS te-fl-native

ARG TE_FL_ARTIFACT
ARG TE_FL_COMMIT
ARG TE_FL_REPOSITORY=https://github.com/FlagOpen/TransformerEngine-FL.git
ARG PLATFORM=ascend

# Special handling for Ascend: TE-FL expects /workspace/TransformerEngine-FL
# This is a known contract from existing Ascend setup
ENV TE_FL_SOURCE_DIR=/workspace/TransformerEngine-FL

# Copy installation script
COPY common/install_te_fl.sh /tmp/install_te_fl.sh
RUN chmod +x /tmp/install_te_fl.sh

# Install TE-FL
RUN if [ -z "${TE_FL_ARTIFACT}" ]; then \
        echo "WARNING: TE_FL_ARTIFACT not specified, falling back to source build" >&2; \
        mkdir -p /workspace && \
        cd /workspace && \
        git clone "${TE_FL_REPOSITORY}" TransformerEngine-FL && \
        cd TransformerEngine-FL && \
        git checkout "${TE_FL_COMMIT}" && \
        pip install --no-cache-dir --no-deps . ; \
    else \
        /tmp/install_te_fl.sh \
            --artifact "${TE_FL_ARTIFACT}" \
            --commit "${TE_FL_COMMIT}" \
            --repository "${TE_FL_REPOSITORY}" \
            --platform "${PLATFORM}" ; \
    fi

# Set environment variables for provenance
ENV TE_FL_COMMIT=${TE_FL_COMMIT}
ENV TE_FL_PLATFORM=${PLATFORM}
ENV TE_FL_REPOSITORY=${TE_FL_REPOSITORY}

# OCI labels for TE-FL provenance
LABEL com.flagos.te_fl.commit="${TE_FL_COMMIT}"
LABEL com.flagos.te_fl.platform="${PLATFORM}"
LABEL com.flagos.te_fl.artifact="${TE_FL_ARTIFACT}"

# ==============================================================================
# Stage 3: ci (test environment)
# ==============================================================================
FROM te-fl-native AS ci

ARG TORCH_VERSION
ARG VENDOR_SDK

# Install test dependencies
RUN pip install --no-cache-dir \
    pytest==8.3.3 \
    pytest-cov==5.0.0 \
    pytest-timeout==2.3.1 \
    pytest-metadata==3.1.1 \
    pytest-json-report==1.5.0 \
    nltk==3.9.1 \
    zarr==2.18.3 \
    tensorstore==0.1.67

# Copy smoke test
COPY common/smoke_test.sh /usr/local/bin/smoke_test.sh
RUN chmod +x /usr/local/bin/smoke_test.sh

# Verify TE-FL installation
RUN smoke_test.sh

# Set working directory
WORKDIR /workspace

# Environment variables for Ascend
ENV ASCEND_HOME=/usr/local/Ascend
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${PYTHONPATH}

# OCI labels
LABEL org.opencontainers.image.title="Megatron-LM-FL CI (Ascend)"
LABEL org.opencontainers.image.description="CI environment for Megatron-LM-FL on Huawei Ascend NPU"
LABEL org.opencontainers.image.vendor="FlagOS"
LABEL com.flagos.megatron.platform="ascend"
LABEL com.flagos.megatron.torch_version="${TORCH_VERSION}"
LABEL com.flagos.megatron.vendor_sdk="${VENDOR_SDK}"

# ==============================================================================
# Stage 4: dev (development environment)
# ==============================================================================
FROM ci AS dev

# Install development tools
RUN pip install --no-cache-dir \
    ipython==8.28.0 \
    jupyter==1.1.1 \
    pre-commit==4.0.1 \
    ruff==0.7.4 \
    black==24.10.0

# Additional debug tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="Megatron-LM-FL Dev (Ascend)"
LABEL org.opencontainers.image.description="Development environment for Megatron-LM-FL on Huawei Ascend NPU"
