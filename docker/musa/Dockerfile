# syntax=docker/dockerfile:1.4
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

# ==============================================================================
# Megatron-LM-FL CI Image - MUSA Platform
# ==============================================================================
# Multi-stage build for Moore Threads MUSA platform
# ==============================================================================

ARG BASE_IMAGE=registry.mthreads.com/mcconline/musa-runtime:4.3.5
ARG PYTHON_VERSION=3.10

# ==============================================================================
# Stage 1: platform-runtime
# ==============================================================================
FROM ${BASE_IMAGE} AS platform-runtime

ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive

# Install build tools and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    cmake \
    ninja-build \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to correct version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ==============================================================================
# Stage 2: te-fl-native
# ==============================================================================
FROM platform-runtime AS te-fl-native

ARG TE_FL_ARTIFACT
ARG TE_FL_COMMIT
ARG TE_FL_REPOSITORY=https://github.com/FlagOpen/TransformerEngine-FL.git
ARG PLATFORM=musa

# Copy installation script
COPY common/install_te_fl.sh /tmp/install_te_fl.sh
RUN chmod +x /tmp/install_te_fl.sh

# Install TE-FL
RUN if [ -z "${TE_FL_ARTIFACT}" ]; then \
        echo "ERROR: TE_FL_ARTIFACT not specified for MUSA platform" >&2; \
        echo "       Build requires pre-built TE-FL artifact" >&2; \
        exit 1; \
    fi && \
    /tmp/install_te_fl.sh \
        --artifact "${TE_FL_ARTIFACT}" \
        --commit "${TE_FL_COMMIT}" \
        --repository "${TE_FL_REPOSITORY}" \
        --platform "${PLATFORM}"

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

# Install optional heavy dependencies
# Note: mamba-ssm and causal-conv1d are CUDA-specific, skip on MUSA
RUN echo "Skipping CUDA-specific optional dependencies on MUSA"

# Copy smoke test
COPY common/smoke_test.sh /usr/local/bin/smoke_test.sh
RUN chmod +x /usr/local/bin/smoke_test.sh

# Verify TE-FL installation
RUN smoke_test.sh

# Set working directory
WORKDIR /workspace

# Environment variables for MUSA
ENV MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV MCCL_SOCKET_IFNAME=eth0

# OCI labels
LABEL org.opencontainers.image.title="Megatron-LM-FL CI (MUSA)"
LABEL org.opencontainers.image.description="CI environment for Megatron-LM-FL on Moore Threads MUSA"
LABEL org.opencontainers.image.vendor="FlagOS"
LABEL com.flagos.megatron.platform="musa"
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

LABEL org.opencontainers.image.title="Megatron-LM-FL Dev (MUSA)"
LABEL org.opencontainers.image.description="Development environment for Megatron-LM-FL on Moore Threads MUSA"
