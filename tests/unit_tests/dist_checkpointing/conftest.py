# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest import mock

import pytest

from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.core.msc_utils import MultiStorageClientFeature


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope='session', autouse=True)
def disable_msc():
    MultiStorageClientFeature.disable()
    yield


@pytest.fixture(scope="class")
def tmp_dir_per_class(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(autouse=True)
def _sync_npu_streams_after_test():
    """Synchronize NPU streams after each test to release HCCL stream resources.

    dist_checkpointing tests call all_gather_object heavily; each call
    creates a short-lived HCCL stream that the Ascend driver may not
    reclaim immediately.  Over many tests the stream pool drains to
    zero, the next rtStreamCreateWithFlags fails, and the rank drops
    out of the collective — the other ranks deadlock.

    A full-device sync between tests lets the driver finish pending
    work and recycle streams, eliminating the leak across test boundaries.
    """
    yield
    try:
        import torch_npu
        torch_npu.npu.synchronize()
    except (ImportError, RuntimeError):
        pass
