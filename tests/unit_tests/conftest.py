# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import gc
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed

from megatron.core import config
from megatron.core.utils import is_te_min_version
from megatron.plugin.platform import get_platform
from tests.test_utils.python_scripts.download_unit_tests_dataset import download_and_extract_asset
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def pytest_addoption(parser):
    """
    Additional command-line arguments passed to pytest.
    For now:
        --experimental: Enable the Mcore experimental flag (DEFAULT: False)
    """
    parser.addoption(
        '--experimental',
        action='store_true',
        help="pass that argument to enable experimental flag during testing (DEFAULT: False)",
    )


@pytest.fixture(autouse=True)
def experimental(request):
    """Simple fixture setting the experimental flag [CPU | GPU]"""
    config.ENABLE_EXPERIMENTAL = request.config.getoption("--experimental") is True


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope="session", autouse=True)
def bind_local_device():
    """Bind each torchrun worker to its device and isolate compiler caches."""
    local_rank = os.getenv("LOCAL_RANK")
    if local_rank is None:
        return

    for cache_var, default_root in (
        ("TORCHINDUCTOR_CACHE_DIR", "/tmp/.torch_inductor_cache"),
        ("TRITON_CACHE_DIR", "/tmp/.triton_cache"),
    ):
        cache_dir = Path(os.getenv(cache_var, default_root)) / f"rank_{local_rank}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ[cache_var] = str(cache_dir)

    platform = get_platform()
    device_count = platform.device_count()
    if device_count > 0:
        platform.set_device(int(local_rank) % device_count)


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Destroy default process group after all tests complete.

    On Enflame: tensor_parallel skips cleanup (cleanup barrier deadlocks with rank counter
    mismatch 9 vs 10 due to post-test timing race). Other Enflame groups run cleanup
    (fixes dist_checkpointing/distributed teardown deadlocks).

    Follows the pattern from Utils.destroy_model_parallel() (test_utilities.py:106-108):
    synchronize device work, then barrier without timeout (GCU has no timeout kwarg),
    then destroy the default process group even if barrier fails.
    """
    yield
    if not torch.distributed.is_initialized():
        return

    platform = get_platform()

    # Enflame: skip cleanup for tensor_parallel (session-end barrier deadlocks)
    if platform.device_name() == 'enflame':
        # Check if any collected test is from tensor_parallel
        session = request.session
        test_paths = [str(item.fspath) for item in session.items]
        if any('tensor_parallel' in path for path in test_paths):
            return  # Skip cleanup, rely on atexit destructor

        # Other Enflame groups: run cleanup (fixes dist_checkpointing/distributed)
        try:
            platform.synchronize()
            torch.distributed.barrier()
        except Exception:
            pass
        torch.distributed.destroy_process_group()
        return

    # Non-Enflame: upstream pattern (barrier with timeout → TypeError → early return)
    try:
        torch.distributed.barrier(timeout=timedelta(seconds=300))
    except Exception:
        return
    torch.distributed.destroy_process_group()


@pytest.fixture(scope="function", autouse=True)
def set_env():
    if is_te_min_version("1.3"):
        os.environ['NVTE_FLASH_ATTN'] = '0'
        os.environ['NVTE_FUSED_ATTN'] = '0'


@pytest.fixture(scope="session")
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.

    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """

    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'

    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
            yield tmp_dir

    else:
        yield tmp_dir


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Ensure test data is available at /opt/data by downloading if necessary."""
    data_path = Path("/opt/data")
    ready_file_name = (
        f"megatron_unit_test_data_ready_{os.getenv('MASTER_PORT', 'single')}_"
        f"{os.getenv('WORLD_SIZE', '1')}"
    )
    ready_file = Path("/tmp") / ready_file_name
    rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    def data_available():
        return data_path.exists() and any(
            path.name != ready_file.name for path in data_path.iterdir()
        )

    def ensure_data_on_rank_zero():
        # Check if data directory exists and has content
        if not data_available():
            print("Test data not found at /opt/data. Downloading...")

            try:
                # Download assets to /opt/data
                download_and_extract_asset(assets_dir=data_path)

                print("Test data downloaded successfully.")

            except ImportError as e:
                print(f"Failed to import download function: {e}")
                # Don't fail the tests, just warn
            except Exception as e:
                print(f"Failed to download test data: {e}")
                # Don't fail the tests, just warn
        else:
            print("Test data already available at /opt/data")

        try:
            data_path.mkdir(parents=True, exist_ok=True)
            ready_file.touch()
        except Exception as e:
            print(f"Failed to mark test data readiness: {e}")

    if world_size <= 1 or rank == 0:
        ensure_data_on_rank_zero()
        return

    # In torchrun jobs, every rank executes pytest collection/setup. If nonzero ranks race
    # ahead while rank 0 is still downloading data, distributed tests can time out in NCCL
    # setup. Wait on a simple file sentinel before entering test setup.
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if ready_file.exists() or data_available():
            return
        time.sleep(1)
    print("Timed out waiting for rank 0 to prepare /opt/data; continuing without downloaded data.")


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables"""
    # Store the original environment variables before the test
    original_env = dict(os.environ)

    # Run the test
    yield

    # After the test, restore the original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Clean up GPU memory after each test to prevent OOM in CI.

    Platform-specific behavior:
    - metax: skip gc.collect() (can abort inside cyclic GC during multi-rank teardown)
    - enflame: skip entirely (empty_cache can deadlock under ECCL)
    """
    yield

    platform = get_platform()
    device_name = platform.device_name()

    # Enflame: skip cleanup entirely due to ECCL empty_cache instability
    if device_name == 'enflame':
        return

    # Metax: skip gc.collect() but still run empty_cache
    if device_name != 'metax':
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

