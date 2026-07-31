import torch

from megatron.core.tensor_parallel.data import broadcast_data
from megatron.plugin.platform import get_platform
from tests.unit_tests.test_utilities import Utils

cur_platform = get_platform()


def test_broadcast_data():
    Utils.initialize_model_parallel(2, 4)
    try:
        device = cur_platform.device()
        dtype = torch.float32
        input_data = {
            rank: torch.ones((8, 8), dtype=dtype, device=device) * rank
            for rank in range(8)
        }
        actual_output = broadcast_data([0, 1], input_data, dtype)
        assert torch.equal(actual_output[0], input_data[0])
        assert torch.equal(actual_output[1], input_data[1])
    finally:
        Utils.destroy_model_parallel()
