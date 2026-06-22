from collections import defaultdict
from types import SimpleNamespace

import pytest

from megatron.plugin.hetero import parallel_context


def test_group_names_and_overlap_mapping():
    assert parallel_context.get_group_name("tp") == "tp"
    assert parallel_context.get_group_name("tp", is_expert=True) == "exp_tp"
    assert parallel_context.get_nccl_option_name("dp-cp") == "dp_cp"
    assert parallel_context.get_nccl_option_name("tp", is_expert=True) == "ep_tp"

    with pytest.raises(ValueError, match="Invalid token"):
        parallel_context.get_nccl_option_name("invalid")
    with pytest.raises(ValueError, match="Invalid token"):
        parallel_context.get_nccl_option_name("invalid", is_expert=True)

    assert parallel_context.find_overlapped_mapping(2, 4) == {
        0: [(0, 0, 1), (1, 1, 2)],
        1: [(2, 0, 1), (3, 1, 2)],
    }
    assert parallel_context.find_overlapped_mapping(2, 3, global_size=6) == {
        0: [(0, 0, 2), (1, 2, 3)],
        1: [(1, 0, 1), (2, 1, 3)],
    }


def test_rank_mapper_orders_devices_and_translates_ranks(monkeypatch):
    rank_infos = [
        {"rank": 0, "device_type": "cuda"},
        {"rank": 1, "device_type": "musa"},
        {"rank": 2, "device_type": "cuda"},
        {"rank": 3, "device_type": "musa"},
    ]
    monkeypatch.setattr(parallel_context.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_context.torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(parallel_context.torch.distributed, "get_rank", lambda: 1)

    def all_gather_object(output, current):
        output[:] = rank_infos

    monkeypatch.setattr(parallel_context.torch.distributed, "all_gather_object", all_gather_object)
    mapper = parallel_context.RankMapper(
        SimpleNamespace(
            hetero_device_types=["cuda", "musa"],
            hetero_current_device_type="musa",
        )
    )

    assert mapper.to_physical_ranks([0, 1, 2, 3]) == [0, 2, 1, 3]
    assert mapper.to_logical_ranks([0, 2, 1, 3]) == [0, 1, 2, 3]


def _bare_parallel_context():
    context = parallel_context.ParallelContext.__new__(parallel_context.ParallelContext)
    context._is_initialized = True
    context._rank = 1
    context._current_process_mesh_index = 0
    context._global_parallel_world_sizes = {}
    context._global_parallel_ranks = {}
    context._global_process_groups = defaultdict(list)
    context._global_group_ranks = defaultdict(list)
    context._global_all_group_ranks = defaultdict(list)
    context._global_process_group_to_ranks = {}
    return context


def test_parallel_context_rank_world_size_and_pipeline_stage_accessors(monkeypatch):
    context = _bare_parallel_context()
    context.set_tensor_model_parallel_world_size(2)
    context.set_pipeline_model_parallel_world_size(4)
    context.set_virtual_pipeline_model_parallel_world_size(2)
    context.set_tensor_model_parallel_rank(1)
    context.set_pipeline_model_parallel_rank(0)
    context.set_pipeline_model_parallel_split_rank(2)
    context.set_virtual_pipeline_model_parallel_rank(0)

    assert context.is_initialized() is True
    assert context.get_tensor_model_parallel_world_size() == 2
    assert context.get_pipeline_model_parallel_world_size() == 4
    assert context.get_virtual_pipeline_model_parallel_world_size() == 2
    assert context.get_tensor_model_parallel_rank() == 1
    assert context.get_pipeline_model_parallel_rank() == 0
    assert context.get_pipeline_model_parallel_split_rank() == 2
    assert context.is_pipeline_first_stage() is True
    assert context.is_pipeline_last_stage() is False
    assert context.is_pipeline_stage_before_split(rank=1) is True
    assert context.is_pipeline_stage_after_split(rank=2) is True
    assert context.is_pipeline_stage_at_split() is False

    context.set_virtual_pipeline_model_parallel_rank(1)
    context.set_pipeline_model_parallel_rank(3)
    assert context.is_pipeline_first_stage() is False
    assert context.is_pipeline_last_stage() is True

    context.set_data_parallel_rank(3)
    context.set_expert_model_parallel_world_size(4)
    context.set_expert_model_parallel_rank(2)
    context.set_expert_tensor_parallel_world_size(2)
    context.set_expert_tensor_parallel_rank(1)
    assert context.get_data_parallel_rank() == 3
    assert context.get_expert_model_parallel_world_size() == 4
    assert context.get_expert_model_parallel_rank() == 2
    assert context.get_expert_tensor_parallel_world_size() == 2
    assert context.get_expert_tensor_parallel_rank() == 1


def test_parallel_context_global_group_accessors_and_dp_coefficient():
    context = _bare_parallel_context()
    group = object()
    context._global_process_groups["tp-pp"] = group
    context._global_group_ranks["tp-pp"] = [0, 1]
    context._global_all_group_ranks["tp-pp"] = [[0, 1], [2, 3]]

    assert context.get_global_process_group("tp-pp", check_initialized=True) is group
    assert context.get_global_group_ranks("tp-pp", check_initialized=True) == [0, 1]
    assert context.get_global_all_group_ranks("tp-pp", check_initialized=True) == [
        [0, 1],
        [2, 3],
    ]
    assert context.get_model_parallel_group() is group
    assert context.get_model_parallel_src_rank() == 0

    with pytest.raises(AssertionError, match="not initialized"):
        context.get_global_process_group("missing", check_initialized=True)

    class Mesh:
        def __init__(self, dp):
            self.dp = dp

        def get_parallel_size(self, token, is_expert=False):
            assert token == "dp"
            return self.dp

    context._args = SimpleNamespace(calculate_per_token_loss=False)
    context._process_meshes = [Mesh(2), Mesh(4)]
    assert context.get_dp_coef_when_recv_backward() == pytest.approx(0.5)
    context._args.calculate_per_token_loss = True
    assert context.get_dp_coef_when_recv_backward() == 1.0
