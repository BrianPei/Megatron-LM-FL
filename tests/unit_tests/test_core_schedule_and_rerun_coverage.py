# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common import model_chunk_schedule_plan as schedule_plan
from megatron.core.models.gpt import fine_grained_callables as fine_callables
from megatron.core import rerun_state_machine as rerun
from megatron.core import timers
from megatron.core.inference import inference_request as inference_request_module
from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.dist_checkpointing import tensor_aware_state_dict as tensor_aware
from megatron.core.transformer import utils as transformer_utils
from megatron.core.resharding import utils as reshard_utils


class _FakeEvent:
    def __init__(self, calls):
        self.calls = calls

    def record(self, stream):
        self.calls.append(("event-record", stream))

    def wait(self, stream):
        self.calls.append(("event-wait", stream))


class _FakePlatform:
    def __init__(self, calls):
        self.calls = calls

    def current_stream(self):
        return "current-stream"

    def current_device(self):
        return 0

    def stream(self, stream):
        self.calls.append(("stream", stream))
        return nullcontext()

    def range_push(self, name):
        self.calls.append(("range-push", name))

    def range_pop(self):
        self.calls.append(("range-pop",))

    def Event(self):
        return _FakeEvent(self.calls)


class _ScheduleNode:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.model_chunk_state = SimpleNamespace()

    def forward(self, value=None):
        self.calls.append((self.name, "forward", value))
        return f"{self.name}.forward({value})"

    def backward(self, value=None):
        self.calls.append((self.name, "backward", value))
        return f"{self.name}.backward({value})"

    def backward_dw(self):
        self.calls.append((self.name, "backward_dw"))


class _LayerPlan:
    def __init__(self, prefix, calls, early_release=False):
        self.prefix = prefix
        self.calls = calls
        self.config = SimpleNamespace(ep_overlap_early_attn_memory_release=early_release)
        self.attn = _ScheduleNode(f"{prefix}.attn", calls)
        self.moe_dispatch = _ScheduleNode(f"{prefix}.dispatch", calls)
        self.mlp = _ScheduleNode(f"{prefix}.mlp", calls)
        self.moe_combine = _ScheduleNode(f"{prefix}.combine", calls)
        self.mtp_post_process = _ScheduleNode(f"{prefix}.mtp", calls)

    def get_fp8_context(self):
        self.calls.append((self.prefix, "fp8-context"))
        return nullcontext()

    def release_state(self):
        self.calls.append((self.prefix, "release"))


class _ChunkPlan:
    def __init__(self, prefix, calls, layers, post_process=True):
        self.prefix = prefix
        self.calls = calls
        self.vp_stage = prefix
        self.pre_process = _ScheduleNode(f"{prefix}.pre", calls)
        self.post_process = _ScheduleNode(f"{prefix}.post", calls) if post_process else None
        self._transformer_layers = list(layers)
        self._model_chunk_state = SimpleNamespace(model=SimpleNamespace(name=prefix))
        self._event = _FakeEvent(calls)

    def __bool__(self):
        return True

    def record_current_stream(self):
        self.calls.append((self.prefix, "record"))

    def wait_current_stream(self):
        self.calls.append((self.prefix, "wait"))

    def get_layer(self, index):
        return self._transformer_layers[index]

    def pop_layer(self):
        return self._transformer_layers.pop()

    def num_layers(self):
        return len(self._transformer_layers)

    def release_state(self):
        self.calls.append((self.prefix, "release_state"))
        self._model_chunk_state.model = None
        self.pre_process.model_chunk_state = None
        if self.post_process is not None:
            self.post_process.model_chunk_state = None


def test_transformer_layer_schedule_run_covers_overlap_orders():
    calls = []
    forward = _LayerPlan("forward", calls)
    backward = _LayerPlan("backward", calls)

    f_output, b_output = schedule_plan.TransformerLayerSchedulePlan.run(
        forward,
        backward,
        f_input="hidden",
        b_grad="grad",
        is_last_layer_in_bwd=False,
    )

    assert f_output == "forward.mtp.forward(forward.combine.forward(forward.mlp.forward(forward.dispatch.forward(forward.attn.forward(hidden)))))"
    assert b_output == "backward.attn.backward(backward.dispatch.backward(backward.mlp.backward(backward.combine.backward(backward.mtp.backward(grad)))))"
    assert calls[:4] == [
        ("backward.mtp", "backward", "grad"),
        ("backward.combine", "backward", "backward.mtp.backward(grad)"),
        ("forward", "fp8-context"),
        ("forward.attn", "forward", "hidden"),
    ]
    assert ("backward.mlp", "backward_dw") in calls
    assert ("backward.attn", "backward_dw") in calls

    early_calls = []
    early = _LayerPlan("early", early_calls, early_release=True)
    _, early_grad = schedule_plan.TransformerLayerSchedulePlan.run(
        None,
        early,
        b_grad="grad",
        is_last_layer_in_bwd=True,
    )

    assert early_grad.endswith("early.attn.backward(early.dispatch.backward(early.mlp.backward(early.combine.backward(early.mtp.backward(grad)))))")
    assert ("early.attn", "backward_dw") not in early_calls
    attn_backward_index = early_calls.index(
        (
            "early.attn",
            "backward",
            "early.dispatch.backward(early.mlp.backward(early.combine.backward(early.mtp.backward(grad))))",
        )
    )
    assert attn_backward_index < len(early_calls)


def test_transformer_model_chunk_schedule_run_and_manual_plan_methods(monkeypatch):
    calls = []
    monkeypatch.setattr(schedule_plan, "cur_platform", _FakePlatform(calls))
    monkeypatch.setattr(schedule_plan, "get_comm_stream", lambda: "comm-stream")

    forward_layers = [_LayerPlan("f0", calls), _LayerPlan("f1", calls), _LayerPlan("f2", calls)]
    backward_layers = [_LayerPlan("b0", calls), _LayerPlan("b1", calls)]
    forward_plan = _ChunkPlan("forward-chunk", calls, forward_layers)
    backward_plan = _ChunkPlan("backward-chunk", calls, backward_layers)

    result = schedule_plan.TransformerModelChunkSchedulePlan.run(
        forward_plan,
        backward_plan,
        b_grad="loss-grad",
        pre_forward=lambda vp: calls.append(("pre-forward", vp)),
        pre_backward=lambda vp: calls.append(("pre-backward", vp)),
        post_forward=lambda value, vp: calls.append(("post-forward", value, vp)),
        post_backward=lambda value, vp: calls.append(("post-backward", value, vp)),
    )

    assert result.startswith("forward-chunk.post.forward(")
    assert ("pre-forward", "forward-chunk") in calls
    assert ("pre-backward", "backward-chunk") in calls
    assert any(call[0] == "post-forward" for call in calls)
    assert any(call[0] == "post-backward" for call in calls)
    assert ("b1", "release") in calls
    assert ("b0", "release") in calls
    assert ("backward-chunk", "release_state") in calls
    assert calls.count(("range-pop",)) == 3

    plan = object.__new__(schedule_plan.TransformerModelChunkSchedulePlan)
    plan._model_chunk_state = SimpleNamespace(model=SimpleNamespace())
    plan._transformer_layers = ["layer-0", "layer-1"]
    plan._event = _FakeEvent(calls)
    plan.pre_process = SimpleNamespace(model_chunk_state=plan._model_chunk_state)
    plan.post_process = SimpleNamespace(model_chunk_state=plan._model_chunk_state)

    assert plan.event is plan._event
    assert plan.state is plan._model_chunk_state
    assert plan.num_layers() == 2
    assert plan.get_layer(1) == "layer-1"
    assert plan.pop_layer() == "layer-1"
    plan.record_current_stream()
    plan.wait_current_stream()
    plan.release_state()

    assert plan._model_chunk_state.model is None
    assert plan.pre_process is None
    assert plan.post_process is None


def test_fine_grained_callable_helpers_and_nodes(monkeypatch):
    calls = []
    config = SimpleNamespace(
        fp8=None,
        fp4=None,
        moe_token_dispatcher_type="alltoall",
        moe_flex_dispatcher_backend=None,
        cuda_graph_scope=[],
    )

    assert fine_callables.should_free_input("mlp", False, config, None) is False
    assert fine_callables.should_free_input("mlp", True, config, 1) is False
    assert fine_callables.should_free_input("moe_dispatch", True, config, 1) is True
    assert fine_callables.should_free_input("moe_combine", True, config, 1) is True
    assert fine_callables.should_free_input("unknown", True, config, 1) is False

    config.fp8 = "enabled"
    assert fine_callables.should_free_input("mlp", True, config, 1) is True
    config.fp8 = None
    config.moe_token_dispatcher_type = "flex"
    config.moe_flex_dispatcher_backend = "deepep"
    assert fine_callables.should_free_input("moe_dispatch", True, config, 2) is False
    config.moe_flex_dispatcher_backend = "hybridep"
    assert fine_callables.should_free_input("mlp", True, config, 2) is False

    class _Owner:
        def method(self, value):
            return f"wrapped-{value}"

    owner = _Owner()
    assert fine_callables.weak_method(owner.method)("value") == "wrapped-value"

    chunk_state = SimpleNamespace(
        input_ids="ids",
        position_ids="positions",
        decoder_input="decoder",
        packed_seq_params="packed",
        padding_mask="padding",
        labels="labels",
        loss_mask="loss-mask",
        attention_mask="attention",
        runtime_gather_output=True,
        extra_block_kwargs={"extra": "kwarg"},
    )

    gpt_model = SimpleNamespace(
        pre_process=False,
        decoder=SimpleNamespace(
            input_tensor="decoder-from-model",
            layers=[object()],
            final_layernorm=None,
        ),
        config=SimpleNamespace(mtp_num_layers=0),
    )

    def preprocess(**kwargs):
        calls.append(("preprocess", kwargs))
        return ("decoder-out", "rotary", "cos", "sin", "offset", "padding-out")

    def postprocess(**kwargs):
        calls.append(("postprocess", kwargs))
        return "loss"

    gpt_model._preprocess = preprocess
    gpt_model._postprocess = postprocess
    monkeypatch.setattr(fine_callables, "float16_to_fp32", lambda value: f"fp32-{value}")

    pre_node = fine_callables.PreProcessNode(gpt_model, chunk_state, _FakeEvent(calls), lambda: "stream")
    assert pre_node.forward_impl() == "decoder-out"
    assert chunk_state.decoder_input == "decoder-out"
    assert chunk_state.rotary_pos_emb == "rotary"
    assert chunk_state.sequence_len_offset == "offset"

    post_node = fine_callables.PostProcessNode(gpt_model, chunk_state, _FakeEvent(calls), lambda: "stream")
    assert post_node.forward_impl("hidden") == "fp32-loss"
    post_kwargs = calls[-1][1]
    assert post_kwargs["hidden_states"] == "hidden"
    assert post_kwargs["decoder_input"] == "decoder-out"
    assert post_kwargs["runtime_gather_output"] is True


def test_transformer_layer_node_backward_and_dw_without_real_autograd(monkeypatch):
    calls = []
    monkeypatch.setattr(fine_callables, "cur_platform", _FakePlatform(calls))
    monkeypatch.setattr(fine_callables, "make_viewless", lambda value: value)

    config = SimpleNamespace(
        fp8=None,
        fp4=None,
        moe_token_dispatcher_type="alltoall",
        moe_flex_dispatcher_backend=None,
        cuda_graph_scope=[],
        delay_wgrad_compute=True,
    )
    layer_state = SimpleNamespace()
    chunk_state = SimpleNamespace()

    class _BackwardDW:
        def backward_dw(self):
            calls.append(("module", "backward_dw"))

    node = fine_callables.TransformerLayerNode(
        stream=lambda: "node-stream",
        event=_FakeEvent(calls),
        layer_state=layer_state,
        chunk_state=chunk_state,
        submodule=lambda this_node, value: f"forward-{value}",
        name="mlp",
        bwd_dw_callables=_BackwardDW(),
        extra_args={"config": config, "delay_wgrad_compute": True},
    )

    assert node.forward_impl("input") == "forward-input"
    node.detached = (SimpleNamespace(grad="detached-grad"),)
    node.before_detached = ("before-detached",)

    def fake_backward(outputs, grads):
        calls.append(("default-backward", outputs, grads))
        return grads

    node.default_backward_func = fake_backward
    grads = node.backward_impl(("output",), ("output-grad",))
    assert grads == ("output-grad", "detached-grad")
    assert node.output_grads == ("output-grad", "detached-grad")
    assert node.delay_grads_release is True

    node.backward_dw()
    assert ("module", "backward_dw") in calls
    assert node.output_grads is None
    assert node.bwd_dw_callables is None

    no_delay = fine_callables.TransformerLayerNode(
        stream=lambda: "node-stream",
        event=_FakeEvent(calls),
        layer_state=layer_state,
        chunk_state=chunk_state,
        submodule=lambda this_node, value: value,
        name="attn",
        extra_args={"config": config, "delay_wgrad_compute": False},
    )
    assert no_delay.backward_dw() is None


def test_timers_start_stop_elapsed_and_log_level_filtering(monkeypatch):
    calls = []
    time_values = iter([0.0, 10.0, 11.5, 20.0, 22.0, 30.0, 31.0])
    monkeypatch.setattr(timers.time, "time", lambda: next(time_values))
    monkeypatch.setattr(timers, "cur_platform", SimpleNamespace(synchronize=lambda: calls.append("sync")))

    timer = timers.Timer("unit")
    timer.start()
    timer.stop()
    assert timer.elapsed(reset=False) == 1.5
    assert timer.active_time() == 1.5
    timer.set_elapsed(4.0)
    assert timer.elapsed(reset=True) == 4.0
    assert timer.elapsed(reset=False) == 0.0

    timer.start()
    assert timer.elapsed(reset=True) == 2.0
    assert timer._started is True
    timer.stop()
    assert timer.active_time() == 4.5
    assert calls.count("sync") == 6

    group = timers.Timers(log_level=1, log_option="minmax")
    enabled = group("enabled", log_level=1)
    assert isinstance(enabled, timers.Timer)
    assert group("enabled", log_level=1) is enabled
    with pytest.raises(AssertionError, match="does not match"):
        group("enabled", log_level=2)

    dummy = group("too-detailed", log_level=2)
    assert isinstance(dummy, timers.DummyTimer)
    with pytest.raises(Exception, match="dummy timer"):
        dummy.elapsed()
    with pytest.raises(Exception, match="active timer"):
        dummy.active_time()

    with pytest.raises(AssertionError, match="invalid"):
        timers.Timers(log_level=1, log_option="average")
    with pytest.raises(AssertionError, match="larger"):
        group("unsupported", log_level=3)


def test_resharding_metadata_name_resolution_and_balanced_selection():
    assert reshard_utils._get_rank_in_group(3, [1, 3, 5]) == 1
    with pytest.raises(ValueError, match="not found"):
        reshard_utils._get_rank_in_group(4, [1, 3, 5])

    assert reshard_utils._detect_expert_index_from_param_name("experts.weight12") == 12
    assert reshard_utils._detect_expert_index_from_param_name("experts.bias7") == 7
    assert reshard_utils._detect_expert_index_from_param_name("experts.weight") is None

    non_ep = reshard_utils.ParameterMetadata(
        name="decoder.layers.0.mlp.weight",
        shape=(4, 4),
        dtype="bf16",
        element_size=2,
        owner_rank=0,
    )
    reshard_utils.assign_ep_resolved_name_inplace(non_ep)
    assert non_ep.resolved_name == "decoder.layers.0.mlp.weight"
    assert non_ep.global_expert_index is None

    ep_meta = reshard_utils.ParameterMetadata(
        name="decoder.layers.0.experts.weight1",
        shape=(4, 4),
        dtype="bf16",
        element_size=2,
        is_ep=True,
        num_experts=8,
        owner_rank=3,
        expert_parallel_group_ranks=[2, 3],
    )
    reshard_utils.assign_ep_resolved_name_inplace(ep_meta)
    assert ep_meta.global_expert_index == 5
    assert ep_meta.resolved_name == "decoder.layers.0.experts.weight5"

    prefix_map = {"decoder.layers.0": "decoder.layers.8"}
    reshard_utils.assign_resolved_name_inplace(
        ep_meta,
        layer_module_prefix_map=prefix_map,
        base_name="decoder.layers.0.experts.bias0",
    )
    assert ep_meta.global_expert_index == 4
    assert ep_meta.resolved_name == "decoder.layers.8.experts.bias4"
    assert (
        reshard_utils._resolve_global_layer_number_in_name(
            "decoder.layers.0.self_attention.weight", prefix_map
        )
        == "decoder.layers.8.self_attention.weight"
    )
    assert reshard_utils._resolve_global_layer_number_in_name("embedding.weight", prefix_map) == "embedding.weight"

    class _Module:
        def named_modules(self):
            return [
                ("", self),
                ("decoder.layers.0", SimpleNamespace(layer_number=9)),
                ("decoder.layers.0.self_attention", SimpleNamespace(layer_number=None)),
                ("decoder.layers.final", SimpleNamespace(layer_number=10)),
            ]

    assert reshard_utils._build_layer_module_prefix_map(_Module()) == {
        "decoder.layers.0": "decoder.layers.8"
    }

    src = [
        reshard_utils.ParameterMetadata(
            name="w",
            shape=(1,),
            dtype="bf16",
            element_size=2,
            owner_rank=0,
            data_parallel_group_ranks=[0, 1],
        ),
        reshard_utils.ParameterMetadata(
            name="w",
            shape=(1,),
            dtype="bf16",
            element_size=2,
            owner_rank=2,
            data_parallel_group_ranks=[2, 3],
        ),
        reshard_utils.ParameterMetadata(
            name="w",
            shape=(1,),
            dtype="bf16",
            element_size=2,
            owner_rank=3,
            data_parallel_group_ranks=[2, 3],
        ),
    ]
    dst = reshard_utils.ParameterMetadata(
        name="w",
        shape=(1,),
        dtype="bf16",
        element_size=2,
        owner_rank=99,
        data_parallel_group_ranks=[99],
    )
    assert reshard_utils.select_src_metadata_balanced(src, dst, dst_rank=3).owner_rank == 3
    assert reshard_utils.select_src_metadata_balanced(src, dst, dst_rank=0).owner_rank == 0

    ep_src = [
        reshard_utils.ParameterMetadata(
            name="w",
            shape=(1,),
            dtype="bf16",
            element_size=2,
            owner_rank=0,
            expert_parallel_group_ranks=[0, 1],
            data_parallel_group_ranks=[0],
        ),
        reshard_utils.ParameterMetadata(
            name="w",
            shape=(1,),
            dtype="bf16",
            element_size=2,
            owner_rank=1,
            expert_parallel_group_ranks=[0, 1],
            data_parallel_group_ranks=[1],
        ),
    ]
    ep_dst = reshard_utils.ParameterMetadata(
        name="w",
        shape=(1,),
        dtype="bf16",
        element_size=2,
        owner_rank=9,
        expert_parallel_group_ranks=[8, 9],
        data_parallel_group_ranks=[9],
    )
    assert reshard_utils.select_src_metadata_balanced(ep_src, ep_dst, dst_rank=9).owner_rank == 1

    bad_dst = reshard_utils.ParameterMetadata(
        name="w",
        shape=(1,),
        dtype="bf16",
        element_size=2,
        owner_rank=10,
        expert_parallel_group_ranks=[8, 9, 10],
        data_parallel_group_ranks=[10],
    )
    assert reshard_utils.select_src_metadata_balanced(ep_src, bad_dst, dst_rank=10).owner_rank in {0, 1}
    with pytest.raises(ValueError, match="non-empty"):
        reshard_utils.select_src_metadata_balanced([], dst, dst_rank=0)


class _ModuleTree:
    def __init__(self, **attrs):
        self._modules = {}
        for key, value in attrs.items():
            setattr(self, key, value)

    def add(self, name, child):
        self._modules[name] = child
        return child

    def modules(self):
        yield self
        for child in self._modules.values():
            yield from child.modules()


def test_transformer_utils_cpu_helpers_and_attribute_caches(monkeypatch):
    scores = torch.zeros(2, 2)
    mask = torch.tensor([[False, True], [False, False]])
    masked = transformer_utils.attention_mask_func(scores, mask)
    assert masked[0, 1].item() == -10000.0
    assert masked[1, 1].item() == 0.0

    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(transformer_utils.openai_gelu(x), transformer_utils.gelu_impl(x))
    assert transformer_utils.erf_gelu(x).shape == x.shape

    linear = transformer_utils.get_linear_layer(
        2,
        3,
        init_method=lambda weight: weight.data.fill_(2.0),
        perform_initialization=True,
    )
    assert torch.all(linear.weight == 2.0)
    assert torch.all(linear.bias == 0.0)

    assert transformer_utils._get_extra_state_offsets(()) == ((1,), (0,))
    assert transformer_utils._get_extra_state_offsets(((1, 4, 8), (0, 2, 6))) == ((6, 8), (2, 4))
    with pytest.raises(AssertionError, match="contiguous"):
        transformer_utils._get_extra_state_offsets(((2, 4, 8),))

    monkeypatch.setattr(
        transformer_utils.parallel_state,
        "get_data_parallel_group",
        lambda with_context_parallel=True: "dp-cp-group",
    )
    assert transformer_utils.ensure_metadata_has_dp_cp_group(None) == {"dp_cp_group": "dp-cp-group"}
    metadata = {}
    assert transformer_utils.ensure_metadata_has_dp_cp_group(metadata) is metadata
    assert metadata["dp_cp_group"] == "dp-cp-group"
    with pytest.raises(AssertionError, match="metadata must be"):
        transformer_utils.ensure_metadata_has_dp_cp_group("bad")

    transformer_utils._sequence_parallel_attr_cache = None
    model = _ModuleTree(
        config=SimpleNamespace(sequence_parallel=False),
        position_embedding_type="rope",
    )
    child = model.add(
        "child",
        _ModuleTree(
            sequence_parallel=False,
            scatter_to_sequence_parallel=False,
            reduce_scatter_embeddings=False,
        ),
    )
    excluded = model.add("excluded", _ModuleTree(sequence_parallel=False))
    transformer_utils.set_model_to_sequence_parallel(model, set_to=True, exclude_modules=[excluded])
    assert model.config.sequence_parallel is True
    assert child.sequence_parallel is True
    assert child.scatter_to_sequence_parallel is True
    assert child.reduce_scatter_embeddings is True
    assert excluded.sequence_parallel is False

    transformer_utils.set_model_to_sequence_parallel(model, set_to=False, exclude_modules=[excluded])
    assert child.sequence_parallel is False

    transformer_utils.cuda_graph_attr_cache = None
    graph_model = _ModuleTree(
        config=SimpleNamespace(cuda_graph_impl="none", recompute_granularity="full"),
    )
    graph_child = graph_model.add(
        "graph_child",
        _ModuleTree(
            cuda_graph_impl=False,
            flash_decode=False,
            cudagraph_manager="cached-manager",
            recompute_granularity="selective",
            config=SimpleNamespace(cuda_graph_impl="none", flash_decode=False),
        ),
    )
    transformer_utils.toggle_cuda_graphs(graph_model, set_to="local")
    assert graph_model.config.cuda_graph_impl == "local"
    assert graph_child.cuda_graph_impl == "local"
    assert graph_child.config.cuda_graph_impl == "local"
    assert graph_child.recompute_granularity is None
    assert graph_child.cudagraph_manager == "cached-manager"

    transformer_utils.toggle_cuda_graphs(graph_model, set_to="none")
    assert graph_child.cuda_graph_impl == "none"
    assert graph_child.config.cuda_graph_impl == "none"
    assert graph_child.recompute_granularity == "selective"
    assert not hasattr(graph_child, "cudagraph_manager")
    with pytest.raises(AssertionError, match="Invalid CUDA graph"):
        transformer_utils.toggle_cuda_graphs(graph_model, set_to="global")

    assert transformer_utils.is_layer_window_attention(None, None, 1) is False
    assert transformer_utils.is_layer_window_attention((4, 4), None, 1) is True
    assert transformer_utils.is_layer_window_attention((4, 4), 2, 2) is False
    assert transformer_utils.is_layer_window_attention((4, 4), 2, 3) is True
    assert transformer_utils.is_layer_window_attention((4, 4), [False, True], 2) is True
    with pytest.raises(ValueError, match="Invalid"):
        transformer_utils.is_layer_window_attention((4, 4), "bad", 1)


def test_dynamic_inference_request_record_checkpoint_merge_and_serialization(monkeypatch):
    nvtx_calls = []
    fake_nvtx = SimpleNamespace(
        range_push=lambda name: nvtx_calls.append(("push", name)),
        range_pop=lambda: nvtx_calls.append(("pop", None)),
    )
    monkeypatch.setattr(inference_request_module.torch.cuda, "nvtx", fake_nvtx, raising=False)
    monkeypatch.setattr(inference_request_module.torch.distributed, "is_initialized", lambda: False)

    request = DynamicInferenceRequest(
        request_id=11,
        prompt="hello",
        prompt_tokens=torch.tensor([1, 2, 3, 4]),
        sampling_params=SamplingParams(num_tokens_to_generate=5, top_k=3),
        block_size_tokens=2,
        enable_prefix_caching=True,
        status=Status.ACTIVE_AND_GENERATING_TOKENS,
    )
    assert request.remaining_prompt_length == 4
    assert len(request.precomputed_block_hashes) == 2
    assert "id 11" in str(request)

    with pytest.warns(UserWarning, match="Defaulting to -1"):
        metadata = request.tracked_metadata
    assert metadata[3] == -1
    assert request.sampling_params.termination_id == -1

    request.add_event_add_engine()
    request.add_event_add_context()
    generated_event = request.add_event_generated_token(
        9,
        blocks_total=8,
        blocks_hashed_total=4,
        blocks_hashed_active=2,
        blocks_ref_count=3,
        pre_fwd_active_token_count=6,
        pre_fwd_step_count=7,
    )
    request.add_event_pause()
    request.add_event_evict()
    request.add_event_finish()
    request.add_event_fail()
    assert generated_event.payload["token_id"] == 9
    assert generated_event.payload["pre_fwd_step_count"] == 7

    request.generated_tokens = [5, 6]
    request.routing_indices = torch.zeros(5, 1, 1, dtype=torch.int64)
    serialized = request.serialize()
    restored = DynamicInferenceRequest.deserialize(serialized)
    assert restored.events[2].payload["blocks_ref_count"] == 3
    assert torch.equal(restored.routing_indices, request.routing_indices)
    assert any(call == ("push", "DynamicInferenceRequest.serialize") for call in nvtx_calls)

    request.routing_indices = torch.zeros(1, 1, 1, dtype=torch.int64)
    with pytest.raises(AssertionError, match="routing_indices first dimension"):
        request.serialize()

    request.routing_indices = torch.zeros(5, 1, 1, dtype=torch.int64)
    request.generated_text = "alpha"
    request.generated_log_probs = [0.1, 0.2]
    request.generated_top_n_logprobs = [{"a": -1.0}, {"b": -2.0}]
    request.tpot = [10, 11]
    request.status = Status.COMPLETED
    assert request.succeeded() is True
    request.status = Status.FAILED
    assert request.failed() is True
    request.status = Status.COMPLETED

    record = DynamicInferenceRequestRecord.from_request(request)
    assert record.request_id == 11
    record.checkpoint()
    checkpointed = record[-1]
    assert torch.equal(checkpointed.prompt_tokens, torch.tensor([1, 2, 3, 4, 5, 6]))
    assert checkpointed.sampling_params.num_tokens_to_generate == 3
    assert checkpointed.event_add_engine is request.event_add_engine

    checkpointed.generated_tokens = [7]
    checkpointed.generated_text = "beta"
    checkpointed.generated_log_probs = [0.3]
    checkpointed.generated_top_n_logprobs = [{"c": -3.0}]
    checkpointed.tpot = [12]
    checkpointed.events = [checkpointed.add_event_finish()]
    checkpointed.status = Status.COMPLETED
    checkpointed.policy_epoch = [(1, 2)]
    checkpointed.kv_cache_epoch = [(3, 4)]
    checkpointed.routing_indices = torch.zeros(6, 1, 1, dtype=torch.int64)
    record.latency = 1.25

    merged = record.merge()
    assert merged.generated_tokens == [5, 6, 7]
    assert merged.generated_text == "alphabeta"
    assert merged.generated_length == 3
    assert merged.latency == 1.25
    assert merged.policy_epoch == [(1, 2)]
    assert merged.kv_cache_epoch == [(3, 4)]
    assert len(merged.events) == len(request.events) + 1

    serialized_record = record.serialize()
    restored_record = DynamicInferenceRequestRecord.deserialize(serialized_record)
    assert restored_record.request_id == 11
    assert len(restored_record.requests) == 2


def test_tensor_aware_state_dict_tensor_lifecycle(monkeypatch):
    class _FakeShardedTensor:
        def __init__(self, key, data):
            self.key = key
            self.data = data
            self.init_calls = []

        def init_data(self, device):
            self.init_calls.append(device)
            self.data = torch.empty(2, device=device)

    class _FakeShardedObject:
        def __init__(self, key, data):
            self.key = key
            self.data = data

    monkeypatch.setattr(tensor_aware, "ShardedTensor", _FakeShardedTensor)
    monkeypatch.setattr(tensor_aware, "ShardedObject", _FakeShardedObject)

    tensor_a = torch.tensor([1.0, 2.0])
    tensor_b = torch.tensor([3.0, 4.0])
    sharded_a = _FakeShardedTensor("a", tensor_a)
    sharded_b = _FakeShardedTensor("b", tensor_b)
    state = tensor_aware.MCoreTensorAwareStateDict(
        common={"metadata": {"step": 1}},
        sharded_state_dict={"layer": [sharded_a, {"nested": sharded_b}]},
    )

    assert state.is_hollow is False
    assert state.common_state_dict["metadata"]["step"] == 1
    tensors = list(state.tensors)
    assert tensors[0] is tensor_a
    assert tensors[1] is tensor_b

    popped = state.pop_tensors()
    assert popped[0] is tensor_a
    assert popped[1] is tensor_b
    assert state.is_hollow is True
    assert sharded_a.data is None
    assert sharded_a.orig_device == "cpu"
    assert list(state._sharded_tensors) == [sharded_a, sharded_b]

    state.insert_tensors([tensor_a, tensor_b])
    assert state.is_hollow is False
    assert sharded_a.data is tensor_a
    assert not hasattr(sharded_a, "orig_device")

    state.pop_tensors()
    state.init_tensors()
    assert state.is_hollow is False
    assert sharded_a.init_calls == ["cpu"]
    assert sharded_b.init_calls == ["cpu"]

    old_data = sharded_a.data
    state.copy_tensors_to_cpu(non_blocking=True)
    assert sharded_a.data.device.type == "cpu"
    assert sharded_a.data is not old_data
    state.restore_tensor_device(non_blocking=False)
    assert sharded_a.data.device.type == "cpu"

    tensor_aware.MCoreTensorAwareStateDict._validate_params("atomic")
    tensor_aware.MCoreTensorAwareStateDict._validate_params("fully_parallel")
    with pytest.raises(NotImplementedError, match="Only"):
        tensor_aware.MCoreTensorAwareStateDict._validate_params("other")

    cached_distribution = ({"a": 0}, {"b": 1}, "unused-a", "unused-b")
    assert (
        tensor_aware.MCoreTensorAwareStateDict._get_distribution(
            False, state.sharded_state_dict, None
        )
        == (None, None, None, None)
    )
    assert (
        tensor_aware.MCoreTensorAwareStateDict._get_distribution(
            True, state.sharded_state_dict, None, cached_distribution
        )
        is cached_distribution
    )


def test_rerun_quick_stats_data_iterator_and_error_injector(monkeypatch):
    stats = rerun.QuickStats(max_size=3)
    stats.record(0.0)
    stats.record(1.0)
    stats.record(2.0)
    stats.record(3.0)
    stats.record(4.0)
    other = rerun.QuickStats()
    other.record(5.0)
    other.record(0.0)
    stats.combine([other])

    printed = stats.print_stats()
    assert "total/identical samples" in printed
    assert stats.max == 5.0
    stats.reset()
    assert stats.print_stats() == "0 samples, all identical"

    machine = SimpleNamespace(get_mode=lambda: rerun.RerunMode.VALIDATE_RESULTS)
    monkeypatch.setattr(rerun, "get_rerun_state_machine", lambda: machine)
    iterator = rerun.RerunDataIterator(iter(["a", "b", "c"]))
    assert next(iterator) == "a"
    assert next(iterator) == "b"
    assert iterator.saved_microbatches == ["a", "b"]
    iterator.rewind()
    assert next(iterator) == "a"
    saved = iterator.state_dict()
    iterator.advance()
    assert iterator.saved_microbatches == []
    iterator.load_state_dict(saved)
    assert iterator.replaying is True
    assert iterator.replay_pos == 1

    monkeypatch.setattr(rerun.random, "randint", lambda start, stop: 0)
    monkeypatch.setattr(rerun, "safe_get_rank", lambda: 0)
    injector = rerun.RerunErrorInjector(
        error_injection_rate=1,
        error_injection_type=rerun.RerunDiagnostic.TRANSIENT_ERROR,
    )
    assert injector.maybe_inject() is True
    assert injector.maybe_inject() is False
    assert (
        injector.maybe_miscompare(lambda left, right: 0.25, 1.0, 2.0, rerun.RerunState.RERUNNING_IN_PLACE)
        == rerun.COMPARISON_MISMATCH
    )
    assert injector.injected_error_type is None
    assert injector.maybe_miscompare(lambda left, right: 0.25, 1.0, 2.0, rerun.RerunState.RERUNNING_IN_PLACE) == 0.25

    persistent = rerun.RerunErrorInjector(
        error_injection_rate=1,
        error_injection_type=rerun.RerunDiagnostic.PERSISTENT_ERROR,
    )
    persistent.injected_error_type = rerun.RerunDiagnostic.PERSISTENT_ERROR
    assert (
        persistent.maybe_miscompare(lambda left, right: 0.0, 1.0, 1.0, rerun.RerunState.RERUNNING_IN_PLACE)
        == rerun.COMPARISON_MATCH
    )
    assert (
        persistent.maybe_miscompare(lambda left, right: 0.0, 1.0, 2.0, rerun.RerunState.RERUNNING_FROM_CHECKPOINT)
        == rerun.COMPARISON_MISMATCH
    )

    checkpoint = persistent.state_dict()
    reloaded = rerun.RerunErrorInjector()
    reloaded.load_state_dict(checkpoint)
    assert reloaded.error_injection_type == rerun.RerunDiagnostic.PERSISTENT_ERROR


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def test_rerun_state_machine_validation_paths_and_checkpoint_helpers(monkeypatch, tmp_path):
    restored = []
    monkeypatch.setattr(rerun, "log_single_rank", lambda *args, **kwargs: None)
    monkeypatch.setattr(rerun, "safe_get_rank", lambda: 0)
    monkeypatch.setattr(rerun, "cur_platform", SimpleNamespace(current_device=lambda: 0))

    disabled = rerun.RerunStateMachine(mode=rerun.RerunMode.DISABLED)
    assert disabled.should_run_forward_backward(None) is True
    assert disabled.should_run_forward_backward(None) is False
    assert disabled.should_checkpoint_and_exit() == (False, False, 0)
    disabled.validate_result("bad", lambda _: True, "nonfatal", fatal=False)
    with pytest.raises(RuntimeError, match="Unexpected result"):
        disabled.validate_result("bad", lambda _: True, "fatal", fatal=True)

    machine = rerun.RerunStateMachine(
        state_save_func=lambda: {"custom": "state"},
        state_restore_func=restored.append,
        mode=rerun.RerunMode.VALIDATE_RESULTS,
    )
    monkeypatch.setattr(rerun, "get_rerun_state_machine", lambda: machine)
    monkeypatch.setattr(machine, "_save_state", lambda: setattr(machine, "saved_state", {"custom": "state"}))
    monkeypatch.setattr(machine, "_restore_state", lambda: restored.append(machine.saved_state))
    monkeypatch.setattr(machine, "_reduce_any", lambda value: tuple(value) if isinstance(value, list) else value)

    data_iterator = rerun.RerunDataIterator(iter(["microbatch"]))
    assert machine.should_run_forward_backward(data_iterator) is True
    assert next(data_iterator) == "microbatch"
    machine.first_iteration_complete = True
    machine.validate_result(1.0, lambda _: True, "loss spike", fatal=False)
    assert machine.rerun_requested is True

    assert machine.should_run_forward_backward(data_iterator) is True
    assert data_iterator.replaying is True
    machine.validate_result(
        2.0,
        lambda _: False,
        "loss spike",
        comparison_func=lambda initial, result: rerun.COMPARISON_MISMATCH,
        fatal=False,
    )
    assert machine.continue_requested is True
    assert machine.should_run_forward_backward(data_iterator) is False
    assert machine.state == rerun.RerunState.NOT_RUNNING_YET
    assert restored

    assert machine.is_unexpectedly_large(_Scalar(1.0), threshold=10.0, context="loss", num_samples=2) is False
    assert machine.is_unexpectedly_large(_Scalar(2.0), threshold=10.0, context="loss", num_samples=2) is False
    assert machine.is_unexpectedly_large(_Scalar(25.0), threshold=10.0, context="loss", num_samples=2) is True
    assert machine.is_unexpectedly_large(_Scalar(float("nan")), threshold=10.0, context="nan") is False
    assert machine.is_unexpectedly_large(_Scalar(float("inf")), threshold=10.0, context="inf") is False
    assert machine.is_unexpectedly_large(_Scalar(3.0), threshold=10.0, context="loss", resample=True) is False

    assert machine.validate_state_dict(None) is False
    assert machine.validate_state_dict({"state": rerun.RerunState.NOT_RUNNING_YET}) is False
    assert machine.validate_state_dict({"state": rerun.RerunState.INITIAL_RUN}) is True
    machine.load_state_dict(
        {
            "mode": rerun.RerunMode.VALIDATE_RESULTS,
            "state": rerun.RerunState.WILL_RERUN_FROM_CHECKPOINT,
            "current_iteration": 7,
            "sharded": {
                "rerun_requested": True,
                "checkpoint_requested": True,
                "restart_again_requested": False,
                "continue_requested": False,
                "error_injector_checkpoint": rerun.RerunErrorInjector().state_dict(),
                "failed_validation_call": "call",
                "initial_result": 3.0,
                "suspicious_node": "node",
                "suspicious_device": 0,
                "data_iterator_checkpoints": [{"saved_microbatches": [], "replaying": False, "replay_pos": 0}],
                "large_value_counts": {"loss": 2},
                "max_values": {"loss": 2.0},
            },
        }
    )
    assert machine.current_iteration == 7
    assert machine.data_iterator_checkpoints[0]["replay_pos"] == 0

    tracker = tmp_path / "rerun-tracker.log"
    tracker.write_text(
        "\n".join(
            [
                "ts=now node=node device=0 jobID=job-a rank=0 iteration=4 status=RerunValidationStatus.RERUN_DISABLED result=nan message='loss'",
                "ts=now node=node device=0 jobID=job-a rank=0 iteration=4 status=RerunValidationStatus.RERUN_DISABLED result=nan message='loss'",
                "ts=now node=node device=0 jobID=job-b rank=0 iteration=5 status=RerunValidationStatus.FIRST_RERUN_REPRODUCIBLE result=nan message='loss'",
                "ts=now node=node device=0 jobID=job-b rank=1 iteration=5 status=RerunValidationStatus.FIRST_RERUN_REPRODUCIBLE result=nan message='loss'",
            ]
        )
    )
    assert rerun.RerunStateMachine.get_skipped_iterations_from_tracker_file(str(tracker)) == [4]
