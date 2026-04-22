# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from megatron.rl import GenericGenerationArgs
from megatron.rl.inference.api import InferenceRequest
from megatron.rl.inference.megatron import MegatronLocal


class _DummyTokenizer:
    bos = None
    eod = 42


@pytest.mark.asyncio
@patch("megatron.rl.inference.megatron.get_tokenizer", return_value=_DummyTokenizer())
@patch("megatron.rl.inference.megatron.get_args")
@patch("megatron.rl.inference.megatron.get_dynamic_inference_engine")
@patch("megatron.rl.inference.megatron.get_static_inference_engine")
async def test_megatron_local_launch_uses_static_engine_when_dynamic_batching_disabled(
    mock_get_static_engine,
    mock_get_dynamic_engine,
    mock_get_args,
    _mock_get_tokenizer,
):
    static_engine = MagicMock()
    mock_get_static_engine.return_value = static_engine
    mock_get_args.return_value = SimpleNamespace(
        inference_dynamic_batching=False,
        inference_wandb_logging_step_interval=0,
        rank=0,
        world_size=1,
    )

    launched = await MegatronLocal.launch(model=MagicMock())

    mock_get_static_engine.assert_called_once()
    assert mock_get_static_engine.call_args.kwargs["legacy"] is True
    mock_get_dynamic_engine.assert_not_called()
    assert launched._client is None
    assert launched._inference_engine is static_engine


@pytest.mark.asyncio
@patch("megatron.rl.inference.megatron.InferenceClient")
@patch("megatron.rl.inference.megatron.dist.get_rank", return_value=0)
@patch("megatron.rl.inference.megatron.get_tokenizer", return_value=_DummyTokenizer())
@patch("megatron.rl.inference.megatron.get_args")
@patch("megatron.rl.inference.megatron.get_dynamic_inference_engine")
@patch("megatron.rl.inference.megatron.get_static_inference_engine")
async def test_megatron_local_launch_uses_dynamic_engine_when_dynamic_batching_enabled(
    mock_get_static_engine,
    mock_get_dynamic_engine,
    mock_get_args,
    _mock_get_tokenizer,
    _mock_get_rank,
    mock_inference_client,
):
    dynamic_engine = MagicMock()
    dynamic_engine.start_listening_to_data_parallel_coordinator = AsyncMock()
    mock_get_dynamic_engine.return_value = dynamic_engine

    client = MagicMock()
    client.start = AsyncMock()
    mock_inference_client.return_value = client

    mock_get_args.return_value = SimpleNamespace(
        inference_dynamic_batching=True,
        inference_wandb_logging_step_interval=0,
        rank=0,
        world_size=1,
    )

    launched = await MegatronLocal.launch(model=MagicMock())

    mock_get_static_engine.assert_not_called()
    mock_get_dynamic_engine.assert_called_once()
    dynamic_engine.start_listening_to_data_parallel_coordinator.assert_awaited_once_with(
        inference_coordinator_port=41521,
        launch_inference_coordinator=True,
    )
    client.start.assert_awaited_once()
    assert launched._client is client
    assert launched._inference_engine is dynamic_engine


@pytest.mark.asyncio
@patch("megatron.rl.inference.megatron.get_tokenizer", return_value=_DummyTokenizer())
async def test_megatron_local_base_generate_without_client_uses_static_engine(
    _mock_get_tokenizer,
):
    static_engine = MagicMock()
    static_engine.controller = SimpleNamespace(tokenizer=_DummyTokenizer())
    static_engine.controller.tokenize_prompt.side_effect = lambda prompt, add_BOS: [10, 11]
    static_engine.get_new_request_id.return_value = 7
    static_engine.generate.return_value = [
        SimpleNamespace(
            generated_text=" world",
            prompt_tokens=[10, 11],
            generated_tokens=[12, 13],
            generated_log_probs=[-0.1, -0.2],
        )
    ]

    server = MegatronLocal()
    server._client = None
    server._inference_engine = static_engine

    request = InferenceRequest(
        prompt=["hello"],
        generation_args=GenericGenerationArgs(max_tokens=2, temperature=1.0, top_p=0.0, top_k=0),
    )

    responses = await server.base_generate(request)

    static_engine.generate.assert_called_once()
    generated_requests = static_engine.generate.call_args.kwargs["inference_requests"]
    assert len(generated_requests) == 1
    assert generated_requests[0].request_id == 7
    assert generated_requests[0].prompt == "hello"
    assert generated_requests[0].prompt_tokens == [10, 11]
    assert generated_requests[0].sampling_params.num_tokens_total is None
    assert generated_requests[0].sampling_params.num_tokens_to_generate == 0
    assert len(responses) == 1
    assert responses[0].response == " world"
    assert responses[0].raw_text == "hello world"
    assert responses[0].token_ids == [10, 11, 12, 13]
    assert responses[0].prompt_length == 2
