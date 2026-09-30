# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard packed prompt-row selection and cancellation without loading a model.

The hook receives final normalized states for a packed batch. The cheapest check
for wrong-row extraction uses uneven prompt spans, reversed request IDs, and
padding. Actual model and HTTP behavior is checked by validate_hidden_state.py.
Chunk transitions and cached offsets must not select an earlier prompt row or
replace an owned result when a later batch advances past the prompt.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from vllm.config import CompilationMode
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from vllm_last_hidden_state.compat import capture_model_forward
from vllm_last_hidden_state.connector import (
    FLAG,
    HANDLE,
    LastHiddenStateConnector,
    VectorMetadata,
)
from vllm_last_hidden_state.endpoint import ChatWithLastHiddenState


@pytest.fixture
def connector():
    instance = object.__new__(LastHiddenStateConnector)
    instance.hidden_size = 3
    instance.results = {}
    instance.cancelled = {}
    return instance


@pytest.fixture
def runner():
    def request(tokens, handle):
        return SimpleNamespace(
            prompt_token_ids=tokens,
            sampling_params=SimpleNamespace(
                max_tokens=1,
                extra_args={"kv_transfer_params": {FLAG: True, HANDLE: handle}},
            ),
        )

    return SimpleNamespace(
        requests={"a": request([10, 11, 12], "ha"), "b": request([20, 21], "hb")},
        input_batch=SimpleNamespace(
            req_ids=["b", "a"], num_computed_tokens_cpu=np.array([0, 0])
        ),
        query_start_loc=SimpleNamespace(np=np.array([0, 2, 5])),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("quantization", [None, "compressed-tensors", "awq", "gptq"])
@pytest.mark.parametrize("method", [None, "mtp", "eagle3"])
def test_runtime_settings_delegate_precision_to_upstream(dtype, quantization, method):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        device_config=SimpleNamespace(device_type="cpu"),
        model_config=SimpleNamespace(
            dtype=dtype, quantization=quantization, get_hidden_size=lambda: 3
        ),
        speculative_config=SimpleNamespace(method=method) if method else None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        lora_config=None,
    )
    if method == "eagle3":
        with pytest.raises(ValueError, match="MTP speculative decoding only"):
            LastHiddenStateConnector(config, KVConnectorRole.WORKER, None)
    else:
        connector = LastHiddenStateConnector(config, KVConnectorRole.WORKER, None)
        assert connector.hidden_size == 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_precision_and_speculative_verification_preserve_prompt_state(
    connector, runner, dtype
):
    states = torch.arange(15, dtype=dtype).reshape(5, 3) / 8
    expected = states[4].float().tolist()
    connector.capture_batch(runner, states)
    # Drafting may reuse output storage; verification packs several decode rows.
    states.zero_()
    runner.input_batch.num_computed_tokens_cpu[:] = [2, 3]
    runner.query_start_loc.np[:] = [0, 3, 6]
    connector.capture_batch(runner, torch.full((6, 3), 99, dtype=dtype))
    assert connector.take("ha")["vector"] == expected


def test_final_prompt_rows_follow_batch_order_and_ignore_padding(connector, runner):
    states = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    connector.capture_batch(runner, (states, torch.zeros_like(states)))
    a, b = connector.take("ha"), connector.take("hb")
    assert a["vector"] == states[4].tolist()
    assert b["vector"] == states[1].tolist()
    assert (a["position"], a["token_id"]) == (2, 12)
    states.zero_()
    assert b["vector"] == [3.0, 4.0, 5.0]
    assert connector.take("ha") is None


def test_cancel_before_capture_discards_only_that_request(connector, runner):
    connector.discard("ha")
    connector.capture_batch(runner, torch.ones(5, 3))
    assert connector.take("ha") is None
    assert connector.take("hb")["vector"] == [1.0] * 3


def test_invalid_prompt_span_returns_error_instead_of_neighbor_state(connector, runner):
    runner.query_start_loc.np[-1] = 6
    connector.capture_batch(runner, torch.ones(5, 3))
    assert "error" in connector.take("ha")
    assert "vector" in connector.take("hb")


def test_cached_prefix_offsets_select_the_uncached_final_rows(connector, runner):
    runner.input_batch.num_computed_tokens_cpu[:] = [1, 2]
    runner.query_start_loc.np[:] = [0, 1, 2]
    states = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    connector.capture_batch(runner, states)
    assert connector.take("hb")["vector"] == states[0].tolist()
    assert connector.take("ha")["vector"] == states[1].tolist()


def test_chunked_prefill_waits_for_final_chunk_and_keeps_completed_result(
    connector, runner
):
    runner.query_start_loc.np[:] = [0, 2, 3]
    connector.capture_batch(runner, torch.ones(3, 3))
    assert "ha" not in connector.results
    assert connector.results["hb"][1]["vector"] == [1.0] * 3
    runner.input_batch.num_computed_tokens_cpu[:] = [2, 1]
    runner.query_start_loc.np[:] = [0, 1, 3]
    states = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    connector.capture_batch(runner, states)
    assert connector.take("ha")["vector"] == states[2].tolist()
    assert connector.take("hb")["vector"] == [1.0] * 3
    connector.capture_batch(runner, states)
    assert connector.take("hb") is None


def test_ordinary_request_does_not_capture_or_delay_blocks(connector, runner):
    for request in runner.requests.values():
        request.sampling_params.extra_args = {}
    connector.capture_batch(runner, torch.ones(5, 3))
    assert connector.results == {}
    assert connector.request_finished(SimpleNamespace(kv_transfer_params=None), []) == (
        False,
        None,
    )
    assert connector.build_connector_meta(None) == VectorMetadata()


def test_completion_returns_handle_without_reporting_kv_transfers(connector):
    """0.30's transfer snapshot must not turn extraction into an async KV send."""
    request = SimpleNamespace(kv_transfer_params={FLAG: True, HANDLE: "handle"})
    assert connector.request_finished_all_groups(request, ([1], [2])) == (
        False,
        {HANDLE: "handle"},
    )
    transfers = connector.get_transfer_results({"request-id"})
    assert not transfers.finished_sending
    assert not transfers.finished_recving
    assert not transfers.failed_recving
    assert not connector.register_finished_partial_tail(request, ([1],), [(0, 1, 2)])
    assert not connector.has_pending_block_frees()


def test_recomputed_final_prompt_replaces_state_from_a_preempted_forward(
    connector, runner
):
    connector.capture_batch(runner, torch.ones(5, 3))
    connector.capture_batch(runner, torch.full((5, 3), 2.0))
    assert connector.take("ha")["vector"] == [2.0] * 3


def test_forward_wrapper_copies_after_execution_before_reused_output_changes(
    connector, runner
):
    """Exercise the GPU wrapper boundary on CPU, without claiming CUDA coverage."""
    states = torch.zeros(5, 3)
    calls = []

    def original(*, value):
        calls.append(value)
        states.fill_(value)
        return states

    runner._model_forward = original
    capture_model_forward(runner, connector)
    assert runner._model_forward(value=7) is states
    result = connector.take("ha")
    runner.input_batch.num_computed_tokens_cpu[:] = [2, 3]
    assert runner._model_forward(value=9) is states
    assert calls == [7, 9]
    assert result["vector"] == [7.0] * 3
    assert connector.take("hb")["vector"] == [7.0] * 3
    assert connector.take("ha") is None


@pytest.mark.asyncio
async def test_cancelled_chat_discards_the_same_internal_handle():
    started = asyncio.Event()
    captured = {}

    async def generate(request, raw_request):
        captured["handle"] = request.kv_transfer_params[HANDLE]
        started.set()
        await asyncio.Event().wait()

    engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=[None]))
    handler = ChatWithLastHiddenState(
        SimpleNamespace(create_chat_completion=generate), engine
    )
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1,
        kv_transfer_params={FLAG: True},
    )
    task = asyncio.create_task(handler.create_chat_completion(request))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    engine.collective_rpc.assert_awaited_once_with(
        "last_hidden_state_discard", args=(captured["handle"],)
    )
