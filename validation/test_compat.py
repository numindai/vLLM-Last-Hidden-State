# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture installation guards and idempotence; no claim of GPU execution."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_last_hidden_state.compat import install_capture


@pytest.fixture
def upstream_types(monkeypatch):
    class Model:
        pass

    class CPU:
        pass

    class GPU:
        pass

    for path, symbol, cls in [
        ("vllm.model_executor.models.qwen3_5", "Qwen3_5Model", Model),
        ("vllm.v1.worker.cpu_model_runner", "CPUModelRunner", CPU),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", GPU),
    ]:
        monkeypatch.setitem(sys.modules, path, SimpleNamespace(**{symbol: cls}))
    return Model, CPU, GPU


def test_cpu_capture_is_attached_once_to_outer_model(upstream_types):
    model_type, cpu_type, _ = upstream_types
    outer = SimpleNamespace(
        language_model=SimpleNamespace(model=model_type()),
        register_forward_hook=Mock(return_value="hook"),
    )
    worker = SimpleNamespace(get_model=lambda: outer, model_runner=cpu_type())
    connector = SimpleNamespace(capture_batch=Mock())
    install_capture(worker, connector)
    install_capture(worker, connector)
    outer.register_forward_hook.assert_called_once()
    hook = outer.register_forward_hook.call_args.args[0]
    output = object()
    hook(outer, (), output)
    connector.capture_batch.assert_called_once_with(worker.model_runner, output)
    assert worker._last_hidden_state_hook == "hook"


def test_gpu_capture_is_wrapped_once_and_preserves_output(upstream_types):
    model_type, _, gpu_type = upstream_types
    output = object()
    runner = gpu_type()
    original = Mock(return_value=output)
    runner._model_forward = original
    worker = SimpleNamespace(
        get_model=lambda: SimpleNamespace(model=model_type()), model_runner=runner
    )
    connector = SimpleNamespace(capture_batch=Mock())
    install_capture(worker, connector)
    wrapper = runner._model_forward
    install_capture(worker, connector)
    assert runner._model_forward is wrapper
    assert runner._model_forward(input_ids="ids") is output
    original.assert_called_once_with(input_ids="ids")
    connector.capture_batch.assert_called_once_with(runner, output)


def test_unknown_runner_is_rejected_before_installing_hook(upstream_types):
    model_type, _, _ = upstream_types
    outer = SimpleNamespace(model=model_type(), register_forward_hook=Mock())
    worker = SimpleNamespace(get_model=lambda: outer, model_runner=object())
    with pytest.raises(ValueError, match="V1 CPU or GPU model runner"):
        install_capture(worker, object())
    outer.register_forward_hook.assert_not_called()


def test_unknown_model_is_rejected(upstream_types):
    _, cpu_type, _ = upstream_types
    worker = SimpleNamespace(get_model=lambda: object(), model_runner=cpu_type())
    with pytest.raises(ValueError, match="Only Qwen3.5"):
        install_capture(worker, object())
