# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture installation guards and idempotence; no claim of GPU execution."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_last_hidden_state import FLAG, HANDLE
from vllm_last_hidden_state.compat import install_capture


@pytest.fixture
def upstream_types(monkeypatch):
    class Model:
        pass

    class GPU:
        pass

    class CPU(GPU):
        pass

    for path, symbol, cls in [
        ("vllm.model_executor.models.qwen3_5", "Qwen3_5Model", Model),
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", GPU),
        ("vllm.v1.worker.cpu.model_runner", "CPUModelRunner", CPU),
    ]:
        monkeypatch.setitem(sys.modules, path, SimpleNamespace(**{symbol: cls}))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.cpu_model_runner", None)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", None)
    return Model, CPU, GPU


def test_unknown_runner_is_rejected_before_installing_hook(upstream_types):
    model_type, _, _ = upstream_types
    outer = SimpleNamespace(model=model_type(), register_forward_hook=Mock())
    worker = SimpleNamespace(get_model=lambda: outer, model_runner=object())
    with pytest.raises(ValueError, match="supported CPU or GPU model runner"):
        install_capture(worker, object())
    outer.register_forward_hook.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("multimodal", [False, True])
def test_v2_capture_is_installed_once_without_model_forward_hook(
    upstream_types, device, multimodal
):
    model_type, _, _ = upstream_types
    module = sys.modules[f"vllm.v1.worker.{device}.model_runner"]
    runner_type = getattr(module, f"{device.upper()}ModelRunner")
    runner = runner_type()
    runner.execute_model = Mock()
    outer = SimpleNamespace(model=model_type(), register_forward_hook=Mock())
    if multimodal:
        outer = SimpleNamespace(language_model=outer, register_forward_hook=Mock())
    worker = SimpleNamespace(get_model=lambda: outer, model_runner=runner)
    install_capture(worker, object())
    wrapper = runner.execute_model
    install_capture(worker, object())
    assert runner.execute_model is wrapper
    outer.register_forward_hook.assert_not_called()


def test_unknown_model_is_rejected(upstream_types):
    _, cpu_type, _ = upstream_types
    worker = SimpleNamespace(get_model=lambda: object(), model_runner=cpu_type())
    with pytest.raises(ValueError, match="Only Qwen3.5"):
        install_capture(worker, object())


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_mtp_draft_forward_does_not_trigger_target_capture(upstream_types, device):
    model_type, cpu_type, gpu_type = upstream_types

    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model_type()

        def forward(self, states):
            return states + 1

    target = Target()
    draft = Target()
    runner = cpu_type() if device == "cpu" else gpu_type()
    runner.drafter = SimpleNamespace(model=draft)
    runner.execute_model_state = None

    def execute(schedule, intermediate_tensors, dummy_run):
        runner.execute_model_state = SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=["request"],
                num_computed_tokens_np=[0],
                query_start_loc_np=[0, 2],
            ),
            hidden_states=target(torch.zeros(2, 3)),
        )

    runner.execute_model = execute
    worker = SimpleNamespace(get_model=lambda: target, model_runner=runner)
    connector = SimpleNamespace(capture_rows=Mock())
    install_capture(worker, connector)
    schedule = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="request",
                prompt_token_ids=[10, 11],
                sampling_params=SimpleNamespace(
                    extra_args={
                        "kv_transfer_params": {
                            FLAG: True,
                            HANDLE: "handle",
                        }
                    }
                ),
            )
        ],
        total_num_scheduled_tokens=2,
    )
    assert runner.execute_model(schedule) is None
    output = runner.execute_model_state.hidden_states
    runner.drafter.model(output)
    connector.capture_rows.assert_called_once()
    assert connector.capture_rows.call_args.args[1] is output
