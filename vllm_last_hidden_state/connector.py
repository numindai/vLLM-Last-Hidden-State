# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-row extraction with request-scoped, in-memory CPU results."""

from dataclasses import dataclass
from time import monotonic
from typing import Any

import torch
from vllm.config import CompilationMode
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)

from . import FLAG, HANDLE
from .compat import prompt_batch_rows

TTL_SECONDS = 120.0
MAX_RESULTS = 1024


@dataclass
class VectorMetadata(KVConnectorMetadata):
    pass


class LastHiddenStateConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        parallel = vllm_config.parallel_config
        if parallel.distributed_executor_backend != "mp":
            raise ValueError(
                "Last hidden state requires the local multiprocessing executor"
            )
        if vllm_config.device_config.device_type not in ("cpu", "cuda"):
            raise ValueError("Last hidden state supports CPU or CUDA workers")
        if any(
            size != 1
            for size in (
                parallel.tensor_parallel_size,
                parallel.pipeline_parallel_size,
                parallel.data_parallel_size,
                parallel.decode_context_parallel_size,
                parallel.prefill_context_parallel_size,
            )
        ):
            raise ValueError("Last hidden state requires a single local worker")
        speculative = vllm_config.speculative_config
        if speculative is not None and speculative.method != "mtp":
            raise ValueError("Last hidden state supports MTP speculative decoding only")
        if vllm_config.compilation_config.mode not in (
            CompilationMode.NONE,
            CompilationMode.DYNAMO_TRACE_ONCE,
            CompilationMode.VLLM_COMPILE,
        ):
            raise ValueError("Unsupported compilation mode for last hidden state")
        if vllm_config.lora_config is not None:
            raise ValueError("Last hidden state does not support LoRA")
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        self.results: dict[str, tuple[float, dict[str, Any]]] = {}
        self.cancelled: dict[str, float] = {}

    @property
    def requires_kv_delivery(self):
        return False

    def start_load_kv(self, *args, **kwargs):
        pass

    def wait_for_layer_load(self, *args, **kwargs):
        pass

    def save_kv_layer(self, *args, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        return VectorMetadata()

    def request_finished(self, request, block_ids):
        params = request.kv_transfer_params or {}
        if not params.get(FLAG) or not params.get(HANDLE):
            return False, None
        return False, {HANDLE: params[HANDLE]}

    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids)

    @torch.inference_mode()
    def capture_batch(self, runner, output):
        """Copy the final prompt row from each opted-in request's packed span."""
        self._expire()
        hidden_states = output[0] if isinstance(output, tuple) else output
        for request, computed_tokens, span_start, span_end in prompt_batch_rows(runner):
            sampling = request.sampling_params
            params = (sampling.extra_args or {}).get("kv_transfer_params") or {}
            handle = params.get(HANDLE)
            if not params.get(FLAG) or not handle or handle in self.cancelled:
                continue
            try:
                if sampling.max_tokens != 1 or not request.prompt_token_ids:
                    raise ValueError("Expected one-token generation with prompt IDs")
                position = len(request.prompt_token_ids) - 1
                relative = position - computed_tokens
                if not 0 <= span_start <= span_end <= hidden_states.shape[0]:
                    raise ValueError("Invalid packed query span")
                if relative < 0 or span_start + relative >= span_end:
                    continue
                row = hidden_states[span_start + relative]
                if row.device.type not in ("cpu", "cuda") or row.shape != (
                    self.hidden_size,
                ):
                    raise ValueError("Unexpected final hidden-state shape or device")
                vector = row.detach().to(device="cpu", dtype=torch.float32, copy=True)
                if not vector.isfinite().all():
                    raise ValueError("Non-finite hidden-state vector")
                result = {
                    "vector": vector.tolist(),
                    "position": position,
                    "token_id": request.prompt_token_ids[position],
                }
            except Exception as exc:
                result = {"error": str(exc)}
            if len(self.results) >= MAX_RESULTS:
                self.results.pop(next(iter(self.results)))
            self.results[handle] = (monotonic(), result)

    def _expire(self):
        cutoff = monotonic() - TTL_SECONDS
        self.results = {
            key: value for key, value in self.results.items() if value[0] > cutoff
        }
        self.cancelled = {
            key: timestamp
            for key, timestamp in self.cancelled.items()
            if timestamp > cutoff
        }

    def take(self, handle):
        self._expire()
        result = self.results.pop(handle, None)
        return result[1] if result else None

    def discard(self, handle):
        self._expire()
        self.results.pop(handle, None)
        if len(self.cancelled) >= MAX_RESULTS:
            self.cancelled.pop(next(iter(self.cancelled)))
        self.cancelled[handle] = monotonic()
