# Last-prompt-state implementation contract

This document records the current package design, evolved from the original
implementation brief. The [README](README.md) is the public introduction and
client guide; [MAINTENANCE.md](MAINTENANCE.md) covers development and upgrades.
The independently installable Apache-2.0 package lives at
[NuMind's GitHub repository](https://github.com/numindai/vLLM-Last-Hidden-State).
Native integration is proposed separately in [vLLM PR #57185](https://github.com/vllm-project/vllm/pull/57185).

## Tensor and API contract

Expose the last active **prompt** token's final post-output-normalization state
inline in a normal Python `/v1/chat/completions` response. It is the vector used
to predict the first generated token. Do not append that generated token and
run another forward pass, pool tokens, select arbitrary layers, return logits,
or apply unit-length normalization.

- Request: `kv_transfer_params.return_last_hidden_state=true`.
- Response: `kv_transfer_params.last_hidden_state`, a finite float32 JSON list
  of length `hidden_size`.
- Extraction requires `n=1`, non-streaming, and an effective output limit of
  one (`max_completion_tokens=1` or `max_tokens=1`). Invalid requests return a
  clear error. Ordinary requests retain their normal budgets and streaming.
- The vector is returned in the same response; no client retrieval endpoint,
  file, or tensor serialization format is needed.
- The package returns the vector only. The upstream proposal additionally
  returns position, layer, and representation metadata.

## Prompt alignment

Compare with the final active row of
`model.model(**inputs, return_dict=True, use_cache=False).last_hidden_state`.
Use the same model/processor revision, image preprocessing, chat template,
thinking mode, and token IDs. Model output normalization is distinct from L2
normalization; only the former belongs in this contract.

For an unfinished assistant prefix such as `{"short description": "`, use
`continue_final_message=True` and `add_generation_prompt=False`. Capture the
last position of the fully formatted input, including that prefix, rather than
assuming it is the last lexical token of the user message. Keep
`chat_template_kwargs={"enable_thinking": False}` explicit in validation.

Packed row selection must account for request order, cached-prefix length,
chunk boundaries, and padding. A partial chunk that has not reached the final
prompt token must not produce a vector. Copy into owned CPU memory before the
runner or drafter can reuse the output buffer.

## Additive implementation

Do not modify upstream source in the package's `vllm/` submodule.

1. `serve.py` validates effective upstream CLI/YAML settings, activates the
   endpoint/worker/connector, supplies `mp` when missing, and selects the Python
   frontend. Runner selection and precision remain upstream/caller decisions.
2. `endpoint.py` validates opt-in requests, creates request handles, delegates
   ordinary chat work, and includes the retrieved vector in the response.
3. `compat.py` checks vLLM 0.30.0 and installs an instance-local V1 CPU model
   hook, V1 GPU forward wrapper, or V2 execute wrapper. V2 captures fresh target
   `execute_model_state` before sampling/drafting; it tracks only opted-in
   metadata through chunking and preemption.
4. `connector.py` selects the final prompt row, stores bounded request results,
   and propagates a handle through completion metadata. `worker.py` exposes
   initialization, take, and discard RPCs. Cleanup covers errors/cancellation.

These capture hooks/wrappers are deliberate runtime adaptations, not a public
upstream hidden-state callback. The direct output path in the upstream PR is
simpler inside vLLM itself; the package's connector/RPC transport keeps this
installation additive. Share API semantics and behavioral tests, not necessarily
transport internals.

## Scope and evidence

Current guards allow native Qwen3.5, one local `mp` worker, CPU/CUDA V1 or V2,
compatible compilation modes, and MTP only among speculative methods. LoRA,
distributed parallelism, other KV connectors, and other model families remain
unsupported. Dtype/quantization validation belongs to upstream vLLM.

V2 remains experimental. Real CPU eager/compiled and eager MTP evidence is in
[V2-v0.30.0.md](validation/V2-v0.30.0.md). The [release report](validation/UPGRADE-v0.30.0.md)
records V1/precision coverage and earlier failures. Preserve the failed image
comparisons; do not infer numerical parity or retrieval quality from passing
transport tests. GPU execution, graph replay, async scheduling, and real
preemption/cancellation still need hardware/runtime validation.

The [research references](README.md#research-background) motivate the readout.
This package is serving infrastructure, not a new embedding algorithm or a
claim of reproduction of any paper's benchmark results.
