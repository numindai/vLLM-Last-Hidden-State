# Agent instructions: one-token hidden states through chat completions

This is a task brief intended to be usable as an `AGENTS.md`. Keep it in
`EMBEDDINGS.md`; do not replace, rename, or edit the existing `AGENTS.md`.
The instructions below define the implementation task for an agent given this
brief.

## Objective

Implement a small, local extension to vLLM so that a client can call the normal
`POST /v1/chat/completions` endpoint with an extra opt-in argument and
`max_tokens=1`, and receive the final hidden state at the last non-padding
**input/prompt position** inline in the response, alongside the normal completion.
This is the state used to predict the first generated token, not the generated
token's own hidden state. It is already computed during prefill; extracting it
does not require feeding the generated token through another forward pass.

Return exactly one vector: one token position, from the final model layer.
Do not return all token states, all layer states, pooled embeddings, input token
embeddings, or logits. Despite this file's name, this is a generation feature,
not a request to implement or use `/v1/embeddings`.

This extension is distributed publicly at
https://github.com/numindai/vLLM-Last-Hidden-State under Apache-2.0. Keep it
independently installable without maintaining a vLLM fork. Minimize installation
effort and maintenance across upgrades. The README owns the current public API
and supported configurations; this brief records the original implementation scope.

## Required API behavior

- Support non-streaming chat completions with one conversation, `n=1`, and
  `max_tokens=1`. An assertion for the one-token restriction is acceptable;
  a clear request validation error is also fine. Apply restrictions only when
  the feature is requested. Do not build multi-token or streaming support.
- Add an explicit request flag and one response field containing the vector.
  Prefer existing extensible fields if they avoid changing protocol classes.
  The exact names and nesting are implementation choices, not requirements.
- Preserve normal completion content, finish behavior, and externally accurate
  usage reporting. Ordinary requests must retain their normal behavior.
- Return the values in the HTTP response itself. The client must not need to
  fetch another endpoint, access server storage, or load a tensor file.
- Prefer a JSON list of numbers of length `hidden_size`. Do not add a general
  tensor serialization framework for a single vector.
- Keep the implementation narrow. Additional restrictions on model, backend,
  parallelism, or caching are acceptable when necessary for minimality, but
  must be explicit and validated rather than silently producing wrong results.

Illustrative API contract, using existing `kv_transfer_params` fields:

```python
response = client.chat.completions.create(
    model="<served-model>",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1,
    n=1,
    stream=False,
    extra_body={"kv_transfer_params": {"return_last_hidden_state": True}},
)
vector = response.model_dump()["kv_transfer_params"]["last_hidden_state"]
```

This flag is a proposed name, not an existing vLLM capability. Document the
actual request, response location, and OpenAI Python client access once tested.

## Resolve token alignment correctly

Match the user's Transformers reference with `numind/NuExtract3`: run
`model.model(**inputs, return_dict=True, use_cache=False)` and select
`outputs.last_hidden_state[batch_indices, last_non_padding_positions]`.
The HTTP call still generates one token, but the returned vector comes from
the input sequence before that token is appended. Do not add a decode step or
increase the token limit to compute the generated token's own state.

The reference supplies an image, extraction template, description instruction,
and an unfinished assistant message ending with this JSON prefix:

```text
{
  "short description": "
```

It uses `continue_final_message=True` and `add_generation_prompt=False` so the
first generated token continues the description string. Preserve this boundary
in the chat API using the corresponding supported options. The selected state
belongs to the final token of the fully formatted input, including the assistant
prefill, not necessarily the final token of the user's message. Inspect actual
tokenization; the opening quote may share a token with adjacent characters.
Support this image-plus-assistant-prefill use case, without hardcoding its field
name or instruction into the extraction implementation.

The current `ExampleHiddenStatesConnector.request_finished` explicitly excludes
the final generated token even with `include_output_tokens=True`, because that
token was never an input to a forward pass. That exclusion is consistent with
this task: extract only the final prompt-position vector. Verify its actual
token ID, position, and layer instead of assuming the last cache slot is correct.

Return the same final representation as the reference's `last_hidden_state`,
including the model's final output normalization if applied there. The existing
extraction documentation says that selecting `num_hidden_layers` gives the
last layer's output **before** output normalization. Do not assume it matches
the reference's `last_hidden_state` without checking and matching normalization.
Do not add pooling or unit-length normalization. No intermediate-layer selection
API is needed.

## Minimal-change priority

1. Best outcome: one additional Python file, copied into an importable location
   and activated through existing vLLM configuration or registration hooks.
2. A few additional files, or a tiny separately installable plugin package with
   registration metadata, are acceptable if a single file cannot be loaded.
3. Prefer existing extension hooks and small subclasses over runtime patching.
   A file that monkey-patches many internals is not a minimal solution merely
   because it is one file. Disclose any unavoidable runtime patch precisely.
4. Aim for zero edits to existing vLLM source, registries, or packaging files.
   If additive files cannot meet the contract, explain the concrete missing
   hook and smallest fallback before expanding into a core patch. Do not turn
   this into the broad upstream RFC implementation.

Configuration flags, an import path, and installation of plugin metadata are
acceptable. State every required activation step; copying an unreferenced
Python file does not automatically make vLLM import it.

## Investigate these extension points first

Verify these against the checkout being used; they are leads, not a proven
end-to-end solution:

- Search for `extract_hidden_states`, `ExampleHiddenStatesConnector`, and
  `eagle_aux_hidden_state_layer_ids` to reuse existing extraction machinery.
- `KVConnectorFactory` accepts `kv_connector_module_path` plus a connector class
  name, allowing an external connector without editing the built-in registry.
- Chat request and response models already expose `kv_transfer_params`; trace
  its propagation through sampling, scheduler, engine outputs, and serving.
- General plugins use installed `vllm.general_plugins` entry points.
  `VLLM_PLUGINS` filters registered plugins; it does not import arbitrary files.
- Endpoint plugins use `vllm.endpoint_plugins`, require explicit allowlisting,
  and load only in API frontend processes. Check whether they can extend the
  existing chat route and verify actual route dispatch and serialization.
- `worker_extension_cls` can add worker methods callable through
  `collective_rpc`; it rejects name conflicts with existing worker attributes.
  Do not assume it can override the model execution method.

Trace the full lifecycle: API frontend, engine/scheduler, GPU worker, and return
path. Check imports and initialization in spawned processes and Ray actors if
supported. An in-process dictionary is not shared across these boundaries.
Register hooks idempotently and install required modules wherever they execute.

Distinguish delaying KV block release from delaying the HTTP response until the
vector is ready. Ensure correct request association, GPU-to-CPU completion, and
cleanup on success or cancellation. Avoid exporting the whole sequence only to
discard it in the frontend. Prefer in-memory transfer; if temporary server-side
storage is the smallest viable bridge, disclose its cost and clean it up within
the server. It must never become a client-visible file workflow.

## Validation and deliverables

- Deliver the smallest additive implementation, exact copy/install locations,
  activation and server commands, and one working chat-completion client example.
- Record the tested vLLM revision, model, backend, normalization convention,
  restrictions, and any overhead from extraction or its buffers.
- Verify one finite vector of length `hidden_size` at the final non-padding input
  position against the Transformers reference with suitable dtype tolerances.
  Match image preprocessing, chat formatting, input token IDs, assistant prefill,
  and final normalization. The reference input must not include the generated
  token. Include the NuExtract3 image-plus-description-prefix case.
- Test the actual HTTP response, unsupported opted-in requests, normal requests
  without the flag, and concurrent requests with distinct prompts to catch
  swapped vectors. Reuse nearby tests and keep validation proportional.
- Follow repository development instructions for Python, linting, and relevant
  serving/model evaluations. Report commands and actual results; distinguish
  checks that ran from checks blocked by unavailable hardware or dependencies.
- Show that existing vLLM files remain unchanged, or itemize and justify every
  unavoidable exception. Do not claim copy-only feasibility without proving the
  complete loading, extraction, transport, and response path.

## References

- [RFC #48743: return extracted hidden states in generation responses](https://github.com/vllm-project/vllm/issues/48743).
  Motivation for inline return; its broad implementation scope is not required.
- [Extracting hidden states from vLLM](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-03-30-extract-hidden-states.md).
  Inspiration for reusing extraction and connector infrastructure.
- Local guides: [hidden state extraction](vllm/docs/features/speculative_decoding/extract_hidden_states.md),
  [plugin system](vllm/docs/design/plugin_system.md), and
  [endpoint plugins](vllm/docs/design/endpoint_plugins.md).
