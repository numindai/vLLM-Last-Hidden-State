# Updating alongside vLLM

This is the upgrade runbook for the private package. The
[README](README.md#development) describes each implementation file and
the request lifecycle; its [compatibility section](README.md#supported-configurations)
describes current restrictions. Version and activation conflicts are checked at startup; model/runtime
restrictions are checked in the connector and compatibility module.

## Start an upgrade

1. Identify the target vLLM revision and the last tested revision from the newest
   recorded validation report. Inspect the worktree before switching checkouts;
   the extension is tracked by this repository and upstream source lives in the
   `vllm/` submodule. Keep the package and its validation history together. An
   upstream clone alone does not contain this extension.
2. Read this package's `AGENTS.md` and the target checkout's applicable agent
   instructions. `EMBEDDINGS.md` in this repository is the original
   brief; the README records later decisions on launcher behavior and supported
   execution modes.
3. Record the target with `git -C vllm rev-parse HEAD`, `git -C vllm describe --always --dirty`,
   and `git -C vllm status --short`. Record Python and installed dependencies with
   `.venv/bin/python --version` and `uv pip freeze`. A checkout SHA alone does not
   identify the installed wheel, native build, or editable source location.
4. Prepare a working vLLM installation using the target repository's environment
   instructions for the actual hardware. Confirm ordinary vLLM can start before
   attributing native build, kernel, or model-loading failures to this package.
5. Reinstall the extension in that environment:
   `uv pip install --no-deps -e .`. Its runtime dependency on `packaging` supports version checks;
   the operator supplies vLLM and its matching PyTorch stack.
   It is not a declaration of compatibility with every vLLM version.

Commands below run from this package repository root. Keep run artifacts in
a dedicated directory, with separate filenames per checkpoint and configuration.
Use [UPGRADE_REPORT_TEMPLATE.md](validation/UPGRADE_REPORT_TEMPLATE.md) for the
durable report; raw vectors and full environment dumps need not enter the package.

The supported version is centralized in `compat.SUPPORTED_VLLM_VERSION`. Hardware
local suffixes are accepted, but prereleases/dev/post releases are not. Update the
version policy only alongside the source audit and a new report; changing the
submodule alone does not certify compatibility. Never change installation metadata
to bypass the guard for an old native build.

The launcher uses the upstream serve parser for preflight checks, then executes
upstream with the original runtime options and missing activation flags. Test YAML,
dotted JSON, aliases, and CLI precedence when upgrading this parser integration.

## Audit the upstream contracts

Search by symbol because upstream file locations change. Use `git -C vllm diff OLD NEW`
on the source files found by these searches, then read the new callers and
implementations. If old history is unavailable, inspect the target source
directly and record that limitation. Successful imports alone cannot validate
these contracts.

| Package integration | Search upstream for | Contract to recheck |
| --- | --- | --- |
| `pyproject.toml`, `vllm_last_hidden_state/serve.py` | `vllm.endpoint_plugins`, `load_endpoint_plugins`, `VLLM_PLUGINS`, `VLLM_USE_RUST_FRONTEND` | Installed entry point is explicitly enabled in the Python frontend; original CLI still accepts activation arguments. |
| `endpoint.py` | `init_app_state`, `init_state`, `openai_serving_chat`, `create_chat_completion` | Plugin initialization runs after chat state exists; the existing route resolves the delegated handler and serializes its response. |
| Request metadata | `kv_transfer_params`, `to_sampling_params`, `extra_args` | Request flag and server-generated handle reach scheduler requests and worker sampling metadata without loss or shared mutation. |
| Result transport | `request_finished`, `request_finished_all_groups`, `EngineCoreOutput`, `collective_rpc` | Completion carries the handle back; RPC reaches the capturing worker and returns one result per worker. |
| `connector.py` registration | `KVConnectorBase_V1`, `SupportsHMA`, `KVConnectorFactory`, `kv_connector_module_path`, `get_transfer_results`, `register_finished_partial_tail` | Constructor/abstract methods and hybrid-cache lifecycle still fit a metadata-only connector with no external tokens, transfer completions, or delayed block release. |
| `worker.py` initialization | `worker_extension_cls`, `get_model`, `get_kv_transfer_group` | Extension methods do not conflict with worker methods; initialization reaches the actual runner after model warmup. |
| `compat.py`: CPU capture | `CPUModelRunner`, `register_forward_hook`, `execute_model` | Outer-model hook still runs outside the compiled decoder and sees its final normalized states. |
| `compat.py`: GPU capture | `GPUModelRunner`, `_model_forward`, `CUDAGraphWrapper`, `execute_model` | Wrapped method executes on every real batch, including graph replay, before corresponding batch metadata or output storage can be reused. |
| `compat.py`: packed metadata | `input_batch`, `req_ids`, `requests`, `query_start_loc`, `num_computed_tokens_cpu` | Request order, CPU offsets, query spans, output rows, and padding refer to the same forward pass. |
| Cache/chunk scheduling | `get_computed_blocks`, `max_cache_hit_length`, `num_computed_tokens`, `async_scheduling` | A cache hit still recomputes the final prompt token; chunk/preemption offsets describe actual computed input tokens. |
| Model representation | `Qwen3_5Model`, `Qwen3NextModel`, `self.norm`, `language_model` | Trace inherited forwards too: captured output is after final output normalization, before logits, for text and multimodal wrappers. |
| MTP | `EagleProposer`, `SpecDecodeBaseProposer`, `Qwen3_5MTP`, `finalize_kv_connector` | Drafter uses a separate outer model, sharing only embeddings/head; target capture precedes drafting and deferred connector finalization. Verify packed prompt offsets during speculative verification. |
| Runtime validation | `CompilationMode`, `device_config`, `parallel_config`, `speculative_config` | Guard names and meanings still match the supported backend, runner, executor, and normalized MTP method. Dtype/quantization validation belongs to upstream. |

For example:

```sh
rg -n 'worker_extension_cls|def collective_rpc|vllm.endpoint_plugins' vllm
rg -n '_model_forward|query_start_loc|num_computed_tokens_cpu' vllm
rg -n 'kv_transfer_params|request_finished_all_groups|max_cache_hit_length' vllm
rg -n 'Qwen3_5Model|Qwen3NextModel|VLLM_USE_RUST_FRONTEND' vllm
```

Prefer a new upstream extension hook if it satisfies the same semantics with
less maintenance. The current CPU module hook and GPU instance-method wrapper
are deliberate, disclosed adaptations. A moved/renamed runner requires tracing
the execution boundary; accepting another class in a type check is insufficient.

## Semantics that are easy to break

- Row selection is `span_start + (prompt_length - 1 - computed_tokens)` in
  actual batch order. A valid span that does not contain that position is an
  earlier prefill chunk or a later decode step, not an extraction failure.
- A completed result survives later chunks. Recomputing the final prompt after
  preemption replaces it; skipping every handle already in the result store
  would preserve stale data.
- Graph replay bypasses inner Python hooks. On GPU, copy the selected row into
  owned CPU memory while both tensor and metadata still describe this batch.
  The current synchronous copy is intentional. Making it nonblocking requires
  explicit completion/lifetime handling, not just changing a `.to()` argument.
- Recheck worker execution/RPC ordering when async machinery changes. Neither
  copying in a later callback nor reading mutable metadata after sampling is
  assumed safe. Concurrent requests must never exchange vectors.
- Preserve the handle lifecycle: server-generated per request, returned by the
  connector, consumed once by RPC, discarded on failure/cancellation. Bounded
  result/cancellation stores and lazy expiry prevent unbounded abandoned state.
- Ordinary requests delegate to the original handler, keep normal generation
  and streaming, and copy no hidden-state row. Opted-in failures must be visible
  errors rather than successful responses missing a vector.
- The existing `extract_hidden_states` speculative path was previously rejected
  because it affected ordinary sampler behavior and allocated sequence-sized
  state. Re-evaluate those costs and final normalization if upstream replaces it.

## Validation sequence

### Fast checks

Use the README's unit-test and lint commands first. These tests exercise row
selection, ownership, chunk/cache offsets, recomputation, and cancellation using
small fixtures. They bypass real connector construction and model execution;
passing them does not establish startup, protocol transport, or GPU support.
Run `ruff check .` and `ruff format --check .`. Pytest collection and lint
settings belong to this package, not the upstream submodule.

### Model and HTTP checks

Use the [historical validation commands](validation/HISTORY.md#validation) and
[recorded comparisons](validation/HISTORY.md#recorded-results) as the starting configurations.
For each model, preserve the checkpoint, tokenizer/processor revision, dtype,
template options, and input token IDs between reference and server. Pin model
revisions using the target vLLM CLI or a resolved local snapshot; the current
validation helper has no `--revision` argument, but accepts a snapshot as `--model`.
Use the same model identifier for the served model and HTTP helper.

1. Generate a fresh Transformers `reference` for the target environment. For
   drawais INT4, use `--text-only --dequantize` for reference generation and
   `--text-only` for HTTP. Detect `compressed-tensors` from model configuration;
   the repository name is not a reason to force the `awq` loader.
2. Start ordinary `.venv/bin/vllm serve` with `VLLM_PLUGINS=""` and the Python
   frontend (`VLLM_USE_RUST_FRONTEND=0`). Record `baseline` responses. Restart
   using `vllm-last-hidden-state` with the same runtime settings and pass that baseline to
   `http`. This isolates the extension's effect on ordinary output.
3. Run both eager and compiled configurations for original BF16 text/image and
   INT4 text. Check logs for actual compilation/backend and quantized kernels;
   accepting CLI flags does not prove they were active. The historical notes describe
   CPU compilation environment variables that can silently disable compilation.
4. Run long prompts with caching and chunking enabled. Start with a cold cache
   and set the batch token budget below the actual prompt length. The recorded
   CPU setup used a 4096-token model limit and 1280-token batch budget. Verify
   lengths again after tokenizer changes. Repeat with
   `--require-prefix-cache-hit`; the helper checks an increase in real cache-hit
   counters. Keep the server otherwise idle so unrelated traffic cannot satisfy
   that assertion.
5. Check sequential and concurrent distinct prompts, finite vector size, exact
   token alignment, first generated token, usage, normal one/three-token output,
   streaming, and invalid opted-in requests. The HTTP helper exercises these.
   Its `--baseline` comparison checks choices, usage, and prompt token IDs.
6. After changing protocol or cleanup handling, additionally exercise
   `max_completion_tokens=1` and its precedence, client cancellation, failed
   capture/RPC, and subsequent successful requests. After changing scheduling,
   exercise real preemption/recomputation and mixed ordinary/extraction traffic.
   These are not all covered by the existing automated HTTP helper.

The helper writes its report only after the complete suite passes. Capture stdout
and stderr as well as the exit status: failed runs print per-case metrics before
the assertion, and absence of the final JSON report must not erase that failure.
Use fresh report paths so a stale successful report cannot stand in for a failure.

### GPU and async scheduling

Use a CUDA installation and the intended GPU command in the historical notes. Confirm the
effective runner, compilation/graph mode, and async scheduling in startup/runtime
logs. Recheck the runner-selection flag in the target checkout rather than
assuming `VLLM_USE_V2_MODEL_RUNNER=0` still selects the supported implementation.

Run eager first, then compilation with real graph capture/replay and async
scheduling, including caching/chunking, repeated requests that reuse buffers,
mixed ordinary/extraction traffic, cancellation, and preemption. Compare against
appropriate same-checkpoint references and ordinary baselines. Record graph modes
and evidence of replay; a compiled prefill alone is not evidence of replay.
If scheduling/capture integration changed, compare async on/off with other
settings held constant. The synchronous per-vector CPU copy is an expected cost;
do not claim zero overhead from correctness tests.

Without a GPU, complete CPU and source-level checks and mark **GPU execution,
CUDA graphs, and async scheduling untested** in the report and compatibility
summary. This checkout disables async scheduling on CPU; recheck that policy
on upgrades. A fixture calling the GPU wrapper with CPU tensors is not runtime
validation of those paths.

## Diagnose comparison failures

First separate startup/transport, tokenization, row selection, and arithmetic.
An ordinary server failure points to environment/upstream problems; missing
handles point to metadata/completion transport. Wrong token IDs require matching
template/processor inputs before investigating vector tolerances. Wrong shape,
position, or request association requires inspecting capture boundaries.

For numerical differences, keep the existing thresholds and record the failure.
The historical long-image Transformers comparison failed even with caching and
chunking disabled; see
[results-caching-chunking.json](validation/results-caching-chunking.json).
Do not treat that as an automatic exemption for future failures.

A useful diagnostic is a same-vLLM control with extraction active, caching and
chunking disabled, and a batch budget large enough for the whole prompt. Collect
its opted-in vectors, exact prompt IDs, and first generated token into the
`reference` helper's JSON schema, labeling `backend` explicitly as a vLLM control.
The helper's `baseline` mode records ordinary responses, **not** hidden vectors;
it cannot create this control reference. Compare candidate and control to
Transformers independently, then candidate to control. Shared capture bugs may
affect both vLLM runs, so control agreement never replaces token/normalization
inspection or the independent Transformers comparison.

## Finish and hand off

Save a new upgrade report and linked metrics under `validation/`, leaving previous
results intact. Record exact server/client commands, environment, actual feature
activation, and pass/fail/untested status separately. Update the README's current
compatibility summary and this guide's contract map if upstream integration moved.
Do not infer a supported version range from a single successful checkout.

Review `git diff` and `git status --short` against the starting worktree so core
edits or unrelated user changes are not confused with this package. Inspect added
files too: `git diff` alone omits untracked package content. Keep the report and
package together in durable storage before discarding an old environment.

Suggested prompt for the next Codex session (replace the target placeholder):

> Update this last-hidden-state package for vLLM TARGET_REVISION. Read its AGENTS.md and
> maintenance guide, audit the upstream contracts, implement compatibility fixes,
> run available validation, and save an upgrade report. Preserve the API and
> pass-through launcher. Document unavailable GPU validation explicitly.
