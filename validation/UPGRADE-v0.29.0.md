# vLLM 0.29.0 compatibility report

> Historical v0.29.0 evidence. Current release coverage is in
> [UPGRADE-v0.30.0.md](UPGRADE-v0.30.0.md) and [V2-v0.30.0.md](V2-v0.30.0.md).
> Commands, versions, and results below are preserved for their original runs;
> they do not describe current launcher defaults or certify v0.30.0.

Date: 2026-09-21. This report supersedes no historical numerical results; the
previous results remain in [HISTORY.md](HISTORY.md) and their original JSON files.

## Revisions and environment

- Previous tested source: `23cfaad49701c497def53552b23317335431f72a`.
- Target: upstream tag `v0.29.0`, commit
  `98dff2a81d747d1dba01a47f939f48c3526d4206` (unmodified tracked source).
- Extension: `0.1.0`, validated before the initial `v0.29.0` Git tag; no distribution wheel.
- Python 3.12.13, PyTorch 2.13.0+cpu, Transformers 5.17.0, OpenAI 3.13.0,
  packaging 26.3; pytest 9.1.1, pytest-asyncio 1.4.0, Ruff 0.16.7.
- Linux x86_64, AMD Ryzen AI MAX+ 395; CUDA is unavailable in this CPU environment.
- Runtime test model: `Qwen/Qwen3.5-0.8B`, snapshot
  `2fc06364715b967f1860aea9cf38778875588b17`, BF16, including image processing and
  assistant prefill. Both reference and server use this exact cached snapshot.

The inherited environment identified vLLM as `0.1.dev1+g23cfaad49.cpu`, despite
checkout of the new tag. The new startup check correctly rejected that version.
Initial upstream-only startup worked with those older native artifacts, but is
**not** evidence for a clean v0.29.0 build. Native CPU extensions were rebuilt
from the target source before recording the runtime results below.

## Source contracts checked

| Contract | v0.29.0 evidence and conclusion |
| --- | --- |
| CLI parsing | `FlexibleArgumentParser` and `ServeSubcommand.subparser_init` resolve aliases, dotted JSON, YAML, and CLI precedence. The launcher now uses that parser for conflict checks. |
| Plugin defaults | `vllm/plugins/__init__.py` loads non-endpoint groups by default when the allowlist is unset; endpoint plugins require explicit names. Preserve those defaults and existing explicit names when adding this plugin. |
| Endpoint lifecycle | `entrypoints/launchers/api_server/app_state.py` calls `init_generate_state` before `init_endpoint_plugins_state`; `state.openai_serving_chat` therefore exists before delegation. Plugin initialization errors propagate. |
| Request metadata | `ChatCompletionRequest.to_sampling_params` places `kv_transfer_params` into `extra_args`; the worker can recover the generated handle and opt-in flag. |
| Connector API | `KVConnectorBase_V1` still takes config, role and KV-cache config. `SupportsHMA.request_finished_all_groups` remains the hybrid-cache handoff. Changed optional transfer APIs are not used by this metadata-only connector. |
| Completion transport | Scheduler `_free_request` passes the connector's transfer params into `EngineCoreOutput`; frontend response and collective RPC paths retain the handle. |
| Worker extension | `worker_base.py` rejects attribute conflicts and adds extension methods to the worker class for RPC. Our extension exposes only initialization/take/discard. |
| CPU capture | `CPUModelRunner` still inherits `GPUModelRunner`; the outer Qwen3.5 model returns the decoder's final states. Install the outer hook after warmup and only once. |
| GPU capture | `GPUModelRunner.execute_model` calls `_model_forward` before extracting logits rows. The wrapper copies after the model/graph call and before the runner advances. Source inspection is not CUDA runtime evidence. |
| Final normalization | `Qwen3_5Model` inherits the `Qwen3NextModel.forward` path, which applies `self.norm` before returning hidden states. The outer text/multimodal wrappers return those states unchanged. |
| Batch metadata | `input_batch.req_ids`, `requests`, `query_start_loc.np`, and `num_computed_tokens_cpu` still identify the packed prompt spans. Access now lives in `compat.prompt_batch_rows`. |
| Cache/chunking | `KVCacheManager` limits cache hits to `request.num_tokens - 1`, retaining final-token recomputation. Selection uses prompt length minus computed-token offset. |
| Async CPU policy | `platforms/cpu.py` explicitly sets `async_scheduling=False`. CPU results cannot certify GPU async execution. |

## Package changes and checks

- `compat.py` owns the exact version policy, upstream parser access, plugin
  discovery defaults, runner/model inspection, batch metadata, and capture hooks.
- Version guards run in the launcher, endpoint initialization, and worker
  initialization. Accept `0.29.0` with optional local hardware suffixes; reject
  other versions, dev/prerelease/post releases, and missing installations.
- The launcher preserves compatible existing settings. It rejects effective
  conflicting workers/connectors and an explicitly requested Rust frontend.
  CLI overrides of YAML follow upstream precedence. Other plugin compatibility
  is not asserted merely because their allowlist entries are preserved.
- The package owns pytest collection, Ruff settings, and development extras.
  Hardware-specific vLLM/PyTorch installation remains external.
- 44 model-free tests passed against target Python source. Coverage includes
  version policy, early rejection, plugin selection, actual upstream CLI parsing,
  YAML/alias/dotted forms, capture installation/idempotence, packed rows,
  chunk/cache offsets, ownership, preemption replacement, and cancellation.
- Ruff checks/formatting and whitespace checks passed. One expected upstream
  deprecation warning concerns the still-supported `max_tokens` request field.

## Runtime validation

All measurements below use the 0.8B snapshot and freshly rebuilt CPU extensions.
The numerical thresholds remain cosine >= 0.999 and relative L2 <= 5%.

| Check | Result | Evidence |
| --- | --- | --- |
| Upstream baseline, short and long text/image requests | Pass | Ordinary one-token and three-token responses recorded before enabling the extension. |
| Eager short suite vs Transformers | Pass | [Metrics](results-v0.29.0-eager-short.json): minimum cosine 0.999007, maximum relative L2 4.467%. |
| Eager long suite vs Transformers | Fail on image | [Failure](results-v0.29.0-eager-long-failure.json): image cosine 0.998963, relative L2 4.554%. Both text cases passed; later suite checks were not reached. |
| Compiled short suite vs Transformers | Fail on image | [Failure](results-v0.29.0-compiled-short-failure.json): image cosine 0.998484, relative L2 5.516%. Both text cases passed; later suite checks were not reached. |
| Compiled long suite vs Transformers | Pass | [Metrics](results-v0.29.0-compiled-long.json): minimum cosine 0.999234, maximum relative L2 3.915%; 12,800 measured cache-hit tokens. |
| Eager long control with caching/chunking disabled vs Transformers | Pass | [Control metrics](results-v0.29.0-control-transformers.json): image cosine 0.999110, relative L2 4.219%. |
| Eager long suite vs vLLM control | Pass | [Metrics](results-v0.29.0-eager-control-long.json): minimum cosine 0.999889, maximum relative L2 1.494%; 12,800 cache-hit tokens. |
| Compiled long suite vs vLLM control | Pass | [Metrics](results-v0.29.0-compiled-control-long.json): minimum cosine 0.999885, maximum relative L2 1.519%; 12,800 cache-hit tokens. |

Passing full suites include exact input-token alignment and first generated token,
1024 finite vector components, sequential and concurrent text/image requests,
ordinary responses matching the plugin-disabled baseline, ordinary streaming,
and rejection of invalid extraction requests. The compiled server logged
`enforce_eager=False`, `DYNAMO_TRACE_ONCE`, backend `inductor`, and a saved AOT
compiled function before successful extraction. Long prompts contained 1628 text
or 1705 image tokens with a 1280-token prefill budget.

The control also verified that `max_completion_tokens=1` takes precedence when
`max_tokens=3` is supplied in the same extraction request.

These image failures remain real limitations of the comparison evidence. They
are not converted into passes or attributed conclusively to a particular kernel.
The extension returns vLLM's state; universal numerical parity with Transformers
is not established. GPU execution, CUDA graphs, GPU async scheduling, 9B and
INT4 runtime validation on this tag remain untested. Historical 9B/INT4 results
apply only to the old revision.

## Reproduction

The source-validation environment uses the existing hardware dependency stack at
`vllm/.venv`. Its original editable paths predate the repository move, so source
validation explicitly sets `PYTHONPATH` to this repository and its submodule.
This is a source-runtime check, not a clean pip-install test.

```sh
# From the extension repository root:
export PYTHONPATH="$PWD:$PWD/vllm"
vllm/.venv/bin/python -m pytest
vllm/.venv/bin/python -m ruff check .
vllm/.venv/bin/python -m ruff format --check .
```

An initial CPU rebuild encountered a stale oneDNN CMake cache referencing the
old checkout path. The retry uses a fresh dependency build directory:

```sh
# Run inside vllm/; this builds native extensions in place, not a wheel.
VLLM_TARGET_DEVICE=cpu MAX_JOBS=8 \
  CMAKE_ARGS='-DFETCHCONTENT_BASE_DIR=/tmp/hidden-state-v029-deps' \
  .venv/bin/python setup.py build_ext --inplace
```

After a successful in-place build, regenerate the source distribution metadata
with `VLLM_TARGET_DEVICE=cpu .venv/bin/python setup.py egg_info` inside `vllm/`.
This records the version from Git (`0.29.0+cpu`); no pretend-version override is
used. With the explicit source `PYTHONPATH`, `importlib.metadata` resolves that
source metadata. The older environment's installed distribution is not rewritten.
The rebuilt CPU binary hashes and exact package versions are recorded in
[environment-v0.29.0.json](environment-v0.29.0.json). Upstream retained its existing
Rust binaries; the Rust frontend and tool parsing are not exercised here.

Runtime commands (from the extension root, using the environment above):

```sh
export HF_HUB_OFFLINE=1
export VLLM_USE_RUST_FRONTEND=0
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_CPU_KVCACHE_SPACE=1
export VLLM_CPU_OMP_THREADS_BIND=nobind
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Exact cached model/tokenizer/processor snapshot used in this run:
validation_model=/home/soren/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17

vllm/.venv/bin/python validation/validate_hidden_state.py reference \
  --model "$validation_model" --reference /tmp/hidden-state-v029-reference.json
vllm/.venv/bin/python validation/validate_hidden_state.py reference \
  --model "$validation_model" --long-prompts \
  --reference /tmp/hidden-state-v029-long-reference.json

# Baseline server; run HTTP checks from another terminal with the same exports.
VLLM_PLUGINS='' vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve \
  "$validation_model" --host 127.0.0.1 --port 18329 --dtype bfloat16 \
  --distributed-executor-backend mp --max-model-len 4096 \
  --max-num-batched-tokens 1280 --max-num-seqs 4 --skip-mm-profiling \
  --enable-prefix-caching --enable-chunked-prefill --enforce-eager

vllm/.venv/bin/python validation/validate_hidden_state.py baseline \
  --model "$validation_model" --url http://127.0.0.1:18329 \
  --reference /tmp/hidden-state-v029-reference.json \
  --baseline /tmp/hidden-state-v029-baseline-short.json
vllm/.venv/bin/python validation/validate_hidden_state.py baseline \
  --model "$validation_model" --url http://127.0.0.1:18329 --long-prompts \
  --reference /tmp/hidden-state-v029-long-reference.json \
  --baseline /tmp/hidden-state-v029-baseline-long.json

# Stop the baseline, then start the extension with identical runtime arguments.
VLLM_PLUGINS='' vllm/.venv/bin/python -m vllm_last_hidden_state.serve \
  "$validation_model" --host 127.0.0.1 --port 18329 --dtype bfloat16 \
  --distributed-executor-backend mp --max-model-len 4096 \
  --max-num-batched-tokens 1280 --max-num-seqs 4 --skip-mm-profiling \
  --enable-prefix-caching --enable-chunked-prefill --enforce-eager

vllm/.venv/bin/python validation/validate_hidden_state.py http \
  --model "$validation_model" --url http://127.0.0.1:18329 \
  --reference /tmp/hidden-state-v029-reference.json \
  --baseline /tmp/hidden-state-v029-baseline-short.json \
  --report validation/results-v0.29.0-eager-short.json
vllm/.venv/bin/python validation/validate_hidden_state.py http \
  --model "$validation_model" --url http://127.0.0.1:18329 \
  --long-prompts --require-prefix-cache-hit \
  --reference /tmp/hidden-state-v029-long-reference.json \
  --baseline /tmp/hidden-state-v029-baseline-long.json \
  --report validation/results-v0.29.0-eager-long.json
```

For the compiled run, restart the extension without `--enforce-eager` and repeat
both HTTP commands with report filenames containing `compiled` instead of `eager`.
The scripts compare exact input token IDs, first generated token, vector shape
and finiteness, cosine >= 0.999 and relative L2 <= 5%, ordinary response content,
usage, and token IDs, concurrent requests, ordinary streaming, and invalid
extraction requests. The long suite also requires a measured prefix-cache hit
counter increase. No numerical thresholds are relaxed.


## Image discrepancy follow-up

To separate feature effects from the independent Transformers comparison, run the
extension in eager mode with `--no-enable-prefix-caching`,
`--no-enable-chunked-prefill`, and `--max-num-batched-tokens 4096`. For each long
case from `validation.validate_hidden_state.cases(True)`, collect the response
vector, prompt token IDs and first generated token. Use the same request options
as the HTTP helper, with `max_completion_tokens=1` and `max_tokens=3` to check
precedence. Compare these vectors to the original Transformers references.

The uncached/unchunked long image passed the threshold (cosine 0.999110), whereas
the eager cached/chunked image narrowly failed (0.998963). This is not evidence
that the Transformers discrepancy is independent of execution settings. The
feature-enabled eager suite did pass against the vLLM control at the unchanged
thresholds, including all long-case protocol, concurrency, baseline and cache-hit
checks. Comparing two vLLM runs is a feature regression check; it does not replace
an independent Transformers comparison or prove the absence of a shared bug.

For reproduction, copy the original long reference JSON, replace each case's
`vector` and `first_generated_token` with the control results, and set `backend`
to `vllm_control_eager_no_cache_no_chunk`. Keep model, prompt IDs, positions,
dtype, and `long_prompts` unchanged. Re-run the full long HTTP suite using this
reference in both eager and compiled modes, with cache/chunking enabled and
`--require-prefix-cache-hit`. Save the resulting reports with `-control-long`
in their names. The compiled short-image failure remains explicitly recorded;
this long-prompt control does not resolve that separate failure.


Both eager and compiled long control suites passed. The compiled control run
encountered an upstream AOT cache-load warning (`function` has no attribute
`finalize_loading`), recompiled, and completed successfully. All temporary servers
were stopped. Tracked upstream source remains unchanged.

The acceptance policy is therefore vLLM 0.29.0, with demonstrated CPU source-runtime
integration and the numerical limitations above. This is not certification of
all Qwen3.5 checkpoints, quantization formats, GPU modes, or clean pip installation.
