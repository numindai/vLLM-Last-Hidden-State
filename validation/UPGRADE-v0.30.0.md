# vLLM 0.30.0 compatibility report

Date: 2026-09-23. Historical [0.29.0](UPGRADE-v0.29.0.md) and
[MTP](MTP-v0.29.0.md) results remain unchanged and do not certify this release.

## Revisions and environment

- Starting extension commit: `d188f611372224e27ed3bacc2759fd4cdfed17c7` (`0.1.0`); upgrade branch:
  `codex/vllm-0.30.0`. The package changes and this report are the upgrade artifact.
- Previous upstream: `98dff2a81d747d1dba01a47f939f48c3526d4206` (v0.29.0).
- Target: tag `v0.30.0`, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
  `git describe --always --dirty` returns `ced6857` because the release tag is
  lightweight; tracked upstream source is unmodified.
- Environment: Python 3.12.13, PyTorch 2.13.0+cpu, Transformers 5.17.0;
  inherited `vllm/.venv`, with explicit source `PYTHONPATH` because its old vLLM
  editable registration points to a previous repository location.
- Hardware: Linux x86_64, AMD Ryzen AI MAX+ 395, approximately 121 GiB RAM;
  CPU PyTorch, no available CUDA device. GPU, CUDA graphs, and async scheduling
  are **untested**. Upstream still forces CPU async scheduling off.
- BF16 checkpoint/tokenizer/processor: `Qwen/Qwen3.5-0.8B`, cached snapshot
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Original BF16 checkpoint/tokenizer/processor: `Qwen/Qwen3.5-9B`, snapshot
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- INT4 checkpoint/tokenizer: `drawais/Qwen3.5-9B-AWQ-INT4`, cached snapshot
  `b3b888195917f6224e3a97e84e7f64c6b1ec8731`. The configuration identifies
  compressed-tensors; no forced AWQ loader. Transformers reference uses
  `--text-only --dequantize`, server/helper use `--text-only`.
- References are freshly generated with Transformers, BF16, including final
  output normalization; all chat requests explicitly use
  `chat_template_kwargs={"enable_thinking": False}`. Text/image cases include
  unfinished assistant messages with `continue_final_message=True` and
  `add_generation_prompt=False`.

## Contract audit and changes

Paths below are relative to the pinned upstream checkout. Compared v0.29.0 to
v0.30.0 and read target callers, rather than relying on import compatibility.

| Integration | Target source inspected | Findings and package changes |
| --- | --- | --- |
| Endpoint loading and Python frontend | `vllm/plugins/__init__.py`; `entrypoints/cli/serve.py`; `entrypoints/launchers/api_server/app_state.py:init_app_state`; `entrypoints/openai/chat_completion/api_router.py:chat/create_chat_completion` | Endpoint plugins still need explicit allowlisting. Initialization follows `init_generate_state`; the route resolves the replaced `state.openai_serving_chat` and serializes the returned response. New tool-server/template initialization does not remove this hook. Preserve Python frontend and non-endpoint plugin defaults. |
| CLI conflict checks | `utils/argparse_utils.py:FlexibleArgumentParser`; `entrypoints/cli/serve.py:ServeSubcommand` | Parser supports the same worker/connector flags, aliases, dotted JSON, YAML, and CLI precedence. Existing tests exercise the actual target parser. Keep launcher pass-through and the two-token YAML spelling adaptation. |
| Request metadata and completion | `entrypoints/openai/chat_completion/protocol.py:to_sampling_params`; `v1/request.py`; `v1/core/sched/scheduler.py:_free_request/_connector_request_finished`; `v1/engine/__init__.py:EngineCoreOutput`; `entrypoints/openai/chat_completion/serving.py` | `kv_transfer_params` reaches sampling `extra_args` and request state; finished request metadata reaches the normal chat response. Per-request server handles and one-shot retrieval remain unchanged. |
| Connector lifecycle and HMA | `distributed/kv_transfer/kv_connector/v1/base.py`; `kv_connector/factory.py`; `v1/worker/kv_connector_model_runner_mixin.py` | New `get_transfer_results` adapts inherited empty `get_finished`; partial-tail registration and pending frees default to false. Our connector sends no KV, claims no external tokens, and delays no block frees. A regression test checks handle return separately from transfer results. New `finish_forward`/`reset_capture_state` hooks receive no hidden states and do not replace the capture boundary. |
| Worker initialization and RPC | `v1/worker/worker_base.py`; `v1/executor/multiproc_executor.py:collective_rpc/worker_busy_loop/_execute_worker_rpc/handle_output` | Extension methods still reject collisions. Initialization runs after engine warmup. Worker main loop executes RPCs serially; async output conversion runs on a separate queue. Capture owns its CPU vector before returning from model execution; retrieval still returns one result per local worker. CUDA ordering remains source-level evidence only. |
| CPU capture and normalization | `v1/worker/cpu_model_runner.py:load_model/get_model/warming_up_model`; `model_executor/models/qwen3_5.py`; `qwen3_next.py:Qwen3NextModel.forward` | CPU runner retains its outer model. Qwen3.5 inherits the forward that applies final `self.norm`, and text/multimodal wrappers return those states before logits. Attach the outer hook after warmup, outside the compiled decoder. |
| GPU capture, graphs, ownership | `v1/worker/gpu_model_runner.py:_model_forward/execute_model/load_model`; `compilation/cuda_graph.py` | Real execution still calls `_model_forward`, which invokes the model including graph wrappers. Capture runs on return, before logits indexing and drafting. Updated nesting of breakable/piecewise and full graph wrappers does not bypass this boundary. Retain synchronous owned float32 CPU copy; no CUDA runtime claim. |
| Packed rows, cache, chunks, preemption | `v1/worker/gpu_model_runner.py:_update_states/_prepare_inputs`; `v1/core/kv_cache_manager.py:get_computed_blocks`; `v1/core/sched/async_scheduler.py` | CPU computed offsets and cumulative scheduled query spans still share batch order. Prefix hits are capped at `request.num_tokens - 1`. Preemption restores request state and blocks; capture can replace an earlier result. Async speculative CPU lengths are optimistic after decoding starts; the final prompt capture precedes that phase and later spans are skipped. This reasoning still needs GPU validation. |
| MTP target/draft isolation | `v1/spec_decode/eagle.py`; `llm_base_proposer.py`; `model_executor/models/qwen3_5_mtp.py`; `gpu_model_runner.py:execute_model/sample_tokens`; `cpu_model_runner.py` | Drafter invokes a separate outer `self.model`; shared embeddings/head do not share the target capture hook. Target copy precedes draft execution and deferred connector finalization. Some CPU fallbacks moved to dispatcher `.kernel` attributes; other fallbacks still replace old module symbols, leaving new dispatchers unbound. Runtime tests below expose this upstream regression. |
| Runtime policy and runner selection | `config/vllm.py:use_v2_model_runner`; `config/speculative.py`; `platforms/cpu.py`; `v1/worker/gpu_worker.py` | `VLLM_USE_V2_MODEL_RUNNER=0` still chooses V1 for supported Qwen3.5 configurations. HiSparse and watermarking can require V2, which the concrete runner guard rejects. Preserve mp/single-worker/CPU-or-CUDA/compilation/LoRA guards and normalized `mtp` method; dtype and quantization stay with upstream. |

The exact supported-version constant, missing-installation message, and version
regression cases now target 0.30.0 (including local hardware suffixes). Other
releases and rc/dev/post builds remain rejected. No extraction algorithm,
request/response contract, performance default, or tracked upstream file changed.
No new upstream hidden-state API replaces the current narrow capture adapters;
`extract_hidden_states` remains a speculative decoding mode, not a passive
one-row post-normalization callback.

## Build and fast-check evidence

The inherited 0.29.0 native binaries are not used as proof of a 0.30.0 build.
The first build could not resolve GitHub in the network sandbox. The network-enabled
retry fetched oneDNN but found old CMake header paths under the deleted
`/tmp/hidden-state-v029-deps`. The next attempt uses a fresh build directory:

```sh
# Inside vllm/:
VLLM_TARGET_DEVICE=cpu MAX_JOBS=8 \
  CMAKE_ARGS='-DFETCHCONTENT_BASE_DIR=/tmp/hidden-state-v030-deps' \
  .venv/bin/python setup.py build_ext --build-temp build/temp-v030 --inplace
# Only after a successful native build:
VLLM_TARGET_DEVICE=cpu .venv/bin/python setup.py egg_info
```

The extension was reinstalled successfully with:

```sh
UV_CACHE_DIR=/tmp/v030-uv-cache uv pip install --python vllm/.venv/bin/python \
  --no-deps --no-build-isolation -e .
```

Model-free validation uses the target Python source, real upstream parser and
connector base, and small fixtures for capture/model hooks:

```sh
export PYTHONPATH="$PWD:$PWD/vllm"
vllm/.venv/bin/python -m pytest -q
vllm/.venv/bin/python -m ruff check .
vllm/.venv/bin/python -m ruff format --check .
git diff --check
```

86 tests pass; [unit output](runs-v0.30.0/unit-tests.txt). These tests do not
establish graph replay, hardware async correctness, or real preemption. No
package-level pre-commit configuration is present; upstream repository hooks
were not installed or run because no upstream source was edited.

CPU build completed successfully (exit 0); [build output](runs-v0.30.0/build.txt).
The failed [sandbox download](runs-v0.30.0/build-sandbox-failure.txt) and
[stale CMake cache](runs-v0.30.0/build-stale-cache-failure.txt) attempts are retained.
[Environment and native hashes](environment-v0.30.0.json) identify the rebuilt
artifacts and source metadata (`0.30.0+cpu`). Setup retained existing Rust
artifacts; neither Rust frontend nor tool parsing was validated. This is a source
runtime check, not a clean wheel-install certification.

## Runtime reproduction

Run from the package root. The `*-commands.json` files in
[runs-v0.30.0](runs-v0.30.0) retain exact executed argument arrays, environment
settings, and helper exit codes. `*-server.txt` files retain startup, actual
compilation/kernel selection, HTTP outcomes, and shutdown output. Failed helpers
retain their per-case metrics and traceback even when no final JSON is written.
Each server is stopped before the next configuration on its port. Original
BF16 9B checks use port 18331, independently of INT4/0.8B checks on port 18330.
No unrelated traffic reaches either server or satisfies its cache-hit checks.
Some CPU work overlaps; recorded timings are not a performance benchmark.

```sh
export PYTHONPATH="$PWD:$PWD/vllm"
export HF_HUB_OFFLINE=1 VLLM_USE_RUST_FRONTEND=0 VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_CPU_KVCACHE_SPACE=1 VLLM_CPU_OMP_THREADS_BIND=nobind
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_PLUGINS=''
# VLLM_CPU_CI_ENV and TORCH_COMPILE_DISABLE are unset.
validation_model=/home/soren/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
vllm/.venv/bin/python validation/validate_hidden_state.py reference \
  --model "$validation_model" --reference /tmp/v030-reference.json
vllm/.venv/bin/python validation/validate_hidden_state.py reference \
  --model "$validation_model" --long-prompts --reference /tmp/v030-reference-long.json

# Baseline server (eager); run helpers separately while it is ready.
vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve "$validation_model" \
  --host 127.0.0.1 --port 18330 --dtype bfloat16 \
  --distributed-executor-backend mp --max-model-len 4096 \
  --max-num-batched-tokens 1280 --max-num-seqs 4 --skip-mm-profiling \
  --enable-chunked-prefill --enable-prefix-caching --enforce-eager
vllm/.venv/bin/python validation/validate_hidden_state.py baseline \
  --model "$validation_model" --url http://127.0.0.1:18330 \
  --reference /tmp/v030-reference.json --baseline /tmp/v030-baseline-eager-short.json
vllm/.venv/bin/python validation/validate_hidden_state.py baseline \
  --model "$validation_model" --url http://127.0.0.1:18330 --long-prompts \
  --reference /tmp/v030-reference-long.json --baseline /tmp/v030-baseline-eager-long.json

# Stop baseline; replace its module with vllm_last_hidden_state.serve and
# omit the separate "serve" subcommand; retain every runtime option above.
vllm/.venv/bin/python validation/validate_hidden_state.py http \
  --model "$validation_model" --url http://127.0.0.1:18330 \
  --reference /tmp/v030-reference.json --baseline /tmp/v030-baseline-eager-short.json \
  --report validation/runs-v0.30.0/eager-short.json
vllm/.venv/bin/python validation/validate_hidden_state.py http \
  --model "$validation_model" --url http://127.0.0.1:18330 --long-prompts \
  --require-prefix-cache-hit --reference /tmp/v030-reference-long.json \
  --baseline /tmp/v030-baseline-eager-long.json \
  --report validation/runs-v0.30.0/eager-long.json
```

Compiled runs omit `--enforce-eager` and use separate `compiled` baseline/report
names. INT4 uses its pinned snapshot, references generated with
`--text-only --dequantize`, and `--text-only` for baseline/HTTP helpers.
The numerical thresholds remain cosine >= 0.999 and relative L2 <= 0.05.
No tolerance changes turn failures into passes.

## CPU MTP regression in upstream 0.30.0

The tested CPU backend has no Triton-CPU. Do not carry forward the working
0.29.0 MTP claims to this release:

- With two speculative tokens, eager extraction fails at the first request with
  `TypeError: Cannot inspect kernel parameters for function` in
  `_eagle_step_slot_mapping_metadata` through the new Triton warmup dispatcher.
  [Extension failure](runs-v0.30.0/mtp-eager-server.txt).
  [Plugin-disabled ordinary baseline](runs-v0.30.0/baseline-mtp-server.txt)
  reproduces the same failure on an ordinary one-token completion, without the
  connector, worker extension, or capture hook.
- With one speculative token and caching disabled, six short/long text/image
  extractions returned finite vectors and one generated token. The helper then
  failed on ordinary 16-token generation in `_rejection_greedy_sample`, with the
  same dispatcher error. Thus the complete suite, numerical control comparison,
  streaming, and mixed-traffic checks did **not** pass.
  [Helper failure](runs-v0.30.0/mtp-single-eager.txt),
  [worker traceback](runs-v0.30.0/mtp-single-eager-server.txt).
- Compiled MTP startup fails during model warmup, before endpoint initialization
  installs capture, for both [two tokens](runs-v0.30.0/mtp-compiled-server.txt)
  and [one token](runs-v0.30.0/mtp-single-compiled-server.txt).
- Prefix-cached MTP additionally fails in
  `mamba_utils.run_fused_precopy:precopy_mamba_align_fused_kernel[grid]` with
  `TypeError: 'function' object is not subscriptable`.
  [Failure](runs-v0.30.0/mtp-prefix-server.txt).
  Disabling caching alone no longer suffices to run CPU MTP on this build.

Source diagnosis: `CPUModelRunner._postprocess_triton` redirects several new
kernel owners through `.kernel`, but still replaces old module-level symbols
for the EAGLE step metadata and rejection sampler. New dispatcher objects retain
the original non-CPU kernels. The existing native CPU fallback kernels are still
present. An upstream repair would need to bind the actual dispatcher owners and
reconcile parameter names (for example CPU `stride` versus dispatcher
`block_table_stride`), then retest greedy/random rejection, warmup, and Mamba
prefix precopy. A single import rename or disabling JIT warmup does not repair
runtime dispatch. No speculative sampler/core patch or silent launcher workaround
was added to this extension. A matching Triton-CPU build is another possible
validation environment, **not a verified workaround**.

MTP target/draft isolation remains implemented and covered by model-free checks.
GPU MTP remains untested. These upstream failures do not establish a GPU failure
and do not justify disabling caching, compilation, async scheduling, or MTP
unconditionally in the launcher.

## Control and protocol checks

For the 0.8B control, extraction stayed enabled while caching/chunking were
explicitly disabled and the token budget raised to 4096. The existing
`validate_mtp.py` helper (without `--require-mtp`) recorded six short/long
extractions, ordinary generation with a 16-token limit, streaming, and concurrent
mixed traffic.
Its [report](runs-v0.30.0/control.json) passes; this control has no MTP configured.
All six control vectors independently pass the original Transformers thresholds:
[metrics](runs-v0.30.0/control-transformers.json).

To reproduce the control references, copy the corresponding fresh Transformers
reference JSON, replace each case's `vector` and `first_generated_token` with the
control's `extracted` response at index `case` (short) or `case + 3` (long), assert
identical prompt IDs, and set `backend` to
`vllm_control_eager_no_cache_no_chunk`. Re-run the standard HTTP helper against
feature-enabled eager/compiled servers using those references and the ordinary
baselines. The recorded `eager-control-*` and `compiled-control-*` command files
show the exact helper invocations. Both full short and long suites pass; each
long suite measures 12,800 cache-hit tokens. This is a feature regression check,
not independent Transformers parity, and a shared capture bug could affect both
vLLM runs.

A separate [HTTP precedence check](runs-v0.30.0/token-limit-precedence.json)
sent `max_completion_tokens=1` with `max_tokens=3`, the normal short assistant
prefill request, explicit `enable_thinking=False`, and extraction enabled. It
returned one completion token and a 1024-component vector. Full passing suites
also compare ordinary choices, usage, and prompt token IDs with plugin-disabled
baselines, test ordinary streaming, and reject invalid extraction requests.

Actual client disconnect/fault-injected RPC recovery and real scheduler
preemption/recomputation were **not exercised** in this upgrade. Cancellation and
recomputation fixtures pass, but are not runtime evidence for those cases.
The 0.8B control does not diagnose the separate 9B long-image comparison failure;
a same-9B uncached/unchunked control remains a follow-up if that numerical case
must meet the Transformers threshold.

## Numerical results

[Machine-readable summary](runs-v0.30.0/numerical-summary.json). Minima and maxima
include concurrent repeats in complete suites. Failed suites include only cases
reached before the first assertion; they do not certify later suite checks.
All reached numerical comparisons first passed exact prompt IDs, final token
position, first generated token, usage, vector shape, and finiteness.

| Suite | Status | Minimum cosine | Maximum relative L2 | Measured cache-hit tokens |
| --- | --- | --- | --- | --- |
| [eager-short](runs-v0.30.0/eager-short.json) | **pass** | 0.999007 | 4.467% | not asserted |
| [eager-long](runs-v0.30.0/eager-long-failure.json) | **fail** | 0.998963 | 4.554% | not asserted |
| [compiled-short](runs-v0.30.0/compiled-short-failure.json) | **fail** | 0.998484 | 5.516% | not asserted |
| [compiled-long](runs-v0.30.0/compiled-long.json) | **pass** | 0.999234 | 3.915% | 12800 |
| [eager-bf16-9b-short](runs-v0.30.0/eager-bf16-9b-short.json) | **pass** | 0.999152 | 4.121% | not asserted |
| [eager-bf16-9b-long](runs-v0.30.0/eager-bf16-9b-long-failure.json) | **fail** | 0.998290 | 5.850% | not asserted |
| [compiled-bf16-9b-short](runs-v0.30.0/compiled-bf16-9b-short.json) | **pass** | 0.999173 | 4.074% | not asserted |
| [compiled-bf16-9b-long](runs-v0.30.0/compiled-bf16-9b-long-failure.json) | **fail** | 0.998397 | 5.663% | not asserted |
| [eager-int4-short](runs-v0.30.0/eager-int4-short.json) | **pass** | 0.999877 | 1.579% | not asserted |
| [compiled-int4-short](runs-v0.30.0/compiled-int4-short.json) | **pass** | 0.999770 | 2.159% | not asserted |
| [eager-control-short](runs-v0.30.0/eager-control-short.json) | **pass** | 1.000000 | 0.000% | not asserted |
| [eager-control-long](runs-v0.30.0/eager-control-long.json) | **pass** | 0.999889 | 1.494% | 12800 |
| [compiled-control-short](runs-v0.30.0/compiled-control-short.json) | **pass** | 0.999827 | 1.862% | not asserted |
| [compiled-control-long](runs-v0.30.0/compiled-control-long.json) | **pass** | 0.999859 | 1.686% | 12800 |

BF16 0.8B vectors have 1024 components; BF16 9B and INT4 vectors have 4096.
Short text/image prompts contain 28/105 tokens; long text/image prompts contain
1628/1705 tokens with a 1280-token batch budget. INT4 checks use short text only.
All four failed suites fail on the image case after both text cases passed.
The 0.8B eager-long and compiled-short values reproduce the previously recorded
0.29.0 discrepancies, but remain failures at the unchanged thresholds.
No general numerical equivalence to Transformers is claimed.

## Validation matrix and conclusion

| Check | Status | Evidence / limits |
| --- | --- | --- |
| Unit tests, lint/format, whitespace | **pass** | 86 tests; [unit output](runs-v0.30.0/unit-tests.txt), [Ruff](runs-v0.30.0/lint.txt); `git diff --check` clean. Upstream hooks not run; upstream source unmodified. |
| CPU BF16 eager, text + image | **pass/fail** | 0.8B and 9B short suites pass; both long-image Transformers comparisons fail. |
| CPU BF16 compiled, text + image | **pass/fail** | 0.8B long and 9B short suites pass; 0.8B short-image and 9B long-image comparisons fail. Logs record `DYNAMO_TRACE_ONCE`, `inductor`, saved AOT functions, and completed warmup. |
| CPU INT4 eager and compiled, text | **pass** | Both full short suites pass; logs select `CPUWNA16LinearKernel for CompressedTensorsWNA16`. Long INT4 and INT4 images untested. |
| Cold long prompts spanning chunks | **pass/fail** | 1628/1705 tokens exceed 1280 budget; reached text comparisons pass; 0.8B compiled full suite passes; other image failures above. |
| Real repeated prefix-cache hits | **pass** | 12,800 measured tokens in 0.8B compiled Transformers and eager/compiled long control suites; no extrapolation to failed 9B long suites. |
| Ordinary output vs plugin-disabled baseline | **pass** | Every complete passing HTTP suite matches choices, usage, and prompt IDs, including one- and three-token completions. |
| Concurrent distinct prompts and ordinary streaming | **pass** | Complete passing BF16/INT4/control suites; no claim for portions of failed suites not reached. |
| Invalid extraction and token-limit precedence | **pass** | Passing suites reject invalid opt-ins; separate real HTTP precedence report linked above. |
| Cancellation/failure cleanup and subsequent requests | **untested at runtime** | Existing cancellation fixture passes. No disconnect or RPC fault injection. Fatal upstream MTP failures are recorded as failures, not successful cleanup tests. |
| Real preemption/recomputation | **untested** | Recomputed-row ownership fixture passes; no forced scheduler preemption. |
| Mixed ordinary/extraction traffic | **pass without MTP** | 0.8B control suite. MTP suite does not reach mixed traffic. |
| CPU MTP eager/compiled, cache on/off | **fail** | Dispatcher regressions, reproduced without extension; prefix precopy also fails. One-token extraction alone is insufficient to certify ordinary multi-token MTP. |
| GPU eager | **untested** | No CUDA device/runtime in this environment. |
| GPU compilation and actual CUDA graph replay | **untested** | Model-free wrapper test is not graph execution. |
| GPU async scheduling with caching/chunking, including MTP | **untested** | CPU explicitly disables async scheduling. Source audit does not substitute for hardware validation. |

The package now targets exactly vLLM 0.30.0 with demonstrated CPU source-runtime
integration and the numerical/upstream limitations above. Supported runtime shape
remains Qwen3.5, one local mp worker, V1 CPU or CUDA runner, no LoRA/distributed
parallelism/other connectors, and non-streaming one-token opt-in extraction.
MTP remains accepted at the package boundary, with target-only capture preserved;
CPU MTP on the tested non-Triton backend is not a working configuration.

Remaining work is explicit: repair/validate upstream CPU speculative dispatcher
fallbacks and Mamba precopy before claiming CPU MTP; diagnose 9B long-image
numerics with a same-checkpoint uncached/unchunked control; obtain real CUDA
execution/graph replay/async evidence; and exercise disconnect/RPC recovery and
real preemption. NuExtract3-W4A16 and other checkpoints are not certified by the
new Qwen/drawais runs. No failure is converted into a pass by historical results.

README compatibility/installation notes and the maintenance contract map were
updated. All historical reports remain intact. Only the upstream submodule pin
changed; no tracked vLLM source, sampler, kernel, parser, or packaging file was
patched. The launcher still forwards caller settings unchanged except its
existing activation/Python-frontend handling.

All temporary validation servers were stopped; ports 18330 and 18331 were
verified closed. Final upstream `git status --short` is empty. Package changes
and new evidence remain in the working tree on `codex/vllm-0.30.0`; no tag,
push, or upstream PR was created.
