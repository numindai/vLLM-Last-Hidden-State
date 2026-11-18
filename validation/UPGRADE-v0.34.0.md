# vLLM 0.34.0 compatibility upgrade — 2026-11-18

## Revisions and environment

- Previous release: v0.33.0, `4ba787ec9959e566d3b587a796dff07673324da5`.
  Its [report](UPGRADE-v0.33.0.md) and failed numerical comparisons are preserved.
- Target: v0.34.0, `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`.
  The upstream submodule is detached at this revision, with unmodified source.
- Extension: 0.34.0, changes based on package commit
  `b3c56ea2689b8f894200c0ceba0c7cc6b3ef70a2`. Package metadata, version guard,
  tests, documentation, and submodule pin are updated together.
  The [source manifest](runs-v0.34.0/source-manifest.json) identifies the tested
  code independently of commit or publication status.
- Official vLLM 0.34.0+cpu wheel, Python 3.12.13, PyTorch 2.14.0+cpu,
  Transformers 5.22.2, Triton CPU 3.8.0+cpu, huggingface-hub 2.4.0.
  [Runtime provenance](runs-v0.34.0/runtime-environment.txt) and
  [dependency freeze](runs-v0.34.0/runtime-freeze.txt) identify the installation.
- Linux x86_64, AMD Ryzen AI MAX+ 395, 32 logical CPUs; no CUDA device.
  Model validation uses eight CPU threads and isolated localhost port 18432.
- Qwen/Qwen3.5-0.8B model/processor snapshot
  `2fc06364715b967f1860aea9cf38778875588b17`, BF16, fresh Transformers eager
  references with thinking disabled and identical assistant prefill/token IDs.

## Breaking change and contract audit

**Unset `VLLM_USE_V2_MODEL_RUNNER`.** Model Runner V1 and
`VllmConfig.use_v2_model_runner` were removed in 0.34.0. Upstream rejects every
environment value except `1`, which is redundant and warns. The launcher still
passes the environment through so upstream owns the diagnostic. Remove old `0`
settings from shell profiles, containers, and services before starting either
the launcher or direct `vllm serve`.

Removed unconditional imports of `cpu_model_runner` and `gpu_model_runner`, the
V1 CPU forward hook, V1 GPU `_model_forward` wrapper, and V1 packed-row adapter.
Both supported concrete runner classes now use the existing V2 execute adapter.
The package does not register logits processors; there is no V1 processor to
port here. Downstream custom processors must use
`vllm.v1.worker.gpu.sample.logits_processor`.

The `vllm.v1` engine namespace and `KVConnectorBase_V1` are still current APIs;
they are distinct from the removed Model Runner V1 and remain in use.

Compared the previous and target source and traced these contracts. Paths are
relative to upstream's `vllm/` Python package.

| Integration | Target source | Finding |
| --- | --- | --- |
| Endpoint initialization and route delegation | `entrypoints/launchers/api_server/app_state.py`, `entrypoints/generate/api_router.py`, `plugins/endpoint_plugins/interface.py`, `entrypoints/openai/chat_completion/api_router.py` | Chat state creation moved to `init_generate_state`; endpoint plugin initialization still follows it. The chat route resolves the wrapped handler and serializes its response. |
| CLI and runner environment | `entrypoints/cli/serve.py`, `utils/argparse_utils.py`, `platforms/interface.py:validate_environ` | Serve parser, YAML/dotted options and executor guards remain usable. New tests exercise upstream rejection of `0`, empty and invalid values, acceptance of unset/`1`, and launcher pass-through. |
| Request and completion metadata | `entrypoints/openai/chat_completion/protocol.py`, `serving.py`, `v1/request.py`, `v1/core/sched/scheduler.py` | KV parameters still reach sampling extra arguments and scheduler requests; completion carries the connector handle into the non-streaming chat response. |
| Connector lifecycle | `distributed/kv_transfer/kv_connector/v1/base.py`, `v1/worker/gpu/kv_connector.py` | New lookahead-hash/cache-access/async-load methods have defaults. Metadata-only capture requires no KV delivery or delayed block release. Extensible GPU cache remains runtime-untested. |
| Worker and RPC | `v1/worker/worker_base.py`, `gpu_worker.py`, `cpu_worker.py`, `v1/executor/multiproc_executor.py` | Post-warmup initialization RPC still reaches the concrete runner. New nonblocking response polling preserves queued RPC ordering. Capture completes before sampling; synchronous owned CPU copies remain intentional. |
| CPU/GPU target execution | `v1/worker/gpu/model_runner.py:execute_model`, `sample_tokens`, `ExecuteModelState`; `v1/worker/cpu/model_runner.py` | CPU inherits the remaining GPU runner. Eager, piecewise and full-graph outputs converge on fresh execution state before returning `None`. Sampling/drafting reads and clears that state afterward. No model forward hook is used. |
| Packed rows and request lifecycle | `prepare_inputs`, `v1/core/sched/output.py`, `scheduler.py` | Batch-order computed counts and query offsets remain available. Resumed requests are new-request data with required prefill IDs. Removed `num_common_prefix_blocks` is dropped from fixtures; finished/preempted metadata cleanup and stale/dummy guards remain. |
| Prefix cache and chunks | `v1/core/kv_cache_manager.py`, `v1/request.py:get_max_cache_hit_length` | The hit cap still recomputes at least the last token. Row selection uses prompt length and batch-order computed offsets, not request-state slots. |
| Representation and MTP | `model_executor/models/qwen3_5.py`, inherited `qwen3_next.py:forward`, `v1/worker/gpu/model_runner.py:sample_tokens`, `gpu/spec_decode/mtp/speculator.py` | Target output remains post-final-normalization. Capture precedes speculative drafting and output-buffer reuse. Only normalized speculative method `mtp` is accepted. |
| CPU and accelerator limits | `platforms/cpu.py`, remaining runner and multiprocess executor | x86 now requires Triton CPU and CPU still disables async scheduling. Source tracing and CPU tests do not establish GPU graph replay or async correctness. |

## Reproduction and fast checks

Commands run from the package root:

```sh
uv venv --python 3.12 /tmp/vllm034-runtime
uv pip install --python /tmp/vllm034-runtime/bin/python \
  'vllm==0.34.0+cpu' \
  'triton @ https://github.com/triton-lang/triton-cpu/releases/download/v3.8.0/triton-3.8.0+cpu-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl' \
  --extra-index-url https://wheels.vllm.ai/0.34.0/cpu --torch-backend cpu
uv pip install --python /tmp/vllm034-runtime/bin/python -e '.[dev]' build twine
uv pip check --python /tmp/vllm034-runtime/bin/python
HF_HUB_OFFLINE=1 /tmp/vllm034-runtime/bin/python -m pytest -q --tb=short
/tmp/vllm034-runtime/bin/ruff check .
/tmp/vllm034-runtime/bin/ruff format --check .
git diff --check
git -C vllm status --short
```

- **Pass:** 116 unit tests, one upstream-field deprecation warning:
  [test output](runs-v0.34.0/unit-tests.txt). Fixtures reject imports of the deleted
  V1 modules and cover CPU/GPU installation, text/multimodal wrappers, target-only
  capture, packed offsets, chunking, owned vectors, cleanup and version policy.
- **Pass:** [dependency consistency](runs-v0.34.0/dependency-check.txt).
  [Runtime installation](runs-v0.34.0/runtime-install.txt) and
  [development installation](runs-v0.34.0/dev-install.txt) are recorded separately.
- **Pass:** all 11 audited upstream integration files match the official wheel
  byte for byte: [source/wheel audit](runs-v0.34.0/upstream-wheel-audit.json).

Download the pinned model before replaying runtime commands:

```sh
/tmp/vllm034-runtime/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17', local_dir='/tmp/vllm034-model', allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'])"
```

[Model file hashes](runs-v0.34.0/model-manifest.json) identify the snapshot.
[Reference commands](runs-v0.34.0/reference-commands.json) and the per-mode
`*-commands.json` records contain the server/client argument arrays, environment,
readiness, and exit statuses. Replace the recorded checkout prefix with your own
checkout path. Unset `VLLM_USE_V2_MODEL_RUNNER` and `PYTHONPATH` for every run.
Run the ordinary eager/compiled/MTP baselines first, then plugin eager, compiled,
and eager MTP. The eager plugin mixed suite creates the non-MTP control snapshot
used by the compiled and MTP suites. Start a fresh server per mode, wait for its
health endpoint, run its clients in order, and stop it before the next mode.

## Runtime validation

All six server configurations reached readiness and stopped with exit code zero.
The obsolete runner environment variable was unset throughout. Mode filenames
retain `v2` to identify the capture path; there is no V1 runner configuration.
[Matrix records](runs-v0.34.0/matrix.json) and
[summary metrics](runs-v0.34.0/summary.json) preserve successes and failures.
Public text logs normalize line endings and trailing whitespace only; original
logs remain in local workspace evidence outside the package.

| Check | Result |
| --- | --- |
| Plugin-disabled eager, compiled and eager-MTP baselines | **Pass**, short and long inputs |
| Eager, compiled and eager-MTP long HTTP suites | **Pass**, exact prompt IDs, 1024 finite vector elements, first-token and usage checks, concurrency, streaming, invalid-request rejection, chunking and measured cache hits |
| Ordinary responses vs matching plugin-disabled baselines | **Pass** in all three long suites; choices, usage and prompt IDs match |
| Mixed ordinary/extraction traffic and repeated requests | **Pass** in all three plugin modes, including `max_completion_tokens=1` |
| Compiled target vs eager control | **Pass**, minimum cosine 0.999901, maximum relative L2 1.410% |
| Eager MTP target vs non-MTP control | **Pass**, minimum cosine 0.999854, maximum relative L2 1.711%; 49 drafted / 27 accepted tokens |
| Short-image comparison to fresh Transformers references | **Fail** in all three modes; thresholds unchanged and failures retained |
| GPU eager, CUDA graphs and async scheduling | **Untested**, no CUDA device; CPU disables async scheduling |
| INT4, other checkpoints, compiled MTP, real preemption and live cancellation | **Untested** on 0.34.0 |

Long prompts contain 1,628 text or 1,705 image-plus-text tokens, exceeding the
1,280-token prefill budget. Long suites record 14,080 cache-hit tokens in eager
and compiled modes, and 7,040 with MTP. Mixed suites record 6,400 non-MTP or
3,200 MTP cache-hit tokens. The compiled worker logs confirm
`DYNAMO_TRACE_ONCE`, backend `inductor`, and a saved AOT function; CUDA graph mode
is `NONE`. Frontend warning messages about disabled Inductor do not describe the
worker's effective configuration. These checks do not establish graph replay.

Numerical thresholds remain cosine >= 0.999 and relative L2 <= 0.05.
Long extrema include sequential and concurrent comparisons:

| Configuration | Long minimum cosine | Long maximum relative L2 | Short-image cosine | Short-image relative L2 |
| --- | --- | --- | --- | --- |
| Eager | 0.999517 | 3.110% | 0.998610 | 5.286% |
| Compiled | 0.999551 | 2.998% | 0.998513 | 5.462% |
| Eager MTP | 0.999526 | 3.083% | 0.998614 | 5.276% |

The short-image case has 105 matching input tokens, final prompt position 104,
correct vector shape and first generated token. Both tolerances fail, matching
the previously recorded behavior. The underlying numerical cause remains
unresolved. Short clients exit with status 1 before writing a final report;
their `*-short.txt` logs retain the per-case metrics and assertions. A missing
short JSON is not a successful result. Passing long suites and same-vLLM
controls does not establish full Transformers image parity or retrieval quality.

## Packaging checks

```sh
/tmp/vllm034-runtime/bin/python -m build --no-isolation
/tmp/vllm034-runtime/bin/python -m twine check --strict dist/vllm_last_hidden_state-0.34.0*
```

**Pass:** [Ruff and formatting](runs-v0.34.0/lint.txt),
[wheel/source build](runs-v0.34.0/build.txt),
[strict Twine validation](runs-v0.34.0/twine.txt), and distribution boundaries.
The [artifact manifest](runs-v0.34.0/artifacts.json) records hashes and contents;
neither distribution includes the upstream submodule or validation directory.

## Remaining scope

GPU eager execution, CUDA graphs, async scheduling, compiled MTP, INT4 and other
checkpoints remain untested on this release. Real preemption and live client
cancellation remain untested; unit lifecycle/error tests are separate evidence.
LoRA, distributed workers, non-MTP speculation, other model families and the Rust
frontend remain unsupported. No upstream core source changes are required.
This report validates only the separately installable package, not PR #57185.
