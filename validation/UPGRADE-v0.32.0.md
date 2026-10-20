# vLLM 0.32.0 compatibility upgrade — 2026-10-20

## Revisions and environment

- Previous release: v0.31.0, `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
  Its [report](UPGRADE-v0.31.0.md), including failed short-image comparisons,
  remains historical evidence.
- Target: v0.32.0, `5df63a7203cdb9328d824a27b682e77c7b9928be`.
  The upstream submodule is detached at this revision, with no source edits.
- Extension: 0.32.0, local changes above package commit
  `06fbf45caa7e6b3197ef111a1eb53105efc2a86c`. Metadata, version policy,
  tests, documentation, and submodule updated together. Nothing published.
- Runtime validation uses a new isolated environment with the official
  v0.32.0 CPU wheel, not the sibling native PR checkout. Exact installed
  versions and import location are in [runtime metadata](runs-v0.32.0/runtime-environment.txt)
  and the [dependency freeze](runs-v0.32.0/runtime-freeze.txt).
- Host: Linux x86_64, AMD Ryzen AI MAX+ 395, 32 logical CPUs. No CUDA device.
  CPU runs use eight threads; existing benchmark servers are not modified.

## Contract audit and changes

Compared the pinned 0.31.0 and 0.32.0 source and traced the following callers.
Paths below are relative to upstream's `vllm/` Python package.
No capture/transport redesign or upstream source changes are required.

| Integration | Source inspected | Finding |
| --- | --- | --- |
| Endpoint initialization and delegation | `plugins/endpoint_plugins/interface.py`, `entrypoints/launchers/api_server/app_state.py`, `entrypoints/generate/api_router.py` | Plugin state still initializes after chat state. New route-conflict resolution does not affect this plugin's empty `attach_router`; existing chat routes resolve the wrapped handler. |
| CLI and guards | `entrypoints/cli/serve.py`, `utils/argparse_utils.py`, `config/kv_transfer.py` | YAML/dotted JSON and precedence remain compatible. Upstream removed `kv_buffer_size`; the preservation test now uses `kv_connector_extra_config`. Version policy accepts only 0.32.0 plus optional local hardware suffixes. Runner selection remains caller-owned. |
| Request/response metadata | `entrypoints/openai/chat_completion/protocol.py`, `v1/request.py`, `v1/core/sched/scheduler.py` | KV parameters still reach request/sampling metadata; final completion carries the connector handle. |
| Connector lifecycle | `distributed/kv_transfer/kv_connector/v1/base.py`, `v1/worker/gpu/kv_connector.py` | No new required abstract methods. New cache-source and loaded-group defaults fit a connector reporting zero external tokens. V2 now derives load kwargs from InputBatch; variadic no-op load methods remain compatible. |
| Worker/RPC | `v1/worker/worker_base.py`, `v1/executor/multiproc_executor.py` | Extension injection and sequential single-worker RPC dispatch preserved. New shared-memory arena and dummy-batch timeout do not change result retrieval. |
| V1 target capture | `v1/worker/cpu_model_runner.py`, `v1/worker/gpu_model_runner.py` | CPU outer-model hook and GPU instance `_model_forward` wrapper still surround target execution outside inner compilation/graph replay and before drafting. |
| V2 target capture | `v1/worker/gpu/model_runner.py`, `v1/worker/cpu/model_runner.py` | CPU inherits GPU V2. Fresh ExecuteModelState retains target hidden states and InputBatch until sampling clears it. New `dp_sync_state` and speculative-budget fields are not used by the adapter. |
| Packed rows and request lifecycle | `v1/worker/gpu/model_runner.py:prepare_inputs`, `v1/core/sched/scheduler.py` | Computed-token arrays remain gathered into batch order. Resumed V2 requests are resent as NewRequestData. Adapter finish/preemption removal, re-registration and stale/no-forward rejection remain applicable. |
| Prefix cache/chunks | `v1/core/kv_cache_manager.py:get_computed_blocks` | Cache hit still capped at `request.num_tokens - 1`, forcing the final prompt token to execute. |
| Representation/MTP | `model_executor/models/qwen3_5.py`, inherited `qwen3_next.py:forward`, V1 EagleProposer, V2 speculator | Qwen configuration classes move to Transformers. Target output remains after final norm. V2 drafter-state selection was factored into `get_drafter_hidden_states`; capture precedes it and drafting. |
| Compilation/async | V1/V2 execution boundaries, `platforms/cpu.py`, multiprocess dispatch | Owned synchronous CPU copy retained. CPU still disables async scheduling. New stock-torch compile mode remains outside the package's existing accepted modes. GPU graph replay and async execution need independent runtime evidence. |

## Installation and fast checks

Commands run from the package root:

```sh
uv venv --python ../vllm/.venv/bin/python /tmp/vllm032-runtime
uv pip install --python /tmp/vllm032-runtime/bin/python \
  'https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcpu-cp38-abi3-manylinux_2_39_x86_64.whl' \
  'https://github.com/triton-lang/triton-cpu/releases/download/v3.8.0/triton-3.8.0+cpu-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl' \
  --torch-backend cpu
uv pip install --python /tmp/vllm032-runtime/bin/python -e '.[dev]' build twine
uv pip check --python /tmp/vllm032-runtime/bin/python
HF_HUB_OFFLINE=1 /tmp/vllm032-runtime/bin/python -m pytest -q --tb=short
/tmp/vllm032-runtime/bin/ruff check .
/tmp/vllm032-runtime/bin/ruff format --check .
```

- **Pass:** 112 tests, 15 warnings against the matching installed CPU wheel:
  [unit output](runs-v0.32.0/runtime-unit-tests.txt).
- **Pass:** dependency consistency:
  [installation](runs-v0.32.0/runtime-install.txt),
  [extension installation](runs-v0.32.0/extension-install.txt),
  [dependency check](runs-v0.32.0/dependency-check.txt).
- The initial source-only check with older native artifacts failed one test
  because its fixture supplied upstream's removed `kv_buffer_size` field:
  [preserved output](runs-v0.32.0/source-unit-tests.txt).
  The fixture was corrected before the matching-wheel run. Source-only results
  using an older native build are not runtime certification.
- **Pass:** Ruff check/format and whitespace checks:
  [lint output](runs-v0.32.0/lint.txt).
- **Pass:** wheel/sdist build and strict Twine checks:
  [build](runs-v0.32.0/build.txt), [Twine](runs-v0.32.0/twine.txt).
  [Artifact hashes/manifests](runs-v0.32.0/artifacts.json) record the built files;
  neither distribution includes the upstream submodule or validation artifacts.

```sh
/tmp/vllm032-runtime/bin/python -m build --no-isolation
/tmp/vllm032-runtime/bin/python -m twine check --strict dist/vllm_last_hidden_state-0.32.0*
git diff --check
git -C vllm status --short
```

## Runtime method

Model and processor: Qwen/Qwen3.5-0.8B snapshot
`2fc06364715b967f1860aea9cf38778875588b17`, BF16, native vLLM implementation.
Fresh Transformers eager references use the same local snapshot, explicit
thinking-disabled chat template and assistant prefill, without changing the
comparison thresholds (cosine >= 0.999; relative L2 <= 0.05).

Each `*-commands.json` in [run artifacts](runs-v0.32.0) records the actual
environment, server/client argument arrays and outcomes. References and vector
snapshots are durable files in that directory. Servers bind only port 18432
and stop after each configuration. Long prompts exceed the 1280-token batch
budget; model length is 4096, with four sequences, prefix caching and chunking
enabled. Plugin-disabled baselines use the same configuration as their
corresponding V2 plugin run.

## Runtime results

| Check | Result |
| --- | --- |
| Plugin-disabled V2 eager/compiled/eager-MTP baselines | **Pass**, short and long inputs |
| V1 eager and V2 eager/compiled/eager-MTP long HTTP suites | **Pass**, exact input IDs, finite 1024-element vectors, first tokens, usage, concurrency, streaming, invalid-request rejection, chunking and measured prefix-cache hits |
| V2 ordinary output vs matching plugin-disabled baselines | **Pass** for all three configurations |
| Mixed ordinary decoding and extraction | **Pass** in all four configurations, six short/long text/image cases each |
| V2 eager sequential vectors vs V1 control | **Pass**, identical vectors |
| V2 compiled vs V1 control | **Pass**, minimum cosine 0.999865; maximum relative L2 1.647% |
| V2 MTP vs V2 non-MTP control | **Pass**, minimum cosine 0.999854; maximum relative L2 1.711%; 49 draft tokens, 27 accepted |
| Short text vs fresh Transformers reference | **Pass** in all four configurations |
| Short image vs fresh Transformers reference | **Fail** in all four configurations; unchanged thresholds |

See [matrix exit codes](runs-v0.32.0/matrix.txt), per-configuration command
records and reports in [run artifacts](runs-v0.32.0), and
[summary metrics](runs-v0.32.0/summary.json). Long-prompt lengths are
1628/1628/1705 tokens. Each non-MTP long suite measures 14,080 prefix-hit tokens;
the MTP suite measures 7,040. Mixed suites measure 6,400 non-MTP or 2,560 MTP
prefix-hit tokens. Compiled logs confirm a saved AOT function using
DYNAMO_TRACE_ONCE/Inductor; CUDA graph mode is NONE.

Long extrema below cover sequential and concurrent comparisons in each HTTP
suite. Short-image values are preserved from failed clients' JSON output before
their assertion; those clients stop before producing a final report JSON.

| Configuration | Long minimum cosine | Long maximum relative L2 | Short-image cosine | Short-image relative L2 |
| --- | --- | --- | --- | --- |
| V1 eager | 0.999448 | 3.329% | 0.998610 | 5.286% |
| V2 eager | 0.999402 | 3.461% | 0.998610 | 5.286% |
| V2 compiled | 0.999551 | 2.997% | 0.998513 | 5.462% |
| V2 eager MTP | 0.999453 | 3.317% | 0.998614 | 5.276% |

The short-image case has 105 input tokens and matching token IDs, final prompt
position, vector size and first generated token, but exceeds both numerical
thresholds. It also failed in prior releases. Its kernel-level cause has not
been established or repaired; no tolerance was relaxed. Passing transport and
control comparisons does not establish full Transformers image parity or
retrieval quality.

## Coverage limits

GPU eager execution, CUDA graph replay, GPU async scheduling, CPU INT4, other
checkpoints/precisions, compiled MTP, V1 compilation, real preemption and live
client cancellation are not runtime-validated on 0.32.0. Unit tests cover
preemption bookkeeping and cancellation/error cleanup. No performance or
retrieval-quality benchmark is claimed. V2 remains experimental.

The native [PR #57185](https://github.com/vllm-project/vllm/pull/57185) is separately
rebased onto upstream main and uses direct runner-output transport. This
package's wheel, model and HTTP results do not validate the native proposal.
