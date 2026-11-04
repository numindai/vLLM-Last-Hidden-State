# vLLM 0.33.0 compatibility upgrade — 2026-11-04

## Revisions and environment

- Previous release: v0.32.0, `5df63a7203cdb9328d824a27b682e77c7b9928be`.
  Its [report](UPGRADE-v0.32.0.md), including failed short-image comparisons,
  remains historical evidence.
- Target: v0.33.0, `4ba787ec9959e566d3b587a796dff07673324da5`.
  The upstream submodule is detached at this revision, with unmodified source.
- Extension: 0.33.0, validated changes based on package commit `22841772e84deaf154470125c29e5a908fe78f9b`.
  Metadata, version policy, tests, documentation, and submodule updated together.
  The [source manifest](runs-v0.33.0/source-manifest.json) identifies the tested
  implementation independently of commit and publication status.
- Official vLLM 0.33.0+cpu wheel, Python 3.12.13, PyTorch 2.14.0+cpu,
  Transformers 5.20.2, Triton CPU 3.8.0+cpu. Full versions/import provenance:
  [runtime metadata](runs-v0.33.0/runtime-environment.txt) and
  [dependency freeze](runs-v0.33.0/runtime-freeze.txt).
- Linux x86_64, AMD Ryzen AI MAX+ 395, 32 logical CPUs, approximately 121 GiB
  RAM; no CUDA device. Tests use eight CPU threads and a separate local port
  18432. Existing benchmark servers are not changed.
- Qwen/Qwen3.5-0.8B model/processor snapshot
  `2fc06364715b967f1860aea9cf38778875588b17`, native BF16 target, fresh
  Transformers eager references. [Downloaded file hashes](runs-v0.33.0/model-manifest.json)
  identify the exact local snapshot.

## Contract audit and changes

Compared the pinned 0.32.0 and 0.33.0 source and traced these callers.
Paths below are relative to upstream's `vllm/` Python package.
No capture/transport redesign or upstream source edits are required.

| Integration | Source inspected | Finding |
| --- | --- | --- |
| Endpoint initialization/delegation | `plugins/endpoint_plugins/interface.py`, `entrypoints/launchers/api_server/app_state.py`, `entrypoints/openai/chat_completion/api_router.py` | Plugin initialization still follows chat state. Route response-schema changes preserve handler lookup and JSON response serialization. |
| CLI and guards | `entrypoints/cli/serve.py`, `entrypoints/common/cli_args.py`, `utils/argparse_utils.py` | Internal CLI helpers moved; the public ServeSubcommand parser remains the package entry point. New enable/disable boolean aliases retain old spellings. Version guard now requires 0.33.0, allowing local hardware suffixes only. |
| Request/result metadata | `entrypoints/openai/chat_completion/protocol.py`, `serving.py`, `v1/request.py`, `v1/core/sched/scheduler.py` | KV parameters still reach sampling metadata; completed non-streaming responses preserve the connector handle. New streaming KV metadata does not enable streaming extraction. |
| Connector lifecycle | `distributed/kv_transfer/kv_connector/v1/base.py`, `v1/worker/gpu/kv_connector.py` | New sleep/wake hooks have no-op defaults; no new abstract requirement. V2 collects post-forward transfer results before drafting and finalizes saves afterward. Metadata-only connector behavior remains applicable. |
| Worker/RPC | `v1/worker/worker_base.py`, `v1/executor/multiproc_executor.py` | New optional worker-extension init/destroy hooks do not replace the package's explicit post-warmup initialization RPC. CPU workers retain serial startup. Watchdog and fault-output handling preserve sequential RPC invocation and error propagation. |
| V1 target capture | `v1/worker/cpu_model_runner.py`, `v1/worker/gpu_model_runner.py` | CPU outer-model hook and GPU instance `_model_forward` wrapper remain outside target execution/compiled internals, before draft execution. |
| V2 target capture | `v1/worker/gpu/model_runner.py`, `v1/worker/cpu/model_runner.py` | CPU inherits GPU V2. Fresh ExecuteModelState still retains target hidden states and InputBatch until sample_tokens clears it. Full and piecewise graphs converge on this same output boundary. New draft-skipping logic does not move capture into the drafter. |
| Packed rows and lifecycle | V2 `prepare_inputs`, `v1/core/sched/scheduler.py` | Computed-token arrays are gathered into batch order. Resumed V2 requests are resent as NewRequestData. Finish/preemption cleanup, re-registration and stale/no-forward rejection remain applicable. |
| Cache/chunks | `v1/core/kv_cache_manager.py:get_computed_blocks` | Default hit cap remains `request.num_tokens - 1`, requiring final prompt execution. New cache planning/lookup machinery retains this boundary. |
| Representation/MTP | `model_executor/models/qwen3_5.py`, inherited `qwen3_next.py:forward`, V1 EagleProposer, V2 speculator | Target output remains after final norm. Capture precedes target-state reuse and MTP drafting. Qwen model source is unchanged from the prior pin. |
| Compilation/async | V1/V2 execution and multiprocess dispatch | Owned synchronous CPU copy retained. GPU eager, graph replay and async execution require independent runtime evidence; CPU results cannot establish them. |

## Commands and evidence

Commands run from the package root unless specified otherwise:

```sh
uv venv --python ../vllm/.venv/bin/python /tmp/vllm033-runtime
uv pip install --python /tmp/vllm033-runtime/bin/python \
  'vllm==0.33.0+cpu' \
  'triton @ https://github.com/triton-lang/triton-cpu/releases/download/v3.8.0/triton-3.8.0+cpu-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl' \
  --extra-index-url https://wheels.vllm.ai/0.33.0/cpu --torch-backend cpu
uv pip install --python /tmp/vllm033-runtime/bin/python -e '.[dev]' build twine
uv pip check --python /tmp/vllm033-runtime/bin/python
HF_HUB_OFFLINE=1 /tmp/vllm033-runtime/bin/python -m pytest -q --tb=short
/tmp/vllm033-runtime/bin/ruff check .
/tmp/vllm033-runtime/bin/ruff format --check .
/tmp/vllm033-runtime/bin/python -m build --no-isolation
/tmp/vllm033-runtime/bin/python -m twine check --strict dist/vllm_last_hidden_state-0.33.0*
git diff --check
git -C vllm status --short
```

- **Pass:** 112 unit tests, one deprecation warning, against the matching
  installed wheel: [unit output](runs-v0.33.0/unit-tests.txt). Includes launcher
  runner pass-through, mp enforcement, YAML/dotted JSON precedence, version
  policy, packed rows, lifecycle and cancellation/error cleanup fixtures.
- **Pass:** [dependency consistency](runs-v0.33.0/dependency-check.txt),
  [Ruff](runs-v0.33.0/lint.txt), [build](runs-v0.33.0/build.txt), and
  [strict Twine checks](runs-v0.33.0/twine.txt).
  [Artifact manifests/hashes](runs-v0.33.0/artifacts.json) verify both distributions
  exclude the upstream submodule and validation directory.
- Initial wheel installation required explicitly listing upstream's Triton CPU
  URL dependency. [Failed attempt](runs-v0.33.0/runtime-install-initial.txt) and
  [successful installation](runs-v0.33.0/runtime-install.txt) are preserved.
  The GitHub release-assets API returned 404; the official release wheel index
  above supplied the matching CPU wheel.

## Runtime method

Download the same snapshot before replaying the command arrays:

```sh
/tmp/vllm033-runtime/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17', local_dir='/tmp/vllm033-model', allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'])"
```

[Reference commands](runs-v0.33.0/reference-commands.json) and per-mode
`*-commands.json` files record actual environments, server/client argument
arrays and exit statuses. These can be replayed from this repository without
local workspace notes (replace the recorded checkout prefix with your own
checkout path). Run matching plugin-disabled baselines before each V2
plugin mode; run V1 before V2 and non-MTP V2 before MTP to create control
snapshots. Start each server cold, wait for `/health`, execute its clients in
order, then stop it before the next mode. Keep stdout/stderr for failing clients.

References use identical processed token IDs, thinking-disabled templates and
assistant prefill. Numerical thresholds are unchanged: cosine >= 0.999 and
relative L2 <= 0.05. Long inputs have 1628/1628/1705 tokens and span the 1280-token
prefill budget. Model length is 4096, four sequences, prefix caching/chunking
enabled. Compilation and MTP each have an ordinary baseline with matching
runtime settings. GPU graphs and async are not established by CPU execution.

## Runtime results

| Check | Result |
| --- | --- |
| Plugin-disabled V2 eager/compiled/eager-MTP baselines | **Pass**, short and long inputs |
| V1 eager and V2 eager/compiled/eager-MTP long HTTP suites | **Pass**, exact input IDs, finite 1024-element vectors, first tokens, usage, concurrency, streaming, invalid-request rejection, chunking, measured prefix-cache hits |
| V2 ordinary output vs matching plugin-disabled baselines | **Pass**, all three configurations |
| Mixed ordinary decoding and extraction | **Pass**, six short/long text/image cases in each configuration |
| V2 eager sequential vectors vs V1 control | **Pass**, identical vectors |
| V2 compiled vs V1 control | **Pass**, minimum cosine 0.999901, maximum relative L2 1.410% |
| V2 MTP vs V2 non-MTP control | **Pass**, minimum cosine 0.999854, maximum relative L2 1.711%; 49 drafted and 27 accepted tokens |
| Short text vs fresh Transformers reference | **Pass**, all four configurations |
| Short image vs fresh Transformers reference | **Fail**, all four configurations; unchanged thresholds |
| Cancellation/error cleanup and preemption bookkeeping | **Pass** in unit fixtures; live cancellation and real preemption **untested** |
| GPU eager, graph replay and async scheduling | **Untested**, no CUDA device |
| CPU INT4, V1 compilation, compiled MTP, other checkpoints | **Untested** on this release |

See [matrix exit codes](runs-v0.33.0/matrix.txt), per-configuration command
records and reports, and [summary metrics](runs-v0.33.0/summary.json).
All seven test servers were stopped after their configuration. Public text logs
normalize trailing whitespace and line endings; numerical records are unchanged.
Original text logs remain in local workspace evidence outside the package.

The long HTTP suites measured 14,080 prefix-hit tokens for each non-MTP mode,
and 7,040 with MTP. Mixed suites measured 6,400 non-MTP or 3,200 MTP prefix-hit
tokens. Compiled server logs confirm a saved AOT function with
DYNAMO_TRACE_ONCE/Inductor; CUDA graph mode is NONE.

Long extrema cover sequential and concurrent comparisons. Short-image values
come from failed clients' stdout before the assertion; those clients stop before
writing a final report JSON, so a missing short JSON is an expected failure
artifact, not a successful run.

| Configuration | Long minimum cosine | Long maximum relative L2 | Short-image cosine | Short-image relative L2 |
| --- | --- | --- | --- | --- |
| V1 eager | 0.999517 | 3.110% | 0.998610 | 5.286% |
| V2 eager | 0.999517 | 3.110% | 0.998610 | 5.286% |
| V2 compiled | 0.999551 | 2.998% | 0.998513 | 5.462% |
| V2 eager MTP | 0.999526 | 3.083% | 0.998614 | 5.276% |

The short-image case has 105 input tokens with matching processed IDs, final
prompt position, vector shape and first generated token. It exceeds both
numerical thresholds, as in prior releases. The kernel-level cause remains
unresolved; no tolerance was relaxed. Passing transport or same-vLLM controls
does not establish full Transformers image parity or retrieval quality.

## Compatibility conclusion and remaining work

The package supports vLLM 0.33.0 with the same additive capture/transport
architecture and scope. Matching-wheel unit checks and the listed Qwen3.5-0.8B
BF16 CPU serving modes pass, apart from the preserved short-image comparison
failure. V2 remains experimental. GPU execution, graph replay, async scheduling,
INT4, other checkpoints/precisions, V1 compilation, compiled MTP, real preemption
and live cancellation require their own runtime evidence. No performance or
retrieval-quality claim is made.

The [0.32.0 report](UPGRADE-v0.32.0.md) and older reports remain historical.
The native [PR #57185](https://github.com/vllm-project/vllm/pull/57185) was
independently updated to upstream main and uses direct runner-output transport;
this package's CPU wheel/model results do not validate that implementation.
