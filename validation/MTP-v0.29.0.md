# MTP and precision validation on vLLM 0.29.0

Date: 2026-09-21. Upstream source remains pinned and unmodified at
`98dff2a81d747d1dba01a47f939f48c3526d4206`. This extends the earlier
[upgrade report](UPGRADE-v0.29.0.md), using the same rebuilt CPU extensions,
Python 3.12.13, Torch 2.13.0+cpu, and AMD Ryzen AI MAX+ 395.

## Implementation and source audit

- Removed the extension's BF16 requirement and quantization whitelist. Model
  loading and kernel precision validation belong to upstream vLLM. The launcher
  neither adds nor rewrites `--dtype` / `--quantization`.
- Permit normalized speculative method `mtp`; retain rejection of other methods
  because their model/output contracts have not been audited here.
- Target hooks are unchanged. `SpeculativeConfig.use_eagle()` selects
  `EagleProposer` for Qwen MTP. `SpecDecodeBaseProposer.load_model` loads its own
  outer model; only embedding and output-head modules can be shared with the
  target. Proposals call that separate model directly.
- CPU capture attaches to the outer target module, not shared submodules. GPU
  capture wraps the target runner's `_model_forward`, which drafting bypasses.
  The copy completes before draft execution or output-buffer reuse.
- `Qwen3_5MTP` returns draft states separately. The captured tensor is still the
  target Qwen3.5 decoder output after its final normalization.
- Speculative verification may pack multiple decode rows per request. Selecting
  `prompt_length - 1 - computed_tokens` skips those rows once the prompt is past.
  Recomputed final prompts still replace results after preemption.
- Upstream defers connector finalization until after drafting. This connector's
  load/save callbacks are no-ops; captured results are owned and do not depend
  on the lifetime of the bound connector metadata.
- Precision regression coverage uses FP16/BF16/FP32 outputs, and verifies
  construction imposes no quantization whitelist. Fixture tests of AWQ/GPTQ
  settings do **not** establish that those kernels work on this CPU.

## Fast checks

85 tests passed; Ruff lint and formatting passed. One existing `max_tokens`
deprecation warning remains in the cancellation fixture. Tests include
CPU/GPU hook isolation from draft forwards, speculative verification rows,
precision-independent copies, and startup settings. GPU tests use fixtures only.

## Runtime setup

All commands below run at repository root. The explicit source path selects the
rebuilt 0.29.0+cpu distribution metadata instead of the old editable install.
No metadata override or distribution wheel was used.

```sh
export PYTHONPATH="$PWD:$PWD/vllm"
export VLLM_PLUGINS='' VLLM_USE_RUST_FRONTEND=0
export VLLM_CPU_KVCACHE_SPACE=1 VLLM_CPU_OMP_THREADS_BIND=nobind
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1
```

Server command (replace `MODEL_SNAPSHOT` with the resolved snapshot path):

```sh
vllm/.venv/bin/python -m vllm_last_hidden_state.serve MODEL_SNAPSHOT \
  --served-model-name mtp-test --distributed-executor-backend mp \
  --max-model-len 4096 --max-num-batched-tokens 1280 --max-num-seqs 4 \
  --enforce-eager --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --port 8137
```

No dtype/quantization flag is used unless a precision test is explicitly noted.
The non-MTP control omits only `--speculative-config` and uses port 8138.

```sh
vllm/.venv/bin/python validation/validate_mtp.py --require-mtp \
  --snapshot /tmp/mtp-snapshot.json --report /tmp/mtp-report.json
vllm/.venv/bin/python validation/validate_mtp.py --url http://127.0.0.1:8138 \
  --reference /tmp/mtp-snapshot.json --snapshot /tmp/control-snapshot.json \
  --report /tmp/control-report.json
```

The integration helper uses two short text prompts, one image prompt, and long
versions (longer than the 1280-token prefill budget), with assistant prefill.
It checks finite vectors, exact prompt and first-output token IDs, ordinary
16-token generation, streaming, repeated extraction and mixed concurrent traffic.
It requires nonzero drafted **and accepted** token counters in MTP mode.
Non-MTP comparisons use unchanged cosine >= 0.999 and relative L2 <= 0.05;
ordinary greedy output token sequences must match exactly. These controls isolate
MTP behavior; they do not replace an independent Transformers reference.

## CPU caching limitation

With MTP and default prefix caching, the first ordinary request fails before
model forward in `mamba_utils.preprocess_mamba` -> `run_fused_precopy` ->
`precopy_mamba_align_fused_kernel[grid]`, with
`TypeError: 'function' object is not subscriptable`. This CPU build has no Triton
CPU backend. The CPU runner substitutes several speculative kernels but not this
fused Mamba align-copy kernel. Qwen MTP also explicitly rejects `mamba_cache_mode=all`.

The same failure was reproduced using `python -m vllm.entrypoints.cli.main serve`
with `VLLM_PLUGINS=''`, no worker extension and no connector. It is independent
of this package.

`--no-enable-prefix-caching` avoids that path. This is an explicit deployment
workaround; the extension does not silently disable caching, change upstream
source, or claim to repair this upstream CPU kernel. Non-MTP cache validation
from the earlier report remains separate.

This workaround applies to the tested CPU build, not as a general MTP
requirement. On CUDA, retain prefix caching and the default Mamba cache mode
`align`: the failing operation has a Triton GPU implementation in the pinned
source. CUDA execution has not been validated here. Do not select `all`, which
Qwen3.5 MTP explicitly rejects on either backend.

## Checkpoints and results

- Qwen/Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`:
  default dtype resolves to BF16. MTP eager suite passes, with 44 draft rounds,
  88 proposed tokens and 33 accepted tokens. See [metrics](mtp-qwen08-eager.json).
  Non-MTP control passes with exact ordinary greedy tokens; minimum prompt-vector
  cosine 0.999866 and maximum relative L2 0.016372.
  See [comparison](mtp-qwen08-control-comparison.json).
- Qwen3.5-0.8B compiled MTP also passes [the same suite](mtp-qwen08-compiled.json),
  including comparison with the non-MTP eager control and exact ordinary tokens.
  Startup logged `DYNAMO_TRACE_ONCE` / Inductor and saved AOT functions for
  target and draft models. This run omitted `--enforce-eager`, used port 8140,
  and added `--limit-mm-per-prompt '{"image":{"count":1,"width":256,"height":256},"video":0}'`
  to bound image warmup. An earlier maximum-image-size warmup was cancelled
  after several minutes and is not counted as a completed test.
- numind/NuExtract3-W4A16 revision `b5028670152c8130a3f362b66981eee16612b7f6`:
  Qwen3.5 architecture, hidden size 2560, one MTP layer, compressed-tensors W4A16
  with group size 128. Configuration excludes `re:^mtp.*` from quantization,
  and its weight index includes `model_mtp.safetensors`. Default activation dtype
  is BF16, chosen by upstream from the checkpoint. The eager MTP suite passes
  with 2560-element vectors, 13 draft rounds, 26 proposed tokens and 26 accepted
  tokens on these fixtures. This is not a general acceptance-rate benchmark.
  See [runtime metrics](mtp-nuextract-eager.json) and
  [checkpoint file hashes](mtp-nuextract-checkpoint.json). Startup selected
  `CPUWNA16LinearKernel for CompressedTensorsWNA16` and loaded both target and
  draft weights; no dtype or quantization flags were supplied. This run used
  the same 256-by-256 image profiling bound as the compiled Qwen test.
  Prompt lengths were 43, 43, 121, 1643, 1643 and 1721 tokens.
  The [non-MTP control comparison](mtp-nuextract-control-comparison.json) also
  passes: exact ordinary greedy token sequences, minimum vector cosine
  0.999359, maximum relative L2 0.035822. NuExtract compilation was not tested;
  compiled MTP evidence above is for Qwen3.5-0.8B.

Failure details are retained in [CPU limitation results](mtp-cpu-limitations.json).

## Precision limitation on this CPU backend

An explicit `--dtype float16` run starts and loads both target and MTP models,
but fails on the first request in upstream GDN attention with
`AssertionError: CPU GDN attention requires BF16.` The package no longer imposes
this restriction; the pinned CPU Qwen3.5 kernel does. Omitting `--dtype` lets
vLLM select the checkpoint default (BF16 here), so no BF16 flag is needed.
FP16 capture is covered by fixture tests, not a successful CPU Qwen3.5 runtime.

GPU, CUDA graphs and async scheduling remain untested in this CPU environment.
