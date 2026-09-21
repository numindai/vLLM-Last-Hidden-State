# Historical implementation and validation notes

Archived on 2026-09-21 before the package maintenance update. Runtime results
below apply to vLLM `23cfaad49701c497def53552b23317335431f72a`, not v0.29.0.
Use the [current README](../README.md) for installation and activation and the
[v0.29.0 report](UPGRADE-v0.29.0.md) for current evidence. Historical descriptions
of launcher behavior and file locations are preserved as context.

## CPU environment used by the historical commands

Run the commands below from the package repository root. These variables were
explicit in the original reproduction instructions; the launcher does not set them:

```sh
export VLLM_CPU_KVCACHE_SPACE=1
export VLLM_CPU_OMP_THREADS_BIND=nobind
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
```

## Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[
        {"role": "user", "content": "Describe a yellow bicycle briefly."},
        {"role": "assistant", "content": '{\n  "short description": "'},
    ],
    max_tokens=1,
    temperature=0,
    stream=False,
    n=1,
    extra_body={
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "kv_transfer_params": {"return_last_hidden_state": True},
    },
)
vector = response.model_dump()["kv_transfer_params"]["last_hidden_state"]
print(response.choices[0].message.content, len(vector))
```

Images use the usual OpenAI `image_url` content blocks. The validation script
contains a self-contained image example. The plugin has no hardcoded template,
instruction, image, or JSON field name. `max_completion_tokens=1` is accepted
instead of `max_tokens=1`, following vLLM's precedence when both are present.

## Files and lifecycle

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Installable package and endpoint entry point. |
| `connector.py` | Return the request handle; select the prompt row from the packed batch; own the CPU result. |
| `worker.py` | Install CPU/GPU capture; retrieve or discard results through engine RPC. |
| `endpoint.py` | Validate opt-in requests and augment the existing chat response. |
| `vllm_last_hidden_state/serve.py` | Forward vLLM serve arguments and add only extension activation. |
| `validation/validate_hidden_state.py` | Actual HTTP versus Transformers comparison, including image and assistant prefill. |
| `validation/test_extension.py` | Small model-free row-selection and cleanup checks. |

On CPU, a PyTorch forward hook on the outer Qwen3.5 model observes the final
normalized states returned by its decoder. On GPU, an instance-local wrapper
around the runner's `_model_forward` captures its returned tensor outside both
compilation and CUDA graph replay. Both use the model runner's actual
request order, packed query boundaries, and computed-token counts to select
`prompt_length - 1` for each opted-in request. Earlier prefill chunks are skipped.
vLLM recomputes the final prompt token even on a full prefix-cache hit, so the
same selection works after a cached prefix. Chunks before or after the final
prompt position leave an existing result untouched. If preemption causes that
position to be recomputed, capture replaces the old state with the new one.
Only that row is copied, and model output is never modified. No additional model
forward or normalization is run.

The request flag and a server-generated handle reach the worker through existing
sampling metadata. The scheduler-side connector returns the same handle in the
normal completion. The frontend retrieves the already-captured vector through
`EngineClient.collective_rpc`, replaces the handle with a JSON list, and sends the
HTTP response. The capture completes before generation returns, so neither cache
block retention nor polling is necessary. No tensor files or additional client
endpoints are involved.

The RPC has a 30-second deadline. Missing/failed capture produces an HTTP error
rather than silently omitting the vector. Successful reads consume results;
cancellation discards results and marks pending work for disposal. Abandoned
results expire after 120 seconds on subsequent capture/RPC activity. Results and
cancellation markers are each capped at 1024 entries. A completely idle worker
retains bounded expired entries until its next operation or shutdown.

## Compatibility and costs

The extension accepts BF16 Qwen3.5 on the V1 CPU or GPU model runner, with
unquantized or `compressed-tensors` weights and one local multiprocessing worker.
It allows eager execution, Dynamo trace-once, and vLLM compilation. Prefix
caching, chunked prefill, and asynchronous scheduling are no longer rejected.
Validated checkpoints and modes are listed below.

CPU runtime validation covers prefix caching and chunked prefill. The CUDA path
is implemented but has not been run on a GPU here, including CUDA graphs and
async scheduling. This checkout's CPU platform unconditionally sets
`async_scheduling=False`, even if requested. Enabling all three features in
execution therefore requires a GPU backend; removing a connector guard cannot
enable async scheduling on CPU.

On a CUDA installation, the intended configuration for all three features is:

```sh
VLLM_USE_V2_MODEL_RUNNER=0 .venv/bin/vllm-last-hidden-state \
    drawais/Qwen3.5-9B-AWQ-INT4 --host 127.0.0.1 --dtype bfloat16 \
    --distributed-executor-backend mp \
    --enable-prefix-caching --enable-chunked-prefill --async-scheduling
```

This GPU command is untested here. The wrapper forwards vLLM compilation and
CUDA graph options unchanged. MTP remains unsupported.
The cached NuExtract3 config uses
this architecture, but its model/template parity is not implied by Qwen testing.
Float32, other quantization formats, LoRA, distributed parallelism, speculative
decoding, and alternate KV connectors are unsupported. The connector validates
configuration at startup; the launcher
does not supply or adjust these runtime settings.

Two runtime adaptations are explicit:

1. The endpoint plugin replaces `state.openai_serving_chat` with a delegating
   wrapper. It keeps the original HTTP route and its request/response handling.
2. On CPU, the worker installs one idempotent PyTorch forward hook on the outer
   model. On GPU, it wraps `_model_forward` on that runner instance after warmup:
   call the original, copy the requested output row, then return the original
   output unchanged. This specific runtime patch is needed because CUDA graph
   replay bypasses inner Python module hooks. No global class is patched.

These integrations depend on the serving handler, V1 batch metadata, and
Qwen3.5 model structure. Existing vLLM source files remain unchanged.

The existing `extract_hidden_states` path was investigated and tested first.
It runs through speculative-decoding machinery, which changes available sampler
features even on ordinary requests and adds a full-sequence hidden-state cache.
Direct post-normalization capture avoids those side effects and avoids rounding
from materializing and renormalizing an auxiliary residual state.

The extension allocates no sequence-sized hidden-state buffers. Each requested
4096-component vector requires a 16 KiB temporary float32 copy plus its Python
list/JSON representation. There is one retrieval RPC per opted-in completion.
Ordinary requests incur a small Python capture/metadata check, with no tensor copy.
For GPU extraction, the selected row is synchronously copied to owned CPU memory
before batch metadata or replay buffers can be reused. This adds synchronization
to opted-in batches; it does not disable async scheduling for ordinary requests.
No serving throughput improvement or zero overhead is claimed.

BF16 vLLM CPU inference and Transformers CPU inference have measurable numerical
differences even when tokenization matches. Direct capture returns exactly the
selected vLLM post-normalization row. It does not make vLLM's arithmetic match
Transformers bit for bit. Float32 was attempted, but the baseline CPU model
failed because `causal_conv1d_fwd_kernel_impl` does not implement Float.

## Validation

Run from the repository root after installing the package:

```sh
.venv/bin/python -m pytest validation/test_extension.py -q \
    --confcutdir=.
.venv/bin/python validation/validate_hidden_state.py reference
# Start vllm-last-hidden-state in another terminal, then:
.venv/bin/python validation/validate_hidden_state.py http
.venv/bin/ruff check vllm_last_hidden_state validation
.venv/bin/ruff format --check vllm_last_hidden_state validation
```

Reference and HTTP phases must use the same model, dtype, processor, chat
formatting, and image. The script compares exact input token IDs before checking
the vector. It includes two distinct text prompts, an image with an extraction
instruction and unfinished JSON assistant message, concurrent requests, ordinary
requests, and invalid opted-in requests.

The CPU example above uses vLLM's default Inductor compilation
(`DYNAMO_TRACE_ONCE` in this checkout). Add the original `--enforce-eager` flag
for eager execution, or pass vLLM's compilation options directly. The
CPU vision encoder remains outside compilation under vLLM's default settings.
The CPU capture hook is on the outer model so installing it after warmup does not
change a previously compiled inner module. `VLLM_CPU_CI_ENV` must be unset or
zero, and `TORCH_COMPILE_DISABLE` must not be `1`, when testing Inductor.
The wrapper leaves these environment variables untouched.

For INT4, use the same server command with the positional model replaced by
`drawais/Qwen3.5-9B-AWQ-INT4`; `--skip-mm-profiling` is unnecessary for this
text-only checkpoint.

For the multimodal CPU test, `--skip-mm-profiling` avoids an expensive
maximum-size dummy-image pass during startup. Real image inference remains
enabled; this is not a test of maximum image memory requirements.

The drawais checkpoint is text-only (`Qwen3_5ForCausalLM`), with symmetric INT4
group-size-128 weights stored in `compressed-tensors` format. Let vLLM detect
this format; do not force `--quantization awq` based on the repository name.
Run both validation phases with `--model drawais/Qwen3.5-9B-AWQ-INT4 --text-only`
and a separate `--reference` path; add `--dequantize` to the reference phase.
This checks the two text cases against the
same quantized checkpoint in Transformers, which decompresses its weights for
CPU inference. It does not compare quantized vectors to unquantized vectors or
imply image support for this checkpoint.

BF16 vector comparison checks relative L2 error <= 5% and cosine similarity
>= 0.999, alongside exact token alignment and matching first generated token.
The report also records maximum and mean absolute component errors. A stricter
initial elementwise check (`atol=0.15`, `rtol=0.05`) failed on BF16 CPU; this is
explicitly not an elementwise-parity claim. Scale and direction checks avoid
unstable per-component relative errors near zero while bounding overall error.
These are numerical extraction checks, not downstream embedding-quality evals.

## Recorded results

Validated on this machine (AMD Ryzen AI MAX+ 395, CPU execution only), with
vLLM `23cfaad49701c497def53552b23317335431f72a`, PyTorch `2.13.0+cpu`,
Transformers `5.17.0`, and OpenAI Python `3.13.0`. Model revision:
`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

| Case | Prompt position | Cosine | Relative L2 error | Max component error |
| --- | ---: | ---: | ---: | ---: |
| Bicycle description | 27 | 0.999681 | 2.54% | 1.390625 |
| Mountain description | 27 | 0.999805 | 1.98% | 0.281250 |
| Image and description prefix | 104 | 0.999152 | 4.12% | 0.375000 |

All vectors contain 4096 finite values. In these examples, the final prompt token
is ID 328 (a space followed by an opening quote), and the first generated token
is ID 32 (`A`). Concurrent cases produced the same vectors as sequential cases.
Ordinary one-token and three-token completion content, finish reasons, token IDs,
and usage matched the baseline server exactly. Ordinary streaming and invalid
opt-in request checks passed. Five unit checks cover packed-row selection,
copy ownership, invalid spans, ordinary bypass, and cancellation cleanup.

Observed sequential opted-in latency was 0.35–1.32 seconds. These are short smoke
tests, not an isolated overhead benchmark: warmup and CPU scheduling affect the
comparison. Both final baseline and plugin use the same 1 GiB KV cache budget;
no hidden-state cache is allocated. Full metrics are in
[`validation/results-qwen3.5-9b.json`](results-qwen3.5-9b.json).

Equivalent commands for the recorded model/HTTP comparisons with the current
pass-through launcher (use the CPU environment exports above in each server
terminal):

```sh
.venv/bin/python validation/validate_hidden_state.py reference
VLLM_PLUGINS="" .venv/bin/vllm serve Qwen/Qwen3.5-9B \
    --host 127.0.0.1 --port 8123 --dtype bfloat16 \
    --distributed-executor-backend mp \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --no-enable-prefix-caching --no-enable-chunked-prefill \
    --no-async-scheduling --enforce-eager
.venv/bin/python validation/validate_hidden_state.py baseline \
    --url http://127.0.0.1:8123 --baseline /tmp/last-hidden-baseline.json
.venv/bin/vllm-last-hidden-state Qwen/Qwen3.5-9B \
    --host 127.0.0.1 --port 8124 --dtype bfloat16 \
    --distributed-executor-backend mp \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --no-enable-prefix-caching --no-enable-chunked-prefill \
    --no-async-scheduling --enforce-eager
.venv/bin/python validation/validate_hidden_state.py http \
    --url http://127.0.0.1:8124 --baseline /tmp/last-hidden-baseline.json
```

The reference was also successfully run in float32, but both baseline and plugin
vLLM float32 startup failed in the existing CPU convolution kernel. CUDA and the
original NuExtract3 checkpoint/template were not validated. No upstream issue,
PR, commit, or publication was created. `git diff --exit-code` confirmed all
tracked vLLM files remained unchanged; all extension deliverables are additive.

Repository pre-commit hooks passed on all added files (including Ruff, Markdown,
SPDX, forbidden-import, and JSON checks). Native build and downloaded weights
are environment artifacts; they are not required additions to vLLM source.

### CPU compilation and INT4 follow-up

After moving capture to the outer model, the complete HTTP suite passed in all
four configurations below. Each row compares with Transformers using the same
checkpoint, with the existing cosine and relative-L2 thresholds unchanged.

| Weights | Execution | Cases | Minimum cosine | Maximum relative L2 error |
| --- | --- | --- | ---: | ---: |
| Original BF16 | Eager regression | Two text, one image | 0.999152 | 4.12% |
| Original BF16 | Inductor | Two text, one image | 0.999173 | 4.07% |
| drawais INT4 | Eager | Two text | 0.999877 | 1.58% |
| drawais INT4 | Inductor | Two text | 0.999770 | 2.16% |

All cases ran sequentially and concurrently. Ordinary one-token and three-token
completions matched each checkpoint's ordinary eager baseline exactly for
choices, token IDs, and usage. Ordinary streaming and invalid opt-in checks also
passed. These checks do not imply bitwise vector equality between eager and
compiled arithmetic.

Both compiled runs logged `enforce_eager=False`, `DYNAMO_TRACE_ONCE`, backend
`inductor`, and a saved AOT compiled artifact before successful HTTP extraction.
Both INT4 runs logged `CPUWNA16LinearKernel`; the vLLM server did not load
unquantized replacement weights. The drawais revision is
`b3b888195917f6224e3a97e84e7f64c6b1ec8731`, using `compressed-tensors 0.17.0`.
Transformers' default compressed loading failed with a missing `Linear.weight`;
its explicit `dequantize=True` loading option succeeded for the reference.

For example, prepare and check the quantized reference with:

```sh
.venv/bin/python validation/validate_hidden_state.py reference \
    --model drawais/Qwen3.5-9B-AWQ-INT4 --text-only --dequantize \
    --reference /tmp/last-hidden-awq-reference.json
# Start the INT4 server using the launcher command above, then:
.venv/bin/python validation/validate_hidden_state.py http \
    --model drawais/Qwen3.5-9B-AWQ-INT4 --text-only \
    --reference /tmp/last-hidden-awq-reference.json
```

Use the original `.venv/bin/vllm serve` with `VLLM_PLUGINS=""` and the validation
script's `baseline` mode to record ordinary responses, then pass that JSON path
with `--baseline` to `http`
for the ordinary-output regression comparison. The follow-up did this for both
checkpoints. Full metrics and runtime evidence are recorded in
[`validation/results-cpu-compilation-int4.json`](results-cpu-compilation-int4.json).
MTP remains disabled and was not tested. CUDA compilation was not tested.

### Prefix caching, chunked prefill, and async scheduling

Capture now skips chunks that do not contain the final prompt position and uses
the computed-token offset when a request resumes from a cached prefix. It keeps
an owned copy of a completed result even if the runner advances to another
batch. The CPU check uses `--max-model-len 4096 --max-num-batched-tokens 1280`
with `--enable-prefix-caching --enable-chunked-prefill`, ensuring the long
prompts cannot fit into one prefill step on a cold cache.

To reproduce, add `--long-prompts` to both reference and HTTP validation phases.
Add `--require-prefix-cache-hit` to the HTTP phase to require an increase in
the server's actual prefix-cache hit-token counter, rather than merely checking
that caching was configured. Use separate reference and report files from the
short-prompt tests. For the INT4 reference, also use `--text-only --dequantize`.

Nine model-free checks cover row selection, cached offsets, chunk transitions,
copy ownership, invalid spans, ordinary bypass, cancellation, and capture after
the wrapped forward call, including replacement after a recomputed prefill.
The wrapper check uses CPU tensors and a reused output
buffer; it is not a CUDA graph or GPU async scheduling test.

The GPU implementation copies the result after `_model_forward` returns,
including after CUDA graph replay, before the next batch can overwrite input
metadata or output buffers. Worker RPCs and capture execute in the worker's
serial command loop; asynchronous output delivery does not read mutable batch
metadata on behalf of this extension. Actual GPU execution, CUDA graphs, and
asynchronous scheduling remain unvalidated on this CPU-only machine.

Recorded CPU results with compilation and both caching and chunking enabled:

| Comparison | Prompt tokens | Minimum cosine | Maximum relative L2 | Cache-hit tokens during suite |
| --- | --- | ---: | ---: | ---: |
| INT4 vs same quantized checkpoint in Transformers | 1628 | 0.999595 | 2.85% | 10240 |
| Original text/image vs vLLM control without caching or chunking | 1628 / 1705 | 0.999938 | 1.13% | 14080 |

Sequential and concurrent requests, ordinary streaming, and invalid opt-in
checks passed for these comparisons. All four ordinary response comparisons
(three one-token cases and one three-token case) matched the feature-disabled
control exactly for choices, usage, and prompt token IDs.

The long image did **not** pass the existing Transformers numerical threshold:
relative L2 was 5.66% with caching/chunking and 5.72% in the control without
them (cosines 0.998397 and 0.998370). The independent Transformers failure is
retained in the results; no tolerance was relaxed. The comparison with the
vLLM control measures the effect of enabling the features and does not establish
Transformers parity. Control vectors were collected through the same chat
endpoint with both features disabled and a 4096-token batch budget, then used
as the reference for the cached/chunked HTTP suite.

Full metrics and the GPU validation status are in
[`validation/results-caching-chunking.json`](results-caching-chunking.json).
