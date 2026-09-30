# vLLM compatibility upgrade: TARGET_REVISION

Copy to a new report named for the target revision. Replace placeholders and
mark each result **pass**, **fail**, or **untested**; do not leave unchecked work
looking like a successful compatibility claim. Link numerical JSON reports and
preserve relevant failed-run output. See the [runbook](../MAINTENANCE.md).

## Revisions and environment

- Previous tested vLLM revision/report:
- Target vLLM revision and dirty state:
- Extension package version (must equal `compat.SUPPORTED_VLLM_VERSION` and target vLLM release) and extension commit or saved source artifact:
- Installed vLLM version/source path and native build provenance:
- Python, PyTorch, Transformers, compressed-tensors, OpenAI client versions:
- OS, CPU/GPU, accelerator runtime/driver, available memory:
- Model and tokenizer/processor identifiers and resolved revisions:
- Reference backend, dtype, quantization, template options:

## Contract audit and changes

| Integration | Target symbols/source locations inspected | Findings and package changes |
| --- | --- | --- |
| Endpoint loading, Python frontend, route delegation | TODO | TODO |
| Request metadata, connector lifecycle, response handle | TODO | TODO |
| Worker initialization and RPC ordering | TODO | TODO |
| V1/V2 CPU/GPU capture boundary, output normalization | TODO | TODO |
| V2 fresh execution state, batch order, preemption/resume metadata | TODO | TODO |
| MTP target/draft ownership and capture-before-drafting | TODO | TODO |
| Packed rows, cached prefix, chunks, preemption | TODO | TODO |
| Async scheduling, graph replay, buffer ownership | TODO | TODO |
| Runtime guards, quantization, runner selection | TODO | TODO |

## Commands and evidence

Record exact install/build, reference, baseline, server, HTTP, unit-test, and lint
commands used, with relevant environment variables and exit statuses. Include
effective runtime settings and log excerpts proving compilation, quantized kernel
selection, graph replay, and async scheduling where claimed. Link durable artifacts
or include the necessary excerpts; temporary absolute paths alone are insufficient.

| Check | Status | Evidence / failure / reason untested |
| --- | --- | --- |
| Unit checks and repository hooks | untested | TODO |
| Launcher leaves runner environment untouched; supplies/validates mp | untested | TODO |
| V2 CPU eager and compiled with Triton CPU | untested | TODO |
| MTP vs non-MTP control with measured draft/accepted counters | untested | TODO |
| CPU BF16 eager: text + image | untested | TODO |
| CPU BF16 compiled: text + image | untested | TODO |
| CPU INT4 eager and compiled: text | untested | TODO |
| Cold-cache long prompts spanning prefill chunks | untested | TODO |
| Repeated prefix-cache hits, measured counter delta | untested | TODO |
| Ordinary output vs plugin-disabled baseline | untested | TODO |
| Concurrent distinct prompts and ordinary streaming | untested | TODO |
| Invalid opt-in requests and token-limit precedence | untested | TODO |
| Cancellation/failure cleanup and subsequent requests | untested | TODO |
| Real preemption/recomputation and mixed traffic | untested | TODO |
| GPU eager | untested | TODO |
| GPU compilation and actual CUDA graph replay | untested | TODO |
| GPU async scheduling with caching/chunking | untested | TODO |

## Numerical results

For each model/mode/reference backend, record prompt lengths, vector size, exact
token alignment, first-token/usage checks, minimum cosine, maximum relative L2,
thresholds, and cache-hit-token delta when applicable. Record failed comparisons
as well as successful controls. A same-vLLM control is not Transformers parity.

## Compatibility conclusion and remaining work

- Validated configurations on this revision:
- Implemented but untested configurations (explicitly list GPU/graphs/async):
- Failures, regressions, or environment blockers and next reproduction steps:
- Unsupported configurations / changed requirements:
- README/runbook updates and any core changes (with justification):
- Relationship to upstream PR #57185; distinguish package evidence from core evidence:
- Earlier reports superseded for current guidance (preserve original results):
