# vLLM Last Hidden State

**Get the representation behind the first generated token—through your chat API.**

An open-source extension by [NuMind](https://numind.ai) that returns one hidden-state
vector alongside a one-token chat completion. Explore instruction-conditioned
representations of text and images using the model you already serve with vLLM.
Ordinary generation and streaming continue on the same server.

[NuExtract Platform](https://nuextract.ai) · [NuExtract3](https://huggingface.co/numind/NuExtract3) · [Upstream PR #57185](https://github.com/vllm-project/vllm/pull/57185) · [Apache-2.0](LICENSE)

## Why use it?

- **One model, two uses.** Serve ordinary chat requests and opt into hidden-state
  extraction on individual one-token requests.
- **Text and images.** Capture the final prompt representation after the model
  processes your instructions, chat template, images, and any assistant prefill.
- **Reuse the existing computation.** Copy one vector from the target model's
  forward pass, without a second forward pass, a pooling server, or token-state files.
- **Use familiar clients.** Send a vLLM-specific extension field through the
  OpenAI Python client and receive the vector in the same response.

This is a building block for representation analysis and experiments in document
retrieval, clustering, and classification. It exposes a raw model representation;
retrieval quality still depends on the model, prompt, and downstream processing.

## Quick start

Use Python 3.12+ with **vLLM 0.30.0 installed for your hardware**. The extension
keeps your existing vLLM/PyTorch installation intact.

```sh
python -m pip install "vllm-last-hidden-state @ https://github.com/numindai/vLLM-Last-Hidden-State/archive/refs/heads/main.zip"
vllm-last-hidden-state Qwen/Qwen3.5-9B
```

For reproducibility, replace `refs/heads/main` with a tested full commit SHA.
Wheel downloads will also be available through [GitHub releases](https://github.com/numindai/vLLM-Last-Hidden-State/releases)
as releases are published. See the [maintenance guide](MAINTENANCE.md) for source
checkouts, development, and release publishing.

With the [OpenAI Python client](https://github.com/openai/openai-python) installed:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[
        {
            "role": "user",
            "content": "Summarize this text in one word: A yellow bicycle beside a stone wall.",
        }
    ],
    max_completion_tokens=1,
    n=1,
    stream=False,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "kv_transfer_params": {"return_last_hidden_state": True},
    },
)
vector = response.model_dump()["kv_transfer_params"]["last_hidden_state"]
print(len(vector), vector[:5])
```

Leave out `return_last_hidden_state` for ordinary generation, with your usual
output length and streaming settings. This flag is an extension to vLLM's
OpenAI-compatible API, not a standard OpenAI API parameter.

### Using `vllm serve` directly

You can use the standard `vllm serve` command instead of the
`vllm-last-hidden-state` launcher. Install the package in the **same Python
environment as vLLM**, then supply its activation settings explicitly:

```sh
VLLM_PLUGINS=last_hidden_state \
VLLM_USE_RUST_FRONTEND=0 \
vllm serve Qwen/Qwen3.5-9B \
    --distributed-executor-backend mp \
    --worker-extension-cls vllm_last_hidden_state.worker.LastHiddenStateWorker \
    --kv-transfer-config '{"kv_connector":"LastHiddenStateConnector","kv_connector_module_path":"vllm_last_hidden_state.connector","kv_role":"kv_producer"}' \
    --host 127.0.0.1 \
    --port 8000
```

| Setting | Purpose |
| --- | --- |
| `VLLM_PLUGINS=last_hidden_state` | Enables the installed endpoint plugin that handles the extraction flag and adds the vector to the chat response. |
| `VLLM_USE_RUST_FRONTEND=0` | Selects the Python frontend required by this package. |
| `--distributed-executor-backend mp` | Selects the local multiprocess executor required for worker RPCs. Keep parallelism at one worker. |
| `--worker-extension-cls …LastHiddenStateWorker` | Adds worker initialization, vector retrieval, and cleanup methods. |
| `--kv-transfer-config …` | Registers the package's connector by module/class and sets its required `kv_producer` role. Include all three JSON fields exactly as shown. |
| `--host`, `--port` | Optional ordinary vLLM server settings; the example serves the client above on localhost port 8000. |

Replace the model and append your usual compatible vLLM options, such as
`--max-model-len` or `--dtype`. Runner selection follows upstream defaults;
set `VLLM_USE_V2_MODEL_RUNNER=0` or `1` yourself if you want an explicit choice.
The [supported configurations](#supported-configurations) apply to either launch
method, including the Triton CPU requirement for CPU V2.

The example enables only this plugin. If you also need other installed plugins,
include their entry-point names in the comma-separated `VLLM_PLUGINS` value,
for example `VLLM_PLUGINS=other_plugin,last_hidden_state`. Compatibility with
plugins that replace the same handler or worker internals is not established.

Use the **same client request shown above**: the launch settings enable the
capability, while `kv_transfer_params.return_last_hidden_state=true` opts in
per request. Running plain `vllm serve` without these activation settings does
not enable this package's extraction endpoint. The convenience launcher supplies
these required settings for you; direct invocation requires you to keep them
consistent with any YAML configuration or other CLI options.

## What exactly is returned?

For a fully processed prompt with T active positions, the vector is the final
layer's state at position T−1, **after the decoder's final output normalization**.
It is the state used to predict the first generated token, not that generated
token's own state. The response contains a JSON list of `hidden_size` float32
values. No token pooling or L2 normalization is applied.

The prompt includes the chat template and assistant boundary. To extract at an
unfinished assistant message, use `continue_final_message=True` and
`add_generation_prompt=False` in `extra_body`. Images use the usual chat
`image_url` content items. The same final-position rule applies in both cases.
Choose and evaluate your prompting and normalization strategy for your task.

## Supported configurations

| Setting | Current scope |
| --- | --- |
| vLLM version | **0.30.0**; local build suffixes such as `+cpu` are accepted |
| Models | Native Qwen3.5 architecture |
| API | Python `/v1/chat/completions`; extraction requires `n=1`, `stream=False`, and one output token |
| Execution | One local `mp` worker; CPU or CUDA adapters; V1 and experimental V2 |
| Scheduling | Prefix caching and chunked prefill implemented and tested on CPU |
| Speculation | MTP target-state capture; other speculative methods unsupported |
| Precision | Model dtype/quantization follow the installed vLLM backend |
| Exclusions | LoRA, distributed parallelism, other KV connectors, and the Rust frontend |

The launcher activates the extension and Python frontend and supplies
`--distributed-executor-backend mp` automatically. It leaves runner selection,
model precision, scheduling, and compilation to upstream vLLM and your settings.
Conflicting executor/worker/connector settings are rejected. See
[advanced configuration](MAINTENANCE.md#launcher-and-advanced-configuration).

V2 has real CPU eager and compiled coverage on Qwen3.5-0.8B BF16, including mixed
traffic, streaming, and measured prefix-cache hits. Eager V2 MTP also passes a
comparison with non-MTP target states while drafting and accepting tokens. CPU
V2 requires Triton CPU. For example, select V2 and MTP explicitly with:

```sh
VLLM_USE_V2_MODEL_RUNNER=1 vllm-last-hidden-state Qwen/Qwen3.5-0.8B \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

**Validation limits:** V2 remains experimental. Some image vectors exceed the
strict Transformers comparison tolerance on both V1 and V2. GPU execution,
CUDA graph replay, and async scheduling are not runtime-validated for this
package. The synchronous vector copy and retrieval RPC have a cost; these checks
are not performance benchmarks. See the [V2 results](validation/V2-v0.30.0.md)
and [v0.30.0 validation report](validation/UPGRADE-v0.30.0.md) for exact coverage,
including older CPU MTP failures without Triton CPU.

## Research background

Prompt-conditioned hidden-state representations have an established academic
history. This package makes one such readout available through a serving API;
it does not introduce a new embedding method or reproduce the papers' results.

- **[Scaling Sentence Embeddings with Large Language Models (PromptEOL)](https://aclanthology.org/2024.findings-emnlp.181/)**
  (Findings of EMNLP 2024) studies one-word completion prompts for causal-language-model
  sentence representations, including settings with and without fine-tuning.
- **[PromptReps](https://aclanthology.org/2024.emnlp-main.250/)** (EMNLP 2024)
  uses the final prompt token's hidden state and next-token logits for zero-shot
  document retrieval. Its dense representation is closely related to the state
  exposed here; this package does not implement its sparse retrieval component.
- **[E5-V: Universal Embeddings with Multimodal Large Language Models](https://arxiv.org/abs/2407.12580)**
  (2024) explores prompted text/image representations, including a training-free
  setting and a model trained contrastively on text pairs.
- **[FreeRet: MLLMs as Training-Free Retrievers](https://arxiv.org/abs/2509.24621)**
  (2025) investigates prompting and representation choice for multimodal retrieval.
  Its attention-level readout differs from the final post-normalization state
  returned here and is not implemented by this extension.

A correctly extracted state is not automatically a good similarity embedding.
Evaluate retrieval quality separately from tensor correctness, and cite the
underlying methods when using them in research.

## Built by NuMind

We build [NuExtract](https://nuextract.ai) to turn documents into usable data.
[NuExtract3](https://huggingface.co/numind/NuExtract3) is our open-weight 4B
vision-language model for structured extraction and document-to-Markdown
conversion, with multilingual text and image inputs.

Working with invoices, receipts, forms, or contracts? **[Try the NuExtract
platform and API](https://nuextract.ai)** to extract structured JSON and document
content without managing an inference server. For deployment in your own
environment, explore [NuExtract Enterprise](https://about.nuextract.ai/).

## Upstream and contributing

We are proposing native support in **[vLLM PR #57185](https://github.com/vllm-project/vllm/pull/57185)**.
The standalone package provides an alternative installation path for this
capability. Both use `kv_transfer_params.return_last_hidden_state` and
`kv_transfer_params.last_hidden_state`; supported configurations and transport
implementations differ. The PR includes additional position/layer metadata,
offline generation, and Python/Rust completion interfaces. Package tests do not
certify the PR, and the plugin is not required to use a build containing it.

Issues and contributions are welcome on [GitHub](https://github.com/numindai/vLLM-Last-Hidden-State).
For implementation details, tests, and releases, see [MAINTENANCE.md](MAINTENANCE.md).
The package is licensed under [Apache-2.0](LICENSE).
