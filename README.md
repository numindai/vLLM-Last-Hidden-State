# vLLM last hidden state

A private extension that returns the final **prompt** token's post-normalization
hidden state alongside a normal chat completion. It copies one vector from the
existing forward pass; it does not run a second model pass, pool tokens, or
normalize the vector to unit length.

## Installation

Use Python 3.12+ and install **vLLM 0.29.0 for your hardware** in your environment
first. This package deliberately does not install or replace vLLM/PyTorch.
Install this extension from the tagged GitLab revision using an SSH key with
access to the repository:

```sh
python -m pip install "git+ssh://git@gitlab.com/numind.tech/vllm_last_hidden_state.git@v0.29.0"
```

For development from a local checkout, use `python -m pip install -e .`.
Both expose the same launcher. Installing the package alone does not enable
extraction. The Git tag selects the extension revision; it is independent of the
Python package version (`0.1.0`).

The `vllm/` submodule is upstream source pinned to `v0.29.0`
(`98dff2a81d747d1dba01a47f939f48c3526d4206`) for inspection and validation.
Initialize it when you need that source:

```sh
git submodule update --init --recursive
```

The submodule does not control the vLLM installed in your Python environment.
The launcher, endpoint initialization, and worker initialization check installed
vLLM metadata. Only `0.29.0` is accepted; local build suffixes such as `+cpu` are
allowed. Other releases, prereleases, development builds, and missing installations
produce an error explaining what to install.

## Usage

Start the extension with the same model and runtime options you would give
`vllm serve`:

```sh
VLLM_USE_V2_MODEL_RUNNER=0 vllm-last-hidden-state Qwen/Qwen3.5-9B \
    --dtype bfloat16 --distributed-executor-backend mp
```

You can also use `python -m vllm_last_hidden_state.serve`. The launcher chooses
the Python frontend and activates the endpoint, worker extension, and connector.
It leaves model, device, scheduling, and compilation settings to you. The example
explicitly selects the supported V1 runner. CPU-specific environment setup is
described in the [validation notes](validation/HISTORY.md#cpu-environment-used-by-the-historical-commands).

Extraction is opt-in per request:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Describe a yellow bicycle briefly."}],
    max_completion_tokens=1,
    n=1,
    stream=False,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "kv_transfer_params": {"return_last_hidden_state": True},
    },
)
vector = response.model_dump()["kv_transfer_params"]["last_hidden_state"]
```

The vector represents the last input token, including any assistant prefill,
not the generated token. Images and assistant prefill use ordinary chat request
fields. Requests without the extraction flag follow the ordinary response path;
the enabled capture hook still performs a small metadata check.

Existing `VLLM_PLUGINS` entries are preserved and `last_hidden_state` is added
once. If unset, normally enabled non-endpoint plugins are preserved; other endpoint
plugins are not automatically enabled. Set `VLLM_PLUGINS=""` before launching
if you want only this extension.

The launcher checks effective worker/connector settings using vLLM's parser,
including YAML configuration and CLI overrides. Matching settings are retained.
A different worker or connector is rejected instead of silently overwritten.
If supplying `--kv-transfer-config` yourself, include all three required fields:

```json
{
  "kv_connector": "LastHiddenStateConnector",
  "kv_connector_module_path": "vllm_last_hidden_state.connector",
  "kv_role": "kv_producer"
}
```

Additional connector options are preserved. An explicit
`VLLM_USE_RUST_FRONTEND=1` is rejected because this endpoint requires the Python
frontend. Preserving other plugins does not establish compatibility with plugins
that replace the same chat handler or worker internals.

## Supported configurations

- Version policy: vLLM **0.29.0** only. See the [validation report](validation/UPGRADE-v0.29.0.md)
  for the exact runtime evidence and remaining gaps. CPU Qwen3.5-0.8B serving
  was exercised in eager and compiled modes. Two image comparisons failed the
  unchanged Transformers numerical thresholds; numerical parity is not guaranteed.
- Qwen3.5, BF16, unquantized or `compressed-tensors` weights; one local worker
  using `--distributed-executor-backend mp`.
- Non-streaming extraction with `n=1` and one generated token, via
  `max_completion_tokens=1` or `max_tokens=1`.
- V1 CPU/GPU runners; eager and compiled capture paths. Prefix caching and chunked
  prefill are implemented. GPU execution, CUDA graphs, and async scheduling need
  hardware validation; CPU execution cannot validate those paths.
- LoRA, speculative decoding (including MTP), distributed parallelism, and other
  KV connectors are unsupported.

Opted-in GPU requests synchronously copy one vector to CPU and use one retrieval
RPC. No zero-overhead claim is made. Historical CPU results and their numerical
limitations are in [HISTORY.md](validation/HISTORY.md).

## Development

With a hardware-appropriate vLLM installation already available:

```sh
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
ruff format --check .
```

Pytest collects only this package's tests in `validation/`; Ruff excludes the
upstream submodule. No distribution wheel is required for this workflow.

`compat.py` owns version checks, upstream CLI parsing, runner inspection,
batch metadata access, and CPU/GPU capture installation. `connector.py` selects
and stores the vector, `worker.py` exposes retrieval/cleanup RPCs, and
`endpoint.py` augments the existing chat response.

For upgrades and actual model/HTTP validation, use [MAINTENANCE.md](MAINTENANCE.md),
the [report template](validation/UPGRADE_REPORT_TEMPLATE.md), and the
[historical reproduction commands](validation/HISTORY.md#validation).
