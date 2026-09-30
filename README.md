# vLLM last hidden state

An Apache-2.0 licensed extension that returns the final **prompt** token's post-normalization
hidden state alongside a normal chat completion. It copies one vector from the
existing forward pass; it does not run a second model pass, pool tokens, or
normalize the vector to unit length.

## Installation

Use Python 3.12+ and install **vLLM 0.30.0 for your hardware** in your environment
first. This package deliberately does not install or replace vLLM/PyTorch.
Install the current GitHub source without Git or SSH credentials:

```sh
python -m pip install "vllm-last-hidden-state @ https://github.com/numindai/vLLM-Last-Hidden-State/archive/refs/heads/main.zip"
```

For reproducible deployments, replace `refs/heads/main` with a tested full commit
SHA. The source archive does not download the upstream vLLM submodule, which is
unnecessary for installation. HTTPS Git installs also work, but require Git and
may fetch the large upstream submodule:

```sh
python -m pip install "vllm-last-hidden-state @ git+https://github.com/numindai/vLLM-Last-Hidden-State.git@main"
```

Published [GitHub releases](https://github.com/numindai/vLLM-Last-Hidden-State/releases)
will carry a small `py3-none-any.whl` file. Install its download URL with
`python -m pip install <wheel-url>`; no source checkout or build is needed.
PyPI publishing is optional and is not enabled by default. See the
[release instructions](MAINTENANCE.md#publishing-releases).

To install from a local checkout:

```sh
python -m pip install .
```

For development, use `python -m pip install -e .`.
Both expose the same launcher. Installing the package alone does not enable
extraction. The package version matches the supported vLLM release: both are
`0.30.0`. Keep the project version in `pyproject.toml`,
`compat.SUPPORTED_VLLM_VERSION`, and this compatibility documentation aligned
when upgrading. The historical extension tag `v0.29.0` is for vLLM 0.29.0 and
does not contain this upgrade.

The `vllm/` submodule is upstream source pinned to `v0.30.0`
(`ced6857afa0ea7b2e3f0846a62e1394e90f15607`) for inspection and validation.
Initialize it when you need that source:

```sh
git submodule update --init --recursive
```

The submodule does not control the vLLM installed in your Python environment.
The launcher, endpoint initialization, and worker initialization check installed
vLLM metadata. Only `0.30.0` is accepted; local build suffixes such as `+cpu` are
allowed. Other releases, prereleases, development builds, and missing installations
produce an error explaining what to install.

## Usage

Start the extension with the same model and runtime options you would give
`vllm serve`:

```sh
VLLM_USE_V2_MODEL_RUNNER=0 vllm-last-hidden-state Qwen/Qwen3.5-9B \
    --distributed-executor-backend mp
```

You can also use `python -m vllm_last_hidden_state.serve`. The launcher chooses
the Python frontend and activates the endpoint, worker extension, and connector.
It leaves model, device, scheduling, and compilation settings to you. The example
explicitly selects the supported V1 runner. CPU-specific environment setup is
described in the [validation notes](validation/HISTORY.md#cpu-environment-used-by-the-historical-commands).

Enable Qwen3.5 MTP with the ordinary vLLM option:

```sh
VLLM_USE_V2_MODEL_RUNNER=0 vllm-last-hidden-state numind/NuExtract3-W4A16 \
    --distributed-executor-backend mp \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

No `--dtype` or `--quantization` flag is required by this package. vLLM resolves
the checkpoint defaults and validates its kernels for your hardware. Explicit
upstream precision options are passed through unchanged. Returned vectors are
copied to float32 for JSON serialization, independently of model precision.
MTP drafts tokens for ordinary multi-token generation; extraction still returns the
target model's last prompt state with a one-token completion.

CPU MTP is currently blocked on the tested vLLM 0.30.0 build without Triton-CPU.
The upstream CPU fallbacks do not bind all of the new speculative-kernel
dispatchers: two speculative tokens fail in draft metadata updates, and one
speculative token still fails during ordinary multi-token rejection sampling.
Ordinary vLLM with the extension disabled reproduces the two-token failure.
Prefix-cached CPU MTP also hits the upstream Mamba precopy kernel error;
`--no-enable-prefix-caching` alone no longer makes this CPU configuration work.
The launcher does not alter these settings or patch upstream sampling kernels.
CUDA MTP remains implemented but untested here; Qwen3.5 MTP rejects Mamba cache
mode `all`. See the [0.30.0 report](validation/UPGRADE-v0.30.0.md) for failures and
[historical MTP validation](validation/MTP-v0.29.0.md) for the older working CPU build.

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

- Version policy: vLLM **0.30.0** only. See the [upgrade report](validation/UPGRADE-v0.30.0.md)
  for source audit, runtime evidence, and remaining gaps. CPU 9B BF16 short
  text/image and INT4 text suites passed in eager and compiled modes. Four BF16
  image comparisons failed the unchanged Transformers thresholds; numerical
  parity is not guaranteed. Historical 0.29.0 results remain separate.
- Qwen3.5; one local worker using `--distributed-executor-backend mp`.
  Dtype and quantization support follow the installed vLLM backend; this extension
  imposes no additional precision whitelist.
- Non-streaming extraction with `n=1` and one generated token, via
  `max_completion_tokens=1` or `max_tokens=1`.
- V1 CPU/GPU runners; eager and compiled capture paths. Prefix caching and chunked
  prefill are implemented. GPU execution, CUDA graphs, and async scheduling need
  hardware validation; CPU execution cannot validate those paths.
- MTP target-state capture is retained; other speculative methods remain
  unsupported. CPU MTP on the tested 0.30.0 backend has the upstream failures
  above. Historical CPU MTP successes on 0.29.0 do not validate 0.30.0.
- LoRA, distributed parallelism, and other
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
