# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 0.29.0 assumptions; audit this module when changing the upstream pin."""

from importlib import metadata

from packaging.version import InvalidVersion, Version

SUPPORTED_VLLM_VERSION = "0.29.0"


def require_supported_vllm():
    """Check installation metadata without importing vLLM or initializing devices."""
    try:
        installed = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        raise RuntimeError(
            "vLLM is not installed. Install vLLM 0.29.0 for your hardware in "
            "this Python environment before using vllm-last-hidden-state."
        ) from None
    try:
        # Local hardware suffixes (e.g. +cpu) are allowed; dev/rc/post releases
        # are different upstream versions and must be validated separately.
        supported = Version(installed).public == SUPPORTED_VLLM_VERSION
    except InvalidVersion:
        supported = False
    if not supported:
        raise RuntimeError(
            f"Unsupported vLLM version {installed!r}; vllm-last-hidden-state "
            f"requires {SUPPORTED_VLLM_VERSION} (hardware suffixes are allowed). "
            "Install the matching vLLM build for your hardware in this Python "
            "environment. Checking out the submodule does not update the installation."
        )
    return installed


def parse_serve_settings(arguments):
    """Let upstream resolve CLI aliases, dotted JSON and YAML precedence."""
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(prog="vllm")
    ServeSubcommand().subparser_init(parser.add_subparsers())
    return parser.parse_args(["serve", *arguments])


def default_plugin_names():
    """Preserve upstream's default non-endpoint plugin selection when unset."""
    groups = (
        "vllm.general_plugins",
        "vllm.io_processor_plugins",
        "vllm.platform_plugins",
        "vllm.stat_logger_plugins",
    )
    return [ep.name for group in groups for ep in metadata.entry_points(group=group)]


def prompt_batch_rows(runner):
    """Read metadata at the forward boundary, before the runner reuses buffers."""
    batch = runner.input_batch
    starts = runner.query_start_loc.np
    for index, request_id in enumerate(batch.req_ids):
        yield (
            runner.requests[request_id],
            int(batch.num_computed_tokens_cpu[index]),
            int(starts[index]),
            int(starts[index + 1]),
        )


def capture_model_forward(runner, connector):
    """Capture the target forward, outside compilation and CUDA graph replay.

    In 0.29.0 Qwen MTP's EagleProposer calls its own model directly, not this
    method. Copying here also precedes drafting and output-buffer reuse.
    """
    original = runner._model_forward

    def forward(*args, **kwargs):
        output = original(*args, **kwargs)
        connector.capture_batch(runner, output)
        return output

    runner._model_forward = forward


def install_capture(worker, connector):
    """Validate the concrete model/runner and attach one instance-local hook."""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    model = worker.get_model()
    language_model = getattr(model, "language_model", model)
    if not isinstance(getattr(language_model, "model", None), Qwen3_5Model):
        raise ValueError("Only Qwen3.5 is currently supported")
    runner = worker.model_runner
    if type(runner) not in (CPUModelRunner, GPUModelRunner):
        raise ValueError("Last hidden state requires the V1 CPU or GPU model runner")
    if getattr(worker, "_last_hidden_state_initialized", False):
        return
    if type(runner) is CPUModelRunner:
        # MTP shares embeddings/lm_head, not this outer target module. Attaching
        # to an inner shared layer would capture draft forwards as well.

        def capture(module, inputs, output):
            connector.capture_batch(runner, output)

        worker._last_hidden_state_hook = model.register_forward_hook(capture)
    else:
        capture_model_forward(runner, connector)
    worker._last_hidden_state_initialized = True
