# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward vLLM serve arguments with explicit, conflict-checked activation."""

import json
import os
import sys

from .compat import (
    default_plugin_names,
    parse_serve_settings,
    require_supported_vllm,
)

WORKER_CLASS = "vllm_last_hidden_state.worker.LastHiddenStateWorker"
CONNECTOR_CONFIG = {
    "kv_connector": "LastHiddenStateConnector",
    "kv_connector_module_path": "vllm_last_hidden_state.connector",
    "kv_role": "kv_producer",
}


def prepare_arguments(arguments):
    # Upstream 0.30.0 expands YAML only for the two-token spelling.
    arguments = [
        part
        for arg in arguments
        for part in (arg.split("=", 1) if arg.startswith("--config=") else [arg])
    ]
    settings = parse_serve_settings(arguments)
    executor = settings.distributed_executor_backend
    if executor not in (None, "mp"):
        raise ValueError(
            "--distributed-executor-backend conflicts with hidden-state extraction: "
            f"expected 'mp', got {executor!r}. Remove the option or use 'mp' "
            "(also check your YAML config)."
        )
    worker = settings.worker_extension_cls
    if worker and worker != WORKER_CLASS:
        raise ValueError(
            f"--worker-extension-cls conflicts with hidden-state extraction: "
            f"expected {WORKER_CLASS!r}, got {worker!r}. Remove the option or "
            "use the required class (also check your YAML config)."
        )
    connector = settings.kv_transfer_config
    if connector is not None:
        for key, expected in CONNECTOR_CONFIG.items():
            actual = getattr(connector, key)
            if actual != expected:
                raise ValueError(
                    f"--kv-transfer-config conflicts with hidden-state extraction: "
                    f"{key} must be {expected!r}, got {actual!r}. Omit the option "
                    "to use the extension's connector, or provide all three "
                    "required connector fields (also check your YAML config)."
                )
    if executor is None:
        arguments += ["--distributed-executor-backend", "mp"]
    if not worker:
        arguments += ["--worker-extension-cls", WORKER_CLASS]
    if connector is None:
        arguments += ["--kv-transfer-config", json.dumps(CONNECTOR_CONFIG)]
    return arguments


def enable_plugins():
    existing = os.environ.get("VLLM_PLUGINS")
    names = (
        default_plugin_names()
        if existing is None
        else [name.strip() for name in existing.split(",") if name.strip()]
    )
    os.environ["VLLM_PLUGINS"] = ",".join(dict.fromkeys([*names, "last_hidden_state"]))
    frontend = os.environ.get("VLLM_USE_RUST_FRONTEND")
    if frontend not in (None, "0"):
        raise ValueError(
            "VLLM_USE_RUST_FRONTEND conflicts with hidden-state extraction: "
            "the Python frontend is required. Unset it or set it to 0."
        )
    os.environ["VLLM_USE_RUST_FRONTEND"] = "0"


def main():
    try:
        require_supported_vllm()
        enable_plugins()
        arguments = prepare_arguments(sys.argv[1:])
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"vllm-last-hidden-state: {exc}") from None
    os.execv(
        sys.executable,
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", *arguments],
    )


if __name__ == "__main__":
    main()
