# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Version policy and real vLLM CLI parsing, without loading a model."""

import json
import sys
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_last_hidden_state import compat, serve
from vllm_last_hidden_state.endpoint import LastHiddenStatePlugin


@pytest.mark.parametrize("version", ["0.29.0", "0.29.0+cpu", "0.29.0+cu130"])
def test_supported_release_with_hardware_suffix(monkeypatch, version):
    monkeypatch.setattr(compat.metadata, "version", lambda _: version)
    assert compat.require_supported_vllm() == version


@pytest.mark.parametrize(
    "version", ["0.28.0", "0.30.0", "0.29.0rc1", "0.29.0.dev1", "0.29.0.post1", "dev"]
)
def test_other_releases_fail_with_actionable_message(monkeypatch, version):
    monkeypatch.setattr(compat.metadata, "version", lambda _: version)
    with pytest.raises(RuntimeError, match="requires 0.29.0"):
        compat.require_supported_vllm()


def test_missing_vllm_does_not_require_importing_it(monkeypatch):
    monkeypatch.setattr(
        compat.metadata, "version", Mock(side_effect=PackageNotFoundError)
    )
    with pytest.raises(RuntimeError, match="Install vLLM 0.29.0 for your hardware"):
        compat.require_supported_vllm()


def test_rejected_version_never_parses_or_executes_vllm(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _: "0.28.0")
    parse = Mock()
    execute = Mock()
    monkeypatch.setattr(serve, "prepare_arguments", parse)
    monkeypatch.setattr(serve.os, "execv", execute)
    with pytest.raises(SystemExit, match="Unsupported vLLM"):
        serve.main()
    parse.assert_not_called()
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_direct_plugin_activation_checks_version(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _: "0.28.0")
    with pytest.raises(RuntimeError, match="Unsupported vLLM"):
        await LastHiddenStatePlugin().init_state(None, None, None)


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        ("", "last_hidden_state"),
        ("other, last_hidden_state,other", "other,last_hidden_state"),
    ],
)
def test_explicit_plugin_allowlist_is_preserved(monkeypatch, existing, expected):
    monkeypatch.setenv("VLLM_PLUGINS", existing)
    monkeypatch.delenv("VLLM_USE_RUST_FRONTEND", raising=False)
    serve.enable_plugins()
    assert serve.os.environ["VLLM_PLUGINS"] == expected
    assert serve.os.environ["VLLM_USE_RUST_FRONTEND"] == "0"


def test_unset_allowlist_preserves_default_plugins_without_enabling_endpoints(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    monkeypatch.delenv("VLLM_USE_RUST_FRONTEND", raising=False)
    groups = []

    def entry_points(*, group):
        groups.append(group)
        return [SimpleNamespace(name=group)]

    monkeypatch.setattr(compat.metadata, "entry_points", entry_points)
    serve.enable_plugins()
    assert "vllm.endpoint_plugins" not in groups
    assert serve.os.environ["VLLM_PLUGINS"].split(",") == [*groups, "last_hidden_state"]


def test_explicit_rust_frontend_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "1")
    monkeypatch.setenv("VLLM_PLUGINS", "")
    with pytest.raises(ValueError, match="Python frontend is required"):
        serve.enable_plugins()


def test_original_arguments_are_preserved_and_activation_is_added():
    original = ["example/model", "--port", "8123", "--enforce-eager"]
    args = serve.prepare_arguments(original)
    assert args[: len(original)] == original
    settings = compat.parse_serve_settings(args)
    assert settings.worker_extension_cls == serve.WORKER_CLASS
    assert settings.kv_transfer_config.kv_connector == "LastHiddenStateConnector"
    assert settings.port == 8123
    assert settings.enforce_eager


@pytest.mark.parametrize("option", ["--worker-extension-cls", "--worker_extension_cls"])
@pytest.mark.parametrize("equals", [False, True])
def test_worker_conflict_in_all_cli_spellings(option, equals):
    option_args = [f"{option}=other.Worker"] if equals else [option, "other.Worker"]
    with pytest.raises(ValueError, match="worker-extension-cls conflicts"):
        serve.prepare_arguments(["example/model", *option_args])


def test_matching_configuration_preserves_extra_connector_options():
    config = {**serve.CONNECTOR_CONFIG, "kv_buffer_size": 2048}
    args = [
        "example/model",
        "--worker-extension-cls",
        serve.WORKER_CLASS,
        "--kv-transfer-config",
        json.dumps(config),
    ]
    assert serve.prepare_arguments(args) == args


@pytest.mark.parametrize(
    "key,value",
    [
        ("kv_connector", "Other"),
        ("kv_role", "kv_consumer"),
        ("kv_connector_module_path", "other"),
    ],
)
def test_connector_conflicts(key, value):
    config = {**serve.CONNECTOR_CONFIG, key: value}
    with pytest.raises(ValueError, match=f"{key} must be"):
        serve.prepare_arguments(
            ["example/model", "--kv-transfer-config", json.dumps(config)]
        )


def test_dotted_connector_configuration_is_checked():
    with pytest.raises(ValueError, match="kv_connector must be"):
        serve.prepare_arguments(
            [
                "example/model",
                "--kv-transfer-config.kv_connector",
                "Other",
                "--kv-transfer-config.kv_role",
                "kv_producer",
            ]
        )


@pytest.mark.parametrize("equals", [False, True])
def test_yaml_configuration_conflict(tmp_path, equals):
    config = tmp_path / "server.yaml"
    config.write_text("model: example/model\nworker-extension-cls: other.Worker\n")
    args = [f"--config={config}"] if equals else ["--config", str(config)]
    with pytest.raises(ValueError, match="worker-extension-cls conflicts"):
        serve.prepare_arguments(args)


def test_cli_override_of_yaml_uses_upstream_precedence(tmp_path):
    config = tmp_path / "server.yaml"
    config.write_text(
        "model: example/model\nworker-extension-cls: other.Worker\nport: 8123\n"
    )
    args = serve.prepare_arguments(
        ["--config", str(config), "--worker-extension-cls", serve.WORKER_CLASS]
    )
    settings = compat.parse_serve_settings(args)
    assert settings.port == 8123
    assert settings.worker_extension_cls == serve.WORKER_CLASS


def test_yaml_connector_conflict(tmp_path):
    config = tmp_path / "server.yaml"
    config.write_text(
        "model: example/model\nkv-transfer-config:\n  kv_connector: Other\n  kv_role: kv_producer\n"
    )
    with pytest.raises(ValueError, match="kv_connector must be"):
        serve.prepare_arguments(["--config", str(config)])


def test_main_executes_same_python_with_forwarded_arguments(monkeypatch):
    monkeypatch.setattr(compat.metadata, "version", lambda _: "0.29.0+cpu")
    monkeypatch.setenv("VLLM_PLUGINS", "other")
    monkeypatch.delenv("VLLM_USE_RUST_FRONTEND", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["vllm-last-hidden-state", "example/model", "--port", "8123"]
    )
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.main()
    executable, args = execute.call_args.args
    assert executable == sys.executable
    assert args[:7] == [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "example/model",
        "--port",
        "8123",
    ]
    assert serve.os.environ["VLLM_PLUGINS"] == "other,last_hidden_state"
