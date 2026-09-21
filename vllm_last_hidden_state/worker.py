# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker initialization and request-scoped result RPC methods."""

from vllm.distributed.kv_transfer import get_kv_transfer_group

from .compat import install_capture, require_supported_vllm
from .connector import LastHiddenStateConnector


class LastHiddenStateWorker:
    def last_hidden_state_initialize(self):
        require_supported_vllm()
        connector = get_kv_transfer_group()
        if not isinstance(connector, LastHiddenStateConnector):
            raise ValueError("LastHiddenStateConnector must be configured")
        install_capture(self, connector)
        return {"hidden_size": connector.hidden_size}

    def last_hidden_state_take(self, handle):
        return get_kv_transfer_group().take(handle)

    def last_hidden_state_discard(self, handle):
        get_kv_transfer_group().discard(handle)
