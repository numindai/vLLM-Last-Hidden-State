# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Augment the existing chat handler through endpoint-plugin initialization."""

import asyncio
import logging
from http import HTTPStatus
from uuid import uuid4

from . import FLAG, HANDLE
from .compat import require_supported_vllm

logger = logging.getLogger(__name__)


class ChatWithLastHiddenState:
    def __init__(self, original, engine_client):
        self.original = original
        self.engine = engine_client

    def __getattr__(self, name):
        return getattr(self.original, name)

    async def create_chat_completion(self, request, raw_request=None):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponse,
        )

        params = request.kv_transfer_params or {}
        if not params.get(FLAG):
            return await self.original.create_chat_completion(request, raw_request)
        limit = (
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens
        )
        if params.get(FLAG) is not True:
            return self.original.create_error_response(f"{FLAG} must be true")
        if request.stream or (request.n or 1) != 1 or limit != 1:
            return self.original.create_error_response(
                "Hidden-state extraction requires stream=false, n=1, and "
                "max_tokens=1 (or max_completion_tokens=1)"
            )
        handle = uuid4().hex
        request = request.model_copy(
            update={"kv_transfer_params": {**params, HANDLE: handle}}
        )
        consumed = False
        try:
            response = await self.original.create_chat_completion(request, raw_request)
            if not isinstance(response, ChatCompletionResponse):
                return response
            if (response.kv_transfer_params or {}).get(HANDLE) != handle:
                raise RuntimeError("Completion did not contain the extraction handle")
            async with asyncio.timeout(30):
                results = await self.engine.collective_rpc(
                    "last_hidden_state_take", args=(handle,)
                )
            if len(results) != 1 or results[0] is None:
                raise RuntimeError("Worker did not capture the requested hidden state")
            result = results[0]
            consumed = True
            if "error" in result:
                raise RuntimeError(result["error"])
            response.kv_transfer_params = {"last_hidden_state": result["vector"]}
            return response
        except (RuntimeError, TimeoutError) as exc:
            logger.exception("Hidden-state extraction failed")
            return self.original.create_error_response(
                f"Hidden-state extraction failed: {exc}",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        finally:
            if not consumed:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(
                            self.engine.collective_rpc(
                                "last_hidden_state_discard", args=(handle,)
                            )
                        ),
                        timeout=5,
                    )
                except (Exception, asyncio.CancelledError):
                    logger.warning("Could not discard hidden state %s", handle)


class LastHiddenStatePlugin:
    name = "last_hidden_state"
    required_tasks = ("generate",)

    def attach_router(self, app):
        pass

    async def init_state(self, engine_client, state, args):
        require_supported_vllm()
        if engine_client is None or state.openai_serving_chat is None:
            raise ValueError("Last hidden state requires a chat generation server")
        if isinstance(state.openai_serving_chat, ChatWithLastHiddenState):
            return
        await engine_client.collective_rpc("last_hidden_state_initialize")
        state.openai_serving_chat = ChatWithLastHiddenState(
            state.openai_serving_chat, engine_client
        )
