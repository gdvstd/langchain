"""Tests for middleware wrapper memoization behavior."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain.agents import factory as agent_factory
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse


class FirstMiddleware(AgentMiddleware):
    """Middleware that records wrap_model_call order."""

    def __init__(self, log: list[str], async_log: list[str]) -> None:
        super().__init__()
        self._log = log
        self._async_log = async_log

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        self._log.append("first-before")
        result = handler(request)
        self._log.append("first-after")
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self._async_log.append("first-before")
        result = await handler(request)
        self._async_log.append("first-after")
        return result


class SecondMiddleware(AgentMiddleware):
    """Middleware that records wrap_model_call order."""

    def __init__(self, log: list[str], async_log: list[str]) -> None:
        super().__init__()
        self._log = log
        self._async_log = async_log

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        self._log.append("second-before")
        result = handler(request)
        self._log.append("second-after")
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self._async_log.append("second-before")
        result = await handler(request)
        self._async_log.append("second-after")
        return result


def _build_request() -> ModelRequest:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    return ModelRequest(model=model, messages=[])


def test_model_call_wrapper_cache_reuse_and_order() -> None:
    agent_factory._WRAP_MODEL_CALL_CACHE.clear()
    agent_factory._AWRAP_MODEL_CALL_CACHE.clear()

    call_log: list[str] = []
    async_log: list[str] = []
    middleware = [FirstMiddleware(call_log, async_log), SecondMiddleware(call_log, async_log)]

    wrap_handler, awrap_handler = agent_factory._compose_middleware_wrappers(
        sync_middleware=middleware,
        async_middleware=middleware,
        sync_attr="wrap_model_call",
        async_attr="awrap_model_call",
        sync_chain=agent_factory._chain_model_call_handlers,
        async_chain=agent_factory._chain_async_model_call_handlers,
        sync_cache=agent_factory._WRAP_MODEL_CALL_CACHE,
        async_cache=agent_factory._AWRAP_MODEL_CALL_CACHE,
    )
    wrap_handler_again, awrap_handler_again = agent_factory._compose_middleware_wrappers(
        sync_middleware=middleware,
        async_middleware=middleware,
        sync_attr="wrap_model_call",
        async_attr="awrap_model_call",
        sync_chain=agent_factory._chain_model_call_handlers,
        async_chain=agent_factory._chain_async_model_call_handlers,
        sync_cache=agent_factory._WRAP_MODEL_CALL_CACHE,
        async_cache=agent_factory._AWRAP_MODEL_CALL_CACHE,
    )

    assert wrap_handler is wrap_handler_again
    assert awrap_handler is awrap_handler_again

    request = _build_request()

    def base_handler(_: ModelRequest) -> ModelResponse:
        call_log.append("handler")
        return ModelResponse(result=[AIMessage(content="ok")], structured_response=None)

    assert wrap_handler is not None
    wrap_handler(request, base_handler)

    assert call_log == [
        "first-before",
        "second-before",
        "handler",
        "second-after",
        "first-after",
    ]

    expected_key = ("FirstMiddleware", "SecondMiddleware")
    assert list(agent_factory._WRAP_MODEL_CALL_CACHE.keys()) == [expected_key]
    assert list(agent_factory._AWRAP_MODEL_CALL_CACHE.keys()) == [expected_key]


@pytest.mark.asyncio
async def test_async_model_call_wrapper_cache_reuse_and_order() -> None:
    agent_factory._WRAP_MODEL_CALL_CACHE.clear()
    agent_factory._AWRAP_MODEL_CALL_CACHE.clear()

    call_log: list[str] = []
    async_log: list[str] = []
    middleware = [FirstMiddleware(call_log, async_log), SecondMiddleware(call_log, async_log)]

    _, awrap_handler = agent_factory._compose_middleware_wrappers(
        sync_middleware=middleware,
        async_middleware=middleware,
        sync_attr="wrap_model_call",
        async_attr="awrap_model_call",
        sync_chain=agent_factory._chain_model_call_handlers,
        async_chain=agent_factory._chain_async_model_call_handlers,
        sync_cache=agent_factory._WRAP_MODEL_CALL_CACHE,
        async_cache=agent_factory._AWRAP_MODEL_CALL_CACHE,
    )

    request = _build_request()

    async def base_handler(_: ModelRequest) -> ModelResponse:
        async_log.append("handler")
        return ModelResponse(result=[AIMessage(content="ok")], structured_response=None)

    assert awrap_handler is not None
    await awrap_handler(request, base_handler)

    assert async_log == [
        "first-before",
        "second-before",
        "handler",
        "second-after",
        "first-after",
    ]
