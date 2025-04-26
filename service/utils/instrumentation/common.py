from typing import Any, Awaitable, Callable, TypeVar

from opentelemetry.context import get_current
from opentelemetry.trace import Tracer

CarrierT = list[tuple[str, bytes]]
ItemT = TypeVar("ItemT")


def context_extraction_handler(tracer: Tracer) -> Callable:
    async def wrapper(
        func: Callable[..., Awaitable[None]],
        _: Any,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        context = get_current()

        with tracer.start_as_current_span(
            f"{func.__qualname__}",
            context=context,
        ):
            return await func(*args, **kwargs)

    return wrapper


def inner_trace_handler(tracer: Tracer) -> Callable:
    async def wrapper(
        func: Callable[..., Awaitable],
        _: Any,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        context = get_current()

        with tracer.start_as_current_span(
            f"{func.__qualname__}",
            context=context,
        ):
            return await func(*args, **kwargs)

    return wrapper


def trace_and_inject_handler(tracer: Tracer) -> Callable:
    async def wrapper(
        func: Callable[..., Awaitable],
        _: Any,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        context = get_current()

        with tracer.start_as_current_span(
            f"{func.__qualname__}",
            context=context,
        ):
            return await func(*args, **kwargs)

    return wrapper
