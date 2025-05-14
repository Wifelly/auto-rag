from time import perf_counter
from traceback import format_exc
from typing import Any, Awaitable, Callable, Final

from opentelemetry.trace import INVALID_SPAN, format_trace_id, get_current_span
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from service.monitoring.logger import contextualize, logger
from service.monitoring.metrics import REQUESTS_COUNT, REQUESTS_PROCESSING_TIME


class LoggingMiddleware(BaseHTTPMiddleware):
    _METADATA_FIELDS_TO_LOG: Final[set[str]] = {
        "user-id",
        "session-id",
        "external-request-id",
        "internal-request-id",
    }

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        headers = dict(request.headers)
        metadata = self._get_metadata(headers)

        with contextualize(metadata=metadata):
            return await call_next(request)

    def _get_metadata(self, headers: dict[str, str]) -> dict[str, Any]:
        metadata = {
            key: self._kebab_to_snake(value) for key in self._METADATA_FIELDS_TO_LOG if (value := headers.get(key))
        }

        if trace_id := _get_trace_id():
            metadata["trace_id"] = trace_id

        return metadata

    @staticmethod
    def _kebab_to_snake(string: str) -> str:
        return string.replace("-", "_")


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        method = request.method

        before_time = perf_counter()
        response = await call_next(request)
        after_time = perf_counter()

        REQUESTS_COUNT.labels(
            method=method,
            path=f"{request.method} {request.url.path}",
            status=response.status_code,
        ).inc()
        REQUESTS_PROCESSING_TIME.labels(
            method=method,
            path=f"{request.method} {request.url.path}",
            status=response.status_code,
        ).observe(
            after_time - before_time,
        )

        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        try:
            return await call_next(request)
        except Exception:
            logger.error(format_exc())

            return JSONResponse(
                status_code=200,
                content={
                    "message": "Internal Server Error",
                    "traceback": format_exc(),
                },
            )


class RequestBodyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            if content_type.startswith("application/json"):
                try:
                    request.state.request_body = await request.json()
                except Exception:
                    request.state.request_body = None
            else:
                request.state.request_body = None
        return await call_next(request)


def _get_trace_id() -> str | None:
    span = get_current_span()
    if span is INVALID_SPAN:
        return None

    context = span.get_span_context()
    return format_trace_id(context.trace_id)
