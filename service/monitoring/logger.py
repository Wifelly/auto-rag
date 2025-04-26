import contextlib
import logging
import sys
from contextvars import ContextVar
from typing import Any, Final, Iterator

from loguru import logger

from service.settings import get_settings

_context: ContextVar[dict[str, Any]] = ContextVar("logging_context", default={})
_LOG_RECORD_BUILT_IN_ATTRS: Final[list[str]] = [
    "asctime", "created", "color_message", "exc_info", "exc_text", "extra",
    "stack_info", "filename", "args", "funcName", "id", "levelname", "levelno",
    "lineno", "module", "msg", "msecs", "message", "name", "pathname", "process",
    "processName", "relativeCreated", "taskName", "thread", "threadName",
]

settings = get_settings()


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def get_extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    extra_fields = {
        key: value
        for key, value in record.__dict__.items()
        if key not in _LOG_RECORD_BUILT_IN_ATTRS
    }
    return {**_context.get(), **extra_fields}


@contextlib.contextmanager
def contextualize(**kwargs: Any) -> Iterator:
    logger.configure(extra=kwargs)
    try:
        yield
    finally:
        logger.configure(extra={})


def setup_logging(log_level: str, use_json_formatter: bool = False) -> None:
    logger.remove()

    log_format = {
        "time": "{time:YYYY-MM-DD HH:mm:ss.SSS}",
        "level": "{level}",
        "message": "{message}",
        "extra": "{extra}",
    }

    if use_json_formatter:
        format_string = "{\"time\": \"" + log_format["time"] + "\", \"level\": \"" + log_format["level"] + \
                        "\", \"message\": \"" + log_format["message"] + "\", \"extra\": " + log_format["extra"] + "}"
    else:
        format_string = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " + \
                        "<level>{level: <8}</level> | " + \
                        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " + \
                        "{message} | {extra}"

    logger.add(
        sys.stdout,
        format=format_string,
        level=log_level,
        serialize=use_json_formatter,
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    external_modules = ["fastapi", "uvicorn", "opentelemetry", "grpc"]
    external_log_level = "DEBUG" if log_level == "DEBUG" else "WARNING"

    for module in external_modules:
        mod_logger = logging.getLogger(module)
        mod_logger.handlers = [InterceptHandler()]
        mod_logger.propagate = False
        mod_logger.setLevel(external_log_level)


logger = logger.bind(service=settings.service_alias)