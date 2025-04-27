from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

from service.database.init_db import init_database
from service.middlewares import (
    ErrorHandlerMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    RequestBodyMiddleware,
)
from service.monitoring.metrics import configure_metrics
from service.monitoring.tracing import setup_tracer
from service.routes import health_router, metrics_router, embeddings
from service.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    settings = get_settings()
    configure_metrics(settings)
    await init_database()
    yield


def get_application() -> FastAPI:
    application = FastAPI(title="AutoRAG Service", lifespan=lifespan)

    application.add_middleware(LoggingMiddleware)
    application.add_middleware(ErrorHandlerMiddleware)
    application.add_middleware(OpenTelemetryMiddleware)
    application.add_middleware(RequestBodyMiddleware)
    application.add_middleware(MetricsMiddleware)

    settings = get_settings()
    setup_tracer(settings)

    application.include_router(health_router, tags=["Health"])
    application.include_router(metrics_router, tags=["Metrics"])
    application.include_router(embeddings.router, prefix="/api/v1", tags=["Embeddings"])
    return application


app = get_application()
