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
from service.monitoring.status import router as status_router
from service.monitoring.tracing import setup_tracer
from service.routes import assistants, chat_router, embeddings, health_router, metrics_router, rags
from service.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    settings = get_settings()
    configure_metrics(settings)
    await init_database()

    yield


def get_application() -> FastAPI:
    application = FastAPI(title="AutoRAG Service", openapi_version="3.0.2", lifespan=lifespan)

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
    application.include_router(rags.router, prefix="/api/v1", tags=["RAG"])
    application.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
    application.include_router(assistants.router, prefix="/api/v1", tags=["Assistant"])
    application.include_router(status_router, prefix="/status", tags=["Status"])

    return application


app = get_application()
