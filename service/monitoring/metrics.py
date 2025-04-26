import time
from math import exp
from threading import active_count

from prometheus_client import (
    GC_COLLECTOR,
    PLATFORM_COLLECTOR,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
)
from prometheus_client.utils import INF

from service.settings import Settings

_STARTUP_TIMESTAMP = time.monotonic()

LATENCY_BUCKETS = (*[round(exp(x / 2) / 22, 3) for x in range(14)], INF)

SERVICE_INFO = Counter(
    "service_info",
    "Service name, tag and environment",
    ["component_name", "name", "image_tag", "environment", "kind"],
    registry=None,
)

REQUESTS_COUNT = Counter(
    "http_calls",
    "http calls count by path and status",
    ["path", "method", "status"],
    registry=None,
)

REQUESTS_PROCESSING_TIME = Histogram(
    "http_call_latency_in_seconds",
    "Histogram of requests latencies by path (in seconds)",
    ["path", "method", "status"],
    registry=None,
    buckets=LATENCY_BUCKETS,
)

UPTIME = Gauge("uptime_in_seconds", "Time elapsed from startup (in seconds)", registry=None)
UPTIME.set_function(lambda: time.monotonic() - _STARTUP_TIMESTAMP)

THREADS_COUNT = Gauge("threads_count", "Threads count", registry=None)
THREADS_COUNT.set_function(active_count)


def configure_metrics(
    settings: Settings,
) -> None:
    REGISTRY.unregister(GC_COLLECTOR)
    REGISTRY.unregister(PLATFORM_COLLECTOR)

    REGISTRY.register(UPTIME)
    REGISTRY.register(THREADS_COUNT)

    REGISTRY.register(REQUESTS_COUNT)
    REGISTRY.register(REQUESTS_PROCESSING_TIME)

    REGISTRY.register(SERVICE_INFO)

    SERVICE_INFO.labels(
        component_name=settings.component_name,
        name=settings.service_name,
        image_tag=settings.version,
        environment=settings.environment,
        kind=settings.service_kind,
    ).inc()
