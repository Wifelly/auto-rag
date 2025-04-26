from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

from service.settings import Settings


def setup_tracer(settings: Settings) -> Tracer:
    resource = Resource(
        attributes={
            SERVICE_NAME: settings.tracing_service_name,
        },
    )
    processor = BatchSpanProcessor(
        span_exporter=OTLPSpanExporter(
            endpoint=f"{settings.jaeger.agent_host}:{settings.jaeger.agent_port}",
            insecure=settings.jaeger.insecure,
        ),
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    return trace.get_tracer(__name__)
