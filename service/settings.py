from functools import cache

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
)


class TracingSettings(BaseSettings):
    agent_host: str = Field(default="otel-exporter")
    agent_port: int = Field(default=6831)
    insecure: bool = Field(default=True)

    class Config:
        env_prefix = "TRACING_"


class Settings(BaseSettings):
    component_name: str = Field(default="AutoRag")
    service_name: str = Field(default="Backend")
    service_alias: str = Field(default="auto-rag")
    service_platform: str = Field(default="generation")
    version: str = Field(default="0.0.1")
    service_kind: str = Field("API Service")

    environment: str = Field(default="local")
    log_level: str = Field(default="INFO")
    log_json_format: bool = Field(default=True)

    @property
    def tracing_service_name(self) -> str:
        return f"{self.service_platform}.{self.environment}.{self.service_alias}"

    host: str = Field(default="127.0.0.1")
    service_port: int = Field(default=8080)
    max_workers: int = Field(default=1)

    jaeger: TracingSettings = Field(default_factory=TracingSettings)
    metrics_server_port: int = Field(default=8000)

    DATABASE_URL: str = "postgresql+asyncpg://user:password@postgres:5432/embeddings_db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@cache
def get_settings() -> Settings:
    return Settings()
