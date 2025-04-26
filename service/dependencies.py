from fastapi import Request

from service.settings import get_settings


def sample_dep(request: Request) -> str:
    settings = get_settings()
    return f"Hello from {settings.component_name} /{request.url}"
