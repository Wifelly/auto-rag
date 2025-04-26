import uvicorn

from service.settings import get_settings


def main() -> None:
    """Entrypoint of the application."""
    settings = get_settings()
    uvicorn.run(
        "service.main:get_application",
        workers=1,
        host=settings.host,
        port=settings.service_port,
        reload=settings.environment == "local",
        factory=True,
    )


if __name__ == "__main__":
    main()
