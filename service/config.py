import os

from dotenv import load_dotenv

load_dotenv(".env")


class ConfigError(Exception):
    pass


class Config:
    GPT_API_KEY: str | None = os.getenv("GPT_API_KEY")
    GPT_URL: str | None = os.getenv("GPT_URL")
    GPT_MODEL: str = os.getenv("GPT_MODEL", "openai/gpt-4.1-mini")

    TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", 0.2))
    DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", 0.9))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", 1024))
    DEFAULT_STOP: list[str] = []

    SYSTEM_PROMPT: str = "Ты — медицинский ассистент. Отвечай чётко, структурировано и обоснованно."

    INDEX_DIR: str = os.getenv("INDEX_DIR", "saved_indexes")

    @classmethod
    def validate(cls):
        if not (cls.GPT_API_KEY and cls.GPT_URL):
            raise ConfigError("Не заданы параметры доступа к удалённому GPT API: GPT_API_KEY и/или GPT_URL.")
        if not cls.TAVILY_API_KEY:
            logger = __import__("service.monitoring.logger", fromlist=["logger"]).logger
            logger.warning("[Config] Не задан TAVILY_API_KEY — веб-поиск отключён.")
