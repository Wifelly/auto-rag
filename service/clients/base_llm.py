from abc import ABC, abstractmethod

from service.config import Config


class BaseLLM(ABC):
    def __init__(
        self,
        default_temperature: float | None = None,
        default_top_p: float | None = None,
        default_max_tokens: int | None = None,
    ):
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_max_tokens = default_max_tokens

    @abstractmethod
    async def generate(self, messages: list[dict], **kwargs) -> str:
        pass

    @abstractmethod
    async def generate_stream(self, messages: list[dict], **kwargs):
        pass

    def get_generation_params(
        self,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        return {
            "temperature": (temperature if temperature is not None else self.default_temperature),
            "top_p": (top_p if top_p is not None else self.default_top_p),
            "max_tokens": (max_tokens if max_tokens is not None else self.default_max_tokens),
            "stop": (stop if stop is not None else Config.DEFAULT_STOP),
        }
