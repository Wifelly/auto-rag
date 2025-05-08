from abc import ABC, abstractmethod

from service.config import Config


class BaseLLM(ABC):
    def __init__(self, temperature: float, top_p: float, max_tokens: int):
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_max_tokens = max_tokens

    def get_generation_params(self, temperature=None, top_p=None, max_tokens=None, stop=None):
        return {
            "temperature": temperature if temperature is not None else self.default_temperature,
            "top_p": top_p if top_p is not None else self.default_top_p,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "stop": stop if stop is not None else Config.DEFAULT_STOP,
        }

    @abstractmethod
    async def generate(self, messages: list[dict], **kwargs):
        pass
