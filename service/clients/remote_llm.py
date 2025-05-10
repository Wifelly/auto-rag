import aiohttp

from service.clients.base_llm import BaseLLM
from service.config import Config
from service.monitoring.logger import logger


class RemoteLLM(BaseLLM):
    def __init__(self):
        super().__init__(
            Config.DEFAULT_TEMPERATURE,
            Config.DEFAULT_TOP_P,
            Config.DEFAULT_MAX_TOKENS,
        )
        Config.validate()
        self.api_key = Config.GPT_API_KEY
        self.model = Config.GPT_MODEL
        self.url = Config.GPT_URL

    async def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        params = self.get_generation_params(temperature, top_p, max_tokens, stop)
        payload = {
            "model": self.model,
            "messages": messages,
            **{k: v for k, v in params.items() if v is not None},
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.post(self.url, json=payload, headers=headers) as resp:
                    data = await resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        return "[Empty response from GPT API]"
                    return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"[RemoteLLM] API error: {e}", exc_info=True)
            return "[GPT API call error]"

    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ):
        params = self.get_generation_params(temperature, top_p, max_tokens, stop=None)
        params.pop("stop", None)

        payload = {
            "model": self.model,
            "messages": messages,
            **{k: v for k, v in params.items() if v is not None},
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(self.url, json=payload, headers=headers) as resp:
                async for chunk in resp.content:
                    yield chunk
