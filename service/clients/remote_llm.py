import aiohttp

from service.clients.base_llm import BaseLLM
from service.config import Config
from service.monitoring.logger import logger


class RemoteLLM(BaseLLM):
    def __init__(self):
        super().__init__(None, None, None)

        self.api_key = Config.VSE_GPT_API_KEY
        self.model = Config.GPT_MODEL
        self.url = Config.GPT_URL

    async def generate(self, messages: list[dict], temperature=None, top_p=None, max_tokens=None, stop=None) -> str:
        params = self.get_generation_params(temperature, top_p, max_tokens, stop)

        payload = {
            "model": self.model,
            "messages": messages,
            **params,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.post(self.url, json=payload, headers=headers) as resp:
                    data = await resp.json()
                    choices = data.get("choices")
                    if not choices or not isinstance(choices, list):
                        return "[Пустой или некорректный ответ от GPT API]"

                    content = choices[0].get("message", {}).get("content", "").strip()
                    return content or "[Пустой ответ от удалённой модели]"
        except Exception as e:
            logger.error(f"[RemoteLLM] Ошибка при вызове API: {e}", exc_info=True)
            return "[Ошибка при вызове GPT API]"
