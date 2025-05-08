# --- local_llm.py ---

import os

from llama_cpp import Llama

from service.clients.base_llm import BaseLLM
from service.clients.prompt_builder import PromptBuilder
from service.config import Config
from service.monitoring.logger import logger


class LocalLLM(BaseLLM):
    def __init__(self):
        super().__init__(Config.LOCAL_GEN_TEMPERATURE, Config.LOCAL_GEN_TOP_P, Config.LOCAL_GEN_MAX_TOKENS)

        path = Config.GGUF_MODEL_PATH
        if not path or not os.path.exists(path):
            raise ValueError("GGUF модель не найдена или путь не указан")

        self.model = Llama(model_path=path, n_ctx=2048, n_threads=4, use_mlock=True)

    async def generate(self, messages: list[dict], temperature=None, top_p=None, max_tokens=None, stop=None) -> str:
        params = self.get_generation_params(temperature, top_p, max_tokens, stop)
        try:
            output = self.model(prompt=PromptBuilder.format_prompt(messages), **params)
            return output.get("choices", [{}])[0].get("text", "").strip() or "[Пустой ответ от локальной модели]"
        except Exception as e:
            logger.error(f"[LocalLLM] Ошибка генерации: {e}", exc_info=True)
            return "[Ошибка генерации локальной моделью]"
