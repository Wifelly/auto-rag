from service.clients.llm_service import LLMService
from service.monitoring.logger import logger

llm_service = LLMService()


class LLMResponder:
    @staticmethod
    async def respond(messages: list[dict], use_local_llm: bool = True, **generation_params) -> str:
        result = await llm_service.call(messages, use_local_llm=use_local_llm, **generation_params)

        if result["success"]:
            return result["response"]

        logger.error(f"[LLMResponder] Ошибка LLM: {result['error']}")
        return "[Ошибка генерации ответа]"
