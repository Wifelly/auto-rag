from service.clients.llm_service import llm_service
from service.monitoring.logger import logger


class LLMResponder:
    @staticmethod
    async def respond(
        messages: list[dict],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        result = await llm_service.call(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )

        if result.get("success"):
            return result["response"]

        error = result.get("error", "Неизвестная ошибка")
        logger.error(f"[LLMResponder] Ошибка LLM: {error}")
        return "[Ошибка генерации ответа]"
