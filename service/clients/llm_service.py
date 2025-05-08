# --- llm_service.py ---

from dotenv import load_dotenv

from service.clients.local_llm import LocalLLM
from service.clients.remote_llm import RemoteLLM

load_dotenv(".env")


class LLMService:
    async def call(
        self,
        messages: list[dict],
        use_local_llm: bool = True,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        stop: list[str] = None,
    ) -> dict:
        llm = LocalLLM() if use_local_llm else RemoteLLM()
        try:
            response = await llm.generate(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
            )
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e), "code": "llm_generation_error"}


llm_service = LLMService()


async def call_gpt(
    messages: list[dict],
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
    stop: list[str] = None,
) -> dict:
    return await llm_service.call(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
    )


__all__ = ["LLMService", "LocalLLM", "RemoteLLM", "RagService", "llm_service"]
