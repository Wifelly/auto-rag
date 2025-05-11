from dotenv import load_dotenv

from service.clients.remote_llm import RemoteLLM
from service.config import Config

load_dotenv(".env")


class LLMService:
    async def call(
        self,
        messages: list[dict],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        llm = RemoteLLM()

        if max_tokens is None:
            max_tokens = Config.DEFAULT_MAX_TOKENS
        if stop is None:
            stop = Config.DEFAULT_STOP

        try:
            resp = await llm.generate(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
            )
            return {"success": True, "response": resp}
        except Exception as e:
            return {"success": False, "error": str(e), "code": "llm_error"}

    async def stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ):
        llm = RemoteLLM()

        if max_tokens is None:
            max_tokens = Config.DEFAULT_MAX_TOKENS
        if stop is None:
            stop = Config.DEFAULT_STOP

        async for token in llm.generate_stream(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        ):
            yield token


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
