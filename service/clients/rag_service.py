import re
from enum import Enum

import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession

from service.clients.context_retriever import ContextRetriever
from service.clients.llm_service import llm_service
from service.clients.prompt_builder import PromptBuilder
from service.config import Config
from service.monitoring.logger import logger
from service.services.embedding_container import embedding_manager


class Mode(str, Enum):
    CHAT = "chat"
    RAG = "rag"
    WEB = "web"


class RagService:
    def __init__(self):
        self.web_enabled = True

    async def _search_web(self, query: str) -> tuple[str, list[str]]:
        api_key = Config.TAVILY_API_KEY
        if not api_key:
            logger.warning("[WebSearch] Tavily API ключ не настроен")
            return "", []

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": False,
        }

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp,
            ):
                data = await resp.json()
        except Exception as e:
            logger.error(f"[WebSearch] Ошибка при запросе: {e}")
            return "", []

        if not isinstance(data, dict):
            logger.error(f"[WebSearch] Ответ не является словарём: {data}")
            return "", []

        answer = ""
        sources = []

        raw_answer = data.get("answer")
        if isinstance(raw_answer, str):
            answer = raw_answer.strip()
        elif isinstance(raw_answer, dict):
            answer = raw_answer.get("text", "").strip()
            sources = raw_answer.get("sources", [])
        else:
            logger.error(f"[WebSearch] Некорректный формат поля 'answer': {raw_answer}")
            return "", []

        if not sources and isinstance(data.get("results"), list):
            sources = [r.get("url") for r in data["results"] if isinstance(r, dict) and "url" in r]

        return answer, sources

    async def answer_query(
        self,
        query: str,
        embedding_ids: list[int],
        db_session: AsyncSession,
        mode: Mode = Mode.RAG,
        top_k: int = 5,
        min_score: float = 0.0,
        use_web: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        user_stop = stop or Config.DEFAULT_STOP
        stop_tokens = [] if mode == Mode.WEB else user_stop

        initial_res = await llm_service.call(
            messages=PromptBuilder.build_initial_prompt(query),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
        )
        initial = initial_res.get("response", "[Ошибка генерации]")
        meta = {"source": Mode.CHAT.value, "files": [], "initial_response": initial}

        if mode == Mode.CHAT:
            return {"response": initial, "meta": meta}

        full_ctx = ""
        source_files = []

        if mode == Mode.WEB and use_web and self.web_enabled:
            web_ctx, urls = await self._search_web(query)
            if web_ctx:
                full_ctx = web_ctx
                source_files = urls
                meta["source"] = Mode.WEB.value
            else:
                return {"response": initial, "meta": meta}

        if mode == Mode.RAG or (mode == Mode.WEB and not full_ctx):
            for emb_id in embedding_ids:
                emb = await ContextRetriever(db_session).emb_svc.get_embedding_by_id(emb_id)
                if emb and emb.index_uid:
                    try:
                        embedding_manager.load_embedding(emb.index_uid)
                    except Exception as e:
                        logger.error(f"[RagService] Не удалось загрузить {emb.index_uid}: {e}")

            ctx_text, ctx_files = await ContextRetriever(db_session).get_context(
                embedding_ids=embedding_ids,
                query=query,
                top_k=top_k,
                min_score=min_score,
            )
            if ctx_text:
                full_ctx = ctx_text
                source_files = ctx_files
                meta["source"] = Mode.RAG.value
            else:
                return {"response": initial, "meta": meta}

        prompt_res = PromptBuilder.build_with_context(initial, query, full_ctx)
        refined_res = await llm_service.call(
            messages=prompt_res,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
        )
        raw_final = refined_res.get("response", initial)

        numbered = []
        num = 1
        for line in raw_final.splitlines():
            if re.match(r"^(\-|\*|\d+\.)\s+", line.strip()):
                clean = re.sub(r"^(\-|\*|\d+\.)\s+", "", line).strip()
                if clean:
                    numbered.append(f"{num}. {clean}")
                    num += 1
        final = "\n".join(numbered) if numbered else raw_final

        meta["files"] = source_files
        return {"response": final, "meta": meta}

    async def stream_answer_query(
        self,
        query: str,
        embedding_ids: list[int],
        db_session: AsyncSession,
        mode: Mode = Mode.RAG,
        top_k: int = 5,
        min_score: float = 0.0,
        use_web: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ):
        user_stop = stop or Config.DEFAULT_STOP
        stop_tokens = [] if mode == Mode.WEB else user_stop

        initial_res = await llm_service.call(
            messages=PromptBuilder.build_initial_prompt(query),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
        )
        initial = initial_res.get("response", "[Ошибка генерации]")

        yield f"data: {initial}\n\n"

        if mode == Mode.CHAT:
            return

        full_ctx = ""
        source_files = []

        if mode == Mode.WEB and use_web and self.web_enabled:
            web_ctx, urls = await self._search_web(query)
            if web_ctx:
                full_ctx = web_ctx
                source_files = urls
            else:
                return

        if mode == Mode.RAG or (mode == Mode.WEB and not full_ctx):
            for emb_id in embedding_ids:
                emb = await ContextRetriever(db_session).emb_svc.get_embedding_by_id(emb_id)
                if emb and emb.index_uid:
                    try:
                        embedding_manager.load_embedding(emb.index_uid)
                    except Exception as e:
                        logger.error(f"[RagService] Не удалось загрузить {emb.index_uid}: {e}")

            ctx_text, ctx_files = await ContextRetriever(db_session).get_context(
                embedding_ids=embedding_ids,
                query=query,
                top_k=top_k,
                min_score=min_score,
            )
            if ctx_text:
                full_ctx = ctx_text
                source_files = ctx_files
            else:
                return

        prompt = PromptBuilder.build_with_context(initial, query, full_ctx)

        async for token in llm_service.stream(
            messages=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
        ):
            yield f"data: {token}\n\n"

        if source_files:
            yield f"data: [FILES] {' | '.join(source_files)}\n\n"


rag_service = RagService()
