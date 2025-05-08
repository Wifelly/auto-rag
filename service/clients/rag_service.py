from enum import Enum
from pathlib import Path

import aiohttp

from service.clients.context_retriever import ContextRetriever
from service.clients.prompt_builder import PromptBuilder
from service.clients.responder import LLMResponder
from service.config import Config
from service.database.models import Embedding
from service.monitoring.logger import logger
from service.services.embedding_container import embedding_manager


class Mode(str, Enum):
    AUTO = "auto"
    RAG = "rag"
    CHAT = "chat"


class RagService:
    def __init__(self):
        self.web_enabled = True

    async def _search_web(self, query: str) -> tuple[str, list[str]]:
        api_key = Config.TAVILY_API_KEY
        if not api_key:
            logger.warning("[WebSearch] Tavily API ключ не настроен")
            return "[Tavily API ключ не настроен]", []

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": False,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    answer = data.get("answer") or ""
                    sources = data.get("results", [])

                    source_urls = []
                    parts = [f"### Результаты из интернета:\n{answer.strip()}"]

                    for src in sources[:3]:
                        title = src.get("title") or "Источник"
                        url = src.get("url")
                        if url:
                            source_urls.append(url)
                            parts.append(f"- {title}: {url}")

                    full_context = "\n".join(parts).strip()
                    return full_context or "[Ничего не найдено]", source_urls

        except Exception as e:
            logger.warning(f"[WebSearch] Ошибка Tavily API: {e}")
            return f"[Ошибка Tavily API: {e}]", []

    async def answer_query(
        self,
        query: str,
        embedding: Embedding,
        db_session,
        mode: Mode = Mode.AUTO,
        top_k: int = 5,
        min_score: float = 0.0,
        use_web: bool = False,
        use_local_llm: bool = True,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        stop: list[str] = None,
    ) -> dict:
        vector_db_path = embedding.vector_db_path
        index_uid = Path(vector_db_path).name

        logger.info(f"[answer_query] Проверка index_uid={index_uid}")
        logger.info(f"[answer_query] Загруженные индексы: {embedding_manager.get_loaded_embeddings()}")

        if index_uid not in embedding_manager.get_loaded_embeddings():
            logger.warning(f"[answer_query] Индекс {index_uid} не загружен в память. Пробуем загрузить.")
            try:
                await embedding_manager.load_embedding(embedding.vector_db_path)
                logger.info(f"[answer_query] Индекс {index_uid} успешно загружен.")
            except Exception as e:
                logger.error(f"[answer_query] Ошибка при загрузке индекса {index_uid}: {e}")
                return {
                    "response": "Ошибка загрузки embedding: " + str(e),
                    "meta": {
                        "context_found": False,
                        "source": "error",
                        "files": [],
                        "top_score": 0.0,
                        "initial_response": None,
                    },
                }

        self.web_enabled = use_web
        context_retriever = ContextRetriever(self, db_session)

        initial_prompt = PromptBuilder.build_initial_prompt(query)
        initial_response = await LLMResponder.respond(
            initial_prompt,
            use_local_llm=use_local_llm,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or Config.DEFAULT_STOP,
        )

        if mode == Mode.CHAT:
            return {
                "response": initial_response,
                "meta": {
                    "context_found": False,
                    "source": "llm",
                    "files": [],
                    "top_score": 0.0,
                    "initial_response": initial_response,
                },
            }

        context, source_label, source_files, top_score = await context_retriever.retrieve(
            query=query,
            emb_id=embedding.id,
            top_k=top_k,
            min_score=min_score,
            use_web=use_web,
        )

        if not context or not context.strip():
            logger.warning(f"[RAG] Контекст не найден для запроса: {query}")
            return {
                "response": initial_response,
                "meta": {
                    "context_found": False,
                    "source": source_label,
                    "files": source_files,
                    "top_score": top_score,
                    "initial_response": initial_response,
                },
            }

        contextual_prompt = PromptBuilder.build_with_context(
            initial=initial_response,
            query=query,
            context=context,
        )

        refined_response = await LLMResponder.respond(
            contextual_prompt,
            use_local_llm=use_local_llm,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or Config.DEFAULT_STOP,
        )

        return {
            "response": refined_response,
            "meta": {
                "context_found": True,
                "source": source_label,
                "files": source_files,
                "top_score": top_score,
                "initial_response": initial_response,
            },
        }


rag_service = RagService()
