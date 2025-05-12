from sqlalchemy.ext.asyncio import AsyncSession

from service.monitoring.logger import logger
from service.services.embedding_container import embedding_manager
from service.services.embedding_service import EmbeddingService


class ContextRetriever:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.emb_svc = EmbeddingService(db)

    async def get_context(
        self,
        embedding_ids: list[int],
        query: str,
        top_k: int = 5,
        min_score: float = 0.2,
    ) -> tuple[str, list[str]]:
        parts: list[str] = []
        files: list[str] = []
        if not query:
            texts = [await self._load_by_id(doc_id) for doc_id in embedding_ids]
            return "\n\n".join(texts), []

        if not embedding_ids:
            logger.warning("[ContextRetriever] embedding_ids пуст — нет контекста")
            return "", []

        for emb_id in embedding_ids:
            emb = await self.emb_svc.get_embedding_by_id(emb_id)
            if not emb or not emb.vector_db_path:
                logger.warning(f"[ContextRetriever] Embedding {emb_id} не найден или без пути")
                continue

            try:
                embedding_manager.load_embedding(emb.vector_db_path)
            except Exception as e:
                logger.error(f"[ContextRetriever] Не удалось загрузить индекс {emb_id}: {e}")
                continue

            docs = embedding_manager.search(
                emb.vector_db_path,
                query=query,
                top_k=top_k,
                min_score=min_score,
            )
            logger.info(f"[ContextRetriever] По embedding {emb_id} найдено {len(docs)} фрагментов")

            for doc in docs:
                meta = doc.get("metadata", {})
                src = meta.get("source_file") or meta.get("filename") or "неизвестно"
                text = doc.get("content", "").strip()
                parts.append(f'Из файла "{src}":\n{text}')
                files.append(src)

        parts = parts[:top_k]
        unique_files = []
        for f in files:
            if f not in unique_files:
                unique_files.append(f)

        return "\n\n".join(parts), unique_files
