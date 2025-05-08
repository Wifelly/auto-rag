from service.monitoring.logger import logger
from service.services.embedding_container import embedding_manager
from service.services.embedding_service import EmbeddingService


class ContextRetriever:
    def __init__(self, rag_service, db_session):
        self.rag_service = rag_service
        self.db_session = db_session

    async def retrieve(self, query: str, emb_id: int, top_k: int, min_score: float, use_web: bool):
        embedding_docs = []
        source_files = set()
        parts = []
        source_label = "llm"
        top_score = 0.0

        if use_web and getattr(self.rag_service, "web_enabled", False):
            web_context, web_sources = await self.rag_service._search_web(query)
            if web_context:
                parts.append(web_context)
            if web_sources:
                source_files.update(web_sources)
            source_label = "web"

        else:
            embedding_service = EmbeddingService(self.db_session)
            embedding = await embedding_service.get_embedding_by_id(emb_id)
            if embedding and embedding.vector_db_path:
                embedding_docs = (
                    embedding_manager.search(
                        vector_db_path=embedding.vector_db_path,
                        query=query,
                        top_k=top_k,
                        min_score=min_score,
                    )
                    or []
                )

                logger.info(
                    f"[RAG] Найдено {len(embedding_docs)} документов из эмбеддинга {embedding.name or embedding.id}"
                )
                source_label = "rag"

        if embedding_docs:
            for idx, doc in enumerate(embedding_docs):
                parts.append(doc["content"] if isinstance(doc, dict) else doc.page_content)
                meta = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
                source = meta.get("source_file") or meta.get("filename") or meta.get("path") or meta.get("source")
                if source:
                    source_files.add(source)

            score = embedding_docs[0].get("score") if isinstance(embedding_docs[0], dict) else None
            if isinstance(score, (int, float)):
                top_score = round(score, 4)

        return "\n\n".join(parts), source_label, sorted(source_files), top_score
