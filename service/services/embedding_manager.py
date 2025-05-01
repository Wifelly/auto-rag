# --- embedding_manager.py ---

from pathlib import Path

from langchain_community.vectorstores import FAISS

from service.models.embedding.embedding import DEFAULT_MODEL_NAME, EmbeddingModel
from service.monitoring.logger import logger


class EmbeddingManager:
    def __init__(self, base_dir: str = "saved_indexes", model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.base_dir = Path(base_dir)
        self.model = EmbeddingModel(model_name=model_name)
        self.loaded_embeddings: dict[str, FAISS] = {}

        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Создана директория для индексов: {self.base_dir}")

    def load_embedding(self, emb_id: str) -> bool:
        index_path = self.base_dir / emb_id
        faiss_file = index_path / "index.faiss"
        pkl_file = index_path / "index.pkl"

        if not faiss_file.exists() or not pkl_file.exists():
            logger.error(f"Не найден index.faiss или index.pkl по пути: {index_path}")
            return False

        try:
            db = FAISS.load_local(str(index_path), self.model, allow_dangerous_deserialization=True)
            self.loaded_embeddings[emb_id] = db
            logger.info(f"Индекс {emb_id} успешно загружен в память.")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса {emb_id}: {e}")
            return False

    def unload_embedding(self, emb_id: str) -> bool:
        if emb_id in self.loaded_embeddings:
            del self.loaded_embeddings[emb_id]
            logger.info(f"Индекс {emb_id} выгружен из памяти.")
            return True
        logger.warning(f"Индекс {emb_id} не был загружен.")
        return False

    def get_loaded_embeddings(self) -> list[str]:
        return list(self.loaded_embeddings.keys())

    def search(self, emb_id: str, query: str, top_k: int = 5, min_score: float = 0.0) -> list[dict] | None:
        if emb_id not in self.loaded_embeddings:
            logger.error(f"Индекс {emb_id} не загружен в память.")
            return None

        try:
            db = self.loaded_embeddings[emb_id]
            docs_and_scores = db.similarity_search_with_score(query, k=top_k)

            results = [
                {"content": doc.page_content, "score": float(score), "metadata": doc.metadata}
                for doc, score in docs_and_scores
                if score >= min_score
            ]

            logger.info(f"Поиск в индексе {emb_id} завершён. Найдено: {len(results)}")
            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске в индексе {emb_id}: {e}")
            return None
