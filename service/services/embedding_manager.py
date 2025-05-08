from collections import OrderedDict
from pathlib import Path

from langchain_community.vectorstores import FAISS

from service.models.embedding.embedding import DEFAULT_MODEL_NAME, EmbeddingModel
from service.monitoring.logger import logger


class EmbeddingManager:
    def __init__(
        self, base_dir: str = "saved_indexes", model_name: str = DEFAULT_MODEL_NAME, max_cache_size: int = 10
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.model = EmbeddingModel(model_name=model_name)
        self.loaded_embeddings: OrderedDict[str, FAISS] = OrderedDict()
        self.max_cache_size = max_cache_size

        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Создана директория для индексов: {self.base_dir}")

    def load_embedding(self, vector_db_path: str) -> bool:
        faiss_file = Path(vector_db_path + ".faiss")
        pkl_file = Path(vector_db_path + ".pkl")

        index_uid = Path(vector_db_path).name

        if index_uid in self.loaded_embeddings:
            self.loaded_embeddings.move_to_end(index_uid)
            return True

        if not faiss_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(f"[FAISS] Файлы не найдены по пути: {vector_db_path}")

        try:
            db = FAISS.load_local(
                str(Path(vector_db_path).parent),
                self.model,
                index_name=index_uid,
                allow_dangerous_deserialization=True,
            )

            if len(self.loaded_embeddings) >= self.max_cache_size:
                self.loaded_embeddings.popitem(last=False)

            self.loaded_embeddings[index_uid] = db
            logger.info(f"[FAISS] Загружен индекс {index_uid}")
            logger.info(f"[DEBUG] Индекс {index_uid} загружен. Сейчас в памяти: {list(self.loaded_embeddings.keys())}")
            return True

        except Exception as e:
            logger.exception(f"[FAISS] Ошибка загрузки индекса {index_uid}: {e}")
            raise RuntimeError(f"[FAISS] Ошибка загрузки: {e}")

    def unload_embedding(self, vector_db_path: str) -> bool:
        index_uid = Path(vector_db_path).name
        return self.loaded_embeddings.pop(index_uid, None) is not None

    def get_loaded_embeddings(self) -> list[str]:
        return list(self.loaded_embeddings.keys())

    def search(self, vector_db_path: str, query: str, top_k: int = 5, min_score: float = 0.0) -> list[dict]:
        index_uid = Path(vector_db_path).name

        if index_uid not in self.loaded_embeddings:
            logger.warning(f"[FAISS] Индекс {index_uid} не загружен в память.")
            return []

        try:
            results = self.loaded_embeddings[index_uid].similarity_search_with_score(query, k=top_k)
            logger.info(f"[FAISS] Поиск в индексе {index_uid} по запросу '{query}', найдено: {len(results)}")

            filtered = [
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                }
                for doc, score in results
                if score >= min_score
            ]

            logger.info(f"[FAISS] После фильтрации по min_score={min_score}: {len(filtered)} результатов")
            return filtered

        except Exception as e:
            logger.exception(f"[FAISS] Ошибка при поиске в индексе {index_uid}: {e}")
            return []


embedding_manager = EmbeddingManager()
