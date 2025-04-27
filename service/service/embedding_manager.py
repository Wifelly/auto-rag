# --- embedding_manager.py ---

import os
from typing import Dict, Optional, List

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from embeddings import EmbeddingModel, DEFAULT_MODEL_NAME
from logger import logger

class EmbeddingManager:
    def __init__(self, base_dir: str = "saved_indexes", model_name: str = DEFAULT_MODEL_NAME):
        """
        base_dir: папка, где сохраняются FAISS индексы
        model_name: имя модели эмбеддингов
        """
        self.base_dir = base_dir
        self.model = EmbeddingModel(model_name=model_name)
        self.loaded_embeddings: Dict[str, FAISS] = {}  # id -> FAISS база

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            logger.info(f"Создана директория для индексов: {self.base_dir}")

    def load_embedding(self, emb_id: str) -> bool:
        """
        Загружает FAISS индекс в память по ID
        """
        index_path = os.path.join(self.base_dir, emb_id)
        faiss_file = os.path.join(index_path, "index.faiss")

        if not os.path.exists(faiss_file):
            logger.error(f"Индекс не найден: {faiss_file}")
            return False

        try:
            db = FAISS.load_local(index_path, self.model, allow_dangerous_deserialization=True)
            self.loaded_embeddings[emb_id] = db
            logger.info(f"Индекс {emb_id} успешно загружен в память.")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса {emb_id}: {e}")
            return False

    def unload_embedding(self, emb_id: str) -> bool:
        """
        Выгружает индекс из памяти
        """
        if emb_id in self.loaded_embeddings:
            del self.loaded_embeddings[emb_id]
            logger.info(f"Индекс {emb_id} выгружен из памяти.")
            return True
        else:
            logger.warning(f"Индекс {emb_id} не был загружен.")
            return False

    def get_loaded_embeddings(self) -> List[str]:
        """
        Возвращает список всех загруженных в память эмбеддингов
        """
        return list(self.loaded_embeddings.keys())

    def search(self, emb_id: str, query: str, top_k: int = 5) -> Optional[List[Document]]:
        """
        Выполняет поиск по загруженному индексу
        """
        if emb_id not in self.loaded_embeddings:
            logger.error(f"Индекс {emb_id} не загружен в память.")
            return None

        db = self.loaded_embeddings[emb_id]

        try:
            query_vector = self.model.embed_query(query)
            docs_and_scores = db.similarity_search_by_vector(query_vector, k=top_k)
            logger.info(f"Поиск в индексе {emb_id} выполнен успешно.")
            return docs_and_scores
        except Exception as e:
            logger.error(f"Ошибка при поиске в индексе {emb_id}: {e}")
            return None

    def list_saved_indexes(self) -> List[str]:
        """
        Возвращает список всех доступных на диске эмбеддингов
        """
        try:
            return [
                name for name in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, name))
            ]
        except Exception as e:
            logger.error(f"Ошибка при чтении директории индексов: {e}")
            return []
