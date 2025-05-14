from collections import OrderedDict
from pathlib import Path

from langchain_community.vectorstores import FAISS

from service.config import Config
from service.models.embedding.embedding import DEFAULT_MODEL_NAME, EmbeddingModel
from service.monitoring.logger import logger


class EmbeddingManager:
    def __init__(
        self,
        base_dir: str = Config.INDEX_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        max_cache_size: int = 10,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.model = EmbeddingModel(model_name=model_name)
        self.loaded_embeddings: OrderedDict[str, FAISS] = OrderedDict()
        self.max_cache_size = max_cache_size

        logger.info(f"[EmbeddingManager] Using index directory: {self.base_dir}")

    def load_embedding(self, identifier: str) -> bool:
        uid = Path(identifier).name
        faiss_file = self.base_dir / f"{uid}.faiss"
        pkl_file = self.base_dir / f"{uid}.pkl"

        if uid in self.loaded_embeddings:
            self.loaded_embeddings.move_to_end(uid)
            return True

        if not faiss_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(f"[FAISS] Files not found: {faiss_file}, {pkl_file}")

        db = FAISS.load_local(
            str(self.base_dir),
            self.model,
            index_name=uid,
            allow_dangerous_deserialization=True,
        )
        if len(self.loaded_embeddings) >= self.max_cache_size:
            self.loaded_embeddings.popitem(last=False)
        self.loaded_embeddings[uid] = db

        logger.info(f"[EmbeddingManager] Loaded index: {uid}")
        return True

    def unload_embedding(self, identifier: str) -> bool:
        uid = Path(identifier).name
        removed = self.loaded_embeddings.pop(uid, None)
        if removed:
            logger.info(f"[EmbeddingManager] Unloaded index: {uid}")
            return True
        return False

    def get_loaded_embeddings(self) -> list[str]:
        return list(self.loaded_embeddings.keys())

    def search_with_meta(
        self,
        identifier: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        uid = Path(identifier).name
        if uid not in self.loaded_embeddings:
            logger.warning(f"[EmbeddingManager] Index {uid} is not loaded.")
            return []

        hits = self.loaded_embeddings[uid].similarity_search_with_score(query, k=top_k)
        results = []
        for doc, score in hits:
            if score < min_score:
                continue
            md = doc.metadata or {}
            results.append(
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "source_file": md.get("source_file") or md.get("filename") or "неизвестно",
                }
            )
        logger.info(f"[EmbeddingManager] search_with_meta in {uid}: {len(results)} hits")
        return results


embedding_manager = EmbeddingManager()
