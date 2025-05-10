import re
from pathlib import Path

from fastapi import UploadFile
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from sqlalchemy.ext.asyncio import AsyncSession

from service.config import Config
from service.models.embedding.embedding import DEFAULT_MODEL_NAME, EmbeddingModel
from service.monitoring.logger import logger
from service.services.embedding_service import EmbeddingService
from service.services.utils import MAX_TEXT_SIZE, Utils

MAX_ALLOWED_CHUNKS = 50000


class HybridTextSplitter:
    def __init__(self, max_chunk_chars: int = 500, overlap_chars: int = 100):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def split(self, text: str) -> list[str]:
        sents = re.split(r"(?<=[\.\!\?])\s+", text)
        chunks: list[str] = []
        current = ""
        for sent in sents:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 <= self.max_chunk_chars:
                current = f"{current} {sent}".strip() if current else sent
            else:
                chunks.append(current)
                overlap = current[-self.overlap_chars :] if self.overlap_chars else ""
                current = f"{overlap} {sent}".strip()
        if current:
            chunks.append(current)
        return chunks


class DocumentPipeline:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.utils = Utils()
        self._embedder: EmbeddingModel | None = None
        Path(Config.INDEX_DIR).mkdir(parents=True, exist_ok=True)

    async def _load_embeddings(self) -> None:
        if self._embedder is None:
            logger.info(f"[DocumentPipeline] Loading embedding model: {self.model_name}")
            self._embedder = EmbeddingModel(model_name=self.model_name)

    async def train_from_uploaded_files(
        self,
        files: list[UploadFile],
        db: AsyncSession,
        embedding_id: int,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        append: bool = False,
    ) -> bool:
        await self._load_embeddings()
        svc = EmbeddingService(db)

        emb = await svc.get_embedding_by_id(embedding_id)
        if not emb or not emb.index_uid:
            logger.error(f"[DocumentPipeline] Embedding {embedding_id} not found or missing UID")
            return False

        uid = emb.index_uid
        faiss_path = Path(Config.INDEX_DIR) / f"{uid}.faiss"
        pkl_path = Path(Config.INDEX_DIR) / f"{uid}.pkl"

        # Mark as "processing"
        await svc.update_embedding_status(embedding_id, status_id=2)

        documents: list[Document] = []
        filenames: list[str] = []
        total_size = 0
        splitter = HybridTextSplitter(max_chunk_chars=chunk_size, overlap_chars=chunk_overlap)

        for file in files:
            try:
                raw = await file.read()
                text = await self.utils.extract_text_from_bytes(file.filename, raw)
                text = self.utils.clean_text(text)
                if not text:
                    continue

                total_size += len(text.encode("utf-8"))
                if total_size > MAX_TEXT_SIZE:
                    logger.warning(f"[DocumentPipeline] Total text size exceeded {MAX_TEXT_SIZE} bytes")
                    await svc.update_embedding_status(embedding_id, status_id=4)
                    return False

                filenames.append(file.filename)
                chunks = splitter.split(text)
                if len(chunks) > MAX_ALLOWED_CHUNKS:
                    logger.warning(f"[DocumentPipeline] {file.filename} produced {len(chunks)} chunks; skipping")
                    continue

                for idx, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"source_file": file.filename, "chunk_id": idx},
                        )
                    )
            except Exception as e:
                logger.error(f"[DocumentPipeline] Ошибка обработки {file.filename}: {e}")

        if not documents:
            await svc.update_embedding_status(embedding_id, status_id=4)
            return False

        try:
            if append and faiss_path.exists() and pkl_path.exists():
                index = FAISS.load_local(
                    Config.INDEX_DIR,
                    self._embedder,
                    index_name=uid,
                    allow_dangerous_deserialization=True,
                )
                index.add_documents(documents)
                index.save_local(Config.INDEX_DIR, index_name=uid)
            else:
                index = FAISS.from_documents(documents, self._embedder)
                index.save_local(Config.INDEX_DIR, index_name=uid)

            combined_files = list(set((emb.files or []) + filenames))
            base_index_path = Path(Config.INDEX_DIR) / uid

            await svc.update_embedding_metadata(
                embedding_id=embedding_id,
                files=combined_files,
                vector_db_path=str(base_index_path),
                index_uid=uid,
            )
            await svc.update_embedding_status(embedding_id, status_id=3)
            logger.info(f"[DocumentPipeline] Индекс сохранен: {faiss_path}")
            return True

        except Exception as e:
            logger.exception(f"[DocumentPipeline] Ошибка создания индекса: {e}")
            await svc.update_embedding_status(embedding_id, status_id=4)
            return False
