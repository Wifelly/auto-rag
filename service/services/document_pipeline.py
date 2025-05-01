# --- document_pipeline.py ---

from pathlib import Path

from fastapi import UploadFile
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sqlalchemy.ext.asyncio import AsyncSession

from service.models.embedding.embedding import DEFAULT_MODEL_NAME, EmbeddingModel
from service.monitoring.logger import logger
from service.services.embedding_service import EmbeddingService
from service.services.utils import MAX_TEXT_SIZE, Utils

MAX_ALLOWED_CHUNKS = 50000


class DocumentPipeline:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.embeddings = None
        self.model_name = model_name
        self.utils = Utils()

    async def _load_embeddings(self) -> None:
        if self.embeddings is None:
            logger.info(f"Lazy loading embedding model: {self.model_name}")
            self.embeddings = EmbeddingModel(model_name=self.model_name)

    async def train_from_uploaded_files(
        self,
        files: list[UploadFile],
        output_dir: str,
        db: AsyncSession,
        embedding_id: int,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
    ) -> bool:
        await self._load_embeddings()
        embedding_service = EmbeddingService(db)

        await embedding_service.update_embedding_status(embedding_id, 2)

        documents: list[Document] = []
        filenames: list[str] = []
        total_text_size = 0

        for file in files:
            content = await file.read()
            try:
                text = await self.utils.extract_text_from_bytes(file.filename, content)
                if not text.strip():
                    continue

                text = self.utils.clean_text(text)
                total_text_size += len(text.encode("utf-8"))
                if total_text_size > MAX_TEXT_SIZE:
                    logger.warning(f"Суммарный размер текстов превышает {MAX_TEXT_SIZE} байт.")
                    await embedding_service.update_embedding_status(embedding_id, 4)
                    return False

                filenames.append(file.filename)

                chunks = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", " "],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ).split_text(text)

                if len(chunks) > MAX_ALLOWED_CHUNKS:
                    logger.warning(f"Файл {file.filename} дал слишком много чанков: {len(chunks)}")
                    continue

                for i, chunk in enumerate(chunks):
                    documents.append(Document(page_content=chunk, metadata={"chunk_id": i, "source": file.filename}))

            except Exception as e:
                logger.error(f"Ошибка обработки файла {file.filename}: {e}")

        if not documents:
            await embedding_service.update_embedding_status(embedding_id, 4)
            return False

        try:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)

            db = FAISS.from_documents(documents, self.embeddings)
            db.save_local(str(path))
        except Exception as e:
            logger.error(f"Ошибка при создании FAISS индекса: {e}")
            await embedding_service.update_embedding_status(embedding_id, 4)
            return False

        await embedding_service.update_embedding_metadata(embedding_id, filenames, f"{output_dir}/index.faiss")
        await embedding_service.update_embedding_status(embedding_id, 3)
        return True
