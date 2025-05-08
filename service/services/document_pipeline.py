from pathlib import Path

from fastapi import UploadFile
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from sqlalchemy.ext.asyncio import AsyncSession

from service.models.embedding.embedding import DEFAULT_MODEL_NAME, EmbeddingModel
from service.monitoring.logger import logger
from service.services.embedding_service import EmbeddingService
from service.services.utils import MAX_TEXT_SIZE, CustomMedicalTextSplitter, Utils

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
        db: AsyncSession,
        embedding_id: int,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        append: bool = False,
    ) -> bool:
        await self._load_embeddings()
        embedding_service = EmbeddingService(db)

        embedding = await embedding_service.get_embedding_by_id(embedding_id)
        if not embedding:
            logger.error(f"[FAISS] Embedding с id={embedding_id} не найден.")
            return False

        faiss_file = Path(embedding.vector_db_path + ".faiss")
        pkl_file = Path(embedding.vector_db_path + ".pkl")
        uid = Path(embedding.vector_db_path).name

        await embedding_service.update_embedding_status(embedding_id, 2)

        documents: list[Document] = []
        filenames: list[str] = []
        total_text_size = 0

        splitter = CustomMedicalTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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

                chunks = splitter.split(text)
                if len(chunks) > MAX_ALLOWED_CHUNKS:
                    logger.warning(f"Файл {file.filename} дал слишком много чанков: {len(chunks)}")
                    continue

                for i, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "chunk_id": i,
                                "source_file": file.filename,
                                "filename": file.filename,
                                "path": file.filename,
                                "source": file.filename,
                            },
                        )
                    )

            except Exception as e:
                logger.error(f"Ошибка обработки файла {file.filename}: {e}")

        if not documents:
            await embedding_service.update_embedding_status(embedding_id, 4)
            return False

        try:
            logger.info(f"[FAISS] Обработка embedding_id={embedding_id}, файлы: {filenames}")
            new_db = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"[FAISS] Новый индекс содержит {len(documents)} чанков.")

            if append and faiss_file.exists():
                logger.info(f"[FAISS] Режим: append. Загрузка существующего индекса: {faiss_file.name}")
                existing_db = FAISS.load_local(
                    "saved_indexes", self.embeddings, index_name=uid, allow_dangerous_deserialization=True
                )

                existing_count = len(existing_db.docstore._dict)
                logger.info(f"[FAISS] Существующий индекс до merge: {existing_count} чанков")

                existing_db.merge_from(new_db)

                after_merge_count = len(existing_db.docstore._dict)
                logger.info(f"[FAISS] После merge: {after_merge_count} чанков")

                existing_db.save_local("saved_indexes", index_name=uid)
            else:
                logger.info("[FAISS] Режим: overwrite. Создание нового индекса.")
                new_db.save_local("saved_indexes", index_name=uid)

            logger.info(f"[FAISS] Сохранён индекс: {faiss_file.name}, embedding_id={embedding_id}, path={faiss_file}")

        except Exception as e:
            logger.exception(f"[FAISS] Ошибка при создании индекса для embedding_id={embedding_id}: {e}")
            await embedding_service.update_embedding_status(embedding_id, 4)
            return False

        existing_files = embedding.files or []
        combined_files = list(set(existing_files + filenames))

        await embedding_service.update_embedding_metadata(
            embedding_id=embedding_id,
            files=combined_files,
            vector_db_path=embedding.vector_db_path,
            index_uid=uid,
        )
        await embedding_service.update_embedding_status(embedding_id, 3)
        return True
