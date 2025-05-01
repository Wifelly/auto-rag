# --- tasks.py ---

import asyncio
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from service.monitoring.logger import logger
from service.services.document_pipeline import DocumentPipeline
from service.services.embedding_service import EmbeddingService


def train_task(
    user_docs_dir: str,
    output_dir: str | None,
    chunk_size: int,
    chunk_overlap: int,
    max_files: int | None,
    append: bool,
    db: AsyncSession,
    embedding_id: int | None = None,
):
    async def _run():
        try:
            logger.info("Запуск фоновой задачи обучения")

            user_dir_name = Path(user_docs_dir).name
            embedding_service = EmbeddingService(db)

            if embedding_id is None:
                output_path = Path("saved_indexes") / user_dir_name
                vector_path = str(output_path / "index.faiss")
                embedding = await embedding_service.create_embedding(
                    name=user_dir_name,
                    files=[],
                    status_id=1,
                    vector_db_path=vector_path,
                )
                embedding_id = embedding.id
                logger.info(f"Создан новый embedding #{embedding_id}")
            else:
                logger.info(f"Обновление embedding #{embedding_id}")
                output_path = Path("saved_indexes") / str(embedding_id)

            pipeline = DocumentPipeline()
            success = await pipeline.run_pipeline(
                user_docs_dir=user_docs_dir,
                output_dir=str(output_path),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_files=max_files,
                append=append,
                db=db,
                embedding_id=embedding_id,
            )

            new_status = 3 if success else 4
            await embedding_service.update_embedding_status(embedding_id, new_status)
            logger.info(f"Статус embedding #{embedding_id} обновлён на {new_status}")

        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")

    asyncio.run(_run())
