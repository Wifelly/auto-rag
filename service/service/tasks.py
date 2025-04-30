import asyncio
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from service.monitoring.logger import logger
from service.service.document_pipeline import DocumentPipeline
from service.services.embedding_service import EmbeddingService


def train_task(
    user_docs_dir: str,
    output_dir: str | None,
    chunk_size: int,
    chunk_overlap: int,
    max_files: int | None,
    append: bool,
    db: AsyncSession,
):
    async def _run():
        try:
            logger.info("Запуск фоновой задачи обучения")

            user_dir_name = Path(user_docs_dir).name

            final_output_dir = output_dir or str(Path("./vectors") / user_dir_name)

            vector_path = str(Path(final_output_dir) / "index.faiss")

            embedding_service = EmbeddingService(db)
            embedding = await embedding_service.create_embedding(
                name=user_dir_name,
                files=[],
                status_id=1,
                vector_db_path=vector_path,
            )
            logger.info(f"Создан embedding #{embedding.id}")

            pipeline = DocumentPipeline()
            success = await pipeline.run_pipeline(
                user_docs_dir=user_docs_dir,
                output_dir=final_output_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_files=max_files,
                append=append,
                db=db,
                embedding_id=embedding.id,
            )

            new_status = 2 if success else 3
            await embedding_service.update_embedding_status(embedding.id, new_status)
            logger.info(f"Статус embedding #{embedding.id} обновлён на {new_status}")

        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")

    asyncio.run(_run())
