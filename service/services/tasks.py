import asyncio
import multiprocessing

from service.database.init_db import async_session_maker
from service.monitoring.logger import logger
from service.services.document_pipeline import DocumentPipeline


async def _run_training(
    files_data: list[tuple[str, bytes]],
    embedding_id: int,
    chunk_size: int,
    chunk_overlap: int,
    append: bool,
):
    async with async_session_maker() as db:
        try:
            await DocumentPipeline().train_from_bytes(
                files_data=files_data,
                db=db,
                embedding_id=embedding_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                append=append,
                return_documents=False,
            )
        except Exception as e:
            logger.exception(f"[BACKGROUND TRAIN] Ошибка обучения эмбеддинга {embedding_id}: {e}")


def train_embeddings_process(
    files_data: list[tuple[str, bytes]],
    embedding_id: int,
    chunk_size: int,
    chunk_overlap: int,
    append: bool,
):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            _run_training(
                files_data,
                embedding_id,
                chunk_size,
                chunk_overlap,
                append,
            )
        )
    finally:
        loop.close()


def schedule_train_in_subprocess(
    files_data: list[tuple[str, bytes]],
    embedding_id: int,
    chunk_size: int,
    chunk_overlap: int,
    append: bool = False,
):
    p = multiprocessing.Process(
        target=train_embeddings_process,
        args=(files_data, embedding_id, chunk_size, chunk_overlap, append),
        daemon=True,
    )
    p.start()
