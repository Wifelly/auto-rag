from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from service.database.config import get_db
from service.monitoring.logger import logger
from service.services.document_pipeline import DocumentPipeline
from service.services.embedding_container import embedding_manager
from service.services.embedding_service import EmbeddingService
from service.services.tasks import train_task

router = APIRouter(tags=["Embeddings"])


class EmbeddingResponse(BaseModel):
    id: int
    name: str
    vector_db_path: str
    status_id: int
    model_config = ConfigDict(from_attributes=True)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.0


@router.post("/embeddings/", response_model=EmbeddingResponse)
async def create_embedding(
    user_id: int = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(1024),
    chunk_overlap: int = Form(200),
    db: AsyncSession = Depends(get_db),
):
    filenames = [file.filename for file in files]
    embedding_service = EmbeddingService(db)

    try:
        embedding = await embedding_service.create_embedding(
            user_id=user_id,
            name=name,
            files=filenames,
            status_id=1,
        )

        pipeline = DocumentPipeline()
        success = await pipeline.train_from_uploaded_files(
            files=files,
            embedding_id=embedding.id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            db=db,
            append=False,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Обработка и обучение не выполнены.")

        return embedding

    except Exception as e:
        logger.exception(f"[CREATE EMBEDDING] Ошибка при создании: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при создании эмбеддинга: {e!s}")


@router.post("/embeddings/{embedding_id}/append", response_model=EmbeddingResponse)
async def append_to_embedding(
    embedding_id: int,
    user_id: int = Form(...),
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(1024),
    chunk_overlap: int = Form(200),
    db: AsyncSession = Depends(get_db),
):
    filenames = [file.filename for file in files]
    embedding_service = EmbeddingService(db)

    embedding = await embedding_service.get_embedding_by_id(embedding_id)
    if not embedding:
        raise HTTPException(status_code=404, detail="Embedding не найден")

    if embedding.user_id != user_id:
        raise HTTPException(status_code=403, detail="Доступ запрещён: чужой embedding")

    if embedding.status_id != 3:
        raise HTTPException(status_code=400, detail="Нельзя дозаписать: индекс не готов (status_id != 3)")

    faiss_file = Path(embedding.vector_db_path + ".faiss")
    if not faiss_file.exists():
        raise HTTPException(status_code=400, detail=f"Файлы индекса не найдены по пути {faiss_file}")

    try:
        logger.info(f"[APPEND] embedding_id={embedding_id}, user_id={user_id}, файлы: {filenames}")
        pipeline = DocumentPipeline()
        success = await pipeline.train_from_uploaded_files(
            files=files,
            embedding_id=embedding_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            db=db,
            append=True,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Дозапись в индекс не удалась")

        return embedding

    except Exception as e:
        logger.exception(f"[APPEND] Ошибка при дозаписи в embedding_id={embedding_id}")
        raise HTTPException(status_code=500, detail=f"Ошибка при дозаписи: {e}")


@router.get("/embeddings/{embedding_id}", response_model=EmbeddingResponse)
async def get_embedding(embedding_id: int, db: AsyncSession = Depends(get_db)):
    embedding_service = EmbeddingService(db)
    embedding = await embedding_service.get_embedding_by_id(embedding_id)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return embedding


@router.get("/embeddings/", response_model=list[EmbeddingResponse])
async def list_embeddings(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    embedding_service = EmbeddingService(db)
    return await embedding_service.get_all_embeddings(skip=skip, limit=limit)


@router.delete("/embeddings/{embedding_id}")
async def delete_embedding(embedding_id: int, db: AsyncSession = Depends(get_db)):
    embedding_service = EmbeddingService(db)
    success = await embedding_service.delete_embedding(embedding_id)
    if not success:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return {"message": "Embedding deleted successfully"}


@router.post("/restore-embedding/{embedding_id}")
async def restore_embedding(embedding_id: int, db: AsyncSession = Depends(get_db)):
    service = EmbeddingService(db)
    success = await service.restore_embedding(embedding_id)
    if not success:
        raise HTTPException(status_code=404, detail="Embedding not found or already active")
    return {"success": True, "message": f"Embedding {embedding_id} восстановлен"}


@router.put("/embeddings/{embedding_id}/status/{status_id}", response_model=EmbeddingResponse)
async def update_embedding_status(embedding_id: int, status_id: int, db: AsyncSession = Depends(get_db)):
    embedding_service = EmbeddingService(db)
    embedding = await embedding_service.update_embedding_status(embedding_id, status_id)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return embedding


@router.post("/embeddings/train")
async def train_embedding(
    background_tasks: BackgroundTasks,
    user_docs_dir: str,
    output_dir: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    max_files: int | None = None,
    append: bool = False,
    embedding_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    background_tasks.add_task(
        train_task,
        user_docs_dir=user_docs_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_files=max_files,
        append=append,
        embedding_id=embedding_id,
        db=db,
    )
    return {"message": "Процесс обучения запущен в фоне."}


@router.post("/embeddings/{embedding_id}/load")
async def load_embedding_to_memory(embedding_id: int, db: AsyncSession = Depends(get_db)):
    service = EmbeddingService(db)
    embedding = await service.get_embedding_by_id(embedding_id)

    if embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    if embedding.status_id != 3:
        raise HTTPException(status_code=400, detail="Embedding is not yet trained")
    if not embedding.vector_db_path:
        raise HTTPException(status_code=400, detail="vector_db_path не задан")

    try:
        logger.info(f"[LOAD] Загрузка FAISS: path={embedding.vector_db_path}")
        embedding_manager.load_embedding(embedding.vector_db_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Файлы индекса не найдены: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки FAISS индекса: {e}")

    return {"message": f"Embedding {embedding_id} успешно загружен в память."}


@router.post("/embeddings/{embedding_id}/unload")
async def unload_embedding_from_memory(embedding_id: int, db: AsyncSession = Depends(get_db)):
    service = EmbeddingService(db)
    embedding = await service.get_embedding_by_id(embedding_id)
    if not embedding:
        raise HTTPException(status_code=404, detail="Embedding not found")

    embedding_manager.unload_embedding(embedding.vector_db_path)
    return {"message": f"Embedding {embedding_id} успешно выгружен из памяти."}


@router.post("/embeddings/{embedding_id}/search")
async def search_embedding(embedding_id: int, request: SearchRequest, db: AsyncSession = Depends(get_db)):
    service = EmbeddingService(db)
    embedding = await service.get_embedding_by_id(embedding_id)
    if not embedding:
        raise HTTPException(status_code=404, detail="Embedding not found")

    index_uid = Path(embedding.vector_db_path).name
    if index_uid not in embedding_manager.get_loaded_embeddings():
        raise HTTPException(status_code=400, detail="Embedding не загружен в память")

    results = embedding_manager.search(
        vector_db_path=embedding.vector_db_path,
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
    )

    if results is None:
        raise HTTPException(status_code=500, detail="Ошибка во время поиска")

    return results


@router.post("/embeddings/{embedding_id}/set-public")
async def set_embedding_public(embedding_id: int, is_public: bool, db: AsyncSession = Depends(get_db)):
    service = EmbeddingService(db)
    embedding = await service.get_embedding_by_id(embedding_id)
    if not embedding:
        raise HTTPException(status_code=404, detail="Embedding не найден")

    embedding.is_public = is_public
    await db.commit()
    return {"message": f"Embedding {embedding_id} публичность изменена на {is_public}"}
