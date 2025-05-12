from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from service.database.config import get_db
from service.database.models import Embedding
from service.monitoring.logger import logger
from service.services.document_pipeline import DocumentPipeline
from service.services.embedding_container import embedding_manager
from service.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


class EmbeddingResponse(BaseModel):
    id: int
    name: str
    vector_db_path: str | None
    index_uid: str | None
    status_id: int
    is_loader: bool
    is_public: bool

    model_config = ConfigDict(from_attributes=True)


class SearchResponse(BaseModel):
    embedding_id: int
    name: str
    results: list[dict]


async def _train_embedding_from_bytes(
    files_data: list[tuple[str, bytes]],
    db: AsyncSession,
    embedding_id: int,
    chunk_size: int,
    chunk_overlap: int,
    append: bool,
):
    try:
        pipeline = DocumentPipeline()
        ok = await pipeline.train_from_bytes(files_data, db, embedding_id, chunk_size, chunk_overlap, append=append)
        if not ok:
            logger.error(f"[BACKGROUND TRAIN] Эмбеддинг {embedding_id} не обучен (append={append})")
    except Exception as e:
        logger.exception(f"[BACKGROUND TRAIN] Ошибка фонового обучения эмбеддинга {embedding_id}: {e}")


@router.post("/", response_model=EmbeddingResponse)
async def create_embedding(
    background_tasks: BackgroundTasks,
    user_id: int = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(1024),
    chunk_overlap: int = Form(200),
    db: AsyncSession = Depends(get_db),
):
    # Считываем файлы в память
    files_data: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        files_data.append((upload.filename, content))

    svc = EmbeddingService(db)
    try:
        emb = await svc.create_embedding(
            user_id,
            name,
            [fn for fn, _ in files_data],
            status_id=1,
        )
    except Exception as e:
        logger.exception(f"[CREATE EMBEDDING] {e}")
        raise HTTPException(500, f"Ошибка создания записи: {e}") from e

    background_tasks.add_task(
        _train_embedding_from_bytes,
        files_data,
        db,
        emb.id,
        chunk_size,
        chunk_overlap,
        False,
    )

    return emb


@router.post("/{embedding_id}/append", response_model=EmbeddingResponse)
async def append_to_embedding(
    background_tasks: BackgroundTasks,
    embedding_id: int,
    user_id: int = Form(...),
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(1024),
    chunk_overlap: int = Form(200),
    db: AsyncSession = Depends(get_db),
):
    svc = EmbeddingService(db)
    emb = await svc.get_embedding_by_id(embedding_id)
    if not emb or emb.user_id != user_id:
        raise HTTPException(404, "Embedding не найден или доступ запрещён")
    if emb.status_id != 3:
        raise HTTPException(400, "Индекс ещё не готов")

    if not embedding_manager.base_dir.joinpath(f"{emb.index_uid}.faiss").exists():
        raise HTTPException(404, "Файл индекса не найден")

    # Считываем файлы в память
    files_data: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        files_data.append((upload.filename, content))

    background_tasks.add_task(
        _train_embedding_from_bytes,
        files_data,
        db,
        embedding_id,
        chunk_size,
        chunk_overlap,
        True,
    )

    return emb


@router.get("/", response_model=list[EmbeddingResponse])
async def list_embeddings(
    user_id: int = Query(..., ge=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, gt=0),
    db: AsyncSession = Depends(get_db),
):
    return await EmbeddingService(db).get_all_embeddings(user_id=user_id, skip=skip, limit=limit)


@router.delete("/{embedding_id}")
async def delete_embedding(
    embedding_id: int,
    db: AsyncSession = Depends(get_db),
):
    deleted = await EmbeddingService(db).delete_embedding(embedding_id)
    if not deleted:
        raise HTTPException(404, "Не найдено")
    return {"message": "Deleted"}


@router.post("/restore/{embedding_id}")
async def restore_embedding(
    embedding_id: int,
    db: AsyncSession = Depends(get_db),
):
    restored = await EmbeddingService(db).restore_embedding(embedding_id)
    if not restored:
        raise HTTPException(404, "Не найдено или уже активно")
    return {"message": "Restored"}


@router.put("/{embedding_id}/status/{status_id}", response_model=EmbeddingResponse)
async def update_status(
    embedding_id: int,
    status_id: int,
    db: AsyncSession = Depends(get_db),
):
    emb = await EmbeddingService(db).update_embedding_status(embedding_id, status_id)
    if not emb:
        raise HTTPException(404, "Не найдено")
    return emb


@router.post("/{embedding_id}/load", response_model=EmbeddingResponse)
async def load_embedding(
    embedding_id: int,
    is_loader: bool = Form(False),
    db: AsyncSession = Depends(get_db),
):
    emb = await EmbeddingService(db).get_embedding_by_id(embedding_id)
    if not emb or not emb.index_uid:
        raise HTTPException(404, "Не найдено")
    try:
        embedding_manager.load_embedding(emb.index_uid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e

    emb.is_loader = is_loader
    await db.commit()
    await db.refresh(emb)
    return emb


@router.post("/{embedding_id}/unload", response_model=EmbeddingResponse)
async def unload_embedding(
    embedding_id: int,
    is_loader: bool = Form(False),
    db: AsyncSession = Depends(get_db),
):
    emb = await EmbeddingService(db).get_embedding_by_id(embedding_id)
    if not emb or not emb.index_uid:
        raise HTTPException(404, "Не найдено")
    try:
        embedding_manager.unload_embedding(emb.index_uid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e

    emb.is_loader = is_loader
    await db.commit()
    await db.refresh(emb)
    return emb


@router.get(
    "/by-name/{name}",
    response_model=EmbeddingResponse,
    summary="Получить эмбеддинг по его уникальному имени",
)
async def get_embedding_by_name(
    name: str = Path(..., description="Уникальное имя эмбеддинга"),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Embedding).where(Embedding.name == name, Embedding.is_deleted == False).limit(1)
    result = await db.execute(stmt)
    emb = result.scalar_one_or_none()
    if not emb:
        raise HTTPException(404, f"Эмбеддинг с именем «{name}» не найден")
    return emb


@router.post("/{embedding_id}/set-public", response_model=EmbeddingResponse)
async def set_public(
    embedding_id: int,
    is_public: bool = Form(...),
    db: AsyncSession = Depends(get_db),
):
    svc = EmbeddingService(db)
    emb = await svc.get_embedding_by_id(embedding_id)
    if not emb:
        raise HTTPException(404, "Не найдено")
    emb.is_public = is_public
    await db.commit()
    await db.refresh(emb)
    return emb
