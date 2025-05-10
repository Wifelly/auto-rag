from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from service.database.config import get_db
from service.monitoring.logger import logger
from service.services.document_pipeline import DocumentPipeline
from service.services.embedding_container import embedding_manager
from service.services.embedding_service import EmbeddingService
from service.services.tasks import train_task

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


class EmbeddingResponse(BaseModel):
    id: int
    name: str
    vector_db_path: str
    index_uid: str
    status_id: int

    model_config = ConfigDict(from_attributes=True)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.0


@router.post("/", response_model=EmbeddingResponse)
async def create_embedding(
    user_id: int = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(1024),
    chunk_overlap: int = Form(200),
    db: AsyncSession = Depends(get_db),
):
    svc = EmbeddingService(db)
    try:
        emb = await svc.create_embedding(user_id, name, [f.filename for f in files], status_id=1)
        pipeline = DocumentPipeline()
        ok = await pipeline.train_from_uploaded_files(files, db, emb.id, chunk_size, chunk_overlap, append=False)
        if not ok:
            raise HTTPException(500, "Не удалось обучить эмбеддинг")
        return emb
    except Exception as e:
        logger.exception(f"[CREATE EMBEDDING] {e}")
        raise HTTPException(500, f"Ошибка: {e}")


@router.post("/{embedding_id}/append", response_model=EmbeddingResponse)
async def append_to_embedding(
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

    uid = emb.index_uid
    base = embedding_manager.base_dir
    if not (base / f"{uid}.faiss").exists():
        raise HTTPException(404, "Файл индекса не найден")

    pipeline = DocumentPipeline()
    ok = await pipeline.train_from_uploaded_files(files, db, embedding_id, chunk_size, chunk_overlap, append=True)
    if not ok:
        raise HTTPException(500, "Не удалось дописать в эмбеддинг")
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
    if not await EmbeddingService(db).delete_embedding(embedding_id):
        raise HTTPException(404, "Не найдено")
    return {"message": "Deleted"}


@router.post("/restore/{embedding_id}")
async def restore_embedding(
    embedding_id: int,
    db: AsyncSession = Depends(get_db),
):
    if not await EmbeddingService(db).restore_embedding(embedding_id):
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


@router.post("/train")
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
        user_docs_dir,
        output_dir,
        chunk_size,
        chunk_overlap,
        max_files,
        append,
        embedding_id,
        db,
    )
    return {"message": "Запущено"}


@router.post("/{embedding_id}/load")
async def load_embedding(
    embedding_id: int,
    db: AsyncSession = Depends(get_db),
):
    emb = await EmbeddingService(db).get_embedding_by_id(embedding_id)
    if not emb or not emb.index_uid:
        raise HTTPException(404, "Не найдено")
    try:
        embedding_manager.load_embedding(emb.index_uid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"message": "Загружен"}


@router.post("/{embedding_id}/unload")
async def unload_embedding(
    embedding_id: int,
    db: AsyncSession = Depends(get_db),
):
    emb = await EmbeddingService(db).get_embedding_by_id(embedding_id)
    if not emb or not emb.index_uid:
        raise HTTPException(404, "Не найдено")
    embedding_manager.unload_embedding(emb.index_uid)
    return {"message": "Выгружен"}


@router.post("/{embedding_id}/search")
async def search_embedding(
    embedding_id: int,
    req: SearchRequest,
    db: AsyncSession = Depends(get_db),
):
    emb = await EmbeddingService(db).get_embedding_by_id(embedding_id)
    if not emb or not emb.index_uid:
        raise HTTPException(404, "Не найдено")
    return embedding_manager.search(emb.index_uid, req.query, top_k=req.top_k, min_score=req.min_score)


@router.post("/{embedding_id}/set-public")
async def set_public(
    embedding_id: int,
    is_public: bool,
    db: AsyncSession = Depends(get_db),
):
    svc = EmbeddingService(db)
    emb = await svc.get_embedding_by_id(embedding_id)
    if not emb:
        raise HTTPException(404, "Не найдено")
    emb.is_public = is_public
    await db.commit()
    return {"message": "OK"}
