from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from service.database.config import get_db
from service.service.tasks import train_task
from service.services.embedding_service import EmbeddingService

router = APIRouter()


class EmbeddingCreate(BaseModel):
    name: str
    vector_db_path: str
    status_id: int


class EmbeddingResponse(BaseModel):
    id: int
    name: str
    vector_db_path: str
    status_id: int

    model_config = ConfigDict(from_attributes=True)


@router.post("/embeddings/", response_model=EmbeddingResponse)
async def create_embedding(
    name: str = Form(...),
    vector_db_path: str = Form(...),
    status_id: int = Form(...),
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    file_names = [file.filename for file in files]
    embedding_service = EmbeddingService(db)
    return await embedding_service.create_embedding(
        name=name, files=file_names, status_id=status_id, vector_db_path=vector_db_path
    )


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


@router.put("/embeddings/{embedding_id}/status/{status_id}", response_model=EmbeddingResponse)
async def update_embedding_status(embedding_id: int, status_id: int, db: AsyncSession = Depends(get_db)):
    embedding_service = EmbeddingService(db)
    embedding = await embedding_service.update_embedding_status(embedding_id, status_id)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return embedding


@router.post("/embeddings/train", tags=["Embedding"])
async def train_embedding(
    background_tasks: BackgroundTasks,
    user_docs_dir: str,
    output_dir: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    max_files: int | None = None,
    append: bool = False,
    db: AsyncSession = Depends(get_db),
):
    try:
        background_tasks.add_task(
            train_task,
            user_docs_dir=user_docs_dir,
            output_dir=output_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_files=max_files,
            append=append,
            db=db,
        )
        return {"message": "Процесс обучения запущен в фоне."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при запуске обучения: {e}")
