from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from service.database.config import get_db
from service.services.embedding_service import EmbeddingService

router = APIRouter()


class EmbeddingCreate(BaseModel):
    name: str
    status_id: int
    vector_db_path: str


class EmbeddingResponse(BaseModel):
    id: int
    name: str
    status_id: int

    class Config:
        orm_mode = True


@router.post("/embeddings/", response_model=EmbeddingResponse)
async def create_embedding(embedding: EmbeddingCreate, files: list[UploadFile], db: AsyncSession = Depends(get_db)):
    embedding_service = EmbeddingService(db)
    return await embedding_service.create_embedding(name=embedding.name, files=files, status_id=embedding.status_id)


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
