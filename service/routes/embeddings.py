# --- embeddings.py ---

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from service.database.config import get_db
from service.services.document_pipeline import DocumentPipeline
from service.services.embedding_manager import EmbeddingManager
from service.services.embedding_service import EmbeddingService
from service.services.tasks import train_task

router = APIRouter()
embedding_manager = EmbeddingManager()


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
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    filenames = [file.filename for file in files]
    output_dir = "saved_indexes/temp"
    embedding_service = EmbeddingService(db)
    return await embedding_service.create_embedding(
        name=name, files=filenames, status_id=1, vector_db_path=f"{output_dir}/index.faiss"
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


@router.post("/embeddings/train/upload", tags=["Embedding"])
async def train_from_upload(
    files: list[UploadFile] = File(...),
    name: str = Form(...),
    chunk_size: int = Form(1024),
    chunk_overlap: int = Form(200),
    db: AsyncSession = Depends(get_db),
):
    embedding_service = EmbeddingService(db)
    temp_embedding = await embedding_service.create_embedding(
        name=name,
        files=[f.filename for f in files],
        status_id=1,
        vector_db_path="temp/index.faiss",
    )

    output_dir = f"saved_indexes/{temp_embedding.id}"
    pipeline = DocumentPipeline()
    success = await pipeline.train_from_uploaded_files(
        files=files,
        embedding_id=temp_embedding.id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        db=db,
        output_dir=output_dir,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Обработка и обучение не выполнены.")
    return {"message": f"Embedding #{temp_embedding.id} успешно обучен из загруженных файлов."}


@router.post("/embeddings/{embedding_id}/load", tags=["Embedding"])
async def load_embedding_to_memory(embedding_id: int, db: AsyncSession = Depends(get_db)):
    service = EmbeddingService(db)
    embedding = await service.get_embedding_by_id(embedding_id)

    if embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")

    if embedding.status_id != 3:
        raise HTTPException(status_code=400, detail="Embedding is not yet trained (status_id != 3)")

    embedding_manager.load_embedding(str(embedding_id))
    return {"message": f"Embedding {embedding_id} успешно загружен в память."}


@router.post("/embeddings/{embedding_id}/unload", tags=["Embedding"])
async def unload_embedding_from_memory(embedding_id: int):
    embedding_manager.unload_embedding(str(embedding_id))
    return {"message": f"Embedding {embedding_id} успешно выгружен из памяти."}


@router.post("/embeddings/{embedding_id}/search", tags=["Embedding"])
async def search_embedding(embedding_id: int, request: SearchRequest):
    emb_id_str = str(embedding_id)
    if emb_id_str not in embedding_manager.get_loaded_embeddings():
        raise HTTPException(status_code=400, detail="Embedding не загружен в память")

    results = embedding_manager.search(
        emb_id=emb_id_str, query=request.query, top_k=request.top_k, min_score=request.min_score
    )

    if results is None:
        raise HTTPException(status_code=500, detail="Ошибка во время поиска")

    return results
