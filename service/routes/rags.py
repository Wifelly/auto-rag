# --- rags.py ---

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from service.clients.rag_service import Mode, RagService
from service.database.config import get_db
from service.database.models import Embedding
from service.monitoring.logger import logger
from service.services.document_pipeline import DocumentPipeline
from service.services.embedding_container import embedding_manager
from service.services.embedding_service import EmbeddingService

router = APIRouter()
rag_service = RagService()


@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    embedding_id: int = Form(...),
    chunk_size: int = Form(1024),
    chunk_overlap: int = Form(200),
    db: AsyncSession = Depends(get_db),
):
    service = EmbeddingService(db)
    embedding = await service.get_embedding_by_id(embedding_id)

    if not embedding:
        raise HTTPException(status_code=404, detail="Embedding не найден")
    if embedding.user_id != user_id:
        raise HTTPException(status_code=403, detail="Нет доступа к embedding")

    index_key = Path(embedding.vector_db_path).name
    if index_key not in embedding_manager.get_loaded_embeddings():
        raise HTTPException(status_code=400, detail="FAISS не загружен в память. Сначала выполните /load")

    pipeline = DocumentPipeline()
    await pipeline._load_embeddings()

    content = await file.read()
    try:
        text = await pipeline.utils.extract_text_from_bytes(file.filename, content)
        text = pipeline.utils.clean_text(text)
    except Exception:
        raise HTTPException(status_code=400, detail="Ошибка при извлечении текста из файла")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata={"chunk_id": i, "source": file.filename})
        for i, chunk in enumerate(chunks)
    ]

    if not documents:
        raise HTTPException(status_code=400, detail="Документ не содержит полезного текста")

    db_instance = embedding_manager.loaded_embeddings[index_key]
    db_instance.add_documents(documents)

    return {"success": True, "message": f"Документ добавлен в память для embedding {embedding_id}"}


@router.get("/my-collections")
async def list_collections(user_id: int, db: AsyncSession = Depends(get_db)):
    service = EmbeddingService(db)
    embeddings = await service.get_all_embeddings(user_id=user_id)
    return {"collections": [e.id for e in embeddings]}


@router.get("/my-collections/full")
async def list_collections_full(user_id: int, db: AsyncSession = Depends(get_db)):
    service = EmbeddingService(db)
    embeddings = await service.get_all_embeddings(user_id=user_id)

    loaded_keys = set(embedding_manager.get_loaded_embeddings())

    result = []
    for e in embeddings:
        is_loaded = Path(e.vector_db_path).name in loaded_keys
        result.append(
            {
                "id": e.id,
                "name": e.name,
                "status": e.status_id,
                "created_at": e.created_at,
                "is_loaded": is_loaded,
                "files": e.files,
            }
        )

    return {"collections": result}


@router.delete("/delete-document")
async def delete_document(user_id: int, embedding_id: int, db: AsyncSession = Depends(get_db)):
    service = EmbeddingService(db)
    embedding = await service.get_embedding_by_id(embedding_id)

    if not embedding:
        raise HTTPException(status_code=404, detail="Embedding не найден")
    if embedding.user_id != user_id:
        raise HTTPException(status_code=403, detail="Нет доступа к embedding")

    key = Path(embedding.vector_db_path).name
    if key not in embedding_manager.get_loaded_embeddings():
        raise HTTPException(status_code=404, detail="Embedding не загружен в память")

    embedding_manager.unload_embedding(embedding.vector_db_path)
    return {"success": True, "message": f"Embedding {embedding_id} выгружен из памяти"}


class QueryRequest(BaseModel):
    user_id: int
    embedding_id: int
    query: str
    mode: Mode = Mode.AUTO
    use_web: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None


@router.post("/answer-query")
async def answer_query(payload: QueryRequest, db: AsyncSession = Depends(get_db)):
    embedding_service = EmbeddingService(db)
    embedding: Embedding | None = await embedding_service.get_embedding_by_id(payload.embedding_id)

    if not embedding:
        raise HTTPException(status_code=404, detail="Embedding не найден")
    if embedding.user_id != payload.user_id and not embedding.is_public:
        raise HTTPException(status_code=403, detail="Нет доступа к embedding (он приватный)")

    logger.info(f"[DEBUG] Проверка vector_db_path: {embedding.vector_db_path}")
    logger.info(f"[DEBUG] Загруженные индексы: {embedding_manager.get_loaded_embeddings()}")

    return await rag_service.answer_query(
        query=payload.query,
        embedding=embedding,
        db_session=db,
        mode=payload.mode,
        use_web=payload.use_web,
        use_local_llm=False,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        stop=payload.stop,
    )
