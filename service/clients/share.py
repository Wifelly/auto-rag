from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, constr
from sqlalchemy.ext.asyncio import AsyncSession

from service.clients.access_manager import access_manager
from service.database.config import get_db
from service.services.embedding_container import embedding_manager
from service.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/share", tags=["Share"])


class ShareRequest(BaseModel):
    owner_id: int
    collection_name: constr(r"^[A-Za-z0-9_]+$", min_length=1, max_length=50)
    target_user_id: int


@router.post("/")
async def share_collection(req: ShareRequest, db: AsyncSession = Depends(get_db)):
    svc = EmbeddingService(db)
    emb = await svc.find_by_name(req.owner_id, req.collection_name)
    if not emb:
        raise HTTPException(status_code=404, detail="Коллекция не найдена")

    if emb.index_uid not in embedding_manager.get_loaded_embeddings():
        raise HTTPException(status_code=400, detail="FAISS не загружен. Сначала выполните /load")

    try:
        await access_manager.require_access(req.owner_id, str(emb.id))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Нет доступа к коллекции")

    await access_manager.grant_access(str(req.owner_id), str(emb.id), req.target_user_id)
    return {"status": "shared", "collection_id": emb.id, "to": req.target_user_id}
