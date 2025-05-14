from pathlib import Path

from fastapi import APIRouter

from service.config import Config
from service.services.embedding_container import embedding_manager

router = APIRouter(prefix="/status", tags=["Status"])


@router.get("/llm")
async def get_llm_status():
    return {
        "llm_enabled": True,
        "remote_url": Config.GPT_URL,
        "model": Config.GPT_MODEL,
        "default_temperature": Config.DEFAULT_TEMPERATURE,
        "default_top_p": Config.DEFAULT_TOP_P,
        "default_max_tokens": Config.DEFAULT_MAX_TOKENS,
        "default_stop": Config.DEFAULT_STOP,
    }


@router.get("/embeddings")
async def get_embedding_status():
    base_dir = Path(embedding_manager.base_dir)
    loaded = embedding_manager.get_loaded_embeddings()
    info = {}
    for key in loaded:
        faiss_index = embedding_manager.loaded_embeddings[key]
        size = getattr(faiss_index.index, "ntotal", None)
        if size is None:
            size = len(getattr(faiss_index.docstore, "_dict", {}))
        path = base_dir / key
        info[key] = {
            "index_size": size,
            "embedding_path": str(path),
        }
    return {
        "loaded_embeddings": loaded,
        "count": len(loaded),
        "index_info": info,
    }
