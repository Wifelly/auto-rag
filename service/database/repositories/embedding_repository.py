# --- embedding_repository.py ---

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Embedding


class EmbeddingRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_embedding(self, name: str, files: list[str], status_id: int, vector_db_path: str) -> Embedding:
        embedding = Embedding(
            name=name,
            files=files,
            status_id=status_id,
            vector_db_path=vector_db_path,
        )
        self.db.add(embedding)
        await self.db.commit()
        await self.db.refresh(embedding)
        return embedding

    async def get_embedding_by_id(self, embedding_id: int) -> Embedding | None:
        result = await self.db.execute(select(Embedding).where(Embedding.id == embedding_id))
        return result.scalar_one_or_none()

    async def get_all_embeddings(self, skip: int = 0, limit: int = 100) -> list[Embedding]:
        result = await self.db.execute(select(Embedding).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def delete_embedding(self, embedding_id: int) -> bool:
        result = await self.db.execute(select(Embedding).where(Embedding.id == embedding_id))
        embedding = result.scalar_one_or_none()
        if embedding:
            await self.db.delete(embedding)
            await self.db.commit()
            return True
        return False

    async def update_embedding_status(self, embedding_id: int, status_id: int) -> Embedding | None:
        embedding = await self.get_embedding_by_id(embedding_id)
        if embedding:
            embedding.status_id = status_id
            await self.db.commit()
            await self.db.refresh(embedding)
            return embedding
        return None

    async def update_embedding_metadata(
        self, embedding_id: int, files: list[str], vector_db_path: str
    ) -> Embedding | None:
        embedding = await self.get_embedding_by_id(embedding_id)
        if embedding:
            embedding.files = files
            embedding.vector_db_path = vector_db_path
            await self.db.commit()
            await self.db.refresh(embedding)
            return embedding
        return None

    async def get_embeddings_by_status(self, status_id: int, skip: int = 0, limit: int = 100) -> list[Embedding]:
        result = await self.db.execute(
            select(Embedding).where(Embedding.status_id == status_id).offset(skip).limit(limit)
        )
        return list(result.scalars().all())
