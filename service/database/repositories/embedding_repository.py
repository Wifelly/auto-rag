from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from ..models import Embedding, EmbeddingStatus


class EmbeddingRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_embedding(self, name: str, file: str, status_id: int) -> Embedding:
        db_embedding = Embedding(
            name=name,
            file=file,
            status_id=status_id
        )
        self.db.add(db_embedding)
        await self.db.commit()
        await self.db.refresh(db_embedding)
        return db_embedding

    async def get_embedding_by_id(self, embedding_id: int) -> Optional[Embedding]:
        query = select(Embedding).where(Embedding.id == embedding_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_all_embeddings(self, skip: int = 0, limit: int = 100) -> List[Embedding]:
        query = select(Embedding).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete_embedding(self, embedding_id: int) -> bool:
        query = select(Embedding).where(Embedding.id == embedding_id)
        result = await self.db.execute(query)
        embedding = result.scalar_one_or_none()

        if embedding:
            await self.db.delete(embedding)
            await self.db.commit()
            return True
        return False

    async def update_embedding_status(self, embedding_id: int, status_id: int) -> Optional[Embedding]:
        query = select(Embedding).where(Embedding.id == embedding_id)
        result = await self.db.execute(query)
        embedding = result.scalar_one_or_none()

        if embedding:
            embedding.status_id = status_id
            await self.db.commit()
            await self.db.refresh(embedding)
            return embedding
        return None

    async def get_embeddings_by_status(self, status_id: int, skip: int = 0, limit: int = 100) -> List[Embedding]:
        query = (
            select(Embedding)
            .where(Embedding.status_id == status_id)
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())