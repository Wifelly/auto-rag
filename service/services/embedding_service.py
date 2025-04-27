from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from service.database.repositories.embedding_repository import EmbeddingRepository
from service.database.models import Embedding

class EmbeddingService:
    def __init__(self, db: AsyncSession):
        self.repository = EmbeddingRepository(db)

    async def create_embedding(self, name: str, file: str, status_id: int) -> Embedding:
        return await self.repository.create_embedding(name, file, status_id)

    async def get_embedding_by_id(self, embedding_id: int) -> Optional[Embedding]:
        return await self.repository.get_embedding_by_id(embedding_id)

    async def get_all_embeddings(self, skip: int = 0, limit: int = 100) -> List[Embedding]:
        return await self.repository.get_all_embeddings(skip, limit)

    async def delete_embedding(self, embedding_id: int) -> bool:
        return await self.repository.delete_embedding(embedding_id)

    async def update_embedding_status(self, embedding_id: int, status_id: int) -> Optional[Embedding]:
        return await self.repository.update_embedding_status(embedding_id, status_id)

    async def get_embeddings_by_status(self, status_id: int, skip: int = 0, limit: int = 100) -> List[Embedding]:
        return await self.repository.get_embeddings_by_status(status_id, skip, limit)