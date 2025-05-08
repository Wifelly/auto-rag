import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from service.database.models import Embedding
from service.database.repositories.embedding_repository import EmbeddingRepository
from service.monitoring.logger import logger


class EmbeddingService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repository = EmbeddingRepository(db)

    async def create_embedding(self, user_id: int, name: str, files: list[str], status_id: int) -> Embedding:
        embedding = await self.repository.create_embedding(user_id=user_id, name=name, files=files, status_id=status_id)

        uid = str(uuid.uuid4())
        vector_db_path = f"saved_indexes/{uid}"

        await self.repository.update_embedding_metadata(
            embedding_id=embedding.id,
            files=files,
            vector_db_path=vector_db_path,
            index_uid=uid,
        )

        embedding.index_uid = uid
        embedding.vector_db_path = vector_db_path

        logger.info(f"[EMBEDDING] Создан embedding_id={embedding.id}, uid={uid}")

        return embedding

    async def get_embedding_by_id(self, embedding_id: int) -> Embedding | None:
        return await self.repository.get_embedding_by_id(embedding_id)

    async def update_embedding_status(self, embedding_id: int, status_id: int) -> Embedding | None:
        return await self.repository.update_embedding_status(embedding_id, status_id)

    async def update_embedding_metadata(
        self, embedding_id: int, files: list[str], vector_db_path: str, index_uid: str
    ) -> Embedding | None:
        return await self.repository.update_embedding_metadata(embedding_id, files, vector_db_path, index_uid)

    async def get_all_embeddings(self, user_id: int, skip: int = 0, limit: int = 100) -> list[Embedding]:
        return await self.repository.get_all_embeddings(user_id=user_id, skip=skip, limit=limit)

    async def delete_embedding(self, embedding_id: int) -> bool:
        return await self.repository.delete_embedding(embedding_id)

    async def restore_embedding(self, embedding_id: int) -> bool:
        return await self.repository.restore_embedding(embedding_id)
