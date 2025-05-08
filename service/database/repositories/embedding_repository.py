from sqlalchemy import not_, select
from sqlalchemy.ext.asyncio import AsyncSession

from service.database.models import Embedding


class EmbeddingRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_embedding(
        self,
        user_id: int,
        name: str,
        files: list[str],
        status_id: int,
        vector_db_path: str = "",
    ) -> Embedding:
        embedding = Embedding(
            user_id=user_id,
            name=name,
            files=files,
            status_id=status_id,
            vector_db_path=vector_db_path,
            is_deleted=False,
        )
        self.db.add(embedding)
        await self.db.commit()
        await self.db.refresh(embedding)
        return embedding

    async def get_embedding_by_id(self, embedding_id: int) -> Embedding | None:
        result = await self.db.execute(
            select(Embedding).where(
                Embedding.id == embedding_id,
                not_(Embedding.is_deleted),
            )
        )
        return result.scalar_one_or_none()

    async def get_all_embeddings(self, user_id: int, skip: int = 0, limit: int = 100) -> list[Embedding]:
        result = await self.db.execute(
            select(Embedding)
            .where(
                Embedding.user_id == user_id,
                not_(Embedding.is_deleted),
            )
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_embedding(self, embedding_id: int) -> bool:
        embedding = await self.get_embedding_by_id(embedding_id)
        if embedding:
            embedding.is_deleted = True
            await self.db.commit()
            await self.db.refresh(embedding)
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
        self, embedding_id: int, files: list[str], vector_db_path: str, index_uid: str
    ) -> Embedding | None:
        embedding = await self.get_embedding_by_id(embedding_id)
        if embedding:
            embedding.files = files
            embedding.vector_db_path = vector_db_path
            embedding.index_uid = index_uid
            await self.db.commit()
            await self.db.refresh(embedding)
            return embedding
        return None

    async def find_by_name(self, user_id: int, name: str) -> Embedding | None:
        result = await self.db.execute(
            select(Embedding).where(
                Embedding.user_id == user_id,
                Embedding.name == name,
                not_(Embedding.is_deleted),
            )
        )
        return result.scalar_one_or_none()

    async def restore_embedding(self, embedding_id: int) -> bool:
        result = await self.db.execute(select(Embedding).where(Embedding.id == embedding_id))
        embedding = result.scalar_one_or_none()
        if embedding and embedding.is_deleted:
            embedding.is_deleted = False
            await self.db.commit()
            await self.db.refresh(embedding)
            return True
        return False
