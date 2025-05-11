from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from service.database.models import ChatMessage, ChatRole, ChatSession, Embedding
from service.monitoring.logger import logger
from service.services.chat_file_pipeline import ChatFileHandler

MAX_MESSAGES_PER_CHAT = 500


class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.file_handler = ChatFileHandler()

    async def create_session(
        self, user_id: int, embedding_ids: list[int] | None = None, title: str | None = None
    ) -> ChatSession:
        session = ChatSession(user_id=user_id, title=title)
        if embedding_ids:
            embeddings = await self.db.execute(
                select(Embedding).where(
                    Embedding.id.in_(embedding_ids),
                    Embedding.user_id == user_id,
                    Embedding.is_deleted.is_(False),
                )
            )
            session.embeddings = embeddings.scalars().all()

        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info(f"[CHAT] Создана сессия {session.id} для user_id={user_id}")
        return session

    async def add_message(
        self,
        chat_id: UUID,
        role: ChatRole,
        content: str,
        source: str | None = None,
        source_files: list[str] | None = None,
    ) -> ChatMessage:
        existing = await self.db.scalar(select(func.count(ChatMessage.id)).where(ChatMessage.chat_id == chat_id))

        if role == ChatRole.USER and existing == 0:
            session = await self.get_session(chat_id)
            if session and not session.title:
                trimmed = content.strip()
                session.title = (trimmed[:100] + "…") if len(trimmed) > 100 else trimmed

        msg = ChatMessage(
            chat_id=chat_id,
            role=role,
            content=content,
            source=source,
            source_files=source_files,
        )
        self.db.add(msg)
        await self.db.commit()
        await self.db.refresh(msg)

        total = await self.db.scalar(select(func.count(ChatMessage.id)).where(ChatMessage.chat_id == chat_id))
        if total > MAX_MESSAGES_PER_CHAT:
            to_delete = total - MAX_MESSAGES_PER_CHAT
            subq = (
                select(ChatMessage.id)
                .where(ChatMessage.chat_id == chat_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(to_delete)
            )
            await self.db.execute(delete(ChatMessage).where(ChatMessage.id.in_(subq)))
            await self.db.commit()
            logger.info(f"[CHAT] Удалено {to_delete} старых сообщений из чата {chat_id}")

        return msg

    async def get_messages(self, chat_id: UUID) -> list[ChatMessage]:
        result = await self.db.execute(
            select(ChatMessage).where(ChatMessage.chat_id == chat_id).order_by(ChatMessage.created_at)
        )
        return result.scalars().all()

    async def get_session(self, chat_id: UUID) -> ChatSession | None:
        result = await self.db.execute(
            select(ChatSession).options(selectinload(ChatSession.embeddings)).where(ChatSession.id == chat_id)
        )
        return result.scalar_one_or_none()

    async def delete_chat(self, chat_id: UUID) -> None:
        await self.db.execute(delete(ChatSession).where(ChatSession.id == chat_id))
        await self.db.commit()
        logger.info(f"[CHAT] Чат {chat_id} удалён")

    async def handle_uploaded_file(self, file, chunk_size: int = 500, chunk_overlap: int = 100):
        return await self.file_handler.handle_uploaded_file(file, chunk_size, chunk_overlap)

    async def get_sessions_by_user(self, user_id: int) -> list[ChatSession]:
        result = await self.db.execute(
            select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc())
        )
        return result.scalars().all()

    async def update_title(self, chat_id: UUID, new_title: str) -> ChatSession:
        stmt = (
            update(ChatSession)
            .where(ChatSession.id == chat_id)
            .values(title=new_title)
            .execution_options(synchronize_session="fetch")
        ).returning(ChatSession)
        result = await self.db.execute(stmt)
        updated = result.scalar_one()
        await self.db.commit()
        return updated
