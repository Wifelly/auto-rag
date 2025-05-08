import enum
import uuid
from datetime import datetime

from sqlalchemy import ARRAY, Boolean, Column, Enum as PgEnum, ForeignKey, String, Table
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .config import Base

chat_embedding_association = Table(
    "chat_embedding_association",
    Base.metadata,
    Column("chat_id", ForeignKey("chat_sessions.id", ondelete="CASCADE"), primary_key=True),
    Column("embedding_id", ForeignKey("embeddings.id", ondelete="CASCADE"), primary_key=True),
)


class EmbeddingStatus(Base):
    __tablename__ = "embedding_status"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)

    embeddings: Mapped[list["Embedding"]] = relationship(back_populates="status_rel", cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped[int] = mapped_column(nullable=False)
    files: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    status_id: Mapped[int] = mapped_column(ForeignKey("embedding_status.id", ondelete="SET NULL"))
    vector_db_path: Mapped[str | None] = mapped_column(nullable=True)
    index_uid: Mapped[str | None] = mapped_column(nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(onupdate=func.now())

    status_rel: Mapped["EmbeddingStatus"] = relationship(back_populates="embeddings")
    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        secondary=chat_embedding_association, back_populates="embeddings"
    )


class ChatRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(nullable=False)
    title: Mapped[str | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    embeddings: Mapped[list["Embedding"]] = relationship(
        secondary=chat_embedding_association, back_populates="chat_sessions"
    )
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[ChatRole] = mapped_column(PgEnum(ChatRole, name="chat_role"), nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    source: Mapped[str | None] = mapped_column(nullable=True)
    source_files: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
