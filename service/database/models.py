from datetime import datetime

from sqlalchemy import ARRAY, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .config import Base


class EmbeddingStatus(Base):
    __tablename__ = "embedding_status"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(nullable=False)
    # Relationship to Embedding
    embeddings: Mapped[list["Embedding"]] = relationship(back_populates="status_rel")

    def __repr__(self) -> str:
        return f"EmbeddingStatus(id={self.id}, name={self.name})"


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(nullable=False)
    files: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    status_id: Mapped[int] = mapped_column(ForeignKey("embedding_status.id"))
    vector_db_path: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(onupdate=func.now(), nullable=True)

    # Relationship to EmbeddingStatus
    status_rel: Mapped[EmbeddingStatus] = relationship(back_populates="embeddings")

    def __repr__(self) -> str:
        return f"Embedding(id={self.id}, name={self.name}, files={self.files}), vector_db_path={self.vector_db_path})"
