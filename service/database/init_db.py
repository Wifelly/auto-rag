from typing import AsyncIterator
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from service.settings import get_settings
from .config import Base
from .models import EmbeddingStatus

settings = get_settings()
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,
    future=True
)

async_session_maker = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with async_session_maker() as session:
        yield session

async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

async def init_db() -> None:
    try:
        await create_tables()
        
        async with async_session_maker() as session:
            initial_statuses = [
                EmbeddingStatus(name="PENDING"),
                EmbeddingStatus(name="PROCESSING"),
                EmbeddingStatus(name="COMPLETED"),
                EmbeddingStatus(name="FAILED")
            ]
            
            session.add_all(initial_statuses)
            await session.commit()
            
        print("Database initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

async def init_database():
    await init_db()

if __name__ == "__main__":
    init_database()