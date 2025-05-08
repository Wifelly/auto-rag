# sessions.py

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from service.settings import get_settings

# Базовый класс для моделей
Base = declarative_base()
settings = get_settings()

# Создание асинхронного движка и фабрики сессий
engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)


# Контекстный менеджер для получения сессии
@asynccontextmanager
async def get_async_session() -> AsyncSession:
    async with async_session() as session:
        yield session
