from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    from .init_db import async_session_maker
    async with async_session_maker() as session:
        yield session
