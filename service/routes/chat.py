from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from service.clients.rag_service import Mode, rag_service
from service.database.config import get_db
from service.database.models import ChatRole
from service.services.chat_file_pipeline import ChatFileHandler
from service.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


class CreateChatRequest(BaseModel):
    user_id: int
    embedding_ids: list[int] = Field(default_factory=list)
    title: str | None = None


class SendMessageRequest(BaseModel):
    role: ChatRole
    content: str


class ChatMessageResponse(BaseModel):
    role: ChatRole
    content: str
    created_at: str
    source: str | None = None
    source_files: list[str] | None = None


class ChatResponseRequest(BaseModel):
    query: str
    embedding_ids: list[int] = Field(default_factory=list)
    top_k: int = 5
    min_score: float = 0.0
    use_rag: bool = True
    use_web: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None


class ChatResponseOutput(BaseModel):
    response: str
    source: str
    files: list[str] | None = None


class UpdateChatTitleRequest(BaseModel):
    title: str


class ChatListResponse(BaseModel):
    chat_id: UUID
    title: str
    created_at: str


@router.post("/")
async def create_chat(req: CreateChatRequest, db: AsyncSession = Depends(get_db)):
    svc = ChatService(db)
    session = await svc.create_session(
        user_id=req.user_id,
        embedding_ids=req.embedding_ids,
        title=req.title,
    )
    return {"chat_id": session.id}


@router.post("/{chat_id}/send")
async def send_message(chat_id: UUID, req: SendMessageRequest, db: AsyncSession = Depends(get_db)):
    svc = ChatService(db)
    if not await svc.get_session(chat_id):
        raise HTTPException(404, "Чат не найден")
    msg = await svc.add_message(chat_id, req.role, req.content)
    return {"message_id": msg.id}


@router.post("/{chat_id}/upload-file")
async def upload_file(
    chat_id: UUID,
    file: UploadFile = File(...),
    query: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    svc = ChatService(db)
    if not await svc.get_session(chat_id):
        raise HTTPException(404, "Чат не найден")
    docs = await ChatFileHandler().handle_uploaded_file(file)
    if not docs:
        raise HTTPException(400, "Ошибка обработки файла")
    text = "\n\n".join(d.page_content for d in docs)
    if query and query.strip():
        text += f"\n\n{query.strip()}"
    await svc.add_message(
        chat_id,
        ChatRole.USER,
        text,
        source="uploaded_file",
        source_files=[file.filename],
    )
    return {"status": "success", "message": f"Файл {file.filename} загружен"}


@router.get("/{chat_id}/history", response_model=list[ChatMessageResponse])
async def get_history(chat_id: UUID, db: AsyncSession = Depends(get_db)):
    svc = ChatService(db)
    if not await svc.get_session(chat_id):
        raise HTTPException(404, "Чат не найден")
    msgs = await svc.get_messages(chat_id)
    return [
        ChatMessageResponse(
            role=m.role,
            content=m.content,
            created_at=m.created_at.isoformat(),
            source=m.source,
            source_files=m.source_files,
        )
        for m in msgs
    ]


@router.post("/{chat_id}/respond", response_model=ChatResponseOutput)
async def respond(chat_id: UUID, req: ChatResponseRequest, db: AsyncSession = Depends(get_db)):
    svc = ChatService(db)
    if not await svc.get_session(chat_id):
        raise HTTPException(404, "Чат не найден")
    if req.use_web:
        mode = Mode.WEB
    elif req.use_rag:
        mode = Mode.RAG
    else:
        mode = Mode.CHAT
    result = await rag_service.answer_query(
        query=req.query,
        embedding_ids=req.embedding_ids,
        db_session=db,
        mode=mode,
        top_k=req.top_k,
        min_score=req.min_score,
        use_web=req.use_web,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        stop=req.stop,
    )
    return ChatResponseOutput(
        response=result["response"],
        source=result["meta"]["source"],
        files=result["meta"].get("files"),
    )


@router.patch("/{chat_id}/title")
async def update_title(chat_id: UUID, req: UpdateChatTitleRequest, db: AsyncSession = Depends(get_db)):
    svc = ChatService(db)
    sess = await svc.get_session(chat_id)
    if not sess:
        raise HTTPException(404, "Чат не найден")
    sess.title = req.title.strip()
    await db.commit()
    return {"chat_id": chat_id, "new_title": sess.title}


@router.delete("/{chat_id}")
async def delete_chat(chat_id: UUID, db: AsyncSession = Depends(get_db)):
    svc = ChatService(db)
    if not await svc.get_session(chat_id):
        raise HTTPException(404, "Чат не найден")
    await svc.delete_chat(chat_id)
    return {"chat_id": chat_id, "status": "deleted"}


@router.get("/user/{user_id}", response_model=list[ChatListResponse])
async def get_user_chats(user_id: int, db: AsyncSession = Depends(get_db)):
    svc = ChatService(db)
    sessions = await svc.get_sessions_by_user(user_id)
    return [
        ChatListResponse(
            chat_id=s.id,
            title=s.title or "Без названия",
            created_at=s.created_at.isoformat(),
        )
        for s in sessions
    ]
