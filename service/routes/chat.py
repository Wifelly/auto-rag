import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from service.clients.rag_service import Mode, rag_service
from service.database.config import get_db
from service.database.models import ChatRole
from service.services.chat_service import ChatService
from service.services.document_pipeline import DocumentPipeline
from service.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatListResponse(BaseModel):
    chat_id: UUID
    title: str | None
    created_at: str


class ChatMessageResponse(BaseModel):
    role: ChatRole
    content: str
    created_at: str
    source: str | None = None
    source_files: list[str] | None = None


class ChatResponseOutput(BaseModel):
    response: str
    source: str | None
    files: list[str] | None = None


class CreateChatRequest(BaseModel):
    user_id: int
    embedding_ids: list[int] = Field(default_factory=list)
    title: str | None = None


class SendMessageRequest(BaseModel):
    role: ChatRole
    content: str


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


class UpdateChatTitleRequest(BaseModel):
    title: str


@router.post("/", response_model=ChatListResponse)
async def create_chat(
    req: CreateChatRequest,
    db: AsyncSession = Depends(get_db),
):
    svc = ChatService(db)
    session = await svc.create_session(
        user_id=req.user_id,
        embedding_ids=req.embedding_ids,
        title=req.title,
    )
    return ChatListResponse(
        chat_id=session.id,
        title=session.title,
        created_at=session.created_at.isoformat(),
    )


@router.get("/user/{user_id}", response_model=list[ChatListResponse])
async def get_user_chats(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    svc = ChatService(db)
    sessions = await svc.get_sessions_by_user(user_id)
    return [
        ChatListResponse(
            chat_id=s.id,
            title=s.title,
            created_at=s.created_at.isoformat(),
        )
        for s in sessions
    ]


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    svc = ChatService(db)
    if not await svc.get_session(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    await svc.delete_chat(chat_id)


@router.post("/{chat_id}/send", status_code=status.HTTP_204_NO_CONTENT)
async def send_message(
    chat_id: UUID,
    req: SendMessageRequest,
    db: AsyncSession = Depends(get_db),
):
    svc = ChatService(db)
    if not await svc.get_session(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    await svc.add_message(chat_id, req.role, req.content)


@router.patch("/{chat_id}", response_model=ChatListResponse)
async def update_chat_title(
    chat_id: UUID,
    req: UpdateChatTitleRequest,
    db: AsyncSession = Depends(get_db),
):
    svc = ChatService(db)
    session = await svc.get_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")
    session = await svc.update_title(chat_id, req.title)
    return ChatListResponse(
        chat_id=session.id,
        title=session.title,
        created_at=session.created_at.isoformat(),
    )


@router.get("/{chat_id}/history", response_model=list[ChatMessageResponse])
async def get_history(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    svc = ChatService(db)
    if not await svc.get_session(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
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
async def respond(
    chat_id: UUID,
    req: ChatResponseRequest,
    db: AsyncSession = Depends(get_db),
):
    svc = ChatService(db)
    if not await svc.get_session(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    mode = Mode.WEB if req.use_web else Mode.RAG if req.use_rag else Mode.CHAT
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
    await svc.add_message(
        chat_id,
        ChatRole.ASSISTANT,
        result["response"],
        source=result["meta"]["source"],
        source_files=result["meta"].get("files"),
    )
    return ChatResponseOutput(
        response=result["response"],
        source=result["meta"]["source"],
        files=result["meta"].get("files"),
    )


@router.get("/{chat_id}/respond-stream")
async def respond_stream(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    query: str = Query(..., description="Текст запроса"),
    embedding_ids: list[int] = Query(
        default=[],
        description="Список ID эмбеддингов, разделённых запятыми",
    ),
    top_k: int = Query(5, ge=1),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    use_rag: bool = Query(True),
    use_web: bool = Query(False),
    temperature: float | None = Query(None, ge=0.0, le=2.0),
    top_p: float | None = Query(None, ge=0.0, le=1.0),
    max_tokens: int | None = Query(None, ge=1),
    stop: list[str] = Query(
        default=[],
        description="Список токенов для остановки генерации",
    ),
):
    svc = ChatService(db)
    if not await svc.get_session(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")

    mode = Mode.WEB if use_web else Mode.RAG if use_rag else Mode.CHAT
    buffer: list[str] = []

    async def event_generator():
        async for chunk in rag_service.stream_answer_query(
            query=query,
            embedding_ids=embedding_ids,
            db_session=db,
            mode=mode,
            top_k=top_k,
            min_score=min_score,
            use_web=use_web,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        ):
            buffer.append(chunk)
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.01)

        full_response = "".join(buffer)
        await svc.add_message(
            chat_id,
            ChatRole.ASSISTANT,
            full_response,
            source=None,
            source_files=None,
        )

        yield "event: done\ndata: complete\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.post("/{chat_id}/upload-file", response_model=list[ChatMessageResponse])
async def upload_files(
    chat_id: UUID,
    files: list[UploadFile] = File(...),
    query: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    svc_chat = ChatService(db)
    session = await svc_chat.get_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")

    emb_svc = EmbeddingService(db)
    emb = await emb_svc.create_embedding(
        user_id=session.user_id,
        name=f"chat_{chat_id}_shared_index",
        files=[f.filename for f in files],
        status_id=1,
    )

    files_data: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        files_data.append((upload.filename, content))

    pipeline = DocumentPipeline()
    chunks = await pipeline.train_from_bytes(
        files_data=files_data,
        db=db,
        embedding_id=emb.id,
        chunk_size=500,
        chunk_overlap=100,
        append=False,
        return_documents=True,
    )
    if not chunks:
        raise HTTPException(status_code=500, detail="Failed to index upload files")

    embedding_ids = [emb.id]
    created_msgs: list[ChatMessageResponse] = []

    for doc in chunks:
        msg = await svc_chat.add_message(
            chat_id,
            ChatRole.SYSTEM,
            doc.page_content,
            source="file_chunk",
            source_files=[doc.metadata.get("source_file")],
        )
        created_msgs.append(
            ChatMessageResponse(
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at.isoformat(),
                source=msg.source,
                source_files=msg.source_files,
            )
        )

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    user_msg = await svc_chat.add_message(
        chat_id,
        ChatRole.USER,
        query.strip(),
        source="uploaded_files",
        source_files=[f.filename for f in files],
    )
    created_msgs.append(
        ChatMessageResponse(
            role=user_msg.role,
            content=user_msg.content,
            created_at=user_msg.created_at.isoformat(),
            source=user_msg.source,
            source_files=user_msg.source_files,
        )
    )

    resp = await rag_service.answer_query(
        query=query.strip(),
        embedding_ids=embedding_ids,
        db_session=db,
        mode=Mode.RAG,
        top_k=5,
        min_score=0.0,
        use_web=False,
    )
    assist_msg = await svc_chat.add_message(
        chat_id,
        ChatRole.ASSISTANT,
        resp["response"],
        source=resp["meta"]["source"],
        source_files=resp["meta"].get("files"),
    )
    created_msgs.append(
        ChatMessageResponse(
            role=assist_msg.role,
            content=assist_msg.content,
            created_at=assist_msg.created_at.isoformat(),
            source=assist_msg.source,
            source_files=assist_msg.source_files,
        )
    )

    return created_msgs
