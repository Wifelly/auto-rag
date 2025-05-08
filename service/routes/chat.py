from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from service.clients.llm_service import llm_service
from service.clients.rag_service import rag_service
from service.database.config import get_db
from service.database.models import ChatRole
from service.monitoring.logger import logger
from service.services.chat_file_pipeline import ChatFileHandler
from service.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


class CreateChatRequest(BaseModel):
    user_id: int
    embedding_ids: list[int] = []
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
    top_k: int = 5
    min_score: float = 0.0
    use_rag: bool = True
    use_web: bool = False
    use_local_llm: bool = True
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class ChatResponseOutput(BaseModel):
    response: str
    source: str
    files: list[str] | None = None


class UpdateChatTitleRequest(BaseModel):
    title: str


@router.post("/", summary="Создать новый чат")
async def create_chat(req: CreateChatRequest, db: AsyncSession = Depends(get_db)):
    service = ChatService(db)
    session = await service.create_session(
        user_id=req.user_id,
        embedding_ids=req.embedding_ids,
        title=req.title,
    )
    return {"chat_id": session.id}


@router.post("/{chat_id}/send", summary="Отправить сообщение")
async def send_message(chat_id: UUID, req: SendMessageRequest, db: AsyncSession = Depends(get_db)):
    service = ChatService(db)
    session = await service.get_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Чат не найден")
    msg = await service.add_message(chat_id, role=req.role, content=req.content)
    return {"message_id": msg.id}


@router.post("/{chat_id}/upload-file", summary="Загрузить файл в чат")
async def upload_file(chat_id: UUID, file: UploadFile, db: AsyncSession = Depends(get_db)):
    service = ChatService(db)
    session = await service.get_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Чат не найден")

    file_handler = ChatFileHandler()
    documents = await file_handler.handle_uploaded_file(file)

    if documents:
        all_files = [file.filename]
        content = "\n\n".join(doc.page_content for doc in documents)
        await service.add_message(
            chat_id,
            ChatRole.USER,
            content=content,
            source="uploaded_file",
            source_files=all_files,
        )

        return {"status": "success", "message": f"Файл {file.filename} успешно загружен и обработан."}
    else:
        return {"status": "failure", "message": "Ошибка обработки файла."}


@router.get("/{chat_id}/history", response_model=list[ChatMessageResponse])
async def get_history(chat_id: UUID, db: AsyncSession = Depends(get_db)):
    service = ChatService(db)
    session = await service.get_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Чат не найден")

    messages = await service.get_messages(chat_id)
    return [
        ChatMessageResponse(
            role=m.role,
            content=m.content,
            created_at=m.created_at.isoformat(),
            source=m.source,
            source_files=m.source_files,
        )
        for m in messages
    ]


@router.post("/{chat_id}/respond", response_model=ChatResponseOutput, summary="Получить ответ от LLM")
async def respond(chat_id: UUID, req: ChatResponseRequest, db: AsyncSession = Depends(get_db)):
    service = ChatService(db)
    session = await service.get_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Чат не найден")

    await service.add_message(chat_id, ChatRole.USER, req.query)

    if req.use_rag and session.embeddings:
        context_parts = []
        all_files: set[str] = set()
        used_embedding_ids: list[str] = []

        for embedding in session.embeddings:
            result = await rag_service.answer_query(
                query=req.query,
                embedding=embedding,
                db_session=db,
                mode="rag",
                top_k=req.top_k,
                min_score=req.min_score,
                use_web=req.use_web,
                use_local_llm=req.use_local_llm,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
            )

            meta = result.get("meta", {})
            initial = meta.get("initial_response", "")
            refined = result.get("response", "") or result.get("refined", "")
            context_found = meta.get("context_found", False)

            if not context_found:
                part = (
                    f"\n### Ответ от LLM:\n{initial}\n\n"
                    f"_(Контекст из эмбеддинга {embedding.name or embedding.id} не найден)_"
                )
            else:
                part = (
                    f"\n### Ответ от LLM:\n{initial}\n\n"
                    f"### Уточнённый ответ с контекстом ({embedding.name or embedding.id}):\n{refined}"
                )

            context_parts.append(part)

            if meta.get("files"):
                all_files.update(meta["files"])
            else:
                logger.warning(f"[CHAT] RAG не вернул files для embedding_id={embedding.id}")

            used_embedding_ids.append(str(embedding.id))

        full_context = "\n".join(context_parts)
        source = f"rag::{','.join(used_embedding_ids)}"

        await service.add_message(
            chat_id,
            ChatRole.ASSISTANT,
            content=full_context,
            source=source,
            source_files=list(all_files),
        )

        return ChatResponseOutput(response=full_context, source=source, files=list(all_files))

    history = await service.get_messages(chat_id)
    messages = [{"role": m.role.value, "content": m.content} for m in history]
    messages.append({"role": "user", "content": req.query})

    answer = await llm_service.call(
        messages,
        use_local_llm=req.use_local_llm,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )

    await service.add_message(chat_id, ChatRole.ASSISTANT, answer, source="llm")

    return ChatResponseOutput(response=answer, source="llm", files=None)


@router.patch("/{chat_id}/title", summary="Обновить заголовок чата вручную")
async def update_chat_title(chat_id: UUID, req: UpdateChatTitleRequest, db: AsyncSession = Depends(get_db)):
    service = ChatService(db)
    session = await service.get_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Чат не найден")

    session.title = req.title.strip()
    await db.commit()
    return {"chat_id": session.id, "new_title": session.title}


@router.delete("/{chat_id}", summary="Удалить чат")
async def delete_chat(chat_id: UUID, db: AsyncSession = Depends(get_db)):
    service = ChatService(db)
    session = await service.get_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Чат не найден")

    await service.delete_chat(chat_id)
    return {"chat_id": chat_id, "status": "deleted"}
