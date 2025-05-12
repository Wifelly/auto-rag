from fastapi import APIRouter, Depends, HTTPException, status
from httpx import ReadTimeout
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from service.clients.context_retriever import ContextRetriever
from service.clients.llm_service import llm_service
from service.clients.prompt_builder import PromptBuilder
from service.database.config import get_db

router = APIRouter(prefix="/assistant", tags=["Assistant"])


class SummarizeRequest(BaseModel):
    text: str = Field(..., max_length=10_000, description="Текст для конспектирования (макс. 10 000 символов)")


class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="Краткий конспект текста")


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Краткое резюме текста",
)
async def summarize(req: SummarizeRequest):
    messages = [
        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
        {"role": "user", "content": f"Сделай краткий конспект текста:\n\n{req.text}"},
    ]
    try:
        result = await llm_service.call(messages=messages)
        resp = result.get("response")
        if resp is None:
            raise ValueError("Empty response from LLM")
    except ReadTimeout:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "LLM service timeout")
    except Exception as err:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM error: {err}") from err

    return SummarizeResponse(summary=resp)


class QuizRequest(BaseModel):
    text: str = Field(..., max_length=10_000, description="Материал для вопросов (макс. 10 000 символов)")
    num_questions: int = Field(5, ge=1, description="Количество вопросов")
    difficulty: str = Field("medium", description="Сложность вопросов")


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    answer: str


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]


@router.post(
    "/quiz",
    response_model=QuizResponse,
    status_code=status.HTTP_200_OK,
    summary="Сгенерировать контрольные вопросы по тексту",
)
async def generate_quiz(req: QuizRequest):
    messages = [
        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Сгенерируй {req.num_questions} контрольных вопросов "
                f"(сложность: {req.difficulty}) по тексту:\n\n{req.text}\n"
                "Верни ответ в формате JSON: "
                '{"questions":[{"question":"...","options":["..."],"answer":"..."}]}'
            ),
        },
    ]
    try:
        result = await llm_service.call(messages=messages)
        resp = result.get("response")
        if resp is None:
            raise ValueError("Empty response from LLM")
        quiz_resp = QuizResponse.parse_raw(resp)
    except ReadTimeout:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "LLM service timeout")
    except Exception as err:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM error: {err}") from err

    return quiz_resp


class QuizFromDocRequest(BaseModel):
    embedding_ids: list[int] = Field(..., description="ID векторных индексов")
    query: str | None = Field(None, max_length=10_000, description="Текстовый запрос (макс. 10 000 символов)")
    num_questions: int = Field(5, ge=1, description="Количество вопросов")
    difficulty: str = Field("medium", description="Сложность вопросов")


@router.post(
    "/quiz-from-doc",
    response_model=QuizResponse,
    status_code=status.HTTP_200_OK,
    summary="Сгенерировать вопросы по контексту из документов",
)
async def quiz_from_doc(
    req: QuizFromDocRequest,
    db: AsyncSession = Depends(get_db),
):
    context_text, _ = await ContextRetriever(db).get_context(
        embedding_ids=req.embedding_ids,
        query=req.query or "",
        top_k=req.num_questions * 2,
        min_score=0.0,
    )
    if not context_text:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Контекст не найден")

    user_content = (
        f"Вот выдержки из документа:\n\n{context_text}\n\n"
        f"Сгенерируй {req.num_questions} вопросов "
        f"(сложность: {req.difficulty}) по этому материалу. "
        "Верни JSON вида: "
        '{"questions":[{"question":"...","options":["..."],"answer":"..."}]}'
    )
    messages = [
        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        result = await llm_service.call(messages=messages)
        resp = result.get("response")
        if resp is None:
            raise ValueError("Empty response from LLM")
        quiz_resp = QuizResponse.parse_raw(resp)
    except ReadTimeout:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "LLM service timeout")
    except Exception as err:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM error: {err}") from err

    return quiz_resp


class EvaluateRequest(BaseModel):
    question: str = Field(..., max_length=10_000, description="Вопрос (макс. 10 000 символов)")
    student_answer: str = Field(..., max_length=10_000, description="Ответ ученика (макс. 10 000 символов)")
    rubric: list[str] = Field(..., min_items=1, description="Критерии оценки")


class EvaluateResponse(BaseModel):
    score: int
    max_score: int
    feedback: str


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    status_code=status.HTTP_200_OK,
    summary="Оценить ответ ученика по рубрике",
)
async def evaluate(req: EvaluateRequest):
    user_msg = (
        f"Оцени ответ ученика на вопрос:\n{req.question}\n\n"
        f"Ответ ученика:\n{req.student_answer}\n\n"
        f"Критерии оценки:\n- " + "\n- ".join(req.rubric) + "\n\nВерни JSON вида: "
        '{"score": <число>, "max_score": <число>, "feedback": "..."}'
    )
    messages = [
        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        result = await llm_service.call(messages=messages)
        resp = result.get("response")
        if resp is None:
            raise ValueError("Empty response from LLM")
        eval_resp = EvaluateResponse.parse_raw(resp)
    except ReadTimeout:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "LLM service timeout")
    except Exception as err:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM error: {err}") from err

    return eval_resp


class ContextPreviewRequest(BaseModel):
    embedding_ids: list[int] = Field(..., description="ID векторных индексов")
    query: str = Field(..., max_length=10_000, description="Текстовый запрос (макс. 10 000 символов)")
    top_k: int = Field(5, ge=1, description="Количество фрагментов")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Минимальный порог скоринга")


class ContextPreviewResponse(BaseModel):
    preview: str = Field(..., description="Предпросмотр контекста")


@router.post(
    "/context-preview",
    response_model=ContextPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Получить превью контекста из документов",
)
async def context_preview(
    req: ContextPreviewRequest,
    db: AsyncSession = Depends(get_db),
):
    text, _ = await ContextRetriever(db).get_context(
        embedding_ids=req.embedding_ids,
        query=req.query,
        top_k=req.top_k,
        min_score=req.min_score,
    )
    if text == "":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Контекст не найден")
    return ContextPreviewResponse(preview=text)
