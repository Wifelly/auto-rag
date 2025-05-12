import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from service.clients.context_retriever import ContextRetriever
from service.clients.llm_service import llm_service
from service.clients.prompt_builder import PromptBuilder
from service.database.config import get_db

router = APIRouter(prefix="/assistant", tags=["Assistant"])


class SummarizeRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: str


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    messages = [
        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
        {"role": "user", "content": f"Сделай краткий конспект следующего текста:\n\n{req.text}"},
    ]
    result = await llm_service.call(messages=messages)
    return SummarizeResponse(summary=result["response"])


class QuizRequest(BaseModel):
    text: str
    num_questions: int = Field(5, ge=1)
    difficulty: str = Field("medium")


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    answer: str


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]


@router.post("/quiz", response_model=QuizResponse)
async def generate_quiz(req: QuizRequest):
    messages = [
        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Сгенерируй {req.num_questions} контрольных вопросов по следующему материалу "
                f"(сложность: {req.difficulty}):\n\n{req.text}"
            ),
        },
    ]
    result = await llm_service.call(messages=messages)
    return QuizResponse.model_validate_json(result["response"])


class QuizFromDocRequest(BaseModel):
    embedding_ids: list[int] = Field(..., description="ID векторных индексов")
    query: str | None = Field(None, description="Текстовый запрос (можно не указывать)")
    num_questions: int = Field(5, ge=1)
    difficulty: str = Field("medium")


class QuizFromDocResponse(BaseModel):
    questions: list[QuizQuestion]


@router.post("/quiz-from-doc", response_model=QuizResponse)
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
        raise HTTPException(status_code=404, detail="Контекст не найден")

    user_content = (
        f"У тебя есть следующие выдержки из документа:\n\n{context_text}\n\n"
        f"Сгенерируй {req.num_questions} контрольных вопросов "
        f"(сложность: {req.difficulty}). "
        "Верни ответ в формате JSON:"
        '{"questions":[{"question":...,"options":[...],"answer":...},...]}'
    )
    messages = [
        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    result = await llm_service.call(messages=messages)
    parsed = json.loads(result["response"])
    return QuizResponse.model_validate(parsed)


class EvaluateRequest(BaseModel):
    question: str
    student_answer: str
    rubric: list[str] = Field(..., description="Критерии оценки")


class EvaluateResponse(BaseModel):
    score: int
    max_score: int
    feedback: str


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    user_msg = (
        f"Оцени ответ ученика на вопрос:\n{req.question}\n\n"
        f"Ответ ученика:\n{req.student_answer}\n\n"
        f"Критерии оценки:\n- "
        + "\n- ".join(req.rubric)
        + '\n\nВерни JSON вида {"score": <число>, "max_score": <число>, "feedback": "..."}'
    )
    messages = [{"role": "system", "content": PromptBuilder.SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
    result = await llm_service.call(messages=messages)
    return EvaluateResponse.parse_raw(result["response"])


class ContextPreviewRequest(BaseModel):
    embedding_ids: list[int] = Field(default_factory=list)
    query: str
    top_k: int = 5
    min_score: float = 0.0


class ContextPreviewResponse(BaseModel):
    preview: str


@router.post("/context-preview", response_model=ContextPreviewResponse)
async def context_preview(
    req: ContextPreviewRequest,
    db: AsyncSession = Depends(get_db),
):
    text, files = await ContextRetriever(db).get_context(
        embedding_ids=req.embedding_ids,
        query=req.query,
        top_k=req.top_k,
        min_score=req.min_score,
    )
    return ContextPreviewResponse(preview=text)
