from fastapi import APIRouter, Depends
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
    num_questions: int = 5
    difficulty: str = "medium"


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
    return QuizResponse.parse_raw(result["response"])


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
        f"Критерии оценки:\n- " + "\n- ".join(req.rubric) + "\n\nВерни JSON вида "
        '{"score": <число>, "max_score": <число>, "feedback": "..."}'
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
    text, _files = await ContextRetriever(db).get_context(
        embedding_ids=req.embedding_ids,
        query=req.query,
        top_k=req.top_k,
        min_score=req.min_score,
    )
    return ContextPreviewResponse(preview=text)
