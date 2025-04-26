from typing import Any

from fastapi import APIRouter
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from starlette.responses import Response

router = APIRouter()


@router.get("/metrics", response_class=Response)
async def get() -> Any:
    return Response(generate_latest(REGISTRY), headers={"Content-Type": CONTENT_TYPE_LATEST})
