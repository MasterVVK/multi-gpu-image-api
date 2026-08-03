"""GET /v1/models — каталог в формате OpenAI."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ...core.security import require_api_key
from ..deps import get_settings_dep
from .mapping import model_list

router = APIRouter(prefix="/v1", tags=["openai"])


@router.get("/models", dependencies=[Depends(require_api_key)])
async def list_models(request: Request) -> dict:
    return {"object": "list", "data": model_list(get_settings_dep(request))}
