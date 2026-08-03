"""OpenAI-совместимый контур.

Живёт под /v1 и намеренно отвечает в конверте ошибок OpenAI
(`{"error": {message, type, param, code}}`), а не в нативном: иначе openai-SDK
не распознаёт ошибку. Выбор конверта делает install_exception_handlers по
префиксу маршрута.
"""

from __future__ import annotations

from fastapi import APIRouter

from . import images, models

router = APIRouter()
router.include_router(images.router)
router.include_router(models.router)
