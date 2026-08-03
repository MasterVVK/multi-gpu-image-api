"""Выдача изображений.

Авторизация здесь отдельная: пускаем либо по API-ключу, либо по HMAC-подписи в
query. Без подписи `<img src>` не работал бы — браузер не добавляет заголовки
к картинкам.

Проверка пути — в ImageStorage.resolve: разрешающая регулярка плюс доказательство
вложенности через is_relative_to. В старом коде проверялись только суффикс .png
и UUID в имени, поэтому путь вида
`2026/08/03/../../../qi-api/outputs/<uuid>.png` успешно читал чужую выдачу.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from ...core.security import authorize_image_access
from ..deps import get_storage

router = APIRouter(tags=["images"])


@router.get("/images/{image_path:path}")
async def get_image(request: Request, image_path: str) -> FileResponse:
    await authorize_image_access(request, image_path)
    target = get_storage(request).resolve(image_path)
    return FileResponse(
        target,
        media_type="image/png",
        headers={
            # Картинки неизменяемы: имя содержит UUID, содержимое не меняется.
            "Cache-Control": "private, max-age=86400, immutable",
        },
    )
