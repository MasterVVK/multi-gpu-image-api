"""POST /v1/images/generations — блокирующая обёртка над асинхронным конвейером.

Два неочевидных решения:

* **Лимит n.** При эксклюзивной аренде n задач выполняются строго по очереди:
  четыре hidream по 50 шагов — это около десяти минут висящего HTTP-запроса.
  Поэтому IMG_OPENAI_MAX_N=2, а для пакетов есть нативный асинхронный API.

* **Таймаут возвращает task_ids.** Отдать 504 и потерять уже запущенную работу
  было бы расточительно: генерация продолжается, и клиент может забрать
  результат через /api/v1/tasks/<id>.
"""

from __future__ import annotations

import asyncio
import base64
import io
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ...core.errors import AppError, ValidationError
from ...core.security import ApiKeyInfo, require_api_key
from ...domain.schemas import GenerateRequest
from ...services.scheduler import enqueue
from ..deps import get_scheduler, get_settings_dep, get_storage, get_tasks
from .mapping import (
    OpenAIImageData,
    OpenAIImageRequest,
    OpenAIImageResponse,
    resolve_model,
    resolve_size,
    resolve_steps,
)

router = APIRouter(prefix="/v1", tags=["openai"])


@router.post("/images/generations")
async def images_generations(
    request: Request,
    payload: OpenAIImageRequest,
    key: ApiKeyInfo = Depends(require_api_key),
):
    settings = get_settings_dep(request)
    scheduler = get_scheduler(request)
    tasks = get_tasks(request)
    storage = get_storage(request)

    if payload.n > settings.openai_max_n:
        raise ValidationError(
            "n=%d превышает лимит %d: при эксклюзивной аренде GPU задачи выполняются "
            "последовательно. Для пакетов используйте POST /api/v1/generate."
            % (payload.n, settings.openai_max_n)
        )

    spec = resolve_model(payload.model, settings)
    width, height, aspect_ratio = resolve_size(spec, payload.size)
    steps = resolve_steps(spec, payload.quality)

    group_id = str(uuid.uuid4())
    task_ids: list[str] = []

    for index in range(payload.n):
        native = GenerateRequest(
            model=spec.id,
            prompt=payload.prompt,
            aspect_ratio=aspect_ratio,
            width=None if aspect_ratio else width,
            height=None if aspect_ratio else height,
            num_inference_steps=steps,
            # seed задаём по индексу, иначе n одинаковых задач дали бы n копий
            # одной картинки.
            seed=(payload.seed + index) if payload.seed is not None else None,
            use_pe=None,
        )
        from ...domain.resolve import resolve as resolve_params

        resolved = resolve_params(native, spec)
        task_ids.append(
            await enqueue(
                scheduler,
                native,
                resolved,
                source="openai",
                group_id=group_id,
                api_key_id=key.key_id,
                request_id=getattr(request.state, "request_id", None),
            )
        )

    try:
        rows = await asyncio.wait_for(
            _await_all(tasks, task_ids), timeout=settings.openai_timeout_s
        )
    except TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "генерация не уложилась в %.0f с; заберите результат через "
                    "/api/v1/tasks/<id>" % settings.openai_timeout_s,
                    "type": "timeout",
                    "param": None,
                    "code": "timeout",
                },
                "task_ids": task_ids,
            },
        )

    data: list[OpenAIImageData] = []
    for row in rows:
        if row["status"] != "completed":
            raise AppError(
                row["error"] or "генерация не удалась (%s)" % row["error_code"],
                {"task_id": row["task_id"], "error_code": row["error_code"]},
            )

        entry = OpenAIImageData(revised_prompt=row["enhanced_prompt"])
        if payload.response_format == "b64_json":
            path = storage.resolve(row["image_path"])
            raw = await asyncio.get_running_loop().run_in_executor(
                None, _read_as, path, payload.output_format
            )
            entry.b64_json = base64.b64encode(raw).decode("ascii")
        else:
            entry.url = storage.image_url(row["image_path"], absolute=True)
        data.append(entry)

    return OpenAIImageResponse(
        created=int(datetime.now(UTC).timestamp()),
        data=data,
    )


@router.post("/images/edits")
async def images_edits() -> JSONResponse:
    return _not_implemented("редактирование изображений")


@router.post("/images/variations")
async def images_variations() -> JSONResponse:
    return _not_implemented("вариации изображений")


def _not_implemented(what: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "%s не поддерживается: ни у одной из подключённых моделей "
                "нет img2img-режима" % what,
                "type": "invalid_request_error",
                "param": None,
                "code": "not_implemented",
            }
        },
    )


async def _await_all(tasks, task_ids: list[str]) -> list:
    """Дождаться завершения всех задач группы, сохраняя их порядок."""
    pending = set(task_ids)
    finished: dict[str, object] = {}

    while pending:
        for task_id in list(pending):
            row = await tasks.get(task_id)
            if row is not None and row["status"] in ("completed", "failed", "cancelled"):
                finished[task_id] = row
                pending.discard(task_id)
        if pending:
            await asyncio.sleep(1.0)

    return [finished[task_id] for task_id in task_ids]


def _read_as(path, output_format: str) -> bytes:
    raw = path.read_bytes()
    if output_format == "png":
        return raw

    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    buffer = io.BytesIO()
    if output_format == "jpeg":
        image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=95)
    else:
        image.save(buffer, format="WEBP", quality=95)
    return buffer.getvalue()
