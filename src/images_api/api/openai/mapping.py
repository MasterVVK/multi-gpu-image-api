"""Перевод запроса OpenAI в нативный.

Принцип: чужой клиент не должен получать 422 из-за наших внутренних правил.
Некратный размер округляем вниз, неизвестный алиас модели — в дефолтную,
`quality` — в множитель шагов. Всё применённое видно в нативной задаче и в
заголовках ответа, так что подмена не молчаливая.
"""

from __future__ import annotations

import re
from typing import Literal

from images_worker.registry import MODELS, ModelSpec
from pydantic import BaseModel, ConfigDict, Field

from ...core.config import Settings
from ...domain.resolve import coerce_size, resolve_spec

# Алиасы моделей OpenAI: клиенты часто шлют их жёстко зашитыми.
MODEL_ALIASES = ("gpt-image-1", "gpt-image-1-mini", "dall-e-3", "dall-e-2")

_SIZE_RE = re.compile(r"^(\d{2,5})\s*[x×]\s*(\d{2,5})$", re.IGNORECASE)

# quality → множитель к числу шагов модели.
_QUALITY_STEPS = {
    "low": 0.6,
    "medium": 1.0,
    "standard": 1.0,
    "auto": 1.0,
    "high": 1.4,
    "hd": 1.4,
}


class OpenAIImageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "gpt-image-1"
    prompt: str = Field(..., min_length=1, max_length=4000)
    n: int = Field(default=1, ge=1, le=10)
    size: str = "auto"
    quality: Literal["auto", "low", "medium", "high", "standard", "hd"] = "auto"
    response_format: Literal["url", "b64_json"] = "url"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    # Принимаются и игнорируются — чтобы SDK не падал на неизвестном поле.
    style: str | None = None
    background: str | None = None
    user: str | None = None
    # Наше расширение: воспроизводимость нужна чаще, чем совместимость.
    seed: int | None = Field(default=None, ge=0)


class OpenAIImageData(BaseModel):
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


class OpenAIImageResponse(BaseModel):
    created: int
    data: list[OpenAIImageData]


def resolve_model(name: str, settings: Settings) -> ModelSpec:
    if name in MODELS:
        return MODELS[name]
    if name in MODEL_ALIASES or not name:
        return resolve_spec(settings.openai_default_model)
    # Незнакомое имя — тоже в дефолтную: клиент OpenAI не обязан знать наш каталог.
    return resolve_spec(settings.openai_default_model)


def resolve_size(spec: ModelSpec, size: str) -> tuple[int, int, str | None]:
    """`size` → (ширина, высота, применённый пресет или None)."""
    value = (size or "auto").strip().lower()
    if value in ("auto", "", "default"):
        width, height = spec.default_size
        return width, height, spec.default_aspect_ratio

    match = _SIZE_RE.match(value)
    if not match:
        width, height = spec.default_size
        return width, height, spec.default_aspect_ratio

    return coerce_size(spec, int(match.group(1)), int(match.group(2)))


def resolve_steps(spec: ModelSpec, quality: str) -> int:
    factor = _QUALITY_STEPS.get(quality, 1.0)
    steps = round(spec.default_steps * factor)
    return max(1, min(steps, spec.max_steps))


def model_list(settings: Settings) -> list[dict]:
    """Каталог для GET /v1/models: наши модели плюс алиасы.

    `created` — фиксированная константа: в скрипте нет доступа ко времени
    сборки, а клиентам это поле нужно лишь для формальной совместимости.
    """
    created = 1735689600  # 2025-01-01T00:00:00Z
    entries = [
        {"id": spec.id, "object": "model", "created": created, "owned_by": "images-api"}
        for spec in sorted(MODELS.values(), key=lambda s: s.id)
    ]
    entries.extend(
        {
            "id": alias,
            "object": "model",
            "created": created,
            "owned_by": "images-api",
            "aliased_to": settings.openai_default_model,
        }
        for alias in MODEL_ALIASES
    )
    return entries
