"""Каталог моделей и разрешений.

Здесь же видно главное отличие от старых сервисов: `available_on` показывает,
на каких узлах веса реально лежат. Пустой список — модель сейчас негде выполнить,
и запрос к ней завершится model_unavailable, а не повиснет в очереди.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from images_worker.registry import MODELS

from ...core.security import require_api_key
from ...domain.resolve import resolve_spec
from ...domain.schemas import (
    ModelInfo,
    ModelsResponse,
    ResolutionInfo,
    ResolutionsResponse,
)
from ..deps import get_pool, get_settings_dep

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse, dependencies=[Depends(require_api_key)])
async def list_models(request: Request) -> ModelsResponse:
    pool = get_pool(request)
    settings = get_settings_dep(request)

    models = []
    for spec in MODELS.values():
        available_on = sorted(n.name for n in pool.nodes.values() if n.serves(spec.id))
        loaded_on = sorted(n.name for n in pool.nodes.values() if n.resident_model == spec.id)
        models.append(
            ModelInfo(
                id=spec.id,
                display_name=spec.display_name,
                backend=spec.backend,
                default_steps=spec.default_steps,
                max_steps=spec.max_steps,
                default_guidance_scale=spec.default_guidance,
                default_shift=spec.default_shift,
                schedulers=list(spec.schedulers),
                aspect_ratios=sorted(spec.aspect_ratios),
                arbitrary_size_supported=spec.arbitrary_size,
                dim_multiple=spec.dim_multiple,
                min_dim=spec.min_dim,
                max_dim=spec.max_dim,
                supports_negative_prompt=spec.supports_negative_prompt,
                supports_pe=spec.supports_pe,
                peak_vram_mb=spec.peak_vram_mb,
                available_on=available_on,
                loaded_on=loaded_on,
                notes=spec.notes,
            )
        )

    models.sort(key=lambda m: m.id)
    return ModelsResponse(models=models, default_model=settings.openai_default_model)


@router.get(
    "/resolutions", response_model=ResolutionsResponse, dependencies=[Depends(require_api_key)]
)
async def resolutions(model: str) -> ResolutionsResponse:
    spec = resolve_spec(model)
    return ResolutionsResponse(
        model=spec.id,
        resolutions=[
            ResolutionInfo(aspect_ratio=name, width=size[0], height=size[1])
            for name, size in sorted(spec.aspect_ratios.items())
        ],
        arbitrary_size_supported=spec.arbitrary_size,
        dim_multiple=spec.dim_multiple,
        min_dim=spec.min_dim,
        max_dim=spec.max_dim,
    )
