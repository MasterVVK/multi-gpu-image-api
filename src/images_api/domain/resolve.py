"""Применение реестра к запросу: размеры, шаги, guidance, планировщик.

Вынесено из валидатора pydantic намеренно: правила зависят от модели, а
валидатор про модель ничего не знает. Здесь же — единственное место, где
принимается решение «этот размер для этой модели недопустим».
"""

from __future__ import annotations

from images_worker.registry import MODELS, ModelSpec, get_spec

from ..core.errors import ValidationError
from .schemas import GenerateRequest, ResolvedParams


def known_models() -> list[str]:
    return sorted(MODELS)


def resolve_spec(model_id: str) -> ModelSpec:
    try:
        return get_spec(model_id)
    except KeyError:
        # Отдельная подсказка для трёх старых значений: раньше {"model": "turbo"}
        # был валиден в трёх сервисах с разным смыслом, и молчаливо угадывать нельзя.
        legacy = {
            "turbo": ["z-image-turbo", "ernie-image-turbo"],
            "base": ["z-image-base"],
            "sft": ["ernie-image"],
        }
        hint = ""
        if model_id in legacy:
            hint = " Раньше это значение было валидно в нескольких сервисах; укажите явно: %s." % (
                ", ".join(legacy[model_id])
            )
        raise ValidationError(
            "неизвестная модель %r. Доступны: %s.%s" % (model_id, ", ".join(known_models()), hint)
        ) from None


def _round_down(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


def nearest_preset(spec: ModelSpec, width: int, height: int) -> tuple[str, int, int]:
    """Ближайший пресет по соотношению сторон — для моделей без произвольных размеров."""
    target = width / height
    best_name, best = None, None
    best_delta = float("inf")
    for name, (pw, ph) in spec.aspect_ratios.items():
        delta = abs((pw / ph) - target)
        if delta < best_delta:
            best_delta, best_name, best = delta, name, (pw, ph)
    assert best_name is not None and best is not None
    return best_name, best[0], best[1]


def resolve(request: GenerateRequest, spec: ModelSpec | None = None) -> ResolvedParams:
    spec = spec or resolve_spec(request.model)

    aspect_ratio: str | None = None

    if request.width is not None and request.height is not None:
        width, height = request.width, request.height

        if not spec.arbitrary_size:
            raise ValidationError(
                "модель %s поддерживает только пресеты aspect_ratio: %s"
                % (spec.id, ", ".join(sorted(spec.aspect_ratios)))
            )

        for name, value in (("width", width), ("height", height)):
            if value % spec.dim_multiple:
                raise ValidationError(
                    "%s должно быть кратно %d для модели %s" % (name, spec.dim_multiple, spec.id)
                )
            if not (spec.min_dim <= value <= spec.max_dim):
                raise ValidationError(
                    "%s для %s должно быть в диапазоне %d..%d"
                    % (name, spec.id, spec.min_dim, spec.max_dim)
                )
    else:
        aspect_ratio = request.aspect_ratio or spec.default_aspect_ratio
        if aspect_ratio not in spec.aspect_ratios:
            raise ValidationError(
                "aspect_ratio %r недоступен для %s. Доступны: %s"
                % (aspect_ratio, spec.id, ", ".join(sorted(spec.aspect_ratios)))
            )
        width, height = spec.aspect_ratios[aspect_ratio]

    steps = request.num_inference_steps or spec.default_steps
    if steps > spec.max_steps:
        raise ValidationError(
            "num_inference_steps=%d превышает максимум %d для %s" % (steps, spec.max_steps, spec.id)
        )

    guidance = request.guidance_scale if request.guidance_scale is not None else spec.default_guidance

    shift = None
    scheduler = None
    if spec.default_shift is not None:
        shift = request.shift if request.shift is not None else spec.default_shift
    elif request.shift is not None:
        raise ValidationError("параметр shift поддерживает только %s" % _models_with_shift())

    if spec.schedulers:
        scheduler = request.scheduler or spec.default_scheduler
        if scheduler not in spec.schedulers:
            raise ValidationError(
                "scheduler %r недоступен для %s. Доступны: %s"
                % (scheduler, spec.id, ", ".join(spec.schedulers))
            )
    elif request.scheduler is not None:
        raise ValidationError("параметр scheduler поддерживает только %s" % _models_with_scheduler())

    negative_prompt = request.negative_prompt or ""
    if negative_prompt and not spec.supports_negative_prompt:
        raise ValidationError("модель %s не принимает negative_prompt" % spec.id)

    return ResolvedParams(
        model_id=spec.id,
        backend=spec.backend,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        num_inference_steps=steps,
        guidance_scale=guidance,
        shift=shift,
        scheduler=scheduler,
        negative_prompt=negative_prompt,
    )


def coerce_size(spec: ModelSpec, width: int, height: int) -> tuple[int, int, str | None]:
    """Подогнать произвольный размер под возможности модели.

    Используется OpenAI-слоем, где `size` приходит от чужого клиента и падать
    с 422 из-за некратности — плохое поведение: OpenAI-клиенты такого не ждут.
    """
    if not spec.arbitrary_size:
        name, pw, ph = nearest_preset(spec, width, height)
        return pw, ph, name

    w = min(max(_round_down(width, spec.dim_multiple), spec.min_dim), spec.max_dim)
    h = min(max(_round_down(height, spec.dim_multiple), spec.min_dim), spec.max_dim)
    w = _round_down(w, spec.dim_multiple)
    h = _round_down(h, spec.dim_multiple)
    return w, h, None


def _models_with_shift() -> str:
    return ", ".join(sorted(m.id for m in MODELS.values() if m.default_shift is not None))


def _models_with_scheduler() -> str:
    return ", ".join(sorted(m.id for m in MODELS.values() if m.schedulers))
