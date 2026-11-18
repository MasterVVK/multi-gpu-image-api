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
        # Только для моделей с произвольным размером: у «только пресеты» явный
        # размер — ошибка запроса по контракту, даже если числа совпали.
        if spec.arbitrary_size:
            aspect_ratio = _preset_name(spec, width, height)

    if aspect_ratio is not None:
        # Размер, совпавший с пресетом, — это пресет: /api/v1/resolutions сам
        # отдаёт эти числа, и отвергать их по max_dim было бы нечестно (у
        # qwen-image-2.1 и hidream часть пресетов длиннее max_dim).
        pass
    elif request.width is not None and request.height is not None:
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
    if spec.fixed_steps and steps != spec.default_steps:
        # Сетка шагов зашита в чекпойнт: записать в задачу другое число значило бы
        # соврать и в истории, и в расчёте тайм-аута.
        raise ValidationError(
            "у модели %s число шагов фиксировано: %d" % (spec.id, spec.default_steps)
        )
    if steps > spec.max_steps:
        raise ValidationError(
            "num_inference_steps=%d превышает максимум %d для %s" % (steps, spec.max_steps, spec.id)
        )

    guidance = request.guidance_scale if request.guidance_scale is not None else spec.default_guidance
    if spec.fixed_guidance and guidance != spec.default_guidance:
        raise ValidationError(
            "модель %s не принимает guidance_scale: он зашит в чекпойнт" % spec.id
        )

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
    threshold = spec.negative_needs_guidance_above
    if negative_prompt and threshold is not None and guidance <= threshold:
        raise ValidationError(
            "negative_prompt у модели %s действует только при guidance_scale > %g "
            "(сейчас %g): увеличьте guidance_scale или уберите negative_prompt"
            % (spec.id, threshold, guidance)
        )

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

    preset = _preset_name(spec, width, height)
    if preset is not None:
        return width, height, preset
    if width > spec.max_dim or height > spec.max_dim:
        # Срезать стороны по отдельности нельзя: 2752x1536 превратилось бы в
        # 2048x1536, то есть 16:9 стало бы 4:3. Ближайший по пропорциям пресет
        # сохраняет кадр, а пресеты крупнее max_dim у таких моделей как раз есть.
        name, pw, ph = nearest_preset(spec, width, height)
        return pw, ph, name

    w = min(max(_round_down(width, spec.dim_multiple), spec.min_dim), spec.max_dim)
    h = min(max(_round_down(height, spec.dim_multiple), spec.min_dim), spec.max_dim)
    w = _round_down(w, spec.dim_multiple)
    h = _round_down(h, spec.dim_multiple)
    return w, h, None


def _preset_name(spec: ModelSpec, width: int, height: int) -> str | None:
    """Имя пресета с точно такими размерами или None."""
    for name, size in spec.aspect_ratios.items():
        if size == (width, height):
            return name
    return None


def _models_with_shift() -> str:
    return ", ".join(sorted(m.id for m in MODELS.values() if m.default_shift is not None))


def _models_with_scheduler() -> str:
    return ", ".join(sorted(m.id for m in MODELS.values() if m.schedulers))
