"""Разрешение параметров запроса по реестру моделей."""

from __future__ import annotations

import pytest
from images_worker.registry import get_spec

from images_api.core.errors import ValidationError
from images_api.domain.resolve import coerce_size, nearest_preset, resolve, resolve_spec
from images_api.domain.schemas import GenerateRequest


def req(**kwargs) -> GenerateRequest:
    kwargs.setdefault("prompt", "test")
    return GenerateRequest(**kwargs)


def test_defaults_from_registry():
    r = resolve(req(model="z-image-turbo"))
    assert (r.width, r.height) == (1024, 1024)
    assert r.num_inference_steps == 9
    # Дистилляция без CFG: ноль здесь осмысленное значение, а не «не задано».
    assert r.guidance_scale == 0.0


def test_cfg_scale_alias_accepted():
    """qi-api называл параметр cfg_scale — старые клиенты не должны сломаться."""
    r = resolve(req(model="qwen-image", cfg_scale=7.5))
    assert r.guidance_scale == 7.5


def test_guidance_scale_wins_over_alias():
    r = resolve(req(model="qwen-image", guidance_scale=3.0))
    assert r.guidance_scale == 3.0


def test_legacy_model_name_names_candidates():
    """{"model": "turbo"} был валиден в трёх сервисах — молча угадывать нельзя."""
    with pytest.raises(ValidationError) as exc:
        resolve_spec("turbo")
    assert "z-image-turbo" in str(exc.value)
    assert "ernie-image-turbo" in str(exc.value)


def test_ernie_rejects_explicit_size():
    """ERNIE умеет только пресеты — явный размер должен давать внятный отказ."""
    with pytest.raises(ValidationError) as exc:
        resolve(req(model="ernie-image-turbo", width=1024, height=1024))
    assert "aspect_ratio" in str(exc.value)


def test_dim_multiple_enforced():
    with pytest.raises(ValidationError) as exc:
        resolve(req(model="hidream-o1", width=1000, height=1024))
    assert "кратно 64" in str(exc.value)


def test_width_without_height_rejected():
    with pytest.raises(ValueError):
        req(model="hidream-o1", width=1024)


def test_aspect_ratio_and_size_are_exclusive():
    with pytest.raises(ValueError):
        req(model="hidream-o1", aspect_ratio="1:1", width=1024, height=1024)


def test_steps_above_model_max_rejected():
    with pytest.raises(ValidationError) as exc:
        resolve(req(model="z-image-turbo", num_inference_steps=50))
    assert "20" in str(exc.value)


def test_shift_only_for_hidream():
    with pytest.raises(ValidationError):
        resolve(req(model="z-image-turbo", shift=3.0))
    assert resolve(req(model="hidream-o1", shift=2.0)).shift == 2.0


def test_hidream_negative_prompt_rejected():
    """У hidream в пайплайне негативного промпта нет — молча ронять его нельзя."""
    with pytest.raises(ValidationError):
        resolve(req(model="hidream-o1", negative_prompt="blurry"))


def test_qwen_4x3_preset_is_divisible_by_8():
    """В qi-api стояло 1472x1140, а 1140 не делится на 8 — diffusers переокруглял."""
    w, h = get_spec("qwen-image").aspect_ratios["4:3"]
    assert w % 8 == 0 and h % 8 == 0


def test_coerce_size_rounds_down_for_arbitrary():
    spec = get_spec("hidream-o1")
    w, h, preset = coerce_size(spec, 1000, 1000)
    assert (w, h) == (960, 960) and preset is None


def test_coerce_size_snaps_to_preset_for_ernie():
    spec = get_spec("ernie-image-turbo")
    w, h, preset = coerce_size(spec, 1920, 1080)
    assert preset == "16:9" and (w, h) == (1376, 768)


def test_nearest_preset_by_aspect_not_area():
    spec = get_spec("ernie-image-turbo")
    name, _w, _h = nearest_preset(spec, 100, 100)
    assert name == "1:1"
