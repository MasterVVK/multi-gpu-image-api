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


# --- Qwen-Image 2.1 ---------------------------------------------------------


def test_qwen21_defaults_follow_vendor_recommendation():
    """40 шагов без CFG — рекомендация Qwen; 1.0 здесь значит «CFG выключен»."""
    r = resolve(req(model="qwen-image-2.1", aspect_ratio="16:9"))
    assert (r.width, r.height) == (2752, 1536)
    assert r.num_inference_steps == 40
    assert r.guidance_scale == 1.0
    assert r.backend == "qwen21"


@pytest.mark.parametrize("model_id", ["qwen-image-2.1", "qwen-image-2.1-turbo"])
def test_qwen21_presets_are_divisible_by_32(model_id):
    """Пайплайн 2.1 требует кратности 32 и некратное молча округляет вниз."""
    for name, (w, h) in get_spec(model_id).aspect_ratios.items():
        assert w % 32 == 0 and h % 32 == 0, name


def test_qwen21_turbo_has_fixed_schedule():
    """Сетка сигм зашита в чекпойнт: принять другое число шагов значило бы
    записать в задачу то, чего пайплайн не сделает."""
    r = resolve(req(model="qwen-image-2.1-turbo"))
    assert r.num_inference_steps == 8
    with pytest.raises(ValidationError):
        resolve(req(model="qwen-image-2.1-turbo", num_inference_steps=20))


def test_qwen21_turbo_rejects_negative_prompt():
    with pytest.raises(ValidationError):
        resolve(req(model="qwen-image-2.1-turbo", negative_prompt="blurry"))


def test_qwen21_turbo_rejects_fewer_steps_too():
    """Не только больше восьми: любое другое число пайплайн не выполнит."""
    with pytest.raises(ValidationError):
        resolve(req(model="qwen-image-2.1-turbo", num_inference_steps=4))


def test_qwen21_turbo_rejects_foreign_guidance():
    with pytest.raises(ValidationError):
        resolve(req(model="qwen-image-2.1-turbo", guidance_scale=6.5))
    # Значение, совпадающее с зашитым, — не ошибка: клиенты шлют дефолт схемы.
    assert resolve(req(model="qwen-image-2.1-turbo", guidance_scale=1.0)).guidance_scale == 1.0


@pytest.mark.parametrize("model_id", ["qwen-image-2.1", "qwen-image", "moss-image"])
def test_negative_prompt_needs_cfg(model_id):
    """Negative без CFG пайплайн игнорирует — принять его молча нельзя."""
    with pytest.raises(ValidationError):
        resolve(req(model=model_id, negative_prompt="blurry", guidance_scale=1.0))
    r = resolve(req(model=model_id, negative_prompt="blurry", guidance_scale=4.0))
    assert r.negative_prompt == "blurry"


def test_qwen21_default_guidance_rejects_negative():
    """Дефолт 2.1 — CFG выключен, значит negative без явного guidance отвергается."""
    with pytest.raises(ValidationError):
        resolve(req(model="qwen-image-2.1", negative_prompt="blurry"))


def test_explicit_size_equal_to_preset_is_accepted_beyond_max_dim():
    """2752x1536 длиннее max_dim=2048, но это пресет из /resolutions."""
    r = resolve(req(model="qwen-image-2.1", width=2752, height=1536))
    assert (r.width, r.height, r.aspect_ratio) == (2752, 1536, "16:9")
    with pytest.raises(ValidationError):
        resolve(req(model="qwen-image-2.1", width=2752, height=1568))


def test_coerce_size_keeps_aspect_when_over_max_dim():
    """Покомпонентный срез превращал 16:9 в 4:3."""
    spec = get_spec("qwen-image-2.1")
    assert coerce_size(spec, 2752, 1536) == (2752, 1536, "16:9")
    assert coerce_size(spec, 3840, 2160) == (2752, 1536, "16:9")


def test_moss_defaults():
    r = resolve(req(model="moss-image", aspect_ratio="16:9"))
    assert (r.width, r.height) == (1920, 1088)
    assert r.num_inference_steps == 50 and r.guidance_scale == 3.0
    assert r.backend == "moss"


def test_qwen21_bf16_differs_only_in_weights():
    """Полные веса — та же модель для клиента: те же пресеты, шаги и правила CFG."""
    fp8, bf16 = get_spec("qwen-image-2.1"), get_spec("qwen-image-2.1-bf16")
    assert bf16.aspect_ratios == fp8.aspect_ratios
    assert (bf16.default_steps, bf16.default_guidance) == (fp8.default_steps, fp8.default_guidance)
    assert bf16.repo_id == fp8.repo_id and bf16.variant is None and fp8.variant == "fp8"
    assert bf16.peak_vram_mb > fp8.peak_vram_mb
