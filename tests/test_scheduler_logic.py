"""Чистая логика планировщика: сколько VRAM просить и сколько ждать."""

from __future__ import annotations

from images_worker.registry import get_spec

from images_api.services.scheduler import generate_timeout, required_mb


def test_offload_model_needs_full_peak_even_when_resident():
    """При cpu_offload резидентность не экономит VRAM: веса в системной RAM."""
    spec = get_spec("z-image-turbo")
    assert spec.weights_vram_mb == 0
    assert required_mb("z-image-turbo", resident=True) == spec.peak_vram_mb
    assert required_mb("z-image-turbo", resident=False) == spec.peak_vram_mb


def test_resident_model_asks_only_for_the_difference():
    """У модели «всё на карте» веса уже в used_mb — требовать их второй раз нельзя,
    иначе резидентная модель не проходила бы собственную проверку."""
    spec = get_spec("flux3-klein-full")
    assert spec.weights_vram_mb > 0
    cold = required_mb("flux3-klein-full", resident=False)
    warm = required_mb("flux3-klein-full", resident=True)
    assert cold == spec.peak_vram_mb
    assert warm == spec.peak_vram_mb - spec.weights_vram_mb
    assert warm < cold


def test_required_never_drops_to_zero():
    """Даже у резидентной модели нужен запас под активации."""
    assert required_mb("flux3-klein-full", resident=True) >= 1024


def test_timeout_has_warmup_allowance():
    """Первый прогон после загрузки заметно медленнее: без запаса тайм-аут
    обрывал честную генерацию z-image на 141-й секунде."""
    assert generate_timeout("z-image-turbo", 9) > 300
    assert generate_timeout("z-image-turbo", None) == generate_timeout("z-image-turbo", 9)


def test_timeout_grows_with_steps():
    assert generate_timeout("flux3-klein-full", 50) > generate_timeout("flux3-klein-full", 10)


def test_every_model_timeout_covers_documented_runtime():
    """Грубая проверка адекватности: тайм-аут не должен быть ниже
    ожидаемого времени работы на дефолтных шагах."""
    for model_id in ("qwen-image", "qwen-image-2.1", "qwen-image-2.1-bf16",
                     "qwen-image-2.1-full", "qwen-image-2.1-turbo", "moss-image",
                     "minimax-h3-image",
                     "z-image-turbo", "z-image-base",
                     "ernie-image-turbo", "ernie-image", "flux3-klein-full"):
        spec = get_spec(model_id)
        expected = spec.default_steps * spec.seconds_per_step_hint
        assert generate_timeout(model_id, None) > expected * 2
