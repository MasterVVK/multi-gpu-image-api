"""Запись реестра minimax-h3-image и то, как её понимает шлюз.

Часть про шлюз (resolve, схема запроса, тайм-аут) требует его исходников; путь к копии
репозитория с применённым registry_patch.py задаётся IMG_GATEWAY_SRC (каталог src),
иначе эти тесты пропускаются.
"""

from __future__ import annotations

import os
import sys

import pytest
from images_worker.registry import (
    BACKENDS,
    DEFAULT_WORKER_PORTS,
    MODELS,
    get_spec,
    models_for_backend,
)

GATEWAY_SRC = os.environ.get("IMG_GATEWAY_SRC", "")
needs_gateway = pytest.mark.skipif(not GATEWAY_SRC, reason="IMG_GATEWAY_SRC is not set")
if GATEWAY_SRC and GATEWAY_SRC not in sys.path:
    sys.path.insert(0, GATEWAY_SRC)

PRESETS = {
    "1:1": (4096, 4096),
    "4:3": (4736, 3552),
    "3:4": (3552, 4736),
    "3:2": (5024, 3360),
    "2:3": (3360, 5024),
    "16:9": (5472, 3072),
    "9:16": (3072, 5472),
    "21:9": (6272, 2688),
    "9:21": (2688, 6272),
}


def test_spec():
    spec = get_spec("minimax-h3-image")
    assert spec.backend == "minimax" and "minimax" in BACKENDS
    assert DEFAULT_WORKER_PORTS["minimax"] == 8107
    assert len(set(DEFAULT_WORKER_PORTS.values())) == len(DEFAULT_WORKER_PORTS)
    assert models_for_backend("minimax") == ("minimax-h3-image",)
    assert spec.repo_id == "MiniMaxAI/MiniMax-H3-Image"
    assert spec.aspect_ratios == PRESETS
    assert spec.default_size == (4096, 4096)
    assert spec.default_steps == spec.max_steps == 6
    assert spec.guidance_param == "none"
    assert spec.supports_negative_prompt is False
    assert spec.cpu_offload is False and spec.weights_vram_mb == 0
    assert spec.peak_vram_mb == 71800 < spec.card_class_mb == 98304
    # Холст пайплайна кратен 32; пресеты ему уже кратны и почти равны по площади.
    for width, height in spec.aspect_ratios.values():
        assert width % 32 == 0 and height % 32 == 0
        assert 16.7e6 < width * height < 16.9e6
    # Произвольный размер не может дать больше пикселей, чем пресет: пик VRAM тот же.
    assert spec.max_dim * spec.max_dim <= max(w * h for w, h in PRESETS.values())


def test_other_models_keep_the_24gb_class():
    for spec in MODELS.values():
        # Модели большого класса перечислены явно: кроме самой MiniMax — Qwen-Image 2.1
        # и ERNIE-Image целиком на карте.
        if spec.id not in {"minimax-h3-image", "qwen-image-2.1-full", "ernie-image-full",
                         "z-image-base-full", "moss-image-full"}:
            assert spec.card_class_mb == 24576 and spec.peak_vram_mb < 24576, spec.id


@needs_gateway
@pytest.mark.parametrize("ratio", sorted(PRESETS))
def test_presets_resolve_through_aspect_ratio(ratio):
    """Стороны до 6272 проходят через aspect_ratio: потолок 4096 стоит только на явных width/height."""
    from images_api.domain.resolve import resolve
    from images_api.domain.schemas import GenerateRequest

    params = resolve(GenerateRequest(model="minimax-h3-image", prompt="x", aspect_ratio=ratio))
    assert (params.width, params.height) == PRESETS[ratio]
    assert params.aspect_ratio == ratio
    assert params.num_inference_steps == 6
    assert params.negative_prompt == ""


@needs_gateway
def test_request_rules():
    import pydantic

    from images_api.core.errors import ValidationError
    from images_api.domain.resolve import resolve
    from images_api.domain.schemas import GenerateRequest

    def req(**kw):
        return GenerateRequest(model="minimax-h3-image", prompt="x", **kw)

    assert (resolve(req()).width, resolve(req()).height) == (4096, 4096)
    assert (resolve(req(width=4096, height=2048)).width, resolve(req(width=4096, height=2048)).height) == (4096, 2048)

    # Явные width/height выше 4096 отсекает схема запроса — до реестра дело не доходит.
    with pytest.raises(pydantic.ValidationError):
        req(width=5472, height=3072)
    with pytest.raises(ValidationError):
        resolve(req(width=1000, height=1024))  # не кратно 32
    with pytest.raises(ValidationError):
        resolve(req(width=512, height=512))  # меньше min_dim
    with pytest.raises(ValidationError):
        resolve(req(num_inference_steps=7))
    with pytest.raises(ValidationError):
        resolve(req(negative_prompt="blurry"))
    with pytest.raises(ValidationError):
        resolve(req(aspect_ratio="5:4"))
    # max_steps == default_steps означает «число шагов фиксировано»: шлюз принимает только 6.
    with pytest.raises(ValidationError):
        resolve(req(num_inference_steps=4))
    assert resolve(req(num_inference_steps=6)).num_inference_steps == 6


@needs_gateway
def test_gateway_budget():
    from images_api.services.scheduler import generate_timeout, required_mb

    # 240 + 6 * 55 * 3 = 1230 с: 20.5 минуты на чтение ~130 ГБ с диска, денойз и декод.
    assert generate_timeout("minimax-h3-image", None) == 1230.0
    assert generate_timeout("minimax-h3-image", 6) == 1230.0
    # Резидентных весов нет: шлюз всегда требует полный пик.
    assert required_mb("minimax-h3-image", resident=False) == 71800
    assert required_mb("minimax-h3-image", resident=True) == 71800


@needs_gateway
def test_job_carries_large_sizes():
    from images_worker.protocol import GenerateJob, GenerateMeta

    job = GenerateJob(
        task_id="t", model_id="minimax-h3-image", prompt="x", width=6272, height=2688, num_inference_steps=6, guidance_scale=1.0
    )
    assert (job.width, job.height) == (6272, 2688)
    meta = GenerateMeta(
        task_id="t", model_id="minimax-h3-image", backend="minimax", node="node-c", width=6272, height=2688,
        num_steps=6, guidance_scale=1.0, seed_used=1, generation_time_seconds=600.0,
    )
    assert GenerateMeta.from_header(meta.to_header()).width == 6272
