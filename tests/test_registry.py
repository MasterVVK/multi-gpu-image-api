"""Инварианты каталога моделей — он общий для шлюза и воркеров."""

from __future__ import annotations

import pytest
from images_worker.registry import BACKENDS, DEFAULT_WORKER_PORTS, MODELS, get_spec


def test_ids_are_flat_and_unique():
    """Плоское пространство имён вместо трёх разных model:"turbo"."""
    assert len(MODELS) == len({m.id for m in MODELS.values()})
    assert "turbo" not in MODELS and "base" not in MODELS and "sft" not in MODELS


@pytest.mark.parametrize("model_id", sorted(MODELS))
def test_spec_is_self_consistent(model_id):
    spec = get_spec(model_id)
    assert spec.backend in BACKENDS
    assert spec.repo_id or spec.local_path, "модель без источника весов"
    assert spec.aspect_ratios, "нет ни одного пресета"
    assert spec.default_aspect_ratio in spec.aspect_ratios
    assert 0 < spec.default_steps <= spec.max_steps
    assert spec.peak_vram_mb > 0
    assert spec.weights_vram_mb <= spec.peak_vram_mb
    assert spec.load_timeout_s > 0
    if spec.schedulers:
        assert spec.default_scheduler in spec.schedulers


@pytest.mark.parametrize("model_id", sorted(MODELS))
def test_presets_fit_the_card(model_id):
    """Пик модели обязан помещаться в карту её класса.

    Класс по умолчанию — 24 ГБ (3090/4090), так что для обычных моделей
    проверка прежняя. Больший класс модель объявляет явно, и таких моделей
    единицы — список ниже не даёт завести ещё одну незаметно.
    """
    spec = get_spec(model_id)
    assert spec.peak_vram_mb < spec.card_class_mb
    if model_id not in {"minimax-h3-image", "qwen-image-2.1-full", "ernie-image-full",
                        "z-image-base-full", "moss-image-full"}:
        assert spec.card_class_mb == 24576


@pytest.mark.parametrize("model_id", sorted(MODELS))
def test_arbitrary_size_models_declare_bounds(model_id):
    spec = get_spec(model_id)
    if spec.arbitrary_size:
        assert spec.dim_multiple > 0
        assert 0 < spec.min_dim <= spec.max_dim


def test_offload_models_hold_no_idle_vram():
    """cpu_offload означает веса в системной RAM: в покое карта свободна."""
    for spec in MODELS.values():
        if spec.cpu_offload:
            assert spec.weights_vram_mb == 0, spec.id


def test_fixed_schedule_models_are_consistent():
    """Зашитый guidance без зашитых шагов (или наоборот) — признак опечатки в спеке."""
    for spec in MODELS.values():
        if spec.fixed_guidance:
            assert spec.fixed_steps, spec.id
            assert not spec.supports_negative_prompt, spec.id


def test_every_backend_has_a_port_and_a_model():
    for backend in BACKENDS:
        assert backend in DEFAULT_WORKER_PORTS
        assert any(m.backend == backend for m in MODELS.values())


def test_big_card_models_are_not_announced_on_small_cards(monkeypatch):
    """Узел с 24 ГБ не должен объявлять модель класса 96 ГБ, даже если её веса
    лежат на диске (у qwen-image-2.1-full они общие с обычной 2.1)."""
    from images_worker import gpu
    from images_worker.backends.base import fits_this_card
    from images_worker.protocol import GpuInfo

    def card(total_mb):
        return lambda force=False: GpuInfo(index=0, available=True, name="x", total_mb=total_mb,
                                           used_mb=0, free_mb=total_mb)

    full, fp8 = get_spec("qwen-image-2.1-full"), get_spec("qwen-image-2.1")
    monkeypatch.setattr(gpu, "gpu_info", card(24576))
    assert not fits_this_card(full) and fits_this_card(fp8)
    monkeypatch.setattr(gpu, "gpu_info", card(97887))
    assert fits_this_card(full) and fits_this_card(get_spec("minimax-h3-image"))
    monkeypatch.setattr(gpu, "gpu_info", lambda force=False: GpuInfo(index=0, available=False))
    assert not fits_this_card(full)


def test_idle_exit_only_when_cuda_context_exists(monkeypatch):
    """Процесс без CUDA-контекста по простою не выходит — отдавать ему нечего."""
    import sys
    import types

    from images_worker import runner

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert runner._holds_cuda_context() is False

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert runner._holds_cuda_context() is True
    fake.cuda.is_initialized = lambda: False
    assert runner._holds_cuda_context() is False


def test_worker_models_allowlist(monkeypatch):
    """IMG_WORKER_MODELS ограничивает объявляемые модели узла."""
    from images_worker.backends import base

    monkeypatch.setattr(base, "weights_present", lambda spec: True)
    monkeypatch.setattr(base, "fits_this_card", lambda spec: True)

    class Probe(base.Backend):
        name = "qwen21"

    monkeypatch.delenv("IMG_WORKER_MODELS", raising=False)
    assert "qwen-image-2.1" in Probe().supported_models()
    monkeypatch.setenv("IMG_WORKER_MODELS", "qwen-image-2.1-full")
    assert Probe().supported_models() == ["qwen-image-2.1-full"]
