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
    """Ни один пресет не должен требовать больше, чем есть на 3090."""
    spec = get_spec(model_id)
    assert spec.peak_vram_mb < 24576


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


def test_every_backend_has_a_port_and_a_model():
    for backend in BACKENDS:
        assert backend in DEFAULT_WORKER_PORTS
        assert any(m.backend == backend for m in MODELS.values())
