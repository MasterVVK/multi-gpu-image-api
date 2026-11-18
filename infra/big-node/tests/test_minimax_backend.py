"""Структурные тесты бэкенда minimax на CPU.

Что здесь настоящее: блоки `MiniMaxH3ImageBlocks`, классы трансформера, VAE, энкодера Qwen3-VL,
токенизатор и процессор из конфигов модели, загрузка с диска из кэша «только на чтение», офлайн.
Что поддельное: размеры (крошечные конфиги) и значения весов (случайные). То есть проверяется
устройство бэкенда — порядок стадий, имена состояния, освобождение, отмена, — но не качество
картинки, не VRAM и не время.
"""

from __future__ import annotations

import os
import shutil
import threading
import time

import pytest
from conftest import REPO_DIR, make_job

TEXT = ("text_encoder", "tokenizer", "processor")
IMAGE = ("transformer", "scheduler", "vae")


# --- устройство пайплайна ---------------------------------------------------


def test_blocks_and_components_match_backend_lists(backend):
    """Имена блоков, входов/выходов и компонентов, на которые опирается бэкенд, существуют."""
    from diffusers import MiniMaxH3ImageBlocks
    from diffusers.image_processor import VaeImageProcessor

    assert list(MiniMaxH3ImageBlocks().sub_blocks) == ["before_encode", "text_encoder", "vae_encoder", "denoise", "decode"]

    stages = backend._pipe
    assert {p.name for p in stages.text_blocks.inputs} >= {"prompt"}
    assert {p.name for p in stages.text_blocks.intermediate_outputs} == {"prompt_embeds", "text_token_tags"}
    image_inputs = {p.name for p in stages.image_blocks.inputs}
    assert image_inputs >= {"prompt_embeds", "text_token_tags", "height", "width", "generator"}
    assert "images" in {p.name for p in stages.image_blocks.outputs}
    # У модели нет ни шагов, ни CFG, ни негатива — таких входов не существует вовсе.
    assert not image_inputs & {"num_inference_steps", "guidance_scale", "negative_prompt", "negative_prompt_embeds"}

    text_pipe = stages.text_blocks.init_pipeline(stages.root)
    assert set(text_pipe.components) == set(TEXT)
    assert all(component is None for component in text_pipe.components.values())

    image_pipe = stages.image_blocks.init_pipeline(stages.root)
    assert set(image_pipe.components) == set(IMAGE) | {"image_processor"}
    # image_processor не грузится с диска: он создаётся из конфига уже при init_pipeline.
    assert isinstance(image_pipe.image_processor, VaeImageProcessor)
    assert image_pipe.get_component_spec("image_processor").default_creation_method == "from_config"
    assert all(image_pipe.components[name] is None for name in IMAGE)
    # Сетка сигм приехала из modular_model_index.json снапшота, а не из дефолта блока.
    assert len(image_pipe.config.sample_sigmas) == 7


def test_load_is_cheap_and_touches_no_weights(tap):
    """/load только разбирает блоки: ни один компонент не грузится, CUDA не трогается."""
    import torch
    from images_worker.backends import get_backend

    backend = get_backend("minimax")
    already, seconds = backend.load("minimax-h3-image")
    assert already is False and seconds < 30
    assert backend.resident_model == "minimax-h3-image"
    assert tap.events == []
    assert not torch.cuda.is_initialized()
    assert backend._pipe.root == str(REPO_DIR / "snapshots" / "8ced60068623def5b74b4d10558117cb61d6482d")


def test_load_and_unload_never_initialize_cuda(monkeypatch):
    """Простаивающий воркер не заводит CUDA-контекст — даже на машине, где карта есть.

    Базовый `_release_memory` зовёт torch.cuda.ipc_collect(), который начинается с
    `_lazy_init()`; бэкенд обязан обходить его, пока сам не работал на карте.
    """
    import torch
    from images_worker.backends import get_backend

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: calls.append("ipc_collect"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: calls.append("_lazy_init"))

    backend = get_backend("minimax")
    backend.load("minimax-h3-image")
    assert backend.unload() == "minimax-h3-image"
    assert backend.resident_model is None
    assert calls == []


# --- полный проход -----------------------------------------------------------


def test_generate_runs_stages_in_order_and_frees_them(backend, tap):
    image, seed, seconds = backend.generate(make_job())

    assert image.size == (1024, 1024) and image.mode == "RGB"
    assert seed == 7

    # Порядок: энкодер загружен -> перенесён -> отработал -> выгружен -> и только потом трансформер с VAE.
    assert tap.names() == (
        ["load", "loaded", "to", "text_encoder", "unload"]
        + ["load", "loaded", "to"]
        + ["transformer"] * 6
        + ["vae.decode", "unload"]
    )
    load_text, load_image = [e for e in tap.events if e[0] == "load"]
    assert load_text[1] == TEXT and load_text[2] == ()
    # Главное требование: к началу загрузки второй стадии от первой в памяти нет ничего.
    assert load_image[1] == IMAGE and load_image[2] == ()
    loaded_text, loaded_image = [e for e in tap.events if e[0] == "loaded"]
    assert loaded_text[1] == tuple(sorted(TEXT))
    assert loaded_image[1] == tuple(sorted(IMAGE + ("image_processor",)))
    assert [e[1] for e in tap.events if e[0] == "transformer"] == [0, 1, 2, 3, 4, 5]
    # Запрос 1024x1024 считается на холсте 4 Мп (2048x2048 -> латент 128x128) и уменьшается
    # обратно — это путь через image_processor.resize.
    assert [e[1] for e in tap.events if e[0] == "vae.decode"] == [(1, 4, 128, 128)]
    assert [e[1] for e in tap.events if e[0] == "to"] == ["cpu", "cpu"]
    assert all(kwargs == {"dtype": __import__("torch").bfloat16} for kwargs in tap.load_kwargs)

    # После задачи не живёт ни один тяжёлый компонент.
    assert tap.alive() == []
    assert backend.current_task is None
    # Без CUDA-контекста таймер выхода из процесса не взводится.
    assert not [t for t in threading.enumerate() if t.name == "minimax-exit"]


def test_same_seed_same_image_and_native_size(backend, tap):
    """Сид воспроизводим; размер без уменьшения (холст = запрос) тоже проходит."""
    job = make_job(width=2048, height=2080, seed=123)
    first, seed_a, _ = backend.generate(job)
    second, seed_b, _ = backend.generate(make_job(task_id="t2", width=2048, height=2080, seed=123))
    assert first.size == (2048, 2080)
    assert seed_a == seed_b == 123
    assert first.tobytes() == second.tobytes()
    assert tap.alive() == []


def test_placement_device_map(monkeypatch, tap):
    """Запасной способ размещения: веса читаются сразу на устройство, `pipe.to()` не вызывается."""
    import torch
    from images_worker.backends import get_backend

    monkeypatch.setenv("IMG_MINIMAX_PLACEMENT", "device_map")
    backend = get_backend("minimax")
    backend.load("minimax-h3-image")
    image, _, _ = backend.generate(make_job())

    assert image.size == (1024, 1024)
    assert "to" not in tap.names()
    assert tap.load_kwargs == [
        {"dtype": torch.bfloat16, "device_map": {"text_encoder": "cpu"}},
        {"dtype": torch.bfloat16, "device_map": {"transformer": "cpu", "vae": "cpu"}},
    ]
    assert tap.alive() == []


# --- отмена ---------------------------------------------------------------------


def test_cancel_between_stages(backend, tap):
    """Отмена, пришедшая во время кодирования текста: вторая стадия не начинается."""
    from images_worker.protocol import GenerationCancelled

    tap.on("text_encoder", lambda _n: backend.cancel("t1"))
    with pytest.raises(GenerationCancelled):
        backend.generate(make_job())
        pytest.fail("generation was not cancelled")

    assert tap.names() == ["load", "loaded", "to", "text_encoder", "unload"]
    assert tap.alive() == []
    assert backend.current_task is None


def test_cancel_inside_denoise_stops_at_next_step(backend, tap):
    """Отмена во время шага денойза: цикл прерывается перед следующим шагом, декода нет."""
    from images_worker.protocol import GenerationCancelled

    alive_when_caught = None

    def cancel_on_second_step(step):
        if step == 1:
            assert backend.cancel("t1") is True

    tap.on("transformer", cancel_on_second_step)
    try:
        backend.generate(make_job())
        pytest.fail("generation was not cancelled")
    except GenerationCancelled:
        # Исключение ещё живо — а веса уже отпущены: трассировка их не держит.
        alive_when_caught = tap.alive()

    assert [e[1] for e in tap.events if e[0] == "transformer"] == [0, 1]
    assert "vae.decode" not in tap.names()
    assert tap.names()[-1] == "unload"
    assert alive_when_caught == []


def test_cancel_before_decode(backend, tap):
    """Отмена на последнем шаге ловится точкой перед декодом VAE."""
    from images_worker.protocol import GenerationCancelled

    tap.on("transformer", lambda step: backend.cancel("t1") if step == 5 else None)
    with pytest.raises(GenerationCancelled):
        backend.generate(make_job())

    assert len([e for e in tap.events if e[0] == "transformer"]) == 6
    assert "vae.decode" not in tap.names()
    assert tap.alive() == []


def test_cancel_of_another_task_is_ignored(backend, tap):
    tap.on("text_encoder", lambda _n: backend.cancel("someone-else"))
    image, _, _ = backend.generate(make_job())
    assert image.size == (1024, 1024)


# --- ошибки ---------------------------------------------------------------------


def test_failure_in_decode_releases_weights_while_exception_is_alive(backend, tap, monkeypatch):
    """Падение внутри модели: к моменту, когда исключение дошло до вызывающего, веса отпущены."""
    from diffusers import AutoencoderKLMiniMaxH3Image
    from images_worker.app import _is_oom

    def boom(self, z, *args, **kwargs):
        activations = z.clone()  # локальная переменная кадра, как активации настоящей модели
        raise RuntimeError("CUDA out of memory (simulated), tried to allocate %d" % activations.numel())

    monkeypatch.setattr(AutoencoderKLMiniMaxH3Image, "decode", boom)
    try:
        backend.generate(make_job())
        pytest.fail("generation did not fail")
    except RuntimeError as exc:
        assert tap.alive() == []
        assert _is_oom(exc)  # шлюз получит vram_oom
        assert exc.__traceback__ is not None  # трассировка для журнала на месте

    # Следующая задача после сбоя проходит как обычно.
    monkeypatch.undo()


def test_component_that_failed_to_load_is_an_error_with_the_reason(backend, monkeypatch):
    """diffusers проглатывает ошибку загрузки компонента; бэкенд обязан поднять её с причиной."""
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec
    from images_worker.app import _is_oom

    real_load = ComponentSpec.load

    def load(self, **kwargs):
        if self.name == "text_encoder":
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return real_load(self, **kwargs)

    monkeypatch.setattr(ComponentSpec, "load", load)
    with pytest.raises(RuntimeError, match="failed to load text_encoder") as info:
        backend.generate(make_job())
    assert _is_oom(info.value)


def test_backend_recovers_after_failure(backend, tap, monkeypatch):
    from diffusers import AutoencoderKLMiniMaxH3Image

    real_decode = AutoencoderKLMiniMaxH3Image.decode
    calls = {"n": 0}

    def flaky(self, z, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real_decode(self, z, *args, **kwargs)

    monkeypatch.setattr(AutoencoderKLMiniMaxH3Image, "decode", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        backend.generate(make_job())
    image, _, _ = backend.generate(make_job(task_id="t2"))
    assert image.size == (1024, 1024)
    assert tap.alive() == []


# --- поиск снапшота и офлайн -------------------------------------------------------


def test_resolve_snapshot(tmp_path):
    from images_worker.backends.minimax import resolve_snapshot
    from images_worker.registry import get_spec

    spec = get_spec("minimax-h3-image")
    assert resolve_snapshot(spec).endswith("/snapshots/8ced60068623def5b74b4d10558117cb61d6482d")

    # Без refs/main берётся единственный снапшот; без снапшотов — внятная ошибка.
    cache = tmp_path / "hub"
    snap = cache / "models--MiniMaxAI--MiniMax-H3-Image" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "modular_model_index.json").write_text("{}")
    old = os.environ["HF_HUB_CACHE"]
    os.environ["HF_HUB_CACHE"] = str(cache)
    try:
        assert resolve_snapshot(spec) == str(snap)
        shutil.rmtree(snap)
        with pytest.raises(FileNotFoundError):
            resolve_snapshot(spec)
    finally:
        os.environ["HF_HUB_CACHE"] = old


def test_repo_id_offline_cannot_load_processor(backend):
    """Почему бэкенд отдаёт пайплайну каталог, а не repo_id.

    С repo_id и HF_HUB_OFFLINE=1 процессор Qwen3-VL не грузится: transformers ищет
    необязательный processor/config.json и без сети не может узнать, что его нет.
    Если этот тест начнёт падать — upstream починили, и обход можно пересмотреть.
    """
    stages = backend._pipe
    by_repo = stages.text_blocks.init_pipeline("MiniMaxAI/MiniMax-H3-Image")
    by_repo.load_components(names=["processor", "tokenizer"])
    assert by_repo.tokenizer is not None
    assert by_repo.processor is None

    by_path = stages.text_blocks.init_pipeline(stages.root)
    by_path.load_components(names=["processor", "tokenizer"])
    assert by_path.processor is not None and by_path.tokenizer is not None


# --- выход из процесса после задачи --------------------------------------------------


@pytest.fixture()
def exits(monkeypatch):
    """Подменить os._exit и притвориться, что CUDA-контекст создан."""
    import torch
    from images_worker.backends import minimax

    recorded = []
    monkeypatch.setattr(minimax.os, "_exit", lambda code: recorded.append(code))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setenv("IMG_MINIMAX_EXIT_AFTER_S", "0.3")
    return recorded


def test_exit_fires_after_task_when_idle(exits):
    from images_worker.backends import get_backend

    backend = get_backend("minimax")
    backend._generation += 1
    backend._arm_exit()
    time.sleep(0.8)
    assert exits == [0]


def test_exit_is_dropped_when_next_task_starts(exits):
    from images_worker.backends import get_backend

    backend = get_backend("minimax")
    backend._generation += 1
    backend._arm_exit()
    backend.load("minimax-h3-image")  # шлюз начал следующую задачу
    time.sleep(0.8)
    assert exits == []


def test_exit_waits_for_running_generation(exits):
    from images_worker.backends import get_backend

    backend = get_backend("minimax")
    backend._generation += 1
    backend._arm_exit()
    with backend.lock:  # идёт генерация
        backend._current_task = "t9"
        time.sleep(0.8)
        assert exits == []
        backend._generation += 1  # generate() увеличивает счётчик
        backend._current_task = None
    time.sleep(0.3)
    assert exits == []


def test_exit_disabled_by_zero(exits, monkeypatch):
    from images_worker.backends import get_backend

    monkeypatch.setenv("IMG_MINIMAX_EXIT_AFTER_S", "0")
    backend = get_backend("minimax")
    backend._arm_exit()
    time.sleep(0.5)
    assert exits == []
    assert not [t for t in threading.enumerate() if t.name == "minimax-exit"]
