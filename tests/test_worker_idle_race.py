"""Гонка сторожа простоя воркера с генерацией (images_worker/runner.py).

24.11.2026 на большой карте: z-image-base-full грузилась 61.8 с при пороге простоя 15 с.
Как только модель стала резидентной, сторож увидел «простой 62 с» и поставил выгрузку в
GPU-поток раньше генерации — задача упала с «model … is not resident (current: None)».
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from images_worker import runner as runner_mod
from images_worker.protocol import GenerateJob
from PIL import Image

MODEL = "z-image-base-full"


class FakeBackend:
    def __init__(self, load_seconds: float = 0.05) -> None:
        self.resident_model = None
        self.loading_model = None
        self.current_task = None
        self.loads = 0
        self.load_seconds = load_seconds

    def load(self, model_id):
        time.sleep(self.load_seconds)
        self.resident_model = model_id
        self.loads += 1
        return False, self.load_seconds

    def unload(self):
        was, self.resident_model = self.resident_model, None
        return was

    def generate(self, job):
        if self.resident_model != job.model_id:
            raise RuntimeError(
                "model %s is not resident (current: %s)" % (job.model_id, self.resident_model)
            )
        return Image.new("RGB", (job.width, job.height)), 1, 0.01


def make_runner(backend: FakeBackend) -> runner_mod.WorkerRunner:
    runner = object.__new__(runner_mod.WorkerRunner)
    runner.settings = SimpleNamespace(backend="zimage", node="test")
    runner.backend = backend
    runner.started_at = time.monotonic()
    runner.last_error = None
    runner._gpu_pool = ThreadPoolExecutor(max_workers=1)
    runner._last_activity = time.monotonic() - 3600  # давно простаивал
    runner._tasks = []
    return runner


def job() -> GenerateJob:
    return GenerateJob(
        task_id="t1",
        model_id=MODEL,
        prompt="p",
        width=8,
        height=8,
        num_inference_steps=1,
        guidance_scale=1.0,
    )


async def test_unload_between_load_and_generate_is_survived(monkeypatch):
    backend = FakeBackend()
    runner = make_runner(backend)
    # Выгрузка встаёт в GPU-поток сразу после загрузки — ровно туда, где 24.11 её
    # поставил сторож простоя, — и выполняется раньше генерации.
    monkeypatch.setattr(runner_mod.gpu, "invalidate", lambda: None)
    monkeypatch.setattr(
        runner_mod.gpu, "gpu_info", lambda force=False: runner._gpu_pool.submit(backend.unload)
    )

    png, meta = await runner.generate(job())

    assert png.startswith(b"\x89PNG")
    assert meta.model_id == MODEL
    assert backend.loads == 2


async def test_idle_is_counted_from_the_end_of_loading(monkeypatch):
    backend = FakeBackend(load_seconds=0.3)
    runner = make_runner(backend)
    monkeypatch.setattr(runner_mod.gpu, "invalidate", lambda: None)
    monkeypatch.setattr(runner_mod.gpu, "gpu_info", lambda force=False: None)

    await runner.load(MODEL)

    assert runner.idle_seconds < 0.25
