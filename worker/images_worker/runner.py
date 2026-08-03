"""Рантайм воркера: очередь на один GPU-поток, простой-выгрузка, простой-выход.

Разделение обязанностей со шлюзом:
  * шлюз — авторитет. Он знает про оба узла, очередь и соседей по карте,
    и он решает, что и когда грузить/выгружать.
  * воркер — страховка. Всё, что здесь есть, работает и тогда, когда шлюз упал
    или потерял связь: иначе модель осталась бы висеть в VRAM навсегда.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import gpu
from .backends import get_backend
from .protocol import (
    GenerateJob,
    GenerateMeta,
    GenerationCancelled,
    GpuInfo,
    WorkerStatus,
)
from .registry import get_spec
from .settings import WorkerSettings

log = logging.getLogger("images_worker.runner")


class WorkerRunner:
    def __init__(self, settings: WorkerSettings) -> None:
        self.settings = settings
        self.backend = get_backend(settings.backend)
        self.started_at = time.monotonic()
        self.last_error: str | None = None

        # Ровно один поток: все GPU-операции строго последовательны.
        # /status и /cancel при этом остаются отзывчивыми — они в пул не ходят.
        self._gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self._last_activity = time.monotonic()
        self._tasks: list = []

    # --- вспомогательное ---

    def touch(self) -> None:
        self._last_activity = time.monotonic()

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    async def _in_gpu_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._gpu_pool, fn, *args)

    # --- операции ---

    def status(self) -> WorkerStatus:
        return WorkerStatus(
            backend=self.settings.backend,
            node=self.settings.node,
            version=__import__("images_worker").__version__,
            pid=os.getpid(),
            uptime_s=round(time.monotonic() - self.started_at, 1),
            supported_models=self.backend.supported_models(),
            resident_model=self.backend.resident_model,
            loading_model=self.backend.loading_model,
            busy=self.backend.current_task is not None,
            current_task_id=self.backend.current_task,
            gpu=gpu.gpu_info(),
            last_error=self.last_error,
        )

    async def load(self, model_id: str) -> tuple[bool, float, GpuInfo]:
        self.touch()
        try:
            already, elapsed = await self._in_gpu_thread(self.backend.load, model_id)
            self.last_error = None
        except Exception as exc:
            self.last_error = "load %s: %s" % (model_id, exc)
            raise
        gpu.invalidate()
        return already, elapsed, gpu.gpu_info(force=True)

    async def unload(self) -> tuple[str | None, GpuInfo]:
        self.touch()
        was = await self._in_gpu_thread(self.backend.unload)
        gpu.invalidate()
        return was, gpu.gpu_info(force=True)

    async def generate(self, job: GenerateJob) -> tuple[bytes, GenerateMeta]:
        self.touch()
        spec = get_spec(job.model_id)

        load_time = 0.0
        if self.backend.resident_model != job.model_id:
            _already, load_time, _gpu = await self.load(job.model_id)

        def _run():
            return self.backend.generate(job)

        try:
            image, seed_used, gen_time = await self._in_gpu_thread(_run)
        except GenerationCancelled:
            # Ловим явно: BaseException обычным `except Exception` не поймается.
            self.touch()
            raise
        except Exception as exc:
            self.last_error = "generate %s: %s" % (job.model_id, exc)
            raise
        finally:
            self.touch()
            gpu.invalidate()

        png = await asyncio.get_running_loop().run_in_executor(None, _encode_png, image)

        # Размер берём у самой картинки, а не из запроса. HiDream примагничивает
        # произвольный размер к ближайшему нативному (запрос 1024x1024 отдаёт
        # 2048x2048), и API не должен сообщать то, чего не отдавал.
        actual_width, actual_height = image.size
        if (actual_width, actual_height) != (job.width, job.height):
            log.info(
                "size adjusted by pipeline: requested %dx%d, produced %dx%d",
                job.width,
                job.height,
                actual_width,
                actual_height,
            )

        meta = GenerateMeta(
            task_id=job.task_id,
            model_id=job.model_id,
            backend=self.settings.backend,
            node=self.settings.node,
            width=actual_width,
            height=actual_height,
            requested_width=job.width,
            requested_height=job.height,
            num_steps=job.num_inference_steps,
            guidance_scale=job.guidance_scale,
            shift=job.shift if spec.default_shift is not None else None,
            scheduler=job.scheduler if spec.schedulers else None,
            seed_used=seed_used,
            generation_time_seconds=round(gen_time, 2),
            load_time_seconds=round(load_time, 2),
        )
        self.last_error = None
        return png, meta

    def cancel(self, task_id: str) -> bool:
        return self.backend.cancel(task_id)

    # --- фоновые сторожа ---

    async def _idle_unloader(self) -> None:
        """Выгрузить модель после простоя.

        Приём взят из hidream_o1_api/queue.py::_idle_unloader — единственного
        места во всех четырёх сервисах, где VRAM освобождалась сама.
        """
        threshold = self.settings.idle_unload_s
        if threshold <= 0:
            return
        interval = max(5.0, min(threshold / 4.0, 30.0))
        while True:
            await asyncio.sleep(interval)
            try:
                if (
                    self.backend.current_task is None
                    and self.backend.resident_model is not None
                    and self.backend.loading_model is None
                    and self.idle_seconds >= threshold
                ):
                    log.info("idle %.0fs — unloading %s", self.idle_seconds, self.backend.resident_model)
                    await self.unload()
            except Exception as exc:  # pragma: no cover
                log.warning("idle unloader: %s", exc)

    async def _idle_exiter(self) -> None:
        """Выйти из процесса после долгого простоя.

        Даже с выгруженной моделью процесс держит CUDA-контекст (~340 МБ —
        столько сейчас висит за простаивающим z-image). Четыре бездействующих
        воркера — это больше гигабайта VRAM впустую. Единственный способ вернуть
        её полностью — умереть; systemd поднимет обратно за пару секунд.
        """
        threshold = self.settings.idle_exit_s
        if threshold <= 0:
            return
        while True:
            await asyncio.sleep(30.0)
            if (
                self.backend.current_task is None
                and self.backend.resident_model is None
                and self.backend.loading_model is None
                and self.idle_seconds >= threshold
            ):
                log.info("idle %.0fs with no model — exiting to release CUDA context", self.idle_seconds)
                # _exit, а не sys.exit: обычный выход ждал бы GPU-пул.
                os._exit(0)

    def start_background(self) -> None:
        self._tasks = [
            asyncio.create_task(self._idle_unloader(), name="idle-unloader"),
            asyncio.create_task(self._idle_exiter(), name="idle-exiter"),
        ]

    async def stop_background(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks = []
        # Выгружаем при штатной остановке — чтобы после `systemctl stop`
        # за процессом не оставалось занятой памяти.
        try:
            self.backend.unload()
        except Exception as exc:  # pragma: no cover
            log.warning("unload on shutdown: %s", exc)
        self._gpu_pool.shutdown(wait=False, cancel_futures=True)


def _encode_png(image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
