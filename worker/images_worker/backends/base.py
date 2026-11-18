"""Базовый класс бэкенда: жизненный цикл модели, выгрузка, отмена.

Главное правило воркера: **на процесс — одна резидентная модель**. Старый
z-image держал turbo и base одновременно, что на 24 ГБ работало только потому,
что обе с cpu-offload и в покое VRAM почти не занимают. Здесь резидентность
единственна всегда — это делает учёт честным и предсказуемым.
"""

from __future__ import annotations

import gc
import glob
import logging
import os
import threading
import time

from ..protocol import GenerateJob, GenerationCancelled
from ..registry import ModelSpec, get_spec, models_for_backend

log = logging.getLogger("images_worker.backend")


def _hf_cache_dir() -> str:
    """Каталог кэша HuggingFace с учётом обеих переменных окружения."""
    cache = os.environ.get("HF_HUB_CACHE", "").strip()
    if cache:
        return cache
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        return os.path.join(home, "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def weights_present(spec: ModelSpec) -> bool:
    """Есть ли веса модели на этом хосте — без обращения в сеть.

    Нужно для `supported_models` в /status: шлюз строит из этого таблицу
    node_models, и «на node-b есть hidream, но нет ernie» получается само,
    без единой строчки конфигурации.
    """
    if spec.local_path:
        if not os.path.isdir(spec.local_path):
            return False
        return bool(
            glob.glob(os.path.join(spec.local_path, "*.safetensors"))
            or glob.glob(os.path.join(spec.local_path, "*.bin"))
        )

    if not spec.repo_id:
        return False

    repo_dir = os.path.join(_hf_cache_dir(), "models--" + spec.repo_id.replace("/", "--"))
    snapshots = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return False

    # В кэше могут лежать одни refs без блобов (так выглядит прерванная закачка),
    # поэтому проверяем, что в снапшоте реально что-то есть.
    for root, _dirs, files in os.walk(snapshots):
        for name in files:
            path = os.path.join(root, name)
            try:
                if os.path.getsize(os.path.realpath(path)) > 1024 * 1024:
                    return True
            except OSError:
                continue
    return False


def _allowed_models() -> set[str] | None:
    """Белый список моделей узла из IMG_WORKER_MODELS (через запятую) или None.

    Нужен, когда у одного бэкенда есть модели с разной ценой для хоста. На узле с
    большой картой qwen21 должен объявлять только qwen-image-2.1-full: остальные
    варианты держат текстовый энкодер в системной RAM (десятки ГБ), а её там
    обещано не трогать — хотя веса у всех вариантов общие и лежат на диске.
    """
    raw = os.environ.get("IMG_WORKER_MODELS", "").strip()
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def fits_this_card(spec: ModelSpec) -> bool:
    """Способна ли карта этого узла вообще вместить модель.

    Обычные модели рассчитаны на 24 ГБ и проходят всегда. Модели большого класса
    (card_class_mb выше) узел объявляет, только если его карта вмещает их пик:
    иначе шлюз считал бы узел кандидатом, а задача ждала бы на нём VRAM, которой
    там не бывает. Без показаний nvidia-smi такие модели не объявляем.
    """
    if spec.card_class_mb <= 24576:
        return True
    from .. import gpu

    info = gpu.gpu_info()
    return bool(info.available and info.total_mb >= spec.peak_vram_mb)


class Backend:
    """Общий жизненный цикл. Наследники реализуют _load_model и _generate."""

    name: str = "base"

    def __init__(self) -> None:
        self._pipe = None
        self._resident: str | None = None
        self._loading: str | None = None
        self._cancel = threading.Event()
        self._current_task: str | None = None
        self._supported_cache: list | None = None
        self._supported_at: float = 0.0
        # Сериализует load/unload/generate внутри процесса. Планировщик и так
        # шлёт по одной задаче на узел, но защищаемся от гонки с idle-выгрузкой.
        self.lock = threading.RLock()

    # --- состояние ---

    @property
    def resident_model(self) -> str | None:
        return self._resident

    @property
    def loading_model(self) -> str | None:
        return self._loading

    @property
    def current_task(self) -> str | None:
        return self._current_task

    # Наличие весов проверяется обходом каталога кэша HuggingFace — недёшево,
    # а /status планировщик дёргает каждые несколько секунд. Кэшируем: веса
    # не появляются и не исчезают между двумя опросами.
    _SUPPORTED_TTL_S = 300.0

    def supported_models(self) -> list:
        now = time.monotonic()
        if self._supported_cache is not None and (now - self._supported_at) < self._SUPPORTED_TTL_S:
            return list(self._supported_cache)

        allowed = _allowed_models()
        found = [
            m
            for m in models_for_backend(self.name)
            if (allowed is None or m in allowed)
            and weights_present(get_spec(m))
            and fits_this_card(get_spec(m))
        ]
        self._supported_cache = found
        self._supported_at = now
        return list(found)

    def invalidate_supported(self) -> None:
        """Сбросить кэш — после докачки весов список меняется."""
        self._supported_cache = None

    # --- жизненный цикл ---

    def load(self, model_id: str) -> tuple[bool, float]:
        """Сделать модель резидентной. Возвращает (была_ли_уже_загружена, секунды)."""
        spec = get_spec(model_id)
        if spec.backend != self.name:
            raise ValueError("model %s belongs to backend %s, not %s" % (model_id, spec.backend, self.name))

        with self.lock:
            if self._resident == model_id:
                return True, 0.0

            # Одна резидентная модель на процесс: перед загрузкой новой — выгрузить старую.
            if self._resident is not None:
                log.info("evicting %s before loading %s", self._resident, model_id)
                self._unload_locked()

            self._loading = model_id
            started = time.monotonic()
            try:
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                self._pipe = self._load_model(spec)
                self._resident = model_id
            except Exception:
                # В отличие от старых сервисов, которые глотали исключение и
                # оставляли «модель не загружена» всплывать позже из generate(),
                # здесь ошибка загрузки видна сразу и приводит к чистому состоянию.
                self._pipe = None
                self._resident = None
                self._release_memory()
                raise
            finally:
                self._loading = None

            elapsed = time.monotonic() - started
            log.info("loaded %s in %.1fs", model_id, elapsed)
            return False, elapsed

    def unload(self) -> str | None:
        with self.lock:
            return self._unload_locked()

    def _unload_locked(self) -> str | None:
        if self._pipe is None and self._resident is None:
            return None
        was = self._resident
        self._pipe = None
        self._resident = None
        self._release_memory()
        log.info("unloaded %s", was)
        return was

    def _release_memory(self) -> None:
        """Освобождение памяти — дословно приём из HiDream, лучший из четырёх сервисов.

        ipc_collect() важен: без него страницы, засвеченные другим процессам,
        остаются висеть за картой.
        """
        gc.collect()
        try:
            import torch

            # is_initialized, а не is_available: ipc_collect() начинается с ленивой
            # инициализации CUDA и создал бы контекст (~340 МБ на карте) в процессе,
            # который карту ещё не трогал, — ровно то, чего мы здесь избегаем.
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:  # pragma: no cover
            log.warning("cuda cleanup failed: %s", exc)

    # --- генерация ---

    def generate(self, job: GenerateJob):
        """Синхронная генерация. Возвращает (PIL.Image, seed_used, seconds)."""
        with self.lock:
            if self._resident != job.model_id:
                raise RuntimeError("model %s is not resident (current: %s)" % (job.model_id, self._resident))

            self._cancel.clear()
            self._current_task = job.task_id
            spec = get_spec(job.model_id)
            started = time.monotonic()
            try:
                image, seed_used = self._generate(spec, job)
            finally:
                self._current_task = None
                self._cancel.clear()
                # Промежуточные тензоры отпускаем всегда, в том числе после OOM:
                # иначе следующая задача упрётся в мусор от предыдущей.
                self._release_memory()

            return image, seed_used, time.monotonic() - started

    def cancel(self, task_id: str) -> bool:
        """Пометить задачу к отмене. Реальный выход — из step-callback'а."""
        if self._current_task == task_id:
            self._cancel.set()
            return True
        return False

    def _raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise GenerationCancelled("cancelled by gateway")

    def _step_callback(self):
        """Callback для diffusers: единственная точка, где генерация умеет прерваться.

        Сигнатура `callback_on_step_end(pipe, step, timestep, kwargs)` есть у
        Qwen/Z-Image/ERNIE. У модульных пайплайнов (flux3) и у moss его нет — там отмена
        встроена в progress_bar цикла денойза.
        """

        def _cb(_pipe, _step, _timestep, callback_kwargs):
            self._raise_if_cancelled()
            return callback_kwargs

        return _cb

    def _seed(self, job: GenerateJob) -> int:
        if job.seed is not None:
            return job.seed
        import torch

        return int(torch.randint(0, 2**32 - 1, (1,)).item())

    def _generator(self, seed: int):
        """Генератор всегда на CPU.

        При enable_model_cpu_offload() модули переезжают между устройствами, и
        CUDA-генератор даёт невоспроизводимые результаты. Все четыре исходных
        сервиса делали именно так.
        """
        import torch

        return torch.Generator(device="cpu").manual_seed(seed)

    # --- реализуют наследники ---

    def _load_model(self, spec: ModelSpec):
        raise NotImplementedError

    def _generate(self, spec: ModelSpec, job: GenerateJob):
        raise NotImplementedError
