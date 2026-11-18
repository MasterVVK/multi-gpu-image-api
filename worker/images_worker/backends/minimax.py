"""Бэкенд MiniMax-H3-Image: 33B DiT + 33B текстовый энкодер, 16 Мп за 6 шагов.

Модель не для 24 ГБ: и трансформер, и энкодер Qwen3-VL весят по ~62 ГБ в bf16.
Узел под неё — карта на 96 ГБ, причём чужая и боевая, отсюда правила, которыми
этот бэкенд отличается от остальных.

1. Резидентной модели нет. `_load_model` только разбирает блоки пайплайна и
   находит снапшот в кэше (секунды, ноль VRAM, CUDA не трогается), а оба
   тяжёлых компонента приезжают на карту внутри `_generate` и уезжают до его
   возврата. «После картинки VRAM свободна» выполняется по построению, а не по
   таймеру простоя.

2. Тяжёлые компоненты живут на карте по очереди, а не вместе: энкодер ->
   выгрузка -> трансформер + VAE -> выгрузка. Вместе они (~124 ГБ) не входят
   даже в 96 ГБ, а штатный auto-offload diffusers держит обе копии в системной
   RAM — на хосте со 125 ГБ и боевыми базами это недопустимо. Цена — чтение
   ~130 ГБ с диска на каждую картинку. Веса при этом не копируются в анонимную
   память: safetensors отображает файл (mmap), страницы файловые и вытесняемые
   (замерено на CPU: RssAnon не растёт), так что лимит памяти контейнера
   ограничивает и их.

3. Пайплайн только модульный (`MiniMaxH3ImageBlocks`): классического
   `...Pipeline.from_pretrained` у модели в diffusers нет. Блоки режутся на
   две под-цепочки — кодирование текста и «денойз + декод», — у каждой свой
   набор компонентов. `image_processor` в списках нет намеренно: он создаётся
   из конфига при `init_pipeline` (default_creation_method="from_config").

4. Пайплайну отдаётся каталог снапшота, а не repo_id. С repo_id и
   HF_HUB_OFFLINE=1 процессор Qwen3-VL не грузится вовсе: transformers ищет
   необязательный `processor/config.json`, без сети не может узнать, что его в
   репозитории нет, и падает (онлайн он записал бы в кэш маркер `.no_exist` —
   а кэш смонтирован только на чтение). С локальным каталогом отсутствие файла
   — просто отсутствие файла: ни сети, ни записи в кэш.

5. После задачи процесс завершается (см. `_arm_exit`). Выгрузка возвращает
   веса, но CUDA-контекст (сотни МБ) torch отдать не умеет — только смерть
   процесса. Контейнер поднимается обратно политикой перезапуска, и новый
   процесс до первой задачи карту не трогает совсем.

Шагов и CFG у модели нет: 6 шагов и сетка сигм зашиты в чекпойнт, guidance
дистиллирован, negative_prompt не существует. Значения из запроса не берутся.

Отмена проверяется перед каждой стадией, после загрузки компонентов, перед
каждым из шести шагов денойза и перед декодом — через собственный блок,
вставленный в цепочку (штатный механизм модульных пайплайнов: у них нет ни
`callback_on_step_end`, ни флага `interrupt`). Внутри `from_pretrained` и
внутри одного шага отмена не действует: чтение 66 ГБ с диска прервать нечем.

Переменные окружения:
  IMG_MINIMAX_DEVICE        устройство, по умолчанию `cuda` (в тестах — `cpu`);
  IMG_MINIMAX_PLACEMENT     `to` (по умолчанию): загрузка и затем `pipe.to(device)`;
                            `device_map`: `load_components(device_map=...)`, веса
                            читаются сразу на карту. Запасной путь на случай
                            сюрпризов первого;
  IMG_MINIMAX_EXIT_AFTER_S  через сколько секунд после задачи выйти из процесса,
                            по умолчанию 120 (ответ с картинкой в 40-50 МБ должен
                            успеть уйти через туннель); 0 — не выходить.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import threading
import time
import traceback

import torch

from ..protocol import GenerateJob
from ..registry import ModelSpec
from .base import Backend, _hf_cache_dir

log = logging.getLogger("images_worker.minimax")

_TEXT_COMPONENTS = ("text_encoder", "tokenizer", "processor")
_IMAGE_COMPONENTS = ("transformer", "scheduler", "vae")
# Компоненты с весами: только им нужен device_map.
_MODULES = ("text_encoder", "transformer", "vae")
_DTYPE = torch.bfloat16
_NUM_STEPS = 6
_MIB = 1024 * 1024
# Логгер, которым diffusers сообщает о компоненте, который не удалось загрузить.
_DIFFUSERS_LOADER_LOG = "diffusers.modular_pipelines.modular_pipeline"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _free_cuda() -> None:
    gc.collect()
    # is_initialized, а не is_available: процесс, который карту ещё не трогал, не должен
    # заводить на ней контекст ради уборки.
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _cuda_allocated_mb() -> int:
    """Сколько torch держит на карте живыми тензорами. Без CUDA-контекста — ноль, и контекст не создаётся."""
    if not torch.cuda.is_initialized():
        return 0
    return int(torch.cuda.memory_allocated() // _MIB)


def _drop_frames(exc: BaseException) -> None:
    """Отпустить локальные переменные завершившихся кадров из трассировки исключения.

    Пока исключение летит наверх, его трассировка держит кадры diffusers и самой
    модели, а те — пайплайн, веса и активации. `empty_cache()` в `finally` при этом
    ничего не вернул бы: память занята живыми тензорами. Сама трассировка (файлы,
    строки) остаётся — в журнал она попадёт целиком.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__


def resolve_snapshot(spec: ModelSpec) -> str:
    """Каталог с файлами модели: `local_path` из реестра или снапшот в кэше HuggingFace.

    Ни сети, ни записи: читается `refs/main` и проверяется, что каталог на месте.
    """
    if spec.local_path:
        root = spec.local_path
    else:
        repo_dir = os.path.join(_hf_cache_dir(), "models--" + (spec.repo_id or "").replace("/", "--"))
        snapshots = os.path.join(repo_dir, "snapshots")
        revision = None
        try:
            with open(os.path.join(repo_dir, "refs", "main")) as fh:
                revision = fh.read().strip()
        except OSError:
            pass
        if not revision or not os.path.isdir(os.path.join(snapshots, revision)):
            found = sorted(os.listdir(snapshots)) if os.path.isdir(snapshots) else []
            if len(found) != 1:
                raise FileNotFoundError(
                    "no usable snapshot of %s in %s (refs/main=%r, snapshots=%s)"
                    % (spec.repo_id, repo_dir, revision, found)
                )
            revision = found[0]
        root = os.path.join(snapshots, revision)

    if not os.path.isfile(os.path.join(root, "modular_model_index.json")):
        raise FileNotFoundError("modular_model_index.json not found in %s" % root)
    return root


class _LoadFailures(logging.Handler):
    """Собирает предупреждения загрузчика diffusers.

    `load_components` не бросает исключение, если компонент не загрузился: он пишет
    предупреждение («можно игнорировать») и оставляет компонент равным None. Причина
    нужна нам в тексте ошибки — в том числе чтобы нехватка VRAM при загрузке дошла
    до шлюза как vram_oom, а не как AttributeError у NoneType.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):
            self.messages.append(record.getMessage())


def _make_checkpoint(check, stage: str):
    """Блок-точка отмены для вставки в цепочку модульного пайплайна.

    Подходит и в обычную цепочку (`block(components, state)`), и внутрь цикла
    денойза (`block(components, block_state, i=..., t=...)`). Функция `check`
    живёт в замыкании, а не в атрибуте: `init_pipeline` делает deepcopy блоков,
    и замыкание при этом остаётся общим, а не копируется вместе с бэкендом.
    """
    from diffusers.modular_pipelines import ModularPipelineBlocks

    class MiniMaxCheckpointStep(ModularPipelineBlocks):
        model_name = "minimax-h3-image"

        @property
        def description(self) -> str:
            return "images-worker: raises if the gateway cancelled the task (%s)." % stage

        def __call__(self, components, state, **kwargs):
            check(stage, kwargs.get("i"))
            return components, state

    return MiniMaxCheckpointStep()


class _Stages:
    """Две под-цепочки модульного пайплайна без загруженных весов."""

    def __init__(self, root: str, check) -> None:
        from diffusers import MiniMaxH3ImageBlocks
        from diffusers.modular_pipelines import SequentialPipelineBlocks

        blocks = MiniMaxH3ImageBlocks()
        self.root = root
        self.text_blocks = blocks.sub_blocks["text_encoder"]

        # Точка отмены перед каждым шагом: первым блоком в цикл денойза ветки text2image.
        denoise_loop = blocks.sub_blocks["denoise"].sub_blocks["text2image"].sub_blocks["denoise"]
        denoise_loop.sub_blocks.insert("checkpoint", _make_checkpoint(check, "denoise"), 0)
        # Полоса tqdm в журнале контейнера только мешает: шаги пишет точка отмены.
        denoise_loop.set_progress_bar_config(disable=True)

        self.image_blocks = SequentialPipelineBlocks.from_blocks_dict(
            {
                "denoise": blocks.sub_blocks["denoise"],
                "checkpoint": _make_checkpoint(check, "decode"),
                "decode": blocks.sub_blocks["decode"],
            }
        )


class MiniMaxBackend(Backend):
    name = "minimax"

    def __init__(self) -> None:
        super().__init__()
        self._device = _env("IMG_MINIMAX_DEVICE", "cuda")
        self._placement = _env("IMG_MINIMAX_PLACEMENT", "to")
        if self._placement not in ("to", "device_map"):
            raise ValueError("IMG_MINIMAX_PLACEMENT must be 'to' or 'device_map', got %r" % self._placement)
        self._exit_after_s = float(_env("IMG_MINIMAX_EXIT_AFTER_S", "120"))
        self._generation = 0
        self._step_started = 0.0

    # --- жизненный цикл ---

    def _load_model(self, spec: ModelSpec):
        # Только разбор блоков и поиск снапшота: секунды и ноль VRAM.
        root = resolve_snapshot(spec)
        log.info("preparing stages for %s from %s (weights load per task)", spec.id, root)
        return _Stages(root, self._checkpoint)

    def load(self, model_id: str) -> tuple[bool, float]:
        # /load — начало новой задачи шлюза: взведённый выход из процесса отменяется.
        self._generation += 1
        return super().load(model_id)

    def _release_memory(self) -> None:
        # Базовая версия зовёт torch.cuda.ipc_collect(), а тот начинается с _lazy_init():
        # у процесса, который карту ещё не трогал (например, /load без /generate, потом
        # /unload по простою), это СОЗДАЛО бы CUDA-контекст — сотни МБ чужой карты, которые
        # некому вернуть. Здесь уборка только там, где контекст уже есть.
        _free_cuda()

    def _checkpoint(self, stage: str, step: int | None) -> None:
        """Вызывается блоками-точками отмены изнутри пайплайна.

        В цикле денойза — перед каждым шагом (step = 0..5), перед декодом — с step=None.
        """
        now = time.monotonic()
        done = step if step is not None else _NUM_STEPS
        if done > 0:
            log.info("denoise step %d/%d done in %.1fs", done, _NUM_STEPS, now - self._step_started)
        self._step_started = now
        self._raise_if_cancelled()

    # --- стадии ---

    def _open(self, blocks, names: tuple[str, ...]):
        """Собрать под-пайплайн и поднять его компоненты на устройство."""
        pipe = blocks.init_pipeline(self._pipe.root)

        kwargs = {"dtype": _DTYPE}
        if self._placement == "device_map":
            # Словарь «компонент -> значение»: строка ушла бы и токенизатору с планировщиком.
            kwargs["device_map"] = {name: self._device for name in names if name in _MODULES}

        failures = _LoadFailures()
        loader_log = logging.getLogger(_DIFFUSERS_LOADER_LOG)
        loader_log.addHandler(failures)
        try:
            pipe.load_components(names=list(names), **kwargs)
        finally:
            loader_log.removeHandler(failures)

        missing = [name for name in names if getattr(pipe, name, None) is None]
        if missing:
            reason = failures.messages[-1].strip().splitlines()[-1] if failures.messages else "no reason logged"
            raise RuntimeError("failed to load %s from %s: %s" % (", ".join(missing), self._pipe.root, reason))

        self._raise_if_cancelled()
        if self._placement == "to":
            pipe.to(self._device)
        return pipe

    def _close(self, pipe, names: tuple[str, ...]) -> None:
        """Снять компоненты с пайплайна и вернуть память карте."""
        if pipe is not None:
            try:
                pipe.unload_components(list(names))
            except Exception as exc:  # pragma: no cover
                log.warning("unload_components failed: %s", exc)
        _free_cuda()

    def _encode(self, prompt: str):
        """Стадия 1: текст -> эмбеддинги. Возвращает тензоры на CPU."""
        pipe = None
        try:
            pipe = self._open(self._pipe.text_blocks, _TEXT_COMPONENTS)
            self._raise_if_cancelled()
            state = pipe(prompt=prompt)
            # На CPU: между стадиями на карте не должно остаться ничего нашего.
            prompt_embeds = state.get("prompt_embeds").to("cpu")
            text_token_tags = state.get("text_token_tags").to("cpu")
            del state
            return prompt_embeds, text_token_tags
        finally:
            self._close(pipe, _TEXT_COMPONENTS)

    def _render(self, prompt_embeds, text_token_tags, job: GenerateJob, seed: int):
        """Стадия 2: эмбеддинги -> картинка (денойз и декод)."""
        pipe = None
        try:
            pipe = self._open(self._pipe.image_blocks, _IMAGE_COMPONENTS)
            self._raise_if_cancelled()
            images = pipe(
                # Денойзер подаёт эмбеддинги в трансформер как есть, без переноса на устройство.
                prompt_embeds=prompt_embeds.to(self._device),
                text_token_tags=text_token_tags,
                height=job.height,
                width=job.width,
                generator=self._generator(seed),
                output="images",
            )
            return images[0]
        finally:
            self._close(pipe, _IMAGE_COMPONENTS)

    def _stage_report(self, name: str, started: float) -> None:
        peak = ""
        if torch.cuda.is_initialized():
            peak = ", peak allocated %d MiB, reserved %d MiB" % (
                torch.cuda.max_memory_allocated() // _MIB,
                torch.cuda.max_memory_reserved() // _MIB,
            )
            torch.cuda.reset_peak_memory_stats()
        left = _cuda_allocated_mb()
        log.info("%s stage done in %.1fs%s, left on card %d MiB", name, time.monotonic() - started, peak, left)
        if left > 64:
            log.warning("%s stage left %d MiB allocated on the card", name, left)

    # --- генерация ---

    def _generate(self, spec: ModelSpec, job: GenerateJob):
        seed = self._seed(job)
        if job.negative_prompt:
            log.warning("negative_prompt ignored: the model is guidance-distilled")
        if job.num_inference_steps != _NUM_STEPS:
            log.info("num_inference_steps=%d ignored: the checkpoint samples in %d", job.num_inference_steps, _NUM_STEPS)

        self._generation += 1
        try:
            self._raise_if_cancelled()
            started = time.monotonic()
            prompt_embeds, text_token_tags = self._encode(job.prompt)
            self._stage_report("text", started)

            self._raise_if_cancelled()
            started = time.monotonic()
            image = self._render(prompt_embeds, text_token_tags, job, seed)
            del prompt_embeds, text_token_tags
            self._stage_report("image", started)
            return image, seed
        except BaseException as exc:
            # И ошибка, и отмена (GenerationCancelled — BaseException): трассировка держит
            # кадры с весами и активациями, без этого карта осталась бы занятой до тех пор,
            # пока исключение не умрёт где-то в HTTP-слое.
            _drop_frames(exc)
            raise
        finally:
            _free_cuda()
            left = _cuda_allocated_mb()
            if left > 64:
                log.error("task %s left %d MiB allocated on the card", job.task_id, left)
            self._arm_exit()

    def _arm_exit(self) -> None:
        """Выйти из процесса вскоре после задачи — так карте возвращается и CUDA-контекст.

        Общий idle-exiter воркера (IMG_WORKER_IDLE_EXIT_S) этому бэкенду не подходит:
        он срабатывает и у процесса, который карту ещё не трогал, то есть перезапускал бы
        контейнер по кругу. Здесь выход только после реальной работы на карте и только
        если за время ожидания не пришла новая задача (она взведёт свой таймер).
        """
        if self._exit_after_s <= 0 or not torch.cuda.is_initialized():
            return
        generation = self._generation

        def _watch() -> None:
            time.sleep(self._exit_after_s)
            with self.lock:  # идущую генерацию дожидаемся, а не обрываем
                if self._generation != generation or self._current_task is not None:
                    return
                log.info("exiting %.0fs after the task to release the CUDA context", self._exit_after_s)
                os._exit(0)

        threading.Thread(target=_watch, name="minimax-exit", daemon=True).start()
