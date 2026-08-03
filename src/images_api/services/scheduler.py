"""Планировщик и арбитр VRAM.

Политика — эксклюзив на GPU плюс генерация по остаточному принципу:

1. **Один узел — одна генерация.** Аренда узла держится asyncio.Lock'ом.
   Это корректно, потому что шлюз однопроцессный (--workers 1); проверка стоит
   в create_app и отказывает в старте при WEB_CONCURRENCY != 1.
2. **Одна резидентная модель на узел.** Перед загрузкой другой предыдущая
   выгружается, в том числе через воркер другого бэкенда.
3. **Преflight по nvidia-smi.** Никогда по torch.cuda.memory_allocated(): при
   cpu-offload torch показывает ~0 у загруженной модели.
4. **Ждём до 10 минут.** Не хватает карты — задача уходит в waiting_vram и
   остаётся в очереди; опрос идёт по всем узлам. Отсчёт начинается в момент,
   когда задачу впервые не удалось разместить из-за занятой памяти, — а не при
   её создании. Стоять в очереди за нашей же задачей это не наказывает.
5. **Никого не вытесняем.** Ollama, RAG, TTS на node-b приоритетнее нас. Мы умеем
   только ждать и отказываться.

Важная деталь: ожидание VRAM происходит БЕЗ удержания аренды. Иначе задача,
ждущая десять минут, заблокировала бы узел для всех остальных.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

import httpx
from images_worker.protocol import GenerateJob
from images_worker.registry import get_spec

from ..core.config import Settings
from ..core.logging import get_logger
from ..domain.schemas import GenerateRequest, ResolvedParams
from .node_pool import NodePool, NodeRuntime
from .prompt_enhancer import PromptEnhancer
from .storage import ImageStorage
from .tasks import TaskRepository, utcnow
from .worker_client import WorkerClient, WorkerRejected, WorkerUnavailable

log = get_logger("scheduler")

# Как долго узел не рассматривается после сбоя воркера.
_DEGRADE_AFTER_FAILURE_S = 30.0
_MAX_ATTEMPTS = 2


class Scheduler:
    def __init__(
        self,
        *,
        tasks: TaskRepository,
        pool: NodePool,
        client: WorkerClient,
        storage: ImageStorage,
        enhancer: PromptEnhancer,
        settings: Settings,
    ) -> None:
        self.tasks = tasks
        self.pool = pool
        self.client = client
        self.storage = storage
        self.enhancer = enhancer
        self.settings = settings

        self.wake = asyncio.Event()
        self._running: dict[str, asyncio.Task] = {}
        self._enhancing: set[str] = set()
        # Живые ссылки на фоновые задачи: без них event loop держит только слабые.
        self._background: set[asyncio.Task] = set()
        self._loops: list[asyncio.Task] = []
        self._stopping = False

    # --- жизненный цикл ---

    async def start(self) -> None:
        self._loops = [
            asyncio.create_task(self._arbiter_loop(), name="arbiter"),
            asyncio.create_task(self._sweeper_loop(), name="sweeper"),
            asyncio.create_task(self._refresh_loop(), name="node-refresh"),
        ]
        log.info("scheduler_started", nodes=len(self.pool.nodes))

    async def stop(self) -> None:
        self._stopping = True
        for task in self._loops:
            task.cancel()
        for task in [*self._running.values(), *self._background]:
            task.cancel()
        for task in [*self._loops, *self._running.values()]:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._loops.clear()
        self._running.clear()
        self._background.clear()

    def notify(self) -> None:
        """Разбудить арбитра — вызывается при постановке задачи в очередь."""
        self.wake.set()

    # --- циклы ---

    async def _arbiter_loop(self) -> None:
        while not self._stopping:
            try:
                with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                    await asyncio.wait_for(
                        self.wake.wait(), timeout=self.settings.vram_poll_interval_s
                    )
                self.wake.clear()
                await self._dispatch_round()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("arbiter_error", error=str(exc))
                await asyncio.sleep(1.0)

    async def _sweeper_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(60.0)
            try:
                await self.tasks.sweep_stale(self.settings.lease_stale_s, _MAX_ATTEMPTS)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("sweeper_error", error=str(exc))

    async def _refresh_loop(self) -> None:
        while not self._stopping:
            try:
                await self.pool.refresh_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("node_refresh_error", error=str(exc))
            await asyncio.sleep(self.settings.guard_interval_s)

    # --- диспетчеризация ---

    async def _dispatch_round(self) -> None:
        pending = await self.tasks.pending()
        if not pending:
            return

        now = utcnow()

        for row in pending:
            task_id = row["task_id"]
            if task_id in self._running or task_id in self._enhancing:
                continue

            if row["cancel_requested"]:
                await self.tasks.mark_cancelled(task_id)
                continue

            # Улучшение промпта делаем ДО аренды узла: держать GPU во время
            # 30-90-секундного вызова LLM было бы расточительно.
            if row["use_pe"] and row["pe_status"] is None:
                self._enhancing.add(task_id)
                # Ссылку обязательно держать: event loop хранит на задачи только
                # слабые ссылки, и без этого сборщик мусора может убить улучшение
                # промпта прямо посреди работы.
                task = asyncio.create_task(self._enhance(task_id))
                self._background.add(task)
                task.add_done_callback(self._background.discard)
                continue

            model_id = row["model_id"]
            candidates = self.pool.candidates(model_id, forced_node=row["node"])

            if not candidates:
                # Модель негде выполнить прямо сейчас: узел мог упасть или
                # перезапускаться. Окно отсчёта даём (иначе задача висела бы
                # вечно), но статус НЕ меняем на waiting_vram — карта тут ни при
                # чём, и врать про причину нельзя.
                if _past_deadline(row, now):
                    await self._fail_no_capacity(task_id, model_id, row["node"])
                else:
                    await self._start_wait(row, row["status"], model_id, [])
                continue

            free_nodes = [n for n in candidates if not n.busy]
            if not free_nodes:
                # Узел занят нашей же задачей — это обычная очередь, а не
                # нехватка карты. Окно ожидания VRAM здесь не запускаем.
                continue

            placed = False
            for node in free_nodes:
                # Аренда берётся ДО проверки памяти: вытеснение чужой резидентной
                # модели — это уже работа с узлом, и делать её без аренды нельзя.
                if not await self.pool.try_acquire(node, task_id):
                    continue

                verdict = await self._vram_verdict(node, model_id)

                if verdict is False and node.resident_model and node.resident_model != model_id:
                    # Карту держит наша же предыдущая модель. Ждать, пока сработает
                    # таймер простоя, незачем — освобождаем сразу и перепроверяем.
                    # Особенно важно для ernie-* : при пике 21.5 ГБ им не хватает
                    # даже остатков соседнего воркера.
                    log.info(
                        "evicting_to_fit",
                        node=node.name,
                        resident=node.resident_model,
                        wanted=model_id,
                        free_mb=node.gpu_free_mb,
                    )
                    await self._evict(node)
                    verdict = await self._vram_verdict(node, model_id)

                if verdict is True:
                    self._running[task_id] = asyncio.create_task(self._run(node, task_id))
                    placed = True
                    break

                await self.pool.release(node)

            if not placed:
                # Ни на одном свободном узле сейчас не помещаемся — вот это и есть
                # нехватка VRAM, с которой начинается отсчёт десяти минут.
                if _past_deadline(row, now):
                    await self._fail_vram(task_id, model_id, candidates)
                else:
                    await self._start_wait(row, "waiting_vram", model_id, free_nodes)
                # Не прерываем цикл: задача подешевле может поместиться там, где
                # эта не помещается. Окно ожидания ограничивает голодание.

    async def _start_wait(
        self, row, status: str, model_id: str, nodes: list[NodeRuntime]
    ) -> None:
        """Начать отсчёт окна ожидания свободной карты (если ещё не начат).

        Ключевой момент: окно запускается только тогда, когда задачу нельзя
        разместить из-за занятой VRAM или отсутствия узла, — но НЕ тогда, когда
        она просто стоит в очереди за нашей же задачей. Иначе десять минут
        истекали бы у всех, кто оказался вторым в очереди.
        """
        task_id = row["task_id"]
        if row["deadline_at"]:
            return

        deadline = await self.tasks.start_vram_wait(task_id, self.settings.vram_wait_timeout_s)
        if row["status"] != status:
            await self.tasks.mark_status(task_id, status)
        log.info(
            "vram_wait_started",
            task_id=task_id,
            model=model_id,
            timeout_s=self.settings.vram_wait_timeout_s,
            deadline=deadline.isoformat(),
            nodes=[n.name for n in nodes],
        )

    async def _vram_verdict(self, node: NodeRuntime, model_id: str) -> bool | None:
        """Помещается ли модель на узел прямо сейчас.

        True — да; False — нет; None — узел не отвечает и исключён из круга.
        """
        spec = get_spec(model_id)
        url = node.url_for(spec.backend)
        if url is None:
            return None

        try:
            status = await self.client.status(
                url, token=node.worker_token, timeout=self.settings.worker_status_timeout_s
            )
        except (WorkerUnavailable, WorkerRejected) as exc:
            node.degrade(_DEGRADE_AFTER_FAILURE_S, "status: %s" % exc)
            return None

        if not status.gpu.available:
            node.degrade(_DEGRADE_AFTER_FAILURE_S, "gpu unavailable: %s" % status.gpu.error)
            return None

        node.gpu_total_mb = status.gpu.total_mb
        node.gpu_used_mb = status.gpu.used_mb
        node.gpu_free_mb = status.gpu.free_mb

        need = required_mb(model_id, resident=(node.resident_model == model_id))
        return status.gpu.free_mb >= need + node.vram_reserve_mb

    # --- исполнение ---

    async def _run(self, node: NodeRuntime, task_id: str) -> None:
        """Выполнить задачу на арендованном узле.

        Инвариант: из этой функции нельзя выйти, оставив за узлом занятую VRAM
        или незакрытую аренду. Отсюда единственный finally на всё.
        """
        heartbeat: asyncio.Task | None = None
        row = await self.tasks.get(task_id)
        if row is None:
            await self.pool.release(node)
            self._running.pop(task_id, None)
            return

        model_id = row["model_id"]
        spec = get_spec(model_id)
        url = node.url_for(spec.backend)
        needs_unload_on_error = False

        try:
            await self.tasks.mark_started(task_id, node.name)
            heartbeat = asyncio.create_task(self._heartbeat(node, task_id))

            # Вытеснение перед загрузкой: одна резидентная модель на узел,
            # в том числе если занят воркер другого бэкенда.
            if node.resident_model and node.resident_model != model_id:
                await self._evict(node)

            await self.tasks.mark_status(task_id, "loading")
            needs_unload_on_error = True
            load_result = await self.client.load(
                url, model_id, token=node.worker_token, timeout=float(spec.load_timeout_s)
            )
            node.resident_backend, node.resident_model = spec.backend, model_id

            prompt = row["enhanced_prompt"] or row["prompt"]
            job = GenerateJob(
                task_id=task_id,
                model_id=model_id,
                prompt=prompt,
                negative_prompt=row["negative_prompt"] or "",
                width=row["width"],
                height=row["height"],
                num_inference_steps=row["num_steps"],
                guidance_scale=row["guidance_scale"],
                shift=row["shift"],
                scheduler=row["scheduler"],
                seed=row["seed_requested"],
            )

            await self.tasks.mark_status(task_id, "generating")
            timeout = generate_timeout(model_id, row["num_steps"])
            payload, meta = await self.client.generate(
                url,
                job,
                token=node.worker_token,
                timeout=timeout,
                max_bytes=self.settings.max_image_bytes,
            )

            await self.tasks.mark_status(task_id, "saving")
            image_id, rel_path, size = await asyncio.get_running_loop().run_in_executor(
                None, self.storage.write, payload
            )
            await self.tasks.mark_completed(
                task_id,
                image_id=image_id,
                image_path=rel_path,
                image_bytes=size,
                seed_used=meta.seed_used,
                generation_time_seconds=meta.generation_time_seconds,
                load_time_seconds=meta.load_time_seconds or load_result.load_time_seconds,
                width=meta.width,
                height=meta.height,
            )
            if (meta.width, meta.height) != (row["width"], row["height"]):
                log.info(
                    "size_adjusted_by_pipeline",
                    task_id=task_id,
                    model=model_id,
                    requested="%dx%d" % (row["width"], row["height"]),
                    produced="%dx%d" % (meta.width, meta.height),
                )

        except asyncio.CancelledError:
            await self._on_cancel(node, task_id, url)
            raise
        except WorkerRejected as exc:
            await self._on_worker_rejected(node, task_id, exc)
        except WorkerUnavailable as exc:
            await self._on_worker_unavailable(node, task_id, row, exc, url)
        except Exception as exc:
            log.exception("task_run_error", task_id=task_id, node=node.name, error=str(exc))
            await self.tasks.mark_failed(task_id, str(exc), "backend_error")
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await heartbeat

            # Если задача упала после начала загрузки — снимаем модель, чтобы
            # за узлом не осталась занятая память по нашей вине.
            final = await self.tasks.get(task_id)
            failed = final is not None and final["status"] in ("failed", "cancelled")
            if failed and needs_unload_on_error:
                await self._safe_unload(node, url)

            await self.pool.release(node)
            self._running.pop(task_id, None)
            self.notify()

    async def _heartbeat(self, node: NodeRuntime, task_id: str) -> None:
        while True:
            await asyncio.sleep(self.settings.lease_heartbeat_s)
            await self.tasks.heartbeat(task_id)
            await self.pool.heartbeat(node)

    async def _evict(self, node: NodeRuntime) -> None:
        backend = node.resident_backend
        url = node.url_for(backend) if backend else None
        if url is None:
            node.resident_backend = node.resident_model = None
            return
        log.info("evict_before_load", node=node.name, model=node.resident_model)
        await self._safe_unload(node, url)

    async def _safe_unload(self, node: NodeRuntime, url: str | None) -> None:
        if url is None:
            return
        try:
            await self.client.unload(url, token=node.worker_token)
            node.resident_backend = node.resident_model = None
        except (WorkerUnavailable, WorkerRejected) as exc:
            log.warning("unload_failed", node=node.name, error=str(exc))

    # --- обработка ошибок ---

    async def _on_cancel(self, node: NodeRuntime, task_id: str, url: str | None) -> None:
        if url:
            with contextlib.suppress(Exception):
                await self.client.cancel(url, task_id, token=node.worker_token)

        # Отличаем отмену пользователем от остановки шлюза. При рестарте задача
        # ни в чём не виновата: возвращаем её в очередь, чтобы восстановление
        # подхватило её после старта, а не молча потеряло.
        if self._stopping:
            row = await self.tasks.get(task_id)
            if row is not None and row["attempts"] < _MAX_ATTEMPTS:
                await self.tasks.requeue(task_id)
                log.info("task_requeued_on_shutdown", task_id=task_id)
                return

        await self.tasks.mark_cancelled(task_id)

    async def _on_worker_rejected(self, node: NodeRuntime, task_id: str, exc: WorkerRejected) -> None:
        if exc.code == "cancelled":
            await self.tasks.mark_cancelled(task_id)
            return
        if exc.code == "vram_oom":
            # Мы прошли преflight, но карта всё равно кончилась: сосед вырос
            # уже после проверки. Оценка пика в реестре, скорее всего, занижена.
            node.degrade(_DEGRADE_AFTER_FAILURE_S, "oom")
            await self.tasks.mark_failed(task_id, exc.message, "vram_oom")
            return
        code = "model_unavailable" if exc.code == "model_unavailable" else "backend_error"
        await self.tasks.mark_failed(task_id, exc.message, code)

    async def _on_worker_unavailable(
        self, node: NodeRuntime, task_id: str, row, exc: WorkerUnavailable, url: str | None = None
    ) -> None:
        node.degrade(_DEGRADE_AFTER_FAILURE_S, "unavailable: %s" % exc)
        timed_out = "timeout" in str(exc).lower()

        # Мы перестали ждать — значит воркер обязан перестать считать. Иначе он
        # продолжит жечь карту как сирота, и убирать за ним придётся сторожу.
        if timed_out and url:
            with contextlib.suppress(Exception):
                await self.client.cancel(url, task_id, token=node.worker_token)

        if row["attempts"] < _MAX_ATTEMPTS:
            # Повтор осмысленен: воркер мог перезапускаться, а на другом узле
            # та же модель может быть доступна.
            await self.tasks.requeue(task_id)
            log.warning("task_requeued", task_id=task_id, node=node.name, reason=str(exc)[:200])
            return

        await self.tasks.mark_failed(
            task_id, str(exc), "worker_timeout" if timed_out else "worker_unavailable"
        )

    async def _fail_vram(self, task_id: str, model_id: str, candidates: list[NodeRuntime]) -> None:
        """Отказ по остаточному принципу — штатный исход, а не поломка."""
        detail = await self._vram_detail(model_id, candidates)
        await self.tasks.mark_failed(task_id, detail, "vram_unavailable")

    async def _vram_detail(self, model_id: str, candidates: list[NodeRuntime]) -> str:
        """Внятный текст отказа: сколько нужно, сколько есть и кто занял карту."""
        need = required_mb(model_id, resident=False)
        parts: list[str] = []
        for node in candidates:
            piece = "%s: free %d МБ, резерв %d МБ, нужно %d МБ" % (
                node.name,
                node.gpu_free_mb,
                node.vram_reserve_mb,
                need,
            )
            holder = await self._ollama_holder(node)
            if holder:
                piece += ", ollama держит %s" % holder
            parts.append(piece)
        return "не дождались свободной VRAM за %d с — %s" % (
            self.settings.vram_wait_timeout_s,
            "; ".join(parts) or "нет доступных узлов",
        )

    async def _ollama_holder(self, node: NodeRuntime) -> str | None:
        """Кто из соседей занял карту — только для текста ошибки.

        Читаем /api/ps и ничего не выгружаем: соседи приоритетнее нас, а попытка
        их подвинуть вдобавок гонялась бы с их собственным watchdog'ом.
        """
        url = self.settings.node_ollama_urls.get(node.name)
        if not url:
            return None
        try:
            async with httpx.AsyncClient() as probe:
                resp = await probe.get("%s/api/ps" % url.rstrip("/"), timeout=3.0)
                resp.raise_for_status()
                models = resp.json().get("models", [])
        except Exception:
            return None
        if not models:
            return None
        return ", ".join(
            "%s (%.1f ГБ)" % (m.get("name", "?"), (m.get("size_vram") or 0) / 1e9) for m in models
        )

    async def _fail_no_capacity(self, task_id: str, model_id: str, forced_node: str | None) -> None:
        serving = self.pool.nodes_serving(model_id)
        if forced_node:
            message = "узел %s не обслуживает модель %s" % (forced_node, model_id)
        elif serving:
            message = "узлы с моделью %s есть (%s), но ни один не готов" % (
                model_id,
                ", ".join(serving),
            )
        else:
            message = "ни один узел не имеет весов модели %s" % model_id
        await self.tasks.mark_failed(task_id, message, "model_unavailable")

    # --- улучшение промпта ---

    async def _enhance(self, task_id: str) -> None:
        try:
            row = await self.tasks.get(task_id)
            if row is None:
                return
            enhanced, status, error = await self.enhancer.enhance(
                row["prompt"], row["width"], row["height"]
            )
            await self.tasks.set_pe_result(
                task_id,
                enhanced if status == "external" else None,
                status,
                error,
            )
        except Exception as exc:
            log.warning("pe_unexpected_error", task_id=task_id, error=str(exc))
            with contextlib.suppress(Exception):
                await self.tasks.set_pe_result(task_id, None, "failed", str(exc))
        finally:
            self._enhancing.discard(task_id)
            self.notify()

    # --- отмена ---

    async def cancel(self, task_id: str) -> bool:
        row = await self.tasks.get(task_id)
        if row is None:
            return False
        if row["status"] in ("completed", "failed", "cancelled"):
            return False

        await self.tasks.request_cancel(task_id)

        running = self._running.get(task_id)
        if running is None:
            await self.tasks.mark_cancelled(task_id)
            return True

        node = self.pool.get(row["node"]) if row["node"] else None
        if node is not None:
            spec = get_spec(row["model_id"])
            url = node.url_for(spec.backend)
            if url:
                with contextlib.suppress(Exception):
                    await self.client.cancel(url, task_id, token=node.worker_token)
        return True

    # --- восстановление при старте ---

    async def recover(self) -> None:
        """Привести узлы к заведомо чистому состоянию до приёма трафика.

        Шлюз мог упасть в любой момент, и за воркерами могла остаться
        загруженная модель, о которой уже никто не знает.
        """
        await self.tasks.recover_on_startup(_MAX_ATTEMPTS)
        await self.pool.refresh_all()

        for node in self.pool.nodes.values():
            for backend, url in node.base_urls.items():
                with contextlib.suppress(Exception):
                    result = await self.client.unload(url, token=node.worker_token)
                    if result.unloaded:
                        log.info("startup_unload", node=node.name, backend=backend, model=result.unloaded)
            node.resident_backend = node.resident_model = None


def required_mb(model_id: str, resident: bool) -> int:
    """Сколько свободной VRAM нужно под задачу (без резерва узла).

    Если модель уже резидентна и не использует cpu-offload, её веса уже учтены
    в used_mb — второй раз их требовать нельзя, иначе резидентная hidream
    никогда бы не проходила собственную проверку.
    """
    spec = get_spec(model_id)
    if resident and spec.weights_vram_mb:
        return max(1024, spec.peak_vram_mb - spec.weights_vram_mb)
    return spec.peak_vram_mb


def generate_timeout(model_id: str, steps: int | None) -> float:
    """Тайм-аут генерации.

    Намеренно щедрый. Слишком длинный тайм-аут лишь откладывает обнаружение
    действительно зависшего воркера — а зависшего всё равно поймает sweeper по
    протухшему heartbeat. Слишком короткий уничтожает уже сделанную работу:
    первая версия ставила 141 с на z-image-turbo, оборвала честную генерацию,
    сторож выгрузил модель, и всё пошло по второму кругу — 8 минут впустую.

    Слагаемое в 240 с — это прогрев: первый прогон после загрузки заметно
    медленнее (компиляция ядер, первый проход текстового энкодера, а у моделей
    с cpu_offload ещё и прокачка весов RAM→VRAM).
    """
    spec = get_spec(model_id)
    steps = steps or spec.default_steps
    return 240.0 + steps * spec.seconds_per_step_hint * 3.0


def _past_deadline(row, now: datetime) -> bool:
    raw = row["deadline_at"]
    if not raw:
        return False
    deadline = datetime.fromisoformat(raw)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return now >= deadline


async def enqueue(
    scheduler: Scheduler,
    request: GenerateRequest,
    resolved: ResolvedParams,
    *,
    source: str = "native",
    group_id: str | None = None,
    api_key_id: str | None = None,
    request_id: str | None = None,
) -> str:
    """Поставить задачу и разбудить арбитра."""
    spec = get_spec(resolved.model_id)
    use_pe = request.use_pe if request.use_pe is not None else False
    if use_pe and not spec.supports_pe:
        use_pe = False

    task_id = await scheduler.tasks.create(
        request,
        resolved,
        vram_wait_timeout_s=scheduler.settings.vram_wait_timeout_s,
        source=source,
        group_id=group_id,
        api_key_id=api_key_id,
        request_id=request_id,
        use_pe=use_pe,
    )
    scheduler.notify()
    return task_id
