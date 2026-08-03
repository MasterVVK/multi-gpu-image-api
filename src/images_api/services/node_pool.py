"""Пул GPU-узлов: состояние, опрос воркеров, эксклюзивная аренда.

Эксклюзивность держится на asyncio.Lock в памяти процесса — это корректно
ровно потому, что шлюз однопроцессный (--workers 1, проверяется в create_app).
В БД аренда дублируется, чтобы после падения шлюза было видно, кто что держал,
и чтобы GET /api/v1/nodes показывал правду.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from images_worker.registry import MODELS, get_spec

from ..core.config import Settings
from ..core.logging import get_logger
from ..db.conn import Database
from ..domain.schemas import NodeGpu, NodeInfo
from .worker_client import WorkerClient, WorkerUnavailable

log = get_logger("node_pool")


def utcnow() -> datetime:
    # datetime.utcnow() во всех четырёх старых сервисах давал наивное время
    # и в 3.12 уже помечен как устаревший. Здесь везде — aware UTC.
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


@dataclass
class NodeRuntime:
    name: str
    kind: str
    base_urls: dict[str, str]
    worker_token: str = ""
    enabled: bool = True
    draining: bool = False
    vram_reserve_mb: int = 2048
    weight: int = 100
    notes: str = ""

    # --- наблюдаемое состояние ---
    gpu_total_mb: int = 0
    gpu_used_mb: int = 0
    gpu_free_mb: int = 0
    gpu_name: str | None = None
    gpu_age_s: float = 0.0
    models: dict[str, bool] = field(default_factory=dict)
    resident_backend: str | None = None
    resident_model: str | None = None
    healthy: bool = False
    last_seen_at: datetime | None = None
    last_error: str | None = None
    # Узел временно исключён из планирования после сбоя.
    degraded_until: float = 0.0

    # --- аренда ---
    busy_task_id: str | None = None
    lease_acquired_at: datetime | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    # Момент, когда узел последний раз что-то делал — для простой-выгрузки.
    last_activity: float = field(default_factory=time.monotonic)

    @property
    def usable_mb(self) -> int:
        """Сколько VRAM шлюз реально готов отдать под генерацию.

        Резерв — запас под соседей, которых мы не вытесняем: probe-модель
        ollama на br-01, RAG/TTS/whisper и своя ollama на nv-02.
        """
        return max(0, self.gpu_free_mb - self.vram_reserve_mb)

    @property
    def is_degraded(self) -> bool:
        return time.monotonic() < self.degraded_until

    @property
    def schedulable(self) -> bool:
        return self.enabled and not self.draining and self.healthy and not self.is_degraded

    @property
    def busy(self) -> bool:
        return self.busy_task_id is not None

    def serves(self, model_id: str) -> bool:
        spec = MODELS.get(model_id)
        if spec is None or spec.backend not in self.base_urls:
            return False
        return self.models.get(model_id, False)

    def url_for(self, backend: str) -> str | None:
        return self.base_urls.get(backend)

    def degrade(self, seconds: float, reason: str) -> None:
        self.degraded_until = time.monotonic() + seconds
        self.last_error = reason
        log.warning("node_degraded", node=self.name, seconds=seconds, reason=reason)

    def to_info(self) -> NodeInfo:
        return NodeInfo(
            name=self.name,
            kind=self.kind,
            enabled=self.enabled,
            draining=self.draining,
            healthy=self.healthy,
            backends=dict(self.base_urls),
            models=sorted(m for m, ok in self.models.items() if ok),
            resident_model=self.resident_model,
            busy_task_id=self.busy_task_id,
            gpu=NodeGpu(
                total_mb=self.gpu_total_mb,
                used_mb=self.gpu_used_mb,
                free_mb=self.gpu_free_mb,
                reserve_mb=self.vram_reserve_mb,
                usable_mb=self.usable_mb,
                name=self.gpu_name,
                age_s=self.gpu_age_s,
            ),
            weight=self.weight,
            last_seen_at=self.last_seen_at,
            last_error=self.last_error,
        )


class NodePool:
    def __init__(self, db: Database, client: WorkerClient, settings: Settings) -> None:
        self.db = db
        self.client = client
        self.settings = settings
        self.nodes: dict[str, NodeRuntime] = {}

    # --- загрузка и сохранение ---

    async def load(self) -> None:
        rows = await self.db.fetch_all("SELECT * FROM nodes ORDER BY name")
        for row in rows:
            node = NodeRuntime(
                name=row["name"],
                kind=row["kind"],
                base_urls=json.loads(row["base_urls_json"]),
                worker_token=row["worker_token"] or self.settings.worker_token,
                enabled=bool(row["enabled"]),
                draining=bool(row["draining"]),
                vram_reserve_mb=row["vram_reserve_mb"],
                weight=row["weight"],
                notes=row["notes"] or "",
                gpu_total_mb=row["gpu_total_mb"] or 0,
            )
            self.nodes[node.name] = node

        if not self.nodes:
            await self._seed_from_settings()

        # Аренда живёт только в памяти: если шлюз перезапустился, все прежние
        # захваты недействительны по определению.
        await self.db.execute(
            "UPDATE nodes SET busy_task_id=NULL, lease_acquired_at=NULL, lease_heartbeat_at=NULL"
        )

    async def _seed_from_settings(self) -> None:
        seeds = self.settings.nodes
        if not seeds:
            return
        for seed in seeds:
            try:
                node = NodeRuntime(
                    name=seed["name"],
                    kind=seed.get("kind", "local"),
                    base_urls=seed["base_urls"],
                    worker_token=seed.get("worker_token", self.settings.worker_token),
                    enabled=bool(seed.get("enabled", True)),
                    vram_reserve_mb=int(seed.get("vram_reserve_mb", 2048)),
                    weight=int(seed.get("weight", 100)),
                    notes=seed.get("notes", ""),
                )
            except KeyError as exc:
                log.error("node_seed_invalid", seed=seed, missing=str(exc))
                continue
            self.nodes[node.name] = node
            await self.persist(node)
            log.info("node_seeded", node=node.name, backends=sorted(node.base_urls))

    async def persist(self, node: NodeRuntime) -> None:
        await self.db.execute(
            """
            INSERT INTO nodes (name, kind, enabled, draining, base_urls_json, worker_token,
                               gpu_total_mb, vram_reserve_mb, weight, busy_task_id,
                               lease_acquired_at, resident_backend, resident_model,
                               last_seen_at, last_error, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                kind=excluded.kind,
                enabled=excluded.enabled,
                draining=excluded.draining,
                base_urls_json=excluded.base_urls_json,
                worker_token=excluded.worker_token,
                gpu_total_mb=excluded.gpu_total_mb,
                vram_reserve_mb=excluded.vram_reserve_mb,
                weight=excluded.weight,
                busy_task_id=excluded.busy_task_id,
                lease_acquired_at=excluded.lease_acquired_at,
                resident_backend=excluded.resident_backend,
                resident_model=excluded.resident_model,
                last_seen_at=excluded.last_seen_at,
                last_error=excluded.last_error,
                notes=excluded.notes
            """,
            (
                node.name,
                node.kind,
                int(node.enabled),
                int(node.draining),
                json.dumps(node.base_urls),
                node.worker_token,
                node.gpu_total_mb,
                node.vram_reserve_mb,
                node.weight,
                node.busy_task_id,
                _iso(node.lease_acquired_at),
                node.resident_backend,
                node.resident_model,
                _iso(node.last_seen_at),
                node.last_error,
                node.notes,
            ),
        )

    async def add(self, node: NodeRuntime) -> None:
        self.nodes[node.name] = node
        await self.persist(node)
        await self.refresh_node(node)

    async def remove(self, name: str) -> None:
        self.nodes.pop(name, None)
        await self.db.execute("DELETE FROM nodes WHERE name=?", (name,))

    # --- опрос ---

    async def refresh_node(self, node: NodeRuntime) -> None:
        """Опросить все воркеры узла и собрать сводное состояние.

        Свободная VRAM удалённого узла берётся ТОЛЬКО из ответа его собственного
        воркера, который выполняет nvidia-smi у себя. Шлюз никогда не
        экстраполирует цифры br-01 на nv-02.
        """
        models: dict[str, bool] = {}
        resident: tuple[str, str] | None = None
        gpu_seen = False
        errors: list[str] = []

        for backend, base_url in node.base_urls.items():
            try:
                status = await self.client.status(
                    base_url, token=node.worker_token, timeout=self.settings.worker_status_timeout_s
                )
            except (WorkerUnavailable, Exception) as exc:
                errors.append("%s: %s" % (backend, exc))
                # Не отвечает — это НЕ то же самое, что «весов нет». Воркер может
                # просто перезапускаться. Сохраняем прошлую известную доступность:
                # иначе после каждого рестарта планировщик решал бы, что модель
                # выполнить негде, и ставил задачам отсчёт отказа по VRAM.
                # От нерабочего воркера защищает healthy/degraded, а не эта карта.
                for model_id in _models_of(backend):
                    models[model_id] = node.models.get(model_id, False)
                continue

            for model_id in _models_of(backend):
                models[model_id] = model_id in status.supported_models

            if status.resident_model:
                resident = (backend, status.resident_model)

            if status.gpu.available and not gpu_seen:
                # Все воркеры узла смотрят на одну карту — берём первый валидный ответ.
                node.gpu_total_mb = status.gpu.total_mb
                node.gpu_used_mb = status.gpu.used_mb
                node.gpu_free_mb = status.gpu.free_mb
                node.gpu_name = status.gpu.name
                node.gpu_age_s = status.gpu.age_s
                gpu_seen = True

        node.models = models
        node.resident_backend, node.resident_model = resident if resident else (None, None)
        # Узел без единого ответившего воркера не имеет достоверной VRAM,
        # а значит не планируется вовсе.
        node.healthy = gpu_seen
        node.last_seen_at = utcnow() if gpu_seen else node.last_seen_at
        node.last_error = "; ".join(errors) if errors else None

        await self.persist(node)
        await self._persist_models(node)

    async def _persist_models(self, node: NodeRuntime) -> None:
        now = _iso(utcnow())
        rows = [
            (node.name, model_id, get_spec(model_id).backend, int(available), now)
            for model_id, available in node.models.items()
        ]
        if not rows:
            return
        await self.db.execute_many(
            """
            INSERT INTO node_models (node, model_id, backend, available, checked_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(node, model_id) DO UPDATE SET
                available=excluded.available, checked_at=excluded.checked_at
            """,
            rows,
        )

    async def refresh_all(self) -> None:
        await asyncio.gather(
            *(self.refresh_node(node) for node in list(self.nodes.values())),
            return_exceptions=True,
        )

    # --- выбор узла ---

    def candidates(self, model_id: str, forced_node: str | None = None) -> list[NodeRuntime]:
        """Узлы, способные выполнить модель, в порядке предпочтения.

        1. уже держит нужную модель резидентной — экономит 30-120 с загрузки;
        2. дальше по весу и по имени, чтобы порядок был детерминированным.
        """
        pool = self.nodes.values()
        if forced_node:
            pool = [n for n in pool if n.name == forced_node]

        usable = [n for n in pool if n.schedulable and n.serves(model_id)]
        usable.sort(
            key=lambda n: (
                # 1) уже держит нужную модель — экономит минуты загрузки;
                0 if n.resident_model == model_id else 1,
                # 2) свободен, а не занят нашей же задачей;
                1 if n.busy else 0,
                # 3) где больше свободной памяти. Узлы у нас разной загруженности:
                #    на br-01 постоянно висит закреплённый эмбеддер vedora, на
                #    nv-02 карта обычно свободнее. Без этого шага задача сначала
                #    пробовала бы br-01 просто потому, что он первый по алфавиту.
                -n.usable_mb,
                -n.weight,
                n.name,
            )
        )
        return usable

    def nodes_serving(self, model_id: str) -> list[str]:
        return sorted(n.name for n in self.nodes.values() if n.serves(model_id))

    # --- аренда ---

    async def try_acquire(self, node: NodeRuntime, task_id: str) -> bool:
        """Взять эксклюзивную аренду узла под задачу. Без ожидания."""
        if node.lock.locked():
            return False
        try:
            await asyncio.wait_for(node.lock.acquire(), timeout=0.01)
        except TimeoutError:
            return False

        node.busy_task_id = task_id
        node.lease_acquired_at = utcnow()
        node.last_activity = time.monotonic()
        await self.db.execute(
            "UPDATE nodes SET busy_task_id=?, lease_acquired_at=?, lease_heartbeat_at=? WHERE name=?",
            (task_id, _iso(node.lease_acquired_at), _iso(node.lease_acquired_at), node.name),
        )
        log.info("lease_acquired", node=node.name, task_id=task_id)
        return True

    async def release(self, node: NodeRuntime) -> None:
        task_id = node.busy_task_id
        node.busy_task_id = None
        node.lease_acquired_at = None
        node.last_activity = time.monotonic()
        if node.lock.locked():
            node.lock.release()
        await self.db.execute(
            "UPDATE nodes SET busy_task_id=NULL, lease_acquired_at=NULL, lease_heartbeat_at=NULL WHERE name=?",
            (node.name,),
        )
        log.info("lease_released", node=node.name, task_id=task_id)

    async def heartbeat(self, node: NodeRuntime) -> None:
        await self.db.execute(
            "UPDATE nodes SET lease_heartbeat_at=? WHERE name=?", (_iso(utcnow()), node.name)
        )

    def get(self, name: str) -> NodeRuntime | None:
        return self.nodes.get(name)

    @property
    def ready_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.schedulable)


def _models_of(backend: str) -> list[str]:
    return [m.id for m in MODELS.values() if m.backend == backend]
