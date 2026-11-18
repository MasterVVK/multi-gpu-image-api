"""Провод между шлюзом и воркером.

Транспорт — HTTP на локальной петле (для node-b — через `ssh -L`). Выбор в пользу
HTTP, а не ZeroMQ/gRPC/pipe, сделан по одной причине: fastapi/uvicorn/pydantic уже
стоят во всех четырёх venv, а новые бинарные колёса в запиненные окружения тащить
нельзя. Побочные плюсы: воркер отлаживается `curl`, у каждого свой юнит с
`Restart=always`, и один и тот же код работает локально и удалённо.

Ответ `POST /generate` — сырые байты PNG, а метаданные едут в заголовке
`X-Images-Meta` (base64url от JSON). Так шлюз получает картинку одним ответом,
без второго запроса и без общей файловой системы.

Модуль импортируется и шлюзом, и воркерами, поэтому тут только pydantic.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from pydantic import BaseModel, Field

META_HEADER = "X-Images-Meta"
TOKEN_HEADER = "X-Worker-Token"


class GpuInfo(BaseModel):
    """Снимок GPU по nvidia-smi.

    Именно nvidia-smi, а не torch.cuda.memory_allocated(): при
    `enable_model_cpu_offload()` torch показывает ~0 ГБ у загруженной модели,
    потому что веса лежат в системной RAM и приезжают на карту помодульно.
    Планировщик должен видеть реальную занятость карты, включая соседей
    (ollama, RAG, chromium), которых мы не контролируем.
    """

    index: int = 0
    name: str | None = None
    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    available: bool = False
    age_s: float = 0.0
    error: str | None = None


class WorkerStatus(BaseModel):
    backend: str
    node: str
    version: str
    pid: int
    uptime_s: float

    # Модели, чьи веса реально доступны на этом хосте. Шлюз строит из этого
    # таблицу node_models, поэтому «на node-b есть hidream, но нет ernie»
    # не требует никакой конфигурации.
    supported_models: list[str] = Field(default_factory=list)

    resident_model: str | None = None
    loading_model: str | None = None
    busy: bool = False
    current_task_id: str | None = None

    gpu: GpuInfo = Field(default_factory=GpuInfo)
    last_error: str | None = None


class LoadRequest(BaseModel):
    model_id: str


class LoadResult(BaseModel):
    model_id: str
    already_loaded: bool = False
    load_time_seconds: float = 0.0
    gpu: GpuInfo = Field(default_factory=GpuInfo)


class UnloadResult(BaseModel):
    unloaded: str | None = None
    gpu: GpuInfo = Field(default_factory=GpuInfo)


class GenerateJob(BaseModel):
    task_id: str
    model_id: str
    prompt: str
    negative_prompt: str = ""
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    shift: float | None = None
    scheduler: str | None = None
    seed: int | None = None


class GenerateMeta(BaseModel):
    """Едет в заголовке X-Images-Meta рядом с байтами PNG."""

    task_id: str
    model_id: str
    backend: str
    node: str
    # Фактический размер отданной картинки. Может отличаться от запрошенного:
    # HiDream примагничивает произвольный размер к ближайшему нативному.
    width: int
    height: int
    requested_width: int | None = None
    requested_height: int | None = None
    num_steps: int
    guidance_scale: float
    shift: float | None = None
    scheduler: str | None = None
    seed_used: int
    generation_time_seconds: float
    load_time_seconds: float = 0.0

    def to_header(self) -> str:
        raw = json.dumps(self.model_dump(), ensure_ascii=False).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii")

    @classmethod
    def from_header(cls, value: str) -> GenerateMeta:
        pad = "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode(value + pad)
        return cls.model_validate(json.loads(raw.decode("utf-8")))


class CancelRequest(BaseModel):
    task_id: str


class WorkerError(BaseModel):
    """Единый конверт ошибки воркера.

    `code` шлюз кладёт в TaskInfo.error_code без перевода, поэтому словарь
    совпадает с доменным: vram_oom | model_unavailable | cancelled | backend_error.
    """

    code: str
    message: str
    detail: dict[str, Any] | None = None


class GenerationCancelled(BaseException):
    """Бросается из step-callback'а, когда шлюз попросил отменить задачу.

    Наследование от BaseException, а не Exception, — намеренное и обязательное.
    Чужие пайплайны бывает, что зовут callback внутри `try: ... except Exception: pass`
    (так делал вендорный код HiDream-O1, удалённой 18.11.2026), и обычное исключение
    молча проглотилось бы — отменить генерацию было бы нечем. BaseException проходит
    сквозь такой перехват — ровно как KeyboardInterrupt.

    Цена: обычные `except Exception` эту отмену не поймают, её нужно ловить явно.
    Все такие места — в backends/base.py и runner.py.
    """
