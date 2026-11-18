"""Единая схема запроса и задачи — надмножество четырёх исходных контрактов.

Что с чем согласовано:

| было                                        | стало                                        |
|---------------------------------------------|----------------------------------------------|
| `cfg_scale` (qwen) / `guidance_scale` (ост.) | одно поле `guidance_scale`; `cfg_scale`      |
|                                             | принимается как алиас и зеркалится в ответе  |
| `model: "turbo"` в трёх сервисах             | плоский namespace: z-image-turbo,            |
|                                             | ernie-image-turbo, …                         |
| `enhanced_prompt` / `rewritten_prompt`       | канон `enhanced_prompt`, второе — зеркало    |
| pe_status из 4 и из 3 значений               | skipped | external | failed                   |
| разные таблицы разрешений                    | пресеты берутся из реестра по модели         |

Смена namespace моделей — ломающее изменение: `{"model": "turbo"}` был валиден
в трёх сервисах с тремя разными смыслами, безопасного автоматического маппинга
нет. Такой запрос получает 422 с перечислением кандидатов.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

TaskStatus = Literal[
    "queued",
    "waiting_vram",
    "leased",
    "loading",
    "generating",
    "saving",
    "completed",
    "failed",
    "cancelled",
]

PEStatus = Literal["skipped", "external", "failed"]

ErrorCode = Literal[
    "vram_unavailable",
    "vram_oom",
    "worker_timeout",
    "worker_unavailable",
    "model_unavailable",
    "cancelled",
    "interrupted",
    "backend_error",
]

ACTIVE_STATUSES: tuple[str, ...] = (
    "queued",
    "waiting_vram",
    "leased",
    "loading",
    "generating",
    "saving",
)
# Статусы, в которых задача уже держала (или могла держать) узел — их надо
# восстанавливать при старте шлюза.
RUNNING_STATUSES: tuple[str, ...] = ("leased", "loading", "generating", "saving")
TERMINAL_STATUSES: tuple[str, ...] = ("completed", "failed", "cancelled")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    model: str = Field(
        ...,
        description="Идентификатор из реестра (полный список — GET /api/v1/models): qwen-image-2.1 | "
        "qwen-image-2.1-turbo | z-image-turbo | z-image-base | ernie-image-turbo | ernie-image | "
        "moss-image | flux3-klein-full | minimax-h3-image и др.",
    )
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=1000)

    # Либо пресет, либо явные размеры — но не оба сразу.
    aspect_ratio: str | None = None
    width: int | None = Field(default=None, ge=256, le=4096)
    height: int | None = Field(default=None, ge=256, le=4096)

    num_inference_steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        validation_alias=AliasChoices("guidance_scale", "cfg_scale"),
        description="CFG. `cfg_scale` — устаревшее написание из qi-api, принимается как алиас.",
    )
    shift: float | None = Field(default=None, ge=0.0, le=20.0, description="Только моделям с параметром shift (сейчас в реестре таких нет)")
    scheduler: str | None = Field(default=None, description="Только моделям со списком планировщиков (сейчас в реестре таких нет)")
    seed: int | None = Field(default=None, ge=0)

    use_pe: bool | None = Field(
        default=None,
        description="Улучшение промпта внешней LLM. Выполняется шлюзом до диспетчеризации, "
        "на чужом GPU — VRAM наших карт не занимает.",
    )
    priority: int = Field(default=100, ge=0, le=1000, description="Меньше — раньше")
    node: str | None = Field(default=None, description="Принудительный узел (node-a | node-b)")
    idempotency_key: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _check_size(self) -> GenerateRequest:
        if (self.width is None) != (self.height is None):
            raise ValueError("width и height задаются только вместе")
        if self.aspect_ratio and self.width is not None:
            raise ValueError("укажите либо aspect_ratio, либо width/height, но не оба")
        return self


class ResolvedParams(BaseModel):
    """Результат применения реестра к запросу — то, что реально уедет воркеру."""

    model_id: str
    backend: str
    width: int
    height: int
    aspect_ratio: str | None = None
    num_inference_steps: int
    guidance_scale: float
    shift: float | None = None
    scheduler: str | None = None
    negative_prompt: str = ""


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    model: str
    backend: str
    node: str | None = None

    prompt: str
    negative_prompt: str | None = None
    enhanced_prompt: str | None = None
    # Устаревшее зеркало enhanced_prompt: так поле называлось у hidream.
    rewritten_prompt: str | None = None
    pe_status: PEStatus | None = None
    pe_error: str | None = None
    use_pe: bool | None = None

    width: int
    height: int
    aspect_ratio: str | None = None
    num_steps: int | None = None
    guidance_scale: float | None = None
    # Устаревшее зеркало guidance_scale: так поле называлось у qi-api.
    cfg_scale: float | None = None
    shift: float | None = None
    scheduler: str | None = None
    seed_used: int | None = None

    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    queue_position: int | None = None
    queue_time_seconds: float | None = None
    generation_time_seconds: float | None = None
    load_time_seconds: float | None = None

    image_id: str | None = None
    image_url: str | None = None

    error: str | None = None
    error_code: ErrorCode | None = None
    attempts: int = 0


class GenerateResponse(BaseModel):
    task_id: str
    status: TaskStatus
    model: str
    node: str | None = None
    queue_position: int
    message: str


class ResolutionInfo(BaseModel):
    aspect_ratio: str
    width: int
    height: int


class ResolutionsResponse(BaseModel):
    model: str
    resolutions: list[ResolutionInfo]
    arbitrary_size_supported: bool
    dim_multiple: int
    min_dim: int
    max_dim: int


class ModelInfo(BaseModel):
    id: str
    display_name: str
    backend: str
    default_steps: int
    max_steps: int
    default_guidance_scale: float
    default_shift: float | None = None
    schedulers: list[str] = Field(default_factory=list)
    aspect_ratios: list[str] = Field(default_factory=list)
    arbitrary_size_supported: bool = False
    dim_multiple: int = 64
    min_dim: int = 512
    max_dim: int = 2048
    supports_negative_prompt: bool = True
    supports_pe: bool = True
    peak_vram_mb: int = 0
    # Узлы, где веса реально лежат. Пустой список — модель сейчас негде выполнить.
    available_on: list[str] = Field(default_factory=list)
    loaded_on: list[str] = Field(default_factory=list)
    notes: str = ""


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default_model: str


class NodeGpu(BaseModel):
    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    reserve_mb: int = 0
    # free_mb - reserve_mb: сколько шлюз реально готов отдать под генерацию.
    usable_mb: int = 0
    name: str | None = None
    age_s: float = 0.0


class NodeInfo(BaseModel):
    name: str
    kind: str
    enabled: bool
    draining: bool
    healthy: bool
    backends: dict[str, str] = Field(default_factory=dict)
    models: list[str] = Field(default_factory=list)
    resident_model: str | None = None
    busy_task_id: str | None = None
    gpu: NodeGpu = Field(default_factory=NodeGpu)
    weight: int = 100
    last_seen_at: datetime | None = None
    last_error: str | None = None


class NodesResponse(BaseModel):
    nodes: list[NodeInfo]


class NodeCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["local", "remote"] = "remote"
    base_urls: dict[str, str]
    worker_token: str | None = None
    vram_reserve_mb: int = Field(default=2048, ge=0, le=24576)
    weight: int = Field(default=100, ge=1, le=10000)
    enabled: bool = False
    notes: str | None = None


class NodeUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    draining: bool | None = None
    vram_reserve_mb: int | None = Field(default=None, ge=0, le=24576)
    weight: int | None = Field(default=None, ge=1, le=10000)
    base_urls: dict[str, str] | None = None
    notes: str | None = None


class QueueEntry(BaseModel):
    task_id: str
    model: str
    status: TaskStatus
    node: str | None = None
    priority: int
    position: int
    waiting_seconds: float
    deadline_in_seconds: float | None = None


class QueueResponse(BaseModel):
    active: list[QueueEntry]
    queued_count: int
    waiting_vram_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    nodes_total: int
    nodes_ready: int
    queued: int
    running: int


class ReadyResponse(BaseModel):
    status: str
    deps: dict[str, str]


class TaskListResponse(BaseModel):
    tasks: list[TaskInfo]
    total: int


def task_row_to_info(row: Any, queue_position: int | None = None, image_url: str | None = None) -> TaskInfo:
    """Строка SQLite → TaskInfo, включая оба устаревших зеркала полей."""
    data = dict(row)
    return TaskInfo(
        task_id=data["task_id"],
        status=data["status"],
        model=data["model_id"],
        backend=data["backend"],
        node=data.get("node"),
        prompt=data["prompt"],
        negative_prompt=data.get("negative_prompt"),
        enhanced_prompt=data.get("enhanced_prompt"),
        rewritten_prompt=data.get("enhanced_prompt"),
        pe_status=data.get("pe_status"),
        pe_error=data.get("pe_error"),
        use_pe=bool(data["use_pe"]) if data.get("use_pe") is not None else None,
        width=data["width"],
        height=data["height"],
        aspect_ratio=data.get("aspect_ratio"),
        num_steps=data.get("num_steps"),
        guidance_scale=data.get("guidance_scale"),
        cfg_scale=data.get("guidance_scale"),
        shift=data.get("shift"),
        scheduler=data.get("scheduler"),
        seed_used=data.get("seed_used"),
        created_at=data["created_at"],
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        queue_position=queue_position,
        queue_time_seconds=data.get("queue_time_seconds"),
        generation_time_seconds=data.get("generation_time_seconds"),
        load_time_seconds=data.get("load_time_seconds"),
        image_id=data.get("image_id"),
        image_url=image_url,
        error=data.get("error"),
        error_code=data.get("error_code"),
        attempts=data.get("attempts", 0),
    )
