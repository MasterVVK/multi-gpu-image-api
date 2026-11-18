"""Настройки шлюза.

Один класс Settings(BaseSettings), секции с комментариями,
@lru_cache get_settings(). У нетривиальных значений в комментарии написано,
откуда взялось число.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from images_worker.registry import PROJECTS_ROOT
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class NodeSeed(dict):
    """Описание узла из .env — валидируется уже в node_pool."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"),),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Сервер ---
    host: str = Field(default="127.0.0.1", alias="IMG_HOST")
    port: int = Field(default=8010, alias="IMG_PORT")
    log_level: str = Field(default="INFO", alias="IMG_LOG_LEVEL")
    log_json: bool = Field(default=True, alias="IMG_LOG_JSON")
    # Базовый URL для абсолютных ссылок на картинки в OpenAI-ответах.
    public_base_url: str = Field(default="http://127.0.0.1:8010", alias="IMG_PUBLIC_BASE_URL")

    # --- Безопасность ---
    api_keys_raw: str = Field(default="", alias="IMG_API_KEYS")
    # Разрешить бинд на не-loopback без ключей. По умолчанию нельзя: именно так
    # четыре старых сервиса оказались открыты на 0.0.0.0 без единой проверки.
    allow_insecure: bool = Field(default=False, alias="IMG_ALLOW_INSECURE")
    cors_origins_raw: str = Field(default="", alias="IMG_CORS_ORIGINS")
    # Секрет для HMAC-подписи ссылок на картинки: нужен, чтобы <img src> работал
    # без заголовка авторизации. Пустой — подпись выключена.
    image_url_secret: str = Field(default="", alias="IMG_IMAGE_URL_SECRET")
    images_require_auth: bool = Field(default=True, alias="IMG_IMAGES_REQUIRE_AUTH")

    # --- Хранилище ---
    output_dir: Path = Field(default=PROJECT_ROOT / "outputs", alias="IMG_OUTPUT_DIR")
    db_path: Path = Field(default=PROJECT_ROOT / "data" / "images.db", alias="IMG_DB_PATH")
    task_retention_days: int = Field(default=30, alias="IMG_TASK_RETENTION_DAYS")
    image_retention_days: int = Field(default=90, alias="IMG_IMAGE_RETENTION_DAYS")
    # Ограничение на размер ответа воркера: 2048x2048 PNG ≈ 8-12 МБ, 64 МБ — с запасом.
    max_image_bytes: int = Field(default=64 * 1024 * 1024, alias="IMG_MAX_IMAGE_BYTES")

    # --- Планировщик и VRAM ---
    # Остаточный принцип: если карты не хватает, ждём в очереди до 10 минут и
    # честно отказываем. Чужие процессы (ollama, RAG) не вытесняем никогда.
    vram_wait_timeout_s: int = Field(default=600, alias="IMG_VRAM_WAIT_TIMEOUT_S")
    vram_poll_interval_s: float = Field(default=5.0, alias="IMG_VRAM_POLL_INTERVAL_S")
    # Выгрузка резидентной модели после простоя узла.
    idle_unload_s: int = Field(default=180, alias="IMG_IDLE_UNLOAD_SECONDS")
    # Сторож утечек: узел без задачи, но с занятой памятью дольше этого — чиним.
    # Должен быть больше idle_unload_s, иначе аварийная ветка срабатывает раньше
    # штатной и нормальный простой попадает в журнал инцидентов как «залипло»
    # (ровно так и было при 120 против 180 — видно в vram_incidents).
    stuck_vram_s: int = Field(default=300, alias="IMG_STUCK_VRAM_SECONDS")
    guard_interval_s: float = Field(default=30.0, alias="IMG_GUARD_INTERVAL_S")
    lease_heartbeat_s: float = Field(default=5.0, alias="IMG_LEASE_HEARTBEAT_S")
    lease_stale_s: int = Field(default=120, alias="IMG_LEASE_STALE_S")
    # Разрешить сторожу перезапускать зависший воркер через systemctl.
    # Требует правила в sudoers, поэтому по умолчанию выключено.
    allow_worker_restart: bool = Field(default=False, alias="IMG_ALLOW_WORKER_RESTART")

    worker_token: str = Field(default="", alias="IMG_WORKER_TOKEN")
    worker_connect_timeout_s: float = Field(default=10.0, alias="IMG_WORKER_CONNECT_TIMEOUT_S")
    worker_status_timeout_s: float = Field(default=15.0, alias="IMG_WORKER_STATUS_TIMEOUT_S")

    # --- Узлы ---
    # JSON-список. Пустой — узлы берутся только из БД (их можно завести через API).
    nodes_json: str = Field(default="", alias="IMG_NODES_JSON")
    # Диагностика отказов: node -> URL ollama. Используется ТОЛЬКО для чтения
    # /api/ps, чтобы в тексте ошибки было видно, кто именно занял карту.
    node_ollama_urls_json: str = Field(default="", alias="IMG_NODE_OLLAMA_URLS")

    # --- Prompt Enhancer ---
    pe_enabled: bool = Field(default=True, alias="IMG_PE_ENABLED")
    # JSON-список [{"base_url": ..., "model": ..., "api_key": ..., "timeout_s": 60}].
    # Порядок = порядок попыток. Правило отбора: эндпоинт не должен занимать VRAM
    # наших двух 3090 — отсюда cloud-модели node-b и туннели к чужим картам.
    pe_endpoints_json: str = Field(default="", alias="IMG_PE_ENDPOINTS")
    pe_timeout_s: float = Field(default=60.0, alias="IMG_PE_TIMEOUT_S")

    # --- OpenAI-совместимый слой ---
    openai_default_model: str = Field(default="z-image-turbo", alias="IMG_OPENAI_DEFAULT_MODEL")
    # n>1 конфликтует с эксклюзивом: 4 задачи по 50 шагов — это ~10 минут
    # блокирующего HTTP-запроса. Для пакетов есть нативный асинхронный API.
    openai_max_n: int = Field(default=2, alias="IMG_OPENAI_MAX_N")
    openai_timeout_s: float = Field(default=600.0, alias="IMG_OPENAI_TIMEOUT_S")

    @field_validator("output_dir", "db_path", mode="before")
    @classmethod
    def _expand(cls, value: Any) -> Any:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    # --- производные ---

    @property
    def api_keys(self) -> list[str]:
        return [k.strip() for k in self.api_keys_raw.split(",") if k.strip()]

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    @property
    def nodes(self) -> list[dict]:
        return _json_or(self.nodes_json, [])

    @property
    def node_ollama_urls(self) -> dict[str, str]:
        return _json_or(self.node_ollama_urls_json, {})

    @property
    def pe_endpoints(self) -> list[dict]:
        return _json_or(self.pe_endpoints_json, [])

    @property
    def is_loopback(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")

    @property
    def legacy_image_roots(self) -> dict[str, Path]:
        """Каталоги выдачи четырёх старых сервисов.

        Отдаются только на чтение по /api/v1/images/legacy/<ключ>/..., чтобы
        ссылки, сохранённые до объединения, продолжали открываться. Каталоги
        проектов ищутся там же, где их ищет реестр моделей, — в IMG_PROJECTS_ROOT.
        """
        root = Path(PROJECTS_ROOT)
        return {
            "qwen": root / "qi-api" / "outputs",
            "zimage": root / "Z-Image-Turbo" / "outputs",
            "ernie": root / "ERNIE-Image" / "outputs",
            "hidream": root / "HiDream-O1-Image" / "outputs",
        }


def _json_or(raw: str, default: Any) -> Any:
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
