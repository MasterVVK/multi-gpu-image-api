-- Долговечное состояние шлюза.
--
-- SQLite, а не Postgres/Redis:
--   * Redis на машине нет вовсе;
--   * Postgres есть, но он принадлежит vedora — второй потребитель создал бы
--     связность между проектами и зависимость по порядку старта ради нуля выгоды;
--   * шлюз однопроцессный (--workers 1), задач десятки в сутки.
-- WAL закрывает главный дефект прежней схемы: очередь переживает рестарт,
-- раньше все четыре сервиса держали задачи в обычном dict в памяти.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tasks (
    task_id                 TEXT PRIMARY KEY,
    status                  TEXT NOT NULL,     -- queued|waiting_vram|leased|loading|generating|saving|completed|failed|cancelled
    source                  TEXT NOT NULL DEFAULT 'native',  -- native|openai
    group_id                TEXT,              -- пакет из одного запроса OpenAI при n>1
    idempotency_key         TEXT,

    model_id                TEXT NOT NULL,
    backend                 TEXT NOT NULL,
    node                    TEXT,
    priority                INTEGER NOT NULL DEFAULT 100,

    request_json            TEXT NOT NULL,
    prompt                  TEXT NOT NULL,
    negative_prompt         TEXT,
    enhanced_prompt         TEXT,
    pe_status               TEXT,              -- skipped|external|failed
    pe_error                TEXT,
    use_pe                  INTEGER,

    width                   INTEGER NOT NULL,
    height                  INTEGER NOT NULL,
    aspect_ratio            TEXT,
    num_steps               INTEGER,
    guidance_scale          REAL,
    shift                   REAL,
    scheduler               TEXT,
    seed_requested          INTEGER,
    seed_used               INTEGER,

    image_id                TEXT,
    image_path              TEXT,              -- относительный путь внутри outputs/
    image_bytes             INTEGER,

    queue_time_seconds      REAL,
    generation_time_seconds REAL,
    load_time_seconds       REAL,

    error                   TEXT,
    error_code              TEXT,
    attempts                INTEGER NOT NULL DEFAULT 0,
    cancel_requested        INTEGER NOT NULL DEFAULT 0,

    created_at              TEXT NOT NULL,
    started_at              TEXT,
    completed_at            TEXT,
    heartbeat_at            TEXT,
    -- Дедлайн ожидания VRAM: created_at + IMG_VRAM_WAIT_TIMEOUT_S.
    -- Считается от создания задачи, а не от начала очередной попытки, иначе
    -- «10 минут» растянулись бы на неограниченное время.
    deadline_at             TEXT,

    api_key_id              TEXT,
    request_id              TEXT
);

CREATE INDEX IF NOT EXISTS ix_tasks_queue  ON tasks (status, priority, created_at);
CREATE INDEX IF NOT EXISTS ix_tasks_recent ON tasks (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_tasks_image  ON tasks (image_id);
CREATE INDEX IF NOT EXISTS ix_tasks_group  ON tasks (group_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_tasks_idem
    ON tasks (idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS nodes (
    name                TEXT PRIMARY KEY,
    kind                TEXT NOT NULL DEFAULT 'local',   -- local|remote
    enabled             INTEGER NOT NULL DEFAULT 1,
    -- draining: доработать текущую задачу и больше не брать новых.
    draining            INTEGER NOT NULL DEFAULT 0,
    base_urls_json      TEXT NOT NULL,                   -- {"hidream": "http://127.0.0.1:8104", ...}
    worker_token        TEXT,

    gpu_total_mb        INTEGER,
    -- Запас под соседей по карте, которых мы не контролируем.
    -- br-01: 2048 (probe-модель ollama ~1000 МБ + X + фрагментация).
    -- nv-02: 4096 (RAG/TTS/whisper/chromium, всплески крупнее).
    vram_reserve_mb     INTEGER NOT NULL DEFAULT 2048,
    weight              INTEGER NOT NULL DEFAULT 100,

    busy_task_id        TEXT,
    lease_acquired_at   TEXT,
    lease_heartbeat_at  TEXT,
    resident_backend    TEXT,
    resident_model      TEXT,

    last_seen_at        TEXT,
    last_error          TEXT,
    notes               TEXT
);

-- Какие модели реально доступны на узле. Заполняется автоматически из
-- /status.supported_models каждого воркера, поэтому «на nv-02 есть hidream,
-- но нет ernie» не требует ни строчки конфигурации.
CREATE TABLE IF NOT EXISTS node_models (
    node        TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    backend     TEXT NOT NULL,
    available   INTEGER NOT NULL DEFAULT 1,
    checked_at  TEXT NOT NULL,
    PRIMARY KEY (node, model_id),
    FOREIGN KEY (node) REFERENCES nodes (name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id       TEXT PRIMARY KEY,
    key_hash     TEXT NOT NULL,
    name         TEXT,
    role         TEXT NOT NULL DEFAULT 'user',   -- user|admin
    enabled      INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL,
    last_used_at TEXT
);

-- Журнал вмешательств сторожа VRAM: когда, на каком узле, что нашли и что сделали.
CREATE TABLE IF NOT EXISTS vram_incidents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node        TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    used_mb     INTEGER,
    free_mb     INTEGER,
    resident    TEXT,
    action      TEXT NOT NULL,   -- unload|restart_worker|gave_up
    outcome     TEXT
);

CREATE INDEX IF NOT EXISTS ix_incidents_recent ON vram_incidents (detected_at DESC);

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
