"""Единый каталог моделей — общий источник правды для шлюза и воркеров.

Заменяет четыре несогласованных конфига:
  qi-api/src/qi_api/config.py            (ASPECT_RATIO_TO_SIZE, default_steps)
  Z-Image-Turbo/src/z_image_api/config.py (MODEL_CONFIGS)
  ERNIE-Image/src/ernie_image_api/config.py
  HiDream-O1-Image/src/hidream_o1_api/config.py (MODEL_CONFIG)

Идентификаторы плоские: раньше `model: "turbo"` был валиден в трёх сервисах
с тремя разными смыслами. Теперь `z-image-turbo` и `ernie-image-turbo` — разные вещи.

Модуль намеренно без зависимостей (даже без pydantic): его импортирует и шлюз,
и каждый из четырёх venv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Корень, где лежат каталоги проектов моделей. Вынесен в переменную, чтобы
# развернуть проект на другой машине можно было без правки кода.
PROJECTS_ROOT = os.environ.get("IMG_PROJECTS_ROOT", "/home/user")

# Бэкенд = отдельный процесс со своим venv. Один бэкенд обслуживает 1..N моделей,
# но резидентной в каждый момент держит ровно одну.
BACKENDS = ("qwen", "zimage", "ernie", "hidream")

# Как называется параметр CFG в вызове конкретного пайплайна.
# qwen: QwenImagePipeline принимает и `guidance_scale`, и `true_cfg_scale`,
#       но реальный CFG с negative_prompt делает именно `true_cfg_scale`.
#       Старый qi-api не передавал ни того, ни другого — см. GUIDANCE_PARAM ниже.
GUIDANCE_PARAMS = ("true_cfg_scale", "guidance_scale", "none")


@dataclass(frozen=True)
class ModelSpec:
    """Полное описание модели: что грузить, чем считать, сколько VRAM просить."""

    id: str
    backend: str
    display_name: str

    # --- источник весов ---
    # repo_id — HuggingFace repo; local_path — каталог на диске (только hidream).
    # files — для zimage, который собирается вручную из отдельных safetensors.
    repo_id: str | None = None
    local_path: str | None = None
    variant: str | None = None
    files: dict[str, str] = field(default_factory=dict)

    # --- дефолты генерации ---
    default_steps: int = 30
    max_steps: int = 100
    default_guidance: float = 5.0
    guidance_param: str = "guidance_scale"
    default_shift: float | None = None
    schedulers: tuple[str, ...] = ()
    default_scheduler: str | None = None

    # --- размеры ---
    aspect_ratios: dict[str, tuple[int, int]] = field(default_factory=dict)
    default_aspect_ratio: str = "1:1"
    arbitrary_size: bool = False
    dim_multiple: int = 64
    min_dim: int = 512
    max_dim: int = 2048

    # --- возможности ---
    supports_negative_prompt: bool = True
    supports_pe: bool = True

    # --- ресурсы ---
    # peak_vram_mb — пик по nvidia-smi во время генерации в дефолтном разрешении.
    # weights_vram_mb — сколько карта держит, когда модель загружена, но простаивает.
    #   Для моделей с cpu_offload это 0: веса лежат в системной RAM и приезжают
    #   помодульно. Ненулевое значение только у hidream (device_map="cuda").
    #   Планировщик вычитает его, если модель уже резидентна, — иначе резидентная
    #   hidream не проходила бы собственную проверку: её 16.4 ГБ уже в used_mb.
    #
    # Замерено scripts/measure_vram.py 2026-08-03 на node-b (RTX 3090), дефолтный
    # пресет, за вычетом фона от соседей по карте; сверху +5%.
    # Почему +5%, а не больше: занижение приводит к vram_oom — задача падает, но
    # память освобождается штатно; завышение приводит к vram_unavailable, то есть
    # модель просто перестаёт запускаться. Первое дешевле.
    # ВНИМАНИЕ: ernie-* НЕ замерены (их негде было выполнить) — там всё ещё оценка
    # из документации проекта. Замеры сделаны на дефолтном разрешении; на
    # максимальном пик будет выше.
    peak_vram_mb: int = 16000
    weights_vram_mb: int = 0
    cpu_offload: bool = False
    # Холодная загрузка с диска реально занимает минуты: замерено 131 с на шард
    # у hidream (33 ГБ, 8 шардов) при непрогретом page cache. Тайм-ауты с запасом.
    load_timeout_s: int = 1800
    seconds_per_step_hint: float = 3.0

    notes: str = ""

    @property
    def default_size(self) -> tuple[int, int]:
        return self.aspect_ratios[self.default_aspect_ratio]


# --- Qwen-Image -------------------------------------------------------------
# Источник: /home/user/qi-api/.env → QI_MODEL_ID=Ilus-AI/Qwen-Image-2512-FP8,
#           QI_QUANTIZATION=none → ветка DiffusionPipeline.from_pretrained(bf16).
# Пресет 4:3 исправлен: в qi_api/config.py стояло 1472x1140, но 1140 не делится
# на 8 и diffusers молча переокругляет. 1136 = 71*16.
QWEN_IMAGE = ModelSpec(
    id="qwen-image",
    backend="qwen",
    display_name="Qwen-Image 2512 FP8",
    repo_id="Ilus-AI/Qwen-Image-2512-FP8",
    default_steps=50,
    max_steps=100,
    default_guidance=4.0,
    guidance_param="true_cfg_scale",
    aspect_ratios={
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1136),
        "3:4": (1136, 1472),
    },
    arbitrary_size=True,
    dim_multiple=16,
    min_dim=512,
    max_dim=2048,
    # Замерено 2026-08-03: пик 20400 МБ чистыми. Оценка «16000» из
    # документации была занижена на 4.4 ГБ — при тесной карте это дало бы OOM.
    peak_vram_mb=21500,
    cpu_offload=True,
    load_timeout_s=2400,
    seconds_per_step_hint=12.0,
    notes="CFG подаётся как true_cfg_scale вместе с negative_prompt. "
    "Старый qi-api принимал cfg_scale и не передавал его в пайплайн вовсе.",
)

# --- Z-Image ----------------------------------------------------------------
# Пайплайн собирается вручную из отдельных файлов репозитория DeepBeepMeep/Z-Image
# (в старом коде REPO_ID был захардкожен, а ZI_TURBO_REPO из .env игнорировался).
_ZIMAGE_COMMON_FILES = {
    "vae": "ZImageTurbo_VAE_bf16.safetensors",
    "vae_config": "ZImageTurbo_VAE_bf16_config.json",
    "scheduler_config": "ZImageTurbo_scheduler_config.json",
    "text_encoder": "Qwen3/qwen3_bf16.safetensors",
    "text_encoder_config": "Qwen3/config.json",
    "tokenizer_config": "Qwen3/tokenizer_config.json",
    "tokenizer": "Qwen3/tokenizer.json",
    "vocab": "Qwen3/vocab.json",
    "merges": "Qwen3/merges.txt",
    "special_tokens": "Qwen3/special_tokens_map.json",
}

_ZIMAGE_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1216, 896),
    "3:4": (896, 1216),
}

Z_IMAGE_TURBO = ModelSpec(
    id="z-image-turbo",
    backend="zimage",
    display_name="Z-Image Turbo",
    repo_id="DeepBeepMeep/Z-Image",
    variant="turbo",
    files=dict(_ZIMAGE_COMMON_FILES, transformer="ZImageTurbo_bf16.safetensors"),
    default_steps=9,
    max_steps=20,
    default_guidance=0.0,
    guidance_param="guidance_scale",
    aspect_ratios=dict(_ZIMAGE_RATIOS),
    arbitrary_size=True,
    dim_multiple=32,
    min_dim=512,
    max_dim=2048,
    # Замерено 2026-08-03: пик 12900 МБ чистыми, +5%.
    peak_vram_mb=13600,
    cpu_offload=True,
    load_timeout_s=1800,
    seconds_per_step_hint=12.0,
    notes="Дистилляция без CFG: guidance_scale=0.0, 9 шагов. Самая дешёвая модель по VRAM.",
)

Z_IMAGE_BASE = ModelSpec(
    id="z-image-base",
    backend="zimage",
    display_name="Z-Image Base",
    repo_id="DeepBeepMeep/Z-Image",
    variant="base",
    files=dict(_ZIMAGE_COMMON_FILES, transformer="ZImageBase_bf16.safetensors"),
    default_steps=30,
    max_steps=100,
    default_guidance=5.0,
    guidance_param="guidance_scale",
    aspect_ratios=dict(_ZIMAGE_RATIOS),
    arbitrary_size=True,
    dim_multiple=32,
    min_dim=512,
    max_dim=2048,
    # Отдельно не замерен: тяжелее turbo, оценка от его замера +5%.
    peak_vram_mb=14300,
    cpu_offload=True,
    load_timeout_s=1800,
    seconds_per_step_hint=12.0,
)

# --- ERNIE-Image ------------------------------------------------------------
# Единственный бэкенд без произвольных размеров: пресеты взяты из карточки модели.
_ERNIE_RATIOS = {
    "1:1": (1024, 1024),
    "3:2": (1264, 848),
    "2:3": (848, 1264),
    "16:9": (1376, 768),
    "9:16": (768, 1376),
    "4:3": (1200, 896),
    "3:4": (896, 1200),
}

ERNIE_IMAGE_TURBO = ModelSpec(
    id="ernie-image-turbo",
    backend="ernie",
    display_name="ERNIE-Image Turbo",
    repo_id="baidu/ERNIE-Image-Turbo",
    variant="turbo",
    default_steps=8,
    max_steps=20,
    default_guidance=1.0,
    guidance_param="guidance_scale",
    aspect_ratios=dict(_ERNIE_RATIOS),
    arbitrary_size=False,
    dim_multiple=16,
    peak_vram_mb=21500,
    cpu_offload=True,
    load_timeout_s=2400,
    seconds_per_step_hint=10.0,
    notes="DMD+RL дистилляция, CFG 1.0. Внутрипайплайновый PE (Ministral-3.3B, +6 ГБ) "
    "отключён принудительно: 22+6 ГБ не помещаются в 24 ГБ.",
)

ERNIE_IMAGE_SFT = ModelSpec(
    id="ernie-image",
    backend="ernie",
    display_name="ERNIE-Image SFT",
    repo_id="baidu/ERNIE-Image",
    variant="sft",
    default_steps=50,
    max_steps=100,
    default_guidance=4.0,
    guidance_param="guidance_scale",
    aspect_ratios=dict(_ERNIE_RATIOS),
    arbitrary_size=False,
    dim_multiple=16,
    peak_vram_mb=21500,
    cpu_offload=True,
    load_timeout_s=2400,
    seconds_per_step_hint=10.0,
    notes="Самая тяжёлая модель: 22 ГБ пика + 2 ГБ резерва = ровно 24 ГБ на node-a. "
    "Отказы vram_unavailable здесь ожидаемы — это и есть остаточный принцип.",
)

# --- HiDream-O1 -------------------------------------------------------------
# Не diffusers: Qwen3VLForConditionalGeneration + вендорный models/pipeline.py.
# device_map="cuda" без offload — единственная модель, которая при выгрузке
# реально возвращает VRAM (16.4 ГБ).
HIDREAM_O1 = ModelSpec(
    id="hidream-o1",
    backend="hidream",
    display_name="HiDream-O1-Image",
    local_path=os.environ.get(
        "IMG_HIDREAM_MODEL_PATH",
        os.path.join(PROJECTS_ROOT, "HiDream-O1-Image", "weights", "HiDream-O1-Image"),
    ),
    default_steps=50,
    max_steps=100,
    default_guidance=5.0,
    guidance_param="guidance_scale",
    default_shift=3.0,
    schedulers=("default", "flow_match", "flash"),
    default_scheduler="default",
    aspect_ratios={
        "1:1": (2048, 2048),
        "4:3": (2304, 1728),
        "3:4": (1728, 2304),
        "16:9": (2560, 1440),
        "9:16": (1440, 2560),
        "3:2": (2496, 1664),
        "2:3": (1664, 2496),
        "21:9": (3104, 1312),
        "9:21": (1312, 3104),
        "9:7": (2304, 1792),
        "7:9": (1792, 2304),
    },
    arbitrary_size=True,
    dim_multiple=64,
    min_dim=512,
    max_dim=2048,
    supports_negative_prompt=False,
    # Замерено 2026-08-03: пик 18375 МБ, в покое держит 15883 МБ.
    peak_vram_mb=19300,
    weights_vram_mb=15900,
    cpu_offload=False,
    load_timeout_s=2400,
    seconds_per_step_hint=3.0,
    notes="Пресеты 16:9/21:9/9:16/9:21 превышают max_dim=2048 и доступны только "
    "через aspect_ratio, но не через явные width/height. "
    "Вендорный pipeline примагничивает произвольный размер к ближайшему нативному: "
    "запрос 1024x1024 отдаёт 2048x2048 — проверено. Поэтому фактический размер "
    "берётся у готовой картинки, а не из запроса.",
)


MODELS: dict[str, ModelSpec] = {
    m.id: m
    for m in (
        QWEN_IMAGE,
        Z_IMAGE_TURBO,
        Z_IMAGE_BASE,
        ERNIE_IMAGE_TURBO,
        ERNIE_IMAGE_SFT,
        HIDREAM_O1,
    )
}

# Какие модели обслуживает каждый бэкенд-процесс.
MODELS_BY_BACKEND: dict[str, tuple[str, ...]] = {
    backend: tuple(m.id for m in MODELS.values() if m.backend == backend)
    for backend in BACKENDS
}

# Порт воркера на локальной петле. На node-b те же порты, наружу — через ssh -L
# на 18101..18104 (см. /etc/ssh-tunnels/node-b.env).
DEFAULT_WORKER_PORTS: dict[str, int] = {
    "qwen": 8101,
    "zimage": 8102,
    "ernie": 8103,
    "hidream": 8104,
}


def get_spec(model_id: str) -> ModelSpec:
    """Вернуть спецификацию или бросить KeyError с внятным списком кандидатов."""
    try:
        return MODELS[model_id]
    except KeyError:
        raise KeyError(
            "unknown model %r; known models: %s" % (model_id, ", ".join(sorted(MODELS)))
        ) from None


def models_for_backend(backend: str) -> tuple[str, ...]:
    return MODELS_BY_BACKEND.get(backend, ())
