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
# развернуть проект на другой машине можно было без правки кода. По умолчанию —
# домашний каталог пользователя, от которого запущен процесс.
PROJECTS_ROOT = os.environ.get("IMG_PROJECTS_ROOT", os.path.expanduser("~"))

# Бэкенд = отдельный процесс со своим venv. Один бэкенд обслуживает 1..N моделей,
# но резидентной в каждый момент держит ровно одну.
BACKENDS = ("qwen", "zimage", "ernie", "hidream", "qwen21", "moss", "minimax")

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
    # У части пайплайнов negative_prompt работает только при включённом CFG:
    # при guidance_scale не выше этого порога он молча игнорируется. Шлюз в
    # таком случае отказывает с 422 — принять параметр и не применить нельзя.
    # None — ограничения нет.
    negative_needs_guidance_above: float | None = None

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
    # Класс карты, под которую рассчитана модель: её пик обязан помещаться в такую карту.
    # По умолчанию 24 ГБ (3090/4090) — это инвариант tests/test_registry.py для всех
    # обычных моделей. Больше — только у моделей, живущих на отдельном большом узле.
    card_class_mb: int = 24576

    notes: str = ""

    @property
    def default_size(self) -> tuple[int, int]:
        return self.aspect_ratios[self.default_aspect_ratio]

    @property
    def fixed_steps(self) -> bool:
        """Число шагов зашито в чекпойнт и из запроса не берётся."""
        return self.max_steps == self.default_steps

    @property
    def fixed_guidance(self) -> bool:
        """Guidance зашит (дистилляция без CFG) и в пайплайн не передаётся."""
        return self.guidance_param == "none"


# --- Qwen-Image -------------------------------------------------------------
# Источник: $IMG_PROJECTS_ROOT/qi-api/.env → QI_MODEL_ID=Ilus-AI/Qwen-Image-2512-FP8,
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
    negative_needs_guidance_above=1.0,
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

# --- Qwen-Image 2.1 ---------------------------------------------------------
# Преемник 2512: 7B DiT, родное разрешение 2K, энкодер Qwen3-VL. Свой бэкенд и
# свой venv — стек (QwenImage21Pipeline, transformers >= 5.17) несовместим с
# venv 2512. Пресеты — из карточки модели, все стороны кратны 32.
# variant="fp8" — веса трансформера хранятся в float8 (layerwise casting);
# от этого зависит пик VRAM, поэтому режим задаётся здесь, а не окружением.
# Лицензия весов — Qwen Research License (только некоммерческое использование),
# в отличие от Apache-2.0 у 2512.
_QWEN21_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

QWEN_IMAGE_21 = ModelSpec(
    id="qwen-image-2.1",
    backend="qwen21",
    display_name="Qwen-Image 2.1",
    repo_id="Qwen/Qwen-Image-2.1",
    variant="fp8",
    # Рекомендация Qwen: 40 шагов без CFG. guidance_scale > 1 включает настоящий
    # CFG (true_cfg_scale + negative_prompt) и удваивает работу на шаг.
    default_steps=40,
    max_steps=100,
    default_guidance=1.0,
    guidance_param="true_cfg_scale",
    negative_needs_guidance_above=1.0,
    aspect_ratios=dict(_QWEN21_RATIOS),
    default_aspect_ratio="1:1",
    arbitrary_size=True,
    dim_multiple=32,
    min_dim=512,
    max_dim=2048,
    # Замерено 2026-11-17 на RTX 3090, пресет 16:9 (2752x1536), 40 шагов:
    # пик 10113 МБ чистыми, +5%. Это режим «энкодер на CPU, трансформер в fp8»
    # (см. backends/qwen21.py); с весами bf16 пик был бы 16.8 ГБ.
    peak_vram_mb=10700,
    cpu_offload=True,
    load_timeout_s=2400,
    # 277 с на 40 шагов вместе с кодированием промпта на CPU (10..20 с) и декодом.
    seconds_per_step_hint=7.0,
    notes="CFG по умолчанию выключен (guidance_scale=1.0); negative_prompt действует "
    "только при guidance_scale > 1. Пресеты 4:3/3:2/16:9 и обратные превышают "
    "max_dim=2048 и доступны только через aspect_ratio.",
)

# Тот же чекпойнт с весами трансформера как есть (bf16), без fp8-хранения.
# Качество — эталонное, цена — пик 16.8 ГБ вместо 10.1: на карте, которую делят
# с соседями, эта модель запускается, только когда соседи ушли. Клиенту, которому
# важнее результат, чем время, имеет смысл просить её первой и откатываться на
# qwen-image-2.1 при отказе vram_unavailable.
QWEN_IMAGE_21_BF16 = ModelSpec(
    id="qwen-image-2.1-bf16",
    backend="qwen21",
    display_name="Qwen-Image 2.1 (полные веса bf16)",
    repo_id="Qwen/Qwen-Image-2.1",
    variant=None,
    default_steps=40,
    max_steps=100,
    default_guidance=1.0,
    guidance_param="true_cfg_scale",
    negative_needs_guidance_above=1.0,
    aspect_ratios=dict(_QWEN21_RATIOS),
    default_aspect_ratio="1:1",
    arbitrary_size=True,
    dim_multiple=32,
    min_dim=512,
    max_dim=2048,
    # Замерено 2026-11-17 на RTX 3090, пресет 16:9 (2752x1536), 40 шагов,
    # энкодер на CPU: пик 16823 МБ чистыми, +5%.
    peak_vram_mb=17700,
    cpu_offload=True,
    load_timeout_s=2400,
    # 290 с на 40 шагов вместе с кодированием промпта и декодом.
    seconds_per_step_hint=7.5,
    notes="Полные веса трансформера (bf16): тот же результат, что у разработчика, "
    "но пик VRAM 16.8 ГБ против 10.1 ГБ у qwen-image-2.1. Остальное — как у qwen-image-2.1.",
)

# Тот же чекпойнт для большой карты: энкодер, трансформер и VAE лежат на GPU
# целиком и постоянно, системная RAM хоста почти не занята. Для узла с картой
# на 96 ГБ, где видеопамяти много, а оперативную нужно беречь. На 24 ГБ модель
# не помещается по определению, и узлы с такими картами её не объявляют.
QWEN_IMAGE_21_FULL = ModelSpec(
    id="qwen-image-2.1-full",
    backend="qwen21",
    display_name="Qwen-Image 2.1 (bf16, всё на карте)",
    repo_id="Qwen/Qwen-Image-2.1",
    variant=None,
    default_steps=40,
    max_steps=100,
    default_guidance=1.0,
    guidance_param="true_cfg_scale",
    negative_needs_guidance_above=1.0,
    aspect_ratios=dict(_QWEN21_RATIOS),
    default_aspect_ratio="1:1",
    arbitrary_size=True,
    dim_multiple=32,
    min_dim=512,
    max_dim=2048,
    # Замерено 2026-11-17 на RTX PRO 6000 (узел node-c), пресет 16:9 (2752x1536),
    # 40 шагов: пик 39323 МБ чистыми по nvidia-smi, +5%. Генерация 89 с,
    # загрузка 42 с; системная RAM хоста выросла на 2.5 ГБ.
    peak_vram_mb=41300,
    weights_vram_mb=32000,
    cpu_offload=False,
    card_class_mb=98304,
    load_timeout_s=2400,
    # 89 с на 40 шагов на RTX PRO 6000; запас на случай медленного диска.
    seconds_per_step_hint=4.0,
    notes="Полные веса, все компоненты на карте: для узла с большой картой. "
    "Результат тот же, что у qwen-image-2.1-bf16; отличается только размещение.",
)

QWEN_IMAGE_21_TURBO = ModelSpec(
    id="qwen-image-2.1-turbo",
    backend="qwen21",
    display_name="Qwen-Image 2.1 Turbo",
    repo_id="Qwen/Qwen-Image-2.1-Turbo",
    variant="fp8",
    # Сетка сигм зашита в чекпойнт: шагов всегда 8, num_inference_steps на неё
    # не влияет. Поэтому max_steps == default_steps — другое число было бы
    # обещанием, которое пайплайн не выполнит.
    default_steps=8,
    max_steps=8,
    default_guidance=1.0,
    guidance_param="none",
    aspect_ratios=dict(_QWEN21_RATIOS),
    default_aspect_ratio="1:1",
    arbitrary_size=True,
    dim_multiple=32,
    min_dim=512,
    max_dim=2048,
    supports_negative_prompt=False,
    # Отдельно не замерена: архитектура и режим те же, что у base, а число
    # шагов на пик не влияет — берём его замер.
    # Уточнить: scripts/measure_vram.py qwen-image-2.1-turbo.
    peak_vram_mb=10700,
    cpu_offload=True,
    load_timeout_s=2400,
    # Шагов всего 8, а кодирование промпта на CPU и декод стоят столько же,
    # сколько у base, — поэтому на шаг закладываем с запасом.
    seconds_per_step_hint=10.0,
    notes="Дистилляция на 8 шагов без CFG: число шагов и guidance зашиты в чекпойнт "
    "и из запроса не берутся, negative_prompt не принимается.",
)

# --- MOSS-Image --------------------------------------------------------------
# OpenMOSS, Apache-2.0. SFT — дообученная версия Base с тем же пайплайном.
# Пайплайн свой (pipeline_moss_image.py в репозитории весов), venv общий с qwen21.
# Рекомендации авторов: 1–2 Мп на картинку, от 50 шагов, CFG 3.0 (3..6), стороны
# кратны 16. Пресеты взяты на ~2 Мп.
MOSS_IMAGE = ModelSpec(
    id="moss-image",
    backend="moss",
    display_name="MOSS-Image SFT",
    repo_id="OpenMOSS-Team/MOSS-Image-SFT",
    default_steps=50,
    max_steps=100,
    default_guidance=3.0,
    guidance_param="guidance_scale",
    negative_needs_guidance_above=1.0,
    aspect_ratios={
        "1:1": (1408, 1408),
        "16:9": (1920, 1088),
        "9:16": (1088, 1920),
        "4:3": (1664, 1248),
        "3:4": (1248, 1664),
        "3:2": (1728, 1152),
        "2:3": (1152, 1728),
    },
    arbitrary_size=True,
    dim_multiple=16,
    min_dim=512,
    max_dim=2048,
    # Замерено 2026-11-17 на RTX 3090, пресет 16:9 (1920x1088), 50 шагов, CFG 3.0:
    # пик 10772 МБ по счётчику torch (reserved) + ~340 МБ CUDA-контекста, +5%.
    # По nvidia-smi снять не удалось: во время прогона карту делили соседи.
    # Уточнить: scripts/measure_vram.py moss-image на свободной карте.
    peak_vram_mb=11700,
    cpu_offload=True,
    load_timeout_s=2400,
    # 290 с на 50 шагов вместе с двумя кодированиями на CPU (промпт и negative, ~45 с).
    seconds_per_step_hint=6.0,
    notes="CFG включён при любом guidance_scale, кроме 1.0 (два прохода на шаг); "
    "пустой negative_prompt допустим.",
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
    # Замерено 2026-08-03: пик 16471 МБ чистыми, +5%. Это УЖЕ без встроенного
    # prompt-enhancer'а: бэкенд выбрасывает его при загрузке. С ним пик уходил
    # за 23.5 ГБ и модель падала с OOM на 24-гигабайтной карте.
    peak_vram_mb=17200,
    cpu_offload=True,
    load_timeout_s=2400,
    seconds_per_step_hint=10.0,
    notes="DMD+RL дистилляция, CFG 1.0. Встроенный PE (Ministral-3.3B) выбрасывается "
    "при загрузке: он грузится как компонент пайплайна независимо от use_pe=False "
    "и стоил около 6 ГБ пика.",
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
    # Отдельно не замерена. Пайплайн тот же, что у turbo, и число шагов на пик
    # почти не влияет — берём его замер. Уточнить: scripts/measure_vram.py ernie-image.
    peak_vram_mb=17200,
    cpu_offload=True,
    load_timeout_s=2400,
    seconds_per_step_hint=10.0,
    notes="Тот же пайплайн, что у turbo, но 50 шагов вместо 8 — дольше, "
    "а по памяти столько же. Встроенный PE так же выбрасывается при загрузке.",
)

# Тот же чекпойнт ERNIE-Image SFT, но все компоненты (энкодер Mistral3 7.2 ГБ,
# трансформер 15 ГБ, VAE) резидентно на карте, без cpu_offload: для узла с большой
# картой, где RAM хоста не трогаем. На 24 ГБ не помещается — такие узлы её не объявляют.
ERNIE_IMAGE_FULL = ModelSpec(
    id="ernie-image-full",
    backend="ernie",
    display_name="ERNIE-Image SFT (всё на карте)",
    repo_id="baidu/ERNIE-Image",
    variant="sft",
    default_steps=50,
    max_steps=100,
    default_guidance=4.0,
    guidance_param="guidance_scale",
    aspect_ratios=dict(_ERNIE_RATIOS),
    arbitrary_size=False,
    dim_multiple=16,
    # Замерено 2026-11-18 на RTX PRO 6000 (узел node-c), пресет 16:9, CFG 4.0: пик
    # ~25400 МБ чистыми по nvidia-smi (соседи по карте шумят ±1 ГБ), резидентно
    # после загрузки ~23.5 ГБ. Загрузка 18 с, 50 шагов — 107 с; RAM хоста +2.3 ГБ.
    # Запас оставлен крупный: на 96 ГБ он ничего не стоит.
    peak_vram_mb=30000,
    weights_vram_mb=23000,
    cpu_offload=False,
    card_class_mb=98304,
    load_timeout_s=2400,
    seconds_per_step_hint=2.0,
    notes="Полные веса, все компоненты на карте: для узла с большой картой. "
    "Результат тот же, что у ernie-image; отличается только размещение.",
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


# --- MiniMax-H3-Image -------------------------------------------------------
# 33B DiT + 33B энкодер Qwen3-VL, по ~62 ГБ в bf16 каждый: модель не для 24 ГБ.
# Живёт на отдельном узле с картой 96 ГБ, в docker-контейнере. Пайплайн в diffusers
# только модульный; бэкенд держит энкодер и трансформер на карте по очереди и грузит
# их с диска на каждую задачу — см. backends/minimax.py.
# Пресеты — рекомендованные MiniMax, ~16.8 Мп каждый, все стороны кратны 32.
# Лицензия весов — MiniMax-H3-Image Community License Agreement (LICENSE в репозитории модели).
_MINIMAX_RATIOS = {
    "1:1": (4096, 4096),
    "4:3": (4736, 3552),
    "3:4": (3552, 4736),
    "3:2": (5024, 3360),
    "2:3": (3360, 5024),
    "16:9": (5472, 3072),
    "9:16": (3072, 5472),
    "21:9": (6272, 2688),
    "9:21": (2688, 6272),
}

MINIMAX_H3_IMAGE = ModelSpec(
    id="minimax-h3-image",
    backend="minimax",
    display_name="MiniMax-H3-Image",
    repo_id="MiniMaxAI/MiniMax-H3-Image",
    # Сетка из 6 сигм зашита в чекпойнт, у трансформера по выходной голове на шаг:
    # num_inference_steps не существует. max_steps == default_steps — как у
    # qwen-image-2.1-turbo: другое число было бы обещанием, которое пайплайн не выполнит.
    default_steps=6,
    max_steps=6,
    # Guidance дистиллирован: ни CFG, ни negative_prompt у модели нет.
    default_guidance=1.0,
    guidance_param="none",
    aspect_ratios=dict(_MINIMAX_RATIOS),
    default_aspect_ratio="1:1",
    # Произвольный размер — до 4096 по стороне (потолок схемы запроса), то есть не
    # больше пикселей, чем в пресетах: пик VRAM пресетов его покрывает. Меньше 4 Мп
    # модель всё равно считает на холсте 4 Мп и уменьшает результат.
    arbitrary_size=True,
    dim_multiple=32,
    min_dim=1024,
    max_dim=4096,
    supports_negative_prompt=False,
    # Замерено 2026-11-17 на RTX PRO 6000 (узел node-c): пресет 16:9 (5472x3072,
    # 16.8 Мп) — пик 68364 МБ чистыми по nvidia-smi (стадия трансформера+VAE;
    # стадия энкодера — 63.8 ГБ), +5%. На 2048x2048 пик 65.6 ГБ. Системная RAM
    # хоста за задачу выросла на 0.4–2 ГБ: веса читаются через mmap.
    peak_vram_mb=71800,
    # Резидентных весов нет: между задачами бэкенд не держит на карте ничего
    # (и сам выходит из процесса, отдавая CUDA-контекст).
    weights_vram_mb=0,
    cpu_offload=False,
    card_class_mb=98304,
    # /load только разбирает блоки пайплайна — секунды. Запас — на холодный импорт
    # diffusers/transformers в свежем контейнере.
    load_timeout_s=300,
    # Чтение ~130 ГБ с диска входит во время ГЕНЕРАЦИИ, а не загрузки, поэтому
    # «секунды на шаг» здесь — не про шаг денойза, а способ получить нужный тайм-аут
    # шлюза: 240 + 6 * 55 * 3 = 1230 с (20.5 мин). Замер 2026-11-17: 267 с на
    # 2048x2048 и 377 с на 5472x3072, из них денойз 5 с, остальное — чтение весов.
    seconds_per_step_hint=55.0,
    notes="6 шагов и guidance зашиты в чекпойнт: num_inference_steps и guidance_scale из "
    "запроса не берутся, negative_prompt не принимается. Пресеты ~16.8 Мп, стороны до 6272 "
    "доступны только через aspect_ratio (явные width/height ограничены 4096). Веса (~130 ГБ) "
    "читаются с диска на каждую задачу: генерация занимает минуты.",
)


MODELS: dict[str, ModelSpec] = {
    m.id: m
    for m in (
        QWEN_IMAGE,
        QWEN_IMAGE_21,
        QWEN_IMAGE_21_BF16,
        QWEN_IMAGE_21_FULL,
        QWEN_IMAGE_21_TURBO,
        MOSS_IMAGE,
        Z_IMAGE_TURBO,
        Z_IMAGE_BASE,
        ERNIE_IMAGE_TURBO,
        ERNIE_IMAGE_SFT,
        ERNIE_IMAGE_FULL,
        HIDREAM_O1,
        MINIMAX_H3_IMAGE,
    )
}

# Какие модели обслуживает каждый бэкенд-процесс.
MODELS_BY_BACKEND: dict[str, tuple[str, ...]] = {
    backend: tuple(m.id for m in MODELS.values() if m.backend == backend)
    for backend in BACKENDS
}

# Порт воркера на локальной петле. На node-b те же порты, наружу — через ssh -L
# на 18101..18106 (см. /etc/ssh-tunnels/node-b.env).
DEFAULT_WORKER_PORTS: dict[str, int] = {
    "qwen": 8101,
    "zimage": 8102,
    "ernie": 8103,
    "hidream": 8104,
    "qwen21": 8105,
    "moss": 8106,
    "minimax": 8107,
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
