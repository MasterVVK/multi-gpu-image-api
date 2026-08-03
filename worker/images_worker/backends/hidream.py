"""Бэкенд HiDream-O1-Image.

Перенос <корень HiDream>/src/hidream_o1_api/model.py. Это не
diffusers: Qwen3VLForConditionalGeneration + вендорный models/pipeline.py из
корня репозитория, который добавляется в sys.path.

Единственная из шести моделей без cpu-offload (`device_map="cuda"`), то есть
единственная, у которой выгрузка реально возвращает VRAM — все 16.4 ГБ.
Для остальных «выгрузка» освобождает системную RAM, а карту защищает
эксклюзивный lease планировщика.

Осторожно с именами: в корне HiDream лежат модуль prompt_agent.py
и пакет models/, и после sys.path.insert они видны глобально. В images_worker
не должно появиться модулей с такими именами.
"""

from __future__ import annotations

import logging
import os
import sys

import torch

from ..protocol import GenerateJob
from ..registry import PROJECTS_ROOT, ModelSpec
from .base import Backend

log = logging.getLogger("images_worker.hidream")

# Корень вендорного кода HiDream. Переопределяется переменной окружения.
REPO_DIR = os.environ.get(
    "IMG_HIDREAM_REPO_DIR", os.path.join(PROJECTS_ROOT, "HiDream-O1-Image")
)


class HiDreamBackend(Backend):
    name = "hidream"

    def __init__(self) -> None:
        super().__init__()
        self._processor = None

    def _load_model(self, spec: ModelSpec):
        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)

        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        from transformers import AutoProcessor, PreTrainedTokenizerBase

        model_path = spec.local_path
        log.info("loading %s", model_path)

        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()

        # Спецтокены пайплайн ждёт на токенизаторе, но в конфиге их нет.
        tokenizer = processor if isinstance(processor, PreTrainedTokenizerBase) else processor.tokenizer
        tokenizer.boi_token = "<|boi_token|>"
        tokenizer.bor_token = "<|bor_token|>"
        tokenizer.eor_token = "<|eor_token|>"
        tokenizer.bot_token = "<|bot_token|>"
        tokenizer.tms_token = "<|tms_token|>"

        self._processor = processor
        return model

    def _unload_locked(self):
        self._processor = None
        return super()._unload_locked()

    def _generate(self, spec: ModelSpec, job: GenerateJob):
        from models.pipeline import generate_image

        # Свой диапазон seed: пайплайн кладёт его в torch.manual_seed, а
        # 2**32-1 у него переполняется — оригинал использовал 2**31-1.
        seed = job.seed if job.seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())

        scheduler_name = job.scheduler or spec.default_scheduler or "default"
        shift = job.shift if job.shift is not None else (spec.default_shift or 3.0)

        extra = {}
        if scheduler_name == "flash":
            extra["noise_scale_start"] = 7.5
            extra["noise_scale_end"] = 7.5
            extra["noise_clip_std"] = 2.5

        def _callback(_step_idx, _total, _preview=None):
            # Отмена проходит только потому, что GenerationCancelled — BaseException:
            # pipeline.py заворачивает этот вызов в `except Exception: pass`.
            self._raise_if_cancelled()

        self._raise_if_cancelled()
        image = generate_image(
            model=self._pipe,
            processor=self._processor,
            prompt=job.prompt,
            ref_image_paths=[],
            height=job.height,
            width=job.width,
            num_inference_steps=job.num_inference_steps,
            guidance_scale=job.guidance_scale,
            shift=shift,
            timesteps_list=None,
            scheduler_name=scheduler_name,
            seed=seed,
            keep_original_aspect=False,
            callback=_callback,
            **extra,
        )
        return image, seed
