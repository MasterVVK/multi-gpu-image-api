"""Бэкенд MOSS-Image (OpenMOSS): 42-слойный DiT, энкодер Qwen3.5, VAE от FLUX.2.

Пайплайн у модели свой — файл `pipeline_moss_image.py` лежит в репозитории весов
и подключается через `custom_pipeline` + `trust_remote_code`. Это исполнение
кода из репозитория модели, поэтому воркер работает с HF_HUB_OFFLINE=1: код
берётся из того же зафиксированного снапшота в кэше, что и веса, и сам собой
не обновится.

Venv общий с qwen21 (тот же стек: torch 2.9, свежие diffusers и transformers),
процесс — свой: на процесс по-прежнему одна резидентная модель.

Как и у Qwen-Image 2.1, текстовый энкодер (~17 ГБ в bf16) на карту не попадает:
он живёт в системной RAM в fp32 и кодирует промпт на процессоре. Иначе пик VRAM
задавал бы он, а не трансформер, и модель не помещалась бы рядом с соседями.
Исключение — moss-image-full для узла с большой картой: там всё на GPU, и промпт
кодирует сам пайплайн.

Чего у пайплайна нет и как это закрыто:
  * нет `callback_on_step_end` — отмена задачи встроена в его progress_bar,
    единственное место, через которое проходит каждый шаг денойза;
  * кодирование промпта привязано к устройству пайплайна, поэтому для CPU оно
    повторено здесь (шаблон чата с enable_thinking, без паддинга) — см.
    `MOSSImagePipeline.encode_prompt` в снапшоте f972bd4b.
"""

from __future__ import annotations

import logging
import time

import torch

from ..protocol import GenerateJob
from ..registry import ModelSpec
from .base import Backend

log = logging.getLogger("images_worker.moss")

_CPU_TE_ATTR = "_images_worker_cpu_text_encoder"
_MAX_SEQUENCE_LENGTH = 4096  # дефолт пайплайна


class MossBackend(Backend):
    name = "moss"

    def _load_model(self, spec: ModelSpec):
        from diffusers import DiffusionPipeline
        from huggingface_hub import snapshot_download

        # Каталог снапшота, а не id репозитория: с custom_pipeline diffusers по id
        # идёт в сеть за ревизией даже при полном кэше и в офлайн-режиме падает.
        # Локальный каталог он читает целиком с диска, включая файл пайплайна.
        model_dir = snapshot_download(spec.repo_id, local_files_only=True)
        log.info("loading %s from %s", spec.repo_id, model_dir)
        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            custom_pipeline="pipeline_moss_image",
            trust_remote_code=True,
            # Трансформер считает в bf16, VAE и начальный шум — в fp32: так велит карточка модели.
            torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
        )
        # Из карточки модели: планировщик пересоздаётся из собственного конфига.
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)

        if not spec.cpu_offload:
            # Большая карта (moss-image-full): всё на GPU, энкодер тоже — промпт
            # кодирует сам пайплайн (_generate отдаёт строки, а не эмбеддинги).
            pipe.to("cuda")
            setattr(pipe, _CPU_TE_ATTR, None)
        else:
            # Энкодер снимаем с пайплайна до offload-хуков — та же причина, что в qwen21.py.
            cpu_text_encoder = pipe.text_encoder
            pipe.register_modules(text_encoder=None)
            cpu_text_encoder.to(device="cpu", dtype=torch.float32)
            cpu_text_encoder.eval()

            pipe.enable_model_cpu_offload()
            setattr(pipe, _CPU_TE_ATTR, cpu_text_encoder)

        # Точка отмены: пайплайн перебирает шаги через self.progress_bar(...).
        plain_bar = pipe.progress_bar

        def cancellable_bar(iterable=None, total=None):
            for item in plain_bar(iterable, total):
                self._raise_if_cancelled()
                yield item

        pipe.progress_bar = cancellable_bar
        return pipe

    def _encode_on_cpu(self, text: str):
        pipe = self._pipe
        encoder = getattr(pipe, _CPU_TE_ATTR)
        started = time.monotonic()
        formatted = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = pipe.tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=_MAX_SEQUENCE_LENGTH
        )
        with torch.no_grad():
            output = encoder(**inputs, use_cache=False, return_dict=True)
        embeds = output.last_hidden_state  # [1, S, C]: один промпт, паддинга нет
        log.info("prompt encoded on cpu in %.1fs (%d tokens)", time.monotonic() - started, embeds.shape[1])
        device = pipe._execution_device
        mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=device)
        return embeds.to(device=device, dtype=torch.bfloat16), mask

    def _generate(self, spec: ModelSpec, job: GenerateJob):
        seed = self._seed(job)
        kwargs = {
            "height": job.height,
            "width": job.width,
            "num_inference_steps": job.num_inference_steps,
            "guidance_scale": job.guidance_scale,
            "generator": self._generator(seed),
        }

        self._raise_if_cancelled()
        if getattr(self._pipe, _CPU_TE_ATTR, None) is None:
            # Энкодер на карте: пайплайн кодирует промпт сам, своим же шаблоном.
            kwargs["prompt"] = job.prompt
            kwargs["negative_prompt"] = job.negative_prompt or ""
            if job.guidance_scale == 1.0 and job.negative_prompt:
                log.warning("negative_prompt ignored: guidance_scale=1.0 turns CFG off")
            result = self._pipe(**kwargs)
            image = result.images[0]
            del result
            return image, seed

        kwargs["prompt_embeds"], kwargs["prompt_attention_mask"] = self._encode_on_cpu(job.prompt)
        # CFG у пайплайна включён при любом guidance_scale, кроме 1.0, и тогда ему
        # нужен negative — пустая строка допустима и является его дефолтом.
        if job.guidance_scale != 1.0:
            kwargs["negative_prompt_embeds"], kwargs["negative_prompt_attention_mask"] = self._encode_on_cpu(
                job.negative_prompt or ""
            )
        elif job.negative_prompt:
            log.warning("negative_prompt ignored: guidance_scale=1.0 turns CFG off")
        self._raise_if_cancelled()

        result = self._pipe(**kwargs)
        image = result.images[0]
        del result
        return image, seed
