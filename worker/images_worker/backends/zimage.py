"""Бэкенд Z-Image (turbo и base).

Перенос /home/user/Z-Image-Turbo/src/z_image_api/model.py. Пайплайн собирается
вручную из отдельных файлов репозитория: готового from_pretrained для этой
раскладки нет.

Два исправления против оригинала:
  * REPO_ID был захардкожен в классе, а ZI_TURBO_REPO/ZI_BASE_REPO из .env
    читались и никуда не шли. Теперь источник один — spec.repo_id/spec.files.
  * turbo и base держались в VRAM одновременно (живой /health показывал
    {"turbo": true, "base": true}). Base.load() выгружает предыдущую модель.
"""

from __future__ import annotations

import json
import logging

import torch

from ..protocol import GenerateJob
from ..registry import ModelSpec
from .base import Backend
from .qwen import supported_call_kwargs

log = logging.getLogger("images_worker.zimage")


class ZImageBackend(Backend):
    name = "zimage"

    def _load_model(self, spec: ModelSpec):
        from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
        from diffusers.models import AutoencoderKL, ZImageTransformer2DModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

        def fetch(key: str) -> str:
            filename = spec.files[key]
            log.info("fetching %s", filename)
            return hf_hub_download(repo_id=spec.repo_id, filename=filename)

        dtype = torch.bfloat16

        transformer_file = fetch("transformer")
        vae_file = fetch("vae")
        vae_config_file = fetch("vae_config")
        scheduler_config_file = fetch("scheduler_config")

        te_file = fetch("text_encoder")
        te_config_file = fetch("text_encoder_config")
        tokenizer_config_file = fetch("tokenizer_config")
        for key in ("tokenizer", "vocab", "merges", "special_tokens"):
            fetch(key)

        with open(scheduler_config_file, encoding="utf-8") as fh:
            scheduler_config = json.load(fh)
        with open(vae_config_file, encoding="utf-8") as fh:
            vae_config = json.load(fh)
        with open(te_config_file, encoding="utf-8") as fh:
            te_config = json.load(fh)

        scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

        import os

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.dirname(tokenizer_config_file), trust_remote_code=True
        )

        log.info("building text encoder (Qwen3)")
        text_encoder = Qwen3ForCausalLM(Qwen3Config(**te_config))
        text_encoder.load_state_dict(load_file(te_file), strict=False)
        text_encoder = text_encoder.to(dtype)

        log.info("building VAE")
        vae_model_config = {k: v for k, v in vae_config.items() if not k.startswith("_")}
        vae = AutoencoderKL(**vae_model_config)
        vae.load_state_dict(load_file(vae_file))
        # VAE держим во float32: в bf16 заметно падает качество декода.
        vae = vae.to(torch.float32)

        log.info("building transformer")
        transformer = ZImageTransformer2DModel.from_single_file(transformer_file, torch_dtype=dtype)

        pipe = ZImagePipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        pipe.enable_model_cpu_offload()
        return pipe

    def _generate(self, spec: ModelSpec, job: GenerateJob):
        seed = self._seed(job)
        accepted = supported_call_kwargs(self._pipe)

        kwargs = {
            "prompt": job.prompt,
            # Пустую строку отдаём как None: turbo работает без CFG вовсе,
            # и пустой negative_prompt только зря гоняет второй проход.
            "negative_prompt": job.negative_prompt or None,
            "width": job.width,
            "height": job.height,
            "num_inference_steps": job.num_inference_steps,
            "guidance_scale": job.guidance_scale,
            "generator": self._generator(seed),
        }
        if "callback_on_step_end" in accepted:
            kwargs["callback_on_step_end"] = self._step_callback()

        self._raise_if_cancelled()
        result = self._pipe(**kwargs)
        image = result.images[0]
        del result
        return image, seed
