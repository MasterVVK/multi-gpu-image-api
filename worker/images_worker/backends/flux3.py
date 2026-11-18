"""Бэкенд FLUX.3 Image Klein (Black Forest Labs), базовый чекпойнт без дистилляции.

Пайплайн модульный (diffusers `ModularPipeline`, блоки `Flux3AutoBlocks`). Перед
генерацией он по умолчанию гоняет отдельную 9B-модель — апсемплер промпта
(`flux-3-image-klein-pup`). Её здесь нет: шлюзу приходят уже полные описания сцены,
которые пишет LLM клиента, а для готовых описаний авторы модели сами советуют снять
шаг `prompt_upsampler`, чтобы текст не расширялся второй раз.

Каталог модели собирается на лету во временном каталоге: ссылки на подкаталоги
снапшота в кэше HF (transformer, text_encoder, vae) и свой `modular_model_index.json`
без апсемплера. Так загрузка идёт офлайн и не зависит от того, лежат ли в кэше индекс
репозитория и веса апсемплера.

Отмена задачи: у модульного пайплайна нет `callback_on_step_end`, поэтому проверка
встроена в progress_bar цикла денойза — его `update()` вызывается на каждом шаге.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import torch

from ..protocol import GenerateJob
from ..registry import ModelSpec
from .base import Backend, _hf_cache_dir

log = logging.getLogger("images_worker.flux3")

# Компоненты репозитория без апсемплера; путь подставляется при загрузке.
_COMPONENTS = {
    "text_encoder": ("transformers", "Qwen3VLForConditionalGeneration", "text_encoder"),
    "tokenizer": ("transformers", "Qwen3VLProcessor", "text_encoder"),
    "transformer": ("diffusers", "Flux3Transformer2DModel", "transformer"),
    "vae": ("diffusers", "AutoencoderKLFlux2", "vae"),
}


def _snapshot_dir(repo_id: str) -> str:
    """Каталог снапшота в кэше HF по refs/main — без проверки полноты.

    snapshot_download(local_files_only=True) сверяет снапшот со всем списком файлов
    репозитория и отказывает: на узле нарочно скачана только часть (без однофайлового
    дубля, fp8/nvfp4 и индекса с апсемплером).
    """
    repo_dir = os.path.join(_hf_cache_dir(), "models--" + repo_id.replace("/", "--"))
    with open(os.path.join(repo_dir, "refs", "main"), encoding="utf-8") as fh:
        commit = fh.read().strip()
    snapshot = os.path.join(repo_dir, "snapshots", commit)
    missing = [sub for _, _, sub in _COMPONENTS.values() if not os.path.isdir(os.path.join(snapshot, sub))]
    if missing:
        raise RuntimeError("в снапшоте %s нет %s" % (snapshot, ", ".join(sorted(set(missing)))))
    return snapshot


def _model_dir(snapshot: str) -> str:
    root = tempfile.mkdtemp(prefix="flux3-")
    index: dict = {"_blocks_class_name": "Flux3AutoBlocks", "_class_name": "Flux3ModularPipeline"}
    for name, (library, cls, subfolder) in _COMPONENTS.items():
        link = os.path.join(root, subfolder)
        if not os.path.exists(link):
            os.symlink(os.path.join(snapshot, subfolder), link)
        index[name] = [library, cls, {"pretrained_model_name_or_path": root, "revision": None,
                                      "subfolder": subfolder, "type_hint": [library, cls], "variant": None}]
    with open(os.path.join(root, "modular_model_index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    return root


class _CancellableBar:
    """Обёртка над tqdm цикла денойза: перед каждым шагом — проверка отмены."""

    def __init__(self, bar, check):
        self._bar, self._check = bar, check

    def __enter__(self):
        self._bar.__enter__()
        return self

    def __exit__(self, *exc):
        return self._bar.__exit__(*exc)

    def update(self, n=1):
        self._check()
        return self._bar.update(n)


class Flux3Backend(Backend):
    name = "flux3"

    def _hook_cancel(self, blocks) -> int:
        """Подменяет progress_bar у циклических блоков (денойз). Возвращает, сколько подменено."""
        from diffusers.modular_pipelines.modular_pipeline import LoopSequentialPipelineBlocks

        hooked = 0
        stack = [blocks]
        while stack:
            block = stack.pop()
            if isinstance(block, LoopSequentialPipelineBlocks):
                plain = block.progress_bar

                def cancellable(iterable=None, total=None, _plain=plain):
                    return _CancellableBar(_plain(iterable=iterable, total=total), self._raise_if_cancelled)

                block.progress_bar = cancellable
                hooked += 1
            stack.extend(getattr(block, "sub_blocks", {}).values())
        return hooked

    def _load_model(self, spec: ModelSpec):
        from diffusers import ModularPipeline

        snapshot = _snapshot_dir(spec.repo_id)
        root = _model_dir(snapshot)
        log.info("loading %s from %s (via %s)", spec.repo_id, snapshot, root)

        blocks = ModularPipeline.from_pretrained(root).blocks
        if "prompt_upsampler" in blocks.sub_blocks:
            blocks.sub_blocks.pop("prompt_upsampler")
        pipe = blocks.init_pipeline(root)
        # VAE в fp32 — так в карточке модели; остальное bf16.
        pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
        if spec.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        if not self._hook_cancel(pipe.blocks):
            log.warning("denoise loop not found: cancellation works only between jobs")
        return pipe

    def _generate(self, spec: ModelSpec, job: GenerateJob):
        from diffusers import ClassifierFreeGuidance

        seed = self._seed(job)
        if getattr(self._pipe.guider, "guidance_scale", None) != job.guidance_scale:
            self._pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=job.guidance_scale))

        kwargs = {
            "prompt": job.prompt,
            "width": job.width,
            "height": job.height,
            "num_inference_steps": job.num_inference_steps,
            "generator": self._generator(seed),
            "output": "images",
        }
        if job.negative_prompt:
            kwargs["negative_prompt"] = job.negative_prompt

        self._raise_if_cancelled()
        images = self._pipe(**kwargs)
        image = images[0]
        del images
        return image, seed
