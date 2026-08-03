"""Бэкенд Qwen-Image.

Перенос /home/user/qi-api/src/qi_api/model.py с одним содержательным исправлением:
старый код принимал `cfg_scale`, валидировал его, складывал в TaskInfo — и не
передавал в пайплайн (model.py:185-192). Пайплайн всегда работал на своём
дефолте `true_cfg_scale=4.0`, а значение из запроса молча терялось.

Следствие фикса: те же запросы с cfg_scale != 4.0 при том же seed теперь дают
другую картинку. Это ожидаемо и зафиксировано в плане.
"""

from __future__ import annotations

import inspect
import logging
import os

import torch

from ..protocol import GenerateJob
from ..registry import ModelSpec
from .base import Backend

log = logging.getLogger("images_worker.qwen")


def supported_call_kwargs(pipe) -> set:
    """Какие kwargs принимает __call__ конкретного пайплайна.

    В четырёх venv стоят разные diffusers (0.36.0 / 0.38.0.dev0), и набор
    параметров у них расходится. Дешевле спросить сигнатуру, чем городить
    ветвления по версиям.
    """
    try:
        return set(inspect.signature(pipe.__call__).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        return set()


class QwenBackend(Backend):
    name = "qwen"

    def _load_model(self, spec: ModelSpec):
        # Квантизация — решение уровня хоста, не модели: на node-a в .env стоит
        # QI_QUANTIZATION=none (веса уже FP8), на другой карте может понадобиться nf4.
        quantization = os.environ.get("IMG_QWEN_QUANTIZATION", "none").strip().lower()
        dtype = torch.bfloat16
        model_id = spec.repo_id

        if quantization in ("nf4", "int8"):
            from diffusers import BitsAndBytesConfig as DiffusersBnb
            from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
            from transformers import BitsAndBytesConfig as TransformersBnb
            from transformers import Qwen2_5_VLForConditionalGeneration

            is_int8 = quantization == "int8"
            log.info("loading transformer with %s", quantization)

            if is_int8:
                tr_quant = DiffusersBnb(
                    load_in_8bit=True,
                    # Пропуск этого слоя критичен для качества — проверено в исходном проекте.
                    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
                )
                te_quant = TransformersBnb(load_in_8bit=True)
            else:
                tr_quant = DiffusersBnb(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
                )
                te_quant = TransformersBnb(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                )

            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_id, subfolder="transformer", quantization_config=tr_quant, torch_dtype=dtype
            )
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, subfolder="text_encoder", quantization_config=te_quant, torch_dtype=dtype
            )
            pipe = QwenImagePipeline.from_pretrained(
                model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype
            )
        else:
            from diffusers import DiffusionPipeline

            log.info("loading pre-quantized/full-precision %s", model_id)
            pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

        # 24 ГБ для FP8 Qwen-Image — впритык, offload обязателен.
        # Веса живут в системной RAM (её тут 184 ГБ) и приезжают на карту помодульно.
        pipe.enable_model_cpu_offload()
        return pipe

    def _generate(self, spec: ModelSpec, job: GenerateJob):
        seed = self._seed(job)
        accepted = supported_call_kwargs(self._pipe)

        kwargs = {
            "prompt": job.prompt,
            "width": job.width,
            "height": job.height,
            "num_inference_steps": job.num_inference_steps,
            "generator": self._generator(seed),
        }
        if job.negative_prompt:
            kwargs["negative_prompt"] = job.negative_prompt

        # Вот тот самый фикс: CFG наконец доезжает до пайплайна.
        if spec.guidance_param in accepted:
            kwargs[spec.guidance_param] = job.guidance_scale
        else:  # pragma: no cover — страховка на случай смены сигнатуры diffusers
            log.warning("pipeline has no %s parameter; CFG ignored", spec.guidance_param)

        if "callback_on_step_end" in accepted:
            kwargs["callback_on_step_end"] = self._step_callback()

        self._raise_if_cancelled()
        result = self._pipe(**kwargs)
        image = result.images[0]
        del result
        return image, seed
