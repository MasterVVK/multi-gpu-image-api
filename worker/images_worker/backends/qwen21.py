"""Бэкенд Qwen-Image 2.1 и 2.1-Turbo.

Отдельный бэкенд, а не ветка в qwen.py, потому что стек несовместим: нужен
`QwenImage21Pipeline` (diffusers из main) и transformers >= 5.17, а venv
Qwen-Image 2512 держит diffusers 0.36 / transformers 4.57. Один venv под обе
модели означал бы обновить стек под работающей 2512 — проще и безопаснее дать
новой модели свой процесс.

Чем 2.1 отличается от 2512 по вызову:
  * CFG выключен по умолчанию (`true_cfg_scale=1.0`) — так рекомендует Qwen.
    Включается только парой «true_cfg_scale > 1 + negative_prompt» и удваивает
    работу на шаг. Negative без CFG пайплайн игнорирует.
  * Turbo несёт свою сетку сигм в конфиге чекпойнта: число шагов (8) зашито,
    `num_inference_steps` на него не влияет, CFG и negative не применяются.
  * Стороны кратны 32, родное разрешение — 2K.

Как модель уложена в общую 24-гигабайтную карту (замеры 2026-11-17, RTX 3090,
2752x1536, 40 шагов, один и тот же seed):

    энкодер   трансформер   пик VRAM    генерация
    на карте  bf16          17.2 ГБ     300..308 с
    на CPU    bf16          16.8 ГБ     290 с
    на CPU    fp8           10.1 ГБ     277 с      <- так и работаем

  * Текстовый энкодер (17.5 ГБ в bf16) на карту не попадает вовсе: он живёт в
    системной RAM в fp32 и кодирует промпт на процессоре. Это 10..20 с на промпт
    (растёт с длиной: 9 с на 92 токена, 21 с на 227) против перекачки 17.5 ГБ на
    карту и обратно — выходит даже быстрее.
  * Веса трансформера хранятся в float8 и приводятся к bf16 послойно в момент
    вычисления (layerwise casting): тот же компромисс, что у 2512-FP8. На глаз
    картинки от bf16 не отличаются, а 10 ГБ вместо 17 — это разница между
    «помещается рядом с соседями по карте» и «ждёт, пока они уйдут».

Оба решения зашиты здесь и в `variant` реестра, а не вынесены в переменные
окружения: от них зависит пик VRAM, по которому планировщик решает, запускать
ли задачу, и расхождение настройки хоста с цифрой в реестре означало бы OOM
вместо честного отказа.
"""

from __future__ import annotations

import logging
import time

import torch

from ..protocol import GenerateJob
from ..registry import ModelSpec
from .base import Backend

log = logging.getLogger("images_worker.qwen21")

# Атрибут, под которым энкодер живёт на объекте пайплайна.
# Не компонент пайплайна намеренно: зарегистрированный модуль получил бы
# offload-хук и поехал на карту, а нам нужно ровно обратное. Привязка к
# объекту пайплайна даёт правильное время жизни: unload() отпускает оба.
_CPU_TE_ATTR = "_images_worker_cpu_text_encoder"

# Размер плитки и шаг плиточного декодирования VAE, в пикселях картинки.
_VAE_TILE = 1024
_VAE_TILE_STRIDE = 768


class Qwen21Backend(Backend):
    name = "qwen21"

    def _load_model(self, spec: ModelSpec):
        from diffusers import QwenImage21Pipeline

        if spec.variant not in ("fp8", None):
            raise ValueError("unknown variant %r for %s" % (spec.variant, spec.id))
        log.info("loading %s (transformer storage: %s)", spec.repo_id, spec.variant or "bf16")

        pipe = QwenImage21Pipeline.from_pretrained(spec.repo_id, torch_dtype=torch.bfloat16)

        # Декодер VAE на родных 2K-пресетах целиком в 24 ГБ не помещается:
        # 1664x928 берёт 10.5 ГБ, а 2048x2048 и 2752x1536 падают с OOM. Плитки
        # 1024 px с шагом 768 (перекрытие 256 px сглаживает стыки) дают пик
        # 7.4 ГБ и ~4 с на картинку. Плитки 512 не экономнее (8.0 ГБ), плитки
        # 1536 съедают 15.8 ГБ.
        pipe.vae.enable_tiling(
            tile_sample_min_height=_VAE_TILE,
            tile_sample_min_width=_VAE_TILE,
            tile_sample_stride_height=_VAE_TILE_STRIDE,
            tile_sample_stride_width=_VAE_TILE_STRIDE,
        )

        if spec.variant == "fp8":
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
            )

        if not spec.cpu_offload:
            # Большая карта (qwen-image-2.1-full): всё на GPU, энкодер в том числе.
            # RAM хоста после загрузки свободна, VRAM занята резидентно — это
            # отражено в weights_vram_mb спецификации.
            pipe.to("cuda")
            setattr(pipe, _CPU_TE_ATTR, None)
            return pipe

        # Снимаем энкодер с пайплайна ДО навешивания offload-хуков: иначе
        # enable_model_cpu_offload() включит его в цепочку и он окажется на
        # карте. fp32, потому что bf16 на CPU без AMX/AVX512 считается
        # программно и на порядок медленнее.
        cpu_text_encoder = pipe.text_encoder
        pipe.register_modules(text_encoder=None)
        cpu_text_encoder.to(device="cpu", dtype=torch.float32)
        cpu_text_encoder.eval()

        # Трансформер и VAE живут в системной RAM и приезжают на карту
        # помодульно — как у остальных diffusers-бэкендов.
        pipe.enable_model_cpu_offload()
        setattr(pipe, _CPU_TE_ATTR, cpu_text_encoder)
        return pipe

    def _encode_on_cpu(self, text: str):
        """Закодировать промпт энкодером, который на карту не ездит.

        Кодирование делает сам пайплайн (у него шаблон чата, отбрасывание
        системного префикса и обход нормализации последнего слоя) — мы лишь
        на время вызова возвращаем ему энкодер и говорим считать на CPU.
        """
        pipe = self._pipe
        started = time.monotonic()
        pipe.text_encoder = getattr(pipe, _CPU_TE_ATTR)
        try:
            with torch.no_grad():
                embeds, mask, _ = pipe.encode_prompt(prompt=text, device=torch.device("cpu"))
        finally:
            pipe.text_encoder = None
        log.info("prompt encoded on cpu in %.1fs (%d tokens)", time.monotonic() - started, embeds.shape[1])
        # Готовые эмбеддинги пайплайн на карту сам не переносит: он строит из них
        # служебные маски на том же устройстве, и трансформер получает смесь
        # cpu и cuda. Поэтому переносим явно, вместе с маской.
        device = pipe._execution_device
        if mask is not None:
            mask = mask.to(device)
        return embeds.to(device=device, dtype=torch.bfloat16), mask

    def _generate(self, spec: ModelSpec, job: GenerateJob):
        seed = self._seed(job)
        pipe = self._pipe
        fixed_schedule = spec.guidance_param == "none"  # Turbo

        kwargs = {
            "width": job.width,
            "height": job.height,
            "generator": self._generator(seed),
            "callback_on_step_end": self._step_callback(),
        }
        if not fixed_schedule:
            kwargs["num_inference_steps"] = job.num_inference_steps

        use_cfg = not fixed_schedule and job.guidance_scale > 1.0
        if job.negative_prompt and not use_cfg:
            log.warning("negative_prompt ignored: CFG is off (guidance_scale=%s)", job.guidance_scale)

        self._raise_if_cancelled()
        if use_cfg:
            kwargs["true_cfg_scale"] = job.guidance_scale
        # Пустой negative CFG не включает (у Qwen нет bos-токена), пробел — включает.
        negative = (job.negative_prompt or " ") if use_cfg else None
        if getattr(pipe, _CPU_TE_ATTR, None) is None:
            # Энкодер на карте вместе с остальным: пайплайн кодирует сам.
            kwargs["prompt"] = job.prompt
            if negative is not None:
                kwargs["negative_prompt"] = negative
        else:
            kwargs["prompt_embeds"], kwargs["prompt_embeds_mask"] = self._encode_on_cpu(job.prompt)
            if negative is not None:
                self._raise_if_cancelled()
                kwargs["negative_prompt_embeds"], kwargs["negative_prompt_embeds_mask"] = self._encode_on_cpu(
                    negative
                )
        self._raise_if_cancelled()

        result = pipe(**kwargs)
        image = result.images[0]
        del result
        return _drop_noise_alpha(image), seed


def _drop_noise_alpha(image):
    """Вернуть RGB, если прозрачности на картинке по сути нет.

    2.1 всегда отдаёт RGBA: модель умеет прозрачный фон, но только когда об этом
    прямо просят в промпте. На обычном запросе альфа — это шум около единицы
    (замер 2026-11-17: минимум 230..240 из 255), и оставлять её вредно дважды:
    просмотрщики рисуют едва заметные полупрозрачные пятна, а PNG тяжелеет на
    ~15% (10.2 МБ против 8.7 МБ на 2752x1536). Цвет при этом не премультиплицирован,
    так что альфу можно просто отбросить, без подложки.

    Настоящая прозрачность (альфа местами ниже половины) остаётся как есть.
    """
    if image.mode != "RGBA":
        return image
    lowest_alpha, _ = image.getchannel("A").getextrema()
    if lowest_alpha >= 128:
        return image.convert("RGB")
    return image
