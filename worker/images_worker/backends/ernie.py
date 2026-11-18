"""Бэкенд ERNIE-Image (turbo и sft).

Перенос $IMG_PROJECTS_ROOT/ERNIE-Image/src/ernie_image_api/model.py. Смена варианта
turbo↔sft — это полная перезагрузка (Base.load сам выгружает предыдущую),
ровно как делал _unload_other() в оригинале.

Ключевое отличие: `use_pe` в пайплайн жёстко передаётся как False.
ErnieImagePipeline.__call__ по умолчанию имеет use_pe=True и тогда поднимает
внутри себя Ministral-3.3B — это +6 ГБ поверх 22 ГБ самой модели, то есть
гарантированный OOM на 24 ГБ. Улучшение промпта делает шлюз, до диспетчеризации
и на чужом GPU (см. services/prompt_enhancer.py).
"""

from __future__ import annotations

import logging

import torch

from ..protocol import GenerateJob
from ..registry import ModelSpec
from .base import Backend
from .qwen import supported_call_kwargs

log = logging.getLogger("images_worker.ernie")


class ErnieBackend(Backend):
    name = "ernie"

    def _load_model(self, spec: ModelSpec):
        import gc

        from diffusers import ErnieImagePipeline

        log.info("loading %s", spec.repo_id)
        # Вариант «всё на карте» (узел с большой картой) не грузит prompt-enhancer
        # вовсе: его веса там не скачаны, а читать 7 ГБ в RAM хоста ради выброса —
        # против правил узла.
        # Обычные узлы по-прежнему выбрасывают его после загрузки (ниже).
        skip_pe = {} if spec.cpu_offload else {"pe": None, "pe_tokenizer": None}
        pipe = ErnieImagePipeline.from_pretrained(spec.repo_id, torch_dtype=torch.bfloat16, **skip_pe)

        # Выбрасываем встроенный prompt-enhancer (Ministral-3.3B) до того, как
        # навесятся offload-хуки.
        #
        # use_pe=False в вызове его НЕ убирает: это отдельный компонент пайплайна,
        # а не ветка в __call__ — он грузится всегда и занимает место наравне с
        # остальными модулями. Улучшением промпта у нас занимается шлюз, причём на
        # чужой карте, так что здесь он бесполезен и вреден: у ernie-* пик и без
        # него упирается в 24 ГБ.
        #
        # Сигнатура компонента допускает None (в логе diffusers это видно как
        # `Expected types for pe: (AutoModelForCausalLM, NoneType)`), поэтому
        # подмена на None легальна.
        if getattr(pipe, "pe", None) is not None:
            try:
                pipe.register_modules(pe=None)
                gc.collect()
                log.info("dropped built-in prompt enhancer (pe) to save memory")
            except Exception as exc:
                # Не критично: без выгрузки просто выше пик. Ронять загрузку из-за
                # смены внутреннего API diffusers не за что.
                log.warning("could not drop pe component: %s", exc)

        if not spec.cpu_offload:
            # Большая карта (ernie-image-full): всё на GPU, RAM хоста после
            # загрузки свободна.
            pipe.to("cuda")
            return pipe
        pipe.enable_model_cpu_offload()
        return pipe

    def _generate(self, spec: ModelSpec, job: GenerateJob):
        seed = self._seed(job)
        accepted = supported_call_kwargs(self._pipe)

        kwargs = {
            "prompt": job.prompt,
            "negative_prompt": job.negative_prompt or None,
            "width": job.width,
            "height": job.height,
            "num_inference_steps": job.num_inference_steps,
            "guidance_scale": job.guidance_scale,
            "generator": self._generator(seed),
        }
        if "use_pe" in accepted:
            # Не убирать. Дефолт пайплайна — True, и он стоит 6 ГБ VRAM.
            kwargs["use_pe"] = False
        if "callback_on_step_end" in accepted:
            kwargs["callback_on_step_end"] = self._step_callback()

        self._raise_if_cancelled()
        result = self._pipe(**kwargs)
        image = result.images[0]
        del result
        return image, seed
