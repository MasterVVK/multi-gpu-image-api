"""Общая обвязка тестов бэкенда minimax: поддельный кэш HF и «прослушка» пайплайна.

Тесты идут на CPU с настоящими классами diffusers/transformers и крошечными случайными
весами (см. make_tiny_repo.py). Запуск — см. run_tests.sh.
"""

from __future__ import annotations

import functools
import gc
import os
import pathlib
import stat
import subprocess
import sys
import weakref

import pytest

HERE = pathlib.Path(__file__).resolve().parent
FAKE_HF = HERE / "fake_hf"
REPO_DIR = FAKE_HF / "hub" / "models--MiniMaxAI--MiniMax-H3-Image"

# До импорта torch/huggingface_hub: карта не используется, сеть запрещена, кэш — поддельный.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = str(FAKE_HF / "hub")
os.environ["HF_HOME"] = str(HERE / ".hf_home_scratch")
os.environ["IMG_MINIMAX_DEVICE"] = "cpu"
os.environ.pop("IMG_MINIMAX_PLACEMENT", None)


def _tree(root: pathlib.Path) -> dict[str, tuple[int, int]]:
    out = {}
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        out[str(path.relative_to(root))] = (st.st_size if stat.S_ISREG(st.st_mode) else -1, st.st_mtime_ns)
    return out


@pytest.fixture(scope="session", autouse=True)
def fake_cache():
    """Кэш собирается один раз и делается доступным только на чтение — как том `:ro` в контейнере."""
    if not (REPO_DIR / "refs" / "main").is_file():
        subprocess.run([sys.executable, str(HERE / "make_tiny_repo.py"), str(FAKE_HF)], check=True)
    subprocess.run(["chmod", "-R", "a-w", str(FAKE_HF)], check=True)
    before = _tree(FAKE_HF)
    yield FAKE_HF
    assert _tree(FAKE_HF) == before, "кэш с весами изменился за время тестов"


class Tap:
    """Журнал событий пайплайна и слабые ссылки на тяжёлые компоненты."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.refs: dict[str, weakref.ref] = {}
        self.load_kwargs: list[dict] = []
        self.hooks: dict[str, list] = {}

    def alive(self) -> list[str]:
        gc.collect()
        return sorted(name for name, ref in self.refs.items() if ref() is not None)

    def on(self, event: str, fn) -> None:
        self.hooks.setdefault(event, []).append(fn)

    def emit(self, *event) -> None:
        self.events.append(event)
        for fn in self.hooks.get(event[0], []):
            fn(*event[1:])

    def names(self) -> list[str]:
        return [event[0] for event in self.events]


@pytest.fixture()
def tap(monkeypatch):
    import torch
    from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3ImageTransformer2DModel
    from diffusers.modular_pipelines import ModularPipeline
    from diffusers.modular_pipelines.minimax_h3_image import encoders

    tap = Tap()

    real_load = ModularPipeline.load_components
    real_unload = ModularPipeline.unload_components
    real_to = ModularPipeline.to
    real_encode = encoders.get_qwen3vl_prompt_embeds
    real_forward = MiniMaxH3ImageTransformer2DModel.forward
    real_decode = AutoencoderKLMiniMaxH3Image.decode

    def load_components(self, names=None, **kwargs):
        # Кто из ранее загруженных ещё жив в момент, когда начинается загрузка следующих.
        tap.emit("load", tuple(names), tuple(tap.alive()))
        tap.load_kwargs.append(dict(kwargs))
        result = real_load(self, names=names, **kwargs)
        for name, component in self.components.items():
            if isinstance(component, torch.nn.Module):
                tap.refs[name] = weakref.ref(component)
        tap.emit("loaded", tuple(sorted(k for k, v in self.components.items() if v is not None)))
        return result

    def unload_components(self, names):
        result = real_unload(self, names)
        tap.emit("unload", tuple(names))
        return result

    def to(self, *args, **kwargs):
        tap.emit("to", str(args[0]) if args else str(kwargs.get("device")))
        return real_to(self, *args, **kwargs)

    def encode(text_encoder, processor, token_ids, *args, **kwargs):
        tap.emit("text_encoder", len(token_ids))
        return real_encode(text_encoder, processor, token_ids, *args, **kwargs)

    @functools.wraps(real_forward)  # денойзер читает сигнатуру forward
    def forward(self, *args, **kwargs):
        tap.emit("transformer", kwargs.get("pdd_step_index"))
        return real_forward(self, *args, **kwargs)

    @functools.wraps(real_decode)
    def decode(self, z, *args, **kwargs):
        tap.emit("vae.decode", tuple(z.shape))
        return real_decode(self, z, *args, **kwargs)

    monkeypatch.setattr(ModularPipeline, "load_components", load_components)
    monkeypatch.setattr(ModularPipeline, "unload_components", unload_components)
    monkeypatch.setattr(ModularPipeline, "to", to)
    monkeypatch.setattr(encoders, "get_qwen3vl_prompt_embeds", encode)
    monkeypatch.setattr(MiniMaxH3ImageTransformer2DModel, "forward", forward)
    monkeypatch.setattr(AutoencoderKLMiniMaxH3Image, "decode", decode)
    return tap


@pytest.fixture()
def backend(monkeypatch):
    from images_worker.backends import get_backend

    backend = get_backend("minimax")
    backend.load("minimax-h3-image")
    return backend


def make_job(task_id: str = "t1", **overrides):
    from images_worker.protocol import GenerateJob

    params = {
        "task_id": task_id,
        "model_id": "minimax-h3-image",
        "prompt": "a red fox in the snow",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 6,
        "guidance_scale": 1.0,
        "seed": 7,
    }
    params.update(overrides)
    return GenerateJob(**params)
