"""Бэкенды генерации: по одному на venv.

Импортируются лениво — `get_backend()` подтягивает только тот модуль, который
нужен этому процессу, потому что каждый тянет свой набор тяжёлых зависимостей
(torch/diffusers/transformers разных версий).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import Backend


def get_backend(name: str) -> Backend:
    if name == "qwen":
        from .qwen import QwenBackend

        return QwenBackend()
    if name == "zimage":
        from .zimage import ZImageBackend

        return ZImageBackend()
    if name == "ernie":
        from .ernie import ErnieBackend

        return ErnieBackend()
    if name == "hidream":
        from .hidream import HiDreamBackend

        return HiDreamBackend()
    raise ValueError("unknown backend %r" % name)
