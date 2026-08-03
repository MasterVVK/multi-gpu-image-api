"""Чтение состояния GPU через nvidia-smi.

Два правила, которые здесь важнее всего остального:

1. **Никакого torch.** `/status` вызывается планировщиком каждые несколько секунд
   у всех воркеров. Если бы он трогал `torch.cuda.*`, каждый простаивающий воркер
   создал бы CUDA-контекст (~340 МБ, как сейчас видно у z-image) и четыре
   бездействующих процесса съели бы больше гигабайта VRAM ни за что.

2. **Кэш + блокировка.** Форма взята из /opt/lab-agent/lab-agent.py::sh_cached:
   один Lock на команду, TTL-кэш и отдача устаревшего значения, если вызов упал.
   Планировщик опрашивает узлы часто, а nvidia-smi — недешёвый форк.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

from .protocol import GpuInfo

_TTL_S = 2.0
_STALE_OK_S = 30.0
_TIMEOUT_S = 5.0

_lock = threading.Lock()
_cached: GpuInfo | None = None
_cached_at: float = 0.0


def _gpu_index() -> int:
    """Индекс карты для nvidia-smi.

    Если процессу сузили видимость через CUDA_VISIBLE_DEVICES, то физический
    индекс берём оттуда: внутри процесса карта будет нулевой, а nvidia-smi
    считает по-настоящему.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible.split(",")[0].isdigit():
        return int(visible.split(",")[0])
    return int(os.environ.get("IMG_WORKER_GPU_INDEX", "0"))


def _query() -> GpuInfo:
    index = _gpu_index()
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=True,
        ).stdout.strip()
    except FileNotFoundError:
        return GpuInfo(index=index, available=False, error="nvidia-smi not found")
    except subprocess.TimeoutExpired:
        return GpuInfo(index=index, available=False, error="nvidia-smi timeout")
    except subprocess.CalledProcessError as exc:
        msg = (exc.stderr or "").strip() or "nvidia-smi exit %s" % exc.returncode
        return GpuInfo(index=index, available=False, error=msg)

    line = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return GpuInfo(index=index, available=False, error="unparsable: %r" % line)

    try:
        total, used, free = int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return GpuInfo(index=index, available=False, error="unparsable numbers: %r" % line)

    return GpuInfo(
        index=index,
        name=parts[0],
        total_mb=total,
        used_mb=used,
        free_mb=free,
        available=True,
    )


def gpu_info(force: bool = False) -> GpuInfo:
    """Свежий (в пределах TTL) снимок GPU."""
    global _cached, _cached_at

    with _lock:
        now = time.monotonic()
        if not force and _cached is not None and (now - _cached_at) < _TTL_S:
            info = _cached.model_copy()
            info.age_s = round(now - _cached_at, 2)
            return info

        info = _query()
        if not info.available and _cached is not None and (now - _cached_at) < _STALE_OK_S:
            # Разовый сбой nvidia-smi не должен выкидывать узел из планирования —
            # отдаём последнее известное значение, честно пометив возраст.
            stale = _cached.model_copy()
            stale.age_s = round(now - _cached_at, 2)
            stale.error = info.error
            return stale

        _cached = info
        _cached_at = now
        info = info.model_copy()
        info.age_s = 0.0
        return info


def invalidate() -> None:
    """Сбросить кэш — вызывается сразу после load/unload, когда цифры заведомо изменились."""
    global _cached_at
    with _lock:
        _cached_at = 0.0
