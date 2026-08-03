#!/usr/bin/env python3
"""Калибровка VRAM: замерить реальный пик по каждой модели.

В реестре сейчас стоят оценки из документации проектов, а не замеры. Этот
скрипт прогоняет по одной генерации на модель, семплируя nvidia-smi, и печатает
готовые строки для worker/images_worker/registry.py.

Замеряются два числа:
  peak_vram_mb    — максимум во время генерации (за вычетом фона от соседей);
  weights_vram_mb — сколько держится после загрузки, но до генерации.
                    Для моделей с cpu_offload это ~0, и так и должно быть.

Запуск (на том узле, который калибруем):
    .venv/bin/python scripts/measure_vram.py
    .venv/bin/python scripts/measure_vram.py z-image-turbo hidream-o1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))

from images_worker.registry import MODELS  # noqa: E402


def local_gpu_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def node_gpu_used_mb(base: str, key: str, node: str | None) -> int:
    """Занятость карты КОНКРЕТНОГО узла.

    nvidia-smi локально показал бы карту node-a, а задача может выполняться на
    node-b — цифры были бы не от той карты вовсе. Правильный источник — тот же,
    которым пользуется планировщик: /api/v1/nodes, куда каждый узел сообщает
    показания собственного nvidia-smi.
    """
    if node is None:
        return local_gpu_used_mb()
    try:
        for entry in api("GET", "/api/v1/nodes", key, base, timeout=15.0)["nodes"]:
            if entry["name"] == node:
                return int(entry["gpu"]["used_mb"])
    except Exception:
        pass
    return 0


class Sampler(threading.Thread):
    """Фоновый семплер занятости карты. Узел задаётся снаружи: он становится
    известен только после того, как планировщик выберет, где считать."""

    def __init__(self, base: str, key: str, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.base, self.key = base, key
        self.interval = interval
        self.peak = 0
        self.node: str | None = None
        # Имя _stop занято внутренним методом threading.Thread — join() падает.
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            with contextlib.suppress(Exception):
                self.peak = max(self.peak, node_gpu_used_mb(self.base, self.key, self.node))
            self._stop_event.wait(self.interval)

    def stop(self) -> int:
        self._stop_event.set()
        self.join(timeout=5)
        return self.peak


def api(method: str, path: str, key: str, base: str, payload: dict | None = None, timeout: float = 60.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method)
    req.add_header("X-API-Key", key)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=None)
    parser.add_argument("--api", default="http://127.0.0.1:8010")
    parser.add_argument("--key", default=None)
    parser.add_argument("--wait", type=float, default=1800.0)
    args = parser.parse_args()

    key = args.key
    if not key:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("IMG_API_KEYS="):
                key = line.split("=", 1)[1].split(",")[0].strip()
                break
    if not key:
        print("не найден API-ключ: задайте --key или IMG_API_KEYS в .env", file=sys.stderr)
        return 2

    models = args.models or sorted(MODELS)
    results: dict[str, tuple[int, int]] = {}

    for model_id in models:
        spec = MODELS[model_id]
        print("\n=== %s (оценка в реестре %d МБ)" % (model_id, spec.peak_vram_mb))

        sampler = Sampler(args.api, key)
        sampler.start()
        baseline = 0

        created = api(
            "POST",
            "/api/v1/generate",
            key,
            args.api,
            {"model": model_id, "prompt": "калибровка VRAM: red cube on a white table", "aspect_ratio": "1:1", "seed": 42},
        )
        task_id = created["task_id"]

        after_load = 0
        deadline = time.monotonic() + args.wait
        status = "queued"
        while time.monotonic() < deadline:
            task = api("GET", "/api/v1/tasks/%s" % task_id, key, args.api)
            status = task["status"]
            if task.get("node") and sampler.node is None:
                # Узел известен только после назначения — до этого момента и
                # фон берём с него же, иначе вычитали бы чужую карту.
                sampler.node = task["node"]
                baseline = node_gpu_used_mb(args.api, key, sampler.node)
                print("  узел %s, фон %d МБ" % (sampler.node, baseline))
            # Момент между «загружено» и «пошли шаги» — это и есть вес модели на карте.
            if status == "generating" and after_load == 0:
                after_load = node_gpu_used_mb(args.api, key, sampler.node)
            if status in ("completed", "failed", "cancelled"):
                break
            time.sleep(1.0)

        peak = sampler.stop()

        if status != "completed":
            print("  НЕ УДАЛОСЬ: %s — %s" % (status, task.get("error")))
            continue

        peak_net = max(0, peak - baseline)
        weights_net = max(0, after_load - baseline)
        results[model_id] = (peak_net, weights_net)
        print(
            "  пик %d МБ (чистыми %d), после загрузки %d МБ (чистыми %d), генерация %.1f с"
            % (peak, peak_net, after_load, weights_net, task.get("generation_time_seconds") or 0)
        )

    if not results:
        return 1

    print("\n\n--- строки для worker/images_worker/registry.py ---")
    print("# Замерено %s, запас +5%%" % time.strftime("%Y-%m-%d"))
    for model_id, (peak_net, weights_net) in results.items():
        spec = MODELS[model_id]
        # +5%: занижение даёт vram_oom (штатно обрабатывается), завышение —
        # vram_unavailable, то есть модель перестаёт запускаться вовсе.
        suggested = int(peak_net * 1.05 // 100 * 100)
        line = "    peak_vram_mb=%d," % suggested
        if spec.weights_vram_mb or weights_net > 1000:
            line += "  weights_vram_mb=%d," % int(weights_net // 100 * 100)
        print("%-20s %s   # было %d" % (model_id, line, spec.peak_vram_mb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
