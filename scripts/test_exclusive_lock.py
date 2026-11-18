#!/usr/bin/env python3
"""Проверка эксклюзивной аренды под нагрузкой.

Ставим несколько задач разом и следим за двумя вещами:
  * на каждом узле в любой момент не больше одной активной задачи;
  * пиковая занятость карты не выходит за предел.

Третья проверка косвенная: суммарное время не должно быть меньше, чем позволяет
число задействованных узлов. Эксклюзив — это «одна задача на КАРТУ», поэтому два
узла законно работают параллельно и порог считается с поправкой на них.

Оговорка: пик VRAM снимается через nvidia-smi ЛОКАЛЬНО, то есть только по node-a.
Занятость удалённых узлов видна через /api/v1/nodes, куда её сообщают их
собственные воркеры.

    .venv/bin/python scripts/test_exclusive_lock.py
    .venv/bin/python scripts/test_exclusive_lock.py --models z-image-turbo qwen-image-2.1 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("IMG_API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip()
    raise SystemExit("не найден IMG_API_KEYS в .env")


def api(method: str, path: str, base: str, key: str, payload: dict | None = None, timeout: float = 30.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method)
    req.add_header("X-API-Key", key)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class NodeWatcher(threading.Thread):
    """Опрашивает /api/v1/nodes и nvidia-smi, копит нарушения."""

    def __init__(self, base: str, key: str, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.base, self.key, self.interval = base, key, interval
        self.violations: list[str] = []
        self.peak_vram = 0
        self.max_busy_seen = 0
        # Имя _stop занято внутренним методом threading.Thread — join() падает.
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                nodes = api("GET", "/api/v1/nodes", self.base, self.key)["nodes"]
                busy = [n for n in nodes if n["busy_task_id"]]
                self.max_busy_seen = max(self.max_busy_seen, len(busy))

                # Один узел — максимум одна задача. Дубли имён узлов среди
                # занятых означали бы протёкшую блокировку.
                names = [n["name"] for n in busy]
                if len(names) != len(set(names)):
                    self.violations.append("узел занят двумя задачами: %s" % names)

                for node in nodes:
                    total = node["gpu"]["total_mb"]
                    used = node["gpu"]["used_mb"]
                    if total and used > total - 512:
                        self.violations.append(
                            "%s: занято %d из %d МБ — почти впритык" % (node["name"], used, total)
                        )
            except Exception:
                pass

            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True,
                ).stdout
                self.peak_vram = max(self.peak_vram, int(out.strip().splitlines()[0]))
            except Exception:
                pass

            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8010")
    parser.add_argument("--models", nargs="*", default=["z-image-turbo", "qwen-image-2.1", "z-image-base"])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--wait", type=float, default=3600.0)
    parser.add_argument("--vram-limit", type=int, default=23000, help="МБ, потолок занятости карты")
    args = parser.parse_args()

    key = read_key()
    jobs = [m for _ in range(args.repeat) for m in args.models]
    print("ставлю %d задач разом: %s" % (len(jobs), ", ".join(jobs)))

    watcher = NodeWatcher(args.api, key)
    watcher.start()
    started = time.monotonic()

    task_ids = []
    for index, model in enumerate(jobs):
        created = api(
            "POST", "/api/v1/generate", args.api, key,
            {
                "model": model,
                "prompt": "нагрузочный тест %d: geometric shapes on a white background" % index,
                "aspect_ratio": "1:1",
                "seed": 1000 + index,
            },
        )
        task_ids.append(created["task_id"])

    results = {}
    deadline = time.monotonic() + args.wait
    while task_ids and time.monotonic() < deadline:
        for task_id in list(task_ids):
            task = api("GET", "/api/v1/tasks/%s" % task_id, args.api, key)
            if task["status"] in ("completed", "failed", "cancelled"):
                results[task_id] = task
                task_ids.remove(task_id)
        time.sleep(2.0)

    total_wall = time.monotonic() - started
    watcher.stop()

    print("\n--- результаты ---")
    sum_individual = 0.0
    ok = failed = 0
    for task in results.values():
        gen = task.get("generation_time_seconds") or 0
        load = task.get("load_time_seconds") or 0
        sum_individual += gen + load
        if task["status"] == "completed":
            ok += 1
        else:
            failed += 1
        print(
            "  %-18s %-11s загрузка=%6.1fс генерация=%6.1fс %s"
            % (task["model"], task["status"], load, gen, task.get("error_code") or "")
        )

    print("\n--- проверки ---")
    problems = []

    print("  задач: %d успешно, %d с ошибкой, %d не дождались" % (ok, failed, len(task_ids)))
    if task_ids:
        problems.append("%d задач не завершились за %.0f с" % (len(task_ids), args.wait))

    print("  максимум одновременно занятых узлов: %d" % watcher.max_busy_seen)
    print("  пик VRAM: %d МБ (предел %d)" % (watcher.peak_vram, args.vram_limit))
    if watcher.peak_vram > args.vram_limit:
        problems.append("пик VRAM %d МБ превысил предел %d" % (watcher.peak_vram, args.vram_limit))

    if watcher.violations:
        problems.extend(sorted(set(watcher.violations)))

    # Проверка по времени имеет смысл только с поправкой на число узлов:
    # два узла ЗАКОННО работают параллельно, и суммарное время падает вдвое.
    # Эксклюзив — это «одна задача на КАРТУ», а не «одна задача на систему».
    nodes_used = len({t.get("node") for t in results.values() if t.get("node")}) or 1
    ratio = total_wall / sum_individual if sum_individual else 0
    floor = 0.6 / nodes_used
    print("  узлов задействовано: %d" % nodes_used)
    print("  общее время %.0fс, сумма индивидуальных %.0fс, отношение %.2f (порог %.2f)"
          % (total_wall, sum_individual, ratio, floor))
    if sum_individual and ratio < floor:
        problems.append(
            "быстрее, чем возможно при %d узлах: отношение %.2f — похоже, две задачи "
            "делили одну карту" % (nodes_used, ratio))

    if problems:
        print("\nПРОБЛЕМЫ:")
        for item in problems:
            print("  - %s" % item)
        return 1

    print("\nэксклюзив держится")
    return 0


if __name__ == "__main__":
    sys.exit(main())
