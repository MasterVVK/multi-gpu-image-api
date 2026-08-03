#!/usr/bin/env python3
"""Проверка остаточного принципа: ждём карту до 10 минут, потом честно отказываем.

Три сценария:

  1. **Ждём и дожидаемся.** Занимаем карту посторонним процессом, ставим задачу,
     убеждаемся что она ушла в waiting_vram. Освобождаем карту — задача должна
     выполниться, а не остаться висеть.

  2. **Ждём и не дожидаемся.** Карту не освобождаем. Через
     IMG_VRAM_WAIT_TIMEOUT_S задача обязана завершиться с error_code=vram_unavailable,
     а не висеть вечно и не падать с CUDA OOM.

  3. **Соседей не трогаем.** Пока задача ждёт, посторонний процесс должен остаться
     жив: images-api не вытесняет чужую работу — в этом весь смысл остаточного принципа.

Занимать карту можно двумя способами:
  --hog N     запустить процесс, съедающий N ГБ через torch (по-настоящему);
  --reserve   временно поднять vram_reserve_mb узла (быстро, без нагрузки на карту).

    .venv/bin/python scripts/test_vram_backpressure.py --quick
    .venv/bin/python scripts/test_vram_backpressure.py --timeout 620
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOG_VENV = "/home/user/HiDream-O1-Image/venv/bin/python"

HOG_CODE = """
import sys, time
import torch
gb = float(sys.argv[1])
seconds = float(sys.argv[2])
block = torch.empty(int(gb * 1024**3 // 2), dtype=torch.float16, device='cuda')
print('заняли %.1f ГБ' % gb, flush=True)
time.sleep(seconds)
del block
"""


def read_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("IMG_API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip()
    raise SystemExit("не найден IMG_API_KEYS в .env")


def api(method: str, path: str, base: str, key: str, payload=None, timeout: float = 30.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method)
    req.add_header("X-API-Key", key)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def gpu_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def wait_status(base, key, task_id, targets, timeout, label):
    """Дождаться одного из статусов. Возвращает задачу или None."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        task = api("GET", "/api/v1/tasks/%s" % task_id, base, key)
        if task["status"] != last:
            print("    [%5.0fс] %s" % (timeout - (deadline - time.monotonic()), task["status"]))
            last = task["status"]
        if task["status"] in targets:
            return task
        time.sleep(3.0)
    print("    не дождались %s за %.0f с (сейчас %s) — %s" % (targets, timeout, last, label))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="ernie-image-turbo", help="модель с высоким пиком VRAM")
    parser.add_argument("--hog", type=float, default=18.0, help="сколько ГБ занять посторонним процессом")
    parser.add_argument("--timeout", type=float, default=660.0, help="ожидание отказа, чуть больше IMG_VRAM_WAIT_TIMEOUT_S")
    parser.add_argument("--quick", action="store_true", help="пропустить сценарий с полным дедлайном")
    args = parser.parse_args()

    key = read_key()
    problems: list[str] = []

    # --- сценарий 1: заняли, задача ждёт, освободили — выполнилась -----------
    print("\n=== 1. Ждём карту и дожидаемся")
    baseline = gpu_used_mb()
    print("  фон %d МБ, занимаю %.0f ГБ на 180 с" % (baseline, args.hog))
    hog = subprocess.Popen(
        [HOG_VENV, "-c", HOG_CODE, str(args.hog), "180"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(25)
    print("  занято сейчас: %d МБ" % gpu_used_mb())

    created = api("POST", "/api/v1/generate", args.api, key,
                  {"model": args.model, "prompt": "backpressure test", "aspect_ratio": "1:1", "seed": 7})
    task_id = created["task_id"]
    print("  задача %s поставлена" % task_id[:8])

    waiting = wait_status(args.api, key, task_id, {"waiting_vram"}, 90, "ожидание waiting_vram")
    if waiting is None:
        problems.append("задача не перешла в waiting_vram при занятой карте")
    else:
        print("  ок: задача ждёт карту, статус waiting_vram")

    if hog.poll() is not None:
        problems.append("посторонний процесс умер, пока задача ждала — мы его вытеснили")
    else:
        print("  ок: посторонний процесс жив — мы его не вытесняли")

    print("  освобождаю карту")
    hog.terminate()
    hog.wait(timeout=60)
    time.sleep(10)

    done = wait_status(args.api, key, task_id, {"completed", "failed", "cancelled"}, 1500, "выполнение")
    if done is None or done["status"] != "completed":
        problems.append("после освобождения карты задача не выполнилась: %s"
                        % (done or {}).get("status"))
    else:
        print("  ок: задача выполнилась за %.0f с генерации" % (done.get("generation_time_seconds") or 0))

    # --- сценарий 2: не освобождаем — отказ по дедлайну ---------------------
    if args.quick:
        print("\n=== 2. Пропущен (--quick)")
    else:
        print("\n=== 2. Ждём карту и не дожидаемся (займёт ~%.0f мин)" % (args.timeout / 60))
        hold = args.timeout + 120
        hog2 = subprocess.Popen(
            [HOG_VENV, "-c", HOG_CODE, str(args.hog), str(hold)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        time.sleep(25)
        created = api("POST", "/api/v1/generate", args.api, key,
                      {"model": args.model, "prompt": "backpressure deadline test", "aspect_ratio": "1:1", "seed": 8})
        task_id = created["task_id"]
        print("  задача %s поставлена, ждём отказа" % task_id[:8])

        failed = wait_status(args.api, key, task_id, {"failed", "completed", "cancelled"},
                             args.timeout, "отказ по дедлайну")
        if failed is None:
            problems.append("задача не завершилась за %.0f с — дедлайн не сработал" % args.timeout)
        elif failed["status"] != "failed" or failed.get("error_code") != "vram_unavailable":
            problems.append("ожидался failed/vram_unavailable, получено %s/%s"
                            % (failed["status"], failed.get("error_code")))
        else:
            print("  ок: отказ vram_unavailable")
            print("  текст: %s" % (failed.get("error") or "")[:200])

        if hog2.poll() is not None:
            problems.append("посторонний процесс умер — мы его вытеснили")
        hog2.terminate()
        hog2.wait(timeout=60)

    print("\n--- итог ---")
    if problems:
        for item in problems:
            print("  ПРОБЛЕМА: %s" % item)
        return 1
    print("  остаточный принцип работает")
    return 0


if __name__ == "__main__":
    sys.exit(main())
