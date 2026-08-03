#!/usr/bin/env python3
"""Разбор ответа /tasks/{id}/wait для smoke.sh.

Отдельным файлом, а не heredoc'ом: heredoc занимает stdin программой, и
json.load(sys.stdin) в такой конструкции читает пустоту.
Отчёт идёт в stderr, image_url — в stdout, чтобы вызывающий скрипт мог его забрать.
"""
import json
import sys

model, raw = sys.argv[1], sys.argv[2]

try:
    t = json.loads(raw)
except json.JSONDecodeError:
    print("%-18s ОШИБКА: нечитаемый ответ: %s" % (model, raw[:200]), file=sys.stderr)
    sys.exit(1)

if t.get("status") != "completed":
    print("%-18s %s: %s (%s)" % (model, t.get("status"), t.get("error"), t.get("error_code")),
          file=sys.stderr)
    sys.exit(1)

print("%-18s %-9s %sx%-5s шагов=%-3s cfg=%-4s seed=%-11s очередь=%5.1fс загрузка=%6.1fс генерация=%6.1fс pe=%s"
      % (model, t["status"], t["width"], t["height"], t["num_steps"], t["guidance_scale"],
         t["seed_used"], t.get("queue_time_seconds") or 0, t.get("load_time_seconds") or 0,
         t.get("generation_time_seconds") or 0, t.get("pe_status")), file=sys.stderr)
print(t["image_url"])
