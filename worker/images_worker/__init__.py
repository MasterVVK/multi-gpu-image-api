"""Тонкий воркер генерации изображений.

Пакет ставится в venv каждой модели (`pip install --no-deps -e`) и в venv шлюза.
Шлюз импортирует отсюда только `registry` и `protocol` — модули без тяжёлых
зависимостей. Всё, что тянет torch, лежит в `images_worker.backends` и
импортируется лениво, уже внутри воркера.

Совместимость: Python 3.10+ (на node-b стоит 3.10.12, на node-a — 3.12.7).
Отсюда `from __future__ import annotations`, `timezone.utc` вместо `datetime.UTC`
и отсутствие `match`.
"""

__version__ = "0.1.0"
