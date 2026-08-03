"""Сборка нативного контура.

Правило из vedora: прикладные роутеры живут под /api/v1, а пробы — снаружи
префикса, в public_router, и без API-ключа.
"""

from __future__ import annotations

from fastapi import APIRouter

from . import generate, health, images, models, nodes, queue

router = APIRouter(prefix="/api/v1")
router.include_router(generate.router)
router.include_router(images.router)
router.include_router(models.router)
router.include_router(nodes.router)
router.include_router(queue.router)

# /healthz, /readyz и совместимый /api/v1/health — свои полные пути внутри.
public_router = APIRouter()
public_router.include_router(health.router)
