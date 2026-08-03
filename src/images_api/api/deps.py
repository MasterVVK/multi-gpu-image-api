"""Зависимости FastAPI.

Всё долгоживущее лежит на app.state (положено в lifespan) и читается отсюда.
Так обработчики не знают ни про порядок инициализации, ни про глобальные объекты.
"""

from __future__ import annotations

from fastapi import Request

from ..core.config import Settings
from ..services.node_pool import NodePool
from ..services.prompt_enhancer import PromptEnhancer
from ..services.scheduler import Scheduler
from ..services.storage import ImageStorage
from ..services.tasks import TaskRepository
from ..services.vram_guard import VramGuard


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_scheduler(request: Request) -> Scheduler:
    return request.app.state.scheduler


def get_tasks(request: Request) -> TaskRepository:
    return request.app.state.tasks


def get_pool(request: Request) -> NodePool:
    return request.app.state.pool


def get_storage(request: Request) -> ImageStorage:
    return request.app.state.storage


def get_enhancer(request: Request) -> PromptEnhancer:
    return request.app.state.enhancer


def get_guard(request: Request) -> VramGuard:
    return request.app.state.guard
