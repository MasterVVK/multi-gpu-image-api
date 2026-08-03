"""Фабрика приложения.

create_app(), middleware запроса, единые обработчики ошибок,
роутеры подключаются явно.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.openai import router as openai_router
from .api.v1 import public_router
from .api.v1 import router as v1_router
from .core.config import get_settings
from .core.errors import install_exception_handlers
from .core.lifespan import lifespan
from .core.logging import RequestContextMiddleware

VERSION = "0.1.0"


def create_app() -> FastAPI:
    settings = get_settings()

    # Эксклюзивность аренды узла держится на asyncio.Lock в памяти процесса.
    # Второй воркер uvicorn сломал бы её молча: два процесса раздавали бы одну
    # карту двум задачам одновременно. Лучше не стартовать вовсе.
    concurrency = os.environ.get("WEB_CONCURRENCY", "1")
    if concurrency != "1":
        raise RuntimeError(
            "WEB_CONCURRENCY=%s: images-api работает только в один процесс, иначе "
            "эксклюзивный арбитраж VRAM перестаёт быть эксклюзивным." % concurrency
        )

    app = FastAPI(
        title="images-api",
        description=(
            "Единый API генерации изображений поверх нескольких GPU-узлов. "
            "Нативный контур — /api/v1, OpenAI-совместимый — /v1."
        ),
        version=VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)

    # allow_origins=["*"] вместе с allow_credentials=True — недопустимая
    # комбинация, а именно так было во всех четырёх старых сервисах.
    origins = settings.cors_origins
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["*"],
        )

    install_exception_handlers(app)

    app.include_router(public_router)
    app.include_router(v1_router)
    app.include_router(openai_router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "service": "images-api",
            "version": VERSION,
            "native_api": "/api/v1",
            "openai_api": "/v1",
            "docs": "/docs",
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "images_api.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
