"""Точка входа воркера: `<venv>/bin/python -m images_worker`.

Один процесс = один бэкенд = один venv. Запускается systemd-юнитом
images-worker-<backend>.service с EnvironmentFile=/etc/images-api/worker-<backend>.env.
"""

from __future__ import annotations

import uvicorn

from .app import create_app
from .settings import load_settings


def main() -> None:
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        # workers=1 обязателен: модель и её блокировка живут в памяти процесса.
        workers=1,
        log_level=settings.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
