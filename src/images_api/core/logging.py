"""structlog → JSON, с X-Request-Id в каждой строке.

merge_contextvars первым, ISO-время в UTC, ensure_ascii=False
(логи читают люди, и там встречается кириллица).
Логгеры называются images_api.<компонент>, события — snake_case-глаголы:
task_enqueued, lease_acquired, vram_wait_started, worker_restarted.
"""

from __future__ import annotations

import logging
import sys
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_ID_HEADER = "X-Request-Id"


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=getattr(logging, level.upper(), logging.INFO))

    # uvicorn.access дублирует наш собственный лог запросов — глушим.
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("httpx").setLevel(logging.WARNING)

    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Заводит/пробрасывает X-Request-Id и кладёт его в contextvars.

    После этого любая строка лога, написанная во время обработки запроса,
    автоматически несёт request_id — включая строки из планировщика, если
    он вызван синхронно из обработчика.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def get_logger(component: str):
    return structlog.get_logger("images_api.%s" % component)
