"""Иерархия ошибок и два конверта ответа.

Конвертов действительно два, и это осознанно:
  * нативный /api/v1  →  {"error": {"code", "message", "request_id"}}
    — единая форма для всех наших сервисов;
  * /v1 (OpenAI)      →  {"error": {"message", "type", "param", "code"}}
    — иначе openai-SDK не распознает ошибку и покажет пользователю мусор.

Выбор конверта — по префиксу маршрута, см. install_exception_handlers.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    code = "internal_error"
    status_code = 500
    message = "internal error"

    def __init__(self, message: str | None = None, detail: dict[str, Any] | None = None) -> None:
        self.message = message or self.message
        self.detail = detail or {}
        super().__init__(self.message)


class ValidationError(AppError):
    code = "validation_error"
    status_code = 422
    message = "validation error"


class AuthenticationError(AppError):
    code = "authentication_error"
    status_code = 401
    message = "invalid or missing API key"


class NotFoundError(AppError):
    code = "not_found"
    status_code = 404
    message = "not found"


class ConflictError(AppError):
    code = "conflict"
    status_code = 409
    message = "conflict"


class ServiceError(AppError):
    code = "service_error"
    status_code = 503
    message = "service unavailable"


class ModelUnavailableError(AppError):
    """Модель известна, но её негде выполнить: ни один узел не имеет весов."""

    code = "model_unavailable"
    status_code = 503
    message = "no node can serve this model"


class VramUnavailableError(AppError):
    """Карты не хватило и за отведённое время она не освободилась.

    Штатный исход остаточного принципа, а не поломка.
    """

    code = "vram_unavailable"
    status_code = 503
    message = "not enough free VRAM"


def _is_openai_route(request: Request) -> bool:
    return request.url.path.startswith("/v1/")


def _openai_body(message: str, code: str, status: int) -> dict:
    type_map = {
        401: "invalid_request_error",
        404: "invalid_request_error",
        409: "invalid_request_error",
        422: "invalid_request_error",
        503: "server_error",
        504: "timeout",
    }
    return {
        "error": {
            "message": message,
            "type": type_map.get(status, "server_error"),
            "param": None,
            "code": code,
        }
    }


def _native_body(message: str, code: str, request: Request, detail: dict | None = None) -> dict:
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
        }
    }
    if detail:
        body["error"]["detail"] = detail
    return body


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        if _is_openai_route(request):
            return JSONResponse(
                status_code=exc.status_code,
                content=_openai_body(exc.message, exc.code, exc.status_code),
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=_native_body(exc.message, exc.code, request, exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        message = "; ".join(
            "%s: %s" % (".".join(str(p) for p in err.get("loc", ())[1:]), err.get("msg", ""))
            for err in exc.errors()
        ) or "validation error"
        if _is_openai_route(request):
            return JSONResponse(status_code=400, content=_openai_body(message, "validation_error", 400))
        return JSONResponse(
            status_code=422, content=_native_body(message, "validation_error", request)
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = str(exc.detail)
        code = {401: "authentication_error", 404: "not_found", 409: "conflict"}.get(
            exc.status_code, "http_error"
        )
        if _is_openai_route(request):
            return JSONResponse(
                status_code=exc.status_code, content=_openai_body(message, code, exc.status_code)
            )
        return JSONResponse(
            status_code=exc.status_code, content=_native_body(message, code, request)
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        import structlog

        structlog.get_logger("images_api.http").exception(
            "unhandled_error", path=request.url.path, error=str(exc)
        )
        if _is_openai_route(request):
            return JSONResponse(status_code=500, content=_openai_body("internal error", "internal_error", 500))
        return JSONResponse(
            status_code=500, content=_native_body("internal error", "internal_error", request)
        )
