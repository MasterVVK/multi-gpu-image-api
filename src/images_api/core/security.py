"""API-ключи и подпись ссылок на картинки.

Два способа авторизации, оба ведут в одно хранилище ключей:
  * `X-API-Key: <key>`        — нативные клиенты;
  * `Authorization: Bearer …` — это то, что шлёт openai-SDK, и без поддержки
    заголовка OpenAI-совместимый слой был бы бесполезен.

Отдельно — подписанные ссылки на изображения. Без них `<img src="...">`
не открывался бы: браузер не умеет добавлять заголовок авторизации к картинке.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

from fastapi import Header, Request

from .config import Settings, get_settings
from .errors import AuthenticationError

SIGNATURE_PARAM = "sig"
_SIGNATURE_LEN = 32


@dataclass(frozen=True)
class ApiKeyInfo:
    key_id: str
    name: str
    role: str  # user | admin

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def key_id(raw_key: str) -> str:
    """Короткий стабильный идентификатор ключа для логов — сам ключ не логируем."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _match(raw_key: str, settings: Settings) -> ApiKeyInfo | None:
    for configured in settings.api_keys:
        # compare_digest, а не ==: сравнение по времени не должно зависеть от ключа.
        if hmac.compare_digest(raw_key, configured):
            # Первый ключ в списке — админский: им заводят и включают узлы.
            role = "admin" if configured == settings.api_keys[0] else "user"
            return ApiKeyInfo(key_id=key_id(configured), name="env", role=role)
    return None


def extract_key(x_api_key: str = "", authorization: str = "") -> str:
    if x_api_key:
        return x_api_key.strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


async def require_api_key(
    x_api_key: str = Header(default="", alias="X-API-Key"),
    authorization: str = Header(default="", alias="Authorization"),
) -> ApiKeyInfo:
    settings = get_settings()
    if not settings.api_keys:
        # Ключи не настроены — авторизация выключена целиком. Такое допустимо
        # только на loopback; иначе приложение не стартует (см. check_startup_security).
        return ApiKeyInfo(key_id="anonymous", name="anonymous", role="admin")

    raw = extract_key(x_api_key, authorization)
    if not raw:
        raise AuthenticationError("missing API key: send X-API-Key or Authorization: Bearer")

    info = _match(raw, settings)
    if info is None:
        raise AuthenticationError("invalid API key")
    return info


async def require_admin(
    x_api_key: str = Header(default="", alias="X-API-Key"),
    authorization: str = Header(default="", alias="Authorization"),
) -> ApiKeyInfo:
    info = await require_api_key(x_api_key, authorization)
    if not info.is_admin:
        raise AuthenticationError("admin API key required")
    return info


def check_startup_security(settings: Settings) -> None:
    """Отказаться стартовать в заведомо небезопасной конфигурации.

    Ровно та проверка, которой не хватало четырём исходным сервисам: все они
    слушали 0.0.0.0 вообще без аутентификации.
    """
    if settings.is_loopback or settings.api_keys or settings.allow_insecure:
        return
    raise RuntimeError(
        "IMG_HOST=%s смотрит наружу, а IMG_API_KEYS пуст. Задайте ключи "
        "или выставьте IMG_ALLOW_INSECURE=true, если это осознанно." % settings.host
    )


# --- подпись ссылок на изображения ---


def sign_path(rel_path: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), rel_path.encode("utf-8"), hashlib.sha256).hexdigest()[
        :_SIGNATURE_LEN
    ]


def verify_path_signature(rel_path: str, signature: str, secret: str) -> bool:
    if not secret or not signature:
        return False
    return hmac.compare_digest(sign_path(rel_path, secret), signature)


async def authorize_image_access(request: Request, rel_path: str) -> None:
    """Пустить к картинке либо по валидному ключу, либо по валидной подписи."""
    settings = get_settings()
    if not settings.images_require_auth or not settings.api_keys:
        return

    signature = request.query_params.get(SIGNATURE_PARAM, "")
    if verify_path_signature(rel_path, signature, settings.image_url_secret):
        return

    raw = extract_key(
        request.headers.get("X-API-Key", ""), request.headers.get("Authorization", "")
    )
    if raw and _match(raw, settings):
        return

    raise AuthenticationError("image access requires a valid API key or signature")


def generate_key() -> str:
    return "img-" + secrets.token_urlsafe(32)
