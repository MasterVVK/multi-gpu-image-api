"""HTTP-клиент воркера.

Один httpx.AsyncClient на всё приложение (пул соединений переиспользуется).
Тайм-ауты разные по операциям: /status должен отвечать за секунды, а /load
холодной модели может честно идти минуты — замерено 131 с на шард при
непрогретом page cache, восемь шардов у hidream.
"""

from __future__ import annotations

import httpx
from images_worker.protocol import (
    META_HEADER,
    TOKEN_HEADER,
    GenerateJob,
    GenerateMeta,
    LoadResult,
    UnloadResult,
    WorkerStatus,
)

from ..core.logging import get_logger

log = get_logger("worker_client")


class WorkerUnavailable(Exception):
    """Воркер не отвечает: не запущен, перезапускается или оборвался туннель."""


class WorkerRejected(Exception):
    """Воркер ответил ошибкой в своём конверте."""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__("%s: %s" % (code, message))


class WorkerClient:
    def __init__(self, client: httpx.AsyncClient, token: str = "") -> None:
        self._client = client
        self._token = token

    def _headers(self, token: str | None = None) -> dict[str, str]:
        value = token if token is not None else self._token
        return {TOKEN_HEADER: value} if value else {}

    async def _raise_for_worker_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            body = response.json()
            code = body.get("code", "backend_error")
            message = body.get("message", response.text[:500])
        except Exception:
            code, message = "backend_error", response.text[:500]
        raise WorkerRejected(code, message, response.status_code)

    async def healthz(self, base_url: str, timeout: float = 5.0) -> dict:
        try:
            resp = await self._client.get("%s/healthz" % base_url.rstrip("/"), timeout=timeout)
        except httpx.HTTPError as exc:
            raise WorkerUnavailable(str(exc)) from exc
        await self._raise_for_worker_error(resp)
        return resp.json()

    async def status(self, base_url: str, token: str | None = None, timeout: float = 15.0) -> WorkerStatus:
        try:
            resp = await self._client.get(
                "%s/status" % base_url.rstrip("/"), headers=self._headers(token), timeout=timeout
            )
        except httpx.HTTPError as exc:
            raise WorkerUnavailable(str(exc)) from exc
        await self._raise_for_worker_error(resp)
        return WorkerStatus.model_validate(resp.json())

    async def load(
        self, base_url: str, model_id: str, token: str | None = None, timeout: float = 2400.0
    ) -> LoadResult:
        try:
            resp = await self._client.post(
                "%s/load" % base_url.rstrip("/"),
                json={"model_id": model_id},
                headers=self._headers(token),
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise WorkerUnavailable(str(exc)) from exc
        await self._raise_for_worker_error(resp)
        return LoadResult.model_validate(resp.json())

    async def unload(self, base_url: str, token: str | None = None, timeout: float = 120.0) -> UnloadResult:
        try:
            resp = await self._client.post(
                "%s/unload" % base_url.rstrip("/"), headers=self._headers(token), timeout=timeout
            )
        except httpx.HTTPError as exc:
            raise WorkerUnavailable(str(exc)) from exc
        await self._raise_for_worker_error(resp)
        return UnloadResult.model_validate(resp.json())

    async def generate(
        self,
        base_url: str,
        job: GenerateJob,
        token: str | None = None,
        timeout: float = 900.0,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> tuple[bytes, GenerateMeta]:
        """Сгенерировать картинку и получить её байтами.

        Байты в ответе, а не путь к файлу: так один и тот же код работает для
        локального узла и для nv-02 через туннель, не требуя общей файловой
        системы, и `completed` означает ровно «файл у шлюза есть».
        """
        try:
            resp = await self._client.post(
                "%s/generate" % base_url.rstrip("/"),
                json=job.model_dump(),
                headers=self._headers(token),
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise WorkerUnavailable("timeout: %s" % exc) from exc
        except httpx.HTTPError as exc:
            raise WorkerUnavailable(str(exc)) from exc

        await self._raise_for_worker_error(resp)

        payload = resp.content
        if len(payload) > max_bytes:
            raise WorkerRejected(
                "backend_error", "image too large: %d bytes" % len(payload), resp.status_code
            )

        raw_meta = resp.headers.get(META_HEADER)
        if not raw_meta:
            raise WorkerRejected("backend_error", "worker response lacks %s" % META_HEADER, 500)

        return payload, GenerateMeta.from_header(raw_meta)

    async def cancel(self, base_url: str, task_id: str, token: str | None = None, timeout: float = 10.0) -> bool:
        try:
            resp = await self._client.post(
                "%s/cancel" % base_url.rstrip("/"),
                json={"task_id": task_id},
                headers=self._headers(token),
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            log.warning("cancel_failed", base_url=base_url, task_id=task_id, error=str(exc))
            return False
        if resp.status_code >= 400:
            return False
        return bool(resp.json().get("cancelled"))
