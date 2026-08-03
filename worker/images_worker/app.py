"""HTTP-фасад воркера.

Слушает только 127.0.0.1. Наружу воркер не смотрит никогда: удалённый узел
доступен шлюзу исключительно через `ssh -L` (см. infra/systemd/images-tunnel@.service).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse

from . import gpu
from .protocol import (
    META_HEADER,
    TOKEN_HEADER,
    CancelRequest,
    GenerateJob,
    GenerationCancelled,
    LoadRequest,
    LoadResult,
    UnloadResult,
    WorkerError,
    WorkerStatus,
)
from .registry import MODELS, get_spec
from .runner import WorkerRunner
from .settings import WorkerSettings, load_settings

log = logging.getLogger("images_worker")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=WorkerError(code=code, message=message).model_dump())


def create_app(settings: WorkerSettings | None = None) -> FastAPI:
    settings = settings or load_settings()
    runner = WorkerRunner(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        log.info(
            "worker %s starting on %s:%s (node=%s, models=%s)",
            settings.backend,
            settings.host,
            settings.port,
            settings.node,
            ",".join(runner.backend.supported_models()) or "none",
        )
        runner.start_background()
        if settings.preload_model:
            # Предзагрузка — только по явному указанию. По умолчанию воркер
            # стартует пустым: модель приезжает по первой задаче от шлюза,
            # и до тех пор карта свободна для соседей.
            try:
                await runner.load(settings.preload_model)
            except Exception as exc:
                log.error("preload %s failed: %s", settings.preload_model, exc)
        try:
            yield
        finally:
            await runner.stop_background()
            log.info("worker %s stopped", settings.backend)

    app = FastAPI(
        title="images-worker: %s" % settings.backend,
        version=__import__("images_worker").__version__,
        lifespan=lifespan,
    )
    app.state.runner = runner
    app.state.settings = settings

    async def require_token(x_worker_token: str = Header(default="", alias=TOKEN_HEADER)) -> None:
        if settings.token and x_worker_token != settings.token:
            raise HTTPException(status_code=401, detail="invalid worker token")

    # /healthz намеренно без токена и без единого обращения к GPU:
    # это проба для systemd и для watchdog'а ssh-туннеля.
    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "backend": settings.backend,
            "node": settings.node,
            "resident_model": runner.backend.resident_model,
        }

    @app.get("/status", response_model=WorkerStatus, dependencies=[Depends(require_token)])
    async def status() -> WorkerStatus:
        return runner.status()

    @app.post("/load", response_model=LoadResult, dependencies=[Depends(require_token)])
    async def load(req: LoadRequest):
        try:
            spec = get_spec(req.model_id)
        except KeyError as exc:
            return _error(404, "model_unavailable", str(exc))
        if spec.backend != settings.backend:
            return _error(
                400,
                "model_unavailable",
                "model %s belongs to backend %s, this worker serves %s"
                % (req.model_id, spec.backend, settings.backend),
            )
        try:
            already, elapsed, info = await runner.load(req.model_id)
        except MemoryError as exc:
            return _error(507, "vram_oom", str(exc))
        except Exception as exc:
            if _is_oom(exc):
                return _error(507, "vram_oom", str(exc))
            return _error(500, "backend_error", str(exc))
        return LoadResult(
            model_id=req.model_id, already_loaded=already, load_time_seconds=round(elapsed, 2), gpu=info
        )

    @app.post("/unload", response_model=UnloadResult, dependencies=[Depends(require_token)])
    async def unload():
        if runner.backend.current_task is not None:
            # Не ждём: шлюз должен узнать про занятость сразу, а не висеть
            # на блокировке до конца пятиминутной генерации.
            return _error(409, "backend_error", "busy with task %s" % runner.backend.current_task)
        was, info = await runner.unload()
        return UnloadResult(unloaded=was, gpu=info)

    @app.post("/generate", dependencies=[Depends(require_token)])
    async def generate(job: GenerateJob):
        try:
            get_spec(job.model_id)
        except KeyError as exc:
            return _error(404, "model_unavailable", str(exc))

        try:
            png, meta = await runner.generate(job)
        except GenerationCancelled:
            return _error(409, "cancelled", "task %s cancelled" % job.task_id)
        except MemoryError as exc:
            return _error(507, "vram_oom", str(exc))
        except Exception as exc:
            if _is_oom(exc):
                return _error(507, "vram_oom", str(exc))
            log.exception("generation failed for task %s", job.task_id)
            return _error(500, "backend_error", "%s: %s" % (type(exc).__name__, exc))

        return Response(
            content=png,
            media_type="image/png",
            headers={META_HEADER: meta.to_header()},
        )

    @app.post("/cancel", dependencies=[Depends(require_token)])
    async def cancel(req: CancelRequest) -> dict:
        return {"cancelled": runner.cancel(req.task_id)}

    @app.get("/models", dependencies=[Depends(require_token)])
    async def models() -> dict:
        supported = set(runner.backend.supported_models())
        return {
            "backend": settings.backend,
            "models": [
                {
                    "id": m.id,
                    "display_name": m.display_name,
                    "weights_present": m.id in supported,
                    "peak_vram_mb": m.peak_vram_mb,
                }
                for m in MODELS.values()
                if m.backend == settings.backend
            ],
            "gpu": gpu.gpu_info().model_dump(),
        }

    return app


def _is_oom(exc: BaseException) -> bool:
    """Распознать нехватку VRAM без импорта torch в модуле верхнего уровня."""
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text
