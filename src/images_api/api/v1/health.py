"""Пробы. Живут вне префикса /api/v1 и без API-ключа — их дёргает systemd и watchdog.

/healthz — процесс жив.
/readyz  — жив и полезен: есть готовый узел и отвечает БД. Иначе 503.

Отдельно оставлен /api/v1/health — сводный, в форме, похожей на старые четыре
сервиса, чтобы существующие дашборды не начали получать 404 после переключения.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ...core.security import require_api_key
from ...domain.schemas import HealthResponse, ReadyResponse
from ..deps import get_enhancer, get_pool, get_tasks

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz", response_model=ReadyResponse)
async def readyz(request: Request) -> JSONResponse:
    deps: dict[str, str] = {}

    try:
        await request.app.state.db.fetch_one("SELECT 1 AS ok")
        deps["sqlite"] = "ok"
    except Exception as exc:
        deps["sqlite"] = "fail: %s" % str(exc)[:120]

    pool = get_pool(request)
    ready = pool.ready_count
    deps["nodes"] = "ok (%d/%d готовы)" % (ready, len(pool.nodes))
    if ready == 0:
        deps["nodes"] = "fail: нет готовых узлов (%d всего)" % len(pool.nodes)

    for name, node in pool.nodes.items():
        deps["node:%s" % name] = (
            "ok, свободно %d МБ" % node.gpu_free_mb
            if node.healthy
            else "fail: %s" % (node.last_error or "не отвечает")[:120]
        )

    ok = all(not value.startswith("fail") for value in deps.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content=ReadyResponse(status="ok" if ok else "degraded", deps=deps).model_dump(),
    )


@router.get("/api/v1/health", response_model=HealthResponse, dependencies=[Depends(require_api_key)])
async def legacy_health(request: Request) -> HealthResponse:
    pool = get_pool(request)
    counts = await get_tasks(request).count_by_status()
    return HealthResponse(
        status="healthy" if pool.ready_count else "waiting",
        version=request.app.version,
        nodes_total=len(pool.nodes),
        nodes_ready=pool.ready_count,
        queued=counts.get("queued", 0) + counts.get("waiting_vram", 0),
        running=sum(counts.get(s, 0) for s in ("leased", "loading", "generating", "saving")),
    )


@router.get("/api/v1/pe/status", dependencies=[Depends(require_api_key)])
async def pe_status(request: Request) -> dict:
    """Состояние цепочки Prompt Enhancer — какой эндпоинт сейчас отвечает."""
    enhancer = get_enhancer(request)
    return {"configured": enhancer.configured, "endpoints": await enhancer.probe()}
