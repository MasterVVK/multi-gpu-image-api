"""Управление GPU-узлами во время работы, без редеплоя.

Порядок ввода node-b в строй:
    POST  /api/v1/nodes            {"name": "node-b", ..., "enabled": false}
    POST  /api/v1/nodes/node-b/refresh
    PATCH /api/v1/nodes/node-b      {"enabled": true}
Вывод из строя — {"draining": true}: узел доработает текущую задачу и больше
новых не возьмёт.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ...core.errors import ConflictError, NotFoundError
from ...core.security import require_admin, require_api_key
from ...domain.schemas import (
    NodeCreateRequest,
    NodeInfo,
    NodesResponse,
    NodeUpdateRequest,
)
from ...services.node_pool import NodeRuntime
from ..deps import get_guard, get_pool, get_scheduler, get_settings_dep

router = APIRouter(tags=["nodes"])


@router.get("/nodes", response_model=NodesResponse, dependencies=[Depends(require_api_key)])
async def list_nodes(request: Request) -> NodesResponse:
    pool = get_pool(request)
    return NodesResponse(nodes=[node.to_info() for node in pool.nodes.values()])


@router.post("/nodes", response_model=NodeInfo, dependencies=[Depends(require_admin)])
async def create_node(request: Request, payload: NodeCreateRequest) -> NodeInfo:
    pool = get_pool(request)
    settings = get_settings_dep(request)

    if pool.get(payload.name) is not None:
        raise ConflictError("узел %s уже зарегистрирован" % payload.name)

    node = NodeRuntime(
        name=payload.name,
        kind=payload.kind,
        base_urls=payload.base_urls,
        worker_token=payload.worker_token or settings.worker_token,
        enabled=payload.enabled,
        vram_reserve_mb=payload.vram_reserve_mb,
        weight=payload.weight,
        notes=payload.notes or "",
    )
    await pool.add(node)
    return node.to_info()


@router.patch("/nodes/{name}", response_model=NodeInfo, dependencies=[Depends(require_admin)])
async def update_node(request: Request, name: str, payload: NodeUpdateRequest) -> NodeInfo:
    pool = get_pool(request)
    node = pool.get(name)
    if node is None:
        raise NotFoundError("узел %s не найден" % name)

    if payload.enabled is not None:
        node.enabled = payload.enabled
    if payload.draining is not None:
        node.draining = payload.draining
    if payload.vram_reserve_mb is not None:
        node.vram_reserve_mb = payload.vram_reserve_mb
    if payload.weight is not None:
        node.weight = payload.weight
    if payload.base_urls is not None:
        node.base_urls = payload.base_urls
    if payload.notes is not None:
        node.notes = payload.notes

    await pool.persist(node)
    if payload.enabled or payload.base_urls:
        await pool.refresh_node(node)
    # Включённый узел должен сразу забрать то, что ждёт в очереди.
    get_scheduler(request).notify()
    return node.to_info()


@router.post("/nodes/{name}/refresh", response_model=NodeInfo, dependencies=[Depends(require_admin)])
async def refresh_node(request: Request, name: str) -> NodeInfo:
    pool = get_pool(request)
    node = pool.get(name)
    if node is None:
        raise NotFoundError("узел %s не найден" % name)
    await pool.refresh_node(node)
    get_scheduler(request).notify()
    return node.to_info()


@router.delete("/nodes/{name}", dependencies=[Depends(require_admin)])
async def delete_node(request: Request, name: str) -> dict:
    pool = get_pool(request)
    node = pool.get(name)
    if node is None:
        raise NotFoundError("узел %s не найден" % name)
    if node.busy:
        raise ConflictError("узел занят задачей %s; сначала выставьте draining" % node.busy_task_id)
    await pool.remove(name)
    return {"deleted": name}


@router.get("/nodes/{name}/incidents", dependencies=[Depends(require_api_key)])
async def node_incidents(request: Request, name: str, limit: int = 20) -> dict:
    """Журнал вмешательств сторожа VRAM по узлу."""
    if get_pool(request).get(name) is None:
        raise NotFoundError("узел %s не найден" % name)
    incidents = await get_guard(request).recent_incidents(limit=limit)
    return {"node": name, "incidents": [i for i in incidents if i["node"] == name]}
