"""Состояние очереди: кто работает, кто ждёт и сколько ещё будет ждать."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request

from ...core.security import require_api_key
from ...domain.schemas import QueueEntry, QueueResponse
from ..deps import get_tasks

router = APIRouter(tags=["queue"])


def _aware(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    return value if value.tzinfo else value.replace(tzinfo=UTC)


@router.get("/queue", response_model=QueueResponse, dependencies=[Depends(require_api_key)])
async def get_queue(request: Request) -> QueueResponse:
    tasks = get_tasks(request)
    rows = await tasks.active()
    now = datetime.now(UTC)

    entries: list[QueueEntry] = []
    for position, row in enumerate(rows, start=1):
        created = _aware(row["created_at"])
        deadline_in = None
        if row["deadline_at"] and row["status"] in ("queued", "waiting_vram"):
            deadline_in = round((_aware(row["deadline_at"]) - now).total_seconds(), 1)

        entries.append(
            QueueEntry(
                task_id=row["task_id"],
                model=row["model_id"],
                status=row["status"],
                node=row["node"],
                priority=row["priority"],
                position=position,
                waiting_seconds=round((now - created).total_seconds(), 1),
                deadline_in_seconds=deadline_in,
            )
        )

    return QueueResponse(
        active=entries,
        queued_count=sum(1 for e in entries if e.status == "queued"),
        # Отдельный счётчик: это те, кому не хватило карты и кто ждёт по
        # остаточному принципу. Рост этого числа — сигнал, что GPU занят соседями.
        waiting_vram_count=sum(1 for e in entries if e.status == "waiting_vram"),
    )
