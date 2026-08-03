"""Запись и выдача изображений.

Шлюз — единственный, кто пишет на диск. Воркер (в том числе на node-b) отдаёт
байты в HTTP-ответе, шлюз кладёт их в свой outputs/. Так один код-путь работает
для локального и удалённого узла, не нужен сетевой ФС, а `completed` означает
ровно «файл существует».
"""

from __future__ import annotations

import contextlib
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..core.config import Settings
from ..core.errors import NotFoundError
from ..core.logging import get_logger
from ..core.security import SIGNATURE_PARAM, sign_path

log = get_logger("storage")

# Разрешающий список: дата + UUID + .png. Ничего другого отдать нельзя.
_SAFE_REL = re.compile(
    r"^\d{4}/\d{2}/\d{2}/"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"\.png$"
)
_LEGACY_REL = re.compile(r"^(?P<root>[a-z]+)/(?P<rest>.+)$")


class ImageStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = Path(settings.output_dir)
        self.tmp = self.root / "tmp"

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        # Хвосты от прерванных записей — при старте их держать незачем.
        for leftover in self.tmp.glob("*.part"):
            with contextlib.suppress(OSError):
                leftover.unlink()

    def write(self, payload: bytes) -> tuple[str, str, int]:
        """Сохранить PNG атомарно. Возвращает (image_id, относительный путь, байт)."""
        now = datetime.now(UTC)
        image_id = str(uuid.uuid4())
        rel = "%04d/%02d/%02d/%s.png" % (now.year, now.month, now.day, image_id)

        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        part = self.tmp / ("%s.png.part" % image_id)
        with open(part, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        # os.replace атомарен в пределах одной ФС: наполовину записанный файл
        # не может быть отдан клиенту никогда.
        os.replace(part, target)

        return image_id, rel, len(payload)

    # --- выдача ---

    def resolve(self, rel_path: str) -> Path:
        """Проверить путь и вернуть файл.

        Два слоя защиты намеренно: регулярка — разрешающий список, а
        is_relative_to — доказательство вложенности, если регулярку когда-нибудь
        ослабят. В старом коде проверялись только суффикс .png и UUID в имени,
        поэтому `2026/08/03/../../../qi-api/outputs/<uuid>.png` читался успешно.
        """
        if "\x00" in rel_path or rel_path.startswith("/"):
            raise NotFoundError("image not found")

        if rel_path.startswith("legacy/"):
            return self._resolve_legacy(rel_path[len("legacy/") :])

        if not _SAFE_REL.fullmatch(rel_path):
            raise NotFoundError("image not found")

        return self._contained(self.root, rel_path)

    def _resolve_legacy(self, rest: str) -> Path:
        """Старая выдача четырёх сервисов — только на чтение.

        Нужна, чтобы ссылки, записанные до объединения, продолжали открываться.
        """
        match = _LEGACY_REL.match(rest)
        if not match:
            raise NotFoundError("image not found")

        roots = self.settings.legacy_image_roots
        # Корень выбирается из фиксированного словаря, никогда из ввода.
        root = roots.get(match.group("root"))
        if root is None:
            raise NotFoundError("image not found")

        tail = match.group("rest")
        if not _SAFE_REL.fullmatch(tail):
            raise NotFoundError("image not found")

        return self._contained(root, tail)

    @staticmethod
    def _contained(root: Path, rel: str) -> Path:
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            raise NotFoundError("image not found") from None

        target = (resolved_root / rel).resolve()
        if not target.is_relative_to(resolved_root) or not target.is_file():
            raise NotFoundError("image not found")
        return target

    # --- ссылки ---

    def image_url(self, rel_path: str, absolute: bool = False) -> str:
        path = "/api/v1/images/%s" % rel_path
        if self.settings.image_url_secret:
            path = "%s?%s=%s" % (
                path,
                SIGNATURE_PARAM,
                sign_path(rel_path, self.settings.image_url_secret),
            )
        if absolute:
            return self.settings.public_base_url.rstrip("/") + path
        return path

    # --- уборка ---

    def delete(self, rel_path: str) -> bool:
        try:
            target = self._contained(self.root, rel_path)
        except NotFoundError:
            return False
        try:
            target.unlink()
            return True
        except OSError as exc:
            log.warning("image_delete_failed", path=rel_path, error=str(exc))
            return False
