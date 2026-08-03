"""Выдача картинок: проверка пути и подписи.

В старом коде проверялись только суффикс .png и UUID в имени, поэтому путь вида
`2026/08/03/../../../qi-api/outputs/<uuid>.png` успешно читал чужую выдачу.
"""

from __future__ import annotations

import uuid

import pytest

from images_api.core.config import Settings
from images_api.core.errors import NotFoundError
from images_api.core.security import sign_path, verify_path_signature
from images_api.services.storage import ImageStorage


@pytest.fixture
def storage(tmp_path):
    s = Settings(IMG_OUTPUT_DIR=str(tmp_path / "outputs"), IMG_DB_PATH=str(tmp_path / "db.sqlite"))
    st = ImageStorage(s)
    st.prepare()
    return st


def test_write_is_atomic_and_readable(storage):
    payload = b"\x89PNG\r\n\x1a\n" + b"x" * 128
    image_id, rel, size = storage.write(payload)
    assert size == len(payload)
    assert uuid.UUID(image_id)
    assert storage.resolve(rel).read_bytes() == payload
    # Временный файл не должен оставаться.
    assert not list(storage.tmp.glob("*.part"))


@pytest.mark.parametrize(
    "path",
    [
        "../../../etc/passwd",
        "/etc/passwd",
        "2026/08/03/../../../../etc/passwd",
        "2026/08/03/../../../qi-api/outputs/00000000-0000-0000-0000-000000000000.png",
        "legacy/../../etc/passwd",
        "legacy/qwen/../../../../etc/passwd",
        "2026/08/03/not-a-uuid.png",
        "2026/08/03/00000000-0000-0000-0000-000000000000.txt",
        "legacy/unknown-root/2026/08/03/00000000-0000-0000-0000-000000000000.png",
        "\x00/etc/passwd",
    ],
)
def test_traversal_rejected(storage, path):
    with pytest.raises(NotFoundError):
        storage.resolve(path)


def test_missing_file_is_not_found(storage):
    with pytest.raises(NotFoundError):
        storage.resolve("2026/08/03/%s.png" % uuid.uuid4())


def test_signature_roundtrip():
    secret = "s3cret"
    rel = "2026/08/03/%s.png" % uuid.uuid4()
    sig = sign_path(rel, secret)
    assert verify_path_signature(rel, sig, secret)
    # Подпись привязана к пути: чужую подставить нельзя.
    assert not verify_path_signature("2026/08/04/x.png", sig, secret)
    assert not verify_path_signature(rel, sig, "другой-секрет")
    assert not verify_path_signature(rel, "", secret)


def test_image_url_carries_signature(tmp_path):
    s = Settings(
        IMG_OUTPUT_DIR=str(tmp_path / "o"),
        IMG_DB_PATH=str(tmp_path / "d.sqlite"),
        IMG_IMAGE_URL_SECRET="secret",
        IMG_PUBLIC_BASE_URL="http://example/",
    )
    st = ImageStorage(s)
    rel = "2026/08/03/%s.png" % uuid.uuid4()
    assert "sig=" in st.image_url(rel)
    assert st.image_url(rel, absolute=True).startswith("http://example/api/v1/images/")
