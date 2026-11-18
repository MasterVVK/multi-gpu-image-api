#!/usr/bin/env bash
# Сделать images_worker импортируемым в venv каждой модели и в venv шлюза.
#
# Почему .pth, а не `pip install -e`:
#   * pip потребовал бы сборочный бэкенд (hatchling), которого нет ни в одном
#     из четырёх venv, — значит скачивание через SOCKS-прокси на каждой машине;
#   * .pth физически не способен тронуть ни один пин: это одна строка с путём,
#     которую site.py добавляет в sys.path при старте интерпретатора;
#   * снимается одной командой (см. --uninstall), без следов в метаданных pip.
#
# Идемпотентно. Запускать после любого изменения структуры каталогов
# (правки внутри images_worker/*.py подхватываются сразу, установка не нужна).

set -euo pipefail

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../worker" && pwd)"
# Корень каталогов проектов моделей; по умолчанию — домашний каталог.
PROJECTS_ROOT="${IMG_PROJECTS_ROOT:-$HOME}"
PTH_NAME="images_worker.pth"

VENVS=(
    "$PROJECTS_ROOT/qi-api/venv"
    "$PROJECTS_ROOT/Z-Image-Turbo/venv"
    "$PROJECTS_ROOT/ERNIE-Image/venv"
    "$PROJECTS_ROOT/HiDream-O1-Image/venv"
    "$PROJECTS_ROOT/Qwen-Image-2.1/venv"
    "$(cd "$WORKER_DIR/.." && pwd)/.venv"
)

UNINSTALL=0
[[ "${1:-}" == "--uninstall" ]] && UNINSTALL=1

for VENV in "${VENVS[@]}"; do
    NAME="$(basename "$(dirname "$VENV")")"
    if [[ ! -x "$VENV/bin/python" ]]; then
        printf '%-20s SKIP (нет %s)\n' "$NAME" "$VENV/bin/python"
        continue
    fi

    SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

    if [[ $UNINSTALL -eq 1 ]]; then
        rm -f "$SITE/$PTH_NAME"
        printf '%-20s удалён %s\n' "$NAME" "$PTH_NAME"
        continue
    fi

    echo "$WORKER_DIR" > "$SITE/$PTH_NAME"

    # Проверка обязательна: убеждаемся, что чужой стек цел и images_worker виден.
    # Именно эта строка ловит случай «поставили и сломали окружение».
    # В venv шлюза torch нет и быть не должно — там проверяем только импорт пакета.
    "$VENV/bin/python" - <<'PY'
import importlib, importlib.util, sys

required = ["images_worker"]
if importlib.util.find_spec("torch") is not None:
    required = ["torch", "transformers", "images_worker"]

report = []
for mod in required:
    try:
        m = importlib.import_module(mod)
        report.append("%s=%s" % (mod, getattr(m, "__version__", "ok")))
    except Exception as exc:
        report.append("%s=FAIL(%s)" % (mod, exc))
        sys.exit("ПРОВЕРКА НЕ ПРОШЛА: " + " ".join(report))
print("  " + " ".join(report))
PY
    printf '%-20s OK\n' "$NAME"
done
