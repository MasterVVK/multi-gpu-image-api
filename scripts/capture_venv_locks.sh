#!/usr/bin/env bash
# Снять точные версии из каждого venv на node-a, чтобы воспроизвести их на node-b.
#
# venv НЕ копируются: в них зашиты абсолютные пути (shebang'и, pyvenv.cfg), да и
# 25 ГБ колёс гонять незачем. Вместо этого фиксируем список и ставим заново.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/infra/remote-node/requirements"
# Корень каталогов проектов моделей; по умолчанию — домашний каталог.
PROJECTS_ROOT="${IMG_PROJECTS_ROOT:-$HOME}"
mkdir -p "$OUT"

declare -A VENVS=(
    [qwen]="$PROJECTS_ROOT/qi-api/venv"
    [zimage]="$PROJECTS_ROOT/Z-Image-Turbo/venv"
    [ernie]="$PROJECTS_ROOT/ERNIE-Image/venv"
    [hidream]="$PROJECTS_ROOT/HiDream-O1-Image/venv"
    [qwen21]="$PROJECTS_ROOT/Qwen-Image-2.1/venv"
)

for name in "${!VENVS[@]}"; do
    venv="${VENVS[$name]}"
    [[ -x "$venv/bin/python" ]] || { echo "пропускаю $name"; continue; }

    # --exclude-editable убирает наш же images_worker (он ставится .pth-файлом).
    "$venv/bin/pip" freeze --exclude-editable > "$OUT/$name.lock.txt"

    {
        echo "# Снято с node-a $(date +%F) из \$IMG_PROJECTS_ROOT/${venv#"$PROJECTS_ROOT"/}"
        echo "# python: $("$venv/bin/python" -V 2>&1)"
        echo "# ВНИМАНИЕ: на node-b Python 3.10 — колёса нужны под cp310."
        if [[ "$name" == "qwen21" ]]; then
            # Единственное место, где записан источник torch: без него bootstrap
            # возьмёт сборку с PyPI, а не ту, на которой сняты замеры.
            echo "# torch ставится с https://download.pytorch.org/whl/cu128, diffusers — коммит main"
            echo "# (см. qwen21.diffusers.direct_url.json). Нужен Python >= 3.11. Этот же venv обслуживает moss."
        fi
        "$venv/bin/python" - <<'PY'
import importlib
for mod in ("torch", "torchvision", "transformers", "diffusers", "accelerate", "safetensors"):
    try:
        m = importlib.import_module(mod)
        print("#   %s==%s" % (mod, getattr(m, "__version__", "?")))
    except Exception:
        pass
PY
    } > "$OUT/$name.notes.txt"

    printf '%-9s %4d пакетов -> %s\n' "$name" "$(wc -l < "$OUT/$name.lock.txt")" "$OUT/$name.lock.txt"
done

# Провенанс diffusers у ERNIE: там нужен не релиз, а конкретный коммит main,
# в котором появился ErnieImagePipeline. Без этой строки воспроизвести нельзя.
ERNIE_DIRECT=$(find "$PROJECTS_ROOT"/ERNIE-Image/venv/lib/*/site-packages/diffusers-*.dist-info -name direct_url.json 2>/dev/null | head -1)
if [[ -n "$ERNIE_DIRECT" ]]; then
    cp "$ERNIE_DIRECT" "$OUT/ernie.diffusers.direct_url.json"
    echo "diffusers у ernie: $(python3 -c "import json;d=json.load(open('$ERNIE_DIRECT'));print(d.get('url'), d.get('vcs_info',{}).get('commit_id'))")"
fi

# То же для Qwen-Image 2.1: QwenImage21Pipeline на момент установки был только
# в main, так что воспроизводимость держится на идентификаторе коммита.
QWEN21_DIRECT=$(find "$PROJECTS_ROOT"/Qwen-Image-2.1/venv/lib/*/site-packages/diffusers-*.dist-info -name direct_url.json 2>/dev/null | head -1)
if [[ -n "$QWEN21_DIRECT" ]]; then
    cp "$QWEN21_DIRECT" "$OUT/qwen21.diffusers.direct_url.json"
    echo "diffusers у qwen21: $(python3 -c "import json;d=json.load(open('$QWEN21_DIRECT'));print(d.get('url'), d.get('vcs_info',{}).get('commit_id'))")"
fi
