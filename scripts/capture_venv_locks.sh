#!/usr/bin/env bash
# Снять точные версии из каждого venv на br-01, чтобы воспроизвести их на nv-02.
#
# venv НЕ копируются: в них зашиты абсолютные пути (shebang'и, pyvenv.cfg), да и
# 25 ГБ колёс гонять незачем. Вместо этого фиксируем список и ставим заново.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/infra/nv-02/requirements"
mkdir -p "$OUT"

declare -A VENVS=(
    [qwen]=/home/user/qi-api/venv
    [zimage]=/home/user/Z-Image-Turbo/venv
    [ernie]=/home/user/ERNIE-Image/venv
    [hidream]=/home/user/HiDream-O1-Image/venv
)

for name in "${!VENVS[@]}"; do
    venv="${VENVS[$name]}"
    [[ -x "$venv/bin/python" ]] || { echo "пропускаю $name"; continue; }

    # --exclude-editable убирает наш же images_worker (он ставится .pth-файлом).
    "$venv/bin/pip" freeze --exclude-editable > "$OUT/$name.lock.txt"

    {
        echo "# Снято с br-01 $(date +%F) из $venv"
        echo "# python: $("$venv/bin/python" -V 2>&1)"
        echo "# ВНИМАНИЕ: на nv-02 Python 3.10 — колёса нужны под cp310."
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
ERNIE_DIRECT=$(find /home/user/ERNIE-Image/venv/lib/*/site-packages/diffusers-*.dist-info -name direct_url.json 2>/dev/null | head -1)
if [[ -n "$ERNIE_DIRECT" ]]; then
    cp "$ERNIE_DIRECT" "$OUT/ernie.diffusers.direct_url.json"
    echo "diffusers у ernie: $(python3 -c "import json;d=json.load(open('$ERNIE_DIRECT'));print(d.get('url'), d.get('vcs_info',{}).get('commit_id'))")"
fi
