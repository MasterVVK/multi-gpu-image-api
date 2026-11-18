#!/usr/bin/env bash
# Развёртывание воркеров на node-b. Запускать НА node-b.
#
# Что делает: создаёт venv под выбранные бэкенды, ставит зафиксированные с node-a
# версии, подключает пакет images_worker .pth-файлом и раскладывает юниты.
# Веса не качает — их привозит scripts/sync_to_node.sh с node-a по LAN.
#
# Чего НЕ делает и почему:
#   * не копирует venv с node-a — в них зашиты абсолютные пути и 25 ГБ колёс;
#   * не ставит ollama и ничего для Prompt Enhancer — PE живёт только в шлюзе.
#
# Особенности node-b: Python 3.10 (на node-a — 3.12), драйвер 580.173.02,
# GPU делится с RAG/TTS/whisper и собственной ollama.
#
#   BACKENDS="zimage moss" PYTHON="$(uv python find 3.12)" WORKER_TOKEN=... ./bootstrap.sh
#   PIP_PROXY=http://proxy:8888 BACKENDS="zimage" WORKER_TOKEN=... ./bootstrap.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKENDS="${BACKENDS:-zimage moss}"
NODE="${IMG_WORKER_NODE:-node-b}"
# Корень каталогов проектов моделей; по умолчанию — домашний каталог.
PROJECTS_ROOT="${IMG_PROJECTS_ROOT:-$HOME}"
WORKER_ROOT="${WORKER_ROOT:-$PROJECTS_ROOT/images-api/worker}"
TOKEN="${WORKER_TOKEN:-}"

declare -A PROJECT=(
    [qwen]="$PROJECTS_ROOT/qi-api"
    [zimage]="$PROJECTS_ROOT/Z-Image-Turbo"
    [ernie]="$PROJECTS_ROOT/ERNIE-Image"
    [qwen21]="$PROJECTS_ROOT/Qwen-Image-2.1"
    # MOSS-Image делит venv с Qwen-Image 2.1 — стек у них общий.
    [moss]="$PROJECTS_ROOT/Qwen-Image-2.1"
)
declare -A PORT=([qwen]=8101 [zimage]=8102 [ernie]=8103 [qwen21]=8105 [moss]=8106)
# Чей лок зависимостей использовать (по умолчанию — одноимённый).
declare -A LOCK_OF=([moss]=qwen21)
# Интерпретатор для новых venv. Стеку qwen21/moss нужен Python >= 3.11
# (transformers >= 5.17), а системный на node-b — 3.10: задайте, например,
# PYTHON="$(uv python find 3.12)".
PYTHON="${PYTHON:-python3}"

# HTTP-прокси для pip. Пустой по умолчанию. Если канал до PyPI медленный,
# разница ощутима: в нашей сети прямой канал давал ~0.27 МБ/с, а через
# прокси ~4 МБ/с — на ~3 ГБ колёс CUDA это полчаса против четырёх минут.
PIP_PROXY="${PIP_PROXY:-}"


if [[ -z "$TOKEN" ]]; then
    echo "Задайте WORKER_TOKEN — тот же IMG_WORKER_TOKEN, что в .env шлюза на node-a" >&2
    exit 1
fi

echo "python: $(python3 -V)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "диск:   $(df -h /home | tail -1 | awk '{print $4" свободно"}')"
echo

for backend in $BACKENDS; do
    root="${PROJECT[$backend]}"
    lock="$HERE/requirements/${LOCK_OF[$backend]:-$backend}.lock.txt"

    [[ -f "$lock" ]] || { echo "нет $lock — снимите его на node-a через scripts/capture_venv_locks.sh"; exit 1; }

    echo "=== $backend → $root"
    mkdir -p "$root"

    if [[ ! -x "$root/venv/bin/python" ]]; then
        if [[ "$backend" == "qwen21" || "$backend" == "moss" ]] \
            && ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
            echo "$backend: нужен Python >= 3.11, а $PYTHON — это $("$PYTHON" -V 2>&1)." >&2
            echo "Задайте PYTHON=/путь/к/python3.12 (например, из 'uv python install 3.12')." >&2
            exit 1
        fi
        "$PYTHON" -m venv "$root/venv"
    fi
    "$root/venv/bin/pip" install --quiet ${PIP_PROXY:+--proxy "$PIP_PROXY"} --upgrade pip wheel

    # torch ставим отдельно и первым: у бэкендов разные сборки CUDA, и если
    # отдать это на откуп общему резолверу, он подтянет не ту.
    case "$backend" in
        ernie)
            "$root/venv/bin/pip" install --quiet ${PIP_PROXY:+--proxy "$PIP_PROXY"} \
                --index-url https://download.pytorch.org/whl/cu121 \
                "torch==2.5.1+cu121" "torchvision==0.20.1+cu121"
            ;;
        qwen)
            "$root/venv/bin/pip" install --quiet ${PIP_PROXY:+--proxy "$PIP_PROXY"} "torch==2.9.1" "torchvision==0.24.1"
            ;;
        zimage)
            "$root/venv/bin/pip" install --quiet ${PIP_PROXY:+--proxy "$PIP_PROXY"} "torch==2.10.0"
            ;;
        qwen21|moss)
            "$root/venv/bin/pip" install --quiet ${PIP_PROXY:+--proxy "$PIP_PROXY"} \
                --index-url https://download.pytorch.org/whl/cu128 \
                "torch==2.9.1" "torchvision==0.24.1"
            ;;
    esac

    # diffusers из git (в локе строка вида «diffusers @ git+https://...@<коммит>»)
    # под шаблон «имя==версия» не попадает, а без него бэкенд не импортируется:
    # ставим отдельно и падаем громко, а не пишем «пропущен» в общем списке.
    direct="$(grep -iE '^diffusers @ ' "$lock" || true)"
    if [[ -n "$direct" ]]; then
        "$root/venv/bin/pip" install --quiet ${PIP_PROXY:+--proxy "$PIP_PROXY"} "$direct" \
            || { echo "    НЕ УДАЛОСЬ поставить $direct — бэкенд работать не будет"; exit 1; }
    fi

    # Из лока берём ТОЧНЫЕ версии только для пакетов, которые определяют
    # поведение модели. Остальное ставим по именам, без версий.
    #
    # Почему не весь лок целиком: он снят на node-a с Python 3.12, и часть
    # транзитивных пакетов там в версиях, которых для 3.10 просто не существует
    # (networkx==3.6.1 требует >=3.11 — на этом первая попытка и встала).
    # Точные версии транзитивных зависимостей нам и не нужны: важны стеки
    # torch/transformers/diffusers, а их мы фиксируем.
    CRITICAL='^(transformers|diffusers|accelerate|tokenizers|safetensors|huggingface[-_]hub|numpy|sentencepiece|einops)=='
    grep -iE "$CRITICAL" "$lock" > /tmp/$backend.pinned.txt || true

    # Остальные — только имена. Хвост == обязателен: без него "^torch" отсекает
    # заодно torchao, а он нужен для FP8-квантизации Qwen (иначе "TorchAoConfig
    # requires torchao").
    grep -viE "^(torch|torchvision|torchaudio|triton)==|^nvidia-|^diffusers @ " "$lock" \
        | grep -viE "$CRITICAL" \
        | cut -d= -f1 | grep -v '^$' > /tmp/$backend.loose.txt

    echo "  точные версии: $(wc -l < /tmp/$backend.pinned.txt), свободные: $(wc -l < /tmp/$backend.loose.txt)"

    # Точную версию пробуем поставить, но если её для этого Python не собирали
    # (numpy==2.4.3 требует >=3.12, transformers может быть таким же) — откатываемся
    # на совместимую в пределах мажорной и громко об этом сообщаем.
    # Молча ставить «что дадут» нельзя: расхождение версии стека меняет поведение
    # модели, и знать об этом нужно до, а не после странных картинок.
    while read -r pinned; do
        [[ -z "$pinned" ]] && continue
        name="${pinned%%==*}"
        version="${pinned##*==}"
        major="${version%%.*}"
        if "$root/venv/bin/pip" install --quiet ${PIP_PROXY:+--proxy "$PIP_PROXY"} "$pinned"; then
            continue
        fi
        echo "    ВНИМАНИЕ: $pinned недоступен для этого Python, беру $name>=$major,<$((major+1))"
        "$root/venv/bin/pip" install --quiet ${PIP_PROXY:+--proxy "$PIP_PROXY"} \
            "$name>=$major,<$((major+1))" \
            || { echo "    НЕ УДАЛОСЬ поставить $name — бэкенд работать не будет"; exit 1; }
    done < /tmp/$backend.pinned.txt
    # Свободные ставим по одному: один недоступный для 3.10 пакет не должен
    # утаскивать за собой весь остальной список.
    while read -r pkg; do
        "$root/venv/bin/pip" install --quiet ${PIP_PROXY:+--proxy "$PIP_PROXY"} "$pkg" \
            || echo "    пропущен: $pkg"
    done < /tmp/$backend.loose.txt

    # images_worker — .pth-файлом, чтобы pip гарантированно не тронул пины.
    site="$("$root/venv/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
    echo "$WORKER_ROOT" > "$site/images_worker.pth"

    if [[ "$backend" == "qwen21" || "$backend" == "moss" ]]; then
        "$root/venv/bin/python" -c "from diffusers import QwenImage21Pipeline" \
            || { echo "    diffusers без QwenImage21Pipeline — бэкенд $backend работать не будет"; exit 1; }
    fi

    "$root/venv/bin/python" -c "
import torch, transformers, images_worker
print('  torch=%s transformers=%s images_worker=%s cuda=%s'
      % (torch.__version__, transformers.__version__, images_worker.__version__,
         torch.cuda.is_available()))
"

    sudo install -d -m 0755 /etc/images-api
    tmp="$(mktemp)"
    cat > "$tmp" <<EOF
IMG_WORKER_BACKEND=$backend
IMG_WORKER_NODE=$NODE
IMG_WORKER_HOST=127.0.0.1
IMG_WORKER_PORT=${PORT[$backend]}
IMG_WORKER_PYTHON=$root/venv/bin/python
IMG_WORKER_TOKEN=$TOKEN
IMG_WORKER_LOG_LEVEL=info
IMG_WORKER_IDLE_UNLOAD_S=300
IMG_WORKER_IDLE_EXIT_S=900
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NO_PROXY=127.0.0.1,localhost,::1
no_proxy=127.0.0.1,localhost,::1
EOF
    # Веса привозит sync_to_node.sh; в сеть этим бэкендам ходить незачем, а MOSS
    # ещё и исполняет код пайплайна из снапшота — он не должен обновиться сам.
    [[ "$backend" == "qwen21" || "$backend" == "moss" ]] && echo "HF_HUB_OFFLINE=1" >> "$tmp"
    sudo install -m 0640 -o root -g root "$tmp" "/etc/images-api/worker-$backend.env"
    rm -f "$tmp"
done

sudo install -m 0755 "$WORKER_ROOT/../infra/systemd/images-worker" /usr/local/bin/images-worker
sudo install -m 0644 "$WORKER_ROOT/../infra/systemd/images-worker@.service" /etc/systemd/system/
sudo systemctl daemon-reload

echo
echo "Готово. Дальше на node-b:"
for backend in $BACKENDS; do
    echo "  sudo systemctl enable --now images-worker@$backend"
done
echo
echo "Затем на node-a зарегистрировать узел (см. README, раздел про node-b)."
