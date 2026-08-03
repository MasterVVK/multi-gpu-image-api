#!/usr/bin/env bash
# Установка юнитов и env-файлов воркеров. Идемпотентно.
#
# Токен воркеров берётся из /home/user/images-api/.env (IMG_WORKER_TOKEN),
# чтобы он гарантированно совпадал с тем, что ждёт шлюз.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR=/etc/images-api
NODE="${IMG_WORKER_NODE:-node-a}"
PROJECTS_ROOT="${IMG_PROJECTS_ROOT:-/home/user}"

# SOCKS-прокси для докачки весов с HuggingFace. Пустые по умолчанию — задайте,
# если прямого выхода в сеть нет. У ERNIE он свой: в нашей сети это был
# отдельный узел, поэтому переменная отдельная.
HF_PROXY="${HF_PROXY:-}"
HF_PROXY_ERNIE="${HF_PROXY_ERNIE:-$HF_PROXY}"

TOKEN="$(grep -E '^IMG_WORKER_TOKEN=' "$ROOT/.env" | cut -d= -f2-)"
if [[ -z "$TOKEN" ]]; then
    echo "IMG_WORKER_TOKEN пуст в $ROOT/.env" >&2
    exit 1
fi

# бэкенд:порт:venv
BACKENDS=(
    "qwen:8101:$PROJECTS_ROOT/qi-api/venv"
    "zimage:8102:$PROJECTS_ROOT/Z-Image-Turbo/venv"
    "ernie:8103:$PROJECTS_ROOT/ERNIE-Image/venv"
    "hidream:8104:$PROJECTS_ROOT/HiDream-O1-Image/venv"
)

sudo install -d -m 0755 "$ENV_DIR"
sudo install -m 0755 "$ROOT/infra/systemd/images-worker" /usr/local/bin/images-worker
sudo install -m 0644 "$ROOT/infra/systemd/images-worker@.service" /etc/systemd/system/
sudo install -m 0644 "$ROOT/infra/systemd/images-api.service" /etc/systemd/system/

for entry in "${BACKENDS[@]}"; do
    IFS=: read -r backend port venv <<< "$entry"
    if [[ ! -x "$venv/bin/python" ]]; then
        echo "пропускаю $backend: нет $venv/bin/python"
        continue
    fi

    tmp="$(mktemp)"
    cat > "$tmp" <<EOF
# Сгенерировано scripts/install_systemd.sh — правки перетрутся.
IMG_WORKER_BACKEND=$backend
IMG_WORKER_NODE=$NODE
IMG_WORKER_HOST=127.0.0.1
IMG_WORKER_PORT=$port
IMG_WORKER_PYTHON=$venv/bin/python
IMG_WORKER_TOKEN=$TOKEN
IMG_WORKER_LOG_LEVEL=info

# Страховка на случай, если упал сам шлюз: он и так управляет выгрузкой,
# но тогда управлять было бы некому.
IMG_WORKER_IDLE_UNLOAD_S=300
# Полный выход процесса после простоя без модели — единственный способ
# отдать CUDA-контекст (~340 МБ на процесс). systemd поднимет обратно.
IMG_WORKER_IDLE_EXIT_SECONDS=900

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Локальные адреса мимо SOCKS: иначе обращения внутри хоста уходят в прокси.
NO_PROXY=127.0.0.1,localhost,::1
no_proxy=127.0.0.1,localhost,::1
EOF

    # Прокси нужен только для докачки весов с HuggingFace.
    case "$backend" in
        ernie|zimage|qwen)
            proxy="$HF_PROXY"
            [[ "$backend" == "ernie" ]] && proxy="$HF_PROXY_ERNIE"
            if [[ -n "$proxy" ]]; then
                cat >> "$tmp" <<EOF
ALL_PROXY=$proxy
HTTP_PROXY=$proxy
HTTPS_PROXY=$proxy
EOF
            fi
            cat >> "$tmp" <<'EOF'
HF_HUB_DOWNLOAD_TIMEOUT=600
HF_HUB_ETAG_TIMEOUT=300
EOF
            ;;
        hidream)
            # Веса локальные, сеть не нужна вовсе.
            echo "IMG_HIDREAM_REPO_DIR=$PROJECTS_ROOT/HiDream-O1-Image" >> "$tmp"
            ;;
    esac

    sudo install -m 0640 -o root -g root "$tmp" "$ENV_DIR/worker-$backend.env"
    rm -f "$tmp"
    echo "настроен $backend → порт $port, venv $venv"
done

sudo systemctl daemon-reload
echo
echo "Готово. Дальше:"
echo "  sudo systemctl enable --now images-worker@{qwen,zimage,ernie,hidream}"
echo "  sudo systemctl enable --now images-api"
