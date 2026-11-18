#!/usr/bin/env bash
# Сторож доступа к GPU у контейнеров воркеров на узле с большой картой. Запускается
# на машине шлюза (там же, где ssh-туннель до узла) таймером
# images-big-node-gpu-watchdog.timer; установка — в шапке
# images-big-node-gpu-watchdog.service.
#
# Зачем. Известная беда nvidia-container-toolkit с cgroup v2: после systemd
# daemon-reload на хосте (обновление пакетов, чужие юниты) уже запущенные
# контейнеры теряют доступ к устройствам — nvidia-smi внутри отвечает
# «Failed to initialize NVML: Unknown Error», воркер объявляет пустой список
# моделей, и шлюз перестаёт слать на узел задачи. Перезапуск процесса не
# помогает, нужен пересоздание контейнера. Впервые поймано 18.11.2026.
#
# Что делает: опрашивает воркеры через туннель; если у воркера gpu.available=false
# и он не выполняет задачу — пересоздаёт его контейнер (docker compose up -d
# --force-recreate <сервис>). Не чаще раза в 10 минут на сервис.
#
# Настройки — в env-файле (по умолчанию /etc/images-api/big-node-watchdog.env,
# другой путь — переменной ENV_FILE). Файл читается как shell-скрипт, поэтому
# значения с пробелами берутся в кавычки:
#
#   IMG_WORKER_TOKEN=<токен воркеров узла>
#   # Команда ssh до узла целиком. BatchMode и ConnectTimeout оставьте: без них
#   # ssh под таймером может повиснуть на вопросе или на соединении.
#   REMOTE_SSH="ssh -o BatchMode=yes -o ConnectTimeout=12 -p 22 -i ~/.ssh/key user@host"
#   # Каталог комплекта на узле (там compose.yml).
#   REMOTE_DIR=minimax-h3-image
#   # Необязательно: «локальный порт туннеля:сервис compose» через пробел.
#   SERVICES="18302:zimage 18303:ernie 18304:hidream 18305:qwen21 18306:moss 18307:worker"
set -uo pipefail

ENV_FILE="${ENV_FILE:-/etc/images-api/big-node-watchdog.env}"
STATE_DIR=/var/tmp/images-big-node-gpu-watchdog
COOLDOWN_S=600

[ -r "$ENV_FILE" ] || { echo "нет доступа к $ENV_FILE"; exit 1; }
# shellcheck source=/dev/null
. "$ENV_FILE"

TOKEN="${IMG_WORKER_TOKEN:-}"
[ -n "$TOKEN" ] || { echo "нет токена в $ENV_FILE"; exit 1; }
REMOTE="${REMOTE_SSH:-}"
[ -n "$REMOTE" ] || { echo "не задан REMOTE_SSH в $ENV_FILE — команда ssh до узла целиком, например: ssh -o BatchMode=yes -p 22 -i ~/.ssh/key user@host"; exit 1; }
REMOTE_DIR="${REMOTE_DIR:-}"
[ -n "$REMOTE_DIR" ] || { echo "не задан REMOTE_DIR в $ENV_FILE — каталог комплекта (compose.yml) на узле"; exit 1; }
# локальный порт туннеля : сервис compose
SERVICES="${SERVICES:-18302:zimage 18303:ernie 18304:hidream 18305:qwen21 18306:moss 18307:worker}"
mkdir -p "$STATE_DIR"

for entry in $SERVICES; do
    port="${entry%%:*}"; svc="${entry#*:}"
    status="$(curl -s -m 15 -H "X-Worker-Token: $TOKEN" "http://127.0.0.1:$port/status")" || continue
    verdict="$(printf '%s' "$status" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print("skip"); sys.exit()
gpu = d.get("gpu") or {}
busy = d.get("current_task") or d.get("loading_model")
if gpu.get("available") is False and not busy:
    print("broken " + str(gpu.get("error") or "")[:80])
else:
    print("ok")')"
    case "$verdict" in
        broken*)
            last="$(cat "$STATE_DIR/$svc" 2>/dev/null || echo 0)"
            now="$(date +%s)"
            if [ $((now - last)) -lt $COOLDOWN_S ]; then
                echo "$svc: GPU недоступна (${verdict#broken }), пересоздание было недавно — жду"
                continue
            fi
            echo "$svc: GPU недоступна (${verdict#broken }) — пересоздаю контейнер"
            echo "$now" > "$STATE_DIR/$svc"
            $REMOTE "cd $REMOTE_DIR && docker compose up -d --force-recreate $svc" 2>&1 | tail -2
            ;;
    esac
done
