#!/usr/bin/env bash
# Перенос кода и весов на node-b по LAN.
#
# rsync, а не tar|ssh: кэш HuggingFace построен на симлинках и хардлинках на
# blob-файлы, и -aH переносит это как есть (tar потребовал бы возни с
# --hard-dereference и раздул бы объём). Плюс rsync возобновляется после обрыва —
# на 60 ГБ это решающее свойство.
#
# Качать заново через SOCKS-прокси бессмысленно: те же байты лежат в метре отсюда.
#
#   REMOTE=user@node-b ./scripts/sync_to_node.sh              # код + веса zimage и moss
#   REMOTE=user@node-b MODELS="zimage" ./scripts/sync_to_node.sh
#   REMOTE=user@node-b DRY=1 ./scripts/sync_to_node.sh        # только показать объём

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Адрес второго узла обязателен: подставлять чей-то конкретный по умолчанию
# нельзя, а угадать невозможно.
REMOTE="${REMOTE:?укажите REMOTE=user@host второго узла}"
SSH_PORT="${SSH_PORT:-22}"
MODELS="${MODELS:-zimage moss}"
HF="$HOME/.cache/huggingface/hub"
# Корень каталогов проектов моделей; по умолчанию — домашний каталог.
PROJECTS_ROOT="${IMG_PROJECTS_ROOT:-$HOME}"
RSYNC_OPTS=(-aH --info=progress2 --partial -e "ssh -p $SSH_PORT")
[[ -n "${DRY:-}" ]] && RSYNC_OPTS+=(--dry-run)

command -v rsync >/dev/null || { echo "нет rsync: sudo apt install rsync"; exit 1; }
ssh -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE" true \
    || { echo "нет доступа по ssh к $REMOTE"; exit 1; }

echo "=== свободно на node-b"
ssh -p "$SSH_PORT" "$REMOTE" 'df -h /home | tail -1'

echo
echo "=== код images-api (без venv, весов и выдачи)"
ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p $PROJECTS_ROOT/images-api"
rsync "${RSYNC_OPTS[@]}" \
    --exclude '.venv/' --exclude 'outputs/' --exclude 'data/' \
    --exclude '__pycache__/' --exclude '.env' \
    "$ROOT/worker" "$ROOT/infra" "$ROOT/scripts" "$ROOT/README.md" \
    "$REMOTE:$PROJECTS_ROOT/images-api/"

for model in $MODELS; do
    echo
    case "$model" in
        zimage)
            echo "=== веса z-image (~31 ГБ, кэш HuggingFace)"
            ssh -p "$SSH_PORT" "$REMOTE" 'mkdir -p ~/.cache/huggingface/hub'
            rsync "${RSYNC_OPTS[@]}" "$HF/models--DeepBeepMeep--Z-Image" "$REMOTE:$HF/"
            ;;
        ernie)
            echo "=== веса ernie (~60 ГБ: turbo + sft)"
            rsync "${RSYNC_OPTS[@]}" \
                "$HF/models--baidu--ERNIE-Image" "$HF/models--baidu--ERNIE-Image-Turbo" \
                "$REMOTE:$HF/"
            ;;
        qwen)
            echo "=== веса qwen (~48 ГБ)"
            rsync "${RSYNC_OPTS[@]}" "$HF/models--Ilus-AI--Qwen-Image-2512-FP8" "$REMOTE:$HF/"
            ;;
        qwen21)
            echo "=== веса qwen-image-2.1 (~66 ГБ: base + turbo)"
            ssh -p "$SSH_PORT" "$REMOTE" 'mkdir -p ~/.cache/huggingface/hub'
            rsync "${RSYNC_OPTS[@]}" \
                "$HF/models--Qwen--Qwen-Image-2.1" "$HF/models--Qwen--Qwen-Image-2.1-Turbo" \
                "$REMOTE:$HF/"
            ;;
        moss)
            echo "=== веса moss-image (~27 ГБ)"
            ssh -p "$SSH_PORT" "$REMOTE" 'mkdir -p ~/.cache/huggingface/hub'
            rsync "${RSYNC_OPTS[@]}" "$HF/models--OpenMOSS-Team--MOSS-Image-SFT" "$REMOTE:$HF/"
            ;;
        *) echo "неизвестная модель: $model"; exit 2 ;;
    esac
done

echo
echo "Готово. Дальше на node-b:"
echo "  cd $PROJECTS_ROOT/images-api/infra/remote-node"
echo "  BACKENDS=\"$MODELS\" WORKER_TOKEN=<IMG_WORKER_TOKEN из .env> ./bootstrap.sh"
