#!/usr/bin/env bash
# Прогон всех моделей через шлюз: постановка → ожидание → скачивание → проверка PNG.
#
# jq на br-01 не установлен, поэтому разбор JSON — через python3.
#
#   ./scripts/smoke.sh                 # все модели, дефолтные шаги
#   ./scripts/smoke.sh z-image-turbo   # одна модель
#   STEPS=8 ./scripts/smoke.sh         # ускоренный прогон

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API="${IMG_API:-http://127.0.0.1:8010}"
KEY="${IMG_KEY:-$(grep -E '^IMG_API_KEYS=' "$ROOT/.env" | cut -d= -f2- | cut -d, -f1)}"
OUT="${OUT_DIR:-/tmp/images-api-smoke}"
PROMPT="${PROMPT:-a red cube on a white table, studio light}"
WAIT="${WAIT:-1750}"   # предел /tasks/{id}/wait — 1800 с

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=(z-image-turbo z-image-base qwen-image ernie-image-turbo ernie-image hidream-o1)
fi

mkdir -p "$OUT"
FAILED=0

for MODEL in "${MODELS[@]}"; do
    BODY="{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"aspect_ratio\":\"1:1\",\"seed\":42"
    [[ -n "${STEPS:-}" ]] && BODY="$BODY,\"num_inference_steps\":$STEPS"
    BODY="$BODY}"

    TID=$(curl -sS -m 30 -X POST "$API/api/v1/generate" -H "X-API-Key: $KEY" \
        -H 'Content-Type: application/json' -d "$BODY" \
        | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("task_id",""))')

    if [[ -z "$TID" ]]; then
        printf '%-18s ОШИБКА: задача не создана\n' "$MODEL"
        FAILED=1
        continue
    fi

    RESULT=$(curl -sS -m $((WAIT + 30)) "$API/api/v1/tasks/$TID/wait?timeout=$WAIT" -H "X-API-Key: $KEY")

    # JSON передаётся аргументом, а не пайпом: heredoc с программой уже занимает
    # stdin, и json.load(sys.stdin) прочитал бы пустоту.
    URL=$(python3 "$ROOT/scripts/_smoke_report.py" "$MODEL" "$RESULT")
    if [[ -z "$URL" ]]; then
        FAILED=1
        continue
    fi

    curl -sS -m 300 -o "$OUT/$MODEL.png" "$API$URL" -H "X-API-Key: $KEY"
    if ! "$ROOT/.venv/bin/python" "$ROOT/scripts/_smoke_verify.py" "$OUT/$MODEL.png" "$MODEL"; then
        FAILED=1
    fi
done

exit $FAILED
