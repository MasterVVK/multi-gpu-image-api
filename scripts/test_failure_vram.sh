#!/usr/bin/env bash
# Проверка, что сервис не удерживает VRAM при сбоях.
#
# Три сценария, каждый — реальный способ уронить генерацию на середине:
#   1. SIGKILL воркеру      — задача должна упасть, память вернуться, воркер подняться;
#   2. DELETE /tasks/{id}   — отмена, память вернуться;
#   3. рестарт шлюза        — задача НЕ теряется (возвращается в очередь), память вернуться.
#
# Модель по умолчанию — z-image-base (бэкенд zimage). У моделей узла с картой 24 ГБ
# включён cpu-offload: веса приезжают на карту только на время генерации, поэтому
# каждый сценарий сначала ждёт статуса generating и только потом роняет задачу.
# Другую модель задавайте вместе с её бэкендом.
#
#   ./scripts/test_failure_vram.sh
#   MODEL=qwen-image-2.1 BACKEND=qwen21 SCENARIO=2 ./scripts/test_failure_vram.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API="${IMG_API:-http://127.0.0.1:8010}"
KEY="${IMG_KEY:-$(grep -E '^IMG_API_KEYS=' "$ROOT/.env" | cut -d= -f2- | cut -d, -f1)}"
MODEL="${MODEL:-z-image-base}"
BACKEND="${BACKEND:-zimage}"
SCENARIO="${SCENARIO:-all}"
# Сторож ждёт IMG_STUCK_VRAM_SECONDS (120) плюс интервал обхода (30).
RECOVER_WAIT="${RECOVER_WAIT:-200}"

FAILED=0

used_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

jq_field() { python3 -c "import sys,json;print(json.load(sys.stdin).get('$1',''))"; }

submit() {
    curl -sS -m 30 -X POST "$API/api/v1/generate" -H "X-API-Key: $KEY" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"failure test\",\"aspect_ratio\":\"1:1\"}" \
        | jq_field task_id
}

task_status() {
    curl -sS -m 15 "$API/api/v1/tasks/$1" -H "X-API-Key: $KEY" | jq_field status
}

# Ждём, пока задача реально начнёт считать: ронять её раньше бессмысленно.
wait_generating() {
    local tid="$1" deadline=$((SECONDS + ${2:-900}))
    while (( SECONDS < deadline )); do
        local s; s=$(task_status "$tid")
        [[ "$s" == "generating" ]] && return 0
        [[ "$s" == "completed" || "$s" == "failed" || "$s" == "cancelled" ]] && return 1
        sleep 5
    done
    return 1
}

wait_vram_back() {
    local baseline="$1" deadline=$((SECONDS + RECOVER_WAIT))
    while (( SECONDS < deadline )); do
        local now; now=$(used_mb)
        if (( now <= baseline + 1500 )); then
            echo "    память вернулась: $now МБ (фон $baseline) за $((SECONDS)) с"
            return 0
        fi
        sleep 10
    done
    echo "    ПАМЯТЬ НЕ ВЕРНУЛАСЬ: $(used_mb) МБ при фоне $baseline МБ"
    return 1
}

check_resident_null() {
    local resident
    resident=$(curl -sS -m 15 "$API/api/v1/nodes" -H "X-API-Key: $KEY" \
        | python3 -c "import sys,json;print(json.load(sys.stdin)['nodes'][0]['resident_model'])")
    if [[ "$resident" == "None" || -z "$resident" ]]; then
        echo "    resident_model = null"
        return 0
    fi
    echo "    ОСТАЛАСЬ РЕЗИДЕНТНАЯ МОДЕЛЬ: $resident"
    return 1
}

scenario_kill() {
    echo
    echo "=== 1. SIGKILL воркеру на середине генерации"
    local base; base=$(used_mb); echo "  фон $base МБ"
    local tid; tid=$(submit); echo "  задача ${tid:0:8}"

    if ! wait_generating "$tid"; then
        echo "  НЕ УДАЛОСЬ дождаться generating (статус $(task_status "$tid"))"; return 1
    fi
    echo "  генерация пошла, занято $(used_mb) МБ — убиваю воркер"
    sudo systemctl kill -s SIGKILL "images-worker@$BACKEND"

    sleep 20
    local st; st=$(task_status "$tid")
    echo "  статус задачи: $st"
    [[ "$st" == "failed" || "$st" == "queued" ]] || { echo "  ОЖИДАЛСЯ failed или queued"; return 1; }

    systemctl is-active "images-worker@$BACKEND" | grep -q active \
        && echo "    воркер поднят systemd" || { echo "    ВОРКЕР НЕ ПОДНЯЛСЯ"; return 1; }

    wait_vram_back "$base" || return 1
    check_resident_null || return 1
    return 0
}

scenario_cancel() {
    echo
    echo "=== 2. Отмена на середине генерации"
    local base; base=$(used_mb); echo "  фон $base МБ"
    local tid; tid=$(submit); echo "  задача ${tid:0:8}"

    if ! wait_generating "$tid"; then
        echo "  НЕ УДАЛОСЬ дождаться generating"; return 1
    fi
    echo "  генерация пошла, занято $(used_mb) МБ — отменяю"
    curl -sS -m 30 -X DELETE "$API/api/v1/tasks/$tid" -H "X-API-Key: $KEY" > /dev/null

    sleep 30
    local st; st=$(task_status "$tid")
    echo "  статус задачи: $st"
    [[ "$st" == "cancelled" ]] || { echo "  ОЖИДАЛСЯ cancelled"; return 1; }

    wait_vram_back "$base" || return 1
    return 0
}

scenario_gateway_restart() {
    echo
    echo "=== 3. Рестарт шлюза на середине генерации"
    local base; base=$(used_mb); echo "  фон $base МБ"
    local tid; tid=$(submit); echo "  задача ${tid:0:8}"

    if ! wait_generating "$tid"; then
        echo "  НЕ УДАЛОСЬ дождаться generating"; return 1
    fi
    echo "  генерация пошла — перезапускаю шлюз"
    sudo systemctl restart images-api
    sleep 25

    local st; st=$(task_status "$tid")
    echo "  статус задачи после рестарта: $st"
    # Работа не должна теряться: задача либо снова в очереди/работе, либо уже готова.
    case "$st" in
        queued|waiting_vram|leased|loading|generating|completed)
            echo "    задача не потеряна" ;;
        *)
            echo "    ЗАДАЧА ПОТЕРЯНА (ожидались queued/generating/completed)"; return 1 ;;
    esac
    return 0
}

case "$SCENARIO" in
    1) scenario_kill || FAILED=1 ;;
    2) scenario_cancel || FAILED=1 ;;
    3) scenario_gateway_restart || FAILED=1 ;;
    all)
        scenario_kill || FAILED=1
        scenario_cancel || FAILED=1
        scenario_gateway_restart || FAILED=1
        ;;
    *) echo "неизвестный SCENARIO=$SCENARIO"; exit 2 ;;
esac

echo
if (( FAILED )); then
    echo "ЕСТЬ ПРОБЛЕМЫ"
else
    echo "VRAM при сбоях не зависает"
fi
exit $FAILED
