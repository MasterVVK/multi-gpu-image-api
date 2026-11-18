#!/usr/bin/env bash
# Структурные тесты бэкенда minimax на CPU: настоящие блоки diffusers, крошечные случайные веса.
# Нужен python со стеком воркера (torch, diffusers d9586a3, transformers) и pytest;
# по умолчанию — venv Qwen-Image 2.1 в $IMG_PROJECTS_ROOT (если не задан — домашний каталог).
#   PY=/path/to/venv/bin/python ./run_tests.sh [аргументы pytest]
#
# Поддельный кэш весов fake_hf/ в git не хранится: conftest.py собирает его сам при первом
# запуске (make_tiny_repo.py). Мелкие конфиги (model_index, scheduler, tokenizer, processor)
# он копирует из настоящего снапшота в ~/.cache/huggingface/hub — без весов хватит такого:
#   hf download MiniMaxAI/MiniMax-H3-Image --include '*.json' --include 'scheduler/*' \
#       --include 'tokenizer/*' --include 'processor/*'
# Собранный кэш остаётся только для чтения; пересобрать: chmod -R u+w fake_hf && rm -rf fake_hf.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PY:-${IMG_PROJECTS_ROOT:-$HOME}/Qwen-Image-2.1/venv/bin/python}"

# Пакет воркера берётся прямо из репозитория шлюза (../../../worker), исходники шлюза —
# из ../../../src: бэкенд minimax и его запись уже в реестре, патч больше не нужен.
REPO="$(cd "$ROOT/../.." && pwd)"
export IMG_GATEWAY_SRC="${IMG_GATEWAY_SRC:-$REPO/src}"

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/worker:$HERE/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
cd "$HERE"
if [ "$#" -eq 0 ]; then
    set -- test_minimax_backend.py test_minimax_registry.py
fi
exec "$PY" -m pytest -p no:cacheprovider -q "$@"
