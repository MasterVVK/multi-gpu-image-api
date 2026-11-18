#!/usr/bin/env bash
# Загрузка весов Z-Image Base на узел временным контейнером (на хост ничего не ставится).
# Бэкенд собирает пайплайн из отдельных файлов репозитория — качаются только они
# (turbo-трансформер не нужен: на узле объявляется только z-image-base-full).
# Плюс transformer/config.json из Tongyi-MAI/Z-Image-Turbo: по нему diffusers в
# from_single_file узнаёт архитектуру, а воркер работает офлайн.
set -euo pipefail
# Каталог комплекта — тот, где лежит скрипт: веса пишутся в ./hf рядом с compose.yml.
ROOT="$(cd "$(dirname "$0")" && pwd)"
FILES="ZImageBase_bf16.safetensors ZImageTurbo_VAE_bf16.safetensors ZImageTurbo_VAE_bf16_config.json \
ZImageTurbo_scheduler_config.json Qwen3/qwen3_bf16.safetensors Qwen3/config.json Qwen3/tokenizer_config.json \
Qwen3/tokenizer.json Qwen3/vocab.json Qwen3/merges.txt Qwen3/special_tokens_map.json"
docker rm -f zimage-dl >/dev/null 2>&1 || true
docker run -d --name zimage-dl --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
  -v "$ROOT/hf:/hf" python:3.12-slim \
  sh -c "pip install -q --user huggingface_hub && until /tmp/.local/bin/hf download DeepBeepMeep/Z-Image $FILES --max-workers 4 && /tmp/.local/bin/hf download Tongyi-MAI/Z-Image-Turbo transformer/config.json; do echo retry; sleep 10; done; echo DOWNLOAD_DONE"
