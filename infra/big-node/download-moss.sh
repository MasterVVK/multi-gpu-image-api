#!/usr/bin/env bash
# Загрузка весов MOSS-Image SFT на узел временным контейнером (на хост ничего не ставится).
# Снапшот целиком: пайплайн подключается как custom_pipeline из каталога снапшота.
set -euo pipefail
# Каталог комплекта — тот, где лежит скрипт: веса пишутся в ./hf рядом с compose.yml.
ROOT="$(cd "$(dirname "$0")" && pwd)"
docker rm -f moss-dl >/dev/null 2>&1 || true
docker run -d --name moss-dl --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
  -v "$ROOT/hf:/hf" python:3.12-slim \
  sh -c "pip install -q --user huggingface_hub && until /tmp/.local/bin/hf download OpenMOSS-Team/MOSS-Image-SFT --max-workers 4; do echo retry; sleep 10; done; echo DOWNLOAD_DONE"
