#!/usr/bin/env bash
# Загрузка весов HiDream-O1-Image на узел временным контейнером (на хост ничего не ставится).
set -euo pipefail
# Каталог комплекта — тот, где лежит скрипт: веса пишутся в ./hf рядом с compose.yml.
ROOT="$(cd "$(dirname "$0")" && pwd)"
docker rm -f hidream-dl >/dev/null 2>&1 || true
docker run -d --name hidream-dl --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
  -v "$ROOT/hf:/hf" python:3.12-slim \
  sh -c "pip install -q --user huggingface_hub && until /tmp/.local/bin/hf download HiDream-ai/HiDream-O1-Image --revision 0b0901d99f200389e138c61946af1185f5f49a13 --exclude 'assets/*' --max-workers 4; do echo retry; sleep 10; done; echo DOWNLOAD_DONE"
