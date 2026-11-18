#!/usr/bin/env bash
# Загрузка весов ERNIE-Image (SFT) на узел временным контейнером (на хост ничего не ставится).
# Встроенный prompt-enhancer (pe, pe_tokenizer, 7 ГБ) не качается: бэкенд его не грузит.
set -euo pipefail
# Каталог комплекта — тот, где лежит скрипт: веса пишутся в ./hf рядом с compose.yml.
ROOT="$(cd "$(dirname "$0")" && pwd)"
REV=5346b31d68c9c23758ba56ef8be5e9dc174c7f99
docker rm -f ernie-dl >/dev/null 2>&1 || true
docker run -d --name ernie-dl --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
  -v "$ROOT/hf:/hf" python:3.12-slim \
  sh -c "pip install -q --user huggingface_hub && until /tmp/.local/bin/hf download baidu/ERNIE-Image --revision $REV --exclude 'pe/*' --exclude 'pe_tokenizer/*' --exclude 'assets/*' --max-workers 4; do echo retry; sleep 10; done; mkdir -p /hf/hub/models--baidu--ERNIE-Image/refs && printf %s $REV > /hf/hub/models--baidu--ERNIE-Image/refs/main; echo DOWNLOAD_DONE"
# refs/main нужен: hf download по хешу коммита его не пишет, а from_pretrained без revision
# ищет снапшот через refs/main и в офлайне иначе падает с «not fully cached locally».
